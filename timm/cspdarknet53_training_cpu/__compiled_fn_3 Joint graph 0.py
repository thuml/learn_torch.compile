from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[32]"; primals_2: "f32[32]"; primals_3: "f32[64]"; primals_4: "f32[64]"; primals_5: "f32[128]"; primals_6: "f32[128]"; primals_7: "f32[32]"; primals_8: "f32[32]"; primals_9: "f32[64]"; primals_10: "f32[64]"; primals_11: "f32[64]"; primals_12: "f32[64]"; primals_13: "f32[64]"; primals_14: "f32[64]"; primals_15: "f32[128]"; primals_16: "f32[128]"; primals_17: "f32[128]"; primals_18: "f32[128]"; primals_19: "f32[64]"; primals_20: "f32[64]"; primals_21: "f32[64]"; primals_22: "f32[64]"; primals_23: "f32[64]"; primals_24: "f32[64]"; primals_25: "f32[64]"; primals_26: "f32[64]"; primals_27: "f32[64]"; primals_28: "f32[64]"; primals_29: "f32[128]"; primals_30: "f32[128]"; primals_31: "f32[256]"; primals_32: "f32[256]"; primals_33: "f32[256]"; primals_34: "f32[256]"; primals_35: "f32[128]"; primals_36: "f32[128]"; primals_37: "f32[128]"; primals_38: "f32[128]"; primals_39: "f32[128]"; primals_40: "f32[128]"; primals_41: "f32[128]"; primals_42: "f32[128]"; primals_43: "f32[128]"; primals_44: "f32[128]"; primals_45: "f32[128]"; primals_46: "f32[128]"; primals_47: "f32[128]"; primals_48: "f32[128]"; primals_49: "f32[128]"; primals_50: "f32[128]"; primals_51: "f32[128]"; primals_52: "f32[128]"; primals_53: "f32[128]"; primals_54: "f32[128]"; primals_55: "f32[128]"; primals_56: "f32[128]"; primals_57: "f32[128]"; primals_58: "f32[128]"; primals_59: "f32[128]"; primals_60: "f32[128]"; primals_61: "f32[128]"; primals_62: "f32[128]"; primals_63: "f32[128]"; primals_64: "f32[128]"; primals_65: "f32[128]"; primals_66: "f32[128]"; primals_67: "f32[128]"; primals_68: "f32[128]"; primals_69: "f32[256]"; primals_70: "f32[256]"; primals_71: "f32[512]"; primals_72: "f32[512]"; primals_73: "f32[512]"; primals_74: "f32[512]"; primals_75: "f32[256]"; primals_76: "f32[256]"; primals_77: "f32[256]"; primals_78: "f32[256]"; primals_79: "f32[256]"; primals_80: "f32[256]"; primals_81: "f32[256]"; primals_82: "f32[256]"; primals_83: "f32[256]"; primals_84: "f32[256]"; primals_85: "f32[256]"; primals_86: "f32[256]"; primals_87: "f32[256]"; primals_88: "f32[256]"; primals_89: "f32[256]"; primals_90: "f32[256]"; primals_91: "f32[256]"; primals_92: "f32[256]"; primals_93: "f32[256]"; primals_94: "f32[256]"; primals_95: "f32[256]"; primals_96: "f32[256]"; primals_97: "f32[256]"; primals_98: "f32[256]"; primals_99: "f32[256]"; primals_100: "f32[256]"; primals_101: "f32[256]"; primals_102: "f32[256]"; primals_103: "f32[256]"; primals_104: "f32[256]"; primals_105: "f32[256]"; primals_106: "f32[256]"; primals_107: "f32[256]"; primals_108: "f32[256]"; primals_109: "f32[512]"; primals_110: "f32[512]"; primals_111: "f32[1024]"; primals_112: "f32[1024]"; primals_113: "f32[1024]"; primals_114: "f32[1024]"; primals_115: "f32[512]"; primals_116: "f32[512]"; primals_117: "f32[512]"; primals_118: "f32[512]"; primals_119: "f32[512]"; primals_120: "f32[512]"; primals_121: "f32[512]"; primals_122: "f32[512]"; primals_123: "f32[512]"; primals_124: "f32[512]"; primals_125: "f32[512]"; primals_126: "f32[512]"; primals_127: "f32[512]"; primals_128: "f32[512]"; primals_129: "f32[512]"; primals_130: "f32[512]"; primals_131: "f32[512]"; primals_132: "f32[512]"; primals_133: "f32[1024]"; primals_134: "f32[1024]"; primals_135: "f32[32, 3, 3, 3]"; primals_136: "f32[64, 32, 3, 3]"; primals_137: "f32[128, 64, 1, 1]"; primals_138: "f32[32, 64, 1, 1]"; primals_139: "f32[64, 32, 3, 3]"; primals_140: "f32[64, 64, 1, 1]"; primals_141: "f32[64, 128, 1, 1]"; primals_142: "f32[128, 64, 3, 3]"; primals_143: "f32[128, 128, 1, 1]"; primals_144: "f32[64, 64, 1, 1]"; primals_145: "f32[64, 64, 3, 3]"; primals_146: "f32[64, 64, 1, 1]"; primals_147: "f32[64, 64, 3, 3]"; primals_148: "f32[64, 64, 1, 1]"; primals_149: "f32[128, 128, 1, 1]"; primals_150: "f32[256, 128, 3, 3]"; primals_151: "f32[256, 256, 1, 1]"; primals_152: "f32[128, 128, 1, 1]"; primals_153: "f32[128, 128, 3, 3]"; primals_154: "f32[128, 128, 1, 1]"; primals_155: "f32[128, 128, 3, 3]"; primals_156: "f32[128, 128, 1, 1]"; primals_157: "f32[128, 128, 3, 3]"; primals_158: "f32[128, 128, 1, 1]"; primals_159: "f32[128, 128, 3, 3]"; primals_160: "f32[128, 128, 1, 1]"; primals_161: "f32[128, 128, 3, 3]"; primals_162: "f32[128, 128, 1, 1]"; primals_163: "f32[128, 128, 3, 3]"; primals_164: "f32[128, 128, 1, 1]"; primals_165: "f32[128, 128, 3, 3]"; primals_166: "f32[128, 128, 1, 1]"; primals_167: "f32[128, 128, 3, 3]"; primals_168: "f32[128, 128, 1, 1]"; primals_169: "f32[256, 256, 1, 1]"; primals_170: "f32[512, 256, 3, 3]"; primals_171: "f32[512, 512, 1, 1]"; primals_172: "f32[256, 256, 1, 1]"; primals_173: "f32[256, 256, 3, 3]"; primals_174: "f32[256, 256, 1, 1]"; primals_175: "f32[256, 256, 3, 3]"; primals_176: "f32[256, 256, 1, 1]"; primals_177: "f32[256, 256, 3, 3]"; primals_178: "f32[256, 256, 1, 1]"; primals_179: "f32[256, 256, 3, 3]"; primals_180: "f32[256, 256, 1, 1]"; primals_181: "f32[256, 256, 3, 3]"; primals_182: "f32[256, 256, 1, 1]"; primals_183: "f32[256, 256, 3, 3]"; primals_184: "f32[256, 256, 1, 1]"; primals_185: "f32[256, 256, 3, 3]"; primals_186: "f32[256, 256, 1, 1]"; primals_187: "f32[256, 256, 3, 3]"; primals_188: "f32[256, 256, 1, 1]"; primals_189: "f32[512, 512, 1, 1]"; primals_190: "f32[1024, 512, 3, 3]"; primals_191: "f32[1024, 1024, 1, 1]"; primals_192: "f32[512, 512, 1, 1]"; primals_193: "f32[512, 512, 3, 3]"; primals_194: "f32[512, 512, 1, 1]"; primals_195: "f32[512, 512, 3, 3]"; primals_196: "f32[512, 512, 1, 1]"; primals_197: "f32[512, 512, 3, 3]"; primals_198: "f32[512, 512, 1, 1]"; primals_199: "f32[512, 512, 3, 3]"; primals_200: "f32[512, 512, 1, 1]"; primals_201: "f32[1024, 1024, 1, 1]"; primals_202: "f32[1000, 1024]"; primals_203: "f32[1000]"; primals_204: "i64[]"; primals_205: "f32[32]"; primals_206: "f32[32]"; primals_207: "i64[]"; primals_208: "f32[64]"; primals_209: "f32[64]"; primals_210: "i64[]"; primals_211: "f32[128]"; primals_212: "f32[128]"; primals_213: "i64[]"; primals_214: "f32[32]"; primals_215: "f32[32]"; primals_216: "i64[]"; primals_217: "f32[64]"; primals_218: "f32[64]"; primals_219: "i64[]"; primals_220: "f32[64]"; primals_221: "f32[64]"; primals_222: "i64[]"; primals_223: "f32[64]"; primals_224: "f32[64]"; primals_225: "i64[]"; primals_226: "f32[128]"; primals_227: "f32[128]"; primals_228: "i64[]"; primals_229: "f32[128]"; primals_230: "f32[128]"; primals_231: "i64[]"; primals_232: "f32[64]"; primals_233: "f32[64]"; primals_234: "i64[]"; primals_235: "f32[64]"; primals_236: "f32[64]"; primals_237: "i64[]"; primals_238: "f32[64]"; primals_239: "f32[64]"; primals_240: "i64[]"; primals_241: "f32[64]"; primals_242: "f32[64]"; primals_243: "i64[]"; primals_244: "f32[64]"; primals_245: "f32[64]"; primals_246: "i64[]"; primals_247: "f32[128]"; primals_248: "f32[128]"; primals_249: "i64[]"; primals_250: "f32[256]"; primals_251: "f32[256]"; primals_252: "i64[]"; primals_253: "f32[256]"; primals_254: "f32[256]"; primals_255: "i64[]"; primals_256: "f32[128]"; primals_257: "f32[128]"; primals_258: "i64[]"; primals_259: "f32[128]"; primals_260: "f32[128]"; primals_261: "i64[]"; primals_262: "f32[128]"; primals_263: "f32[128]"; primals_264: "i64[]"; primals_265: "f32[128]"; primals_266: "f32[128]"; primals_267: "i64[]"; primals_268: "f32[128]"; primals_269: "f32[128]"; primals_270: "i64[]"; primals_271: "f32[128]"; primals_272: "f32[128]"; primals_273: "i64[]"; primals_274: "f32[128]"; primals_275: "f32[128]"; primals_276: "i64[]"; primals_277: "f32[128]"; primals_278: "f32[128]"; primals_279: "i64[]"; primals_280: "f32[128]"; primals_281: "f32[128]"; primals_282: "i64[]"; primals_283: "f32[128]"; primals_284: "f32[128]"; primals_285: "i64[]"; primals_286: "f32[128]"; primals_287: "f32[128]"; primals_288: "i64[]"; primals_289: "f32[128]"; primals_290: "f32[128]"; primals_291: "i64[]"; primals_292: "f32[128]"; primals_293: "f32[128]"; primals_294: "i64[]"; primals_295: "f32[128]"; primals_296: "f32[128]"; primals_297: "i64[]"; primals_298: "f32[128]"; primals_299: "f32[128]"; primals_300: "i64[]"; primals_301: "f32[128]"; primals_302: "f32[128]"; primals_303: "i64[]"; primals_304: "f32[128]"; primals_305: "f32[128]"; primals_306: "i64[]"; primals_307: "f32[256]"; primals_308: "f32[256]"; primals_309: "i64[]"; primals_310: "f32[512]"; primals_311: "f32[512]"; primals_312: "i64[]"; primals_313: "f32[512]"; primals_314: "f32[512]"; primals_315: "i64[]"; primals_316: "f32[256]"; primals_317: "f32[256]"; primals_318: "i64[]"; primals_319: "f32[256]"; primals_320: "f32[256]"; primals_321: "i64[]"; primals_322: "f32[256]"; primals_323: "f32[256]"; primals_324: "i64[]"; primals_325: "f32[256]"; primals_326: "f32[256]"; primals_327: "i64[]"; primals_328: "f32[256]"; primals_329: "f32[256]"; primals_330: "i64[]"; primals_331: "f32[256]"; primals_332: "f32[256]"; primals_333: "i64[]"; primals_334: "f32[256]"; primals_335: "f32[256]"; primals_336: "i64[]"; primals_337: "f32[256]"; primals_338: "f32[256]"; primals_339: "i64[]"; primals_340: "f32[256]"; primals_341: "f32[256]"; primals_342: "i64[]"; primals_343: "f32[256]"; primals_344: "f32[256]"; primals_345: "i64[]"; primals_346: "f32[256]"; primals_347: "f32[256]"; primals_348: "i64[]"; primals_349: "f32[256]"; primals_350: "f32[256]"; primals_351: "i64[]"; primals_352: "f32[256]"; primals_353: "f32[256]"; primals_354: "i64[]"; primals_355: "f32[256]"; primals_356: "f32[256]"; primals_357: "i64[]"; primals_358: "f32[256]"; primals_359: "f32[256]"; primals_360: "i64[]"; primals_361: "f32[256]"; primals_362: "f32[256]"; primals_363: "i64[]"; primals_364: "f32[256]"; primals_365: "f32[256]"; primals_366: "i64[]"; primals_367: "f32[512]"; primals_368: "f32[512]"; primals_369: "i64[]"; primals_370: "f32[1024]"; primals_371: "f32[1024]"; primals_372: "i64[]"; primals_373: "f32[1024]"; primals_374: "f32[1024]"; primals_375: "i64[]"; primals_376: "f32[512]"; primals_377: "f32[512]"; primals_378: "i64[]"; primals_379: "f32[512]"; primals_380: "f32[512]"; primals_381: "i64[]"; primals_382: "f32[512]"; primals_383: "f32[512]"; primals_384: "i64[]"; primals_385: "f32[512]"; primals_386: "f32[512]"; primals_387: "i64[]"; primals_388: "f32[512]"; primals_389: "f32[512]"; primals_390: "i64[]"; primals_391: "f32[512]"; primals_392: "f32[512]"; primals_393: "i64[]"; primals_394: "f32[512]"; primals_395: "f32[512]"; primals_396: "i64[]"; primals_397: "f32[512]"; primals_398: "f32[512]"; primals_399: "i64[]"; primals_400: "f32[512]"; primals_401: "f32[512]"; primals_402: "i64[]"; primals_403: "f32[1024]"; primals_404: "f32[1024]"; primals_405: "f32[8, 3, 256, 256]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution: "f32[8, 32, 256, 256]" = torch.ops.aten.convolution.default(primals_405, primals_135, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_204, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
    getitem: "f32[1, 32, 1, 1]" = var_mean[0]
    getitem_1: "f32[1, 32, 1, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
    rsqrt: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 32, 256, 256]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
    mul: "f32[8, 32, 256, 256]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    squeeze: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
    squeeze_1: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
    mul_1: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
    mul_2: "f32[32]" = torch.ops.aten.mul.Tensor(primals_205, 0.9)
    add_2: "f32[32]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[32]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000019073522708);  squeeze_2 = None
    mul_4: "f32[32]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[32]" = torch.ops.aten.mul.Tensor(primals_206, 0.9)
    add_3: "f32[32]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_1, -1)
    unsqueeze_1: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 32, 256, 256]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1);  primals_2 = None
    unsqueeze_3: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 32, 256, 256]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt: "b8[8, 32, 256, 256]" = torch.ops.aten.gt.Scalar(add_4, 0)
    mul_7: "f32[8, 32, 256, 256]" = torch.ops.aten.mul.Tensor(add_4, 0.01)
    where: "f32[8, 32, 256, 256]" = torch.ops.aten.where.self(gt, add_4, mul_7);  gt = add_4 = mul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:126, code: x = self.conv(x)
    convolution_1: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(where, primals_136, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_207, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[1, 64, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 64, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
    rsqrt_1: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_1: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
    mul_8: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
    squeeze_4: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_9: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_10: "f32[64]" = torch.ops.aten.mul.Tensor(primals_208, 0.9)
    add_7: "f32[64]" = torch.ops.aten.add.Tensor(mul_9, mul_10);  mul_9 = mul_10 = None
    squeeze_5: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_11: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000076294527394);  squeeze_5 = None
    mul_12: "f32[64]" = torch.ops.aten.mul.Tensor(mul_11, 0.1);  mul_11 = None
    mul_13: "f32[64]" = torch.ops.aten.mul.Tensor(primals_209, 0.9)
    add_8: "f32[64]" = torch.ops.aten.add.Tensor(mul_12, mul_13);  mul_12 = mul_13 = None
    unsqueeze_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1)
    unsqueeze_5: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_14: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_8, unsqueeze_5);  mul_8 = unsqueeze_5 = None
    unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_4, -1);  primals_4 = None
    unsqueeze_7: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_9: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_7);  mul_14 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_1: "b8[8, 64, 128, 128]" = torch.ops.aten.gt.Scalar(add_9, 0)
    mul_15: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(add_9, 0.01)
    where_1: "f32[8, 64, 128, 128]" = torch.ops.aten.where.self(gt_1, add_9, mul_15);  gt_1 = add_9 = mul_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_2: "f32[8, 128, 128, 128]" = torch.ops.aten.convolution.default(where_1, primals_137, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_210, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 128, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 128, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
    rsqrt_2: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_2: "f32[8, 128, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
    mul_16: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_17: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_18: "f32[128]" = torch.ops.aten.mul.Tensor(primals_211, 0.9)
    add_12: "f32[128]" = torch.ops.aten.add.Tensor(mul_17, mul_18);  mul_17 = mul_18 = None
    squeeze_8: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_19: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0000076294527394);  squeeze_8 = None
    mul_20: "f32[128]" = torch.ops.aten.mul.Tensor(mul_19, 0.1);  mul_19 = None
    mul_21: "f32[128]" = torch.ops.aten.mul.Tensor(primals_212, 0.9)
    add_13: "f32[128]" = torch.ops.aten.add.Tensor(mul_20, mul_21);  mul_20 = mul_21 = None
    unsqueeze_8: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_9: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_22: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_9);  mul_16 = unsqueeze_9 = None
    unsqueeze_10: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_11: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_14: "f32[8, 128, 128, 128]" = torch.ops.aten.add.Tensor(mul_22, unsqueeze_11);  mul_22 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_2: "b8[8, 128, 128, 128]" = torch.ops.aten.gt.Scalar(add_14, 0)
    mul_23: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(add_14, 0.01)
    where_2: "f32[8, 128, 128, 128]" = torch.ops.aten.where.self(gt_2, add_14, mul_23);  gt_2 = add_14 = mul_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(where_2, [64, 64], 1)
    getitem_9: "f32[8, 64, 128, 128]" = split_with_sizes_1[1];  split_with_sizes_1 = None
    convolution_3: "f32[8, 32, 128, 128]" = torch.ops.aten.convolution.default(getitem_9, primals_138, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_15: "i64[]" = torch.ops.aten.add.Tensor(primals_213, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
    getitem_10: "f32[1, 32, 1, 1]" = var_mean_3[0]
    getitem_11: "f32[1, 32, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_16: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
    rsqrt_3: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_3: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_11)
    mul_24: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
    squeeze_10: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_25: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_26: "f32[32]" = torch.ops.aten.mul.Tensor(primals_214, 0.9)
    add_17: "f32[32]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
    squeeze_11: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
    mul_27: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0000076294527394);  squeeze_11 = None
    mul_28: "f32[32]" = torch.ops.aten.mul.Tensor(mul_27, 0.1);  mul_27 = None
    mul_29: "f32[32]" = torch.ops.aten.mul.Tensor(primals_215, 0.9)
    add_18: "f32[32]" = torch.ops.aten.add.Tensor(mul_28, mul_29);  mul_28 = mul_29 = None
    unsqueeze_12: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1)
    unsqueeze_13: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_30: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(mul_24, unsqueeze_13);  mul_24 = unsqueeze_13 = None
    unsqueeze_14: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1);  primals_8 = None
    unsqueeze_15: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_19: "f32[8, 32, 128, 128]" = torch.ops.aten.add.Tensor(mul_30, unsqueeze_15);  mul_30 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_3: "b8[8, 32, 128, 128]" = torch.ops.aten.gt.Scalar(add_19, 0)
    mul_31: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(add_19, 0.01)
    where_3: "f32[8, 32, 128, 128]" = torch.ops.aten.where.self(gt_3, add_19, mul_31);  gt_3 = add_19 = mul_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_4: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(where_3, primals_139, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_20: "i64[]" = torch.ops.aten.add.Tensor(primals_216, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
    getitem_12: "f32[1, 64, 1, 1]" = var_mean_4[0]
    getitem_13: "f32[1, 64, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_21: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
    rsqrt_4: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_4: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_13)
    mul_32: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
    squeeze_13: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
    mul_33: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_34: "f32[64]" = torch.ops.aten.mul.Tensor(primals_217, 0.9)
    add_22: "f32[64]" = torch.ops.aten.add.Tensor(mul_33, mul_34);  mul_33 = mul_34 = None
    squeeze_14: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
    mul_35: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000076294527394);  squeeze_14 = None
    mul_36: "f32[64]" = torch.ops.aten.mul.Tensor(mul_35, 0.1);  mul_35 = None
    mul_37: "f32[64]" = torch.ops.aten.mul.Tensor(primals_218, 0.9)
    add_23: "f32[64]" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
    unsqueeze_16: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1)
    unsqueeze_17: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    mul_38: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_32, unsqueeze_17);  mul_32 = unsqueeze_17 = None
    unsqueeze_18: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_10, -1);  primals_10 = None
    unsqueeze_19: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    add_24: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_38, unsqueeze_19);  mul_38 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_4: "b8[8, 64, 128, 128]" = torch.ops.aten.gt.Scalar(add_24, 0)
    mul_39: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(add_24, 0.01)
    where_4: "f32[8, 64, 128, 128]" = torch.ops.aten.where.self(gt_4, add_24, mul_39);  gt_4 = add_24 = mul_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_25: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(where_4, getitem_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_5: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(add_25, primals_140, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_26: "i64[]" = torch.ops.aten.add.Tensor(primals_219, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
    getitem_14: "f32[1, 64, 1, 1]" = var_mean_5[0]
    getitem_15: "f32[1, 64, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_27: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
    rsqrt_5: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    sub_5: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_15)
    mul_40: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_15: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
    squeeze_16: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    mul_41: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_42: "f32[64]" = torch.ops.aten.mul.Tensor(primals_220, 0.9)
    add_28: "f32[64]" = torch.ops.aten.add.Tensor(mul_41, mul_42);  mul_41 = mul_42 = None
    squeeze_17: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
    mul_43: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000076294527394);  squeeze_17 = None
    mul_44: "f32[64]" = torch.ops.aten.mul.Tensor(mul_43, 0.1);  mul_43 = None
    mul_45: "f32[64]" = torch.ops.aten.mul.Tensor(primals_221, 0.9)
    add_29: "f32[64]" = torch.ops.aten.add.Tensor(mul_44, mul_45);  mul_44 = mul_45 = None
    unsqueeze_20: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_21: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_46: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_40, unsqueeze_21);  mul_40 = unsqueeze_21 = None
    unsqueeze_22: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_23: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_30: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_46, unsqueeze_23);  mul_46 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_5: "b8[8, 64, 128, 128]" = torch.ops.aten.gt.Scalar(add_30, 0)
    mul_47: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(add_30, 0.01)
    where_5: "f32[8, 64, 128, 128]" = torch.ops.aten.where.self(gt_5, add_30, mul_47);  gt_5 = add_30 = mul_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:339, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
    split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(where_2, [64, 64], 1)
    getitem_16: "f32[8, 64, 128, 128]" = split_with_sizes_2[0];  split_with_sizes_2 = None
    cat: "f32[8, 128, 128, 128]" = torch.ops.aten.cat.default([getitem_16, where_5], 1);  getitem_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_6: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(cat, primals_141, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_31: "i64[]" = torch.ops.aten.add.Tensor(primals_222, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_6 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
    getitem_18: "f32[1, 64, 1, 1]" = var_mean_6[0]
    getitem_19: "f32[1, 64, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_32: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
    rsqrt_6: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_6: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_19)
    mul_48: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_18: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
    squeeze_19: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
    mul_49: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_50: "f32[64]" = torch.ops.aten.mul.Tensor(primals_223, 0.9)
    add_33: "f32[64]" = torch.ops.aten.add.Tensor(mul_49, mul_50);  mul_49 = mul_50 = None
    squeeze_20: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
    mul_51: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0000076294527394);  squeeze_20 = None
    mul_52: "f32[64]" = torch.ops.aten.mul.Tensor(mul_51, 0.1);  mul_51 = None
    mul_53: "f32[64]" = torch.ops.aten.mul.Tensor(primals_224, 0.9)
    add_34: "f32[64]" = torch.ops.aten.add.Tensor(mul_52, mul_53);  mul_52 = mul_53 = None
    unsqueeze_24: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1)
    unsqueeze_25: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    mul_54: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_48, unsqueeze_25);  mul_48 = unsqueeze_25 = None
    unsqueeze_26: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1);  primals_14 = None
    unsqueeze_27: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    add_35: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_54, unsqueeze_27);  mul_54 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_6: "b8[8, 64, 128, 128]" = torch.ops.aten.gt.Scalar(add_35, 0)
    mul_55: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(add_35, 0.01)
    where_6: "f32[8, 64, 128, 128]" = torch.ops.aten.where.self(gt_6, add_35, mul_55);  gt_6 = add_35 = mul_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:126, code: x = self.conv(x)
    convolution_7: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(where_6, primals_142, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_36: "i64[]" = torch.ops.aten.add.Tensor(primals_225, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_7 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[1, 128, 1, 1]" = var_mean_7[0]
    getitem_21: "f32[1, 128, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_37: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
    rsqrt_7: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_7: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_21)
    mul_56: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_21: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
    squeeze_22: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
    mul_57: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_58: "f32[128]" = torch.ops.aten.mul.Tensor(primals_226, 0.9)
    add_38: "f32[128]" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
    squeeze_23: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
    mul_59: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.000030518509476);  squeeze_23 = None
    mul_60: "f32[128]" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
    mul_61: "f32[128]" = torch.ops.aten.mul.Tensor(primals_227, 0.9)
    add_39: "f32[128]" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
    unsqueeze_28: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1)
    unsqueeze_29: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_62: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_29);  mul_56 = unsqueeze_29 = None
    unsqueeze_30: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1);  primals_16 = None
    unsqueeze_31: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_40: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_31);  mul_62 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_7: "b8[8, 128, 64, 64]" = torch.ops.aten.gt.Scalar(add_40, 0)
    mul_63: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(add_40, 0.01)
    where_7: "f32[8, 128, 64, 64]" = torch.ops.aten.where.self(gt_7, add_40, mul_63);  gt_7 = add_40 = mul_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_8: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(where_7, primals_143, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_41: "i64[]" = torch.ops.aten.add.Tensor(primals_228, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_8 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[1, 128, 1, 1]" = var_mean_8[0]
    getitem_23: "f32[1, 128, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_42: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
    rsqrt_8: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_8: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_23)
    mul_64: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_24: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
    squeeze_25: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
    mul_65: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_66: "f32[128]" = torch.ops.aten.mul.Tensor(primals_229, 0.9)
    add_43: "f32[128]" = torch.ops.aten.add.Tensor(mul_65, mul_66);  mul_65 = mul_66 = None
    squeeze_26: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
    mul_67: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.000030518509476);  squeeze_26 = None
    mul_68: "f32[128]" = torch.ops.aten.mul.Tensor(mul_67, 0.1);  mul_67 = None
    mul_69: "f32[128]" = torch.ops.aten.mul.Tensor(primals_230, 0.9)
    add_44: "f32[128]" = torch.ops.aten.add.Tensor(mul_68, mul_69);  mul_68 = mul_69 = None
    unsqueeze_32: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_33: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    mul_70: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_64, unsqueeze_33);  mul_64 = unsqueeze_33 = None
    unsqueeze_34: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_35: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    add_45: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_70, unsqueeze_35);  mul_70 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_8: "b8[8, 128, 64, 64]" = torch.ops.aten.gt.Scalar(add_45, 0)
    mul_71: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(add_45, 0.01)
    where_8: "f32[8, 128, 64, 64]" = torch.ops.aten.where.self(gt_8, add_45, mul_71);  gt_8 = add_45 = mul_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    split_with_sizes_4 = torch.ops.aten.split_with_sizes.default(where_8, [64, 64], 1)
    getitem_27: "f32[8, 64, 64, 64]" = split_with_sizes_4[1];  split_with_sizes_4 = None
    convolution_9: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(getitem_27, primals_144, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_46: "i64[]" = torch.ops.aten.add.Tensor(primals_231, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_9 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
    getitem_28: "f32[1, 64, 1, 1]" = var_mean_9[0]
    getitem_29: "f32[1, 64, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_47: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
    rsqrt_9: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_9: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_29)
    mul_72: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_27: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
    squeeze_28: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_73: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_74: "f32[64]" = torch.ops.aten.mul.Tensor(primals_232, 0.9)
    add_48: "f32[64]" = torch.ops.aten.add.Tensor(mul_73, mul_74);  mul_73 = mul_74 = None
    squeeze_29: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
    mul_75: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.000030518509476);  squeeze_29 = None
    mul_76: "f32[64]" = torch.ops.aten.mul.Tensor(mul_75, 0.1);  mul_75 = None
    mul_77: "f32[64]" = torch.ops.aten.mul.Tensor(primals_233, 0.9)
    add_49: "f32[64]" = torch.ops.aten.add.Tensor(mul_76, mul_77);  mul_76 = mul_77 = None
    unsqueeze_36: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
    unsqueeze_37: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_78: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_72, unsqueeze_37);  mul_72 = unsqueeze_37 = None
    unsqueeze_38: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
    unsqueeze_39: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_50: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_78, unsqueeze_39);  mul_78 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_9: "b8[8, 64, 64, 64]" = torch.ops.aten.gt.Scalar(add_50, 0)
    mul_79: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_50, 0.01)
    where_9: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(gt_9, add_50, mul_79);  gt_9 = add_50 = mul_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_10: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(where_9, primals_145, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_51: "i64[]" = torch.ops.aten.add.Tensor(primals_234, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_10 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
    getitem_30: "f32[1, 64, 1, 1]" = var_mean_10[0]
    getitem_31: "f32[1, 64, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_52: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
    rsqrt_10: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_10: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_31)
    mul_80: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_30: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
    squeeze_31: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
    mul_81: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_82: "f32[64]" = torch.ops.aten.mul.Tensor(primals_235, 0.9)
    add_53: "f32[64]" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
    squeeze_32: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
    mul_83: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.000030518509476);  squeeze_32 = None
    mul_84: "f32[64]" = torch.ops.aten.mul.Tensor(mul_83, 0.1);  mul_83 = None
    mul_85: "f32[64]" = torch.ops.aten.mul.Tensor(primals_236, 0.9)
    add_54: "f32[64]" = torch.ops.aten.add.Tensor(mul_84, mul_85);  mul_84 = mul_85 = None
    unsqueeze_40: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1)
    unsqueeze_41: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    mul_86: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_80, unsqueeze_41);  mul_80 = unsqueeze_41 = None
    unsqueeze_42: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1);  primals_22 = None
    unsqueeze_43: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    add_55: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_86, unsqueeze_43);  mul_86 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_10: "b8[8, 64, 64, 64]" = torch.ops.aten.gt.Scalar(add_55, 0)
    mul_87: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_55, 0.01)
    where_10: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(gt_10, add_55, mul_87);  gt_10 = add_55 = mul_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_56: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(where_10, getitem_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_11: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(add_56, primals_146, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_57: "i64[]" = torch.ops.aten.add.Tensor(primals_237, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_11 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
    getitem_32: "f32[1, 64, 1, 1]" = var_mean_11[0]
    getitem_33: "f32[1, 64, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_58: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
    rsqrt_11: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_11: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_33)
    mul_88: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    squeeze_33: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
    squeeze_34: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
    mul_89: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_90: "f32[64]" = torch.ops.aten.mul.Tensor(primals_238, 0.9)
    add_59: "f32[64]" = torch.ops.aten.add.Tensor(mul_89, mul_90);  mul_89 = mul_90 = None
    squeeze_35: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
    mul_91: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.000030518509476);  squeeze_35 = None
    mul_92: "f32[64]" = torch.ops.aten.mul.Tensor(mul_91, 0.1);  mul_91 = None
    mul_93: "f32[64]" = torch.ops.aten.mul.Tensor(primals_239, 0.9)
    add_60: "f32[64]" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
    unsqueeze_44: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_45: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_94: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_88, unsqueeze_45);  mul_88 = unsqueeze_45 = None
    unsqueeze_46: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_47: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_61: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_94, unsqueeze_47);  mul_94 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_11: "b8[8, 64, 64, 64]" = torch.ops.aten.gt.Scalar(add_61, 0)
    mul_95: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_61, 0.01)
    where_11: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(gt_11, add_61, mul_95);  gt_11 = add_61 = mul_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_12: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(where_11, primals_147, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_62: "i64[]" = torch.ops.aten.add.Tensor(primals_240, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_12 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
    getitem_34: "f32[1, 64, 1, 1]" = var_mean_12[0]
    getitem_35: "f32[1, 64, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_63: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
    rsqrt_12: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_12: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_35)
    mul_96: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_36: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
    squeeze_37: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
    mul_97: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_98: "f32[64]" = torch.ops.aten.mul.Tensor(primals_241, 0.9)
    add_64: "f32[64]" = torch.ops.aten.add.Tensor(mul_97, mul_98);  mul_97 = mul_98 = None
    squeeze_38: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
    mul_99: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.000030518509476);  squeeze_38 = None
    mul_100: "f32[64]" = torch.ops.aten.mul.Tensor(mul_99, 0.1);  mul_99 = None
    mul_101: "f32[64]" = torch.ops.aten.mul.Tensor(primals_242, 0.9)
    add_65: "f32[64]" = torch.ops.aten.add.Tensor(mul_100, mul_101);  mul_100 = mul_101 = None
    unsqueeze_48: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1)
    unsqueeze_49: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    mul_102: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_96, unsqueeze_49);  mul_96 = unsqueeze_49 = None
    unsqueeze_50: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1);  primals_26 = None
    unsqueeze_51: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    add_66: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_102, unsqueeze_51);  mul_102 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_12: "b8[8, 64, 64, 64]" = torch.ops.aten.gt.Scalar(add_66, 0)
    mul_103: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_66, 0.01)
    where_12: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(gt_12, add_66, mul_103);  gt_12 = add_66 = mul_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_67: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(where_12, add_56)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_13: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(add_67, primals_148, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_68: "i64[]" = torch.ops.aten.add.Tensor(primals_243, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_13 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
    getitem_36: "f32[1, 64, 1, 1]" = var_mean_13[0]
    getitem_37: "f32[1, 64, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_69: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
    rsqrt_13: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
    sub_13: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_37)
    mul_104: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_39: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
    squeeze_40: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_105: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_106: "f32[64]" = torch.ops.aten.mul.Tensor(primals_244, 0.9)
    add_70: "f32[64]" = torch.ops.aten.add.Tensor(mul_105, mul_106);  mul_105 = mul_106 = None
    squeeze_41: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
    mul_107: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.000030518509476);  squeeze_41 = None
    mul_108: "f32[64]" = torch.ops.aten.mul.Tensor(mul_107, 0.1);  mul_107 = None
    mul_109: "f32[64]" = torch.ops.aten.mul.Tensor(primals_245, 0.9)
    add_71: "f32[64]" = torch.ops.aten.add.Tensor(mul_108, mul_109);  mul_108 = mul_109 = None
    unsqueeze_52: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1)
    unsqueeze_53: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_110: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_104, unsqueeze_53);  mul_104 = unsqueeze_53 = None
    unsqueeze_54: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1);  primals_28 = None
    unsqueeze_55: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_72: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_110, unsqueeze_55);  mul_110 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_13: "b8[8, 64, 64, 64]" = torch.ops.aten.gt.Scalar(add_72, 0)
    mul_111: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_72, 0.01)
    where_13: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(gt_13, add_72, mul_111);  gt_13 = add_72 = mul_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:339, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
    split_with_sizes_5 = torch.ops.aten.split_with_sizes.default(where_8, [64, 64], 1)
    getitem_38: "f32[8, 64, 64, 64]" = split_with_sizes_5[0];  split_with_sizes_5 = None
    cat_1: "f32[8, 128, 64, 64]" = torch.ops.aten.cat.default([getitem_38, where_13], 1);  getitem_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_14: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(cat_1, primals_149, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_73: "i64[]" = torch.ops.aten.add.Tensor(primals_246, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_14 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
    getitem_40: "f32[1, 128, 1, 1]" = var_mean_14[0]
    getitem_41: "f32[1, 128, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_74: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
    rsqrt_14: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_14: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_41)
    mul_112: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_42: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
    squeeze_43: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
    mul_113: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_114: "f32[128]" = torch.ops.aten.mul.Tensor(primals_247, 0.9)
    add_75: "f32[128]" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
    squeeze_44: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
    mul_115: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.000030518509476);  squeeze_44 = None
    mul_116: "f32[128]" = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
    mul_117: "f32[128]" = torch.ops.aten.mul.Tensor(primals_248, 0.9)
    add_76: "f32[128]" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
    unsqueeze_56: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_57: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    mul_118: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_57);  mul_112 = unsqueeze_57 = None
    unsqueeze_58: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_59: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    add_77: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_59);  mul_118 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_14: "b8[8, 128, 64, 64]" = torch.ops.aten.gt.Scalar(add_77, 0)
    mul_119: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(add_77, 0.01)
    where_14: "f32[8, 128, 64, 64]" = torch.ops.aten.where.self(gt_14, add_77, mul_119);  gt_14 = add_77 = mul_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:126, code: x = self.conv(x)
    convolution_15: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(where_14, primals_150, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_78: "i64[]" = torch.ops.aten.add.Tensor(primals_249, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_15 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
    getitem_42: "f32[1, 256, 1, 1]" = var_mean_15[0]
    getitem_43: "f32[1, 256, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_79: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
    rsqrt_15: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
    sub_15: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_43)
    mul_120: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    squeeze_45: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
    squeeze_46: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
    mul_121: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_122: "f32[256]" = torch.ops.aten.mul.Tensor(primals_250, 0.9)
    add_80: "f32[256]" = torch.ops.aten.add.Tensor(mul_121, mul_122);  mul_121 = mul_122 = None
    squeeze_47: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
    mul_123: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0001220852154804);  squeeze_47 = None
    mul_124: "f32[256]" = torch.ops.aten.mul.Tensor(mul_123, 0.1);  mul_123 = None
    mul_125: "f32[256]" = torch.ops.aten.mul.Tensor(primals_251, 0.9)
    add_81: "f32[256]" = torch.ops.aten.add.Tensor(mul_124, mul_125);  mul_124 = mul_125 = None
    unsqueeze_60: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1)
    unsqueeze_61: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_126: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_120, unsqueeze_61);  mul_120 = unsqueeze_61 = None
    unsqueeze_62: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1);  primals_32 = None
    unsqueeze_63: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_82: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_126, unsqueeze_63);  mul_126 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_15: "b8[8, 256, 32, 32]" = torch.ops.aten.gt.Scalar(add_82, 0)
    mul_127: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(add_82, 0.01)
    where_15: "f32[8, 256, 32, 32]" = torch.ops.aten.where.self(gt_15, add_82, mul_127);  gt_15 = add_82 = mul_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_16: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(where_15, primals_151, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_83: "i64[]" = torch.ops.aten.add.Tensor(primals_252, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_16 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
    getitem_44: "f32[1, 256, 1, 1]" = var_mean_16[0]
    getitem_45: "f32[1, 256, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_84: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05)
    rsqrt_16: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_16: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_45)
    mul_128: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_48: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
    squeeze_49: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
    mul_129: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_130: "f32[256]" = torch.ops.aten.mul.Tensor(primals_253, 0.9)
    add_85: "f32[256]" = torch.ops.aten.add.Tensor(mul_129, mul_130);  mul_129 = mul_130 = None
    squeeze_50: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
    mul_131: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0001220852154804);  squeeze_50 = None
    mul_132: "f32[256]" = torch.ops.aten.mul.Tensor(mul_131, 0.1);  mul_131 = None
    mul_133: "f32[256]" = torch.ops.aten.mul.Tensor(primals_254, 0.9)
    add_86: "f32[256]" = torch.ops.aten.add.Tensor(mul_132, mul_133);  mul_132 = mul_133 = None
    unsqueeze_64: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1)
    unsqueeze_65: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    mul_134: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_128, unsqueeze_65);  mul_128 = unsqueeze_65 = None
    unsqueeze_66: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1);  primals_34 = None
    unsqueeze_67: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    add_87: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_134, unsqueeze_67);  mul_134 = unsqueeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_16: "b8[8, 256, 32, 32]" = torch.ops.aten.gt.Scalar(add_87, 0)
    mul_135: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(add_87, 0.01)
    where_16: "f32[8, 256, 32, 32]" = torch.ops.aten.where.self(gt_16, add_87, mul_135);  gt_16 = add_87 = mul_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    split_with_sizes_7 = torch.ops.aten.split_with_sizes.default(where_16, [128, 128], 1)
    getitem_49: "f32[8, 128, 32, 32]" = split_with_sizes_7[1];  split_with_sizes_7 = None
    convolution_17: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(getitem_49, primals_152, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_88: "i64[]" = torch.ops.aten.add.Tensor(primals_255, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_17 = torch.ops.aten.var_mean.correction(convolution_17, [0, 2, 3], correction = 0, keepdim = True)
    getitem_50: "f32[1, 128, 1, 1]" = var_mean_17[0]
    getitem_51: "f32[1, 128, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_89: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
    rsqrt_17: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
    sub_17: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_17, getitem_51)
    mul_136: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    squeeze_51: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
    squeeze_52: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
    mul_137: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_138: "f32[128]" = torch.ops.aten.mul.Tensor(primals_256, 0.9)
    add_90: "f32[128]" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
    squeeze_53: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
    mul_139: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0001220852154804);  squeeze_53 = None
    mul_140: "f32[128]" = torch.ops.aten.mul.Tensor(mul_139, 0.1);  mul_139 = None
    mul_141: "f32[128]" = torch.ops.aten.mul.Tensor(primals_257, 0.9)
    add_91: "f32[128]" = torch.ops.aten.add.Tensor(mul_140, mul_141);  mul_140 = mul_141 = None
    unsqueeze_68: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_69: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_142: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_69);  mul_136 = unsqueeze_69 = None
    unsqueeze_70: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_71: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_92: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_142, unsqueeze_71);  mul_142 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_17: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_92, 0)
    mul_143: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_92, 0.01)
    where_17: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_17, add_92, mul_143);  gt_17 = add_92 = mul_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_18: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(where_17, primals_153, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_93: "i64[]" = torch.ops.aten.add.Tensor(primals_258, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_18 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
    getitem_52: "f32[1, 128, 1, 1]" = var_mean_18[0]
    getitem_53: "f32[1, 128, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_94: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05)
    rsqrt_18: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    sub_18: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_53)
    mul_144: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    squeeze_54: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
    squeeze_55: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
    mul_145: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_146: "f32[128]" = torch.ops.aten.mul.Tensor(primals_259, 0.9)
    add_95: "f32[128]" = torch.ops.aten.add.Tensor(mul_145, mul_146);  mul_145 = mul_146 = None
    squeeze_56: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
    mul_147: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0001220852154804);  squeeze_56 = None
    mul_148: "f32[128]" = torch.ops.aten.mul.Tensor(mul_147, 0.1);  mul_147 = None
    mul_149: "f32[128]" = torch.ops.aten.mul.Tensor(primals_260, 0.9)
    add_96: "f32[128]" = torch.ops.aten.add.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
    unsqueeze_72: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1)
    unsqueeze_73: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    mul_150: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_144, unsqueeze_73);  mul_144 = unsqueeze_73 = None
    unsqueeze_74: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1);  primals_38 = None
    unsqueeze_75: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    add_97: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_150, unsqueeze_75);  mul_150 = unsqueeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_18: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_97, 0)
    mul_151: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_97, 0.01)
    where_18: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_18, add_97, mul_151);  gt_18 = add_97 = mul_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_98: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(where_18, getitem_49)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_19: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(add_98, primals_154, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_99: "i64[]" = torch.ops.aten.add.Tensor(primals_261, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_19 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
    getitem_54: "f32[1, 128, 1, 1]" = var_mean_19[0]
    getitem_55: "f32[1, 128, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_100: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05)
    rsqrt_19: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
    sub_19: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_55)
    mul_152: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    squeeze_57: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
    squeeze_58: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
    mul_153: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_154: "f32[128]" = torch.ops.aten.mul.Tensor(primals_262, 0.9)
    add_101: "f32[128]" = torch.ops.aten.add.Tensor(mul_153, mul_154);  mul_153 = mul_154 = None
    squeeze_59: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
    mul_155: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0001220852154804);  squeeze_59 = None
    mul_156: "f32[128]" = torch.ops.aten.mul.Tensor(mul_155, 0.1);  mul_155 = None
    mul_157: "f32[128]" = torch.ops.aten.mul.Tensor(primals_263, 0.9)
    add_102: "f32[128]" = torch.ops.aten.add.Tensor(mul_156, mul_157);  mul_156 = mul_157 = None
    unsqueeze_76: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1)
    unsqueeze_77: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_158: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_152, unsqueeze_77);  mul_152 = unsqueeze_77 = None
    unsqueeze_78: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1);  primals_40 = None
    unsqueeze_79: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_103: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_158, unsqueeze_79);  mul_158 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_19: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_103, 0)
    mul_159: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_103, 0.01)
    where_19: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_19, add_103, mul_159);  gt_19 = add_103 = mul_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_20: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(where_19, primals_155, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_104: "i64[]" = torch.ops.aten.add.Tensor(primals_264, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_20 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
    getitem_56: "f32[1, 128, 1, 1]" = var_mean_20[0]
    getitem_57: "f32[1, 128, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_105: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05)
    rsqrt_20: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    sub_20: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_57)
    mul_160: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    squeeze_60: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
    squeeze_61: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
    mul_161: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_162: "f32[128]" = torch.ops.aten.mul.Tensor(primals_265, 0.9)
    add_106: "f32[128]" = torch.ops.aten.add.Tensor(mul_161, mul_162);  mul_161 = mul_162 = None
    squeeze_62: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
    mul_163: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0001220852154804);  squeeze_62 = None
    mul_164: "f32[128]" = torch.ops.aten.mul.Tensor(mul_163, 0.1);  mul_163 = None
    mul_165: "f32[128]" = torch.ops.aten.mul.Tensor(primals_266, 0.9)
    add_107: "f32[128]" = torch.ops.aten.add.Tensor(mul_164, mul_165);  mul_164 = mul_165 = None
    unsqueeze_80: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_81: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    mul_166: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_160, unsqueeze_81);  mul_160 = unsqueeze_81 = None
    unsqueeze_82: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_83: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    add_108: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_166, unsqueeze_83);  mul_166 = unsqueeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_20: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_108, 0)
    mul_167: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_108, 0.01)
    where_20: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_20, add_108, mul_167);  gt_20 = add_108 = mul_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_109: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(where_20, add_98)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_21: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(add_109, primals_156, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_110: "i64[]" = torch.ops.aten.add.Tensor(primals_267, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_21 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
    getitem_58: "f32[1, 128, 1, 1]" = var_mean_21[0]
    getitem_59: "f32[1, 128, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_111: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
    rsqrt_21: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
    sub_21: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_59)
    mul_168: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    squeeze_63: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
    squeeze_64: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
    mul_169: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_170: "f32[128]" = torch.ops.aten.mul.Tensor(primals_268, 0.9)
    add_112: "f32[128]" = torch.ops.aten.add.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
    squeeze_65: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
    mul_171: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0001220852154804);  squeeze_65 = None
    mul_172: "f32[128]" = torch.ops.aten.mul.Tensor(mul_171, 0.1);  mul_171 = None
    mul_173: "f32[128]" = torch.ops.aten.mul.Tensor(primals_269, 0.9)
    add_113: "f32[128]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
    unsqueeze_84: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1)
    unsqueeze_85: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_174: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_168, unsqueeze_85);  mul_168 = unsqueeze_85 = None
    unsqueeze_86: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1);  primals_44 = None
    unsqueeze_87: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_114: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_174, unsqueeze_87);  mul_174 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_21: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_114, 0)
    mul_175: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_114, 0.01)
    where_21: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_21, add_114, mul_175);  gt_21 = add_114 = mul_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_22: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(where_21, primals_157, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_115: "i64[]" = torch.ops.aten.add.Tensor(primals_270, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_22 = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
    getitem_60: "f32[1, 128, 1, 1]" = var_mean_22[0]
    getitem_61: "f32[1, 128, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_116: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
    rsqrt_22: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
    sub_22: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_22, getitem_61)
    mul_176: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    squeeze_66: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
    squeeze_67: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
    mul_177: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_178: "f32[128]" = torch.ops.aten.mul.Tensor(primals_271, 0.9)
    add_117: "f32[128]" = torch.ops.aten.add.Tensor(mul_177, mul_178);  mul_177 = mul_178 = None
    squeeze_68: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
    mul_179: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0001220852154804);  squeeze_68 = None
    mul_180: "f32[128]" = torch.ops.aten.mul.Tensor(mul_179, 0.1);  mul_179 = None
    mul_181: "f32[128]" = torch.ops.aten.mul.Tensor(primals_272, 0.9)
    add_118: "f32[128]" = torch.ops.aten.add.Tensor(mul_180, mul_181);  mul_180 = mul_181 = None
    unsqueeze_88: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1)
    unsqueeze_89: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    mul_182: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_176, unsqueeze_89);  mul_176 = unsqueeze_89 = None
    unsqueeze_90: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1);  primals_46 = None
    unsqueeze_91: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    add_119: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_182, unsqueeze_91);  mul_182 = unsqueeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_22: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_119, 0)
    mul_183: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_119, 0.01)
    where_22: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_22, add_119, mul_183);  gt_22 = add_119 = mul_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_120: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(where_22, add_109)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_23: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(add_120, primals_158, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_121: "i64[]" = torch.ops.aten.add.Tensor(primals_273, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_23 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
    getitem_62: "f32[1, 128, 1, 1]" = var_mean_23[0]
    getitem_63: "f32[1, 128, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_122: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
    rsqrt_23: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    sub_23: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_63)
    mul_184: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    squeeze_69: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
    squeeze_70: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
    mul_185: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
    mul_186: "f32[128]" = torch.ops.aten.mul.Tensor(primals_274, 0.9)
    add_123: "f32[128]" = torch.ops.aten.add.Tensor(mul_185, mul_186);  mul_185 = mul_186 = None
    squeeze_71: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
    mul_187: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0001220852154804);  squeeze_71 = None
    mul_188: "f32[128]" = torch.ops.aten.mul.Tensor(mul_187, 0.1);  mul_187 = None
    mul_189: "f32[128]" = torch.ops.aten.mul.Tensor(primals_275, 0.9)
    add_124: "f32[128]" = torch.ops.aten.add.Tensor(mul_188, mul_189);  mul_188 = mul_189 = None
    unsqueeze_92: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_93: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_190: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_184, unsqueeze_93);  mul_184 = unsqueeze_93 = None
    unsqueeze_94: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_95: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_125: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_190, unsqueeze_95);  mul_190 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_23: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_125, 0)
    mul_191: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_125, 0.01)
    where_23: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_23, add_125, mul_191);  gt_23 = add_125 = mul_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_24: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(where_23, primals_159, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_126: "i64[]" = torch.ops.aten.add.Tensor(primals_276, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_24 = torch.ops.aten.var_mean.correction(convolution_24, [0, 2, 3], correction = 0, keepdim = True)
    getitem_64: "f32[1, 128, 1, 1]" = var_mean_24[0]
    getitem_65: "f32[1, 128, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_127: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
    rsqrt_24: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
    sub_24: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_24, getitem_65)
    mul_192: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    squeeze_72: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
    squeeze_73: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
    mul_193: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_194: "f32[128]" = torch.ops.aten.mul.Tensor(primals_277, 0.9)
    add_128: "f32[128]" = torch.ops.aten.add.Tensor(mul_193, mul_194);  mul_193 = mul_194 = None
    squeeze_74: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
    mul_195: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0001220852154804);  squeeze_74 = None
    mul_196: "f32[128]" = torch.ops.aten.mul.Tensor(mul_195, 0.1);  mul_195 = None
    mul_197: "f32[128]" = torch.ops.aten.mul.Tensor(primals_278, 0.9)
    add_129: "f32[128]" = torch.ops.aten.add.Tensor(mul_196, mul_197);  mul_196 = mul_197 = None
    unsqueeze_96: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1)
    unsqueeze_97: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    mul_198: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_192, unsqueeze_97);  mul_192 = unsqueeze_97 = None
    unsqueeze_98: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1);  primals_50 = None
    unsqueeze_99: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    add_130: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_198, unsqueeze_99);  mul_198 = unsqueeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_24: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_130, 0)
    mul_199: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_130, 0.01)
    where_24: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_24, add_130, mul_199);  gt_24 = add_130 = mul_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_131: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(where_24, add_120)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_25: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(add_131, primals_160, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_132: "i64[]" = torch.ops.aten.add.Tensor(primals_279, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_25 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
    getitem_66: "f32[1, 128, 1, 1]" = var_mean_25[0]
    getitem_67: "f32[1, 128, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_133: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
    rsqrt_25: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
    sub_25: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_67)
    mul_200: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    squeeze_75: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
    squeeze_76: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
    mul_201: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_202: "f32[128]" = torch.ops.aten.mul.Tensor(primals_280, 0.9)
    add_134: "f32[128]" = torch.ops.aten.add.Tensor(mul_201, mul_202);  mul_201 = mul_202 = None
    squeeze_77: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
    mul_203: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0001220852154804);  squeeze_77 = None
    mul_204: "f32[128]" = torch.ops.aten.mul.Tensor(mul_203, 0.1);  mul_203 = None
    mul_205: "f32[128]" = torch.ops.aten.mul.Tensor(primals_281, 0.9)
    add_135: "f32[128]" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
    unsqueeze_100: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1)
    unsqueeze_101: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_206: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_200, unsqueeze_101);  mul_200 = unsqueeze_101 = None
    unsqueeze_102: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1);  primals_52 = None
    unsqueeze_103: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_136: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_206, unsqueeze_103);  mul_206 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_25: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_136, 0)
    mul_207: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_136, 0.01)
    where_25: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_25, add_136, mul_207);  gt_25 = add_136 = mul_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_26: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(where_25, primals_161, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_137: "i64[]" = torch.ops.aten.add.Tensor(primals_282, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_26 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
    getitem_68: "f32[1, 128, 1, 1]" = var_mean_26[0]
    getitem_69: "f32[1, 128, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_138: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05)
    rsqrt_26: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
    sub_26: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_69)
    mul_208: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    squeeze_78: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
    squeeze_79: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
    mul_209: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_210: "f32[128]" = torch.ops.aten.mul.Tensor(primals_283, 0.9)
    add_139: "f32[128]" = torch.ops.aten.add.Tensor(mul_209, mul_210);  mul_209 = mul_210 = None
    squeeze_80: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
    mul_211: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0001220852154804);  squeeze_80 = None
    mul_212: "f32[128]" = torch.ops.aten.mul.Tensor(mul_211, 0.1);  mul_211 = None
    mul_213: "f32[128]" = torch.ops.aten.mul.Tensor(primals_284, 0.9)
    add_140: "f32[128]" = torch.ops.aten.add.Tensor(mul_212, mul_213);  mul_212 = mul_213 = None
    unsqueeze_104: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_105: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    mul_214: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_208, unsqueeze_105);  mul_208 = unsqueeze_105 = None
    unsqueeze_106: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_107: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    add_141: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_214, unsqueeze_107);  mul_214 = unsqueeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_26: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_141, 0)
    mul_215: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_141, 0.01)
    where_26: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_26, add_141, mul_215);  gt_26 = add_141 = mul_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_142: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(where_26, add_131)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_27: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(add_142, primals_162, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_143: "i64[]" = torch.ops.aten.add.Tensor(primals_285, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_27 = torch.ops.aten.var_mean.correction(convolution_27, [0, 2, 3], correction = 0, keepdim = True)
    getitem_70: "f32[1, 128, 1, 1]" = var_mean_27[0]
    getitem_71: "f32[1, 128, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_144: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05)
    rsqrt_27: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    sub_27: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_27, getitem_71)
    mul_216: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    squeeze_81: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
    squeeze_82: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
    mul_217: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
    mul_218: "f32[128]" = torch.ops.aten.mul.Tensor(primals_286, 0.9)
    add_145: "f32[128]" = torch.ops.aten.add.Tensor(mul_217, mul_218);  mul_217 = mul_218 = None
    squeeze_83: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
    mul_219: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0001220852154804);  squeeze_83 = None
    mul_220: "f32[128]" = torch.ops.aten.mul.Tensor(mul_219, 0.1);  mul_219 = None
    mul_221: "f32[128]" = torch.ops.aten.mul.Tensor(primals_287, 0.9)
    add_146: "f32[128]" = torch.ops.aten.add.Tensor(mul_220, mul_221);  mul_220 = mul_221 = None
    unsqueeze_108: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1)
    unsqueeze_109: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_222: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_216, unsqueeze_109);  mul_216 = unsqueeze_109 = None
    unsqueeze_110: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1);  primals_56 = None
    unsqueeze_111: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_147: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_222, unsqueeze_111);  mul_222 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_27: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_147, 0)
    mul_223: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_147, 0.01)
    where_27: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_27, add_147, mul_223);  gt_27 = add_147 = mul_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_28: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(where_27, primals_163, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_148: "i64[]" = torch.ops.aten.add.Tensor(primals_288, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_28 = torch.ops.aten.var_mean.correction(convolution_28, [0, 2, 3], correction = 0, keepdim = True)
    getitem_72: "f32[1, 128, 1, 1]" = var_mean_28[0]
    getitem_73: "f32[1, 128, 1, 1]" = var_mean_28[1];  var_mean_28 = None
    add_149: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05)
    rsqrt_28: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
    sub_28: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_28, getitem_73)
    mul_224: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    squeeze_84: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
    squeeze_85: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
    mul_225: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
    mul_226: "f32[128]" = torch.ops.aten.mul.Tensor(primals_289, 0.9)
    add_150: "f32[128]" = torch.ops.aten.add.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
    squeeze_86: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
    mul_227: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0001220852154804);  squeeze_86 = None
    mul_228: "f32[128]" = torch.ops.aten.mul.Tensor(mul_227, 0.1);  mul_227 = None
    mul_229: "f32[128]" = torch.ops.aten.mul.Tensor(primals_290, 0.9)
    add_151: "f32[128]" = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
    unsqueeze_112: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1)
    unsqueeze_113: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    mul_230: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_224, unsqueeze_113);  mul_224 = unsqueeze_113 = None
    unsqueeze_114: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_58, -1);  primals_58 = None
    unsqueeze_115: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    add_152: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_115);  mul_230 = unsqueeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_28: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_152, 0)
    mul_231: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_152, 0.01)
    where_28: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_28, add_152, mul_231);  gt_28 = add_152 = mul_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_153: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(where_28, add_142)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_29: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(add_153, primals_164, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_154: "i64[]" = torch.ops.aten.add.Tensor(primals_291, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_29 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
    getitem_74: "f32[1, 128, 1, 1]" = var_mean_29[0]
    getitem_75: "f32[1, 128, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_155: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05)
    rsqrt_29: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
    sub_29: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_29, getitem_75)
    mul_232: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    squeeze_87: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
    squeeze_88: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
    mul_233: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
    mul_234: "f32[128]" = torch.ops.aten.mul.Tensor(primals_292, 0.9)
    add_156: "f32[128]" = torch.ops.aten.add.Tensor(mul_233, mul_234);  mul_233 = mul_234 = None
    squeeze_89: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
    mul_235: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0001220852154804);  squeeze_89 = None
    mul_236: "f32[128]" = torch.ops.aten.mul.Tensor(mul_235, 0.1);  mul_235 = None
    mul_237: "f32[128]" = torch.ops.aten.mul.Tensor(primals_293, 0.9)
    add_157: "f32[128]" = torch.ops.aten.add.Tensor(mul_236, mul_237);  mul_236 = mul_237 = None
    unsqueeze_116: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_117: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_238: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_232, unsqueeze_117);  mul_232 = unsqueeze_117 = None
    unsqueeze_118: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_119: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_158: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_238, unsqueeze_119);  mul_238 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_29: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_158, 0)
    mul_239: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_158, 0.01)
    where_29: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_29, add_158, mul_239);  gt_29 = add_158 = mul_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_30: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(where_29, primals_165, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_159: "i64[]" = torch.ops.aten.add.Tensor(primals_294, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_30 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
    getitem_76: "f32[1, 128, 1, 1]" = var_mean_30[0]
    getitem_77: "f32[1, 128, 1, 1]" = var_mean_30[1];  var_mean_30 = None
    add_160: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
    rsqrt_30: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
    sub_30: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_30, getitem_77)
    mul_240: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    squeeze_90: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
    squeeze_91: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
    mul_241: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
    mul_242: "f32[128]" = torch.ops.aten.mul.Tensor(primals_295, 0.9)
    add_161: "f32[128]" = torch.ops.aten.add.Tensor(mul_241, mul_242);  mul_241 = mul_242 = None
    squeeze_92: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
    mul_243: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0001220852154804);  squeeze_92 = None
    mul_244: "f32[128]" = torch.ops.aten.mul.Tensor(mul_243, 0.1);  mul_243 = None
    mul_245: "f32[128]" = torch.ops.aten.mul.Tensor(primals_296, 0.9)
    add_162: "f32[128]" = torch.ops.aten.add.Tensor(mul_244, mul_245);  mul_244 = mul_245 = None
    unsqueeze_120: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_61, -1)
    unsqueeze_121: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    mul_246: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_240, unsqueeze_121);  mul_240 = unsqueeze_121 = None
    unsqueeze_122: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1);  primals_62 = None
    unsqueeze_123: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    add_163: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_246, unsqueeze_123);  mul_246 = unsqueeze_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_30: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_163, 0)
    mul_247: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_163, 0.01)
    where_30: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_30, add_163, mul_247);  gt_30 = add_163 = mul_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_164: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(where_30, add_153)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_31: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(add_164, primals_166, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_165: "i64[]" = torch.ops.aten.add.Tensor(primals_297, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_31 = torch.ops.aten.var_mean.correction(convolution_31, [0, 2, 3], correction = 0, keepdim = True)
    getitem_78: "f32[1, 128, 1, 1]" = var_mean_31[0]
    getitem_79: "f32[1, 128, 1, 1]" = var_mean_31[1];  var_mean_31 = None
    add_166: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
    rsqrt_31: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
    sub_31: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_31, getitem_79)
    mul_248: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    squeeze_93: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
    squeeze_94: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
    mul_249: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
    mul_250: "f32[128]" = torch.ops.aten.mul.Tensor(primals_298, 0.9)
    add_167: "f32[128]" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
    squeeze_95: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
    mul_251: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0001220852154804);  squeeze_95 = None
    mul_252: "f32[128]" = torch.ops.aten.mul.Tensor(mul_251, 0.1);  mul_251 = None
    mul_253: "f32[128]" = torch.ops.aten.mul.Tensor(primals_299, 0.9)
    add_168: "f32[128]" = torch.ops.aten.add.Tensor(mul_252, mul_253);  mul_252 = mul_253 = None
    unsqueeze_124: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1)
    unsqueeze_125: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_254: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_248, unsqueeze_125);  mul_248 = unsqueeze_125 = None
    unsqueeze_126: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_64, -1);  primals_64 = None
    unsqueeze_127: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_169: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_254, unsqueeze_127);  mul_254 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_31: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_169, 0)
    mul_255: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_169, 0.01)
    where_31: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_31, add_169, mul_255);  gt_31 = add_169 = mul_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_32: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(where_31, primals_167, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_170: "i64[]" = torch.ops.aten.add.Tensor(primals_300, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_32 = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
    getitem_80: "f32[1, 128, 1, 1]" = var_mean_32[0]
    getitem_81: "f32[1, 128, 1, 1]" = var_mean_32[1];  var_mean_32 = None
    add_171: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05)
    rsqrt_32: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
    sub_32: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_32, getitem_81)
    mul_256: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    squeeze_96: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
    squeeze_97: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
    mul_257: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
    mul_258: "f32[128]" = torch.ops.aten.mul.Tensor(primals_301, 0.9)
    add_172: "f32[128]" = torch.ops.aten.add.Tensor(mul_257, mul_258);  mul_257 = mul_258 = None
    squeeze_98: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
    mul_259: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0001220852154804);  squeeze_98 = None
    mul_260: "f32[128]" = torch.ops.aten.mul.Tensor(mul_259, 0.1);  mul_259 = None
    mul_261: "f32[128]" = torch.ops.aten.mul.Tensor(primals_302, 0.9)
    add_173: "f32[128]" = torch.ops.aten.add.Tensor(mul_260, mul_261);  mul_260 = mul_261 = None
    unsqueeze_128: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
    unsqueeze_129: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    mul_262: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_256, unsqueeze_129);  mul_256 = unsqueeze_129 = None
    unsqueeze_130: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
    unsqueeze_131: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    add_174: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_262, unsqueeze_131);  mul_262 = unsqueeze_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_32: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_174, 0)
    mul_263: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_174, 0.01)
    where_32: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_32, add_174, mul_263);  gt_32 = add_174 = mul_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_175: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(where_32, add_164)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_33: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(add_175, primals_168, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_176: "i64[]" = torch.ops.aten.add.Tensor(primals_303, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_33 = torch.ops.aten.var_mean.correction(convolution_33, [0, 2, 3], correction = 0, keepdim = True)
    getitem_82: "f32[1, 128, 1, 1]" = var_mean_33[0]
    getitem_83: "f32[1, 128, 1, 1]" = var_mean_33[1];  var_mean_33 = None
    add_177: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05)
    rsqrt_33: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
    sub_33: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_33, getitem_83)
    mul_264: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    squeeze_99: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
    squeeze_100: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
    mul_265: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
    mul_266: "f32[128]" = torch.ops.aten.mul.Tensor(primals_304, 0.9)
    add_178: "f32[128]" = torch.ops.aten.add.Tensor(mul_265, mul_266);  mul_265 = mul_266 = None
    squeeze_101: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
    mul_267: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0001220852154804);  squeeze_101 = None
    mul_268: "f32[128]" = torch.ops.aten.mul.Tensor(mul_267, 0.1);  mul_267 = None
    mul_269: "f32[128]" = torch.ops.aten.mul.Tensor(primals_305, 0.9)
    add_179: "f32[128]" = torch.ops.aten.add.Tensor(mul_268, mul_269);  mul_268 = mul_269 = None
    unsqueeze_132: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_67, -1)
    unsqueeze_133: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_270: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_264, unsqueeze_133);  mul_264 = unsqueeze_133 = None
    unsqueeze_134: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1);  primals_68 = None
    unsqueeze_135: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_180: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_270, unsqueeze_135);  mul_270 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_33: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_180, 0)
    mul_271: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_180, 0.01)
    where_33: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_33, add_180, mul_271);  gt_33 = add_180 = mul_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:339, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
    split_with_sizes_8 = torch.ops.aten.split_with_sizes.default(where_16, [128, 128], 1)
    getitem_84: "f32[8, 128, 32, 32]" = split_with_sizes_8[0];  split_with_sizes_8 = None
    cat_2: "f32[8, 256, 32, 32]" = torch.ops.aten.cat.default([getitem_84, where_33], 1);  getitem_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_34: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(cat_2, primals_169, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_181: "i64[]" = torch.ops.aten.add.Tensor(primals_306, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_34 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
    getitem_86: "f32[1, 256, 1, 1]" = var_mean_34[0]
    getitem_87: "f32[1, 256, 1, 1]" = var_mean_34[1];  var_mean_34 = None
    add_182: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05)
    rsqrt_34: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
    sub_34: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_87)
    mul_272: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    squeeze_102: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
    squeeze_103: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
    mul_273: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
    mul_274: "f32[256]" = torch.ops.aten.mul.Tensor(primals_307, 0.9)
    add_183: "f32[256]" = torch.ops.aten.add.Tensor(mul_273, mul_274);  mul_273 = mul_274 = None
    squeeze_104: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
    mul_275: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0001220852154804);  squeeze_104 = None
    mul_276: "f32[256]" = torch.ops.aten.mul.Tensor(mul_275, 0.1);  mul_275 = None
    mul_277: "f32[256]" = torch.ops.aten.mul.Tensor(primals_308, 0.9)
    add_184: "f32[256]" = torch.ops.aten.add.Tensor(mul_276, mul_277);  mul_276 = mul_277 = None
    unsqueeze_136: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1)
    unsqueeze_137: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    mul_278: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_272, unsqueeze_137);  mul_272 = unsqueeze_137 = None
    unsqueeze_138: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_70, -1);  primals_70 = None
    unsqueeze_139: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    add_185: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_278, unsqueeze_139);  mul_278 = unsqueeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_34: "b8[8, 256, 32, 32]" = torch.ops.aten.gt.Scalar(add_185, 0)
    mul_279: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(add_185, 0.01)
    where_34: "f32[8, 256, 32, 32]" = torch.ops.aten.where.self(gt_34, add_185, mul_279);  gt_34 = add_185 = mul_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:126, code: x = self.conv(x)
    convolution_35: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(where_34, primals_170, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_186: "i64[]" = torch.ops.aten.add.Tensor(primals_309, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_35 = torch.ops.aten.var_mean.correction(convolution_35, [0, 2, 3], correction = 0, keepdim = True)
    getitem_88: "f32[1, 512, 1, 1]" = var_mean_35[0]
    getitem_89: "f32[1, 512, 1, 1]" = var_mean_35[1];  var_mean_35 = None
    add_187: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05)
    rsqrt_35: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_187);  add_187 = None
    sub_35: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_35, getitem_89)
    mul_280: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    squeeze_105: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
    squeeze_106: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
    mul_281: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
    mul_282: "f32[512]" = torch.ops.aten.mul.Tensor(primals_310, 0.9)
    add_188: "f32[512]" = torch.ops.aten.add.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
    squeeze_107: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
    mul_283: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0004885197850513);  squeeze_107 = None
    mul_284: "f32[512]" = torch.ops.aten.mul.Tensor(mul_283, 0.1);  mul_283 = None
    mul_285: "f32[512]" = torch.ops.aten.mul.Tensor(primals_311, 0.9)
    add_189: "f32[512]" = torch.ops.aten.add.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
    unsqueeze_140: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_141: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_286: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_141);  mul_280 = unsqueeze_141 = None
    unsqueeze_142: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_143: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_190: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_286, unsqueeze_143);  mul_286 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_35: "b8[8, 512, 16, 16]" = torch.ops.aten.gt.Scalar(add_190, 0)
    mul_287: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(add_190, 0.01)
    where_35: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(gt_35, add_190, mul_287);  gt_35 = add_190 = mul_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_36: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(where_35, primals_171, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_191: "i64[]" = torch.ops.aten.add.Tensor(primals_312, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_36 = torch.ops.aten.var_mean.correction(convolution_36, [0, 2, 3], correction = 0, keepdim = True)
    getitem_90: "f32[1, 512, 1, 1]" = var_mean_36[0]
    getitem_91: "f32[1, 512, 1, 1]" = var_mean_36[1];  var_mean_36 = None
    add_192: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05)
    rsqrt_36: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
    sub_36: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_36, getitem_91)
    mul_288: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    squeeze_108: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
    squeeze_109: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
    mul_289: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
    mul_290: "f32[512]" = torch.ops.aten.mul.Tensor(primals_313, 0.9)
    add_193: "f32[512]" = torch.ops.aten.add.Tensor(mul_289, mul_290);  mul_289 = mul_290 = None
    squeeze_110: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
    mul_291: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0004885197850513);  squeeze_110 = None
    mul_292: "f32[512]" = torch.ops.aten.mul.Tensor(mul_291, 0.1);  mul_291 = None
    mul_293: "f32[512]" = torch.ops.aten.mul.Tensor(primals_314, 0.9)
    add_194: "f32[512]" = torch.ops.aten.add.Tensor(mul_292, mul_293);  mul_292 = mul_293 = None
    unsqueeze_144: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_73, -1)
    unsqueeze_145: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    mul_294: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_288, unsqueeze_145);  mul_288 = unsqueeze_145 = None
    unsqueeze_146: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1);  primals_74 = None
    unsqueeze_147: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    add_195: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_294, unsqueeze_147);  mul_294 = unsqueeze_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_36: "b8[8, 512, 16, 16]" = torch.ops.aten.gt.Scalar(add_195, 0)
    mul_295: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(add_195, 0.01)
    where_36: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(gt_36, add_195, mul_295);  gt_36 = add_195 = mul_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    split_with_sizes_10 = torch.ops.aten.split_with_sizes.default(where_36, [256, 256], 1)
    getitem_95: "f32[8, 256, 16, 16]" = split_with_sizes_10[1];  split_with_sizes_10 = None
    convolution_37: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(getitem_95, primals_172, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_196: "i64[]" = torch.ops.aten.add.Tensor(primals_315, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_37 = torch.ops.aten.var_mean.correction(convolution_37, [0, 2, 3], correction = 0, keepdim = True)
    getitem_96: "f32[1, 256, 1, 1]" = var_mean_37[0]
    getitem_97: "f32[1, 256, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_197: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05)
    rsqrt_37: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
    sub_37: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_37, getitem_97)
    mul_296: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    squeeze_111: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
    squeeze_112: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
    mul_297: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
    mul_298: "f32[256]" = torch.ops.aten.mul.Tensor(primals_316, 0.9)
    add_198: "f32[256]" = torch.ops.aten.add.Tensor(mul_297, mul_298);  mul_297 = mul_298 = None
    squeeze_113: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
    mul_299: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0004885197850513);  squeeze_113 = None
    mul_300: "f32[256]" = torch.ops.aten.mul.Tensor(mul_299, 0.1);  mul_299 = None
    mul_301: "f32[256]" = torch.ops.aten.mul.Tensor(primals_317, 0.9)
    add_199: "f32[256]" = torch.ops.aten.add.Tensor(mul_300, mul_301);  mul_300 = mul_301 = None
    unsqueeze_148: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1)
    unsqueeze_149: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_302: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_296, unsqueeze_149);  mul_296 = unsqueeze_149 = None
    unsqueeze_150: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_76, -1);  primals_76 = None
    unsqueeze_151: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_200: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_302, unsqueeze_151);  mul_302 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_37: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_200, 0)
    mul_303: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_200, 0.01)
    where_37: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_37, add_200, mul_303);  gt_37 = add_200 = mul_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_38: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(where_37, primals_173, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_201: "i64[]" = torch.ops.aten.add.Tensor(primals_318, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_38 = torch.ops.aten.var_mean.correction(convolution_38, [0, 2, 3], correction = 0, keepdim = True)
    getitem_98: "f32[1, 256, 1, 1]" = var_mean_38[0]
    getitem_99: "f32[1, 256, 1, 1]" = var_mean_38[1];  var_mean_38 = None
    add_202: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05)
    rsqrt_38: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_202);  add_202 = None
    sub_38: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_38, getitem_99)
    mul_304: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    squeeze_114: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
    squeeze_115: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
    mul_305: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
    mul_306: "f32[256]" = torch.ops.aten.mul.Tensor(primals_319, 0.9)
    add_203: "f32[256]" = torch.ops.aten.add.Tensor(mul_305, mul_306);  mul_305 = mul_306 = None
    squeeze_116: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
    mul_307: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0004885197850513);  squeeze_116 = None
    mul_308: "f32[256]" = torch.ops.aten.mul.Tensor(mul_307, 0.1);  mul_307 = None
    mul_309: "f32[256]" = torch.ops.aten.mul.Tensor(primals_320, 0.9)
    add_204: "f32[256]" = torch.ops.aten.add.Tensor(mul_308, mul_309);  mul_308 = mul_309 = None
    unsqueeze_152: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
    unsqueeze_153: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    mul_310: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_304, unsqueeze_153);  mul_304 = unsqueeze_153 = None
    unsqueeze_154: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
    unsqueeze_155: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    add_205: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_310, unsqueeze_155);  mul_310 = unsqueeze_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_38: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_205, 0)
    mul_311: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_205, 0.01)
    where_38: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_38, add_205, mul_311);  gt_38 = add_205 = mul_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_206: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(where_38, getitem_95)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_39: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(add_206, primals_174, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_207: "i64[]" = torch.ops.aten.add.Tensor(primals_321, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_39 = torch.ops.aten.var_mean.correction(convolution_39, [0, 2, 3], correction = 0, keepdim = True)
    getitem_100: "f32[1, 256, 1, 1]" = var_mean_39[0]
    getitem_101: "f32[1, 256, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_208: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05)
    rsqrt_39: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_208);  add_208 = None
    sub_39: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_39, getitem_101)
    mul_312: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    squeeze_117: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
    squeeze_118: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
    mul_313: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
    mul_314: "f32[256]" = torch.ops.aten.mul.Tensor(primals_322, 0.9)
    add_209: "f32[256]" = torch.ops.aten.add.Tensor(mul_313, mul_314);  mul_313 = mul_314 = None
    squeeze_119: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_100, [0, 2, 3]);  getitem_100 = None
    mul_315: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0004885197850513);  squeeze_119 = None
    mul_316: "f32[256]" = torch.ops.aten.mul.Tensor(mul_315, 0.1);  mul_315 = None
    mul_317: "f32[256]" = torch.ops.aten.mul.Tensor(primals_323, 0.9)
    add_210: "f32[256]" = torch.ops.aten.add.Tensor(mul_316, mul_317);  mul_316 = mul_317 = None
    unsqueeze_156: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_79, -1)
    unsqueeze_157: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_318: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_312, unsqueeze_157);  mul_312 = unsqueeze_157 = None
    unsqueeze_158: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1);  primals_80 = None
    unsqueeze_159: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_211: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_318, unsqueeze_159);  mul_318 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_39: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_211, 0)
    mul_319: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_211, 0.01)
    where_39: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_39, add_211, mul_319);  gt_39 = add_211 = mul_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_40: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(where_39, primals_175, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_212: "i64[]" = torch.ops.aten.add.Tensor(primals_324, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_40 = torch.ops.aten.var_mean.correction(convolution_40, [0, 2, 3], correction = 0, keepdim = True)
    getitem_102: "f32[1, 256, 1, 1]" = var_mean_40[0]
    getitem_103: "f32[1, 256, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_213: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05)
    rsqrt_40: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
    sub_40: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_103)
    mul_320: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    squeeze_120: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
    squeeze_121: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
    mul_321: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
    mul_322: "f32[256]" = torch.ops.aten.mul.Tensor(primals_325, 0.9)
    add_214: "f32[256]" = torch.ops.aten.add.Tensor(mul_321, mul_322);  mul_321 = mul_322 = None
    squeeze_122: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
    mul_323: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0004885197850513);  squeeze_122 = None
    mul_324: "f32[256]" = torch.ops.aten.mul.Tensor(mul_323, 0.1);  mul_323 = None
    mul_325: "f32[256]" = torch.ops.aten.mul.Tensor(primals_326, 0.9)
    add_215: "f32[256]" = torch.ops.aten.add.Tensor(mul_324, mul_325);  mul_324 = mul_325 = None
    unsqueeze_160: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1)
    unsqueeze_161: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    mul_326: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_320, unsqueeze_161);  mul_320 = unsqueeze_161 = None
    unsqueeze_162: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_82, -1);  primals_82 = None
    unsqueeze_163: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    add_216: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_326, unsqueeze_163);  mul_326 = unsqueeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_40: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_216, 0)
    mul_327: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_216, 0.01)
    where_40: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_40, add_216, mul_327);  gt_40 = add_216 = mul_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_217: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(where_40, add_206)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_41: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(add_217, primals_176, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_218: "i64[]" = torch.ops.aten.add.Tensor(primals_327, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_41 = torch.ops.aten.var_mean.correction(convolution_41, [0, 2, 3], correction = 0, keepdim = True)
    getitem_104: "f32[1, 256, 1, 1]" = var_mean_41[0]
    getitem_105: "f32[1, 256, 1, 1]" = var_mean_41[1];  var_mean_41 = None
    add_219: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05)
    rsqrt_41: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_219);  add_219 = None
    sub_41: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_41, getitem_105)
    mul_328: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
    squeeze_123: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2, 3]);  getitem_105 = None
    squeeze_124: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
    mul_329: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
    mul_330: "f32[256]" = torch.ops.aten.mul.Tensor(primals_328, 0.9)
    add_220: "f32[256]" = torch.ops.aten.add.Tensor(mul_329, mul_330);  mul_329 = mul_330 = None
    squeeze_125: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_104, [0, 2, 3]);  getitem_104 = None
    mul_331: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0004885197850513);  squeeze_125 = None
    mul_332: "f32[256]" = torch.ops.aten.mul.Tensor(mul_331, 0.1);  mul_331 = None
    mul_333: "f32[256]" = torch.ops.aten.mul.Tensor(primals_329, 0.9)
    add_221: "f32[256]" = torch.ops.aten.add.Tensor(mul_332, mul_333);  mul_332 = mul_333 = None
    unsqueeze_164: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1)
    unsqueeze_165: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_334: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_328, unsqueeze_165);  mul_328 = unsqueeze_165 = None
    unsqueeze_166: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
    unsqueeze_167: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_222: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_334, unsqueeze_167);  mul_334 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_41: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_222, 0)
    mul_335: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_222, 0.01)
    where_41: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_41, add_222, mul_335);  gt_41 = add_222 = mul_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_42: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(where_41, primals_177, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_223: "i64[]" = torch.ops.aten.add.Tensor(primals_330, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_42 = torch.ops.aten.var_mean.correction(convolution_42, [0, 2, 3], correction = 0, keepdim = True)
    getitem_106: "f32[1, 256, 1, 1]" = var_mean_42[0]
    getitem_107: "f32[1, 256, 1, 1]" = var_mean_42[1];  var_mean_42 = None
    add_224: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05)
    rsqrt_42: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_224);  add_224 = None
    sub_42: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_42, getitem_107)
    mul_336: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
    squeeze_126: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
    squeeze_127: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
    mul_337: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
    mul_338: "f32[256]" = torch.ops.aten.mul.Tensor(primals_331, 0.9)
    add_225: "f32[256]" = torch.ops.aten.add.Tensor(mul_337, mul_338);  mul_337 = mul_338 = None
    squeeze_128: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
    mul_339: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0004885197850513);  squeeze_128 = None
    mul_340: "f32[256]" = torch.ops.aten.mul.Tensor(mul_339, 0.1);  mul_339 = None
    mul_341: "f32[256]" = torch.ops.aten.mul.Tensor(primals_332, 0.9)
    add_226: "f32[256]" = torch.ops.aten.add.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
    unsqueeze_168: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_85, -1)
    unsqueeze_169: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    mul_342: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_336, unsqueeze_169);  mul_336 = unsqueeze_169 = None
    unsqueeze_170: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1);  primals_86 = None
    unsqueeze_171: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    add_227: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_342, unsqueeze_171);  mul_342 = unsqueeze_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_42: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_227, 0)
    mul_343: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_227, 0.01)
    where_42: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_42, add_227, mul_343);  gt_42 = add_227 = mul_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_228: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(where_42, add_217)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_43: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(add_228, primals_178, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_229: "i64[]" = torch.ops.aten.add.Tensor(primals_333, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_43 = torch.ops.aten.var_mean.correction(convolution_43, [0, 2, 3], correction = 0, keepdim = True)
    getitem_108: "f32[1, 256, 1, 1]" = var_mean_43[0]
    getitem_109: "f32[1, 256, 1, 1]" = var_mean_43[1];  var_mean_43 = None
    add_230: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05)
    rsqrt_43: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_230);  add_230 = None
    sub_43: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_43, getitem_109)
    mul_344: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
    squeeze_129: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_109, [0, 2, 3]);  getitem_109 = None
    squeeze_130: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
    mul_345: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
    mul_346: "f32[256]" = torch.ops.aten.mul.Tensor(primals_334, 0.9)
    add_231: "f32[256]" = torch.ops.aten.add.Tensor(mul_345, mul_346);  mul_345 = mul_346 = None
    squeeze_131: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_108, [0, 2, 3]);  getitem_108 = None
    mul_347: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0004885197850513);  squeeze_131 = None
    mul_348: "f32[256]" = torch.ops.aten.mul.Tensor(mul_347, 0.1);  mul_347 = None
    mul_349: "f32[256]" = torch.ops.aten.mul.Tensor(primals_335, 0.9)
    add_232: "f32[256]" = torch.ops.aten.add.Tensor(mul_348, mul_349);  mul_348 = mul_349 = None
    unsqueeze_172: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1)
    unsqueeze_173: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_350: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_344, unsqueeze_173);  mul_344 = unsqueeze_173 = None
    unsqueeze_174: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_88, -1);  primals_88 = None
    unsqueeze_175: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_233: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_350, unsqueeze_175);  mul_350 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_43: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_233, 0)
    mul_351: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_233, 0.01)
    where_43: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_43, add_233, mul_351);  gt_43 = add_233 = mul_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_44: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(where_43, primals_179, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_234: "i64[]" = torch.ops.aten.add.Tensor(primals_336, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_44 = torch.ops.aten.var_mean.correction(convolution_44, [0, 2, 3], correction = 0, keepdim = True)
    getitem_110: "f32[1, 256, 1, 1]" = var_mean_44[0]
    getitem_111: "f32[1, 256, 1, 1]" = var_mean_44[1];  var_mean_44 = None
    add_235: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05)
    rsqrt_44: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_235);  add_235 = None
    sub_44: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_44, getitem_111)
    mul_352: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
    squeeze_132: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_111, [0, 2, 3]);  getitem_111 = None
    squeeze_133: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
    mul_353: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1)
    mul_354: "f32[256]" = torch.ops.aten.mul.Tensor(primals_337, 0.9)
    add_236: "f32[256]" = torch.ops.aten.add.Tensor(mul_353, mul_354);  mul_353 = mul_354 = None
    squeeze_134: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_110, [0, 2, 3]);  getitem_110 = None
    mul_355: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_134, 1.0004885197850513);  squeeze_134 = None
    mul_356: "f32[256]" = torch.ops.aten.mul.Tensor(mul_355, 0.1);  mul_355 = None
    mul_357: "f32[256]" = torch.ops.aten.mul.Tensor(primals_338, 0.9)
    add_237: "f32[256]" = torch.ops.aten.add.Tensor(mul_356, mul_357);  mul_356 = mul_357 = None
    unsqueeze_176: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1)
    unsqueeze_177: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    mul_358: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_352, unsqueeze_177);  mul_352 = unsqueeze_177 = None
    unsqueeze_178: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1);  primals_90 = None
    unsqueeze_179: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    add_238: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_358, unsqueeze_179);  mul_358 = unsqueeze_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_44: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_238, 0)
    mul_359: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_238, 0.01)
    where_44: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_44, add_238, mul_359);  gt_44 = add_238 = mul_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_239: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(where_44, add_228)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_45: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(add_239, primals_180, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_240: "i64[]" = torch.ops.aten.add.Tensor(primals_339, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_45 = torch.ops.aten.var_mean.correction(convolution_45, [0, 2, 3], correction = 0, keepdim = True)
    getitem_112: "f32[1, 256, 1, 1]" = var_mean_45[0]
    getitem_113: "f32[1, 256, 1, 1]" = var_mean_45[1];  var_mean_45 = None
    add_241: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05)
    rsqrt_45: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_241);  add_241 = None
    sub_45: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_45, getitem_113)
    mul_360: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
    squeeze_135: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3]);  getitem_113 = None
    squeeze_136: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
    mul_361: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1)
    mul_362: "f32[256]" = torch.ops.aten.mul.Tensor(primals_340, 0.9)
    add_242: "f32[256]" = torch.ops.aten.add.Tensor(mul_361, mul_362);  mul_361 = mul_362 = None
    squeeze_137: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_112, [0, 2, 3]);  getitem_112 = None
    mul_363: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_137, 1.0004885197850513);  squeeze_137 = None
    mul_364: "f32[256]" = torch.ops.aten.mul.Tensor(mul_363, 0.1);  mul_363 = None
    mul_365: "f32[256]" = torch.ops.aten.mul.Tensor(primals_341, 0.9)
    add_243: "f32[256]" = torch.ops.aten.add.Tensor(mul_364, mul_365);  mul_364 = mul_365 = None
    unsqueeze_180: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_91, -1)
    unsqueeze_181: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_366: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_360, unsqueeze_181);  mul_360 = unsqueeze_181 = None
    unsqueeze_182: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1);  primals_92 = None
    unsqueeze_183: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_244: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_366, unsqueeze_183);  mul_366 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_45: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_244, 0)
    mul_367: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_244, 0.01)
    where_45: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_45, add_244, mul_367);  gt_45 = add_244 = mul_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_46: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(where_45, primals_181, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_245: "i64[]" = torch.ops.aten.add.Tensor(primals_342, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_46 = torch.ops.aten.var_mean.correction(convolution_46, [0, 2, 3], correction = 0, keepdim = True)
    getitem_114: "f32[1, 256, 1, 1]" = var_mean_46[0]
    getitem_115: "f32[1, 256, 1, 1]" = var_mean_46[1];  var_mean_46 = None
    add_246: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-05)
    rsqrt_46: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_246);  add_246 = None
    sub_46: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_46, getitem_115)
    mul_368: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
    squeeze_138: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_115, [0, 2, 3]);  getitem_115 = None
    squeeze_139: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
    mul_369: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_138, 0.1)
    mul_370: "f32[256]" = torch.ops.aten.mul.Tensor(primals_343, 0.9)
    add_247: "f32[256]" = torch.ops.aten.add.Tensor(mul_369, mul_370);  mul_369 = mul_370 = None
    squeeze_140: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_114, [0, 2, 3]);  getitem_114 = None
    mul_371: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_140, 1.0004885197850513);  squeeze_140 = None
    mul_372: "f32[256]" = torch.ops.aten.mul.Tensor(mul_371, 0.1);  mul_371 = None
    mul_373: "f32[256]" = torch.ops.aten.mul.Tensor(primals_344, 0.9)
    add_248: "f32[256]" = torch.ops.aten.add.Tensor(mul_372, mul_373);  mul_372 = mul_373 = None
    unsqueeze_184: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_93, -1)
    unsqueeze_185: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    mul_374: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_368, unsqueeze_185);  mul_368 = unsqueeze_185 = None
    unsqueeze_186: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_94, -1);  primals_94 = None
    unsqueeze_187: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    add_249: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_374, unsqueeze_187);  mul_374 = unsqueeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_46: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_249, 0)
    mul_375: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_249, 0.01)
    where_46: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_46, add_249, mul_375);  gt_46 = add_249 = mul_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_250: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(where_46, add_239)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_47: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(add_250, primals_182, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_251: "i64[]" = torch.ops.aten.add.Tensor(primals_345, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_47 = torch.ops.aten.var_mean.correction(convolution_47, [0, 2, 3], correction = 0, keepdim = True)
    getitem_116: "f32[1, 256, 1, 1]" = var_mean_47[0]
    getitem_117: "f32[1, 256, 1, 1]" = var_mean_47[1];  var_mean_47 = None
    add_252: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-05)
    rsqrt_47: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_252);  add_252 = None
    sub_47: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_47, getitem_117)
    mul_376: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
    squeeze_141: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_117, [0, 2, 3]);  getitem_117 = None
    squeeze_142: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
    mul_377: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_141, 0.1)
    mul_378: "f32[256]" = torch.ops.aten.mul.Tensor(primals_346, 0.9)
    add_253: "f32[256]" = torch.ops.aten.add.Tensor(mul_377, mul_378);  mul_377 = mul_378 = None
    squeeze_143: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_116, [0, 2, 3]);  getitem_116 = None
    mul_379: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_143, 1.0004885197850513);  squeeze_143 = None
    mul_380: "f32[256]" = torch.ops.aten.mul.Tensor(mul_379, 0.1);  mul_379 = None
    mul_381: "f32[256]" = torch.ops.aten.mul.Tensor(primals_347, 0.9)
    add_254: "f32[256]" = torch.ops.aten.add.Tensor(mul_380, mul_381);  mul_380 = mul_381 = None
    unsqueeze_188: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1)
    unsqueeze_189: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_382: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_376, unsqueeze_189);  mul_376 = unsqueeze_189 = None
    unsqueeze_190: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1);  primals_96 = None
    unsqueeze_191: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_255: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_382, unsqueeze_191);  mul_382 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_47: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_255, 0)
    mul_383: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_255, 0.01)
    where_47: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_47, add_255, mul_383);  gt_47 = add_255 = mul_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_48: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(where_47, primals_183, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_256: "i64[]" = torch.ops.aten.add.Tensor(primals_348, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_48 = torch.ops.aten.var_mean.correction(convolution_48, [0, 2, 3], correction = 0, keepdim = True)
    getitem_118: "f32[1, 256, 1, 1]" = var_mean_48[0]
    getitem_119: "f32[1, 256, 1, 1]" = var_mean_48[1];  var_mean_48 = None
    add_257: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05)
    rsqrt_48: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_257);  add_257 = None
    sub_48: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_48, getitem_119)
    mul_384: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
    squeeze_144: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
    squeeze_145: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
    mul_385: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_144, 0.1)
    mul_386: "f32[256]" = torch.ops.aten.mul.Tensor(primals_349, 0.9)
    add_258: "f32[256]" = torch.ops.aten.add.Tensor(mul_385, mul_386);  mul_385 = mul_386 = None
    squeeze_146: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_118, [0, 2, 3]);  getitem_118 = None
    mul_387: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_146, 1.0004885197850513);  squeeze_146 = None
    mul_388: "f32[256]" = torch.ops.aten.mul.Tensor(mul_387, 0.1);  mul_387 = None
    mul_389: "f32[256]" = torch.ops.aten.mul.Tensor(primals_350, 0.9)
    add_259: "f32[256]" = torch.ops.aten.add.Tensor(mul_388, mul_389);  mul_388 = mul_389 = None
    unsqueeze_192: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_97, -1)
    unsqueeze_193: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    mul_390: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_384, unsqueeze_193);  mul_384 = unsqueeze_193 = None
    unsqueeze_194: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1);  primals_98 = None
    unsqueeze_195: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    add_260: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_390, unsqueeze_195);  mul_390 = unsqueeze_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_48: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_260, 0)
    mul_391: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_260, 0.01)
    where_48: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_48, add_260, mul_391);  gt_48 = add_260 = mul_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_261: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(where_48, add_250)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_49: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(add_261, primals_184, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_262: "i64[]" = torch.ops.aten.add.Tensor(primals_351, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_49 = torch.ops.aten.var_mean.correction(convolution_49, [0, 2, 3], correction = 0, keepdim = True)
    getitem_120: "f32[1, 256, 1, 1]" = var_mean_49[0]
    getitem_121: "f32[1, 256, 1, 1]" = var_mean_49[1];  var_mean_49 = None
    add_263: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-05)
    rsqrt_49: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_263);  add_263 = None
    sub_49: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_49, getitem_121)
    mul_392: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
    squeeze_147: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3]);  getitem_121 = None
    squeeze_148: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
    mul_393: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_147, 0.1)
    mul_394: "f32[256]" = torch.ops.aten.mul.Tensor(primals_352, 0.9)
    add_264: "f32[256]" = torch.ops.aten.add.Tensor(mul_393, mul_394);  mul_393 = mul_394 = None
    squeeze_149: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_120, [0, 2, 3]);  getitem_120 = None
    mul_395: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_149, 1.0004885197850513);  squeeze_149 = None
    mul_396: "f32[256]" = torch.ops.aten.mul.Tensor(mul_395, 0.1);  mul_395 = None
    mul_397: "f32[256]" = torch.ops.aten.mul.Tensor(primals_353, 0.9)
    add_265: "f32[256]" = torch.ops.aten.add.Tensor(mul_396, mul_397);  mul_396 = mul_397 = None
    unsqueeze_196: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1)
    unsqueeze_197: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_398: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_392, unsqueeze_197);  mul_392 = unsqueeze_197 = None
    unsqueeze_198: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_100, -1);  primals_100 = None
    unsqueeze_199: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_266: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_398, unsqueeze_199);  mul_398 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_49: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_266, 0)
    mul_399: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_266, 0.01)
    where_49: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_49, add_266, mul_399);  gt_49 = add_266 = mul_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_50: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(where_49, primals_185, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_267: "i64[]" = torch.ops.aten.add.Tensor(primals_354, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_50 = torch.ops.aten.var_mean.correction(convolution_50, [0, 2, 3], correction = 0, keepdim = True)
    getitem_122: "f32[1, 256, 1, 1]" = var_mean_50[0]
    getitem_123: "f32[1, 256, 1, 1]" = var_mean_50[1];  var_mean_50 = None
    add_268: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-05)
    rsqrt_50: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_268);  add_268 = None
    sub_50: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_50, getitem_123)
    mul_400: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
    squeeze_150: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_123, [0, 2, 3]);  getitem_123 = None
    squeeze_151: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
    mul_401: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_150, 0.1)
    mul_402: "f32[256]" = torch.ops.aten.mul.Tensor(primals_355, 0.9)
    add_269: "f32[256]" = torch.ops.aten.add.Tensor(mul_401, mul_402);  mul_401 = mul_402 = None
    squeeze_152: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_122, [0, 2, 3]);  getitem_122 = None
    mul_403: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_152, 1.0004885197850513);  squeeze_152 = None
    mul_404: "f32[256]" = torch.ops.aten.mul.Tensor(mul_403, 0.1);  mul_403 = None
    mul_405: "f32[256]" = torch.ops.aten.mul.Tensor(primals_356, 0.9)
    add_270: "f32[256]" = torch.ops.aten.add.Tensor(mul_404, mul_405);  mul_404 = mul_405 = None
    unsqueeze_200: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_101, -1)
    unsqueeze_201: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    mul_406: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_400, unsqueeze_201);  mul_400 = unsqueeze_201 = None
    unsqueeze_202: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1);  primals_102 = None
    unsqueeze_203: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    add_271: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_406, unsqueeze_203);  mul_406 = unsqueeze_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_50: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_271, 0)
    mul_407: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_271, 0.01)
    where_50: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_50, add_271, mul_407);  gt_50 = add_271 = mul_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_272: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(where_50, add_261)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_51: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(add_272, primals_186, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_273: "i64[]" = torch.ops.aten.add.Tensor(primals_357, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_51 = torch.ops.aten.var_mean.correction(convolution_51, [0, 2, 3], correction = 0, keepdim = True)
    getitem_124: "f32[1, 256, 1, 1]" = var_mean_51[0]
    getitem_125: "f32[1, 256, 1, 1]" = var_mean_51[1];  var_mean_51 = None
    add_274: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-05)
    rsqrt_51: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_274);  add_274 = None
    sub_51: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_51, getitem_125)
    mul_408: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
    squeeze_153: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_125, [0, 2, 3]);  getitem_125 = None
    squeeze_154: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
    mul_409: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_153, 0.1)
    mul_410: "f32[256]" = torch.ops.aten.mul.Tensor(primals_358, 0.9)
    add_275: "f32[256]" = torch.ops.aten.add.Tensor(mul_409, mul_410);  mul_409 = mul_410 = None
    squeeze_155: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_124, [0, 2, 3]);  getitem_124 = None
    mul_411: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_155, 1.0004885197850513);  squeeze_155 = None
    mul_412: "f32[256]" = torch.ops.aten.mul.Tensor(mul_411, 0.1);  mul_411 = None
    mul_413: "f32[256]" = torch.ops.aten.mul.Tensor(primals_359, 0.9)
    add_276: "f32[256]" = torch.ops.aten.add.Tensor(mul_412, mul_413);  mul_412 = mul_413 = None
    unsqueeze_204: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_103, -1)
    unsqueeze_205: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_414: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_408, unsqueeze_205);  mul_408 = unsqueeze_205 = None
    unsqueeze_206: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_104, -1);  primals_104 = None
    unsqueeze_207: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_277: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_414, unsqueeze_207);  mul_414 = unsqueeze_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_51: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_277, 0)
    mul_415: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_277, 0.01)
    where_51: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_51, add_277, mul_415);  gt_51 = add_277 = mul_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_52: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(where_51, primals_187, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_278: "i64[]" = torch.ops.aten.add.Tensor(primals_360, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_52 = torch.ops.aten.var_mean.correction(convolution_52, [0, 2, 3], correction = 0, keepdim = True)
    getitem_126: "f32[1, 256, 1, 1]" = var_mean_52[0]
    getitem_127: "f32[1, 256, 1, 1]" = var_mean_52[1];  var_mean_52 = None
    add_279: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-05)
    rsqrt_52: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_279);  add_279 = None
    sub_52: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_52, getitem_127)
    mul_416: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
    squeeze_156: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_127, [0, 2, 3]);  getitem_127 = None
    squeeze_157: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
    mul_417: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_156, 0.1)
    mul_418: "f32[256]" = torch.ops.aten.mul.Tensor(primals_361, 0.9)
    add_280: "f32[256]" = torch.ops.aten.add.Tensor(mul_417, mul_418);  mul_417 = mul_418 = None
    squeeze_158: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_126, [0, 2, 3]);  getitem_126 = None
    mul_419: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_158, 1.0004885197850513);  squeeze_158 = None
    mul_420: "f32[256]" = torch.ops.aten.mul.Tensor(mul_419, 0.1);  mul_419 = None
    mul_421: "f32[256]" = torch.ops.aten.mul.Tensor(primals_362, 0.9)
    add_281: "f32[256]" = torch.ops.aten.add.Tensor(mul_420, mul_421);  mul_420 = mul_421 = None
    unsqueeze_208: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_105, -1)
    unsqueeze_209: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    mul_422: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_416, unsqueeze_209);  mul_416 = unsqueeze_209 = None
    unsqueeze_210: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_106, -1);  primals_106 = None
    unsqueeze_211: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    add_282: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_422, unsqueeze_211);  mul_422 = unsqueeze_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_52: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_282, 0)
    mul_423: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_282, 0.01)
    where_52: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_52, add_282, mul_423);  gt_52 = add_282 = mul_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_283: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(where_52, add_272)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_53: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(add_283, primals_188, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_284: "i64[]" = torch.ops.aten.add.Tensor(primals_363, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_53 = torch.ops.aten.var_mean.correction(convolution_53, [0, 2, 3], correction = 0, keepdim = True)
    getitem_128: "f32[1, 256, 1, 1]" = var_mean_53[0]
    getitem_129: "f32[1, 256, 1, 1]" = var_mean_53[1];  var_mean_53 = None
    add_285: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-05)
    rsqrt_53: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_285);  add_285 = None
    sub_53: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_53, getitem_129)
    mul_424: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
    squeeze_159: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_129, [0, 2, 3]);  getitem_129 = None
    squeeze_160: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2, 3]);  rsqrt_53 = None
    mul_425: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_159, 0.1)
    mul_426: "f32[256]" = torch.ops.aten.mul.Tensor(primals_364, 0.9)
    add_286: "f32[256]" = torch.ops.aten.add.Tensor(mul_425, mul_426);  mul_425 = mul_426 = None
    squeeze_161: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_128, [0, 2, 3]);  getitem_128 = None
    mul_427: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_161, 1.0004885197850513);  squeeze_161 = None
    mul_428: "f32[256]" = torch.ops.aten.mul.Tensor(mul_427, 0.1);  mul_427 = None
    mul_429: "f32[256]" = torch.ops.aten.mul.Tensor(primals_365, 0.9)
    add_287: "f32[256]" = torch.ops.aten.add.Tensor(mul_428, mul_429);  mul_428 = mul_429 = None
    unsqueeze_212: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_107, -1)
    unsqueeze_213: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_430: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_424, unsqueeze_213);  mul_424 = unsqueeze_213 = None
    unsqueeze_214: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1);  primals_108 = None
    unsqueeze_215: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_288: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_430, unsqueeze_215);  mul_430 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_53: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_288, 0)
    mul_431: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_288, 0.01)
    where_53: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_53, add_288, mul_431);  gt_53 = add_288 = mul_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:339, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
    split_with_sizes_11 = torch.ops.aten.split_with_sizes.default(where_36, [256, 256], 1)
    getitem_130: "f32[8, 256, 16, 16]" = split_with_sizes_11[0];  split_with_sizes_11 = None
    cat_3: "f32[8, 512, 16, 16]" = torch.ops.aten.cat.default([getitem_130, where_53], 1);  getitem_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_54: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(cat_3, primals_189, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_289: "i64[]" = torch.ops.aten.add.Tensor(primals_366, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_54 = torch.ops.aten.var_mean.correction(convolution_54, [0, 2, 3], correction = 0, keepdim = True)
    getitem_132: "f32[1, 512, 1, 1]" = var_mean_54[0]
    getitem_133: "f32[1, 512, 1, 1]" = var_mean_54[1];  var_mean_54 = None
    add_290: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-05)
    rsqrt_54: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_290);  add_290 = None
    sub_54: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_54, getitem_133)
    mul_432: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
    squeeze_162: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_133, [0, 2, 3]);  getitem_133 = None
    squeeze_163: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2, 3]);  rsqrt_54 = None
    mul_433: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_162, 0.1)
    mul_434: "f32[512]" = torch.ops.aten.mul.Tensor(primals_367, 0.9)
    add_291: "f32[512]" = torch.ops.aten.add.Tensor(mul_433, mul_434);  mul_433 = mul_434 = None
    squeeze_164: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_132, [0, 2, 3]);  getitem_132 = None
    mul_435: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_164, 1.0004885197850513);  squeeze_164 = None
    mul_436: "f32[512]" = torch.ops.aten.mul.Tensor(mul_435, 0.1);  mul_435 = None
    mul_437: "f32[512]" = torch.ops.aten.mul.Tensor(primals_368, 0.9)
    add_292: "f32[512]" = torch.ops.aten.add.Tensor(mul_436, mul_437);  mul_436 = mul_437 = None
    unsqueeze_216: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_109, -1)
    unsqueeze_217: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    mul_438: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_432, unsqueeze_217);  mul_432 = unsqueeze_217 = None
    unsqueeze_218: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_110, -1);  primals_110 = None
    unsqueeze_219: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    add_293: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_438, unsqueeze_219);  mul_438 = unsqueeze_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_54: "b8[8, 512, 16, 16]" = torch.ops.aten.gt.Scalar(add_293, 0)
    mul_439: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(add_293, 0.01)
    where_54: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(gt_54, add_293, mul_439);  gt_54 = add_293 = mul_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:126, code: x = self.conv(x)
    convolution_55: "f32[8, 1024, 8, 8]" = torch.ops.aten.convolution.default(where_54, primals_190, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_294: "i64[]" = torch.ops.aten.add.Tensor(primals_369, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_55 = torch.ops.aten.var_mean.correction(convolution_55, [0, 2, 3], correction = 0, keepdim = True)
    getitem_134: "f32[1, 1024, 1, 1]" = var_mean_55[0]
    getitem_135: "f32[1, 1024, 1, 1]" = var_mean_55[1];  var_mean_55 = None
    add_295: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_134, 1e-05)
    rsqrt_55: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_295);  add_295 = None
    sub_55: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_55, getitem_135)
    mul_440: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = None
    squeeze_165: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_135, [0, 2, 3]);  getitem_135 = None
    squeeze_166: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2, 3]);  rsqrt_55 = None
    mul_441: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_165, 0.1)
    mul_442: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_370, 0.9)
    add_296: "f32[1024]" = torch.ops.aten.add.Tensor(mul_441, mul_442);  mul_441 = mul_442 = None
    squeeze_167: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_134, [0, 2, 3]);  getitem_134 = None
    mul_443: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_167, 1.0019569471624266);  squeeze_167 = None
    mul_444: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_443, 0.1);  mul_443 = None
    mul_445: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_371, 0.9)
    add_297: "f32[1024]" = torch.ops.aten.add.Tensor(mul_444, mul_445);  mul_444 = mul_445 = None
    unsqueeze_220: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_111, -1)
    unsqueeze_221: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_446: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(mul_440, unsqueeze_221);  mul_440 = unsqueeze_221 = None
    unsqueeze_222: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_112, -1);  primals_112 = None
    unsqueeze_223: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_298: "f32[8, 1024, 8, 8]" = torch.ops.aten.add.Tensor(mul_446, unsqueeze_223);  mul_446 = unsqueeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_55: "b8[8, 1024, 8, 8]" = torch.ops.aten.gt.Scalar(add_298, 0)
    mul_447: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(add_298, 0.01)
    where_55: "f32[8, 1024, 8, 8]" = torch.ops.aten.where.self(gt_55, add_298, mul_447);  gt_55 = add_298 = mul_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_56: "f32[8, 1024, 8, 8]" = torch.ops.aten.convolution.default(where_55, primals_191, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_299: "i64[]" = torch.ops.aten.add.Tensor(primals_372, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_56 = torch.ops.aten.var_mean.correction(convolution_56, [0, 2, 3], correction = 0, keepdim = True)
    getitem_136: "f32[1, 1024, 1, 1]" = var_mean_56[0]
    getitem_137: "f32[1, 1024, 1, 1]" = var_mean_56[1];  var_mean_56 = None
    add_300: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_136, 1e-05)
    rsqrt_56: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_300);  add_300 = None
    sub_56: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_56, getitem_137)
    mul_448: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = None
    squeeze_168: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_137, [0, 2, 3]);  getitem_137 = None
    squeeze_169: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2, 3]);  rsqrt_56 = None
    mul_449: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_168, 0.1)
    mul_450: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_373, 0.9)
    add_301: "f32[1024]" = torch.ops.aten.add.Tensor(mul_449, mul_450);  mul_449 = mul_450 = None
    squeeze_170: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_136, [0, 2, 3]);  getitem_136 = None
    mul_451: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_170, 1.0019569471624266);  squeeze_170 = None
    mul_452: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_451, 0.1);  mul_451 = None
    mul_453: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_374, 0.9)
    add_302: "f32[1024]" = torch.ops.aten.add.Tensor(mul_452, mul_453);  mul_452 = mul_453 = None
    unsqueeze_224: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_113, -1)
    unsqueeze_225: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    mul_454: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(mul_448, unsqueeze_225);  mul_448 = unsqueeze_225 = None
    unsqueeze_226: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_114, -1);  primals_114 = None
    unsqueeze_227: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    add_303: "f32[8, 1024, 8, 8]" = torch.ops.aten.add.Tensor(mul_454, unsqueeze_227);  mul_454 = unsqueeze_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_56: "b8[8, 1024, 8, 8]" = torch.ops.aten.gt.Scalar(add_303, 0)
    mul_455: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(add_303, 0.01)
    where_56: "f32[8, 1024, 8, 8]" = torch.ops.aten.where.self(gt_56, add_303, mul_455);  gt_56 = add_303 = mul_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    split_with_sizes_13 = torch.ops.aten.split_with_sizes.default(where_56, [512, 512], 1)
    getitem_141: "f32[8, 512, 8, 8]" = split_with_sizes_13[1];  split_with_sizes_13 = None
    convolution_57: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(getitem_141, primals_192, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_304: "i64[]" = torch.ops.aten.add.Tensor(primals_375, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_57 = torch.ops.aten.var_mean.correction(convolution_57, [0, 2, 3], correction = 0, keepdim = True)
    getitem_142: "f32[1, 512, 1, 1]" = var_mean_57[0]
    getitem_143: "f32[1, 512, 1, 1]" = var_mean_57[1];  var_mean_57 = None
    add_305: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-05)
    rsqrt_57: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_305);  add_305 = None
    sub_57: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_57, getitem_143)
    mul_456: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = None
    squeeze_171: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_143, [0, 2, 3]);  getitem_143 = None
    squeeze_172: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_57, [0, 2, 3]);  rsqrt_57 = None
    mul_457: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_171, 0.1)
    mul_458: "f32[512]" = torch.ops.aten.mul.Tensor(primals_376, 0.9)
    add_306: "f32[512]" = torch.ops.aten.add.Tensor(mul_457, mul_458);  mul_457 = mul_458 = None
    squeeze_173: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_142, [0, 2, 3]);  getitem_142 = None
    mul_459: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_173, 1.0019569471624266);  squeeze_173 = None
    mul_460: "f32[512]" = torch.ops.aten.mul.Tensor(mul_459, 0.1);  mul_459 = None
    mul_461: "f32[512]" = torch.ops.aten.mul.Tensor(primals_377, 0.9)
    add_307: "f32[512]" = torch.ops.aten.add.Tensor(mul_460, mul_461);  mul_460 = mul_461 = None
    unsqueeze_228: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_115, -1)
    unsqueeze_229: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_462: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_456, unsqueeze_229);  mul_456 = unsqueeze_229 = None
    unsqueeze_230: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_116, -1);  primals_116 = None
    unsqueeze_231: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_308: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_462, unsqueeze_231);  mul_462 = unsqueeze_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_57: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_308, 0)
    mul_463: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_308, 0.01)
    where_57: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_57, add_308, mul_463);  gt_57 = add_308 = mul_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_58: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(where_57, primals_193, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_309: "i64[]" = torch.ops.aten.add.Tensor(primals_378, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_58 = torch.ops.aten.var_mean.correction(convolution_58, [0, 2, 3], correction = 0, keepdim = True)
    getitem_144: "f32[1, 512, 1, 1]" = var_mean_58[0]
    getitem_145: "f32[1, 512, 1, 1]" = var_mean_58[1];  var_mean_58 = None
    add_310: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-05)
    rsqrt_58: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_310);  add_310 = None
    sub_58: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_58, getitem_145)
    mul_464: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = None
    squeeze_174: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_145, [0, 2, 3]);  getitem_145 = None
    squeeze_175: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_58, [0, 2, 3]);  rsqrt_58 = None
    mul_465: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_174, 0.1)
    mul_466: "f32[512]" = torch.ops.aten.mul.Tensor(primals_379, 0.9)
    add_311: "f32[512]" = torch.ops.aten.add.Tensor(mul_465, mul_466);  mul_465 = mul_466 = None
    squeeze_176: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_144, [0, 2, 3]);  getitem_144 = None
    mul_467: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_176, 1.0019569471624266);  squeeze_176 = None
    mul_468: "f32[512]" = torch.ops.aten.mul.Tensor(mul_467, 0.1);  mul_467 = None
    mul_469: "f32[512]" = torch.ops.aten.mul.Tensor(primals_380, 0.9)
    add_312: "f32[512]" = torch.ops.aten.add.Tensor(mul_468, mul_469);  mul_468 = mul_469 = None
    unsqueeze_232: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_117, -1)
    unsqueeze_233: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    mul_470: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_464, unsqueeze_233);  mul_464 = unsqueeze_233 = None
    unsqueeze_234: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_118, -1);  primals_118 = None
    unsqueeze_235: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    add_313: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_470, unsqueeze_235);  mul_470 = unsqueeze_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_58: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_313, 0)
    mul_471: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_313, 0.01)
    where_58: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_58, add_313, mul_471);  gt_58 = add_313 = mul_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_314: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(where_58, getitem_141)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_59: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(add_314, primals_194, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_315: "i64[]" = torch.ops.aten.add.Tensor(primals_381, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_59 = torch.ops.aten.var_mean.correction(convolution_59, [0, 2, 3], correction = 0, keepdim = True)
    getitem_146: "f32[1, 512, 1, 1]" = var_mean_59[0]
    getitem_147: "f32[1, 512, 1, 1]" = var_mean_59[1];  var_mean_59 = None
    add_316: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_146, 1e-05)
    rsqrt_59: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_316);  add_316 = None
    sub_59: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_59, getitem_147)
    mul_472: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = None
    squeeze_177: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_147, [0, 2, 3]);  getitem_147 = None
    squeeze_178: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_59, [0, 2, 3]);  rsqrt_59 = None
    mul_473: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_177, 0.1)
    mul_474: "f32[512]" = torch.ops.aten.mul.Tensor(primals_382, 0.9)
    add_317: "f32[512]" = torch.ops.aten.add.Tensor(mul_473, mul_474);  mul_473 = mul_474 = None
    squeeze_179: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_146, [0, 2, 3]);  getitem_146 = None
    mul_475: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_179, 1.0019569471624266);  squeeze_179 = None
    mul_476: "f32[512]" = torch.ops.aten.mul.Tensor(mul_475, 0.1);  mul_475 = None
    mul_477: "f32[512]" = torch.ops.aten.mul.Tensor(primals_383, 0.9)
    add_318: "f32[512]" = torch.ops.aten.add.Tensor(mul_476, mul_477);  mul_476 = mul_477 = None
    unsqueeze_236: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_119, -1)
    unsqueeze_237: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_478: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_472, unsqueeze_237);  mul_472 = unsqueeze_237 = None
    unsqueeze_238: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_120, -1);  primals_120 = None
    unsqueeze_239: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_319: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_478, unsqueeze_239);  mul_478 = unsqueeze_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_59: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_319, 0)
    mul_479: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_319, 0.01)
    where_59: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_59, add_319, mul_479);  gt_59 = add_319 = mul_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_60: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(where_59, primals_195, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_320: "i64[]" = torch.ops.aten.add.Tensor(primals_384, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_60 = torch.ops.aten.var_mean.correction(convolution_60, [0, 2, 3], correction = 0, keepdim = True)
    getitem_148: "f32[1, 512, 1, 1]" = var_mean_60[0]
    getitem_149: "f32[1, 512, 1, 1]" = var_mean_60[1];  var_mean_60 = None
    add_321: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_148, 1e-05)
    rsqrt_60: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_321);  add_321 = None
    sub_60: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_60, getitem_149)
    mul_480: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = None
    squeeze_180: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_149, [0, 2, 3]);  getitem_149 = None
    squeeze_181: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_60, [0, 2, 3]);  rsqrt_60 = None
    mul_481: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_180, 0.1)
    mul_482: "f32[512]" = torch.ops.aten.mul.Tensor(primals_385, 0.9)
    add_322: "f32[512]" = torch.ops.aten.add.Tensor(mul_481, mul_482);  mul_481 = mul_482 = None
    squeeze_182: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_148, [0, 2, 3]);  getitem_148 = None
    mul_483: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_182, 1.0019569471624266);  squeeze_182 = None
    mul_484: "f32[512]" = torch.ops.aten.mul.Tensor(mul_483, 0.1);  mul_483 = None
    mul_485: "f32[512]" = torch.ops.aten.mul.Tensor(primals_386, 0.9)
    add_323: "f32[512]" = torch.ops.aten.add.Tensor(mul_484, mul_485);  mul_484 = mul_485 = None
    unsqueeze_240: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_121, -1)
    unsqueeze_241: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    mul_486: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_480, unsqueeze_241);  mul_480 = unsqueeze_241 = None
    unsqueeze_242: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_122, -1);  primals_122 = None
    unsqueeze_243: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    add_324: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_486, unsqueeze_243);  mul_486 = unsqueeze_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_60: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_324, 0)
    mul_487: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_324, 0.01)
    where_60: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_60, add_324, mul_487);  gt_60 = add_324 = mul_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_325: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(where_60, add_314)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_61: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(add_325, primals_196, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_326: "i64[]" = torch.ops.aten.add.Tensor(primals_387, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_61 = torch.ops.aten.var_mean.correction(convolution_61, [0, 2, 3], correction = 0, keepdim = True)
    getitem_150: "f32[1, 512, 1, 1]" = var_mean_61[0]
    getitem_151: "f32[1, 512, 1, 1]" = var_mean_61[1];  var_mean_61 = None
    add_327: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_150, 1e-05)
    rsqrt_61: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_327);  add_327 = None
    sub_61: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_61, getitem_151)
    mul_488: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_61);  sub_61 = None
    squeeze_183: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_151, [0, 2, 3]);  getitem_151 = None
    squeeze_184: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_61, [0, 2, 3]);  rsqrt_61 = None
    mul_489: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_183, 0.1)
    mul_490: "f32[512]" = torch.ops.aten.mul.Tensor(primals_388, 0.9)
    add_328: "f32[512]" = torch.ops.aten.add.Tensor(mul_489, mul_490);  mul_489 = mul_490 = None
    squeeze_185: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_150, [0, 2, 3]);  getitem_150 = None
    mul_491: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_185, 1.0019569471624266);  squeeze_185 = None
    mul_492: "f32[512]" = torch.ops.aten.mul.Tensor(mul_491, 0.1);  mul_491 = None
    mul_493: "f32[512]" = torch.ops.aten.mul.Tensor(primals_389, 0.9)
    add_329: "f32[512]" = torch.ops.aten.add.Tensor(mul_492, mul_493);  mul_492 = mul_493 = None
    unsqueeze_244: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_123, -1)
    unsqueeze_245: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_494: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_488, unsqueeze_245);  mul_488 = unsqueeze_245 = None
    unsqueeze_246: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_124, -1);  primals_124 = None
    unsqueeze_247: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_330: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_494, unsqueeze_247);  mul_494 = unsqueeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_61: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_330, 0)
    mul_495: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_330, 0.01)
    where_61: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_61, add_330, mul_495);  gt_61 = add_330 = mul_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_62: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(where_61, primals_197, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_331: "i64[]" = torch.ops.aten.add.Tensor(primals_390, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_62 = torch.ops.aten.var_mean.correction(convolution_62, [0, 2, 3], correction = 0, keepdim = True)
    getitem_152: "f32[1, 512, 1, 1]" = var_mean_62[0]
    getitem_153: "f32[1, 512, 1, 1]" = var_mean_62[1];  var_mean_62 = None
    add_332: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_152, 1e-05)
    rsqrt_62: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_332);  add_332 = None
    sub_62: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_62, getitem_153)
    mul_496: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_62);  sub_62 = None
    squeeze_186: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_153, [0, 2, 3]);  getitem_153 = None
    squeeze_187: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_62, [0, 2, 3]);  rsqrt_62 = None
    mul_497: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_186, 0.1)
    mul_498: "f32[512]" = torch.ops.aten.mul.Tensor(primals_391, 0.9)
    add_333: "f32[512]" = torch.ops.aten.add.Tensor(mul_497, mul_498);  mul_497 = mul_498 = None
    squeeze_188: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_152, [0, 2, 3]);  getitem_152 = None
    mul_499: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_188, 1.0019569471624266);  squeeze_188 = None
    mul_500: "f32[512]" = torch.ops.aten.mul.Tensor(mul_499, 0.1);  mul_499 = None
    mul_501: "f32[512]" = torch.ops.aten.mul.Tensor(primals_392, 0.9)
    add_334: "f32[512]" = torch.ops.aten.add.Tensor(mul_500, mul_501);  mul_500 = mul_501 = None
    unsqueeze_248: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_125, -1)
    unsqueeze_249: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    mul_502: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_496, unsqueeze_249);  mul_496 = unsqueeze_249 = None
    unsqueeze_250: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_126, -1);  primals_126 = None
    unsqueeze_251: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    add_335: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_502, unsqueeze_251);  mul_502 = unsqueeze_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_62: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_335, 0)
    mul_503: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_335, 0.01)
    where_62: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_62, add_335, mul_503);  gt_62 = add_335 = mul_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_336: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(where_62, add_325)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_63: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(add_336, primals_198, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_337: "i64[]" = torch.ops.aten.add.Tensor(primals_393, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_63 = torch.ops.aten.var_mean.correction(convolution_63, [0, 2, 3], correction = 0, keepdim = True)
    getitem_154: "f32[1, 512, 1, 1]" = var_mean_63[0]
    getitem_155: "f32[1, 512, 1, 1]" = var_mean_63[1];  var_mean_63 = None
    add_338: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_154, 1e-05)
    rsqrt_63: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_338);  add_338 = None
    sub_63: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_63, getitem_155)
    mul_504: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_63);  sub_63 = None
    squeeze_189: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_155, [0, 2, 3]);  getitem_155 = None
    squeeze_190: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_63, [0, 2, 3]);  rsqrt_63 = None
    mul_505: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_189, 0.1)
    mul_506: "f32[512]" = torch.ops.aten.mul.Tensor(primals_394, 0.9)
    add_339: "f32[512]" = torch.ops.aten.add.Tensor(mul_505, mul_506);  mul_505 = mul_506 = None
    squeeze_191: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_154, [0, 2, 3]);  getitem_154 = None
    mul_507: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_191, 1.0019569471624266);  squeeze_191 = None
    mul_508: "f32[512]" = torch.ops.aten.mul.Tensor(mul_507, 0.1);  mul_507 = None
    mul_509: "f32[512]" = torch.ops.aten.mul.Tensor(primals_395, 0.9)
    add_340: "f32[512]" = torch.ops.aten.add.Tensor(mul_508, mul_509);  mul_508 = mul_509 = None
    unsqueeze_252: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_127, -1)
    unsqueeze_253: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_510: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_504, unsqueeze_253);  mul_504 = unsqueeze_253 = None
    unsqueeze_254: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_128, -1);  primals_128 = None
    unsqueeze_255: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_341: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_510, unsqueeze_255);  mul_510 = unsqueeze_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_63: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_341, 0)
    mul_511: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_341, 0.01)
    where_63: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_63, add_341, mul_511);  gt_63 = add_341 = mul_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_64: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(where_63, primals_199, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_342: "i64[]" = torch.ops.aten.add.Tensor(primals_396, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_64 = torch.ops.aten.var_mean.correction(convolution_64, [0, 2, 3], correction = 0, keepdim = True)
    getitem_156: "f32[1, 512, 1, 1]" = var_mean_64[0]
    getitem_157: "f32[1, 512, 1, 1]" = var_mean_64[1];  var_mean_64 = None
    add_343: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-05)
    rsqrt_64: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_343);  add_343 = None
    sub_64: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_64, getitem_157)
    mul_512: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_64);  sub_64 = None
    squeeze_192: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_157, [0, 2, 3]);  getitem_157 = None
    squeeze_193: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_64, [0, 2, 3]);  rsqrt_64 = None
    mul_513: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_192, 0.1)
    mul_514: "f32[512]" = torch.ops.aten.mul.Tensor(primals_397, 0.9)
    add_344: "f32[512]" = torch.ops.aten.add.Tensor(mul_513, mul_514);  mul_513 = mul_514 = None
    squeeze_194: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_156, [0, 2, 3]);  getitem_156 = None
    mul_515: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_194, 1.0019569471624266);  squeeze_194 = None
    mul_516: "f32[512]" = torch.ops.aten.mul.Tensor(mul_515, 0.1);  mul_515 = None
    mul_517: "f32[512]" = torch.ops.aten.mul.Tensor(primals_398, 0.9)
    add_345: "f32[512]" = torch.ops.aten.add.Tensor(mul_516, mul_517);  mul_516 = mul_517 = None
    unsqueeze_256: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_129, -1)
    unsqueeze_257: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    mul_518: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_512, unsqueeze_257);  mul_512 = unsqueeze_257 = None
    unsqueeze_258: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_130, -1);  primals_130 = None
    unsqueeze_259: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    add_346: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_518, unsqueeze_259);  mul_518 = unsqueeze_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_64: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_346, 0)
    mul_519: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_346, 0.01)
    where_64: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_64, add_346, mul_519);  gt_64 = add_346 = mul_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_347: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(where_64, add_336)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_65: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(add_347, primals_200, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_348: "i64[]" = torch.ops.aten.add.Tensor(primals_399, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_65 = torch.ops.aten.var_mean.correction(convolution_65, [0, 2, 3], correction = 0, keepdim = True)
    getitem_158: "f32[1, 512, 1, 1]" = var_mean_65[0]
    getitem_159: "f32[1, 512, 1, 1]" = var_mean_65[1];  var_mean_65 = None
    add_349: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-05)
    rsqrt_65: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_349);  add_349 = None
    sub_65: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_65, getitem_159)
    mul_520: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_65);  sub_65 = None
    squeeze_195: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_159, [0, 2, 3]);  getitem_159 = None
    squeeze_196: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_65, [0, 2, 3]);  rsqrt_65 = None
    mul_521: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_195, 0.1)
    mul_522: "f32[512]" = torch.ops.aten.mul.Tensor(primals_400, 0.9)
    add_350: "f32[512]" = torch.ops.aten.add.Tensor(mul_521, mul_522);  mul_521 = mul_522 = None
    squeeze_197: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_158, [0, 2, 3]);  getitem_158 = None
    mul_523: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_197, 1.0019569471624266);  squeeze_197 = None
    mul_524: "f32[512]" = torch.ops.aten.mul.Tensor(mul_523, 0.1);  mul_523 = None
    mul_525: "f32[512]" = torch.ops.aten.mul.Tensor(primals_401, 0.9)
    add_351: "f32[512]" = torch.ops.aten.add.Tensor(mul_524, mul_525);  mul_524 = mul_525 = None
    unsqueeze_260: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_131, -1)
    unsqueeze_261: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
    mul_526: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_520, unsqueeze_261);  mul_520 = unsqueeze_261 = None
    unsqueeze_262: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_132, -1);  primals_132 = None
    unsqueeze_263: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
    add_352: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_526, unsqueeze_263);  mul_526 = unsqueeze_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_65: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_352, 0)
    mul_527: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_352, 0.01)
    where_65: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_65, add_352, mul_527);  gt_65 = add_352 = mul_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:339, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
    split_with_sizes_14 = torch.ops.aten.split_with_sizes.default(where_56, [512, 512], 1)
    getitem_160: "f32[8, 512, 8, 8]" = split_with_sizes_14[0];  split_with_sizes_14 = None
    cat_4: "f32[8, 1024, 8, 8]" = torch.ops.aten.cat.default([getitem_160, where_65], 1);  getitem_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_66: "f32[8, 1024, 8, 8]" = torch.ops.aten.convolution.default(cat_4, primals_201, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_353: "i64[]" = torch.ops.aten.add.Tensor(primals_402, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_66 = torch.ops.aten.var_mean.correction(convolution_66, [0, 2, 3], correction = 0, keepdim = True)
    getitem_162: "f32[1, 1024, 1, 1]" = var_mean_66[0]
    getitem_163: "f32[1, 1024, 1, 1]" = var_mean_66[1];  var_mean_66 = None
    add_354: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_162, 1e-05)
    rsqrt_66: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_354);  add_354 = None
    sub_66: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_66, getitem_163)
    mul_528: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_66);  sub_66 = None
    squeeze_198: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_163, [0, 2, 3]);  getitem_163 = None
    squeeze_199: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_66, [0, 2, 3]);  rsqrt_66 = None
    mul_529: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_198, 0.1)
    mul_530: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_403, 0.9)
    add_355: "f32[1024]" = torch.ops.aten.add.Tensor(mul_529, mul_530);  mul_529 = mul_530 = None
    squeeze_200: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_162, [0, 2, 3]);  getitem_162 = None
    mul_531: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_200, 1.0019569471624266);  squeeze_200 = None
    mul_532: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_531, 0.1);  mul_531 = None
    mul_533: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_404, 0.9)
    add_356: "f32[1024]" = torch.ops.aten.add.Tensor(mul_532, mul_533);  mul_532 = mul_533 = None
    unsqueeze_264: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_133, -1)
    unsqueeze_265: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
    mul_534: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(mul_528, unsqueeze_265);  mul_528 = unsqueeze_265 = None
    unsqueeze_266: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_134, -1);  primals_134 = None
    unsqueeze_267: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
    add_357: "f32[8, 1024, 8, 8]" = torch.ops.aten.add.Tensor(mul_534, unsqueeze_267);  mul_534 = unsqueeze_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_66: "b8[8, 1024, 8, 8]" = torch.ops.aten.gt.Scalar(add_357, 0)
    mul_535: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(add_357, 0.01)
    where_66: "f32[8, 1024, 8, 8]" = torch.ops.aten.where.self(gt_66, add_357, mul_535);  gt_66 = add_357 = mul_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean: "f32[8, 1024, 1, 1]" = torch.ops.aten.mean.dim(where_66, [-1, -2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view: "f32[8, 1024]" = torch.ops.aten.view.default(mean, [8, 1024]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:131, code: x = self.drop(x)
    clone: "f32[8, 1024]" = torch.ops.aten.clone.default(view);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    permute: "f32[1024, 1000]" = torch.ops.aten.permute.default(primals_202, [1, 0]);  primals_202 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_203, clone, permute);  primals_203 = None
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
    expand: "f32[8, 1024, 8, 8]" = torch.ops.aten.expand.default(view_2, [8, 1024, 8, 8]);  view_2 = None
    div: "f32[8, 1024, 8, 8]" = torch.ops.aten.div.Scalar(expand, 64);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_68: "f32[8, 1024, 8, 8]" = torch.ops.aten.alias.default(where_66);  where_66 = None
    alias_69: "f32[8, 1024, 8, 8]" = torch.ops.aten.alias.default(alias_68);  alias_68 = None
    gt_67: "b8[8, 1024, 8, 8]" = torch.ops.aten.gt.Scalar(alias_69, 0);  alias_69 = None
    mul_536: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(div, 0.01)
    where_67: "f32[8, 1024, 8, 8]" = torch.ops.aten.where.self(gt_67, div, mul_536);  gt_67 = div = mul_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_268: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_198, 0);  squeeze_198 = None
    unsqueeze_269: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 2);  unsqueeze_268 = None
    unsqueeze_270: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 3);  unsqueeze_269 = None
    sum_2: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_67, [0, 2, 3])
    sub_67: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_270)
    mul_537: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(where_67, sub_67);  sub_67 = None
    sum_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_537, [0, 2, 3]);  mul_537 = None
    mul_538: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_2, 0.001953125)
    unsqueeze_271: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_538, 0);  mul_538 = None
    unsqueeze_272: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 2);  unsqueeze_271 = None
    unsqueeze_273: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 3);  unsqueeze_272 = None
    mul_539: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_3, 0.001953125)
    mul_540: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_199, squeeze_199)
    mul_541: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_539, mul_540);  mul_539 = mul_540 = None
    unsqueeze_274: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_541, 0);  mul_541 = None
    unsqueeze_275: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 2);  unsqueeze_274 = None
    unsqueeze_276: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 3);  unsqueeze_275 = None
    mul_542: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_199, primals_133);  primals_133 = None
    unsqueeze_277: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_542, 0);  mul_542 = None
    unsqueeze_278: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None
    unsqueeze_279: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 3);  unsqueeze_278 = None
    sub_68: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_270);  convolution_66 = unsqueeze_270 = None
    mul_543: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_276);  sub_68 = unsqueeze_276 = None
    sub_69: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(where_67, mul_543);  where_67 = mul_543 = None
    sub_70: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(sub_69, unsqueeze_273);  sub_69 = unsqueeze_273 = None
    mul_544: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_279);  sub_70 = unsqueeze_279 = None
    mul_545: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_199);  sum_3 = squeeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_544, cat_4, primals_201, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_544 = cat_4 = primals_201 = None
    getitem_164: "f32[8, 1024, 8, 8]" = convolution_backward[0]
    getitem_165: "f32[1024, 1024, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:339, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
    slice_1: "f32[8, 512, 8, 8]" = torch.ops.aten.slice.Tensor(getitem_164, 1, 0, 512)
    slice_2: "f32[8, 512, 8, 8]" = torch.ops.aten.slice.Tensor(getitem_164, 1, 512, 1024);  getitem_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_71: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(where_65);  where_65 = None
    alias_72: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(alias_71);  alias_71 = None
    gt_68: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(alias_72, 0);  alias_72 = None
    mul_546: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(slice_2, 0.01)
    where_68: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_68, slice_2, mul_546);  gt_68 = slice_2 = mul_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_280: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_195, 0);  squeeze_195 = None
    unsqueeze_281: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 2);  unsqueeze_280 = None
    unsqueeze_282: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 3);  unsqueeze_281 = None
    sum_4: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_68, [0, 2, 3])
    sub_71: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_282)
    mul_547: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(where_68, sub_71);  sub_71 = None
    sum_5: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_547, [0, 2, 3]);  mul_547 = None
    mul_548: "f32[512]" = torch.ops.aten.mul.Tensor(sum_4, 0.001953125)
    unsqueeze_283: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_548, 0);  mul_548 = None
    unsqueeze_284: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 2);  unsqueeze_283 = None
    unsqueeze_285: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 3);  unsqueeze_284 = None
    mul_549: "f32[512]" = torch.ops.aten.mul.Tensor(sum_5, 0.001953125)
    mul_550: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_196, squeeze_196)
    mul_551: "f32[512]" = torch.ops.aten.mul.Tensor(mul_549, mul_550);  mul_549 = mul_550 = None
    unsqueeze_286: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_551, 0);  mul_551 = None
    unsqueeze_287: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 2);  unsqueeze_286 = None
    unsqueeze_288: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 3);  unsqueeze_287 = None
    mul_552: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_196, primals_131);  primals_131 = None
    unsqueeze_289: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_552, 0);  mul_552 = None
    unsqueeze_290: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 2);  unsqueeze_289 = None
    unsqueeze_291: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 3);  unsqueeze_290 = None
    sub_72: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_282);  convolution_65 = unsqueeze_282 = None
    mul_553: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_288);  sub_72 = unsqueeze_288 = None
    sub_73: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(where_68, mul_553);  where_68 = mul_553 = None
    sub_74: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_73, unsqueeze_285);  sub_73 = unsqueeze_285 = None
    mul_554: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_291);  sub_74 = unsqueeze_291 = None
    mul_555: "f32[512]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_196);  sum_5 = squeeze_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_554, add_347, primals_200, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_554 = add_347 = primals_200 = None
    getitem_167: "f32[8, 512, 8, 8]" = convolution_backward_1[0]
    getitem_168: "f32[512, 512, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_74: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(where_64);  where_64 = None
    alias_75: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(alias_74);  alias_74 = None
    gt_69: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(alias_75, 0);  alias_75 = None
    mul_556: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_167, 0.01)
    where_69: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_69, getitem_167, mul_556);  gt_69 = mul_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_292: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_192, 0);  squeeze_192 = None
    unsqueeze_293: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 2);  unsqueeze_292 = None
    unsqueeze_294: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 3);  unsqueeze_293 = None
    sum_6: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_69, [0, 2, 3])
    sub_75: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_294)
    mul_557: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(where_69, sub_75);  sub_75 = None
    sum_7: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_557, [0, 2, 3]);  mul_557 = None
    mul_558: "f32[512]" = torch.ops.aten.mul.Tensor(sum_6, 0.001953125)
    unsqueeze_295: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_558, 0);  mul_558 = None
    unsqueeze_296: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 2);  unsqueeze_295 = None
    unsqueeze_297: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 3);  unsqueeze_296 = None
    mul_559: "f32[512]" = torch.ops.aten.mul.Tensor(sum_7, 0.001953125)
    mul_560: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_193, squeeze_193)
    mul_561: "f32[512]" = torch.ops.aten.mul.Tensor(mul_559, mul_560);  mul_559 = mul_560 = None
    unsqueeze_298: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_561, 0);  mul_561 = None
    unsqueeze_299: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 2);  unsqueeze_298 = None
    unsqueeze_300: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 3);  unsqueeze_299 = None
    mul_562: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_193, primals_129);  primals_129 = None
    unsqueeze_301: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_562, 0);  mul_562 = None
    unsqueeze_302: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None
    unsqueeze_303: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 3);  unsqueeze_302 = None
    sub_76: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_294);  convolution_64 = unsqueeze_294 = None
    mul_563: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_300);  sub_76 = unsqueeze_300 = None
    sub_77: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(where_69, mul_563);  where_69 = mul_563 = None
    sub_78: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_77, unsqueeze_297);  sub_77 = unsqueeze_297 = None
    mul_564: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_303);  sub_78 = unsqueeze_303 = None
    mul_565: "f32[512]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_193);  sum_7 = squeeze_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_564, where_63, primals_199, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_564 = primals_199 = None
    getitem_170: "f32[8, 512, 8, 8]" = convolution_backward_2[0]
    getitem_171: "f32[512, 512, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_77: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(where_63);  where_63 = None
    alias_78: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(alias_77);  alias_77 = None
    gt_70: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(alias_78, 0);  alias_78 = None
    mul_566: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_170, 0.01)
    where_70: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_70, getitem_170, mul_566);  gt_70 = getitem_170 = mul_566 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_304: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_189, 0);  squeeze_189 = None
    unsqueeze_305: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 2);  unsqueeze_304 = None
    unsqueeze_306: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 3);  unsqueeze_305 = None
    sum_8: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_70, [0, 2, 3])
    sub_79: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_306)
    mul_567: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(where_70, sub_79);  sub_79 = None
    sum_9: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_567, [0, 2, 3]);  mul_567 = None
    mul_568: "f32[512]" = torch.ops.aten.mul.Tensor(sum_8, 0.001953125)
    unsqueeze_307: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_568, 0);  mul_568 = None
    unsqueeze_308: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 2);  unsqueeze_307 = None
    unsqueeze_309: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 3);  unsqueeze_308 = None
    mul_569: "f32[512]" = torch.ops.aten.mul.Tensor(sum_9, 0.001953125)
    mul_570: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_190, squeeze_190)
    mul_571: "f32[512]" = torch.ops.aten.mul.Tensor(mul_569, mul_570);  mul_569 = mul_570 = None
    unsqueeze_310: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_571, 0);  mul_571 = None
    unsqueeze_311: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 2);  unsqueeze_310 = None
    unsqueeze_312: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 3);  unsqueeze_311 = None
    mul_572: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_190, primals_127);  primals_127 = None
    unsqueeze_313: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_572, 0);  mul_572 = None
    unsqueeze_314: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 2);  unsqueeze_313 = None
    unsqueeze_315: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 3);  unsqueeze_314 = None
    sub_80: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_306);  convolution_63 = unsqueeze_306 = None
    mul_573: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_312);  sub_80 = unsqueeze_312 = None
    sub_81: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(where_70, mul_573);  where_70 = mul_573 = None
    sub_82: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_81, unsqueeze_309);  sub_81 = unsqueeze_309 = None
    mul_574: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_315);  sub_82 = unsqueeze_315 = None
    mul_575: "f32[512]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_190);  sum_9 = squeeze_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_574, add_336, primals_198, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_574 = add_336 = primals_198 = None
    getitem_173: "f32[8, 512, 8, 8]" = convolution_backward_3[0]
    getitem_174: "f32[512, 512, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_358: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(getitem_167, getitem_173);  getitem_167 = getitem_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_80: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(where_62);  where_62 = None
    alias_81: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(alias_80);  alias_80 = None
    gt_71: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(alias_81, 0);  alias_81 = None
    mul_576: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_358, 0.01)
    where_71: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_71, add_358, mul_576);  gt_71 = mul_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_316: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_186, 0);  squeeze_186 = None
    unsqueeze_317: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, 2);  unsqueeze_316 = None
    unsqueeze_318: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 3);  unsqueeze_317 = None
    sum_10: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_71, [0, 2, 3])
    sub_83: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_318)
    mul_577: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(where_71, sub_83);  sub_83 = None
    sum_11: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_577, [0, 2, 3]);  mul_577 = None
    mul_578: "f32[512]" = torch.ops.aten.mul.Tensor(sum_10, 0.001953125)
    unsqueeze_319: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_578, 0);  mul_578 = None
    unsqueeze_320: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 2);  unsqueeze_319 = None
    unsqueeze_321: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 3);  unsqueeze_320 = None
    mul_579: "f32[512]" = torch.ops.aten.mul.Tensor(sum_11, 0.001953125)
    mul_580: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_187, squeeze_187)
    mul_581: "f32[512]" = torch.ops.aten.mul.Tensor(mul_579, mul_580);  mul_579 = mul_580 = None
    unsqueeze_322: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_581, 0);  mul_581 = None
    unsqueeze_323: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 2);  unsqueeze_322 = None
    unsqueeze_324: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 3);  unsqueeze_323 = None
    mul_582: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_187, primals_125);  primals_125 = None
    unsqueeze_325: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_582, 0);  mul_582 = None
    unsqueeze_326: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None
    unsqueeze_327: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 3);  unsqueeze_326 = None
    sub_84: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_318);  convolution_62 = unsqueeze_318 = None
    mul_583: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_324);  sub_84 = unsqueeze_324 = None
    sub_85: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(where_71, mul_583);  where_71 = mul_583 = None
    sub_86: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_85, unsqueeze_321);  sub_85 = unsqueeze_321 = None
    mul_584: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_327);  sub_86 = unsqueeze_327 = None
    mul_585: "f32[512]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_187);  sum_11 = squeeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_584, where_61, primals_197, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_584 = primals_197 = None
    getitem_176: "f32[8, 512, 8, 8]" = convolution_backward_4[0]
    getitem_177: "f32[512, 512, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_83: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(where_61);  where_61 = None
    alias_84: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(alias_83);  alias_83 = None
    gt_72: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(alias_84, 0);  alias_84 = None
    mul_586: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_176, 0.01)
    where_72: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_72, getitem_176, mul_586);  gt_72 = getitem_176 = mul_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_328: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_183, 0);  squeeze_183 = None
    unsqueeze_329: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 2);  unsqueeze_328 = None
    unsqueeze_330: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 3);  unsqueeze_329 = None
    sum_12: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_72, [0, 2, 3])
    sub_87: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_330)
    mul_587: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(where_72, sub_87);  sub_87 = None
    sum_13: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_587, [0, 2, 3]);  mul_587 = None
    mul_588: "f32[512]" = torch.ops.aten.mul.Tensor(sum_12, 0.001953125)
    unsqueeze_331: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_588, 0);  mul_588 = None
    unsqueeze_332: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 2);  unsqueeze_331 = None
    unsqueeze_333: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 3);  unsqueeze_332 = None
    mul_589: "f32[512]" = torch.ops.aten.mul.Tensor(sum_13, 0.001953125)
    mul_590: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_184, squeeze_184)
    mul_591: "f32[512]" = torch.ops.aten.mul.Tensor(mul_589, mul_590);  mul_589 = mul_590 = None
    unsqueeze_334: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_591, 0);  mul_591 = None
    unsqueeze_335: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 2);  unsqueeze_334 = None
    unsqueeze_336: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 3);  unsqueeze_335 = None
    mul_592: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_184, primals_123);  primals_123 = None
    unsqueeze_337: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_592, 0);  mul_592 = None
    unsqueeze_338: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 2);  unsqueeze_337 = None
    unsqueeze_339: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 3);  unsqueeze_338 = None
    sub_88: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_330);  convolution_61 = unsqueeze_330 = None
    mul_593: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_336);  sub_88 = unsqueeze_336 = None
    sub_89: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(where_72, mul_593);  where_72 = mul_593 = None
    sub_90: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_89, unsqueeze_333);  sub_89 = unsqueeze_333 = None
    mul_594: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_339);  sub_90 = unsqueeze_339 = None
    mul_595: "f32[512]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_184);  sum_13 = squeeze_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_594, add_325, primals_196, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_594 = add_325 = primals_196 = None
    getitem_179: "f32[8, 512, 8, 8]" = convolution_backward_5[0]
    getitem_180: "f32[512, 512, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_359: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(add_358, getitem_179);  add_358 = getitem_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_86: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(where_60);  where_60 = None
    alias_87: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(alias_86);  alias_86 = None
    gt_73: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(alias_87, 0);  alias_87 = None
    mul_596: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_359, 0.01)
    where_73: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_73, add_359, mul_596);  gt_73 = mul_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_340: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_180, 0);  squeeze_180 = None
    unsqueeze_341: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 2);  unsqueeze_340 = None
    unsqueeze_342: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 3);  unsqueeze_341 = None
    sum_14: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_73, [0, 2, 3])
    sub_91: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_342)
    mul_597: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(where_73, sub_91);  sub_91 = None
    sum_15: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_597, [0, 2, 3]);  mul_597 = None
    mul_598: "f32[512]" = torch.ops.aten.mul.Tensor(sum_14, 0.001953125)
    unsqueeze_343: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_598, 0);  mul_598 = None
    unsqueeze_344: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 2);  unsqueeze_343 = None
    unsqueeze_345: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 3);  unsqueeze_344 = None
    mul_599: "f32[512]" = torch.ops.aten.mul.Tensor(sum_15, 0.001953125)
    mul_600: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
    mul_601: "f32[512]" = torch.ops.aten.mul.Tensor(mul_599, mul_600);  mul_599 = mul_600 = None
    unsqueeze_346: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_601, 0);  mul_601 = None
    unsqueeze_347: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 2);  unsqueeze_346 = None
    unsqueeze_348: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 3);  unsqueeze_347 = None
    mul_602: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_181, primals_121);  primals_121 = None
    unsqueeze_349: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_602, 0);  mul_602 = None
    unsqueeze_350: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None
    unsqueeze_351: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 3);  unsqueeze_350 = None
    sub_92: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_342);  convolution_60 = unsqueeze_342 = None
    mul_603: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_348);  sub_92 = unsqueeze_348 = None
    sub_93: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(where_73, mul_603);  where_73 = mul_603 = None
    sub_94: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_93, unsqueeze_345);  sub_93 = unsqueeze_345 = None
    mul_604: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_351);  sub_94 = unsqueeze_351 = None
    mul_605: "f32[512]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_181);  sum_15 = squeeze_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_604, where_59, primals_195, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_604 = primals_195 = None
    getitem_182: "f32[8, 512, 8, 8]" = convolution_backward_6[0]
    getitem_183: "f32[512, 512, 3, 3]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_89: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(where_59);  where_59 = None
    alias_90: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(alias_89);  alias_89 = None
    gt_74: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(alias_90, 0);  alias_90 = None
    mul_606: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_182, 0.01)
    where_74: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_74, getitem_182, mul_606);  gt_74 = getitem_182 = mul_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_352: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_177, 0);  squeeze_177 = None
    unsqueeze_353: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 2);  unsqueeze_352 = None
    unsqueeze_354: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 3);  unsqueeze_353 = None
    sum_16: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_74, [0, 2, 3])
    sub_95: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_354)
    mul_607: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(where_74, sub_95);  sub_95 = None
    sum_17: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_607, [0, 2, 3]);  mul_607 = None
    mul_608: "f32[512]" = torch.ops.aten.mul.Tensor(sum_16, 0.001953125)
    unsqueeze_355: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_608, 0);  mul_608 = None
    unsqueeze_356: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 2);  unsqueeze_355 = None
    unsqueeze_357: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 3);  unsqueeze_356 = None
    mul_609: "f32[512]" = torch.ops.aten.mul.Tensor(sum_17, 0.001953125)
    mul_610: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_178, squeeze_178)
    mul_611: "f32[512]" = torch.ops.aten.mul.Tensor(mul_609, mul_610);  mul_609 = mul_610 = None
    unsqueeze_358: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_611, 0);  mul_611 = None
    unsqueeze_359: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 2);  unsqueeze_358 = None
    unsqueeze_360: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 3);  unsqueeze_359 = None
    mul_612: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_178, primals_119);  primals_119 = None
    unsqueeze_361: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_612, 0);  mul_612 = None
    unsqueeze_362: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
    unsqueeze_363: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 3);  unsqueeze_362 = None
    sub_96: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_354);  convolution_59 = unsqueeze_354 = None
    mul_613: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_360);  sub_96 = unsqueeze_360 = None
    sub_97: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(where_74, mul_613);  where_74 = mul_613 = None
    sub_98: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_97, unsqueeze_357);  sub_97 = unsqueeze_357 = None
    mul_614: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_363);  sub_98 = unsqueeze_363 = None
    mul_615: "f32[512]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_178);  sum_17 = squeeze_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_614, add_314, primals_194, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_614 = add_314 = primals_194 = None
    getitem_185: "f32[8, 512, 8, 8]" = convolution_backward_7[0]
    getitem_186: "f32[512, 512, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_360: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(add_359, getitem_185);  add_359 = getitem_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_92: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(where_58);  where_58 = None
    alias_93: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(alias_92);  alias_92 = None
    gt_75: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(alias_93, 0);  alias_93 = None
    mul_616: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_360, 0.01)
    where_75: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_75, add_360, mul_616);  gt_75 = mul_616 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_364: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_174, 0);  squeeze_174 = None
    unsqueeze_365: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 2);  unsqueeze_364 = None
    unsqueeze_366: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 3);  unsqueeze_365 = None
    sum_18: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_75, [0, 2, 3])
    sub_99: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_366)
    mul_617: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(where_75, sub_99);  sub_99 = None
    sum_19: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_617, [0, 2, 3]);  mul_617 = None
    mul_618: "f32[512]" = torch.ops.aten.mul.Tensor(sum_18, 0.001953125)
    unsqueeze_367: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_618, 0);  mul_618 = None
    unsqueeze_368: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
    unsqueeze_369: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 3);  unsqueeze_368 = None
    mul_619: "f32[512]" = torch.ops.aten.mul.Tensor(sum_19, 0.001953125)
    mul_620: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
    mul_621: "f32[512]" = torch.ops.aten.mul.Tensor(mul_619, mul_620);  mul_619 = mul_620 = None
    unsqueeze_370: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_621, 0);  mul_621 = None
    unsqueeze_371: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 2);  unsqueeze_370 = None
    unsqueeze_372: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 3);  unsqueeze_371 = None
    mul_622: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_175, primals_117);  primals_117 = None
    unsqueeze_373: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_622, 0);  mul_622 = None
    unsqueeze_374: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
    unsqueeze_375: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 3);  unsqueeze_374 = None
    sub_100: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_366);  convolution_58 = unsqueeze_366 = None
    mul_623: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_372);  sub_100 = unsqueeze_372 = None
    sub_101: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(where_75, mul_623);  where_75 = mul_623 = None
    sub_102: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_101, unsqueeze_369);  sub_101 = unsqueeze_369 = None
    mul_624: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_375);  sub_102 = unsqueeze_375 = None
    mul_625: "f32[512]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_175);  sum_19 = squeeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_624, where_57, primals_193, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_624 = primals_193 = None
    getitem_188: "f32[8, 512, 8, 8]" = convolution_backward_8[0]
    getitem_189: "f32[512, 512, 3, 3]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_95: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(where_57);  where_57 = None
    alias_96: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(alias_95);  alias_95 = None
    gt_76: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(alias_96, 0);  alias_96 = None
    mul_626: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_188, 0.01)
    where_76: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_76, getitem_188, mul_626);  gt_76 = getitem_188 = mul_626 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_376: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_171, 0);  squeeze_171 = None
    unsqueeze_377: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 2);  unsqueeze_376 = None
    unsqueeze_378: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 3);  unsqueeze_377 = None
    sum_20: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_76, [0, 2, 3])
    sub_103: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_378)
    mul_627: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(where_76, sub_103);  sub_103 = None
    sum_21: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_627, [0, 2, 3]);  mul_627 = None
    mul_628: "f32[512]" = torch.ops.aten.mul.Tensor(sum_20, 0.001953125)
    unsqueeze_379: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_628, 0);  mul_628 = None
    unsqueeze_380: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
    unsqueeze_381: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 3);  unsqueeze_380 = None
    mul_629: "f32[512]" = torch.ops.aten.mul.Tensor(sum_21, 0.001953125)
    mul_630: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
    mul_631: "f32[512]" = torch.ops.aten.mul.Tensor(mul_629, mul_630);  mul_629 = mul_630 = None
    unsqueeze_382: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_631, 0);  mul_631 = None
    unsqueeze_383: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 2);  unsqueeze_382 = None
    unsqueeze_384: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 3);  unsqueeze_383 = None
    mul_632: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_172, primals_115);  primals_115 = None
    unsqueeze_385: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_632, 0);  mul_632 = None
    unsqueeze_386: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
    unsqueeze_387: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
    sub_104: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_378);  convolution_57 = unsqueeze_378 = None
    mul_633: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_384);  sub_104 = unsqueeze_384 = None
    sub_105: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(where_76, mul_633);  where_76 = mul_633 = None
    sub_106: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_105, unsqueeze_381);  sub_105 = unsqueeze_381 = None
    mul_634: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_387);  sub_106 = unsqueeze_387 = None
    mul_635: "f32[512]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_172);  sum_21 = squeeze_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_634, getitem_141, primals_192, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_634 = getitem_141 = primals_192 = None
    getitem_191: "f32[8, 512, 8, 8]" = convolution_backward_9[0]
    getitem_192: "f32[512, 512, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_361: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(add_360, getitem_191);  add_360 = getitem_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:336, code: xs, xb = x.split(self.expand_chs // 2, dim=1)
    cat_5: "f32[8, 1024, 8, 8]" = torch.ops.aten.cat.default([slice_1, add_361], 1);  slice_1 = add_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_98: "f32[8, 1024, 8, 8]" = torch.ops.aten.alias.default(where_56);  where_56 = None
    alias_99: "f32[8, 1024, 8, 8]" = torch.ops.aten.alias.default(alias_98);  alias_98 = None
    gt_77: "b8[8, 1024, 8, 8]" = torch.ops.aten.gt.Scalar(alias_99, 0);  alias_99 = None
    mul_636: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(cat_5, 0.01)
    where_77: "f32[8, 1024, 8, 8]" = torch.ops.aten.where.self(gt_77, cat_5, mul_636);  gt_77 = cat_5 = mul_636 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_388: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_168, 0);  squeeze_168 = None
    unsqueeze_389: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 2);  unsqueeze_388 = None
    unsqueeze_390: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 3);  unsqueeze_389 = None
    sum_22: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_77, [0, 2, 3])
    sub_107: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_390)
    mul_637: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(where_77, sub_107);  sub_107 = None
    sum_23: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_637, [0, 2, 3]);  mul_637 = None
    mul_638: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_22, 0.001953125)
    unsqueeze_391: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_638, 0);  mul_638 = None
    unsqueeze_392: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
    unsqueeze_393: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 3);  unsqueeze_392 = None
    mul_639: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_23, 0.001953125)
    mul_640: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
    mul_641: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_639, mul_640);  mul_639 = mul_640 = None
    unsqueeze_394: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_641, 0);  mul_641 = None
    unsqueeze_395: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 2);  unsqueeze_394 = None
    unsqueeze_396: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 3);  unsqueeze_395 = None
    mul_642: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_169, primals_113);  primals_113 = None
    unsqueeze_397: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_642, 0);  mul_642 = None
    unsqueeze_398: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
    unsqueeze_399: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 3);  unsqueeze_398 = None
    sub_108: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_390);  convolution_56 = unsqueeze_390 = None
    mul_643: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_396);  sub_108 = unsqueeze_396 = None
    sub_109: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(where_77, mul_643);  where_77 = mul_643 = None
    sub_110: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(sub_109, unsqueeze_393);  sub_109 = unsqueeze_393 = None
    mul_644: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_399);  sub_110 = unsqueeze_399 = None
    mul_645: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_169);  sum_23 = squeeze_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_644, where_55, primals_191, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_644 = primals_191 = None
    getitem_194: "f32[8, 1024, 8, 8]" = convolution_backward_10[0]
    getitem_195: "f32[1024, 1024, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_101: "f32[8, 1024, 8, 8]" = torch.ops.aten.alias.default(where_55);  where_55 = None
    alias_102: "f32[8, 1024, 8, 8]" = torch.ops.aten.alias.default(alias_101);  alias_101 = None
    gt_78: "b8[8, 1024, 8, 8]" = torch.ops.aten.gt.Scalar(alias_102, 0);  alias_102 = None
    mul_646: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_194, 0.01)
    where_78: "f32[8, 1024, 8, 8]" = torch.ops.aten.where.self(gt_78, getitem_194, mul_646);  gt_78 = getitem_194 = mul_646 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_400: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_165, 0);  squeeze_165 = None
    unsqueeze_401: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 2);  unsqueeze_400 = None
    unsqueeze_402: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 3);  unsqueeze_401 = None
    sum_24: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_78, [0, 2, 3])
    sub_111: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_402)
    mul_647: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(where_78, sub_111);  sub_111 = None
    sum_25: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_647, [0, 2, 3]);  mul_647 = None
    mul_648: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_24, 0.001953125)
    unsqueeze_403: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_648, 0);  mul_648 = None
    unsqueeze_404: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 2);  unsqueeze_403 = None
    unsqueeze_405: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 3);  unsqueeze_404 = None
    mul_649: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_25, 0.001953125)
    mul_650: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
    mul_651: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_649, mul_650);  mul_649 = mul_650 = None
    unsqueeze_406: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_651, 0);  mul_651 = None
    unsqueeze_407: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 2);  unsqueeze_406 = None
    unsqueeze_408: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 3);  unsqueeze_407 = None
    mul_652: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_166, primals_111);  primals_111 = None
    unsqueeze_409: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_652, 0);  mul_652 = None
    unsqueeze_410: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 2);  unsqueeze_409 = None
    unsqueeze_411: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 3);  unsqueeze_410 = None
    sub_112: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_402);  convolution_55 = unsqueeze_402 = None
    mul_653: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_408);  sub_112 = unsqueeze_408 = None
    sub_113: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(where_78, mul_653);  where_78 = mul_653 = None
    sub_114: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(sub_113, unsqueeze_405);  sub_113 = unsqueeze_405 = None
    mul_654: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_411);  sub_114 = unsqueeze_411 = None
    mul_655: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_166);  sum_25 = squeeze_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:126, code: x = self.conv(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_654, where_54, primals_190, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_654 = primals_190 = None
    getitem_197: "f32[8, 512, 16, 16]" = convolution_backward_11[0]
    getitem_198: "f32[1024, 512, 3, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_104: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(where_54);  where_54 = None
    alias_105: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(alias_104);  alias_104 = None
    gt_79: "b8[8, 512, 16, 16]" = torch.ops.aten.gt.Scalar(alias_105, 0);  alias_105 = None
    mul_656: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_197, 0.01)
    where_79: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(gt_79, getitem_197, mul_656);  gt_79 = getitem_197 = mul_656 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_412: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_162, 0);  squeeze_162 = None
    unsqueeze_413: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 2);  unsqueeze_412 = None
    unsqueeze_414: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 3);  unsqueeze_413 = None
    sum_26: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_79, [0, 2, 3])
    sub_115: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_414)
    mul_657: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(where_79, sub_115);  sub_115 = None
    sum_27: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_657, [0, 2, 3]);  mul_657 = None
    mul_658: "f32[512]" = torch.ops.aten.mul.Tensor(sum_26, 0.00048828125)
    unsqueeze_415: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_658, 0);  mul_658 = None
    unsqueeze_416: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 2);  unsqueeze_415 = None
    unsqueeze_417: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 3);  unsqueeze_416 = None
    mul_659: "f32[512]" = torch.ops.aten.mul.Tensor(sum_27, 0.00048828125)
    mul_660: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
    mul_661: "f32[512]" = torch.ops.aten.mul.Tensor(mul_659, mul_660);  mul_659 = mul_660 = None
    unsqueeze_418: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_661, 0);  mul_661 = None
    unsqueeze_419: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 2);  unsqueeze_418 = None
    unsqueeze_420: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 3);  unsqueeze_419 = None
    mul_662: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_163, primals_109);  primals_109 = None
    unsqueeze_421: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_662, 0);  mul_662 = None
    unsqueeze_422: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None
    unsqueeze_423: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 3);  unsqueeze_422 = None
    sub_116: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_414);  convolution_54 = unsqueeze_414 = None
    mul_663: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_420);  sub_116 = unsqueeze_420 = None
    sub_117: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(where_79, mul_663);  where_79 = mul_663 = None
    sub_118: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_117, unsqueeze_417);  sub_117 = unsqueeze_417 = None
    mul_664: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_423);  sub_118 = unsqueeze_423 = None
    mul_665: "f32[512]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_163);  sum_27 = squeeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_664, cat_3, primals_189, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_664 = cat_3 = primals_189 = None
    getitem_200: "f32[8, 512, 16, 16]" = convolution_backward_12[0]
    getitem_201: "f32[512, 512, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:339, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
    slice_3: "f32[8, 256, 16, 16]" = torch.ops.aten.slice.Tensor(getitem_200, 1, 0, 256)
    slice_4: "f32[8, 256, 16, 16]" = torch.ops.aten.slice.Tensor(getitem_200, 1, 256, 512);  getitem_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_107: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(where_53);  where_53 = None
    alias_108: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_107);  alias_107 = None
    gt_80: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(alias_108, 0);  alias_108 = None
    mul_666: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(slice_4, 0.01)
    where_80: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_80, slice_4, mul_666);  gt_80 = slice_4 = mul_666 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_424: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_159, 0);  squeeze_159 = None
    unsqueeze_425: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 2);  unsqueeze_424 = None
    unsqueeze_426: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 3);  unsqueeze_425 = None
    sum_28: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_80, [0, 2, 3])
    sub_119: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_426)
    mul_667: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_80, sub_119);  sub_119 = None
    sum_29: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_667, [0, 2, 3]);  mul_667 = None
    mul_668: "f32[256]" = torch.ops.aten.mul.Tensor(sum_28, 0.00048828125)
    unsqueeze_427: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_668, 0);  mul_668 = None
    unsqueeze_428: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 2);  unsqueeze_427 = None
    unsqueeze_429: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 3);  unsqueeze_428 = None
    mul_669: "f32[256]" = torch.ops.aten.mul.Tensor(sum_29, 0.00048828125)
    mul_670: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
    mul_671: "f32[256]" = torch.ops.aten.mul.Tensor(mul_669, mul_670);  mul_669 = mul_670 = None
    unsqueeze_430: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_671, 0);  mul_671 = None
    unsqueeze_431: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 2);  unsqueeze_430 = None
    unsqueeze_432: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 3);  unsqueeze_431 = None
    mul_672: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_160, primals_107);  primals_107 = None
    unsqueeze_433: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_672, 0);  mul_672 = None
    unsqueeze_434: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
    unsqueeze_435: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 3);  unsqueeze_434 = None
    sub_120: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_426);  convolution_53 = unsqueeze_426 = None
    mul_673: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_432);  sub_120 = unsqueeze_432 = None
    sub_121: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_80, mul_673);  where_80 = mul_673 = None
    sub_122: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_121, unsqueeze_429);  sub_121 = unsqueeze_429 = None
    mul_674: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_435);  sub_122 = unsqueeze_435 = None
    mul_675: "f32[256]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_160);  sum_29 = squeeze_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_674, add_283, primals_188, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_674 = add_283 = primals_188 = None
    getitem_203: "f32[8, 256, 16, 16]" = convolution_backward_13[0]
    getitem_204: "f32[256, 256, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_110: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(where_52);  where_52 = None
    alias_111: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_110);  alias_110 = None
    gt_81: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(alias_111, 0);  alias_111 = None
    mul_676: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_203, 0.01)
    where_81: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_81, getitem_203, mul_676);  gt_81 = mul_676 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_436: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_156, 0);  squeeze_156 = None
    unsqueeze_437: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 2);  unsqueeze_436 = None
    unsqueeze_438: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 3);  unsqueeze_437 = None
    sum_30: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_81, [0, 2, 3])
    sub_123: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_438)
    mul_677: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_81, sub_123);  sub_123 = None
    sum_31: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_677, [0, 2, 3]);  mul_677 = None
    mul_678: "f32[256]" = torch.ops.aten.mul.Tensor(sum_30, 0.00048828125)
    unsqueeze_439: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_678, 0);  mul_678 = None
    unsqueeze_440: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 2);  unsqueeze_439 = None
    unsqueeze_441: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 3);  unsqueeze_440 = None
    mul_679: "f32[256]" = torch.ops.aten.mul.Tensor(sum_31, 0.00048828125)
    mul_680: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
    mul_681: "f32[256]" = torch.ops.aten.mul.Tensor(mul_679, mul_680);  mul_679 = mul_680 = None
    unsqueeze_442: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_681, 0);  mul_681 = None
    unsqueeze_443: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 2);  unsqueeze_442 = None
    unsqueeze_444: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 3);  unsqueeze_443 = None
    mul_682: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_157, primals_105);  primals_105 = None
    unsqueeze_445: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_682, 0);  mul_682 = None
    unsqueeze_446: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
    unsqueeze_447: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 3);  unsqueeze_446 = None
    sub_124: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_438);  convolution_52 = unsqueeze_438 = None
    mul_683: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_444);  sub_124 = unsqueeze_444 = None
    sub_125: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_81, mul_683);  where_81 = mul_683 = None
    sub_126: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_125, unsqueeze_441);  sub_125 = unsqueeze_441 = None
    mul_684: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_447);  sub_126 = unsqueeze_447 = None
    mul_685: "f32[256]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_157);  sum_31 = squeeze_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_684, where_51, primals_187, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_684 = primals_187 = None
    getitem_206: "f32[8, 256, 16, 16]" = convolution_backward_14[0]
    getitem_207: "f32[256, 256, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_113: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(where_51);  where_51 = None
    alias_114: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_113);  alias_113 = None
    gt_82: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(alias_114, 0);  alias_114 = None
    mul_686: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_206, 0.01)
    where_82: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_82, getitem_206, mul_686);  gt_82 = getitem_206 = mul_686 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_448: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
    unsqueeze_449: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, 2);  unsqueeze_448 = None
    unsqueeze_450: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 3);  unsqueeze_449 = None
    sum_32: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_82, [0, 2, 3])
    sub_127: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_450)
    mul_687: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_82, sub_127);  sub_127 = None
    sum_33: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_687, [0, 2, 3]);  mul_687 = None
    mul_688: "f32[256]" = torch.ops.aten.mul.Tensor(sum_32, 0.00048828125)
    unsqueeze_451: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_688, 0);  mul_688 = None
    unsqueeze_452: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 2);  unsqueeze_451 = None
    unsqueeze_453: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 3);  unsqueeze_452 = None
    mul_689: "f32[256]" = torch.ops.aten.mul.Tensor(sum_33, 0.00048828125)
    mul_690: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
    mul_691: "f32[256]" = torch.ops.aten.mul.Tensor(mul_689, mul_690);  mul_689 = mul_690 = None
    unsqueeze_454: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_691, 0);  mul_691 = None
    unsqueeze_455: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 2);  unsqueeze_454 = None
    unsqueeze_456: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 3);  unsqueeze_455 = None
    mul_692: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_103);  primals_103 = None
    unsqueeze_457: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_692, 0);  mul_692 = None
    unsqueeze_458: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 2);  unsqueeze_457 = None
    unsqueeze_459: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 3);  unsqueeze_458 = None
    sub_128: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_450);  convolution_51 = unsqueeze_450 = None
    mul_693: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_456);  sub_128 = unsqueeze_456 = None
    sub_129: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_82, mul_693);  where_82 = mul_693 = None
    sub_130: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_129, unsqueeze_453);  sub_129 = unsqueeze_453 = None
    mul_694: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_459);  sub_130 = unsqueeze_459 = None
    mul_695: "f32[256]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_154);  sum_33 = squeeze_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_694, add_272, primals_186, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_694 = add_272 = primals_186 = None
    getitem_209: "f32[8, 256, 16, 16]" = convolution_backward_15[0]
    getitem_210: "f32[256, 256, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_362: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(getitem_203, getitem_209);  getitem_203 = getitem_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_116: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(where_50);  where_50 = None
    alias_117: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_116);  alias_116 = None
    gt_83: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(alias_117, 0);  alias_117 = None
    mul_696: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_362, 0.01)
    where_83: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_83, add_362, mul_696);  gt_83 = mul_696 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_460: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_150, 0);  squeeze_150 = None
    unsqueeze_461: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, 2);  unsqueeze_460 = None
    unsqueeze_462: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 3);  unsqueeze_461 = None
    sum_34: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_83, [0, 2, 3])
    sub_131: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_462)
    mul_697: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_83, sub_131);  sub_131 = None
    sum_35: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_697, [0, 2, 3]);  mul_697 = None
    mul_698: "f32[256]" = torch.ops.aten.mul.Tensor(sum_34, 0.00048828125)
    unsqueeze_463: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_698, 0);  mul_698 = None
    unsqueeze_464: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 2);  unsqueeze_463 = None
    unsqueeze_465: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 3);  unsqueeze_464 = None
    mul_699: "f32[256]" = torch.ops.aten.mul.Tensor(sum_35, 0.00048828125)
    mul_700: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
    mul_701: "f32[256]" = torch.ops.aten.mul.Tensor(mul_699, mul_700);  mul_699 = mul_700 = None
    unsqueeze_466: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_701, 0);  mul_701 = None
    unsqueeze_467: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 2);  unsqueeze_466 = None
    unsqueeze_468: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 3);  unsqueeze_467 = None
    mul_702: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_101);  primals_101 = None
    unsqueeze_469: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_702, 0);  mul_702 = None
    unsqueeze_470: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None
    unsqueeze_471: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 3);  unsqueeze_470 = None
    sub_132: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_462);  convolution_50 = unsqueeze_462 = None
    mul_703: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_468);  sub_132 = unsqueeze_468 = None
    sub_133: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_83, mul_703);  where_83 = mul_703 = None
    sub_134: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_133, unsqueeze_465);  sub_133 = unsqueeze_465 = None
    mul_704: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_471);  sub_134 = unsqueeze_471 = None
    mul_705: "f32[256]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_151);  sum_35 = squeeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_704, where_49, primals_185, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_704 = primals_185 = None
    getitem_212: "f32[8, 256, 16, 16]" = convolution_backward_16[0]
    getitem_213: "f32[256, 256, 3, 3]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_119: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(where_49);  where_49 = None
    alias_120: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_119);  alias_119 = None
    gt_84: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(alias_120, 0);  alias_120 = None
    mul_706: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_212, 0.01)
    where_84: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_84, getitem_212, mul_706);  gt_84 = getitem_212 = mul_706 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_472: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_147, 0);  squeeze_147 = None
    unsqueeze_473: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, 2);  unsqueeze_472 = None
    unsqueeze_474: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 3);  unsqueeze_473 = None
    sum_36: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_84, [0, 2, 3])
    sub_135: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_474)
    mul_707: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_84, sub_135);  sub_135 = None
    sum_37: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_707, [0, 2, 3]);  mul_707 = None
    mul_708: "f32[256]" = torch.ops.aten.mul.Tensor(sum_36, 0.00048828125)
    unsqueeze_475: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_708, 0);  mul_708 = None
    unsqueeze_476: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 2);  unsqueeze_475 = None
    unsqueeze_477: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 3);  unsqueeze_476 = None
    mul_709: "f32[256]" = torch.ops.aten.mul.Tensor(sum_37, 0.00048828125)
    mul_710: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
    mul_711: "f32[256]" = torch.ops.aten.mul.Tensor(mul_709, mul_710);  mul_709 = mul_710 = None
    unsqueeze_478: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_711, 0);  mul_711 = None
    unsqueeze_479: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 2);  unsqueeze_478 = None
    unsqueeze_480: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 3);  unsqueeze_479 = None
    mul_712: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_99);  primals_99 = None
    unsqueeze_481: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_712, 0);  mul_712 = None
    unsqueeze_482: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 2);  unsqueeze_481 = None
    unsqueeze_483: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 3);  unsqueeze_482 = None
    sub_136: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_474);  convolution_49 = unsqueeze_474 = None
    mul_713: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_480);  sub_136 = unsqueeze_480 = None
    sub_137: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_84, mul_713);  where_84 = mul_713 = None
    sub_138: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_137, unsqueeze_477);  sub_137 = unsqueeze_477 = None
    mul_714: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_483);  sub_138 = unsqueeze_483 = None
    mul_715: "f32[256]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_148);  sum_37 = squeeze_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_714, add_261, primals_184, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_714 = add_261 = primals_184 = None
    getitem_215: "f32[8, 256, 16, 16]" = convolution_backward_17[0]
    getitem_216: "f32[256, 256, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_363: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(add_362, getitem_215);  add_362 = getitem_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_122: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(where_48);  where_48 = None
    alias_123: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_122);  alias_122 = None
    gt_85: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(alias_123, 0);  alias_123 = None
    mul_716: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_363, 0.01)
    where_85: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_85, add_363, mul_716);  gt_85 = mul_716 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_484: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
    unsqueeze_485: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 2);  unsqueeze_484 = None
    unsqueeze_486: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 3);  unsqueeze_485 = None
    sum_38: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_85, [0, 2, 3])
    sub_139: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_486)
    mul_717: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_85, sub_139);  sub_139 = None
    sum_39: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_717, [0, 2, 3]);  mul_717 = None
    mul_718: "f32[256]" = torch.ops.aten.mul.Tensor(sum_38, 0.00048828125)
    unsqueeze_487: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_718, 0);  mul_718 = None
    unsqueeze_488: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
    unsqueeze_489: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 3);  unsqueeze_488 = None
    mul_719: "f32[256]" = torch.ops.aten.mul.Tensor(sum_39, 0.00048828125)
    mul_720: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
    mul_721: "f32[256]" = torch.ops.aten.mul.Tensor(mul_719, mul_720);  mul_719 = mul_720 = None
    unsqueeze_490: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_721, 0);  mul_721 = None
    unsqueeze_491: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 2);  unsqueeze_490 = None
    unsqueeze_492: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 3);  unsqueeze_491 = None
    mul_722: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_97);  primals_97 = None
    unsqueeze_493: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_722, 0);  mul_722 = None
    unsqueeze_494: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
    unsqueeze_495: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 3);  unsqueeze_494 = None
    sub_140: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_486);  convolution_48 = unsqueeze_486 = None
    mul_723: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_492);  sub_140 = unsqueeze_492 = None
    sub_141: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_85, mul_723);  where_85 = mul_723 = None
    sub_142: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_141, unsqueeze_489);  sub_141 = unsqueeze_489 = None
    mul_724: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_495);  sub_142 = unsqueeze_495 = None
    mul_725: "f32[256]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_145);  sum_39 = squeeze_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_724, where_47, primals_183, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_724 = primals_183 = None
    getitem_218: "f32[8, 256, 16, 16]" = convolution_backward_18[0]
    getitem_219: "f32[256, 256, 3, 3]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_125: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(where_47);  where_47 = None
    alias_126: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_125);  alias_125 = None
    gt_86: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(alias_126, 0);  alias_126 = None
    mul_726: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_218, 0.01)
    where_86: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_86, getitem_218, mul_726);  gt_86 = getitem_218 = mul_726 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_496: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
    unsqueeze_497: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 2);  unsqueeze_496 = None
    unsqueeze_498: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 3);  unsqueeze_497 = None
    sum_40: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_86, [0, 2, 3])
    sub_143: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_498)
    mul_727: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_86, sub_143);  sub_143 = None
    sum_41: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_727, [0, 2, 3]);  mul_727 = None
    mul_728: "f32[256]" = torch.ops.aten.mul.Tensor(sum_40, 0.00048828125)
    unsqueeze_499: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_728, 0);  mul_728 = None
    unsqueeze_500: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 2);  unsqueeze_499 = None
    unsqueeze_501: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 3);  unsqueeze_500 = None
    mul_729: "f32[256]" = torch.ops.aten.mul.Tensor(sum_41, 0.00048828125)
    mul_730: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
    mul_731: "f32[256]" = torch.ops.aten.mul.Tensor(mul_729, mul_730);  mul_729 = mul_730 = None
    unsqueeze_502: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_731, 0);  mul_731 = None
    unsqueeze_503: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 2);  unsqueeze_502 = None
    unsqueeze_504: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 3);  unsqueeze_503 = None
    mul_732: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_95);  primals_95 = None
    unsqueeze_505: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_732, 0);  mul_732 = None
    unsqueeze_506: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 2);  unsqueeze_505 = None
    unsqueeze_507: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 3);  unsqueeze_506 = None
    sub_144: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_498);  convolution_47 = unsqueeze_498 = None
    mul_733: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_504);  sub_144 = unsqueeze_504 = None
    sub_145: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_86, mul_733);  where_86 = mul_733 = None
    sub_146: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_145, unsqueeze_501);  sub_145 = unsqueeze_501 = None
    mul_734: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_507);  sub_146 = unsqueeze_507 = None
    mul_735: "f32[256]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_142);  sum_41 = squeeze_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_734, add_250, primals_182, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_734 = add_250 = primals_182 = None
    getitem_221: "f32[8, 256, 16, 16]" = convolution_backward_19[0]
    getitem_222: "f32[256, 256, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_364: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(add_363, getitem_221);  add_363 = getitem_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_128: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(where_46);  where_46 = None
    alias_129: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_128);  alias_128 = None
    gt_87: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(alias_129, 0);  alias_129 = None
    mul_736: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_364, 0.01)
    where_87: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_87, add_364, mul_736);  gt_87 = mul_736 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_508: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
    unsqueeze_509: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, 2);  unsqueeze_508 = None
    unsqueeze_510: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 3);  unsqueeze_509 = None
    sum_42: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_87, [0, 2, 3])
    sub_147: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_510)
    mul_737: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_87, sub_147);  sub_147 = None
    sum_43: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_737, [0, 2, 3]);  mul_737 = None
    mul_738: "f32[256]" = torch.ops.aten.mul.Tensor(sum_42, 0.00048828125)
    unsqueeze_511: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_738, 0);  mul_738 = None
    unsqueeze_512: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 2);  unsqueeze_511 = None
    unsqueeze_513: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 3);  unsqueeze_512 = None
    mul_739: "f32[256]" = torch.ops.aten.mul.Tensor(sum_43, 0.00048828125)
    mul_740: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
    mul_741: "f32[256]" = torch.ops.aten.mul.Tensor(mul_739, mul_740);  mul_739 = mul_740 = None
    unsqueeze_514: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_741, 0);  mul_741 = None
    unsqueeze_515: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 2);  unsqueeze_514 = None
    unsqueeze_516: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 3);  unsqueeze_515 = None
    mul_742: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_93);  primals_93 = None
    unsqueeze_517: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_742, 0);  mul_742 = None
    unsqueeze_518: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 2);  unsqueeze_517 = None
    unsqueeze_519: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 3);  unsqueeze_518 = None
    sub_148: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_510);  convolution_46 = unsqueeze_510 = None
    mul_743: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_516);  sub_148 = unsqueeze_516 = None
    sub_149: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_87, mul_743);  where_87 = mul_743 = None
    sub_150: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_149, unsqueeze_513);  sub_149 = unsqueeze_513 = None
    mul_744: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_519);  sub_150 = unsqueeze_519 = None
    mul_745: "f32[256]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_139);  sum_43 = squeeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_744, where_45, primals_181, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_744 = primals_181 = None
    getitem_224: "f32[8, 256, 16, 16]" = convolution_backward_20[0]
    getitem_225: "f32[256, 256, 3, 3]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_131: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(where_45);  where_45 = None
    alias_132: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_131);  alias_131 = None
    gt_88: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(alias_132, 0);  alias_132 = None
    mul_746: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_224, 0.01)
    where_88: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_88, getitem_224, mul_746);  gt_88 = getitem_224 = mul_746 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_520: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
    unsqueeze_521: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 2);  unsqueeze_520 = None
    unsqueeze_522: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 3);  unsqueeze_521 = None
    sum_44: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_88, [0, 2, 3])
    sub_151: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_522)
    mul_747: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_88, sub_151);  sub_151 = None
    sum_45: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_747, [0, 2, 3]);  mul_747 = None
    mul_748: "f32[256]" = torch.ops.aten.mul.Tensor(sum_44, 0.00048828125)
    unsqueeze_523: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_748, 0);  mul_748 = None
    unsqueeze_524: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 2);  unsqueeze_523 = None
    unsqueeze_525: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 3);  unsqueeze_524 = None
    mul_749: "f32[256]" = torch.ops.aten.mul.Tensor(sum_45, 0.00048828125)
    mul_750: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
    mul_751: "f32[256]" = torch.ops.aten.mul.Tensor(mul_749, mul_750);  mul_749 = mul_750 = None
    unsqueeze_526: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_751, 0);  mul_751 = None
    unsqueeze_527: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 2);  unsqueeze_526 = None
    unsqueeze_528: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 3);  unsqueeze_527 = None
    mul_752: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_91);  primals_91 = None
    unsqueeze_529: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_752, 0);  mul_752 = None
    unsqueeze_530: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 2);  unsqueeze_529 = None
    unsqueeze_531: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 3);  unsqueeze_530 = None
    sub_152: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_522);  convolution_45 = unsqueeze_522 = None
    mul_753: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_528);  sub_152 = unsqueeze_528 = None
    sub_153: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_88, mul_753);  where_88 = mul_753 = None
    sub_154: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_153, unsqueeze_525);  sub_153 = unsqueeze_525 = None
    mul_754: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_531);  sub_154 = unsqueeze_531 = None
    mul_755: "f32[256]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_136);  sum_45 = squeeze_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_754, add_239, primals_180, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_754 = add_239 = primals_180 = None
    getitem_227: "f32[8, 256, 16, 16]" = convolution_backward_21[0]
    getitem_228: "f32[256, 256, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_365: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(add_364, getitem_227);  add_364 = getitem_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_134: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(where_44);  where_44 = None
    alias_135: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_134);  alias_134 = None
    gt_89: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(alias_135, 0);  alias_135 = None
    mul_756: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_365, 0.01)
    where_89: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_89, add_365, mul_756);  gt_89 = mul_756 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_532: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
    unsqueeze_533: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 2);  unsqueeze_532 = None
    unsqueeze_534: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 3);  unsqueeze_533 = None
    sum_46: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_89, [0, 2, 3])
    sub_155: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_534)
    mul_757: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_89, sub_155);  sub_155 = None
    sum_47: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_757, [0, 2, 3]);  mul_757 = None
    mul_758: "f32[256]" = torch.ops.aten.mul.Tensor(sum_46, 0.00048828125)
    unsqueeze_535: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_758, 0);  mul_758 = None
    unsqueeze_536: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 2);  unsqueeze_535 = None
    unsqueeze_537: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 3);  unsqueeze_536 = None
    mul_759: "f32[256]" = torch.ops.aten.mul.Tensor(sum_47, 0.00048828125)
    mul_760: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
    mul_761: "f32[256]" = torch.ops.aten.mul.Tensor(mul_759, mul_760);  mul_759 = mul_760 = None
    unsqueeze_538: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_761, 0);  mul_761 = None
    unsqueeze_539: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 2);  unsqueeze_538 = None
    unsqueeze_540: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 3);  unsqueeze_539 = None
    mul_762: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_89);  primals_89 = None
    unsqueeze_541: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_762, 0);  mul_762 = None
    unsqueeze_542: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 2);  unsqueeze_541 = None
    unsqueeze_543: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 3);  unsqueeze_542 = None
    sub_156: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_534);  convolution_44 = unsqueeze_534 = None
    mul_763: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_540);  sub_156 = unsqueeze_540 = None
    sub_157: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_89, mul_763);  where_89 = mul_763 = None
    sub_158: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_157, unsqueeze_537);  sub_157 = unsqueeze_537 = None
    mul_764: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_543);  sub_158 = unsqueeze_543 = None
    mul_765: "f32[256]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_133);  sum_47 = squeeze_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_764, where_43, primals_179, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_764 = primals_179 = None
    getitem_230: "f32[8, 256, 16, 16]" = convolution_backward_22[0]
    getitem_231: "f32[256, 256, 3, 3]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_137: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(where_43);  where_43 = None
    alias_138: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_137);  alias_137 = None
    gt_90: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(alias_138, 0);  alias_138 = None
    mul_766: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_230, 0.01)
    where_90: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_90, getitem_230, mul_766);  gt_90 = getitem_230 = mul_766 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_544: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
    unsqueeze_545: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, 2);  unsqueeze_544 = None
    unsqueeze_546: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 3);  unsqueeze_545 = None
    sum_48: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_90, [0, 2, 3])
    sub_159: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_546)
    mul_767: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_90, sub_159);  sub_159 = None
    sum_49: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_767, [0, 2, 3]);  mul_767 = None
    mul_768: "f32[256]" = torch.ops.aten.mul.Tensor(sum_48, 0.00048828125)
    unsqueeze_547: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_768, 0);  mul_768 = None
    unsqueeze_548: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_547, 2);  unsqueeze_547 = None
    unsqueeze_549: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 3);  unsqueeze_548 = None
    mul_769: "f32[256]" = torch.ops.aten.mul.Tensor(sum_49, 0.00048828125)
    mul_770: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
    mul_771: "f32[256]" = torch.ops.aten.mul.Tensor(mul_769, mul_770);  mul_769 = mul_770 = None
    unsqueeze_550: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_771, 0);  mul_771 = None
    unsqueeze_551: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 2);  unsqueeze_550 = None
    unsqueeze_552: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 3);  unsqueeze_551 = None
    mul_772: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_87);  primals_87 = None
    unsqueeze_553: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_772, 0);  mul_772 = None
    unsqueeze_554: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 2);  unsqueeze_553 = None
    unsqueeze_555: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 3);  unsqueeze_554 = None
    sub_160: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_546);  convolution_43 = unsqueeze_546 = None
    mul_773: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_552);  sub_160 = unsqueeze_552 = None
    sub_161: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_90, mul_773);  where_90 = mul_773 = None
    sub_162: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_161, unsqueeze_549);  sub_161 = unsqueeze_549 = None
    mul_774: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_555);  sub_162 = unsqueeze_555 = None
    mul_775: "f32[256]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_130);  sum_49 = squeeze_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_774, add_228, primals_178, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_774 = add_228 = primals_178 = None
    getitem_233: "f32[8, 256, 16, 16]" = convolution_backward_23[0]
    getitem_234: "f32[256, 256, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_366: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(add_365, getitem_233);  add_365 = getitem_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_140: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(where_42);  where_42 = None
    alias_141: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_140);  alias_140 = None
    gt_91: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(alias_141, 0);  alias_141 = None
    mul_776: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_366, 0.01)
    where_91: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_91, add_366, mul_776);  gt_91 = mul_776 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_556: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
    unsqueeze_557: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 2);  unsqueeze_556 = None
    unsqueeze_558: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 3);  unsqueeze_557 = None
    sum_50: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_91, [0, 2, 3])
    sub_163: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_558)
    mul_777: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_91, sub_163);  sub_163 = None
    sum_51: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_777, [0, 2, 3]);  mul_777 = None
    mul_778: "f32[256]" = torch.ops.aten.mul.Tensor(sum_50, 0.00048828125)
    unsqueeze_559: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_778, 0);  mul_778 = None
    unsqueeze_560: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 2);  unsqueeze_559 = None
    unsqueeze_561: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 3);  unsqueeze_560 = None
    mul_779: "f32[256]" = torch.ops.aten.mul.Tensor(sum_51, 0.00048828125)
    mul_780: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
    mul_781: "f32[256]" = torch.ops.aten.mul.Tensor(mul_779, mul_780);  mul_779 = mul_780 = None
    unsqueeze_562: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_781, 0);  mul_781 = None
    unsqueeze_563: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 2);  unsqueeze_562 = None
    unsqueeze_564: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 3);  unsqueeze_563 = None
    mul_782: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_85);  primals_85 = None
    unsqueeze_565: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_782, 0);  mul_782 = None
    unsqueeze_566: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 2);  unsqueeze_565 = None
    unsqueeze_567: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 3);  unsqueeze_566 = None
    sub_164: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_558);  convolution_42 = unsqueeze_558 = None
    mul_783: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_564);  sub_164 = unsqueeze_564 = None
    sub_165: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_91, mul_783);  where_91 = mul_783 = None
    sub_166: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_165, unsqueeze_561);  sub_165 = unsqueeze_561 = None
    mul_784: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_567);  sub_166 = unsqueeze_567 = None
    mul_785: "f32[256]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_127);  sum_51 = squeeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_784, where_41, primals_177, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_784 = primals_177 = None
    getitem_236: "f32[8, 256, 16, 16]" = convolution_backward_24[0]
    getitem_237: "f32[256, 256, 3, 3]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_143: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(where_41);  where_41 = None
    alias_144: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_143);  alias_143 = None
    gt_92: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(alias_144, 0);  alias_144 = None
    mul_786: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_236, 0.01)
    where_92: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_92, getitem_236, mul_786);  gt_92 = getitem_236 = mul_786 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_568: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
    unsqueeze_569: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, 2);  unsqueeze_568 = None
    unsqueeze_570: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 3);  unsqueeze_569 = None
    sum_52: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_92, [0, 2, 3])
    sub_167: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_570)
    mul_787: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_92, sub_167);  sub_167 = None
    sum_53: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_787, [0, 2, 3]);  mul_787 = None
    mul_788: "f32[256]" = torch.ops.aten.mul.Tensor(sum_52, 0.00048828125)
    unsqueeze_571: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_788, 0);  mul_788 = None
    unsqueeze_572: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_571, 2);  unsqueeze_571 = None
    unsqueeze_573: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 3);  unsqueeze_572 = None
    mul_789: "f32[256]" = torch.ops.aten.mul.Tensor(sum_53, 0.00048828125)
    mul_790: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
    mul_791: "f32[256]" = torch.ops.aten.mul.Tensor(mul_789, mul_790);  mul_789 = mul_790 = None
    unsqueeze_574: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_791, 0);  mul_791 = None
    unsqueeze_575: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, 2);  unsqueeze_574 = None
    unsqueeze_576: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 3);  unsqueeze_575 = None
    mul_792: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_83);  primals_83 = None
    unsqueeze_577: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_792, 0);  mul_792 = None
    unsqueeze_578: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 2);  unsqueeze_577 = None
    unsqueeze_579: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 3);  unsqueeze_578 = None
    sub_168: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_570);  convolution_41 = unsqueeze_570 = None
    mul_793: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_576);  sub_168 = unsqueeze_576 = None
    sub_169: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_92, mul_793);  where_92 = mul_793 = None
    sub_170: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_169, unsqueeze_573);  sub_169 = unsqueeze_573 = None
    mul_794: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_170, unsqueeze_579);  sub_170 = unsqueeze_579 = None
    mul_795: "f32[256]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_124);  sum_53 = squeeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_794, add_217, primals_176, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_794 = add_217 = primals_176 = None
    getitem_239: "f32[8, 256, 16, 16]" = convolution_backward_25[0]
    getitem_240: "f32[256, 256, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_367: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(add_366, getitem_239);  add_366 = getitem_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_146: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(where_40);  where_40 = None
    alias_147: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_146);  alias_146 = None
    gt_93: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(alias_147, 0);  alias_147 = None
    mul_796: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_367, 0.01)
    where_93: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_93, add_367, mul_796);  gt_93 = mul_796 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_580: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
    unsqueeze_581: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 2);  unsqueeze_580 = None
    unsqueeze_582: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 3);  unsqueeze_581 = None
    sum_54: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_93, [0, 2, 3])
    sub_171: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_582)
    mul_797: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_93, sub_171);  sub_171 = None
    sum_55: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_797, [0, 2, 3]);  mul_797 = None
    mul_798: "f32[256]" = torch.ops.aten.mul.Tensor(sum_54, 0.00048828125)
    unsqueeze_583: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_798, 0);  mul_798 = None
    unsqueeze_584: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_583, 2);  unsqueeze_583 = None
    unsqueeze_585: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 3);  unsqueeze_584 = None
    mul_799: "f32[256]" = torch.ops.aten.mul.Tensor(sum_55, 0.00048828125)
    mul_800: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_801: "f32[256]" = torch.ops.aten.mul.Tensor(mul_799, mul_800);  mul_799 = mul_800 = None
    unsqueeze_586: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_801, 0);  mul_801 = None
    unsqueeze_587: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 2);  unsqueeze_586 = None
    unsqueeze_588: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 3);  unsqueeze_587 = None
    mul_802: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_81);  primals_81 = None
    unsqueeze_589: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_802, 0);  mul_802 = None
    unsqueeze_590: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 2);  unsqueeze_589 = None
    unsqueeze_591: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 3);  unsqueeze_590 = None
    sub_172: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_582);  convolution_40 = unsqueeze_582 = None
    mul_803: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_588);  sub_172 = unsqueeze_588 = None
    sub_173: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_93, mul_803);  where_93 = mul_803 = None
    sub_174: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_173, unsqueeze_585);  sub_173 = unsqueeze_585 = None
    mul_804: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_591);  sub_174 = unsqueeze_591 = None
    mul_805: "f32[256]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_121);  sum_55 = squeeze_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_804, where_39, primals_175, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_804 = primals_175 = None
    getitem_242: "f32[8, 256, 16, 16]" = convolution_backward_26[0]
    getitem_243: "f32[256, 256, 3, 3]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_149: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(where_39);  where_39 = None
    alias_150: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_149);  alias_149 = None
    gt_94: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(alias_150, 0);  alias_150 = None
    mul_806: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_242, 0.01)
    where_94: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_94, getitem_242, mul_806);  gt_94 = getitem_242 = mul_806 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_592: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
    unsqueeze_593: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 2);  unsqueeze_592 = None
    unsqueeze_594: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 3);  unsqueeze_593 = None
    sum_56: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_94, [0, 2, 3])
    sub_175: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_594)
    mul_807: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_94, sub_175);  sub_175 = None
    sum_57: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_807, [0, 2, 3]);  mul_807 = None
    mul_808: "f32[256]" = torch.ops.aten.mul.Tensor(sum_56, 0.00048828125)
    unsqueeze_595: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_808, 0);  mul_808 = None
    unsqueeze_596: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 2);  unsqueeze_595 = None
    unsqueeze_597: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 3);  unsqueeze_596 = None
    mul_809: "f32[256]" = torch.ops.aten.mul.Tensor(sum_57, 0.00048828125)
    mul_810: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_811: "f32[256]" = torch.ops.aten.mul.Tensor(mul_809, mul_810);  mul_809 = mul_810 = None
    unsqueeze_598: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_811, 0);  mul_811 = None
    unsqueeze_599: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 2);  unsqueeze_598 = None
    unsqueeze_600: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 3);  unsqueeze_599 = None
    mul_812: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_79);  primals_79 = None
    unsqueeze_601: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_812, 0);  mul_812 = None
    unsqueeze_602: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 2);  unsqueeze_601 = None
    unsqueeze_603: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 3);  unsqueeze_602 = None
    sub_176: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_594);  convolution_39 = unsqueeze_594 = None
    mul_813: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_600);  sub_176 = unsqueeze_600 = None
    sub_177: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_94, mul_813);  where_94 = mul_813 = None
    sub_178: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_177, unsqueeze_597);  sub_177 = unsqueeze_597 = None
    mul_814: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_178, unsqueeze_603);  sub_178 = unsqueeze_603 = None
    mul_815: "f32[256]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_118);  sum_57 = squeeze_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_814, add_206, primals_174, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_814 = add_206 = primals_174 = None
    getitem_245: "f32[8, 256, 16, 16]" = convolution_backward_27[0]
    getitem_246: "f32[256, 256, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_368: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(add_367, getitem_245);  add_367 = getitem_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_152: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(where_38);  where_38 = None
    alias_153: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_152);  alias_152 = None
    gt_95: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(alias_153, 0);  alias_153 = None
    mul_816: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_368, 0.01)
    where_95: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_95, add_368, mul_816);  gt_95 = mul_816 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_604: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
    unsqueeze_605: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, 2);  unsqueeze_604 = None
    unsqueeze_606: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 3);  unsqueeze_605 = None
    sum_58: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_95, [0, 2, 3])
    sub_179: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_606)
    mul_817: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_95, sub_179);  sub_179 = None
    sum_59: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_817, [0, 2, 3]);  mul_817 = None
    mul_818: "f32[256]" = torch.ops.aten.mul.Tensor(sum_58, 0.00048828125)
    unsqueeze_607: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_818, 0);  mul_818 = None
    unsqueeze_608: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_607, 2);  unsqueeze_607 = None
    unsqueeze_609: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 3);  unsqueeze_608 = None
    mul_819: "f32[256]" = torch.ops.aten.mul.Tensor(sum_59, 0.00048828125)
    mul_820: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_821: "f32[256]" = torch.ops.aten.mul.Tensor(mul_819, mul_820);  mul_819 = mul_820 = None
    unsqueeze_610: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_821, 0);  mul_821 = None
    unsqueeze_611: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 2);  unsqueeze_610 = None
    unsqueeze_612: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 3);  unsqueeze_611 = None
    mul_822: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_77);  primals_77 = None
    unsqueeze_613: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_822, 0);  mul_822 = None
    unsqueeze_614: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 2);  unsqueeze_613 = None
    unsqueeze_615: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 3);  unsqueeze_614 = None
    sub_180: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_606);  convolution_38 = unsqueeze_606 = None
    mul_823: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_612);  sub_180 = unsqueeze_612 = None
    sub_181: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_95, mul_823);  where_95 = mul_823 = None
    sub_182: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_181, unsqueeze_609);  sub_181 = unsqueeze_609 = None
    mul_824: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_615);  sub_182 = unsqueeze_615 = None
    mul_825: "f32[256]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_115);  sum_59 = squeeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_824, where_37, primals_173, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_824 = primals_173 = None
    getitem_248: "f32[8, 256, 16, 16]" = convolution_backward_28[0]
    getitem_249: "f32[256, 256, 3, 3]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_155: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(where_37);  where_37 = None
    alias_156: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_155);  alias_155 = None
    gt_96: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(alias_156, 0);  alias_156 = None
    mul_826: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_248, 0.01)
    where_96: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_96, getitem_248, mul_826);  gt_96 = getitem_248 = mul_826 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_616: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
    unsqueeze_617: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 2);  unsqueeze_616 = None
    unsqueeze_618: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 3);  unsqueeze_617 = None
    sum_60: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_96, [0, 2, 3])
    sub_183: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_618)
    mul_827: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_96, sub_183);  sub_183 = None
    sum_61: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_827, [0, 2, 3]);  mul_827 = None
    mul_828: "f32[256]" = torch.ops.aten.mul.Tensor(sum_60, 0.00048828125)
    unsqueeze_619: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_828, 0);  mul_828 = None
    unsqueeze_620: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_619, 2);  unsqueeze_619 = None
    unsqueeze_621: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 3);  unsqueeze_620 = None
    mul_829: "f32[256]" = torch.ops.aten.mul.Tensor(sum_61, 0.00048828125)
    mul_830: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_831: "f32[256]" = torch.ops.aten.mul.Tensor(mul_829, mul_830);  mul_829 = mul_830 = None
    unsqueeze_622: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_831, 0);  mul_831 = None
    unsqueeze_623: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 2);  unsqueeze_622 = None
    unsqueeze_624: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 3);  unsqueeze_623 = None
    mul_832: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_75);  primals_75 = None
    unsqueeze_625: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_832, 0);  mul_832 = None
    unsqueeze_626: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 2);  unsqueeze_625 = None
    unsqueeze_627: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 3);  unsqueeze_626 = None
    sub_184: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_618);  convolution_37 = unsqueeze_618 = None
    mul_833: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_624);  sub_184 = unsqueeze_624 = None
    sub_185: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_96, mul_833);  where_96 = mul_833 = None
    sub_186: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_185, unsqueeze_621);  sub_185 = unsqueeze_621 = None
    mul_834: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_186, unsqueeze_627);  sub_186 = unsqueeze_627 = None
    mul_835: "f32[256]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_112);  sum_61 = squeeze_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_834, getitem_95, primals_172, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_834 = getitem_95 = primals_172 = None
    getitem_251: "f32[8, 256, 16, 16]" = convolution_backward_29[0]
    getitem_252: "f32[256, 256, 1, 1]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_369: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(add_368, getitem_251);  add_368 = getitem_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:336, code: xs, xb = x.split(self.expand_chs // 2, dim=1)
    cat_6: "f32[8, 512, 16, 16]" = torch.ops.aten.cat.default([slice_3, add_369], 1);  slice_3 = add_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_158: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(where_36);  where_36 = None
    alias_159: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(alias_158);  alias_158 = None
    gt_97: "b8[8, 512, 16, 16]" = torch.ops.aten.gt.Scalar(alias_159, 0);  alias_159 = None
    mul_836: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(cat_6, 0.01)
    where_97: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(gt_97, cat_6, mul_836);  gt_97 = cat_6 = mul_836 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_628: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
    unsqueeze_629: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, 2);  unsqueeze_628 = None
    unsqueeze_630: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 3);  unsqueeze_629 = None
    sum_62: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_97, [0, 2, 3])
    sub_187: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_630)
    mul_837: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(where_97, sub_187);  sub_187 = None
    sum_63: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_837, [0, 2, 3]);  mul_837 = None
    mul_838: "f32[512]" = torch.ops.aten.mul.Tensor(sum_62, 0.00048828125)
    unsqueeze_631: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_838, 0);  mul_838 = None
    unsqueeze_632: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_631, 2);  unsqueeze_631 = None
    unsqueeze_633: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 3);  unsqueeze_632 = None
    mul_839: "f32[512]" = torch.ops.aten.mul.Tensor(sum_63, 0.00048828125)
    mul_840: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_841: "f32[512]" = torch.ops.aten.mul.Tensor(mul_839, mul_840);  mul_839 = mul_840 = None
    unsqueeze_634: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_841, 0);  mul_841 = None
    unsqueeze_635: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, 2);  unsqueeze_634 = None
    unsqueeze_636: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 3);  unsqueeze_635 = None
    mul_842: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_73);  primals_73 = None
    unsqueeze_637: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_842, 0);  mul_842 = None
    unsqueeze_638: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 2);  unsqueeze_637 = None
    unsqueeze_639: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 3);  unsqueeze_638 = None
    sub_188: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_630);  convolution_36 = unsqueeze_630 = None
    mul_843: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_636);  sub_188 = unsqueeze_636 = None
    sub_189: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(where_97, mul_843);  where_97 = mul_843 = None
    sub_190: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_189, unsqueeze_633);  sub_189 = unsqueeze_633 = None
    mul_844: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_639);  sub_190 = unsqueeze_639 = None
    mul_845: "f32[512]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_109);  sum_63 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_844, where_35, primals_171, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_844 = primals_171 = None
    getitem_254: "f32[8, 512, 16, 16]" = convolution_backward_30[0]
    getitem_255: "f32[512, 512, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_161: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(where_35);  where_35 = None
    alias_162: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(alias_161);  alias_161 = None
    gt_98: "b8[8, 512, 16, 16]" = torch.ops.aten.gt.Scalar(alias_162, 0);  alias_162 = None
    mul_846: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_254, 0.01)
    where_98: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(gt_98, getitem_254, mul_846);  gt_98 = getitem_254 = mul_846 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_640: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
    unsqueeze_641: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, 2);  unsqueeze_640 = None
    unsqueeze_642: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 3);  unsqueeze_641 = None
    sum_64: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_98, [0, 2, 3])
    sub_191: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_642)
    mul_847: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(where_98, sub_191);  sub_191 = None
    sum_65: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_847, [0, 2, 3]);  mul_847 = None
    mul_848: "f32[512]" = torch.ops.aten.mul.Tensor(sum_64, 0.00048828125)
    unsqueeze_643: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_848, 0);  mul_848 = None
    unsqueeze_644: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_643, 2);  unsqueeze_643 = None
    unsqueeze_645: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 3);  unsqueeze_644 = None
    mul_849: "f32[512]" = torch.ops.aten.mul.Tensor(sum_65, 0.00048828125)
    mul_850: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_851: "f32[512]" = torch.ops.aten.mul.Tensor(mul_849, mul_850);  mul_849 = mul_850 = None
    unsqueeze_646: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_851, 0);  mul_851 = None
    unsqueeze_647: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, 2);  unsqueeze_646 = None
    unsqueeze_648: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 3);  unsqueeze_647 = None
    mul_852: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_71);  primals_71 = None
    unsqueeze_649: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_852, 0);  mul_852 = None
    unsqueeze_650: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 2);  unsqueeze_649 = None
    unsqueeze_651: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 3);  unsqueeze_650 = None
    sub_192: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_642);  convolution_35 = unsqueeze_642 = None
    mul_853: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_648);  sub_192 = unsqueeze_648 = None
    sub_193: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(where_98, mul_853);  where_98 = mul_853 = None
    sub_194: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_193, unsqueeze_645);  sub_193 = unsqueeze_645 = None
    mul_854: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_194, unsqueeze_651);  sub_194 = unsqueeze_651 = None
    mul_855: "f32[512]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_106);  sum_65 = squeeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:126, code: x = self.conv(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_854, where_34, primals_170, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_854 = primals_170 = None
    getitem_257: "f32[8, 256, 32, 32]" = convolution_backward_31[0]
    getitem_258: "f32[512, 256, 3, 3]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_164: "f32[8, 256, 32, 32]" = torch.ops.aten.alias.default(where_34);  where_34 = None
    alias_165: "f32[8, 256, 32, 32]" = torch.ops.aten.alias.default(alias_164);  alias_164 = None
    gt_99: "b8[8, 256, 32, 32]" = torch.ops.aten.gt.Scalar(alias_165, 0);  alias_165 = None
    mul_856: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_257, 0.01)
    where_99: "f32[8, 256, 32, 32]" = torch.ops.aten.where.self(gt_99, getitem_257, mul_856);  gt_99 = getitem_257 = mul_856 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_652: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
    unsqueeze_653: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, 2);  unsqueeze_652 = None
    unsqueeze_654: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 3);  unsqueeze_653 = None
    sum_66: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_99, [0, 2, 3])
    sub_195: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_654)
    mul_857: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(where_99, sub_195);  sub_195 = None
    sum_67: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_857, [0, 2, 3]);  mul_857 = None
    mul_858: "f32[256]" = torch.ops.aten.mul.Tensor(sum_66, 0.0001220703125)
    unsqueeze_655: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_858, 0);  mul_858 = None
    unsqueeze_656: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_655, 2);  unsqueeze_655 = None
    unsqueeze_657: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 3);  unsqueeze_656 = None
    mul_859: "f32[256]" = torch.ops.aten.mul.Tensor(sum_67, 0.0001220703125)
    mul_860: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_861: "f32[256]" = torch.ops.aten.mul.Tensor(mul_859, mul_860);  mul_859 = mul_860 = None
    unsqueeze_658: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_861, 0);  mul_861 = None
    unsqueeze_659: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, 2);  unsqueeze_658 = None
    unsqueeze_660: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 3);  unsqueeze_659 = None
    mul_862: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_69);  primals_69 = None
    unsqueeze_661: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_862, 0);  mul_862 = None
    unsqueeze_662: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 2);  unsqueeze_661 = None
    unsqueeze_663: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 3);  unsqueeze_662 = None
    sub_196: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_654);  convolution_34 = unsqueeze_654 = None
    mul_863: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_660);  sub_196 = unsqueeze_660 = None
    sub_197: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(where_99, mul_863);  where_99 = mul_863 = None
    sub_198: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(sub_197, unsqueeze_657);  sub_197 = unsqueeze_657 = None
    mul_864: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_663);  sub_198 = unsqueeze_663 = None
    mul_865: "f32[256]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_103);  sum_67 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_864, cat_2, primals_169, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_864 = cat_2 = primals_169 = None
    getitem_260: "f32[8, 256, 32, 32]" = convolution_backward_32[0]
    getitem_261: "f32[256, 256, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:339, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
    slice_5: "f32[8, 128, 32, 32]" = torch.ops.aten.slice.Tensor(getitem_260, 1, 0, 128)
    slice_6: "f32[8, 128, 32, 32]" = torch.ops.aten.slice.Tensor(getitem_260, 1, 128, 256);  getitem_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_167: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(where_33);  where_33 = None
    alias_168: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_167);  alias_167 = None
    gt_100: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(alias_168, 0);  alias_168 = None
    mul_866: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(slice_6, 0.01)
    where_100: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_100, slice_6, mul_866);  gt_100 = slice_6 = mul_866 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_664: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
    unsqueeze_665: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, 2);  unsqueeze_664 = None
    unsqueeze_666: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 3);  unsqueeze_665 = None
    sum_68: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_100, [0, 2, 3])
    sub_199: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_666)
    mul_867: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_100, sub_199);  sub_199 = None
    sum_69: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_867, [0, 2, 3]);  mul_867 = None
    mul_868: "f32[128]" = torch.ops.aten.mul.Tensor(sum_68, 0.0001220703125)
    unsqueeze_667: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_868, 0);  mul_868 = None
    unsqueeze_668: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_667, 2);  unsqueeze_667 = None
    unsqueeze_669: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 3);  unsqueeze_668 = None
    mul_869: "f32[128]" = torch.ops.aten.mul.Tensor(sum_69, 0.0001220703125)
    mul_870: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_871: "f32[128]" = torch.ops.aten.mul.Tensor(mul_869, mul_870);  mul_869 = mul_870 = None
    unsqueeze_670: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_871, 0);  mul_871 = None
    unsqueeze_671: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, 2);  unsqueeze_670 = None
    unsqueeze_672: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 3);  unsqueeze_671 = None
    mul_872: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_67);  primals_67 = None
    unsqueeze_673: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_872, 0);  mul_872 = None
    unsqueeze_674: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 2);  unsqueeze_673 = None
    unsqueeze_675: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 3);  unsqueeze_674 = None
    sub_200: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_666);  convolution_33 = unsqueeze_666 = None
    mul_873: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_672);  sub_200 = unsqueeze_672 = None
    sub_201: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_100, mul_873);  where_100 = mul_873 = None
    sub_202: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_201, unsqueeze_669);  sub_201 = unsqueeze_669 = None
    mul_874: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_675);  sub_202 = unsqueeze_675 = None
    mul_875: "f32[128]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_100);  sum_69 = squeeze_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_874, add_175, primals_168, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_874 = add_175 = primals_168 = None
    getitem_263: "f32[8, 128, 32, 32]" = convolution_backward_33[0]
    getitem_264: "f32[128, 128, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_170: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(where_32);  where_32 = None
    alias_171: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_170);  alias_170 = None
    gt_101: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(alias_171, 0);  alias_171 = None
    mul_876: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_263, 0.01)
    where_101: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_101, getitem_263, mul_876);  gt_101 = mul_876 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_676: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
    unsqueeze_677: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, 2);  unsqueeze_676 = None
    unsqueeze_678: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 3);  unsqueeze_677 = None
    sum_70: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_101, [0, 2, 3])
    sub_203: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_678)
    mul_877: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_101, sub_203);  sub_203 = None
    sum_71: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_877, [0, 2, 3]);  mul_877 = None
    mul_878: "f32[128]" = torch.ops.aten.mul.Tensor(sum_70, 0.0001220703125)
    unsqueeze_679: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_878, 0);  mul_878 = None
    unsqueeze_680: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_679, 2);  unsqueeze_679 = None
    unsqueeze_681: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 3);  unsqueeze_680 = None
    mul_879: "f32[128]" = torch.ops.aten.mul.Tensor(sum_71, 0.0001220703125)
    mul_880: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_881: "f32[128]" = torch.ops.aten.mul.Tensor(mul_879, mul_880);  mul_879 = mul_880 = None
    unsqueeze_682: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_881, 0);  mul_881 = None
    unsqueeze_683: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, 2);  unsqueeze_682 = None
    unsqueeze_684: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 3);  unsqueeze_683 = None
    mul_882: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_65);  primals_65 = None
    unsqueeze_685: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_882, 0);  mul_882 = None
    unsqueeze_686: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 2);  unsqueeze_685 = None
    unsqueeze_687: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 3);  unsqueeze_686 = None
    sub_204: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_678);  convolution_32 = unsqueeze_678 = None
    mul_883: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_684);  sub_204 = unsqueeze_684 = None
    sub_205: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_101, mul_883);  where_101 = mul_883 = None
    sub_206: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_205, unsqueeze_681);  sub_205 = unsqueeze_681 = None
    mul_884: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_687);  sub_206 = unsqueeze_687 = None
    mul_885: "f32[128]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_97);  sum_71 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_884, where_31, primals_167, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_884 = primals_167 = None
    getitem_266: "f32[8, 128, 32, 32]" = convolution_backward_34[0]
    getitem_267: "f32[128, 128, 3, 3]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_173: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(where_31);  where_31 = None
    alias_174: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_173);  alias_173 = None
    gt_102: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(alias_174, 0);  alias_174 = None
    mul_886: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_266, 0.01)
    where_102: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_102, getitem_266, mul_886);  gt_102 = getitem_266 = mul_886 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_688: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
    unsqueeze_689: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, 2);  unsqueeze_688 = None
    unsqueeze_690: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 3);  unsqueeze_689 = None
    sum_72: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_102, [0, 2, 3])
    sub_207: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_690)
    mul_887: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_102, sub_207);  sub_207 = None
    sum_73: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_887, [0, 2, 3]);  mul_887 = None
    mul_888: "f32[128]" = torch.ops.aten.mul.Tensor(sum_72, 0.0001220703125)
    unsqueeze_691: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_888, 0);  mul_888 = None
    unsqueeze_692: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_691, 2);  unsqueeze_691 = None
    unsqueeze_693: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 3);  unsqueeze_692 = None
    mul_889: "f32[128]" = torch.ops.aten.mul.Tensor(sum_73, 0.0001220703125)
    mul_890: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_891: "f32[128]" = torch.ops.aten.mul.Tensor(mul_889, mul_890);  mul_889 = mul_890 = None
    unsqueeze_694: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_891, 0);  mul_891 = None
    unsqueeze_695: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 2);  unsqueeze_694 = None
    unsqueeze_696: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 3);  unsqueeze_695 = None
    mul_892: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_63);  primals_63 = None
    unsqueeze_697: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_892, 0);  mul_892 = None
    unsqueeze_698: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 2);  unsqueeze_697 = None
    unsqueeze_699: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 3);  unsqueeze_698 = None
    sub_208: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_690);  convolution_31 = unsqueeze_690 = None
    mul_893: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_696);  sub_208 = unsqueeze_696 = None
    sub_209: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_102, mul_893);  where_102 = mul_893 = None
    sub_210: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_209, unsqueeze_693);  sub_209 = unsqueeze_693 = None
    mul_894: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_210, unsqueeze_699);  sub_210 = unsqueeze_699 = None
    mul_895: "f32[128]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_94);  sum_73 = squeeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_894, add_164, primals_166, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_894 = add_164 = primals_166 = None
    getitem_269: "f32[8, 128, 32, 32]" = convolution_backward_35[0]
    getitem_270: "f32[128, 128, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_370: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(getitem_263, getitem_269);  getitem_263 = getitem_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_176: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(where_30);  where_30 = None
    alias_177: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_176);  alias_176 = None
    gt_103: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(alias_177, 0);  alias_177 = None
    mul_896: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_370, 0.01)
    where_103: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_103, add_370, mul_896);  gt_103 = mul_896 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_700: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_701: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, 2);  unsqueeze_700 = None
    unsqueeze_702: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 3);  unsqueeze_701 = None
    sum_74: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_103, [0, 2, 3])
    sub_211: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_702)
    mul_897: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_103, sub_211);  sub_211 = None
    sum_75: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_897, [0, 2, 3]);  mul_897 = None
    mul_898: "f32[128]" = torch.ops.aten.mul.Tensor(sum_74, 0.0001220703125)
    unsqueeze_703: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_898, 0);  mul_898 = None
    unsqueeze_704: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_703, 2);  unsqueeze_703 = None
    unsqueeze_705: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 3);  unsqueeze_704 = None
    mul_899: "f32[128]" = torch.ops.aten.mul.Tensor(sum_75, 0.0001220703125)
    mul_900: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_901: "f32[128]" = torch.ops.aten.mul.Tensor(mul_899, mul_900);  mul_899 = mul_900 = None
    unsqueeze_706: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_901, 0);  mul_901 = None
    unsqueeze_707: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 2);  unsqueeze_706 = None
    unsqueeze_708: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 3);  unsqueeze_707 = None
    mul_902: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_61);  primals_61 = None
    unsqueeze_709: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_902, 0);  mul_902 = None
    unsqueeze_710: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 2);  unsqueeze_709 = None
    unsqueeze_711: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 3);  unsqueeze_710 = None
    sub_212: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_702);  convolution_30 = unsqueeze_702 = None
    mul_903: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_708);  sub_212 = unsqueeze_708 = None
    sub_213: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_103, mul_903);  where_103 = mul_903 = None
    sub_214: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_213, unsqueeze_705);  sub_213 = unsqueeze_705 = None
    mul_904: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_711);  sub_214 = unsqueeze_711 = None
    mul_905: "f32[128]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_91);  sum_75 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_904, where_29, primals_165, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_904 = primals_165 = None
    getitem_272: "f32[8, 128, 32, 32]" = convolution_backward_36[0]
    getitem_273: "f32[128, 128, 3, 3]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_179: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(where_29);  where_29 = None
    alias_180: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_179);  alias_179 = None
    gt_104: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(alias_180, 0);  alias_180 = None
    mul_906: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_272, 0.01)
    where_104: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_104, getitem_272, mul_906);  gt_104 = getitem_272 = mul_906 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_712: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    unsqueeze_713: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, 2);  unsqueeze_712 = None
    unsqueeze_714: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 3);  unsqueeze_713 = None
    sum_76: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_104, [0, 2, 3])
    sub_215: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_714)
    mul_907: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_104, sub_215);  sub_215 = None
    sum_77: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_907, [0, 2, 3]);  mul_907 = None
    mul_908: "f32[128]" = torch.ops.aten.mul.Tensor(sum_76, 0.0001220703125)
    unsqueeze_715: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_908, 0);  mul_908 = None
    unsqueeze_716: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_715, 2);  unsqueeze_715 = None
    unsqueeze_717: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 3);  unsqueeze_716 = None
    mul_909: "f32[128]" = torch.ops.aten.mul.Tensor(sum_77, 0.0001220703125)
    mul_910: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_911: "f32[128]" = torch.ops.aten.mul.Tensor(mul_909, mul_910);  mul_909 = mul_910 = None
    unsqueeze_718: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_911, 0);  mul_911 = None
    unsqueeze_719: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 2);  unsqueeze_718 = None
    unsqueeze_720: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 3);  unsqueeze_719 = None
    mul_912: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_59);  primals_59 = None
    unsqueeze_721: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_912, 0);  mul_912 = None
    unsqueeze_722: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 2);  unsqueeze_721 = None
    unsqueeze_723: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 3);  unsqueeze_722 = None
    sub_216: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_714);  convolution_29 = unsqueeze_714 = None
    mul_913: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_720);  sub_216 = unsqueeze_720 = None
    sub_217: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_104, mul_913);  where_104 = mul_913 = None
    sub_218: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_217, unsqueeze_717);  sub_217 = unsqueeze_717 = None
    mul_914: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_218, unsqueeze_723);  sub_218 = unsqueeze_723 = None
    mul_915: "f32[128]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_88);  sum_77 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_914, add_153, primals_164, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_914 = add_153 = primals_164 = None
    getitem_275: "f32[8, 128, 32, 32]" = convolution_backward_37[0]
    getitem_276: "f32[128, 128, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_371: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(add_370, getitem_275);  add_370 = getitem_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_182: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(where_28);  where_28 = None
    alias_183: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_182);  alias_182 = None
    gt_105: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(alias_183, 0);  alias_183 = None
    mul_916: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_371, 0.01)
    where_105: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_105, add_371, mul_916);  gt_105 = mul_916 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_724: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_725: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, 2);  unsqueeze_724 = None
    unsqueeze_726: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 3);  unsqueeze_725 = None
    sum_78: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_105, [0, 2, 3])
    sub_219: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_726)
    mul_917: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_105, sub_219);  sub_219 = None
    sum_79: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_917, [0, 2, 3]);  mul_917 = None
    mul_918: "f32[128]" = torch.ops.aten.mul.Tensor(sum_78, 0.0001220703125)
    unsqueeze_727: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_918, 0);  mul_918 = None
    unsqueeze_728: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_727, 2);  unsqueeze_727 = None
    unsqueeze_729: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 3);  unsqueeze_728 = None
    mul_919: "f32[128]" = torch.ops.aten.mul.Tensor(sum_79, 0.0001220703125)
    mul_920: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_921: "f32[128]" = torch.ops.aten.mul.Tensor(mul_919, mul_920);  mul_919 = mul_920 = None
    unsqueeze_730: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_921, 0);  mul_921 = None
    unsqueeze_731: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, 2);  unsqueeze_730 = None
    unsqueeze_732: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 3);  unsqueeze_731 = None
    mul_922: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_57);  primals_57 = None
    unsqueeze_733: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_922, 0);  mul_922 = None
    unsqueeze_734: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 2);  unsqueeze_733 = None
    unsqueeze_735: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 3);  unsqueeze_734 = None
    sub_220: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_726);  convolution_28 = unsqueeze_726 = None
    mul_923: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_732);  sub_220 = unsqueeze_732 = None
    sub_221: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_105, mul_923);  where_105 = mul_923 = None
    sub_222: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_221, unsqueeze_729);  sub_221 = unsqueeze_729 = None
    mul_924: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_735);  sub_222 = unsqueeze_735 = None
    mul_925: "f32[128]" = torch.ops.aten.mul.Tensor(sum_79, squeeze_85);  sum_79 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_924, where_27, primals_163, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_924 = primals_163 = None
    getitem_278: "f32[8, 128, 32, 32]" = convolution_backward_38[0]
    getitem_279: "f32[128, 128, 3, 3]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_185: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(where_27);  where_27 = None
    alias_186: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_185);  alias_185 = None
    gt_106: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(alias_186, 0);  alias_186 = None
    mul_926: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_278, 0.01)
    where_106: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_106, getitem_278, mul_926);  gt_106 = getitem_278 = mul_926 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_736: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_737: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, 2);  unsqueeze_736 = None
    unsqueeze_738: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 3);  unsqueeze_737 = None
    sum_80: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_106, [0, 2, 3])
    sub_223: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_738)
    mul_927: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_106, sub_223);  sub_223 = None
    sum_81: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_927, [0, 2, 3]);  mul_927 = None
    mul_928: "f32[128]" = torch.ops.aten.mul.Tensor(sum_80, 0.0001220703125)
    unsqueeze_739: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_928, 0);  mul_928 = None
    unsqueeze_740: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_739, 2);  unsqueeze_739 = None
    unsqueeze_741: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 3);  unsqueeze_740 = None
    mul_929: "f32[128]" = torch.ops.aten.mul.Tensor(sum_81, 0.0001220703125)
    mul_930: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_931: "f32[128]" = torch.ops.aten.mul.Tensor(mul_929, mul_930);  mul_929 = mul_930 = None
    unsqueeze_742: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_931, 0);  mul_931 = None
    unsqueeze_743: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, 2);  unsqueeze_742 = None
    unsqueeze_744: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 3);  unsqueeze_743 = None
    mul_932: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_55);  primals_55 = None
    unsqueeze_745: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_932, 0);  mul_932 = None
    unsqueeze_746: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_745, 2);  unsqueeze_745 = None
    unsqueeze_747: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 3);  unsqueeze_746 = None
    sub_224: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_738);  convolution_27 = unsqueeze_738 = None
    mul_933: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_744);  sub_224 = unsqueeze_744 = None
    sub_225: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_106, mul_933);  where_106 = mul_933 = None
    sub_226: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_225, unsqueeze_741);  sub_225 = unsqueeze_741 = None
    mul_934: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_226, unsqueeze_747);  sub_226 = unsqueeze_747 = None
    mul_935: "f32[128]" = torch.ops.aten.mul.Tensor(sum_81, squeeze_82);  sum_81 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_934, add_142, primals_162, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_934 = add_142 = primals_162 = None
    getitem_281: "f32[8, 128, 32, 32]" = convolution_backward_39[0]
    getitem_282: "f32[128, 128, 1, 1]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_372: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(add_371, getitem_281);  add_371 = getitem_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_188: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(where_26);  where_26 = None
    alias_189: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_188);  alias_188 = None
    gt_107: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(alias_189, 0);  alias_189 = None
    mul_936: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_372, 0.01)
    where_107: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_107, add_372, mul_936);  gt_107 = mul_936 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_748: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_749: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, 2);  unsqueeze_748 = None
    unsqueeze_750: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 3);  unsqueeze_749 = None
    sum_82: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_107, [0, 2, 3])
    sub_227: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_750)
    mul_937: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_107, sub_227);  sub_227 = None
    sum_83: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_937, [0, 2, 3]);  mul_937 = None
    mul_938: "f32[128]" = torch.ops.aten.mul.Tensor(sum_82, 0.0001220703125)
    unsqueeze_751: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_938, 0);  mul_938 = None
    unsqueeze_752: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_751, 2);  unsqueeze_751 = None
    unsqueeze_753: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 3);  unsqueeze_752 = None
    mul_939: "f32[128]" = torch.ops.aten.mul.Tensor(sum_83, 0.0001220703125)
    mul_940: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_941: "f32[128]" = torch.ops.aten.mul.Tensor(mul_939, mul_940);  mul_939 = mul_940 = None
    unsqueeze_754: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_941, 0);  mul_941 = None
    unsqueeze_755: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, 2);  unsqueeze_754 = None
    unsqueeze_756: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 3);  unsqueeze_755 = None
    mul_942: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_53);  primals_53 = None
    unsqueeze_757: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_942, 0);  mul_942 = None
    unsqueeze_758: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_757, 2);  unsqueeze_757 = None
    unsqueeze_759: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 3);  unsqueeze_758 = None
    sub_228: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_750);  convolution_26 = unsqueeze_750 = None
    mul_943: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_756);  sub_228 = unsqueeze_756 = None
    sub_229: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_107, mul_943);  where_107 = mul_943 = None
    sub_230: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_229, unsqueeze_753);  sub_229 = unsqueeze_753 = None
    mul_944: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_759);  sub_230 = unsqueeze_759 = None
    mul_945: "f32[128]" = torch.ops.aten.mul.Tensor(sum_83, squeeze_79);  sum_83 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_944, where_25, primals_161, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_944 = primals_161 = None
    getitem_284: "f32[8, 128, 32, 32]" = convolution_backward_40[0]
    getitem_285: "f32[128, 128, 3, 3]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_191: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(where_25);  where_25 = None
    alias_192: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_191);  alias_191 = None
    gt_108: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(alias_192, 0);  alias_192 = None
    mul_946: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_284, 0.01)
    where_108: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_108, getitem_284, mul_946);  gt_108 = getitem_284 = mul_946 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_760: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_761: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, 2);  unsqueeze_760 = None
    unsqueeze_762: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 3);  unsqueeze_761 = None
    sum_84: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_108, [0, 2, 3])
    sub_231: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_762)
    mul_947: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_108, sub_231);  sub_231 = None
    sum_85: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_947, [0, 2, 3]);  mul_947 = None
    mul_948: "f32[128]" = torch.ops.aten.mul.Tensor(sum_84, 0.0001220703125)
    unsqueeze_763: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_948, 0);  mul_948 = None
    unsqueeze_764: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_763, 2);  unsqueeze_763 = None
    unsqueeze_765: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 3);  unsqueeze_764 = None
    mul_949: "f32[128]" = torch.ops.aten.mul.Tensor(sum_85, 0.0001220703125)
    mul_950: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_951: "f32[128]" = torch.ops.aten.mul.Tensor(mul_949, mul_950);  mul_949 = mul_950 = None
    unsqueeze_766: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_951, 0);  mul_951 = None
    unsqueeze_767: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, 2);  unsqueeze_766 = None
    unsqueeze_768: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 3);  unsqueeze_767 = None
    mul_952: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_51);  primals_51 = None
    unsqueeze_769: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_952, 0);  mul_952 = None
    unsqueeze_770: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_769, 2);  unsqueeze_769 = None
    unsqueeze_771: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 3);  unsqueeze_770 = None
    sub_232: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_762);  convolution_25 = unsqueeze_762 = None
    mul_953: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_768);  sub_232 = unsqueeze_768 = None
    sub_233: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_108, mul_953);  where_108 = mul_953 = None
    sub_234: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_233, unsqueeze_765);  sub_233 = unsqueeze_765 = None
    mul_954: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_234, unsqueeze_771);  sub_234 = unsqueeze_771 = None
    mul_955: "f32[128]" = torch.ops.aten.mul.Tensor(sum_85, squeeze_76);  sum_85 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_954, add_131, primals_160, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_954 = add_131 = primals_160 = None
    getitem_287: "f32[8, 128, 32, 32]" = convolution_backward_41[0]
    getitem_288: "f32[128, 128, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_373: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(add_372, getitem_287);  add_372 = getitem_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_194: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(where_24);  where_24 = None
    alias_195: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_194);  alias_194 = None
    gt_109: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(alias_195, 0);  alias_195 = None
    mul_956: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_373, 0.01)
    where_109: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_109, add_373, mul_956);  gt_109 = mul_956 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_772: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_773: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, 2);  unsqueeze_772 = None
    unsqueeze_774: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 3);  unsqueeze_773 = None
    sum_86: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_109, [0, 2, 3])
    sub_235: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_774)
    mul_957: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_109, sub_235);  sub_235 = None
    sum_87: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_957, [0, 2, 3]);  mul_957 = None
    mul_958: "f32[128]" = torch.ops.aten.mul.Tensor(sum_86, 0.0001220703125)
    unsqueeze_775: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_958, 0);  mul_958 = None
    unsqueeze_776: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_775, 2);  unsqueeze_775 = None
    unsqueeze_777: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 3);  unsqueeze_776 = None
    mul_959: "f32[128]" = torch.ops.aten.mul.Tensor(sum_87, 0.0001220703125)
    mul_960: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_961: "f32[128]" = torch.ops.aten.mul.Tensor(mul_959, mul_960);  mul_959 = mul_960 = None
    unsqueeze_778: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_961, 0);  mul_961 = None
    unsqueeze_779: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, 2);  unsqueeze_778 = None
    unsqueeze_780: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 3);  unsqueeze_779 = None
    mul_962: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_49);  primals_49 = None
    unsqueeze_781: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_962, 0);  mul_962 = None
    unsqueeze_782: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_781, 2);  unsqueeze_781 = None
    unsqueeze_783: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 3);  unsqueeze_782 = None
    sub_236: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_774);  convolution_24 = unsqueeze_774 = None
    mul_963: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_780);  sub_236 = unsqueeze_780 = None
    sub_237: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_109, mul_963);  where_109 = mul_963 = None
    sub_238: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_237, unsqueeze_777);  sub_237 = unsqueeze_777 = None
    mul_964: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_238, unsqueeze_783);  sub_238 = unsqueeze_783 = None
    mul_965: "f32[128]" = torch.ops.aten.mul.Tensor(sum_87, squeeze_73);  sum_87 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_964, where_23, primals_159, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_964 = primals_159 = None
    getitem_290: "f32[8, 128, 32, 32]" = convolution_backward_42[0]
    getitem_291: "f32[128, 128, 3, 3]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_197: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(where_23);  where_23 = None
    alias_198: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_197);  alias_197 = None
    gt_110: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(alias_198, 0);  alias_198 = None
    mul_966: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_290, 0.01)
    where_110: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_110, getitem_290, mul_966);  gt_110 = getitem_290 = mul_966 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_784: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_785: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, 2);  unsqueeze_784 = None
    unsqueeze_786: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 3);  unsqueeze_785 = None
    sum_88: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_110, [0, 2, 3])
    sub_239: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_786)
    mul_967: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_110, sub_239);  sub_239 = None
    sum_89: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_967, [0, 2, 3]);  mul_967 = None
    mul_968: "f32[128]" = torch.ops.aten.mul.Tensor(sum_88, 0.0001220703125)
    unsqueeze_787: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_968, 0);  mul_968 = None
    unsqueeze_788: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_787, 2);  unsqueeze_787 = None
    unsqueeze_789: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 3);  unsqueeze_788 = None
    mul_969: "f32[128]" = torch.ops.aten.mul.Tensor(sum_89, 0.0001220703125)
    mul_970: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_971: "f32[128]" = torch.ops.aten.mul.Tensor(mul_969, mul_970);  mul_969 = mul_970 = None
    unsqueeze_790: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_971, 0);  mul_971 = None
    unsqueeze_791: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, 2);  unsqueeze_790 = None
    unsqueeze_792: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_791, 3);  unsqueeze_791 = None
    mul_972: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_47);  primals_47 = None
    unsqueeze_793: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_972, 0);  mul_972 = None
    unsqueeze_794: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_793, 2);  unsqueeze_793 = None
    unsqueeze_795: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 3);  unsqueeze_794 = None
    sub_240: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_786);  convolution_23 = unsqueeze_786 = None
    mul_973: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_792);  sub_240 = unsqueeze_792 = None
    sub_241: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_110, mul_973);  where_110 = mul_973 = None
    sub_242: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_241, unsqueeze_789);  sub_241 = unsqueeze_789 = None
    mul_974: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_242, unsqueeze_795);  sub_242 = unsqueeze_795 = None
    mul_975: "f32[128]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_70);  sum_89 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_974, add_120, primals_158, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_974 = add_120 = primals_158 = None
    getitem_293: "f32[8, 128, 32, 32]" = convolution_backward_43[0]
    getitem_294: "f32[128, 128, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_374: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(add_373, getitem_293);  add_373 = getitem_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_200: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(where_22);  where_22 = None
    alias_201: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_200);  alias_200 = None
    gt_111: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(alias_201, 0);  alias_201 = None
    mul_976: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_374, 0.01)
    where_111: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_111, add_374, mul_976);  gt_111 = mul_976 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_796: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_797: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, 2);  unsqueeze_796 = None
    unsqueeze_798: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 3);  unsqueeze_797 = None
    sum_90: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_111, [0, 2, 3])
    sub_243: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_798)
    mul_977: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_111, sub_243);  sub_243 = None
    sum_91: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_977, [0, 2, 3]);  mul_977 = None
    mul_978: "f32[128]" = torch.ops.aten.mul.Tensor(sum_90, 0.0001220703125)
    unsqueeze_799: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_978, 0);  mul_978 = None
    unsqueeze_800: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_799, 2);  unsqueeze_799 = None
    unsqueeze_801: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 3);  unsqueeze_800 = None
    mul_979: "f32[128]" = torch.ops.aten.mul.Tensor(sum_91, 0.0001220703125)
    mul_980: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_981: "f32[128]" = torch.ops.aten.mul.Tensor(mul_979, mul_980);  mul_979 = mul_980 = None
    unsqueeze_802: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_981, 0);  mul_981 = None
    unsqueeze_803: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_802, 2);  unsqueeze_802 = None
    unsqueeze_804: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_803, 3);  unsqueeze_803 = None
    mul_982: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_45);  primals_45 = None
    unsqueeze_805: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_982, 0);  mul_982 = None
    unsqueeze_806: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_805, 2);  unsqueeze_805 = None
    unsqueeze_807: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 3);  unsqueeze_806 = None
    sub_244: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_798);  convolution_22 = unsqueeze_798 = None
    mul_983: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_804);  sub_244 = unsqueeze_804 = None
    sub_245: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_111, mul_983);  where_111 = mul_983 = None
    sub_246: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_245, unsqueeze_801);  sub_245 = unsqueeze_801 = None
    mul_984: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_246, unsqueeze_807);  sub_246 = unsqueeze_807 = None
    mul_985: "f32[128]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_67);  sum_91 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_984, where_21, primals_157, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_984 = primals_157 = None
    getitem_296: "f32[8, 128, 32, 32]" = convolution_backward_44[0]
    getitem_297: "f32[128, 128, 3, 3]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_203: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(where_21);  where_21 = None
    alias_204: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_203);  alias_203 = None
    gt_112: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(alias_204, 0);  alias_204 = None
    mul_986: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_296, 0.01)
    where_112: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_112, getitem_296, mul_986);  gt_112 = getitem_296 = mul_986 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_808: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_809: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, 2);  unsqueeze_808 = None
    unsqueeze_810: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 3);  unsqueeze_809 = None
    sum_92: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_112, [0, 2, 3])
    sub_247: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_810)
    mul_987: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_112, sub_247);  sub_247 = None
    sum_93: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_987, [0, 2, 3]);  mul_987 = None
    mul_988: "f32[128]" = torch.ops.aten.mul.Tensor(sum_92, 0.0001220703125)
    unsqueeze_811: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_988, 0);  mul_988 = None
    unsqueeze_812: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_811, 2);  unsqueeze_811 = None
    unsqueeze_813: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 3);  unsqueeze_812 = None
    mul_989: "f32[128]" = torch.ops.aten.mul.Tensor(sum_93, 0.0001220703125)
    mul_990: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_991: "f32[128]" = torch.ops.aten.mul.Tensor(mul_989, mul_990);  mul_989 = mul_990 = None
    unsqueeze_814: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_991, 0);  mul_991 = None
    unsqueeze_815: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_814, 2);  unsqueeze_814 = None
    unsqueeze_816: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_815, 3);  unsqueeze_815 = None
    mul_992: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_43);  primals_43 = None
    unsqueeze_817: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_992, 0);  mul_992 = None
    unsqueeze_818: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_817, 2);  unsqueeze_817 = None
    unsqueeze_819: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 3);  unsqueeze_818 = None
    sub_248: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_810);  convolution_21 = unsqueeze_810 = None
    mul_993: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_816);  sub_248 = unsqueeze_816 = None
    sub_249: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_112, mul_993);  where_112 = mul_993 = None
    sub_250: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_249, unsqueeze_813);  sub_249 = unsqueeze_813 = None
    mul_994: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_250, unsqueeze_819);  sub_250 = unsqueeze_819 = None
    mul_995: "f32[128]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_64);  sum_93 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_994, add_109, primals_156, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_994 = add_109 = primals_156 = None
    getitem_299: "f32[8, 128, 32, 32]" = convolution_backward_45[0]
    getitem_300: "f32[128, 128, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_375: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(add_374, getitem_299);  add_374 = getitem_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_206: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(where_20);  where_20 = None
    alias_207: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_206);  alias_206 = None
    gt_113: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(alias_207, 0);  alias_207 = None
    mul_996: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_375, 0.01)
    where_113: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_113, add_375, mul_996);  gt_113 = mul_996 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_820: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_821: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, 2);  unsqueeze_820 = None
    unsqueeze_822: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 3);  unsqueeze_821 = None
    sum_94: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_113, [0, 2, 3])
    sub_251: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_822)
    mul_997: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_113, sub_251);  sub_251 = None
    sum_95: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_997, [0, 2, 3]);  mul_997 = None
    mul_998: "f32[128]" = torch.ops.aten.mul.Tensor(sum_94, 0.0001220703125)
    unsqueeze_823: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_998, 0);  mul_998 = None
    unsqueeze_824: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_823, 2);  unsqueeze_823 = None
    unsqueeze_825: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 3);  unsqueeze_824 = None
    mul_999: "f32[128]" = torch.ops.aten.mul.Tensor(sum_95, 0.0001220703125)
    mul_1000: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_1001: "f32[128]" = torch.ops.aten.mul.Tensor(mul_999, mul_1000);  mul_999 = mul_1000 = None
    unsqueeze_826: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1001, 0);  mul_1001 = None
    unsqueeze_827: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, 2);  unsqueeze_826 = None
    unsqueeze_828: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_827, 3);  unsqueeze_827 = None
    mul_1002: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_41);  primals_41 = None
    unsqueeze_829: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1002, 0);  mul_1002 = None
    unsqueeze_830: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_829, 2);  unsqueeze_829 = None
    unsqueeze_831: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 3);  unsqueeze_830 = None
    sub_252: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_822);  convolution_20 = unsqueeze_822 = None
    mul_1003: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_828);  sub_252 = unsqueeze_828 = None
    sub_253: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_113, mul_1003);  where_113 = mul_1003 = None
    sub_254: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_253, unsqueeze_825);  sub_253 = unsqueeze_825 = None
    mul_1004: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_254, unsqueeze_831);  sub_254 = unsqueeze_831 = None
    mul_1005: "f32[128]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_61);  sum_95 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_1004, where_19, primals_155, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1004 = primals_155 = None
    getitem_302: "f32[8, 128, 32, 32]" = convolution_backward_46[0]
    getitem_303: "f32[128, 128, 3, 3]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_209: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(where_19);  where_19 = None
    alias_210: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_209);  alias_209 = None
    gt_114: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(alias_210, 0);  alias_210 = None
    mul_1006: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_302, 0.01)
    where_114: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_114, getitem_302, mul_1006);  gt_114 = getitem_302 = mul_1006 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_832: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_833: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, 2);  unsqueeze_832 = None
    unsqueeze_834: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 3);  unsqueeze_833 = None
    sum_96: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_114, [0, 2, 3])
    sub_255: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_834)
    mul_1007: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_114, sub_255);  sub_255 = None
    sum_97: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1007, [0, 2, 3]);  mul_1007 = None
    mul_1008: "f32[128]" = torch.ops.aten.mul.Tensor(sum_96, 0.0001220703125)
    unsqueeze_835: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1008, 0);  mul_1008 = None
    unsqueeze_836: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_835, 2);  unsqueeze_835 = None
    unsqueeze_837: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, 3);  unsqueeze_836 = None
    mul_1009: "f32[128]" = torch.ops.aten.mul.Tensor(sum_97, 0.0001220703125)
    mul_1010: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_1011: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1009, mul_1010);  mul_1009 = mul_1010 = None
    unsqueeze_838: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1011, 0);  mul_1011 = None
    unsqueeze_839: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_838, 2);  unsqueeze_838 = None
    unsqueeze_840: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_839, 3);  unsqueeze_839 = None
    mul_1012: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_39);  primals_39 = None
    unsqueeze_841: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1012, 0);  mul_1012 = None
    unsqueeze_842: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_841, 2);  unsqueeze_841 = None
    unsqueeze_843: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, 3);  unsqueeze_842 = None
    sub_256: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_834);  convolution_19 = unsqueeze_834 = None
    mul_1013: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_840);  sub_256 = unsqueeze_840 = None
    sub_257: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_114, mul_1013);  where_114 = mul_1013 = None
    sub_258: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_257, unsqueeze_837);  sub_257 = unsqueeze_837 = None
    mul_1014: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_258, unsqueeze_843);  sub_258 = unsqueeze_843 = None
    mul_1015: "f32[128]" = torch.ops.aten.mul.Tensor(sum_97, squeeze_58);  sum_97 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_1014, add_98, primals_154, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1014 = add_98 = primals_154 = None
    getitem_305: "f32[8, 128, 32, 32]" = convolution_backward_47[0]
    getitem_306: "f32[128, 128, 1, 1]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_376: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(add_375, getitem_305);  add_375 = getitem_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_212: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(where_18);  where_18 = None
    alias_213: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_212);  alias_212 = None
    gt_115: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(alias_213, 0);  alias_213 = None
    mul_1016: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_376, 0.01)
    where_115: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_115, add_376, mul_1016);  gt_115 = mul_1016 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_844: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_845: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_844, 2);  unsqueeze_844 = None
    unsqueeze_846: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 3);  unsqueeze_845 = None
    sum_98: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_115, [0, 2, 3])
    sub_259: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_846)
    mul_1017: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_115, sub_259);  sub_259 = None
    sum_99: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1017, [0, 2, 3]);  mul_1017 = None
    mul_1018: "f32[128]" = torch.ops.aten.mul.Tensor(sum_98, 0.0001220703125)
    unsqueeze_847: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1018, 0);  mul_1018 = None
    unsqueeze_848: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_847, 2);  unsqueeze_847 = None
    unsqueeze_849: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, 3);  unsqueeze_848 = None
    mul_1019: "f32[128]" = torch.ops.aten.mul.Tensor(sum_99, 0.0001220703125)
    mul_1020: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_1021: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1019, mul_1020);  mul_1019 = mul_1020 = None
    unsqueeze_850: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1021, 0);  mul_1021 = None
    unsqueeze_851: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_850, 2);  unsqueeze_850 = None
    unsqueeze_852: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_851, 3);  unsqueeze_851 = None
    mul_1022: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_37);  primals_37 = None
    unsqueeze_853: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1022, 0);  mul_1022 = None
    unsqueeze_854: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_853, 2);  unsqueeze_853 = None
    unsqueeze_855: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, 3);  unsqueeze_854 = None
    sub_260: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_846);  convolution_18 = unsqueeze_846 = None
    mul_1023: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_852);  sub_260 = unsqueeze_852 = None
    sub_261: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_115, mul_1023);  where_115 = mul_1023 = None
    sub_262: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_261, unsqueeze_849);  sub_261 = unsqueeze_849 = None
    mul_1024: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_262, unsqueeze_855);  sub_262 = unsqueeze_855 = None
    mul_1025: "f32[128]" = torch.ops.aten.mul.Tensor(sum_99, squeeze_55);  sum_99 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_1024, where_17, primals_153, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1024 = primals_153 = None
    getitem_308: "f32[8, 128, 32, 32]" = convolution_backward_48[0]
    getitem_309: "f32[128, 128, 3, 3]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_215: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(where_17);  where_17 = None
    alias_216: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_215);  alias_215 = None
    gt_116: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(alias_216, 0);  alias_216 = None
    mul_1026: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_308, 0.01)
    where_116: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_116, getitem_308, mul_1026);  gt_116 = getitem_308 = mul_1026 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_856: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_857: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, 2);  unsqueeze_856 = None
    unsqueeze_858: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 3);  unsqueeze_857 = None
    sum_100: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_116, [0, 2, 3])
    sub_263: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_858)
    mul_1027: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_116, sub_263);  sub_263 = None
    sum_101: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1027, [0, 2, 3]);  mul_1027 = None
    mul_1028: "f32[128]" = torch.ops.aten.mul.Tensor(sum_100, 0.0001220703125)
    unsqueeze_859: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1028, 0);  mul_1028 = None
    unsqueeze_860: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_859, 2);  unsqueeze_859 = None
    unsqueeze_861: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, 3);  unsqueeze_860 = None
    mul_1029: "f32[128]" = torch.ops.aten.mul.Tensor(sum_101, 0.0001220703125)
    mul_1030: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_1031: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1029, mul_1030);  mul_1029 = mul_1030 = None
    unsqueeze_862: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1031, 0);  mul_1031 = None
    unsqueeze_863: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_862, 2);  unsqueeze_862 = None
    unsqueeze_864: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_863, 3);  unsqueeze_863 = None
    mul_1032: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_35);  primals_35 = None
    unsqueeze_865: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1032, 0);  mul_1032 = None
    unsqueeze_866: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_865, 2);  unsqueeze_865 = None
    unsqueeze_867: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, 3);  unsqueeze_866 = None
    sub_264: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_858);  convolution_17 = unsqueeze_858 = None
    mul_1033: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_864);  sub_264 = unsqueeze_864 = None
    sub_265: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_116, mul_1033);  where_116 = mul_1033 = None
    sub_266: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_265, unsqueeze_861);  sub_265 = unsqueeze_861 = None
    mul_1034: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_266, unsqueeze_867);  sub_266 = unsqueeze_867 = None
    mul_1035: "f32[128]" = torch.ops.aten.mul.Tensor(sum_101, squeeze_52);  sum_101 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_1034, getitem_49, primals_152, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1034 = getitem_49 = primals_152 = None
    getitem_311: "f32[8, 128, 32, 32]" = convolution_backward_49[0]
    getitem_312: "f32[128, 128, 1, 1]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_377: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(add_376, getitem_311);  add_376 = getitem_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:336, code: xs, xb = x.split(self.expand_chs // 2, dim=1)
    cat_7: "f32[8, 256, 32, 32]" = torch.ops.aten.cat.default([slice_5, add_377], 1);  slice_5 = add_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_218: "f32[8, 256, 32, 32]" = torch.ops.aten.alias.default(where_16);  where_16 = None
    alias_219: "f32[8, 256, 32, 32]" = torch.ops.aten.alias.default(alias_218);  alias_218 = None
    gt_117: "b8[8, 256, 32, 32]" = torch.ops.aten.gt.Scalar(alias_219, 0);  alias_219 = None
    mul_1036: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(cat_7, 0.01)
    where_117: "f32[8, 256, 32, 32]" = torch.ops.aten.where.self(gt_117, cat_7, mul_1036);  gt_117 = cat_7 = mul_1036 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_868: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_869: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, 2);  unsqueeze_868 = None
    unsqueeze_870: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 3);  unsqueeze_869 = None
    sum_102: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_117, [0, 2, 3])
    sub_267: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_870)
    mul_1037: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(where_117, sub_267);  sub_267 = None
    sum_103: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1037, [0, 2, 3]);  mul_1037 = None
    mul_1038: "f32[256]" = torch.ops.aten.mul.Tensor(sum_102, 0.0001220703125)
    unsqueeze_871: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1038, 0);  mul_1038 = None
    unsqueeze_872: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_871, 2);  unsqueeze_871 = None
    unsqueeze_873: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, 3);  unsqueeze_872 = None
    mul_1039: "f32[256]" = torch.ops.aten.mul.Tensor(sum_103, 0.0001220703125)
    mul_1040: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_1041: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1039, mul_1040);  mul_1039 = mul_1040 = None
    unsqueeze_874: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1041, 0);  mul_1041 = None
    unsqueeze_875: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_874, 2);  unsqueeze_874 = None
    unsqueeze_876: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_875, 3);  unsqueeze_875 = None
    mul_1042: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_33);  primals_33 = None
    unsqueeze_877: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1042, 0);  mul_1042 = None
    unsqueeze_878: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_877, 2);  unsqueeze_877 = None
    unsqueeze_879: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, 3);  unsqueeze_878 = None
    sub_268: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_870);  convolution_16 = unsqueeze_870 = None
    mul_1043: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_876);  sub_268 = unsqueeze_876 = None
    sub_269: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(where_117, mul_1043);  where_117 = mul_1043 = None
    sub_270: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(sub_269, unsqueeze_873);  sub_269 = unsqueeze_873 = None
    mul_1044: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_270, unsqueeze_879);  sub_270 = unsqueeze_879 = None
    mul_1045: "f32[256]" = torch.ops.aten.mul.Tensor(sum_103, squeeze_49);  sum_103 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_1044, where_15, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1044 = primals_151 = None
    getitem_314: "f32[8, 256, 32, 32]" = convolution_backward_50[0]
    getitem_315: "f32[256, 256, 1, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_221: "f32[8, 256, 32, 32]" = torch.ops.aten.alias.default(where_15);  where_15 = None
    alias_222: "f32[8, 256, 32, 32]" = torch.ops.aten.alias.default(alias_221);  alias_221 = None
    gt_118: "b8[8, 256, 32, 32]" = torch.ops.aten.gt.Scalar(alias_222, 0);  alias_222 = None
    mul_1046: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_314, 0.01)
    where_118: "f32[8, 256, 32, 32]" = torch.ops.aten.where.self(gt_118, getitem_314, mul_1046);  gt_118 = getitem_314 = mul_1046 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_880: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_881: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_880, 2);  unsqueeze_880 = None
    unsqueeze_882: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 3);  unsqueeze_881 = None
    sum_104: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_118, [0, 2, 3])
    sub_271: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_882)
    mul_1047: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(where_118, sub_271);  sub_271 = None
    sum_105: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1047, [0, 2, 3]);  mul_1047 = None
    mul_1048: "f32[256]" = torch.ops.aten.mul.Tensor(sum_104, 0.0001220703125)
    unsqueeze_883: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1048, 0);  mul_1048 = None
    unsqueeze_884: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_883, 2);  unsqueeze_883 = None
    unsqueeze_885: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, 3);  unsqueeze_884 = None
    mul_1049: "f32[256]" = torch.ops.aten.mul.Tensor(sum_105, 0.0001220703125)
    mul_1050: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_1051: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1049, mul_1050);  mul_1049 = mul_1050 = None
    unsqueeze_886: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1051, 0);  mul_1051 = None
    unsqueeze_887: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_886, 2);  unsqueeze_886 = None
    unsqueeze_888: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_887, 3);  unsqueeze_887 = None
    mul_1052: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_31);  primals_31 = None
    unsqueeze_889: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1052, 0);  mul_1052 = None
    unsqueeze_890: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_889, 2);  unsqueeze_889 = None
    unsqueeze_891: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, 3);  unsqueeze_890 = None
    sub_272: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_882);  convolution_15 = unsqueeze_882 = None
    mul_1053: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_888);  sub_272 = unsqueeze_888 = None
    sub_273: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(where_118, mul_1053);  where_118 = mul_1053 = None
    sub_274: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(sub_273, unsqueeze_885);  sub_273 = unsqueeze_885 = None
    mul_1054: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_274, unsqueeze_891);  sub_274 = unsqueeze_891 = None
    mul_1055: "f32[256]" = torch.ops.aten.mul.Tensor(sum_105, squeeze_46);  sum_105 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:126, code: x = self.conv(x)
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_1054, where_14, primals_150, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1054 = primals_150 = None
    getitem_317: "f32[8, 128, 64, 64]" = convolution_backward_51[0]
    getitem_318: "f32[256, 128, 3, 3]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_224: "f32[8, 128, 64, 64]" = torch.ops.aten.alias.default(where_14);  where_14 = None
    alias_225: "f32[8, 128, 64, 64]" = torch.ops.aten.alias.default(alias_224);  alias_224 = None
    gt_119: "b8[8, 128, 64, 64]" = torch.ops.aten.gt.Scalar(alias_225, 0);  alias_225 = None
    mul_1056: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_317, 0.01)
    where_119: "f32[8, 128, 64, 64]" = torch.ops.aten.where.self(gt_119, getitem_317, mul_1056);  gt_119 = getitem_317 = mul_1056 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_892: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_893: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_892, 2);  unsqueeze_892 = None
    unsqueeze_894: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_893, 3);  unsqueeze_893 = None
    sum_106: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_119, [0, 2, 3])
    sub_275: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_894)
    mul_1057: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(where_119, sub_275);  sub_275 = None
    sum_107: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1057, [0, 2, 3]);  mul_1057 = None
    mul_1058: "f32[128]" = torch.ops.aten.mul.Tensor(sum_106, 3.0517578125e-05)
    unsqueeze_895: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1058, 0);  mul_1058 = None
    unsqueeze_896: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_895, 2);  unsqueeze_895 = None
    unsqueeze_897: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, 3);  unsqueeze_896 = None
    mul_1059: "f32[128]" = torch.ops.aten.mul.Tensor(sum_107, 3.0517578125e-05)
    mul_1060: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_1061: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1059, mul_1060);  mul_1059 = mul_1060 = None
    unsqueeze_898: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1061, 0);  mul_1061 = None
    unsqueeze_899: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_898, 2);  unsqueeze_898 = None
    unsqueeze_900: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_899, 3);  unsqueeze_899 = None
    mul_1062: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_29);  primals_29 = None
    unsqueeze_901: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1062, 0);  mul_1062 = None
    unsqueeze_902: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_901, 2);  unsqueeze_901 = None
    unsqueeze_903: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, 3);  unsqueeze_902 = None
    sub_276: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_894);  convolution_14 = unsqueeze_894 = None
    mul_1063: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_276, unsqueeze_900);  sub_276 = unsqueeze_900 = None
    sub_277: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(where_119, mul_1063);  where_119 = mul_1063 = None
    sub_278: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(sub_277, unsqueeze_897);  sub_277 = unsqueeze_897 = None
    mul_1064: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_278, unsqueeze_903);  sub_278 = unsqueeze_903 = None
    mul_1065: "f32[128]" = torch.ops.aten.mul.Tensor(sum_107, squeeze_43);  sum_107 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_1064, cat_1, primals_149, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1064 = cat_1 = primals_149 = None
    getitem_320: "f32[8, 128, 64, 64]" = convolution_backward_52[0]
    getitem_321: "f32[128, 128, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:339, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
    slice_7: "f32[8, 64, 64, 64]" = torch.ops.aten.slice.Tensor(getitem_320, 1, 0, 64)
    slice_8: "f32[8, 64, 64, 64]" = torch.ops.aten.slice.Tensor(getitem_320, 1, 64, 128);  getitem_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_227: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(where_13);  where_13 = None
    alias_228: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(alias_227);  alias_227 = None
    gt_120: "b8[8, 64, 64, 64]" = torch.ops.aten.gt.Scalar(alias_228, 0);  alias_228 = None
    mul_1066: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(slice_8, 0.01)
    where_120: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(gt_120, slice_8, mul_1066);  gt_120 = slice_8 = mul_1066 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_904: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_905: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_904, 2);  unsqueeze_904 = None
    unsqueeze_906: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_905, 3);  unsqueeze_905 = None
    sum_108: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_120, [0, 2, 3])
    sub_279: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_906)
    mul_1067: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(where_120, sub_279);  sub_279 = None
    sum_109: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1067, [0, 2, 3]);  mul_1067 = None
    mul_1068: "f32[64]" = torch.ops.aten.mul.Tensor(sum_108, 3.0517578125e-05)
    unsqueeze_907: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1068, 0);  mul_1068 = None
    unsqueeze_908: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_907, 2);  unsqueeze_907 = None
    unsqueeze_909: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, 3);  unsqueeze_908 = None
    mul_1069: "f32[64]" = torch.ops.aten.mul.Tensor(sum_109, 3.0517578125e-05)
    mul_1070: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_1071: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1069, mul_1070);  mul_1069 = mul_1070 = None
    unsqueeze_910: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1071, 0);  mul_1071 = None
    unsqueeze_911: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_910, 2);  unsqueeze_910 = None
    unsqueeze_912: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_911, 3);  unsqueeze_911 = None
    mul_1072: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_27);  primals_27 = None
    unsqueeze_913: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1072, 0);  mul_1072 = None
    unsqueeze_914: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_913, 2);  unsqueeze_913 = None
    unsqueeze_915: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, 3);  unsqueeze_914 = None
    sub_280: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_906);  convolution_13 = unsqueeze_906 = None
    mul_1073: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_280, unsqueeze_912);  sub_280 = unsqueeze_912 = None
    sub_281: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(where_120, mul_1073);  where_120 = mul_1073 = None
    sub_282: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_281, unsqueeze_909);  sub_281 = unsqueeze_909 = None
    mul_1074: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_282, unsqueeze_915);  sub_282 = unsqueeze_915 = None
    mul_1075: "f32[64]" = torch.ops.aten.mul.Tensor(sum_109, squeeze_40);  sum_109 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_1074, add_67, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1074 = add_67 = primals_148 = None
    getitem_323: "f32[8, 64, 64, 64]" = convolution_backward_53[0]
    getitem_324: "f32[64, 64, 1, 1]" = convolution_backward_53[1];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_230: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(where_12);  where_12 = None
    alias_231: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(alias_230);  alias_230 = None
    gt_121: "b8[8, 64, 64, 64]" = torch.ops.aten.gt.Scalar(alias_231, 0);  alias_231 = None
    mul_1076: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_323, 0.01)
    where_121: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(gt_121, getitem_323, mul_1076);  gt_121 = mul_1076 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_916: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_917: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_916, 2);  unsqueeze_916 = None
    unsqueeze_918: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_917, 3);  unsqueeze_917 = None
    sum_110: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_121, [0, 2, 3])
    sub_283: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_918)
    mul_1077: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(where_121, sub_283);  sub_283 = None
    sum_111: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1077, [0, 2, 3]);  mul_1077 = None
    mul_1078: "f32[64]" = torch.ops.aten.mul.Tensor(sum_110, 3.0517578125e-05)
    unsqueeze_919: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1078, 0);  mul_1078 = None
    unsqueeze_920: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_919, 2);  unsqueeze_919 = None
    unsqueeze_921: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_920, 3);  unsqueeze_920 = None
    mul_1079: "f32[64]" = torch.ops.aten.mul.Tensor(sum_111, 3.0517578125e-05)
    mul_1080: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_1081: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1079, mul_1080);  mul_1079 = mul_1080 = None
    unsqueeze_922: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1081, 0);  mul_1081 = None
    unsqueeze_923: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_922, 2);  unsqueeze_922 = None
    unsqueeze_924: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_923, 3);  unsqueeze_923 = None
    mul_1082: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_25);  primals_25 = None
    unsqueeze_925: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1082, 0);  mul_1082 = None
    unsqueeze_926: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_925, 2);  unsqueeze_925 = None
    unsqueeze_927: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, 3);  unsqueeze_926 = None
    sub_284: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_918);  convolution_12 = unsqueeze_918 = None
    mul_1083: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_924);  sub_284 = unsqueeze_924 = None
    sub_285: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(where_121, mul_1083);  where_121 = mul_1083 = None
    sub_286: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_285, unsqueeze_921);  sub_285 = unsqueeze_921 = None
    mul_1084: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_286, unsqueeze_927);  sub_286 = unsqueeze_927 = None
    mul_1085: "f32[64]" = torch.ops.aten.mul.Tensor(sum_111, squeeze_37);  sum_111 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_1084, where_11, primals_147, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1084 = primals_147 = None
    getitem_326: "f32[8, 64, 64, 64]" = convolution_backward_54[0]
    getitem_327: "f32[64, 64, 3, 3]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_233: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(where_11);  where_11 = None
    alias_234: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(alias_233);  alias_233 = None
    gt_122: "b8[8, 64, 64, 64]" = torch.ops.aten.gt.Scalar(alias_234, 0);  alias_234 = None
    mul_1086: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_326, 0.01)
    where_122: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(gt_122, getitem_326, mul_1086);  gt_122 = getitem_326 = mul_1086 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_928: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_929: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_928, 2);  unsqueeze_928 = None
    unsqueeze_930: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_929, 3);  unsqueeze_929 = None
    sum_112: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_122, [0, 2, 3])
    sub_287: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_930)
    mul_1087: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(where_122, sub_287);  sub_287 = None
    sum_113: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1087, [0, 2, 3]);  mul_1087 = None
    mul_1088: "f32[64]" = torch.ops.aten.mul.Tensor(sum_112, 3.0517578125e-05)
    unsqueeze_931: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1088, 0);  mul_1088 = None
    unsqueeze_932: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_931, 2);  unsqueeze_931 = None
    unsqueeze_933: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_932, 3);  unsqueeze_932 = None
    mul_1089: "f32[64]" = torch.ops.aten.mul.Tensor(sum_113, 3.0517578125e-05)
    mul_1090: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_1091: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1089, mul_1090);  mul_1089 = mul_1090 = None
    unsqueeze_934: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1091, 0);  mul_1091 = None
    unsqueeze_935: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_934, 2);  unsqueeze_934 = None
    unsqueeze_936: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_935, 3);  unsqueeze_935 = None
    mul_1092: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_23);  primals_23 = None
    unsqueeze_937: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1092, 0);  mul_1092 = None
    unsqueeze_938: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_937, 2);  unsqueeze_937 = None
    unsqueeze_939: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_938, 3);  unsqueeze_938 = None
    sub_288: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_930);  convolution_11 = unsqueeze_930 = None
    mul_1093: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_288, unsqueeze_936);  sub_288 = unsqueeze_936 = None
    sub_289: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(where_122, mul_1093);  where_122 = mul_1093 = None
    sub_290: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_289, unsqueeze_933);  sub_289 = unsqueeze_933 = None
    mul_1094: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_290, unsqueeze_939);  sub_290 = unsqueeze_939 = None
    mul_1095: "f32[64]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_34);  sum_113 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_1094, add_56, primals_146, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1094 = add_56 = primals_146 = None
    getitem_329: "f32[8, 64, 64, 64]" = convolution_backward_55[0]
    getitem_330: "f32[64, 64, 1, 1]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_378: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(getitem_323, getitem_329);  getitem_323 = getitem_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_236: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(where_10);  where_10 = None
    alias_237: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(alias_236);  alias_236 = None
    gt_123: "b8[8, 64, 64, 64]" = torch.ops.aten.gt.Scalar(alias_237, 0);  alias_237 = None
    mul_1096: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_378, 0.01)
    where_123: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(gt_123, add_378, mul_1096);  gt_123 = mul_1096 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_940: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_941: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_940, 2);  unsqueeze_940 = None
    unsqueeze_942: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_941, 3);  unsqueeze_941 = None
    sum_114: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_123, [0, 2, 3])
    sub_291: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_942)
    mul_1097: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(where_123, sub_291);  sub_291 = None
    sum_115: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1097, [0, 2, 3]);  mul_1097 = None
    mul_1098: "f32[64]" = torch.ops.aten.mul.Tensor(sum_114, 3.0517578125e-05)
    unsqueeze_943: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1098, 0);  mul_1098 = None
    unsqueeze_944: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_943, 2);  unsqueeze_943 = None
    unsqueeze_945: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_944, 3);  unsqueeze_944 = None
    mul_1099: "f32[64]" = torch.ops.aten.mul.Tensor(sum_115, 3.0517578125e-05)
    mul_1100: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_1101: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1099, mul_1100);  mul_1099 = mul_1100 = None
    unsqueeze_946: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1101, 0);  mul_1101 = None
    unsqueeze_947: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_946, 2);  unsqueeze_946 = None
    unsqueeze_948: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_947, 3);  unsqueeze_947 = None
    mul_1102: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_21);  primals_21 = None
    unsqueeze_949: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1102, 0);  mul_1102 = None
    unsqueeze_950: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_949, 2);  unsqueeze_949 = None
    unsqueeze_951: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_950, 3);  unsqueeze_950 = None
    sub_292: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_942);  convolution_10 = unsqueeze_942 = None
    mul_1103: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_292, unsqueeze_948);  sub_292 = unsqueeze_948 = None
    sub_293: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(where_123, mul_1103);  where_123 = mul_1103 = None
    sub_294: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_293, unsqueeze_945);  sub_293 = unsqueeze_945 = None
    mul_1104: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_294, unsqueeze_951);  sub_294 = unsqueeze_951 = None
    mul_1105: "f32[64]" = torch.ops.aten.mul.Tensor(sum_115, squeeze_31);  sum_115 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_1104, where_9, primals_145, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1104 = primals_145 = None
    getitem_332: "f32[8, 64, 64, 64]" = convolution_backward_56[0]
    getitem_333: "f32[64, 64, 3, 3]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_239: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(where_9);  where_9 = None
    alias_240: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(alias_239);  alias_239 = None
    gt_124: "b8[8, 64, 64, 64]" = torch.ops.aten.gt.Scalar(alias_240, 0);  alias_240 = None
    mul_1106: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_332, 0.01)
    where_124: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(gt_124, getitem_332, mul_1106);  gt_124 = getitem_332 = mul_1106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_952: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_953: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_952, 2);  unsqueeze_952 = None
    unsqueeze_954: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_953, 3);  unsqueeze_953 = None
    sum_116: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_124, [0, 2, 3])
    sub_295: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_954)
    mul_1107: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(where_124, sub_295);  sub_295 = None
    sum_117: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1107, [0, 2, 3]);  mul_1107 = None
    mul_1108: "f32[64]" = torch.ops.aten.mul.Tensor(sum_116, 3.0517578125e-05)
    unsqueeze_955: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1108, 0);  mul_1108 = None
    unsqueeze_956: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_955, 2);  unsqueeze_955 = None
    unsqueeze_957: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_956, 3);  unsqueeze_956 = None
    mul_1109: "f32[64]" = torch.ops.aten.mul.Tensor(sum_117, 3.0517578125e-05)
    mul_1110: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_1111: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1109, mul_1110);  mul_1109 = mul_1110 = None
    unsqueeze_958: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1111, 0);  mul_1111 = None
    unsqueeze_959: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_958, 2);  unsqueeze_958 = None
    unsqueeze_960: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_959, 3);  unsqueeze_959 = None
    mul_1112: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_19);  primals_19 = None
    unsqueeze_961: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1112, 0);  mul_1112 = None
    unsqueeze_962: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_961, 2);  unsqueeze_961 = None
    unsqueeze_963: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_962, 3);  unsqueeze_962 = None
    sub_296: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_954);  convolution_9 = unsqueeze_954 = None
    mul_1113: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_296, unsqueeze_960);  sub_296 = unsqueeze_960 = None
    sub_297: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(where_124, mul_1113);  where_124 = mul_1113 = None
    sub_298: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_297, unsqueeze_957);  sub_297 = unsqueeze_957 = None
    mul_1114: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_298, unsqueeze_963);  sub_298 = unsqueeze_963 = None
    mul_1115: "f32[64]" = torch.ops.aten.mul.Tensor(sum_117, squeeze_28);  sum_117 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_1114, getitem_27, primals_144, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1114 = getitem_27 = primals_144 = None
    getitem_335: "f32[8, 64, 64, 64]" = convolution_backward_57[0]
    getitem_336: "f32[64, 64, 1, 1]" = convolution_backward_57[1];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_379: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(add_378, getitem_335);  add_378 = getitem_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:336, code: xs, xb = x.split(self.expand_chs // 2, dim=1)
    cat_8: "f32[8, 128, 64, 64]" = torch.ops.aten.cat.default([slice_7, add_379], 1);  slice_7 = add_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_242: "f32[8, 128, 64, 64]" = torch.ops.aten.alias.default(where_8);  where_8 = None
    alias_243: "f32[8, 128, 64, 64]" = torch.ops.aten.alias.default(alias_242);  alias_242 = None
    gt_125: "b8[8, 128, 64, 64]" = torch.ops.aten.gt.Scalar(alias_243, 0);  alias_243 = None
    mul_1116: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(cat_8, 0.01)
    where_125: "f32[8, 128, 64, 64]" = torch.ops.aten.where.self(gt_125, cat_8, mul_1116);  gt_125 = cat_8 = mul_1116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_964: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_965: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_964, 2);  unsqueeze_964 = None
    unsqueeze_966: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_965, 3);  unsqueeze_965 = None
    sum_118: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_125, [0, 2, 3])
    sub_299: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_966)
    mul_1117: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(where_125, sub_299);  sub_299 = None
    sum_119: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1117, [0, 2, 3]);  mul_1117 = None
    mul_1118: "f32[128]" = torch.ops.aten.mul.Tensor(sum_118, 3.0517578125e-05)
    unsqueeze_967: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1118, 0);  mul_1118 = None
    unsqueeze_968: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_967, 2);  unsqueeze_967 = None
    unsqueeze_969: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_968, 3);  unsqueeze_968 = None
    mul_1119: "f32[128]" = torch.ops.aten.mul.Tensor(sum_119, 3.0517578125e-05)
    mul_1120: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_1121: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1119, mul_1120);  mul_1119 = mul_1120 = None
    unsqueeze_970: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1121, 0);  mul_1121 = None
    unsqueeze_971: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_970, 2);  unsqueeze_970 = None
    unsqueeze_972: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_971, 3);  unsqueeze_971 = None
    mul_1122: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_17);  primals_17 = None
    unsqueeze_973: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1122, 0);  mul_1122 = None
    unsqueeze_974: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_973, 2);  unsqueeze_973 = None
    unsqueeze_975: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_974, 3);  unsqueeze_974 = None
    sub_300: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_966);  convolution_8 = unsqueeze_966 = None
    mul_1123: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_300, unsqueeze_972);  sub_300 = unsqueeze_972 = None
    sub_301: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(where_125, mul_1123);  where_125 = mul_1123 = None
    sub_302: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(sub_301, unsqueeze_969);  sub_301 = unsqueeze_969 = None
    mul_1124: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_302, unsqueeze_975);  sub_302 = unsqueeze_975 = None
    mul_1125: "f32[128]" = torch.ops.aten.mul.Tensor(sum_119, squeeze_25);  sum_119 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_1124, where_7, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1124 = primals_143 = None
    getitem_338: "f32[8, 128, 64, 64]" = convolution_backward_58[0]
    getitem_339: "f32[128, 128, 1, 1]" = convolution_backward_58[1];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_245: "f32[8, 128, 64, 64]" = torch.ops.aten.alias.default(where_7);  where_7 = None
    alias_246: "f32[8, 128, 64, 64]" = torch.ops.aten.alias.default(alias_245);  alias_245 = None
    gt_126: "b8[8, 128, 64, 64]" = torch.ops.aten.gt.Scalar(alias_246, 0);  alias_246 = None
    mul_1126: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_338, 0.01)
    where_126: "f32[8, 128, 64, 64]" = torch.ops.aten.where.self(gt_126, getitem_338, mul_1126);  gt_126 = getitem_338 = mul_1126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_976: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_977: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_976, 2);  unsqueeze_976 = None
    unsqueeze_978: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_977, 3);  unsqueeze_977 = None
    sum_120: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_126, [0, 2, 3])
    sub_303: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_978)
    mul_1127: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(where_126, sub_303);  sub_303 = None
    sum_121: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1127, [0, 2, 3]);  mul_1127 = None
    mul_1128: "f32[128]" = torch.ops.aten.mul.Tensor(sum_120, 3.0517578125e-05)
    unsqueeze_979: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1128, 0);  mul_1128 = None
    unsqueeze_980: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_979, 2);  unsqueeze_979 = None
    unsqueeze_981: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_980, 3);  unsqueeze_980 = None
    mul_1129: "f32[128]" = torch.ops.aten.mul.Tensor(sum_121, 3.0517578125e-05)
    mul_1130: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_1131: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1129, mul_1130);  mul_1129 = mul_1130 = None
    unsqueeze_982: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1131, 0);  mul_1131 = None
    unsqueeze_983: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_982, 2);  unsqueeze_982 = None
    unsqueeze_984: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_983, 3);  unsqueeze_983 = None
    mul_1132: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_15);  primals_15 = None
    unsqueeze_985: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1132, 0);  mul_1132 = None
    unsqueeze_986: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_985, 2);  unsqueeze_985 = None
    unsqueeze_987: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_986, 3);  unsqueeze_986 = None
    sub_304: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_978);  convolution_7 = unsqueeze_978 = None
    mul_1133: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_304, unsqueeze_984);  sub_304 = unsqueeze_984 = None
    sub_305: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(where_126, mul_1133);  where_126 = mul_1133 = None
    sub_306: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(sub_305, unsqueeze_981);  sub_305 = unsqueeze_981 = None
    mul_1134: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_306, unsqueeze_987);  sub_306 = unsqueeze_987 = None
    mul_1135: "f32[128]" = torch.ops.aten.mul.Tensor(sum_121, squeeze_22);  sum_121 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:126, code: x = self.conv(x)
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(mul_1134, where_6, primals_142, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1134 = primals_142 = None
    getitem_341: "f32[8, 64, 128, 128]" = convolution_backward_59[0]
    getitem_342: "f32[128, 64, 3, 3]" = convolution_backward_59[1];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_248: "f32[8, 64, 128, 128]" = torch.ops.aten.alias.default(where_6);  where_6 = None
    alias_249: "f32[8, 64, 128, 128]" = torch.ops.aten.alias.default(alias_248);  alias_248 = None
    gt_127: "b8[8, 64, 128, 128]" = torch.ops.aten.gt.Scalar(alias_249, 0);  alias_249 = None
    mul_1136: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(getitem_341, 0.01)
    where_127: "f32[8, 64, 128, 128]" = torch.ops.aten.where.self(gt_127, getitem_341, mul_1136);  gt_127 = getitem_341 = mul_1136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_988: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_989: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_988, 2);  unsqueeze_988 = None
    unsqueeze_990: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_989, 3);  unsqueeze_989 = None
    sum_122: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_127, [0, 2, 3])
    sub_307: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_990)
    mul_1137: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(where_127, sub_307);  sub_307 = None
    sum_123: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1137, [0, 2, 3]);  mul_1137 = None
    mul_1138: "f32[64]" = torch.ops.aten.mul.Tensor(sum_122, 7.62939453125e-06)
    unsqueeze_991: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1138, 0);  mul_1138 = None
    unsqueeze_992: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_991, 2);  unsqueeze_991 = None
    unsqueeze_993: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_992, 3);  unsqueeze_992 = None
    mul_1139: "f32[64]" = torch.ops.aten.mul.Tensor(sum_123, 7.62939453125e-06)
    mul_1140: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_1141: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1139, mul_1140);  mul_1139 = mul_1140 = None
    unsqueeze_994: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1141, 0);  mul_1141 = None
    unsqueeze_995: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_994, 2);  unsqueeze_994 = None
    unsqueeze_996: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_995, 3);  unsqueeze_995 = None
    mul_1142: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_13);  primals_13 = None
    unsqueeze_997: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1142, 0);  mul_1142 = None
    unsqueeze_998: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_997, 2);  unsqueeze_997 = None
    unsqueeze_999: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_998, 3);  unsqueeze_998 = None
    sub_308: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_990);  convolution_6 = unsqueeze_990 = None
    mul_1143: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_308, unsqueeze_996);  sub_308 = unsqueeze_996 = None
    sub_309: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(where_127, mul_1143);  where_127 = mul_1143 = None
    sub_310: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(sub_309, unsqueeze_993);  sub_309 = unsqueeze_993 = None
    mul_1144: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_310, unsqueeze_999);  sub_310 = unsqueeze_999 = None
    mul_1145: "f32[64]" = torch.ops.aten.mul.Tensor(sum_123, squeeze_19);  sum_123 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_1144, cat, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1144 = cat = primals_141 = None
    getitem_344: "f32[8, 128, 128, 128]" = convolution_backward_60[0]
    getitem_345: "f32[64, 128, 1, 1]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:339, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
    slice_9: "f32[8, 64, 128, 128]" = torch.ops.aten.slice.Tensor(getitem_344, 1, 0, 64)
    slice_10: "f32[8, 64, 128, 128]" = torch.ops.aten.slice.Tensor(getitem_344, 1, 64, 128);  getitem_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_251: "f32[8, 64, 128, 128]" = torch.ops.aten.alias.default(where_5);  where_5 = None
    alias_252: "f32[8, 64, 128, 128]" = torch.ops.aten.alias.default(alias_251);  alias_251 = None
    gt_128: "b8[8, 64, 128, 128]" = torch.ops.aten.gt.Scalar(alias_252, 0);  alias_252 = None
    mul_1146: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(slice_10, 0.01)
    where_128: "f32[8, 64, 128, 128]" = torch.ops.aten.where.self(gt_128, slice_10, mul_1146);  gt_128 = slice_10 = mul_1146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1000: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_1001: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1000, 2);  unsqueeze_1000 = None
    unsqueeze_1002: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1001, 3);  unsqueeze_1001 = None
    sum_124: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_128, [0, 2, 3])
    sub_311: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_1002)
    mul_1147: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(where_128, sub_311);  sub_311 = None
    sum_125: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1147, [0, 2, 3]);  mul_1147 = None
    mul_1148: "f32[64]" = torch.ops.aten.mul.Tensor(sum_124, 7.62939453125e-06)
    unsqueeze_1003: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1148, 0);  mul_1148 = None
    unsqueeze_1004: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1003, 2);  unsqueeze_1003 = None
    unsqueeze_1005: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1004, 3);  unsqueeze_1004 = None
    mul_1149: "f32[64]" = torch.ops.aten.mul.Tensor(sum_125, 7.62939453125e-06)
    mul_1150: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_1151: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1149, mul_1150);  mul_1149 = mul_1150 = None
    unsqueeze_1006: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1151, 0);  mul_1151 = None
    unsqueeze_1007: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1006, 2);  unsqueeze_1006 = None
    unsqueeze_1008: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1007, 3);  unsqueeze_1007 = None
    mul_1152: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_11);  primals_11 = None
    unsqueeze_1009: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1152, 0);  mul_1152 = None
    unsqueeze_1010: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1009, 2);  unsqueeze_1009 = None
    unsqueeze_1011: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1010, 3);  unsqueeze_1010 = None
    sub_312: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_1002);  convolution_5 = unsqueeze_1002 = None
    mul_1153: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_312, unsqueeze_1008);  sub_312 = unsqueeze_1008 = None
    sub_313: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(where_128, mul_1153);  where_128 = mul_1153 = None
    sub_314: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(sub_313, unsqueeze_1005);  sub_313 = unsqueeze_1005 = None
    mul_1154: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_314, unsqueeze_1011);  sub_314 = unsqueeze_1011 = None
    mul_1155: "f32[64]" = torch.ops.aten.mul.Tensor(sum_125, squeeze_16);  sum_125 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_1154, add_25, primals_140, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1154 = add_25 = primals_140 = None
    getitem_347: "f32[8, 64, 128, 128]" = convolution_backward_61[0]
    getitem_348: "f32[64, 64, 1, 1]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_254: "f32[8, 64, 128, 128]" = torch.ops.aten.alias.default(where_4);  where_4 = None
    alias_255: "f32[8, 64, 128, 128]" = torch.ops.aten.alias.default(alias_254);  alias_254 = None
    gt_129: "b8[8, 64, 128, 128]" = torch.ops.aten.gt.Scalar(alias_255, 0);  alias_255 = None
    mul_1156: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(getitem_347, 0.01)
    where_129: "f32[8, 64, 128, 128]" = torch.ops.aten.where.self(gt_129, getitem_347, mul_1156);  gt_129 = mul_1156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1012: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_1013: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1012, 2);  unsqueeze_1012 = None
    unsqueeze_1014: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1013, 3);  unsqueeze_1013 = None
    sum_126: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_129, [0, 2, 3])
    sub_315: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_1014)
    mul_1157: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(where_129, sub_315);  sub_315 = None
    sum_127: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1157, [0, 2, 3]);  mul_1157 = None
    mul_1158: "f32[64]" = torch.ops.aten.mul.Tensor(sum_126, 7.62939453125e-06)
    unsqueeze_1015: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1158, 0);  mul_1158 = None
    unsqueeze_1016: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1015, 2);  unsqueeze_1015 = None
    unsqueeze_1017: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1016, 3);  unsqueeze_1016 = None
    mul_1159: "f32[64]" = torch.ops.aten.mul.Tensor(sum_127, 7.62939453125e-06)
    mul_1160: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_1161: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1159, mul_1160);  mul_1159 = mul_1160 = None
    unsqueeze_1018: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1161, 0);  mul_1161 = None
    unsqueeze_1019: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1018, 2);  unsqueeze_1018 = None
    unsqueeze_1020: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1019, 3);  unsqueeze_1019 = None
    mul_1162: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_9);  primals_9 = None
    unsqueeze_1021: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1162, 0);  mul_1162 = None
    unsqueeze_1022: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1021, 2);  unsqueeze_1021 = None
    unsqueeze_1023: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1022, 3);  unsqueeze_1022 = None
    sub_316: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_1014);  convolution_4 = unsqueeze_1014 = None
    mul_1163: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_316, unsqueeze_1020);  sub_316 = unsqueeze_1020 = None
    sub_317: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(where_129, mul_1163);  where_129 = mul_1163 = None
    sub_318: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(sub_317, unsqueeze_1017);  sub_317 = unsqueeze_1017 = None
    mul_1164: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_318, unsqueeze_1023);  sub_318 = unsqueeze_1023 = None
    mul_1165: "f32[64]" = torch.ops.aten.mul.Tensor(sum_127, squeeze_13);  sum_127 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_1164, where_3, primals_139, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1164 = primals_139 = None
    getitem_350: "f32[8, 32, 128, 128]" = convolution_backward_62[0]
    getitem_351: "f32[64, 32, 3, 3]" = convolution_backward_62[1];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_257: "f32[8, 32, 128, 128]" = torch.ops.aten.alias.default(where_3);  where_3 = None
    alias_258: "f32[8, 32, 128, 128]" = torch.ops.aten.alias.default(alias_257);  alias_257 = None
    gt_130: "b8[8, 32, 128, 128]" = torch.ops.aten.gt.Scalar(alias_258, 0);  alias_258 = None
    mul_1166: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(getitem_350, 0.01)
    where_130: "f32[8, 32, 128, 128]" = torch.ops.aten.where.self(gt_130, getitem_350, mul_1166);  gt_130 = getitem_350 = mul_1166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1024: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_1025: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1024, 2);  unsqueeze_1024 = None
    unsqueeze_1026: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1025, 3);  unsqueeze_1025 = None
    sum_128: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_130, [0, 2, 3])
    sub_319: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_1026)
    mul_1167: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(where_130, sub_319);  sub_319 = None
    sum_129: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1167, [0, 2, 3]);  mul_1167 = None
    mul_1168: "f32[32]" = torch.ops.aten.mul.Tensor(sum_128, 7.62939453125e-06)
    unsqueeze_1027: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1168, 0);  mul_1168 = None
    unsqueeze_1028: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1027, 2);  unsqueeze_1027 = None
    unsqueeze_1029: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1028, 3);  unsqueeze_1028 = None
    mul_1169: "f32[32]" = torch.ops.aten.mul.Tensor(sum_129, 7.62939453125e-06)
    mul_1170: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_1171: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1169, mul_1170);  mul_1169 = mul_1170 = None
    unsqueeze_1030: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1171, 0);  mul_1171 = None
    unsqueeze_1031: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1030, 2);  unsqueeze_1030 = None
    unsqueeze_1032: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1031, 3);  unsqueeze_1031 = None
    mul_1172: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_7);  primals_7 = None
    unsqueeze_1033: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1172, 0);  mul_1172 = None
    unsqueeze_1034: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1033, 2);  unsqueeze_1033 = None
    unsqueeze_1035: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1034, 3);  unsqueeze_1034 = None
    sub_320: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_1026);  convolution_3 = unsqueeze_1026 = None
    mul_1173: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_320, unsqueeze_1032);  sub_320 = unsqueeze_1032 = None
    sub_321: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(where_130, mul_1173);  where_130 = mul_1173 = None
    sub_322: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(sub_321, unsqueeze_1029);  sub_321 = unsqueeze_1029 = None
    mul_1174: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_322, unsqueeze_1035);  sub_322 = unsqueeze_1035 = None
    mul_1175: "f32[32]" = torch.ops.aten.mul.Tensor(sum_129, squeeze_10);  sum_129 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_1174, getitem_9, primals_138, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1174 = getitem_9 = primals_138 = None
    getitem_353: "f32[8, 64, 128, 128]" = convolution_backward_63[0]
    getitem_354: "f32[32, 64, 1, 1]" = convolution_backward_63[1];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_380: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(getitem_347, getitem_353);  getitem_347 = getitem_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:336, code: xs, xb = x.split(self.expand_chs // 2, dim=1)
    cat_9: "f32[8, 128, 128, 128]" = torch.ops.aten.cat.default([slice_9, add_380], 1);  slice_9 = add_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_260: "f32[8, 128, 128, 128]" = torch.ops.aten.alias.default(where_2);  where_2 = None
    alias_261: "f32[8, 128, 128, 128]" = torch.ops.aten.alias.default(alias_260);  alias_260 = None
    gt_131: "b8[8, 128, 128, 128]" = torch.ops.aten.gt.Scalar(alias_261, 0);  alias_261 = None
    mul_1176: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(cat_9, 0.01)
    where_131: "f32[8, 128, 128, 128]" = torch.ops.aten.where.self(gt_131, cat_9, mul_1176);  gt_131 = cat_9 = mul_1176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1036: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_1037: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1036, 2);  unsqueeze_1036 = None
    unsqueeze_1038: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1037, 3);  unsqueeze_1037 = None
    sum_130: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_131, [0, 2, 3])
    sub_323: "f32[8, 128, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_1038)
    mul_1177: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(where_131, sub_323);  sub_323 = None
    sum_131: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1177, [0, 2, 3]);  mul_1177 = None
    mul_1178: "f32[128]" = torch.ops.aten.mul.Tensor(sum_130, 7.62939453125e-06)
    unsqueeze_1039: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1178, 0);  mul_1178 = None
    unsqueeze_1040: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1039, 2);  unsqueeze_1039 = None
    unsqueeze_1041: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1040, 3);  unsqueeze_1040 = None
    mul_1179: "f32[128]" = torch.ops.aten.mul.Tensor(sum_131, 7.62939453125e-06)
    mul_1180: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_1181: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1179, mul_1180);  mul_1179 = mul_1180 = None
    unsqueeze_1042: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1181, 0);  mul_1181 = None
    unsqueeze_1043: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1042, 2);  unsqueeze_1042 = None
    unsqueeze_1044: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1043, 3);  unsqueeze_1043 = None
    mul_1182: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_5);  primals_5 = None
    unsqueeze_1045: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1182, 0);  mul_1182 = None
    unsqueeze_1046: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1045, 2);  unsqueeze_1045 = None
    unsqueeze_1047: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1046, 3);  unsqueeze_1046 = None
    sub_324: "f32[8, 128, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_1038);  convolution_2 = unsqueeze_1038 = None
    mul_1183: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(sub_324, unsqueeze_1044);  sub_324 = unsqueeze_1044 = None
    sub_325: "f32[8, 128, 128, 128]" = torch.ops.aten.sub.Tensor(where_131, mul_1183);  where_131 = mul_1183 = None
    sub_326: "f32[8, 128, 128, 128]" = torch.ops.aten.sub.Tensor(sub_325, unsqueeze_1041);  sub_325 = unsqueeze_1041 = None
    mul_1184: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(sub_326, unsqueeze_1047);  sub_326 = unsqueeze_1047 = None
    mul_1185: "f32[128]" = torch.ops.aten.mul.Tensor(sum_131, squeeze_7);  sum_131 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(mul_1184, where_1, primals_137, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1184 = primals_137 = None
    getitem_356: "f32[8, 64, 128, 128]" = convolution_backward_64[0]
    getitem_357: "f32[128, 64, 1, 1]" = convolution_backward_64[1];  convolution_backward_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_263: "f32[8, 64, 128, 128]" = torch.ops.aten.alias.default(where_1);  where_1 = None
    alias_264: "f32[8, 64, 128, 128]" = torch.ops.aten.alias.default(alias_263);  alias_263 = None
    gt_132: "b8[8, 64, 128, 128]" = torch.ops.aten.gt.Scalar(alias_264, 0);  alias_264 = None
    mul_1186: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(getitem_356, 0.01)
    where_132: "f32[8, 64, 128, 128]" = torch.ops.aten.where.self(gt_132, getitem_356, mul_1186);  gt_132 = getitem_356 = mul_1186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1048: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_1049: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1048, 2);  unsqueeze_1048 = None
    unsqueeze_1050: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1049, 3);  unsqueeze_1049 = None
    sum_132: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_132, [0, 2, 3])
    sub_327: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1050)
    mul_1187: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(where_132, sub_327);  sub_327 = None
    sum_133: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1187, [0, 2, 3]);  mul_1187 = None
    mul_1188: "f32[64]" = torch.ops.aten.mul.Tensor(sum_132, 7.62939453125e-06)
    unsqueeze_1051: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1188, 0);  mul_1188 = None
    unsqueeze_1052: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1051, 2);  unsqueeze_1051 = None
    unsqueeze_1053: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1052, 3);  unsqueeze_1052 = None
    mul_1189: "f32[64]" = torch.ops.aten.mul.Tensor(sum_133, 7.62939453125e-06)
    mul_1190: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_1191: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1189, mul_1190);  mul_1189 = mul_1190 = None
    unsqueeze_1054: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1191, 0);  mul_1191 = None
    unsqueeze_1055: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1054, 2);  unsqueeze_1054 = None
    unsqueeze_1056: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1055, 3);  unsqueeze_1055 = None
    mul_1192: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_3);  primals_3 = None
    unsqueeze_1057: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1192, 0);  mul_1192 = None
    unsqueeze_1058: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1057, 2);  unsqueeze_1057 = None
    unsqueeze_1059: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1058, 3);  unsqueeze_1058 = None
    sub_328: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1050);  convolution_1 = unsqueeze_1050 = None
    mul_1193: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_328, unsqueeze_1056);  sub_328 = unsqueeze_1056 = None
    sub_329: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(where_132, mul_1193);  where_132 = mul_1193 = None
    sub_330: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(sub_329, unsqueeze_1053);  sub_329 = unsqueeze_1053 = None
    mul_1194: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_330, unsqueeze_1059);  sub_330 = unsqueeze_1059 = None
    mul_1195: "f32[64]" = torch.ops.aten.mul.Tensor(sum_133, squeeze_4);  sum_133 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:126, code: x = self.conv(x)
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(mul_1194, where, primals_136, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1194 = primals_136 = None
    getitem_359: "f32[8, 32, 256, 256]" = convolution_backward_65[0]
    getitem_360: "f32[64, 32, 3, 3]" = convolution_backward_65[1];  convolution_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_266: "f32[8, 32, 256, 256]" = torch.ops.aten.alias.default(where);  where = None
    alias_267: "f32[8, 32, 256, 256]" = torch.ops.aten.alias.default(alias_266);  alias_266 = None
    gt_133: "b8[8, 32, 256, 256]" = torch.ops.aten.gt.Scalar(alias_267, 0);  alias_267 = None
    mul_1196: "f32[8, 32, 256, 256]" = torch.ops.aten.mul.Tensor(getitem_359, 0.01)
    where_133: "f32[8, 32, 256, 256]" = torch.ops.aten.where.self(gt_133, getitem_359, mul_1196);  gt_133 = getitem_359 = mul_1196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1060: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_1061: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1060, 2);  unsqueeze_1060 = None
    unsqueeze_1062: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1061, 3);  unsqueeze_1061 = None
    sum_134: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_133, [0, 2, 3])
    sub_331: "f32[8, 32, 256, 256]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1062)
    mul_1197: "f32[8, 32, 256, 256]" = torch.ops.aten.mul.Tensor(where_133, sub_331);  sub_331 = None
    sum_135: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1197, [0, 2, 3]);  mul_1197 = None
    mul_1198: "f32[32]" = torch.ops.aten.mul.Tensor(sum_134, 1.9073486328125e-06)
    unsqueeze_1063: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1198, 0);  mul_1198 = None
    unsqueeze_1064: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1063, 2);  unsqueeze_1063 = None
    unsqueeze_1065: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1064, 3);  unsqueeze_1064 = None
    mul_1199: "f32[32]" = torch.ops.aten.mul.Tensor(sum_135, 1.9073486328125e-06)
    mul_1200: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_1201: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1199, mul_1200);  mul_1199 = mul_1200 = None
    unsqueeze_1066: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1201, 0);  mul_1201 = None
    unsqueeze_1067: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1066, 2);  unsqueeze_1066 = None
    unsqueeze_1068: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1067, 3);  unsqueeze_1067 = None
    mul_1202: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_1);  primals_1 = None
    unsqueeze_1069: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1202, 0);  mul_1202 = None
    unsqueeze_1070: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1069, 2);  unsqueeze_1069 = None
    unsqueeze_1071: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1070, 3);  unsqueeze_1070 = None
    sub_332: "f32[8, 32, 256, 256]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1062);  convolution = unsqueeze_1062 = None
    mul_1203: "f32[8, 32, 256, 256]" = torch.ops.aten.mul.Tensor(sub_332, unsqueeze_1068);  sub_332 = unsqueeze_1068 = None
    sub_333: "f32[8, 32, 256, 256]" = torch.ops.aten.sub.Tensor(where_133, mul_1203);  where_133 = mul_1203 = None
    sub_334: "f32[8, 32, 256, 256]" = torch.ops.aten.sub.Tensor(sub_333, unsqueeze_1065);  sub_333 = unsqueeze_1065 = None
    mul_1204: "f32[8, 32, 256, 256]" = torch.ops.aten.mul.Tensor(sub_334, unsqueeze_1071);  sub_334 = unsqueeze_1071 = None
    mul_1205: "f32[32]" = torch.ops.aten.mul.Tensor(sum_135, squeeze_1);  sum_135 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_1204, primals_405, primals_135, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1204 = primals_405 = primals_135 = None
    getitem_363: "f32[32, 3, 3, 3]" = convolution_backward_66[1];  convolution_backward_66 = None
    
    # No stacktrace found for following nodes
    copy_: "i64[]" = torch.ops.aten.copy_.default(primals_204, add);  primals_204 = add = None
    copy__1: "f32[32]" = torch.ops.aten.copy_.default(primals_205, add_2);  primals_205 = add_2 = None
    copy__2: "f32[32]" = torch.ops.aten.copy_.default(primals_206, add_3);  primals_206 = add_3 = None
    copy__3: "i64[]" = torch.ops.aten.copy_.default(primals_207, add_5);  primals_207 = add_5 = None
    copy__4: "f32[64]" = torch.ops.aten.copy_.default(primals_208, add_7);  primals_208 = add_7 = None
    copy__5: "f32[64]" = torch.ops.aten.copy_.default(primals_209, add_8);  primals_209 = add_8 = None
    copy__6: "i64[]" = torch.ops.aten.copy_.default(primals_210, add_10);  primals_210 = add_10 = None
    copy__7: "f32[128]" = torch.ops.aten.copy_.default(primals_211, add_12);  primals_211 = add_12 = None
    copy__8: "f32[128]" = torch.ops.aten.copy_.default(primals_212, add_13);  primals_212 = add_13 = None
    copy__9: "i64[]" = torch.ops.aten.copy_.default(primals_213, add_15);  primals_213 = add_15 = None
    copy__10: "f32[32]" = torch.ops.aten.copy_.default(primals_214, add_17);  primals_214 = add_17 = None
    copy__11: "f32[32]" = torch.ops.aten.copy_.default(primals_215, add_18);  primals_215 = add_18 = None
    copy__12: "i64[]" = torch.ops.aten.copy_.default(primals_216, add_20);  primals_216 = add_20 = None
    copy__13: "f32[64]" = torch.ops.aten.copy_.default(primals_217, add_22);  primals_217 = add_22 = None
    copy__14: "f32[64]" = torch.ops.aten.copy_.default(primals_218, add_23);  primals_218 = add_23 = None
    copy__15: "i64[]" = torch.ops.aten.copy_.default(primals_219, add_26);  primals_219 = add_26 = None
    copy__16: "f32[64]" = torch.ops.aten.copy_.default(primals_220, add_28);  primals_220 = add_28 = None
    copy__17: "f32[64]" = torch.ops.aten.copy_.default(primals_221, add_29);  primals_221 = add_29 = None
    copy__18: "i64[]" = torch.ops.aten.copy_.default(primals_222, add_31);  primals_222 = add_31 = None
    copy__19: "f32[64]" = torch.ops.aten.copy_.default(primals_223, add_33);  primals_223 = add_33 = None
    copy__20: "f32[64]" = torch.ops.aten.copy_.default(primals_224, add_34);  primals_224 = add_34 = None
    copy__21: "i64[]" = torch.ops.aten.copy_.default(primals_225, add_36);  primals_225 = add_36 = None
    copy__22: "f32[128]" = torch.ops.aten.copy_.default(primals_226, add_38);  primals_226 = add_38 = None
    copy__23: "f32[128]" = torch.ops.aten.copy_.default(primals_227, add_39);  primals_227 = add_39 = None
    copy__24: "i64[]" = torch.ops.aten.copy_.default(primals_228, add_41);  primals_228 = add_41 = None
    copy__25: "f32[128]" = torch.ops.aten.copy_.default(primals_229, add_43);  primals_229 = add_43 = None
    copy__26: "f32[128]" = torch.ops.aten.copy_.default(primals_230, add_44);  primals_230 = add_44 = None
    copy__27: "i64[]" = torch.ops.aten.copy_.default(primals_231, add_46);  primals_231 = add_46 = None
    copy__28: "f32[64]" = torch.ops.aten.copy_.default(primals_232, add_48);  primals_232 = add_48 = None
    copy__29: "f32[64]" = torch.ops.aten.copy_.default(primals_233, add_49);  primals_233 = add_49 = None
    copy__30: "i64[]" = torch.ops.aten.copy_.default(primals_234, add_51);  primals_234 = add_51 = None
    copy__31: "f32[64]" = torch.ops.aten.copy_.default(primals_235, add_53);  primals_235 = add_53 = None
    copy__32: "f32[64]" = torch.ops.aten.copy_.default(primals_236, add_54);  primals_236 = add_54 = None
    copy__33: "i64[]" = torch.ops.aten.copy_.default(primals_237, add_57);  primals_237 = add_57 = None
    copy__34: "f32[64]" = torch.ops.aten.copy_.default(primals_238, add_59);  primals_238 = add_59 = None
    copy__35: "f32[64]" = torch.ops.aten.copy_.default(primals_239, add_60);  primals_239 = add_60 = None
    copy__36: "i64[]" = torch.ops.aten.copy_.default(primals_240, add_62);  primals_240 = add_62 = None
    copy__37: "f32[64]" = torch.ops.aten.copy_.default(primals_241, add_64);  primals_241 = add_64 = None
    copy__38: "f32[64]" = torch.ops.aten.copy_.default(primals_242, add_65);  primals_242 = add_65 = None
    copy__39: "i64[]" = torch.ops.aten.copy_.default(primals_243, add_68);  primals_243 = add_68 = None
    copy__40: "f32[64]" = torch.ops.aten.copy_.default(primals_244, add_70);  primals_244 = add_70 = None
    copy__41: "f32[64]" = torch.ops.aten.copy_.default(primals_245, add_71);  primals_245 = add_71 = None
    copy__42: "i64[]" = torch.ops.aten.copy_.default(primals_246, add_73);  primals_246 = add_73 = None
    copy__43: "f32[128]" = torch.ops.aten.copy_.default(primals_247, add_75);  primals_247 = add_75 = None
    copy__44: "f32[128]" = torch.ops.aten.copy_.default(primals_248, add_76);  primals_248 = add_76 = None
    copy__45: "i64[]" = torch.ops.aten.copy_.default(primals_249, add_78);  primals_249 = add_78 = None
    copy__46: "f32[256]" = torch.ops.aten.copy_.default(primals_250, add_80);  primals_250 = add_80 = None
    copy__47: "f32[256]" = torch.ops.aten.copy_.default(primals_251, add_81);  primals_251 = add_81 = None
    copy__48: "i64[]" = torch.ops.aten.copy_.default(primals_252, add_83);  primals_252 = add_83 = None
    copy__49: "f32[256]" = torch.ops.aten.copy_.default(primals_253, add_85);  primals_253 = add_85 = None
    copy__50: "f32[256]" = torch.ops.aten.copy_.default(primals_254, add_86);  primals_254 = add_86 = None
    copy__51: "i64[]" = torch.ops.aten.copy_.default(primals_255, add_88);  primals_255 = add_88 = None
    copy__52: "f32[128]" = torch.ops.aten.copy_.default(primals_256, add_90);  primals_256 = add_90 = None
    copy__53: "f32[128]" = torch.ops.aten.copy_.default(primals_257, add_91);  primals_257 = add_91 = None
    copy__54: "i64[]" = torch.ops.aten.copy_.default(primals_258, add_93);  primals_258 = add_93 = None
    copy__55: "f32[128]" = torch.ops.aten.copy_.default(primals_259, add_95);  primals_259 = add_95 = None
    copy__56: "f32[128]" = torch.ops.aten.copy_.default(primals_260, add_96);  primals_260 = add_96 = None
    copy__57: "i64[]" = torch.ops.aten.copy_.default(primals_261, add_99);  primals_261 = add_99 = None
    copy__58: "f32[128]" = torch.ops.aten.copy_.default(primals_262, add_101);  primals_262 = add_101 = None
    copy__59: "f32[128]" = torch.ops.aten.copy_.default(primals_263, add_102);  primals_263 = add_102 = None
    copy__60: "i64[]" = torch.ops.aten.copy_.default(primals_264, add_104);  primals_264 = add_104 = None
    copy__61: "f32[128]" = torch.ops.aten.copy_.default(primals_265, add_106);  primals_265 = add_106 = None
    copy__62: "f32[128]" = torch.ops.aten.copy_.default(primals_266, add_107);  primals_266 = add_107 = None
    copy__63: "i64[]" = torch.ops.aten.copy_.default(primals_267, add_110);  primals_267 = add_110 = None
    copy__64: "f32[128]" = torch.ops.aten.copy_.default(primals_268, add_112);  primals_268 = add_112 = None
    copy__65: "f32[128]" = torch.ops.aten.copy_.default(primals_269, add_113);  primals_269 = add_113 = None
    copy__66: "i64[]" = torch.ops.aten.copy_.default(primals_270, add_115);  primals_270 = add_115 = None
    copy__67: "f32[128]" = torch.ops.aten.copy_.default(primals_271, add_117);  primals_271 = add_117 = None
    copy__68: "f32[128]" = torch.ops.aten.copy_.default(primals_272, add_118);  primals_272 = add_118 = None
    copy__69: "i64[]" = torch.ops.aten.copy_.default(primals_273, add_121);  primals_273 = add_121 = None
    copy__70: "f32[128]" = torch.ops.aten.copy_.default(primals_274, add_123);  primals_274 = add_123 = None
    copy__71: "f32[128]" = torch.ops.aten.copy_.default(primals_275, add_124);  primals_275 = add_124 = None
    copy__72: "i64[]" = torch.ops.aten.copy_.default(primals_276, add_126);  primals_276 = add_126 = None
    copy__73: "f32[128]" = torch.ops.aten.copy_.default(primals_277, add_128);  primals_277 = add_128 = None
    copy__74: "f32[128]" = torch.ops.aten.copy_.default(primals_278, add_129);  primals_278 = add_129 = None
    copy__75: "i64[]" = torch.ops.aten.copy_.default(primals_279, add_132);  primals_279 = add_132 = None
    copy__76: "f32[128]" = torch.ops.aten.copy_.default(primals_280, add_134);  primals_280 = add_134 = None
    copy__77: "f32[128]" = torch.ops.aten.copy_.default(primals_281, add_135);  primals_281 = add_135 = None
    copy__78: "i64[]" = torch.ops.aten.copy_.default(primals_282, add_137);  primals_282 = add_137 = None
    copy__79: "f32[128]" = torch.ops.aten.copy_.default(primals_283, add_139);  primals_283 = add_139 = None
    copy__80: "f32[128]" = torch.ops.aten.copy_.default(primals_284, add_140);  primals_284 = add_140 = None
    copy__81: "i64[]" = torch.ops.aten.copy_.default(primals_285, add_143);  primals_285 = add_143 = None
    copy__82: "f32[128]" = torch.ops.aten.copy_.default(primals_286, add_145);  primals_286 = add_145 = None
    copy__83: "f32[128]" = torch.ops.aten.copy_.default(primals_287, add_146);  primals_287 = add_146 = None
    copy__84: "i64[]" = torch.ops.aten.copy_.default(primals_288, add_148);  primals_288 = add_148 = None
    copy__85: "f32[128]" = torch.ops.aten.copy_.default(primals_289, add_150);  primals_289 = add_150 = None
    copy__86: "f32[128]" = torch.ops.aten.copy_.default(primals_290, add_151);  primals_290 = add_151 = None
    copy__87: "i64[]" = torch.ops.aten.copy_.default(primals_291, add_154);  primals_291 = add_154 = None
    copy__88: "f32[128]" = torch.ops.aten.copy_.default(primals_292, add_156);  primals_292 = add_156 = None
    copy__89: "f32[128]" = torch.ops.aten.copy_.default(primals_293, add_157);  primals_293 = add_157 = None
    copy__90: "i64[]" = torch.ops.aten.copy_.default(primals_294, add_159);  primals_294 = add_159 = None
    copy__91: "f32[128]" = torch.ops.aten.copy_.default(primals_295, add_161);  primals_295 = add_161 = None
    copy__92: "f32[128]" = torch.ops.aten.copy_.default(primals_296, add_162);  primals_296 = add_162 = None
    copy__93: "i64[]" = torch.ops.aten.copy_.default(primals_297, add_165);  primals_297 = add_165 = None
    copy__94: "f32[128]" = torch.ops.aten.copy_.default(primals_298, add_167);  primals_298 = add_167 = None
    copy__95: "f32[128]" = torch.ops.aten.copy_.default(primals_299, add_168);  primals_299 = add_168 = None
    copy__96: "i64[]" = torch.ops.aten.copy_.default(primals_300, add_170);  primals_300 = add_170 = None
    copy__97: "f32[128]" = torch.ops.aten.copy_.default(primals_301, add_172);  primals_301 = add_172 = None
    copy__98: "f32[128]" = torch.ops.aten.copy_.default(primals_302, add_173);  primals_302 = add_173 = None
    copy__99: "i64[]" = torch.ops.aten.copy_.default(primals_303, add_176);  primals_303 = add_176 = None
    copy__100: "f32[128]" = torch.ops.aten.copy_.default(primals_304, add_178);  primals_304 = add_178 = None
    copy__101: "f32[128]" = torch.ops.aten.copy_.default(primals_305, add_179);  primals_305 = add_179 = None
    copy__102: "i64[]" = torch.ops.aten.copy_.default(primals_306, add_181);  primals_306 = add_181 = None
    copy__103: "f32[256]" = torch.ops.aten.copy_.default(primals_307, add_183);  primals_307 = add_183 = None
    copy__104: "f32[256]" = torch.ops.aten.copy_.default(primals_308, add_184);  primals_308 = add_184 = None
    copy__105: "i64[]" = torch.ops.aten.copy_.default(primals_309, add_186);  primals_309 = add_186 = None
    copy__106: "f32[512]" = torch.ops.aten.copy_.default(primals_310, add_188);  primals_310 = add_188 = None
    copy__107: "f32[512]" = torch.ops.aten.copy_.default(primals_311, add_189);  primals_311 = add_189 = None
    copy__108: "i64[]" = torch.ops.aten.copy_.default(primals_312, add_191);  primals_312 = add_191 = None
    copy__109: "f32[512]" = torch.ops.aten.copy_.default(primals_313, add_193);  primals_313 = add_193 = None
    copy__110: "f32[512]" = torch.ops.aten.copy_.default(primals_314, add_194);  primals_314 = add_194 = None
    copy__111: "i64[]" = torch.ops.aten.copy_.default(primals_315, add_196);  primals_315 = add_196 = None
    copy__112: "f32[256]" = torch.ops.aten.copy_.default(primals_316, add_198);  primals_316 = add_198 = None
    copy__113: "f32[256]" = torch.ops.aten.copy_.default(primals_317, add_199);  primals_317 = add_199 = None
    copy__114: "i64[]" = torch.ops.aten.copy_.default(primals_318, add_201);  primals_318 = add_201 = None
    copy__115: "f32[256]" = torch.ops.aten.copy_.default(primals_319, add_203);  primals_319 = add_203 = None
    copy__116: "f32[256]" = torch.ops.aten.copy_.default(primals_320, add_204);  primals_320 = add_204 = None
    copy__117: "i64[]" = torch.ops.aten.copy_.default(primals_321, add_207);  primals_321 = add_207 = None
    copy__118: "f32[256]" = torch.ops.aten.copy_.default(primals_322, add_209);  primals_322 = add_209 = None
    copy__119: "f32[256]" = torch.ops.aten.copy_.default(primals_323, add_210);  primals_323 = add_210 = None
    copy__120: "i64[]" = torch.ops.aten.copy_.default(primals_324, add_212);  primals_324 = add_212 = None
    copy__121: "f32[256]" = torch.ops.aten.copy_.default(primals_325, add_214);  primals_325 = add_214 = None
    copy__122: "f32[256]" = torch.ops.aten.copy_.default(primals_326, add_215);  primals_326 = add_215 = None
    copy__123: "i64[]" = torch.ops.aten.copy_.default(primals_327, add_218);  primals_327 = add_218 = None
    copy__124: "f32[256]" = torch.ops.aten.copy_.default(primals_328, add_220);  primals_328 = add_220 = None
    copy__125: "f32[256]" = torch.ops.aten.copy_.default(primals_329, add_221);  primals_329 = add_221 = None
    copy__126: "i64[]" = torch.ops.aten.copy_.default(primals_330, add_223);  primals_330 = add_223 = None
    copy__127: "f32[256]" = torch.ops.aten.copy_.default(primals_331, add_225);  primals_331 = add_225 = None
    copy__128: "f32[256]" = torch.ops.aten.copy_.default(primals_332, add_226);  primals_332 = add_226 = None
    copy__129: "i64[]" = torch.ops.aten.copy_.default(primals_333, add_229);  primals_333 = add_229 = None
    copy__130: "f32[256]" = torch.ops.aten.copy_.default(primals_334, add_231);  primals_334 = add_231 = None
    copy__131: "f32[256]" = torch.ops.aten.copy_.default(primals_335, add_232);  primals_335 = add_232 = None
    copy__132: "i64[]" = torch.ops.aten.copy_.default(primals_336, add_234);  primals_336 = add_234 = None
    copy__133: "f32[256]" = torch.ops.aten.copy_.default(primals_337, add_236);  primals_337 = add_236 = None
    copy__134: "f32[256]" = torch.ops.aten.copy_.default(primals_338, add_237);  primals_338 = add_237 = None
    copy__135: "i64[]" = torch.ops.aten.copy_.default(primals_339, add_240);  primals_339 = add_240 = None
    copy__136: "f32[256]" = torch.ops.aten.copy_.default(primals_340, add_242);  primals_340 = add_242 = None
    copy__137: "f32[256]" = torch.ops.aten.copy_.default(primals_341, add_243);  primals_341 = add_243 = None
    copy__138: "i64[]" = torch.ops.aten.copy_.default(primals_342, add_245);  primals_342 = add_245 = None
    copy__139: "f32[256]" = torch.ops.aten.copy_.default(primals_343, add_247);  primals_343 = add_247 = None
    copy__140: "f32[256]" = torch.ops.aten.copy_.default(primals_344, add_248);  primals_344 = add_248 = None
    copy__141: "i64[]" = torch.ops.aten.copy_.default(primals_345, add_251);  primals_345 = add_251 = None
    copy__142: "f32[256]" = torch.ops.aten.copy_.default(primals_346, add_253);  primals_346 = add_253 = None
    copy__143: "f32[256]" = torch.ops.aten.copy_.default(primals_347, add_254);  primals_347 = add_254 = None
    copy__144: "i64[]" = torch.ops.aten.copy_.default(primals_348, add_256);  primals_348 = add_256 = None
    copy__145: "f32[256]" = torch.ops.aten.copy_.default(primals_349, add_258);  primals_349 = add_258 = None
    copy__146: "f32[256]" = torch.ops.aten.copy_.default(primals_350, add_259);  primals_350 = add_259 = None
    copy__147: "i64[]" = torch.ops.aten.copy_.default(primals_351, add_262);  primals_351 = add_262 = None
    copy__148: "f32[256]" = torch.ops.aten.copy_.default(primals_352, add_264);  primals_352 = add_264 = None
    copy__149: "f32[256]" = torch.ops.aten.copy_.default(primals_353, add_265);  primals_353 = add_265 = None
    copy__150: "i64[]" = torch.ops.aten.copy_.default(primals_354, add_267);  primals_354 = add_267 = None
    copy__151: "f32[256]" = torch.ops.aten.copy_.default(primals_355, add_269);  primals_355 = add_269 = None
    copy__152: "f32[256]" = torch.ops.aten.copy_.default(primals_356, add_270);  primals_356 = add_270 = None
    copy__153: "i64[]" = torch.ops.aten.copy_.default(primals_357, add_273);  primals_357 = add_273 = None
    copy__154: "f32[256]" = torch.ops.aten.copy_.default(primals_358, add_275);  primals_358 = add_275 = None
    copy__155: "f32[256]" = torch.ops.aten.copy_.default(primals_359, add_276);  primals_359 = add_276 = None
    copy__156: "i64[]" = torch.ops.aten.copy_.default(primals_360, add_278);  primals_360 = add_278 = None
    copy__157: "f32[256]" = torch.ops.aten.copy_.default(primals_361, add_280);  primals_361 = add_280 = None
    copy__158: "f32[256]" = torch.ops.aten.copy_.default(primals_362, add_281);  primals_362 = add_281 = None
    copy__159: "i64[]" = torch.ops.aten.copy_.default(primals_363, add_284);  primals_363 = add_284 = None
    copy__160: "f32[256]" = torch.ops.aten.copy_.default(primals_364, add_286);  primals_364 = add_286 = None
    copy__161: "f32[256]" = torch.ops.aten.copy_.default(primals_365, add_287);  primals_365 = add_287 = None
    copy__162: "i64[]" = torch.ops.aten.copy_.default(primals_366, add_289);  primals_366 = add_289 = None
    copy__163: "f32[512]" = torch.ops.aten.copy_.default(primals_367, add_291);  primals_367 = add_291 = None
    copy__164: "f32[512]" = torch.ops.aten.copy_.default(primals_368, add_292);  primals_368 = add_292 = None
    copy__165: "i64[]" = torch.ops.aten.copy_.default(primals_369, add_294);  primals_369 = add_294 = None
    copy__166: "f32[1024]" = torch.ops.aten.copy_.default(primals_370, add_296);  primals_370 = add_296 = None
    copy__167: "f32[1024]" = torch.ops.aten.copy_.default(primals_371, add_297);  primals_371 = add_297 = None
    copy__168: "i64[]" = torch.ops.aten.copy_.default(primals_372, add_299);  primals_372 = add_299 = None
    copy__169: "f32[1024]" = torch.ops.aten.copy_.default(primals_373, add_301);  primals_373 = add_301 = None
    copy__170: "f32[1024]" = torch.ops.aten.copy_.default(primals_374, add_302);  primals_374 = add_302 = None
    copy__171: "i64[]" = torch.ops.aten.copy_.default(primals_375, add_304);  primals_375 = add_304 = None
    copy__172: "f32[512]" = torch.ops.aten.copy_.default(primals_376, add_306);  primals_376 = add_306 = None
    copy__173: "f32[512]" = torch.ops.aten.copy_.default(primals_377, add_307);  primals_377 = add_307 = None
    copy__174: "i64[]" = torch.ops.aten.copy_.default(primals_378, add_309);  primals_378 = add_309 = None
    copy__175: "f32[512]" = torch.ops.aten.copy_.default(primals_379, add_311);  primals_379 = add_311 = None
    copy__176: "f32[512]" = torch.ops.aten.copy_.default(primals_380, add_312);  primals_380 = add_312 = None
    copy__177: "i64[]" = torch.ops.aten.copy_.default(primals_381, add_315);  primals_381 = add_315 = None
    copy__178: "f32[512]" = torch.ops.aten.copy_.default(primals_382, add_317);  primals_382 = add_317 = None
    copy__179: "f32[512]" = torch.ops.aten.copy_.default(primals_383, add_318);  primals_383 = add_318 = None
    copy__180: "i64[]" = torch.ops.aten.copy_.default(primals_384, add_320);  primals_384 = add_320 = None
    copy__181: "f32[512]" = torch.ops.aten.copy_.default(primals_385, add_322);  primals_385 = add_322 = None
    copy__182: "f32[512]" = torch.ops.aten.copy_.default(primals_386, add_323);  primals_386 = add_323 = None
    copy__183: "i64[]" = torch.ops.aten.copy_.default(primals_387, add_326);  primals_387 = add_326 = None
    copy__184: "f32[512]" = torch.ops.aten.copy_.default(primals_388, add_328);  primals_388 = add_328 = None
    copy__185: "f32[512]" = torch.ops.aten.copy_.default(primals_389, add_329);  primals_389 = add_329 = None
    copy__186: "i64[]" = torch.ops.aten.copy_.default(primals_390, add_331);  primals_390 = add_331 = None
    copy__187: "f32[512]" = torch.ops.aten.copy_.default(primals_391, add_333);  primals_391 = add_333 = None
    copy__188: "f32[512]" = torch.ops.aten.copy_.default(primals_392, add_334);  primals_392 = add_334 = None
    copy__189: "i64[]" = torch.ops.aten.copy_.default(primals_393, add_337);  primals_393 = add_337 = None
    copy__190: "f32[512]" = torch.ops.aten.copy_.default(primals_394, add_339);  primals_394 = add_339 = None
    copy__191: "f32[512]" = torch.ops.aten.copy_.default(primals_395, add_340);  primals_395 = add_340 = None
    copy__192: "i64[]" = torch.ops.aten.copy_.default(primals_396, add_342);  primals_396 = add_342 = None
    copy__193: "f32[512]" = torch.ops.aten.copy_.default(primals_397, add_344);  primals_397 = add_344 = None
    copy__194: "f32[512]" = torch.ops.aten.copy_.default(primals_398, add_345);  primals_398 = add_345 = None
    copy__195: "i64[]" = torch.ops.aten.copy_.default(primals_399, add_348);  primals_399 = add_348 = None
    copy__196: "f32[512]" = torch.ops.aten.copy_.default(primals_400, add_350);  primals_400 = add_350 = None
    copy__197: "f32[512]" = torch.ops.aten.copy_.default(primals_401, add_351);  primals_401 = add_351 = None
    copy__198: "i64[]" = torch.ops.aten.copy_.default(primals_402, add_353);  primals_402 = add_353 = None
    copy__199: "f32[1024]" = torch.ops.aten.copy_.default(primals_403, add_355);  primals_403 = add_355 = None
    copy__200: "f32[1024]" = torch.ops.aten.copy_.default(primals_404, add_356);  primals_404 = add_356 = None
    return pytree.tree_unflatten([addmm, mul_1205, sum_134, mul_1195, sum_132, mul_1185, sum_130, mul_1175, sum_128, mul_1165, sum_126, mul_1155, sum_124, mul_1145, sum_122, mul_1135, sum_120, mul_1125, sum_118, mul_1115, sum_116, mul_1105, sum_114, mul_1095, sum_112, mul_1085, sum_110, mul_1075, sum_108, mul_1065, sum_106, mul_1055, sum_104, mul_1045, sum_102, mul_1035, sum_100, mul_1025, sum_98, mul_1015, sum_96, mul_1005, sum_94, mul_995, sum_92, mul_985, sum_90, mul_975, sum_88, mul_965, sum_86, mul_955, sum_84, mul_945, sum_82, mul_935, sum_80, mul_925, sum_78, mul_915, sum_76, mul_905, sum_74, mul_895, sum_72, mul_885, sum_70, mul_875, sum_68, mul_865, sum_66, mul_855, sum_64, mul_845, sum_62, mul_835, sum_60, mul_825, sum_58, mul_815, sum_56, mul_805, sum_54, mul_795, sum_52, mul_785, sum_50, mul_775, sum_48, mul_765, sum_46, mul_755, sum_44, mul_745, sum_42, mul_735, sum_40, mul_725, sum_38, mul_715, sum_36, mul_705, sum_34, mul_695, sum_32, mul_685, sum_30, mul_675, sum_28, mul_665, sum_26, mul_655, sum_24, mul_645, sum_22, mul_635, sum_20, mul_625, sum_18, mul_615, sum_16, mul_605, sum_14, mul_595, sum_12, mul_585, sum_10, mul_575, sum_8, mul_565, sum_6, mul_555, sum_4, mul_545, sum_2, getitem_363, getitem_360, getitem_357, getitem_354, getitem_351, getitem_348, getitem_345, getitem_342, getitem_339, getitem_336, getitem_333, getitem_330, getitem_327, getitem_324, getitem_321, getitem_318, getitem_315, getitem_312, getitem_309, getitem_306, getitem_303, getitem_300, getitem_297, getitem_294, getitem_291, getitem_288, getitem_285, getitem_282, getitem_279, getitem_276, getitem_273, getitem_270, getitem_267, getitem_264, getitem_261, getitem_258, getitem_255, getitem_252, getitem_249, getitem_246, getitem_243, getitem_240, getitem_237, getitem_234, getitem_231, getitem_228, getitem_225, getitem_222, getitem_219, getitem_216, getitem_213, getitem_210, getitem_207, getitem_204, getitem_201, getitem_198, getitem_195, getitem_192, getitem_189, getitem_186, getitem_183, getitem_180, getitem_177, getitem_174, getitem_171, getitem_168, getitem_165, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    