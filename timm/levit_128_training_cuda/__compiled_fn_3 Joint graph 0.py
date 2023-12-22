from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[4, 196]"; primals_2: "f32[4, 196]"; primals_3: "f32[4, 196]"; primals_4: "f32[4, 196]"; primals_5: "f32[8, 196]"; primals_6: "f32[8, 49]"; primals_7: "f32[8, 49]"; primals_8: "f32[8, 49]"; primals_9: "f32[8, 49]"; primals_10: "f32[16, 49]"; primals_11: "f32[12, 16]"; primals_12: "f32[12, 16]"; primals_13: "f32[12, 16]"; primals_14: "f32[12, 16]"; primals_15: "f32[16, 3, 3, 3]"; primals_16: "f32[16]"; primals_17: "f32[16]"; primals_18: "f32[32, 16, 3, 3]"; primals_19: "f32[32]"; primals_20: "f32[32]"; primals_21: "f32[64, 32, 3, 3]"; primals_22: "f32[64]"; primals_23: "f32[64]"; primals_24: "f32[128, 64, 3, 3]"; primals_25: "f32[128]"; primals_26: "f32[128]"; primals_27: "f32[256, 128]"; primals_28: "f32[256]"; primals_29: "f32[256]"; primals_30: "f32[128, 128]"; primals_31: "f32[128]"; primals_32: "f32[128]"; primals_33: "f32[256, 128]"; primals_34: "f32[256]"; primals_35: "f32[256]"; primals_36: "f32[128, 256]"; primals_37: "f32[128]"; primals_38: "f32[128]"; primals_39: "f32[256, 128]"; primals_40: "f32[256]"; primals_41: "f32[256]"; primals_42: "f32[128, 128]"; primals_43: "f32[128]"; primals_44: "f32[128]"; primals_45: "f32[256, 128]"; primals_46: "f32[256]"; primals_47: "f32[256]"; primals_48: "f32[128, 256]"; primals_49: "f32[128]"; primals_50: "f32[128]"; primals_51: "f32[256, 128]"; primals_52: "f32[256]"; primals_53: "f32[256]"; primals_54: "f32[128, 128]"; primals_55: "f32[128]"; primals_56: "f32[128]"; primals_57: "f32[256, 128]"; primals_58: "f32[256]"; primals_59: "f32[256]"; primals_60: "f32[128, 256]"; primals_61: "f32[128]"; primals_62: "f32[128]"; primals_63: "f32[256, 128]"; primals_64: "f32[256]"; primals_65: "f32[256]"; primals_66: "f32[128, 128]"; primals_67: "f32[128]"; primals_68: "f32[128]"; primals_69: "f32[256, 128]"; primals_70: "f32[256]"; primals_71: "f32[256]"; primals_72: "f32[128, 256]"; primals_73: "f32[128]"; primals_74: "f32[128]"; primals_75: "f32[640, 128]"; primals_76: "f32[640]"; primals_77: "f32[640]"; primals_78: "f32[128, 128]"; primals_79: "f32[128]"; primals_80: "f32[128]"; primals_81: "f32[256, 512]"; primals_82: "f32[256]"; primals_83: "f32[256]"; primals_84: "f32[512, 256]"; primals_85: "f32[512]"; primals_86: "f32[512]"; primals_87: "f32[256, 512]"; primals_88: "f32[256]"; primals_89: "f32[256]"; primals_90: "f32[512, 256]"; primals_91: "f32[512]"; primals_92: "f32[512]"; primals_93: "f32[256, 256]"; primals_94: "f32[256]"; primals_95: "f32[256]"; primals_96: "f32[512, 256]"; primals_97: "f32[512]"; primals_98: "f32[512]"; primals_99: "f32[256, 512]"; primals_100: "f32[256]"; primals_101: "f32[256]"; primals_102: "f32[512, 256]"; primals_103: "f32[512]"; primals_104: "f32[512]"; primals_105: "f32[256, 256]"; primals_106: "f32[256]"; primals_107: "f32[256]"; primals_108: "f32[512, 256]"; primals_109: "f32[512]"; primals_110: "f32[512]"; primals_111: "f32[256, 512]"; primals_112: "f32[256]"; primals_113: "f32[256]"; primals_114: "f32[512, 256]"; primals_115: "f32[512]"; primals_116: "f32[512]"; primals_117: "f32[256, 256]"; primals_118: "f32[256]"; primals_119: "f32[256]"; primals_120: "f32[512, 256]"; primals_121: "f32[512]"; primals_122: "f32[512]"; primals_123: "f32[256, 512]"; primals_124: "f32[256]"; primals_125: "f32[256]"; primals_126: "f32[512, 256]"; primals_127: "f32[512]"; primals_128: "f32[512]"; primals_129: "f32[256, 256]"; primals_130: "f32[256]"; primals_131: "f32[256]"; primals_132: "f32[512, 256]"; primals_133: "f32[512]"; primals_134: "f32[512]"; primals_135: "f32[256, 512]"; primals_136: "f32[256]"; primals_137: "f32[256]"; primals_138: "f32[1280, 256]"; primals_139: "f32[1280]"; primals_140: "f32[1280]"; primals_141: "f32[256, 256]"; primals_142: "f32[256]"; primals_143: "f32[256]"; primals_144: "f32[384, 1024]"; primals_145: "f32[384]"; primals_146: "f32[384]"; primals_147: "f32[768, 384]"; primals_148: "f32[768]"; primals_149: "f32[768]"; primals_150: "f32[384, 768]"; primals_151: "f32[384]"; primals_152: "f32[384]"; primals_153: "f32[768, 384]"; primals_154: "f32[768]"; primals_155: "f32[768]"; primals_156: "f32[384, 384]"; primals_157: "f32[384]"; primals_158: "f32[384]"; primals_159: "f32[768, 384]"; primals_160: "f32[768]"; primals_161: "f32[768]"; primals_162: "f32[384, 768]"; primals_163: "f32[384]"; primals_164: "f32[384]"; primals_165: "f32[768, 384]"; primals_166: "f32[768]"; primals_167: "f32[768]"; primals_168: "f32[384, 384]"; primals_169: "f32[384]"; primals_170: "f32[384]"; primals_171: "f32[768, 384]"; primals_172: "f32[768]"; primals_173: "f32[768]"; primals_174: "f32[384, 768]"; primals_175: "f32[384]"; primals_176: "f32[384]"; primals_177: "f32[768, 384]"; primals_178: "f32[768]"; primals_179: "f32[768]"; primals_180: "f32[384, 384]"; primals_181: "f32[384]"; primals_182: "f32[384]"; primals_183: "f32[768, 384]"; primals_184: "f32[768]"; primals_185: "f32[768]"; primals_186: "f32[384, 768]"; primals_187: "f32[384]"; primals_188: "f32[384]"; primals_189: "f32[768, 384]"; primals_190: "f32[768]"; primals_191: "f32[768]"; primals_192: "f32[384, 384]"; primals_193: "f32[384]"; primals_194: "f32[384]"; primals_195: "f32[768, 384]"; primals_196: "f32[768]"; primals_197: "f32[768]"; primals_198: "f32[384, 768]"; primals_199: "f32[384]"; primals_200: "f32[384]"; primals_201: "f32[384]"; primals_202: "f32[384]"; primals_203: "f32[1000, 384]"; primals_204: "f32[1000]"; primals_205: "f32[384]"; primals_206: "f32[384]"; primals_207: "f32[1000, 384]"; primals_208: "f32[1000]"; primals_209: "i64[196, 196]"; primals_210: "i64[196, 196]"; primals_211: "i64[196, 196]"; primals_212: "i64[196, 196]"; primals_213: "i64[49, 196]"; primals_214: "i64[49, 49]"; primals_215: "i64[49, 49]"; primals_216: "i64[49, 49]"; primals_217: "i64[49, 49]"; primals_218: "i64[16, 49]"; primals_219: "i64[16, 16]"; primals_220: "i64[16, 16]"; primals_221: "i64[16, 16]"; primals_222: "i64[16, 16]"; primals_223: "f32[16]"; primals_224: "f32[16]"; primals_225: "i64[]"; primals_226: "f32[32]"; primals_227: "f32[32]"; primals_228: "i64[]"; primals_229: "f32[64]"; primals_230: "f32[64]"; primals_231: "i64[]"; primals_232: "f32[128]"; primals_233: "f32[128]"; primals_234: "i64[]"; primals_235: "f32[256]"; primals_236: "f32[256]"; primals_237: "i64[]"; primals_238: "f32[128]"; primals_239: "f32[128]"; primals_240: "i64[]"; primals_241: "f32[256]"; primals_242: "f32[256]"; primals_243: "i64[]"; primals_244: "f32[128]"; primals_245: "f32[128]"; primals_246: "i64[]"; primals_247: "f32[256]"; primals_248: "f32[256]"; primals_249: "i64[]"; primals_250: "f32[128]"; primals_251: "f32[128]"; primals_252: "i64[]"; primals_253: "f32[256]"; primals_254: "f32[256]"; primals_255: "i64[]"; primals_256: "f32[128]"; primals_257: "f32[128]"; primals_258: "i64[]"; primals_259: "f32[256]"; primals_260: "f32[256]"; primals_261: "i64[]"; primals_262: "f32[128]"; primals_263: "f32[128]"; primals_264: "i64[]"; primals_265: "f32[256]"; primals_266: "f32[256]"; primals_267: "i64[]"; primals_268: "f32[128]"; primals_269: "f32[128]"; primals_270: "i64[]"; primals_271: "f32[256]"; primals_272: "f32[256]"; primals_273: "i64[]"; primals_274: "f32[128]"; primals_275: "f32[128]"; primals_276: "i64[]"; primals_277: "f32[256]"; primals_278: "f32[256]"; primals_279: "i64[]"; primals_280: "f32[128]"; primals_281: "f32[128]"; primals_282: "i64[]"; primals_283: "f32[640]"; primals_284: "f32[640]"; primals_285: "i64[]"; primals_286: "f32[128]"; primals_287: "f32[128]"; primals_288: "i64[]"; primals_289: "f32[256]"; primals_290: "f32[256]"; primals_291: "i64[]"; primals_292: "f32[512]"; primals_293: "f32[512]"; primals_294: "i64[]"; primals_295: "f32[256]"; primals_296: "f32[256]"; primals_297: "i64[]"; primals_298: "f32[512]"; primals_299: "f32[512]"; primals_300: "i64[]"; primals_301: "f32[256]"; primals_302: "f32[256]"; primals_303: "i64[]"; primals_304: "f32[512]"; primals_305: "f32[512]"; primals_306: "i64[]"; primals_307: "f32[256]"; primals_308: "f32[256]"; primals_309: "i64[]"; primals_310: "f32[512]"; primals_311: "f32[512]"; primals_312: "i64[]"; primals_313: "f32[256]"; primals_314: "f32[256]"; primals_315: "i64[]"; primals_316: "f32[512]"; primals_317: "f32[512]"; primals_318: "i64[]"; primals_319: "f32[256]"; primals_320: "f32[256]"; primals_321: "i64[]"; primals_322: "f32[512]"; primals_323: "f32[512]"; primals_324: "i64[]"; primals_325: "f32[256]"; primals_326: "f32[256]"; primals_327: "i64[]"; primals_328: "f32[512]"; primals_329: "f32[512]"; primals_330: "i64[]"; primals_331: "f32[256]"; primals_332: "f32[256]"; primals_333: "i64[]"; primals_334: "f32[512]"; primals_335: "f32[512]"; primals_336: "i64[]"; primals_337: "f32[256]"; primals_338: "f32[256]"; primals_339: "i64[]"; primals_340: "f32[512]"; primals_341: "f32[512]"; primals_342: "i64[]"; primals_343: "f32[256]"; primals_344: "f32[256]"; primals_345: "i64[]"; primals_346: "f32[1280]"; primals_347: "f32[1280]"; primals_348: "i64[]"; primals_349: "f32[256]"; primals_350: "f32[256]"; primals_351: "i64[]"; primals_352: "f32[384]"; primals_353: "f32[384]"; primals_354: "i64[]"; primals_355: "f32[768]"; primals_356: "f32[768]"; primals_357: "i64[]"; primals_358: "f32[384]"; primals_359: "f32[384]"; primals_360: "i64[]"; primals_361: "f32[768]"; primals_362: "f32[768]"; primals_363: "i64[]"; primals_364: "f32[384]"; primals_365: "f32[384]"; primals_366: "i64[]"; primals_367: "f32[768]"; primals_368: "f32[768]"; primals_369: "i64[]"; primals_370: "f32[384]"; primals_371: "f32[384]"; primals_372: "i64[]"; primals_373: "f32[768]"; primals_374: "f32[768]"; primals_375: "i64[]"; primals_376: "f32[384]"; primals_377: "f32[384]"; primals_378: "i64[]"; primals_379: "f32[768]"; primals_380: "f32[768]"; primals_381: "i64[]"; primals_382: "f32[384]"; primals_383: "f32[384]"; primals_384: "i64[]"; primals_385: "f32[768]"; primals_386: "f32[768]"; primals_387: "i64[]"; primals_388: "f32[384]"; primals_389: "f32[384]"; primals_390: "i64[]"; primals_391: "f32[768]"; primals_392: "f32[768]"; primals_393: "i64[]"; primals_394: "f32[384]"; primals_395: "f32[384]"; primals_396: "i64[]"; primals_397: "f32[768]"; primals_398: "f32[768]"; primals_399: "i64[]"; primals_400: "f32[384]"; primals_401: "f32[384]"; primals_402: "i64[]"; primals_403: "f32[768]"; primals_404: "f32[768]"; primals_405: "i64[]"; primals_406: "f32[384]"; primals_407: "f32[384]"; primals_408: "i64[]"; primals_409: "f32[384]"; primals_410: "f32[384]"; primals_411: "i64[]"; primals_412: "f32[384]"; primals_413: "f32[384]"; primals_414: "i64[]"; primals_415: "f32[8, 3, 224, 224]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:65, code: return self.bn(self.linear(x))
    convolution: "f32[8, 16, 112, 112]" = torch.ops.aten.convolution.default(primals_415, primals_15, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_225, 1)
    var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
    getitem: "f32[1, 16, 1, 1]" = var_mean[0]
    getitem_1: "f32[1, 16, 1, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
    rsqrt: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
    mul: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    squeeze: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
    squeeze_1: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
    mul_1: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
    mul_2: "f32[16]" = torch.ops.aten.mul.Tensor(primals_223, 0.9)
    add_2: "f32[16]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[16]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.00000996502277);  squeeze_2 = None
    mul_4: "f32[16]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[16]" = torch.ops.aten.mul.Tensor(primals_224, 0.9)
    add_3: "f32[16]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1)
    unsqueeze_1: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1);  primals_17 = None
    unsqueeze_3: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:637, code: x = self.stem(x)
    add_5: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(add_4, 3)
    clamp_min: "f32[8, 16, 112, 112]" = torch.ops.aten.clamp_min.default(add_5, 0);  add_5 = None
    clamp_max: "f32[8, 16, 112, 112]" = torch.ops.aten.clamp_max.default(clamp_min, 6);  clamp_min = None
    mul_7: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(add_4, clamp_max);  clamp_max = None
    div: "f32[8, 16, 112, 112]" = torch.ops.aten.div.Tensor(mul_7, 6);  mul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:65, code: return self.bn(self.linear(x))
    convolution_1: "f32[8, 32, 56, 56]" = torch.ops.aten.convolution.default(div, primals_18, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    add_6: "i64[]" = torch.ops.aten.add.Tensor(primals_228, 1)
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[1, 32, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 32, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_7: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
    rsqrt_1: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    sub_1: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
    mul_8: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
    squeeze_4: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_9: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_10: "f32[32]" = torch.ops.aten.mul.Tensor(primals_226, 0.9)
    add_8: "f32[32]" = torch.ops.aten.add.Tensor(mul_9, mul_10);  mul_9 = mul_10 = None
    squeeze_5: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_11: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000398612827361);  squeeze_5 = None
    mul_12: "f32[32]" = torch.ops.aten.mul.Tensor(mul_11, 0.1);  mul_11 = None
    mul_13: "f32[32]" = torch.ops.aten.mul.Tensor(primals_227, 0.9)
    add_9: "f32[32]" = torch.ops.aten.add.Tensor(mul_12, mul_13);  mul_12 = mul_13 = None
    unsqueeze_4: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
    unsqueeze_5: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_14: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(mul_8, unsqueeze_5);  mul_8 = unsqueeze_5 = None
    unsqueeze_6: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
    unsqueeze_7: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_10: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_7);  mul_14 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:637, code: x = self.stem(x)
    add_11: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(add_10, 3)
    clamp_min_1: "f32[8, 32, 56, 56]" = torch.ops.aten.clamp_min.default(add_11, 0);  add_11 = None
    clamp_max_1: "f32[8, 32, 56, 56]" = torch.ops.aten.clamp_max.default(clamp_min_1, 6);  clamp_min_1 = None
    mul_15: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(add_10, clamp_max_1);  clamp_max_1 = None
    div_1: "f32[8, 32, 56, 56]" = torch.ops.aten.div.Tensor(mul_15, 6);  mul_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:65, code: return self.bn(self.linear(x))
    convolution_2: "f32[8, 64, 28, 28]" = torch.ops.aten.convolution.default(div_1, primals_21, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    add_12: "i64[]" = torch.ops.aten.add.Tensor(primals_231, 1)
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 64, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 64, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_13: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
    rsqrt_2: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
    sub_2: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
    mul_16: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_17: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_18: "f32[64]" = torch.ops.aten.mul.Tensor(primals_229, 0.9)
    add_14: "f32[64]" = torch.ops.aten.add.Tensor(mul_17, mul_18);  mul_17 = mul_18 = None
    squeeze_8: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_19: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0001594642002871);  squeeze_8 = None
    mul_20: "f32[64]" = torch.ops.aten.mul.Tensor(mul_19, 0.1);  mul_19 = None
    mul_21: "f32[64]" = torch.ops.aten.mul.Tensor(primals_230, 0.9)
    add_15: "f32[64]" = torch.ops.aten.add.Tensor(mul_20, mul_21);  mul_20 = mul_21 = None
    unsqueeze_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1)
    unsqueeze_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_22: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_9);  mul_16 = unsqueeze_9 = None
    unsqueeze_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1);  primals_23 = None
    unsqueeze_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_16: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(mul_22, unsqueeze_11);  mul_22 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:637, code: x = self.stem(x)
    add_17: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(add_16, 3)
    clamp_min_2: "f32[8, 64, 28, 28]" = torch.ops.aten.clamp_min.default(add_17, 0);  add_17 = None
    clamp_max_2: "f32[8, 64, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_2, 6);  clamp_min_2 = None
    mul_23: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(add_16, clamp_max_2);  clamp_max_2 = None
    div_2: "f32[8, 64, 28, 28]" = torch.ops.aten.div.Tensor(mul_23, 6);  mul_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:65, code: return self.bn(self.linear(x))
    convolution_3: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_2, primals_24, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    add_18: "i64[]" = torch.ops.aten.add.Tensor(primals_234, 1)
    var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
    getitem_6: "f32[1, 128, 1, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 128, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_19: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
    rsqrt_3: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
    sub_3: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_7)
    mul_24: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
    squeeze_10: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_25: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_26: "f32[128]" = torch.ops.aten.mul.Tensor(primals_232, 0.9)
    add_20: "f32[128]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
    squeeze_11: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
    mul_27: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0006381620931717);  squeeze_11 = None
    mul_28: "f32[128]" = torch.ops.aten.mul.Tensor(mul_27, 0.1);  mul_27 = None
    mul_29: "f32[128]" = torch.ops.aten.mul.Tensor(primals_233, 0.9)
    add_21: "f32[128]" = torch.ops.aten.add.Tensor(mul_28, mul_29);  mul_28 = mul_29 = None
    unsqueeze_12: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1)
    unsqueeze_13: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_30: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_24, unsqueeze_13);  mul_24 = unsqueeze_13 = None
    unsqueeze_14: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1);  primals_26 = None
    unsqueeze_15: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_22: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_30, unsqueeze_15);  mul_30 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:639, code: x = x.flatten(2).transpose(1, 2)
    view: "f32[8, 128, 196]" = torch.ops.aten.view.default(add_22, [8, 128, 196]);  add_22 = None
    permute: "f32[8, 196, 128]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_1: "f32[128, 256]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
    clone: "f32[8, 196, 128]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format)
    view_1: "f32[1568, 128]" = torch.ops.aten.view.default(clone, [1568, 128]);  clone = None
    mm: "f32[1568, 256]" = torch.ops.aten.mm.default(view_1, permute_1)
    view_2: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm, [8, 196, 256]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_3: "f32[1568, 256]" = torch.ops.aten.view.default(view_2, [1568, 256]);  view_2 = None
    add_23: "i64[]" = torch.ops.aten.add.Tensor(primals_237, 1)
    var_mean_4 = torch.ops.aten.var_mean.correction(view_3, [0], correction = 0, keepdim = True)
    getitem_8: "f32[1, 256]" = var_mean_4[0]
    getitem_9: "f32[1, 256]" = var_mean_4[1];  var_mean_4 = None
    add_24: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
    rsqrt_4: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    sub_4: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_3, getitem_9)
    mul_31: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_9, [0]);  getitem_9 = None
    squeeze_13: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0]);  rsqrt_4 = None
    mul_32: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_33: "f32[256]" = torch.ops.aten.mul.Tensor(primals_235, 0.9)
    add_25: "f32[256]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
    squeeze_14: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_8, [0]);  getitem_8 = None
    mul_34: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0006381620931717);  squeeze_14 = None
    mul_35: "f32[256]" = torch.ops.aten.mul.Tensor(mul_34, 0.1);  mul_34 = None
    mul_36: "f32[256]" = torch.ops.aten.mul.Tensor(primals_236, 0.9)
    add_26: "f32[256]" = torch.ops.aten.add.Tensor(mul_35, mul_36);  mul_35 = mul_36 = None
    mul_37: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(mul_31, primals_28);  mul_31 = None
    add_27: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mul_37, primals_29);  mul_37 = primals_29 = None
    view_4: "f32[8, 196, 256]" = torch.ops.aten.view.default(add_27, [8, 196, 256]);  add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_5: "f32[8, 196, 4, 64]" = torch.ops.aten.view.default(view_4, [8, 196, 4, -1]);  view_4 = None
    split_with_sizes = torch.ops.aten.split_with_sizes.default(view_5, [16, 16, 32], 3);  view_5 = None
    getitem_10: "f32[8, 196, 4, 16]" = split_with_sizes[0]
    getitem_11: "f32[8, 196, 4, 16]" = split_with_sizes[1]
    getitem_12: "f32[8, 196, 4, 32]" = split_with_sizes[2];  split_with_sizes = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_2: "f32[8, 4, 196, 16]" = torch.ops.aten.permute.default(getitem_10, [0, 2, 1, 3]);  getitem_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_3: "f32[8, 4, 16, 196]" = torch.ops.aten.permute.default(getitem_11, [0, 2, 3, 1]);  getitem_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_4: "f32[8, 4, 196, 32]" = torch.ops.aten.permute.default(getitem_12, [0, 2, 1, 3]);  getitem_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand: "f32[8, 4, 196, 16]" = torch.ops.aten.expand.default(permute_2, [8, 4, 196, 16]);  permute_2 = None
    clone_1: "f32[8, 4, 196, 16]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
    view_6: "f32[32, 196, 16]" = torch.ops.aten.view.default(clone_1, [32, 196, 16]);  clone_1 = None
    expand_1: "f32[8, 4, 16, 196]" = torch.ops.aten.expand.default(permute_3, [8, 4, 16, 196]);  permute_3 = None
    clone_2: "f32[8, 4, 16, 196]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_7: "f32[32, 16, 196]" = torch.ops.aten.view.default(clone_2, [32, 16, 196]);  clone_2 = None
    bmm: "f32[32, 196, 196]" = torch.ops.aten.bmm.default(view_6, view_7)
    view_8: "f32[8, 4, 196, 196]" = torch.ops.aten.view.default(bmm, [8, 4, 196, 196]);  bmm = None
    mul_38: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(view_8, 0.25);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    slice_1: "f32[4, 196]" = torch.ops.aten.slice.Tensor(primals_1, 0, 0, 9223372036854775807);  primals_1 = None
    index: "f32[4, 196, 196]" = torch.ops.aten.index.Tensor(slice_1, [None, primals_209]);  slice_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_28: "f32[8, 4, 196, 196]" = torch.ops.aten.add.Tensor(mul_38, index);  mul_38 = index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax: "f32[8, 4, 196, 1]" = torch.ops.aten.amax.default(add_28, [-1], True)
    sub_5: "f32[8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(add_28, amax);  add_28 = amax = None
    exp: "f32[8, 4, 196, 196]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
    sum_1: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_3: "f32[8, 4, 196, 196]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[8, 4, 196, 196]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_2: "f32[8, 4, 196, 196]" = torch.ops.aten.expand.default(div_3, [8, 4, 196, 196]);  div_3 = None
    view_9: "f32[32, 196, 196]" = torch.ops.aten.view.default(expand_2, [32, 196, 196]);  expand_2 = None
    expand_3: "f32[8, 4, 196, 32]" = torch.ops.aten.expand.default(permute_4, [8, 4, 196, 32]);  permute_4 = None
    clone_3: "f32[8, 4, 196, 32]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    view_10: "f32[32, 196, 32]" = torch.ops.aten.view.default(clone_3, [32, 196, 32]);  clone_3 = None
    bmm_1: "f32[32, 196, 32]" = torch.ops.aten.bmm.default(view_9, view_10)
    view_11: "f32[8, 4, 196, 32]" = torch.ops.aten.view.default(bmm_1, [8, 4, 196, 32]);  bmm_1 = None
    permute_5: "f32[8, 196, 4, 32]" = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None
    clone_4: "f32[8, 196, 4, 32]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    view_12: "f32[8, 196, 128]" = torch.ops.aten.view.default(clone_4, [8, 196, 128]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_29: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(view_12, 3)
    clamp_min_3: "f32[8, 196, 128]" = torch.ops.aten.clamp_min.default(add_29, 0);  add_29 = None
    clamp_max_3: "f32[8, 196, 128]" = torch.ops.aten.clamp_max.default(clamp_min_3, 6);  clamp_min_3 = None
    mul_39: "f32[8, 196, 128]" = torch.ops.aten.mul.Tensor(view_12, clamp_max_3);  clamp_max_3 = None
    div_4: "f32[8, 196, 128]" = torch.ops.aten.div.Tensor(mul_39, 6);  mul_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_6: "f32[128, 128]" = torch.ops.aten.permute.default(primals_30, [1, 0]);  primals_30 = None
    view_13: "f32[1568, 128]" = torch.ops.aten.view.default(div_4, [1568, 128]);  div_4 = None
    mm_1: "f32[1568, 128]" = torch.ops.aten.mm.default(view_13, permute_6)
    view_14: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_1, [8, 196, 128]);  mm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_15: "f32[1568, 128]" = torch.ops.aten.view.default(view_14, [1568, 128]);  view_14 = None
    add_30: "i64[]" = torch.ops.aten.add.Tensor(primals_240, 1)
    var_mean_5 = torch.ops.aten.var_mean.correction(view_15, [0], correction = 0, keepdim = True)
    getitem_13: "f32[1, 128]" = var_mean_5[0]
    getitem_14: "f32[1, 128]" = var_mean_5[1];  var_mean_5 = None
    add_31: "f32[1, 128]" = torch.ops.aten.add.Tensor(getitem_13, 1e-05)
    rsqrt_5: "f32[1, 128]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_6: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_15, getitem_14)
    mul_40: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_5);  sub_6 = None
    squeeze_15: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_14, [0]);  getitem_14 = None
    squeeze_16: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0]);  rsqrt_5 = None
    mul_41: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_42: "f32[128]" = torch.ops.aten.mul.Tensor(primals_238, 0.9)
    add_32: "f32[128]" = torch.ops.aten.add.Tensor(mul_41, mul_42);  mul_41 = mul_42 = None
    squeeze_17: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_13, [0]);  getitem_13 = None
    mul_43: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0006381620931717);  squeeze_17 = None
    mul_44: "f32[128]" = torch.ops.aten.mul.Tensor(mul_43, 0.1);  mul_43 = None
    mul_45: "f32[128]" = torch.ops.aten.mul.Tensor(primals_239, 0.9)
    add_33: "f32[128]" = torch.ops.aten.add.Tensor(mul_44, mul_45);  mul_44 = mul_45 = None
    mul_46: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(mul_40, primals_31);  mul_40 = None
    add_34: "f32[1568, 128]" = torch.ops.aten.add.Tensor(mul_46, primals_32);  mul_46 = primals_32 = None
    view_16: "f32[8, 196, 128]" = torch.ops.aten.view.default(add_34, [8, 196, 128]);  add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_35: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(permute, view_16);  permute = view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_7: "f32[128, 256]" = torch.ops.aten.permute.default(primals_33, [1, 0]);  primals_33 = None
    clone_5: "f32[8, 196, 128]" = torch.ops.aten.clone.default(add_35, memory_format = torch.contiguous_format)
    view_17: "f32[1568, 128]" = torch.ops.aten.view.default(clone_5, [1568, 128]);  clone_5 = None
    mm_2: "f32[1568, 256]" = torch.ops.aten.mm.default(view_17, permute_7)
    view_18: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_2, [8, 196, 256]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_19: "f32[1568, 256]" = torch.ops.aten.view.default(view_18, [1568, 256]);  view_18 = None
    add_36: "i64[]" = torch.ops.aten.add.Tensor(primals_243, 1)
    var_mean_6 = torch.ops.aten.var_mean.correction(view_19, [0], correction = 0, keepdim = True)
    getitem_15: "f32[1, 256]" = var_mean_6[0]
    getitem_16: "f32[1, 256]" = var_mean_6[1];  var_mean_6 = None
    add_37: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_15, 1e-05)
    rsqrt_6: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_7: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_19, getitem_16)
    mul_47: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_6);  sub_7 = None
    squeeze_18: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_16, [0]);  getitem_16 = None
    squeeze_19: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0]);  rsqrt_6 = None
    mul_48: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_49: "f32[256]" = torch.ops.aten.mul.Tensor(primals_241, 0.9)
    add_38: "f32[256]" = torch.ops.aten.add.Tensor(mul_48, mul_49);  mul_48 = mul_49 = None
    squeeze_20: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_15, [0]);  getitem_15 = None
    mul_50: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0006381620931717);  squeeze_20 = None
    mul_51: "f32[256]" = torch.ops.aten.mul.Tensor(mul_50, 0.1);  mul_50 = None
    mul_52: "f32[256]" = torch.ops.aten.mul.Tensor(primals_242, 0.9)
    add_39: "f32[256]" = torch.ops.aten.add.Tensor(mul_51, mul_52);  mul_51 = mul_52 = None
    mul_53: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(mul_47, primals_34);  mul_47 = None
    add_40: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mul_53, primals_35);  mul_53 = primals_35 = None
    view_20: "f32[8, 196, 256]" = torch.ops.aten.view.default(add_40, [8, 196, 256]);  add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_41: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(view_20, 3)
    clamp_min_4: "f32[8, 196, 256]" = torch.ops.aten.clamp_min.default(add_41, 0);  add_41 = None
    clamp_max_4: "f32[8, 196, 256]" = torch.ops.aten.clamp_max.default(clamp_min_4, 6);  clamp_min_4 = None
    mul_54: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_20, clamp_max_4);  clamp_max_4 = None
    div_5: "f32[8, 196, 256]" = torch.ops.aten.div.Tensor(mul_54, 6);  mul_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    clone_6: "f32[8, 196, 256]" = torch.ops.aten.clone.default(div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_8: "f32[256, 128]" = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
    view_21: "f32[1568, 256]" = torch.ops.aten.view.default(clone_6, [1568, 256]);  clone_6 = None
    mm_3: "f32[1568, 128]" = torch.ops.aten.mm.default(view_21, permute_8)
    view_22: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_3, [8, 196, 128]);  mm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_23: "f32[1568, 128]" = torch.ops.aten.view.default(view_22, [1568, 128]);  view_22 = None
    add_42: "i64[]" = torch.ops.aten.add.Tensor(primals_246, 1)
    var_mean_7 = torch.ops.aten.var_mean.correction(view_23, [0], correction = 0, keepdim = True)
    getitem_17: "f32[1, 128]" = var_mean_7[0]
    getitem_18: "f32[1, 128]" = var_mean_7[1];  var_mean_7 = None
    add_43: "f32[1, 128]" = torch.ops.aten.add.Tensor(getitem_17, 1e-05)
    rsqrt_7: "f32[1, 128]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
    sub_8: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_23, getitem_18)
    mul_55: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_7);  sub_8 = None
    squeeze_21: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_18, [0]);  getitem_18 = None
    squeeze_22: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0]);  rsqrt_7 = None
    mul_56: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_57: "f32[128]" = torch.ops.aten.mul.Tensor(primals_244, 0.9)
    add_44: "f32[128]" = torch.ops.aten.add.Tensor(mul_56, mul_57);  mul_56 = mul_57 = None
    squeeze_23: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_17, [0]);  getitem_17 = None
    mul_58: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0006381620931717);  squeeze_23 = None
    mul_59: "f32[128]" = torch.ops.aten.mul.Tensor(mul_58, 0.1);  mul_58 = None
    mul_60: "f32[128]" = torch.ops.aten.mul.Tensor(primals_245, 0.9)
    add_45: "f32[128]" = torch.ops.aten.add.Tensor(mul_59, mul_60);  mul_59 = mul_60 = None
    mul_61: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(mul_55, primals_37);  mul_55 = None
    add_46: "f32[1568, 128]" = torch.ops.aten.add.Tensor(mul_61, primals_38);  mul_61 = primals_38 = None
    view_24: "f32[8, 196, 128]" = torch.ops.aten.view.default(add_46, [8, 196, 128]);  add_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_47: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_35, view_24);  add_35 = view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_9: "f32[128, 256]" = torch.ops.aten.permute.default(primals_39, [1, 0]);  primals_39 = None
    clone_7: "f32[8, 196, 128]" = torch.ops.aten.clone.default(add_47, memory_format = torch.contiguous_format)
    view_25: "f32[1568, 128]" = torch.ops.aten.view.default(clone_7, [1568, 128]);  clone_7 = None
    mm_4: "f32[1568, 256]" = torch.ops.aten.mm.default(view_25, permute_9)
    view_26: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_4, [8, 196, 256]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_27: "f32[1568, 256]" = torch.ops.aten.view.default(view_26, [1568, 256]);  view_26 = None
    add_48: "i64[]" = torch.ops.aten.add.Tensor(primals_249, 1)
    var_mean_8 = torch.ops.aten.var_mean.correction(view_27, [0], correction = 0, keepdim = True)
    getitem_19: "f32[1, 256]" = var_mean_8[0]
    getitem_20: "f32[1, 256]" = var_mean_8[1];  var_mean_8 = None
    add_49: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_19, 1e-05)
    rsqrt_8: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_9: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_27, getitem_20)
    mul_62: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_8);  sub_9 = None
    squeeze_24: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_20, [0]);  getitem_20 = None
    squeeze_25: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0]);  rsqrt_8 = None
    mul_63: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_64: "f32[256]" = torch.ops.aten.mul.Tensor(primals_247, 0.9)
    add_50: "f32[256]" = torch.ops.aten.add.Tensor(mul_63, mul_64);  mul_63 = mul_64 = None
    squeeze_26: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_19, [0]);  getitem_19 = None
    mul_65: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0006381620931717);  squeeze_26 = None
    mul_66: "f32[256]" = torch.ops.aten.mul.Tensor(mul_65, 0.1);  mul_65 = None
    mul_67: "f32[256]" = torch.ops.aten.mul.Tensor(primals_248, 0.9)
    add_51: "f32[256]" = torch.ops.aten.add.Tensor(mul_66, mul_67);  mul_66 = mul_67 = None
    mul_68: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(mul_62, primals_40);  mul_62 = None
    add_52: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mul_68, primals_41);  mul_68 = primals_41 = None
    view_28: "f32[8, 196, 256]" = torch.ops.aten.view.default(add_52, [8, 196, 256]);  add_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_29: "f32[8, 196, 4, 64]" = torch.ops.aten.view.default(view_28, [8, 196, 4, -1]);  view_28 = None
    split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(view_29, [16, 16, 32], 3);  view_29 = None
    getitem_21: "f32[8, 196, 4, 16]" = split_with_sizes_1[0]
    getitem_22: "f32[8, 196, 4, 16]" = split_with_sizes_1[1]
    getitem_23: "f32[8, 196, 4, 32]" = split_with_sizes_1[2];  split_with_sizes_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_10: "f32[8, 4, 196, 16]" = torch.ops.aten.permute.default(getitem_21, [0, 2, 1, 3]);  getitem_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_11: "f32[8, 4, 16, 196]" = torch.ops.aten.permute.default(getitem_22, [0, 2, 3, 1]);  getitem_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_12: "f32[8, 4, 196, 32]" = torch.ops.aten.permute.default(getitem_23, [0, 2, 1, 3]);  getitem_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_4: "f32[8, 4, 196, 16]" = torch.ops.aten.expand.default(permute_10, [8, 4, 196, 16]);  permute_10 = None
    clone_8: "f32[8, 4, 196, 16]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
    view_30: "f32[32, 196, 16]" = torch.ops.aten.view.default(clone_8, [32, 196, 16]);  clone_8 = None
    expand_5: "f32[8, 4, 16, 196]" = torch.ops.aten.expand.default(permute_11, [8, 4, 16, 196]);  permute_11 = None
    clone_9: "f32[8, 4, 16, 196]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_31: "f32[32, 16, 196]" = torch.ops.aten.view.default(clone_9, [32, 16, 196]);  clone_9 = None
    bmm_2: "f32[32, 196, 196]" = torch.ops.aten.bmm.default(view_30, view_31)
    view_32: "f32[8, 4, 196, 196]" = torch.ops.aten.view.default(bmm_2, [8, 4, 196, 196]);  bmm_2 = None
    mul_69: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(view_32, 0.25);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    slice_2: "f32[4, 196]" = torch.ops.aten.slice.Tensor(primals_2, 0, 0, 9223372036854775807);  primals_2 = None
    index_1: "f32[4, 196, 196]" = torch.ops.aten.index.Tensor(slice_2, [None, primals_210]);  slice_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_53: "f32[8, 4, 196, 196]" = torch.ops.aten.add.Tensor(mul_69, index_1);  mul_69 = index_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax_1: "f32[8, 4, 196, 1]" = torch.ops.aten.amax.default(add_53, [-1], True)
    sub_10: "f32[8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(add_53, amax_1);  add_53 = amax_1 = None
    exp_1: "f32[8, 4, 196, 196]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_2: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_6: "f32[8, 4, 196, 196]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_1: "f32[8, 4, 196, 196]" = torch.ops.aten.alias.default(div_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_6: "f32[8, 4, 196, 196]" = torch.ops.aten.expand.default(div_6, [8, 4, 196, 196]);  div_6 = None
    view_33: "f32[32, 196, 196]" = torch.ops.aten.view.default(expand_6, [32, 196, 196]);  expand_6 = None
    expand_7: "f32[8, 4, 196, 32]" = torch.ops.aten.expand.default(permute_12, [8, 4, 196, 32]);  permute_12 = None
    clone_10: "f32[8, 4, 196, 32]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
    view_34: "f32[32, 196, 32]" = torch.ops.aten.view.default(clone_10, [32, 196, 32]);  clone_10 = None
    bmm_3: "f32[32, 196, 32]" = torch.ops.aten.bmm.default(view_33, view_34)
    view_35: "f32[8, 4, 196, 32]" = torch.ops.aten.view.default(bmm_3, [8, 4, 196, 32]);  bmm_3 = None
    permute_13: "f32[8, 196, 4, 32]" = torch.ops.aten.permute.default(view_35, [0, 2, 1, 3]);  view_35 = None
    clone_11: "f32[8, 196, 4, 32]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    view_36: "f32[8, 196, 128]" = torch.ops.aten.view.default(clone_11, [8, 196, 128]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_54: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(view_36, 3)
    clamp_min_5: "f32[8, 196, 128]" = torch.ops.aten.clamp_min.default(add_54, 0);  add_54 = None
    clamp_max_5: "f32[8, 196, 128]" = torch.ops.aten.clamp_max.default(clamp_min_5, 6);  clamp_min_5 = None
    mul_70: "f32[8, 196, 128]" = torch.ops.aten.mul.Tensor(view_36, clamp_max_5);  clamp_max_5 = None
    div_7: "f32[8, 196, 128]" = torch.ops.aten.div.Tensor(mul_70, 6);  mul_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_14: "f32[128, 128]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
    view_37: "f32[1568, 128]" = torch.ops.aten.view.default(div_7, [1568, 128]);  div_7 = None
    mm_5: "f32[1568, 128]" = torch.ops.aten.mm.default(view_37, permute_14)
    view_38: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_5, [8, 196, 128]);  mm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_39: "f32[1568, 128]" = torch.ops.aten.view.default(view_38, [1568, 128]);  view_38 = None
    add_55: "i64[]" = torch.ops.aten.add.Tensor(primals_252, 1)
    var_mean_9 = torch.ops.aten.var_mean.correction(view_39, [0], correction = 0, keepdim = True)
    getitem_24: "f32[1, 128]" = var_mean_9[0]
    getitem_25: "f32[1, 128]" = var_mean_9[1];  var_mean_9 = None
    add_56: "f32[1, 128]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
    rsqrt_9: "f32[1, 128]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_11: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_39, getitem_25)
    mul_71: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_9);  sub_11 = None
    squeeze_27: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_25, [0]);  getitem_25 = None
    squeeze_28: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0]);  rsqrt_9 = None
    mul_72: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_73: "f32[128]" = torch.ops.aten.mul.Tensor(primals_250, 0.9)
    add_57: "f32[128]" = torch.ops.aten.add.Tensor(mul_72, mul_73);  mul_72 = mul_73 = None
    squeeze_29: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_24, [0]);  getitem_24 = None
    mul_74: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0006381620931717);  squeeze_29 = None
    mul_75: "f32[128]" = torch.ops.aten.mul.Tensor(mul_74, 0.1);  mul_74 = None
    mul_76: "f32[128]" = torch.ops.aten.mul.Tensor(primals_251, 0.9)
    add_58: "f32[128]" = torch.ops.aten.add.Tensor(mul_75, mul_76);  mul_75 = mul_76 = None
    mul_77: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(mul_71, primals_43);  mul_71 = None
    add_59: "f32[1568, 128]" = torch.ops.aten.add.Tensor(mul_77, primals_44);  mul_77 = primals_44 = None
    view_40: "f32[8, 196, 128]" = torch.ops.aten.view.default(add_59, [8, 196, 128]);  add_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_60: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_47, view_40);  add_47 = view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_15: "f32[128, 256]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
    clone_12: "f32[8, 196, 128]" = torch.ops.aten.clone.default(add_60, memory_format = torch.contiguous_format)
    view_41: "f32[1568, 128]" = torch.ops.aten.view.default(clone_12, [1568, 128]);  clone_12 = None
    mm_6: "f32[1568, 256]" = torch.ops.aten.mm.default(view_41, permute_15)
    view_42: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_6, [8, 196, 256]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_43: "f32[1568, 256]" = torch.ops.aten.view.default(view_42, [1568, 256]);  view_42 = None
    add_61: "i64[]" = torch.ops.aten.add.Tensor(primals_255, 1)
    var_mean_10 = torch.ops.aten.var_mean.correction(view_43, [0], correction = 0, keepdim = True)
    getitem_26: "f32[1, 256]" = var_mean_10[0]
    getitem_27: "f32[1, 256]" = var_mean_10[1];  var_mean_10 = None
    add_62: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
    rsqrt_10: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    sub_12: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_43, getitem_27)
    mul_78: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_10);  sub_12 = None
    squeeze_30: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_27, [0]);  getitem_27 = None
    squeeze_31: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0]);  rsqrt_10 = None
    mul_79: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_80: "f32[256]" = torch.ops.aten.mul.Tensor(primals_253, 0.9)
    add_63: "f32[256]" = torch.ops.aten.add.Tensor(mul_79, mul_80);  mul_79 = mul_80 = None
    squeeze_32: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_26, [0]);  getitem_26 = None
    mul_81: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0006381620931717);  squeeze_32 = None
    mul_82: "f32[256]" = torch.ops.aten.mul.Tensor(mul_81, 0.1);  mul_81 = None
    mul_83: "f32[256]" = torch.ops.aten.mul.Tensor(primals_254, 0.9)
    add_64: "f32[256]" = torch.ops.aten.add.Tensor(mul_82, mul_83);  mul_82 = mul_83 = None
    mul_84: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(mul_78, primals_46);  mul_78 = None
    add_65: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mul_84, primals_47);  mul_84 = primals_47 = None
    view_44: "f32[8, 196, 256]" = torch.ops.aten.view.default(add_65, [8, 196, 256]);  add_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_66: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(view_44, 3)
    clamp_min_6: "f32[8, 196, 256]" = torch.ops.aten.clamp_min.default(add_66, 0);  add_66 = None
    clamp_max_6: "f32[8, 196, 256]" = torch.ops.aten.clamp_max.default(clamp_min_6, 6);  clamp_min_6 = None
    mul_85: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_44, clamp_max_6);  clamp_max_6 = None
    div_8: "f32[8, 196, 256]" = torch.ops.aten.div.Tensor(mul_85, 6);  mul_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    clone_13: "f32[8, 196, 256]" = torch.ops.aten.clone.default(div_8);  div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_16: "f32[256, 128]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
    view_45: "f32[1568, 256]" = torch.ops.aten.view.default(clone_13, [1568, 256]);  clone_13 = None
    mm_7: "f32[1568, 128]" = torch.ops.aten.mm.default(view_45, permute_16)
    view_46: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_7, [8, 196, 128]);  mm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_47: "f32[1568, 128]" = torch.ops.aten.view.default(view_46, [1568, 128]);  view_46 = None
    add_67: "i64[]" = torch.ops.aten.add.Tensor(primals_258, 1)
    var_mean_11 = torch.ops.aten.var_mean.correction(view_47, [0], correction = 0, keepdim = True)
    getitem_28: "f32[1, 128]" = var_mean_11[0]
    getitem_29: "f32[1, 128]" = var_mean_11[1];  var_mean_11 = None
    add_68: "f32[1, 128]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
    rsqrt_11: "f32[1, 128]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    sub_13: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_47, getitem_29)
    mul_86: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_11);  sub_13 = None
    squeeze_33: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_29, [0]);  getitem_29 = None
    squeeze_34: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0]);  rsqrt_11 = None
    mul_87: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_88: "f32[128]" = torch.ops.aten.mul.Tensor(primals_256, 0.9)
    add_69: "f32[128]" = torch.ops.aten.add.Tensor(mul_87, mul_88);  mul_87 = mul_88 = None
    squeeze_35: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_28, [0]);  getitem_28 = None
    mul_89: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0006381620931717);  squeeze_35 = None
    mul_90: "f32[128]" = torch.ops.aten.mul.Tensor(mul_89, 0.1);  mul_89 = None
    mul_91: "f32[128]" = torch.ops.aten.mul.Tensor(primals_257, 0.9)
    add_70: "f32[128]" = torch.ops.aten.add.Tensor(mul_90, mul_91);  mul_90 = mul_91 = None
    mul_92: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(mul_86, primals_49);  mul_86 = None
    add_71: "f32[1568, 128]" = torch.ops.aten.add.Tensor(mul_92, primals_50);  mul_92 = primals_50 = None
    view_48: "f32[8, 196, 128]" = torch.ops.aten.view.default(add_71, [8, 196, 128]);  add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_72: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_60, view_48);  add_60 = view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_17: "f32[128, 256]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
    clone_14: "f32[8, 196, 128]" = torch.ops.aten.clone.default(add_72, memory_format = torch.contiguous_format)
    view_49: "f32[1568, 128]" = torch.ops.aten.view.default(clone_14, [1568, 128]);  clone_14 = None
    mm_8: "f32[1568, 256]" = torch.ops.aten.mm.default(view_49, permute_17)
    view_50: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_8, [8, 196, 256]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_51: "f32[1568, 256]" = torch.ops.aten.view.default(view_50, [1568, 256]);  view_50 = None
    add_73: "i64[]" = torch.ops.aten.add.Tensor(primals_261, 1)
    var_mean_12 = torch.ops.aten.var_mean.correction(view_51, [0], correction = 0, keepdim = True)
    getitem_30: "f32[1, 256]" = var_mean_12[0]
    getitem_31: "f32[1, 256]" = var_mean_12[1];  var_mean_12 = None
    add_74: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
    rsqrt_12: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_14: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_51, getitem_31)
    mul_93: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_12);  sub_14 = None
    squeeze_36: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_31, [0]);  getitem_31 = None
    squeeze_37: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0]);  rsqrt_12 = None
    mul_94: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_95: "f32[256]" = torch.ops.aten.mul.Tensor(primals_259, 0.9)
    add_75: "f32[256]" = torch.ops.aten.add.Tensor(mul_94, mul_95);  mul_94 = mul_95 = None
    squeeze_38: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_30, [0]);  getitem_30 = None
    mul_96: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0006381620931717);  squeeze_38 = None
    mul_97: "f32[256]" = torch.ops.aten.mul.Tensor(mul_96, 0.1);  mul_96 = None
    mul_98: "f32[256]" = torch.ops.aten.mul.Tensor(primals_260, 0.9)
    add_76: "f32[256]" = torch.ops.aten.add.Tensor(mul_97, mul_98);  mul_97 = mul_98 = None
    mul_99: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(mul_93, primals_52);  mul_93 = None
    add_77: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mul_99, primals_53);  mul_99 = primals_53 = None
    view_52: "f32[8, 196, 256]" = torch.ops.aten.view.default(add_77, [8, 196, 256]);  add_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_53: "f32[8, 196, 4, 64]" = torch.ops.aten.view.default(view_52, [8, 196, 4, -1]);  view_52 = None
    split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(view_53, [16, 16, 32], 3);  view_53 = None
    getitem_32: "f32[8, 196, 4, 16]" = split_with_sizes_2[0]
    getitem_33: "f32[8, 196, 4, 16]" = split_with_sizes_2[1]
    getitem_34: "f32[8, 196, 4, 32]" = split_with_sizes_2[2];  split_with_sizes_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_18: "f32[8, 4, 196, 16]" = torch.ops.aten.permute.default(getitem_32, [0, 2, 1, 3]);  getitem_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_19: "f32[8, 4, 16, 196]" = torch.ops.aten.permute.default(getitem_33, [0, 2, 3, 1]);  getitem_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_20: "f32[8, 4, 196, 32]" = torch.ops.aten.permute.default(getitem_34, [0, 2, 1, 3]);  getitem_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_8: "f32[8, 4, 196, 16]" = torch.ops.aten.expand.default(permute_18, [8, 4, 196, 16]);  permute_18 = None
    clone_15: "f32[8, 4, 196, 16]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    view_54: "f32[32, 196, 16]" = torch.ops.aten.view.default(clone_15, [32, 196, 16]);  clone_15 = None
    expand_9: "f32[8, 4, 16, 196]" = torch.ops.aten.expand.default(permute_19, [8, 4, 16, 196]);  permute_19 = None
    clone_16: "f32[8, 4, 16, 196]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_55: "f32[32, 16, 196]" = torch.ops.aten.view.default(clone_16, [32, 16, 196]);  clone_16 = None
    bmm_4: "f32[32, 196, 196]" = torch.ops.aten.bmm.default(view_54, view_55)
    view_56: "f32[8, 4, 196, 196]" = torch.ops.aten.view.default(bmm_4, [8, 4, 196, 196]);  bmm_4 = None
    mul_100: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(view_56, 0.25);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    slice_3: "f32[4, 196]" = torch.ops.aten.slice.Tensor(primals_3, 0, 0, 9223372036854775807);  primals_3 = None
    index_2: "f32[4, 196, 196]" = torch.ops.aten.index.Tensor(slice_3, [None, primals_211]);  slice_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_78: "f32[8, 4, 196, 196]" = torch.ops.aten.add.Tensor(mul_100, index_2);  mul_100 = index_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax_2: "f32[8, 4, 196, 1]" = torch.ops.aten.amax.default(add_78, [-1], True)
    sub_15: "f32[8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(add_78, amax_2);  add_78 = amax_2 = None
    exp_2: "f32[8, 4, 196, 196]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
    sum_3: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_9: "f32[8, 4, 196, 196]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_2: "f32[8, 4, 196, 196]" = torch.ops.aten.alias.default(div_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_10: "f32[8, 4, 196, 196]" = torch.ops.aten.expand.default(div_9, [8, 4, 196, 196]);  div_9 = None
    view_57: "f32[32, 196, 196]" = torch.ops.aten.view.default(expand_10, [32, 196, 196]);  expand_10 = None
    expand_11: "f32[8, 4, 196, 32]" = torch.ops.aten.expand.default(permute_20, [8, 4, 196, 32]);  permute_20 = None
    clone_17: "f32[8, 4, 196, 32]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
    view_58: "f32[32, 196, 32]" = torch.ops.aten.view.default(clone_17, [32, 196, 32]);  clone_17 = None
    bmm_5: "f32[32, 196, 32]" = torch.ops.aten.bmm.default(view_57, view_58)
    view_59: "f32[8, 4, 196, 32]" = torch.ops.aten.view.default(bmm_5, [8, 4, 196, 32]);  bmm_5 = None
    permute_21: "f32[8, 196, 4, 32]" = torch.ops.aten.permute.default(view_59, [0, 2, 1, 3]);  view_59 = None
    clone_18: "f32[8, 196, 4, 32]" = torch.ops.aten.clone.default(permute_21, memory_format = torch.contiguous_format);  permute_21 = None
    view_60: "f32[8, 196, 128]" = torch.ops.aten.view.default(clone_18, [8, 196, 128]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_79: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(view_60, 3)
    clamp_min_7: "f32[8, 196, 128]" = torch.ops.aten.clamp_min.default(add_79, 0);  add_79 = None
    clamp_max_7: "f32[8, 196, 128]" = torch.ops.aten.clamp_max.default(clamp_min_7, 6);  clamp_min_7 = None
    mul_101: "f32[8, 196, 128]" = torch.ops.aten.mul.Tensor(view_60, clamp_max_7);  clamp_max_7 = None
    div_10: "f32[8, 196, 128]" = torch.ops.aten.div.Tensor(mul_101, 6);  mul_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_22: "f32[128, 128]" = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
    view_61: "f32[1568, 128]" = torch.ops.aten.view.default(div_10, [1568, 128]);  div_10 = None
    mm_9: "f32[1568, 128]" = torch.ops.aten.mm.default(view_61, permute_22)
    view_62: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_9, [8, 196, 128]);  mm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_63: "f32[1568, 128]" = torch.ops.aten.view.default(view_62, [1568, 128]);  view_62 = None
    add_80: "i64[]" = torch.ops.aten.add.Tensor(primals_264, 1)
    var_mean_13 = torch.ops.aten.var_mean.correction(view_63, [0], correction = 0, keepdim = True)
    getitem_35: "f32[1, 128]" = var_mean_13[0]
    getitem_36: "f32[1, 128]" = var_mean_13[1];  var_mean_13 = None
    add_81: "f32[1, 128]" = torch.ops.aten.add.Tensor(getitem_35, 1e-05)
    rsqrt_13: "f32[1, 128]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    sub_16: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_63, getitem_36)
    mul_102: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_13);  sub_16 = None
    squeeze_39: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_36, [0]);  getitem_36 = None
    squeeze_40: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0]);  rsqrt_13 = None
    mul_103: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_104: "f32[128]" = torch.ops.aten.mul.Tensor(primals_262, 0.9)
    add_82: "f32[128]" = torch.ops.aten.add.Tensor(mul_103, mul_104);  mul_103 = mul_104 = None
    squeeze_41: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_35, [0]);  getitem_35 = None
    mul_105: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0006381620931717);  squeeze_41 = None
    mul_106: "f32[128]" = torch.ops.aten.mul.Tensor(mul_105, 0.1);  mul_105 = None
    mul_107: "f32[128]" = torch.ops.aten.mul.Tensor(primals_263, 0.9)
    add_83: "f32[128]" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
    mul_108: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(mul_102, primals_55);  mul_102 = None
    add_84: "f32[1568, 128]" = torch.ops.aten.add.Tensor(mul_108, primals_56);  mul_108 = primals_56 = None
    view_64: "f32[8, 196, 128]" = torch.ops.aten.view.default(add_84, [8, 196, 128]);  add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_85: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_72, view_64);  add_72 = view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_23: "f32[128, 256]" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
    clone_19: "f32[8, 196, 128]" = torch.ops.aten.clone.default(add_85, memory_format = torch.contiguous_format)
    view_65: "f32[1568, 128]" = torch.ops.aten.view.default(clone_19, [1568, 128]);  clone_19 = None
    mm_10: "f32[1568, 256]" = torch.ops.aten.mm.default(view_65, permute_23)
    view_66: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_10, [8, 196, 256]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_67: "f32[1568, 256]" = torch.ops.aten.view.default(view_66, [1568, 256]);  view_66 = None
    add_86: "i64[]" = torch.ops.aten.add.Tensor(primals_267, 1)
    var_mean_14 = torch.ops.aten.var_mean.correction(view_67, [0], correction = 0, keepdim = True)
    getitem_37: "f32[1, 256]" = var_mean_14[0]
    getitem_38: "f32[1, 256]" = var_mean_14[1];  var_mean_14 = None
    add_87: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_37, 1e-05)
    rsqrt_14: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    sub_17: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_67, getitem_38)
    mul_109: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_14);  sub_17 = None
    squeeze_42: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_38, [0]);  getitem_38 = None
    squeeze_43: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0]);  rsqrt_14 = None
    mul_110: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_111: "f32[256]" = torch.ops.aten.mul.Tensor(primals_265, 0.9)
    add_88: "f32[256]" = torch.ops.aten.add.Tensor(mul_110, mul_111);  mul_110 = mul_111 = None
    squeeze_44: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_37, [0]);  getitem_37 = None
    mul_112: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0006381620931717);  squeeze_44 = None
    mul_113: "f32[256]" = torch.ops.aten.mul.Tensor(mul_112, 0.1);  mul_112 = None
    mul_114: "f32[256]" = torch.ops.aten.mul.Tensor(primals_266, 0.9)
    add_89: "f32[256]" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
    mul_115: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(mul_109, primals_58);  mul_109 = None
    add_90: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mul_115, primals_59);  mul_115 = primals_59 = None
    view_68: "f32[8, 196, 256]" = torch.ops.aten.view.default(add_90, [8, 196, 256]);  add_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_91: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(view_68, 3)
    clamp_min_8: "f32[8, 196, 256]" = torch.ops.aten.clamp_min.default(add_91, 0);  add_91 = None
    clamp_max_8: "f32[8, 196, 256]" = torch.ops.aten.clamp_max.default(clamp_min_8, 6);  clamp_min_8 = None
    mul_116: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_68, clamp_max_8);  clamp_max_8 = None
    div_11: "f32[8, 196, 256]" = torch.ops.aten.div.Tensor(mul_116, 6);  mul_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    clone_20: "f32[8, 196, 256]" = torch.ops.aten.clone.default(div_11);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_24: "f32[256, 128]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
    view_69: "f32[1568, 256]" = torch.ops.aten.view.default(clone_20, [1568, 256]);  clone_20 = None
    mm_11: "f32[1568, 128]" = torch.ops.aten.mm.default(view_69, permute_24)
    view_70: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_11, [8, 196, 128]);  mm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_71: "f32[1568, 128]" = torch.ops.aten.view.default(view_70, [1568, 128]);  view_70 = None
    add_92: "i64[]" = torch.ops.aten.add.Tensor(primals_270, 1)
    var_mean_15 = torch.ops.aten.var_mean.correction(view_71, [0], correction = 0, keepdim = True)
    getitem_39: "f32[1, 128]" = var_mean_15[0]
    getitem_40: "f32[1, 128]" = var_mean_15[1];  var_mean_15 = None
    add_93: "f32[1, 128]" = torch.ops.aten.add.Tensor(getitem_39, 1e-05)
    rsqrt_15: "f32[1, 128]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
    sub_18: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_71, getitem_40)
    mul_117: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_15);  sub_18 = None
    squeeze_45: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_40, [0]);  getitem_40 = None
    squeeze_46: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0]);  rsqrt_15 = None
    mul_118: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_119: "f32[128]" = torch.ops.aten.mul.Tensor(primals_268, 0.9)
    add_94: "f32[128]" = torch.ops.aten.add.Tensor(mul_118, mul_119);  mul_118 = mul_119 = None
    squeeze_47: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_39, [0]);  getitem_39 = None
    mul_120: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0006381620931717);  squeeze_47 = None
    mul_121: "f32[128]" = torch.ops.aten.mul.Tensor(mul_120, 0.1);  mul_120 = None
    mul_122: "f32[128]" = torch.ops.aten.mul.Tensor(primals_269, 0.9)
    add_95: "f32[128]" = torch.ops.aten.add.Tensor(mul_121, mul_122);  mul_121 = mul_122 = None
    mul_123: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(mul_117, primals_61);  mul_117 = None
    add_96: "f32[1568, 128]" = torch.ops.aten.add.Tensor(mul_123, primals_62);  mul_123 = primals_62 = None
    view_72: "f32[8, 196, 128]" = torch.ops.aten.view.default(add_96, [8, 196, 128]);  add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_97: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_85, view_72);  add_85 = view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_25: "f32[128, 256]" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
    clone_21: "f32[8, 196, 128]" = torch.ops.aten.clone.default(add_97, memory_format = torch.contiguous_format)
    view_73: "f32[1568, 128]" = torch.ops.aten.view.default(clone_21, [1568, 128]);  clone_21 = None
    mm_12: "f32[1568, 256]" = torch.ops.aten.mm.default(view_73, permute_25)
    view_74: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_12, [8, 196, 256]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_75: "f32[1568, 256]" = torch.ops.aten.view.default(view_74, [1568, 256]);  view_74 = None
    add_98: "i64[]" = torch.ops.aten.add.Tensor(primals_273, 1)
    var_mean_16 = torch.ops.aten.var_mean.correction(view_75, [0], correction = 0, keepdim = True)
    getitem_41: "f32[1, 256]" = var_mean_16[0]
    getitem_42: "f32[1, 256]" = var_mean_16[1];  var_mean_16 = None
    add_99: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_41, 1e-05)
    rsqrt_16: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
    sub_19: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_75, getitem_42)
    mul_124: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_16);  sub_19 = None
    squeeze_48: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_42, [0]);  getitem_42 = None
    squeeze_49: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0]);  rsqrt_16 = None
    mul_125: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_126: "f32[256]" = torch.ops.aten.mul.Tensor(primals_271, 0.9)
    add_100: "f32[256]" = torch.ops.aten.add.Tensor(mul_125, mul_126);  mul_125 = mul_126 = None
    squeeze_50: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_41, [0]);  getitem_41 = None
    mul_127: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0006381620931717);  squeeze_50 = None
    mul_128: "f32[256]" = torch.ops.aten.mul.Tensor(mul_127, 0.1);  mul_127 = None
    mul_129: "f32[256]" = torch.ops.aten.mul.Tensor(primals_272, 0.9)
    add_101: "f32[256]" = torch.ops.aten.add.Tensor(mul_128, mul_129);  mul_128 = mul_129 = None
    mul_130: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(mul_124, primals_64);  mul_124 = None
    add_102: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mul_130, primals_65);  mul_130 = primals_65 = None
    view_76: "f32[8, 196, 256]" = torch.ops.aten.view.default(add_102, [8, 196, 256]);  add_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_77: "f32[8, 196, 4, 64]" = torch.ops.aten.view.default(view_76, [8, 196, 4, -1]);  view_76 = None
    split_with_sizes_3 = torch.ops.aten.split_with_sizes.default(view_77, [16, 16, 32], 3);  view_77 = None
    getitem_43: "f32[8, 196, 4, 16]" = split_with_sizes_3[0]
    getitem_44: "f32[8, 196, 4, 16]" = split_with_sizes_3[1]
    getitem_45: "f32[8, 196, 4, 32]" = split_with_sizes_3[2];  split_with_sizes_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_26: "f32[8, 4, 196, 16]" = torch.ops.aten.permute.default(getitem_43, [0, 2, 1, 3]);  getitem_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_27: "f32[8, 4, 16, 196]" = torch.ops.aten.permute.default(getitem_44, [0, 2, 3, 1]);  getitem_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_28: "f32[8, 4, 196, 32]" = torch.ops.aten.permute.default(getitem_45, [0, 2, 1, 3]);  getitem_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_12: "f32[8, 4, 196, 16]" = torch.ops.aten.expand.default(permute_26, [8, 4, 196, 16]);  permute_26 = None
    clone_22: "f32[8, 4, 196, 16]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
    view_78: "f32[32, 196, 16]" = torch.ops.aten.view.default(clone_22, [32, 196, 16]);  clone_22 = None
    expand_13: "f32[8, 4, 16, 196]" = torch.ops.aten.expand.default(permute_27, [8, 4, 16, 196]);  permute_27 = None
    clone_23: "f32[8, 4, 16, 196]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
    view_79: "f32[32, 16, 196]" = torch.ops.aten.view.default(clone_23, [32, 16, 196]);  clone_23 = None
    bmm_6: "f32[32, 196, 196]" = torch.ops.aten.bmm.default(view_78, view_79)
    view_80: "f32[8, 4, 196, 196]" = torch.ops.aten.view.default(bmm_6, [8, 4, 196, 196]);  bmm_6 = None
    mul_131: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(view_80, 0.25);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    slice_4: "f32[4, 196]" = torch.ops.aten.slice.Tensor(primals_4, 0, 0, 9223372036854775807);  primals_4 = None
    index_3: "f32[4, 196, 196]" = torch.ops.aten.index.Tensor(slice_4, [None, primals_212]);  slice_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_103: "f32[8, 4, 196, 196]" = torch.ops.aten.add.Tensor(mul_131, index_3);  mul_131 = index_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax_3: "f32[8, 4, 196, 1]" = torch.ops.aten.amax.default(add_103, [-1], True)
    sub_20: "f32[8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(add_103, amax_3);  add_103 = amax_3 = None
    exp_3: "f32[8, 4, 196, 196]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
    sum_4: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_12: "f32[8, 4, 196, 196]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_3: "f32[8, 4, 196, 196]" = torch.ops.aten.alias.default(div_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_14: "f32[8, 4, 196, 196]" = torch.ops.aten.expand.default(div_12, [8, 4, 196, 196]);  div_12 = None
    view_81: "f32[32, 196, 196]" = torch.ops.aten.view.default(expand_14, [32, 196, 196]);  expand_14 = None
    expand_15: "f32[8, 4, 196, 32]" = torch.ops.aten.expand.default(permute_28, [8, 4, 196, 32]);  permute_28 = None
    clone_24: "f32[8, 4, 196, 32]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
    view_82: "f32[32, 196, 32]" = torch.ops.aten.view.default(clone_24, [32, 196, 32]);  clone_24 = None
    bmm_7: "f32[32, 196, 32]" = torch.ops.aten.bmm.default(view_81, view_82)
    view_83: "f32[8, 4, 196, 32]" = torch.ops.aten.view.default(bmm_7, [8, 4, 196, 32]);  bmm_7 = None
    permute_29: "f32[8, 196, 4, 32]" = torch.ops.aten.permute.default(view_83, [0, 2, 1, 3]);  view_83 = None
    clone_25: "f32[8, 196, 4, 32]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_84: "f32[8, 196, 128]" = torch.ops.aten.view.default(clone_25, [8, 196, 128]);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_104: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(view_84, 3)
    clamp_min_9: "f32[8, 196, 128]" = torch.ops.aten.clamp_min.default(add_104, 0);  add_104 = None
    clamp_max_9: "f32[8, 196, 128]" = torch.ops.aten.clamp_max.default(clamp_min_9, 6);  clamp_min_9 = None
    mul_132: "f32[8, 196, 128]" = torch.ops.aten.mul.Tensor(view_84, clamp_max_9);  clamp_max_9 = None
    div_13: "f32[8, 196, 128]" = torch.ops.aten.div.Tensor(mul_132, 6);  mul_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_30: "f32[128, 128]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
    view_85: "f32[1568, 128]" = torch.ops.aten.view.default(div_13, [1568, 128]);  div_13 = None
    mm_13: "f32[1568, 128]" = torch.ops.aten.mm.default(view_85, permute_30)
    view_86: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_13, [8, 196, 128]);  mm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_87: "f32[1568, 128]" = torch.ops.aten.view.default(view_86, [1568, 128]);  view_86 = None
    add_105: "i64[]" = torch.ops.aten.add.Tensor(primals_276, 1)
    var_mean_17 = torch.ops.aten.var_mean.correction(view_87, [0], correction = 0, keepdim = True)
    getitem_46: "f32[1, 128]" = var_mean_17[0]
    getitem_47: "f32[1, 128]" = var_mean_17[1];  var_mean_17 = None
    add_106: "f32[1, 128]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
    rsqrt_17: "f32[1, 128]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
    sub_21: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_87, getitem_47)
    mul_133: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_17);  sub_21 = None
    squeeze_51: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_47, [0]);  getitem_47 = None
    squeeze_52: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0]);  rsqrt_17 = None
    mul_134: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_135: "f32[128]" = torch.ops.aten.mul.Tensor(primals_274, 0.9)
    add_107: "f32[128]" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
    squeeze_53: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_46, [0]);  getitem_46 = None
    mul_136: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0006381620931717);  squeeze_53 = None
    mul_137: "f32[128]" = torch.ops.aten.mul.Tensor(mul_136, 0.1);  mul_136 = None
    mul_138: "f32[128]" = torch.ops.aten.mul.Tensor(primals_275, 0.9)
    add_108: "f32[128]" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
    mul_139: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(mul_133, primals_67);  mul_133 = None
    add_109: "f32[1568, 128]" = torch.ops.aten.add.Tensor(mul_139, primals_68);  mul_139 = primals_68 = None
    view_88: "f32[8, 196, 128]" = torch.ops.aten.view.default(add_109, [8, 196, 128]);  add_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_110: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_97, view_88);  add_97 = view_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_31: "f32[128, 256]" = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
    clone_26: "f32[8, 196, 128]" = torch.ops.aten.clone.default(add_110, memory_format = torch.contiguous_format)
    view_89: "f32[1568, 128]" = torch.ops.aten.view.default(clone_26, [1568, 128]);  clone_26 = None
    mm_14: "f32[1568, 256]" = torch.ops.aten.mm.default(view_89, permute_31)
    view_90: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_14, [8, 196, 256]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_91: "f32[1568, 256]" = torch.ops.aten.view.default(view_90, [1568, 256]);  view_90 = None
    add_111: "i64[]" = torch.ops.aten.add.Tensor(primals_279, 1)
    var_mean_18 = torch.ops.aten.var_mean.correction(view_91, [0], correction = 0, keepdim = True)
    getitem_48: "f32[1, 256]" = var_mean_18[0]
    getitem_49: "f32[1, 256]" = var_mean_18[1];  var_mean_18 = None
    add_112: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05)
    rsqrt_18: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
    sub_22: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_91, getitem_49)
    mul_140: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_18);  sub_22 = None
    squeeze_54: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_49, [0]);  getitem_49 = None
    squeeze_55: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0]);  rsqrt_18 = None
    mul_141: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_142: "f32[256]" = torch.ops.aten.mul.Tensor(primals_277, 0.9)
    add_113: "f32[256]" = torch.ops.aten.add.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
    squeeze_56: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_48, [0]);  getitem_48 = None
    mul_143: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0006381620931717);  squeeze_56 = None
    mul_144: "f32[256]" = torch.ops.aten.mul.Tensor(mul_143, 0.1);  mul_143 = None
    mul_145: "f32[256]" = torch.ops.aten.mul.Tensor(primals_278, 0.9)
    add_114: "f32[256]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
    mul_146: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(mul_140, primals_70);  mul_140 = None
    add_115: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mul_146, primals_71);  mul_146 = primals_71 = None
    view_92: "f32[8, 196, 256]" = torch.ops.aten.view.default(add_115, [8, 196, 256]);  add_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_116: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(view_92, 3)
    clamp_min_10: "f32[8, 196, 256]" = torch.ops.aten.clamp_min.default(add_116, 0);  add_116 = None
    clamp_max_10: "f32[8, 196, 256]" = torch.ops.aten.clamp_max.default(clamp_min_10, 6);  clamp_min_10 = None
    mul_147: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_92, clamp_max_10);  clamp_max_10 = None
    div_14: "f32[8, 196, 256]" = torch.ops.aten.div.Tensor(mul_147, 6);  mul_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    clone_27: "f32[8, 196, 256]" = torch.ops.aten.clone.default(div_14);  div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_32: "f32[256, 128]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
    view_93: "f32[1568, 256]" = torch.ops.aten.view.default(clone_27, [1568, 256]);  clone_27 = None
    mm_15: "f32[1568, 128]" = torch.ops.aten.mm.default(view_93, permute_32)
    view_94: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_15, [8, 196, 128]);  mm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_95: "f32[1568, 128]" = torch.ops.aten.view.default(view_94, [1568, 128]);  view_94 = None
    add_117: "i64[]" = torch.ops.aten.add.Tensor(primals_282, 1)
    var_mean_19 = torch.ops.aten.var_mean.correction(view_95, [0], correction = 0, keepdim = True)
    getitem_50: "f32[1, 128]" = var_mean_19[0]
    getitem_51: "f32[1, 128]" = var_mean_19[1];  var_mean_19 = None
    add_118: "f32[1, 128]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
    rsqrt_19: "f32[1, 128]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    sub_23: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_95, getitem_51)
    mul_148: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_19);  sub_23 = None
    squeeze_57: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_51, [0]);  getitem_51 = None
    squeeze_58: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0]);  rsqrt_19 = None
    mul_149: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_150: "f32[128]" = torch.ops.aten.mul.Tensor(primals_280, 0.9)
    add_119: "f32[128]" = torch.ops.aten.add.Tensor(mul_149, mul_150);  mul_149 = mul_150 = None
    squeeze_59: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_50, [0]);  getitem_50 = None
    mul_151: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0006381620931717);  squeeze_59 = None
    mul_152: "f32[128]" = torch.ops.aten.mul.Tensor(mul_151, 0.1);  mul_151 = None
    mul_153: "f32[128]" = torch.ops.aten.mul.Tensor(primals_281, 0.9)
    add_120: "f32[128]" = torch.ops.aten.add.Tensor(mul_152, mul_153);  mul_152 = mul_153 = None
    mul_154: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(mul_148, primals_73);  mul_148 = None
    add_121: "f32[1568, 128]" = torch.ops.aten.add.Tensor(mul_154, primals_74);  mul_154 = primals_74 = None
    view_96: "f32[8, 196, 128]" = torch.ops.aten.view.default(add_121, [8, 196, 128]);  add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_122: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_110, view_96);  add_110 = view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_33: "f32[128, 640]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
    clone_28: "f32[8, 196, 128]" = torch.ops.aten.clone.default(add_122, memory_format = torch.contiguous_format)
    view_97: "f32[1568, 128]" = torch.ops.aten.view.default(clone_28, [1568, 128]);  clone_28 = None
    mm_16: "f32[1568, 640]" = torch.ops.aten.mm.default(view_97, permute_33)
    view_98: "f32[8, 196, 640]" = torch.ops.aten.view.default(mm_16, [8, 196, 640]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_99: "f32[1568, 640]" = torch.ops.aten.view.default(view_98, [1568, 640]);  view_98 = None
    add_123: "i64[]" = torch.ops.aten.add.Tensor(primals_285, 1)
    var_mean_20 = torch.ops.aten.var_mean.correction(view_99, [0], correction = 0, keepdim = True)
    getitem_52: "f32[1, 640]" = var_mean_20[0]
    getitem_53: "f32[1, 640]" = var_mean_20[1];  var_mean_20 = None
    add_124: "f32[1, 640]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05)
    rsqrt_20: "f32[1, 640]" = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
    sub_24: "f32[1568, 640]" = torch.ops.aten.sub.Tensor(view_99, getitem_53)
    mul_155: "f32[1568, 640]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_20);  sub_24 = None
    squeeze_60: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_53, [0]);  getitem_53 = None
    squeeze_61: "f32[640]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0]);  rsqrt_20 = None
    mul_156: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_157: "f32[640]" = torch.ops.aten.mul.Tensor(primals_283, 0.9)
    add_125: "f32[640]" = torch.ops.aten.add.Tensor(mul_156, mul_157);  mul_156 = mul_157 = None
    squeeze_62: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_52, [0]);  getitem_52 = None
    mul_158: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0006381620931717);  squeeze_62 = None
    mul_159: "f32[640]" = torch.ops.aten.mul.Tensor(mul_158, 0.1);  mul_158 = None
    mul_160: "f32[640]" = torch.ops.aten.mul.Tensor(primals_284, 0.9)
    add_126: "f32[640]" = torch.ops.aten.add.Tensor(mul_159, mul_160);  mul_159 = mul_160 = None
    mul_161: "f32[1568, 640]" = torch.ops.aten.mul.Tensor(mul_155, primals_76);  mul_155 = None
    add_127: "f32[1568, 640]" = torch.ops.aten.add.Tensor(mul_161, primals_77);  mul_161 = primals_77 = None
    view_100: "f32[8, 196, 640]" = torch.ops.aten.view.default(add_127, [8, 196, 640]);  add_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:331, code: k, v = self.kv(x).view(B, N, self.num_heads, -1).split([self.key_dim, self.val_dim], dim=3)
    view_101: "f32[8, 196, 8, 80]" = torch.ops.aten.view.default(view_100, [8, 196, 8, -1]);  view_100 = None
    split_with_sizes_4 = torch.ops.aten.split_with_sizes.default(view_101, [16, 64], 3);  view_101 = None
    getitem_54: "f32[8, 196, 8, 16]" = split_with_sizes_4[0]
    getitem_55: "f32[8, 196, 8, 64]" = split_with_sizes_4[1];  split_with_sizes_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:332, code: k = k.permute(0, 2, 3, 1)  # BHCN
    permute_34: "f32[8, 8, 16, 196]" = torch.ops.aten.permute.default(getitem_54, [0, 2, 3, 1]);  getitem_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:333, code: v = v.permute(0, 2, 1, 3)  # BHNC
    permute_35: "f32[8, 8, 196, 64]" = torch.ops.aten.permute.default(getitem_55, [0, 2, 1, 3]);  getitem_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:157, code: x = x.view(B, self.resolution[0], self.resolution[1], C)
    view_102: "f32[8, 14, 14, 128]" = torch.ops.aten.view.default(add_122, [8, 14, 14, 128]);  add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:161, code: x = x[:, ::self.stride, ::self.stride]
    slice_5: "f32[8, 14, 14, 128]" = torch.ops.aten.slice.Tensor(view_102, 0, 0, 9223372036854775807);  view_102 = None
    slice_6: "f32[8, 7, 14, 128]" = torch.ops.aten.slice.Tensor(slice_5, 1, 0, 9223372036854775807, 2);  slice_5 = None
    slice_7: "f32[8, 7, 7, 128]" = torch.ops.aten.slice.Tensor(slice_6, 2, 0, 9223372036854775807, 2);  slice_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:162, code: return x.reshape(B, -1, C)
    clone_29: "f32[8, 7, 7, 128]" = torch.ops.aten.clone.default(slice_7, memory_format = torch.contiguous_format);  slice_7 = None
    view_103: "f32[8, 49, 128]" = torch.ops.aten.view.default(clone_29, [8, 49, 128]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_36: "f32[128, 128]" = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
    view_104: "f32[392, 128]" = torch.ops.aten.view.default(view_103, [392, 128]);  view_103 = None
    mm_17: "f32[392, 128]" = torch.ops.aten.mm.default(view_104, permute_36)
    view_105: "f32[8, 49, 128]" = torch.ops.aten.view.default(mm_17, [8, 49, 128]);  mm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_106: "f32[392, 128]" = torch.ops.aten.view.default(view_105, [392, 128]);  view_105 = None
    add_128: "i64[]" = torch.ops.aten.add.Tensor(primals_288, 1)
    var_mean_21 = torch.ops.aten.var_mean.correction(view_106, [0], correction = 0, keepdim = True)
    getitem_56: "f32[1, 128]" = var_mean_21[0]
    getitem_57: "f32[1, 128]" = var_mean_21[1];  var_mean_21 = None
    add_129: "f32[1, 128]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05)
    rsqrt_21: "f32[1, 128]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
    sub_25: "f32[392, 128]" = torch.ops.aten.sub.Tensor(view_106, getitem_57)
    mul_162: "f32[392, 128]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_21);  sub_25 = None
    squeeze_63: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_57, [0]);  getitem_57 = None
    squeeze_64: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0]);  rsqrt_21 = None
    mul_163: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_164: "f32[128]" = torch.ops.aten.mul.Tensor(primals_286, 0.9)
    add_130: "f32[128]" = torch.ops.aten.add.Tensor(mul_163, mul_164);  mul_163 = mul_164 = None
    squeeze_65: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_56, [0]);  getitem_56 = None
    mul_165: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0025575447570332);  squeeze_65 = None
    mul_166: "f32[128]" = torch.ops.aten.mul.Tensor(mul_165, 0.1);  mul_165 = None
    mul_167: "f32[128]" = torch.ops.aten.mul.Tensor(primals_287, 0.9)
    add_131: "f32[128]" = torch.ops.aten.add.Tensor(mul_166, mul_167);  mul_166 = mul_167 = None
    mul_168: "f32[392, 128]" = torch.ops.aten.mul.Tensor(mul_162, primals_79);  mul_162 = None
    add_132: "f32[392, 128]" = torch.ops.aten.add.Tensor(mul_168, primals_80);  mul_168 = primals_80 = None
    view_107: "f32[8, 49, 128]" = torch.ops.aten.view.default(add_132, [8, 49, 128]);  add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:334, code: q = self.q(x).view(B, -1, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
    view_108: "f32[8, 49, 8, 16]" = torch.ops.aten.view.default(view_107, [8, -1, 8, 16]);  view_107 = None
    permute_37: "f32[8, 8, 49, 16]" = torch.ops.aten.permute.default(view_108, [0, 2, 1, 3]);  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:336, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_16: "f32[8, 8, 49, 16]" = torch.ops.aten.expand.default(permute_37, [8, 8, 49, 16]);  permute_37 = None
    clone_30: "f32[8, 8, 49, 16]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_109: "f32[64, 49, 16]" = torch.ops.aten.view.default(clone_30, [64, 49, 16]);  clone_30 = None
    expand_17: "f32[8, 8, 16, 196]" = torch.ops.aten.expand.default(permute_34, [8, 8, 16, 196]);  permute_34 = None
    clone_31: "f32[8, 8, 16, 196]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_110: "f32[64, 16, 196]" = torch.ops.aten.view.default(clone_31, [64, 16, 196]);  clone_31 = None
    bmm_8: "f32[64, 49, 196]" = torch.ops.aten.bmm.default(view_109, view_110)
    view_111: "f32[8, 8, 49, 196]" = torch.ops.aten.view.default(bmm_8, [8, 8, 49, 196]);  bmm_8 = None
    mul_169: "f32[8, 8, 49, 196]" = torch.ops.aten.mul.Tensor(view_111, 0.25);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:311, code: return self.attention_biases[:, self.attention_bias_idxs]
    slice_8: "f32[8, 196]" = torch.ops.aten.slice.Tensor(primals_5, 0, 0, 9223372036854775807);  primals_5 = None
    index_4: "f32[8, 49, 196]" = torch.ops.aten.index.Tensor(slice_8, [None, primals_213]);  slice_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:336, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_133: "f32[8, 8, 49, 196]" = torch.ops.aten.add.Tensor(mul_169, index_4);  mul_169 = index_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:337, code: attn = attn.softmax(dim=-1)
    amax_4: "f32[8, 8, 49, 1]" = torch.ops.aten.amax.default(add_133, [-1], True)
    sub_26: "f32[8, 8, 49, 196]" = torch.ops.aten.sub.Tensor(add_133, amax_4);  add_133 = amax_4 = None
    exp_4: "f32[8, 8, 49, 196]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
    sum_5: "f32[8, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_15: "f32[8, 8, 49, 196]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_4: "f32[8, 8, 49, 196]" = torch.ops.aten.alias.default(div_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:339, code: x = (attn @ v).transpose(1, 2).reshape(B, -1, self.val_attn_dim)
    expand_18: "f32[8, 8, 49, 196]" = torch.ops.aten.expand.default(div_15, [8, 8, 49, 196]);  div_15 = None
    view_112: "f32[64, 49, 196]" = torch.ops.aten.view.default(expand_18, [64, 49, 196]);  expand_18 = None
    expand_19: "f32[8, 8, 196, 64]" = torch.ops.aten.expand.default(permute_35, [8, 8, 196, 64]);  permute_35 = None
    clone_32: "f32[8, 8, 196, 64]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
    view_113: "f32[64, 196, 64]" = torch.ops.aten.view.default(clone_32, [64, 196, 64]);  clone_32 = None
    bmm_9: "f32[64, 49, 64]" = torch.ops.aten.bmm.default(view_112, view_113)
    view_114: "f32[8, 8, 49, 64]" = torch.ops.aten.view.default(bmm_9, [8, 8, 49, 64]);  bmm_9 = None
    permute_38: "f32[8, 49, 8, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
    clone_33: "f32[8, 49, 8, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    view_115: "f32[8, 49, 512]" = torch.ops.aten.view.default(clone_33, [8, 49, 512]);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:340, code: x = self.proj(x)
    add_134: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(view_115, 3)
    clamp_min_11: "f32[8, 49, 512]" = torch.ops.aten.clamp_min.default(add_134, 0);  add_134 = None
    clamp_max_11: "f32[8, 49, 512]" = torch.ops.aten.clamp_max.default(clamp_min_11, 6);  clamp_min_11 = None
    mul_170: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_115, clamp_max_11);  clamp_max_11 = None
    div_16: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(mul_170, 6);  mul_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_39: "f32[512, 256]" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
    view_116: "f32[392, 512]" = torch.ops.aten.view.default(div_16, [392, 512]);  div_16 = None
    mm_18: "f32[392, 256]" = torch.ops.aten.mm.default(view_116, permute_39)
    view_117: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_18, [8, 49, 256]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_118: "f32[392, 256]" = torch.ops.aten.view.default(view_117, [392, 256]);  view_117 = None
    add_135: "i64[]" = torch.ops.aten.add.Tensor(primals_291, 1)
    var_mean_22 = torch.ops.aten.var_mean.correction(view_118, [0], correction = 0, keepdim = True)
    getitem_58: "f32[1, 256]" = var_mean_22[0]
    getitem_59: "f32[1, 256]" = var_mean_22[1];  var_mean_22 = None
    add_136: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
    rsqrt_22: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
    sub_27: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_118, getitem_59)
    mul_171: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_22);  sub_27 = None
    squeeze_66: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_59, [0]);  getitem_59 = None
    squeeze_67: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0]);  rsqrt_22 = None
    mul_172: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_173: "f32[256]" = torch.ops.aten.mul.Tensor(primals_289, 0.9)
    add_137: "f32[256]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
    squeeze_68: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_58, [0]);  getitem_58 = None
    mul_174: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0025575447570332);  squeeze_68 = None
    mul_175: "f32[256]" = torch.ops.aten.mul.Tensor(mul_174, 0.1);  mul_174 = None
    mul_176: "f32[256]" = torch.ops.aten.mul.Tensor(primals_290, 0.9)
    add_138: "f32[256]" = torch.ops.aten.add.Tensor(mul_175, mul_176);  mul_175 = mul_176 = None
    mul_177: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_171, primals_82);  mul_171 = None
    add_139: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_177, primals_83);  mul_177 = primals_83 = None
    view_119: "f32[8, 49, 256]" = torch.ops.aten.view.default(add_139, [8, 49, 256]);  add_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_40: "f32[256, 512]" = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
    view_120: "f32[392, 256]" = torch.ops.aten.view.default(view_119, [392, 256])
    mm_19: "f32[392, 512]" = torch.ops.aten.mm.default(view_120, permute_40)
    view_121: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_19, [8, 49, 512]);  mm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_122: "f32[392, 512]" = torch.ops.aten.view.default(view_121, [392, 512]);  view_121 = None
    add_140: "i64[]" = torch.ops.aten.add.Tensor(primals_294, 1)
    var_mean_23 = torch.ops.aten.var_mean.correction(view_122, [0], correction = 0, keepdim = True)
    getitem_60: "f32[1, 512]" = var_mean_23[0]
    getitem_61: "f32[1, 512]" = var_mean_23[1];  var_mean_23 = None
    add_141: "f32[1, 512]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
    rsqrt_23: "f32[1, 512]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
    sub_28: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_122, getitem_61)
    mul_178: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_23);  sub_28 = None
    squeeze_69: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_61, [0]);  getitem_61 = None
    squeeze_70: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0]);  rsqrt_23 = None
    mul_179: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
    mul_180: "f32[512]" = torch.ops.aten.mul.Tensor(primals_292, 0.9)
    add_142: "f32[512]" = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
    squeeze_71: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_60, [0]);  getitem_60 = None
    mul_181: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0025575447570332);  squeeze_71 = None
    mul_182: "f32[512]" = torch.ops.aten.mul.Tensor(mul_181, 0.1);  mul_181 = None
    mul_183: "f32[512]" = torch.ops.aten.mul.Tensor(primals_293, 0.9)
    add_143: "f32[512]" = torch.ops.aten.add.Tensor(mul_182, mul_183);  mul_182 = mul_183 = None
    mul_184: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_178, primals_85);  mul_178 = None
    add_144: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_184, primals_86);  mul_184 = primals_86 = None
    view_123: "f32[8, 49, 512]" = torch.ops.aten.view.default(add_144, [8, 49, 512]);  add_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_145: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(view_123, 3)
    clamp_min_12: "f32[8, 49, 512]" = torch.ops.aten.clamp_min.default(add_145, 0);  add_145 = None
    clamp_max_12: "f32[8, 49, 512]" = torch.ops.aten.clamp_max.default(clamp_min_12, 6);  clamp_min_12 = None
    mul_185: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_123, clamp_max_12);  clamp_max_12 = None
    div_17: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(mul_185, 6);  mul_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    clone_34: "f32[8, 49, 512]" = torch.ops.aten.clone.default(div_17);  div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_41: "f32[512, 256]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
    view_124: "f32[392, 512]" = torch.ops.aten.view.default(clone_34, [392, 512]);  clone_34 = None
    mm_20: "f32[392, 256]" = torch.ops.aten.mm.default(view_124, permute_41)
    view_125: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_20, [8, 49, 256]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_126: "f32[392, 256]" = torch.ops.aten.view.default(view_125, [392, 256]);  view_125 = None
    add_146: "i64[]" = torch.ops.aten.add.Tensor(primals_297, 1)
    var_mean_24 = torch.ops.aten.var_mean.correction(view_126, [0], correction = 0, keepdim = True)
    getitem_62: "f32[1, 256]" = var_mean_24[0]
    getitem_63: "f32[1, 256]" = var_mean_24[1];  var_mean_24 = None
    add_147: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
    rsqrt_24: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
    sub_29: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_126, getitem_63)
    mul_186: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_24);  sub_29 = None
    squeeze_72: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_63, [0]);  getitem_63 = None
    squeeze_73: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0]);  rsqrt_24 = None
    mul_187: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_188: "f32[256]" = torch.ops.aten.mul.Tensor(primals_295, 0.9)
    add_148: "f32[256]" = torch.ops.aten.add.Tensor(mul_187, mul_188);  mul_187 = mul_188 = None
    squeeze_74: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_62, [0]);  getitem_62 = None
    mul_189: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0025575447570332);  squeeze_74 = None
    mul_190: "f32[256]" = torch.ops.aten.mul.Tensor(mul_189, 0.1);  mul_189 = None
    mul_191: "f32[256]" = torch.ops.aten.mul.Tensor(primals_296, 0.9)
    add_149: "f32[256]" = torch.ops.aten.add.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
    mul_192: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_186, primals_88);  mul_186 = None
    add_150: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_192, primals_89);  mul_192 = primals_89 = None
    view_127: "f32[8, 49, 256]" = torch.ops.aten.view.default(add_150, [8, 49, 256]);  add_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:415, code: x = x + self.drop_path(self.mlp(x))
    add_151: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(view_119, view_127);  view_119 = view_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_42: "f32[256, 512]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
    view_128: "f32[392, 256]" = torch.ops.aten.view.default(add_151, [392, 256])
    mm_21: "f32[392, 512]" = torch.ops.aten.mm.default(view_128, permute_42)
    view_129: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_21, [8, 49, 512]);  mm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_130: "f32[392, 512]" = torch.ops.aten.view.default(view_129, [392, 512]);  view_129 = None
    add_152: "i64[]" = torch.ops.aten.add.Tensor(primals_300, 1)
    var_mean_25 = torch.ops.aten.var_mean.correction(view_130, [0], correction = 0, keepdim = True)
    getitem_64: "f32[1, 512]" = var_mean_25[0]
    getitem_65: "f32[1, 512]" = var_mean_25[1];  var_mean_25 = None
    add_153: "f32[1, 512]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
    rsqrt_25: "f32[1, 512]" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
    sub_30: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_130, getitem_65)
    mul_193: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_25);  sub_30 = None
    squeeze_75: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_65, [0]);  getitem_65 = None
    squeeze_76: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0]);  rsqrt_25 = None
    mul_194: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_195: "f32[512]" = torch.ops.aten.mul.Tensor(primals_298, 0.9)
    add_154: "f32[512]" = torch.ops.aten.add.Tensor(mul_194, mul_195);  mul_194 = mul_195 = None
    squeeze_77: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_64, [0]);  getitem_64 = None
    mul_196: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0025575447570332);  squeeze_77 = None
    mul_197: "f32[512]" = torch.ops.aten.mul.Tensor(mul_196, 0.1);  mul_196 = None
    mul_198: "f32[512]" = torch.ops.aten.mul.Tensor(primals_299, 0.9)
    add_155: "f32[512]" = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
    mul_199: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_193, primals_91);  mul_193 = None
    add_156: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_199, primals_92);  mul_199 = primals_92 = None
    view_131: "f32[8, 49, 512]" = torch.ops.aten.view.default(add_156, [8, 49, 512]);  add_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_132: "f32[8, 49, 8, 64]" = torch.ops.aten.view.default(view_131, [8, 49, 8, -1]);  view_131 = None
    split_with_sizes_5 = torch.ops.aten.split_with_sizes.default(view_132, [16, 16, 32], 3);  view_132 = None
    getitem_66: "f32[8, 49, 8, 16]" = split_with_sizes_5[0]
    getitem_67: "f32[8, 49, 8, 16]" = split_with_sizes_5[1]
    getitem_68: "f32[8, 49, 8, 32]" = split_with_sizes_5[2];  split_with_sizes_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_43: "f32[8, 8, 49, 16]" = torch.ops.aten.permute.default(getitem_66, [0, 2, 1, 3]);  getitem_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_44: "f32[8, 8, 16, 49]" = torch.ops.aten.permute.default(getitem_67, [0, 2, 3, 1]);  getitem_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_45: "f32[8, 8, 49, 32]" = torch.ops.aten.permute.default(getitem_68, [0, 2, 1, 3]);  getitem_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_20: "f32[8, 8, 49, 16]" = torch.ops.aten.expand.default(permute_43, [8, 8, 49, 16]);  permute_43 = None
    clone_35: "f32[8, 8, 49, 16]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
    view_133: "f32[64, 49, 16]" = torch.ops.aten.view.default(clone_35, [64, 49, 16]);  clone_35 = None
    expand_21: "f32[8, 8, 16, 49]" = torch.ops.aten.expand.default(permute_44, [8, 8, 16, 49]);  permute_44 = None
    clone_36: "f32[8, 8, 16, 49]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_134: "f32[64, 16, 49]" = torch.ops.aten.view.default(clone_36, [64, 16, 49]);  clone_36 = None
    bmm_10: "f32[64, 49, 49]" = torch.ops.aten.bmm.default(view_133, view_134)
    view_135: "f32[8, 8, 49, 49]" = torch.ops.aten.view.default(bmm_10, [8, 8, 49, 49]);  bmm_10 = None
    mul_200: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(view_135, 0.25);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    slice_9: "f32[8, 49]" = torch.ops.aten.slice.Tensor(primals_6, 0, 0, 9223372036854775807);  primals_6 = None
    index_5: "f32[8, 49, 49]" = torch.ops.aten.index.Tensor(slice_9, [None, primals_214]);  slice_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_157: "f32[8, 8, 49, 49]" = torch.ops.aten.add.Tensor(mul_200, index_5);  mul_200 = index_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax_5: "f32[8, 8, 49, 1]" = torch.ops.aten.amax.default(add_157, [-1], True)
    sub_31: "f32[8, 8, 49, 49]" = torch.ops.aten.sub.Tensor(add_157, amax_5);  add_157 = amax_5 = None
    exp_5: "f32[8, 8, 49, 49]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_6: "f32[8, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_18: "f32[8, 8, 49, 49]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_5: "f32[8, 8, 49, 49]" = torch.ops.aten.alias.default(div_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_22: "f32[8, 8, 49, 49]" = torch.ops.aten.expand.default(div_18, [8, 8, 49, 49]);  div_18 = None
    view_136: "f32[64, 49, 49]" = torch.ops.aten.view.default(expand_22, [64, 49, 49]);  expand_22 = None
    expand_23: "f32[8, 8, 49, 32]" = torch.ops.aten.expand.default(permute_45, [8, 8, 49, 32]);  permute_45 = None
    clone_37: "f32[8, 8, 49, 32]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
    view_137: "f32[64, 49, 32]" = torch.ops.aten.view.default(clone_37, [64, 49, 32]);  clone_37 = None
    bmm_11: "f32[64, 49, 32]" = torch.ops.aten.bmm.default(view_136, view_137)
    view_138: "f32[8, 8, 49, 32]" = torch.ops.aten.view.default(bmm_11, [8, 8, 49, 32]);  bmm_11 = None
    permute_46: "f32[8, 49, 8, 32]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
    clone_38: "f32[8, 49, 8, 32]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
    view_139: "f32[8, 49, 256]" = torch.ops.aten.view.default(clone_38, [8, 49, 256]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_158: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(view_139, 3)
    clamp_min_13: "f32[8, 49, 256]" = torch.ops.aten.clamp_min.default(add_158, 0);  add_158 = None
    clamp_max_13: "f32[8, 49, 256]" = torch.ops.aten.clamp_max.default(clamp_min_13, 6);  clamp_min_13 = None
    mul_201: "f32[8, 49, 256]" = torch.ops.aten.mul.Tensor(view_139, clamp_max_13);  clamp_max_13 = None
    div_19: "f32[8, 49, 256]" = torch.ops.aten.div.Tensor(mul_201, 6);  mul_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_47: "f32[256, 256]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
    view_140: "f32[392, 256]" = torch.ops.aten.view.default(div_19, [392, 256]);  div_19 = None
    mm_22: "f32[392, 256]" = torch.ops.aten.mm.default(view_140, permute_47)
    view_141: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_22, [8, 49, 256]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_142: "f32[392, 256]" = torch.ops.aten.view.default(view_141, [392, 256]);  view_141 = None
    add_159: "i64[]" = torch.ops.aten.add.Tensor(primals_303, 1)
    var_mean_26 = torch.ops.aten.var_mean.correction(view_142, [0], correction = 0, keepdim = True)
    getitem_69: "f32[1, 256]" = var_mean_26[0]
    getitem_70: "f32[1, 256]" = var_mean_26[1];  var_mean_26 = None
    add_160: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_69, 1e-05)
    rsqrt_26: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
    sub_32: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_142, getitem_70)
    mul_202: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_26);  sub_32 = None
    squeeze_78: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_70, [0]);  getitem_70 = None
    squeeze_79: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0]);  rsqrt_26 = None
    mul_203: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_204: "f32[256]" = torch.ops.aten.mul.Tensor(primals_301, 0.9)
    add_161: "f32[256]" = torch.ops.aten.add.Tensor(mul_203, mul_204);  mul_203 = mul_204 = None
    squeeze_80: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_69, [0]);  getitem_69 = None
    mul_205: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0025575447570332);  squeeze_80 = None
    mul_206: "f32[256]" = torch.ops.aten.mul.Tensor(mul_205, 0.1);  mul_205 = None
    mul_207: "f32[256]" = torch.ops.aten.mul.Tensor(primals_302, 0.9)
    add_162: "f32[256]" = torch.ops.aten.add.Tensor(mul_206, mul_207);  mul_206 = mul_207 = None
    mul_208: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_202, primals_94);  mul_202 = None
    add_163: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_208, primals_95);  mul_208 = primals_95 = None
    view_143: "f32[8, 49, 256]" = torch.ops.aten.view.default(add_163, [8, 49, 256]);  add_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_164: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_151, view_143);  add_151 = view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_48: "f32[256, 512]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
    view_144: "f32[392, 256]" = torch.ops.aten.view.default(add_164, [392, 256])
    mm_23: "f32[392, 512]" = torch.ops.aten.mm.default(view_144, permute_48)
    view_145: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_23, [8, 49, 512]);  mm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_146: "f32[392, 512]" = torch.ops.aten.view.default(view_145, [392, 512]);  view_145 = None
    add_165: "i64[]" = torch.ops.aten.add.Tensor(primals_306, 1)
    var_mean_27 = torch.ops.aten.var_mean.correction(view_146, [0], correction = 0, keepdim = True)
    getitem_71: "f32[1, 512]" = var_mean_27[0]
    getitem_72: "f32[1, 512]" = var_mean_27[1];  var_mean_27 = None
    add_166: "f32[1, 512]" = torch.ops.aten.add.Tensor(getitem_71, 1e-05)
    rsqrt_27: "f32[1, 512]" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
    sub_33: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_146, getitem_72)
    mul_209: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_27);  sub_33 = None
    squeeze_81: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_72, [0]);  getitem_72 = None
    squeeze_82: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0]);  rsqrt_27 = None
    mul_210: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
    mul_211: "f32[512]" = torch.ops.aten.mul.Tensor(primals_304, 0.9)
    add_167: "f32[512]" = torch.ops.aten.add.Tensor(mul_210, mul_211);  mul_210 = mul_211 = None
    squeeze_83: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_71, [0]);  getitem_71 = None
    mul_212: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0025575447570332);  squeeze_83 = None
    mul_213: "f32[512]" = torch.ops.aten.mul.Tensor(mul_212, 0.1);  mul_212 = None
    mul_214: "f32[512]" = torch.ops.aten.mul.Tensor(primals_305, 0.9)
    add_168: "f32[512]" = torch.ops.aten.add.Tensor(mul_213, mul_214);  mul_213 = mul_214 = None
    mul_215: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_209, primals_97);  mul_209 = None
    add_169: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_215, primals_98);  mul_215 = primals_98 = None
    view_147: "f32[8, 49, 512]" = torch.ops.aten.view.default(add_169, [8, 49, 512]);  add_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_170: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(view_147, 3)
    clamp_min_14: "f32[8, 49, 512]" = torch.ops.aten.clamp_min.default(add_170, 0);  add_170 = None
    clamp_max_14: "f32[8, 49, 512]" = torch.ops.aten.clamp_max.default(clamp_min_14, 6);  clamp_min_14 = None
    mul_216: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_147, clamp_max_14);  clamp_max_14 = None
    div_20: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(mul_216, 6);  mul_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    clone_39: "f32[8, 49, 512]" = torch.ops.aten.clone.default(div_20);  div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_49: "f32[512, 256]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
    view_148: "f32[392, 512]" = torch.ops.aten.view.default(clone_39, [392, 512]);  clone_39 = None
    mm_24: "f32[392, 256]" = torch.ops.aten.mm.default(view_148, permute_49)
    view_149: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_24, [8, 49, 256]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_150: "f32[392, 256]" = torch.ops.aten.view.default(view_149, [392, 256]);  view_149 = None
    add_171: "i64[]" = torch.ops.aten.add.Tensor(primals_309, 1)
    var_mean_28 = torch.ops.aten.var_mean.correction(view_150, [0], correction = 0, keepdim = True)
    getitem_73: "f32[1, 256]" = var_mean_28[0]
    getitem_74: "f32[1, 256]" = var_mean_28[1];  var_mean_28 = None
    add_172: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_73, 1e-05)
    rsqrt_28: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
    sub_34: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_150, getitem_74)
    mul_217: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_28);  sub_34 = None
    squeeze_84: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_74, [0]);  getitem_74 = None
    squeeze_85: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0]);  rsqrt_28 = None
    mul_218: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
    mul_219: "f32[256]" = torch.ops.aten.mul.Tensor(primals_307, 0.9)
    add_173: "f32[256]" = torch.ops.aten.add.Tensor(mul_218, mul_219);  mul_218 = mul_219 = None
    squeeze_86: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_73, [0]);  getitem_73 = None
    mul_220: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0025575447570332);  squeeze_86 = None
    mul_221: "f32[256]" = torch.ops.aten.mul.Tensor(mul_220, 0.1);  mul_220 = None
    mul_222: "f32[256]" = torch.ops.aten.mul.Tensor(primals_308, 0.9)
    add_174: "f32[256]" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
    mul_223: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_217, primals_100);  mul_217 = None
    add_175: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_223, primals_101);  mul_223 = primals_101 = None
    view_151: "f32[8, 49, 256]" = torch.ops.aten.view.default(add_175, [8, 49, 256]);  add_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_176: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_164, view_151);  add_164 = view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_50: "f32[256, 512]" = torch.ops.aten.permute.default(primals_102, [1, 0]);  primals_102 = None
    view_152: "f32[392, 256]" = torch.ops.aten.view.default(add_176, [392, 256])
    mm_25: "f32[392, 512]" = torch.ops.aten.mm.default(view_152, permute_50)
    view_153: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_25, [8, 49, 512]);  mm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_154: "f32[392, 512]" = torch.ops.aten.view.default(view_153, [392, 512]);  view_153 = None
    add_177: "i64[]" = torch.ops.aten.add.Tensor(primals_312, 1)
    var_mean_29 = torch.ops.aten.var_mean.correction(view_154, [0], correction = 0, keepdim = True)
    getitem_75: "f32[1, 512]" = var_mean_29[0]
    getitem_76: "f32[1, 512]" = var_mean_29[1];  var_mean_29 = None
    add_178: "f32[1, 512]" = torch.ops.aten.add.Tensor(getitem_75, 1e-05)
    rsqrt_29: "f32[1, 512]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
    sub_35: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_154, getitem_76)
    mul_224: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_29);  sub_35 = None
    squeeze_87: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_76, [0]);  getitem_76 = None
    squeeze_88: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0]);  rsqrt_29 = None
    mul_225: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
    mul_226: "f32[512]" = torch.ops.aten.mul.Tensor(primals_310, 0.9)
    add_179: "f32[512]" = torch.ops.aten.add.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
    squeeze_89: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_75, [0]);  getitem_75 = None
    mul_227: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0025575447570332);  squeeze_89 = None
    mul_228: "f32[512]" = torch.ops.aten.mul.Tensor(mul_227, 0.1);  mul_227 = None
    mul_229: "f32[512]" = torch.ops.aten.mul.Tensor(primals_311, 0.9)
    add_180: "f32[512]" = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
    mul_230: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_224, primals_103);  mul_224 = None
    add_181: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_230, primals_104);  mul_230 = primals_104 = None
    view_155: "f32[8, 49, 512]" = torch.ops.aten.view.default(add_181, [8, 49, 512]);  add_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_156: "f32[8, 49, 8, 64]" = torch.ops.aten.view.default(view_155, [8, 49, 8, -1]);  view_155 = None
    split_with_sizes_6 = torch.ops.aten.split_with_sizes.default(view_156, [16, 16, 32], 3);  view_156 = None
    getitem_77: "f32[8, 49, 8, 16]" = split_with_sizes_6[0]
    getitem_78: "f32[8, 49, 8, 16]" = split_with_sizes_6[1]
    getitem_79: "f32[8, 49, 8, 32]" = split_with_sizes_6[2];  split_with_sizes_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_51: "f32[8, 8, 49, 16]" = torch.ops.aten.permute.default(getitem_77, [0, 2, 1, 3]);  getitem_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_52: "f32[8, 8, 16, 49]" = torch.ops.aten.permute.default(getitem_78, [0, 2, 3, 1]);  getitem_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_53: "f32[8, 8, 49, 32]" = torch.ops.aten.permute.default(getitem_79, [0, 2, 1, 3]);  getitem_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_24: "f32[8, 8, 49, 16]" = torch.ops.aten.expand.default(permute_51, [8, 8, 49, 16]);  permute_51 = None
    clone_40: "f32[8, 8, 49, 16]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    view_157: "f32[64, 49, 16]" = torch.ops.aten.view.default(clone_40, [64, 49, 16]);  clone_40 = None
    expand_25: "f32[8, 8, 16, 49]" = torch.ops.aten.expand.default(permute_52, [8, 8, 16, 49]);  permute_52 = None
    clone_41: "f32[8, 8, 16, 49]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_158: "f32[64, 16, 49]" = torch.ops.aten.view.default(clone_41, [64, 16, 49]);  clone_41 = None
    bmm_12: "f32[64, 49, 49]" = torch.ops.aten.bmm.default(view_157, view_158)
    view_159: "f32[8, 8, 49, 49]" = torch.ops.aten.view.default(bmm_12, [8, 8, 49, 49]);  bmm_12 = None
    mul_231: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(view_159, 0.25);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    slice_10: "f32[8, 49]" = torch.ops.aten.slice.Tensor(primals_7, 0, 0, 9223372036854775807);  primals_7 = None
    index_6: "f32[8, 49, 49]" = torch.ops.aten.index.Tensor(slice_10, [None, primals_215]);  slice_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_182: "f32[8, 8, 49, 49]" = torch.ops.aten.add.Tensor(mul_231, index_6);  mul_231 = index_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax_6: "f32[8, 8, 49, 1]" = torch.ops.aten.amax.default(add_182, [-1], True)
    sub_36: "f32[8, 8, 49, 49]" = torch.ops.aten.sub.Tensor(add_182, amax_6);  add_182 = amax_6 = None
    exp_6: "f32[8, 8, 49, 49]" = torch.ops.aten.exp.default(sub_36);  sub_36 = None
    sum_7: "f32[8, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_21: "f32[8, 8, 49, 49]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    alias_6: "f32[8, 8, 49, 49]" = torch.ops.aten.alias.default(div_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_26: "f32[8, 8, 49, 49]" = torch.ops.aten.expand.default(div_21, [8, 8, 49, 49]);  div_21 = None
    view_160: "f32[64, 49, 49]" = torch.ops.aten.view.default(expand_26, [64, 49, 49]);  expand_26 = None
    expand_27: "f32[8, 8, 49, 32]" = torch.ops.aten.expand.default(permute_53, [8, 8, 49, 32]);  permute_53 = None
    clone_42: "f32[8, 8, 49, 32]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
    view_161: "f32[64, 49, 32]" = torch.ops.aten.view.default(clone_42, [64, 49, 32]);  clone_42 = None
    bmm_13: "f32[64, 49, 32]" = torch.ops.aten.bmm.default(view_160, view_161)
    view_162: "f32[8, 8, 49, 32]" = torch.ops.aten.view.default(bmm_13, [8, 8, 49, 32]);  bmm_13 = None
    permute_54: "f32[8, 49, 8, 32]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    clone_43: "f32[8, 49, 8, 32]" = torch.ops.aten.clone.default(permute_54, memory_format = torch.contiguous_format);  permute_54 = None
    view_163: "f32[8, 49, 256]" = torch.ops.aten.view.default(clone_43, [8, 49, 256]);  clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_183: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(view_163, 3)
    clamp_min_15: "f32[8, 49, 256]" = torch.ops.aten.clamp_min.default(add_183, 0);  add_183 = None
    clamp_max_15: "f32[8, 49, 256]" = torch.ops.aten.clamp_max.default(clamp_min_15, 6);  clamp_min_15 = None
    mul_232: "f32[8, 49, 256]" = torch.ops.aten.mul.Tensor(view_163, clamp_max_15);  clamp_max_15 = None
    div_22: "f32[8, 49, 256]" = torch.ops.aten.div.Tensor(mul_232, 6);  mul_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_55: "f32[256, 256]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
    view_164: "f32[392, 256]" = torch.ops.aten.view.default(div_22, [392, 256]);  div_22 = None
    mm_26: "f32[392, 256]" = torch.ops.aten.mm.default(view_164, permute_55)
    view_165: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_26, [8, 49, 256]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_166: "f32[392, 256]" = torch.ops.aten.view.default(view_165, [392, 256]);  view_165 = None
    add_184: "i64[]" = torch.ops.aten.add.Tensor(primals_315, 1)
    var_mean_30 = torch.ops.aten.var_mean.correction(view_166, [0], correction = 0, keepdim = True)
    getitem_80: "f32[1, 256]" = var_mean_30[0]
    getitem_81: "f32[1, 256]" = var_mean_30[1];  var_mean_30 = None
    add_185: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05)
    rsqrt_30: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
    sub_37: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_166, getitem_81)
    mul_233: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_30);  sub_37 = None
    squeeze_90: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_81, [0]);  getitem_81 = None
    squeeze_91: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0]);  rsqrt_30 = None
    mul_234: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
    mul_235: "f32[256]" = torch.ops.aten.mul.Tensor(primals_313, 0.9)
    add_186: "f32[256]" = torch.ops.aten.add.Tensor(mul_234, mul_235);  mul_234 = mul_235 = None
    squeeze_92: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_80, [0]);  getitem_80 = None
    mul_236: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0025575447570332);  squeeze_92 = None
    mul_237: "f32[256]" = torch.ops.aten.mul.Tensor(mul_236, 0.1);  mul_236 = None
    mul_238: "f32[256]" = torch.ops.aten.mul.Tensor(primals_314, 0.9)
    add_187: "f32[256]" = torch.ops.aten.add.Tensor(mul_237, mul_238);  mul_237 = mul_238 = None
    mul_239: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_233, primals_106);  mul_233 = None
    add_188: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_239, primals_107);  mul_239 = primals_107 = None
    view_167: "f32[8, 49, 256]" = torch.ops.aten.view.default(add_188, [8, 49, 256]);  add_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_189: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_176, view_167);  add_176 = view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_56: "f32[256, 512]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
    view_168: "f32[392, 256]" = torch.ops.aten.view.default(add_189, [392, 256])
    mm_27: "f32[392, 512]" = torch.ops.aten.mm.default(view_168, permute_56)
    view_169: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_27, [8, 49, 512]);  mm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_170: "f32[392, 512]" = torch.ops.aten.view.default(view_169, [392, 512]);  view_169 = None
    add_190: "i64[]" = torch.ops.aten.add.Tensor(primals_318, 1)
    var_mean_31 = torch.ops.aten.var_mean.correction(view_170, [0], correction = 0, keepdim = True)
    getitem_82: "f32[1, 512]" = var_mean_31[0]
    getitem_83: "f32[1, 512]" = var_mean_31[1];  var_mean_31 = None
    add_191: "f32[1, 512]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05)
    rsqrt_31: "f32[1, 512]" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
    sub_38: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_170, getitem_83)
    mul_240: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_31);  sub_38 = None
    squeeze_93: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_83, [0]);  getitem_83 = None
    squeeze_94: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0]);  rsqrt_31 = None
    mul_241: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
    mul_242: "f32[512]" = torch.ops.aten.mul.Tensor(primals_316, 0.9)
    add_192: "f32[512]" = torch.ops.aten.add.Tensor(mul_241, mul_242);  mul_241 = mul_242 = None
    squeeze_95: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_82, [0]);  getitem_82 = None
    mul_243: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0025575447570332);  squeeze_95 = None
    mul_244: "f32[512]" = torch.ops.aten.mul.Tensor(mul_243, 0.1);  mul_243 = None
    mul_245: "f32[512]" = torch.ops.aten.mul.Tensor(primals_317, 0.9)
    add_193: "f32[512]" = torch.ops.aten.add.Tensor(mul_244, mul_245);  mul_244 = mul_245 = None
    mul_246: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_240, primals_109);  mul_240 = None
    add_194: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_246, primals_110);  mul_246 = primals_110 = None
    view_171: "f32[8, 49, 512]" = torch.ops.aten.view.default(add_194, [8, 49, 512]);  add_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_195: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(view_171, 3)
    clamp_min_16: "f32[8, 49, 512]" = torch.ops.aten.clamp_min.default(add_195, 0);  add_195 = None
    clamp_max_16: "f32[8, 49, 512]" = torch.ops.aten.clamp_max.default(clamp_min_16, 6);  clamp_min_16 = None
    mul_247: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_171, clamp_max_16);  clamp_max_16 = None
    div_23: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(mul_247, 6);  mul_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    clone_44: "f32[8, 49, 512]" = torch.ops.aten.clone.default(div_23);  div_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_57: "f32[512, 256]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
    view_172: "f32[392, 512]" = torch.ops.aten.view.default(clone_44, [392, 512]);  clone_44 = None
    mm_28: "f32[392, 256]" = torch.ops.aten.mm.default(view_172, permute_57)
    view_173: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_28, [8, 49, 256]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_174: "f32[392, 256]" = torch.ops.aten.view.default(view_173, [392, 256]);  view_173 = None
    add_196: "i64[]" = torch.ops.aten.add.Tensor(primals_321, 1)
    var_mean_32 = torch.ops.aten.var_mean.correction(view_174, [0], correction = 0, keepdim = True)
    getitem_84: "f32[1, 256]" = var_mean_32[0]
    getitem_85: "f32[1, 256]" = var_mean_32[1];  var_mean_32 = None
    add_197: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05)
    rsqrt_32: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
    sub_39: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_174, getitem_85)
    mul_248: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_32);  sub_39 = None
    squeeze_96: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_85, [0]);  getitem_85 = None
    squeeze_97: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0]);  rsqrt_32 = None
    mul_249: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
    mul_250: "f32[256]" = torch.ops.aten.mul.Tensor(primals_319, 0.9)
    add_198: "f32[256]" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
    squeeze_98: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_84, [0]);  getitem_84 = None
    mul_251: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0025575447570332);  squeeze_98 = None
    mul_252: "f32[256]" = torch.ops.aten.mul.Tensor(mul_251, 0.1);  mul_251 = None
    mul_253: "f32[256]" = torch.ops.aten.mul.Tensor(primals_320, 0.9)
    add_199: "f32[256]" = torch.ops.aten.add.Tensor(mul_252, mul_253);  mul_252 = mul_253 = None
    mul_254: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_248, primals_112);  mul_248 = None
    add_200: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_254, primals_113);  mul_254 = primals_113 = None
    view_175: "f32[8, 49, 256]" = torch.ops.aten.view.default(add_200, [8, 49, 256]);  add_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_201: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_189, view_175);  add_189 = view_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_58: "f32[256, 512]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
    view_176: "f32[392, 256]" = torch.ops.aten.view.default(add_201, [392, 256])
    mm_29: "f32[392, 512]" = torch.ops.aten.mm.default(view_176, permute_58)
    view_177: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_29, [8, 49, 512]);  mm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_178: "f32[392, 512]" = torch.ops.aten.view.default(view_177, [392, 512]);  view_177 = None
    add_202: "i64[]" = torch.ops.aten.add.Tensor(primals_324, 1)
    var_mean_33 = torch.ops.aten.var_mean.correction(view_178, [0], correction = 0, keepdim = True)
    getitem_86: "f32[1, 512]" = var_mean_33[0]
    getitem_87: "f32[1, 512]" = var_mean_33[1];  var_mean_33 = None
    add_203: "f32[1, 512]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05)
    rsqrt_33: "f32[1, 512]" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
    sub_40: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_178, getitem_87)
    mul_255: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_33);  sub_40 = None
    squeeze_99: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_87, [0]);  getitem_87 = None
    squeeze_100: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0]);  rsqrt_33 = None
    mul_256: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
    mul_257: "f32[512]" = torch.ops.aten.mul.Tensor(primals_322, 0.9)
    add_204: "f32[512]" = torch.ops.aten.add.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
    squeeze_101: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_86, [0]);  getitem_86 = None
    mul_258: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0025575447570332);  squeeze_101 = None
    mul_259: "f32[512]" = torch.ops.aten.mul.Tensor(mul_258, 0.1);  mul_258 = None
    mul_260: "f32[512]" = torch.ops.aten.mul.Tensor(primals_323, 0.9)
    add_205: "f32[512]" = torch.ops.aten.add.Tensor(mul_259, mul_260);  mul_259 = mul_260 = None
    mul_261: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_255, primals_115);  mul_255 = None
    add_206: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_261, primals_116);  mul_261 = primals_116 = None
    view_179: "f32[8, 49, 512]" = torch.ops.aten.view.default(add_206, [8, 49, 512]);  add_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_180: "f32[8, 49, 8, 64]" = torch.ops.aten.view.default(view_179, [8, 49, 8, -1]);  view_179 = None
    split_with_sizes_7 = torch.ops.aten.split_with_sizes.default(view_180, [16, 16, 32], 3);  view_180 = None
    getitem_88: "f32[8, 49, 8, 16]" = split_with_sizes_7[0]
    getitem_89: "f32[8, 49, 8, 16]" = split_with_sizes_7[1]
    getitem_90: "f32[8, 49, 8, 32]" = split_with_sizes_7[2];  split_with_sizes_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_59: "f32[8, 8, 49, 16]" = torch.ops.aten.permute.default(getitem_88, [0, 2, 1, 3]);  getitem_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_60: "f32[8, 8, 16, 49]" = torch.ops.aten.permute.default(getitem_89, [0, 2, 3, 1]);  getitem_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_61: "f32[8, 8, 49, 32]" = torch.ops.aten.permute.default(getitem_90, [0, 2, 1, 3]);  getitem_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_28: "f32[8, 8, 49, 16]" = torch.ops.aten.expand.default(permute_59, [8, 8, 49, 16]);  permute_59 = None
    clone_45: "f32[8, 8, 49, 16]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
    view_181: "f32[64, 49, 16]" = torch.ops.aten.view.default(clone_45, [64, 49, 16]);  clone_45 = None
    expand_29: "f32[8, 8, 16, 49]" = torch.ops.aten.expand.default(permute_60, [8, 8, 16, 49]);  permute_60 = None
    clone_46: "f32[8, 8, 16, 49]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    view_182: "f32[64, 16, 49]" = torch.ops.aten.view.default(clone_46, [64, 16, 49]);  clone_46 = None
    bmm_14: "f32[64, 49, 49]" = torch.ops.aten.bmm.default(view_181, view_182)
    view_183: "f32[8, 8, 49, 49]" = torch.ops.aten.view.default(bmm_14, [8, 8, 49, 49]);  bmm_14 = None
    mul_262: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(view_183, 0.25);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    slice_11: "f32[8, 49]" = torch.ops.aten.slice.Tensor(primals_8, 0, 0, 9223372036854775807);  primals_8 = None
    index_7: "f32[8, 49, 49]" = torch.ops.aten.index.Tensor(slice_11, [None, primals_216]);  slice_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_207: "f32[8, 8, 49, 49]" = torch.ops.aten.add.Tensor(mul_262, index_7);  mul_262 = index_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax_7: "f32[8, 8, 49, 1]" = torch.ops.aten.amax.default(add_207, [-1], True)
    sub_41: "f32[8, 8, 49, 49]" = torch.ops.aten.sub.Tensor(add_207, amax_7);  add_207 = amax_7 = None
    exp_7: "f32[8, 8, 49, 49]" = torch.ops.aten.exp.default(sub_41);  sub_41 = None
    sum_8: "f32[8, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_24: "f32[8, 8, 49, 49]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    alias_7: "f32[8, 8, 49, 49]" = torch.ops.aten.alias.default(div_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_30: "f32[8, 8, 49, 49]" = torch.ops.aten.expand.default(div_24, [8, 8, 49, 49]);  div_24 = None
    view_184: "f32[64, 49, 49]" = torch.ops.aten.view.default(expand_30, [64, 49, 49]);  expand_30 = None
    expand_31: "f32[8, 8, 49, 32]" = torch.ops.aten.expand.default(permute_61, [8, 8, 49, 32]);  permute_61 = None
    clone_47: "f32[8, 8, 49, 32]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
    view_185: "f32[64, 49, 32]" = torch.ops.aten.view.default(clone_47, [64, 49, 32]);  clone_47 = None
    bmm_15: "f32[64, 49, 32]" = torch.ops.aten.bmm.default(view_184, view_185)
    view_186: "f32[8, 8, 49, 32]" = torch.ops.aten.view.default(bmm_15, [8, 8, 49, 32]);  bmm_15 = None
    permute_62: "f32[8, 49, 8, 32]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
    clone_48: "f32[8, 49, 8, 32]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_187: "f32[8, 49, 256]" = torch.ops.aten.view.default(clone_48, [8, 49, 256]);  clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_208: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(view_187, 3)
    clamp_min_17: "f32[8, 49, 256]" = torch.ops.aten.clamp_min.default(add_208, 0);  add_208 = None
    clamp_max_17: "f32[8, 49, 256]" = torch.ops.aten.clamp_max.default(clamp_min_17, 6);  clamp_min_17 = None
    mul_263: "f32[8, 49, 256]" = torch.ops.aten.mul.Tensor(view_187, clamp_max_17);  clamp_max_17 = None
    div_25: "f32[8, 49, 256]" = torch.ops.aten.div.Tensor(mul_263, 6);  mul_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_63: "f32[256, 256]" = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
    view_188: "f32[392, 256]" = torch.ops.aten.view.default(div_25, [392, 256]);  div_25 = None
    mm_30: "f32[392, 256]" = torch.ops.aten.mm.default(view_188, permute_63)
    view_189: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_30, [8, 49, 256]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_190: "f32[392, 256]" = torch.ops.aten.view.default(view_189, [392, 256]);  view_189 = None
    add_209: "i64[]" = torch.ops.aten.add.Tensor(primals_327, 1)
    var_mean_34 = torch.ops.aten.var_mean.correction(view_190, [0], correction = 0, keepdim = True)
    getitem_91: "f32[1, 256]" = var_mean_34[0]
    getitem_92: "f32[1, 256]" = var_mean_34[1];  var_mean_34 = None
    add_210: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_91, 1e-05)
    rsqrt_34: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
    sub_42: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_190, getitem_92)
    mul_264: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_34);  sub_42 = None
    squeeze_102: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_92, [0]);  getitem_92 = None
    squeeze_103: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0]);  rsqrt_34 = None
    mul_265: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
    mul_266: "f32[256]" = torch.ops.aten.mul.Tensor(primals_325, 0.9)
    add_211: "f32[256]" = torch.ops.aten.add.Tensor(mul_265, mul_266);  mul_265 = mul_266 = None
    squeeze_104: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_91, [0]);  getitem_91 = None
    mul_267: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0025575447570332);  squeeze_104 = None
    mul_268: "f32[256]" = torch.ops.aten.mul.Tensor(mul_267, 0.1);  mul_267 = None
    mul_269: "f32[256]" = torch.ops.aten.mul.Tensor(primals_326, 0.9)
    add_212: "f32[256]" = torch.ops.aten.add.Tensor(mul_268, mul_269);  mul_268 = mul_269 = None
    mul_270: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_264, primals_118);  mul_264 = None
    add_213: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_270, primals_119);  mul_270 = primals_119 = None
    view_191: "f32[8, 49, 256]" = torch.ops.aten.view.default(add_213, [8, 49, 256]);  add_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_214: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_201, view_191);  add_201 = view_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_64: "f32[256, 512]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
    view_192: "f32[392, 256]" = torch.ops.aten.view.default(add_214, [392, 256])
    mm_31: "f32[392, 512]" = torch.ops.aten.mm.default(view_192, permute_64)
    view_193: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_31, [8, 49, 512]);  mm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_194: "f32[392, 512]" = torch.ops.aten.view.default(view_193, [392, 512]);  view_193 = None
    add_215: "i64[]" = torch.ops.aten.add.Tensor(primals_330, 1)
    var_mean_35 = torch.ops.aten.var_mean.correction(view_194, [0], correction = 0, keepdim = True)
    getitem_93: "f32[1, 512]" = var_mean_35[0]
    getitem_94: "f32[1, 512]" = var_mean_35[1];  var_mean_35 = None
    add_216: "f32[1, 512]" = torch.ops.aten.add.Tensor(getitem_93, 1e-05)
    rsqrt_35: "f32[1, 512]" = torch.ops.aten.rsqrt.default(add_216);  add_216 = None
    sub_43: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_194, getitem_94)
    mul_271: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_35);  sub_43 = None
    squeeze_105: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_94, [0]);  getitem_94 = None
    squeeze_106: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0]);  rsqrt_35 = None
    mul_272: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
    mul_273: "f32[512]" = torch.ops.aten.mul.Tensor(primals_328, 0.9)
    add_217: "f32[512]" = torch.ops.aten.add.Tensor(mul_272, mul_273);  mul_272 = mul_273 = None
    squeeze_107: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_93, [0]);  getitem_93 = None
    mul_274: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0025575447570332);  squeeze_107 = None
    mul_275: "f32[512]" = torch.ops.aten.mul.Tensor(mul_274, 0.1);  mul_274 = None
    mul_276: "f32[512]" = torch.ops.aten.mul.Tensor(primals_329, 0.9)
    add_218: "f32[512]" = torch.ops.aten.add.Tensor(mul_275, mul_276);  mul_275 = mul_276 = None
    mul_277: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_271, primals_121);  mul_271 = None
    add_219: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_277, primals_122);  mul_277 = primals_122 = None
    view_195: "f32[8, 49, 512]" = torch.ops.aten.view.default(add_219, [8, 49, 512]);  add_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_220: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(view_195, 3)
    clamp_min_18: "f32[8, 49, 512]" = torch.ops.aten.clamp_min.default(add_220, 0);  add_220 = None
    clamp_max_18: "f32[8, 49, 512]" = torch.ops.aten.clamp_max.default(clamp_min_18, 6);  clamp_min_18 = None
    mul_278: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_195, clamp_max_18);  clamp_max_18 = None
    div_26: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(mul_278, 6);  mul_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    clone_49: "f32[8, 49, 512]" = torch.ops.aten.clone.default(div_26);  div_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_65: "f32[512, 256]" = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
    view_196: "f32[392, 512]" = torch.ops.aten.view.default(clone_49, [392, 512]);  clone_49 = None
    mm_32: "f32[392, 256]" = torch.ops.aten.mm.default(view_196, permute_65)
    view_197: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_32, [8, 49, 256]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_198: "f32[392, 256]" = torch.ops.aten.view.default(view_197, [392, 256]);  view_197 = None
    add_221: "i64[]" = torch.ops.aten.add.Tensor(primals_333, 1)
    var_mean_36 = torch.ops.aten.var_mean.correction(view_198, [0], correction = 0, keepdim = True)
    getitem_95: "f32[1, 256]" = var_mean_36[0]
    getitem_96: "f32[1, 256]" = var_mean_36[1];  var_mean_36 = None
    add_222: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_95, 1e-05)
    rsqrt_36: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_222);  add_222 = None
    sub_44: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_198, getitem_96)
    mul_279: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_36);  sub_44 = None
    squeeze_108: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_96, [0]);  getitem_96 = None
    squeeze_109: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0]);  rsqrt_36 = None
    mul_280: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
    mul_281: "f32[256]" = torch.ops.aten.mul.Tensor(primals_331, 0.9)
    add_223: "f32[256]" = torch.ops.aten.add.Tensor(mul_280, mul_281);  mul_280 = mul_281 = None
    squeeze_110: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_95, [0]);  getitem_95 = None
    mul_282: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0025575447570332);  squeeze_110 = None
    mul_283: "f32[256]" = torch.ops.aten.mul.Tensor(mul_282, 0.1);  mul_282 = None
    mul_284: "f32[256]" = torch.ops.aten.mul.Tensor(primals_332, 0.9)
    add_224: "f32[256]" = torch.ops.aten.add.Tensor(mul_283, mul_284);  mul_283 = mul_284 = None
    mul_285: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_279, primals_124);  mul_279 = None
    add_225: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_285, primals_125);  mul_285 = primals_125 = None
    view_199: "f32[8, 49, 256]" = torch.ops.aten.view.default(add_225, [8, 49, 256]);  add_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_226: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_214, view_199);  add_214 = view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_66: "f32[256, 512]" = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
    view_200: "f32[392, 256]" = torch.ops.aten.view.default(add_226, [392, 256])
    mm_33: "f32[392, 512]" = torch.ops.aten.mm.default(view_200, permute_66)
    view_201: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_33, [8, 49, 512]);  mm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_202: "f32[392, 512]" = torch.ops.aten.view.default(view_201, [392, 512]);  view_201 = None
    add_227: "i64[]" = torch.ops.aten.add.Tensor(primals_336, 1)
    var_mean_37 = torch.ops.aten.var_mean.correction(view_202, [0], correction = 0, keepdim = True)
    getitem_97: "f32[1, 512]" = var_mean_37[0]
    getitem_98: "f32[1, 512]" = var_mean_37[1];  var_mean_37 = None
    add_228: "f32[1, 512]" = torch.ops.aten.add.Tensor(getitem_97, 1e-05)
    rsqrt_37: "f32[1, 512]" = torch.ops.aten.rsqrt.default(add_228);  add_228 = None
    sub_45: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_202, getitem_98)
    mul_286: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_37);  sub_45 = None
    squeeze_111: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_98, [0]);  getitem_98 = None
    squeeze_112: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0]);  rsqrt_37 = None
    mul_287: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
    mul_288: "f32[512]" = torch.ops.aten.mul.Tensor(primals_334, 0.9)
    add_229: "f32[512]" = torch.ops.aten.add.Tensor(mul_287, mul_288);  mul_287 = mul_288 = None
    squeeze_113: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_97, [0]);  getitem_97 = None
    mul_289: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0025575447570332);  squeeze_113 = None
    mul_290: "f32[512]" = torch.ops.aten.mul.Tensor(mul_289, 0.1);  mul_289 = None
    mul_291: "f32[512]" = torch.ops.aten.mul.Tensor(primals_335, 0.9)
    add_230: "f32[512]" = torch.ops.aten.add.Tensor(mul_290, mul_291);  mul_290 = mul_291 = None
    mul_292: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_286, primals_127);  mul_286 = None
    add_231: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_292, primals_128);  mul_292 = primals_128 = None
    view_203: "f32[8, 49, 512]" = torch.ops.aten.view.default(add_231, [8, 49, 512]);  add_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_204: "f32[8, 49, 8, 64]" = torch.ops.aten.view.default(view_203, [8, 49, 8, -1]);  view_203 = None
    split_with_sizes_8 = torch.ops.aten.split_with_sizes.default(view_204, [16, 16, 32], 3);  view_204 = None
    getitem_99: "f32[8, 49, 8, 16]" = split_with_sizes_8[0]
    getitem_100: "f32[8, 49, 8, 16]" = split_with_sizes_8[1]
    getitem_101: "f32[8, 49, 8, 32]" = split_with_sizes_8[2];  split_with_sizes_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_67: "f32[8, 8, 49, 16]" = torch.ops.aten.permute.default(getitem_99, [0, 2, 1, 3]);  getitem_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_68: "f32[8, 8, 16, 49]" = torch.ops.aten.permute.default(getitem_100, [0, 2, 3, 1]);  getitem_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_69: "f32[8, 8, 49, 32]" = torch.ops.aten.permute.default(getitem_101, [0, 2, 1, 3]);  getitem_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_32: "f32[8, 8, 49, 16]" = torch.ops.aten.expand.default(permute_67, [8, 8, 49, 16]);  permute_67 = None
    clone_50: "f32[8, 8, 49, 16]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    view_205: "f32[64, 49, 16]" = torch.ops.aten.view.default(clone_50, [64, 49, 16]);  clone_50 = None
    expand_33: "f32[8, 8, 16, 49]" = torch.ops.aten.expand.default(permute_68, [8, 8, 16, 49]);  permute_68 = None
    clone_51: "f32[8, 8, 16, 49]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    view_206: "f32[64, 16, 49]" = torch.ops.aten.view.default(clone_51, [64, 16, 49]);  clone_51 = None
    bmm_16: "f32[64, 49, 49]" = torch.ops.aten.bmm.default(view_205, view_206)
    view_207: "f32[8, 8, 49, 49]" = torch.ops.aten.view.default(bmm_16, [8, 8, 49, 49]);  bmm_16 = None
    mul_293: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(view_207, 0.25);  view_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    slice_12: "f32[8, 49]" = torch.ops.aten.slice.Tensor(primals_9, 0, 0, 9223372036854775807);  primals_9 = None
    index_8: "f32[8, 49, 49]" = torch.ops.aten.index.Tensor(slice_12, [None, primals_217]);  slice_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_232: "f32[8, 8, 49, 49]" = torch.ops.aten.add.Tensor(mul_293, index_8);  mul_293 = index_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax_8: "f32[8, 8, 49, 1]" = torch.ops.aten.amax.default(add_232, [-1], True)
    sub_46: "f32[8, 8, 49, 49]" = torch.ops.aten.sub.Tensor(add_232, amax_8);  add_232 = amax_8 = None
    exp_8: "f32[8, 8, 49, 49]" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
    sum_9: "f32[8, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_27: "f32[8, 8, 49, 49]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    alias_8: "f32[8, 8, 49, 49]" = torch.ops.aten.alias.default(div_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_34: "f32[8, 8, 49, 49]" = torch.ops.aten.expand.default(div_27, [8, 8, 49, 49]);  div_27 = None
    view_208: "f32[64, 49, 49]" = torch.ops.aten.view.default(expand_34, [64, 49, 49]);  expand_34 = None
    expand_35: "f32[8, 8, 49, 32]" = torch.ops.aten.expand.default(permute_69, [8, 8, 49, 32]);  permute_69 = None
    clone_52: "f32[8, 8, 49, 32]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
    view_209: "f32[64, 49, 32]" = torch.ops.aten.view.default(clone_52, [64, 49, 32]);  clone_52 = None
    bmm_17: "f32[64, 49, 32]" = torch.ops.aten.bmm.default(view_208, view_209)
    view_210: "f32[8, 8, 49, 32]" = torch.ops.aten.view.default(bmm_17, [8, 8, 49, 32]);  bmm_17 = None
    permute_70: "f32[8, 49, 8, 32]" = torch.ops.aten.permute.default(view_210, [0, 2, 1, 3]);  view_210 = None
    clone_53: "f32[8, 49, 8, 32]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
    view_211: "f32[8, 49, 256]" = torch.ops.aten.view.default(clone_53, [8, 49, 256]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_233: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(view_211, 3)
    clamp_min_19: "f32[8, 49, 256]" = torch.ops.aten.clamp_min.default(add_233, 0);  add_233 = None
    clamp_max_19: "f32[8, 49, 256]" = torch.ops.aten.clamp_max.default(clamp_min_19, 6);  clamp_min_19 = None
    mul_294: "f32[8, 49, 256]" = torch.ops.aten.mul.Tensor(view_211, clamp_max_19);  clamp_max_19 = None
    div_28: "f32[8, 49, 256]" = torch.ops.aten.div.Tensor(mul_294, 6);  mul_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_71: "f32[256, 256]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
    view_212: "f32[392, 256]" = torch.ops.aten.view.default(div_28, [392, 256]);  div_28 = None
    mm_34: "f32[392, 256]" = torch.ops.aten.mm.default(view_212, permute_71)
    view_213: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_34, [8, 49, 256]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_214: "f32[392, 256]" = torch.ops.aten.view.default(view_213, [392, 256]);  view_213 = None
    add_234: "i64[]" = torch.ops.aten.add.Tensor(primals_339, 1)
    var_mean_38 = torch.ops.aten.var_mean.correction(view_214, [0], correction = 0, keepdim = True)
    getitem_102: "f32[1, 256]" = var_mean_38[0]
    getitem_103: "f32[1, 256]" = var_mean_38[1];  var_mean_38 = None
    add_235: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05)
    rsqrt_38: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_235);  add_235 = None
    sub_47: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_214, getitem_103)
    mul_295: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_38);  sub_47 = None
    squeeze_114: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_103, [0]);  getitem_103 = None
    squeeze_115: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0]);  rsqrt_38 = None
    mul_296: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
    mul_297: "f32[256]" = torch.ops.aten.mul.Tensor(primals_337, 0.9)
    add_236: "f32[256]" = torch.ops.aten.add.Tensor(mul_296, mul_297);  mul_296 = mul_297 = None
    squeeze_116: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_102, [0]);  getitem_102 = None
    mul_298: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0025575447570332);  squeeze_116 = None
    mul_299: "f32[256]" = torch.ops.aten.mul.Tensor(mul_298, 0.1);  mul_298 = None
    mul_300: "f32[256]" = torch.ops.aten.mul.Tensor(primals_338, 0.9)
    add_237: "f32[256]" = torch.ops.aten.add.Tensor(mul_299, mul_300);  mul_299 = mul_300 = None
    mul_301: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_295, primals_130);  mul_295 = None
    add_238: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_301, primals_131);  mul_301 = primals_131 = None
    view_215: "f32[8, 49, 256]" = torch.ops.aten.view.default(add_238, [8, 49, 256]);  add_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_239: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_226, view_215);  add_226 = view_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_72: "f32[256, 512]" = torch.ops.aten.permute.default(primals_132, [1, 0]);  primals_132 = None
    view_216: "f32[392, 256]" = torch.ops.aten.view.default(add_239, [392, 256])
    mm_35: "f32[392, 512]" = torch.ops.aten.mm.default(view_216, permute_72)
    view_217: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_35, [8, 49, 512]);  mm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_218: "f32[392, 512]" = torch.ops.aten.view.default(view_217, [392, 512]);  view_217 = None
    add_240: "i64[]" = torch.ops.aten.add.Tensor(primals_342, 1)
    var_mean_39 = torch.ops.aten.var_mean.correction(view_218, [0], correction = 0, keepdim = True)
    getitem_104: "f32[1, 512]" = var_mean_39[0]
    getitem_105: "f32[1, 512]" = var_mean_39[1];  var_mean_39 = None
    add_241: "f32[1, 512]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05)
    rsqrt_39: "f32[1, 512]" = torch.ops.aten.rsqrt.default(add_241);  add_241 = None
    sub_48: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_218, getitem_105)
    mul_302: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_39);  sub_48 = None
    squeeze_117: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_105, [0]);  getitem_105 = None
    squeeze_118: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0]);  rsqrt_39 = None
    mul_303: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
    mul_304: "f32[512]" = torch.ops.aten.mul.Tensor(primals_340, 0.9)
    add_242: "f32[512]" = torch.ops.aten.add.Tensor(mul_303, mul_304);  mul_303 = mul_304 = None
    squeeze_119: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_104, [0]);  getitem_104 = None
    mul_305: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0025575447570332);  squeeze_119 = None
    mul_306: "f32[512]" = torch.ops.aten.mul.Tensor(mul_305, 0.1);  mul_305 = None
    mul_307: "f32[512]" = torch.ops.aten.mul.Tensor(primals_341, 0.9)
    add_243: "f32[512]" = torch.ops.aten.add.Tensor(mul_306, mul_307);  mul_306 = mul_307 = None
    mul_308: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_302, primals_133);  mul_302 = None
    add_244: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_308, primals_134);  mul_308 = primals_134 = None
    view_219: "f32[8, 49, 512]" = torch.ops.aten.view.default(add_244, [8, 49, 512]);  add_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_245: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(view_219, 3)
    clamp_min_20: "f32[8, 49, 512]" = torch.ops.aten.clamp_min.default(add_245, 0);  add_245 = None
    clamp_max_20: "f32[8, 49, 512]" = torch.ops.aten.clamp_max.default(clamp_min_20, 6);  clamp_min_20 = None
    mul_309: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_219, clamp_max_20);  clamp_max_20 = None
    div_29: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(mul_309, 6);  mul_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    clone_54: "f32[8, 49, 512]" = torch.ops.aten.clone.default(div_29);  div_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_73: "f32[512, 256]" = torch.ops.aten.permute.default(primals_135, [1, 0]);  primals_135 = None
    view_220: "f32[392, 512]" = torch.ops.aten.view.default(clone_54, [392, 512]);  clone_54 = None
    mm_36: "f32[392, 256]" = torch.ops.aten.mm.default(view_220, permute_73)
    view_221: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_36, [8, 49, 256]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_222: "f32[392, 256]" = torch.ops.aten.view.default(view_221, [392, 256]);  view_221 = None
    add_246: "i64[]" = torch.ops.aten.add.Tensor(primals_345, 1)
    var_mean_40 = torch.ops.aten.var_mean.correction(view_222, [0], correction = 0, keepdim = True)
    getitem_106: "f32[1, 256]" = var_mean_40[0]
    getitem_107: "f32[1, 256]" = var_mean_40[1];  var_mean_40 = None
    add_247: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05)
    rsqrt_40: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_247);  add_247 = None
    sub_49: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_222, getitem_107)
    mul_310: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_40);  sub_49 = None
    squeeze_120: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_107, [0]);  getitem_107 = None
    squeeze_121: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0]);  rsqrt_40 = None
    mul_311: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
    mul_312: "f32[256]" = torch.ops.aten.mul.Tensor(primals_343, 0.9)
    add_248: "f32[256]" = torch.ops.aten.add.Tensor(mul_311, mul_312);  mul_311 = mul_312 = None
    squeeze_122: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_106, [0]);  getitem_106 = None
    mul_313: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0025575447570332);  squeeze_122 = None
    mul_314: "f32[256]" = torch.ops.aten.mul.Tensor(mul_313, 0.1);  mul_313 = None
    mul_315: "f32[256]" = torch.ops.aten.mul.Tensor(primals_344, 0.9)
    add_249: "f32[256]" = torch.ops.aten.add.Tensor(mul_314, mul_315);  mul_314 = mul_315 = None
    mul_316: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_310, primals_136);  mul_310 = None
    add_250: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_316, primals_137);  mul_316 = primals_137 = None
    view_223: "f32[8, 49, 256]" = torch.ops.aten.view.default(add_250, [8, 49, 256]);  add_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_251: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_239, view_223);  add_239 = view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_74: "f32[256, 1280]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
    view_224: "f32[392, 256]" = torch.ops.aten.view.default(add_251, [392, 256])
    mm_37: "f32[392, 1280]" = torch.ops.aten.mm.default(view_224, permute_74)
    view_225: "f32[8, 49, 1280]" = torch.ops.aten.view.default(mm_37, [8, 49, 1280]);  mm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_226: "f32[392, 1280]" = torch.ops.aten.view.default(view_225, [392, 1280]);  view_225 = None
    add_252: "i64[]" = torch.ops.aten.add.Tensor(primals_348, 1)
    var_mean_41 = torch.ops.aten.var_mean.correction(view_226, [0], correction = 0, keepdim = True)
    getitem_108: "f32[1, 1280]" = var_mean_41[0]
    getitem_109: "f32[1, 1280]" = var_mean_41[1];  var_mean_41 = None
    add_253: "f32[1, 1280]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05)
    rsqrt_41: "f32[1, 1280]" = torch.ops.aten.rsqrt.default(add_253);  add_253 = None
    sub_50: "f32[392, 1280]" = torch.ops.aten.sub.Tensor(view_226, getitem_109)
    mul_317: "f32[392, 1280]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_41);  sub_50 = None
    squeeze_123: "f32[1280]" = torch.ops.aten.squeeze.dims(getitem_109, [0]);  getitem_109 = None
    squeeze_124: "f32[1280]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0]);  rsqrt_41 = None
    mul_318: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
    mul_319: "f32[1280]" = torch.ops.aten.mul.Tensor(primals_346, 0.9)
    add_254: "f32[1280]" = torch.ops.aten.add.Tensor(mul_318, mul_319);  mul_318 = mul_319 = None
    squeeze_125: "f32[1280]" = torch.ops.aten.squeeze.dims(getitem_108, [0]);  getitem_108 = None
    mul_320: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0025575447570332);  squeeze_125 = None
    mul_321: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_320, 0.1);  mul_320 = None
    mul_322: "f32[1280]" = torch.ops.aten.mul.Tensor(primals_347, 0.9)
    add_255: "f32[1280]" = torch.ops.aten.add.Tensor(mul_321, mul_322);  mul_321 = mul_322 = None
    mul_323: "f32[392, 1280]" = torch.ops.aten.mul.Tensor(mul_317, primals_139);  mul_317 = None
    add_256: "f32[392, 1280]" = torch.ops.aten.add.Tensor(mul_323, primals_140);  mul_323 = primals_140 = None
    view_227: "f32[8, 49, 1280]" = torch.ops.aten.view.default(add_256, [8, 49, 1280]);  add_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:331, code: k, v = self.kv(x).view(B, N, self.num_heads, -1).split([self.key_dim, self.val_dim], dim=3)
    view_228: "f32[8, 49, 16, 80]" = torch.ops.aten.view.default(view_227, [8, 49, 16, -1]);  view_227 = None
    split_with_sizes_9 = torch.ops.aten.split_with_sizes.default(view_228, [16, 64], 3);  view_228 = None
    getitem_110: "f32[8, 49, 16, 16]" = split_with_sizes_9[0]
    getitem_111: "f32[8, 49, 16, 64]" = split_with_sizes_9[1];  split_with_sizes_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:332, code: k = k.permute(0, 2, 3, 1)  # BHCN
    permute_75: "f32[8, 16, 16, 49]" = torch.ops.aten.permute.default(getitem_110, [0, 2, 3, 1]);  getitem_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:333, code: v = v.permute(0, 2, 1, 3)  # BHNC
    permute_76: "f32[8, 16, 49, 64]" = torch.ops.aten.permute.default(getitem_111, [0, 2, 1, 3]);  getitem_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:157, code: x = x.view(B, self.resolution[0], self.resolution[1], C)
    view_229: "f32[8, 7, 7, 256]" = torch.ops.aten.view.default(add_251, [8, 7, 7, 256]);  add_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:161, code: x = x[:, ::self.stride, ::self.stride]
    slice_13: "f32[8, 7, 7, 256]" = torch.ops.aten.slice.Tensor(view_229, 0, 0, 9223372036854775807);  view_229 = None
    slice_14: "f32[8, 4, 7, 256]" = torch.ops.aten.slice.Tensor(slice_13, 1, 0, 9223372036854775807, 2);  slice_13 = None
    slice_15: "f32[8, 4, 4, 256]" = torch.ops.aten.slice.Tensor(slice_14, 2, 0, 9223372036854775807, 2);  slice_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:162, code: return x.reshape(B, -1, C)
    clone_55: "f32[8, 4, 4, 256]" = torch.ops.aten.clone.default(slice_15, memory_format = torch.contiguous_format);  slice_15 = None
    view_230: "f32[8, 16, 256]" = torch.ops.aten.view.default(clone_55, [8, 16, 256]);  clone_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_77: "f32[256, 256]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
    view_231: "f32[128, 256]" = torch.ops.aten.view.default(view_230, [128, 256]);  view_230 = None
    mm_38: "f32[128, 256]" = torch.ops.aten.mm.default(view_231, permute_77)
    view_232: "f32[8, 16, 256]" = torch.ops.aten.view.default(mm_38, [8, 16, 256]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_233: "f32[128, 256]" = torch.ops.aten.view.default(view_232, [128, 256]);  view_232 = None
    add_257: "i64[]" = torch.ops.aten.add.Tensor(primals_351, 1)
    var_mean_42 = torch.ops.aten.var_mean.correction(view_233, [0], correction = 0, keepdim = True)
    getitem_112: "f32[1, 256]" = var_mean_42[0]
    getitem_113: "f32[1, 256]" = var_mean_42[1];  var_mean_42 = None
    add_258: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05)
    rsqrt_42: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_258);  add_258 = None
    sub_51: "f32[128, 256]" = torch.ops.aten.sub.Tensor(view_233, getitem_113)
    mul_324: "f32[128, 256]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_42);  sub_51 = None
    squeeze_126: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_113, [0]);  getitem_113 = None
    squeeze_127: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0]);  rsqrt_42 = None
    mul_325: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
    mul_326: "f32[256]" = torch.ops.aten.mul.Tensor(primals_349, 0.9)
    add_259: "f32[256]" = torch.ops.aten.add.Tensor(mul_325, mul_326);  mul_325 = mul_326 = None
    squeeze_128: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_112, [0]);  getitem_112 = None
    mul_327: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0078740157480315);  squeeze_128 = None
    mul_328: "f32[256]" = torch.ops.aten.mul.Tensor(mul_327, 0.1);  mul_327 = None
    mul_329: "f32[256]" = torch.ops.aten.mul.Tensor(primals_350, 0.9)
    add_260: "f32[256]" = torch.ops.aten.add.Tensor(mul_328, mul_329);  mul_328 = mul_329 = None
    mul_330: "f32[128, 256]" = torch.ops.aten.mul.Tensor(mul_324, primals_142);  mul_324 = None
    add_261: "f32[128, 256]" = torch.ops.aten.add.Tensor(mul_330, primals_143);  mul_330 = primals_143 = None
    view_234: "f32[8, 16, 256]" = torch.ops.aten.view.default(add_261, [8, 16, 256]);  add_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:334, code: q = self.q(x).view(B, -1, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
    view_235: "f32[8, 16, 16, 16]" = torch.ops.aten.view.default(view_234, [8, -1, 16, 16]);  view_234 = None
    permute_78: "f32[8, 16, 16, 16]" = torch.ops.aten.permute.default(view_235, [0, 2, 1, 3]);  view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:336, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_36: "f32[8, 16, 16, 16]" = torch.ops.aten.expand.default(permute_78, [8, 16, 16, 16]);  permute_78 = None
    clone_56: "f32[8, 16, 16, 16]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
    view_236: "f32[128, 16, 16]" = torch.ops.aten.view.default(clone_56, [128, 16, 16]);  clone_56 = None
    expand_37: "f32[8, 16, 16, 49]" = torch.ops.aten.expand.default(permute_75, [8, 16, 16, 49]);  permute_75 = None
    clone_57: "f32[8, 16, 16, 49]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
    view_237: "f32[128, 16, 49]" = torch.ops.aten.view.default(clone_57, [128, 16, 49]);  clone_57 = None
    bmm_18: "f32[128, 16, 49]" = torch.ops.aten.bmm.default(view_236, view_237)
    view_238: "f32[8, 16, 16, 49]" = torch.ops.aten.view.default(bmm_18, [8, 16, 16, 49]);  bmm_18 = None
    mul_331: "f32[8, 16, 16, 49]" = torch.ops.aten.mul.Tensor(view_238, 0.25);  view_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:311, code: return self.attention_biases[:, self.attention_bias_idxs]
    slice_16: "f32[16, 49]" = torch.ops.aten.slice.Tensor(primals_10, 0, 0, 9223372036854775807);  primals_10 = None
    index_9: "f32[16, 16, 49]" = torch.ops.aten.index.Tensor(slice_16, [None, primals_218]);  slice_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:336, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_262: "f32[8, 16, 16, 49]" = torch.ops.aten.add.Tensor(mul_331, index_9);  mul_331 = index_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:337, code: attn = attn.softmax(dim=-1)
    amax_9: "f32[8, 16, 16, 1]" = torch.ops.aten.amax.default(add_262, [-1], True)
    sub_52: "f32[8, 16, 16, 49]" = torch.ops.aten.sub.Tensor(add_262, amax_9);  add_262 = amax_9 = None
    exp_9: "f32[8, 16, 16, 49]" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
    sum_10: "f32[8, 16, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_30: "f32[8, 16, 16, 49]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    alias_9: "f32[8, 16, 16, 49]" = torch.ops.aten.alias.default(div_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:339, code: x = (attn @ v).transpose(1, 2).reshape(B, -1, self.val_attn_dim)
    expand_38: "f32[8, 16, 16, 49]" = torch.ops.aten.expand.default(div_30, [8, 16, 16, 49]);  div_30 = None
    view_239: "f32[128, 16, 49]" = torch.ops.aten.view.default(expand_38, [128, 16, 49]);  expand_38 = None
    expand_39: "f32[8, 16, 49, 64]" = torch.ops.aten.expand.default(permute_76, [8, 16, 49, 64]);  permute_76 = None
    clone_58: "f32[8, 16, 49, 64]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
    view_240: "f32[128, 49, 64]" = torch.ops.aten.view.default(clone_58, [128, 49, 64]);  clone_58 = None
    bmm_19: "f32[128, 16, 64]" = torch.ops.aten.bmm.default(view_239, view_240)
    view_241: "f32[8, 16, 16, 64]" = torch.ops.aten.view.default(bmm_19, [8, 16, 16, 64]);  bmm_19 = None
    permute_79: "f32[8, 16, 16, 64]" = torch.ops.aten.permute.default(view_241, [0, 2, 1, 3]);  view_241 = None
    clone_59: "f32[8, 16, 16, 64]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
    view_242: "f32[8, 16, 1024]" = torch.ops.aten.view.default(clone_59, [8, 16, 1024]);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:340, code: x = self.proj(x)
    add_263: "f32[8, 16, 1024]" = torch.ops.aten.add.Tensor(view_242, 3)
    clamp_min_21: "f32[8, 16, 1024]" = torch.ops.aten.clamp_min.default(add_263, 0);  add_263 = None
    clamp_max_21: "f32[8, 16, 1024]" = torch.ops.aten.clamp_max.default(clamp_min_21, 6);  clamp_min_21 = None
    mul_332: "f32[8, 16, 1024]" = torch.ops.aten.mul.Tensor(view_242, clamp_max_21);  clamp_max_21 = None
    div_31: "f32[8, 16, 1024]" = torch.ops.aten.div.Tensor(mul_332, 6);  mul_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_80: "f32[1024, 384]" = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
    view_243: "f32[128, 1024]" = torch.ops.aten.view.default(div_31, [128, 1024]);  div_31 = None
    mm_39: "f32[128, 384]" = torch.ops.aten.mm.default(view_243, permute_80)
    view_244: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_39, [8, 16, 384]);  mm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_245: "f32[128, 384]" = torch.ops.aten.view.default(view_244, [128, 384]);  view_244 = None
    add_264: "i64[]" = torch.ops.aten.add.Tensor(primals_354, 1)
    var_mean_43 = torch.ops.aten.var_mean.correction(view_245, [0], correction = 0, keepdim = True)
    getitem_114: "f32[1, 384]" = var_mean_43[0]
    getitem_115: "f32[1, 384]" = var_mean_43[1];  var_mean_43 = None
    add_265: "f32[1, 384]" = torch.ops.aten.add.Tensor(getitem_114, 1e-05)
    rsqrt_43: "f32[1, 384]" = torch.ops.aten.rsqrt.default(add_265);  add_265 = None
    sub_53: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_245, getitem_115)
    mul_333: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_43);  sub_53 = None
    squeeze_129: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_115, [0]);  getitem_115 = None
    squeeze_130: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0]);  rsqrt_43 = None
    mul_334: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
    mul_335: "f32[384]" = torch.ops.aten.mul.Tensor(primals_352, 0.9)
    add_266: "f32[384]" = torch.ops.aten.add.Tensor(mul_334, mul_335);  mul_334 = mul_335 = None
    squeeze_131: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_114, [0]);  getitem_114 = None
    mul_336: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0078740157480315);  squeeze_131 = None
    mul_337: "f32[384]" = torch.ops.aten.mul.Tensor(mul_336, 0.1);  mul_336 = None
    mul_338: "f32[384]" = torch.ops.aten.mul.Tensor(primals_353, 0.9)
    add_267: "f32[384]" = torch.ops.aten.add.Tensor(mul_337, mul_338);  mul_337 = mul_338 = None
    mul_339: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_333, primals_145);  mul_333 = None
    add_268: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_339, primals_146);  mul_339 = primals_146 = None
    view_246: "f32[8, 16, 384]" = torch.ops.aten.view.default(add_268, [8, 16, 384]);  add_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_81: "f32[384, 768]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
    view_247: "f32[128, 384]" = torch.ops.aten.view.default(view_246, [128, 384])
    mm_40: "f32[128, 768]" = torch.ops.aten.mm.default(view_247, permute_81)
    view_248: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_40, [8, 16, 768]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_249: "f32[128, 768]" = torch.ops.aten.view.default(view_248, [128, 768]);  view_248 = None
    add_269: "i64[]" = torch.ops.aten.add.Tensor(primals_357, 1)
    var_mean_44 = torch.ops.aten.var_mean.correction(view_249, [0], correction = 0, keepdim = True)
    getitem_116: "f32[1, 768]" = var_mean_44[0]
    getitem_117: "f32[1, 768]" = var_mean_44[1];  var_mean_44 = None
    add_270: "f32[1, 768]" = torch.ops.aten.add.Tensor(getitem_116, 1e-05)
    rsqrt_44: "f32[1, 768]" = torch.ops.aten.rsqrt.default(add_270);  add_270 = None
    sub_54: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_249, getitem_117)
    mul_340: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_44);  sub_54 = None
    squeeze_132: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_117, [0]);  getitem_117 = None
    squeeze_133: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0]);  rsqrt_44 = None
    mul_341: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1)
    mul_342: "f32[768]" = torch.ops.aten.mul.Tensor(primals_355, 0.9)
    add_271: "f32[768]" = torch.ops.aten.add.Tensor(mul_341, mul_342);  mul_341 = mul_342 = None
    squeeze_134: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_116, [0]);  getitem_116 = None
    mul_343: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_134, 1.0078740157480315);  squeeze_134 = None
    mul_344: "f32[768]" = torch.ops.aten.mul.Tensor(mul_343, 0.1);  mul_343 = None
    mul_345: "f32[768]" = torch.ops.aten.mul.Tensor(primals_356, 0.9)
    add_272: "f32[768]" = torch.ops.aten.add.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
    mul_346: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_340, primals_148);  mul_340 = None
    add_273: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_346, primals_149);  mul_346 = primals_149 = None
    view_250: "f32[8, 16, 768]" = torch.ops.aten.view.default(add_273, [8, 16, 768]);  add_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_274: "f32[8, 16, 768]" = torch.ops.aten.add.Tensor(view_250, 3)
    clamp_min_22: "f32[8, 16, 768]" = torch.ops.aten.clamp_min.default(add_274, 0);  add_274 = None
    clamp_max_22: "f32[8, 16, 768]" = torch.ops.aten.clamp_max.default(clamp_min_22, 6);  clamp_min_22 = None
    mul_347: "f32[8, 16, 768]" = torch.ops.aten.mul.Tensor(view_250, clamp_max_22);  clamp_max_22 = None
    div_32: "f32[8, 16, 768]" = torch.ops.aten.div.Tensor(mul_347, 6);  mul_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    clone_60: "f32[8, 16, 768]" = torch.ops.aten.clone.default(div_32);  div_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_82: "f32[768, 384]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
    view_251: "f32[128, 768]" = torch.ops.aten.view.default(clone_60, [128, 768]);  clone_60 = None
    mm_41: "f32[128, 384]" = torch.ops.aten.mm.default(view_251, permute_82)
    view_252: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_41, [8, 16, 384]);  mm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_253: "f32[128, 384]" = torch.ops.aten.view.default(view_252, [128, 384]);  view_252 = None
    add_275: "i64[]" = torch.ops.aten.add.Tensor(primals_360, 1)
    var_mean_45 = torch.ops.aten.var_mean.correction(view_253, [0], correction = 0, keepdim = True)
    getitem_118: "f32[1, 384]" = var_mean_45[0]
    getitem_119: "f32[1, 384]" = var_mean_45[1];  var_mean_45 = None
    add_276: "f32[1, 384]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05)
    rsqrt_45: "f32[1, 384]" = torch.ops.aten.rsqrt.default(add_276);  add_276 = None
    sub_55: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_253, getitem_119)
    mul_348: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_45);  sub_55 = None
    squeeze_135: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_119, [0]);  getitem_119 = None
    squeeze_136: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0]);  rsqrt_45 = None
    mul_349: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1)
    mul_350: "f32[384]" = torch.ops.aten.mul.Tensor(primals_358, 0.9)
    add_277: "f32[384]" = torch.ops.aten.add.Tensor(mul_349, mul_350);  mul_349 = mul_350 = None
    squeeze_137: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_118, [0]);  getitem_118 = None
    mul_351: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_137, 1.0078740157480315);  squeeze_137 = None
    mul_352: "f32[384]" = torch.ops.aten.mul.Tensor(mul_351, 0.1);  mul_351 = None
    mul_353: "f32[384]" = torch.ops.aten.mul.Tensor(primals_359, 0.9)
    add_278: "f32[384]" = torch.ops.aten.add.Tensor(mul_352, mul_353);  mul_352 = mul_353 = None
    mul_354: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_348, primals_151);  mul_348 = None
    add_279: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_354, primals_152);  mul_354 = primals_152 = None
    view_254: "f32[8, 16, 384]" = torch.ops.aten.view.default(add_279, [8, 16, 384]);  add_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:415, code: x = x + self.drop_path(self.mlp(x))
    add_280: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(view_246, view_254);  view_246 = view_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_83: "f32[384, 768]" = torch.ops.aten.permute.default(primals_153, [1, 0]);  primals_153 = None
    view_255: "f32[128, 384]" = torch.ops.aten.view.default(add_280, [128, 384])
    mm_42: "f32[128, 768]" = torch.ops.aten.mm.default(view_255, permute_83)
    view_256: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_42, [8, 16, 768]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_257: "f32[128, 768]" = torch.ops.aten.view.default(view_256, [128, 768]);  view_256 = None
    add_281: "i64[]" = torch.ops.aten.add.Tensor(primals_363, 1)
    var_mean_46 = torch.ops.aten.var_mean.correction(view_257, [0], correction = 0, keepdim = True)
    getitem_120: "f32[1, 768]" = var_mean_46[0]
    getitem_121: "f32[1, 768]" = var_mean_46[1];  var_mean_46 = None
    add_282: "f32[1, 768]" = torch.ops.aten.add.Tensor(getitem_120, 1e-05)
    rsqrt_46: "f32[1, 768]" = torch.ops.aten.rsqrt.default(add_282);  add_282 = None
    sub_56: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_257, getitem_121)
    mul_355: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_46);  sub_56 = None
    squeeze_138: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_121, [0]);  getitem_121 = None
    squeeze_139: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0]);  rsqrt_46 = None
    mul_356: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_138, 0.1)
    mul_357: "f32[768]" = torch.ops.aten.mul.Tensor(primals_361, 0.9)
    add_283: "f32[768]" = torch.ops.aten.add.Tensor(mul_356, mul_357);  mul_356 = mul_357 = None
    squeeze_140: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_120, [0]);  getitem_120 = None
    mul_358: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_140, 1.0078740157480315);  squeeze_140 = None
    mul_359: "f32[768]" = torch.ops.aten.mul.Tensor(mul_358, 0.1);  mul_358 = None
    mul_360: "f32[768]" = torch.ops.aten.mul.Tensor(primals_362, 0.9)
    add_284: "f32[768]" = torch.ops.aten.add.Tensor(mul_359, mul_360);  mul_359 = mul_360 = None
    mul_361: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_355, primals_154);  mul_355 = None
    add_285: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_361, primals_155);  mul_361 = primals_155 = None
    view_258: "f32[8, 16, 768]" = torch.ops.aten.view.default(add_285, [8, 16, 768]);  add_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_259: "f32[8, 16, 12, 64]" = torch.ops.aten.view.default(view_258, [8, 16, 12, -1]);  view_258 = None
    split_with_sizes_10 = torch.ops.aten.split_with_sizes.default(view_259, [16, 16, 32], 3);  view_259 = None
    getitem_122: "f32[8, 16, 12, 16]" = split_with_sizes_10[0]
    getitem_123: "f32[8, 16, 12, 16]" = split_with_sizes_10[1]
    getitem_124: "f32[8, 16, 12, 32]" = split_with_sizes_10[2];  split_with_sizes_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_84: "f32[8, 12, 16, 16]" = torch.ops.aten.permute.default(getitem_122, [0, 2, 1, 3]);  getitem_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_85: "f32[8, 12, 16, 16]" = torch.ops.aten.permute.default(getitem_123, [0, 2, 3, 1]);  getitem_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_86: "f32[8, 12, 16, 32]" = torch.ops.aten.permute.default(getitem_124, [0, 2, 1, 3]);  getitem_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_40: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(permute_84, [8, 12, 16, 16]);  permute_84 = None
    clone_61: "f32[8, 12, 16, 16]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
    view_260: "f32[96, 16, 16]" = torch.ops.aten.view.default(clone_61, [96, 16, 16]);  clone_61 = None
    expand_41: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(permute_85, [8, 12, 16, 16]);  permute_85 = None
    clone_62: "f32[8, 12, 16, 16]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
    view_261: "f32[96, 16, 16]" = torch.ops.aten.view.default(clone_62, [96, 16, 16]);  clone_62 = None
    bmm_20: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(view_260, view_261)
    view_262: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_20, [8, 12, 16, 16]);  bmm_20 = None
    mul_362: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(view_262, 0.25);  view_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    slice_17: "f32[12, 16]" = torch.ops.aten.slice.Tensor(primals_11, 0, 0, 9223372036854775807);  primals_11 = None
    index_10: "f32[12, 16, 16]" = torch.ops.aten.index.Tensor(slice_17, [None, primals_219]);  slice_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_286: "f32[8, 12, 16, 16]" = torch.ops.aten.add.Tensor(mul_362, index_10);  mul_362 = index_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax_10: "f32[8, 12, 16, 1]" = torch.ops.aten.amax.default(add_286, [-1], True)
    sub_57: "f32[8, 12, 16, 16]" = torch.ops.aten.sub.Tensor(add_286, amax_10);  add_286 = amax_10 = None
    exp_10: "f32[8, 12, 16, 16]" = torch.ops.aten.exp.default(sub_57);  sub_57 = None
    sum_11: "f32[8, 12, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_33: "f32[8, 12, 16, 16]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    alias_10: "f32[8, 12, 16, 16]" = torch.ops.aten.alias.default(div_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_42: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(div_33, [8, 12, 16, 16]);  div_33 = None
    view_263: "f32[96, 16, 16]" = torch.ops.aten.view.default(expand_42, [96, 16, 16]);  expand_42 = None
    expand_43: "f32[8, 12, 16, 32]" = torch.ops.aten.expand.default(permute_86, [8, 12, 16, 32]);  permute_86 = None
    clone_63: "f32[8, 12, 16, 32]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
    view_264: "f32[96, 16, 32]" = torch.ops.aten.view.default(clone_63, [96, 16, 32]);  clone_63 = None
    bmm_21: "f32[96, 16, 32]" = torch.ops.aten.bmm.default(view_263, view_264)
    view_265: "f32[8, 12, 16, 32]" = torch.ops.aten.view.default(bmm_21, [8, 12, 16, 32]);  bmm_21 = None
    permute_87: "f32[8, 16, 12, 32]" = torch.ops.aten.permute.default(view_265, [0, 2, 1, 3]);  view_265 = None
    clone_64: "f32[8, 16, 12, 32]" = torch.ops.aten.clone.default(permute_87, memory_format = torch.contiguous_format);  permute_87 = None
    view_266: "f32[8, 16, 384]" = torch.ops.aten.view.default(clone_64, [8, 16, 384]);  clone_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_287: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(view_266, 3)
    clamp_min_23: "f32[8, 16, 384]" = torch.ops.aten.clamp_min.default(add_287, 0);  add_287 = None
    clamp_max_23: "f32[8, 16, 384]" = torch.ops.aten.clamp_max.default(clamp_min_23, 6);  clamp_min_23 = None
    mul_363: "f32[8, 16, 384]" = torch.ops.aten.mul.Tensor(view_266, clamp_max_23);  clamp_max_23 = None
    div_34: "f32[8, 16, 384]" = torch.ops.aten.div.Tensor(mul_363, 6);  mul_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_88: "f32[384, 384]" = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
    view_267: "f32[128, 384]" = torch.ops.aten.view.default(div_34, [128, 384]);  div_34 = None
    mm_43: "f32[128, 384]" = torch.ops.aten.mm.default(view_267, permute_88)
    view_268: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_43, [8, 16, 384]);  mm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_269: "f32[128, 384]" = torch.ops.aten.view.default(view_268, [128, 384]);  view_268 = None
    add_288: "i64[]" = torch.ops.aten.add.Tensor(primals_366, 1)
    var_mean_47 = torch.ops.aten.var_mean.correction(view_269, [0], correction = 0, keepdim = True)
    getitem_125: "f32[1, 384]" = var_mean_47[0]
    getitem_126: "f32[1, 384]" = var_mean_47[1];  var_mean_47 = None
    add_289: "f32[1, 384]" = torch.ops.aten.add.Tensor(getitem_125, 1e-05)
    rsqrt_47: "f32[1, 384]" = torch.ops.aten.rsqrt.default(add_289);  add_289 = None
    sub_58: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_269, getitem_126)
    mul_364: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_47);  sub_58 = None
    squeeze_141: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_126, [0]);  getitem_126 = None
    squeeze_142: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0]);  rsqrt_47 = None
    mul_365: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_141, 0.1)
    mul_366: "f32[384]" = torch.ops.aten.mul.Tensor(primals_364, 0.9)
    add_290: "f32[384]" = torch.ops.aten.add.Tensor(mul_365, mul_366);  mul_365 = mul_366 = None
    squeeze_143: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_125, [0]);  getitem_125 = None
    mul_367: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_143, 1.0078740157480315);  squeeze_143 = None
    mul_368: "f32[384]" = torch.ops.aten.mul.Tensor(mul_367, 0.1);  mul_367 = None
    mul_369: "f32[384]" = torch.ops.aten.mul.Tensor(primals_365, 0.9)
    add_291: "f32[384]" = torch.ops.aten.add.Tensor(mul_368, mul_369);  mul_368 = mul_369 = None
    mul_370: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_364, primals_157);  mul_364 = None
    add_292: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_370, primals_158);  mul_370 = primals_158 = None
    view_270: "f32[8, 16, 384]" = torch.ops.aten.view.default(add_292, [8, 16, 384]);  add_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_293: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_280, view_270);  add_280 = view_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_89: "f32[384, 768]" = torch.ops.aten.permute.default(primals_159, [1, 0]);  primals_159 = None
    view_271: "f32[128, 384]" = torch.ops.aten.view.default(add_293, [128, 384])
    mm_44: "f32[128, 768]" = torch.ops.aten.mm.default(view_271, permute_89)
    view_272: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_44, [8, 16, 768]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_273: "f32[128, 768]" = torch.ops.aten.view.default(view_272, [128, 768]);  view_272 = None
    add_294: "i64[]" = torch.ops.aten.add.Tensor(primals_369, 1)
    var_mean_48 = torch.ops.aten.var_mean.correction(view_273, [0], correction = 0, keepdim = True)
    getitem_127: "f32[1, 768]" = var_mean_48[0]
    getitem_128: "f32[1, 768]" = var_mean_48[1];  var_mean_48 = None
    add_295: "f32[1, 768]" = torch.ops.aten.add.Tensor(getitem_127, 1e-05)
    rsqrt_48: "f32[1, 768]" = torch.ops.aten.rsqrt.default(add_295);  add_295 = None
    sub_59: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_273, getitem_128)
    mul_371: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_48);  sub_59 = None
    squeeze_144: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_128, [0]);  getitem_128 = None
    squeeze_145: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0]);  rsqrt_48 = None
    mul_372: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_144, 0.1)
    mul_373: "f32[768]" = torch.ops.aten.mul.Tensor(primals_367, 0.9)
    add_296: "f32[768]" = torch.ops.aten.add.Tensor(mul_372, mul_373);  mul_372 = mul_373 = None
    squeeze_146: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_127, [0]);  getitem_127 = None
    mul_374: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_146, 1.0078740157480315);  squeeze_146 = None
    mul_375: "f32[768]" = torch.ops.aten.mul.Tensor(mul_374, 0.1);  mul_374 = None
    mul_376: "f32[768]" = torch.ops.aten.mul.Tensor(primals_368, 0.9)
    add_297: "f32[768]" = torch.ops.aten.add.Tensor(mul_375, mul_376);  mul_375 = mul_376 = None
    mul_377: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_371, primals_160);  mul_371 = None
    add_298: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_377, primals_161);  mul_377 = primals_161 = None
    view_274: "f32[8, 16, 768]" = torch.ops.aten.view.default(add_298, [8, 16, 768]);  add_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_299: "f32[8, 16, 768]" = torch.ops.aten.add.Tensor(view_274, 3)
    clamp_min_24: "f32[8, 16, 768]" = torch.ops.aten.clamp_min.default(add_299, 0);  add_299 = None
    clamp_max_24: "f32[8, 16, 768]" = torch.ops.aten.clamp_max.default(clamp_min_24, 6);  clamp_min_24 = None
    mul_378: "f32[8, 16, 768]" = torch.ops.aten.mul.Tensor(view_274, clamp_max_24);  clamp_max_24 = None
    div_35: "f32[8, 16, 768]" = torch.ops.aten.div.Tensor(mul_378, 6);  mul_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    clone_65: "f32[8, 16, 768]" = torch.ops.aten.clone.default(div_35);  div_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_90: "f32[768, 384]" = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
    view_275: "f32[128, 768]" = torch.ops.aten.view.default(clone_65, [128, 768]);  clone_65 = None
    mm_45: "f32[128, 384]" = torch.ops.aten.mm.default(view_275, permute_90)
    view_276: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_45, [8, 16, 384]);  mm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_277: "f32[128, 384]" = torch.ops.aten.view.default(view_276, [128, 384]);  view_276 = None
    add_300: "i64[]" = torch.ops.aten.add.Tensor(primals_372, 1)
    var_mean_49 = torch.ops.aten.var_mean.correction(view_277, [0], correction = 0, keepdim = True)
    getitem_129: "f32[1, 384]" = var_mean_49[0]
    getitem_130: "f32[1, 384]" = var_mean_49[1];  var_mean_49 = None
    add_301: "f32[1, 384]" = torch.ops.aten.add.Tensor(getitem_129, 1e-05)
    rsqrt_49: "f32[1, 384]" = torch.ops.aten.rsqrt.default(add_301);  add_301 = None
    sub_60: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_277, getitem_130)
    mul_379: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_49);  sub_60 = None
    squeeze_147: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_130, [0]);  getitem_130 = None
    squeeze_148: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0]);  rsqrt_49 = None
    mul_380: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_147, 0.1)
    mul_381: "f32[384]" = torch.ops.aten.mul.Tensor(primals_370, 0.9)
    add_302: "f32[384]" = torch.ops.aten.add.Tensor(mul_380, mul_381);  mul_380 = mul_381 = None
    squeeze_149: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_129, [0]);  getitem_129 = None
    mul_382: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_149, 1.0078740157480315);  squeeze_149 = None
    mul_383: "f32[384]" = torch.ops.aten.mul.Tensor(mul_382, 0.1);  mul_382 = None
    mul_384: "f32[384]" = torch.ops.aten.mul.Tensor(primals_371, 0.9)
    add_303: "f32[384]" = torch.ops.aten.add.Tensor(mul_383, mul_384);  mul_383 = mul_384 = None
    mul_385: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_379, primals_163);  mul_379 = None
    add_304: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_385, primals_164);  mul_385 = primals_164 = None
    view_278: "f32[8, 16, 384]" = torch.ops.aten.view.default(add_304, [8, 16, 384]);  add_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_305: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_293, view_278);  add_293 = view_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_91: "f32[384, 768]" = torch.ops.aten.permute.default(primals_165, [1, 0]);  primals_165 = None
    view_279: "f32[128, 384]" = torch.ops.aten.view.default(add_305, [128, 384])
    mm_46: "f32[128, 768]" = torch.ops.aten.mm.default(view_279, permute_91)
    view_280: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_46, [8, 16, 768]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_281: "f32[128, 768]" = torch.ops.aten.view.default(view_280, [128, 768]);  view_280 = None
    add_306: "i64[]" = torch.ops.aten.add.Tensor(primals_375, 1)
    var_mean_50 = torch.ops.aten.var_mean.correction(view_281, [0], correction = 0, keepdim = True)
    getitem_131: "f32[1, 768]" = var_mean_50[0]
    getitem_132: "f32[1, 768]" = var_mean_50[1];  var_mean_50 = None
    add_307: "f32[1, 768]" = torch.ops.aten.add.Tensor(getitem_131, 1e-05)
    rsqrt_50: "f32[1, 768]" = torch.ops.aten.rsqrt.default(add_307);  add_307 = None
    sub_61: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_281, getitem_132)
    mul_386: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_50);  sub_61 = None
    squeeze_150: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_132, [0]);  getitem_132 = None
    squeeze_151: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0]);  rsqrt_50 = None
    mul_387: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_150, 0.1)
    mul_388: "f32[768]" = torch.ops.aten.mul.Tensor(primals_373, 0.9)
    add_308: "f32[768]" = torch.ops.aten.add.Tensor(mul_387, mul_388);  mul_387 = mul_388 = None
    squeeze_152: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_131, [0]);  getitem_131 = None
    mul_389: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_152, 1.0078740157480315);  squeeze_152 = None
    mul_390: "f32[768]" = torch.ops.aten.mul.Tensor(mul_389, 0.1);  mul_389 = None
    mul_391: "f32[768]" = torch.ops.aten.mul.Tensor(primals_374, 0.9)
    add_309: "f32[768]" = torch.ops.aten.add.Tensor(mul_390, mul_391);  mul_390 = mul_391 = None
    mul_392: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_386, primals_166);  mul_386 = None
    add_310: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_392, primals_167);  mul_392 = primals_167 = None
    view_282: "f32[8, 16, 768]" = torch.ops.aten.view.default(add_310, [8, 16, 768]);  add_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_283: "f32[8, 16, 12, 64]" = torch.ops.aten.view.default(view_282, [8, 16, 12, -1]);  view_282 = None
    split_with_sizes_11 = torch.ops.aten.split_with_sizes.default(view_283, [16, 16, 32], 3);  view_283 = None
    getitem_133: "f32[8, 16, 12, 16]" = split_with_sizes_11[0]
    getitem_134: "f32[8, 16, 12, 16]" = split_with_sizes_11[1]
    getitem_135: "f32[8, 16, 12, 32]" = split_with_sizes_11[2];  split_with_sizes_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_92: "f32[8, 12, 16, 16]" = torch.ops.aten.permute.default(getitem_133, [0, 2, 1, 3]);  getitem_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_93: "f32[8, 12, 16, 16]" = torch.ops.aten.permute.default(getitem_134, [0, 2, 3, 1]);  getitem_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_94: "f32[8, 12, 16, 32]" = torch.ops.aten.permute.default(getitem_135, [0, 2, 1, 3]);  getitem_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_44: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(permute_92, [8, 12, 16, 16]);  permute_92 = None
    clone_66: "f32[8, 12, 16, 16]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
    view_284: "f32[96, 16, 16]" = torch.ops.aten.view.default(clone_66, [96, 16, 16]);  clone_66 = None
    expand_45: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(permute_93, [8, 12, 16, 16]);  permute_93 = None
    clone_67: "f32[8, 12, 16, 16]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
    view_285: "f32[96, 16, 16]" = torch.ops.aten.view.default(clone_67, [96, 16, 16]);  clone_67 = None
    bmm_22: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(view_284, view_285)
    view_286: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_22, [8, 12, 16, 16]);  bmm_22 = None
    mul_393: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(view_286, 0.25);  view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    slice_18: "f32[12, 16]" = torch.ops.aten.slice.Tensor(primals_12, 0, 0, 9223372036854775807);  primals_12 = None
    index_11: "f32[12, 16, 16]" = torch.ops.aten.index.Tensor(slice_18, [None, primals_220]);  slice_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_311: "f32[8, 12, 16, 16]" = torch.ops.aten.add.Tensor(mul_393, index_11);  mul_393 = index_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax_11: "f32[8, 12, 16, 1]" = torch.ops.aten.amax.default(add_311, [-1], True)
    sub_62: "f32[8, 12, 16, 16]" = torch.ops.aten.sub.Tensor(add_311, amax_11);  add_311 = amax_11 = None
    exp_11: "f32[8, 12, 16, 16]" = torch.ops.aten.exp.default(sub_62);  sub_62 = None
    sum_12: "f32[8, 12, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_36: "f32[8, 12, 16, 16]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    alias_11: "f32[8, 12, 16, 16]" = torch.ops.aten.alias.default(div_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_46: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(div_36, [8, 12, 16, 16]);  div_36 = None
    view_287: "f32[96, 16, 16]" = torch.ops.aten.view.default(expand_46, [96, 16, 16]);  expand_46 = None
    expand_47: "f32[8, 12, 16, 32]" = torch.ops.aten.expand.default(permute_94, [8, 12, 16, 32]);  permute_94 = None
    clone_68: "f32[8, 12, 16, 32]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
    view_288: "f32[96, 16, 32]" = torch.ops.aten.view.default(clone_68, [96, 16, 32]);  clone_68 = None
    bmm_23: "f32[96, 16, 32]" = torch.ops.aten.bmm.default(view_287, view_288)
    view_289: "f32[8, 12, 16, 32]" = torch.ops.aten.view.default(bmm_23, [8, 12, 16, 32]);  bmm_23 = None
    permute_95: "f32[8, 16, 12, 32]" = torch.ops.aten.permute.default(view_289, [0, 2, 1, 3]);  view_289 = None
    clone_69: "f32[8, 16, 12, 32]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    view_290: "f32[8, 16, 384]" = torch.ops.aten.view.default(clone_69, [8, 16, 384]);  clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_312: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(view_290, 3)
    clamp_min_25: "f32[8, 16, 384]" = torch.ops.aten.clamp_min.default(add_312, 0);  add_312 = None
    clamp_max_25: "f32[8, 16, 384]" = torch.ops.aten.clamp_max.default(clamp_min_25, 6);  clamp_min_25 = None
    mul_394: "f32[8, 16, 384]" = torch.ops.aten.mul.Tensor(view_290, clamp_max_25);  clamp_max_25 = None
    div_37: "f32[8, 16, 384]" = torch.ops.aten.div.Tensor(mul_394, 6);  mul_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_96: "f32[384, 384]" = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
    view_291: "f32[128, 384]" = torch.ops.aten.view.default(div_37, [128, 384]);  div_37 = None
    mm_47: "f32[128, 384]" = torch.ops.aten.mm.default(view_291, permute_96)
    view_292: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_47, [8, 16, 384]);  mm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_293: "f32[128, 384]" = torch.ops.aten.view.default(view_292, [128, 384]);  view_292 = None
    add_313: "i64[]" = torch.ops.aten.add.Tensor(primals_378, 1)
    var_mean_51 = torch.ops.aten.var_mean.correction(view_293, [0], correction = 0, keepdim = True)
    getitem_136: "f32[1, 384]" = var_mean_51[0]
    getitem_137: "f32[1, 384]" = var_mean_51[1];  var_mean_51 = None
    add_314: "f32[1, 384]" = torch.ops.aten.add.Tensor(getitem_136, 1e-05)
    rsqrt_51: "f32[1, 384]" = torch.ops.aten.rsqrt.default(add_314);  add_314 = None
    sub_63: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_293, getitem_137)
    mul_395: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_51);  sub_63 = None
    squeeze_153: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_137, [0]);  getitem_137 = None
    squeeze_154: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0]);  rsqrt_51 = None
    mul_396: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_153, 0.1)
    mul_397: "f32[384]" = torch.ops.aten.mul.Tensor(primals_376, 0.9)
    add_315: "f32[384]" = torch.ops.aten.add.Tensor(mul_396, mul_397);  mul_396 = mul_397 = None
    squeeze_155: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_136, [0]);  getitem_136 = None
    mul_398: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_155, 1.0078740157480315);  squeeze_155 = None
    mul_399: "f32[384]" = torch.ops.aten.mul.Tensor(mul_398, 0.1);  mul_398 = None
    mul_400: "f32[384]" = torch.ops.aten.mul.Tensor(primals_377, 0.9)
    add_316: "f32[384]" = torch.ops.aten.add.Tensor(mul_399, mul_400);  mul_399 = mul_400 = None
    mul_401: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_395, primals_169);  mul_395 = None
    add_317: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_401, primals_170);  mul_401 = primals_170 = None
    view_294: "f32[8, 16, 384]" = torch.ops.aten.view.default(add_317, [8, 16, 384]);  add_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_318: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_305, view_294);  add_305 = view_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_97: "f32[384, 768]" = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
    view_295: "f32[128, 384]" = torch.ops.aten.view.default(add_318, [128, 384])
    mm_48: "f32[128, 768]" = torch.ops.aten.mm.default(view_295, permute_97)
    view_296: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_48, [8, 16, 768]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_297: "f32[128, 768]" = torch.ops.aten.view.default(view_296, [128, 768]);  view_296 = None
    add_319: "i64[]" = torch.ops.aten.add.Tensor(primals_381, 1)
    var_mean_52 = torch.ops.aten.var_mean.correction(view_297, [0], correction = 0, keepdim = True)
    getitem_138: "f32[1, 768]" = var_mean_52[0]
    getitem_139: "f32[1, 768]" = var_mean_52[1];  var_mean_52 = None
    add_320: "f32[1, 768]" = torch.ops.aten.add.Tensor(getitem_138, 1e-05)
    rsqrt_52: "f32[1, 768]" = torch.ops.aten.rsqrt.default(add_320);  add_320 = None
    sub_64: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_297, getitem_139)
    mul_402: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_52);  sub_64 = None
    squeeze_156: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_139, [0]);  getitem_139 = None
    squeeze_157: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0]);  rsqrt_52 = None
    mul_403: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_156, 0.1)
    mul_404: "f32[768]" = torch.ops.aten.mul.Tensor(primals_379, 0.9)
    add_321: "f32[768]" = torch.ops.aten.add.Tensor(mul_403, mul_404);  mul_403 = mul_404 = None
    squeeze_158: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_138, [0]);  getitem_138 = None
    mul_405: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_158, 1.0078740157480315);  squeeze_158 = None
    mul_406: "f32[768]" = torch.ops.aten.mul.Tensor(mul_405, 0.1);  mul_405 = None
    mul_407: "f32[768]" = torch.ops.aten.mul.Tensor(primals_380, 0.9)
    add_322: "f32[768]" = torch.ops.aten.add.Tensor(mul_406, mul_407);  mul_406 = mul_407 = None
    mul_408: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_402, primals_172);  mul_402 = None
    add_323: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_408, primals_173);  mul_408 = primals_173 = None
    view_298: "f32[8, 16, 768]" = torch.ops.aten.view.default(add_323, [8, 16, 768]);  add_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_324: "f32[8, 16, 768]" = torch.ops.aten.add.Tensor(view_298, 3)
    clamp_min_26: "f32[8, 16, 768]" = torch.ops.aten.clamp_min.default(add_324, 0);  add_324 = None
    clamp_max_26: "f32[8, 16, 768]" = torch.ops.aten.clamp_max.default(clamp_min_26, 6);  clamp_min_26 = None
    mul_409: "f32[8, 16, 768]" = torch.ops.aten.mul.Tensor(view_298, clamp_max_26);  clamp_max_26 = None
    div_38: "f32[8, 16, 768]" = torch.ops.aten.div.Tensor(mul_409, 6);  mul_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    clone_70: "f32[8, 16, 768]" = torch.ops.aten.clone.default(div_38);  div_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_98: "f32[768, 384]" = torch.ops.aten.permute.default(primals_174, [1, 0]);  primals_174 = None
    view_299: "f32[128, 768]" = torch.ops.aten.view.default(clone_70, [128, 768]);  clone_70 = None
    mm_49: "f32[128, 384]" = torch.ops.aten.mm.default(view_299, permute_98)
    view_300: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_49, [8, 16, 384]);  mm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_301: "f32[128, 384]" = torch.ops.aten.view.default(view_300, [128, 384]);  view_300 = None
    add_325: "i64[]" = torch.ops.aten.add.Tensor(primals_384, 1)
    var_mean_53 = torch.ops.aten.var_mean.correction(view_301, [0], correction = 0, keepdim = True)
    getitem_140: "f32[1, 384]" = var_mean_53[0]
    getitem_141: "f32[1, 384]" = var_mean_53[1];  var_mean_53 = None
    add_326: "f32[1, 384]" = torch.ops.aten.add.Tensor(getitem_140, 1e-05)
    rsqrt_53: "f32[1, 384]" = torch.ops.aten.rsqrt.default(add_326);  add_326 = None
    sub_65: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_301, getitem_141)
    mul_410: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_53);  sub_65 = None
    squeeze_159: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_141, [0]);  getitem_141 = None
    squeeze_160: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_53, [0]);  rsqrt_53 = None
    mul_411: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_159, 0.1)
    mul_412: "f32[384]" = torch.ops.aten.mul.Tensor(primals_382, 0.9)
    add_327: "f32[384]" = torch.ops.aten.add.Tensor(mul_411, mul_412);  mul_411 = mul_412 = None
    squeeze_161: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_140, [0]);  getitem_140 = None
    mul_413: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_161, 1.0078740157480315);  squeeze_161 = None
    mul_414: "f32[384]" = torch.ops.aten.mul.Tensor(mul_413, 0.1);  mul_413 = None
    mul_415: "f32[384]" = torch.ops.aten.mul.Tensor(primals_383, 0.9)
    add_328: "f32[384]" = torch.ops.aten.add.Tensor(mul_414, mul_415);  mul_414 = mul_415 = None
    mul_416: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_410, primals_175);  mul_410 = None
    add_329: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_416, primals_176);  mul_416 = primals_176 = None
    view_302: "f32[8, 16, 384]" = torch.ops.aten.view.default(add_329, [8, 16, 384]);  add_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_330: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_318, view_302);  add_318 = view_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_99: "f32[384, 768]" = torch.ops.aten.permute.default(primals_177, [1, 0]);  primals_177 = None
    view_303: "f32[128, 384]" = torch.ops.aten.view.default(add_330, [128, 384])
    mm_50: "f32[128, 768]" = torch.ops.aten.mm.default(view_303, permute_99)
    view_304: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_50, [8, 16, 768]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_305: "f32[128, 768]" = torch.ops.aten.view.default(view_304, [128, 768]);  view_304 = None
    add_331: "i64[]" = torch.ops.aten.add.Tensor(primals_387, 1)
    var_mean_54 = torch.ops.aten.var_mean.correction(view_305, [0], correction = 0, keepdim = True)
    getitem_142: "f32[1, 768]" = var_mean_54[0]
    getitem_143: "f32[1, 768]" = var_mean_54[1];  var_mean_54 = None
    add_332: "f32[1, 768]" = torch.ops.aten.add.Tensor(getitem_142, 1e-05)
    rsqrt_54: "f32[1, 768]" = torch.ops.aten.rsqrt.default(add_332);  add_332 = None
    sub_66: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_305, getitem_143)
    mul_417: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_54);  sub_66 = None
    squeeze_162: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_143, [0]);  getitem_143 = None
    squeeze_163: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_54, [0]);  rsqrt_54 = None
    mul_418: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_162, 0.1)
    mul_419: "f32[768]" = torch.ops.aten.mul.Tensor(primals_385, 0.9)
    add_333: "f32[768]" = torch.ops.aten.add.Tensor(mul_418, mul_419);  mul_418 = mul_419 = None
    squeeze_164: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_142, [0]);  getitem_142 = None
    mul_420: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_164, 1.0078740157480315);  squeeze_164 = None
    mul_421: "f32[768]" = torch.ops.aten.mul.Tensor(mul_420, 0.1);  mul_420 = None
    mul_422: "f32[768]" = torch.ops.aten.mul.Tensor(primals_386, 0.9)
    add_334: "f32[768]" = torch.ops.aten.add.Tensor(mul_421, mul_422);  mul_421 = mul_422 = None
    mul_423: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_417, primals_178);  mul_417 = None
    add_335: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_423, primals_179);  mul_423 = primals_179 = None
    view_306: "f32[8, 16, 768]" = torch.ops.aten.view.default(add_335, [8, 16, 768]);  add_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_307: "f32[8, 16, 12, 64]" = torch.ops.aten.view.default(view_306, [8, 16, 12, -1]);  view_306 = None
    split_with_sizes_12 = torch.ops.aten.split_with_sizes.default(view_307, [16, 16, 32], 3);  view_307 = None
    getitem_144: "f32[8, 16, 12, 16]" = split_with_sizes_12[0]
    getitem_145: "f32[8, 16, 12, 16]" = split_with_sizes_12[1]
    getitem_146: "f32[8, 16, 12, 32]" = split_with_sizes_12[2];  split_with_sizes_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_100: "f32[8, 12, 16, 16]" = torch.ops.aten.permute.default(getitem_144, [0, 2, 1, 3]);  getitem_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_101: "f32[8, 12, 16, 16]" = torch.ops.aten.permute.default(getitem_145, [0, 2, 3, 1]);  getitem_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_102: "f32[8, 12, 16, 32]" = torch.ops.aten.permute.default(getitem_146, [0, 2, 1, 3]);  getitem_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_48: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(permute_100, [8, 12, 16, 16]);  permute_100 = None
    clone_71: "f32[8, 12, 16, 16]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
    view_308: "f32[96, 16, 16]" = torch.ops.aten.view.default(clone_71, [96, 16, 16]);  clone_71 = None
    expand_49: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(permute_101, [8, 12, 16, 16]);  permute_101 = None
    clone_72: "f32[8, 12, 16, 16]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
    view_309: "f32[96, 16, 16]" = torch.ops.aten.view.default(clone_72, [96, 16, 16]);  clone_72 = None
    bmm_24: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(view_308, view_309)
    view_310: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_24, [8, 12, 16, 16]);  bmm_24 = None
    mul_424: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(view_310, 0.25);  view_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    slice_19: "f32[12, 16]" = torch.ops.aten.slice.Tensor(primals_13, 0, 0, 9223372036854775807);  primals_13 = None
    index_12: "f32[12, 16, 16]" = torch.ops.aten.index.Tensor(slice_19, [None, primals_221]);  slice_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_336: "f32[8, 12, 16, 16]" = torch.ops.aten.add.Tensor(mul_424, index_12);  mul_424 = index_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax_12: "f32[8, 12, 16, 1]" = torch.ops.aten.amax.default(add_336, [-1], True)
    sub_67: "f32[8, 12, 16, 16]" = torch.ops.aten.sub.Tensor(add_336, amax_12);  add_336 = amax_12 = None
    exp_12: "f32[8, 12, 16, 16]" = torch.ops.aten.exp.default(sub_67);  sub_67 = None
    sum_13: "f32[8, 12, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_39: "f32[8, 12, 16, 16]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    alias_12: "f32[8, 12, 16, 16]" = torch.ops.aten.alias.default(div_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_50: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(div_39, [8, 12, 16, 16]);  div_39 = None
    view_311: "f32[96, 16, 16]" = torch.ops.aten.view.default(expand_50, [96, 16, 16]);  expand_50 = None
    expand_51: "f32[8, 12, 16, 32]" = torch.ops.aten.expand.default(permute_102, [8, 12, 16, 32]);  permute_102 = None
    clone_73: "f32[8, 12, 16, 32]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
    view_312: "f32[96, 16, 32]" = torch.ops.aten.view.default(clone_73, [96, 16, 32]);  clone_73 = None
    bmm_25: "f32[96, 16, 32]" = torch.ops.aten.bmm.default(view_311, view_312)
    view_313: "f32[8, 12, 16, 32]" = torch.ops.aten.view.default(bmm_25, [8, 12, 16, 32]);  bmm_25 = None
    permute_103: "f32[8, 16, 12, 32]" = torch.ops.aten.permute.default(view_313, [0, 2, 1, 3]);  view_313 = None
    clone_74: "f32[8, 16, 12, 32]" = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
    view_314: "f32[8, 16, 384]" = torch.ops.aten.view.default(clone_74, [8, 16, 384]);  clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_337: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(view_314, 3)
    clamp_min_27: "f32[8, 16, 384]" = torch.ops.aten.clamp_min.default(add_337, 0);  add_337 = None
    clamp_max_27: "f32[8, 16, 384]" = torch.ops.aten.clamp_max.default(clamp_min_27, 6);  clamp_min_27 = None
    mul_425: "f32[8, 16, 384]" = torch.ops.aten.mul.Tensor(view_314, clamp_max_27);  clamp_max_27 = None
    div_40: "f32[8, 16, 384]" = torch.ops.aten.div.Tensor(mul_425, 6);  mul_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_104: "f32[384, 384]" = torch.ops.aten.permute.default(primals_180, [1, 0]);  primals_180 = None
    view_315: "f32[128, 384]" = torch.ops.aten.view.default(div_40, [128, 384]);  div_40 = None
    mm_51: "f32[128, 384]" = torch.ops.aten.mm.default(view_315, permute_104)
    view_316: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_51, [8, 16, 384]);  mm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_317: "f32[128, 384]" = torch.ops.aten.view.default(view_316, [128, 384]);  view_316 = None
    add_338: "i64[]" = torch.ops.aten.add.Tensor(primals_390, 1)
    var_mean_55 = torch.ops.aten.var_mean.correction(view_317, [0], correction = 0, keepdim = True)
    getitem_147: "f32[1, 384]" = var_mean_55[0]
    getitem_148: "f32[1, 384]" = var_mean_55[1];  var_mean_55 = None
    add_339: "f32[1, 384]" = torch.ops.aten.add.Tensor(getitem_147, 1e-05)
    rsqrt_55: "f32[1, 384]" = torch.ops.aten.rsqrt.default(add_339);  add_339 = None
    sub_68: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_317, getitem_148)
    mul_426: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_55);  sub_68 = None
    squeeze_165: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_148, [0]);  getitem_148 = None
    squeeze_166: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0]);  rsqrt_55 = None
    mul_427: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_165, 0.1)
    mul_428: "f32[384]" = torch.ops.aten.mul.Tensor(primals_388, 0.9)
    add_340: "f32[384]" = torch.ops.aten.add.Tensor(mul_427, mul_428);  mul_427 = mul_428 = None
    squeeze_167: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_147, [0]);  getitem_147 = None
    mul_429: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_167, 1.0078740157480315);  squeeze_167 = None
    mul_430: "f32[384]" = torch.ops.aten.mul.Tensor(mul_429, 0.1);  mul_429 = None
    mul_431: "f32[384]" = torch.ops.aten.mul.Tensor(primals_389, 0.9)
    add_341: "f32[384]" = torch.ops.aten.add.Tensor(mul_430, mul_431);  mul_430 = mul_431 = None
    mul_432: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_426, primals_181);  mul_426 = None
    add_342: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_432, primals_182);  mul_432 = primals_182 = None
    view_318: "f32[8, 16, 384]" = torch.ops.aten.view.default(add_342, [8, 16, 384]);  add_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_343: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_330, view_318);  add_330 = view_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_105: "f32[384, 768]" = torch.ops.aten.permute.default(primals_183, [1, 0]);  primals_183 = None
    view_319: "f32[128, 384]" = torch.ops.aten.view.default(add_343, [128, 384])
    mm_52: "f32[128, 768]" = torch.ops.aten.mm.default(view_319, permute_105)
    view_320: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_52, [8, 16, 768]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_321: "f32[128, 768]" = torch.ops.aten.view.default(view_320, [128, 768]);  view_320 = None
    add_344: "i64[]" = torch.ops.aten.add.Tensor(primals_393, 1)
    var_mean_56 = torch.ops.aten.var_mean.correction(view_321, [0], correction = 0, keepdim = True)
    getitem_149: "f32[1, 768]" = var_mean_56[0]
    getitem_150: "f32[1, 768]" = var_mean_56[1];  var_mean_56 = None
    add_345: "f32[1, 768]" = torch.ops.aten.add.Tensor(getitem_149, 1e-05)
    rsqrt_56: "f32[1, 768]" = torch.ops.aten.rsqrt.default(add_345);  add_345 = None
    sub_69: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_321, getitem_150)
    mul_433: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_56);  sub_69 = None
    squeeze_168: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_150, [0]);  getitem_150 = None
    squeeze_169: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_56, [0]);  rsqrt_56 = None
    mul_434: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_168, 0.1)
    mul_435: "f32[768]" = torch.ops.aten.mul.Tensor(primals_391, 0.9)
    add_346: "f32[768]" = torch.ops.aten.add.Tensor(mul_434, mul_435);  mul_434 = mul_435 = None
    squeeze_170: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_149, [0]);  getitem_149 = None
    mul_436: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_170, 1.0078740157480315);  squeeze_170 = None
    mul_437: "f32[768]" = torch.ops.aten.mul.Tensor(mul_436, 0.1);  mul_436 = None
    mul_438: "f32[768]" = torch.ops.aten.mul.Tensor(primals_392, 0.9)
    add_347: "f32[768]" = torch.ops.aten.add.Tensor(mul_437, mul_438);  mul_437 = mul_438 = None
    mul_439: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_433, primals_184);  mul_433 = None
    add_348: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_439, primals_185);  mul_439 = primals_185 = None
    view_322: "f32[8, 16, 768]" = torch.ops.aten.view.default(add_348, [8, 16, 768]);  add_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_349: "f32[8, 16, 768]" = torch.ops.aten.add.Tensor(view_322, 3)
    clamp_min_28: "f32[8, 16, 768]" = torch.ops.aten.clamp_min.default(add_349, 0);  add_349 = None
    clamp_max_28: "f32[8, 16, 768]" = torch.ops.aten.clamp_max.default(clamp_min_28, 6);  clamp_min_28 = None
    mul_440: "f32[8, 16, 768]" = torch.ops.aten.mul.Tensor(view_322, clamp_max_28);  clamp_max_28 = None
    div_41: "f32[8, 16, 768]" = torch.ops.aten.div.Tensor(mul_440, 6);  mul_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    clone_75: "f32[8, 16, 768]" = torch.ops.aten.clone.default(div_41);  div_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_106: "f32[768, 384]" = torch.ops.aten.permute.default(primals_186, [1, 0]);  primals_186 = None
    view_323: "f32[128, 768]" = torch.ops.aten.view.default(clone_75, [128, 768]);  clone_75 = None
    mm_53: "f32[128, 384]" = torch.ops.aten.mm.default(view_323, permute_106)
    view_324: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_53, [8, 16, 384]);  mm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_325: "f32[128, 384]" = torch.ops.aten.view.default(view_324, [128, 384]);  view_324 = None
    add_350: "i64[]" = torch.ops.aten.add.Tensor(primals_396, 1)
    var_mean_57 = torch.ops.aten.var_mean.correction(view_325, [0], correction = 0, keepdim = True)
    getitem_151: "f32[1, 384]" = var_mean_57[0]
    getitem_152: "f32[1, 384]" = var_mean_57[1];  var_mean_57 = None
    add_351: "f32[1, 384]" = torch.ops.aten.add.Tensor(getitem_151, 1e-05)
    rsqrt_57: "f32[1, 384]" = torch.ops.aten.rsqrt.default(add_351);  add_351 = None
    sub_70: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_325, getitem_152)
    mul_441: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_57);  sub_70 = None
    squeeze_171: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_152, [0]);  getitem_152 = None
    squeeze_172: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_57, [0]);  rsqrt_57 = None
    mul_442: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_171, 0.1)
    mul_443: "f32[384]" = torch.ops.aten.mul.Tensor(primals_394, 0.9)
    add_352: "f32[384]" = torch.ops.aten.add.Tensor(mul_442, mul_443);  mul_442 = mul_443 = None
    squeeze_173: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_151, [0]);  getitem_151 = None
    mul_444: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_173, 1.0078740157480315);  squeeze_173 = None
    mul_445: "f32[384]" = torch.ops.aten.mul.Tensor(mul_444, 0.1);  mul_444 = None
    mul_446: "f32[384]" = torch.ops.aten.mul.Tensor(primals_395, 0.9)
    add_353: "f32[384]" = torch.ops.aten.add.Tensor(mul_445, mul_446);  mul_445 = mul_446 = None
    mul_447: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_441, primals_187);  mul_441 = None
    add_354: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_447, primals_188);  mul_447 = primals_188 = None
    view_326: "f32[8, 16, 384]" = torch.ops.aten.view.default(add_354, [8, 16, 384]);  add_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_355: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_343, view_326);  add_343 = view_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_107: "f32[384, 768]" = torch.ops.aten.permute.default(primals_189, [1, 0]);  primals_189 = None
    view_327: "f32[128, 384]" = torch.ops.aten.view.default(add_355, [128, 384])
    mm_54: "f32[128, 768]" = torch.ops.aten.mm.default(view_327, permute_107)
    view_328: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_54, [8, 16, 768]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_329: "f32[128, 768]" = torch.ops.aten.view.default(view_328, [128, 768]);  view_328 = None
    add_356: "i64[]" = torch.ops.aten.add.Tensor(primals_399, 1)
    var_mean_58 = torch.ops.aten.var_mean.correction(view_329, [0], correction = 0, keepdim = True)
    getitem_153: "f32[1, 768]" = var_mean_58[0]
    getitem_154: "f32[1, 768]" = var_mean_58[1];  var_mean_58 = None
    add_357: "f32[1, 768]" = torch.ops.aten.add.Tensor(getitem_153, 1e-05)
    rsqrt_58: "f32[1, 768]" = torch.ops.aten.rsqrt.default(add_357);  add_357 = None
    sub_71: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_329, getitem_154)
    mul_448: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_58);  sub_71 = None
    squeeze_174: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_154, [0]);  getitem_154 = None
    squeeze_175: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_58, [0]);  rsqrt_58 = None
    mul_449: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_174, 0.1)
    mul_450: "f32[768]" = torch.ops.aten.mul.Tensor(primals_397, 0.9)
    add_358: "f32[768]" = torch.ops.aten.add.Tensor(mul_449, mul_450);  mul_449 = mul_450 = None
    squeeze_176: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_153, [0]);  getitem_153 = None
    mul_451: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_176, 1.0078740157480315);  squeeze_176 = None
    mul_452: "f32[768]" = torch.ops.aten.mul.Tensor(mul_451, 0.1);  mul_451 = None
    mul_453: "f32[768]" = torch.ops.aten.mul.Tensor(primals_398, 0.9)
    add_359: "f32[768]" = torch.ops.aten.add.Tensor(mul_452, mul_453);  mul_452 = mul_453 = None
    mul_454: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_448, primals_190);  mul_448 = None
    add_360: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_454, primals_191);  mul_454 = primals_191 = None
    view_330: "f32[8, 16, 768]" = torch.ops.aten.view.default(add_360, [8, 16, 768]);  add_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_331: "f32[8, 16, 12, 64]" = torch.ops.aten.view.default(view_330, [8, 16, 12, -1]);  view_330 = None
    split_with_sizes_13 = torch.ops.aten.split_with_sizes.default(view_331, [16, 16, 32], 3);  view_331 = None
    getitem_155: "f32[8, 16, 12, 16]" = split_with_sizes_13[0]
    getitem_156: "f32[8, 16, 12, 16]" = split_with_sizes_13[1]
    getitem_157: "f32[8, 16, 12, 32]" = split_with_sizes_13[2];  split_with_sizes_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_108: "f32[8, 12, 16, 16]" = torch.ops.aten.permute.default(getitem_155, [0, 2, 1, 3]);  getitem_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_109: "f32[8, 12, 16, 16]" = torch.ops.aten.permute.default(getitem_156, [0, 2, 3, 1]);  getitem_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_110: "f32[8, 12, 16, 32]" = torch.ops.aten.permute.default(getitem_157, [0, 2, 1, 3]);  getitem_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_52: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(permute_108, [8, 12, 16, 16]);  permute_108 = None
    clone_76: "f32[8, 12, 16, 16]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
    view_332: "f32[96, 16, 16]" = torch.ops.aten.view.default(clone_76, [96, 16, 16]);  clone_76 = None
    expand_53: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(permute_109, [8, 12, 16, 16]);  permute_109 = None
    clone_77: "f32[8, 12, 16, 16]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
    view_333: "f32[96, 16, 16]" = torch.ops.aten.view.default(clone_77, [96, 16, 16]);  clone_77 = None
    bmm_26: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(view_332, view_333)
    view_334: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_26, [8, 12, 16, 16]);  bmm_26 = None
    mul_455: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(view_334, 0.25);  view_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    slice_20: "f32[12, 16]" = torch.ops.aten.slice.Tensor(primals_14, 0, 0, 9223372036854775807);  primals_14 = None
    index_13: "f32[12, 16, 16]" = torch.ops.aten.index.Tensor(slice_20, [None, primals_222]);  slice_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_361: "f32[8, 12, 16, 16]" = torch.ops.aten.add.Tensor(mul_455, index_13);  mul_455 = index_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax_13: "f32[8, 12, 16, 1]" = torch.ops.aten.amax.default(add_361, [-1], True)
    sub_72: "f32[8, 12, 16, 16]" = torch.ops.aten.sub.Tensor(add_361, amax_13);  add_361 = amax_13 = None
    exp_13: "f32[8, 12, 16, 16]" = torch.ops.aten.exp.default(sub_72);  sub_72 = None
    sum_14: "f32[8, 12, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_42: "f32[8, 12, 16, 16]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    alias_13: "f32[8, 12, 16, 16]" = torch.ops.aten.alias.default(div_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_54: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(div_42, [8, 12, 16, 16]);  div_42 = None
    view_335: "f32[96, 16, 16]" = torch.ops.aten.view.default(expand_54, [96, 16, 16]);  expand_54 = None
    expand_55: "f32[8, 12, 16, 32]" = torch.ops.aten.expand.default(permute_110, [8, 12, 16, 32]);  permute_110 = None
    clone_78: "f32[8, 12, 16, 32]" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
    view_336: "f32[96, 16, 32]" = torch.ops.aten.view.default(clone_78, [96, 16, 32]);  clone_78 = None
    bmm_27: "f32[96, 16, 32]" = torch.ops.aten.bmm.default(view_335, view_336)
    view_337: "f32[8, 12, 16, 32]" = torch.ops.aten.view.default(bmm_27, [8, 12, 16, 32]);  bmm_27 = None
    permute_111: "f32[8, 16, 12, 32]" = torch.ops.aten.permute.default(view_337, [0, 2, 1, 3]);  view_337 = None
    clone_79: "f32[8, 16, 12, 32]" = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None
    view_338: "f32[8, 16, 384]" = torch.ops.aten.view.default(clone_79, [8, 16, 384]);  clone_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_362: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(view_338, 3)
    clamp_min_29: "f32[8, 16, 384]" = torch.ops.aten.clamp_min.default(add_362, 0);  add_362 = None
    clamp_max_29: "f32[8, 16, 384]" = torch.ops.aten.clamp_max.default(clamp_min_29, 6);  clamp_min_29 = None
    mul_456: "f32[8, 16, 384]" = torch.ops.aten.mul.Tensor(view_338, clamp_max_29);  clamp_max_29 = None
    div_43: "f32[8, 16, 384]" = torch.ops.aten.div.Tensor(mul_456, 6);  mul_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_112: "f32[384, 384]" = torch.ops.aten.permute.default(primals_192, [1, 0]);  primals_192 = None
    view_339: "f32[128, 384]" = torch.ops.aten.view.default(div_43, [128, 384]);  div_43 = None
    mm_55: "f32[128, 384]" = torch.ops.aten.mm.default(view_339, permute_112)
    view_340: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_55, [8, 16, 384]);  mm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_341: "f32[128, 384]" = torch.ops.aten.view.default(view_340, [128, 384]);  view_340 = None
    add_363: "i64[]" = torch.ops.aten.add.Tensor(primals_402, 1)
    var_mean_59 = torch.ops.aten.var_mean.correction(view_341, [0], correction = 0, keepdim = True)
    getitem_158: "f32[1, 384]" = var_mean_59[0]
    getitem_159: "f32[1, 384]" = var_mean_59[1];  var_mean_59 = None
    add_364: "f32[1, 384]" = torch.ops.aten.add.Tensor(getitem_158, 1e-05)
    rsqrt_59: "f32[1, 384]" = torch.ops.aten.rsqrt.default(add_364);  add_364 = None
    sub_73: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_341, getitem_159)
    mul_457: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_59);  sub_73 = None
    squeeze_177: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_159, [0]);  getitem_159 = None
    squeeze_178: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_59, [0]);  rsqrt_59 = None
    mul_458: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_177, 0.1)
    mul_459: "f32[384]" = torch.ops.aten.mul.Tensor(primals_400, 0.9)
    add_365: "f32[384]" = torch.ops.aten.add.Tensor(mul_458, mul_459);  mul_458 = mul_459 = None
    squeeze_179: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_158, [0]);  getitem_158 = None
    mul_460: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_179, 1.0078740157480315);  squeeze_179 = None
    mul_461: "f32[384]" = torch.ops.aten.mul.Tensor(mul_460, 0.1);  mul_460 = None
    mul_462: "f32[384]" = torch.ops.aten.mul.Tensor(primals_401, 0.9)
    add_366: "f32[384]" = torch.ops.aten.add.Tensor(mul_461, mul_462);  mul_461 = mul_462 = None
    mul_463: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_457, primals_193);  mul_457 = None
    add_367: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_463, primals_194);  mul_463 = primals_194 = None
    view_342: "f32[8, 16, 384]" = torch.ops.aten.view.default(add_367, [8, 16, 384]);  add_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_368: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_355, view_342);  add_355 = view_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_113: "f32[384, 768]" = torch.ops.aten.permute.default(primals_195, [1, 0]);  primals_195 = None
    view_343: "f32[128, 384]" = torch.ops.aten.view.default(add_368, [128, 384])
    mm_56: "f32[128, 768]" = torch.ops.aten.mm.default(view_343, permute_113)
    view_344: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_56, [8, 16, 768]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_345: "f32[128, 768]" = torch.ops.aten.view.default(view_344, [128, 768]);  view_344 = None
    add_369: "i64[]" = torch.ops.aten.add.Tensor(primals_405, 1)
    var_mean_60 = torch.ops.aten.var_mean.correction(view_345, [0], correction = 0, keepdim = True)
    getitem_160: "f32[1, 768]" = var_mean_60[0]
    getitem_161: "f32[1, 768]" = var_mean_60[1];  var_mean_60 = None
    add_370: "f32[1, 768]" = torch.ops.aten.add.Tensor(getitem_160, 1e-05)
    rsqrt_60: "f32[1, 768]" = torch.ops.aten.rsqrt.default(add_370);  add_370 = None
    sub_74: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_345, getitem_161)
    mul_464: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_60);  sub_74 = None
    squeeze_180: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_161, [0]);  getitem_161 = None
    squeeze_181: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_60, [0]);  rsqrt_60 = None
    mul_465: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_180, 0.1)
    mul_466: "f32[768]" = torch.ops.aten.mul.Tensor(primals_403, 0.9)
    add_371: "f32[768]" = torch.ops.aten.add.Tensor(mul_465, mul_466);  mul_465 = mul_466 = None
    squeeze_182: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_160, [0]);  getitem_160 = None
    mul_467: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_182, 1.0078740157480315);  squeeze_182 = None
    mul_468: "f32[768]" = torch.ops.aten.mul.Tensor(mul_467, 0.1);  mul_467 = None
    mul_469: "f32[768]" = torch.ops.aten.mul.Tensor(primals_404, 0.9)
    add_372: "f32[768]" = torch.ops.aten.add.Tensor(mul_468, mul_469);  mul_468 = mul_469 = None
    mul_470: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_464, primals_196);  mul_464 = None
    add_373: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_470, primals_197);  mul_470 = primals_197 = None
    view_346: "f32[8, 16, 768]" = torch.ops.aten.view.default(add_373, [8, 16, 768]);  add_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_374: "f32[8, 16, 768]" = torch.ops.aten.add.Tensor(view_346, 3)
    clamp_min_30: "f32[8, 16, 768]" = torch.ops.aten.clamp_min.default(add_374, 0);  add_374 = None
    clamp_max_30: "f32[8, 16, 768]" = torch.ops.aten.clamp_max.default(clamp_min_30, 6);  clamp_min_30 = None
    mul_471: "f32[8, 16, 768]" = torch.ops.aten.mul.Tensor(view_346, clamp_max_30);  clamp_max_30 = None
    div_44: "f32[8, 16, 768]" = torch.ops.aten.div.Tensor(mul_471, 6);  mul_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    clone_80: "f32[8, 16, 768]" = torch.ops.aten.clone.default(div_44);  div_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_114: "f32[768, 384]" = torch.ops.aten.permute.default(primals_198, [1, 0]);  primals_198 = None
    view_347: "f32[128, 768]" = torch.ops.aten.view.default(clone_80, [128, 768]);  clone_80 = None
    mm_57: "f32[128, 384]" = torch.ops.aten.mm.default(view_347, permute_114)
    view_348: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_57, [8, 16, 384]);  mm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_349: "f32[128, 384]" = torch.ops.aten.view.default(view_348, [128, 384]);  view_348 = None
    add_375: "i64[]" = torch.ops.aten.add.Tensor(primals_408, 1)
    var_mean_61 = torch.ops.aten.var_mean.correction(view_349, [0], correction = 0, keepdim = True)
    getitem_162: "f32[1, 384]" = var_mean_61[0]
    getitem_163: "f32[1, 384]" = var_mean_61[1];  var_mean_61 = None
    add_376: "f32[1, 384]" = torch.ops.aten.add.Tensor(getitem_162, 1e-05)
    rsqrt_61: "f32[1, 384]" = torch.ops.aten.rsqrt.default(add_376);  add_376 = None
    sub_75: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_349, getitem_163)
    mul_472: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_61);  sub_75 = None
    squeeze_183: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_163, [0]);  getitem_163 = None
    squeeze_184: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_61, [0]);  rsqrt_61 = None
    mul_473: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_183, 0.1)
    mul_474: "f32[384]" = torch.ops.aten.mul.Tensor(primals_406, 0.9)
    add_377: "f32[384]" = torch.ops.aten.add.Tensor(mul_473, mul_474);  mul_473 = mul_474 = None
    squeeze_185: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_162, [0]);  getitem_162 = None
    mul_475: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_185, 1.0078740157480315);  squeeze_185 = None
    mul_476: "f32[384]" = torch.ops.aten.mul.Tensor(mul_475, 0.1);  mul_475 = None
    mul_477: "f32[384]" = torch.ops.aten.mul.Tensor(primals_407, 0.9)
    add_378: "f32[384]" = torch.ops.aten.add.Tensor(mul_476, mul_477);  mul_476 = mul_477 = None
    mul_478: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_472, primals_199);  mul_472 = None
    add_379: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_478, primals_200);  mul_478 = primals_200 = None
    view_350: "f32[8, 16, 384]" = torch.ops.aten.view.default(add_379, [8, 16, 384]);  add_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_380: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_368, view_350);  add_368 = view_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:681, code: x = x.mean(dim=(-2, -1)) if self.use_conv else x.mean(dim=1)
    mean: "f32[8, 384]" = torch.ops.aten.mean.dim(add_380, [1]);  add_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:119, code: return self.linear(self.drop(self.bn(x)))
    add_381: "i64[]" = torch.ops.aten.add.Tensor(primals_411, 1)
    var_mean_62 = torch.ops.aten.var_mean.correction(mean, [0], correction = 0, keepdim = True)
    getitem_164: "f32[1, 384]" = var_mean_62[0]
    getitem_165: "f32[1, 384]" = var_mean_62[1];  var_mean_62 = None
    add_382: "f32[1, 384]" = torch.ops.aten.add.Tensor(getitem_164, 1e-05)
    rsqrt_62: "f32[1, 384]" = torch.ops.aten.rsqrt.default(add_382);  add_382 = None
    sub_76: "f32[8, 384]" = torch.ops.aten.sub.Tensor(mean, getitem_165)
    mul_479: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_62);  sub_76 = None
    squeeze_186: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_165, [0]);  getitem_165 = None
    squeeze_187: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_62, [0]);  rsqrt_62 = None
    mul_480: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_186, 0.1)
    mul_481: "f32[384]" = torch.ops.aten.mul.Tensor(primals_409, 0.9)
    add_383: "f32[384]" = torch.ops.aten.add.Tensor(mul_480, mul_481);  mul_480 = mul_481 = None
    squeeze_188: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_164, [0]);  getitem_164 = None
    mul_482: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_188, 1.1428571428571428);  squeeze_188 = None
    mul_483: "f32[384]" = torch.ops.aten.mul.Tensor(mul_482, 0.1);  mul_482 = None
    mul_484: "f32[384]" = torch.ops.aten.mul.Tensor(primals_410, 0.9)
    add_384: "f32[384]" = torch.ops.aten.add.Tensor(mul_483, mul_484);  mul_483 = mul_484 = None
    mul_485: "f32[8, 384]" = torch.ops.aten.mul.Tensor(mul_479, primals_201);  mul_479 = None
    add_385: "f32[8, 384]" = torch.ops.aten.add.Tensor(mul_485, primals_202);  mul_485 = primals_202 = None
    clone_81: "f32[8, 384]" = torch.ops.aten.clone.default(add_385);  add_385 = None
    permute_115: "f32[384, 1000]" = torch.ops.aten.permute.default(primals_203, [1, 0]);  primals_203 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_204, clone_81, permute_115);  primals_204 = None
    add_386: "i64[]" = torch.ops.aten.add.Tensor(primals_414, 1)
    var_mean_63 = torch.ops.aten.var_mean.correction(mean, [0], correction = 0, keepdim = True)
    getitem_166: "f32[1, 384]" = var_mean_63[0]
    getitem_167: "f32[1, 384]" = var_mean_63[1];  var_mean_63 = None
    add_387: "f32[1, 384]" = torch.ops.aten.add.Tensor(getitem_166, 1e-05)
    rsqrt_63: "f32[1, 384]" = torch.ops.aten.rsqrt.default(add_387);  add_387 = None
    sub_77: "f32[8, 384]" = torch.ops.aten.sub.Tensor(mean, getitem_167)
    mul_486: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_63);  sub_77 = None
    squeeze_189: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_167, [0]);  getitem_167 = None
    squeeze_190: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_63, [0]);  rsqrt_63 = None
    mul_487: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_189, 0.1)
    mul_488: "f32[384]" = torch.ops.aten.mul.Tensor(primals_412, 0.9)
    add_388: "f32[384]" = torch.ops.aten.add.Tensor(mul_487, mul_488);  mul_487 = mul_488 = None
    squeeze_191: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_166, [0]);  getitem_166 = None
    mul_489: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_191, 1.1428571428571428);  squeeze_191 = None
    mul_490: "f32[384]" = torch.ops.aten.mul.Tensor(mul_489, 0.1);  mul_489 = None
    mul_491: "f32[384]" = torch.ops.aten.mul.Tensor(primals_413, 0.9)
    add_389: "f32[384]" = torch.ops.aten.add.Tensor(mul_490, mul_491);  mul_490 = mul_491 = None
    mul_492: "f32[8, 384]" = torch.ops.aten.mul.Tensor(mul_486, primals_205);  mul_486 = None
    add_390: "f32[8, 384]" = torch.ops.aten.add.Tensor(mul_492, primals_206);  mul_492 = primals_206 = None
    clone_82: "f32[8, 384]" = torch.ops.aten.clone.default(add_390);  add_390 = None
    permute_116: "f32[384, 1000]" = torch.ops.aten.permute.default(primals_207, [1, 0]);  primals_207 = None
    addmm_1: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_208, clone_82, permute_116);  primals_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:690, code: return (x + x_dist) / 2
    add_391: "f32[8, 1000]" = torch.ops.aten.add.Tensor(addmm, addmm_1);  addmm = addmm_1 = None
    div_45: "f32[8, 1000]" = torch.ops.aten.div.Tensor(add_391, 2);  add_391 = None
    div_46: "f32[8, 1000]" = torch.ops.aten.div.Tensor(tangents_1, 2);  tangents_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:119, code: return self.linear(self.drop(self.bn(x)))
    permute_117: "f32[1000, 384]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
    mm_58: "f32[8, 384]" = torch.ops.aten.mm.default(div_46, permute_117);  permute_117 = None
    permute_118: "f32[1000, 8]" = torch.ops.aten.permute.default(div_46, [1, 0])
    mm_59: "f32[1000, 384]" = torch.ops.aten.mm.default(permute_118, clone_82);  permute_118 = clone_82 = None
    permute_119: "f32[384, 1000]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_15: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(div_46, [0], True)
    view_351: "f32[1000]" = torch.ops.aten.view.default(sum_15, [1000]);  sum_15 = None
    permute_120: "f32[1000, 384]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    unsqueeze_16: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_189, 0);  squeeze_189 = None
    sum_16: "f32[384]" = torch.ops.aten.sum.dim_IntList(mm_58, [0])
    sub_78: "f32[8, 384]" = torch.ops.aten.sub.Tensor(mean, unsqueeze_16)
    mul_493: "f32[8, 384]" = torch.ops.aten.mul.Tensor(mm_58, sub_78);  sub_78 = None
    sum_17: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_493, [0]);  mul_493 = None
    mul_494: "f32[384]" = torch.ops.aten.mul.Tensor(sum_16, 0.125)
    unsqueeze_17: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_494, 0);  mul_494 = None
    mul_495: "f32[384]" = torch.ops.aten.mul.Tensor(sum_17, 0.125)
    mul_496: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_190, squeeze_190)
    mul_497: "f32[384]" = torch.ops.aten.mul.Tensor(mul_495, mul_496);  mul_495 = mul_496 = None
    unsqueeze_18: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_497, 0);  mul_497 = None
    mul_498: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_190, primals_205);  primals_205 = None
    unsqueeze_19: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_498, 0);  mul_498 = None
    sub_79: "f32[8, 384]" = torch.ops.aten.sub.Tensor(mean, unsqueeze_16);  unsqueeze_16 = None
    mul_499: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_18);  sub_79 = unsqueeze_18 = None
    sub_80: "f32[8, 384]" = torch.ops.aten.sub.Tensor(mm_58, mul_499);  mm_58 = mul_499 = None
    sub_81: "f32[8, 384]" = torch.ops.aten.sub.Tensor(sub_80, unsqueeze_17);  sub_80 = unsqueeze_17 = None
    mul_500: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_19);  sub_81 = unsqueeze_19 = None
    mul_501: "f32[384]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_190);  sum_17 = squeeze_190 = None
    permute_121: "f32[1000, 384]" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None
    mm_60: "f32[8, 384]" = torch.ops.aten.mm.default(div_46, permute_121);  permute_121 = None
    permute_122: "f32[1000, 8]" = torch.ops.aten.permute.default(div_46, [1, 0])
    mm_61: "f32[1000, 384]" = torch.ops.aten.mm.default(permute_122, clone_81);  permute_122 = clone_81 = None
    permute_123: "f32[384, 1000]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_18: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(div_46, [0], True);  div_46 = None
    view_352: "f32[1000]" = torch.ops.aten.view.default(sum_18, [1000]);  sum_18 = None
    permute_124: "f32[1000, 384]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
    unsqueeze_20: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_186, 0);  squeeze_186 = None
    sum_19: "f32[384]" = torch.ops.aten.sum.dim_IntList(mm_60, [0])
    sub_82: "f32[8, 384]" = torch.ops.aten.sub.Tensor(mean, unsqueeze_20)
    mul_502: "f32[8, 384]" = torch.ops.aten.mul.Tensor(mm_60, sub_82);  sub_82 = None
    sum_20: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_502, [0]);  mul_502 = None
    mul_503: "f32[384]" = torch.ops.aten.mul.Tensor(sum_19, 0.125)
    unsqueeze_21: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_503, 0);  mul_503 = None
    mul_504: "f32[384]" = torch.ops.aten.mul.Tensor(sum_20, 0.125)
    mul_505: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_187, squeeze_187)
    mul_506: "f32[384]" = torch.ops.aten.mul.Tensor(mul_504, mul_505);  mul_504 = mul_505 = None
    unsqueeze_22: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_506, 0);  mul_506 = None
    mul_507: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_187, primals_201);  primals_201 = None
    unsqueeze_23: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_507, 0);  mul_507 = None
    sub_83: "f32[8, 384]" = torch.ops.aten.sub.Tensor(mean, unsqueeze_20);  mean = unsqueeze_20 = None
    mul_508: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_22);  sub_83 = unsqueeze_22 = None
    sub_84: "f32[8, 384]" = torch.ops.aten.sub.Tensor(mm_60, mul_508);  mm_60 = mul_508 = None
    sub_85: "f32[8, 384]" = torch.ops.aten.sub.Tensor(sub_84, unsqueeze_21);  sub_84 = unsqueeze_21 = None
    mul_509: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_23);  sub_85 = unsqueeze_23 = None
    mul_510: "f32[384]" = torch.ops.aten.mul.Tensor(sum_20, squeeze_187);  sum_20 = squeeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:119, code: return self.linear(self.drop(self.bn(x)))
    add_392: "f32[8, 384]" = torch.ops.aten.add.Tensor(mul_500, mul_509);  mul_500 = mul_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:681, code: x = x.mean(dim=(-2, -1)) if self.use_conv else x.mean(dim=1)
    unsqueeze_24: "f32[8, 1, 384]" = torch.ops.aten.unsqueeze.default(add_392, 1);  add_392 = None
    expand_56: "f32[8, 16, 384]" = torch.ops.aten.expand.default(unsqueeze_24, [8, 16, 384]);  unsqueeze_24 = None
    div_47: "f32[8, 16, 384]" = torch.ops.aten.div.Scalar(expand_56, 16);  expand_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_353: "f32[128, 384]" = torch.ops.aten.view.default(div_47, [128, 384])
    unsqueeze_25: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_183, 0);  squeeze_183 = None
    sum_21: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_353, [0])
    sub_86: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_349, unsqueeze_25)
    mul_511: "f32[128, 384]" = torch.ops.aten.mul.Tensor(view_353, sub_86);  sub_86 = None
    sum_22: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_511, [0]);  mul_511 = None
    mul_512: "f32[384]" = torch.ops.aten.mul.Tensor(sum_21, 0.0078125)
    unsqueeze_26: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_512, 0);  mul_512 = None
    mul_513: "f32[384]" = torch.ops.aten.mul.Tensor(sum_22, 0.0078125)
    mul_514: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_184, squeeze_184)
    mul_515: "f32[384]" = torch.ops.aten.mul.Tensor(mul_513, mul_514);  mul_513 = mul_514 = None
    unsqueeze_27: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_515, 0);  mul_515 = None
    mul_516: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_184, primals_199);  primals_199 = None
    unsqueeze_28: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_516, 0);  mul_516 = None
    sub_87: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_349, unsqueeze_25);  view_349 = unsqueeze_25 = None
    mul_517: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_27);  sub_87 = unsqueeze_27 = None
    sub_88: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_353, mul_517);  view_353 = mul_517 = None
    sub_89: "f32[128, 384]" = torch.ops.aten.sub.Tensor(sub_88, unsqueeze_26);  sub_88 = unsqueeze_26 = None
    mul_518: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_28);  sub_89 = unsqueeze_28 = None
    mul_519: "f32[384]" = torch.ops.aten.mul.Tensor(sum_22, squeeze_184);  sum_22 = squeeze_184 = None
    view_354: "f32[8, 16, 384]" = torch.ops.aten.view.default(mul_518, [8, 16, 384]);  mul_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_355: "f32[128, 384]" = torch.ops.aten.view.default(view_354, [128, 384]);  view_354 = None
    permute_125: "f32[384, 128]" = torch.ops.aten.permute.default(view_355, [1, 0])
    mm_62: "f32[384, 768]" = torch.ops.aten.mm.default(permute_125, view_347);  permute_125 = view_347 = None
    permute_126: "f32[768, 384]" = torch.ops.aten.permute.default(mm_62, [1, 0]);  mm_62 = None
    permute_127: "f32[384, 768]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
    mm_63: "f32[128, 768]" = torch.ops.aten.mm.default(view_355, permute_127);  view_355 = permute_127 = None
    view_356: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_63, [8, 16, 768]);  mm_63 = None
    permute_128: "f32[384, 768]" = torch.ops.aten.permute.default(permute_126, [1, 0]);  permute_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    lt: "b8[8, 16, 768]" = torch.ops.aten.lt.Scalar(view_346, -3)
    le: "b8[8, 16, 768]" = torch.ops.aten.le.Scalar(view_346, 3)
    div_48: "f32[8, 16, 768]" = torch.ops.aten.div.Tensor(view_346, 3);  view_346 = None
    add_393: "f32[8, 16, 768]" = torch.ops.aten.add.Tensor(div_48, 0.5);  div_48 = None
    mul_520: "f32[8, 16, 768]" = torch.ops.aten.mul.Tensor(view_356, add_393);  add_393 = None
    where: "f32[8, 16, 768]" = torch.ops.aten.where.self(le, mul_520, view_356);  le = mul_520 = view_356 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_1: "f32[8, 16, 768]" = torch.ops.aten.where.self(lt, scalar_tensor, where);  lt = scalar_tensor = where = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_357: "f32[128, 768]" = torch.ops.aten.view.default(where_1, [128, 768]);  where_1 = None
    unsqueeze_29: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_180, 0);  squeeze_180 = None
    sum_23: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_357, [0])
    sub_90: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_345, unsqueeze_29)
    mul_521: "f32[128, 768]" = torch.ops.aten.mul.Tensor(view_357, sub_90);  sub_90 = None
    sum_24: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_521, [0]);  mul_521 = None
    mul_522: "f32[768]" = torch.ops.aten.mul.Tensor(sum_23, 0.0078125)
    unsqueeze_30: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_522, 0);  mul_522 = None
    mul_523: "f32[768]" = torch.ops.aten.mul.Tensor(sum_24, 0.0078125)
    mul_524: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
    mul_525: "f32[768]" = torch.ops.aten.mul.Tensor(mul_523, mul_524);  mul_523 = mul_524 = None
    unsqueeze_31: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_525, 0);  mul_525 = None
    mul_526: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_181, primals_196);  primals_196 = None
    unsqueeze_32: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_526, 0);  mul_526 = None
    sub_91: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_345, unsqueeze_29);  view_345 = unsqueeze_29 = None
    mul_527: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_31);  sub_91 = unsqueeze_31 = None
    sub_92: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_357, mul_527);  view_357 = mul_527 = None
    sub_93: "f32[128, 768]" = torch.ops.aten.sub.Tensor(sub_92, unsqueeze_30);  sub_92 = unsqueeze_30 = None
    mul_528: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_32);  sub_93 = unsqueeze_32 = None
    mul_529: "f32[768]" = torch.ops.aten.mul.Tensor(sum_24, squeeze_181);  sum_24 = squeeze_181 = None
    view_358: "f32[8, 16, 768]" = torch.ops.aten.view.default(mul_528, [8, 16, 768]);  mul_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_359: "f32[128, 768]" = torch.ops.aten.view.default(view_358, [128, 768]);  view_358 = None
    permute_129: "f32[768, 128]" = torch.ops.aten.permute.default(view_359, [1, 0])
    mm_64: "f32[768, 384]" = torch.ops.aten.mm.default(permute_129, view_343);  permute_129 = view_343 = None
    permute_130: "f32[384, 768]" = torch.ops.aten.permute.default(mm_64, [1, 0]);  mm_64 = None
    permute_131: "f32[768, 384]" = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None
    mm_65: "f32[128, 384]" = torch.ops.aten.mm.default(view_359, permute_131);  view_359 = permute_131 = None
    view_360: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_65, [8, 16, 384]);  mm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_394: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(div_47, view_360);  div_47 = view_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_132: "f32[768, 384]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_361: "f32[128, 384]" = torch.ops.aten.view.default(add_394, [128, 384])
    unsqueeze_33: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_177, 0);  squeeze_177 = None
    sum_25: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_361, [0])
    sub_94: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_341, unsqueeze_33)
    mul_530: "f32[128, 384]" = torch.ops.aten.mul.Tensor(view_361, sub_94);  sub_94 = None
    sum_26: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_530, [0]);  mul_530 = None
    mul_531: "f32[384]" = torch.ops.aten.mul.Tensor(sum_25, 0.0078125)
    unsqueeze_34: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_531, 0);  mul_531 = None
    mul_532: "f32[384]" = torch.ops.aten.mul.Tensor(sum_26, 0.0078125)
    mul_533: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_178, squeeze_178)
    mul_534: "f32[384]" = torch.ops.aten.mul.Tensor(mul_532, mul_533);  mul_532 = mul_533 = None
    unsqueeze_35: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_534, 0);  mul_534 = None
    mul_535: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_178, primals_193);  primals_193 = None
    unsqueeze_36: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_535, 0);  mul_535 = None
    sub_95: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_341, unsqueeze_33);  view_341 = unsqueeze_33 = None
    mul_536: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_35);  sub_95 = unsqueeze_35 = None
    sub_96: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_361, mul_536);  view_361 = mul_536 = None
    sub_97: "f32[128, 384]" = torch.ops.aten.sub.Tensor(sub_96, unsqueeze_34);  sub_96 = unsqueeze_34 = None
    mul_537: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_36);  sub_97 = unsqueeze_36 = None
    mul_538: "f32[384]" = torch.ops.aten.mul.Tensor(sum_26, squeeze_178);  sum_26 = squeeze_178 = None
    view_362: "f32[8, 16, 384]" = torch.ops.aten.view.default(mul_537, [8, 16, 384]);  mul_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_363: "f32[128, 384]" = torch.ops.aten.view.default(view_362, [128, 384]);  view_362 = None
    permute_133: "f32[384, 128]" = torch.ops.aten.permute.default(view_363, [1, 0])
    mm_66: "f32[384, 384]" = torch.ops.aten.mm.default(permute_133, view_339);  permute_133 = view_339 = None
    permute_134: "f32[384, 384]" = torch.ops.aten.permute.default(mm_66, [1, 0]);  mm_66 = None
    permute_135: "f32[384, 384]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
    mm_67: "f32[128, 384]" = torch.ops.aten.mm.default(view_363, permute_135);  view_363 = permute_135 = None
    view_364: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_67, [8, 16, 384]);  mm_67 = None
    permute_136: "f32[384, 384]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    lt_1: "b8[8, 16, 384]" = torch.ops.aten.lt.Scalar(view_338, -3)
    le_1: "b8[8, 16, 384]" = torch.ops.aten.le.Scalar(view_338, 3)
    div_49: "f32[8, 16, 384]" = torch.ops.aten.div.Tensor(view_338, 3);  view_338 = None
    add_395: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(div_49, 0.5);  div_49 = None
    mul_539: "f32[8, 16, 384]" = torch.ops.aten.mul.Tensor(view_364, add_395);  add_395 = None
    where_2: "f32[8, 16, 384]" = torch.ops.aten.where.self(le_1, mul_539, view_364);  le_1 = mul_539 = view_364 = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_3: "f32[8, 16, 384]" = torch.ops.aten.where.self(lt_1, scalar_tensor_1, where_2);  lt_1 = scalar_tensor_1 = where_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    view_365: "f32[8, 16, 12, 32]" = torch.ops.aten.view.default(where_3, [8, 16, 12, 32]);  where_3 = None
    permute_137: "f32[8, 12, 16, 32]" = torch.ops.aten.permute.default(view_365, [0, 2, 1, 3]);  view_365 = None
    clone_83: "f32[8, 12, 16, 32]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
    view_366: "f32[96, 16, 32]" = torch.ops.aten.view.default(clone_83, [96, 16, 32]);  clone_83 = None
    permute_138: "f32[96, 16, 16]" = torch.ops.aten.permute.default(view_335, [0, 2, 1]);  view_335 = None
    bmm_28: "f32[96, 16, 32]" = torch.ops.aten.bmm.default(permute_138, view_366);  permute_138 = None
    permute_139: "f32[96, 32, 16]" = torch.ops.aten.permute.default(view_336, [0, 2, 1]);  view_336 = None
    bmm_29: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(view_366, permute_139);  view_366 = permute_139 = None
    view_367: "f32[8, 12, 16, 32]" = torch.ops.aten.view.default(bmm_28, [8, 12, 16, 32]);  bmm_28 = None
    view_368: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_29, [8, 12, 16, 16]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    alias_14: "f32[8, 12, 16, 16]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    mul_540: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(view_368, alias_14);  view_368 = None
    sum_27: "f32[8, 12, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_540, [-1], True)
    mul_541: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(alias_14, sum_27);  alias_14 = sum_27 = None
    sub_98: "f32[8, 12, 16, 16]" = torch.ops.aten.sub.Tensor(mul_540, mul_541);  mul_540 = mul_541 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    sum_28: "f32[1, 12, 16, 16]" = torch.ops.aten.sum.dim_IntList(sub_98, [0], True)
    view_369: "f32[12, 16, 16]" = torch.ops.aten.view.default(sum_28, [12, 16, 16]);  sum_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    full: "f32[12, 16]" = torch.ops.aten.full.default([12, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    index_put: "f32[12, 16]" = torch.ops.aten.index_put.default(full, [None, primals_222], view_369, True);  full = primals_222 = view_369 = None
    full_1: "f32[12, 16]" = torch.ops.aten.full.default([12, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter: "f32[12, 16]" = torch.ops.aten.slice_scatter.default(full_1, index_put, 0, 0, 9223372036854775807);  full_1 = index_put = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    mul_542: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(sub_98, 0.25);  sub_98 = None
    view_370: "f32[96, 16, 16]" = torch.ops.aten.view.default(mul_542, [96, 16, 16]);  mul_542 = None
    permute_140: "f32[96, 16, 16]" = torch.ops.aten.permute.default(view_332, [0, 2, 1]);  view_332 = None
    bmm_30: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(permute_140, view_370);  permute_140 = None
    permute_141: "f32[96, 16, 16]" = torch.ops.aten.permute.default(view_333, [0, 2, 1]);  view_333 = None
    bmm_31: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(view_370, permute_141);  view_370 = permute_141 = None
    view_371: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_30, [8, 12, 16, 16]);  bmm_30 = None
    view_372: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_31, [8, 12, 16, 16]);  bmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_142: "f32[8, 16, 12, 32]" = torch.ops.aten.permute.default(view_367, [0, 2, 1, 3]);  view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_143: "f32[8, 16, 12, 16]" = torch.ops.aten.permute.default(view_371, [0, 3, 1, 2]);  view_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_144: "f32[8, 16, 12, 16]" = torch.ops.aten.permute.default(view_372, [0, 2, 1, 3]);  view_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    cat: "f32[8, 16, 12, 64]" = torch.ops.aten.cat.default([permute_144, permute_143, permute_142], 3);  permute_144 = permute_143 = permute_142 = None
    view_373: "f32[8, 16, 768]" = torch.ops.aten.view.default(cat, [8, 16, 768]);  cat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_374: "f32[128, 768]" = torch.ops.aten.view.default(view_373, [128, 768]);  view_373 = None
    unsqueeze_37: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_174, 0);  squeeze_174 = None
    sum_29: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_374, [0])
    sub_99: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_329, unsqueeze_37)
    mul_543: "f32[128, 768]" = torch.ops.aten.mul.Tensor(view_374, sub_99);  sub_99 = None
    sum_30: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_543, [0]);  mul_543 = None
    mul_544: "f32[768]" = torch.ops.aten.mul.Tensor(sum_29, 0.0078125)
    unsqueeze_38: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_544, 0);  mul_544 = None
    mul_545: "f32[768]" = torch.ops.aten.mul.Tensor(sum_30, 0.0078125)
    mul_546: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
    mul_547: "f32[768]" = torch.ops.aten.mul.Tensor(mul_545, mul_546);  mul_545 = mul_546 = None
    unsqueeze_39: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_547, 0);  mul_547 = None
    mul_548: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_175, primals_190);  primals_190 = None
    unsqueeze_40: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_548, 0);  mul_548 = None
    sub_100: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_329, unsqueeze_37);  view_329 = unsqueeze_37 = None
    mul_549: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_39);  sub_100 = unsqueeze_39 = None
    sub_101: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_374, mul_549);  view_374 = mul_549 = None
    sub_102: "f32[128, 768]" = torch.ops.aten.sub.Tensor(sub_101, unsqueeze_38);  sub_101 = unsqueeze_38 = None
    mul_550: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_40);  sub_102 = unsqueeze_40 = None
    mul_551: "f32[768]" = torch.ops.aten.mul.Tensor(sum_30, squeeze_175);  sum_30 = squeeze_175 = None
    view_375: "f32[8, 16, 768]" = torch.ops.aten.view.default(mul_550, [8, 16, 768]);  mul_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_376: "f32[128, 768]" = torch.ops.aten.view.default(view_375, [128, 768]);  view_375 = None
    permute_145: "f32[768, 128]" = torch.ops.aten.permute.default(view_376, [1, 0])
    mm_68: "f32[768, 384]" = torch.ops.aten.mm.default(permute_145, view_327);  permute_145 = view_327 = None
    permute_146: "f32[384, 768]" = torch.ops.aten.permute.default(mm_68, [1, 0]);  mm_68 = None
    permute_147: "f32[768, 384]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    mm_69: "f32[128, 384]" = torch.ops.aten.mm.default(view_376, permute_147);  view_376 = permute_147 = None
    view_377: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_69, [8, 16, 384]);  mm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_396: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_394, view_377);  add_394 = view_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_148: "f32[768, 384]" = torch.ops.aten.permute.default(permute_146, [1, 0]);  permute_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_378: "f32[128, 384]" = torch.ops.aten.view.default(add_396, [128, 384])
    unsqueeze_41: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_171, 0);  squeeze_171 = None
    sum_31: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_378, [0])
    sub_103: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_325, unsqueeze_41)
    mul_552: "f32[128, 384]" = torch.ops.aten.mul.Tensor(view_378, sub_103);  sub_103 = None
    sum_32: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_552, [0]);  mul_552 = None
    mul_553: "f32[384]" = torch.ops.aten.mul.Tensor(sum_31, 0.0078125)
    unsqueeze_42: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_553, 0);  mul_553 = None
    mul_554: "f32[384]" = torch.ops.aten.mul.Tensor(sum_32, 0.0078125)
    mul_555: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
    mul_556: "f32[384]" = torch.ops.aten.mul.Tensor(mul_554, mul_555);  mul_554 = mul_555 = None
    unsqueeze_43: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_556, 0);  mul_556 = None
    mul_557: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_172, primals_187);  primals_187 = None
    unsqueeze_44: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_557, 0);  mul_557 = None
    sub_104: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_325, unsqueeze_41);  view_325 = unsqueeze_41 = None
    mul_558: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_43);  sub_104 = unsqueeze_43 = None
    sub_105: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_378, mul_558);  view_378 = mul_558 = None
    sub_106: "f32[128, 384]" = torch.ops.aten.sub.Tensor(sub_105, unsqueeze_42);  sub_105 = unsqueeze_42 = None
    mul_559: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_44);  sub_106 = unsqueeze_44 = None
    mul_560: "f32[384]" = torch.ops.aten.mul.Tensor(sum_32, squeeze_172);  sum_32 = squeeze_172 = None
    view_379: "f32[8, 16, 384]" = torch.ops.aten.view.default(mul_559, [8, 16, 384]);  mul_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_380: "f32[128, 384]" = torch.ops.aten.view.default(view_379, [128, 384]);  view_379 = None
    permute_149: "f32[384, 128]" = torch.ops.aten.permute.default(view_380, [1, 0])
    mm_70: "f32[384, 768]" = torch.ops.aten.mm.default(permute_149, view_323);  permute_149 = view_323 = None
    permute_150: "f32[768, 384]" = torch.ops.aten.permute.default(mm_70, [1, 0]);  mm_70 = None
    permute_151: "f32[384, 768]" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
    mm_71: "f32[128, 768]" = torch.ops.aten.mm.default(view_380, permute_151);  view_380 = permute_151 = None
    view_381: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_71, [8, 16, 768]);  mm_71 = None
    permute_152: "f32[384, 768]" = torch.ops.aten.permute.default(permute_150, [1, 0]);  permute_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    lt_2: "b8[8, 16, 768]" = torch.ops.aten.lt.Scalar(view_322, -3)
    le_2: "b8[8, 16, 768]" = torch.ops.aten.le.Scalar(view_322, 3)
    div_50: "f32[8, 16, 768]" = torch.ops.aten.div.Tensor(view_322, 3);  view_322 = None
    add_397: "f32[8, 16, 768]" = torch.ops.aten.add.Tensor(div_50, 0.5);  div_50 = None
    mul_561: "f32[8, 16, 768]" = torch.ops.aten.mul.Tensor(view_381, add_397);  add_397 = None
    where_4: "f32[8, 16, 768]" = torch.ops.aten.where.self(le_2, mul_561, view_381);  le_2 = mul_561 = view_381 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_5: "f32[8, 16, 768]" = torch.ops.aten.where.self(lt_2, scalar_tensor_2, where_4);  lt_2 = scalar_tensor_2 = where_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_382: "f32[128, 768]" = torch.ops.aten.view.default(where_5, [128, 768]);  where_5 = None
    unsqueeze_45: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_168, 0);  squeeze_168 = None
    sum_33: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_382, [0])
    sub_107: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_321, unsqueeze_45)
    mul_562: "f32[128, 768]" = torch.ops.aten.mul.Tensor(view_382, sub_107);  sub_107 = None
    sum_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_562, [0]);  mul_562 = None
    mul_563: "f32[768]" = torch.ops.aten.mul.Tensor(sum_33, 0.0078125)
    unsqueeze_46: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_563, 0);  mul_563 = None
    mul_564: "f32[768]" = torch.ops.aten.mul.Tensor(sum_34, 0.0078125)
    mul_565: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
    mul_566: "f32[768]" = torch.ops.aten.mul.Tensor(mul_564, mul_565);  mul_564 = mul_565 = None
    unsqueeze_47: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_566, 0);  mul_566 = None
    mul_567: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_169, primals_184);  primals_184 = None
    unsqueeze_48: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_567, 0);  mul_567 = None
    sub_108: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_321, unsqueeze_45);  view_321 = unsqueeze_45 = None
    mul_568: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_47);  sub_108 = unsqueeze_47 = None
    sub_109: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_382, mul_568);  view_382 = mul_568 = None
    sub_110: "f32[128, 768]" = torch.ops.aten.sub.Tensor(sub_109, unsqueeze_46);  sub_109 = unsqueeze_46 = None
    mul_569: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_48);  sub_110 = unsqueeze_48 = None
    mul_570: "f32[768]" = torch.ops.aten.mul.Tensor(sum_34, squeeze_169);  sum_34 = squeeze_169 = None
    view_383: "f32[8, 16, 768]" = torch.ops.aten.view.default(mul_569, [8, 16, 768]);  mul_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_384: "f32[128, 768]" = torch.ops.aten.view.default(view_383, [128, 768]);  view_383 = None
    permute_153: "f32[768, 128]" = torch.ops.aten.permute.default(view_384, [1, 0])
    mm_72: "f32[768, 384]" = torch.ops.aten.mm.default(permute_153, view_319);  permute_153 = view_319 = None
    permute_154: "f32[384, 768]" = torch.ops.aten.permute.default(mm_72, [1, 0]);  mm_72 = None
    permute_155: "f32[768, 384]" = torch.ops.aten.permute.default(permute_105, [1, 0]);  permute_105 = None
    mm_73: "f32[128, 384]" = torch.ops.aten.mm.default(view_384, permute_155);  view_384 = permute_155 = None
    view_385: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_73, [8, 16, 384]);  mm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_398: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_396, view_385);  add_396 = view_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_156: "f32[768, 384]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_386: "f32[128, 384]" = torch.ops.aten.view.default(add_398, [128, 384])
    unsqueeze_49: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_165, 0);  squeeze_165 = None
    sum_35: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_386, [0])
    sub_111: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_317, unsqueeze_49)
    mul_571: "f32[128, 384]" = torch.ops.aten.mul.Tensor(view_386, sub_111);  sub_111 = None
    sum_36: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_571, [0]);  mul_571 = None
    mul_572: "f32[384]" = torch.ops.aten.mul.Tensor(sum_35, 0.0078125)
    unsqueeze_50: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_572, 0);  mul_572 = None
    mul_573: "f32[384]" = torch.ops.aten.mul.Tensor(sum_36, 0.0078125)
    mul_574: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
    mul_575: "f32[384]" = torch.ops.aten.mul.Tensor(mul_573, mul_574);  mul_573 = mul_574 = None
    unsqueeze_51: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_575, 0);  mul_575 = None
    mul_576: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_166, primals_181);  primals_181 = None
    unsqueeze_52: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_576, 0);  mul_576 = None
    sub_112: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_317, unsqueeze_49);  view_317 = unsqueeze_49 = None
    mul_577: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_51);  sub_112 = unsqueeze_51 = None
    sub_113: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_386, mul_577);  view_386 = mul_577 = None
    sub_114: "f32[128, 384]" = torch.ops.aten.sub.Tensor(sub_113, unsqueeze_50);  sub_113 = unsqueeze_50 = None
    mul_578: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_52);  sub_114 = unsqueeze_52 = None
    mul_579: "f32[384]" = torch.ops.aten.mul.Tensor(sum_36, squeeze_166);  sum_36 = squeeze_166 = None
    view_387: "f32[8, 16, 384]" = torch.ops.aten.view.default(mul_578, [8, 16, 384]);  mul_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_388: "f32[128, 384]" = torch.ops.aten.view.default(view_387, [128, 384]);  view_387 = None
    permute_157: "f32[384, 128]" = torch.ops.aten.permute.default(view_388, [1, 0])
    mm_74: "f32[384, 384]" = torch.ops.aten.mm.default(permute_157, view_315);  permute_157 = view_315 = None
    permute_158: "f32[384, 384]" = torch.ops.aten.permute.default(mm_74, [1, 0]);  mm_74 = None
    permute_159: "f32[384, 384]" = torch.ops.aten.permute.default(permute_104, [1, 0]);  permute_104 = None
    mm_75: "f32[128, 384]" = torch.ops.aten.mm.default(view_388, permute_159);  view_388 = permute_159 = None
    view_389: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_75, [8, 16, 384]);  mm_75 = None
    permute_160: "f32[384, 384]" = torch.ops.aten.permute.default(permute_158, [1, 0]);  permute_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    lt_3: "b8[8, 16, 384]" = torch.ops.aten.lt.Scalar(view_314, -3)
    le_3: "b8[8, 16, 384]" = torch.ops.aten.le.Scalar(view_314, 3)
    div_51: "f32[8, 16, 384]" = torch.ops.aten.div.Tensor(view_314, 3);  view_314 = None
    add_399: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(div_51, 0.5);  div_51 = None
    mul_580: "f32[8, 16, 384]" = torch.ops.aten.mul.Tensor(view_389, add_399);  add_399 = None
    where_6: "f32[8, 16, 384]" = torch.ops.aten.where.self(le_3, mul_580, view_389);  le_3 = mul_580 = view_389 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_7: "f32[8, 16, 384]" = torch.ops.aten.where.self(lt_3, scalar_tensor_3, where_6);  lt_3 = scalar_tensor_3 = where_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    view_390: "f32[8, 16, 12, 32]" = torch.ops.aten.view.default(where_7, [8, 16, 12, 32]);  where_7 = None
    permute_161: "f32[8, 12, 16, 32]" = torch.ops.aten.permute.default(view_390, [0, 2, 1, 3]);  view_390 = None
    clone_84: "f32[8, 12, 16, 32]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
    view_391: "f32[96, 16, 32]" = torch.ops.aten.view.default(clone_84, [96, 16, 32]);  clone_84 = None
    permute_162: "f32[96, 16, 16]" = torch.ops.aten.permute.default(view_311, [0, 2, 1]);  view_311 = None
    bmm_32: "f32[96, 16, 32]" = torch.ops.aten.bmm.default(permute_162, view_391);  permute_162 = None
    permute_163: "f32[96, 32, 16]" = torch.ops.aten.permute.default(view_312, [0, 2, 1]);  view_312 = None
    bmm_33: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(view_391, permute_163);  view_391 = permute_163 = None
    view_392: "f32[8, 12, 16, 32]" = torch.ops.aten.view.default(bmm_32, [8, 12, 16, 32]);  bmm_32 = None
    view_393: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_33, [8, 12, 16, 16]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    alias_15: "f32[8, 12, 16, 16]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    mul_581: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(view_393, alias_15);  view_393 = None
    sum_37: "f32[8, 12, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_581, [-1], True)
    mul_582: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(alias_15, sum_37);  alias_15 = sum_37 = None
    sub_115: "f32[8, 12, 16, 16]" = torch.ops.aten.sub.Tensor(mul_581, mul_582);  mul_581 = mul_582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    sum_38: "f32[1, 12, 16, 16]" = torch.ops.aten.sum.dim_IntList(sub_115, [0], True)
    view_394: "f32[12, 16, 16]" = torch.ops.aten.view.default(sum_38, [12, 16, 16]);  sum_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    full_2: "f32[12, 16]" = torch.ops.aten.full.default([12, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    index_put_1: "f32[12, 16]" = torch.ops.aten.index_put.default(full_2, [None, primals_221], view_394, True);  full_2 = primals_221 = view_394 = None
    full_3: "f32[12, 16]" = torch.ops.aten.full.default([12, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_1: "f32[12, 16]" = torch.ops.aten.slice_scatter.default(full_3, index_put_1, 0, 0, 9223372036854775807);  full_3 = index_put_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    mul_583: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(sub_115, 0.25);  sub_115 = None
    view_395: "f32[96, 16, 16]" = torch.ops.aten.view.default(mul_583, [96, 16, 16]);  mul_583 = None
    permute_164: "f32[96, 16, 16]" = torch.ops.aten.permute.default(view_308, [0, 2, 1]);  view_308 = None
    bmm_34: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(permute_164, view_395);  permute_164 = None
    permute_165: "f32[96, 16, 16]" = torch.ops.aten.permute.default(view_309, [0, 2, 1]);  view_309 = None
    bmm_35: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(view_395, permute_165);  view_395 = permute_165 = None
    view_396: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_34, [8, 12, 16, 16]);  bmm_34 = None
    view_397: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_35, [8, 12, 16, 16]);  bmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_166: "f32[8, 16, 12, 32]" = torch.ops.aten.permute.default(view_392, [0, 2, 1, 3]);  view_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_167: "f32[8, 16, 12, 16]" = torch.ops.aten.permute.default(view_396, [0, 3, 1, 2]);  view_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_168: "f32[8, 16, 12, 16]" = torch.ops.aten.permute.default(view_397, [0, 2, 1, 3]);  view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    cat_1: "f32[8, 16, 12, 64]" = torch.ops.aten.cat.default([permute_168, permute_167, permute_166], 3);  permute_168 = permute_167 = permute_166 = None
    view_398: "f32[8, 16, 768]" = torch.ops.aten.view.default(cat_1, [8, 16, 768]);  cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_399: "f32[128, 768]" = torch.ops.aten.view.default(view_398, [128, 768]);  view_398 = None
    unsqueeze_53: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_162, 0);  squeeze_162 = None
    sum_39: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_399, [0])
    sub_116: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_305, unsqueeze_53)
    mul_584: "f32[128, 768]" = torch.ops.aten.mul.Tensor(view_399, sub_116);  sub_116 = None
    sum_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_584, [0]);  mul_584 = None
    mul_585: "f32[768]" = torch.ops.aten.mul.Tensor(sum_39, 0.0078125)
    unsqueeze_54: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_585, 0);  mul_585 = None
    mul_586: "f32[768]" = torch.ops.aten.mul.Tensor(sum_40, 0.0078125)
    mul_587: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
    mul_588: "f32[768]" = torch.ops.aten.mul.Tensor(mul_586, mul_587);  mul_586 = mul_587 = None
    unsqueeze_55: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_588, 0);  mul_588 = None
    mul_589: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_163, primals_178);  primals_178 = None
    unsqueeze_56: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_589, 0);  mul_589 = None
    sub_117: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_305, unsqueeze_53);  view_305 = unsqueeze_53 = None
    mul_590: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_55);  sub_117 = unsqueeze_55 = None
    sub_118: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_399, mul_590);  view_399 = mul_590 = None
    sub_119: "f32[128, 768]" = torch.ops.aten.sub.Tensor(sub_118, unsqueeze_54);  sub_118 = unsqueeze_54 = None
    mul_591: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_56);  sub_119 = unsqueeze_56 = None
    mul_592: "f32[768]" = torch.ops.aten.mul.Tensor(sum_40, squeeze_163);  sum_40 = squeeze_163 = None
    view_400: "f32[8, 16, 768]" = torch.ops.aten.view.default(mul_591, [8, 16, 768]);  mul_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_401: "f32[128, 768]" = torch.ops.aten.view.default(view_400, [128, 768]);  view_400 = None
    permute_169: "f32[768, 128]" = torch.ops.aten.permute.default(view_401, [1, 0])
    mm_76: "f32[768, 384]" = torch.ops.aten.mm.default(permute_169, view_303);  permute_169 = view_303 = None
    permute_170: "f32[384, 768]" = torch.ops.aten.permute.default(mm_76, [1, 0]);  mm_76 = None
    permute_171: "f32[768, 384]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    mm_77: "f32[128, 384]" = torch.ops.aten.mm.default(view_401, permute_171);  view_401 = permute_171 = None
    view_402: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_77, [8, 16, 384]);  mm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_400: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_398, view_402);  add_398 = view_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_172: "f32[768, 384]" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_403: "f32[128, 384]" = torch.ops.aten.view.default(add_400, [128, 384])
    unsqueeze_57: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_159, 0);  squeeze_159 = None
    sum_41: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_403, [0])
    sub_120: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_301, unsqueeze_57)
    mul_593: "f32[128, 384]" = torch.ops.aten.mul.Tensor(view_403, sub_120);  sub_120 = None
    sum_42: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_593, [0]);  mul_593 = None
    mul_594: "f32[384]" = torch.ops.aten.mul.Tensor(sum_41, 0.0078125)
    unsqueeze_58: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_594, 0);  mul_594 = None
    mul_595: "f32[384]" = torch.ops.aten.mul.Tensor(sum_42, 0.0078125)
    mul_596: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
    mul_597: "f32[384]" = torch.ops.aten.mul.Tensor(mul_595, mul_596);  mul_595 = mul_596 = None
    unsqueeze_59: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_597, 0);  mul_597 = None
    mul_598: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_160, primals_175);  primals_175 = None
    unsqueeze_60: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_598, 0);  mul_598 = None
    sub_121: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_301, unsqueeze_57);  view_301 = unsqueeze_57 = None
    mul_599: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_59);  sub_121 = unsqueeze_59 = None
    sub_122: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_403, mul_599);  view_403 = mul_599 = None
    sub_123: "f32[128, 384]" = torch.ops.aten.sub.Tensor(sub_122, unsqueeze_58);  sub_122 = unsqueeze_58 = None
    mul_600: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_60);  sub_123 = unsqueeze_60 = None
    mul_601: "f32[384]" = torch.ops.aten.mul.Tensor(sum_42, squeeze_160);  sum_42 = squeeze_160 = None
    view_404: "f32[8, 16, 384]" = torch.ops.aten.view.default(mul_600, [8, 16, 384]);  mul_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_405: "f32[128, 384]" = torch.ops.aten.view.default(view_404, [128, 384]);  view_404 = None
    permute_173: "f32[384, 128]" = torch.ops.aten.permute.default(view_405, [1, 0])
    mm_78: "f32[384, 768]" = torch.ops.aten.mm.default(permute_173, view_299);  permute_173 = view_299 = None
    permute_174: "f32[768, 384]" = torch.ops.aten.permute.default(mm_78, [1, 0]);  mm_78 = None
    permute_175: "f32[384, 768]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    mm_79: "f32[128, 768]" = torch.ops.aten.mm.default(view_405, permute_175);  view_405 = permute_175 = None
    view_406: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_79, [8, 16, 768]);  mm_79 = None
    permute_176: "f32[384, 768]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    lt_4: "b8[8, 16, 768]" = torch.ops.aten.lt.Scalar(view_298, -3)
    le_4: "b8[8, 16, 768]" = torch.ops.aten.le.Scalar(view_298, 3)
    div_52: "f32[8, 16, 768]" = torch.ops.aten.div.Tensor(view_298, 3);  view_298 = None
    add_401: "f32[8, 16, 768]" = torch.ops.aten.add.Tensor(div_52, 0.5);  div_52 = None
    mul_602: "f32[8, 16, 768]" = torch.ops.aten.mul.Tensor(view_406, add_401);  add_401 = None
    where_8: "f32[8, 16, 768]" = torch.ops.aten.where.self(le_4, mul_602, view_406);  le_4 = mul_602 = view_406 = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_9: "f32[8, 16, 768]" = torch.ops.aten.where.self(lt_4, scalar_tensor_4, where_8);  lt_4 = scalar_tensor_4 = where_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_407: "f32[128, 768]" = torch.ops.aten.view.default(where_9, [128, 768]);  where_9 = None
    unsqueeze_61: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_156, 0);  squeeze_156 = None
    sum_43: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_407, [0])
    sub_124: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_297, unsqueeze_61)
    mul_603: "f32[128, 768]" = torch.ops.aten.mul.Tensor(view_407, sub_124);  sub_124 = None
    sum_44: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_603, [0]);  mul_603 = None
    mul_604: "f32[768]" = torch.ops.aten.mul.Tensor(sum_43, 0.0078125)
    unsqueeze_62: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_604, 0);  mul_604 = None
    mul_605: "f32[768]" = torch.ops.aten.mul.Tensor(sum_44, 0.0078125)
    mul_606: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
    mul_607: "f32[768]" = torch.ops.aten.mul.Tensor(mul_605, mul_606);  mul_605 = mul_606 = None
    unsqueeze_63: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_607, 0);  mul_607 = None
    mul_608: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_157, primals_172);  primals_172 = None
    unsqueeze_64: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_608, 0);  mul_608 = None
    sub_125: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_297, unsqueeze_61);  view_297 = unsqueeze_61 = None
    mul_609: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_63);  sub_125 = unsqueeze_63 = None
    sub_126: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_407, mul_609);  view_407 = mul_609 = None
    sub_127: "f32[128, 768]" = torch.ops.aten.sub.Tensor(sub_126, unsqueeze_62);  sub_126 = unsqueeze_62 = None
    mul_610: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_64);  sub_127 = unsqueeze_64 = None
    mul_611: "f32[768]" = torch.ops.aten.mul.Tensor(sum_44, squeeze_157);  sum_44 = squeeze_157 = None
    view_408: "f32[8, 16, 768]" = torch.ops.aten.view.default(mul_610, [8, 16, 768]);  mul_610 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_409: "f32[128, 768]" = torch.ops.aten.view.default(view_408, [128, 768]);  view_408 = None
    permute_177: "f32[768, 128]" = torch.ops.aten.permute.default(view_409, [1, 0])
    mm_80: "f32[768, 384]" = torch.ops.aten.mm.default(permute_177, view_295);  permute_177 = view_295 = None
    permute_178: "f32[384, 768]" = torch.ops.aten.permute.default(mm_80, [1, 0]);  mm_80 = None
    permute_179: "f32[768, 384]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    mm_81: "f32[128, 384]" = torch.ops.aten.mm.default(view_409, permute_179);  view_409 = permute_179 = None
    view_410: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_81, [8, 16, 384]);  mm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_402: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_400, view_410);  add_400 = view_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_180: "f32[768, 384]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_411: "f32[128, 384]" = torch.ops.aten.view.default(add_402, [128, 384])
    unsqueeze_65: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
    sum_45: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_411, [0])
    sub_128: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_293, unsqueeze_65)
    mul_612: "f32[128, 384]" = torch.ops.aten.mul.Tensor(view_411, sub_128);  sub_128 = None
    sum_46: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_612, [0]);  mul_612 = None
    mul_613: "f32[384]" = torch.ops.aten.mul.Tensor(sum_45, 0.0078125)
    unsqueeze_66: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_613, 0);  mul_613 = None
    mul_614: "f32[384]" = torch.ops.aten.mul.Tensor(sum_46, 0.0078125)
    mul_615: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
    mul_616: "f32[384]" = torch.ops.aten.mul.Tensor(mul_614, mul_615);  mul_614 = mul_615 = None
    unsqueeze_67: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_616, 0);  mul_616 = None
    mul_617: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_169);  primals_169 = None
    unsqueeze_68: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_617, 0);  mul_617 = None
    sub_129: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_293, unsqueeze_65);  view_293 = unsqueeze_65 = None
    mul_618: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_67);  sub_129 = unsqueeze_67 = None
    sub_130: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_411, mul_618);  view_411 = mul_618 = None
    sub_131: "f32[128, 384]" = torch.ops.aten.sub.Tensor(sub_130, unsqueeze_66);  sub_130 = unsqueeze_66 = None
    mul_619: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_68);  sub_131 = unsqueeze_68 = None
    mul_620: "f32[384]" = torch.ops.aten.mul.Tensor(sum_46, squeeze_154);  sum_46 = squeeze_154 = None
    view_412: "f32[8, 16, 384]" = torch.ops.aten.view.default(mul_619, [8, 16, 384]);  mul_619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_413: "f32[128, 384]" = torch.ops.aten.view.default(view_412, [128, 384]);  view_412 = None
    permute_181: "f32[384, 128]" = torch.ops.aten.permute.default(view_413, [1, 0])
    mm_82: "f32[384, 384]" = torch.ops.aten.mm.default(permute_181, view_291);  permute_181 = view_291 = None
    permute_182: "f32[384, 384]" = torch.ops.aten.permute.default(mm_82, [1, 0]);  mm_82 = None
    permute_183: "f32[384, 384]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    mm_83: "f32[128, 384]" = torch.ops.aten.mm.default(view_413, permute_183);  view_413 = permute_183 = None
    view_414: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_83, [8, 16, 384]);  mm_83 = None
    permute_184: "f32[384, 384]" = torch.ops.aten.permute.default(permute_182, [1, 0]);  permute_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    lt_5: "b8[8, 16, 384]" = torch.ops.aten.lt.Scalar(view_290, -3)
    le_5: "b8[8, 16, 384]" = torch.ops.aten.le.Scalar(view_290, 3)
    div_53: "f32[8, 16, 384]" = torch.ops.aten.div.Tensor(view_290, 3);  view_290 = None
    add_403: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(div_53, 0.5);  div_53 = None
    mul_621: "f32[8, 16, 384]" = torch.ops.aten.mul.Tensor(view_414, add_403);  add_403 = None
    where_10: "f32[8, 16, 384]" = torch.ops.aten.where.self(le_5, mul_621, view_414);  le_5 = mul_621 = view_414 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_11: "f32[8, 16, 384]" = torch.ops.aten.where.self(lt_5, scalar_tensor_5, where_10);  lt_5 = scalar_tensor_5 = where_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    view_415: "f32[8, 16, 12, 32]" = torch.ops.aten.view.default(where_11, [8, 16, 12, 32]);  where_11 = None
    permute_185: "f32[8, 12, 16, 32]" = torch.ops.aten.permute.default(view_415, [0, 2, 1, 3]);  view_415 = None
    clone_85: "f32[8, 12, 16, 32]" = torch.ops.aten.clone.default(permute_185, memory_format = torch.contiguous_format);  permute_185 = None
    view_416: "f32[96, 16, 32]" = torch.ops.aten.view.default(clone_85, [96, 16, 32]);  clone_85 = None
    permute_186: "f32[96, 16, 16]" = torch.ops.aten.permute.default(view_287, [0, 2, 1]);  view_287 = None
    bmm_36: "f32[96, 16, 32]" = torch.ops.aten.bmm.default(permute_186, view_416);  permute_186 = None
    permute_187: "f32[96, 32, 16]" = torch.ops.aten.permute.default(view_288, [0, 2, 1]);  view_288 = None
    bmm_37: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(view_416, permute_187);  view_416 = permute_187 = None
    view_417: "f32[8, 12, 16, 32]" = torch.ops.aten.view.default(bmm_36, [8, 12, 16, 32]);  bmm_36 = None
    view_418: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_37, [8, 12, 16, 16]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    alias_16: "f32[8, 12, 16, 16]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    mul_622: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(view_418, alias_16);  view_418 = None
    sum_47: "f32[8, 12, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_622, [-1], True)
    mul_623: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(alias_16, sum_47);  alias_16 = sum_47 = None
    sub_132: "f32[8, 12, 16, 16]" = torch.ops.aten.sub.Tensor(mul_622, mul_623);  mul_622 = mul_623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    sum_48: "f32[1, 12, 16, 16]" = torch.ops.aten.sum.dim_IntList(sub_132, [0], True)
    view_419: "f32[12, 16, 16]" = torch.ops.aten.view.default(sum_48, [12, 16, 16]);  sum_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    full_4: "f32[12, 16]" = torch.ops.aten.full.default([12, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    index_put_2: "f32[12, 16]" = torch.ops.aten.index_put.default(full_4, [None, primals_220], view_419, True);  full_4 = primals_220 = view_419 = None
    full_5: "f32[12, 16]" = torch.ops.aten.full.default([12, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_2: "f32[12, 16]" = torch.ops.aten.slice_scatter.default(full_5, index_put_2, 0, 0, 9223372036854775807);  full_5 = index_put_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    mul_624: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(sub_132, 0.25);  sub_132 = None
    view_420: "f32[96, 16, 16]" = torch.ops.aten.view.default(mul_624, [96, 16, 16]);  mul_624 = None
    permute_188: "f32[96, 16, 16]" = torch.ops.aten.permute.default(view_284, [0, 2, 1]);  view_284 = None
    bmm_38: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(permute_188, view_420);  permute_188 = None
    permute_189: "f32[96, 16, 16]" = torch.ops.aten.permute.default(view_285, [0, 2, 1]);  view_285 = None
    bmm_39: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(view_420, permute_189);  view_420 = permute_189 = None
    view_421: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_38, [8, 12, 16, 16]);  bmm_38 = None
    view_422: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_39, [8, 12, 16, 16]);  bmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_190: "f32[8, 16, 12, 32]" = torch.ops.aten.permute.default(view_417, [0, 2, 1, 3]);  view_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_191: "f32[8, 16, 12, 16]" = torch.ops.aten.permute.default(view_421, [0, 3, 1, 2]);  view_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_192: "f32[8, 16, 12, 16]" = torch.ops.aten.permute.default(view_422, [0, 2, 1, 3]);  view_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    cat_2: "f32[8, 16, 12, 64]" = torch.ops.aten.cat.default([permute_192, permute_191, permute_190], 3);  permute_192 = permute_191 = permute_190 = None
    view_423: "f32[8, 16, 768]" = torch.ops.aten.view.default(cat_2, [8, 16, 768]);  cat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_424: "f32[128, 768]" = torch.ops.aten.view.default(view_423, [128, 768]);  view_423 = None
    unsqueeze_69: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_150, 0);  squeeze_150 = None
    sum_49: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_424, [0])
    sub_133: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_281, unsqueeze_69)
    mul_625: "f32[128, 768]" = torch.ops.aten.mul.Tensor(view_424, sub_133);  sub_133 = None
    sum_50: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_625, [0]);  mul_625 = None
    mul_626: "f32[768]" = torch.ops.aten.mul.Tensor(sum_49, 0.0078125)
    unsqueeze_70: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_626, 0);  mul_626 = None
    mul_627: "f32[768]" = torch.ops.aten.mul.Tensor(sum_50, 0.0078125)
    mul_628: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
    mul_629: "f32[768]" = torch.ops.aten.mul.Tensor(mul_627, mul_628);  mul_627 = mul_628 = None
    unsqueeze_71: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_629, 0);  mul_629 = None
    mul_630: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_166);  primals_166 = None
    unsqueeze_72: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_630, 0);  mul_630 = None
    sub_134: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_281, unsqueeze_69);  view_281 = unsqueeze_69 = None
    mul_631: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_71);  sub_134 = unsqueeze_71 = None
    sub_135: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_424, mul_631);  view_424 = mul_631 = None
    sub_136: "f32[128, 768]" = torch.ops.aten.sub.Tensor(sub_135, unsqueeze_70);  sub_135 = unsqueeze_70 = None
    mul_632: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_72);  sub_136 = unsqueeze_72 = None
    mul_633: "f32[768]" = torch.ops.aten.mul.Tensor(sum_50, squeeze_151);  sum_50 = squeeze_151 = None
    view_425: "f32[8, 16, 768]" = torch.ops.aten.view.default(mul_632, [8, 16, 768]);  mul_632 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_426: "f32[128, 768]" = torch.ops.aten.view.default(view_425, [128, 768]);  view_425 = None
    permute_193: "f32[768, 128]" = torch.ops.aten.permute.default(view_426, [1, 0])
    mm_84: "f32[768, 384]" = torch.ops.aten.mm.default(permute_193, view_279);  permute_193 = view_279 = None
    permute_194: "f32[384, 768]" = torch.ops.aten.permute.default(mm_84, [1, 0]);  mm_84 = None
    permute_195: "f32[768, 384]" = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
    mm_85: "f32[128, 384]" = torch.ops.aten.mm.default(view_426, permute_195);  view_426 = permute_195 = None
    view_427: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_85, [8, 16, 384]);  mm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_404: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_402, view_427);  add_402 = view_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_196: "f32[768, 384]" = torch.ops.aten.permute.default(permute_194, [1, 0]);  permute_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_428: "f32[128, 384]" = torch.ops.aten.view.default(add_404, [128, 384])
    unsqueeze_73: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_147, 0);  squeeze_147 = None
    sum_51: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_428, [0])
    sub_137: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_277, unsqueeze_73)
    mul_634: "f32[128, 384]" = torch.ops.aten.mul.Tensor(view_428, sub_137);  sub_137 = None
    sum_52: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_634, [0]);  mul_634 = None
    mul_635: "f32[384]" = torch.ops.aten.mul.Tensor(sum_51, 0.0078125)
    unsqueeze_74: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_635, 0);  mul_635 = None
    mul_636: "f32[384]" = torch.ops.aten.mul.Tensor(sum_52, 0.0078125)
    mul_637: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
    mul_638: "f32[384]" = torch.ops.aten.mul.Tensor(mul_636, mul_637);  mul_636 = mul_637 = None
    unsqueeze_75: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_638, 0);  mul_638 = None
    mul_639: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_163);  primals_163 = None
    unsqueeze_76: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_639, 0);  mul_639 = None
    sub_138: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_277, unsqueeze_73);  view_277 = unsqueeze_73 = None
    mul_640: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_75);  sub_138 = unsqueeze_75 = None
    sub_139: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_428, mul_640);  view_428 = mul_640 = None
    sub_140: "f32[128, 384]" = torch.ops.aten.sub.Tensor(sub_139, unsqueeze_74);  sub_139 = unsqueeze_74 = None
    mul_641: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_76);  sub_140 = unsqueeze_76 = None
    mul_642: "f32[384]" = torch.ops.aten.mul.Tensor(sum_52, squeeze_148);  sum_52 = squeeze_148 = None
    view_429: "f32[8, 16, 384]" = torch.ops.aten.view.default(mul_641, [8, 16, 384]);  mul_641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_430: "f32[128, 384]" = torch.ops.aten.view.default(view_429, [128, 384]);  view_429 = None
    permute_197: "f32[384, 128]" = torch.ops.aten.permute.default(view_430, [1, 0])
    mm_86: "f32[384, 768]" = torch.ops.aten.mm.default(permute_197, view_275);  permute_197 = view_275 = None
    permute_198: "f32[768, 384]" = torch.ops.aten.permute.default(mm_86, [1, 0]);  mm_86 = None
    permute_199: "f32[384, 768]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
    mm_87: "f32[128, 768]" = torch.ops.aten.mm.default(view_430, permute_199);  view_430 = permute_199 = None
    view_431: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_87, [8, 16, 768]);  mm_87 = None
    permute_200: "f32[384, 768]" = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    lt_6: "b8[8, 16, 768]" = torch.ops.aten.lt.Scalar(view_274, -3)
    le_6: "b8[8, 16, 768]" = torch.ops.aten.le.Scalar(view_274, 3)
    div_54: "f32[8, 16, 768]" = torch.ops.aten.div.Tensor(view_274, 3);  view_274 = None
    add_405: "f32[8, 16, 768]" = torch.ops.aten.add.Tensor(div_54, 0.5);  div_54 = None
    mul_643: "f32[8, 16, 768]" = torch.ops.aten.mul.Tensor(view_431, add_405);  add_405 = None
    where_12: "f32[8, 16, 768]" = torch.ops.aten.where.self(le_6, mul_643, view_431);  le_6 = mul_643 = view_431 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_13: "f32[8, 16, 768]" = torch.ops.aten.where.self(lt_6, scalar_tensor_6, where_12);  lt_6 = scalar_tensor_6 = where_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_432: "f32[128, 768]" = torch.ops.aten.view.default(where_13, [128, 768]);  where_13 = None
    unsqueeze_77: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
    sum_53: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_432, [0])
    sub_141: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_273, unsqueeze_77)
    mul_644: "f32[128, 768]" = torch.ops.aten.mul.Tensor(view_432, sub_141);  sub_141 = None
    sum_54: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_644, [0]);  mul_644 = None
    mul_645: "f32[768]" = torch.ops.aten.mul.Tensor(sum_53, 0.0078125)
    unsqueeze_78: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_645, 0);  mul_645 = None
    mul_646: "f32[768]" = torch.ops.aten.mul.Tensor(sum_54, 0.0078125)
    mul_647: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
    mul_648: "f32[768]" = torch.ops.aten.mul.Tensor(mul_646, mul_647);  mul_646 = mul_647 = None
    unsqueeze_79: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_648, 0);  mul_648 = None
    mul_649: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_160);  primals_160 = None
    unsqueeze_80: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_649, 0);  mul_649 = None
    sub_142: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_273, unsqueeze_77);  view_273 = unsqueeze_77 = None
    mul_650: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_79);  sub_142 = unsqueeze_79 = None
    sub_143: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_432, mul_650);  view_432 = mul_650 = None
    sub_144: "f32[128, 768]" = torch.ops.aten.sub.Tensor(sub_143, unsqueeze_78);  sub_143 = unsqueeze_78 = None
    mul_651: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_80);  sub_144 = unsqueeze_80 = None
    mul_652: "f32[768]" = torch.ops.aten.mul.Tensor(sum_54, squeeze_145);  sum_54 = squeeze_145 = None
    view_433: "f32[8, 16, 768]" = torch.ops.aten.view.default(mul_651, [8, 16, 768]);  mul_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_434: "f32[128, 768]" = torch.ops.aten.view.default(view_433, [128, 768]);  view_433 = None
    permute_201: "f32[768, 128]" = torch.ops.aten.permute.default(view_434, [1, 0])
    mm_88: "f32[768, 384]" = torch.ops.aten.mm.default(permute_201, view_271);  permute_201 = view_271 = None
    permute_202: "f32[384, 768]" = torch.ops.aten.permute.default(mm_88, [1, 0]);  mm_88 = None
    permute_203: "f32[768, 384]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    mm_89: "f32[128, 384]" = torch.ops.aten.mm.default(view_434, permute_203);  view_434 = permute_203 = None
    view_435: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_89, [8, 16, 384]);  mm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_406: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_404, view_435);  add_404 = view_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_204: "f32[768, 384]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_436: "f32[128, 384]" = torch.ops.aten.view.default(add_406, [128, 384])
    unsqueeze_81: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
    sum_55: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_436, [0])
    sub_145: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_269, unsqueeze_81)
    mul_653: "f32[128, 384]" = torch.ops.aten.mul.Tensor(view_436, sub_145);  sub_145 = None
    sum_56: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_653, [0]);  mul_653 = None
    mul_654: "f32[384]" = torch.ops.aten.mul.Tensor(sum_55, 0.0078125)
    unsqueeze_82: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_654, 0);  mul_654 = None
    mul_655: "f32[384]" = torch.ops.aten.mul.Tensor(sum_56, 0.0078125)
    mul_656: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
    mul_657: "f32[384]" = torch.ops.aten.mul.Tensor(mul_655, mul_656);  mul_655 = mul_656 = None
    unsqueeze_83: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_657, 0);  mul_657 = None
    mul_658: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_157);  primals_157 = None
    unsqueeze_84: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_658, 0);  mul_658 = None
    sub_146: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_269, unsqueeze_81);  view_269 = unsqueeze_81 = None
    mul_659: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_83);  sub_146 = unsqueeze_83 = None
    sub_147: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_436, mul_659);  view_436 = mul_659 = None
    sub_148: "f32[128, 384]" = torch.ops.aten.sub.Tensor(sub_147, unsqueeze_82);  sub_147 = unsqueeze_82 = None
    mul_660: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_84);  sub_148 = unsqueeze_84 = None
    mul_661: "f32[384]" = torch.ops.aten.mul.Tensor(sum_56, squeeze_142);  sum_56 = squeeze_142 = None
    view_437: "f32[8, 16, 384]" = torch.ops.aten.view.default(mul_660, [8, 16, 384]);  mul_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_438: "f32[128, 384]" = torch.ops.aten.view.default(view_437, [128, 384]);  view_437 = None
    permute_205: "f32[384, 128]" = torch.ops.aten.permute.default(view_438, [1, 0])
    mm_90: "f32[384, 384]" = torch.ops.aten.mm.default(permute_205, view_267);  permute_205 = view_267 = None
    permute_206: "f32[384, 384]" = torch.ops.aten.permute.default(mm_90, [1, 0]);  mm_90 = None
    permute_207: "f32[384, 384]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    mm_91: "f32[128, 384]" = torch.ops.aten.mm.default(view_438, permute_207);  view_438 = permute_207 = None
    view_439: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_91, [8, 16, 384]);  mm_91 = None
    permute_208: "f32[384, 384]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    lt_7: "b8[8, 16, 384]" = torch.ops.aten.lt.Scalar(view_266, -3)
    le_7: "b8[8, 16, 384]" = torch.ops.aten.le.Scalar(view_266, 3)
    div_55: "f32[8, 16, 384]" = torch.ops.aten.div.Tensor(view_266, 3);  view_266 = None
    add_407: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(div_55, 0.5);  div_55 = None
    mul_662: "f32[8, 16, 384]" = torch.ops.aten.mul.Tensor(view_439, add_407);  add_407 = None
    where_14: "f32[8, 16, 384]" = torch.ops.aten.where.self(le_7, mul_662, view_439);  le_7 = mul_662 = view_439 = None
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_15: "f32[8, 16, 384]" = torch.ops.aten.where.self(lt_7, scalar_tensor_7, where_14);  lt_7 = scalar_tensor_7 = where_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    view_440: "f32[8, 16, 12, 32]" = torch.ops.aten.view.default(where_15, [8, 16, 12, 32]);  where_15 = None
    permute_209: "f32[8, 12, 16, 32]" = torch.ops.aten.permute.default(view_440, [0, 2, 1, 3]);  view_440 = None
    clone_86: "f32[8, 12, 16, 32]" = torch.ops.aten.clone.default(permute_209, memory_format = torch.contiguous_format);  permute_209 = None
    view_441: "f32[96, 16, 32]" = torch.ops.aten.view.default(clone_86, [96, 16, 32]);  clone_86 = None
    permute_210: "f32[96, 16, 16]" = torch.ops.aten.permute.default(view_263, [0, 2, 1]);  view_263 = None
    bmm_40: "f32[96, 16, 32]" = torch.ops.aten.bmm.default(permute_210, view_441);  permute_210 = None
    permute_211: "f32[96, 32, 16]" = torch.ops.aten.permute.default(view_264, [0, 2, 1]);  view_264 = None
    bmm_41: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(view_441, permute_211);  view_441 = permute_211 = None
    view_442: "f32[8, 12, 16, 32]" = torch.ops.aten.view.default(bmm_40, [8, 12, 16, 32]);  bmm_40 = None
    view_443: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_41, [8, 12, 16, 16]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    alias_17: "f32[8, 12, 16, 16]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    mul_663: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(view_443, alias_17);  view_443 = None
    sum_57: "f32[8, 12, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_663, [-1], True)
    mul_664: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(alias_17, sum_57);  alias_17 = sum_57 = None
    sub_149: "f32[8, 12, 16, 16]" = torch.ops.aten.sub.Tensor(mul_663, mul_664);  mul_663 = mul_664 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    sum_58: "f32[1, 12, 16, 16]" = torch.ops.aten.sum.dim_IntList(sub_149, [0], True)
    view_444: "f32[12, 16, 16]" = torch.ops.aten.view.default(sum_58, [12, 16, 16]);  sum_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    full_6: "f32[12, 16]" = torch.ops.aten.full.default([12, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    index_put_3: "f32[12, 16]" = torch.ops.aten.index_put.default(full_6, [None, primals_219], view_444, True);  full_6 = primals_219 = view_444 = None
    full_7: "f32[12, 16]" = torch.ops.aten.full.default([12, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_3: "f32[12, 16]" = torch.ops.aten.slice_scatter.default(full_7, index_put_3, 0, 0, 9223372036854775807);  full_7 = index_put_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    mul_665: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(sub_149, 0.25);  sub_149 = None
    view_445: "f32[96, 16, 16]" = torch.ops.aten.view.default(mul_665, [96, 16, 16]);  mul_665 = None
    permute_212: "f32[96, 16, 16]" = torch.ops.aten.permute.default(view_260, [0, 2, 1]);  view_260 = None
    bmm_42: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(permute_212, view_445);  permute_212 = None
    permute_213: "f32[96, 16, 16]" = torch.ops.aten.permute.default(view_261, [0, 2, 1]);  view_261 = None
    bmm_43: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(view_445, permute_213);  view_445 = permute_213 = None
    view_446: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_42, [8, 12, 16, 16]);  bmm_42 = None
    view_447: "f32[8, 12, 16, 16]" = torch.ops.aten.view.default(bmm_43, [8, 12, 16, 16]);  bmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_214: "f32[8, 16, 12, 32]" = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_215: "f32[8, 16, 12, 16]" = torch.ops.aten.permute.default(view_446, [0, 3, 1, 2]);  view_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_216: "f32[8, 16, 12, 16]" = torch.ops.aten.permute.default(view_447, [0, 2, 1, 3]);  view_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    cat_3: "f32[8, 16, 12, 64]" = torch.ops.aten.cat.default([permute_216, permute_215, permute_214], 3);  permute_216 = permute_215 = permute_214 = None
    view_448: "f32[8, 16, 768]" = torch.ops.aten.view.default(cat_3, [8, 16, 768]);  cat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_449: "f32[128, 768]" = torch.ops.aten.view.default(view_448, [128, 768]);  view_448 = None
    unsqueeze_85: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
    sum_59: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_449, [0])
    sub_150: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_257, unsqueeze_85)
    mul_666: "f32[128, 768]" = torch.ops.aten.mul.Tensor(view_449, sub_150);  sub_150 = None
    sum_60: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_666, [0]);  mul_666 = None
    mul_667: "f32[768]" = torch.ops.aten.mul.Tensor(sum_59, 0.0078125)
    unsqueeze_86: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_667, 0);  mul_667 = None
    mul_668: "f32[768]" = torch.ops.aten.mul.Tensor(sum_60, 0.0078125)
    mul_669: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
    mul_670: "f32[768]" = torch.ops.aten.mul.Tensor(mul_668, mul_669);  mul_668 = mul_669 = None
    unsqueeze_87: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_670, 0);  mul_670 = None
    mul_671: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_154);  primals_154 = None
    unsqueeze_88: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_671, 0);  mul_671 = None
    sub_151: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_257, unsqueeze_85);  view_257 = unsqueeze_85 = None
    mul_672: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_87);  sub_151 = unsqueeze_87 = None
    sub_152: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_449, mul_672);  view_449 = mul_672 = None
    sub_153: "f32[128, 768]" = torch.ops.aten.sub.Tensor(sub_152, unsqueeze_86);  sub_152 = unsqueeze_86 = None
    mul_673: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_88);  sub_153 = unsqueeze_88 = None
    mul_674: "f32[768]" = torch.ops.aten.mul.Tensor(sum_60, squeeze_139);  sum_60 = squeeze_139 = None
    view_450: "f32[8, 16, 768]" = torch.ops.aten.view.default(mul_673, [8, 16, 768]);  mul_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_451: "f32[128, 768]" = torch.ops.aten.view.default(view_450, [128, 768]);  view_450 = None
    permute_217: "f32[768, 128]" = torch.ops.aten.permute.default(view_451, [1, 0])
    mm_92: "f32[768, 384]" = torch.ops.aten.mm.default(permute_217, view_255);  permute_217 = view_255 = None
    permute_218: "f32[384, 768]" = torch.ops.aten.permute.default(mm_92, [1, 0]);  mm_92 = None
    permute_219: "f32[768, 384]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
    mm_93: "f32[128, 384]" = torch.ops.aten.mm.default(view_451, permute_219);  view_451 = permute_219 = None
    view_452: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_93, [8, 16, 384]);  mm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_408: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_406, view_452);  add_406 = view_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_220: "f32[768, 384]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_453: "f32[128, 384]" = torch.ops.aten.view.default(add_408, [128, 384])
    unsqueeze_89: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
    sum_61: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_453, [0])
    sub_154: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_253, unsqueeze_89)
    mul_675: "f32[128, 384]" = torch.ops.aten.mul.Tensor(view_453, sub_154);  sub_154 = None
    sum_62: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_675, [0]);  mul_675 = None
    mul_676: "f32[384]" = torch.ops.aten.mul.Tensor(sum_61, 0.0078125)
    unsqueeze_90: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_676, 0);  mul_676 = None
    mul_677: "f32[384]" = torch.ops.aten.mul.Tensor(sum_62, 0.0078125)
    mul_678: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
    mul_679: "f32[384]" = torch.ops.aten.mul.Tensor(mul_677, mul_678);  mul_677 = mul_678 = None
    unsqueeze_91: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_679, 0);  mul_679 = None
    mul_680: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_151);  primals_151 = None
    unsqueeze_92: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_680, 0);  mul_680 = None
    sub_155: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_253, unsqueeze_89);  view_253 = unsqueeze_89 = None
    mul_681: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_91);  sub_155 = unsqueeze_91 = None
    sub_156: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_453, mul_681);  view_453 = mul_681 = None
    sub_157: "f32[128, 384]" = torch.ops.aten.sub.Tensor(sub_156, unsqueeze_90);  sub_156 = unsqueeze_90 = None
    mul_682: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_92);  sub_157 = unsqueeze_92 = None
    mul_683: "f32[384]" = torch.ops.aten.mul.Tensor(sum_62, squeeze_136);  sum_62 = squeeze_136 = None
    view_454: "f32[8, 16, 384]" = torch.ops.aten.view.default(mul_682, [8, 16, 384]);  mul_682 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_455: "f32[128, 384]" = torch.ops.aten.view.default(view_454, [128, 384]);  view_454 = None
    permute_221: "f32[384, 128]" = torch.ops.aten.permute.default(view_455, [1, 0])
    mm_94: "f32[384, 768]" = torch.ops.aten.mm.default(permute_221, view_251);  permute_221 = view_251 = None
    permute_222: "f32[768, 384]" = torch.ops.aten.permute.default(mm_94, [1, 0]);  mm_94 = None
    permute_223: "f32[384, 768]" = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None
    mm_95: "f32[128, 768]" = torch.ops.aten.mm.default(view_455, permute_223);  view_455 = permute_223 = None
    view_456: "f32[8, 16, 768]" = torch.ops.aten.view.default(mm_95, [8, 16, 768]);  mm_95 = None
    permute_224: "f32[384, 768]" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    lt_8: "b8[8, 16, 768]" = torch.ops.aten.lt.Scalar(view_250, -3)
    le_8: "b8[8, 16, 768]" = torch.ops.aten.le.Scalar(view_250, 3)
    div_56: "f32[8, 16, 768]" = torch.ops.aten.div.Tensor(view_250, 3);  view_250 = None
    add_409: "f32[8, 16, 768]" = torch.ops.aten.add.Tensor(div_56, 0.5);  div_56 = None
    mul_684: "f32[8, 16, 768]" = torch.ops.aten.mul.Tensor(view_456, add_409);  add_409 = None
    where_16: "f32[8, 16, 768]" = torch.ops.aten.where.self(le_8, mul_684, view_456);  le_8 = mul_684 = view_456 = None
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_17: "f32[8, 16, 768]" = torch.ops.aten.where.self(lt_8, scalar_tensor_8, where_16);  lt_8 = scalar_tensor_8 = where_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_457: "f32[128, 768]" = torch.ops.aten.view.default(where_17, [128, 768]);  where_17 = None
    unsqueeze_93: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
    sum_63: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_457, [0])
    sub_158: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_249, unsqueeze_93)
    mul_685: "f32[128, 768]" = torch.ops.aten.mul.Tensor(view_457, sub_158);  sub_158 = None
    sum_64: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_685, [0]);  mul_685 = None
    mul_686: "f32[768]" = torch.ops.aten.mul.Tensor(sum_63, 0.0078125)
    unsqueeze_94: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_686, 0);  mul_686 = None
    mul_687: "f32[768]" = torch.ops.aten.mul.Tensor(sum_64, 0.0078125)
    mul_688: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
    mul_689: "f32[768]" = torch.ops.aten.mul.Tensor(mul_687, mul_688);  mul_687 = mul_688 = None
    unsqueeze_95: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_689, 0);  mul_689 = None
    mul_690: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_148);  primals_148 = None
    unsqueeze_96: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_690, 0);  mul_690 = None
    sub_159: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_249, unsqueeze_93);  view_249 = unsqueeze_93 = None
    mul_691: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_95);  sub_159 = unsqueeze_95 = None
    sub_160: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_457, mul_691);  view_457 = mul_691 = None
    sub_161: "f32[128, 768]" = torch.ops.aten.sub.Tensor(sub_160, unsqueeze_94);  sub_160 = unsqueeze_94 = None
    mul_692: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_96);  sub_161 = unsqueeze_96 = None
    mul_693: "f32[768]" = torch.ops.aten.mul.Tensor(sum_64, squeeze_133);  sum_64 = squeeze_133 = None
    view_458: "f32[8, 16, 768]" = torch.ops.aten.view.default(mul_692, [8, 16, 768]);  mul_692 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_459: "f32[128, 768]" = torch.ops.aten.view.default(view_458, [128, 768]);  view_458 = None
    permute_225: "f32[768, 128]" = torch.ops.aten.permute.default(view_459, [1, 0])
    mm_96: "f32[768, 384]" = torch.ops.aten.mm.default(permute_225, view_247);  permute_225 = view_247 = None
    permute_226: "f32[384, 768]" = torch.ops.aten.permute.default(mm_96, [1, 0]);  mm_96 = None
    permute_227: "f32[768, 384]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
    mm_97: "f32[128, 384]" = torch.ops.aten.mm.default(view_459, permute_227);  view_459 = permute_227 = None
    view_460: "f32[8, 16, 384]" = torch.ops.aten.view.default(mm_97, [8, 16, 384]);  mm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_410: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_408, view_460);  add_408 = view_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_228: "f32[768, 384]" = torch.ops.aten.permute.default(permute_226, [1, 0]);  permute_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_461: "f32[128, 384]" = torch.ops.aten.view.default(add_410, [128, 384]);  add_410 = None
    unsqueeze_97: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
    sum_65: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_461, [0])
    sub_162: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_245, unsqueeze_97)
    mul_694: "f32[128, 384]" = torch.ops.aten.mul.Tensor(view_461, sub_162);  sub_162 = None
    sum_66: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_694, [0]);  mul_694 = None
    mul_695: "f32[384]" = torch.ops.aten.mul.Tensor(sum_65, 0.0078125)
    unsqueeze_98: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_695, 0);  mul_695 = None
    mul_696: "f32[384]" = torch.ops.aten.mul.Tensor(sum_66, 0.0078125)
    mul_697: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
    mul_698: "f32[384]" = torch.ops.aten.mul.Tensor(mul_696, mul_697);  mul_696 = mul_697 = None
    unsqueeze_99: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_698, 0);  mul_698 = None
    mul_699: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_145);  primals_145 = None
    unsqueeze_100: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_699, 0);  mul_699 = None
    sub_163: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_245, unsqueeze_97);  view_245 = unsqueeze_97 = None
    mul_700: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_99);  sub_163 = unsqueeze_99 = None
    sub_164: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_461, mul_700);  view_461 = mul_700 = None
    sub_165: "f32[128, 384]" = torch.ops.aten.sub.Tensor(sub_164, unsqueeze_98);  sub_164 = unsqueeze_98 = None
    mul_701: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_100);  sub_165 = unsqueeze_100 = None
    mul_702: "f32[384]" = torch.ops.aten.mul.Tensor(sum_66, squeeze_130);  sum_66 = squeeze_130 = None
    view_462: "f32[8, 16, 384]" = torch.ops.aten.view.default(mul_701, [8, 16, 384]);  mul_701 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_463: "f32[128, 384]" = torch.ops.aten.view.default(view_462, [128, 384]);  view_462 = None
    permute_229: "f32[384, 128]" = torch.ops.aten.permute.default(view_463, [1, 0])
    mm_98: "f32[384, 1024]" = torch.ops.aten.mm.default(permute_229, view_243);  permute_229 = view_243 = None
    permute_230: "f32[1024, 384]" = torch.ops.aten.permute.default(mm_98, [1, 0]);  mm_98 = None
    permute_231: "f32[384, 1024]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
    mm_99: "f32[128, 1024]" = torch.ops.aten.mm.default(view_463, permute_231);  view_463 = permute_231 = None
    view_464: "f32[8, 16, 1024]" = torch.ops.aten.view.default(mm_99, [8, 16, 1024]);  mm_99 = None
    permute_232: "f32[384, 1024]" = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:340, code: x = self.proj(x)
    lt_9: "b8[8, 16, 1024]" = torch.ops.aten.lt.Scalar(view_242, -3)
    le_9: "b8[8, 16, 1024]" = torch.ops.aten.le.Scalar(view_242, 3)
    div_57: "f32[8, 16, 1024]" = torch.ops.aten.div.Tensor(view_242, 3);  view_242 = None
    add_411: "f32[8, 16, 1024]" = torch.ops.aten.add.Tensor(div_57, 0.5);  div_57 = None
    mul_703: "f32[8, 16, 1024]" = torch.ops.aten.mul.Tensor(view_464, add_411);  add_411 = None
    where_18: "f32[8, 16, 1024]" = torch.ops.aten.where.self(le_9, mul_703, view_464);  le_9 = mul_703 = view_464 = None
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_19: "f32[8, 16, 1024]" = torch.ops.aten.where.self(lt_9, scalar_tensor_9, where_18);  lt_9 = scalar_tensor_9 = where_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:339, code: x = (attn @ v).transpose(1, 2).reshape(B, -1, self.val_attn_dim)
    view_465: "f32[8, 16, 16, 64]" = torch.ops.aten.view.default(where_19, [8, 16, 16, 64]);  where_19 = None
    permute_233: "f32[8, 16, 16, 64]" = torch.ops.aten.permute.default(view_465, [0, 2, 1, 3]);  view_465 = None
    clone_87: "f32[8, 16, 16, 64]" = torch.ops.aten.clone.default(permute_233, memory_format = torch.contiguous_format);  permute_233 = None
    view_466: "f32[128, 16, 64]" = torch.ops.aten.view.default(clone_87, [128, 16, 64]);  clone_87 = None
    permute_234: "f32[128, 49, 16]" = torch.ops.aten.permute.default(view_239, [0, 2, 1]);  view_239 = None
    bmm_44: "f32[128, 49, 64]" = torch.ops.aten.bmm.default(permute_234, view_466);  permute_234 = None
    permute_235: "f32[128, 64, 49]" = torch.ops.aten.permute.default(view_240, [0, 2, 1]);  view_240 = None
    bmm_45: "f32[128, 16, 49]" = torch.ops.aten.bmm.default(view_466, permute_235);  view_466 = permute_235 = None
    view_467: "f32[8, 16, 49, 64]" = torch.ops.aten.view.default(bmm_44, [8, 16, 49, 64]);  bmm_44 = None
    view_468: "f32[8, 16, 16, 49]" = torch.ops.aten.view.default(bmm_45, [8, 16, 16, 49]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:337, code: attn = attn.softmax(dim=-1)
    alias_18: "f32[8, 16, 16, 49]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    mul_704: "f32[8, 16, 16, 49]" = torch.ops.aten.mul.Tensor(view_468, alias_18);  view_468 = None
    sum_67: "f32[8, 16, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_704, [-1], True)
    mul_705: "f32[8, 16, 16, 49]" = torch.ops.aten.mul.Tensor(alias_18, sum_67);  alias_18 = sum_67 = None
    sub_166: "f32[8, 16, 16, 49]" = torch.ops.aten.sub.Tensor(mul_704, mul_705);  mul_704 = mul_705 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:336, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    sum_68: "f32[1, 16, 16, 49]" = torch.ops.aten.sum.dim_IntList(sub_166, [0], True)
    view_469: "f32[16, 16, 49]" = torch.ops.aten.view.default(sum_68, [16, 16, 49]);  sum_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:311, code: return self.attention_biases[:, self.attention_bias_idxs]
    full_8: "f32[16, 49]" = torch.ops.aten.full.default([16, 49], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    index_put_4: "f32[16, 49]" = torch.ops.aten.index_put.default(full_8, [None, primals_218], view_469, True);  full_8 = primals_218 = view_469 = None
    full_9: "f32[16, 49]" = torch.ops.aten.full.default([16, 49], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_4: "f32[16, 49]" = torch.ops.aten.slice_scatter.default(full_9, index_put_4, 0, 0, 9223372036854775807);  full_9 = index_put_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:336, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    mul_706: "f32[8, 16, 16, 49]" = torch.ops.aten.mul.Tensor(sub_166, 0.25);  sub_166 = None
    view_470: "f32[128, 16, 49]" = torch.ops.aten.view.default(mul_706, [128, 16, 49]);  mul_706 = None
    permute_236: "f32[128, 16, 16]" = torch.ops.aten.permute.default(view_236, [0, 2, 1]);  view_236 = None
    bmm_46: "f32[128, 16, 49]" = torch.ops.aten.bmm.default(permute_236, view_470);  permute_236 = None
    permute_237: "f32[128, 49, 16]" = torch.ops.aten.permute.default(view_237, [0, 2, 1]);  view_237 = None
    bmm_47: "f32[128, 16, 16]" = torch.ops.aten.bmm.default(view_470, permute_237);  view_470 = permute_237 = None
    view_471: "f32[8, 16, 16, 49]" = torch.ops.aten.view.default(bmm_46, [8, 16, 16, 49]);  bmm_46 = None
    view_472: "f32[8, 16, 16, 16]" = torch.ops.aten.view.default(bmm_47, [8, 16, 16, 16]);  bmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:334, code: q = self.q(x).view(B, -1, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
    permute_238: "f32[8, 16, 16, 16]" = torch.ops.aten.permute.default(view_472, [0, 2, 1, 3]);  view_472 = None
    clone_88: "f32[8, 16, 16, 16]" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None
    view_473: "f32[8, 16, 256]" = torch.ops.aten.view.default(clone_88, [8, 16, 256]);  clone_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_474: "f32[128, 256]" = torch.ops.aten.view.default(view_473, [128, 256]);  view_473 = None
    unsqueeze_101: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
    sum_69: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_474, [0])
    sub_167: "f32[128, 256]" = torch.ops.aten.sub.Tensor(view_233, unsqueeze_101)
    mul_707: "f32[128, 256]" = torch.ops.aten.mul.Tensor(view_474, sub_167);  sub_167 = None
    sum_70: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_707, [0]);  mul_707 = None
    mul_708: "f32[256]" = torch.ops.aten.mul.Tensor(sum_69, 0.0078125)
    unsqueeze_102: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_708, 0);  mul_708 = None
    mul_709: "f32[256]" = torch.ops.aten.mul.Tensor(sum_70, 0.0078125)
    mul_710: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
    mul_711: "f32[256]" = torch.ops.aten.mul.Tensor(mul_709, mul_710);  mul_709 = mul_710 = None
    unsqueeze_103: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_711, 0);  mul_711 = None
    mul_712: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_142);  primals_142 = None
    unsqueeze_104: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_712, 0);  mul_712 = None
    sub_168: "f32[128, 256]" = torch.ops.aten.sub.Tensor(view_233, unsqueeze_101);  view_233 = unsqueeze_101 = None
    mul_713: "f32[128, 256]" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_103);  sub_168 = unsqueeze_103 = None
    sub_169: "f32[128, 256]" = torch.ops.aten.sub.Tensor(view_474, mul_713);  view_474 = mul_713 = None
    sub_170: "f32[128, 256]" = torch.ops.aten.sub.Tensor(sub_169, unsqueeze_102);  sub_169 = unsqueeze_102 = None
    mul_714: "f32[128, 256]" = torch.ops.aten.mul.Tensor(sub_170, unsqueeze_104);  sub_170 = unsqueeze_104 = None
    mul_715: "f32[256]" = torch.ops.aten.mul.Tensor(sum_70, squeeze_127);  sum_70 = squeeze_127 = None
    view_475: "f32[8, 16, 256]" = torch.ops.aten.view.default(mul_714, [8, 16, 256]);  mul_714 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_476: "f32[128, 256]" = torch.ops.aten.view.default(view_475, [128, 256]);  view_475 = None
    permute_239: "f32[256, 128]" = torch.ops.aten.permute.default(view_476, [1, 0])
    mm_100: "f32[256, 256]" = torch.ops.aten.mm.default(permute_239, view_231);  permute_239 = view_231 = None
    permute_240: "f32[256, 256]" = torch.ops.aten.permute.default(mm_100, [1, 0]);  mm_100 = None
    permute_241: "f32[256, 256]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    mm_101: "f32[128, 256]" = torch.ops.aten.mm.default(view_476, permute_241);  view_476 = permute_241 = None
    view_477: "f32[8, 16, 256]" = torch.ops.aten.view.default(mm_101, [8, 16, 256]);  mm_101 = None
    permute_242: "f32[256, 256]" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:162, code: return x.reshape(B, -1, C)
    view_478: "f32[8, 4, 4, 256]" = torch.ops.aten.view.default(view_477, [8, 4, 4, 256]);  view_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:161, code: x = x[:, ::self.stride, ::self.stride]
    full_10: "f32[8, 4, 7, 256]" = torch.ops.aten.full.default([8, 4, 7, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_5: "f32[8, 4, 7, 256]" = torch.ops.aten.slice_scatter.default(full_10, view_478, 2, 0, 9223372036854775807, 2);  full_10 = view_478 = None
    full_11: "f32[8, 7, 7, 256]" = torch.ops.aten.full.default([8, 7, 7, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_6: "f32[8, 7, 7, 256]" = torch.ops.aten.slice_scatter.default(full_11, slice_scatter_5, 1, 0, 9223372036854775807, 2);  full_11 = slice_scatter_5 = None
    full_12: "f32[8, 7, 7, 256]" = torch.ops.aten.full.default([8, 7, 7, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_7: "f32[8, 7, 7, 256]" = torch.ops.aten.slice_scatter.default(full_12, slice_scatter_6, 0, 0, 9223372036854775807);  full_12 = slice_scatter_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:157, code: x = x.view(B, self.resolution[0], self.resolution[1], C)
    view_479: "f32[8, 49, 256]" = torch.ops.aten.view.default(slice_scatter_7, [8, 49, 256]);  slice_scatter_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:333, code: v = v.permute(0, 2, 1, 3)  # BHNC
    permute_243: "f32[8, 49, 16, 64]" = torch.ops.aten.permute.default(view_467, [0, 2, 1, 3]);  view_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:332, code: k = k.permute(0, 2, 3, 1)  # BHCN
    permute_244: "f32[8, 49, 16, 16]" = torch.ops.aten.permute.default(view_471, [0, 3, 1, 2]);  view_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:331, code: k, v = self.kv(x).view(B, N, self.num_heads, -1).split([self.key_dim, self.val_dim], dim=3)
    cat_4: "f32[8, 49, 16, 80]" = torch.ops.aten.cat.default([permute_244, permute_243], 3);  permute_244 = permute_243 = None
    view_480: "f32[8, 49, 1280]" = torch.ops.aten.view.default(cat_4, [8, 49, 1280]);  cat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_481: "f32[392, 1280]" = torch.ops.aten.view.default(view_480, [392, 1280]);  view_480 = None
    unsqueeze_105: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
    sum_71: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_481, [0])
    sub_171: "f32[392, 1280]" = torch.ops.aten.sub.Tensor(view_226, unsqueeze_105)
    mul_716: "f32[392, 1280]" = torch.ops.aten.mul.Tensor(view_481, sub_171);  sub_171 = None
    sum_72: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_716, [0]);  mul_716 = None
    mul_717: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_71, 0.002551020408163265)
    unsqueeze_106: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_717, 0);  mul_717 = None
    mul_718: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_72, 0.002551020408163265)
    mul_719: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
    mul_720: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_718, mul_719);  mul_718 = mul_719 = None
    unsqueeze_107: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_720, 0);  mul_720 = None
    mul_721: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_139);  primals_139 = None
    unsqueeze_108: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_721, 0);  mul_721 = None
    sub_172: "f32[392, 1280]" = torch.ops.aten.sub.Tensor(view_226, unsqueeze_105);  view_226 = unsqueeze_105 = None
    mul_722: "f32[392, 1280]" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_107);  sub_172 = unsqueeze_107 = None
    sub_173: "f32[392, 1280]" = torch.ops.aten.sub.Tensor(view_481, mul_722);  view_481 = mul_722 = None
    sub_174: "f32[392, 1280]" = torch.ops.aten.sub.Tensor(sub_173, unsqueeze_106);  sub_173 = unsqueeze_106 = None
    mul_723: "f32[392, 1280]" = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_108);  sub_174 = unsqueeze_108 = None
    mul_724: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_72, squeeze_124);  sum_72 = squeeze_124 = None
    view_482: "f32[8, 49, 1280]" = torch.ops.aten.view.default(mul_723, [8, 49, 1280]);  mul_723 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_483: "f32[392, 1280]" = torch.ops.aten.view.default(view_482, [392, 1280]);  view_482 = None
    permute_245: "f32[1280, 392]" = torch.ops.aten.permute.default(view_483, [1, 0])
    mm_102: "f32[1280, 256]" = torch.ops.aten.mm.default(permute_245, view_224);  permute_245 = view_224 = None
    permute_246: "f32[256, 1280]" = torch.ops.aten.permute.default(mm_102, [1, 0]);  mm_102 = None
    permute_247: "f32[1280, 256]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
    mm_103: "f32[392, 256]" = torch.ops.aten.mm.default(view_483, permute_247);  view_483 = permute_247 = None
    view_484: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_103, [8, 49, 256]);  mm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_412: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(view_479, view_484);  view_479 = view_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_248: "f32[1280, 256]" = torch.ops.aten.permute.default(permute_246, [1, 0]);  permute_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_485: "f32[392, 256]" = torch.ops.aten.view.default(add_412, [392, 256])
    unsqueeze_109: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
    sum_73: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_485, [0])
    sub_175: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_222, unsqueeze_109)
    mul_725: "f32[392, 256]" = torch.ops.aten.mul.Tensor(view_485, sub_175);  sub_175 = None
    sum_74: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_725, [0]);  mul_725 = None
    mul_726: "f32[256]" = torch.ops.aten.mul.Tensor(sum_73, 0.002551020408163265)
    unsqueeze_110: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_726, 0);  mul_726 = None
    mul_727: "f32[256]" = torch.ops.aten.mul.Tensor(sum_74, 0.002551020408163265)
    mul_728: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_729: "f32[256]" = torch.ops.aten.mul.Tensor(mul_727, mul_728);  mul_727 = mul_728 = None
    unsqueeze_111: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_729, 0);  mul_729 = None
    mul_730: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_136);  primals_136 = None
    unsqueeze_112: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_730, 0);  mul_730 = None
    sub_176: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_222, unsqueeze_109);  view_222 = unsqueeze_109 = None
    mul_731: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_111);  sub_176 = unsqueeze_111 = None
    sub_177: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_485, mul_731);  view_485 = mul_731 = None
    sub_178: "f32[392, 256]" = torch.ops.aten.sub.Tensor(sub_177, unsqueeze_110);  sub_177 = unsqueeze_110 = None
    mul_732: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_178, unsqueeze_112);  sub_178 = unsqueeze_112 = None
    mul_733: "f32[256]" = torch.ops.aten.mul.Tensor(sum_74, squeeze_121);  sum_74 = squeeze_121 = None
    view_486: "f32[8, 49, 256]" = torch.ops.aten.view.default(mul_732, [8, 49, 256]);  mul_732 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_487: "f32[392, 256]" = torch.ops.aten.view.default(view_486, [392, 256]);  view_486 = None
    permute_249: "f32[256, 392]" = torch.ops.aten.permute.default(view_487, [1, 0])
    mm_104: "f32[256, 512]" = torch.ops.aten.mm.default(permute_249, view_220);  permute_249 = view_220 = None
    permute_250: "f32[512, 256]" = torch.ops.aten.permute.default(mm_104, [1, 0]);  mm_104 = None
    permute_251: "f32[256, 512]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
    mm_105: "f32[392, 512]" = torch.ops.aten.mm.default(view_487, permute_251);  view_487 = permute_251 = None
    view_488: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_105, [8, 49, 512]);  mm_105 = None
    permute_252: "f32[256, 512]" = torch.ops.aten.permute.default(permute_250, [1, 0]);  permute_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    lt_10: "b8[8, 49, 512]" = torch.ops.aten.lt.Scalar(view_219, -3)
    le_10: "b8[8, 49, 512]" = torch.ops.aten.le.Scalar(view_219, 3)
    div_58: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(view_219, 3);  view_219 = None
    add_413: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(div_58, 0.5);  div_58 = None
    mul_734: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_488, add_413);  add_413 = None
    where_20: "f32[8, 49, 512]" = torch.ops.aten.where.self(le_10, mul_734, view_488);  le_10 = mul_734 = view_488 = None
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_21: "f32[8, 49, 512]" = torch.ops.aten.where.self(lt_10, scalar_tensor_10, where_20);  lt_10 = scalar_tensor_10 = where_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_489: "f32[392, 512]" = torch.ops.aten.view.default(where_21, [392, 512]);  where_21 = None
    unsqueeze_113: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
    sum_75: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_489, [0])
    sub_179: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_218, unsqueeze_113)
    mul_735: "f32[392, 512]" = torch.ops.aten.mul.Tensor(view_489, sub_179);  sub_179 = None
    sum_76: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_735, [0]);  mul_735 = None
    mul_736: "f32[512]" = torch.ops.aten.mul.Tensor(sum_75, 0.002551020408163265)
    unsqueeze_114: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_736, 0);  mul_736 = None
    mul_737: "f32[512]" = torch.ops.aten.mul.Tensor(sum_76, 0.002551020408163265)
    mul_738: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_739: "f32[512]" = torch.ops.aten.mul.Tensor(mul_737, mul_738);  mul_737 = mul_738 = None
    unsqueeze_115: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_739, 0);  mul_739 = None
    mul_740: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_133);  primals_133 = None
    unsqueeze_116: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_740, 0);  mul_740 = None
    sub_180: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_218, unsqueeze_113);  view_218 = unsqueeze_113 = None
    mul_741: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_115);  sub_180 = unsqueeze_115 = None
    sub_181: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_489, mul_741);  view_489 = mul_741 = None
    sub_182: "f32[392, 512]" = torch.ops.aten.sub.Tensor(sub_181, unsqueeze_114);  sub_181 = unsqueeze_114 = None
    mul_742: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_116);  sub_182 = unsqueeze_116 = None
    mul_743: "f32[512]" = torch.ops.aten.mul.Tensor(sum_76, squeeze_118);  sum_76 = squeeze_118 = None
    view_490: "f32[8, 49, 512]" = torch.ops.aten.view.default(mul_742, [8, 49, 512]);  mul_742 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_491: "f32[392, 512]" = torch.ops.aten.view.default(view_490, [392, 512]);  view_490 = None
    permute_253: "f32[512, 392]" = torch.ops.aten.permute.default(view_491, [1, 0])
    mm_106: "f32[512, 256]" = torch.ops.aten.mm.default(permute_253, view_216);  permute_253 = view_216 = None
    permute_254: "f32[256, 512]" = torch.ops.aten.permute.default(mm_106, [1, 0]);  mm_106 = None
    permute_255: "f32[512, 256]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
    mm_107: "f32[392, 256]" = torch.ops.aten.mm.default(view_491, permute_255);  view_491 = permute_255 = None
    view_492: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_107, [8, 49, 256]);  mm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_414: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_412, view_492);  add_412 = view_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_256: "f32[512, 256]" = torch.ops.aten.permute.default(permute_254, [1, 0]);  permute_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_493: "f32[392, 256]" = torch.ops.aten.view.default(add_414, [392, 256])
    unsqueeze_117: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
    sum_77: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_493, [0])
    sub_183: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_214, unsqueeze_117)
    mul_744: "f32[392, 256]" = torch.ops.aten.mul.Tensor(view_493, sub_183);  sub_183 = None
    sum_78: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_744, [0]);  mul_744 = None
    mul_745: "f32[256]" = torch.ops.aten.mul.Tensor(sum_77, 0.002551020408163265)
    unsqueeze_118: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_745, 0);  mul_745 = None
    mul_746: "f32[256]" = torch.ops.aten.mul.Tensor(sum_78, 0.002551020408163265)
    mul_747: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_748: "f32[256]" = torch.ops.aten.mul.Tensor(mul_746, mul_747);  mul_746 = mul_747 = None
    unsqueeze_119: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_748, 0);  mul_748 = None
    mul_749: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_130);  primals_130 = None
    unsqueeze_120: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_749, 0);  mul_749 = None
    sub_184: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_214, unsqueeze_117);  view_214 = unsqueeze_117 = None
    mul_750: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_119);  sub_184 = unsqueeze_119 = None
    sub_185: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_493, mul_750);  view_493 = mul_750 = None
    sub_186: "f32[392, 256]" = torch.ops.aten.sub.Tensor(sub_185, unsqueeze_118);  sub_185 = unsqueeze_118 = None
    mul_751: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_186, unsqueeze_120);  sub_186 = unsqueeze_120 = None
    mul_752: "f32[256]" = torch.ops.aten.mul.Tensor(sum_78, squeeze_115);  sum_78 = squeeze_115 = None
    view_494: "f32[8, 49, 256]" = torch.ops.aten.view.default(mul_751, [8, 49, 256]);  mul_751 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_495: "f32[392, 256]" = torch.ops.aten.view.default(view_494, [392, 256]);  view_494 = None
    permute_257: "f32[256, 392]" = torch.ops.aten.permute.default(view_495, [1, 0])
    mm_108: "f32[256, 256]" = torch.ops.aten.mm.default(permute_257, view_212);  permute_257 = view_212 = None
    permute_258: "f32[256, 256]" = torch.ops.aten.permute.default(mm_108, [1, 0]);  mm_108 = None
    permute_259: "f32[256, 256]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
    mm_109: "f32[392, 256]" = torch.ops.aten.mm.default(view_495, permute_259);  view_495 = permute_259 = None
    view_496: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_109, [8, 49, 256]);  mm_109 = None
    permute_260: "f32[256, 256]" = torch.ops.aten.permute.default(permute_258, [1, 0]);  permute_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    lt_11: "b8[8, 49, 256]" = torch.ops.aten.lt.Scalar(view_211, -3)
    le_11: "b8[8, 49, 256]" = torch.ops.aten.le.Scalar(view_211, 3)
    div_59: "f32[8, 49, 256]" = torch.ops.aten.div.Tensor(view_211, 3);  view_211 = None
    add_415: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(div_59, 0.5);  div_59 = None
    mul_753: "f32[8, 49, 256]" = torch.ops.aten.mul.Tensor(view_496, add_415);  add_415 = None
    where_22: "f32[8, 49, 256]" = torch.ops.aten.where.self(le_11, mul_753, view_496);  le_11 = mul_753 = view_496 = None
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_23: "f32[8, 49, 256]" = torch.ops.aten.where.self(lt_11, scalar_tensor_11, where_22);  lt_11 = scalar_tensor_11 = where_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    view_497: "f32[8, 49, 8, 32]" = torch.ops.aten.view.default(where_23, [8, 49, 8, 32]);  where_23 = None
    permute_261: "f32[8, 8, 49, 32]" = torch.ops.aten.permute.default(view_497, [0, 2, 1, 3]);  view_497 = None
    clone_89: "f32[8, 8, 49, 32]" = torch.ops.aten.clone.default(permute_261, memory_format = torch.contiguous_format);  permute_261 = None
    view_498: "f32[64, 49, 32]" = torch.ops.aten.view.default(clone_89, [64, 49, 32]);  clone_89 = None
    permute_262: "f32[64, 49, 49]" = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None
    bmm_48: "f32[64, 49, 32]" = torch.ops.aten.bmm.default(permute_262, view_498);  permute_262 = None
    permute_263: "f32[64, 32, 49]" = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None
    bmm_49: "f32[64, 49, 49]" = torch.ops.aten.bmm.default(view_498, permute_263);  view_498 = permute_263 = None
    view_499: "f32[8, 8, 49, 32]" = torch.ops.aten.view.default(bmm_48, [8, 8, 49, 32]);  bmm_48 = None
    view_500: "f32[8, 8, 49, 49]" = torch.ops.aten.view.default(bmm_49, [8, 8, 49, 49]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    alias_19: "f32[8, 8, 49, 49]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    mul_754: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(view_500, alias_19);  view_500 = None
    sum_79: "f32[8, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_754, [-1], True)
    mul_755: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(alias_19, sum_79);  alias_19 = sum_79 = None
    sub_187: "f32[8, 8, 49, 49]" = torch.ops.aten.sub.Tensor(mul_754, mul_755);  mul_754 = mul_755 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    sum_80: "f32[1, 8, 49, 49]" = torch.ops.aten.sum.dim_IntList(sub_187, [0], True)
    view_501: "f32[8, 49, 49]" = torch.ops.aten.view.default(sum_80, [8, 49, 49]);  sum_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    full_13: "f32[8, 49]" = torch.ops.aten.full.default([8, 49], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    index_put_5: "f32[8, 49]" = torch.ops.aten.index_put.default(full_13, [None, primals_217], view_501, True);  full_13 = primals_217 = view_501 = None
    full_14: "f32[8, 49]" = torch.ops.aten.full.default([8, 49], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_8: "f32[8, 49]" = torch.ops.aten.slice_scatter.default(full_14, index_put_5, 0, 0, 9223372036854775807);  full_14 = index_put_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    mul_756: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(sub_187, 0.25);  sub_187 = None
    view_502: "f32[64, 49, 49]" = torch.ops.aten.view.default(mul_756, [64, 49, 49]);  mul_756 = None
    permute_264: "f32[64, 16, 49]" = torch.ops.aten.permute.default(view_205, [0, 2, 1]);  view_205 = None
    bmm_50: "f32[64, 16, 49]" = torch.ops.aten.bmm.default(permute_264, view_502);  permute_264 = None
    permute_265: "f32[64, 49, 16]" = torch.ops.aten.permute.default(view_206, [0, 2, 1]);  view_206 = None
    bmm_51: "f32[64, 49, 16]" = torch.ops.aten.bmm.default(view_502, permute_265);  view_502 = permute_265 = None
    view_503: "f32[8, 8, 16, 49]" = torch.ops.aten.view.default(bmm_50, [8, 8, 16, 49]);  bmm_50 = None
    view_504: "f32[8, 8, 49, 16]" = torch.ops.aten.view.default(bmm_51, [8, 8, 49, 16]);  bmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_266: "f32[8, 49, 8, 32]" = torch.ops.aten.permute.default(view_499, [0, 2, 1, 3]);  view_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_267: "f32[8, 49, 8, 16]" = torch.ops.aten.permute.default(view_503, [0, 3, 1, 2]);  view_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_268: "f32[8, 49, 8, 16]" = torch.ops.aten.permute.default(view_504, [0, 2, 1, 3]);  view_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    cat_5: "f32[8, 49, 8, 64]" = torch.ops.aten.cat.default([permute_268, permute_267, permute_266], 3);  permute_268 = permute_267 = permute_266 = None
    view_505: "f32[8, 49, 512]" = torch.ops.aten.view.default(cat_5, [8, 49, 512]);  cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_506: "f32[392, 512]" = torch.ops.aten.view.default(view_505, [392, 512]);  view_505 = None
    unsqueeze_121: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
    sum_81: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_506, [0])
    sub_188: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_202, unsqueeze_121)
    mul_757: "f32[392, 512]" = torch.ops.aten.mul.Tensor(view_506, sub_188);  sub_188 = None
    sum_82: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_757, [0]);  mul_757 = None
    mul_758: "f32[512]" = torch.ops.aten.mul.Tensor(sum_81, 0.002551020408163265)
    unsqueeze_122: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_758, 0);  mul_758 = None
    mul_759: "f32[512]" = torch.ops.aten.mul.Tensor(sum_82, 0.002551020408163265)
    mul_760: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_761: "f32[512]" = torch.ops.aten.mul.Tensor(mul_759, mul_760);  mul_759 = mul_760 = None
    unsqueeze_123: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_761, 0);  mul_761 = None
    mul_762: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_127);  primals_127 = None
    unsqueeze_124: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_762, 0);  mul_762 = None
    sub_189: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_202, unsqueeze_121);  view_202 = unsqueeze_121 = None
    mul_763: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_123);  sub_189 = unsqueeze_123 = None
    sub_190: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_506, mul_763);  view_506 = mul_763 = None
    sub_191: "f32[392, 512]" = torch.ops.aten.sub.Tensor(sub_190, unsqueeze_122);  sub_190 = unsqueeze_122 = None
    mul_764: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_124);  sub_191 = unsqueeze_124 = None
    mul_765: "f32[512]" = torch.ops.aten.mul.Tensor(sum_82, squeeze_112);  sum_82 = squeeze_112 = None
    view_507: "f32[8, 49, 512]" = torch.ops.aten.view.default(mul_764, [8, 49, 512]);  mul_764 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_508: "f32[392, 512]" = torch.ops.aten.view.default(view_507, [392, 512]);  view_507 = None
    permute_269: "f32[512, 392]" = torch.ops.aten.permute.default(view_508, [1, 0])
    mm_110: "f32[512, 256]" = torch.ops.aten.mm.default(permute_269, view_200);  permute_269 = view_200 = None
    permute_270: "f32[256, 512]" = torch.ops.aten.permute.default(mm_110, [1, 0]);  mm_110 = None
    permute_271: "f32[512, 256]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    mm_111: "f32[392, 256]" = torch.ops.aten.mm.default(view_508, permute_271);  view_508 = permute_271 = None
    view_509: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_111, [8, 49, 256]);  mm_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_416: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_414, view_509);  add_414 = view_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_272: "f32[512, 256]" = torch.ops.aten.permute.default(permute_270, [1, 0]);  permute_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_510: "f32[392, 256]" = torch.ops.aten.view.default(add_416, [392, 256])
    unsqueeze_125: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
    sum_83: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_510, [0])
    sub_192: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_198, unsqueeze_125)
    mul_766: "f32[392, 256]" = torch.ops.aten.mul.Tensor(view_510, sub_192);  sub_192 = None
    sum_84: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_766, [0]);  mul_766 = None
    mul_767: "f32[256]" = torch.ops.aten.mul.Tensor(sum_83, 0.002551020408163265)
    unsqueeze_126: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_767, 0);  mul_767 = None
    mul_768: "f32[256]" = torch.ops.aten.mul.Tensor(sum_84, 0.002551020408163265)
    mul_769: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_770: "f32[256]" = torch.ops.aten.mul.Tensor(mul_768, mul_769);  mul_768 = mul_769 = None
    unsqueeze_127: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_770, 0);  mul_770 = None
    mul_771: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_124);  primals_124 = None
    unsqueeze_128: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_771, 0);  mul_771 = None
    sub_193: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_198, unsqueeze_125);  view_198 = unsqueeze_125 = None
    mul_772: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_127);  sub_193 = unsqueeze_127 = None
    sub_194: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_510, mul_772);  view_510 = mul_772 = None
    sub_195: "f32[392, 256]" = torch.ops.aten.sub.Tensor(sub_194, unsqueeze_126);  sub_194 = unsqueeze_126 = None
    mul_773: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_195, unsqueeze_128);  sub_195 = unsqueeze_128 = None
    mul_774: "f32[256]" = torch.ops.aten.mul.Tensor(sum_84, squeeze_109);  sum_84 = squeeze_109 = None
    view_511: "f32[8, 49, 256]" = torch.ops.aten.view.default(mul_773, [8, 49, 256]);  mul_773 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_512: "f32[392, 256]" = torch.ops.aten.view.default(view_511, [392, 256]);  view_511 = None
    permute_273: "f32[256, 392]" = torch.ops.aten.permute.default(view_512, [1, 0])
    mm_112: "f32[256, 512]" = torch.ops.aten.mm.default(permute_273, view_196);  permute_273 = view_196 = None
    permute_274: "f32[512, 256]" = torch.ops.aten.permute.default(mm_112, [1, 0]);  mm_112 = None
    permute_275: "f32[256, 512]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    mm_113: "f32[392, 512]" = torch.ops.aten.mm.default(view_512, permute_275);  view_512 = permute_275 = None
    view_513: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_113, [8, 49, 512]);  mm_113 = None
    permute_276: "f32[256, 512]" = torch.ops.aten.permute.default(permute_274, [1, 0]);  permute_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    lt_12: "b8[8, 49, 512]" = torch.ops.aten.lt.Scalar(view_195, -3)
    le_12: "b8[8, 49, 512]" = torch.ops.aten.le.Scalar(view_195, 3)
    div_60: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(view_195, 3);  view_195 = None
    add_417: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(div_60, 0.5);  div_60 = None
    mul_775: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_513, add_417);  add_417 = None
    where_24: "f32[8, 49, 512]" = torch.ops.aten.where.self(le_12, mul_775, view_513);  le_12 = mul_775 = view_513 = None
    scalar_tensor_12: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_25: "f32[8, 49, 512]" = torch.ops.aten.where.self(lt_12, scalar_tensor_12, where_24);  lt_12 = scalar_tensor_12 = where_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_514: "f32[392, 512]" = torch.ops.aten.view.default(where_25, [392, 512]);  where_25 = None
    unsqueeze_129: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
    sum_85: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_514, [0])
    sub_196: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_194, unsqueeze_129)
    mul_776: "f32[392, 512]" = torch.ops.aten.mul.Tensor(view_514, sub_196);  sub_196 = None
    sum_86: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_776, [0]);  mul_776 = None
    mul_777: "f32[512]" = torch.ops.aten.mul.Tensor(sum_85, 0.002551020408163265)
    unsqueeze_130: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_777, 0);  mul_777 = None
    mul_778: "f32[512]" = torch.ops.aten.mul.Tensor(sum_86, 0.002551020408163265)
    mul_779: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_780: "f32[512]" = torch.ops.aten.mul.Tensor(mul_778, mul_779);  mul_778 = mul_779 = None
    unsqueeze_131: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_780, 0);  mul_780 = None
    mul_781: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_121);  primals_121 = None
    unsqueeze_132: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_781, 0);  mul_781 = None
    sub_197: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_194, unsqueeze_129);  view_194 = unsqueeze_129 = None
    mul_782: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_131);  sub_197 = unsqueeze_131 = None
    sub_198: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_514, mul_782);  view_514 = mul_782 = None
    sub_199: "f32[392, 512]" = torch.ops.aten.sub.Tensor(sub_198, unsqueeze_130);  sub_198 = unsqueeze_130 = None
    mul_783: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_132);  sub_199 = unsqueeze_132 = None
    mul_784: "f32[512]" = torch.ops.aten.mul.Tensor(sum_86, squeeze_106);  sum_86 = squeeze_106 = None
    view_515: "f32[8, 49, 512]" = torch.ops.aten.view.default(mul_783, [8, 49, 512]);  mul_783 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_516: "f32[392, 512]" = torch.ops.aten.view.default(view_515, [392, 512]);  view_515 = None
    permute_277: "f32[512, 392]" = torch.ops.aten.permute.default(view_516, [1, 0])
    mm_114: "f32[512, 256]" = torch.ops.aten.mm.default(permute_277, view_192);  permute_277 = view_192 = None
    permute_278: "f32[256, 512]" = torch.ops.aten.permute.default(mm_114, [1, 0]);  mm_114 = None
    permute_279: "f32[512, 256]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    mm_115: "f32[392, 256]" = torch.ops.aten.mm.default(view_516, permute_279);  view_516 = permute_279 = None
    view_517: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_115, [8, 49, 256]);  mm_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_418: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_416, view_517);  add_416 = view_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_280: "f32[512, 256]" = torch.ops.aten.permute.default(permute_278, [1, 0]);  permute_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_518: "f32[392, 256]" = torch.ops.aten.view.default(add_418, [392, 256])
    unsqueeze_133: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
    sum_87: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_518, [0])
    sub_200: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_190, unsqueeze_133)
    mul_785: "f32[392, 256]" = torch.ops.aten.mul.Tensor(view_518, sub_200);  sub_200 = None
    sum_88: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_785, [0]);  mul_785 = None
    mul_786: "f32[256]" = torch.ops.aten.mul.Tensor(sum_87, 0.002551020408163265)
    unsqueeze_134: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_786, 0);  mul_786 = None
    mul_787: "f32[256]" = torch.ops.aten.mul.Tensor(sum_88, 0.002551020408163265)
    mul_788: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_789: "f32[256]" = torch.ops.aten.mul.Tensor(mul_787, mul_788);  mul_787 = mul_788 = None
    unsqueeze_135: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_789, 0);  mul_789 = None
    mul_790: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_118);  primals_118 = None
    unsqueeze_136: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_790, 0);  mul_790 = None
    sub_201: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_190, unsqueeze_133);  view_190 = unsqueeze_133 = None
    mul_791: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_135);  sub_201 = unsqueeze_135 = None
    sub_202: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_518, mul_791);  view_518 = mul_791 = None
    sub_203: "f32[392, 256]" = torch.ops.aten.sub.Tensor(sub_202, unsqueeze_134);  sub_202 = unsqueeze_134 = None
    mul_792: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_136);  sub_203 = unsqueeze_136 = None
    mul_793: "f32[256]" = torch.ops.aten.mul.Tensor(sum_88, squeeze_103);  sum_88 = squeeze_103 = None
    view_519: "f32[8, 49, 256]" = torch.ops.aten.view.default(mul_792, [8, 49, 256]);  mul_792 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_520: "f32[392, 256]" = torch.ops.aten.view.default(view_519, [392, 256]);  view_519 = None
    permute_281: "f32[256, 392]" = torch.ops.aten.permute.default(view_520, [1, 0])
    mm_116: "f32[256, 256]" = torch.ops.aten.mm.default(permute_281, view_188);  permute_281 = view_188 = None
    permute_282: "f32[256, 256]" = torch.ops.aten.permute.default(mm_116, [1, 0]);  mm_116 = None
    permute_283: "f32[256, 256]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    mm_117: "f32[392, 256]" = torch.ops.aten.mm.default(view_520, permute_283);  view_520 = permute_283 = None
    view_521: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_117, [8, 49, 256]);  mm_117 = None
    permute_284: "f32[256, 256]" = torch.ops.aten.permute.default(permute_282, [1, 0]);  permute_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    lt_13: "b8[8, 49, 256]" = torch.ops.aten.lt.Scalar(view_187, -3)
    le_13: "b8[8, 49, 256]" = torch.ops.aten.le.Scalar(view_187, 3)
    div_61: "f32[8, 49, 256]" = torch.ops.aten.div.Tensor(view_187, 3);  view_187 = None
    add_419: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(div_61, 0.5);  div_61 = None
    mul_794: "f32[8, 49, 256]" = torch.ops.aten.mul.Tensor(view_521, add_419);  add_419 = None
    where_26: "f32[8, 49, 256]" = torch.ops.aten.where.self(le_13, mul_794, view_521);  le_13 = mul_794 = view_521 = None
    scalar_tensor_13: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_27: "f32[8, 49, 256]" = torch.ops.aten.where.self(lt_13, scalar_tensor_13, where_26);  lt_13 = scalar_tensor_13 = where_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    view_522: "f32[8, 49, 8, 32]" = torch.ops.aten.view.default(where_27, [8, 49, 8, 32]);  where_27 = None
    permute_285: "f32[8, 8, 49, 32]" = torch.ops.aten.permute.default(view_522, [0, 2, 1, 3]);  view_522 = None
    clone_90: "f32[8, 8, 49, 32]" = torch.ops.aten.clone.default(permute_285, memory_format = torch.contiguous_format);  permute_285 = None
    view_523: "f32[64, 49, 32]" = torch.ops.aten.view.default(clone_90, [64, 49, 32]);  clone_90 = None
    permute_286: "f32[64, 49, 49]" = torch.ops.aten.permute.default(view_184, [0, 2, 1]);  view_184 = None
    bmm_52: "f32[64, 49, 32]" = torch.ops.aten.bmm.default(permute_286, view_523);  permute_286 = None
    permute_287: "f32[64, 32, 49]" = torch.ops.aten.permute.default(view_185, [0, 2, 1]);  view_185 = None
    bmm_53: "f32[64, 49, 49]" = torch.ops.aten.bmm.default(view_523, permute_287);  view_523 = permute_287 = None
    view_524: "f32[8, 8, 49, 32]" = torch.ops.aten.view.default(bmm_52, [8, 8, 49, 32]);  bmm_52 = None
    view_525: "f32[8, 8, 49, 49]" = torch.ops.aten.view.default(bmm_53, [8, 8, 49, 49]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    alias_20: "f32[8, 8, 49, 49]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    mul_795: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(view_525, alias_20);  view_525 = None
    sum_89: "f32[8, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_795, [-1], True)
    mul_796: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(alias_20, sum_89);  alias_20 = sum_89 = None
    sub_204: "f32[8, 8, 49, 49]" = torch.ops.aten.sub.Tensor(mul_795, mul_796);  mul_795 = mul_796 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    sum_90: "f32[1, 8, 49, 49]" = torch.ops.aten.sum.dim_IntList(sub_204, [0], True)
    view_526: "f32[8, 49, 49]" = torch.ops.aten.view.default(sum_90, [8, 49, 49]);  sum_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    full_15: "f32[8, 49]" = torch.ops.aten.full.default([8, 49], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    index_put_6: "f32[8, 49]" = torch.ops.aten.index_put.default(full_15, [None, primals_216], view_526, True);  full_15 = primals_216 = view_526 = None
    full_16: "f32[8, 49]" = torch.ops.aten.full.default([8, 49], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_9: "f32[8, 49]" = torch.ops.aten.slice_scatter.default(full_16, index_put_6, 0, 0, 9223372036854775807);  full_16 = index_put_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    mul_797: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(sub_204, 0.25);  sub_204 = None
    view_527: "f32[64, 49, 49]" = torch.ops.aten.view.default(mul_797, [64, 49, 49]);  mul_797 = None
    permute_288: "f32[64, 16, 49]" = torch.ops.aten.permute.default(view_181, [0, 2, 1]);  view_181 = None
    bmm_54: "f32[64, 16, 49]" = torch.ops.aten.bmm.default(permute_288, view_527);  permute_288 = None
    permute_289: "f32[64, 49, 16]" = torch.ops.aten.permute.default(view_182, [0, 2, 1]);  view_182 = None
    bmm_55: "f32[64, 49, 16]" = torch.ops.aten.bmm.default(view_527, permute_289);  view_527 = permute_289 = None
    view_528: "f32[8, 8, 16, 49]" = torch.ops.aten.view.default(bmm_54, [8, 8, 16, 49]);  bmm_54 = None
    view_529: "f32[8, 8, 49, 16]" = torch.ops.aten.view.default(bmm_55, [8, 8, 49, 16]);  bmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_290: "f32[8, 49, 8, 32]" = torch.ops.aten.permute.default(view_524, [0, 2, 1, 3]);  view_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_291: "f32[8, 49, 8, 16]" = torch.ops.aten.permute.default(view_528, [0, 3, 1, 2]);  view_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_292: "f32[8, 49, 8, 16]" = torch.ops.aten.permute.default(view_529, [0, 2, 1, 3]);  view_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    cat_6: "f32[8, 49, 8, 64]" = torch.ops.aten.cat.default([permute_292, permute_291, permute_290], 3);  permute_292 = permute_291 = permute_290 = None
    view_530: "f32[8, 49, 512]" = torch.ops.aten.view.default(cat_6, [8, 49, 512]);  cat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_531: "f32[392, 512]" = torch.ops.aten.view.default(view_530, [392, 512]);  view_530 = None
    unsqueeze_137: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
    sum_91: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_531, [0])
    sub_205: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_178, unsqueeze_137)
    mul_798: "f32[392, 512]" = torch.ops.aten.mul.Tensor(view_531, sub_205);  sub_205 = None
    sum_92: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_798, [0]);  mul_798 = None
    mul_799: "f32[512]" = torch.ops.aten.mul.Tensor(sum_91, 0.002551020408163265)
    unsqueeze_138: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_799, 0);  mul_799 = None
    mul_800: "f32[512]" = torch.ops.aten.mul.Tensor(sum_92, 0.002551020408163265)
    mul_801: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_802: "f32[512]" = torch.ops.aten.mul.Tensor(mul_800, mul_801);  mul_800 = mul_801 = None
    unsqueeze_139: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_802, 0);  mul_802 = None
    mul_803: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_115);  primals_115 = None
    unsqueeze_140: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_803, 0);  mul_803 = None
    sub_206: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_178, unsqueeze_137);  view_178 = unsqueeze_137 = None
    mul_804: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_139);  sub_206 = unsqueeze_139 = None
    sub_207: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_531, mul_804);  view_531 = mul_804 = None
    sub_208: "f32[392, 512]" = torch.ops.aten.sub.Tensor(sub_207, unsqueeze_138);  sub_207 = unsqueeze_138 = None
    mul_805: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_140);  sub_208 = unsqueeze_140 = None
    mul_806: "f32[512]" = torch.ops.aten.mul.Tensor(sum_92, squeeze_100);  sum_92 = squeeze_100 = None
    view_532: "f32[8, 49, 512]" = torch.ops.aten.view.default(mul_805, [8, 49, 512]);  mul_805 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_533: "f32[392, 512]" = torch.ops.aten.view.default(view_532, [392, 512]);  view_532 = None
    permute_293: "f32[512, 392]" = torch.ops.aten.permute.default(view_533, [1, 0])
    mm_118: "f32[512, 256]" = torch.ops.aten.mm.default(permute_293, view_176);  permute_293 = view_176 = None
    permute_294: "f32[256, 512]" = torch.ops.aten.permute.default(mm_118, [1, 0]);  mm_118 = None
    permute_295: "f32[512, 256]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    mm_119: "f32[392, 256]" = torch.ops.aten.mm.default(view_533, permute_295);  view_533 = permute_295 = None
    view_534: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_119, [8, 49, 256]);  mm_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_420: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_418, view_534);  add_418 = view_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_296: "f32[512, 256]" = torch.ops.aten.permute.default(permute_294, [1, 0]);  permute_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_535: "f32[392, 256]" = torch.ops.aten.view.default(add_420, [392, 256])
    unsqueeze_141: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
    sum_93: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_535, [0])
    sub_209: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_174, unsqueeze_141)
    mul_807: "f32[392, 256]" = torch.ops.aten.mul.Tensor(view_535, sub_209);  sub_209 = None
    sum_94: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_807, [0]);  mul_807 = None
    mul_808: "f32[256]" = torch.ops.aten.mul.Tensor(sum_93, 0.002551020408163265)
    unsqueeze_142: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_808, 0);  mul_808 = None
    mul_809: "f32[256]" = torch.ops.aten.mul.Tensor(sum_94, 0.002551020408163265)
    mul_810: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_811: "f32[256]" = torch.ops.aten.mul.Tensor(mul_809, mul_810);  mul_809 = mul_810 = None
    unsqueeze_143: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_811, 0);  mul_811 = None
    mul_812: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_112);  primals_112 = None
    unsqueeze_144: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_812, 0);  mul_812 = None
    sub_210: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_174, unsqueeze_141);  view_174 = unsqueeze_141 = None
    mul_813: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_210, unsqueeze_143);  sub_210 = unsqueeze_143 = None
    sub_211: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_535, mul_813);  view_535 = mul_813 = None
    sub_212: "f32[392, 256]" = torch.ops.aten.sub.Tensor(sub_211, unsqueeze_142);  sub_211 = unsqueeze_142 = None
    mul_814: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_144);  sub_212 = unsqueeze_144 = None
    mul_815: "f32[256]" = torch.ops.aten.mul.Tensor(sum_94, squeeze_97);  sum_94 = squeeze_97 = None
    view_536: "f32[8, 49, 256]" = torch.ops.aten.view.default(mul_814, [8, 49, 256]);  mul_814 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_537: "f32[392, 256]" = torch.ops.aten.view.default(view_536, [392, 256]);  view_536 = None
    permute_297: "f32[256, 392]" = torch.ops.aten.permute.default(view_537, [1, 0])
    mm_120: "f32[256, 512]" = torch.ops.aten.mm.default(permute_297, view_172);  permute_297 = view_172 = None
    permute_298: "f32[512, 256]" = torch.ops.aten.permute.default(mm_120, [1, 0]);  mm_120 = None
    permute_299: "f32[256, 512]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    mm_121: "f32[392, 512]" = torch.ops.aten.mm.default(view_537, permute_299);  view_537 = permute_299 = None
    view_538: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_121, [8, 49, 512]);  mm_121 = None
    permute_300: "f32[256, 512]" = torch.ops.aten.permute.default(permute_298, [1, 0]);  permute_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    lt_14: "b8[8, 49, 512]" = torch.ops.aten.lt.Scalar(view_171, -3)
    le_14: "b8[8, 49, 512]" = torch.ops.aten.le.Scalar(view_171, 3)
    div_62: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(view_171, 3);  view_171 = None
    add_421: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(div_62, 0.5);  div_62 = None
    mul_816: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_538, add_421);  add_421 = None
    where_28: "f32[8, 49, 512]" = torch.ops.aten.where.self(le_14, mul_816, view_538);  le_14 = mul_816 = view_538 = None
    scalar_tensor_14: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_29: "f32[8, 49, 512]" = torch.ops.aten.where.self(lt_14, scalar_tensor_14, where_28);  lt_14 = scalar_tensor_14 = where_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_539: "f32[392, 512]" = torch.ops.aten.view.default(where_29, [392, 512]);  where_29 = None
    unsqueeze_145: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
    sum_95: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_539, [0])
    sub_213: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_170, unsqueeze_145)
    mul_817: "f32[392, 512]" = torch.ops.aten.mul.Tensor(view_539, sub_213);  sub_213 = None
    sum_96: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_817, [0]);  mul_817 = None
    mul_818: "f32[512]" = torch.ops.aten.mul.Tensor(sum_95, 0.002551020408163265)
    unsqueeze_146: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_818, 0);  mul_818 = None
    mul_819: "f32[512]" = torch.ops.aten.mul.Tensor(sum_96, 0.002551020408163265)
    mul_820: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_821: "f32[512]" = torch.ops.aten.mul.Tensor(mul_819, mul_820);  mul_819 = mul_820 = None
    unsqueeze_147: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_821, 0);  mul_821 = None
    mul_822: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_109);  primals_109 = None
    unsqueeze_148: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_822, 0);  mul_822 = None
    sub_214: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_170, unsqueeze_145);  view_170 = unsqueeze_145 = None
    mul_823: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_147);  sub_214 = unsqueeze_147 = None
    sub_215: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_539, mul_823);  view_539 = mul_823 = None
    sub_216: "f32[392, 512]" = torch.ops.aten.sub.Tensor(sub_215, unsqueeze_146);  sub_215 = unsqueeze_146 = None
    mul_824: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_148);  sub_216 = unsqueeze_148 = None
    mul_825: "f32[512]" = torch.ops.aten.mul.Tensor(sum_96, squeeze_94);  sum_96 = squeeze_94 = None
    view_540: "f32[8, 49, 512]" = torch.ops.aten.view.default(mul_824, [8, 49, 512]);  mul_824 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_541: "f32[392, 512]" = torch.ops.aten.view.default(view_540, [392, 512]);  view_540 = None
    permute_301: "f32[512, 392]" = torch.ops.aten.permute.default(view_541, [1, 0])
    mm_122: "f32[512, 256]" = torch.ops.aten.mm.default(permute_301, view_168);  permute_301 = view_168 = None
    permute_302: "f32[256, 512]" = torch.ops.aten.permute.default(mm_122, [1, 0]);  mm_122 = None
    permute_303: "f32[512, 256]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    mm_123: "f32[392, 256]" = torch.ops.aten.mm.default(view_541, permute_303);  view_541 = permute_303 = None
    view_542: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_123, [8, 49, 256]);  mm_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_422: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_420, view_542);  add_420 = view_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_304: "f32[512, 256]" = torch.ops.aten.permute.default(permute_302, [1, 0]);  permute_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_543: "f32[392, 256]" = torch.ops.aten.view.default(add_422, [392, 256])
    unsqueeze_149: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    sum_97: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_543, [0])
    sub_217: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_166, unsqueeze_149)
    mul_826: "f32[392, 256]" = torch.ops.aten.mul.Tensor(view_543, sub_217);  sub_217 = None
    sum_98: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_826, [0]);  mul_826 = None
    mul_827: "f32[256]" = torch.ops.aten.mul.Tensor(sum_97, 0.002551020408163265)
    unsqueeze_150: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_827, 0);  mul_827 = None
    mul_828: "f32[256]" = torch.ops.aten.mul.Tensor(sum_98, 0.002551020408163265)
    mul_829: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_830: "f32[256]" = torch.ops.aten.mul.Tensor(mul_828, mul_829);  mul_828 = mul_829 = None
    unsqueeze_151: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_830, 0);  mul_830 = None
    mul_831: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_106);  primals_106 = None
    unsqueeze_152: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_831, 0);  mul_831 = None
    sub_218: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_166, unsqueeze_149);  view_166 = unsqueeze_149 = None
    mul_832: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_218, unsqueeze_151);  sub_218 = unsqueeze_151 = None
    sub_219: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_543, mul_832);  view_543 = mul_832 = None
    sub_220: "f32[392, 256]" = torch.ops.aten.sub.Tensor(sub_219, unsqueeze_150);  sub_219 = unsqueeze_150 = None
    mul_833: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_152);  sub_220 = unsqueeze_152 = None
    mul_834: "f32[256]" = torch.ops.aten.mul.Tensor(sum_98, squeeze_91);  sum_98 = squeeze_91 = None
    view_544: "f32[8, 49, 256]" = torch.ops.aten.view.default(mul_833, [8, 49, 256]);  mul_833 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_545: "f32[392, 256]" = torch.ops.aten.view.default(view_544, [392, 256]);  view_544 = None
    permute_305: "f32[256, 392]" = torch.ops.aten.permute.default(view_545, [1, 0])
    mm_124: "f32[256, 256]" = torch.ops.aten.mm.default(permute_305, view_164);  permute_305 = view_164 = None
    permute_306: "f32[256, 256]" = torch.ops.aten.permute.default(mm_124, [1, 0]);  mm_124 = None
    permute_307: "f32[256, 256]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    mm_125: "f32[392, 256]" = torch.ops.aten.mm.default(view_545, permute_307);  view_545 = permute_307 = None
    view_546: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_125, [8, 49, 256]);  mm_125 = None
    permute_308: "f32[256, 256]" = torch.ops.aten.permute.default(permute_306, [1, 0]);  permute_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    lt_15: "b8[8, 49, 256]" = torch.ops.aten.lt.Scalar(view_163, -3)
    le_15: "b8[8, 49, 256]" = torch.ops.aten.le.Scalar(view_163, 3)
    div_63: "f32[8, 49, 256]" = torch.ops.aten.div.Tensor(view_163, 3);  view_163 = None
    add_423: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(div_63, 0.5);  div_63 = None
    mul_835: "f32[8, 49, 256]" = torch.ops.aten.mul.Tensor(view_546, add_423);  add_423 = None
    where_30: "f32[8, 49, 256]" = torch.ops.aten.where.self(le_15, mul_835, view_546);  le_15 = mul_835 = view_546 = None
    scalar_tensor_15: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_31: "f32[8, 49, 256]" = torch.ops.aten.where.self(lt_15, scalar_tensor_15, where_30);  lt_15 = scalar_tensor_15 = where_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    view_547: "f32[8, 49, 8, 32]" = torch.ops.aten.view.default(where_31, [8, 49, 8, 32]);  where_31 = None
    permute_309: "f32[8, 8, 49, 32]" = torch.ops.aten.permute.default(view_547, [0, 2, 1, 3]);  view_547 = None
    clone_91: "f32[8, 8, 49, 32]" = torch.ops.aten.clone.default(permute_309, memory_format = torch.contiguous_format);  permute_309 = None
    view_548: "f32[64, 49, 32]" = torch.ops.aten.view.default(clone_91, [64, 49, 32]);  clone_91 = None
    permute_310: "f32[64, 49, 49]" = torch.ops.aten.permute.default(view_160, [0, 2, 1]);  view_160 = None
    bmm_56: "f32[64, 49, 32]" = torch.ops.aten.bmm.default(permute_310, view_548);  permute_310 = None
    permute_311: "f32[64, 32, 49]" = torch.ops.aten.permute.default(view_161, [0, 2, 1]);  view_161 = None
    bmm_57: "f32[64, 49, 49]" = torch.ops.aten.bmm.default(view_548, permute_311);  view_548 = permute_311 = None
    view_549: "f32[8, 8, 49, 32]" = torch.ops.aten.view.default(bmm_56, [8, 8, 49, 32]);  bmm_56 = None
    view_550: "f32[8, 8, 49, 49]" = torch.ops.aten.view.default(bmm_57, [8, 8, 49, 49]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    alias_21: "f32[8, 8, 49, 49]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    mul_836: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(view_550, alias_21);  view_550 = None
    sum_99: "f32[8, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_836, [-1], True)
    mul_837: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(alias_21, sum_99);  alias_21 = sum_99 = None
    sub_221: "f32[8, 8, 49, 49]" = torch.ops.aten.sub.Tensor(mul_836, mul_837);  mul_836 = mul_837 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    sum_100: "f32[1, 8, 49, 49]" = torch.ops.aten.sum.dim_IntList(sub_221, [0], True)
    view_551: "f32[8, 49, 49]" = torch.ops.aten.view.default(sum_100, [8, 49, 49]);  sum_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    full_17: "f32[8, 49]" = torch.ops.aten.full.default([8, 49], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    index_put_7: "f32[8, 49]" = torch.ops.aten.index_put.default(full_17, [None, primals_215], view_551, True);  full_17 = primals_215 = view_551 = None
    full_18: "f32[8, 49]" = torch.ops.aten.full.default([8, 49], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_10: "f32[8, 49]" = torch.ops.aten.slice_scatter.default(full_18, index_put_7, 0, 0, 9223372036854775807);  full_18 = index_put_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    mul_838: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(sub_221, 0.25);  sub_221 = None
    view_552: "f32[64, 49, 49]" = torch.ops.aten.view.default(mul_838, [64, 49, 49]);  mul_838 = None
    permute_312: "f32[64, 16, 49]" = torch.ops.aten.permute.default(view_157, [0, 2, 1]);  view_157 = None
    bmm_58: "f32[64, 16, 49]" = torch.ops.aten.bmm.default(permute_312, view_552);  permute_312 = None
    permute_313: "f32[64, 49, 16]" = torch.ops.aten.permute.default(view_158, [0, 2, 1]);  view_158 = None
    bmm_59: "f32[64, 49, 16]" = torch.ops.aten.bmm.default(view_552, permute_313);  view_552 = permute_313 = None
    view_553: "f32[8, 8, 16, 49]" = torch.ops.aten.view.default(bmm_58, [8, 8, 16, 49]);  bmm_58 = None
    view_554: "f32[8, 8, 49, 16]" = torch.ops.aten.view.default(bmm_59, [8, 8, 49, 16]);  bmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_314: "f32[8, 49, 8, 32]" = torch.ops.aten.permute.default(view_549, [0, 2, 1, 3]);  view_549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_315: "f32[8, 49, 8, 16]" = torch.ops.aten.permute.default(view_553, [0, 3, 1, 2]);  view_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_316: "f32[8, 49, 8, 16]" = torch.ops.aten.permute.default(view_554, [0, 2, 1, 3]);  view_554 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    cat_7: "f32[8, 49, 8, 64]" = torch.ops.aten.cat.default([permute_316, permute_315, permute_314], 3);  permute_316 = permute_315 = permute_314 = None
    view_555: "f32[8, 49, 512]" = torch.ops.aten.view.default(cat_7, [8, 49, 512]);  cat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_556: "f32[392, 512]" = torch.ops.aten.view.default(view_555, [392, 512]);  view_555 = None
    unsqueeze_153: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    sum_101: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_556, [0])
    sub_222: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_154, unsqueeze_153)
    mul_839: "f32[392, 512]" = torch.ops.aten.mul.Tensor(view_556, sub_222);  sub_222 = None
    sum_102: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_839, [0]);  mul_839 = None
    mul_840: "f32[512]" = torch.ops.aten.mul.Tensor(sum_101, 0.002551020408163265)
    unsqueeze_154: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_840, 0);  mul_840 = None
    mul_841: "f32[512]" = torch.ops.aten.mul.Tensor(sum_102, 0.002551020408163265)
    mul_842: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_843: "f32[512]" = torch.ops.aten.mul.Tensor(mul_841, mul_842);  mul_841 = mul_842 = None
    unsqueeze_155: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_843, 0);  mul_843 = None
    mul_844: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_103);  primals_103 = None
    unsqueeze_156: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_844, 0);  mul_844 = None
    sub_223: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_154, unsqueeze_153);  view_154 = unsqueeze_153 = None
    mul_845: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_223, unsqueeze_155);  sub_223 = unsqueeze_155 = None
    sub_224: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_556, mul_845);  view_556 = mul_845 = None
    sub_225: "f32[392, 512]" = torch.ops.aten.sub.Tensor(sub_224, unsqueeze_154);  sub_224 = unsqueeze_154 = None
    mul_846: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_156);  sub_225 = unsqueeze_156 = None
    mul_847: "f32[512]" = torch.ops.aten.mul.Tensor(sum_102, squeeze_88);  sum_102 = squeeze_88 = None
    view_557: "f32[8, 49, 512]" = torch.ops.aten.view.default(mul_846, [8, 49, 512]);  mul_846 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_558: "f32[392, 512]" = torch.ops.aten.view.default(view_557, [392, 512]);  view_557 = None
    permute_317: "f32[512, 392]" = torch.ops.aten.permute.default(view_558, [1, 0])
    mm_126: "f32[512, 256]" = torch.ops.aten.mm.default(permute_317, view_152);  permute_317 = view_152 = None
    permute_318: "f32[256, 512]" = torch.ops.aten.permute.default(mm_126, [1, 0]);  mm_126 = None
    permute_319: "f32[512, 256]" = torch.ops.aten.permute.default(permute_50, [1, 0]);  permute_50 = None
    mm_127: "f32[392, 256]" = torch.ops.aten.mm.default(view_558, permute_319);  view_558 = permute_319 = None
    view_559: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_127, [8, 49, 256]);  mm_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_424: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_422, view_559);  add_422 = view_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_320: "f32[512, 256]" = torch.ops.aten.permute.default(permute_318, [1, 0]);  permute_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_560: "f32[392, 256]" = torch.ops.aten.view.default(add_424, [392, 256])
    unsqueeze_157: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    sum_103: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_560, [0])
    sub_226: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_150, unsqueeze_157)
    mul_848: "f32[392, 256]" = torch.ops.aten.mul.Tensor(view_560, sub_226);  sub_226 = None
    sum_104: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_848, [0]);  mul_848 = None
    mul_849: "f32[256]" = torch.ops.aten.mul.Tensor(sum_103, 0.002551020408163265)
    unsqueeze_158: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_849, 0);  mul_849 = None
    mul_850: "f32[256]" = torch.ops.aten.mul.Tensor(sum_104, 0.002551020408163265)
    mul_851: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_852: "f32[256]" = torch.ops.aten.mul.Tensor(mul_850, mul_851);  mul_850 = mul_851 = None
    unsqueeze_159: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_852, 0);  mul_852 = None
    mul_853: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_100);  primals_100 = None
    unsqueeze_160: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_853, 0);  mul_853 = None
    sub_227: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_150, unsqueeze_157);  view_150 = unsqueeze_157 = None
    mul_854: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_159);  sub_227 = unsqueeze_159 = None
    sub_228: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_560, mul_854);  view_560 = mul_854 = None
    sub_229: "f32[392, 256]" = torch.ops.aten.sub.Tensor(sub_228, unsqueeze_158);  sub_228 = unsqueeze_158 = None
    mul_855: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_160);  sub_229 = unsqueeze_160 = None
    mul_856: "f32[256]" = torch.ops.aten.mul.Tensor(sum_104, squeeze_85);  sum_104 = squeeze_85 = None
    view_561: "f32[8, 49, 256]" = torch.ops.aten.view.default(mul_855, [8, 49, 256]);  mul_855 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_562: "f32[392, 256]" = torch.ops.aten.view.default(view_561, [392, 256]);  view_561 = None
    permute_321: "f32[256, 392]" = torch.ops.aten.permute.default(view_562, [1, 0])
    mm_128: "f32[256, 512]" = torch.ops.aten.mm.default(permute_321, view_148);  permute_321 = view_148 = None
    permute_322: "f32[512, 256]" = torch.ops.aten.permute.default(mm_128, [1, 0]);  mm_128 = None
    permute_323: "f32[256, 512]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
    mm_129: "f32[392, 512]" = torch.ops.aten.mm.default(view_562, permute_323);  view_562 = permute_323 = None
    view_563: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_129, [8, 49, 512]);  mm_129 = None
    permute_324: "f32[256, 512]" = torch.ops.aten.permute.default(permute_322, [1, 0]);  permute_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    lt_16: "b8[8, 49, 512]" = torch.ops.aten.lt.Scalar(view_147, -3)
    le_16: "b8[8, 49, 512]" = torch.ops.aten.le.Scalar(view_147, 3)
    div_64: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(view_147, 3);  view_147 = None
    add_425: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(div_64, 0.5);  div_64 = None
    mul_857: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_563, add_425);  add_425 = None
    where_32: "f32[8, 49, 512]" = torch.ops.aten.where.self(le_16, mul_857, view_563);  le_16 = mul_857 = view_563 = None
    scalar_tensor_16: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_33: "f32[8, 49, 512]" = torch.ops.aten.where.self(lt_16, scalar_tensor_16, where_32);  lt_16 = scalar_tensor_16 = where_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_564: "f32[392, 512]" = torch.ops.aten.view.default(where_33, [392, 512]);  where_33 = None
    unsqueeze_161: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    sum_105: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_564, [0])
    sub_230: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_146, unsqueeze_161)
    mul_858: "f32[392, 512]" = torch.ops.aten.mul.Tensor(view_564, sub_230);  sub_230 = None
    sum_106: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_858, [0]);  mul_858 = None
    mul_859: "f32[512]" = torch.ops.aten.mul.Tensor(sum_105, 0.002551020408163265)
    unsqueeze_162: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_859, 0);  mul_859 = None
    mul_860: "f32[512]" = torch.ops.aten.mul.Tensor(sum_106, 0.002551020408163265)
    mul_861: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_862: "f32[512]" = torch.ops.aten.mul.Tensor(mul_860, mul_861);  mul_860 = mul_861 = None
    unsqueeze_163: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_862, 0);  mul_862 = None
    mul_863: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_97);  primals_97 = None
    unsqueeze_164: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_863, 0);  mul_863 = None
    sub_231: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_146, unsqueeze_161);  view_146 = unsqueeze_161 = None
    mul_864: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_231, unsqueeze_163);  sub_231 = unsqueeze_163 = None
    sub_232: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_564, mul_864);  view_564 = mul_864 = None
    sub_233: "f32[392, 512]" = torch.ops.aten.sub.Tensor(sub_232, unsqueeze_162);  sub_232 = unsqueeze_162 = None
    mul_865: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_233, unsqueeze_164);  sub_233 = unsqueeze_164 = None
    mul_866: "f32[512]" = torch.ops.aten.mul.Tensor(sum_106, squeeze_82);  sum_106 = squeeze_82 = None
    view_565: "f32[8, 49, 512]" = torch.ops.aten.view.default(mul_865, [8, 49, 512]);  mul_865 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_566: "f32[392, 512]" = torch.ops.aten.view.default(view_565, [392, 512]);  view_565 = None
    permute_325: "f32[512, 392]" = torch.ops.aten.permute.default(view_566, [1, 0])
    mm_130: "f32[512, 256]" = torch.ops.aten.mm.default(permute_325, view_144);  permute_325 = view_144 = None
    permute_326: "f32[256, 512]" = torch.ops.aten.permute.default(mm_130, [1, 0]);  mm_130 = None
    permute_327: "f32[512, 256]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
    mm_131: "f32[392, 256]" = torch.ops.aten.mm.default(view_566, permute_327);  view_566 = permute_327 = None
    view_567: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_131, [8, 49, 256]);  mm_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_426: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_424, view_567);  add_424 = view_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_328: "f32[512, 256]" = torch.ops.aten.permute.default(permute_326, [1, 0]);  permute_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_568: "f32[392, 256]" = torch.ops.aten.view.default(add_426, [392, 256])
    unsqueeze_165: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    sum_107: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_568, [0])
    sub_234: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_142, unsqueeze_165)
    mul_867: "f32[392, 256]" = torch.ops.aten.mul.Tensor(view_568, sub_234);  sub_234 = None
    sum_108: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_867, [0]);  mul_867 = None
    mul_868: "f32[256]" = torch.ops.aten.mul.Tensor(sum_107, 0.002551020408163265)
    unsqueeze_166: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_868, 0);  mul_868 = None
    mul_869: "f32[256]" = torch.ops.aten.mul.Tensor(sum_108, 0.002551020408163265)
    mul_870: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_871: "f32[256]" = torch.ops.aten.mul.Tensor(mul_869, mul_870);  mul_869 = mul_870 = None
    unsqueeze_167: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_871, 0);  mul_871 = None
    mul_872: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_94);  primals_94 = None
    unsqueeze_168: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_872, 0);  mul_872 = None
    sub_235: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_142, unsqueeze_165);  view_142 = unsqueeze_165 = None
    mul_873: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_235, unsqueeze_167);  sub_235 = unsqueeze_167 = None
    sub_236: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_568, mul_873);  view_568 = mul_873 = None
    sub_237: "f32[392, 256]" = torch.ops.aten.sub.Tensor(sub_236, unsqueeze_166);  sub_236 = unsqueeze_166 = None
    mul_874: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_168);  sub_237 = unsqueeze_168 = None
    mul_875: "f32[256]" = torch.ops.aten.mul.Tensor(sum_108, squeeze_79);  sum_108 = squeeze_79 = None
    view_569: "f32[8, 49, 256]" = torch.ops.aten.view.default(mul_874, [8, 49, 256]);  mul_874 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_570: "f32[392, 256]" = torch.ops.aten.view.default(view_569, [392, 256]);  view_569 = None
    permute_329: "f32[256, 392]" = torch.ops.aten.permute.default(view_570, [1, 0])
    mm_132: "f32[256, 256]" = torch.ops.aten.mm.default(permute_329, view_140);  permute_329 = view_140 = None
    permute_330: "f32[256, 256]" = torch.ops.aten.permute.default(mm_132, [1, 0]);  mm_132 = None
    permute_331: "f32[256, 256]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    mm_133: "f32[392, 256]" = torch.ops.aten.mm.default(view_570, permute_331);  view_570 = permute_331 = None
    view_571: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_133, [8, 49, 256]);  mm_133 = None
    permute_332: "f32[256, 256]" = torch.ops.aten.permute.default(permute_330, [1, 0]);  permute_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    lt_17: "b8[8, 49, 256]" = torch.ops.aten.lt.Scalar(view_139, -3)
    le_17: "b8[8, 49, 256]" = torch.ops.aten.le.Scalar(view_139, 3)
    div_65: "f32[8, 49, 256]" = torch.ops.aten.div.Tensor(view_139, 3);  view_139 = None
    add_427: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(div_65, 0.5);  div_65 = None
    mul_876: "f32[8, 49, 256]" = torch.ops.aten.mul.Tensor(view_571, add_427);  add_427 = None
    where_34: "f32[8, 49, 256]" = torch.ops.aten.where.self(le_17, mul_876, view_571);  le_17 = mul_876 = view_571 = None
    scalar_tensor_17: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_35: "f32[8, 49, 256]" = torch.ops.aten.where.self(lt_17, scalar_tensor_17, where_34);  lt_17 = scalar_tensor_17 = where_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    view_572: "f32[8, 49, 8, 32]" = torch.ops.aten.view.default(where_35, [8, 49, 8, 32]);  where_35 = None
    permute_333: "f32[8, 8, 49, 32]" = torch.ops.aten.permute.default(view_572, [0, 2, 1, 3]);  view_572 = None
    clone_92: "f32[8, 8, 49, 32]" = torch.ops.aten.clone.default(permute_333, memory_format = torch.contiguous_format);  permute_333 = None
    view_573: "f32[64, 49, 32]" = torch.ops.aten.view.default(clone_92, [64, 49, 32]);  clone_92 = None
    permute_334: "f32[64, 49, 49]" = torch.ops.aten.permute.default(view_136, [0, 2, 1]);  view_136 = None
    bmm_60: "f32[64, 49, 32]" = torch.ops.aten.bmm.default(permute_334, view_573);  permute_334 = None
    permute_335: "f32[64, 32, 49]" = torch.ops.aten.permute.default(view_137, [0, 2, 1]);  view_137 = None
    bmm_61: "f32[64, 49, 49]" = torch.ops.aten.bmm.default(view_573, permute_335);  view_573 = permute_335 = None
    view_574: "f32[8, 8, 49, 32]" = torch.ops.aten.view.default(bmm_60, [8, 8, 49, 32]);  bmm_60 = None
    view_575: "f32[8, 8, 49, 49]" = torch.ops.aten.view.default(bmm_61, [8, 8, 49, 49]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    alias_22: "f32[8, 8, 49, 49]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul_877: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(view_575, alias_22);  view_575 = None
    sum_109: "f32[8, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_877, [-1], True)
    mul_878: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(alias_22, sum_109);  alias_22 = sum_109 = None
    sub_238: "f32[8, 8, 49, 49]" = torch.ops.aten.sub.Tensor(mul_877, mul_878);  mul_877 = mul_878 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    sum_110: "f32[1, 8, 49, 49]" = torch.ops.aten.sum.dim_IntList(sub_238, [0], True)
    view_576: "f32[8, 49, 49]" = torch.ops.aten.view.default(sum_110, [8, 49, 49]);  sum_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    full_19: "f32[8, 49]" = torch.ops.aten.full.default([8, 49], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    index_put_8: "f32[8, 49]" = torch.ops.aten.index_put.default(full_19, [None, primals_214], view_576, True);  full_19 = primals_214 = view_576 = None
    full_20: "f32[8, 49]" = torch.ops.aten.full.default([8, 49], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_11: "f32[8, 49]" = torch.ops.aten.slice_scatter.default(full_20, index_put_8, 0, 0, 9223372036854775807);  full_20 = index_put_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    mul_879: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(sub_238, 0.25);  sub_238 = None
    view_577: "f32[64, 49, 49]" = torch.ops.aten.view.default(mul_879, [64, 49, 49]);  mul_879 = None
    permute_336: "f32[64, 16, 49]" = torch.ops.aten.permute.default(view_133, [0, 2, 1]);  view_133 = None
    bmm_62: "f32[64, 16, 49]" = torch.ops.aten.bmm.default(permute_336, view_577);  permute_336 = None
    permute_337: "f32[64, 49, 16]" = torch.ops.aten.permute.default(view_134, [0, 2, 1]);  view_134 = None
    bmm_63: "f32[64, 49, 16]" = torch.ops.aten.bmm.default(view_577, permute_337);  view_577 = permute_337 = None
    view_578: "f32[8, 8, 16, 49]" = torch.ops.aten.view.default(bmm_62, [8, 8, 16, 49]);  bmm_62 = None
    view_579: "f32[8, 8, 49, 16]" = torch.ops.aten.view.default(bmm_63, [8, 8, 49, 16]);  bmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_338: "f32[8, 49, 8, 32]" = torch.ops.aten.permute.default(view_574, [0, 2, 1, 3]);  view_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_339: "f32[8, 49, 8, 16]" = torch.ops.aten.permute.default(view_578, [0, 3, 1, 2]);  view_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_340: "f32[8, 49, 8, 16]" = torch.ops.aten.permute.default(view_579, [0, 2, 1, 3]);  view_579 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    cat_8: "f32[8, 49, 8, 64]" = torch.ops.aten.cat.default([permute_340, permute_339, permute_338], 3);  permute_340 = permute_339 = permute_338 = None
    view_580: "f32[8, 49, 512]" = torch.ops.aten.view.default(cat_8, [8, 49, 512]);  cat_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_581: "f32[392, 512]" = torch.ops.aten.view.default(view_580, [392, 512]);  view_580 = None
    unsqueeze_169: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    sum_111: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_581, [0])
    sub_239: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_130, unsqueeze_169)
    mul_880: "f32[392, 512]" = torch.ops.aten.mul.Tensor(view_581, sub_239);  sub_239 = None
    sum_112: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_880, [0]);  mul_880 = None
    mul_881: "f32[512]" = torch.ops.aten.mul.Tensor(sum_111, 0.002551020408163265)
    unsqueeze_170: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_881, 0);  mul_881 = None
    mul_882: "f32[512]" = torch.ops.aten.mul.Tensor(sum_112, 0.002551020408163265)
    mul_883: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_884: "f32[512]" = torch.ops.aten.mul.Tensor(mul_882, mul_883);  mul_882 = mul_883 = None
    unsqueeze_171: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_884, 0);  mul_884 = None
    mul_885: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_91);  primals_91 = None
    unsqueeze_172: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_885, 0);  mul_885 = None
    sub_240: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_130, unsqueeze_169);  view_130 = unsqueeze_169 = None
    mul_886: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_171);  sub_240 = unsqueeze_171 = None
    sub_241: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_581, mul_886);  view_581 = mul_886 = None
    sub_242: "f32[392, 512]" = torch.ops.aten.sub.Tensor(sub_241, unsqueeze_170);  sub_241 = unsqueeze_170 = None
    mul_887: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_242, unsqueeze_172);  sub_242 = unsqueeze_172 = None
    mul_888: "f32[512]" = torch.ops.aten.mul.Tensor(sum_112, squeeze_76);  sum_112 = squeeze_76 = None
    view_582: "f32[8, 49, 512]" = torch.ops.aten.view.default(mul_887, [8, 49, 512]);  mul_887 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_583: "f32[392, 512]" = torch.ops.aten.view.default(view_582, [392, 512]);  view_582 = None
    permute_341: "f32[512, 392]" = torch.ops.aten.permute.default(view_583, [1, 0])
    mm_134: "f32[512, 256]" = torch.ops.aten.mm.default(permute_341, view_128);  permute_341 = view_128 = None
    permute_342: "f32[256, 512]" = torch.ops.aten.permute.default(mm_134, [1, 0]);  mm_134 = None
    permute_343: "f32[512, 256]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    mm_135: "f32[392, 256]" = torch.ops.aten.mm.default(view_583, permute_343);  view_583 = permute_343 = None
    view_584: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_135, [8, 49, 256]);  mm_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_428: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_426, view_584);  add_426 = view_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_344: "f32[512, 256]" = torch.ops.aten.permute.default(permute_342, [1, 0]);  permute_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_585: "f32[392, 256]" = torch.ops.aten.view.default(add_428, [392, 256])
    unsqueeze_173: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    sum_113: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_585, [0])
    sub_243: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_126, unsqueeze_173)
    mul_889: "f32[392, 256]" = torch.ops.aten.mul.Tensor(view_585, sub_243);  sub_243 = None
    sum_114: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_889, [0]);  mul_889 = None
    mul_890: "f32[256]" = torch.ops.aten.mul.Tensor(sum_113, 0.002551020408163265)
    unsqueeze_174: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_890, 0);  mul_890 = None
    mul_891: "f32[256]" = torch.ops.aten.mul.Tensor(sum_114, 0.002551020408163265)
    mul_892: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_893: "f32[256]" = torch.ops.aten.mul.Tensor(mul_891, mul_892);  mul_891 = mul_892 = None
    unsqueeze_175: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_893, 0);  mul_893 = None
    mul_894: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_88);  primals_88 = None
    unsqueeze_176: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_894, 0);  mul_894 = None
    sub_244: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_126, unsqueeze_173);  view_126 = unsqueeze_173 = None
    mul_895: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_175);  sub_244 = unsqueeze_175 = None
    sub_245: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_585, mul_895);  view_585 = mul_895 = None
    sub_246: "f32[392, 256]" = torch.ops.aten.sub.Tensor(sub_245, unsqueeze_174);  sub_245 = unsqueeze_174 = None
    mul_896: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_246, unsqueeze_176);  sub_246 = unsqueeze_176 = None
    mul_897: "f32[256]" = torch.ops.aten.mul.Tensor(sum_114, squeeze_73);  sum_114 = squeeze_73 = None
    view_586: "f32[8, 49, 256]" = torch.ops.aten.view.default(mul_896, [8, 49, 256]);  mul_896 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_587: "f32[392, 256]" = torch.ops.aten.view.default(view_586, [392, 256]);  view_586 = None
    permute_345: "f32[256, 392]" = torch.ops.aten.permute.default(view_587, [1, 0])
    mm_136: "f32[256, 512]" = torch.ops.aten.mm.default(permute_345, view_124);  permute_345 = view_124 = None
    permute_346: "f32[512, 256]" = torch.ops.aten.permute.default(mm_136, [1, 0]);  mm_136 = None
    permute_347: "f32[256, 512]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    mm_137: "f32[392, 512]" = torch.ops.aten.mm.default(view_587, permute_347);  view_587 = permute_347 = None
    view_588: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_137, [8, 49, 512]);  mm_137 = None
    permute_348: "f32[256, 512]" = torch.ops.aten.permute.default(permute_346, [1, 0]);  permute_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    lt_18: "b8[8, 49, 512]" = torch.ops.aten.lt.Scalar(view_123, -3)
    le_18: "b8[8, 49, 512]" = torch.ops.aten.le.Scalar(view_123, 3)
    div_66: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(view_123, 3);  view_123 = None
    add_429: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(div_66, 0.5);  div_66 = None
    mul_898: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_588, add_429);  add_429 = None
    where_36: "f32[8, 49, 512]" = torch.ops.aten.where.self(le_18, mul_898, view_588);  le_18 = mul_898 = view_588 = None
    scalar_tensor_18: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_37: "f32[8, 49, 512]" = torch.ops.aten.where.self(lt_18, scalar_tensor_18, where_36);  lt_18 = scalar_tensor_18 = where_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_589: "f32[392, 512]" = torch.ops.aten.view.default(where_37, [392, 512]);  where_37 = None
    unsqueeze_177: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    sum_115: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_589, [0])
    sub_247: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_122, unsqueeze_177)
    mul_899: "f32[392, 512]" = torch.ops.aten.mul.Tensor(view_589, sub_247);  sub_247 = None
    sum_116: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_899, [0]);  mul_899 = None
    mul_900: "f32[512]" = torch.ops.aten.mul.Tensor(sum_115, 0.002551020408163265)
    unsqueeze_178: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_900, 0);  mul_900 = None
    mul_901: "f32[512]" = torch.ops.aten.mul.Tensor(sum_116, 0.002551020408163265)
    mul_902: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_903: "f32[512]" = torch.ops.aten.mul.Tensor(mul_901, mul_902);  mul_901 = mul_902 = None
    unsqueeze_179: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_903, 0);  mul_903 = None
    mul_904: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_85);  primals_85 = None
    unsqueeze_180: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_904, 0);  mul_904 = None
    sub_248: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_122, unsqueeze_177);  view_122 = unsqueeze_177 = None
    mul_905: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_179);  sub_248 = unsqueeze_179 = None
    sub_249: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_589, mul_905);  view_589 = mul_905 = None
    sub_250: "f32[392, 512]" = torch.ops.aten.sub.Tensor(sub_249, unsqueeze_178);  sub_249 = unsqueeze_178 = None
    mul_906: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_250, unsqueeze_180);  sub_250 = unsqueeze_180 = None
    mul_907: "f32[512]" = torch.ops.aten.mul.Tensor(sum_116, squeeze_70);  sum_116 = squeeze_70 = None
    view_590: "f32[8, 49, 512]" = torch.ops.aten.view.default(mul_906, [8, 49, 512]);  mul_906 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_591: "f32[392, 512]" = torch.ops.aten.view.default(view_590, [392, 512]);  view_590 = None
    permute_349: "f32[512, 392]" = torch.ops.aten.permute.default(view_591, [1, 0])
    mm_138: "f32[512, 256]" = torch.ops.aten.mm.default(permute_349, view_120);  permute_349 = view_120 = None
    permute_350: "f32[256, 512]" = torch.ops.aten.permute.default(mm_138, [1, 0]);  mm_138 = None
    permute_351: "f32[512, 256]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
    mm_139: "f32[392, 256]" = torch.ops.aten.mm.default(view_591, permute_351);  view_591 = permute_351 = None
    view_592: "f32[8, 49, 256]" = torch.ops.aten.view.default(mm_139, [8, 49, 256]);  mm_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_430: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_428, view_592);  add_428 = view_592 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_352: "f32[512, 256]" = torch.ops.aten.permute.default(permute_350, [1, 0]);  permute_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_593: "f32[392, 256]" = torch.ops.aten.view.default(add_430, [392, 256]);  add_430 = None
    unsqueeze_181: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    sum_117: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_593, [0])
    sub_251: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_118, unsqueeze_181)
    mul_908: "f32[392, 256]" = torch.ops.aten.mul.Tensor(view_593, sub_251);  sub_251 = None
    sum_118: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_908, [0]);  mul_908 = None
    mul_909: "f32[256]" = torch.ops.aten.mul.Tensor(sum_117, 0.002551020408163265)
    unsqueeze_182: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_909, 0);  mul_909 = None
    mul_910: "f32[256]" = torch.ops.aten.mul.Tensor(sum_118, 0.002551020408163265)
    mul_911: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_912: "f32[256]" = torch.ops.aten.mul.Tensor(mul_910, mul_911);  mul_910 = mul_911 = None
    unsqueeze_183: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_912, 0);  mul_912 = None
    mul_913: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_82);  primals_82 = None
    unsqueeze_184: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_913, 0);  mul_913 = None
    sub_252: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_118, unsqueeze_181);  view_118 = unsqueeze_181 = None
    mul_914: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_183);  sub_252 = unsqueeze_183 = None
    sub_253: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_593, mul_914);  view_593 = mul_914 = None
    sub_254: "f32[392, 256]" = torch.ops.aten.sub.Tensor(sub_253, unsqueeze_182);  sub_253 = unsqueeze_182 = None
    mul_915: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_254, unsqueeze_184);  sub_254 = unsqueeze_184 = None
    mul_916: "f32[256]" = torch.ops.aten.mul.Tensor(sum_118, squeeze_67);  sum_118 = squeeze_67 = None
    view_594: "f32[8, 49, 256]" = torch.ops.aten.view.default(mul_915, [8, 49, 256]);  mul_915 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_595: "f32[392, 256]" = torch.ops.aten.view.default(view_594, [392, 256]);  view_594 = None
    permute_353: "f32[256, 392]" = torch.ops.aten.permute.default(view_595, [1, 0])
    mm_140: "f32[256, 512]" = torch.ops.aten.mm.default(permute_353, view_116);  permute_353 = view_116 = None
    permute_354: "f32[512, 256]" = torch.ops.aten.permute.default(mm_140, [1, 0]);  mm_140 = None
    permute_355: "f32[256, 512]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
    mm_141: "f32[392, 512]" = torch.ops.aten.mm.default(view_595, permute_355);  view_595 = permute_355 = None
    view_596: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_141, [8, 49, 512]);  mm_141 = None
    permute_356: "f32[256, 512]" = torch.ops.aten.permute.default(permute_354, [1, 0]);  permute_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:340, code: x = self.proj(x)
    lt_19: "b8[8, 49, 512]" = torch.ops.aten.lt.Scalar(view_115, -3)
    le_19: "b8[8, 49, 512]" = torch.ops.aten.le.Scalar(view_115, 3)
    div_67: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(view_115, 3);  view_115 = None
    add_431: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(div_67, 0.5);  div_67 = None
    mul_917: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_596, add_431);  add_431 = None
    where_38: "f32[8, 49, 512]" = torch.ops.aten.where.self(le_19, mul_917, view_596);  le_19 = mul_917 = view_596 = None
    scalar_tensor_19: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_39: "f32[8, 49, 512]" = torch.ops.aten.where.self(lt_19, scalar_tensor_19, where_38);  lt_19 = scalar_tensor_19 = where_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:339, code: x = (attn @ v).transpose(1, 2).reshape(B, -1, self.val_attn_dim)
    view_597: "f32[8, 49, 8, 64]" = torch.ops.aten.view.default(where_39, [8, 49, 8, 64]);  where_39 = None
    permute_357: "f32[8, 8, 49, 64]" = torch.ops.aten.permute.default(view_597, [0, 2, 1, 3]);  view_597 = None
    clone_93: "f32[8, 8, 49, 64]" = torch.ops.aten.clone.default(permute_357, memory_format = torch.contiguous_format);  permute_357 = None
    view_598: "f32[64, 49, 64]" = torch.ops.aten.view.default(clone_93, [64, 49, 64]);  clone_93 = None
    permute_358: "f32[64, 196, 49]" = torch.ops.aten.permute.default(view_112, [0, 2, 1]);  view_112 = None
    bmm_64: "f32[64, 196, 64]" = torch.ops.aten.bmm.default(permute_358, view_598);  permute_358 = None
    permute_359: "f32[64, 64, 196]" = torch.ops.aten.permute.default(view_113, [0, 2, 1]);  view_113 = None
    bmm_65: "f32[64, 49, 196]" = torch.ops.aten.bmm.default(view_598, permute_359);  view_598 = permute_359 = None
    view_599: "f32[8, 8, 196, 64]" = torch.ops.aten.view.default(bmm_64, [8, 8, 196, 64]);  bmm_64 = None
    view_600: "f32[8, 8, 49, 196]" = torch.ops.aten.view.default(bmm_65, [8, 8, 49, 196]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:337, code: attn = attn.softmax(dim=-1)
    alias_23: "f32[8, 8, 49, 196]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    mul_918: "f32[8, 8, 49, 196]" = torch.ops.aten.mul.Tensor(view_600, alias_23);  view_600 = None
    sum_119: "f32[8, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_918, [-1], True)
    mul_919: "f32[8, 8, 49, 196]" = torch.ops.aten.mul.Tensor(alias_23, sum_119);  alias_23 = sum_119 = None
    sub_255: "f32[8, 8, 49, 196]" = torch.ops.aten.sub.Tensor(mul_918, mul_919);  mul_918 = mul_919 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:336, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    sum_120: "f32[1, 8, 49, 196]" = torch.ops.aten.sum.dim_IntList(sub_255, [0], True)
    view_601: "f32[8, 49, 196]" = torch.ops.aten.view.default(sum_120, [8, 49, 196]);  sum_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:311, code: return self.attention_biases[:, self.attention_bias_idxs]
    full_21: "f32[8, 196]" = torch.ops.aten.full.default([8, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    index_put_9: "f32[8, 196]" = torch.ops.aten.index_put.default(full_21, [None, primals_213], view_601, True);  full_21 = primals_213 = view_601 = None
    full_22: "f32[8, 196]" = torch.ops.aten.full.default([8, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_12: "f32[8, 196]" = torch.ops.aten.slice_scatter.default(full_22, index_put_9, 0, 0, 9223372036854775807);  full_22 = index_put_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:336, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    mul_920: "f32[8, 8, 49, 196]" = torch.ops.aten.mul.Tensor(sub_255, 0.25);  sub_255 = None
    view_602: "f32[64, 49, 196]" = torch.ops.aten.view.default(mul_920, [64, 49, 196]);  mul_920 = None
    permute_360: "f32[64, 16, 49]" = torch.ops.aten.permute.default(view_109, [0, 2, 1]);  view_109 = None
    bmm_66: "f32[64, 16, 196]" = torch.ops.aten.bmm.default(permute_360, view_602);  permute_360 = None
    permute_361: "f32[64, 196, 16]" = torch.ops.aten.permute.default(view_110, [0, 2, 1]);  view_110 = None
    bmm_67: "f32[64, 49, 16]" = torch.ops.aten.bmm.default(view_602, permute_361);  view_602 = permute_361 = None
    view_603: "f32[8, 8, 16, 196]" = torch.ops.aten.view.default(bmm_66, [8, 8, 16, 196]);  bmm_66 = None
    view_604: "f32[8, 8, 49, 16]" = torch.ops.aten.view.default(bmm_67, [8, 8, 49, 16]);  bmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:334, code: q = self.q(x).view(B, -1, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
    permute_362: "f32[8, 49, 8, 16]" = torch.ops.aten.permute.default(view_604, [0, 2, 1, 3]);  view_604 = None
    clone_94: "f32[8, 49, 8, 16]" = torch.ops.aten.clone.default(permute_362, memory_format = torch.contiguous_format);  permute_362 = None
    view_605: "f32[8, 49, 128]" = torch.ops.aten.view.default(clone_94, [8, 49, 128]);  clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_606: "f32[392, 128]" = torch.ops.aten.view.default(view_605, [392, 128]);  view_605 = None
    unsqueeze_185: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    sum_121: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_606, [0])
    sub_256: "f32[392, 128]" = torch.ops.aten.sub.Tensor(view_106, unsqueeze_185)
    mul_921: "f32[392, 128]" = torch.ops.aten.mul.Tensor(view_606, sub_256);  sub_256 = None
    sum_122: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_921, [0]);  mul_921 = None
    mul_922: "f32[128]" = torch.ops.aten.mul.Tensor(sum_121, 0.002551020408163265)
    unsqueeze_186: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_922, 0);  mul_922 = None
    mul_923: "f32[128]" = torch.ops.aten.mul.Tensor(sum_122, 0.002551020408163265)
    mul_924: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_925: "f32[128]" = torch.ops.aten.mul.Tensor(mul_923, mul_924);  mul_923 = mul_924 = None
    unsqueeze_187: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_925, 0);  mul_925 = None
    mul_926: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_79);  primals_79 = None
    unsqueeze_188: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_926, 0);  mul_926 = None
    sub_257: "f32[392, 128]" = torch.ops.aten.sub.Tensor(view_106, unsqueeze_185);  view_106 = unsqueeze_185 = None
    mul_927: "f32[392, 128]" = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_187);  sub_257 = unsqueeze_187 = None
    sub_258: "f32[392, 128]" = torch.ops.aten.sub.Tensor(view_606, mul_927);  view_606 = mul_927 = None
    sub_259: "f32[392, 128]" = torch.ops.aten.sub.Tensor(sub_258, unsqueeze_186);  sub_258 = unsqueeze_186 = None
    mul_928: "f32[392, 128]" = torch.ops.aten.mul.Tensor(sub_259, unsqueeze_188);  sub_259 = unsqueeze_188 = None
    mul_929: "f32[128]" = torch.ops.aten.mul.Tensor(sum_122, squeeze_64);  sum_122 = squeeze_64 = None
    view_607: "f32[8, 49, 128]" = torch.ops.aten.view.default(mul_928, [8, 49, 128]);  mul_928 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_608: "f32[392, 128]" = torch.ops.aten.view.default(view_607, [392, 128]);  view_607 = None
    permute_363: "f32[128, 392]" = torch.ops.aten.permute.default(view_608, [1, 0])
    mm_142: "f32[128, 128]" = torch.ops.aten.mm.default(permute_363, view_104);  permute_363 = view_104 = None
    permute_364: "f32[128, 128]" = torch.ops.aten.permute.default(mm_142, [1, 0]);  mm_142 = None
    permute_365: "f32[128, 128]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    mm_143: "f32[392, 128]" = torch.ops.aten.mm.default(view_608, permute_365);  view_608 = permute_365 = None
    view_609: "f32[8, 49, 128]" = torch.ops.aten.view.default(mm_143, [8, 49, 128]);  mm_143 = None
    permute_366: "f32[128, 128]" = torch.ops.aten.permute.default(permute_364, [1, 0]);  permute_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:162, code: return x.reshape(B, -1, C)
    view_610: "f32[8, 7, 7, 128]" = torch.ops.aten.view.default(view_609, [8, 7, 7, 128]);  view_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:161, code: x = x[:, ::self.stride, ::self.stride]
    full_23: "f32[8, 7, 14, 128]" = torch.ops.aten.full.default([8, 7, 14, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_13: "f32[8, 7, 14, 128]" = torch.ops.aten.slice_scatter.default(full_23, view_610, 2, 0, 9223372036854775807, 2);  full_23 = view_610 = None
    full_24: "f32[8, 14, 14, 128]" = torch.ops.aten.full.default([8, 14, 14, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_14: "f32[8, 14, 14, 128]" = torch.ops.aten.slice_scatter.default(full_24, slice_scatter_13, 1, 0, 9223372036854775807, 2);  full_24 = slice_scatter_13 = None
    full_25: "f32[8, 14, 14, 128]" = torch.ops.aten.full.default([8, 14, 14, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_15: "f32[8, 14, 14, 128]" = torch.ops.aten.slice_scatter.default(full_25, slice_scatter_14, 0, 0, 9223372036854775807);  full_25 = slice_scatter_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:157, code: x = x.view(B, self.resolution[0], self.resolution[1], C)
    view_611: "f32[8, 196, 128]" = torch.ops.aten.view.default(slice_scatter_15, [8, 196, 128]);  slice_scatter_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:333, code: v = v.permute(0, 2, 1, 3)  # BHNC
    permute_367: "f32[8, 196, 8, 64]" = torch.ops.aten.permute.default(view_599, [0, 2, 1, 3]);  view_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:332, code: k = k.permute(0, 2, 3, 1)  # BHCN
    permute_368: "f32[8, 196, 8, 16]" = torch.ops.aten.permute.default(view_603, [0, 3, 1, 2]);  view_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:331, code: k, v = self.kv(x).view(B, N, self.num_heads, -1).split([self.key_dim, self.val_dim], dim=3)
    cat_9: "f32[8, 196, 8, 80]" = torch.ops.aten.cat.default([permute_368, permute_367], 3);  permute_368 = permute_367 = None
    view_612: "f32[8, 196, 640]" = torch.ops.aten.view.default(cat_9, [8, 196, 640]);  cat_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_613: "f32[1568, 640]" = torch.ops.aten.view.default(view_612, [1568, 640]);  view_612 = None
    unsqueeze_189: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    sum_123: "f32[640]" = torch.ops.aten.sum.dim_IntList(view_613, [0])
    sub_260: "f32[1568, 640]" = torch.ops.aten.sub.Tensor(view_99, unsqueeze_189)
    mul_930: "f32[1568, 640]" = torch.ops.aten.mul.Tensor(view_613, sub_260);  sub_260 = None
    sum_124: "f32[640]" = torch.ops.aten.sum.dim_IntList(mul_930, [0]);  mul_930 = None
    mul_931: "f32[640]" = torch.ops.aten.mul.Tensor(sum_123, 0.0006377551020408163)
    unsqueeze_190: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_931, 0);  mul_931 = None
    mul_932: "f32[640]" = torch.ops.aten.mul.Tensor(sum_124, 0.0006377551020408163)
    mul_933: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_934: "f32[640]" = torch.ops.aten.mul.Tensor(mul_932, mul_933);  mul_932 = mul_933 = None
    unsqueeze_191: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_934, 0);  mul_934 = None
    mul_935: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_76);  primals_76 = None
    unsqueeze_192: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_935, 0);  mul_935 = None
    sub_261: "f32[1568, 640]" = torch.ops.aten.sub.Tensor(view_99, unsqueeze_189);  view_99 = unsqueeze_189 = None
    mul_936: "f32[1568, 640]" = torch.ops.aten.mul.Tensor(sub_261, unsqueeze_191);  sub_261 = unsqueeze_191 = None
    sub_262: "f32[1568, 640]" = torch.ops.aten.sub.Tensor(view_613, mul_936);  view_613 = mul_936 = None
    sub_263: "f32[1568, 640]" = torch.ops.aten.sub.Tensor(sub_262, unsqueeze_190);  sub_262 = unsqueeze_190 = None
    mul_937: "f32[1568, 640]" = torch.ops.aten.mul.Tensor(sub_263, unsqueeze_192);  sub_263 = unsqueeze_192 = None
    mul_938: "f32[640]" = torch.ops.aten.mul.Tensor(sum_124, squeeze_61);  sum_124 = squeeze_61 = None
    view_614: "f32[8, 196, 640]" = torch.ops.aten.view.default(mul_937, [8, 196, 640]);  mul_937 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_615: "f32[1568, 640]" = torch.ops.aten.view.default(view_614, [1568, 640]);  view_614 = None
    permute_369: "f32[640, 1568]" = torch.ops.aten.permute.default(view_615, [1, 0])
    mm_144: "f32[640, 128]" = torch.ops.aten.mm.default(permute_369, view_97);  permute_369 = view_97 = None
    permute_370: "f32[128, 640]" = torch.ops.aten.permute.default(mm_144, [1, 0]);  mm_144 = None
    permute_371: "f32[640, 128]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    mm_145: "f32[1568, 128]" = torch.ops.aten.mm.default(view_615, permute_371);  view_615 = permute_371 = None
    view_616: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_145, [8, 196, 128]);  mm_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_432: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(view_611, view_616);  view_611 = view_616 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_372: "f32[640, 128]" = torch.ops.aten.permute.default(permute_370, [1, 0]);  permute_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_617: "f32[1568, 128]" = torch.ops.aten.view.default(add_432, [1568, 128])
    unsqueeze_193: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    sum_125: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_617, [0])
    sub_264: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_95, unsqueeze_193)
    mul_939: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(view_617, sub_264);  sub_264 = None
    sum_126: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_939, [0]);  mul_939 = None
    mul_940: "f32[128]" = torch.ops.aten.mul.Tensor(sum_125, 0.0006377551020408163)
    unsqueeze_194: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_940, 0);  mul_940 = None
    mul_941: "f32[128]" = torch.ops.aten.mul.Tensor(sum_126, 0.0006377551020408163)
    mul_942: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_943: "f32[128]" = torch.ops.aten.mul.Tensor(mul_941, mul_942);  mul_941 = mul_942 = None
    unsqueeze_195: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_943, 0);  mul_943 = None
    mul_944: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_73);  primals_73 = None
    unsqueeze_196: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_944, 0);  mul_944 = None
    sub_265: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_95, unsqueeze_193);  view_95 = unsqueeze_193 = None
    mul_945: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_265, unsqueeze_195);  sub_265 = unsqueeze_195 = None
    sub_266: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_617, mul_945);  view_617 = mul_945 = None
    sub_267: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(sub_266, unsqueeze_194);  sub_266 = unsqueeze_194 = None
    mul_946: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_267, unsqueeze_196);  sub_267 = unsqueeze_196 = None
    mul_947: "f32[128]" = torch.ops.aten.mul.Tensor(sum_126, squeeze_58);  sum_126 = squeeze_58 = None
    view_618: "f32[8, 196, 128]" = torch.ops.aten.view.default(mul_946, [8, 196, 128]);  mul_946 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_619: "f32[1568, 128]" = torch.ops.aten.view.default(view_618, [1568, 128]);  view_618 = None
    permute_373: "f32[128, 1568]" = torch.ops.aten.permute.default(view_619, [1, 0])
    mm_146: "f32[128, 256]" = torch.ops.aten.mm.default(permute_373, view_93);  permute_373 = view_93 = None
    permute_374: "f32[256, 128]" = torch.ops.aten.permute.default(mm_146, [1, 0]);  mm_146 = None
    permute_375: "f32[128, 256]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    mm_147: "f32[1568, 256]" = torch.ops.aten.mm.default(view_619, permute_375);  view_619 = permute_375 = None
    view_620: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_147, [8, 196, 256]);  mm_147 = None
    permute_376: "f32[128, 256]" = torch.ops.aten.permute.default(permute_374, [1, 0]);  permute_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    lt_20: "b8[8, 196, 256]" = torch.ops.aten.lt.Scalar(view_92, -3)
    le_20: "b8[8, 196, 256]" = torch.ops.aten.le.Scalar(view_92, 3)
    div_68: "f32[8, 196, 256]" = torch.ops.aten.div.Tensor(view_92, 3);  view_92 = None
    add_433: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(div_68, 0.5);  div_68 = None
    mul_948: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_620, add_433);  add_433 = None
    where_40: "f32[8, 196, 256]" = torch.ops.aten.where.self(le_20, mul_948, view_620);  le_20 = mul_948 = view_620 = None
    scalar_tensor_20: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_41: "f32[8, 196, 256]" = torch.ops.aten.where.self(lt_20, scalar_tensor_20, where_40);  lt_20 = scalar_tensor_20 = where_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_621: "f32[1568, 256]" = torch.ops.aten.view.default(where_41, [1568, 256]);  where_41 = None
    unsqueeze_197: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    sum_127: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_621, [0])
    sub_268: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_91, unsqueeze_197)
    mul_949: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(view_621, sub_268);  sub_268 = None
    sum_128: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_949, [0]);  mul_949 = None
    mul_950: "f32[256]" = torch.ops.aten.mul.Tensor(sum_127, 0.0006377551020408163)
    unsqueeze_198: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_950, 0);  mul_950 = None
    mul_951: "f32[256]" = torch.ops.aten.mul.Tensor(sum_128, 0.0006377551020408163)
    mul_952: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_953: "f32[256]" = torch.ops.aten.mul.Tensor(mul_951, mul_952);  mul_951 = mul_952 = None
    unsqueeze_199: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_953, 0);  mul_953 = None
    mul_954: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_70);  primals_70 = None
    unsqueeze_200: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_954, 0);  mul_954 = None
    sub_269: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_91, unsqueeze_197);  view_91 = unsqueeze_197 = None
    mul_955: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_269, unsqueeze_199);  sub_269 = unsqueeze_199 = None
    sub_270: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_621, mul_955);  view_621 = mul_955 = None
    sub_271: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(sub_270, unsqueeze_198);  sub_270 = unsqueeze_198 = None
    mul_956: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_271, unsqueeze_200);  sub_271 = unsqueeze_200 = None
    mul_957: "f32[256]" = torch.ops.aten.mul.Tensor(sum_128, squeeze_55);  sum_128 = squeeze_55 = None
    view_622: "f32[8, 196, 256]" = torch.ops.aten.view.default(mul_956, [8, 196, 256]);  mul_956 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_623: "f32[1568, 256]" = torch.ops.aten.view.default(view_622, [1568, 256]);  view_622 = None
    permute_377: "f32[256, 1568]" = torch.ops.aten.permute.default(view_623, [1, 0])
    mm_148: "f32[256, 128]" = torch.ops.aten.mm.default(permute_377, view_89);  permute_377 = view_89 = None
    permute_378: "f32[128, 256]" = torch.ops.aten.permute.default(mm_148, [1, 0]);  mm_148 = None
    permute_379: "f32[256, 128]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    mm_149: "f32[1568, 128]" = torch.ops.aten.mm.default(view_623, permute_379);  view_623 = permute_379 = None
    view_624: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_149, [8, 196, 128]);  mm_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_434: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_432, view_624);  add_432 = view_624 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_380: "f32[256, 128]" = torch.ops.aten.permute.default(permute_378, [1, 0]);  permute_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_625: "f32[1568, 128]" = torch.ops.aten.view.default(add_434, [1568, 128])
    unsqueeze_201: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    sum_129: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_625, [0])
    sub_272: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_87, unsqueeze_201)
    mul_958: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(view_625, sub_272);  sub_272 = None
    sum_130: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_958, [0]);  mul_958 = None
    mul_959: "f32[128]" = torch.ops.aten.mul.Tensor(sum_129, 0.0006377551020408163)
    unsqueeze_202: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_959, 0);  mul_959 = None
    mul_960: "f32[128]" = torch.ops.aten.mul.Tensor(sum_130, 0.0006377551020408163)
    mul_961: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_962: "f32[128]" = torch.ops.aten.mul.Tensor(mul_960, mul_961);  mul_960 = mul_961 = None
    unsqueeze_203: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_962, 0);  mul_962 = None
    mul_963: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_67);  primals_67 = None
    unsqueeze_204: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_963, 0);  mul_963 = None
    sub_273: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_87, unsqueeze_201);  view_87 = unsqueeze_201 = None
    mul_964: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_273, unsqueeze_203);  sub_273 = unsqueeze_203 = None
    sub_274: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_625, mul_964);  view_625 = mul_964 = None
    sub_275: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(sub_274, unsqueeze_202);  sub_274 = unsqueeze_202 = None
    mul_965: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_275, unsqueeze_204);  sub_275 = unsqueeze_204 = None
    mul_966: "f32[128]" = torch.ops.aten.mul.Tensor(sum_130, squeeze_52);  sum_130 = squeeze_52 = None
    view_626: "f32[8, 196, 128]" = torch.ops.aten.view.default(mul_965, [8, 196, 128]);  mul_965 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_627: "f32[1568, 128]" = torch.ops.aten.view.default(view_626, [1568, 128]);  view_626 = None
    permute_381: "f32[128, 1568]" = torch.ops.aten.permute.default(view_627, [1, 0])
    mm_150: "f32[128, 128]" = torch.ops.aten.mm.default(permute_381, view_85);  permute_381 = view_85 = None
    permute_382: "f32[128, 128]" = torch.ops.aten.permute.default(mm_150, [1, 0]);  mm_150 = None
    permute_383: "f32[128, 128]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    mm_151: "f32[1568, 128]" = torch.ops.aten.mm.default(view_627, permute_383);  view_627 = permute_383 = None
    view_628: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_151, [8, 196, 128]);  mm_151 = None
    permute_384: "f32[128, 128]" = torch.ops.aten.permute.default(permute_382, [1, 0]);  permute_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    lt_21: "b8[8, 196, 128]" = torch.ops.aten.lt.Scalar(view_84, -3)
    le_21: "b8[8, 196, 128]" = torch.ops.aten.le.Scalar(view_84, 3)
    div_69: "f32[8, 196, 128]" = torch.ops.aten.div.Tensor(view_84, 3);  view_84 = None
    add_435: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(div_69, 0.5);  div_69 = None
    mul_967: "f32[8, 196, 128]" = torch.ops.aten.mul.Tensor(view_628, add_435);  add_435 = None
    where_42: "f32[8, 196, 128]" = torch.ops.aten.where.self(le_21, mul_967, view_628);  le_21 = mul_967 = view_628 = None
    scalar_tensor_21: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_43: "f32[8, 196, 128]" = torch.ops.aten.where.self(lt_21, scalar_tensor_21, where_42);  lt_21 = scalar_tensor_21 = where_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    view_629: "f32[8, 196, 4, 32]" = torch.ops.aten.view.default(where_43, [8, 196, 4, 32]);  where_43 = None
    permute_385: "f32[8, 4, 196, 32]" = torch.ops.aten.permute.default(view_629, [0, 2, 1, 3]);  view_629 = None
    clone_95: "f32[8, 4, 196, 32]" = torch.ops.aten.clone.default(permute_385, memory_format = torch.contiguous_format);  permute_385 = None
    view_630: "f32[32, 196, 32]" = torch.ops.aten.view.default(clone_95, [32, 196, 32]);  clone_95 = None
    permute_386: "f32[32, 196, 196]" = torch.ops.aten.permute.default(view_81, [0, 2, 1]);  view_81 = None
    bmm_68: "f32[32, 196, 32]" = torch.ops.aten.bmm.default(permute_386, view_630);  permute_386 = None
    permute_387: "f32[32, 32, 196]" = torch.ops.aten.permute.default(view_82, [0, 2, 1]);  view_82 = None
    bmm_69: "f32[32, 196, 196]" = torch.ops.aten.bmm.default(view_630, permute_387);  view_630 = permute_387 = None
    view_631: "f32[8, 4, 196, 32]" = torch.ops.aten.view.default(bmm_68, [8, 4, 196, 32]);  bmm_68 = None
    view_632: "f32[8, 4, 196, 196]" = torch.ops.aten.view.default(bmm_69, [8, 4, 196, 196]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    alias_24: "f32[8, 4, 196, 196]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    mul_968: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(view_632, alias_24);  view_632 = None
    sum_131: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_968, [-1], True)
    mul_969: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(alias_24, sum_131);  alias_24 = sum_131 = None
    sub_276: "f32[8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(mul_968, mul_969);  mul_968 = mul_969 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    sum_132: "f32[1, 4, 196, 196]" = torch.ops.aten.sum.dim_IntList(sub_276, [0], True)
    view_633: "f32[4, 196, 196]" = torch.ops.aten.view.default(sum_132, [4, 196, 196]);  sum_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    full_26: "f32[4, 196]" = torch.ops.aten.full.default([4, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    index_put_10: "f32[4, 196]" = torch.ops.aten.index_put.default(full_26, [None, primals_212], view_633, True);  full_26 = primals_212 = view_633 = None
    full_27: "f32[4, 196]" = torch.ops.aten.full.default([4, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_16: "f32[4, 196]" = torch.ops.aten.slice_scatter.default(full_27, index_put_10, 0, 0, 9223372036854775807);  full_27 = index_put_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    mul_970: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(sub_276, 0.25);  sub_276 = None
    view_634: "f32[32, 196, 196]" = torch.ops.aten.view.default(mul_970, [32, 196, 196]);  mul_970 = None
    permute_388: "f32[32, 16, 196]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
    bmm_70: "f32[32, 16, 196]" = torch.ops.aten.bmm.default(permute_388, view_634);  permute_388 = None
    permute_389: "f32[32, 196, 16]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
    bmm_71: "f32[32, 196, 16]" = torch.ops.aten.bmm.default(view_634, permute_389);  view_634 = permute_389 = None
    view_635: "f32[8, 4, 16, 196]" = torch.ops.aten.view.default(bmm_70, [8, 4, 16, 196]);  bmm_70 = None
    view_636: "f32[8, 4, 196, 16]" = torch.ops.aten.view.default(bmm_71, [8, 4, 196, 16]);  bmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_390: "f32[8, 196, 4, 32]" = torch.ops.aten.permute.default(view_631, [0, 2, 1, 3]);  view_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_391: "f32[8, 196, 4, 16]" = torch.ops.aten.permute.default(view_635, [0, 3, 1, 2]);  view_635 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_392: "f32[8, 196, 4, 16]" = torch.ops.aten.permute.default(view_636, [0, 2, 1, 3]);  view_636 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    cat_10: "f32[8, 196, 4, 64]" = torch.ops.aten.cat.default([permute_392, permute_391, permute_390], 3);  permute_392 = permute_391 = permute_390 = None
    view_637: "f32[8, 196, 256]" = torch.ops.aten.view.default(cat_10, [8, 196, 256]);  cat_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_638: "f32[1568, 256]" = torch.ops.aten.view.default(view_637, [1568, 256]);  view_637 = None
    unsqueeze_205: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    sum_133: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_638, [0])
    sub_277: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_75, unsqueeze_205)
    mul_971: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(view_638, sub_277);  sub_277 = None
    sum_134: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_971, [0]);  mul_971 = None
    mul_972: "f32[256]" = torch.ops.aten.mul.Tensor(sum_133, 0.0006377551020408163)
    unsqueeze_206: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_972, 0);  mul_972 = None
    mul_973: "f32[256]" = torch.ops.aten.mul.Tensor(sum_134, 0.0006377551020408163)
    mul_974: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_975: "f32[256]" = torch.ops.aten.mul.Tensor(mul_973, mul_974);  mul_973 = mul_974 = None
    unsqueeze_207: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_975, 0);  mul_975 = None
    mul_976: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_64);  primals_64 = None
    unsqueeze_208: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_976, 0);  mul_976 = None
    sub_278: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_75, unsqueeze_205);  view_75 = unsqueeze_205 = None
    mul_977: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_278, unsqueeze_207);  sub_278 = unsqueeze_207 = None
    sub_279: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_638, mul_977);  view_638 = mul_977 = None
    sub_280: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(sub_279, unsqueeze_206);  sub_279 = unsqueeze_206 = None
    mul_978: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_280, unsqueeze_208);  sub_280 = unsqueeze_208 = None
    mul_979: "f32[256]" = torch.ops.aten.mul.Tensor(sum_134, squeeze_49);  sum_134 = squeeze_49 = None
    view_639: "f32[8, 196, 256]" = torch.ops.aten.view.default(mul_978, [8, 196, 256]);  mul_978 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_640: "f32[1568, 256]" = torch.ops.aten.view.default(view_639, [1568, 256]);  view_639 = None
    permute_393: "f32[256, 1568]" = torch.ops.aten.permute.default(view_640, [1, 0])
    mm_152: "f32[256, 128]" = torch.ops.aten.mm.default(permute_393, view_73);  permute_393 = view_73 = None
    permute_394: "f32[128, 256]" = torch.ops.aten.permute.default(mm_152, [1, 0]);  mm_152 = None
    permute_395: "f32[256, 128]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    mm_153: "f32[1568, 128]" = torch.ops.aten.mm.default(view_640, permute_395);  view_640 = permute_395 = None
    view_641: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_153, [8, 196, 128]);  mm_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_436: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_434, view_641);  add_434 = view_641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_396: "f32[256, 128]" = torch.ops.aten.permute.default(permute_394, [1, 0]);  permute_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_642: "f32[1568, 128]" = torch.ops.aten.view.default(add_436, [1568, 128])
    unsqueeze_209: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    sum_135: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_642, [0])
    sub_281: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_71, unsqueeze_209)
    mul_980: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(view_642, sub_281);  sub_281 = None
    sum_136: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_980, [0]);  mul_980 = None
    mul_981: "f32[128]" = torch.ops.aten.mul.Tensor(sum_135, 0.0006377551020408163)
    unsqueeze_210: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_981, 0);  mul_981 = None
    mul_982: "f32[128]" = torch.ops.aten.mul.Tensor(sum_136, 0.0006377551020408163)
    mul_983: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_984: "f32[128]" = torch.ops.aten.mul.Tensor(mul_982, mul_983);  mul_982 = mul_983 = None
    unsqueeze_211: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_984, 0);  mul_984 = None
    mul_985: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_61);  primals_61 = None
    unsqueeze_212: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_985, 0);  mul_985 = None
    sub_282: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_71, unsqueeze_209);  view_71 = unsqueeze_209 = None
    mul_986: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_282, unsqueeze_211);  sub_282 = unsqueeze_211 = None
    sub_283: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_642, mul_986);  view_642 = mul_986 = None
    sub_284: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(sub_283, unsqueeze_210);  sub_283 = unsqueeze_210 = None
    mul_987: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_212);  sub_284 = unsqueeze_212 = None
    mul_988: "f32[128]" = torch.ops.aten.mul.Tensor(sum_136, squeeze_46);  sum_136 = squeeze_46 = None
    view_643: "f32[8, 196, 128]" = torch.ops.aten.view.default(mul_987, [8, 196, 128]);  mul_987 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_644: "f32[1568, 128]" = torch.ops.aten.view.default(view_643, [1568, 128]);  view_643 = None
    permute_397: "f32[128, 1568]" = torch.ops.aten.permute.default(view_644, [1, 0])
    mm_154: "f32[128, 256]" = torch.ops.aten.mm.default(permute_397, view_69);  permute_397 = view_69 = None
    permute_398: "f32[256, 128]" = torch.ops.aten.permute.default(mm_154, [1, 0]);  mm_154 = None
    permute_399: "f32[128, 256]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    mm_155: "f32[1568, 256]" = torch.ops.aten.mm.default(view_644, permute_399);  view_644 = permute_399 = None
    view_645: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_155, [8, 196, 256]);  mm_155 = None
    permute_400: "f32[128, 256]" = torch.ops.aten.permute.default(permute_398, [1, 0]);  permute_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    lt_22: "b8[8, 196, 256]" = torch.ops.aten.lt.Scalar(view_68, -3)
    le_22: "b8[8, 196, 256]" = torch.ops.aten.le.Scalar(view_68, 3)
    div_70: "f32[8, 196, 256]" = torch.ops.aten.div.Tensor(view_68, 3);  view_68 = None
    add_437: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(div_70, 0.5);  div_70 = None
    mul_989: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_645, add_437);  add_437 = None
    where_44: "f32[8, 196, 256]" = torch.ops.aten.where.self(le_22, mul_989, view_645);  le_22 = mul_989 = view_645 = None
    scalar_tensor_22: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_45: "f32[8, 196, 256]" = torch.ops.aten.where.self(lt_22, scalar_tensor_22, where_44);  lt_22 = scalar_tensor_22 = where_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_646: "f32[1568, 256]" = torch.ops.aten.view.default(where_45, [1568, 256]);  where_45 = None
    unsqueeze_213: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    sum_137: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_646, [0])
    sub_285: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_67, unsqueeze_213)
    mul_990: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(view_646, sub_285);  sub_285 = None
    sum_138: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_990, [0]);  mul_990 = None
    mul_991: "f32[256]" = torch.ops.aten.mul.Tensor(sum_137, 0.0006377551020408163)
    unsqueeze_214: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_991, 0);  mul_991 = None
    mul_992: "f32[256]" = torch.ops.aten.mul.Tensor(sum_138, 0.0006377551020408163)
    mul_993: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_994: "f32[256]" = torch.ops.aten.mul.Tensor(mul_992, mul_993);  mul_992 = mul_993 = None
    unsqueeze_215: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_994, 0);  mul_994 = None
    mul_995: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_58);  primals_58 = None
    unsqueeze_216: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_995, 0);  mul_995 = None
    sub_286: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_67, unsqueeze_213);  view_67 = unsqueeze_213 = None
    mul_996: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_286, unsqueeze_215);  sub_286 = unsqueeze_215 = None
    sub_287: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_646, mul_996);  view_646 = mul_996 = None
    sub_288: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(sub_287, unsqueeze_214);  sub_287 = unsqueeze_214 = None
    mul_997: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_288, unsqueeze_216);  sub_288 = unsqueeze_216 = None
    mul_998: "f32[256]" = torch.ops.aten.mul.Tensor(sum_138, squeeze_43);  sum_138 = squeeze_43 = None
    view_647: "f32[8, 196, 256]" = torch.ops.aten.view.default(mul_997, [8, 196, 256]);  mul_997 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_648: "f32[1568, 256]" = torch.ops.aten.view.default(view_647, [1568, 256]);  view_647 = None
    permute_401: "f32[256, 1568]" = torch.ops.aten.permute.default(view_648, [1, 0])
    mm_156: "f32[256, 128]" = torch.ops.aten.mm.default(permute_401, view_65);  permute_401 = view_65 = None
    permute_402: "f32[128, 256]" = torch.ops.aten.permute.default(mm_156, [1, 0]);  mm_156 = None
    permute_403: "f32[256, 128]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    mm_157: "f32[1568, 128]" = torch.ops.aten.mm.default(view_648, permute_403);  view_648 = permute_403 = None
    view_649: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_157, [8, 196, 128]);  mm_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_438: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_436, view_649);  add_436 = view_649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_404: "f32[256, 128]" = torch.ops.aten.permute.default(permute_402, [1, 0]);  permute_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_650: "f32[1568, 128]" = torch.ops.aten.view.default(add_438, [1568, 128])
    unsqueeze_217: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    sum_139: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_650, [0])
    sub_289: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_63, unsqueeze_217)
    mul_999: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(view_650, sub_289);  sub_289 = None
    sum_140: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_999, [0]);  mul_999 = None
    mul_1000: "f32[128]" = torch.ops.aten.mul.Tensor(sum_139, 0.0006377551020408163)
    unsqueeze_218: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1000, 0);  mul_1000 = None
    mul_1001: "f32[128]" = torch.ops.aten.mul.Tensor(sum_140, 0.0006377551020408163)
    mul_1002: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_1003: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1001, mul_1002);  mul_1001 = mul_1002 = None
    unsqueeze_219: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1003, 0);  mul_1003 = None
    mul_1004: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_55);  primals_55 = None
    unsqueeze_220: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1004, 0);  mul_1004 = None
    sub_290: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_63, unsqueeze_217);  view_63 = unsqueeze_217 = None
    mul_1005: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_290, unsqueeze_219);  sub_290 = unsqueeze_219 = None
    sub_291: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_650, mul_1005);  view_650 = mul_1005 = None
    sub_292: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(sub_291, unsqueeze_218);  sub_291 = unsqueeze_218 = None
    mul_1006: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_292, unsqueeze_220);  sub_292 = unsqueeze_220 = None
    mul_1007: "f32[128]" = torch.ops.aten.mul.Tensor(sum_140, squeeze_40);  sum_140 = squeeze_40 = None
    view_651: "f32[8, 196, 128]" = torch.ops.aten.view.default(mul_1006, [8, 196, 128]);  mul_1006 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_652: "f32[1568, 128]" = torch.ops.aten.view.default(view_651, [1568, 128]);  view_651 = None
    permute_405: "f32[128, 1568]" = torch.ops.aten.permute.default(view_652, [1, 0])
    mm_158: "f32[128, 128]" = torch.ops.aten.mm.default(permute_405, view_61);  permute_405 = view_61 = None
    permute_406: "f32[128, 128]" = torch.ops.aten.permute.default(mm_158, [1, 0]);  mm_158 = None
    permute_407: "f32[128, 128]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    mm_159: "f32[1568, 128]" = torch.ops.aten.mm.default(view_652, permute_407);  view_652 = permute_407 = None
    view_653: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_159, [8, 196, 128]);  mm_159 = None
    permute_408: "f32[128, 128]" = torch.ops.aten.permute.default(permute_406, [1, 0]);  permute_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    lt_23: "b8[8, 196, 128]" = torch.ops.aten.lt.Scalar(view_60, -3)
    le_23: "b8[8, 196, 128]" = torch.ops.aten.le.Scalar(view_60, 3)
    div_71: "f32[8, 196, 128]" = torch.ops.aten.div.Tensor(view_60, 3);  view_60 = None
    add_439: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(div_71, 0.5);  div_71 = None
    mul_1008: "f32[8, 196, 128]" = torch.ops.aten.mul.Tensor(view_653, add_439);  add_439 = None
    where_46: "f32[8, 196, 128]" = torch.ops.aten.where.self(le_23, mul_1008, view_653);  le_23 = mul_1008 = view_653 = None
    scalar_tensor_23: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_47: "f32[8, 196, 128]" = torch.ops.aten.where.self(lt_23, scalar_tensor_23, where_46);  lt_23 = scalar_tensor_23 = where_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    view_654: "f32[8, 196, 4, 32]" = torch.ops.aten.view.default(where_47, [8, 196, 4, 32]);  where_47 = None
    permute_409: "f32[8, 4, 196, 32]" = torch.ops.aten.permute.default(view_654, [0, 2, 1, 3]);  view_654 = None
    clone_96: "f32[8, 4, 196, 32]" = torch.ops.aten.clone.default(permute_409, memory_format = torch.contiguous_format);  permute_409 = None
    view_655: "f32[32, 196, 32]" = torch.ops.aten.view.default(clone_96, [32, 196, 32]);  clone_96 = None
    permute_410: "f32[32, 196, 196]" = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None
    bmm_72: "f32[32, 196, 32]" = torch.ops.aten.bmm.default(permute_410, view_655);  permute_410 = None
    permute_411: "f32[32, 32, 196]" = torch.ops.aten.permute.default(view_58, [0, 2, 1]);  view_58 = None
    bmm_73: "f32[32, 196, 196]" = torch.ops.aten.bmm.default(view_655, permute_411);  view_655 = permute_411 = None
    view_656: "f32[8, 4, 196, 32]" = torch.ops.aten.view.default(bmm_72, [8, 4, 196, 32]);  bmm_72 = None
    view_657: "f32[8, 4, 196, 196]" = torch.ops.aten.view.default(bmm_73, [8, 4, 196, 196]);  bmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    alias_25: "f32[8, 4, 196, 196]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    mul_1009: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(view_657, alias_25);  view_657 = None
    sum_141: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_1009, [-1], True)
    mul_1010: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(alias_25, sum_141);  alias_25 = sum_141 = None
    sub_293: "f32[8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(mul_1009, mul_1010);  mul_1009 = mul_1010 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    sum_142: "f32[1, 4, 196, 196]" = torch.ops.aten.sum.dim_IntList(sub_293, [0], True)
    view_658: "f32[4, 196, 196]" = torch.ops.aten.view.default(sum_142, [4, 196, 196]);  sum_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    full_28: "f32[4, 196]" = torch.ops.aten.full.default([4, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    index_put_11: "f32[4, 196]" = torch.ops.aten.index_put.default(full_28, [None, primals_211], view_658, True);  full_28 = primals_211 = view_658 = None
    full_29: "f32[4, 196]" = torch.ops.aten.full.default([4, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_17: "f32[4, 196]" = torch.ops.aten.slice_scatter.default(full_29, index_put_11, 0, 0, 9223372036854775807);  full_29 = index_put_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    mul_1011: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(sub_293, 0.25);  sub_293 = None
    view_659: "f32[32, 196, 196]" = torch.ops.aten.view.default(mul_1011, [32, 196, 196]);  mul_1011 = None
    permute_412: "f32[32, 16, 196]" = torch.ops.aten.permute.default(view_54, [0, 2, 1]);  view_54 = None
    bmm_74: "f32[32, 16, 196]" = torch.ops.aten.bmm.default(permute_412, view_659);  permute_412 = None
    permute_413: "f32[32, 196, 16]" = torch.ops.aten.permute.default(view_55, [0, 2, 1]);  view_55 = None
    bmm_75: "f32[32, 196, 16]" = torch.ops.aten.bmm.default(view_659, permute_413);  view_659 = permute_413 = None
    view_660: "f32[8, 4, 16, 196]" = torch.ops.aten.view.default(bmm_74, [8, 4, 16, 196]);  bmm_74 = None
    view_661: "f32[8, 4, 196, 16]" = torch.ops.aten.view.default(bmm_75, [8, 4, 196, 16]);  bmm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_414: "f32[8, 196, 4, 32]" = torch.ops.aten.permute.default(view_656, [0, 2, 1, 3]);  view_656 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_415: "f32[8, 196, 4, 16]" = torch.ops.aten.permute.default(view_660, [0, 3, 1, 2]);  view_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_416: "f32[8, 196, 4, 16]" = torch.ops.aten.permute.default(view_661, [0, 2, 1, 3]);  view_661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    cat_11: "f32[8, 196, 4, 64]" = torch.ops.aten.cat.default([permute_416, permute_415, permute_414], 3);  permute_416 = permute_415 = permute_414 = None
    view_662: "f32[8, 196, 256]" = torch.ops.aten.view.default(cat_11, [8, 196, 256]);  cat_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_663: "f32[1568, 256]" = torch.ops.aten.view.default(view_662, [1568, 256]);  view_662 = None
    unsqueeze_221: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    sum_143: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_663, [0])
    sub_294: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_51, unsqueeze_221)
    mul_1012: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(view_663, sub_294);  sub_294 = None
    sum_144: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1012, [0]);  mul_1012 = None
    mul_1013: "f32[256]" = torch.ops.aten.mul.Tensor(sum_143, 0.0006377551020408163)
    unsqueeze_222: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1013, 0);  mul_1013 = None
    mul_1014: "f32[256]" = torch.ops.aten.mul.Tensor(sum_144, 0.0006377551020408163)
    mul_1015: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_1016: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1014, mul_1015);  mul_1014 = mul_1015 = None
    unsqueeze_223: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1016, 0);  mul_1016 = None
    mul_1017: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_52);  primals_52 = None
    unsqueeze_224: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1017, 0);  mul_1017 = None
    sub_295: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_51, unsqueeze_221);  view_51 = unsqueeze_221 = None
    mul_1018: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_295, unsqueeze_223);  sub_295 = unsqueeze_223 = None
    sub_296: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_663, mul_1018);  view_663 = mul_1018 = None
    sub_297: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(sub_296, unsqueeze_222);  sub_296 = unsqueeze_222 = None
    mul_1019: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_297, unsqueeze_224);  sub_297 = unsqueeze_224 = None
    mul_1020: "f32[256]" = torch.ops.aten.mul.Tensor(sum_144, squeeze_37);  sum_144 = squeeze_37 = None
    view_664: "f32[8, 196, 256]" = torch.ops.aten.view.default(mul_1019, [8, 196, 256]);  mul_1019 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_665: "f32[1568, 256]" = torch.ops.aten.view.default(view_664, [1568, 256]);  view_664 = None
    permute_417: "f32[256, 1568]" = torch.ops.aten.permute.default(view_665, [1, 0])
    mm_160: "f32[256, 128]" = torch.ops.aten.mm.default(permute_417, view_49);  permute_417 = view_49 = None
    permute_418: "f32[128, 256]" = torch.ops.aten.permute.default(mm_160, [1, 0]);  mm_160 = None
    permute_419: "f32[256, 128]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    mm_161: "f32[1568, 128]" = torch.ops.aten.mm.default(view_665, permute_419);  view_665 = permute_419 = None
    view_666: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_161, [8, 196, 128]);  mm_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_440: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_438, view_666);  add_438 = view_666 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_420: "f32[256, 128]" = torch.ops.aten.permute.default(permute_418, [1, 0]);  permute_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_667: "f32[1568, 128]" = torch.ops.aten.view.default(add_440, [1568, 128])
    unsqueeze_225: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    sum_145: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_667, [0])
    sub_298: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_47, unsqueeze_225)
    mul_1021: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(view_667, sub_298);  sub_298 = None
    sum_146: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1021, [0]);  mul_1021 = None
    mul_1022: "f32[128]" = torch.ops.aten.mul.Tensor(sum_145, 0.0006377551020408163)
    unsqueeze_226: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1022, 0);  mul_1022 = None
    mul_1023: "f32[128]" = torch.ops.aten.mul.Tensor(sum_146, 0.0006377551020408163)
    mul_1024: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_1025: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1023, mul_1024);  mul_1023 = mul_1024 = None
    unsqueeze_227: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1025, 0);  mul_1025 = None
    mul_1026: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_49);  primals_49 = None
    unsqueeze_228: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1026, 0);  mul_1026 = None
    sub_299: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_47, unsqueeze_225);  view_47 = unsqueeze_225 = None
    mul_1027: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_299, unsqueeze_227);  sub_299 = unsqueeze_227 = None
    sub_300: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_667, mul_1027);  view_667 = mul_1027 = None
    sub_301: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(sub_300, unsqueeze_226);  sub_300 = unsqueeze_226 = None
    mul_1028: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_301, unsqueeze_228);  sub_301 = unsqueeze_228 = None
    mul_1029: "f32[128]" = torch.ops.aten.mul.Tensor(sum_146, squeeze_34);  sum_146 = squeeze_34 = None
    view_668: "f32[8, 196, 128]" = torch.ops.aten.view.default(mul_1028, [8, 196, 128]);  mul_1028 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_669: "f32[1568, 128]" = torch.ops.aten.view.default(view_668, [1568, 128]);  view_668 = None
    permute_421: "f32[128, 1568]" = torch.ops.aten.permute.default(view_669, [1, 0])
    mm_162: "f32[128, 256]" = torch.ops.aten.mm.default(permute_421, view_45);  permute_421 = view_45 = None
    permute_422: "f32[256, 128]" = torch.ops.aten.permute.default(mm_162, [1, 0]);  mm_162 = None
    permute_423: "f32[128, 256]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
    mm_163: "f32[1568, 256]" = torch.ops.aten.mm.default(view_669, permute_423);  view_669 = permute_423 = None
    view_670: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_163, [8, 196, 256]);  mm_163 = None
    permute_424: "f32[128, 256]" = torch.ops.aten.permute.default(permute_422, [1, 0]);  permute_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    lt_24: "b8[8, 196, 256]" = torch.ops.aten.lt.Scalar(view_44, -3)
    le_24: "b8[8, 196, 256]" = torch.ops.aten.le.Scalar(view_44, 3)
    div_72: "f32[8, 196, 256]" = torch.ops.aten.div.Tensor(view_44, 3);  view_44 = None
    add_441: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(div_72, 0.5);  div_72 = None
    mul_1030: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_670, add_441);  add_441 = None
    where_48: "f32[8, 196, 256]" = torch.ops.aten.where.self(le_24, mul_1030, view_670);  le_24 = mul_1030 = view_670 = None
    scalar_tensor_24: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_49: "f32[8, 196, 256]" = torch.ops.aten.where.self(lt_24, scalar_tensor_24, where_48);  lt_24 = scalar_tensor_24 = where_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_671: "f32[1568, 256]" = torch.ops.aten.view.default(where_49, [1568, 256]);  where_49 = None
    unsqueeze_229: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    sum_147: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_671, [0])
    sub_302: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_43, unsqueeze_229)
    mul_1031: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(view_671, sub_302);  sub_302 = None
    sum_148: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1031, [0]);  mul_1031 = None
    mul_1032: "f32[256]" = torch.ops.aten.mul.Tensor(sum_147, 0.0006377551020408163)
    unsqueeze_230: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1032, 0);  mul_1032 = None
    mul_1033: "f32[256]" = torch.ops.aten.mul.Tensor(sum_148, 0.0006377551020408163)
    mul_1034: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_1035: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1033, mul_1034);  mul_1033 = mul_1034 = None
    unsqueeze_231: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1035, 0);  mul_1035 = None
    mul_1036: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_46);  primals_46 = None
    unsqueeze_232: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1036, 0);  mul_1036 = None
    sub_303: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_43, unsqueeze_229);  view_43 = unsqueeze_229 = None
    mul_1037: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_303, unsqueeze_231);  sub_303 = unsqueeze_231 = None
    sub_304: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_671, mul_1037);  view_671 = mul_1037 = None
    sub_305: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(sub_304, unsqueeze_230);  sub_304 = unsqueeze_230 = None
    mul_1038: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_305, unsqueeze_232);  sub_305 = unsqueeze_232 = None
    mul_1039: "f32[256]" = torch.ops.aten.mul.Tensor(sum_148, squeeze_31);  sum_148 = squeeze_31 = None
    view_672: "f32[8, 196, 256]" = torch.ops.aten.view.default(mul_1038, [8, 196, 256]);  mul_1038 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_673: "f32[1568, 256]" = torch.ops.aten.view.default(view_672, [1568, 256]);  view_672 = None
    permute_425: "f32[256, 1568]" = torch.ops.aten.permute.default(view_673, [1, 0])
    mm_164: "f32[256, 128]" = torch.ops.aten.mm.default(permute_425, view_41);  permute_425 = view_41 = None
    permute_426: "f32[128, 256]" = torch.ops.aten.permute.default(mm_164, [1, 0]);  mm_164 = None
    permute_427: "f32[256, 128]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
    mm_165: "f32[1568, 128]" = torch.ops.aten.mm.default(view_673, permute_427);  view_673 = permute_427 = None
    view_674: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_165, [8, 196, 128]);  mm_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_442: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_440, view_674);  add_440 = view_674 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_428: "f32[256, 128]" = torch.ops.aten.permute.default(permute_426, [1, 0]);  permute_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_675: "f32[1568, 128]" = torch.ops.aten.view.default(add_442, [1568, 128])
    unsqueeze_233: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    sum_149: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_675, [0])
    sub_306: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_39, unsqueeze_233)
    mul_1040: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(view_675, sub_306);  sub_306 = None
    sum_150: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1040, [0]);  mul_1040 = None
    mul_1041: "f32[128]" = torch.ops.aten.mul.Tensor(sum_149, 0.0006377551020408163)
    unsqueeze_234: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1041, 0);  mul_1041 = None
    mul_1042: "f32[128]" = torch.ops.aten.mul.Tensor(sum_150, 0.0006377551020408163)
    mul_1043: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_1044: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1042, mul_1043);  mul_1042 = mul_1043 = None
    unsqueeze_235: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1044, 0);  mul_1044 = None
    mul_1045: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_43);  primals_43 = None
    unsqueeze_236: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1045, 0);  mul_1045 = None
    sub_307: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_39, unsqueeze_233);  view_39 = unsqueeze_233 = None
    mul_1046: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_307, unsqueeze_235);  sub_307 = unsqueeze_235 = None
    sub_308: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_675, mul_1046);  view_675 = mul_1046 = None
    sub_309: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(sub_308, unsqueeze_234);  sub_308 = unsqueeze_234 = None
    mul_1047: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_309, unsqueeze_236);  sub_309 = unsqueeze_236 = None
    mul_1048: "f32[128]" = torch.ops.aten.mul.Tensor(sum_150, squeeze_28);  sum_150 = squeeze_28 = None
    view_676: "f32[8, 196, 128]" = torch.ops.aten.view.default(mul_1047, [8, 196, 128]);  mul_1047 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_677: "f32[1568, 128]" = torch.ops.aten.view.default(view_676, [1568, 128]);  view_676 = None
    permute_429: "f32[128, 1568]" = torch.ops.aten.permute.default(view_677, [1, 0])
    mm_166: "f32[128, 128]" = torch.ops.aten.mm.default(permute_429, view_37);  permute_429 = view_37 = None
    permute_430: "f32[128, 128]" = torch.ops.aten.permute.default(mm_166, [1, 0]);  mm_166 = None
    permute_431: "f32[128, 128]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    mm_167: "f32[1568, 128]" = torch.ops.aten.mm.default(view_677, permute_431);  view_677 = permute_431 = None
    view_678: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_167, [8, 196, 128]);  mm_167 = None
    permute_432: "f32[128, 128]" = torch.ops.aten.permute.default(permute_430, [1, 0]);  permute_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    lt_25: "b8[8, 196, 128]" = torch.ops.aten.lt.Scalar(view_36, -3)
    le_25: "b8[8, 196, 128]" = torch.ops.aten.le.Scalar(view_36, 3)
    div_73: "f32[8, 196, 128]" = torch.ops.aten.div.Tensor(view_36, 3);  view_36 = None
    add_443: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(div_73, 0.5);  div_73 = None
    mul_1049: "f32[8, 196, 128]" = torch.ops.aten.mul.Tensor(view_678, add_443);  add_443 = None
    where_50: "f32[8, 196, 128]" = torch.ops.aten.where.self(le_25, mul_1049, view_678);  le_25 = mul_1049 = view_678 = None
    scalar_tensor_25: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_51: "f32[8, 196, 128]" = torch.ops.aten.where.self(lt_25, scalar_tensor_25, where_50);  lt_25 = scalar_tensor_25 = where_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    view_679: "f32[8, 196, 4, 32]" = torch.ops.aten.view.default(where_51, [8, 196, 4, 32]);  where_51 = None
    permute_433: "f32[8, 4, 196, 32]" = torch.ops.aten.permute.default(view_679, [0, 2, 1, 3]);  view_679 = None
    clone_97: "f32[8, 4, 196, 32]" = torch.ops.aten.clone.default(permute_433, memory_format = torch.contiguous_format);  permute_433 = None
    view_680: "f32[32, 196, 32]" = torch.ops.aten.view.default(clone_97, [32, 196, 32]);  clone_97 = None
    permute_434: "f32[32, 196, 196]" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None
    bmm_76: "f32[32, 196, 32]" = torch.ops.aten.bmm.default(permute_434, view_680);  permute_434 = None
    permute_435: "f32[32, 32, 196]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
    bmm_77: "f32[32, 196, 196]" = torch.ops.aten.bmm.default(view_680, permute_435);  view_680 = permute_435 = None
    view_681: "f32[8, 4, 196, 32]" = torch.ops.aten.view.default(bmm_76, [8, 4, 196, 32]);  bmm_76 = None
    view_682: "f32[8, 4, 196, 196]" = torch.ops.aten.view.default(bmm_77, [8, 4, 196, 196]);  bmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    alias_26: "f32[8, 4, 196, 196]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    mul_1050: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(view_682, alias_26);  view_682 = None
    sum_151: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_1050, [-1], True)
    mul_1051: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(alias_26, sum_151);  alias_26 = sum_151 = None
    sub_310: "f32[8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(mul_1050, mul_1051);  mul_1050 = mul_1051 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    sum_152: "f32[1, 4, 196, 196]" = torch.ops.aten.sum.dim_IntList(sub_310, [0], True)
    view_683: "f32[4, 196, 196]" = torch.ops.aten.view.default(sum_152, [4, 196, 196]);  sum_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    full_30: "f32[4, 196]" = torch.ops.aten.full.default([4, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    index_put_12: "f32[4, 196]" = torch.ops.aten.index_put.default(full_30, [None, primals_210], view_683, True);  full_30 = primals_210 = view_683 = None
    full_31: "f32[4, 196]" = torch.ops.aten.full.default([4, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_18: "f32[4, 196]" = torch.ops.aten.slice_scatter.default(full_31, index_put_12, 0, 0, 9223372036854775807);  full_31 = index_put_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    mul_1052: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(sub_310, 0.25);  sub_310 = None
    view_684: "f32[32, 196, 196]" = torch.ops.aten.view.default(mul_1052, [32, 196, 196]);  mul_1052 = None
    permute_436: "f32[32, 16, 196]" = torch.ops.aten.permute.default(view_30, [0, 2, 1]);  view_30 = None
    bmm_78: "f32[32, 16, 196]" = torch.ops.aten.bmm.default(permute_436, view_684);  permute_436 = None
    permute_437: "f32[32, 196, 16]" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
    bmm_79: "f32[32, 196, 16]" = torch.ops.aten.bmm.default(view_684, permute_437);  view_684 = permute_437 = None
    view_685: "f32[8, 4, 16, 196]" = torch.ops.aten.view.default(bmm_78, [8, 4, 16, 196]);  bmm_78 = None
    view_686: "f32[8, 4, 196, 16]" = torch.ops.aten.view.default(bmm_79, [8, 4, 196, 16]);  bmm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_438: "f32[8, 196, 4, 32]" = torch.ops.aten.permute.default(view_681, [0, 2, 1, 3]);  view_681 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_439: "f32[8, 196, 4, 16]" = torch.ops.aten.permute.default(view_685, [0, 3, 1, 2]);  view_685 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_440: "f32[8, 196, 4, 16]" = torch.ops.aten.permute.default(view_686, [0, 2, 1, 3]);  view_686 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    cat_12: "f32[8, 196, 4, 64]" = torch.ops.aten.cat.default([permute_440, permute_439, permute_438], 3);  permute_440 = permute_439 = permute_438 = None
    view_687: "f32[8, 196, 256]" = torch.ops.aten.view.default(cat_12, [8, 196, 256]);  cat_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_688: "f32[1568, 256]" = torch.ops.aten.view.default(view_687, [1568, 256]);  view_687 = None
    unsqueeze_237: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    sum_153: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_688, [0])
    sub_311: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_27, unsqueeze_237)
    mul_1053: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(view_688, sub_311);  sub_311 = None
    sum_154: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1053, [0]);  mul_1053 = None
    mul_1054: "f32[256]" = torch.ops.aten.mul.Tensor(sum_153, 0.0006377551020408163)
    unsqueeze_238: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1054, 0);  mul_1054 = None
    mul_1055: "f32[256]" = torch.ops.aten.mul.Tensor(sum_154, 0.0006377551020408163)
    mul_1056: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_1057: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1055, mul_1056);  mul_1055 = mul_1056 = None
    unsqueeze_239: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1057, 0);  mul_1057 = None
    mul_1058: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_40);  primals_40 = None
    unsqueeze_240: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1058, 0);  mul_1058 = None
    sub_312: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_27, unsqueeze_237);  view_27 = unsqueeze_237 = None
    mul_1059: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_312, unsqueeze_239);  sub_312 = unsqueeze_239 = None
    sub_313: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_688, mul_1059);  view_688 = mul_1059 = None
    sub_314: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(sub_313, unsqueeze_238);  sub_313 = unsqueeze_238 = None
    mul_1060: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_314, unsqueeze_240);  sub_314 = unsqueeze_240 = None
    mul_1061: "f32[256]" = torch.ops.aten.mul.Tensor(sum_154, squeeze_25);  sum_154 = squeeze_25 = None
    view_689: "f32[8, 196, 256]" = torch.ops.aten.view.default(mul_1060, [8, 196, 256]);  mul_1060 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_690: "f32[1568, 256]" = torch.ops.aten.view.default(view_689, [1568, 256]);  view_689 = None
    permute_441: "f32[256, 1568]" = torch.ops.aten.permute.default(view_690, [1, 0])
    mm_168: "f32[256, 128]" = torch.ops.aten.mm.default(permute_441, view_25);  permute_441 = view_25 = None
    permute_442: "f32[128, 256]" = torch.ops.aten.permute.default(mm_168, [1, 0]);  mm_168 = None
    permute_443: "f32[256, 128]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    mm_169: "f32[1568, 128]" = torch.ops.aten.mm.default(view_690, permute_443);  view_690 = permute_443 = None
    view_691: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_169, [8, 196, 128]);  mm_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_444: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_442, view_691);  add_442 = view_691 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_444: "f32[256, 128]" = torch.ops.aten.permute.default(permute_442, [1, 0]);  permute_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_692: "f32[1568, 128]" = torch.ops.aten.view.default(add_444, [1568, 128])
    unsqueeze_241: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    sum_155: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_692, [0])
    sub_315: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_23, unsqueeze_241)
    mul_1062: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(view_692, sub_315);  sub_315 = None
    sum_156: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1062, [0]);  mul_1062 = None
    mul_1063: "f32[128]" = torch.ops.aten.mul.Tensor(sum_155, 0.0006377551020408163)
    unsqueeze_242: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1063, 0);  mul_1063 = None
    mul_1064: "f32[128]" = torch.ops.aten.mul.Tensor(sum_156, 0.0006377551020408163)
    mul_1065: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_1066: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1064, mul_1065);  mul_1064 = mul_1065 = None
    unsqueeze_243: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1066, 0);  mul_1066 = None
    mul_1067: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_37);  primals_37 = None
    unsqueeze_244: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1067, 0);  mul_1067 = None
    sub_316: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_23, unsqueeze_241);  view_23 = unsqueeze_241 = None
    mul_1068: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_316, unsqueeze_243);  sub_316 = unsqueeze_243 = None
    sub_317: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_692, mul_1068);  view_692 = mul_1068 = None
    sub_318: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(sub_317, unsqueeze_242);  sub_317 = unsqueeze_242 = None
    mul_1069: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_318, unsqueeze_244);  sub_318 = unsqueeze_244 = None
    mul_1070: "f32[128]" = torch.ops.aten.mul.Tensor(sum_156, squeeze_22);  sum_156 = squeeze_22 = None
    view_693: "f32[8, 196, 128]" = torch.ops.aten.view.default(mul_1069, [8, 196, 128]);  mul_1069 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_694: "f32[1568, 128]" = torch.ops.aten.view.default(view_693, [1568, 128]);  view_693 = None
    permute_445: "f32[128, 1568]" = torch.ops.aten.permute.default(view_694, [1, 0])
    mm_170: "f32[128, 256]" = torch.ops.aten.mm.default(permute_445, view_21);  permute_445 = view_21 = None
    permute_446: "f32[256, 128]" = torch.ops.aten.permute.default(mm_170, [1, 0]);  mm_170 = None
    permute_447: "f32[128, 256]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    mm_171: "f32[1568, 256]" = torch.ops.aten.mm.default(view_694, permute_447);  view_694 = permute_447 = None
    view_695: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_171, [8, 196, 256]);  mm_171 = None
    permute_448: "f32[128, 256]" = torch.ops.aten.permute.default(permute_446, [1, 0]);  permute_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    lt_26: "b8[8, 196, 256]" = torch.ops.aten.lt.Scalar(view_20, -3)
    le_26: "b8[8, 196, 256]" = torch.ops.aten.le.Scalar(view_20, 3)
    div_74: "f32[8, 196, 256]" = torch.ops.aten.div.Tensor(view_20, 3);  view_20 = None
    add_445: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(div_74, 0.5);  div_74 = None
    mul_1071: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_695, add_445);  add_445 = None
    where_52: "f32[8, 196, 256]" = torch.ops.aten.where.self(le_26, mul_1071, view_695);  le_26 = mul_1071 = view_695 = None
    scalar_tensor_26: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_53: "f32[8, 196, 256]" = torch.ops.aten.where.self(lt_26, scalar_tensor_26, where_52);  lt_26 = scalar_tensor_26 = where_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_696: "f32[1568, 256]" = torch.ops.aten.view.default(where_53, [1568, 256]);  where_53 = None
    unsqueeze_245: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    sum_157: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_696, [0])
    sub_319: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_19, unsqueeze_245)
    mul_1072: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(view_696, sub_319);  sub_319 = None
    sum_158: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1072, [0]);  mul_1072 = None
    mul_1073: "f32[256]" = torch.ops.aten.mul.Tensor(sum_157, 0.0006377551020408163)
    unsqueeze_246: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1073, 0);  mul_1073 = None
    mul_1074: "f32[256]" = torch.ops.aten.mul.Tensor(sum_158, 0.0006377551020408163)
    mul_1075: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_1076: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1074, mul_1075);  mul_1074 = mul_1075 = None
    unsqueeze_247: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1076, 0);  mul_1076 = None
    mul_1077: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_34);  primals_34 = None
    unsqueeze_248: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1077, 0);  mul_1077 = None
    sub_320: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_19, unsqueeze_245);  view_19 = unsqueeze_245 = None
    mul_1078: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_320, unsqueeze_247);  sub_320 = unsqueeze_247 = None
    sub_321: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_696, mul_1078);  view_696 = mul_1078 = None
    sub_322: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(sub_321, unsqueeze_246);  sub_321 = unsqueeze_246 = None
    mul_1079: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_322, unsqueeze_248);  sub_322 = unsqueeze_248 = None
    mul_1080: "f32[256]" = torch.ops.aten.mul.Tensor(sum_158, squeeze_19);  sum_158 = squeeze_19 = None
    view_697: "f32[8, 196, 256]" = torch.ops.aten.view.default(mul_1079, [8, 196, 256]);  mul_1079 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_698: "f32[1568, 256]" = torch.ops.aten.view.default(view_697, [1568, 256]);  view_697 = None
    permute_449: "f32[256, 1568]" = torch.ops.aten.permute.default(view_698, [1, 0])
    mm_172: "f32[256, 128]" = torch.ops.aten.mm.default(permute_449, view_17);  permute_449 = view_17 = None
    permute_450: "f32[128, 256]" = torch.ops.aten.permute.default(mm_172, [1, 0]);  mm_172 = None
    permute_451: "f32[256, 128]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
    mm_173: "f32[1568, 128]" = torch.ops.aten.mm.default(view_698, permute_451);  view_698 = permute_451 = None
    view_699: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_173, [8, 196, 128]);  mm_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_446: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_444, view_699);  add_444 = view_699 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_452: "f32[256, 128]" = torch.ops.aten.permute.default(permute_450, [1, 0]);  permute_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_700: "f32[1568, 128]" = torch.ops.aten.view.default(add_446, [1568, 128])
    unsqueeze_249: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    sum_159: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_700, [0])
    sub_323: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_15, unsqueeze_249)
    mul_1081: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(view_700, sub_323);  sub_323 = None
    sum_160: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1081, [0]);  mul_1081 = None
    mul_1082: "f32[128]" = torch.ops.aten.mul.Tensor(sum_159, 0.0006377551020408163)
    unsqueeze_250: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1082, 0);  mul_1082 = None
    mul_1083: "f32[128]" = torch.ops.aten.mul.Tensor(sum_160, 0.0006377551020408163)
    mul_1084: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_1085: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1083, mul_1084);  mul_1083 = mul_1084 = None
    unsqueeze_251: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1085, 0);  mul_1085 = None
    mul_1086: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_31);  primals_31 = None
    unsqueeze_252: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1086, 0);  mul_1086 = None
    sub_324: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_15, unsqueeze_249);  view_15 = unsqueeze_249 = None
    mul_1087: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_324, unsqueeze_251);  sub_324 = unsqueeze_251 = None
    sub_325: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_700, mul_1087);  view_700 = mul_1087 = None
    sub_326: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(sub_325, unsqueeze_250);  sub_325 = unsqueeze_250 = None
    mul_1088: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_326, unsqueeze_252);  sub_326 = unsqueeze_252 = None
    mul_1089: "f32[128]" = torch.ops.aten.mul.Tensor(sum_160, squeeze_16);  sum_160 = squeeze_16 = None
    view_701: "f32[8, 196, 128]" = torch.ops.aten.view.default(mul_1088, [8, 196, 128]);  mul_1088 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_702: "f32[1568, 128]" = torch.ops.aten.view.default(view_701, [1568, 128]);  view_701 = None
    permute_453: "f32[128, 1568]" = torch.ops.aten.permute.default(view_702, [1, 0])
    mm_174: "f32[128, 128]" = torch.ops.aten.mm.default(permute_453, view_13);  permute_453 = view_13 = None
    permute_454: "f32[128, 128]" = torch.ops.aten.permute.default(mm_174, [1, 0]);  mm_174 = None
    permute_455: "f32[128, 128]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
    mm_175: "f32[1568, 128]" = torch.ops.aten.mm.default(view_702, permute_455);  view_702 = permute_455 = None
    view_703: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_175, [8, 196, 128]);  mm_175 = None
    permute_456: "f32[128, 128]" = torch.ops.aten.permute.default(permute_454, [1, 0]);  permute_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    lt_27: "b8[8, 196, 128]" = torch.ops.aten.lt.Scalar(view_12, -3)
    le_27: "b8[8, 196, 128]" = torch.ops.aten.le.Scalar(view_12, 3)
    div_75: "f32[8, 196, 128]" = torch.ops.aten.div.Tensor(view_12, 3);  view_12 = None
    add_447: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(div_75, 0.5);  div_75 = None
    mul_1090: "f32[8, 196, 128]" = torch.ops.aten.mul.Tensor(view_703, add_447);  add_447 = None
    where_54: "f32[8, 196, 128]" = torch.ops.aten.where.self(le_27, mul_1090, view_703);  le_27 = mul_1090 = view_703 = None
    scalar_tensor_27: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_55: "f32[8, 196, 128]" = torch.ops.aten.where.self(lt_27, scalar_tensor_27, where_54);  lt_27 = scalar_tensor_27 = where_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    view_704: "f32[8, 196, 4, 32]" = torch.ops.aten.view.default(where_55, [8, 196, 4, 32]);  where_55 = None
    permute_457: "f32[8, 4, 196, 32]" = torch.ops.aten.permute.default(view_704, [0, 2, 1, 3]);  view_704 = None
    clone_98: "f32[8, 4, 196, 32]" = torch.ops.aten.clone.default(permute_457, memory_format = torch.contiguous_format);  permute_457 = None
    view_705: "f32[32, 196, 32]" = torch.ops.aten.view.default(clone_98, [32, 196, 32]);  clone_98 = None
    permute_458: "f32[32, 196, 196]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
    bmm_80: "f32[32, 196, 32]" = torch.ops.aten.bmm.default(permute_458, view_705);  permute_458 = None
    permute_459: "f32[32, 32, 196]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    bmm_81: "f32[32, 196, 196]" = torch.ops.aten.bmm.default(view_705, permute_459);  view_705 = permute_459 = None
    view_706: "f32[8, 4, 196, 32]" = torch.ops.aten.view.default(bmm_80, [8, 4, 196, 32]);  bmm_80 = None
    view_707: "f32[8, 4, 196, 196]" = torch.ops.aten.view.default(bmm_81, [8, 4, 196, 196]);  bmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    alias_27: "f32[8, 4, 196, 196]" = torch.ops.aten.alias.default(alias);  alias = None
    mul_1091: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(view_707, alias_27);  view_707 = None
    sum_161: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_1091, [-1], True)
    mul_1092: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(alias_27, sum_161);  alias_27 = sum_161 = None
    sub_327: "f32[8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(mul_1091, mul_1092);  mul_1091 = mul_1092 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    sum_162: "f32[1, 4, 196, 196]" = torch.ops.aten.sum.dim_IntList(sub_327, [0], True)
    view_708: "f32[4, 196, 196]" = torch.ops.aten.view.default(sum_162, [4, 196, 196]);  sum_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    full_32: "f32[4, 196]" = torch.ops.aten.full.default([4, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    index_put_13: "f32[4, 196]" = torch.ops.aten.index_put.default(full_32, [None, primals_209], view_708, True);  full_32 = primals_209 = view_708 = None
    full_33: "f32[4, 196]" = torch.ops.aten.full.default([4, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_19: "f32[4, 196]" = torch.ops.aten.slice_scatter.default(full_33, index_put_13, 0, 0, 9223372036854775807);  full_33 = index_put_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    mul_1093: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(sub_327, 0.25);  sub_327 = None
    view_709: "f32[32, 196, 196]" = torch.ops.aten.view.default(mul_1093, [32, 196, 196]);  mul_1093 = None
    permute_460: "f32[32, 16, 196]" = torch.ops.aten.permute.default(view_6, [0, 2, 1]);  view_6 = None
    bmm_82: "f32[32, 16, 196]" = torch.ops.aten.bmm.default(permute_460, view_709);  permute_460 = None
    permute_461: "f32[32, 196, 16]" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
    bmm_83: "f32[32, 196, 16]" = torch.ops.aten.bmm.default(view_709, permute_461);  view_709 = permute_461 = None
    view_710: "f32[8, 4, 16, 196]" = torch.ops.aten.view.default(bmm_82, [8, 4, 16, 196]);  bmm_82 = None
    view_711: "f32[8, 4, 196, 16]" = torch.ops.aten.view.default(bmm_83, [8, 4, 196, 16]);  bmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_462: "f32[8, 196, 4, 32]" = torch.ops.aten.permute.default(view_706, [0, 2, 1, 3]);  view_706 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_463: "f32[8, 196, 4, 16]" = torch.ops.aten.permute.default(view_710, [0, 3, 1, 2]);  view_710 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_464: "f32[8, 196, 4, 16]" = torch.ops.aten.permute.default(view_711, [0, 2, 1, 3]);  view_711 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    cat_13: "f32[8, 196, 4, 64]" = torch.ops.aten.cat.default([permute_464, permute_463, permute_462], 3);  permute_464 = permute_463 = permute_462 = None
    view_712: "f32[8, 196, 256]" = torch.ops.aten.view.default(cat_13, [8, 196, 256]);  cat_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_713: "f32[1568, 256]" = torch.ops.aten.view.default(view_712, [1568, 256]);  view_712 = None
    unsqueeze_253: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    sum_163: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_713, [0])
    sub_328: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_3, unsqueeze_253)
    mul_1094: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(view_713, sub_328);  sub_328 = None
    sum_164: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1094, [0]);  mul_1094 = None
    mul_1095: "f32[256]" = torch.ops.aten.mul.Tensor(sum_163, 0.0006377551020408163)
    unsqueeze_254: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1095, 0);  mul_1095 = None
    mul_1096: "f32[256]" = torch.ops.aten.mul.Tensor(sum_164, 0.0006377551020408163)
    mul_1097: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_1098: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1096, mul_1097);  mul_1096 = mul_1097 = None
    unsqueeze_255: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1098, 0);  mul_1098 = None
    mul_1099: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_28);  primals_28 = None
    unsqueeze_256: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1099, 0);  mul_1099 = None
    sub_329: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_3, unsqueeze_253);  view_3 = unsqueeze_253 = None
    mul_1100: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_329, unsqueeze_255);  sub_329 = unsqueeze_255 = None
    sub_330: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_713, mul_1100);  view_713 = mul_1100 = None
    sub_331: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(sub_330, unsqueeze_254);  sub_330 = unsqueeze_254 = None
    mul_1101: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_331, unsqueeze_256);  sub_331 = unsqueeze_256 = None
    mul_1102: "f32[256]" = torch.ops.aten.mul.Tensor(sum_164, squeeze_13);  sum_164 = squeeze_13 = None
    view_714: "f32[8, 196, 256]" = torch.ops.aten.view.default(mul_1101, [8, 196, 256]);  mul_1101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    view_715: "f32[1568, 256]" = torch.ops.aten.view.default(view_714, [1568, 256]);  view_714 = None
    permute_465: "f32[256, 1568]" = torch.ops.aten.permute.default(view_715, [1, 0])
    mm_176: "f32[256, 128]" = torch.ops.aten.mm.default(permute_465, view_1);  permute_465 = view_1 = None
    permute_466: "f32[128, 256]" = torch.ops.aten.permute.default(mm_176, [1, 0]);  mm_176 = None
    permute_467: "f32[256, 128]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    mm_177: "f32[1568, 128]" = torch.ops.aten.mm.default(view_715, permute_467);  view_715 = permute_467 = None
    view_716: "f32[8, 196, 128]" = torch.ops.aten.view.default(mm_177, [8, 196, 128]);  mm_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    add_448: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_446, view_716);  add_446 = view_716 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_468: "f32[256, 128]" = torch.ops.aten.permute.default(permute_466, [1, 0]);  permute_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:639, code: x = x.flatten(2).transpose(1, 2)
    permute_469: "f32[8, 128, 196]" = torch.ops.aten.permute.default(add_448, [0, 2, 1]);  add_448 = None
    view_717: "f32[8, 128, 14, 14]" = torch.ops.aten.view.default(permute_469, [8, 128, 14, 14]);  permute_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:65, code: return self.bn(self.linear(x))
    unsqueeze_257: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_258: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 2);  unsqueeze_257 = None
    unsqueeze_259: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, 3);  unsqueeze_258 = None
    sum_165: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_717, [0, 2, 3])
    sub_332: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_259)
    mul_1103: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(view_717, sub_332);  sub_332 = None
    sum_166: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1103, [0, 2, 3]);  mul_1103 = None
    mul_1104: "f32[128]" = torch.ops.aten.mul.Tensor(sum_165, 0.0006377551020408163)
    unsqueeze_260: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1104, 0);  mul_1104 = None
    unsqueeze_261: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 2);  unsqueeze_260 = None
    unsqueeze_262: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 3);  unsqueeze_261 = None
    mul_1105: "f32[128]" = torch.ops.aten.mul.Tensor(sum_166, 0.0006377551020408163)
    mul_1106: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_1107: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1105, mul_1106);  mul_1105 = mul_1106 = None
    unsqueeze_263: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1107, 0);  mul_1107 = None
    unsqueeze_264: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 2);  unsqueeze_263 = None
    unsqueeze_265: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 3);  unsqueeze_264 = None
    mul_1108: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_25);  primals_25 = None
    unsqueeze_266: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1108, 0);  mul_1108 = None
    unsqueeze_267: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 2);  unsqueeze_266 = None
    unsqueeze_268: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 3);  unsqueeze_267 = None
    sub_333: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_259);  convolution_3 = unsqueeze_259 = None
    mul_1109: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_333, unsqueeze_265);  sub_333 = unsqueeze_265 = None
    sub_334: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(view_717, mul_1109);  view_717 = mul_1109 = None
    sub_335: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_334, unsqueeze_262);  sub_334 = unsqueeze_262 = None
    mul_1110: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_335, unsqueeze_268);  sub_335 = unsqueeze_268 = None
    mul_1111: "f32[128]" = torch.ops.aten.mul.Tensor(sum_166, squeeze_10);  sum_166 = squeeze_10 = None
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_1110, div_2, primals_24, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1110 = div_2 = primals_24 = None
    getitem_168: "f32[8, 64, 28, 28]" = convolution_backward[0]
    getitem_169: "f32[128, 64, 3, 3]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:637, code: x = self.stem(x)
    lt_28: "b8[8, 64, 28, 28]" = torch.ops.aten.lt.Scalar(add_16, -3)
    le_28: "b8[8, 64, 28, 28]" = torch.ops.aten.le.Scalar(add_16, 3)
    div_76: "f32[8, 64, 28, 28]" = torch.ops.aten.div.Tensor(add_16, 3);  add_16 = None
    add_449: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(div_76, 0.5);  div_76 = None
    mul_1112: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_168, add_449);  add_449 = None
    where_56: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(le_28, mul_1112, getitem_168);  le_28 = mul_1112 = getitem_168 = None
    scalar_tensor_28: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_57: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(lt_28, scalar_tensor_28, where_56);  lt_28 = scalar_tensor_28 = where_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:65, code: return self.bn(self.linear(x))
    unsqueeze_269: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_270: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 2);  unsqueeze_269 = None
    unsqueeze_271: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 3);  unsqueeze_270 = None
    sum_167: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_57, [0, 2, 3])
    sub_336: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_271)
    mul_1113: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(where_57, sub_336);  sub_336 = None
    sum_168: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1113, [0, 2, 3]);  mul_1113 = None
    mul_1114: "f32[64]" = torch.ops.aten.mul.Tensor(sum_167, 0.00015943877551020407)
    unsqueeze_272: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1114, 0);  mul_1114 = None
    unsqueeze_273: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 2);  unsqueeze_272 = None
    unsqueeze_274: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 3);  unsqueeze_273 = None
    mul_1115: "f32[64]" = torch.ops.aten.mul.Tensor(sum_168, 0.00015943877551020407)
    mul_1116: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_1117: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1115, mul_1116);  mul_1115 = mul_1116 = None
    unsqueeze_275: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1117, 0);  mul_1117 = None
    unsqueeze_276: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 2);  unsqueeze_275 = None
    unsqueeze_277: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 3);  unsqueeze_276 = None
    mul_1118: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_22);  primals_22 = None
    unsqueeze_278: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1118, 0);  mul_1118 = None
    unsqueeze_279: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 2);  unsqueeze_278 = None
    unsqueeze_280: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 3);  unsqueeze_279 = None
    sub_337: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_271);  convolution_2 = unsqueeze_271 = None
    mul_1119: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_337, unsqueeze_277);  sub_337 = unsqueeze_277 = None
    sub_338: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(where_57, mul_1119);  where_57 = mul_1119 = None
    sub_339: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(sub_338, unsqueeze_274);  sub_338 = unsqueeze_274 = None
    mul_1120: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_339, unsqueeze_280);  sub_339 = unsqueeze_280 = None
    mul_1121: "f32[64]" = torch.ops.aten.mul.Tensor(sum_168, squeeze_7);  sum_168 = squeeze_7 = None
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_1120, div_1, primals_21, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1120 = div_1 = primals_21 = None
    getitem_171: "f32[8, 32, 56, 56]" = convolution_backward_1[0]
    getitem_172: "f32[64, 32, 3, 3]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:637, code: x = self.stem(x)
    lt_29: "b8[8, 32, 56, 56]" = torch.ops.aten.lt.Scalar(add_10, -3)
    le_29: "b8[8, 32, 56, 56]" = torch.ops.aten.le.Scalar(add_10, 3)
    div_77: "f32[8, 32, 56, 56]" = torch.ops.aten.div.Tensor(add_10, 3);  add_10 = None
    add_450: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(div_77, 0.5);  div_77 = None
    mul_1122: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_171, add_450);  add_450 = None
    where_58: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_29, mul_1122, getitem_171);  le_29 = mul_1122 = getitem_171 = None
    scalar_tensor_29: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_59: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(lt_29, scalar_tensor_29, where_58);  lt_29 = scalar_tensor_29 = where_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:65, code: return self.bn(self.linear(x))
    unsqueeze_281: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_282: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 2);  unsqueeze_281 = None
    unsqueeze_283: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 3);  unsqueeze_282 = None
    sum_169: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_59, [0, 2, 3])
    sub_340: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_283)
    mul_1123: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_59, sub_340);  sub_340 = None
    sum_170: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1123, [0, 2, 3]);  mul_1123 = None
    mul_1124: "f32[32]" = torch.ops.aten.mul.Tensor(sum_169, 3.985969387755102e-05)
    unsqueeze_284: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1124, 0);  mul_1124 = None
    unsqueeze_285: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 2);  unsqueeze_284 = None
    unsqueeze_286: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 3);  unsqueeze_285 = None
    mul_1125: "f32[32]" = torch.ops.aten.mul.Tensor(sum_170, 3.985969387755102e-05)
    mul_1126: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_1127: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1125, mul_1126);  mul_1125 = mul_1126 = None
    unsqueeze_287: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1127, 0);  mul_1127 = None
    unsqueeze_288: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 2);  unsqueeze_287 = None
    unsqueeze_289: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 3);  unsqueeze_288 = None
    mul_1128: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_19);  primals_19 = None
    unsqueeze_290: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1128, 0);  mul_1128 = None
    unsqueeze_291: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 2);  unsqueeze_290 = None
    unsqueeze_292: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 3);  unsqueeze_291 = None
    sub_341: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_283);  convolution_1 = unsqueeze_283 = None
    mul_1129: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_341, unsqueeze_289);  sub_341 = unsqueeze_289 = None
    sub_342: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(where_59, mul_1129);  where_59 = mul_1129 = None
    sub_343: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(sub_342, unsqueeze_286);  sub_342 = unsqueeze_286 = None
    mul_1130: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_343, unsqueeze_292);  sub_343 = unsqueeze_292 = None
    mul_1131: "f32[32]" = torch.ops.aten.mul.Tensor(sum_170, squeeze_4);  sum_170 = squeeze_4 = None
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_1130, div, primals_18, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1130 = div = primals_18 = None
    getitem_174: "f32[8, 16, 112, 112]" = convolution_backward_2[0]
    getitem_175: "f32[32, 16, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:637, code: x = self.stem(x)
    lt_30: "b8[8, 16, 112, 112]" = torch.ops.aten.lt.Scalar(add_4, -3)
    le_30: "b8[8, 16, 112, 112]" = torch.ops.aten.le.Scalar(add_4, 3)
    div_78: "f32[8, 16, 112, 112]" = torch.ops.aten.div.Tensor(add_4, 3);  add_4 = None
    add_451: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(div_78, 0.5);  div_78 = None
    mul_1132: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_174, add_451);  add_451 = None
    where_60: "f32[8, 16, 112, 112]" = torch.ops.aten.where.self(le_30, mul_1132, getitem_174);  le_30 = mul_1132 = getitem_174 = None
    scalar_tensor_30: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_61: "f32[8, 16, 112, 112]" = torch.ops.aten.where.self(lt_30, scalar_tensor_30, where_60);  lt_30 = scalar_tensor_30 = where_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:65, code: return self.bn(self.linear(x))
    unsqueeze_293: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_294: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 2);  unsqueeze_293 = None
    unsqueeze_295: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 3);  unsqueeze_294 = None
    sum_171: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_61, [0, 2, 3])
    sub_344: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_295)
    mul_1133: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(where_61, sub_344);  sub_344 = None
    sum_172: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1133, [0, 2, 3]);  mul_1133 = None
    mul_1134: "f32[16]" = torch.ops.aten.mul.Tensor(sum_171, 9.964923469387754e-06)
    unsqueeze_296: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1134, 0);  mul_1134 = None
    unsqueeze_297: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 2);  unsqueeze_296 = None
    unsqueeze_298: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 3);  unsqueeze_297 = None
    mul_1135: "f32[16]" = torch.ops.aten.mul.Tensor(sum_172, 9.964923469387754e-06)
    mul_1136: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_1137: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1135, mul_1136);  mul_1135 = mul_1136 = None
    unsqueeze_299: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1137, 0);  mul_1137 = None
    unsqueeze_300: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 2);  unsqueeze_299 = None
    unsqueeze_301: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 3);  unsqueeze_300 = None
    mul_1138: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_16);  primals_16 = None
    unsqueeze_302: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1138, 0);  mul_1138 = None
    unsqueeze_303: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 2);  unsqueeze_302 = None
    unsqueeze_304: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 3);  unsqueeze_303 = None
    sub_345: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_295);  convolution = unsqueeze_295 = None
    mul_1139: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_345, unsqueeze_301);  sub_345 = unsqueeze_301 = None
    sub_346: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(where_61, mul_1139);  where_61 = mul_1139 = None
    sub_347: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_346, unsqueeze_298);  sub_346 = unsqueeze_298 = None
    mul_1140: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_347, unsqueeze_304);  sub_347 = unsqueeze_304 = None
    mul_1141: "f32[16]" = torch.ops.aten.mul.Tensor(sum_172, squeeze_1);  sum_172 = squeeze_1 = None
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_1140, primals_415, primals_15, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1140 = primals_415 = primals_15 = None
    getitem_178: "f32[16, 3, 3, 3]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # No stacktrace found for following nodes
    copy_: "f32[16]" = torch.ops.aten.copy_.default(primals_223, add_2);  primals_223 = add_2 = None
    copy__1: "f32[16]" = torch.ops.aten.copy_.default(primals_224, add_3);  primals_224 = add_3 = None
    copy__2: "i64[]" = torch.ops.aten.copy_.default(primals_225, add);  primals_225 = add = None
    copy__3: "f32[32]" = torch.ops.aten.copy_.default(primals_226, add_8);  primals_226 = add_8 = None
    copy__4: "f32[32]" = torch.ops.aten.copy_.default(primals_227, add_9);  primals_227 = add_9 = None
    copy__5: "i64[]" = torch.ops.aten.copy_.default(primals_228, add_6);  primals_228 = add_6 = None
    copy__6: "f32[64]" = torch.ops.aten.copy_.default(primals_229, add_14);  primals_229 = add_14 = None
    copy__7: "f32[64]" = torch.ops.aten.copy_.default(primals_230, add_15);  primals_230 = add_15 = None
    copy__8: "i64[]" = torch.ops.aten.copy_.default(primals_231, add_12);  primals_231 = add_12 = None
    copy__9: "f32[128]" = torch.ops.aten.copy_.default(primals_232, add_20);  primals_232 = add_20 = None
    copy__10: "f32[128]" = torch.ops.aten.copy_.default(primals_233, add_21);  primals_233 = add_21 = None
    copy__11: "i64[]" = torch.ops.aten.copy_.default(primals_234, add_18);  primals_234 = add_18 = None
    copy__12: "f32[256]" = torch.ops.aten.copy_.default(primals_235, add_25);  primals_235 = add_25 = None
    copy__13: "f32[256]" = torch.ops.aten.copy_.default(primals_236, add_26);  primals_236 = add_26 = None
    copy__14: "i64[]" = torch.ops.aten.copy_.default(primals_237, add_23);  primals_237 = add_23 = None
    copy__15: "f32[128]" = torch.ops.aten.copy_.default(primals_238, add_32);  primals_238 = add_32 = None
    copy__16: "f32[128]" = torch.ops.aten.copy_.default(primals_239, add_33);  primals_239 = add_33 = None
    copy__17: "i64[]" = torch.ops.aten.copy_.default(primals_240, add_30);  primals_240 = add_30 = None
    copy__18: "f32[256]" = torch.ops.aten.copy_.default(primals_241, add_38);  primals_241 = add_38 = None
    copy__19: "f32[256]" = torch.ops.aten.copy_.default(primals_242, add_39);  primals_242 = add_39 = None
    copy__20: "i64[]" = torch.ops.aten.copy_.default(primals_243, add_36);  primals_243 = add_36 = None
    copy__21: "f32[128]" = torch.ops.aten.copy_.default(primals_244, add_44);  primals_244 = add_44 = None
    copy__22: "f32[128]" = torch.ops.aten.copy_.default(primals_245, add_45);  primals_245 = add_45 = None
    copy__23: "i64[]" = torch.ops.aten.copy_.default(primals_246, add_42);  primals_246 = add_42 = None
    copy__24: "f32[256]" = torch.ops.aten.copy_.default(primals_247, add_50);  primals_247 = add_50 = None
    copy__25: "f32[256]" = torch.ops.aten.copy_.default(primals_248, add_51);  primals_248 = add_51 = None
    copy__26: "i64[]" = torch.ops.aten.copy_.default(primals_249, add_48);  primals_249 = add_48 = None
    copy__27: "f32[128]" = torch.ops.aten.copy_.default(primals_250, add_57);  primals_250 = add_57 = None
    copy__28: "f32[128]" = torch.ops.aten.copy_.default(primals_251, add_58);  primals_251 = add_58 = None
    copy__29: "i64[]" = torch.ops.aten.copy_.default(primals_252, add_55);  primals_252 = add_55 = None
    copy__30: "f32[256]" = torch.ops.aten.copy_.default(primals_253, add_63);  primals_253 = add_63 = None
    copy__31: "f32[256]" = torch.ops.aten.copy_.default(primals_254, add_64);  primals_254 = add_64 = None
    copy__32: "i64[]" = torch.ops.aten.copy_.default(primals_255, add_61);  primals_255 = add_61 = None
    copy__33: "f32[128]" = torch.ops.aten.copy_.default(primals_256, add_69);  primals_256 = add_69 = None
    copy__34: "f32[128]" = torch.ops.aten.copy_.default(primals_257, add_70);  primals_257 = add_70 = None
    copy__35: "i64[]" = torch.ops.aten.copy_.default(primals_258, add_67);  primals_258 = add_67 = None
    copy__36: "f32[256]" = torch.ops.aten.copy_.default(primals_259, add_75);  primals_259 = add_75 = None
    copy__37: "f32[256]" = torch.ops.aten.copy_.default(primals_260, add_76);  primals_260 = add_76 = None
    copy__38: "i64[]" = torch.ops.aten.copy_.default(primals_261, add_73);  primals_261 = add_73 = None
    copy__39: "f32[128]" = torch.ops.aten.copy_.default(primals_262, add_82);  primals_262 = add_82 = None
    copy__40: "f32[128]" = torch.ops.aten.copy_.default(primals_263, add_83);  primals_263 = add_83 = None
    copy__41: "i64[]" = torch.ops.aten.copy_.default(primals_264, add_80);  primals_264 = add_80 = None
    copy__42: "f32[256]" = torch.ops.aten.copy_.default(primals_265, add_88);  primals_265 = add_88 = None
    copy__43: "f32[256]" = torch.ops.aten.copy_.default(primals_266, add_89);  primals_266 = add_89 = None
    copy__44: "i64[]" = torch.ops.aten.copy_.default(primals_267, add_86);  primals_267 = add_86 = None
    copy__45: "f32[128]" = torch.ops.aten.copy_.default(primals_268, add_94);  primals_268 = add_94 = None
    copy__46: "f32[128]" = torch.ops.aten.copy_.default(primals_269, add_95);  primals_269 = add_95 = None
    copy__47: "i64[]" = torch.ops.aten.copy_.default(primals_270, add_92);  primals_270 = add_92 = None
    copy__48: "f32[256]" = torch.ops.aten.copy_.default(primals_271, add_100);  primals_271 = add_100 = None
    copy__49: "f32[256]" = torch.ops.aten.copy_.default(primals_272, add_101);  primals_272 = add_101 = None
    copy__50: "i64[]" = torch.ops.aten.copy_.default(primals_273, add_98);  primals_273 = add_98 = None
    copy__51: "f32[128]" = torch.ops.aten.copy_.default(primals_274, add_107);  primals_274 = add_107 = None
    copy__52: "f32[128]" = torch.ops.aten.copy_.default(primals_275, add_108);  primals_275 = add_108 = None
    copy__53: "i64[]" = torch.ops.aten.copy_.default(primals_276, add_105);  primals_276 = add_105 = None
    copy__54: "f32[256]" = torch.ops.aten.copy_.default(primals_277, add_113);  primals_277 = add_113 = None
    copy__55: "f32[256]" = torch.ops.aten.copy_.default(primals_278, add_114);  primals_278 = add_114 = None
    copy__56: "i64[]" = torch.ops.aten.copy_.default(primals_279, add_111);  primals_279 = add_111 = None
    copy__57: "f32[128]" = torch.ops.aten.copy_.default(primals_280, add_119);  primals_280 = add_119 = None
    copy__58: "f32[128]" = torch.ops.aten.copy_.default(primals_281, add_120);  primals_281 = add_120 = None
    copy__59: "i64[]" = torch.ops.aten.copy_.default(primals_282, add_117);  primals_282 = add_117 = None
    copy__60: "f32[640]" = torch.ops.aten.copy_.default(primals_283, add_125);  primals_283 = add_125 = None
    copy__61: "f32[640]" = torch.ops.aten.copy_.default(primals_284, add_126);  primals_284 = add_126 = None
    copy__62: "i64[]" = torch.ops.aten.copy_.default(primals_285, add_123);  primals_285 = add_123 = None
    copy__63: "f32[128]" = torch.ops.aten.copy_.default(primals_286, add_130);  primals_286 = add_130 = None
    copy__64: "f32[128]" = torch.ops.aten.copy_.default(primals_287, add_131);  primals_287 = add_131 = None
    copy__65: "i64[]" = torch.ops.aten.copy_.default(primals_288, add_128);  primals_288 = add_128 = None
    copy__66: "f32[256]" = torch.ops.aten.copy_.default(primals_289, add_137);  primals_289 = add_137 = None
    copy__67: "f32[256]" = torch.ops.aten.copy_.default(primals_290, add_138);  primals_290 = add_138 = None
    copy__68: "i64[]" = torch.ops.aten.copy_.default(primals_291, add_135);  primals_291 = add_135 = None
    copy__69: "f32[512]" = torch.ops.aten.copy_.default(primals_292, add_142);  primals_292 = add_142 = None
    copy__70: "f32[512]" = torch.ops.aten.copy_.default(primals_293, add_143);  primals_293 = add_143 = None
    copy__71: "i64[]" = torch.ops.aten.copy_.default(primals_294, add_140);  primals_294 = add_140 = None
    copy__72: "f32[256]" = torch.ops.aten.copy_.default(primals_295, add_148);  primals_295 = add_148 = None
    copy__73: "f32[256]" = torch.ops.aten.copy_.default(primals_296, add_149);  primals_296 = add_149 = None
    copy__74: "i64[]" = torch.ops.aten.copy_.default(primals_297, add_146);  primals_297 = add_146 = None
    copy__75: "f32[512]" = torch.ops.aten.copy_.default(primals_298, add_154);  primals_298 = add_154 = None
    copy__76: "f32[512]" = torch.ops.aten.copy_.default(primals_299, add_155);  primals_299 = add_155 = None
    copy__77: "i64[]" = torch.ops.aten.copy_.default(primals_300, add_152);  primals_300 = add_152 = None
    copy__78: "f32[256]" = torch.ops.aten.copy_.default(primals_301, add_161);  primals_301 = add_161 = None
    copy__79: "f32[256]" = torch.ops.aten.copy_.default(primals_302, add_162);  primals_302 = add_162 = None
    copy__80: "i64[]" = torch.ops.aten.copy_.default(primals_303, add_159);  primals_303 = add_159 = None
    copy__81: "f32[512]" = torch.ops.aten.copy_.default(primals_304, add_167);  primals_304 = add_167 = None
    copy__82: "f32[512]" = torch.ops.aten.copy_.default(primals_305, add_168);  primals_305 = add_168 = None
    copy__83: "i64[]" = torch.ops.aten.copy_.default(primals_306, add_165);  primals_306 = add_165 = None
    copy__84: "f32[256]" = torch.ops.aten.copy_.default(primals_307, add_173);  primals_307 = add_173 = None
    copy__85: "f32[256]" = torch.ops.aten.copy_.default(primals_308, add_174);  primals_308 = add_174 = None
    copy__86: "i64[]" = torch.ops.aten.copy_.default(primals_309, add_171);  primals_309 = add_171 = None
    copy__87: "f32[512]" = torch.ops.aten.copy_.default(primals_310, add_179);  primals_310 = add_179 = None
    copy__88: "f32[512]" = torch.ops.aten.copy_.default(primals_311, add_180);  primals_311 = add_180 = None
    copy__89: "i64[]" = torch.ops.aten.copy_.default(primals_312, add_177);  primals_312 = add_177 = None
    copy__90: "f32[256]" = torch.ops.aten.copy_.default(primals_313, add_186);  primals_313 = add_186 = None
    copy__91: "f32[256]" = torch.ops.aten.copy_.default(primals_314, add_187);  primals_314 = add_187 = None
    copy__92: "i64[]" = torch.ops.aten.copy_.default(primals_315, add_184);  primals_315 = add_184 = None
    copy__93: "f32[512]" = torch.ops.aten.copy_.default(primals_316, add_192);  primals_316 = add_192 = None
    copy__94: "f32[512]" = torch.ops.aten.copy_.default(primals_317, add_193);  primals_317 = add_193 = None
    copy__95: "i64[]" = torch.ops.aten.copy_.default(primals_318, add_190);  primals_318 = add_190 = None
    copy__96: "f32[256]" = torch.ops.aten.copy_.default(primals_319, add_198);  primals_319 = add_198 = None
    copy__97: "f32[256]" = torch.ops.aten.copy_.default(primals_320, add_199);  primals_320 = add_199 = None
    copy__98: "i64[]" = torch.ops.aten.copy_.default(primals_321, add_196);  primals_321 = add_196 = None
    copy__99: "f32[512]" = torch.ops.aten.copy_.default(primals_322, add_204);  primals_322 = add_204 = None
    copy__100: "f32[512]" = torch.ops.aten.copy_.default(primals_323, add_205);  primals_323 = add_205 = None
    copy__101: "i64[]" = torch.ops.aten.copy_.default(primals_324, add_202);  primals_324 = add_202 = None
    copy__102: "f32[256]" = torch.ops.aten.copy_.default(primals_325, add_211);  primals_325 = add_211 = None
    copy__103: "f32[256]" = torch.ops.aten.copy_.default(primals_326, add_212);  primals_326 = add_212 = None
    copy__104: "i64[]" = torch.ops.aten.copy_.default(primals_327, add_209);  primals_327 = add_209 = None
    copy__105: "f32[512]" = torch.ops.aten.copy_.default(primals_328, add_217);  primals_328 = add_217 = None
    copy__106: "f32[512]" = torch.ops.aten.copy_.default(primals_329, add_218);  primals_329 = add_218 = None
    copy__107: "i64[]" = torch.ops.aten.copy_.default(primals_330, add_215);  primals_330 = add_215 = None
    copy__108: "f32[256]" = torch.ops.aten.copy_.default(primals_331, add_223);  primals_331 = add_223 = None
    copy__109: "f32[256]" = torch.ops.aten.copy_.default(primals_332, add_224);  primals_332 = add_224 = None
    copy__110: "i64[]" = torch.ops.aten.copy_.default(primals_333, add_221);  primals_333 = add_221 = None
    copy__111: "f32[512]" = torch.ops.aten.copy_.default(primals_334, add_229);  primals_334 = add_229 = None
    copy__112: "f32[512]" = torch.ops.aten.copy_.default(primals_335, add_230);  primals_335 = add_230 = None
    copy__113: "i64[]" = torch.ops.aten.copy_.default(primals_336, add_227);  primals_336 = add_227 = None
    copy__114: "f32[256]" = torch.ops.aten.copy_.default(primals_337, add_236);  primals_337 = add_236 = None
    copy__115: "f32[256]" = torch.ops.aten.copy_.default(primals_338, add_237);  primals_338 = add_237 = None
    copy__116: "i64[]" = torch.ops.aten.copy_.default(primals_339, add_234);  primals_339 = add_234 = None
    copy__117: "f32[512]" = torch.ops.aten.copy_.default(primals_340, add_242);  primals_340 = add_242 = None
    copy__118: "f32[512]" = torch.ops.aten.copy_.default(primals_341, add_243);  primals_341 = add_243 = None
    copy__119: "i64[]" = torch.ops.aten.copy_.default(primals_342, add_240);  primals_342 = add_240 = None
    copy__120: "f32[256]" = torch.ops.aten.copy_.default(primals_343, add_248);  primals_343 = add_248 = None
    copy__121: "f32[256]" = torch.ops.aten.copy_.default(primals_344, add_249);  primals_344 = add_249 = None
    copy__122: "i64[]" = torch.ops.aten.copy_.default(primals_345, add_246);  primals_345 = add_246 = None
    copy__123: "f32[1280]" = torch.ops.aten.copy_.default(primals_346, add_254);  primals_346 = add_254 = None
    copy__124: "f32[1280]" = torch.ops.aten.copy_.default(primals_347, add_255);  primals_347 = add_255 = None
    copy__125: "i64[]" = torch.ops.aten.copy_.default(primals_348, add_252);  primals_348 = add_252 = None
    copy__126: "f32[256]" = torch.ops.aten.copy_.default(primals_349, add_259);  primals_349 = add_259 = None
    copy__127: "f32[256]" = torch.ops.aten.copy_.default(primals_350, add_260);  primals_350 = add_260 = None
    copy__128: "i64[]" = torch.ops.aten.copy_.default(primals_351, add_257);  primals_351 = add_257 = None
    copy__129: "f32[384]" = torch.ops.aten.copy_.default(primals_352, add_266);  primals_352 = add_266 = None
    copy__130: "f32[384]" = torch.ops.aten.copy_.default(primals_353, add_267);  primals_353 = add_267 = None
    copy__131: "i64[]" = torch.ops.aten.copy_.default(primals_354, add_264);  primals_354 = add_264 = None
    copy__132: "f32[768]" = torch.ops.aten.copy_.default(primals_355, add_271);  primals_355 = add_271 = None
    copy__133: "f32[768]" = torch.ops.aten.copy_.default(primals_356, add_272);  primals_356 = add_272 = None
    copy__134: "i64[]" = torch.ops.aten.copy_.default(primals_357, add_269);  primals_357 = add_269 = None
    copy__135: "f32[384]" = torch.ops.aten.copy_.default(primals_358, add_277);  primals_358 = add_277 = None
    copy__136: "f32[384]" = torch.ops.aten.copy_.default(primals_359, add_278);  primals_359 = add_278 = None
    copy__137: "i64[]" = torch.ops.aten.copy_.default(primals_360, add_275);  primals_360 = add_275 = None
    copy__138: "f32[768]" = torch.ops.aten.copy_.default(primals_361, add_283);  primals_361 = add_283 = None
    copy__139: "f32[768]" = torch.ops.aten.copy_.default(primals_362, add_284);  primals_362 = add_284 = None
    copy__140: "i64[]" = torch.ops.aten.copy_.default(primals_363, add_281);  primals_363 = add_281 = None
    copy__141: "f32[384]" = torch.ops.aten.copy_.default(primals_364, add_290);  primals_364 = add_290 = None
    copy__142: "f32[384]" = torch.ops.aten.copy_.default(primals_365, add_291);  primals_365 = add_291 = None
    copy__143: "i64[]" = torch.ops.aten.copy_.default(primals_366, add_288);  primals_366 = add_288 = None
    copy__144: "f32[768]" = torch.ops.aten.copy_.default(primals_367, add_296);  primals_367 = add_296 = None
    copy__145: "f32[768]" = torch.ops.aten.copy_.default(primals_368, add_297);  primals_368 = add_297 = None
    copy__146: "i64[]" = torch.ops.aten.copy_.default(primals_369, add_294);  primals_369 = add_294 = None
    copy__147: "f32[384]" = torch.ops.aten.copy_.default(primals_370, add_302);  primals_370 = add_302 = None
    copy__148: "f32[384]" = torch.ops.aten.copy_.default(primals_371, add_303);  primals_371 = add_303 = None
    copy__149: "i64[]" = torch.ops.aten.copy_.default(primals_372, add_300);  primals_372 = add_300 = None
    copy__150: "f32[768]" = torch.ops.aten.copy_.default(primals_373, add_308);  primals_373 = add_308 = None
    copy__151: "f32[768]" = torch.ops.aten.copy_.default(primals_374, add_309);  primals_374 = add_309 = None
    copy__152: "i64[]" = torch.ops.aten.copy_.default(primals_375, add_306);  primals_375 = add_306 = None
    copy__153: "f32[384]" = torch.ops.aten.copy_.default(primals_376, add_315);  primals_376 = add_315 = None
    copy__154: "f32[384]" = torch.ops.aten.copy_.default(primals_377, add_316);  primals_377 = add_316 = None
    copy__155: "i64[]" = torch.ops.aten.copy_.default(primals_378, add_313);  primals_378 = add_313 = None
    copy__156: "f32[768]" = torch.ops.aten.copy_.default(primals_379, add_321);  primals_379 = add_321 = None
    copy__157: "f32[768]" = torch.ops.aten.copy_.default(primals_380, add_322);  primals_380 = add_322 = None
    copy__158: "i64[]" = torch.ops.aten.copy_.default(primals_381, add_319);  primals_381 = add_319 = None
    copy__159: "f32[384]" = torch.ops.aten.copy_.default(primals_382, add_327);  primals_382 = add_327 = None
    copy__160: "f32[384]" = torch.ops.aten.copy_.default(primals_383, add_328);  primals_383 = add_328 = None
    copy__161: "i64[]" = torch.ops.aten.copy_.default(primals_384, add_325);  primals_384 = add_325 = None
    copy__162: "f32[768]" = torch.ops.aten.copy_.default(primals_385, add_333);  primals_385 = add_333 = None
    copy__163: "f32[768]" = torch.ops.aten.copy_.default(primals_386, add_334);  primals_386 = add_334 = None
    copy__164: "i64[]" = torch.ops.aten.copy_.default(primals_387, add_331);  primals_387 = add_331 = None
    copy__165: "f32[384]" = torch.ops.aten.copy_.default(primals_388, add_340);  primals_388 = add_340 = None
    copy__166: "f32[384]" = torch.ops.aten.copy_.default(primals_389, add_341);  primals_389 = add_341 = None
    copy__167: "i64[]" = torch.ops.aten.copy_.default(primals_390, add_338);  primals_390 = add_338 = None
    copy__168: "f32[768]" = torch.ops.aten.copy_.default(primals_391, add_346);  primals_391 = add_346 = None
    copy__169: "f32[768]" = torch.ops.aten.copy_.default(primals_392, add_347);  primals_392 = add_347 = None
    copy__170: "i64[]" = torch.ops.aten.copy_.default(primals_393, add_344);  primals_393 = add_344 = None
    copy__171: "f32[384]" = torch.ops.aten.copy_.default(primals_394, add_352);  primals_394 = add_352 = None
    copy__172: "f32[384]" = torch.ops.aten.copy_.default(primals_395, add_353);  primals_395 = add_353 = None
    copy__173: "i64[]" = torch.ops.aten.copy_.default(primals_396, add_350);  primals_396 = add_350 = None
    copy__174: "f32[768]" = torch.ops.aten.copy_.default(primals_397, add_358);  primals_397 = add_358 = None
    copy__175: "f32[768]" = torch.ops.aten.copy_.default(primals_398, add_359);  primals_398 = add_359 = None
    copy__176: "i64[]" = torch.ops.aten.copy_.default(primals_399, add_356);  primals_399 = add_356 = None
    copy__177: "f32[384]" = torch.ops.aten.copy_.default(primals_400, add_365);  primals_400 = add_365 = None
    copy__178: "f32[384]" = torch.ops.aten.copy_.default(primals_401, add_366);  primals_401 = add_366 = None
    copy__179: "i64[]" = torch.ops.aten.copy_.default(primals_402, add_363);  primals_402 = add_363 = None
    copy__180: "f32[768]" = torch.ops.aten.copy_.default(primals_403, add_371);  primals_403 = add_371 = None
    copy__181: "f32[768]" = torch.ops.aten.copy_.default(primals_404, add_372);  primals_404 = add_372 = None
    copy__182: "i64[]" = torch.ops.aten.copy_.default(primals_405, add_369);  primals_405 = add_369 = None
    copy__183: "f32[384]" = torch.ops.aten.copy_.default(primals_406, add_377);  primals_406 = add_377 = None
    copy__184: "f32[384]" = torch.ops.aten.copy_.default(primals_407, add_378);  primals_407 = add_378 = None
    copy__185: "i64[]" = torch.ops.aten.copy_.default(primals_408, add_375);  primals_408 = add_375 = None
    copy__186: "f32[384]" = torch.ops.aten.copy_.default(primals_409, add_383);  primals_409 = add_383 = None
    copy__187: "f32[384]" = torch.ops.aten.copy_.default(primals_410, add_384);  primals_410 = add_384 = None
    copy__188: "i64[]" = torch.ops.aten.copy_.default(primals_411, add_381);  primals_411 = add_381 = None
    copy__189: "f32[384]" = torch.ops.aten.copy_.default(primals_412, add_388);  primals_412 = add_388 = None
    copy__190: "f32[384]" = torch.ops.aten.copy_.default(primals_413, add_389);  primals_413 = add_389 = None
    copy__191: "i64[]" = torch.ops.aten.copy_.default(primals_414, add_386);  primals_414 = add_386 = None
    return pytree.tree_unflatten([div_45, slice_scatter_19, slice_scatter_18, slice_scatter_17, slice_scatter_16, slice_scatter_12, slice_scatter_11, slice_scatter_10, slice_scatter_9, slice_scatter_8, slice_scatter_4, slice_scatter_3, slice_scatter_2, slice_scatter_1, slice_scatter, getitem_178, mul_1141, sum_171, getitem_175, mul_1131, sum_169, getitem_172, mul_1121, sum_167, getitem_169, mul_1111, sum_165, permute_468, mul_1102, sum_163, permute_456, mul_1089, sum_159, permute_452, mul_1080, sum_157, permute_448, mul_1070, sum_155, permute_444, mul_1061, sum_153, permute_432, mul_1048, sum_149, permute_428, mul_1039, sum_147, permute_424, mul_1029, sum_145, permute_420, mul_1020, sum_143, permute_408, mul_1007, sum_139, permute_404, mul_998, sum_137, permute_400, mul_988, sum_135, permute_396, mul_979, sum_133, permute_384, mul_966, sum_129, permute_380, mul_957, sum_127, permute_376, mul_947, sum_125, permute_372, mul_938, sum_123, permute_366, mul_929, sum_121, permute_356, mul_916, sum_117, permute_352, mul_907, sum_115, permute_348, mul_897, sum_113, permute_344, mul_888, sum_111, permute_332, mul_875, sum_107, permute_328, mul_866, sum_105, permute_324, mul_856, sum_103, permute_320, mul_847, sum_101, permute_308, mul_834, sum_97, permute_304, mul_825, sum_95, permute_300, mul_815, sum_93, permute_296, mul_806, sum_91, permute_284, mul_793, sum_87, permute_280, mul_784, sum_85, permute_276, mul_774, sum_83, permute_272, mul_765, sum_81, permute_260, mul_752, sum_77, permute_256, mul_743, sum_75, permute_252, mul_733, sum_73, permute_248, mul_724, sum_71, permute_242, mul_715, sum_69, permute_232, mul_702, sum_65, permute_228, mul_693, sum_63, permute_224, mul_683, sum_61, permute_220, mul_674, sum_59, permute_208, mul_661, sum_55, permute_204, mul_652, sum_53, permute_200, mul_642, sum_51, permute_196, mul_633, sum_49, permute_184, mul_620, sum_45, permute_180, mul_611, sum_43, permute_176, mul_601, sum_41, permute_172, mul_592, sum_39, permute_160, mul_579, sum_35, permute_156, mul_570, sum_33, permute_152, mul_560, sum_31, permute_148, mul_551, sum_29, permute_136, mul_538, sum_25, permute_132, mul_529, sum_23, permute_128, mul_519, sum_21, mul_510, sum_19, permute_124, view_352, mul_501, sum_16, permute_120, view_351, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    