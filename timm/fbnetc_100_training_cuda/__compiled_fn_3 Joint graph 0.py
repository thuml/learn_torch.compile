from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[16]"; primals_2: "f32[16]"; primals_3: "f32[16]"; primals_4: "f32[16]"; primals_5: "f32[16]"; primals_6: "f32[16]"; primals_7: "f32[16]"; primals_8: "f32[16]"; primals_9: "f32[96]"; primals_10: "f32[96]"; primals_11: "f32[96]"; primals_12: "f32[96]"; primals_13: "f32[24]"; primals_14: "f32[24]"; primals_15: "f32[24]"; primals_16: "f32[24]"; primals_17: "f32[24]"; primals_18: "f32[24]"; primals_19: "f32[24]"; primals_20: "f32[24]"; primals_21: "f32[24]"; primals_22: "f32[24]"; primals_23: "f32[24]"; primals_24: "f32[24]"; primals_25: "f32[24]"; primals_26: "f32[24]"; primals_27: "f32[144]"; primals_28: "f32[144]"; primals_29: "f32[144]"; primals_30: "f32[144]"; primals_31: "f32[32]"; primals_32: "f32[32]"; primals_33: "f32[96]"; primals_34: "f32[96]"; primals_35: "f32[96]"; primals_36: "f32[96]"; primals_37: "f32[32]"; primals_38: "f32[32]"; primals_39: "f32[192]"; primals_40: "f32[192]"; primals_41: "f32[192]"; primals_42: "f32[192]"; primals_43: "f32[32]"; primals_44: "f32[32]"; primals_45: "f32[192]"; primals_46: "f32[192]"; primals_47: "f32[192]"; primals_48: "f32[192]"; primals_49: "f32[32]"; primals_50: "f32[32]"; primals_51: "f32[192]"; primals_52: "f32[192]"; primals_53: "f32[192]"; primals_54: "f32[192]"; primals_55: "f32[64]"; primals_56: "f32[64]"; primals_57: "f32[192]"; primals_58: "f32[192]"; primals_59: "f32[192]"; primals_60: "f32[192]"; primals_61: "f32[64]"; primals_62: "f32[64]"; primals_63: "f32[384]"; primals_64: "f32[384]"; primals_65: "f32[384]"; primals_66: "f32[384]"; primals_67: "f32[64]"; primals_68: "f32[64]"; primals_69: "f32[384]"; primals_70: "f32[384]"; primals_71: "f32[384]"; primals_72: "f32[384]"; primals_73: "f32[64]"; primals_74: "f32[64]"; primals_75: "f32[384]"; primals_76: "f32[384]"; primals_77: "f32[384]"; primals_78: "f32[384]"; primals_79: "f32[112]"; primals_80: "f32[112]"; primals_81: "f32[672]"; primals_82: "f32[672]"; primals_83: "f32[672]"; primals_84: "f32[672]"; primals_85: "f32[112]"; primals_86: "f32[112]"; primals_87: "f32[672]"; primals_88: "f32[672]"; primals_89: "f32[672]"; primals_90: "f32[672]"; primals_91: "f32[112]"; primals_92: "f32[112]"; primals_93: "f32[336]"; primals_94: "f32[336]"; primals_95: "f32[336]"; primals_96: "f32[336]"; primals_97: "f32[112]"; primals_98: "f32[112]"; primals_99: "f32[672]"; primals_100: "f32[672]"; primals_101: "f32[672]"; primals_102: "f32[672]"; primals_103: "f32[184]"; primals_104: "f32[184]"; primals_105: "f32[1104]"; primals_106: "f32[1104]"; primals_107: "f32[1104]"; primals_108: "f32[1104]"; primals_109: "f32[184]"; primals_110: "f32[184]"; primals_111: "f32[1104]"; primals_112: "f32[1104]"; primals_113: "f32[1104]"; primals_114: "f32[1104]"; primals_115: "f32[184]"; primals_116: "f32[184]"; primals_117: "f32[1104]"; primals_118: "f32[1104]"; primals_119: "f32[1104]"; primals_120: "f32[1104]"; primals_121: "f32[184]"; primals_122: "f32[184]"; primals_123: "f32[1104]"; primals_124: "f32[1104]"; primals_125: "f32[1104]"; primals_126: "f32[1104]"; primals_127: "f32[352]"; primals_128: "f32[352]"; primals_129: "f32[1984]"; primals_130: "f32[1984]"; primals_131: "f32[16, 3, 3, 3]"; primals_132: "f32[16, 16, 1, 1]"; primals_133: "f32[16, 1, 3, 3]"; primals_134: "f32[16, 16, 1, 1]"; primals_135: "f32[96, 16, 1, 1]"; primals_136: "f32[96, 1, 3, 3]"; primals_137: "f32[24, 96, 1, 1]"; primals_138: "f32[24, 24, 1, 1]"; primals_139: "f32[24, 1, 3, 3]"; primals_140: "f32[24, 24, 1, 1]"; primals_141: "f32[24, 24, 1, 1]"; primals_142: "f32[24, 1, 3, 3]"; primals_143: "f32[24, 24, 1, 1]"; primals_144: "f32[144, 24, 1, 1]"; primals_145: "f32[144, 1, 5, 5]"; primals_146: "f32[32, 144, 1, 1]"; primals_147: "f32[96, 32, 1, 1]"; primals_148: "f32[96, 1, 5, 5]"; primals_149: "f32[32, 96, 1, 1]"; primals_150: "f32[192, 32, 1, 1]"; primals_151: "f32[192, 1, 5, 5]"; primals_152: "f32[32, 192, 1, 1]"; primals_153: "f32[192, 32, 1, 1]"; primals_154: "f32[192, 1, 3, 3]"; primals_155: "f32[32, 192, 1, 1]"; primals_156: "f32[192, 32, 1, 1]"; primals_157: "f32[192, 1, 5, 5]"; primals_158: "f32[64, 192, 1, 1]"; primals_159: "f32[192, 64, 1, 1]"; primals_160: "f32[192, 1, 5, 5]"; primals_161: "f32[64, 192, 1, 1]"; primals_162: "f32[384, 64, 1, 1]"; primals_163: "f32[384, 1, 5, 5]"; primals_164: "f32[64, 384, 1, 1]"; primals_165: "f32[384, 64, 1, 1]"; primals_166: "f32[384, 1, 5, 5]"; primals_167: "f32[64, 384, 1, 1]"; primals_168: "f32[384, 64, 1, 1]"; primals_169: "f32[384, 1, 5, 5]"; primals_170: "f32[112, 384, 1, 1]"; primals_171: "f32[672, 112, 1, 1]"; primals_172: "f32[672, 1, 5, 5]"; primals_173: "f32[112, 672, 1, 1]"; primals_174: "f32[672, 112, 1, 1]"; primals_175: "f32[672, 1, 5, 5]"; primals_176: "f32[112, 672, 1, 1]"; primals_177: "f32[336, 112, 1, 1]"; primals_178: "f32[336, 1, 5, 5]"; primals_179: "f32[112, 336, 1, 1]"; primals_180: "f32[672, 112, 1, 1]"; primals_181: "f32[672, 1, 5, 5]"; primals_182: "f32[184, 672, 1, 1]"; primals_183: "f32[1104, 184, 1, 1]"; primals_184: "f32[1104, 1, 5, 5]"; primals_185: "f32[184, 1104, 1, 1]"; primals_186: "f32[1104, 184, 1, 1]"; primals_187: "f32[1104, 1, 5, 5]"; primals_188: "f32[184, 1104, 1, 1]"; primals_189: "f32[1104, 184, 1, 1]"; primals_190: "f32[1104, 1, 5, 5]"; primals_191: "f32[184, 1104, 1, 1]"; primals_192: "f32[1104, 184, 1, 1]"; primals_193: "f32[1104, 1, 3, 3]"; primals_194: "f32[352, 1104, 1, 1]"; primals_195: "f32[1984, 352, 1, 1]"; primals_196: "f32[1000, 1984]"; primals_197: "f32[1000]"; primals_198: "i64[]"; primals_199: "f32[16]"; primals_200: "f32[16]"; primals_201: "i64[]"; primals_202: "f32[16]"; primals_203: "f32[16]"; primals_204: "i64[]"; primals_205: "f32[16]"; primals_206: "f32[16]"; primals_207: "i64[]"; primals_208: "f32[16]"; primals_209: "f32[16]"; primals_210: "i64[]"; primals_211: "f32[96]"; primals_212: "f32[96]"; primals_213: "i64[]"; primals_214: "f32[96]"; primals_215: "f32[96]"; primals_216: "i64[]"; primals_217: "f32[24]"; primals_218: "f32[24]"; primals_219: "i64[]"; primals_220: "f32[24]"; primals_221: "f32[24]"; primals_222: "i64[]"; primals_223: "f32[24]"; primals_224: "f32[24]"; primals_225: "i64[]"; primals_226: "f32[24]"; primals_227: "f32[24]"; primals_228: "i64[]"; primals_229: "f32[24]"; primals_230: "f32[24]"; primals_231: "i64[]"; primals_232: "f32[24]"; primals_233: "f32[24]"; primals_234: "i64[]"; primals_235: "f32[24]"; primals_236: "f32[24]"; primals_237: "i64[]"; primals_238: "f32[144]"; primals_239: "f32[144]"; primals_240: "i64[]"; primals_241: "f32[144]"; primals_242: "f32[144]"; primals_243: "i64[]"; primals_244: "f32[32]"; primals_245: "f32[32]"; primals_246: "i64[]"; primals_247: "f32[96]"; primals_248: "f32[96]"; primals_249: "i64[]"; primals_250: "f32[96]"; primals_251: "f32[96]"; primals_252: "i64[]"; primals_253: "f32[32]"; primals_254: "f32[32]"; primals_255: "i64[]"; primals_256: "f32[192]"; primals_257: "f32[192]"; primals_258: "i64[]"; primals_259: "f32[192]"; primals_260: "f32[192]"; primals_261: "i64[]"; primals_262: "f32[32]"; primals_263: "f32[32]"; primals_264: "i64[]"; primals_265: "f32[192]"; primals_266: "f32[192]"; primals_267: "i64[]"; primals_268: "f32[192]"; primals_269: "f32[192]"; primals_270: "i64[]"; primals_271: "f32[32]"; primals_272: "f32[32]"; primals_273: "i64[]"; primals_274: "f32[192]"; primals_275: "f32[192]"; primals_276: "i64[]"; primals_277: "f32[192]"; primals_278: "f32[192]"; primals_279: "i64[]"; primals_280: "f32[64]"; primals_281: "f32[64]"; primals_282: "i64[]"; primals_283: "f32[192]"; primals_284: "f32[192]"; primals_285: "i64[]"; primals_286: "f32[192]"; primals_287: "f32[192]"; primals_288: "i64[]"; primals_289: "f32[64]"; primals_290: "f32[64]"; primals_291: "i64[]"; primals_292: "f32[384]"; primals_293: "f32[384]"; primals_294: "i64[]"; primals_295: "f32[384]"; primals_296: "f32[384]"; primals_297: "i64[]"; primals_298: "f32[64]"; primals_299: "f32[64]"; primals_300: "i64[]"; primals_301: "f32[384]"; primals_302: "f32[384]"; primals_303: "i64[]"; primals_304: "f32[384]"; primals_305: "f32[384]"; primals_306: "i64[]"; primals_307: "f32[64]"; primals_308: "f32[64]"; primals_309: "i64[]"; primals_310: "f32[384]"; primals_311: "f32[384]"; primals_312: "i64[]"; primals_313: "f32[384]"; primals_314: "f32[384]"; primals_315: "i64[]"; primals_316: "f32[112]"; primals_317: "f32[112]"; primals_318: "i64[]"; primals_319: "f32[672]"; primals_320: "f32[672]"; primals_321: "i64[]"; primals_322: "f32[672]"; primals_323: "f32[672]"; primals_324: "i64[]"; primals_325: "f32[112]"; primals_326: "f32[112]"; primals_327: "i64[]"; primals_328: "f32[672]"; primals_329: "f32[672]"; primals_330: "i64[]"; primals_331: "f32[672]"; primals_332: "f32[672]"; primals_333: "i64[]"; primals_334: "f32[112]"; primals_335: "f32[112]"; primals_336: "i64[]"; primals_337: "f32[336]"; primals_338: "f32[336]"; primals_339: "i64[]"; primals_340: "f32[336]"; primals_341: "f32[336]"; primals_342: "i64[]"; primals_343: "f32[112]"; primals_344: "f32[112]"; primals_345: "i64[]"; primals_346: "f32[672]"; primals_347: "f32[672]"; primals_348: "i64[]"; primals_349: "f32[672]"; primals_350: "f32[672]"; primals_351: "i64[]"; primals_352: "f32[184]"; primals_353: "f32[184]"; primals_354: "i64[]"; primals_355: "f32[1104]"; primals_356: "f32[1104]"; primals_357: "i64[]"; primals_358: "f32[1104]"; primals_359: "f32[1104]"; primals_360: "i64[]"; primals_361: "f32[184]"; primals_362: "f32[184]"; primals_363: "i64[]"; primals_364: "f32[1104]"; primals_365: "f32[1104]"; primals_366: "i64[]"; primals_367: "f32[1104]"; primals_368: "f32[1104]"; primals_369: "i64[]"; primals_370: "f32[184]"; primals_371: "f32[184]"; primals_372: "i64[]"; primals_373: "f32[1104]"; primals_374: "f32[1104]"; primals_375: "i64[]"; primals_376: "f32[1104]"; primals_377: "f32[1104]"; primals_378: "i64[]"; primals_379: "f32[184]"; primals_380: "f32[184]"; primals_381: "i64[]"; primals_382: "f32[1104]"; primals_383: "f32[1104]"; primals_384: "i64[]"; primals_385: "f32[1104]"; primals_386: "f32[1104]"; primals_387: "i64[]"; primals_388: "f32[352]"; primals_389: "f32[352]"; primals_390: "i64[]"; primals_391: "f32[1984]"; primals_392: "f32[1984]"; primals_393: "f32[8, 3, 224, 224]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:162, code: x = self.conv_stem(x)
    convolution: "f32[8, 16, 112, 112]" = torch.ops.aten.convolution.default(primals_393, primals_131, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_198, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
    getitem: "f32[1, 16, 1, 1]" = var_mean[0]
    getitem_1: "f32[1, 16, 1, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 0.001)
    rsqrt: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
    mul: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    squeeze: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
    squeeze_1: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
    mul_1: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
    mul_2: "f32[16]" = torch.ops.aten.mul.Tensor(primals_199, 0.9)
    add_2: "f32[16]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[16]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.00000996502277);  squeeze_2 = None
    mul_4: "f32[16]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[16]" = torch.ops.aten.mul.Tensor(primals_200, 0.9)
    add_3: "f32[16]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_1, -1)
    unsqueeze_1: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1);  primals_2 = None
    unsqueeze_3: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu: "f32[8, 16, 112, 112]" = torch.ops.aten.relu.default(add_4);  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_1: "f32[8, 16, 112, 112]" = torch.ops.aten.convolution.default(relu, primals_132, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_201, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[1, 16, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 16, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 0.001)
    rsqrt_1: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_1: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
    mul_7: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
    squeeze_4: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_8: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_9: "f32[16]" = torch.ops.aten.mul.Tensor(primals_202, 0.9)
    add_7: "f32[16]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
    squeeze_5: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_10: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.00000996502277);  squeeze_5 = None
    mul_11: "f32[16]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
    mul_12: "f32[16]" = torch.ops.aten.mul.Tensor(primals_203, 0.9)
    add_8: "f32[16]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
    unsqueeze_4: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1)
    unsqueeze_5: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_13: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
    unsqueeze_6: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_4, -1);  primals_4 = None
    unsqueeze_7: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_9: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_1: "f32[8, 16, 112, 112]" = torch.ops.aten.relu.default(add_9);  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_2: "f32[8, 16, 112, 112]" = torch.ops.aten.convolution.default(relu_1, primals_133, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_204, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 16, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 16, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 0.001)
    rsqrt_2: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_2: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
    mul_14: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_15: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_16: "f32[16]" = torch.ops.aten.mul.Tensor(primals_205, 0.9)
    add_12: "f32[16]" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
    squeeze_8: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_17: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.00000996502277);  squeeze_8 = None
    mul_18: "f32[16]" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
    mul_19: "f32[16]" = torch.ops.aten.mul.Tensor(primals_206, 0.9)
    add_13: "f32[16]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
    unsqueeze_8: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_9: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_20: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
    unsqueeze_10: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_11: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_14: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_2: "f32[8, 16, 112, 112]" = torch.ops.aten.relu.default(add_14);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_3: "f32[8, 16, 112, 112]" = torch.ops.aten.convolution.default(relu_2, primals_134, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_15: "i64[]" = torch.ops.aten.add.Tensor(primals_207, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
    getitem_6: "f32[1, 16, 1, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 16, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_16: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 0.001)
    rsqrt_3: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_3: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_7)
    mul_21: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
    squeeze_10: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_22: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_23: "f32[16]" = torch.ops.aten.mul.Tensor(primals_208, 0.9)
    add_17: "f32[16]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
    squeeze_11: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
    mul_24: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.00000996502277);  squeeze_11 = None
    mul_25: "f32[16]" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
    mul_26: "f32[16]" = torch.ops.aten.mul.Tensor(primals_209, 0.9)
    add_18: "f32[16]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
    unsqueeze_12: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1)
    unsqueeze_13: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_27: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
    unsqueeze_14: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1);  primals_8 = None
    unsqueeze_15: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_19: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_20: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(add_19, relu);  add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_4: "f32[8, 96, 112, 112]" = torch.ops.aten.convolution.default(add_20, primals_135, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_21: "i64[]" = torch.ops.aten.add.Tensor(primals_210, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
    getitem_8: "f32[1, 96, 1, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 96, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_22: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 0.001)
    rsqrt_4: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    sub_4: "f32[8, 96, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_9)
    mul_28: "f32[8, 96, 112, 112]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
    squeeze_13: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
    mul_29: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_30: "f32[96]" = torch.ops.aten.mul.Tensor(primals_211, 0.9)
    add_23: "f32[96]" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
    squeeze_14: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
    mul_31: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.00000996502277);  squeeze_14 = None
    mul_32: "f32[96]" = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
    mul_33: "f32[96]" = torch.ops.aten.mul.Tensor(primals_212, 0.9)
    add_24: "f32[96]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
    unsqueeze_16: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1)
    unsqueeze_17: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    mul_34: "f32[8, 96, 112, 112]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
    unsqueeze_18: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_10, -1);  primals_10 = None
    unsqueeze_19: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    add_25: "f32[8, 96, 112, 112]" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_3: "f32[8, 96, 112, 112]" = torch.ops.aten.relu.default(add_25);  add_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_5: "f32[8, 96, 56, 56]" = torch.ops.aten.convolution.default(relu_3, primals_136, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 96)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_26: "i64[]" = torch.ops.aten.add.Tensor(primals_213, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
    getitem_10: "f32[1, 96, 1, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 96, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_27: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 0.001)
    rsqrt_5: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    sub_5: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_11)
    mul_35: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_15: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
    squeeze_16: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    mul_36: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_37: "f32[96]" = torch.ops.aten.mul.Tensor(primals_214, 0.9)
    add_28: "f32[96]" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
    squeeze_17: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
    mul_38: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000398612827361);  squeeze_17 = None
    mul_39: "f32[96]" = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
    mul_40: "f32[96]" = torch.ops.aten.mul.Tensor(primals_215, 0.9)
    add_29: "f32[96]" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
    unsqueeze_20: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_21: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_41: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
    unsqueeze_22: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_23: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_30: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_4: "f32[8, 96, 56, 56]" = torch.ops.aten.relu.default(add_30);  add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_6: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(relu_4, primals_137, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_31: "i64[]" = torch.ops.aten.add.Tensor(primals_216, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_6 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
    getitem_12: "f32[1, 24, 1, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 24, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_32: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 0.001)
    rsqrt_6: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_6: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_13)
    mul_42: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_18: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
    squeeze_19: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
    mul_43: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_44: "f32[24]" = torch.ops.aten.mul.Tensor(primals_217, 0.9)
    add_33: "f32[24]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
    squeeze_20: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
    mul_45: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0000398612827361);  squeeze_20 = None
    mul_46: "f32[24]" = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
    mul_47: "f32[24]" = torch.ops.aten.mul.Tensor(primals_218, 0.9)
    add_34: "f32[24]" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
    unsqueeze_24: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1)
    unsqueeze_25: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    mul_48: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
    unsqueeze_26: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1);  primals_14 = None
    unsqueeze_27: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    add_35: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_7: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(add_35, primals_138, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_36: "i64[]" = torch.ops.aten.add.Tensor(primals_219, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_7 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
    getitem_14: "f32[1, 24, 1, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 24, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_37: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 0.001)
    rsqrt_7: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_7: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_15)
    mul_49: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_21: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
    squeeze_22: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
    mul_50: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_51: "f32[24]" = torch.ops.aten.mul.Tensor(primals_220, 0.9)
    add_38: "f32[24]" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
    squeeze_23: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
    mul_52: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0000398612827361);  squeeze_23 = None
    mul_53: "f32[24]" = torch.ops.aten.mul.Tensor(mul_52, 0.1);  mul_52 = None
    mul_54: "f32[24]" = torch.ops.aten.mul.Tensor(primals_221, 0.9)
    add_39: "f32[24]" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
    unsqueeze_28: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1)
    unsqueeze_29: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_55: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
    unsqueeze_30: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1);  primals_16 = None
    unsqueeze_31: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_40: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_5: "f32[8, 24, 56, 56]" = torch.ops.aten.relu.default(add_40);  add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_8: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(relu_5, primals_139, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_41: "i64[]" = torch.ops.aten.add.Tensor(primals_222, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_8 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
    getitem_16: "f32[1, 24, 1, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 24, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_42: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 0.001)
    rsqrt_8: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_8: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_17)
    mul_56: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_24: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
    squeeze_25: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
    mul_57: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_58: "f32[24]" = torch.ops.aten.mul.Tensor(primals_223, 0.9)
    add_43: "f32[24]" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
    squeeze_26: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
    mul_59: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0000398612827361);  squeeze_26 = None
    mul_60: "f32[24]" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
    mul_61: "f32[24]" = torch.ops.aten.mul.Tensor(primals_224, 0.9)
    add_44: "f32[24]" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
    unsqueeze_32: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_33: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    mul_62: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
    unsqueeze_34: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_35: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    add_45: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_6: "f32[8, 24, 56, 56]" = torch.ops.aten.relu.default(add_45);  add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_9: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(relu_6, primals_140, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_46: "i64[]" = torch.ops.aten.add.Tensor(primals_225, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_9 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
    getitem_18: "f32[1, 24, 1, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 24, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_47: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 0.001)
    rsqrt_9: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_9: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_19)
    mul_63: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_27: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
    squeeze_28: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_64: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_65: "f32[24]" = torch.ops.aten.mul.Tensor(primals_226, 0.9)
    add_48: "f32[24]" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
    squeeze_29: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
    mul_66: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0000398612827361);  squeeze_29 = None
    mul_67: "f32[24]" = torch.ops.aten.mul.Tensor(mul_66, 0.1);  mul_66 = None
    mul_68: "f32[24]" = torch.ops.aten.mul.Tensor(primals_227, 0.9)
    add_49: "f32[24]" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
    unsqueeze_36: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
    unsqueeze_37: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_69: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
    unsqueeze_38: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
    unsqueeze_39: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_50: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_51: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(add_50, add_35);  add_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_10: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(add_51, primals_141, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_52: "i64[]" = torch.ops.aten.add.Tensor(primals_228, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_10 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[1, 24, 1, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 24, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_53: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 0.001)
    rsqrt_10: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    sub_10: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_21)
    mul_70: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_30: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
    squeeze_31: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
    mul_71: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_72: "f32[24]" = torch.ops.aten.mul.Tensor(primals_229, 0.9)
    add_54: "f32[24]" = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
    squeeze_32: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
    mul_73: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0000398612827361);  squeeze_32 = None
    mul_74: "f32[24]" = torch.ops.aten.mul.Tensor(mul_73, 0.1);  mul_73 = None
    mul_75: "f32[24]" = torch.ops.aten.mul.Tensor(primals_230, 0.9)
    add_55: "f32[24]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
    unsqueeze_40: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1)
    unsqueeze_41: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    mul_76: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_41);  mul_70 = unsqueeze_41 = None
    unsqueeze_42: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1);  primals_22 = None
    unsqueeze_43: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    add_56: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_43);  mul_76 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_7: "f32[8, 24, 56, 56]" = torch.ops.aten.relu.default(add_56);  add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_11: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(relu_7, primals_142, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_57: "i64[]" = torch.ops.aten.add.Tensor(primals_231, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_11 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[1, 24, 1, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 24, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_58: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 0.001)
    rsqrt_11: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_11: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_23)
    mul_77: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    squeeze_33: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
    squeeze_34: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
    mul_78: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_79: "f32[24]" = torch.ops.aten.mul.Tensor(primals_232, 0.9)
    add_59: "f32[24]" = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
    squeeze_35: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
    mul_80: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0000398612827361);  squeeze_35 = None
    mul_81: "f32[24]" = torch.ops.aten.mul.Tensor(mul_80, 0.1);  mul_80 = None
    mul_82: "f32[24]" = torch.ops.aten.mul.Tensor(primals_233, 0.9)
    add_60: "f32[24]" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
    unsqueeze_44: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_45: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_83: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
    unsqueeze_46: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_47: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_61: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_8: "f32[8, 24, 56, 56]" = torch.ops.aten.relu.default(add_61);  add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_12: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(relu_8, primals_143, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_62: "i64[]" = torch.ops.aten.add.Tensor(primals_234, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_12 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
    getitem_24: "f32[1, 24, 1, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 24, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_63: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 0.001)
    rsqrt_12: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_12: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_25)
    mul_84: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_36: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
    squeeze_37: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
    mul_85: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_86: "f32[24]" = torch.ops.aten.mul.Tensor(primals_235, 0.9)
    add_64: "f32[24]" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
    squeeze_38: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
    mul_87: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0000398612827361);  squeeze_38 = None
    mul_88: "f32[24]" = torch.ops.aten.mul.Tensor(mul_87, 0.1);  mul_87 = None
    mul_89: "f32[24]" = torch.ops.aten.mul.Tensor(primals_236, 0.9)
    add_65: "f32[24]" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
    unsqueeze_48: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1)
    unsqueeze_49: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    mul_90: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_49);  mul_84 = unsqueeze_49 = None
    unsqueeze_50: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1);  primals_26 = None
    unsqueeze_51: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    add_66: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_51);  mul_90 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_67: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(add_66, add_51);  add_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_13: "f32[8, 144, 56, 56]" = torch.ops.aten.convolution.default(add_67, primals_144, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_68: "i64[]" = torch.ops.aten.add.Tensor(primals_237, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_13 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
    getitem_26: "f32[1, 144, 1, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 144, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_69: "f32[1, 144, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 0.001)
    rsqrt_13: "f32[1, 144, 1, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
    sub_13: "f32[8, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_27)
    mul_91: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_39: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
    squeeze_40: "f32[144]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_92: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_93: "f32[144]" = torch.ops.aten.mul.Tensor(primals_238, 0.9)
    add_70: "f32[144]" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
    squeeze_41: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
    mul_94: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0000398612827361);  squeeze_41 = None
    mul_95: "f32[144]" = torch.ops.aten.mul.Tensor(mul_94, 0.1);  mul_94 = None
    mul_96: "f32[144]" = torch.ops.aten.mul.Tensor(primals_239, 0.9)
    add_71: "f32[144]" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
    unsqueeze_52: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1)
    unsqueeze_53: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_97: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
    unsqueeze_54: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1);  primals_28 = None
    unsqueeze_55: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_72: "f32[8, 144, 56, 56]" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_9: "f32[8, 144, 56, 56]" = torch.ops.aten.relu.default(add_72);  add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_14: "f32[8, 144, 28, 28]" = torch.ops.aten.convolution.default(relu_9, primals_145, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 144)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_73: "i64[]" = torch.ops.aten.add.Tensor(primals_240, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_14 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
    getitem_28: "f32[1, 144, 1, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 144, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_74: "f32[1, 144, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 0.001)
    rsqrt_14: "f32[1, 144, 1, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_14: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_29)
    mul_98: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_42: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
    squeeze_43: "f32[144]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
    mul_99: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_100: "f32[144]" = torch.ops.aten.mul.Tensor(primals_241, 0.9)
    add_75: "f32[144]" = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
    squeeze_44: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
    mul_101: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0001594642002871);  squeeze_44 = None
    mul_102: "f32[144]" = torch.ops.aten.mul.Tensor(mul_101, 0.1);  mul_101 = None
    mul_103: "f32[144]" = torch.ops.aten.mul.Tensor(primals_242, 0.9)
    add_76: "f32[144]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
    unsqueeze_56: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_57: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    mul_104: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
    unsqueeze_58: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_59: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    add_77: "f32[8, 144, 28, 28]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_10: "f32[8, 144, 28, 28]" = torch.ops.aten.relu.default(add_77);  add_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_15: "f32[8, 32, 28, 28]" = torch.ops.aten.convolution.default(relu_10, primals_146, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_78: "i64[]" = torch.ops.aten.add.Tensor(primals_243, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_15 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
    getitem_30: "f32[1, 32, 1, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 32, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_79: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 0.001)
    rsqrt_15: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
    sub_15: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_31)
    mul_105: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    squeeze_45: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
    squeeze_46: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
    mul_106: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_107: "f32[32]" = torch.ops.aten.mul.Tensor(primals_244, 0.9)
    add_80: "f32[32]" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
    squeeze_47: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
    mul_108: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0001594642002871);  squeeze_47 = None
    mul_109: "f32[32]" = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
    mul_110: "f32[32]" = torch.ops.aten.mul.Tensor(primals_245, 0.9)
    add_81: "f32[32]" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
    unsqueeze_60: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1)
    unsqueeze_61: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_111: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_61);  mul_105 = unsqueeze_61 = None
    unsqueeze_62: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1);  primals_32 = None
    unsqueeze_63: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_82: "f32[8, 32, 28, 28]" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_63);  mul_111 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_16: "f32[8, 96, 28, 28]" = torch.ops.aten.convolution.default(add_82, primals_147, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_83: "i64[]" = torch.ops.aten.add.Tensor(primals_246, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_16 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
    getitem_32: "f32[1, 96, 1, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 96, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_84: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 0.001)
    rsqrt_16: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_16: "f32[8, 96, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_33)
    mul_112: "f32[8, 96, 28, 28]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_48: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
    squeeze_49: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
    mul_113: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_114: "f32[96]" = torch.ops.aten.mul.Tensor(primals_247, 0.9)
    add_85: "f32[96]" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
    squeeze_50: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
    mul_115: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0001594642002871);  squeeze_50 = None
    mul_116: "f32[96]" = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
    mul_117: "f32[96]" = torch.ops.aten.mul.Tensor(primals_248, 0.9)
    add_86: "f32[96]" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
    unsqueeze_64: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1)
    unsqueeze_65: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    mul_118: "f32[8, 96, 28, 28]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
    unsqueeze_66: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1);  primals_34 = None
    unsqueeze_67: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    add_87: "f32[8, 96, 28, 28]" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_11: "f32[8, 96, 28, 28]" = torch.ops.aten.relu.default(add_87);  add_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_17: "f32[8, 96, 28, 28]" = torch.ops.aten.convolution.default(relu_11, primals_148, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 96)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_88: "i64[]" = torch.ops.aten.add.Tensor(primals_249, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_17 = torch.ops.aten.var_mean.correction(convolution_17, [0, 2, 3], correction = 0, keepdim = True)
    getitem_34: "f32[1, 96, 1, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 96, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_89: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 0.001)
    rsqrt_17: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
    sub_17: "f32[8, 96, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, getitem_35)
    mul_119: "f32[8, 96, 28, 28]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    squeeze_51: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
    squeeze_52: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
    mul_120: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_121: "f32[96]" = torch.ops.aten.mul.Tensor(primals_250, 0.9)
    add_90: "f32[96]" = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
    squeeze_53: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
    mul_122: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0001594642002871);  squeeze_53 = None
    mul_123: "f32[96]" = torch.ops.aten.mul.Tensor(mul_122, 0.1);  mul_122 = None
    mul_124: "f32[96]" = torch.ops.aten.mul.Tensor(primals_251, 0.9)
    add_91: "f32[96]" = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
    unsqueeze_68: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_69: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_125: "f32[8, 96, 28, 28]" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_69);  mul_119 = unsqueeze_69 = None
    unsqueeze_70: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_71: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_92: "f32[8, 96, 28, 28]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_71);  mul_125 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_12: "f32[8, 96, 28, 28]" = torch.ops.aten.relu.default(add_92);  add_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_18: "f32[8, 32, 28, 28]" = torch.ops.aten.convolution.default(relu_12, primals_149, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_93: "i64[]" = torch.ops.aten.add.Tensor(primals_252, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_18 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
    getitem_36: "f32[1, 32, 1, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 32, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_94: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 0.001)
    rsqrt_18: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    sub_18: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_37)
    mul_126: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    squeeze_54: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
    squeeze_55: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
    mul_127: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_128: "f32[32]" = torch.ops.aten.mul.Tensor(primals_253, 0.9)
    add_95: "f32[32]" = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
    squeeze_56: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
    mul_129: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0001594642002871);  squeeze_56 = None
    mul_130: "f32[32]" = torch.ops.aten.mul.Tensor(mul_129, 0.1);  mul_129 = None
    mul_131: "f32[32]" = torch.ops.aten.mul.Tensor(primals_254, 0.9)
    add_96: "f32[32]" = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
    unsqueeze_72: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1)
    unsqueeze_73: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    mul_132: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_73);  mul_126 = unsqueeze_73 = None
    unsqueeze_74: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1);  primals_38 = None
    unsqueeze_75: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    add_97: "f32[8, 32, 28, 28]" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_75);  mul_132 = unsqueeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_98: "f32[8, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_97, add_82);  add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_19: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(add_98, primals_150, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_99: "i64[]" = torch.ops.aten.add.Tensor(primals_255, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_19 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
    getitem_38: "f32[1, 192, 1, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 192, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_100: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 0.001)
    rsqrt_19: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
    sub_19: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_39)
    mul_133: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    squeeze_57: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
    squeeze_58: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
    mul_134: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_135: "f32[192]" = torch.ops.aten.mul.Tensor(primals_256, 0.9)
    add_101: "f32[192]" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
    squeeze_59: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
    mul_136: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0001594642002871);  squeeze_59 = None
    mul_137: "f32[192]" = torch.ops.aten.mul.Tensor(mul_136, 0.1);  mul_136 = None
    mul_138: "f32[192]" = torch.ops.aten.mul.Tensor(primals_257, 0.9)
    add_102: "f32[192]" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
    unsqueeze_76: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1)
    unsqueeze_77: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_139: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_77);  mul_133 = unsqueeze_77 = None
    unsqueeze_78: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1);  primals_40 = None
    unsqueeze_79: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_103: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_139, unsqueeze_79);  mul_139 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_13: "f32[8, 192, 28, 28]" = torch.ops.aten.relu.default(add_103);  add_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_20: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(relu_13, primals_151, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 192)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_104: "i64[]" = torch.ops.aten.add.Tensor(primals_258, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_20 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
    getitem_40: "f32[1, 192, 1, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 192, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_105: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 0.001)
    rsqrt_20: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    sub_20: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_41)
    mul_140: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    squeeze_60: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
    squeeze_61: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
    mul_141: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_142: "f32[192]" = torch.ops.aten.mul.Tensor(primals_259, 0.9)
    add_106: "f32[192]" = torch.ops.aten.add.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
    squeeze_62: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
    mul_143: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0001594642002871);  squeeze_62 = None
    mul_144: "f32[192]" = torch.ops.aten.mul.Tensor(mul_143, 0.1);  mul_143 = None
    mul_145: "f32[192]" = torch.ops.aten.mul.Tensor(primals_260, 0.9)
    add_107: "f32[192]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
    unsqueeze_80: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_81: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    mul_146: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_140, unsqueeze_81);  mul_140 = unsqueeze_81 = None
    unsqueeze_82: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_83: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    add_108: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_83);  mul_146 = unsqueeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_14: "f32[8, 192, 28, 28]" = torch.ops.aten.relu.default(add_108);  add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_21: "f32[8, 32, 28, 28]" = torch.ops.aten.convolution.default(relu_14, primals_152, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_109: "i64[]" = torch.ops.aten.add.Tensor(primals_261, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_21 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
    getitem_42: "f32[1, 32, 1, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 32, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_110: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 0.001)
    rsqrt_21: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
    sub_21: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_43)
    mul_147: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    squeeze_63: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
    squeeze_64: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
    mul_148: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_149: "f32[32]" = torch.ops.aten.mul.Tensor(primals_262, 0.9)
    add_111: "f32[32]" = torch.ops.aten.add.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
    squeeze_65: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
    mul_150: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0001594642002871);  squeeze_65 = None
    mul_151: "f32[32]" = torch.ops.aten.mul.Tensor(mul_150, 0.1);  mul_150 = None
    mul_152: "f32[32]" = torch.ops.aten.mul.Tensor(primals_263, 0.9)
    add_112: "f32[32]" = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
    unsqueeze_84: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1)
    unsqueeze_85: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_153: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(mul_147, unsqueeze_85);  mul_147 = unsqueeze_85 = None
    unsqueeze_86: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1);  primals_44 = None
    unsqueeze_87: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_113: "f32[8, 32, 28, 28]" = torch.ops.aten.add.Tensor(mul_153, unsqueeze_87);  mul_153 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_114: "f32[8, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_113, add_98);  add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_22: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(add_114, primals_153, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_115: "i64[]" = torch.ops.aten.add.Tensor(primals_264, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_22 = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
    getitem_44: "f32[1, 192, 1, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 192, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_116: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 0.001)
    rsqrt_22: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
    sub_22: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, getitem_45)
    mul_154: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    squeeze_66: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
    squeeze_67: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
    mul_155: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_156: "f32[192]" = torch.ops.aten.mul.Tensor(primals_265, 0.9)
    add_117: "f32[192]" = torch.ops.aten.add.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
    squeeze_68: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
    mul_157: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0001594642002871);  squeeze_68 = None
    mul_158: "f32[192]" = torch.ops.aten.mul.Tensor(mul_157, 0.1);  mul_157 = None
    mul_159: "f32[192]" = torch.ops.aten.mul.Tensor(primals_266, 0.9)
    add_118: "f32[192]" = torch.ops.aten.add.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
    unsqueeze_88: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1)
    unsqueeze_89: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    mul_160: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_89);  mul_154 = unsqueeze_89 = None
    unsqueeze_90: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1);  primals_46 = None
    unsqueeze_91: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    add_119: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_160, unsqueeze_91);  mul_160 = unsqueeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_15: "f32[8, 192, 28, 28]" = torch.ops.aten.relu.default(add_119);  add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_23: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(relu_15, primals_154, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 192)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_120: "i64[]" = torch.ops.aten.add.Tensor(primals_267, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_23 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
    getitem_46: "f32[1, 192, 1, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 192, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_121: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 0.001)
    rsqrt_23: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
    sub_23: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_47)
    mul_161: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    squeeze_69: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
    squeeze_70: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
    mul_162: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
    mul_163: "f32[192]" = torch.ops.aten.mul.Tensor(primals_268, 0.9)
    add_122: "f32[192]" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
    squeeze_71: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
    mul_164: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0001594642002871);  squeeze_71 = None
    mul_165: "f32[192]" = torch.ops.aten.mul.Tensor(mul_164, 0.1);  mul_164 = None
    mul_166: "f32[192]" = torch.ops.aten.mul.Tensor(primals_269, 0.9)
    add_123: "f32[192]" = torch.ops.aten.add.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
    unsqueeze_92: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_93: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_167: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_93);  mul_161 = unsqueeze_93 = None
    unsqueeze_94: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_95: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_124: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_95);  mul_167 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_16: "f32[8, 192, 28, 28]" = torch.ops.aten.relu.default(add_124);  add_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_24: "f32[8, 32, 28, 28]" = torch.ops.aten.convolution.default(relu_16, primals_155, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_125: "i64[]" = torch.ops.aten.add.Tensor(primals_270, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_24 = torch.ops.aten.var_mean.correction(convolution_24, [0, 2, 3], correction = 0, keepdim = True)
    getitem_48: "f32[1, 32, 1, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 32, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_126: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_48, 0.001)
    rsqrt_24: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    sub_24: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, getitem_49)
    mul_168: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    squeeze_72: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
    squeeze_73: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
    mul_169: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_170: "f32[32]" = torch.ops.aten.mul.Tensor(primals_271, 0.9)
    add_127: "f32[32]" = torch.ops.aten.add.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
    squeeze_74: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
    mul_171: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0001594642002871);  squeeze_74 = None
    mul_172: "f32[32]" = torch.ops.aten.mul.Tensor(mul_171, 0.1);  mul_171 = None
    mul_173: "f32[32]" = torch.ops.aten.mul.Tensor(primals_272, 0.9)
    add_128: "f32[32]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
    unsqueeze_96: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1)
    unsqueeze_97: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    mul_174: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(mul_168, unsqueeze_97);  mul_168 = unsqueeze_97 = None
    unsqueeze_98: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1);  primals_50 = None
    unsqueeze_99: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    add_129: "f32[8, 32, 28, 28]" = torch.ops.aten.add.Tensor(mul_174, unsqueeze_99);  mul_174 = unsqueeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_130: "f32[8, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_129, add_114);  add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_25: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(add_130, primals_156, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_131: "i64[]" = torch.ops.aten.add.Tensor(primals_273, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_25 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
    getitem_50: "f32[1, 192, 1, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 192, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_132: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 0.001)
    rsqrt_25: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
    sub_25: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_51)
    mul_175: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    squeeze_75: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
    squeeze_76: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
    mul_176: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_177: "f32[192]" = torch.ops.aten.mul.Tensor(primals_274, 0.9)
    add_133: "f32[192]" = torch.ops.aten.add.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
    squeeze_77: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
    mul_178: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0001594642002871);  squeeze_77 = None
    mul_179: "f32[192]" = torch.ops.aten.mul.Tensor(mul_178, 0.1);  mul_178 = None
    mul_180: "f32[192]" = torch.ops.aten.mul.Tensor(primals_275, 0.9)
    add_134: "f32[192]" = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
    unsqueeze_100: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1)
    unsqueeze_101: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_181: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_101);  mul_175 = unsqueeze_101 = None
    unsqueeze_102: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1);  primals_52 = None
    unsqueeze_103: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_135: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_103);  mul_181 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_17: "f32[8, 192, 28, 28]" = torch.ops.aten.relu.default(add_135);  add_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_26: "f32[8, 192, 14, 14]" = torch.ops.aten.convolution.default(relu_17, primals_157, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 192)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_136: "i64[]" = torch.ops.aten.add.Tensor(primals_276, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_26 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
    getitem_52: "f32[1, 192, 1, 1]" = var_mean_26[0]
    getitem_53: "f32[1, 192, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_137: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_52, 0.001)
    rsqrt_26: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
    sub_26: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_53)
    mul_182: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    squeeze_78: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
    squeeze_79: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
    mul_183: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_184: "f32[192]" = torch.ops.aten.mul.Tensor(primals_277, 0.9)
    add_138: "f32[192]" = torch.ops.aten.add.Tensor(mul_183, mul_184);  mul_183 = mul_184 = None
    squeeze_80: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
    mul_185: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0006381620931717);  squeeze_80 = None
    mul_186: "f32[192]" = torch.ops.aten.mul.Tensor(mul_185, 0.1);  mul_185 = None
    mul_187: "f32[192]" = torch.ops.aten.mul.Tensor(primals_278, 0.9)
    add_139: "f32[192]" = torch.ops.aten.add.Tensor(mul_186, mul_187);  mul_186 = mul_187 = None
    unsqueeze_104: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_105: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    mul_188: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_105);  mul_182 = unsqueeze_105 = None
    unsqueeze_106: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_107: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    add_140: "f32[8, 192, 14, 14]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_107);  mul_188 = unsqueeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_18: "f32[8, 192, 14, 14]" = torch.ops.aten.relu.default(add_140);  add_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_27: "f32[8, 64, 14, 14]" = torch.ops.aten.convolution.default(relu_18, primals_158, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_141: "i64[]" = torch.ops.aten.add.Tensor(primals_279, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_27 = torch.ops.aten.var_mean.correction(convolution_27, [0, 2, 3], correction = 0, keepdim = True)
    getitem_54: "f32[1, 64, 1, 1]" = var_mean_27[0]
    getitem_55: "f32[1, 64, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_142: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 0.001)
    rsqrt_27: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
    sub_27: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, getitem_55)
    mul_189: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    squeeze_81: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
    squeeze_82: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
    mul_190: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
    mul_191: "f32[64]" = torch.ops.aten.mul.Tensor(primals_280, 0.9)
    add_143: "f32[64]" = torch.ops.aten.add.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
    squeeze_83: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
    mul_192: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0006381620931717);  squeeze_83 = None
    mul_193: "f32[64]" = torch.ops.aten.mul.Tensor(mul_192, 0.1);  mul_192 = None
    mul_194: "f32[64]" = torch.ops.aten.mul.Tensor(primals_281, 0.9)
    add_144: "f32[64]" = torch.ops.aten.add.Tensor(mul_193, mul_194);  mul_193 = mul_194 = None
    unsqueeze_108: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1)
    unsqueeze_109: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_195: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(mul_189, unsqueeze_109);  mul_189 = unsqueeze_109 = None
    unsqueeze_110: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1);  primals_56 = None
    unsqueeze_111: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_145: "f32[8, 64, 14, 14]" = torch.ops.aten.add.Tensor(mul_195, unsqueeze_111);  mul_195 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_28: "f32[8, 192, 14, 14]" = torch.ops.aten.convolution.default(add_145, primals_159, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_146: "i64[]" = torch.ops.aten.add.Tensor(primals_282, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_28 = torch.ops.aten.var_mean.correction(convolution_28, [0, 2, 3], correction = 0, keepdim = True)
    getitem_56: "f32[1, 192, 1, 1]" = var_mean_28[0]
    getitem_57: "f32[1, 192, 1, 1]" = var_mean_28[1];  var_mean_28 = None
    add_147: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_56, 0.001)
    rsqrt_28: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
    sub_28: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, getitem_57)
    mul_196: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    squeeze_84: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
    squeeze_85: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
    mul_197: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
    mul_198: "f32[192]" = torch.ops.aten.mul.Tensor(primals_283, 0.9)
    add_148: "f32[192]" = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
    squeeze_86: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
    mul_199: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0006381620931717);  squeeze_86 = None
    mul_200: "f32[192]" = torch.ops.aten.mul.Tensor(mul_199, 0.1);  mul_199 = None
    mul_201: "f32[192]" = torch.ops.aten.mul.Tensor(primals_284, 0.9)
    add_149: "f32[192]" = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
    unsqueeze_112: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1)
    unsqueeze_113: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    mul_202: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_113);  mul_196 = unsqueeze_113 = None
    unsqueeze_114: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_58, -1);  primals_58 = None
    unsqueeze_115: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    add_150: "f32[8, 192, 14, 14]" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_115);  mul_202 = unsqueeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_19: "f32[8, 192, 14, 14]" = torch.ops.aten.relu.default(add_150);  add_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_29: "f32[8, 192, 14, 14]" = torch.ops.aten.convolution.default(relu_19, primals_160, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 192)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_151: "i64[]" = torch.ops.aten.add.Tensor(primals_285, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_29 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
    getitem_58: "f32[1, 192, 1, 1]" = var_mean_29[0]
    getitem_59: "f32[1, 192, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_152: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 0.001)
    rsqrt_29: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
    sub_29: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, getitem_59)
    mul_203: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    squeeze_87: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
    squeeze_88: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
    mul_204: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
    mul_205: "f32[192]" = torch.ops.aten.mul.Tensor(primals_286, 0.9)
    add_153: "f32[192]" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
    squeeze_89: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
    mul_206: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0006381620931717);  squeeze_89 = None
    mul_207: "f32[192]" = torch.ops.aten.mul.Tensor(mul_206, 0.1);  mul_206 = None
    mul_208: "f32[192]" = torch.ops.aten.mul.Tensor(primals_287, 0.9)
    add_154: "f32[192]" = torch.ops.aten.add.Tensor(mul_207, mul_208);  mul_207 = mul_208 = None
    unsqueeze_116: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_117: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_209: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_117);  mul_203 = unsqueeze_117 = None
    unsqueeze_118: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_119: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_155: "f32[8, 192, 14, 14]" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_119);  mul_209 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_20: "f32[8, 192, 14, 14]" = torch.ops.aten.relu.default(add_155);  add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_30: "f32[8, 64, 14, 14]" = torch.ops.aten.convolution.default(relu_20, primals_161, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_156: "i64[]" = torch.ops.aten.add.Tensor(primals_288, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_30 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
    getitem_60: "f32[1, 64, 1, 1]" = var_mean_30[0]
    getitem_61: "f32[1, 64, 1, 1]" = var_mean_30[1];  var_mean_30 = None
    add_157: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 0.001)
    rsqrt_30: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
    sub_30: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, getitem_61)
    mul_210: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    squeeze_90: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
    squeeze_91: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
    mul_211: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
    mul_212: "f32[64]" = torch.ops.aten.mul.Tensor(primals_289, 0.9)
    add_158: "f32[64]" = torch.ops.aten.add.Tensor(mul_211, mul_212);  mul_211 = mul_212 = None
    squeeze_92: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
    mul_213: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0006381620931717);  squeeze_92 = None
    mul_214: "f32[64]" = torch.ops.aten.mul.Tensor(mul_213, 0.1);  mul_213 = None
    mul_215: "f32[64]" = torch.ops.aten.mul.Tensor(primals_290, 0.9)
    add_159: "f32[64]" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
    unsqueeze_120: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_61, -1)
    unsqueeze_121: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    mul_216: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_121);  mul_210 = unsqueeze_121 = None
    unsqueeze_122: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1);  primals_62 = None
    unsqueeze_123: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    add_160: "f32[8, 64, 14, 14]" = torch.ops.aten.add.Tensor(mul_216, unsqueeze_123);  mul_216 = unsqueeze_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_161: "f32[8, 64, 14, 14]" = torch.ops.aten.add.Tensor(add_160, add_145);  add_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_31: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(add_161, primals_162, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_162: "i64[]" = torch.ops.aten.add.Tensor(primals_291, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_31 = torch.ops.aten.var_mean.correction(convolution_31, [0, 2, 3], correction = 0, keepdim = True)
    getitem_62: "f32[1, 384, 1, 1]" = var_mean_31[0]
    getitem_63: "f32[1, 384, 1, 1]" = var_mean_31[1];  var_mean_31 = None
    add_163: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 0.001)
    rsqrt_31: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
    sub_31: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, getitem_63)
    mul_217: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    squeeze_93: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
    squeeze_94: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
    mul_218: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
    mul_219: "f32[384]" = torch.ops.aten.mul.Tensor(primals_292, 0.9)
    add_164: "f32[384]" = torch.ops.aten.add.Tensor(mul_218, mul_219);  mul_218 = mul_219 = None
    squeeze_95: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
    mul_220: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0006381620931717);  squeeze_95 = None
    mul_221: "f32[384]" = torch.ops.aten.mul.Tensor(mul_220, 0.1);  mul_220 = None
    mul_222: "f32[384]" = torch.ops.aten.mul.Tensor(primals_293, 0.9)
    add_165: "f32[384]" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
    unsqueeze_124: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1)
    unsqueeze_125: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_223: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_125);  mul_217 = unsqueeze_125 = None
    unsqueeze_126: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_64, -1);  primals_64 = None
    unsqueeze_127: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_166: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_223, unsqueeze_127);  mul_223 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_21: "f32[8, 384, 14, 14]" = torch.ops.aten.relu.default(add_166);  add_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_32: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(relu_21, primals_163, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 384)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_167: "i64[]" = torch.ops.aten.add.Tensor(primals_294, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_32 = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
    getitem_64: "f32[1, 384, 1, 1]" = var_mean_32[0]
    getitem_65: "f32[1, 384, 1, 1]" = var_mean_32[1];  var_mean_32 = None
    add_168: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 0.001)
    rsqrt_32: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    sub_32: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, getitem_65)
    mul_224: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    squeeze_96: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
    squeeze_97: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
    mul_225: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
    mul_226: "f32[384]" = torch.ops.aten.mul.Tensor(primals_295, 0.9)
    add_169: "f32[384]" = torch.ops.aten.add.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
    squeeze_98: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
    mul_227: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0006381620931717);  squeeze_98 = None
    mul_228: "f32[384]" = torch.ops.aten.mul.Tensor(mul_227, 0.1);  mul_227 = None
    mul_229: "f32[384]" = torch.ops.aten.mul.Tensor(primals_296, 0.9)
    add_170: "f32[384]" = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
    unsqueeze_128: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
    unsqueeze_129: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    mul_230: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_224, unsqueeze_129);  mul_224 = unsqueeze_129 = None
    unsqueeze_130: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
    unsqueeze_131: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    add_171: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_131);  mul_230 = unsqueeze_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_22: "f32[8, 384, 14, 14]" = torch.ops.aten.relu.default(add_171);  add_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_33: "f32[8, 64, 14, 14]" = torch.ops.aten.convolution.default(relu_22, primals_164, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_172: "i64[]" = torch.ops.aten.add.Tensor(primals_297, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_33 = torch.ops.aten.var_mean.correction(convolution_33, [0, 2, 3], correction = 0, keepdim = True)
    getitem_66: "f32[1, 64, 1, 1]" = var_mean_33[0]
    getitem_67: "f32[1, 64, 1, 1]" = var_mean_33[1];  var_mean_33 = None
    add_173: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 0.001)
    rsqrt_33: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_173);  add_173 = None
    sub_33: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, getitem_67)
    mul_231: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    squeeze_99: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
    squeeze_100: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
    mul_232: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
    mul_233: "f32[64]" = torch.ops.aten.mul.Tensor(primals_298, 0.9)
    add_174: "f32[64]" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
    squeeze_101: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
    mul_234: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0006381620931717);  squeeze_101 = None
    mul_235: "f32[64]" = torch.ops.aten.mul.Tensor(mul_234, 0.1);  mul_234 = None
    mul_236: "f32[64]" = torch.ops.aten.mul.Tensor(primals_299, 0.9)
    add_175: "f32[64]" = torch.ops.aten.add.Tensor(mul_235, mul_236);  mul_235 = mul_236 = None
    unsqueeze_132: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_67, -1)
    unsqueeze_133: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_237: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(mul_231, unsqueeze_133);  mul_231 = unsqueeze_133 = None
    unsqueeze_134: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1);  primals_68 = None
    unsqueeze_135: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_176: "f32[8, 64, 14, 14]" = torch.ops.aten.add.Tensor(mul_237, unsqueeze_135);  mul_237 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_177: "f32[8, 64, 14, 14]" = torch.ops.aten.add.Tensor(add_176, add_161);  add_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_34: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(add_177, primals_165, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_178: "i64[]" = torch.ops.aten.add.Tensor(primals_300, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_34 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
    getitem_68: "f32[1, 384, 1, 1]" = var_mean_34[0]
    getitem_69: "f32[1, 384, 1, 1]" = var_mean_34[1];  var_mean_34 = None
    add_179: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 0.001)
    rsqrt_34: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_179);  add_179 = None
    sub_34: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_69)
    mul_238: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    squeeze_102: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
    squeeze_103: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
    mul_239: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
    mul_240: "f32[384]" = torch.ops.aten.mul.Tensor(primals_301, 0.9)
    add_180: "f32[384]" = torch.ops.aten.add.Tensor(mul_239, mul_240);  mul_239 = mul_240 = None
    squeeze_104: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
    mul_241: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0006381620931717);  squeeze_104 = None
    mul_242: "f32[384]" = torch.ops.aten.mul.Tensor(mul_241, 0.1);  mul_241 = None
    mul_243: "f32[384]" = torch.ops.aten.mul.Tensor(primals_302, 0.9)
    add_181: "f32[384]" = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
    unsqueeze_136: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1)
    unsqueeze_137: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    mul_244: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_137);  mul_238 = unsqueeze_137 = None
    unsqueeze_138: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_70, -1);  primals_70 = None
    unsqueeze_139: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    add_182: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_244, unsqueeze_139);  mul_244 = unsqueeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_23: "f32[8, 384, 14, 14]" = torch.ops.aten.relu.default(add_182);  add_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_35: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(relu_23, primals_166, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 384)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_183: "i64[]" = torch.ops.aten.add.Tensor(primals_303, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_35 = torch.ops.aten.var_mean.correction(convolution_35, [0, 2, 3], correction = 0, keepdim = True)
    getitem_70: "f32[1, 384, 1, 1]" = var_mean_35[0]
    getitem_71: "f32[1, 384, 1, 1]" = var_mean_35[1];  var_mean_35 = None
    add_184: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_70, 0.001)
    rsqrt_35: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_184);  add_184 = None
    sub_35: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, getitem_71)
    mul_245: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    squeeze_105: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
    squeeze_106: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
    mul_246: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
    mul_247: "f32[384]" = torch.ops.aten.mul.Tensor(primals_304, 0.9)
    add_185: "f32[384]" = torch.ops.aten.add.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
    squeeze_107: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
    mul_248: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0006381620931717);  squeeze_107 = None
    mul_249: "f32[384]" = torch.ops.aten.mul.Tensor(mul_248, 0.1);  mul_248 = None
    mul_250: "f32[384]" = torch.ops.aten.mul.Tensor(primals_305, 0.9)
    add_186: "f32[384]" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
    unsqueeze_140: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_141: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_251: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_141);  mul_245 = unsqueeze_141 = None
    unsqueeze_142: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_143: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_187: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_143);  mul_251 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_24: "f32[8, 384, 14, 14]" = torch.ops.aten.relu.default(add_187);  add_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_36: "f32[8, 64, 14, 14]" = torch.ops.aten.convolution.default(relu_24, primals_167, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_188: "i64[]" = torch.ops.aten.add.Tensor(primals_306, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_36 = torch.ops.aten.var_mean.correction(convolution_36, [0, 2, 3], correction = 0, keepdim = True)
    getitem_72: "f32[1, 64, 1, 1]" = var_mean_36[0]
    getitem_73: "f32[1, 64, 1, 1]" = var_mean_36[1];  var_mean_36 = None
    add_189: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_72, 0.001)
    rsqrt_36: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_189);  add_189 = None
    sub_36: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, getitem_73)
    mul_252: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    squeeze_108: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
    squeeze_109: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
    mul_253: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
    mul_254: "f32[64]" = torch.ops.aten.mul.Tensor(primals_307, 0.9)
    add_190: "f32[64]" = torch.ops.aten.add.Tensor(mul_253, mul_254);  mul_253 = mul_254 = None
    squeeze_110: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
    mul_255: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0006381620931717);  squeeze_110 = None
    mul_256: "f32[64]" = torch.ops.aten.mul.Tensor(mul_255, 0.1);  mul_255 = None
    mul_257: "f32[64]" = torch.ops.aten.mul.Tensor(primals_308, 0.9)
    add_191: "f32[64]" = torch.ops.aten.add.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
    unsqueeze_144: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_73, -1)
    unsqueeze_145: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    mul_258: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_145);  mul_252 = unsqueeze_145 = None
    unsqueeze_146: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1);  primals_74 = None
    unsqueeze_147: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    add_192: "f32[8, 64, 14, 14]" = torch.ops.aten.add.Tensor(mul_258, unsqueeze_147);  mul_258 = unsqueeze_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_193: "f32[8, 64, 14, 14]" = torch.ops.aten.add.Tensor(add_192, add_177);  add_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_37: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(add_193, primals_168, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_194: "i64[]" = torch.ops.aten.add.Tensor(primals_309, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_37 = torch.ops.aten.var_mean.correction(convolution_37, [0, 2, 3], correction = 0, keepdim = True)
    getitem_74: "f32[1, 384, 1, 1]" = var_mean_37[0]
    getitem_75: "f32[1, 384, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_195: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_74, 0.001)
    rsqrt_37: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_195);  add_195 = None
    sub_37: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, getitem_75)
    mul_259: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    squeeze_111: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
    squeeze_112: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
    mul_260: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
    mul_261: "f32[384]" = torch.ops.aten.mul.Tensor(primals_310, 0.9)
    add_196: "f32[384]" = torch.ops.aten.add.Tensor(mul_260, mul_261);  mul_260 = mul_261 = None
    squeeze_113: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
    mul_262: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0006381620931717);  squeeze_113 = None
    mul_263: "f32[384]" = torch.ops.aten.mul.Tensor(mul_262, 0.1);  mul_262 = None
    mul_264: "f32[384]" = torch.ops.aten.mul.Tensor(primals_311, 0.9)
    add_197: "f32[384]" = torch.ops.aten.add.Tensor(mul_263, mul_264);  mul_263 = mul_264 = None
    unsqueeze_148: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1)
    unsqueeze_149: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_265: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_149);  mul_259 = unsqueeze_149 = None
    unsqueeze_150: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_76, -1);  primals_76 = None
    unsqueeze_151: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_198: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_265, unsqueeze_151);  mul_265 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_25: "f32[8, 384, 14, 14]" = torch.ops.aten.relu.default(add_198);  add_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_38: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(relu_25, primals_169, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 384)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_199: "i64[]" = torch.ops.aten.add.Tensor(primals_312, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_38 = torch.ops.aten.var_mean.correction(convolution_38, [0, 2, 3], correction = 0, keepdim = True)
    getitem_76: "f32[1, 384, 1, 1]" = var_mean_38[0]
    getitem_77: "f32[1, 384, 1, 1]" = var_mean_38[1];  var_mean_38 = None
    add_200: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_76, 0.001)
    rsqrt_38: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_200);  add_200 = None
    sub_38: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_38, getitem_77)
    mul_266: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    squeeze_114: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
    squeeze_115: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
    mul_267: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
    mul_268: "f32[384]" = torch.ops.aten.mul.Tensor(primals_313, 0.9)
    add_201: "f32[384]" = torch.ops.aten.add.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
    squeeze_116: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
    mul_269: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0006381620931717);  squeeze_116 = None
    mul_270: "f32[384]" = torch.ops.aten.mul.Tensor(mul_269, 0.1);  mul_269 = None
    mul_271: "f32[384]" = torch.ops.aten.mul.Tensor(primals_314, 0.9)
    add_202: "f32[384]" = torch.ops.aten.add.Tensor(mul_270, mul_271);  mul_270 = mul_271 = None
    unsqueeze_152: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
    unsqueeze_153: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    mul_272: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_266, unsqueeze_153);  mul_266 = unsqueeze_153 = None
    unsqueeze_154: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
    unsqueeze_155: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    add_203: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_272, unsqueeze_155);  mul_272 = unsqueeze_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_26: "f32[8, 384, 14, 14]" = torch.ops.aten.relu.default(add_203);  add_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_39: "f32[8, 112, 14, 14]" = torch.ops.aten.convolution.default(relu_26, primals_170, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_204: "i64[]" = torch.ops.aten.add.Tensor(primals_315, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_39 = torch.ops.aten.var_mean.correction(convolution_39, [0, 2, 3], correction = 0, keepdim = True)
    getitem_78: "f32[1, 112, 1, 1]" = var_mean_39[0]
    getitem_79: "f32[1, 112, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_205: "f32[1, 112, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 0.001)
    rsqrt_39: "f32[1, 112, 1, 1]" = torch.ops.aten.rsqrt.default(add_205);  add_205 = None
    sub_39: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, getitem_79)
    mul_273: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    squeeze_117: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
    squeeze_118: "f32[112]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
    mul_274: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
    mul_275: "f32[112]" = torch.ops.aten.mul.Tensor(primals_316, 0.9)
    add_206: "f32[112]" = torch.ops.aten.add.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None
    squeeze_119: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
    mul_276: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0006381620931717);  squeeze_119 = None
    mul_277: "f32[112]" = torch.ops.aten.mul.Tensor(mul_276, 0.1);  mul_276 = None
    mul_278: "f32[112]" = torch.ops.aten.mul.Tensor(primals_317, 0.9)
    add_207: "f32[112]" = torch.ops.aten.add.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
    unsqueeze_156: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_79, -1)
    unsqueeze_157: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_279: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(mul_273, unsqueeze_157);  mul_273 = unsqueeze_157 = None
    unsqueeze_158: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1);  primals_80 = None
    unsqueeze_159: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_208: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(mul_279, unsqueeze_159);  mul_279 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_40: "f32[8, 672, 14, 14]" = torch.ops.aten.convolution.default(add_208, primals_171, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_209: "i64[]" = torch.ops.aten.add.Tensor(primals_318, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_40 = torch.ops.aten.var_mean.correction(convolution_40, [0, 2, 3], correction = 0, keepdim = True)
    getitem_80: "f32[1, 672, 1, 1]" = var_mean_40[0]
    getitem_81: "f32[1, 672, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_210: "f32[1, 672, 1, 1]" = torch.ops.aten.add.Tensor(getitem_80, 0.001)
    rsqrt_40: "f32[1, 672, 1, 1]" = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
    sub_40: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_81)
    mul_280: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    squeeze_120: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
    squeeze_121: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
    mul_281: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
    mul_282: "f32[672]" = torch.ops.aten.mul.Tensor(primals_319, 0.9)
    add_211: "f32[672]" = torch.ops.aten.add.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
    squeeze_122: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
    mul_283: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0006381620931717);  squeeze_122 = None
    mul_284: "f32[672]" = torch.ops.aten.mul.Tensor(mul_283, 0.1);  mul_283 = None
    mul_285: "f32[672]" = torch.ops.aten.mul.Tensor(primals_320, 0.9)
    add_212: "f32[672]" = torch.ops.aten.add.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
    unsqueeze_160: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1)
    unsqueeze_161: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    mul_286: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_161);  mul_280 = unsqueeze_161 = None
    unsqueeze_162: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_82, -1);  primals_82 = None
    unsqueeze_163: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    add_213: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_286, unsqueeze_163);  mul_286 = unsqueeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_27: "f32[8, 672, 14, 14]" = torch.ops.aten.relu.default(add_213);  add_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_41: "f32[8, 672, 14, 14]" = torch.ops.aten.convolution.default(relu_27, primals_172, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 672)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_214: "i64[]" = torch.ops.aten.add.Tensor(primals_321, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_41 = torch.ops.aten.var_mean.correction(convolution_41, [0, 2, 3], correction = 0, keepdim = True)
    getitem_82: "f32[1, 672, 1, 1]" = var_mean_41[0]
    getitem_83: "f32[1, 672, 1, 1]" = var_mean_41[1];  var_mean_41 = None
    add_215: "f32[1, 672, 1, 1]" = torch.ops.aten.add.Tensor(getitem_82, 0.001)
    rsqrt_41: "f32[1, 672, 1, 1]" = torch.ops.aten.rsqrt.default(add_215);  add_215 = None
    sub_41: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, getitem_83)
    mul_287: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
    squeeze_123: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
    squeeze_124: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
    mul_288: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
    mul_289: "f32[672]" = torch.ops.aten.mul.Tensor(primals_322, 0.9)
    add_216: "f32[672]" = torch.ops.aten.add.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
    squeeze_125: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
    mul_290: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0006381620931717);  squeeze_125 = None
    mul_291: "f32[672]" = torch.ops.aten.mul.Tensor(mul_290, 0.1);  mul_290 = None
    mul_292: "f32[672]" = torch.ops.aten.mul.Tensor(primals_323, 0.9)
    add_217: "f32[672]" = torch.ops.aten.add.Tensor(mul_291, mul_292);  mul_291 = mul_292 = None
    unsqueeze_164: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1)
    unsqueeze_165: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_293: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_287, unsqueeze_165);  mul_287 = unsqueeze_165 = None
    unsqueeze_166: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
    unsqueeze_167: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_218: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_293, unsqueeze_167);  mul_293 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_28: "f32[8, 672, 14, 14]" = torch.ops.aten.relu.default(add_218);  add_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_42: "f32[8, 112, 14, 14]" = torch.ops.aten.convolution.default(relu_28, primals_173, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_219: "i64[]" = torch.ops.aten.add.Tensor(primals_324, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_42 = torch.ops.aten.var_mean.correction(convolution_42, [0, 2, 3], correction = 0, keepdim = True)
    getitem_84: "f32[1, 112, 1, 1]" = var_mean_42[0]
    getitem_85: "f32[1, 112, 1, 1]" = var_mean_42[1];  var_mean_42 = None
    add_220: "f32[1, 112, 1, 1]" = torch.ops.aten.add.Tensor(getitem_84, 0.001)
    rsqrt_42: "f32[1, 112, 1, 1]" = torch.ops.aten.rsqrt.default(add_220);  add_220 = None
    sub_42: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, getitem_85)
    mul_294: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
    squeeze_126: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
    squeeze_127: "f32[112]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
    mul_295: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
    mul_296: "f32[112]" = torch.ops.aten.mul.Tensor(primals_325, 0.9)
    add_221: "f32[112]" = torch.ops.aten.add.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None
    squeeze_128: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
    mul_297: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0006381620931717);  squeeze_128 = None
    mul_298: "f32[112]" = torch.ops.aten.mul.Tensor(mul_297, 0.1);  mul_297 = None
    mul_299: "f32[112]" = torch.ops.aten.mul.Tensor(primals_326, 0.9)
    add_222: "f32[112]" = torch.ops.aten.add.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
    unsqueeze_168: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_85, -1)
    unsqueeze_169: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    mul_300: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(mul_294, unsqueeze_169);  mul_294 = unsqueeze_169 = None
    unsqueeze_170: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1);  primals_86 = None
    unsqueeze_171: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    add_223: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(mul_300, unsqueeze_171);  mul_300 = unsqueeze_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_224: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(add_223, add_208);  add_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_43: "f32[8, 672, 14, 14]" = torch.ops.aten.convolution.default(add_224, primals_174, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_225: "i64[]" = torch.ops.aten.add.Tensor(primals_327, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_43 = torch.ops.aten.var_mean.correction(convolution_43, [0, 2, 3], correction = 0, keepdim = True)
    getitem_86: "f32[1, 672, 1, 1]" = var_mean_43[0]
    getitem_87: "f32[1, 672, 1, 1]" = var_mean_43[1];  var_mean_43 = None
    add_226: "f32[1, 672, 1, 1]" = torch.ops.aten.add.Tensor(getitem_86, 0.001)
    rsqrt_43: "f32[1, 672, 1, 1]" = torch.ops.aten.rsqrt.default(add_226);  add_226 = None
    sub_43: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, getitem_87)
    mul_301: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
    squeeze_129: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
    squeeze_130: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
    mul_302: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
    mul_303: "f32[672]" = torch.ops.aten.mul.Tensor(primals_328, 0.9)
    add_227: "f32[672]" = torch.ops.aten.add.Tensor(mul_302, mul_303);  mul_302 = mul_303 = None
    squeeze_131: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
    mul_304: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0006381620931717);  squeeze_131 = None
    mul_305: "f32[672]" = torch.ops.aten.mul.Tensor(mul_304, 0.1);  mul_304 = None
    mul_306: "f32[672]" = torch.ops.aten.mul.Tensor(primals_329, 0.9)
    add_228: "f32[672]" = torch.ops.aten.add.Tensor(mul_305, mul_306);  mul_305 = mul_306 = None
    unsqueeze_172: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1)
    unsqueeze_173: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_307: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_173);  mul_301 = unsqueeze_173 = None
    unsqueeze_174: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_88, -1);  primals_88 = None
    unsqueeze_175: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_229: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_307, unsqueeze_175);  mul_307 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_29: "f32[8, 672, 14, 14]" = torch.ops.aten.relu.default(add_229);  add_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_44: "f32[8, 672, 14, 14]" = torch.ops.aten.convolution.default(relu_29, primals_175, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 672)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_230: "i64[]" = torch.ops.aten.add.Tensor(primals_330, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_44 = torch.ops.aten.var_mean.correction(convolution_44, [0, 2, 3], correction = 0, keepdim = True)
    getitem_88: "f32[1, 672, 1, 1]" = var_mean_44[0]
    getitem_89: "f32[1, 672, 1, 1]" = var_mean_44[1];  var_mean_44 = None
    add_231: "f32[1, 672, 1, 1]" = torch.ops.aten.add.Tensor(getitem_88, 0.001)
    rsqrt_44: "f32[1, 672, 1, 1]" = torch.ops.aten.rsqrt.default(add_231);  add_231 = None
    sub_44: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, getitem_89)
    mul_308: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
    squeeze_132: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
    squeeze_133: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
    mul_309: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1)
    mul_310: "f32[672]" = torch.ops.aten.mul.Tensor(primals_331, 0.9)
    add_232: "f32[672]" = torch.ops.aten.add.Tensor(mul_309, mul_310);  mul_309 = mul_310 = None
    squeeze_134: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
    mul_311: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_134, 1.0006381620931717);  squeeze_134 = None
    mul_312: "f32[672]" = torch.ops.aten.mul.Tensor(mul_311, 0.1);  mul_311 = None
    mul_313: "f32[672]" = torch.ops.aten.mul.Tensor(primals_332, 0.9)
    add_233: "f32[672]" = torch.ops.aten.add.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None
    unsqueeze_176: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1)
    unsqueeze_177: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    mul_314: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_308, unsqueeze_177);  mul_308 = unsqueeze_177 = None
    unsqueeze_178: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1);  primals_90 = None
    unsqueeze_179: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    add_234: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_314, unsqueeze_179);  mul_314 = unsqueeze_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_30: "f32[8, 672, 14, 14]" = torch.ops.aten.relu.default(add_234);  add_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_45: "f32[8, 112, 14, 14]" = torch.ops.aten.convolution.default(relu_30, primals_176, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_235: "i64[]" = torch.ops.aten.add.Tensor(primals_333, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_45 = torch.ops.aten.var_mean.correction(convolution_45, [0, 2, 3], correction = 0, keepdim = True)
    getitem_90: "f32[1, 112, 1, 1]" = var_mean_45[0]
    getitem_91: "f32[1, 112, 1, 1]" = var_mean_45[1];  var_mean_45 = None
    add_236: "f32[1, 112, 1, 1]" = torch.ops.aten.add.Tensor(getitem_90, 0.001)
    rsqrt_45: "f32[1, 112, 1, 1]" = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
    sub_45: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, getitem_91)
    mul_315: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
    squeeze_135: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
    squeeze_136: "f32[112]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
    mul_316: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1)
    mul_317: "f32[112]" = torch.ops.aten.mul.Tensor(primals_334, 0.9)
    add_237: "f32[112]" = torch.ops.aten.add.Tensor(mul_316, mul_317);  mul_316 = mul_317 = None
    squeeze_137: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
    mul_318: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_137, 1.0006381620931717);  squeeze_137 = None
    mul_319: "f32[112]" = torch.ops.aten.mul.Tensor(mul_318, 0.1);  mul_318 = None
    mul_320: "f32[112]" = torch.ops.aten.mul.Tensor(primals_335, 0.9)
    add_238: "f32[112]" = torch.ops.aten.add.Tensor(mul_319, mul_320);  mul_319 = mul_320 = None
    unsqueeze_180: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_91, -1)
    unsqueeze_181: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_321: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(mul_315, unsqueeze_181);  mul_315 = unsqueeze_181 = None
    unsqueeze_182: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1);  primals_92 = None
    unsqueeze_183: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_239: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(mul_321, unsqueeze_183);  mul_321 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_240: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(add_239, add_224);  add_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_46: "f32[8, 336, 14, 14]" = torch.ops.aten.convolution.default(add_240, primals_177, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_241: "i64[]" = torch.ops.aten.add.Tensor(primals_336, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_46 = torch.ops.aten.var_mean.correction(convolution_46, [0, 2, 3], correction = 0, keepdim = True)
    getitem_92: "f32[1, 336, 1, 1]" = var_mean_46[0]
    getitem_93: "f32[1, 336, 1, 1]" = var_mean_46[1];  var_mean_46 = None
    add_242: "f32[1, 336, 1, 1]" = torch.ops.aten.add.Tensor(getitem_92, 0.001)
    rsqrt_46: "f32[1, 336, 1, 1]" = torch.ops.aten.rsqrt.default(add_242);  add_242 = None
    sub_46: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, getitem_93)
    mul_322: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
    squeeze_138: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
    squeeze_139: "f32[336]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
    mul_323: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_138, 0.1)
    mul_324: "f32[336]" = torch.ops.aten.mul.Tensor(primals_337, 0.9)
    add_243: "f32[336]" = torch.ops.aten.add.Tensor(mul_323, mul_324);  mul_323 = mul_324 = None
    squeeze_140: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_92, [0, 2, 3]);  getitem_92 = None
    mul_325: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_140, 1.0006381620931717);  squeeze_140 = None
    mul_326: "f32[336]" = torch.ops.aten.mul.Tensor(mul_325, 0.1);  mul_325 = None
    mul_327: "f32[336]" = torch.ops.aten.mul.Tensor(primals_338, 0.9)
    add_244: "f32[336]" = torch.ops.aten.add.Tensor(mul_326, mul_327);  mul_326 = mul_327 = None
    unsqueeze_184: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_93, -1)
    unsqueeze_185: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    mul_328: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(mul_322, unsqueeze_185);  mul_322 = unsqueeze_185 = None
    unsqueeze_186: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_94, -1);  primals_94 = None
    unsqueeze_187: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    add_245: "f32[8, 336, 14, 14]" = torch.ops.aten.add.Tensor(mul_328, unsqueeze_187);  mul_328 = unsqueeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_31: "f32[8, 336, 14, 14]" = torch.ops.aten.relu.default(add_245);  add_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_47: "f32[8, 336, 14, 14]" = torch.ops.aten.convolution.default(relu_31, primals_178, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 336)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_246: "i64[]" = torch.ops.aten.add.Tensor(primals_339, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_47 = torch.ops.aten.var_mean.correction(convolution_47, [0, 2, 3], correction = 0, keepdim = True)
    getitem_94: "f32[1, 336, 1, 1]" = var_mean_47[0]
    getitem_95: "f32[1, 336, 1, 1]" = var_mean_47[1];  var_mean_47 = None
    add_247: "f32[1, 336, 1, 1]" = torch.ops.aten.add.Tensor(getitem_94, 0.001)
    rsqrt_47: "f32[1, 336, 1, 1]" = torch.ops.aten.rsqrt.default(add_247);  add_247 = None
    sub_47: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, getitem_95)
    mul_329: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
    squeeze_141: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
    squeeze_142: "f32[336]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
    mul_330: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_141, 0.1)
    mul_331: "f32[336]" = torch.ops.aten.mul.Tensor(primals_340, 0.9)
    add_248: "f32[336]" = torch.ops.aten.add.Tensor(mul_330, mul_331);  mul_330 = mul_331 = None
    squeeze_143: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
    mul_332: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_143, 1.0006381620931717);  squeeze_143 = None
    mul_333: "f32[336]" = torch.ops.aten.mul.Tensor(mul_332, 0.1);  mul_332 = None
    mul_334: "f32[336]" = torch.ops.aten.mul.Tensor(primals_341, 0.9)
    add_249: "f32[336]" = torch.ops.aten.add.Tensor(mul_333, mul_334);  mul_333 = mul_334 = None
    unsqueeze_188: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1)
    unsqueeze_189: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_335: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(mul_329, unsqueeze_189);  mul_329 = unsqueeze_189 = None
    unsqueeze_190: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1);  primals_96 = None
    unsqueeze_191: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_250: "f32[8, 336, 14, 14]" = torch.ops.aten.add.Tensor(mul_335, unsqueeze_191);  mul_335 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_32: "f32[8, 336, 14, 14]" = torch.ops.aten.relu.default(add_250);  add_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_48: "f32[8, 112, 14, 14]" = torch.ops.aten.convolution.default(relu_32, primals_179, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_251: "i64[]" = torch.ops.aten.add.Tensor(primals_342, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_48 = torch.ops.aten.var_mean.correction(convolution_48, [0, 2, 3], correction = 0, keepdim = True)
    getitem_96: "f32[1, 112, 1, 1]" = var_mean_48[0]
    getitem_97: "f32[1, 112, 1, 1]" = var_mean_48[1];  var_mean_48 = None
    add_252: "f32[1, 112, 1, 1]" = torch.ops.aten.add.Tensor(getitem_96, 0.001)
    rsqrt_48: "f32[1, 112, 1, 1]" = torch.ops.aten.rsqrt.default(add_252);  add_252 = None
    sub_48: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, getitem_97)
    mul_336: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
    squeeze_144: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
    squeeze_145: "f32[112]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
    mul_337: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_144, 0.1)
    mul_338: "f32[112]" = torch.ops.aten.mul.Tensor(primals_343, 0.9)
    add_253: "f32[112]" = torch.ops.aten.add.Tensor(mul_337, mul_338);  mul_337 = mul_338 = None
    squeeze_146: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
    mul_339: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_146, 1.0006381620931717);  squeeze_146 = None
    mul_340: "f32[112]" = torch.ops.aten.mul.Tensor(mul_339, 0.1);  mul_339 = None
    mul_341: "f32[112]" = torch.ops.aten.mul.Tensor(primals_344, 0.9)
    add_254: "f32[112]" = torch.ops.aten.add.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
    unsqueeze_192: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_97, -1)
    unsqueeze_193: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    mul_342: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(mul_336, unsqueeze_193);  mul_336 = unsqueeze_193 = None
    unsqueeze_194: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1);  primals_98 = None
    unsqueeze_195: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    add_255: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(mul_342, unsqueeze_195);  mul_342 = unsqueeze_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_256: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(add_255, add_240);  add_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_49: "f32[8, 672, 14, 14]" = torch.ops.aten.convolution.default(add_256, primals_180, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_257: "i64[]" = torch.ops.aten.add.Tensor(primals_345, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_49 = torch.ops.aten.var_mean.correction(convolution_49, [0, 2, 3], correction = 0, keepdim = True)
    getitem_98: "f32[1, 672, 1, 1]" = var_mean_49[0]
    getitem_99: "f32[1, 672, 1, 1]" = var_mean_49[1];  var_mean_49 = None
    add_258: "f32[1, 672, 1, 1]" = torch.ops.aten.add.Tensor(getitem_98, 0.001)
    rsqrt_49: "f32[1, 672, 1, 1]" = torch.ops.aten.rsqrt.default(add_258);  add_258 = None
    sub_49: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, getitem_99)
    mul_343: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
    squeeze_147: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
    squeeze_148: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
    mul_344: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_147, 0.1)
    mul_345: "f32[672]" = torch.ops.aten.mul.Tensor(primals_346, 0.9)
    add_259: "f32[672]" = torch.ops.aten.add.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
    squeeze_149: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
    mul_346: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_149, 1.0006381620931717);  squeeze_149 = None
    mul_347: "f32[672]" = torch.ops.aten.mul.Tensor(mul_346, 0.1);  mul_346 = None
    mul_348: "f32[672]" = torch.ops.aten.mul.Tensor(primals_347, 0.9)
    add_260: "f32[672]" = torch.ops.aten.add.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
    unsqueeze_196: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1)
    unsqueeze_197: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_349: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_197);  mul_343 = unsqueeze_197 = None
    unsqueeze_198: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_100, -1);  primals_100 = None
    unsqueeze_199: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_261: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_349, unsqueeze_199);  mul_349 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_33: "f32[8, 672, 14, 14]" = torch.ops.aten.relu.default(add_261);  add_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_50: "f32[8, 672, 7, 7]" = torch.ops.aten.convolution.default(relu_33, primals_181, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 672)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_262: "i64[]" = torch.ops.aten.add.Tensor(primals_348, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_50 = torch.ops.aten.var_mean.correction(convolution_50, [0, 2, 3], correction = 0, keepdim = True)
    getitem_100: "f32[1, 672, 1, 1]" = var_mean_50[0]
    getitem_101: "f32[1, 672, 1, 1]" = var_mean_50[1];  var_mean_50 = None
    add_263: "f32[1, 672, 1, 1]" = torch.ops.aten.add.Tensor(getitem_100, 0.001)
    rsqrt_50: "f32[1, 672, 1, 1]" = torch.ops.aten.rsqrt.default(add_263);  add_263 = None
    sub_50: "f32[8, 672, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, getitem_101)
    mul_350: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
    squeeze_150: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
    squeeze_151: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
    mul_351: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_150, 0.1)
    mul_352: "f32[672]" = torch.ops.aten.mul.Tensor(primals_349, 0.9)
    add_264: "f32[672]" = torch.ops.aten.add.Tensor(mul_351, mul_352);  mul_351 = mul_352 = None
    squeeze_152: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_100, [0, 2, 3]);  getitem_100 = None
    mul_353: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_152, 1.0025575447570332);  squeeze_152 = None
    mul_354: "f32[672]" = torch.ops.aten.mul.Tensor(mul_353, 0.1);  mul_353 = None
    mul_355: "f32[672]" = torch.ops.aten.mul.Tensor(primals_350, 0.9)
    add_265: "f32[672]" = torch.ops.aten.add.Tensor(mul_354, mul_355);  mul_354 = mul_355 = None
    unsqueeze_200: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_101, -1)
    unsqueeze_201: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    mul_356: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(mul_350, unsqueeze_201);  mul_350 = unsqueeze_201 = None
    unsqueeze_202: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1);  primals_102 = None
    unsqueeze_203: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    add_266: "f32[8, 672, 7, 7]" = torch.ops.aten.add.Tensor(mul_356, unsqueeze_203);  mul_356 = unsqueeze_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_34: "f32[8, 672, 7, 7]" = torch.ops.aten.relu.default(add_266);  add_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_51: "f32[8, 184, 7, 7]" = torch.ops.aten.convolution.default(relu_34, primals_182, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_267: "i64[]" = torch.ops.aten.add.Tensor(primals_351, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_51 = torch.ops.aten.var_mean.correction(convolution_51, [0, 2, 3], correction = 0, keepdim = True)
    getitem_102: "f32[1, 184, 1, 1]" = var_mean_51[0]
    getitem_103: "f32[1, 184, 1, 1]" = var_mean_51[1];  var_mean_51 = None
    add_268: "f32[1, 184, 1, 1]" = torch.ops.aten.add.Tensor(getitem_102, 0.001)
    rsqrt_51: "f32[1, 184, 1, 1]" = torch.ops.aten.rsqrt.default(add_268);  add_268 = None
    sub_51: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, getitem_103)
    mul_357: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
    squeeze_153: "f32[184]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
    squeeze_154: "f32[184]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
    mul_358: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_153, 0.1)
    mul_359: "f32[184]" = torch.ops.aten.mul.Tensor(primals_352, 0.9)
    add_269: "f32[184]" = torch.ops.aten.add.Tensor(mul_358, mul_359);  mul_358 = mul_359 = None
    squeeze_155: "f32[184]" = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
    mul_360: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_155, 1.0025575447570332);  squeeze_155 = None
    mul_361: "f32[184]" = torch.ops.aten.mul.Tensor(mul_360, 0.1);  mul_360 = None
    mul_362: "f32[184]" = torch.ops.aten.mul.Tensor(primals_353, 0.9)
    add_270: "f32[184]" = torch.ops.aten.add.Tensor(mul_361, mul_362);  mul_361 = mul_362 = None
    unsqueeze_204: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_103, -1)
    unsqueeze_205: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_363: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(mul_357, unsqueeze_205);  mul_357 = unsqueeze_205 = None
    unsqueeze_206: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_104, -1);  primals_104 = None
    unsqueeze_207: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_271: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(mul_363, unsqueeze_207);  mul_363 = unsqueeze_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_52: "f32[8, 1104, 7, 7]" = torch.ops.aten.convolution.default(add_271, primals_183, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_272: "i64[]" = torch.ops.aten.add.Tensor(primals_354, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_52 = torch.ops.aten.var_mean.correction(convolution_52, [0, 2, 3], correction = 0, keepdim = True)
    getitem_104: "f32[1, 1104, 1, 1]" = var_mean_52[0]
    getitem_105: "f32[1, 1104, 1, 1]" = var_mean_52[1];  var_mean_52 = None
    add_273: "f32[1, 1104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_104, 0.001)
    rsqrt_52: "f32[1, 1104, 1, 1]" = torch.ops.aten.rsqrt.default(add_273);  add_273 = None
    sub_52: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_52, getitem_105)
    mul_364: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
    squeeze_156: "f32[1104]" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2, 3]);  getitem_105 = None
    squeeze_157: "f32[1104]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
    mul_365: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_156, 0.1)
    mul_366: "f32[1104]" = torch.ops.aten.mul.Tensor(primals_355, 0.9)
    add_274: "f32[1104]" = torch.ops.aten.add.Tensor(mul_365, mul_366);  mul_365 = mul_366 = None
    squeeze_158: "f32[1104]" = torch.ops.aten.squeeze.dims(getitem_104, [0, 2, 3]);  getitem_104 = None
    mul_367: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_158, 1.0025575447570332);  squeeze_158 = None
    mul_368: "f32[1104]" = torch.ops.aten.mul.Tensor(mul_367, 0.1);  mul_367 = None
    mul_369: "f32[1104]" = torch.ops.aten.mul.Tensor(primals_356, 0.9)
    add_275: "f32[1104]" = torch.ops.aten.add.Tensor(mul_368, mul_369);  mul_368 = mul_369 = None
    unsqueeze_208: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(primals_105, -1)
    unsqueeze_209: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    mul_370: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(mul_364, unsqueeze_209);  mul_364 = unsqueeze_209 = None
    unsqueeze_210: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(primals_106, -1);  primals_106 = None
    unsqueeze_211: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    add_276: "f32[8, 1104, 7, 7]" = torch.ops.aten.add.Tensor(mul_370, unsqueeze_211);  mul_370 = unsqueeze_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_35: "f32[8, 1104, 7, 7]" = torch.ops.aten.relu.default(add_276);  add_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_53: "f32[8, 1104, 7, 7]" = torch.ops.aten.convolution.default(relu_35, primals_184, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1104)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_277: "i64[]" = torch.ops.aten.add.Tensor(primals_357, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_53 = torch.ops.aten.var_mean.correction(convolution_53, [0, 2, 3], correction = 0, keepdim = True)
    getitem_106: "f32[1, 1104, 1, 1]" = var_mean_53[0]
    getitem_107: "f32[1, 1104, 1, 1]" = var_mean_53[1];  var_mean_53 = None
    add_278: "f32[1, 1104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_106, 0.001)
    rsqrt_53: "f32[1, 1104, 1, 1]" = torch.ops.aten.rsqrt.default(add_278);  add_278 = None
    sub_53: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_53, getitem_107)
    mul_371: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
    squeeze_159: "f32[1104]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
    squeeze_160: "f32[1104]" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2, 3]);  rsqrt_53 = None
    mul_372: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_159, 0.1)
    mul_373: "f32[1104]" = torch.ops.aten.mul.Tensor(primals_358, 0.9)
    add_279: "f32[1104]" = torch.ops.aten.add.Tensor(mul_372, mul_373);  mul_372 = mul_373 = None
    squeeze_161: "f32[1104]" = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
    mul_374: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_161, 1.0025575447570332);  squeeze_161 = None
    mul_375: "f32[1104]" = torch.ops.aten.mul.Tensor(mul_374, 0.1);  mul_374 = None
    mul_376: "f32[1104]" = torch.ops.aten.mul.Tensor(primals_359, 0.9)
    add_280: "f32[1104]" = torch.ops.aten.add.Tensor(mul_375, mul_376);  mul_375 = mul_376 = None
    unsqueeze_212: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(primals_107, -1)
    unsqueeze_213: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_377: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(mul_371, unsqueeze_213);  mul_371 = unsqueeze_213 = None
    unsqueeze_214: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1);  primals_108 = None
    unsqueeze_215: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_281: "f32[8, 1104, 7, 7]" = torch.ops.aten.add.Tensor(mul_377, unsqueeze_215);  mul_377 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_36: "f32[8, 1104, 7, 7]" = torch.ops.aten.relu.default(add_281);  add_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_54: "f32[8, 184, 7, 7]" = torch.ops.aten.convolution.default(relu_36, primals_185, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_282: "i64[]" = torch.ops.aten.add.Tensor(primals_360, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_54 = torch.ops.aten.var_mean.correction(convolution_54, [0, 2, 3], correction = 0, keepdim = True)
    getitem_108: "f32[1, 184, 1, 1]" = var_mean_54[0]
    getitem_109: "f32[1, 184, 1, 1]" = var_mean_54[1];  var_mean_54 = None
    add_283: "f32[1, 184, 1, 1]" = torch.ops.aten.add.Tensor(getitem_108, 0.001)
    rsqrt_54: "f32[1, 184, 1, 1]" = torch.ops.aten.rsqrt.default(add_283);  add_283 = None
    sub_54: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_54, getitem_109)
    mul_378: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
    squeeze_162: "f32[184]" = torch.ops.aten.squeeze.dims(getitem_109, [0, 2, 3]);  getitem_109 = None
    squeeze_163: "f32[184]" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2, 3]);  rsqrt_54 = None
    mul_379: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_162, 0.1)
    mul_380: "f32[184]" = torch.ops.aten.mul.Tensor(primals_361, 0.9)
    add_284: "f32[184]" = torch.ops.aten.add.Tensor(mul_379, mul_380);  mul_379 = mul_380 = None
    squeeze_164: "f32[184]" = torch.ops.aten.squeeze.dims(getitem_108, [0, 2, 3]);  getitem_108 = None
    mul_381: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_164, 1.0025575447570332);  squeeze_164 = None
    mul_382: "f32[184]" = torch.ops.aten.mul.Tensor(mul_381, 0.1);  mul_381 = None
    mul_383: "f32[184]" = torch.ops.aten.mul.Tensor(primals_362, 0.9)
    add_285: "f32[184]" = torch.ops.aten.add.Tensor(mul_382, mul_383);  mul_382 = mul_383 = None
    unsqueeze_216: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_109, -1)
    unsqueeze_217: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    mul_384: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(mul_378, unsqueeze_217);  mul_378 = unsqueeze_217 = None
    unsqueeze_218: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_110, -1);  primals_110 = None
    unsqueeze_219: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    add_286: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(mul_384, unsqueeze_219);  mul_384 = unsqueeze_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_287: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(add_286, add_271);  add_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_55: "f32[8, 1104, 7, 7]" = torch.ops.aten.convolution.default(add_287, primals_186, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_288: "i64[]" = torch.ops.aten.add.Tensor(primals_363, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_55 = torch.ops.aten.var_mean.correction(convolution_55, [0, 2, 3], correction = 0, keepdim = True)
    getitem_110: "f32[1, 1104, 1, 1]" = var_mean_55[0]
    getitem_111: "f32[1, 1104, 1, 1]" = var_mean_55[1];  var_mean_55 = None
    add_289: "f32[1, 1104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_110, 0.001)
    rsqrt_55: "f32[1, 1104, 1, 1]" = torch.ops.aten.rsqrt.default(add_289);  add_289 = None
    sub_55: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_55, getitem_111)
    mul_385: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = None
    squeeze_165: "f32[1104]" = torch.ops.aten.squeeze.dims(getitem_111, [0, 2, 3]);  getitem_111 = None
    squeeze_166: "f32[1104]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2, 3]);  rsqrt_55 = None
    mul_386: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_165, 0.1)
    mul_387: "f32[1104]" = torch.ops.aten.mul.Tensor(primals_364, 0.9)
    add_290: "f32[1104]" = torch.ops.aten.add.Tensor(mul_386, mul_387);  mul_386 = mul_387 = None
    squeeze_167: "f32[1104]" = torch.ops.aten.squeeze.dims(getitem_110, [0, 2, 3]);  getitem_110 = None
    mul_388: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_167, 1.0025575447570332);  squeeze_167 = None
    mul_389: "f32[1104]" = torch.ops.aten.mul.Tensor(mul_388, 0.1);  mul_388 = None
    mul_390: "f32[1104]" = torch.ops.aten.mul.Tensor(primals_365, 0.9)
    add_291: "f32[1104]" = torch.ops.aten.add.Tensor(mul_389, mul_390);  mul_389 = mul_390 = None
    unsqueeze_220: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(primals_111, -1)
    unsqueeze_221: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_391: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(mul_385, unsqueeze_221);  mul_385 = unsqueeze_221 = None
    unsqueeze_222: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(primals_112, -1);  primals_112 = None
    unsqueeze_223: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_292: "f32[8, 1104, 7, 7]" = torch.ops.aten.add.Tensor(mul_391, unsqueeze_223);  mul_391 = unsqueeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_37: "f32[8, 1104, 7, 7]" = torch.ops.aten.relu.default(add_292);  add_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_56: "f32[8, 1104, 7, 7]" = torch.ops.aten.convolution.default(relu_37, primals_187, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1104)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_293: "i64[]" = torch.ops.aten.add.Tensor(primals_366, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_56 = torch.ops.aten.var_mean.correction(convolution_56, [0, 2, 3], correction = 0, keepdim = True)
    getitem_112: "f32[1, 1104, 1, 1]" = var_mean_56[0]
    getitem_113: "f32[1, 1104, 1, 1]" = var_mean_56[1];  var_mean_56 = None
    add_294: "f32[1, 1104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_112, 0.001)
    rsqrt_56: "f32[1, 1104, 1, 1]" = torch.ops.aten.rsqrt.default(add_294);  add_294 = None
    sub_56: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_56, getitem_113)
    mul_392: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = None
    squeeze_168: "f32[1104]" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3]);  getitem_113 = None
    squeeze_169: "f32[1104]" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2, 3]);  rsqrt_56 = None
    mul_393: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_168, 0.1)
    mul_394: "f32[1104]" = torch.ops.aten.mul.Tensor(primals_367, 0.9)
    add_295: "f32[1104]" = torch.ops.aten.add.Tensor(mul_393, mul_394);  mul_393 = mul_394 = None
    squeeze_170: "f32[1104]" = torch.ops.aten.squeeze.dims(getitem_112, [0, 2, 3]);  getitem_112 = None
    mul_395: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_170, 1.0025575447570332);  squeeze_170 = None
    mul_396: "f32[1104]" = torch.ops.aten.mul.Tensor(mul_395, 0.1);  mul_395 = None
    mul_397: "f32[1104]" = torch.ops.aten.mul.Tensor(primals_368, 0.9)
    add_296: "f32[1104]" = torch.ops.aten.add.Tensor(mul_396, mul_397);  mul_396 = mul_397 = None
    unsqueeze_224: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(primals_113, -1)
    unsqueeze_225: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    mul_398: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(mul_392, unsqueeze_225);  mul_392 = unsqueeze_225 = None
    unsqueeze_226: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(primals_114, -1);  primals_114 = None
    unsqueeze_227: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    add_297: "f32[8, 1104, 7, 7]" = torch.ops.aten.add.Tensor(mul_398, unsqueeze_227);  mul_398 = unsqueeze_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_38: "f32[8, 1104, 7, 7]" = torch.ops.aten.relu.default(add_297);  add_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_57: "f32[8, 184, 7, 7]" = torch.ops.aten.convolution.default(relu_38, primals_188, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_298: "i64[]" = torch.ops.aten.add.Tensor(primals_369, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_57 = torch.ops.aten.var_mean.correction(convolution_57, [0, 2, 3], correction = 0, keepdim = True)
    getitem_114: "f32[1, 184, 1, 1]" = var_mean_57[0]
    getitem_115: "f32[1, 184, 1, 1]" = var_mean_57[1];  var_mean_57 = None
    add_299: "f32[1, 184, 1, 1]" = torch.ops.aten.add.Tensor(getitem_114, 0.001)
    rsqrt_57: "f32[1, 184, 1, 1]" = torch.ops.aten.rsqrt.default(add_299);  add_299 = None
    sub_57: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_57, getitem_115)
    mul_399: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = None
    squeeze_171: "f32[184]" = torch.ops.aten.squeeze.dims(getitem_115, [0, 2, 3]);  getitem_115 = None
    squeeze_172: "f32[184]" = torch.ops.aten.squeeze.dims(rsqrt_57, [0, 2, 3]);  rsqrt_57 = None
    mul_400: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_171, 0.1)
    mul_401: "f32[184]" = torch.ops.aten.mul.Tensor(primals_370, 0.9)
    add_300: "f32[184]" = torch.ops.aten.add.Tensor(mul_400, mul_401);  mul_400 = mul_401 = None
    squeeze_173: "f32[184]" = torch.ops.aten.squeeze.dims(getitem_114, [0, 2, 3]);  getitem_114 = None
    mul_402: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_173, 1.0025575447570332);  squeeze_173 = None
    mul_403: "f32[184]" = torch.ops.aten.mul.Tensor(mul_402, 0.1);  mul_402 = None
    mul_404: "f32[184]" = torch.ops.aten.mul.Tensor(primals_371, 0.9)
    add_301: "f32[184]" = torch.ops.aten.add.Tensor(mul_403, mul_404);  mul_403 = mul_404 = None
    unsqueeze_228: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_115, -1)
    unsqueeze_229: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_405: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(mul_399, unsqueeze_229);  mul_399 = unsqueeze_229 = None
    unsqueeze_230: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_116, -1);  primals_116 = None
    unsqueeze_231: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_302: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(mul_405, unsqueeze_231);  mul_405 = unsqueeze_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_303: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(add_302, add_287);  add_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_58: "f32[8, 1104, 7, 7]" = torch.ops.aten.convolution.default(add_303, primals_189, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_304: "i64[]" = torch.ops.aten.add.Tensor(primals_372, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_58 = torch.ops.aten.var_mean.correction(convolution_58, [0, 2, 3], correction = 0, keepdim = True)
    getitem_116: "f32[1, 1104, 1, 1]" = var_mean_58[0]
    getitem_117: "f32[1, 1104, 1, 1]" = var_mean_58[1];  var_mean_58 = None
    add_305: "f32[1, 1104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_116, 0.001)
    rsqrt_58: "f32[1, 1104, 1, 1]" = torch.ops.aten.rsqrt.default(add_305);  add_305 = None
    sub_58: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_58, getitem_117)
    mul_406: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = None
    squeeze_174: "f32[1104]" = torch.ops.aten.squeeze.dims(getitem_117, [0, 2, 3]);  getitem_117 = None
    squeeze_175: "f32[1104]" = torch.ops.aten.squeeze.dims(rsqrt_58, [0, 2, 3]);  rsqrt_58 = None
    mul_407: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_174, 0.1)
    mul_408: "f32[1104]" = torch.ops.aten.mul.Tensor(primals_373, 0.9)
    add_306: "f32[1104]" = torch.ops.aten.add.Tensor(mul_407, mul_408);  mul_407 = mul_408 = None
    squeeze_176: "f32[1104]" = torch.ops.aten.squeeze.dims(getitem_116, [0, 2, 3]);  getitem_116 = None
    mul_409: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_176, 1.0025575447570332);  squeeze_176 = None
    mul_410: "f32[1104]" = torch.ops.aten.mul.Tensor(mul_409, 0.1);  mul_409 = None
    mul_411: "f32[1104]" = torch.ops.aten.mul.Tensor(primals_374, 0.9)
    add_307: "f32[1104]" = torch.ops.aten.add.Tensor(mul_410, mul_411);  mul_410 = mul_411 = None
    unsqueeze_232: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(primals_117, -1)
    unsqueeze_233: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    mul_412: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(mul_406, unsqueeze_233);  mul_406 = unsqueeze_233 = None
    unsqueeze_234: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(primals_118, -1);  primals_118 = None
    unsqueeze_235: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    add_308: "f32[8, 1104, 7, 7]" = torch.ops.aten.add.Tensor(mul_412, unsqueeze_235);  mul_412 = unsqueeze_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_39: "f32[8, 1104, 7, 7]" = torch.ops.aten.relu.default(add_308);  add_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_59: "f32[8, 1104, 7, 7]" = torch.ops.aten.convolution.default(relu_39, primals_190, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1104)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_309: "i64[]" = torch.ops.aten.add.Tensor(primals_375, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_59 = torch.ops.aten.var_mean.correction(convolution_59, [0, 2, 3], correction = 0, keepdim = True)
    getitem_118: "f32[1, 1104, 1, 1]" = var_mean_59[0]
    getitem_119: "f32[1, 1104, 1, 1]" = var_mean_59[1];  var_mean_59 = None
    add_310: "f32[1, 1104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_118, 0.001)
    rsqrt_59: "f32[1, 1104, 1, 1]" = torch.ops.aten.rsqrt.default(add_310);  add_310 = None
    sub_59: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_59, getitem_119)
    mul_413: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = None
    squeeze_177: "f32[1104]" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
    squeeze_178: "f32[1104]" = torch.ops.aten.squeeze.dims(rsqrt_59, [0, 2, 3]);  rsqrt_59 = None
    mul_414: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_177, 0.1)
    mul_415: "f32[1104]" = torch.ops.aten.mul.Tensor(primals_376, 0.9)
    add_311: "f32[1104]" = torch.ops.aten.add.Tensor(mul_414, mul_415);  mul_414 = mul_415 = None
    squeeze_179: "f32[1104]" = torch.ops.aten.squeeze.dims(getitem_118, [0, 2, 3]);  getitem_118 = None
    mul_416: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_179, 1.0025575447570332);  squeeze_179 = None
    mul_417: "f32[1104]" = torch.ops.aten.mul.Tensor(mul_416, 0.1);  mul_416 = None
    mul_418: "f32[1104]" = torch.ops.aten.mul.Tensor(primals_377, 0.9)
    add_312: "f32[1104]" = torch.ops.aten.add.Tensor(mul_417, mul_418);  mul_417 = mul_418 = None
    unsqueeze_236: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(primals_119, -1)
    unsqueeze_237: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_419: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(mul_413, unsqueeze_237);  mul_413 = unsqueeze_237 = None
    unsqueeze_238: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(primals_120, -1);  primals_120 = None
    unsqueeze_239: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_313: "f32[8, 1104, 7, 7]" = torch.ops.aten.add.Tensor(mul_419, unsqueeze_239);  mul_419 = unsqueeze_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_40: "f32[8, 1104, 7, 7]" = torch.ops.aten.relu.default(add_313);  add_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_60: "f32[8, 184, 7, 7]" = torch.ops.aten.convolution.default(relu_40, primals_191, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_314: "i64[]" = torch.ops.aten.add.Tensor(primals_378, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_60 = torch.ops.aten.var_mean.correction(convolution_60, [0, 2, 3], correction = 0, keepdim = True)
    getitem_120: "f32[1, 184, 1, 1]" = var_mean_60[0]
    getitem_121: "f32[1, 184, 1, 1]" = var_mean_60[1];  var_mean_60 = None
    add_315: "f32[1, 184, 1, 1]" = torch.ops.aten.add.Tensor(getitem_120, 0.001)
    rsqrt_60: "f32[1, 184, 1, 1]" = torch.ops.aten.rsqrt.default(add_315);  add_315 = None
    sub_60: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_60, getitem_121)
    mul_420: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = None
    squeeze_180: "f32[184]" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3]);  getitem_121 = None
    squeeze_181: "f32[184]" = torch.ops.aten.squeeze.dims(rsqrt_60, [0, 2, 3]);  rsqrt_60 = None
    mul_421: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_180, 0.1)
    mul_422: "f32[184]" = torch.ops.aten.mul.Tensor(primals_379, 0.9)
    add_316: "f32[184]" = torch.ops.aten.add.Tensor(mul_421, mul_422);  mul_421 = mul_422 = None
    squeeze_182: "f32[184]" = torch.ops.aten.squeeze.dims(getitem_120, [0, 2, 3]);  getitem_120 = None
    mul_423: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_182, 1.0025575447570332);  squeeze_182 = None
    mul_424: "f32[184]" = torch.ops.aten.mul.Tensor(mul_423, 0.1);  mul_423 = None
    mul_425: "f32[184]" = torch.ops.aten.mul.Tensor(primals_380, 0.9)
    add_317: "f32[184]" = torch.ops.aten.add.Tensor(mul_424, mul_425);  mul_424 = mul_425 = None
    unsqueeze_240: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_121, -1)
    unsqueeze_241: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    mul_426: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(mul_420, unsqueeze_241);  mul_420 = unsqueeze_241 = None
    unsqueeze_242: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_122, -1);  primals_122 = None
    unsqueeze_243: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    add_318: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(mul_426, unsqueeze_243);  mul_426 = unsqueeze_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_319: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(add_318, add_303);  add_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_61: "f32[8, 1104, 7, 7]" = torch.ops.aten.convolution.default(add_319, primals_192, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_320: "i64[]" = torch.ops.aten.add.Tensor(primals_381, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_61 = torch.ops.aten.var_mean.correction(convolution_61, [0, 2, 3], correction = 0, keepdim = True)
    getitem_122: "f32[1, 1104, 1, 1]" = var_mean_61[0]
    getitem_123: "f32[1, 1104, 1, 1]" = var_mean_61[1];  var_mean_61 = None
    add_321: "f32[1, 1104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_122, 0.001)
    rsqrt_61: "f32[1, 1104, 1, 1]" = torch.ops.aten.rsqrt.default(add_321);  add_321 = None
    sub_61: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_61, getitem_123)
    mul_427: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_61);  sub_61 = None
    squeeze_183: "f32[1104]" = torch.ops.aten.squeeze.dims(getitem_123, [0, 2, 3]);  getitem_123 = None
    squeeze_184: "f32[1104]" = torch.ops.aten.squeeze.dims(rsqrt_61, [0, 2, 3]);  rsqrt_61 = None
    mul_428: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_183, 0.1)
    mul_429: "f32[1104]" = torch.ops.aten.mul.Tensor(primals_382, 0.9)
    add_322: "f32[1104]" = torch.ops.aten.add.Tensor(mul_428, mul_429);  mul_428 = mul_429 = None
    squeeze_185: "f32[1104]" = torch.ops.aten.squeeze.dims(getitem_122, [0, 2, 3]);  getitem_122 = None
    mul_430: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_185, 1.0025575447570332);  squeeze_185 = None
    mul_431: "f32[1104]" = torch.ops.aten.mul.Tensor(mul_430, 0.1);  mul_430 = None
    mul_432: "f32[1104]" = torch.ops.aten.mul.Tensor(primals_383, 0.9)
    add_323: "f32[1104]" = torch.ops.aten.add.Tensor(mul_431, mul_432);  mul_431 = mul_432 = None
    unsqueeze_244: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(primals_123, -1)
    unsqueeze_245: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_433: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(mul_427, unsqueeze_245);  mul_427 = unsqueeze_245 = None
    unsqueeze_246: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(primals_124, -1);  primals_124 = None
    unsqueeze_247: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_324: "f32[8, 1104, 7, 7]" = torch.ops.aten.add.Tensor(mul_433, unsqueeze_247);  mul_433 = unsqueeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_41: "f32[8, 1104, 7, 7]" = torch.ops.aten.relu.default(add_324);  add_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_62: "f32[8, 1104, 7, 7]" = torch.ops.aten.convolution.default(relu_41, primals_193, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1104)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_325: "i64[]" = torch.ops.aten.add.Tensor(primals_384, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_62 = torch.ops.aten.var_mean.correction(convolution_62, [0, 2, 3], correction = 0, keepdim = True)
    getitem_124: "f32[1, 1104, 1, 1]" = var_mean_62[0]
    getitem_125: "f32[1, 1104, 1, 1]" = var_mean_62[1];  var_mean_62 = None
    add_326: "f32[1, 1104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_124, 0.001)
    rsqrt_62: "f32[1, 1104, 1, 1]" = torch.ops.aten.rsqrt.default(add_326);  add_326 = None
    sub_62: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_62, getitem_125)
    mul_434: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_62);  sub_62 = None
    squeeze_186: "f32[1104]" = torch.ops.aten.squeeze.dims(getitem_125, [0, 2, 3]);  getitem_125 = None
    squeeze_187: "f32[1104]" = torch.ops.aten.squeeze.dims(rsqrt_62, [0, 2, 3]);  rsqrt_62 = None
    mul_435: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_186, 0.1)
    mul_436: "f32[1104]" = torch.ops.aten.mul.Tensor(primals_385, 0.9)
    add_327: "f32[1104]" = torch.ops.aten.add.Tensor(mul_435, mul_436);  mul_435 = mul_436 = None
    squeeze_188: "f32[1104]" = torch.ops.aten.squeeze.dims(getitem_124, [0, 2, 3]);  getitem_124 = None
    mul_437: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_188, 1.0025575447570332);  squeeze_188 = None
    mul_438: "f32[1104]" = torch.ops.aten.mul.Tensor(mul_437, 0.1);  mul_437 = None
    mul_439: "f32[1104]" = torch.ops.aten.mul.Tensor(primals_386, 0.9)
    add_328: "f32[1104]" = torch.ops.aten.add.Tensor(mul_438, mul_439);  mul_438 = mul_439 = None
    unsqueeze_248: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(primals_125, -1)
    unsqueeze_249: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    mul_440: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(mul_434, unsqueeze_249);  mul_434 = unsqueeze_249 = None
    unsqueeze_250: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(primals_126, -1);  primals_126 = None
    unsqueeze_251: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    add_329: "f32[8, 1104, 7, 7]" = torch.ops.aten.add.Tensor(mul_440, unsqueeze_251);  mul_440 = unsqueeze_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_42: "f32[8, 1104, 7, 7]" = torch.ops.aten.relu.default(add_329);  add_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_63: "f32[8, 352, 7, 7]" = torch.ops.aten.convolution.default(relu_42, primals_194, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_330: "i64[]" = torch.ops.aten.add.Tensor(primals_387, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_63 = torch.ops.aten.var_mean.correction(convolution_63, [0, 2, 3], correction = 0, keepdim = True)
    getitem_126: "f32[1, 352, 1, 1]" = var_mean_63[0]
    getitem_127: "f32[1, 352, 1, 1]" = var_mean_63[1];  var_mean_63 = None
    add_331: "f32[1, 352, 1, 1]" = torch.ops.aten.add.Tensor(getitem_126, 0.001)
    rsqrt_63: "f32[1, 352, 1, 1]" = torch.ops.aten.rsqrt.default(add_331);  add_331 = None
    sub_63: "f32[8, 352, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_63, getitem_127)
    mul_441: "f32[8, 352, 7, 7]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_63);  sub_63 = None
    squeeze_189: "f32[352]" = torch.ops.aten.squeeze.dims(getitem_127, [0, 2, 3]);  getitem_127 = None
    squeeze_190: "f32[352]" = torch.ops.aten.squeeze.dims(rsqrt_63, [0, 2, 3]);  rsqrt_63 = None
    mul_442: "f32[352]" = torch.ops.aten.mul.Tensor(squeeze_189, 0.1)
    mul_443: "f32[352]" = torch.ops.aten.mul.Tensor(primals_388, 0.9)
    add_332: "f32[352]" = torch.ops.aten.add.Tensor(mul_442, mul_443);  mul_442 = mul_443 = None
    squeeze_191: "f32[352]" = torch.ops.aten.squeeze.dims(getitem_126, [0, 2, 3]);  getitem_126 = None
    mul_444: "f32[352]" = torch.ops.aten.mul.Tensor(squeeze_191, 1.0025575447570332);  squeeze_191 = None
    mul_445: "f32[352]" = torch.ops.aten.mul.Tensor(mul_444, 0.1);  mul_444 = None
    mul_446: "f32[352]" = torch.ops.aten.mul.Tensor(primals_389, 0.9)
    add_333: "f32[352]" = torch.ops.aten.add.Tensor(mul_445, mul_446);  mul_445 = mul_446 = None
    unsqueeze_252: "f32[352, 1]" = torch.ops.aten.unsqueeze.default(primals_127, -1)
    unsqueeze_253: "f32[352, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_447: "f32[8, 352, 7, 7]" = torch.ops.aten.mul.Tensor(mul_441, unsqueeze_253);  mul_441 = unsqueeze_253 = None
    unsqueeze_254: "f32[352, 1]" = torch.ops.aten.unsqueeze.default(primals_128, -1);  primals_128 = None
    unsqueeze_255: "f32[352, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_334: "f32[8, 352, 7, 7]" = torch.ops.aten.add.Tensor(mul_447, unsqueeze_255);  mul_447 = unsqueeze_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:168, code: x = self.conv_head(x)
    convolution_64: "f32[8, 1984, 7, 7]" = torch.ops.aten.convolution.default(add_334, primals_195, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_335: "i64[]" = torch.ops.aten.add.Tensor(primals_390, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_64 = torch.ops.aten.var_mean.correction(convolution_64, [0, 2, 3], correction = 0, keepdim = True)
    getitem_128: "f32[1, 1984, 1, 1]" = var_mean_64[0]
    getitem_129: "f32[1, 1984, 1, 1]" = var_mean_64[1];  var_mean_64 = None
    add_336: "f32[1, 1984, 1, 1]" = torch.ops.aten.add.Tensor(getitem_128, 0.001)
    rsqrt_64: "f32[1, 1984, 1, 1]" = torch.ops.aten.rsqrt.default(add_336);  add_336 = None
    sub_64: "f32[8, 1984, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_64, getitem_129)
    mul_448: "f32[8, 1984, 7, 7]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_64);  sub_64 = None
    squeeze_192: "f32[1984]" = torch.ops.aten.squeeze.dims(getitem_129, [0, 2, 3]);  getitem_129 = None
    squeeze_193: "f32[1984]" = torch.ops.aten.squeeze.dims(rsqrt_64, [0, 2, 3]);  rsqrt_64 = None
    mul_449: "f32[1984]" = torch.ops.aten.mul.Tensor(squeeze_192, 0.1)
    mul_450: "f32[1984]" = torch.ops.aten.mul.Tensor(primals_391, 0.9)
    add_337: "f32[1984]" = torch.ops.aten.add.Tensor(mul_449, mul_450);  mul_449 = mul_450 = None
    squeeze_194: "f32[1984]" = torch.ops.aten.squeeze.dims(getitem_128, [0, 2, 3]);  getitem_128 = None
    mul_451: "f32[1984]" = torch.ops.aten.mul.Tensor(squeeze_194, 1.0025575447570332);  squeeze_194 = None
    mul_452: "f32[1984]" = torch.ops.aten.mul.Tensor(mul_451, 0.1);  mul_451 = None
    mul_453: "f32[1984]" = torch.ops.aten.mul.Tensor(primals_392, 0.9)
    add_338: "f32[1984]" = torch.ops.aten.add.Tensor(mul_452, mul_453);  mul_452 = mul_453 = None
    unsqueeze_256: "f32[1984, 1]" = torch.ops.aten.unsqueeze.default(primals_129, -1)
    unsqueeze_257: "f32[1984, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    mul_454: "f32[8, 1984, 7, 7]" = torch.ops.aten.mul.Tensor(mul_448, unsqueeze_257);  mul_448 = unsqueeze_257 = None
    unsqueeze_258: "f32[1984, 1]" = torch.ops.aten.unsqueeze.default(primals_130, -1);  primals_130 = None
    unsqueeze_259: "f32[1984, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    add_339: "f32[8, 1984, 7, 7]" = torch.ops.aten.add.Tensor(mul_454, unsqueeze_259);  mul_454 = unsqueeze_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_43: "f32[8, 1984, 7, 7]" = torch.ops.aten.relu.default(add_339);  add_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean: "f32[8, 1984, 1, 1]" = torch.ops.aten.mean.dim(relu_43, [-1, -2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view: "f32[8, 1984]" = torch.ops.aten.view.default(mean, [8, 1984]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:176, code: return x if pre_logits else self.classifier(x)
    permute: "f32[1984, 1000]" = torch.ops.aten.permute.default(primals_196, [1, 0]);  primals_196 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_197, view, permute);  primals_197 = None
    permute_1: "f32[1000, 1984]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm: "f32[8, 1984]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 1984]" = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
    permute_3: "f32[1984, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 1984]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_2: "f32[8, 1984, 1, 1]" = torch.ops.aten.view.default(mm, [8, 1984, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 1984, 7, 7]" = torch.ops.aten.expand.default(view_2, [8, 1984, 7, 7]);  view_2 = None
    div: "f32[8, 1984, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_45: "f32[8, 1984, 7, 7]" = torch.ops.aten.alias.default(relu_43);  relu_43 = None
    alias_46: "f32[8, 1984, 7, 7]" = torch.ops.aten.alias.default(alias_45);  alias_45 = None
    le: "b8[8, 1984, 7, 7]" = torch.ops.aten.le.Scalar(alias_46, 0);  alias_46 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where: "f32[8, 1984, 7, 7]" = torch.ops.aten.where.self(le, scalar_tensor, div);  le = scalar_tensor = div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_260: "f32[1, 1984]" = torch.ops.aten.unsqueeze.default(squeeze_192, 0);  squeeze_192 = None
    unsqueeze_261: "f32[1, 1984, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 2);  unsqueeze_260 = None
    unsqueeze_262: "f32[1, 1984, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 3);  unsqueeze_261 = None
    sum_2: "f32[1984]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_65: "f32[8, 1984, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_262)
    mul_455: "f32[8, 1984, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_65);  sub_65 = None
    sum_3: "f32[1984]" = torch.ops.aten.sum.dim_IntList(mul_455, [0, 2, 3]);  mul_455 = None
    mul_456: "f32[1984]" = torch.ops.aten.mul.Tensor(sum_2, 0.002551020408163265)
    unsqueeze_263: "f32[1, 1984]" = torch.ops.aten.unsqueeze.default(mul_456, 0);  mul_456 = None
    unsqueeze_264: "f32[1, 1984, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 2);  unsqueeze_263 = None
    unsqueeze_265: "f32[1, 1984, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 3);  unsqueeze_264 = None
    mul_457: "f32[1984]" = torch.ops.aten.mul.Tensor(sum_3, 0.002551020408163265)
    mul_458: "f32[1984]" = torch.ops.aten.mul.Tensor(squeeze_193, squeeze_193)
    mul_459: "f32[1984]" = torch.ops.aten.mul.Tensor(mul_457, mul_458);  mul_457 = mul_458 = None
    unsqueeze_266: "f32[1, 1984]" = torch.ops.aten.unsqueeze.default(mul_459, 0);  mul_459 = None
    unsqueeze_267: "f32[1, 1984, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 2);  unsqueeze_266 = None
    unsqueeze_268: "f32[1, 1984, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 3);  unsqueeze_267 = None
    mul_460: "f32[1984]" = torch.ops.aten.mul.Tensor(squeeze_193, primals_129);  primals_129 = None
    unsqueeze_269: "f32[1, 1984]" = torch.ops.aten.unsqueeze.default(mul_460, 0);  mul_460 = None
    unsqueeze_270: "f32[1, 1984, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 2);  unsqueeze_269 = None
    unsqueeze_271: "f32[1, 1984, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 3);  unsqueeze_270 = None
    sub_66: "f32[8, 1984, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_262);  convolution_64 = unsqueeze_262 = None
    mul_461: "f32[8, 1984, 7, 7]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_268);  sub_66 = unsqueeze_268 = None
    sub_67: "f32[8, 1984, 7, 7]" = torch.ops.aten.sub.Tensor(where, mul_461);  where = mul_461 = None
    sub_68: "f32[8, 1984, 7, 7]" = torch.ops.aten.sub.Tensor(sub_67, unsqueeze_265);  sub_67 = unsqueeze_265 = None
    mul_462: "f32[8, 1984, 7, 7]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_271);  sub_68 = unsqueeze_271 = None
    mul_463: "f32[1984]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_193);  sum_3 = squeeze_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:168, code: x = self.conv_head(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_462, add_334, primals_195, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_462 = add_334 = primals_195 = None
    getitem_130: "f32[8, 352, 7, 7]" = convolution_backward[0]
    getitem_131: "f32[1984, 352, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_272: "f32[1, 352]" = torch.ops.aten.unsqueeze.default(squeeze_189, 0);  squeeze_189 = None
    unsqueeze_273: "f32[1, 352, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 2);  unsqueeze_272 = None
    unsqueeze_274: "f32[1, 352, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 3);  unsqueeze_273 = None
    sum_4: "f32[352]" = torch.ops.aten.sum.dim_IntList(getitem_130, [0, 2, 3])
    sub_69: "f32[8, 352, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_274)
    mul_464: "f32[8, 352, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_130, sub_69);  sub_69 = None
    sum_5: "f32[352]" = torch.ops.aten.sum.dim_IntList(mul_464, [0, 2, 3]);  mul_464 = None
    mul_465: "f32[352]" = torch.ops.aten.mul.Tensor(sum_4, 0.002551020408163265)
    unsqueeze_275: "f32[1, 352]" = torch.ops.aten.unsqueeze.default(mul_465, 0);  mul_465 = None
    unsqueeze_276: "f32[1, 352, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 2);  unsqueeze_275 = None
    unsqueeze_277: "f32[1, 352, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 3);  unsqueeze_276 = None
    mul_466: "f32[352]" = torch.ops.aten.mul.Tensor(sum_5, 0.002551020408163265)
    mul_467: "f32[352]" = torch.ops.aten.mul.Tensor(squeeze_190, squeeze_190)
    mul_468: "f32[352]" = torch.ops.aten.mul.Tensor(mul_466, mul_467);  mul_466 = mul_467 = None
    unsqueeze_278: "f32[1, 352]" = torch.ops.aten.unsqueeze.default(mul_468, 0);  mul_468 = None
    unsqueeze_279: "f32[1, 352, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 2);  unsqueeze_278 = None
    unsqueeze_280: "f32[1, 352, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 3);  unsqueeze_279 = None
    mul_469: "f32[352]" = torch.ops.aten.mul.Tensor(squeeze_190, primals_127);  primals_127 = None
    unsqueeze_281: "f32[1, 352]" = torch.ops.aten.unsqueeze.default(mul_469, 0);  mul_469 = None
    unsqueeze_282: "f32[1, 352, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 2);  unsqueeze_281 = None
    unsqueeze_283: "f32[1, 352, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 3);  unsqueeze_282 = None
    sub_70: "f32[8, 352, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_274);  convolution_63 = unsqueeze_274 = None
    mul_470: "f32[8, 352, 7, 7]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_280);  sub_70 = unsqueeze_280 = None
    sub_71: "f32[8, 352, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_130, mul_470);  getitem_130 = mul_470 = None
    sub_72: "f32[8, 352, 7, 7]" = torch.ops.aten.sub.Tensor(sub_71, unsqueeze_277);  sub_71 = unsqueeze_277 = None
    mul_471: "f32[8, 352, 7, 7]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_283);  sub_72 = unsqueeze_283 = None
    mul_472: "f32[352]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_190);  sum_5 = squeeze_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_471, relu_42, primals_194, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_471 = primals_194 = None
    getitem_133: "f32[8, 1104, 7, 7]" = convolution_backward_1[0]
    getitem_134: "f32[352, 1104, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_48: "f32[8, 1104, 7, 7]" = torch.ops.aten.alias.default(relu_42);  relu_42 = None
    alias_49: "f32[8, 1104, 7, 7]" = torch.ops.aten.alias.default(alias_48);  alias_48 = None
    le_1: "b8[8, 1104, 7, 7]" = torch.ops.aten.le.Scalar(alias_49, 0);  alias_49 = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_1: "f32[8, 1104, 7, 7]" = torch.ops.aten.where.self(le_1, scalar_tensor_1, getitem_133);  le_1 = scalar_tensor_1 = getitem_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_284: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(squeeze_186, 0);  squeeze_186 = None
    unsqueeze_285: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 2);  unsqueeze_284 = None
    unsqueeze_286: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 3);  unsqueeze_285 = None
    sum_6: "f32[1104]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_73: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_286)
    mul_473: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, sub_73);  sub_73 = None
    sum_7: "f32[1104]" = torch.ops.aten.sum.dim_IntList(mul_473, [0, 2, 3]);  mul_473 = None
    mul_474: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_6, 0.002551020408163265)
    unsqueeze_287: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_474, 0);  mul_474 = None
    unsqueeze_288: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 2);  unsqueeze_287 = None
    unsqueeze_289: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 3);  unsqueeze_288 = None
    mul_475: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_7, 0.002551020408163265)
    mul_476: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_187, squeeze_187)
    mul_477: "f32[1104]" = torch.ops.aten.mul.Tensor(mul_475, mul_476);  mul_475 = mul_476 = None
    unsqueeze_290: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_477, 0);  mul_477 = None
    unsqueeze_291: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 2);  unsqueeze_290 = None
    unsqueeze_292: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 3);  unsqueeze_291 = None
    mul_478: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_187, primals_125);  primals_125 = None
    unsqueeze_293: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_478, 0);  mul_478 = None
    unsqueeze_294: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 2);  unsqueeze_293 = None
    unsqueeze_295: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 3);  unsqueeze_294 = None
    sub_74: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_286);  convolution_62 = unsqueeze_286 = None
    mul_479: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_292);  sub_74 = unsqueeze_292 = None
    sub_75: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(where_1, mul_479);  where_1 = mul_479 = None
    sub_76: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(sub_75, unsqueeze_289);  sub_75 = unsqueeze_289 = None
    mul_480: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_295);  sub_76 = unsqueeze_295 = None
    mul_481: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_187);  sum_7 = squeeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_480, relu_41, primals_193, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1104, [True, True, False]);  mul_480 = primals_193 = None
    getitem_136: "f32[8, 1104, 7, 7]" = convolution_backward_2[0]
    getitem_137: "f32[1104, 1, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_51: "f32[8, 1104, 7, 7]" = torch.ops.aten.alias.default(relu_41);  relu_41 = None
    alias_52: "f32[8, 1104, 7, 7]" = torch.ops.aten.alias.default(alias_51);  alias_51 = None
    le_2: "b8[8, 1104, 7, 7]" = torch.ops.aten.le.Scalar(alias_52, 0);  alias_52 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_2: "f32[8, 1104, 7, 7]" = torch.ops.aten.where.self(le_2, scalar_tensor_2, getitem_136);  le_2 = scalar_tensor_2 = getitem_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_296: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(squeeze_183, 0);  squeeze_183 = None
    unsqueeze_297: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 2);  unsqueeze_296 = None
    unsqueeze_298: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 3);  unsqueeze_297 = None
    sum_8: "f32[1104]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_77: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_298)
    mul_482: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_77);  sub_77 = None
    sum_9: "f32[1104]" = torch.ops.aten.sum.dim_IntList(mul_482, [0, 2, 3]);  mul_482 = None
    mul_483: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_8, 0.002551020408163265)
    unsqueeze_299: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_483, 0);  mul_483 = None
    unsqueeze_300: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 2);  unsqueeze_299 = None
    unsqueeze_301: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 3);  unsqueeze_300 = None
    mul_484: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_9, 0.002551020408163265)
    mul_485: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_184, squeeze_184)
    mul_486: "f32[1104]" = torch.ops.aten.mul.Tensor(mul_484, mul_485);  mul_484 = mul_485 = None
    unsqueeze_302: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_486, 0);  mul_486 = None
    unsqueeze_303: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 2);  unsqueeze_302 = None
    unsqueeze_304: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 3);  unsqueeze_303 = None
    mul_487: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_184, primals_123);  primals_123 = None
    unsqueeze_305: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_487, 0);  mul_487 = None
    unsqueeze_306: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 2);  unsqueeze_305 = None
    unsqueeze_307: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, 3);  unsqueeze_306 = None
    sub_78: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_298);  convolution_61 = unsqueeze_298 = None
    mul_488: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_304);  sub_78 = unsqueeze_304 = None
    sub_79: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(where_2, mul_488);  where_2 = mul_488 = None
    sub_80: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(sub_79, unsqueeze_301);  sub_79 = unsqueeze_301 = None
    mul_489: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_307);  sub_80 = unsqueeze_307 = None
    mul_490: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_184);  sum_9 = squeeze_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_489, add_319, primals_192, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_489 = add_319 = primals_192 = None
    getitem_139: "f32[8, 184, 7, 7]" = convolution_backward_3[0]
    getitem_140: "f32[1104, 184, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_308: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(squeeze_180, 0);  squeeze_180 = None
    unsqueeze_309: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 2);  unsqueeze_308 = None
    unsqueeze_310: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 3);  unsqueeze_309 = None
    sum_10: "f32[184]" = torch.ops.aten.sum.dim_IntList(getitem_139, [0, 2, 3])
    sub_81: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_310)
    mul_491: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_139, sub_81);  sub_81 = None
    sum_11: "f32[184]" = torch.ops.aten.sum.dim_IntList(mul_491, [0, 2, 3]);  mul_491 = None
    mul_492: "f32[184]" = torch.ops.aten.mul.Tensor(sum_10, 0.002551020408163265)
    unsqueeze_311: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_492, 0);  mul_492 = None
    unsqueeze_312: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 2);  unsqueeze_311 = None
    unsqueeze_313: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 3);  unsqueeze_312 = None
    mul_493: "f32[184]" = torch.ops.aten.mul.Tensor(sum_11, 0.002551020408163265)
    mul_494: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
    mul_495: "f32[184]" = torch.ops.aten.mul.Tensor(mul_493, mul_494);  mul_493 = mul_494 = None
    unsqueeze_314: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_495, 0);  mul_495 = None
    unsqueeze_315: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 2);  unsqueeze_314 = None
    unsqueeze_316: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 3);  unsqueeze_315 = None
    mul_496: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_181, primals_121);  primals_121 = None
    unsqueeze_317: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_496, 0);  mul_496 = None
    unsqueeze_318: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None
    unsqueeze_319: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, 3);  unsqueeze_318 = None
    sub_82: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_310);  convolution_60 = unsqueeze_310 = None
    mul_497: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_316);  sub_82 = unsqueeze_316 = None
    sub_83: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_139, mul_497);  mul_497 = None
    sub_84: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(sub_83, unsqueeze_313);  sub_83 = unsqueeze_313 = None
    mul_498: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_319);  sub_84 = unsqueeze_319 = None
    mul_499: "f32[184]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_181);  sum_11 = squeeze_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_498, relu_40, primals_191, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_498 = primals_191 = None
    getitem_142: "f32[8, 1104, 7, 7]" = convolution_backward_4[0]
    getitem_143: "f32[184, 1104, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_54: "f32[8, 1104, 7, 7]" = torch.ops.aten.alias.default(relu_40);  relu_40 = None
    alias_55: "f32[8, 1104, 7, 7]" = torch.ops.aten.alias.default(alias_54);  alias_54 = None
    le_3: "b8[8, 1104, 7, 7]" = torch.ops.aten.le.Scalar(alias_55, 0);  alias_55 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_3: "f32[8, 1104, 7, 7]" = torch.ops.aten.where.self(le_3, scalar_tensor_3, getitem_142);  le_3 = scalar_tensor_3 = getitem_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_320: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(squeeze_177, 0);  squeeze_177 = None
    unsqueeze_321: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 2);  unsqueeze_320 = None
    unsqueeze_322: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 3);  unsqueeze_321 = None
    sum_12: "f32[1104]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_85: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_322)
    mul_500: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_85);  sub_85 = None
    sum_13: "f32[1104]" = torch.ops.aten.sum.dim_IntList(mul_500, [0, 2, 3]);  mul_500 = None
    mul_501: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_12, 0.002551020408163265)
    unsqueeze_323: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_501, 0);  mul_501 = None
    unsqueeze_324: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 2);  unsqueeze_323 = None
    unsqueeze_325: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, 3);  unsqueeze_324 = None
    mul_502: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_13, 0.002551020408163265)
    mul_503: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_178, squeeze_178)
    mul_504: "f32[1104]" = torch.ops.aten.mul.Tensor(mul_502, mul_503);  mul_502 = mul_503 = None
    unsqueeze_326: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_504, 0);  mul_504 = None
    unsqueeze_327: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 2);  unsqueeze_326 = None
    unsqueeze_328: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 3);  unsqueeze_327 = None
    mul_505: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_178, primals_119);  primals_119 = None
    unsqueeze_329: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_505, 0);  mul_505 = None
    unsqueeze_330: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 2);  unsqueeze_329 = None
    unsqueeze_331: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 3);  unsqueeze_330 = None
    sub_86: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_322);  convolution_59 = unsqueeze_322 = None
    mul_506: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_328);  sub_86 = unsqueeze_328 = None
    sub_87: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(where_3, mul_506);  where_3 = mul_506 = None
    sub_88: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(sub_87, unsqueeze_325);  sub_87 = unsqueeze_325 = None
    mul_507: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_331);  sub_88 = unsqueeze_331 = None
    mul_508: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_178);  sum_13 = squeeze_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_507, relu_39, primals_190, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1104, [True, True, False]);  mul_507 = primals_190 = None
    getitem_145: "f32[8, 1104, 7, 7]" = convolution_backward_5[0]
    getitem_146: "f32[1104, 1, 5, 5]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_57: "f32[8, 1104, 7, 7]" = torch.ops.aten.alias.default(relu_39);  relu_39 = None
    alias_58: "f32[8, 1104, 7, 7]" = torch.ops.aten.alias.default(alias_57);  alias_57 = None
    le_4: "b8[8, 1104, 7, 7]" = torch.ops.aten.le.Scalar(alias_58, 0);  alias_58 = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_4: "f32[8, 1104, 7, 7]" = torch.ops.aten.where.self(le_4, scalar_tensor_4, getitem_145);  le_4 = scalar_tensor_4 = getitem_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_332: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(squeeze_174, 0);  squeeze_174 = None
    unsqueeze_333: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 2);  unsqueeze_332 = None
    unsqueeze_334: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 3);  unsqueeze_333 = None
    sum_14: "f32[1104]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_89: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_334)
    mul_509: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, sub_89);  sub_89 = None
    sum_15: "f32[1104]" = torch.ops.aten.sum.dim_IntList(mul_509, [0, 2, 3]);  mul_509 = None
    mul_510: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_14, 0.002551020408163265)
    unsqueeze_335: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_510, 0);  mul_510 = None
    unsqueeze_336: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 2);  unsqueeze_335 = None
    unsqueeze_337: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 3);  unsqueeze_336 = None
    mul_511: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_15, 0.002551020408163265)
    mul_512: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
    mul_513: "f32[1104]" = torch.ops.aten.mul.Tensor(mul_511, mul_512);  mul_511 = mul_512 = None
    unsqueeze_338: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_513, 0);  mul_513 = None
    unsqueeze_339: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 2);  unsqueeze_338 = None
    unsqueeze_340: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 3);  unsqueeze_339 = None
    mul_514: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_175, primals_117);  primals_117 = None
    unsqueeze_341: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_514, 0);  mul_514 = None
    unsqueeze_342: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 2);  unsqueeze_341 = None
    unsqueeze_343: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 3);  unsqueeze_342 = None
    sub_90: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_334);  convolution_58 = unsqueeze_334 = None
    mul_515: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_340);  sub_90 = unsqueeze_340 = None
    sub_91: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(where_4, mul_515);  where_4 = mul_515 = None
    sub_92: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(sub_91, unsqueeze_337);  sub_91 = unsqueeze_337 = None
    mul_516: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_343);  sub_92 = unsqueeze_343 = None
    mul_517: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_175);  sum_15 = squeeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_516, add_303, primals_189, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_516 = add_303 = primals_189 = None
    getitem_148: "f32[8, 184, 7, 7]" = convolution_backward_6[0]
    getitem_149: "f32[1104, 184, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_340: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(getitem_139, getitem_148);  getitem_139 = getitem_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_344: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(squeeze_171, 0);  squeeze_171 = None
    unsqueeze_345: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 2);  unsqueeze_344 = None
    unsqueeze_346: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 3);  unsqueeze_345 = None
    sum_16: "f32[184]" = torch.ops.aten.sum.dim_IntList(add_340, [0, 2, 3])
    sub_93: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_346)
    mul_518: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(add_340, sub_93);  sub_93 = None
    sum_17: "f32[184]" = torch.ops.aten.sum.dim_IntList(mul_518, [0, 2, 3]);  mul_518 = None
    mul_519: "f32[184]" = torch.ops.aten.mul.Tensor(sum_16, 0.002551020408163265)
    unsqueeze_347: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_519, 0);  mul_519 = None
    unsqueeze_348: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 2);  unsqueeze_347 = None
    unsqueeze_349: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 3);  unsqueeze_348 = None
    mul_520: "f32[184]" = torch.ops.aten.mul.Tensor(sum_17, 0.002551020408163265)
    mul_521: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
    mul_522: "f32[184]" = torch.ops.aten.mul.Tensor(mul_520, mul_521);  mul_520 = mul_521 = None
    unsqueeze_350: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_522, 0);  mul_522 = None
    unsqueeze_351: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 2);  unsqueeze_350 = None
    unsqueeze_352: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 3);  unsqueeze_351 = None
    mul_523: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_172, primals_115);  primals_115 = None
    unsqueeze_353: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_523, 0);  mul_523 = None
    unsqueeze_354: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 2);  unsqueeze_353 = None
    unsqueeze_355: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, 3);  unsqueeze_354 = None
    sub_94: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_346);  convolution_57 = unsqueeze_346 = None
    mul_524: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_352);  sub_94 = unsqueeze_352 = None
    sub_95: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(add_340, mul_524);  mul_524 = None
    sub_96: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(sub_95, unsqueeze_349);  sub_95 = unsqueeze_349 = None
    mul_525: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_355);  sub_96 = unsqueeze_355 = None
    mul_526: "f32[184]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_172);  sum_17 = squeeze_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_525, relu_38, primals_188, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_525 = primals_188 = None
    getitem_151: "f32[8, 1104, 7, 7]" = convolution_backward_7[0]
    getitem_152: "f32[184, 1104, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_60: "f32[8, 1104, 7, 7]" = torch.ops.aten.alias.default(relu_38);  relu_38 = None
    alias_61: "f32[8, 1104, 7, 7]" = torch.ops.aten.alias.default(alias_60);  alias_60 = None
    le_5: "b8[8, 1104, 7, 7]" = torch.ops.aten.le.Scalar(alias_61, 0);  alias_61 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_5: "f32[8, 1104, 7, 7]" = torch.ops.aten.where.self(le_5, scalar_tensor_5, getitem_151);  le_5 = scalar_tensor_5 = getitem_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_356: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(squeeze_168, 0);  squeeze_168 = None
    unsqueeze_357: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 2);  unsqueeze_356 = None
    unsqueeze_358: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 3);  unsqueeze_357 = None
    sum_18: "f32[1104]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_97: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_358)
    mul_527: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, sub_97);  sub_97 = None
    sum_19: "f32[1104]" = torch.ops.aten.sum.dim_IntList(mul_527, [0, 2, 3]);  mul_527 = None
    mul_528: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_18, 0.002551020408163265)
    unsqueeze_359: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_528, 0);  mul_528 = None
    unsqueeze_360: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 2);  unsqueeze_359 = None
    unsqueeze_361: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, 3);  unsqueeze_360 = None
    mul_529: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_19, 0.002551020408163265)
    mul_530: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
    mul_531: "f32[1104]" = torch.ops.aten.mul.Tensor(mul_529, mul_530);  mul_529 = mul_530 = None
    unsqueeze_362: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_531, 0);  mul_531 = None
    unsqueeze_363: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 2);  unsqueeze_362 = None
    unsqueeze_364: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 3);  unsqueeze_363 = None
    mul_532: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_169, primals_113);  primals_113 = None
    unsqueeze_365: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_532, 0);  mul_532 = None
    unsqueeze_366: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 2);  unsqueeze_365 = None
    unsqueeze_367: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, 3);  unsqueeze_366 = None
    sub_98: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_358);  convolution_56 = unsqueeze_358 = None
    mul_533: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_364);  sub_98 = unsqueeze_364 = None
    sub_99: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(where_5, mul_533);  where_5 = mul_533 = None
    sub_100: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(sub_99, unsqueeze_361);  sub_99 = unsqueeze_361 = None
    mul_534: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_367);  sub_100 = unsqueeze_367 = None
    mul_535: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_169);  sum_19 = squeeze_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_534, relu_37, primals_187, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1104, [True, True, False]);  mul_534 = primals_187 = None
    getitem_154: "f32[8, 1104, 7, 7]" = convolution_backward_8[0]
    getitem_155: "f32[1104, 1, 5, 5]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_63: "f32[8, 1104, 7, 7]" = torch.ops.aten.alias.default(relu_37);  relu_37 = None
    alias_64: "f32[8, 1104, 7, 7]" = torch.ops.aten.alias.default(alias_63);  alias_63 = None
    le_6: "b8[8, 1104, 7, 7]" = torch.ops.aten.le.Scalar(alias_64, 0);  alias_64 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_6: "f32[8, 1104, 7, 7]" = torch.ops.aten.where.self(le_6, scalar_tensor_6, getitem_154);  le_6 = scalar_tensor_6 = getitem_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_368: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(squeeze_165, 0);  squeeze_165 = None
    unsqueeze_369: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 2);  unsqueeze_368 = None
    unsqueeze_370: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 3);  unsqueeze_369 = None
    sum_20: "f32[1104]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_101: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_370)
    mul_536: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, sub_101);  sub_101 = None
    sum_21: "f32[1104]" = torch.ops.aten.sum.dim_IntList(mul_536, [0, 2, 3]);  mul_536 = None
    mul_537: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_20, 0.002551020408163265)
    unsqueeze_371: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_537, 0);  mul_537 = None
    unsqueeze_372: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 2);  unsqueeze_371 = None
    unsqueeze_373: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, 3);  unsqueeze_372 = None
    mul_538: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_21, 0.002551020408163265)
    mul_539: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
    mul_540: "f32[1104]" = torch.ops.aten.mul.Tensor(mul_538, mul_539);  mul_538 = mul_539 = None
    unsqueeze_374: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_540, 0);  mul_540 = None
    unsqueeze_375: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 2);  unsqueeze_374 = None
    unsqueeze_376: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 3);  unsqueeze_375 = None
    mul_541: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_166, primals_111);  primals_111 = None
    unsqueeze_377: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_541, 0);  mul_541 = None
    unsqueeze_378: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 2);  unsqueeze_377 = None
    unsqueeze_379: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 3);  unsqueeze_378 = None
    sub_102: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_370);  convolution_55 = unsqueeze_370 = None
    mul_542: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_376);  sub_102 = unsqueeze_376 = None
    sub_103: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(where_6, mul_542);  where_6 = mul_542 = None
    sub_104: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(sub_103, unsqueeze_373);  sub_103 = unsqueeze_373 = None
    mul_543: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_379);  sub_104 = unsqueeze_379 = None
    mul_544: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_166);  sum_21 = squeeze_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_543, add_287, primals_186, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_543 = add_287 = primals_186 = None
    getitem_157: "f32[8, 184, 7, 7]" = convolution_backward_9[0]
    getitem_158: "f32[1104, 184, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_341: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(add_340, getitem_157);  add_340 = getitem_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_380: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(squeeze_162, 0);  squeeze_162 = None
    unsqueeze_381: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 2);  unsqueeze_380 = None
    unsqueeze_382: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 3);  unsqueeze_381 = None
    sum_22: "f32[184]" = torch.ops.aten.sum.dim_IntList(add_341, [0, 2, 3])
    sub_105: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_382)
    mul_545: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(add_341, sub_105);  sub_105 = None
    sum_23: "f32[184]" = torch.ops.aten.sum.dim_IntList(mul_545, [0, 2, 3]);  mul_545 = None
    mul_546: "f32[184]" = torch.ops.aten.mul.Tensor(sum_22, 0.002551020408163265)
    unsqueeze_383: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_546, 0);  mul_546 = None
    unsqueeze_384: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 2);  unsqueeze_383 = None
    unsqueeze_385: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, 3);  unsqueeze_384 = None
    mul_547: "f32[184]" = torch.ops.aten.mul.Tensor(sum_23, 0.002551020408163265)
    mul_548: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
    mul_549: "f32[184]" = torch.ops.aten.mul.Tensor(mul_547, mul_548);  mul_547 = mul_548 = None
    unsqueeze_386: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_549, 0);  mul_549 = None
    unsqueeze_387: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 2);  unsqueeze_386 = None
    unsqueeze_388: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 3);  unsqueeze_387 = None
    mul_550: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_163, primals_109);  primals_109 = None
    unsqueeze_389: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_550, 0);  mul_550 = None
    unsqueeze_390: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 2);  unsqueeze_389 = None
    unsqueeze_391: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, 3);  unsqueeze_390 = None
    sub_106: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_382);  convolution_54 = unsqueeze_382 = None
    mul_551: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_388);  sub_106 = unsqueeze_388 = None
    sub_107: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(add_341, mul_551);  mul_551 = None
    sub_108: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(sub_107, unsqueeze_385);  sub_107 = unsqueeze_385 = None
    mul_552: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_391);  sub_108 = unsqueeze_391 = None
    mul_553: "f32[184]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_163);  sum_23 = squeeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_552, relu_36, primals_185, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_552 = primals_185 = None
    getitem_160: "f32[8, 1104, 7, 7]" = convolution_backward_10[0]
    getitem_161: "f32[184, 1104, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_66: "f32[8, 1104, 7, 7]" = torch.ops.aten.alias.default(relu_36);  relu_36 = None
    alias_67: "f32[8, 1104, 7, 7]" = torch.ops.aten.alias.default(alias_66);  alias_66 = None
    le_7: "b8[8, 1104, 7, 7]" = torch.ops.aten.le.Scalar(alias_67, 0);  alias_67 = None
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_7: "f32[8, 1104, 7, 7]" = torch.ops.aten.where.self(le_7, scalar_tensor_7, getitem_160);  le_7 = scalar_tensor_7 = getitem_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_392: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(squeeze_159, 0);  squeeze_159 = None
    unsqueeze_393: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 2);  unsqueeze_392 = None
    unsqueeze_394: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 3);  unsqueeze_393 = None
    sum_24: "f32[1104]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_109: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_394)
    mul_554: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, sub_109);  sub_109 = None
    sum_25: "f32[1104]" = torch.ops.aten.sum.dim_IntList(mul_554, [0, 2, 3]);  mul_554 = None
    mul_555: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_24, 0.002551020408163265)
    unsqueeze_395: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_555, 0);  mul_555 = None
    unsqueeze_396: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 2);  unsqueeze_395 = None
    unsqueeze_397: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, 3);  unsqueeze_396 = None
    mul_556: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_25, 0.002551020408163265)
    mul_557: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
    mul_558: "f32[1104]" = torch.ops.aten.mul.Tensor(mul_556, mul_557);  mul_556 = mul_557 = None
    unsqueeze_398: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_558, 0);  mul_558 = None
    unsqueeze_399: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 2);  unsqueeze_398 = None
    unsqueeze_400: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 3);  unsqueeze_399 = None
    mul_559: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_160, primals_107);  primals_107 = None
    unsqueeze_401: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_559, 0);  mul_559 = None
    unsqueeze_402: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 2);  unsqueeze_401 = None
    unsqueeze_403: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 3);  unsqueeze_402 = None
    sub_110: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_394);  convolution_53 = unsqueeze_394 = None
    mul_560: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_400);  sub_110 = unsqueeze_400 = None
    sub_111: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(where_7, mul_560);  where_7 = mul_560 = None
    sub_112: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(sub_111, unsqueeze_397);  sub_111 = unsqueeze_397 = None
    mul_561: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_403);  sub_112 = unsqueeze_403 = None
    mul_562: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_160);  sum_25 = squeeze_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_561, relu_35, primals_184, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1104, [True, True, False]);  mul_561 = primals_184 = None
    getitem_163: "f32[8, 1104, 7, 7]" = convolution_backward_11[0]
    getitem_164: "f32[1104, 1, 5, 5]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_69: "f32[8, 1104, 7, 7]" = torch.ops.aten.alias.default(relu_35);  relu_35 = None
    alias_70: "f32[8, 1104, 7, 7]" = torch.ops.aten.alias.default(alias_69);  alias_69 = None
    le_8: "b8[8, 1104, 7, 7]" = torch.ops.aten.le.Scalar(alias_70, 0);  alias_70 = None
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_8: "f32[8, 1104, 7, 7]" = torch.ops.aten.where.self(le_8, scalar_tensor_8, getitem_163);  le_8 = scalar_tensor_8 = getitem_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_404: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(squeeze_156, 0);  squeeze_156 = None
    unsqueeze_405: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 2);  unsqueeze_404 = None
    unsqueeze_406: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 3);  unsqueeze_405 = None
    sum_26: "f32[1104]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_113: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_406)
    mul_563: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(where_8, sub_113);  sub_113 = None
    sum_27: "f32[1104]" = torch.ops.aten.sum.dim_IntList(mul_563, [0, 2, 3]);  mul_563 = None
    mul_564: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_26, 0.002551020408163265)
    unsqueeze_407: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_564, 0);  mul_564 = None
    unsqueeze_408: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 2);  unsqueeze_407 = None
    unsqueeze_409: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, 3);  unsqueeze_408 = None
    mul_565: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_27, 0.002551020408163265)
    mul_566: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
    mul_567: "f32[1104]" = torch.ops.aten.mul.Tensor(mul_565, mul_566);  mul_565 = mul_566 = None
    unsqueeze_410: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_567, 0);  mul_567 = None
    unsqueeze_411: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 2);  unsqueeze_410 = None
    unsqueeze_412: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 3);  unsqueeze_411 = None
    mul_568: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_157, primals_105);  primals_105 = None
    unsqueeze_413: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_568, 0);  mul_568 = None
    unsqueeze_414: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 2);  unsqueeze_413 = None
    unsqueeze_415: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 3);  unsqueeze_414 = None
    sub_114: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_406);  convolution_52 = unsqueeze_406 = None
    mul_569: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_412);  sub_114 = unsqueeze_412 = None
    sub_115: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(where_8, mul_569);  where_8 = mul_569 = None
    sub_116: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(sub_115, unsqueeze_409);  sub_115 = unsqueeze_409 = None
    mul_570: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_415);  sub_116 = unsqueeze_415 = None
    mul_571: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_157);  sum_27 = squeeze_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_570, add_271, primals_183, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_570 = add_271 = primals_183 = None
    getitem_166: "f32[8, 184, 7, 7]" = convolution_backward_12[0]
    getitem_167: "f32[1104, 184, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_342: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(add_341, getitem_166);  add_341 = getitem_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_416: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
    unsqueeze_417: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 2);  unsqueeze_416 = None
    unsqueeze_418: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 3);  unsqueeze_417 = None
    sum_28: "f32[184]" = torch.ops.aten.sum.dim_IntList(add_342, [0, 2, 3])
    sub_117: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_418)
    mul_572: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(add_342, sub_117);  sub_117 = None
    sum_29: "f32[184]" = torch.ops.aten.sum.dim_IntList(mul_572, [0, 2, 3]);  mul_572 = None
    mul_573: "f32[184]" = torch.ops.aten.mul.Tensor(sum_28, 0.002551020408163265)
    unsqueeze_419: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_573, 0);  mul_573 = None
    unsqueeze_420: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 2);  unsqueeze_419 = None
    unsqueeze_421: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 3);  unsqueeze_420 = None
    mul_574: "f32[184]" = torch.ops.aten.mul.Tensor(sum_29, 0.002551020408163265)
    mul_575: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
    mul_576: "f32[184]" = torch.ops.aten.mul.Tensor(mul_574, mul_575);  mul_574 = mul_575 = None
    unsqueeze_422: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_576, 0);  mul_576 = None
    unsqueeze_423: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 2);  unsqueeze_422 = None
    unsqueeze_424: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 3);  unsqueeze_423 = None
    mul_577: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_103);  primals_103 = None
    unsqueeze_425: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_577, 0);  mul_577 = None
    unsqueeze_426: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 2);  unsqueeze_425 = None
    unsqueeze_427: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 3);  unsqueeze_426 = None
    sub_118: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_418);  convolution_51 = unsqueeze_418 = None
    mul_578: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_424);  sub_118 = unsqueeze_424 = None
    sub_119: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(add_342, mul_578);  add_342 = mul_578 = None
    sub_120: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(sub_119, unsqueeze_421);  sub_119 = unsqueeze_421 = None
    mul_579: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_427);  sub_120 = unsqueeze_427 = None
    mul_580: "f32[184]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_154);  sum_29 = squeeze_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_579, relu_34, primals_182, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_579 = primals_182 = None
    getitem_169: "f32[8, 672, 7, 7]" = convolution_backward_13[0]
    getitem_170: "f32[184, 672, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_72: "f32[8, 672, 7, 7]" = torch.ops.aten.alias.default(relu_34);  relu_34 = None
    alias_73: "f32[8, 672, 7, 7]" = torch.ops.aten.alias.default(alias_72);  alias_72 = None
    le_9: "b8[8, 672, 7, 7]" = torch.ops.aten.le.Scalar(alias_73, 0);  alias_73 = None
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_9: "f32[8, 672, 7, 7]" = torch.ops.aten.where.self(le_9, scalar_tensor_9, getitem_169);  le_9 = scalar_tensor_9 = getitem_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_428: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_150, 0);  squeeze_150 = None
    unsqueeze_429: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 2);  unsqueeze_428 = None
    unsqueeze_430: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 3);  unsqueeze_429 = None
    sum_30: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_121: "f32[8, 672, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_430)
    mul_581: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(where_9, sub_121);  sub_121 = None
    sum_31: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_581, [0, 2, 3]);  mul_581 = None
    mul_582: "f32[672]" = torch.ops.aten.mul.Tensor(sum_30, 0.002551020408163265)
    unsqueeze_431: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_582, 0);  mul_582 = None
    unsqueeze_432: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 2);  unsqueeze_431 = None
    unsqueeze_433: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 3);  unsqueeze_432 = None
    mul_583: "f32[672]" = torch.ops.aten.mul.Tensor(sum_31, 0.002551020408163265)
    mul_584: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
    mul_585: "f32[672]" = torch.ops.aten.mul.Tensor(mul_583, mul_584);  mul_583 = mul_584 = None
    unsqueeze_434: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_585, 0);  mul_585 = None
    unsqueeze_435: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 2);  unsqueeze_434 = None
    unsqueeze_436: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 3);  unsqueeze_435 = None
    mul_586: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_101);  primals_101 = None
    unsqueeze_437: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_586, 0);  mul_586 = None
    unsqueeze_438: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 2);  unsqueeze_437 = None
    unsqueeze_439: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 3);  unsqueeze_438 = None
    sub_122: "f32[8, 672, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_430);  convolution_50 = unsqueeze_430 = None
    mul_587: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_436);  sub_122 = unsqueeze_436 = None
    sub_123: "f32[8, 672, 7, 7]" = torch.ops.aten.sub.Tensor(where_9, mul_587);  where_9 = mul_587 = None
    sub_124: "f32[8, 672, 7, 7]" = torch.ops.aten.sub.Tensor(sub_123, unsqueeze_433);  sub_123 = unsqueeze_433 = None
    mul_588: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_439);  sub_124 = unsqueeze_439 = None
    mul_589: "f32[672]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_151);  sum_31 = squeeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_588, relu_33, primals_181, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False]);  mul_588 = primals_181 = None
    getitem_172: "f32[8, 672, 14, 14]" = convolution_backward_14[0]
    getitem_173: "f32[672, 1, 5, 5]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_75: "f32[8, 672, 14, 14]" = torch.ops.aten.alias.default(relu_33);  relu_33 = None
    alias_76: "f32[8, 672, 14, 14]" = torch.ops.aten.alias.default(alias_75);  alias_75 = None
    le_10: "b8[8, 672, 14, 14]" = torch.ops.aten.le.Scalar(alias_76, 0);  alias_76 = None
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_10: "f32[8, 672, 14, 14]" = torch.ops.aten.where.self(le_10, scalar_tensor_10, getitem_172);  le_10 = scalar_tensor_10 = getitem_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_440: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_147, 0);  squeeze_147 = None
    unsqueeze_441: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 2);  unsqueeze_440 = None
    unsqueeze_442: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 3);  unsqueeze_441 = None
    sum_32: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_125: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_442)
    mul_590: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, sub_125);  sub_125 = None
    sum_33: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_590, [0, 2, 3]);  mul_590 = None
    mul_591: "f32[672]" = torch.ops.aten.mul.Tensor(sum_32, 0.0006377551020408163)
    unsqueeze_443: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_591, 0);  mul_591 = None
    unsqueeze_444: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 2);  unsqueeze_443 = None
    unsqueeze_445: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, 3);  unsqueeze_444 = None
    mul_592: "f32[672]" = torch.ops.aten.mul.Tensor(sum_33, 0.0006377551020408163)
    mul_593: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
    mul_594: "f32[672]" = torch.ops.aten.mul.Tensor(mul_592, mul_593);  mul_592 = mul_593 = None
    unsqueeze_446: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_594, 0);  mul_594 = None
    unsqueeze_447: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 2);  unsqueeze_446 = None
    unsqueeze_448: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 3);  unsqueeze_447 = None
    mul_595: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_99);  primals_99 = None
    unsqueeze_449: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_595, 0);  mul_595 = None
    unsqueeze_450: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
    unsqueeze_451: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 3);  unsqueeze_450 = None
    sub_126: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_442);  convolution_49 = unsqueeze_442 = None
    mul_596: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_448);  sub_126 = unsqueeze_448 = None
    sub_127: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(where_10, mul_596);  where_10 = mul_596 = None
    sub_128: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(sub_127, unsqueeze_445);  sub_127 = unsqueeze_445 = None
    mul_597: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_451);  sub_128 = unsqueeze_451 = None
    mul_598: "f32[672]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_148);  sum_33 = squeeze_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_597, add_256, primals_180, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_597 = add_256 = primals_180 = None
    getitem_175: "f32[8, 112, 14, 14]" = convolution_backward_15[0]
    getitem_176: "f32[672, 112, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_452: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
    unsqueeze_453: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 2);  unsqueeze_452 = None
    unsqueeze_454: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 3);  unsqueeze_453 = None
    sum_34: "f32[112]" = torch.ops.aten.sum.dim_IntList(getitem_175, [0, 2, 3])
    sub_129: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_454)
    mul_599: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_175, sub_129);  sub_129 = None
    sum_35: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_599, [0, 2, 3]);  mul_599 = None
    mul_600: "f32[112]" = torch.ops.aten.mul.Tensor(sum_34, 0.0006377551020408163)
    unsqueeze_455: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_600, 0);  mul_600 = None
    unsqueeze_456: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 2);  unsqueeze_455 = None
    unsqueeze_457: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, 3);  unsqueeze_456 = None
    mul_601: "f32[112]" = torch.ops.aten.mul.Tensor(sum_35, 0.0006377551020408163)
    mul_602: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
    mul_603: "f32[112]" = torch.ops.aten.mul.Tensor(mul_601, mul_602);  mul_601 = mul_602 = None
    unsqueeze_458: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_603, 0);  mul_603 = None
    unsqueeze_459: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 2);  unsqueeze_458 = None
    unsqueeze_460: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 3);  unsqueeze_459 = None
    mul_604: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_97);  primals_97 = None
    unsqueeze_461: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_604, 0);  mul_604 = None
    unsqueeze_462: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
    unsqueeze_463: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 3);  unsqueeze_462 = None
    sub_130: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_454);  convolution_48 = unsqueeze_454 = None
    mul_605: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_460);  sub_130 = unsqueeze_460 = None
    sub_131: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_175, mul_605);  mul_605 = None
    sub_132: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(sub_131, unsqueeze_457);  sub_131 = unsqueeze_457 = None
    mul_606: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_463);  sub_132 = unsqueeze_463 = None
    mul_607: "f32[112]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_145);  sum_35 = squeeze_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_606, relu_32, primals_179, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_606 = primals_179 = None
    getitem_178: "f32[8, 336, 14, 14]" = convolution_backward_16[0]
    getitem_179: "f32[112, 336, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_78: "f32[8, 336, 14, 14]" = torch.ops.aten.alias.default(relu_32);  relu_32 = None
    alias_79: "f32[8, 336, 14, 14]" = torch.ops.aten.alias.default(alias_78);  alias_78 = None
    le_11: "b8[8, 336, 14, 14]" = torch.ops.aten.le.Scalar(alias_79, 0);  alias_79 = None
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_11: "f32[8, 336, 14, 14]" = torch.ops.aten.where.self(le_11, scalar_tensor_11, getitem_178);  le_11 = scalar_tensor_11 = getitem_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_464: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
    unsqueeze_465: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 2);  unsqueeze_464 = None
    unsqueeze_466: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 3);  unsqueeze_465 = None
    sum_36: "f32[336]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_133: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_466)
    mul_608: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, sub_133);  sub_133 = None
    sum_37: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_608, [0, 2, 3]);  mul_608 = None
    mul_609: "f32[336]" = torch.ops.aten.mul.Tensor(sum_36, 0.0006377551020408163)
    unsqueeze_467: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_609, 0);  mul_609 = None
    unsqueeze_468: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 2);  unsqueeze_467 = None
    unsqueeze_469: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, 3);  unsqueeze_468 = None
    mul_610: "f32[336]" = torch.ops.aten.mul.Tensor(sum_37, 0.0006377551020408163)
    mul_611: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
    mul_612: "f32[336]" = torch.ops.aten.mul.Tensor(mul_610, mul_611);  mul_610 = mul_611 = None
    unsqueeze_470: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_612, 0);  mul_612 = None
    unsqueeze_471: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 2);  unsqueeze_470 = None
    unsqueeze_472: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 3);  unsqueeze_471 = None
    mul_613: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_95);  primals_95 = None
    unsqueeze_473: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_613, 0);  mul_613 = None
    unsqueeze_474: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
    unsqueeze_475: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 3);  unsqueeze_474 = None
    sub_134: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_466);  convolution_47 = unsqueeze_466 = None
    mul_614: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_472);  sub_134 = unsqueeze_472 = None
    sub_135: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(where_11, mul_614);  where_11 = mul_614 = None
    sub_136: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(sub_135, unsqueeze_469);  sub_135 = unsqueeze_469 = None
    mul_615: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_475);  sub_136 = unsqueeze_475 = None
    mul_616: "f32[336]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_142);  sum_37 = squeeze_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_615, relu_31, primals_178, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 336, [True, True, False]);  mul_615 = primals_178 = None
    getitem_181: "f32[8, 336, 14, 14]" = convolution_backward_17[0]
    getitem_182: "f32[336, 1, 5, 5]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_81: "f32[8, 336, 14, 14]" = torch.ops.aten.alias.default(relu_31);  relu_31 = None
    alias_82: "f32[8, 336, 14, 14]" = torch.ops.aten.alias.default(alias_81);  alias_81 = None
    le_12: "b8[8, 336, 14, 14]" = torch.ops.aten.le.Scalar(alias_82, 0);  alias_82 = None
    scalar_tensor_12: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_12: "f32[8, 336, 14, 14]" = torch.ops.aten.where.self(le_12, scalar_tensor_12, getitem_181);  le_12 = scalar_tensor_12 = getitem_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_476: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
    unsqueeze_477: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 2);  unsqueeze_476 = None
    unsqueeze_478: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 3);  unsqueeze_477 = None
    sum_38: "f32[336]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_137: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_478)
    mul_617: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, sub_137);  sub_137 = None
    sum_39: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_617, [0, 2, 3]);  mul_617 = None
    mul_618: "f32[336]" = torch.ops.aten.mul.Tensor(sum_38, 0.0006377551020408163)
    unsqueeze_479: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_618, 0);  mul_618 = None
    unsqueeze_480: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 2);  unsqueeze_479 = None
    unsqueeze_481: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 3);  unsqueeze_480 = None
    mul_619: "f32[336]" = torch.ops.aten.mul.Tensor(sum_39, 0.0006377551020408163)
    mul_620: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
    mul_621: "f32[336]" = torch.ops.aten.mul.Tensor(mul_619, mul_620);  mul_619 = mul_620 = None
    unsqueeze_482: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_621, 0);  mul_621 = None
    unsqueeze_483: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 2);  unsqueeze_482 = None
    unsqueeze_484: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 3);  unsqueeze_483 = None
    mul_622: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_93);  primals_93 = None
    unsqueeze_485: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_622, 0);  mul_622 = None
    unsqueeze_486: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
    unsqueeze_487: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 3);  unsqueeze_486 = None
    sub_138: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_478);  convolution_46 = unsqueeze_478 = None
    mul_623: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_484);  sub_138 = unsqueeze_484 = None
    sub_139: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(where_12, mul_623);  where_12 = mul_623 = None
    sub_140: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(sub_139, unsqueeze_481);  sub_139 = unsqueeze_481 = None
    mul_624: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_487);  sub_140 = unsqueeze_487 = None
    mul_625: "f32[336]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_139);  sum_39 = squeeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_624, add_240, primals_177, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_624 = add_240 = primals_177 = None
    getitem_184: "f32[8, 112, 14, 14]" = convolution_backward_18[0]
    getitem_185: "f32[336, 112, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_343: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(getitem_175, getitem_184);  getitem_175 = getitem_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_488: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
    unsqueeze_489: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 2);  unsqueeze_488 = None
    unsqueeze_490: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 3);  unsqueeze_489 = None
    sum_40: "f32[112]" = torch.ops.aten.sum.dim_IntList(add_343, [0, 2, 3])
    sub_141: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_490)
    mul_626: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(add_343, sub_141);  sub_141 = None
    sum_41: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_626, [0, 2, 3]);  mul_626 = None
    mul_627: "f32[112]" = torch.ops.aten.mul.Tensor(sum_40, 0.0006377551020408163)
    unsqueeze_491: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_627, 0);  mul_627 = None
    unsqueeze_492: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 2);  unsqueeze_491 = None
    unsqueeze_493: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, 3);  unsqueeze_492 = None
    mul_628: "f32[112]" = torch.ops.aten.mul.Tensor(sum_41, 0.0006377551020408163)
    mul_629: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
    mul_630: "f32[112]" = torch.ops.aten.mul.Tensor(mul_628, mul_629);  mul_628 = mul_629 = None
    unsqueeze_494: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_630, 0);  mul_630 = None
    unsqueeze_495: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 2);  unsqueeze_494 = None
    unsqueeze_496: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 3);  unsqueeze_495 = None
    mul_631: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_91);  primals_91 = None
    unsqueeze_497: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_631, 0);  mul_631 = None
    unsqueeze_498: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
    unsqueeze_499: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 3);  unsqueeze_498 = None
    sub_142: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_490);  convolution_45 = unsqueeze_490 = None
    mul_632: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_496);  sub_142 = unsqueeze_496 = None
    sub_143: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(add_343, mul_632);  mul_632 = None
    sub_144: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(sub_143, unsqueeze_493);  sub_143 = unsqueeze_493 = None
    mul_633: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_499);  sub_144 = unsqueeze_499 = None
    mul_634: "f32[112]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_136);  sum_41 = squeeze_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_633, relu_30, primals_176, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_633 = primals_176 = None
    getitem_187: "f32[8, 672, 14, 14]" = convolution_backward_19[0]
    getitem_188: "f32[112, 672, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_84: "f32[8, 672, 14, 14]" = torch.ops.aten.alias.default(relu_30);  relu_30 = None
    alias_85: "f32[8, 672, 14, 14]" = torch.ops.aten.alias.default(alias_84);  alias_84 = None
    le_13: "b8[8, 672, 14, 14]" = torch.ops.aten.le.Scalar(alias_85, 0);  alias_85 = None
    scalar_tensor_13: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_13: "f32[8, 672, 14, 14]" = torch.ops.aten.where.self(le_13, scalar_tensor_13, getitem_187);  le_13 = scalar_tensor_13 = getitem_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_500: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
    unsqueeze_501: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 2);  unsqueeze_500 = None
    unsqueeze_502: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 3);  unsqueeze_501 = None
    sum_42: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_145: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_502)
    mul_635: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(where_13, sub_145);  sub_145 = None
    sum_43: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_635, [0, 2, 3]);  mul_635 = None
    mul_636: "f32[672]" = torch.ops.aten.mul.Tensor(sum_42, 0.0006377551020408163)
    unsqueeze_503: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_636, 0);  mul_636 = None
    unsqueeze_504: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 2);  unsqueeze_503 = None
    unsqueeze_505: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, 3);  unsqueeze_504 = None
    mul_637: "f32[672]" = torch.ops.aten.mul.Tensor(sum_43, 0.0006377551020408163)
    mul_638: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
    mul_639: "f32[672]" = torch.ops.aten.mul.Tensor(mul_637, mul_638);  mul_637 = mul_638 = None
    unsqueeze_506: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_639, 0);  mul_639 = None
    unsqueeze_507: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 2);  unsqueeze_506 = None
    unsqueeze_508: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 3);  unsqueeze_507 = None
    mul_640: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_89);  primals_89 = None
    unsqueeze_509: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_640, 0);  mul_640 = None
    unsqueeze_510: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 2);  unsqueeze_509 = None
    unsqueeze_511: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 3);  unsqueeze_510 = None
    sub_146: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_502);  convolution_44 = unsqueeze_502 = None
    mul_641: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_508);  sub_146 = unsqueeze_508 = None
    sub_147: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(where_13, mul_641);  where_13 = mul_641 = None
    sub_148: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(sub_147, unsqueeze_505);  sub_147 = unsqueeze_505 = None
    mul_642: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_511);  sub_148 = unsqueeze_511 = None
    mul_643: "f32[672]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_133);  sum_43 = squeeze_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_642, relu_29, primals_175, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False]);  mul_642 = primals_175 = None
    getitem_190: "f32[8, 672, 14, 14]" = convolution_backward_20[0]
    getitem_191: "f32[672, 1, 5, 5]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_87: "f32[8, 672, 14, 14]" = torch.ops.aten.alias.default(relu_29);  relu_29 = None
    alias_88: "f32[8, 672, 14, 14]" = torch.ops.aten.alias.default(alias_87);  alias_87 = None
    le_14: "b8[8, 672, 14, 14]" = torch.ops.aten.le.Scalar(alias_88, 0);  alias_88 = None
    scalar_tensor_14: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_14: "f32[8, 672, 14, 14]" = torch.ops.aten.where.self(le_14, scalar_tensor_14, getitem_190);  le_14 = scalar_tensor_14 = getitem_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_512: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
    unsqueeze_513: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 2);  unsqueeze_512 = None
    unsqueeze_514: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 3);  unsqueeze_513 = None
    sum_44: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_149: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_514)
    mul_644: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, sub_149);  sub_149 = None
    sum_45: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_644, [0, 2, 3]);  mul_644 = None
    mul_645: "f32[672]" = torch.ops.aten.mul.Tensor(sum_44, 0.0006377551020408163)
    unsqueeze_515: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_645, 0);  mul_645 = None
    unsqueeze_516: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 2);  unsqueeze_515 = None
    unsqueeze_517: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, 3);  unsqueeze_516 = None
    mul_646: "f32[672]" = torch.ops.aten.mul.Tensor(sum_45, 0.0006377551020408163)
    mul_647: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
    mul_648: "f32[672]" = torch.ops.aten.mul.Tensor(mul_646, mul_647);  mul_646 = mul_647 = None
    unsqueeze_518: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_648, 0);  mul_648 = None
    unsqueeze_519: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 2);  unsqueeze_518 = None
    unsqueeze_520: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_519, 3);  unsqueeze_519 = None
    mul_649: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_87);  primals_87 = None
    unsqueeze_521: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_649, 0);  mul_649 = None
    unsqueeze_522: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 2);  unsqueeze_521 = None
    unsqueeze_523: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, 3);  unsqueeze_522 = None
    sub_150: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_514);  convolution_43 = unsqueeze_514 = None
    mul_650: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_520);  sub_150 = unsqueeze_520 = None
    sub_151: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(where_14, mul_650);  where_14 = mul_650 = None
    sub_152: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(sub_151, unsqueeze_517);  sub_151 = unsqueeze_517 = None
    mul_651: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_523);  sub_152 = unsqueeze_523 = None
    mul_652: "f32[672]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_130);  sum_45 = squeeze_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_651, add_224, primals_174, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_651 = add_224 = primals_174 = None
    getitem_193: "f32[8, 112, 14, 14]" = convolution_backward_21[0]
    getitem_194: "f32[672, 112, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_344: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(add_343, getitem_193);  add_343 = getitem_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_524: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
    unsqueeze_525: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 2);  unsqueeze_524 = None
    unsqueeze_526: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_525, 3);  unsqueeze_525 = None
    sum_46: "f32[112]" = torch.ops.aten.sum.dim_IntList(add_344, [0, 2, 3])
    sub_153: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_526)
    mul_653: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(add_344, sub_153);  sub_153 = None
    sum_47: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_653, [0, 2, 3]);  mul_653 = None
    mul_654: "f32[112]" = torch.ops.aten.mul.Tensor(sum_46, 0.0006377551020408163)
    unsqueeze_527: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_654, 0);  mul_654 = None
    unsqueeze_528: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 2);  unsqueeze_527 = None
    unsqueeze_529: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, 3);  unsqueeze_528 = None
    mul_655: "f32[112]" = torch.ops.aten.mul.Tensor(sum_47, 0.0006377551020408163)
    mul_656: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
    mul_657: "f32[112]" = torch.ops.aten.mul.Tensor(mul_655, mul_656);  mul_655 = mul_656 = None
    unsqueeze_530: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_657, 0);  mul_657 = None
    unsqueeze_531: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 2);  unsqueeze_530 = None
    unsqueeze_532: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_531, 3);  unsqueeze_531 = None
    mul_658: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_85);  primals_85 = None
    unsqueeze_533: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_658, 0);  mul_658 = None
    unsqueeze_534: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 2);  unsqueeze_533 = None
    unsqueeze_535: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 3);  unsqueeze_534 = None
    sub_154: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_526);  convolution_42 = unsqueeze_526 = None
    mul_659: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_532);  sub_154 = unsqueeze_532 = None
    sub_155: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(add_344, mul_659);  mul_659 = None
    sub_156: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(sub_155, unsqueeze_529);  sub_155 = unsqueeze_529 = None
    mul_660: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_535);  sub_156 = unsqueeze_535 = None
    mul_661: "f32[112]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_127);  sum_47 = squeeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_660, relu_28, primals_173, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_660 = primals_173 = None
    getitem_196: "f32[8, 672, 14, 14]" = convolution_backward_22[0]
    getitem_197: "f32[112, 672, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_90: "f32[8, 672, 14, 14]" = torch.ops.aten.alias.default(relu_28);  relu_28 = None
    alias_91: "f32[8, 672, 14, 14]" = torch.ops.aten.alias.default(alias_90);  alias_90 = None
    le_15: "b8[8, 672, 14, 14]" = torch.ops.aten.le.Scalar(alias_91, 0);  alias_91 = None
    scalar_tensor_15: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_15: "f32[8, 672, 14, 14]" = torch.ops.aten.where.self(le_15, scalar_tensor_15, getitem_196);  le_15 = scalar_tensor_15 = getitem_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_536: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
    unsqueeze_537: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 2);  unsqueeze_536 = None
    unsqueeze_538: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_537, 3);  unsqueeze_537 = None
    sum_48: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_157: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_538)
    mul_662: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, sub_157);  sub_157 = None
    sum_49: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_662, [0, 2, 3]);  mul_662 = None
    mul_663: "f32[672]" = torch.ops.aten.mul.Tensor(sum_48, 0.0006377551020408163)
    unsqueeze_539: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_663, 0);  mul_663 = None
    unsqueeze_540: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 2);  unsqueeze_539 = None
    unsqueeze_541: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, 3);  unsqueeze_540 = None
    mul_664: "f32[672]" = torch.ops.aten.mul.Tensor(sum_49, 0.0006377551020408163)
    mul_665: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
    mul_666: "f32[672]" = torch.ops.aten.mul.Tensor(mul_664, mul_665);  mul_664 = mul_665 = None
    unsqueeze_542: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_666, 0);  mul_666 = None
    unsqueeze_543: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 2);  unsqueeze_542 = None
    unsqueeze_544: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_543, 3);  unsqueeze_543 = None
    mul_667: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_83);  primals_83 = None
    unsqueeze_545: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_667, 0);  mul_667 = None
    unsqueeze_546: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 2);  unsqueeze_545 = None
    unsqueeze_547: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, 3);  unsqueeze_546 = None
    sub_158: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_538);  convolution_41 = unsqueeze_538 = None
    mul_668: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_544);  sub_158 = unsqueeze_544 = None
    sub_159: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(where_15, mul_668);  where_15 = mul_668 = None
    sub_160: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(sub_159, unsqueeze_541);  sub_159 = unsqueeze_541 = None
    mul_669: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_547);  sub_160 = unsqueeze_547 = None
    mul_670: "f32[672]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_124);  sum_49 = squeeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_669, relu_27, primals_172, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False]);  mul_669 = primals_172 = None
    getitem_199: "f32[8, 672, 14, 14]" = convolution_backward_23[0]
    getitem_200: "f32[672, 1, 5, 5]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_93: "f32[8, 672, 14, 14]" = torch.ops.aten.alias.default(relu_27);  relu_27 = None
    alias_94: "f32[8, 672, 14, 14]" = torch.ops.aten.alias.default(alias_93);  alias_93 = None
    le_16: "b8[8, 672, 14, 14]" = torch.ops.aten.le.Scalar(alias_94, 0);  alias_94 = None
    scalar_tensor_16: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_16: "f32[8, 672, 14, 14]" = torch.ops.aten.where.self(le_16, scalar_tensor_16, getitem_199);  le_16 = scalar_tensor_16 = getitem_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_548: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
    unsqueeze_549: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 2);  unsqueeze_548 = None
    unsqueeze_550: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_549, 3);  unsqueeze_549 = None
    sum_50: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_161: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_550)
    mul_671: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, sub_161);  sub_161 = None
    sum_51: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_671, [0, 2, 3]);  mul_671 = None
    mul_672: "f32[672]" = torch.ops.aten.mul.Tensor(sum_50, 0.0006377551020408163)
    unsqueeze_551: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_672, 0);  mul_672 = None
    unsqueeze_552: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 2);  unsqueeze_551 = None
    unsqueeze_553: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, 3);  unsqueeze_552 = None
    mul_673: "f32[672]" = torch.ops.aten.mul.Tensor(sum_51, 0.0006377551020408163)
    mul_674: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_675: "f32[672]" = torch.ops.aten.mul.Tensor(mul_673, mul_674);  mul_673 = mul_674 = None
    unsqueeze_554: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_675, 0);  mul_675 = None
    unsqueeze_555: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 2);  unsqueeze_554 = None
    unsqueeze_556: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_555, 3);  unsqueeze_555 = None
    mul_676: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_81);  primals_81 = None
    unsqueeze_557: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_676, 0);  mul_676 = None
    unsqueeze_558: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 2);  unsqueeze_557 = None
    unsqueeze_559: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, 3);  unsqueeze_558 = None
    sub_162: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_550);  convolution_40 = unsqueeze_550 = None
    mul_677: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_556);  sub_162 = unsqueeze_556 = None
    sub_163: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(where_16, mul_677);  where_16 = mul_677 = None
    sub_164: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(sub_163, unsqueeze_553);  sub_163 = unsqueeze_553 = None
    mul_678: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_559);  sub_164 = unsqueeze_559 = None
    mul_679: "f32[672]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_121);  sum_51 = squeeze_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_678, add_208, primals_171, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_678 = add_208 = primals_171 = None
    getitem_202: "f32[8, 112, 14, 14]" = convolution_backward_24[0]
    getitem_203: "f32[672, 112, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_345: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(add_344, getitem_202);  add_344 = getitem_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_560: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
    unsqueeze_561: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 2);  unsqueeze_560 = None
    unsqueeze_562: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_561, 3);  unsqueeze_561 = None
    sum_52: "f32[112]" = torch.ops.aten.sum.dim_IntList(add_345, [0, 2, 3])
    sub_165: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_562)
    mul_680: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(add_345, sub_165);  sub_165 = None
    sum_53: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_680, [0, 2, 3]);  mul_680 = None
    mul_681: "f32[112]" = torch.ops.aten.mul.Tensor(sum_52, 0.0006377551020408163)
    unsqueeze_563: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_681, 0);  mul_681 = None
    unsqueeze_564: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 2);  unsqueeze_563 = None
    unsqueeze_565: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, 3);  unsqueeze_564 = None
    mul_682: "f32[112]" = torch.ops.aten.mul.Tensor(sum_53, 0.0006377551020408163)
    mul_683: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_684: "f32[112]" = torch.ops.aten.mul.Tensor(mul_682, mul_683);  mul_682 = mul_683 = None
    unsqueeze_566: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_684, 0);  mul_684 = None
    unsqueeze_567: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 2);  unsqueeze_566 = None
    unsqueeze_568: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_567, 3);  unsqueeze_567 = None
    mul_685: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_79);  primals_79 = None
    unsqueeze_569: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_685, 0);  mul_685 = None
    unsqueeze_570: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 2);  unsqueeze_569 = None
    unsqueeze_571: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, 3);  unsqueeze_570 = None
    sub_166: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_562);  convolution_39 = unsqueeze_562 = None
    mul_686: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_568);  sub_166 = unsqueeze_568 = None
    sub_167: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(add_345, mul_686);  add_345 = mul_686 = None
    sub_168: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(sub_167, unsqueeze_565);  sub_167 = unsqueeze_565 = None
    mul_687: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_571);  sub_168 = unsqueeze_571 = None
    mul_688: "f32[112]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_118);  sum_53 = squeeze_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_687, relu_26, primals_170, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_687 = primals_170 = None
    getitem_205: "f32[8, 384, 14, 14]" = convolution_backward_25[0]
    getitem_206: "f32[112, 384, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_96: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(relu_26);  relu_26 = None
    alias_97: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(alias_96);  alias_96 = None
    le_17: "b8[8, 384, 14, 14]" = torch.ops.aten.le.Scalar(alias_97, 0);  alias_97 = None
    scalar_tensor_17: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_17: "f32[8, 384, 14, 14]" = torch.ops.aten.where.self(le_17, scalar_tensor_17, getitem_205);  le_17 = scalar_tensor_17 = getitem_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_572: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
    unsqueeze_573: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 2);  unsqueeze_572 = None
    unsqueeze_574: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_573, 3);  unsqueeze_573 = None
    sum_54: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_169: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_574)
    mul_689: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_17, sub_169);  sub_169 = None
    sum_55: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_689, [0, 2, 3]);  mul_689 = None
    mul_690: "f32[384]" = torch.ops.aten.mul.Tensor(sum_54, 0.0006377551020408163)
    unsqueeze_575: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_690, 0);  mul_690 = None
    unsqueeze_576: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 2);  unsqueeze_575 = None
    unsqueeze_577: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, 3);  unsqueeze_576 = None
    mul_691: "f32[384]" = torch.ops.aten.mul.Tensor(sum_55, 0.0006377551020408163)
    mul_692: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_693: "f32[384]" = torch.ops.aten.mul.Tensor(mul_691, mul_692);  mul_691 = mul_692 = None
    unsqueeze_578: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_693, 0);  mul_693 = None
    unsqueeze_579: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 2);  unsqueeze_578 = None
    unsqueeze_580: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_579, 3);  unsqueeze_579 = None
    mul_694: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_77);  primals_77 = None
    unsqueeze_581: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_694, 0);  mul_694 = None
    unsqueeze_582: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 2);  unsqueeze_581 = None
    unsqueeze_583: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, 3);  unsqueeze_582 = None
    sub_170: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_574);  convolution_38 = unsqueeze_574 = None
    mul_695: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_170, unsqueeze_580);  sub_170 = unsqueeze_580 = None
    sub_171: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_17, mul_695);  where_17 = mul_695 = None
    sub_172: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_171, unsqueeze_577);  sub_171 = unsqueeze_577 = None
    mul_696: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_583);  sub_172 = unsqueeze_583 = None
    mul_697: "f32[384]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_115);  sum_55 = squeeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_696, relu_25, primals_169, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 384, [True, True, False]);  mul_696 = primals_169 = None
    getitem_208: "f32[8, 384, 14, 14]" = convolution_backward_26[0]
    getitem_209: "f32[384, 1, 5, 5]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_99: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(relu_25);  relu_25 = None
    alias_100: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(alias_99);  alias_99 = None
    le_18: "b8[8, 384, 14, 14]" = torch.ops.aten.le.Scalar(alias_100, 0);  alias_100 = None
    scalar_tensor_18: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_18: "f32[8, 384, 14, 14]" = torch.ops.aten.where.self(le_18, scalar_tensor_18, getitem_208);  le_18 = scalar_tensor_18 = getitem_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_584: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
    unsqueeze_585: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 2);  unsqueeze_584 = None
    unsqueeze_586: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_585, 3);  unsqueeze_585 = None
    sum_56: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_173: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_586)
    mul_698: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, sub_173);  sub_173 = None
    sum_57: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_698, [0, 2, 3]);  mul_698 = None
    mul_699: "f32[384]" = torch.ops.aten.mul.Tensor(sum_56, 0.0006377551020408163)
    unsqueeze_587: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_699, 0);  mul_699 = None
    unsqueeze_588: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 2);  unsqueeze_587 = None
    unsqueeze_589: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, 3);  unsqueeze_588 = None
    mul_700: "f32[384]" = torch.ops.aten.mul.Tensor(sum_57, 0.0006377551020408163)
    mul_701: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_702: "f32[384]" = torch.ops.aten.mul.Tensor(mul_700, mul_701);  mul_700 = mul_701 = None
    unsqueeze_590: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_702, 0);  mul_702 = None
    unsqueeze_591: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 2);  unsqueeze_590 = None
    unsqueeze_592: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_591, 3);  unsqueeze_591 = None
    mul_703: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_75);  primals_75 = None
    unsqueeze_593: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_703, 0);  mul_703 = None
    unsqueeze_594: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 2);  unsqueeze_593 = None
    unsqueeze_595: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, 3);  unsqueeze_594 = None
    sub_174: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_586);  convolution_37 = unsqueeze_586 = None
    mul_704: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_592);  sub_174 = unsqueeze_592 = None
    sub_175: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_18, mul_704);  where_18 = mul_704 = None
    sub_176: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_175, unsqueeze_589);  sub_175 = unsqueeze_589 = None
    mul_705: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_595);  sub_176 = unsqueeze_595 = None
    mul_706: "f32[384]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_112);  sum_57 = squeeze_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_705, add_193, primals_168, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_705 = add_193 = primals_168 = None
    getitem_211: "f32[8, 64, 14, 14]" = convolution_backward_27[0]
    getitem_212: "f32[384, 64, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_596: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
    unsqueeze_597: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 2);  unsqueeze_596 = None
    unsqueeze_598: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_597, 3);  unsqueeze_597 = None
    sum_58: "f32[64]" = torch.ops.aten.sum.dim_IntList(getitem_211, [0, 2, 3])
    sub_177: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_598)
    mul_707: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_211, sub_177);  sub_177 = None
    sum_59: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_707, [0, 2, 3]);  mul_707 = None
    mul_708: "f32[64]" = torch.ops.aten.mul.Tensor(sum_58, 0.0006377551020408163)
    unsqueeze_599: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_708, 0);  mul_708 = None
    unsqueeze_600: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 2);  unsqueeze_599 = None
    unsqueeze_601: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, 3);  unsqueeze_600 = None
    mul_709: "f32[64]" = torch.ops.aten.mul.Tensor(sum_59, 0.0006377551020408163)
    mul_710: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_711: "f32[64]" = torch.ops.aten.mul.Tensor(mul_709, mul_710);  mul_709 = mul_710 = None
    unsqueeze_602: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_711, 0);  mul_711 = None
    unsqueeze_603: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 2);  unsqueeze_602 = None
    unsqueeze_604: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_603, 3);  unsqueeze_603 = None
    mul_712: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_73);  primals_73 = None
    unsqueeze_605: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_712, 0);  mul_712 = None
    unsqueeze_606: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 2);  unsqueeze_605 = None
    unsqueeze_607: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, 3);  unsqueeze_606 = None
    sub_178: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_598);  convolution_36 = unsqueeze_598 = None
    mul_713: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(sub_178, unsqueeze_604);  sub_178 = unsqueeze_604 = None
    sub_179: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_211, mul_713);  mul_713 = None
    sub_180: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(sub_179, unsqueeze_601);  sub_179 = unsqueeze_601 = None
    mul_714: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_607);  sub_180 = unsqueeze_607 = None
    mul_715: "f32[64]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_109);  sum_59 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_714, relu_24, primals_167, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_714 = primals_167 = None
    getitem_214: "f32[8, 384, 14, 14]" = convolution_backward_28[0]
    getitem_215: "f32[64, 384, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_102: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(relu_24);  relu_24 = None
    alias_103: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(alias_102);  alias_102 = None
    le_19: "b8[8, 384, 14, 14]" = torch.ops.aten.le.Scalar(alias_103, 0);  alias_103 = None
    scalar_tensor_19: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_19: "f32[8, 384, 14, 14]" = torch.ops.aten.where.self(le_19, scalar_tensor_19, getitem_214);  le_19 = scalar_tensor_19 = getitem_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_608: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
    unsqueeze_609: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 2);  unsqueeze_608 = None
    unsqueeze_610: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_609, 3);  unsqueeze_609 = None
    sum_60: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_181: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_610)
    mul_716: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, sub_181);  sub_181 = None
    sum_61: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_716, [0, 2, 3]);  mul_716 = None
    mul_717: "f32[384]" = torch.ops.aten.mul.Tensor(sum_60, 0.0006377551020408163)
    unsqueeze_611: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_717, 0);  mul_717 = None
    unsqueeze_612: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 2);  unsqueeze_611 = None
    unsqueeze_613: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, 3);  unsqueeze_612 = None
    mul_718: "f32[384]" = torch.ops.aten.mul.Tensor(sum_61, 0.0006377551020408163)
    mul_719: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_720: "f32[384]" = torch.ops.aten.mul.Tensor(mul_718, mul_719);  mul_718 = mul_719 = None
    unsqueeze_614: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_720, 0);  mul_720 = None
    unsqueeze_615: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 2);  unsqueeze_614 = None
    unsqueeze_616: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_615, 3);  unsqueeze_615 = None
    mul_721: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_71);  primals_71 = None
    unsqueeze_617: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_721, 0);  mul_721 = None
    unsqueeze_618: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 2);  unsqueeze_617 = None
    unsqueeze_619: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, 3);  unsqueeze_618 = None
    sub_182: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_610);  convolution_35 = unsqueeze_610 = None
    mul_722: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_616);  sub_182 = unsqueeze_616 = None
    sub_183: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_19, mul_722);  where_19 = mul_722 = None
    sub_184: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_183, unsqueeze_613);  sub_183 = unsqueeze_613 = None
    mul_723: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_619);  sub_184 = unsqueeze_619 = None
    mul_724: "f32[384]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_106);  sum_61 = squeeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_723, relu_23, primals_166, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 384, [True, True, False]);  mul_723 = primals_166 = None
    getitem_217: "f32[8, 384, 14, 14]" = convolution_backward_29[0]
    getitem_218: "f32[384, 1, 5, 5]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_105: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(relu_23);  relu_23 = None
    alias_106: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(alias_105);  alias_105 = None
    le_20: "b8[8, 384, 14, 14]" = torch.ops.aten.le.Scalar(alias_106, 0);  alias_106 = None
    scalar_tensor_20: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_20: "f32[8, 384, 14, 14]" = torch.ops.aten.where.self(le_20, scalar_tensor_20, getitem_217);  le_20 = scalar_tensor_20 = getitem_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_620: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
    unsqueeze_621: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 2);  unsqueeze_620 = None
    unsqueeze_622: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_621, 3);  unsqueeze_621 = None
    sum_62: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_185: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_622)
    mul_725: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_20, sub_185);  sub_185 = None
    sum_63: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_725, [0, 2, 3]);  mul_725 = None
    mul_726: "f32[384]" = torch.ops.aten.mul.Tensor(sum_62, 0.0006377551020408163)
    unsqueeze_623: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_726, 0);  mul_726 = None
    unsqueeze_624: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 2);  unsqueeze_623 = None
    unsqueeze_625: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, 3);  unsqueeze_624 = None
    mul_727: "f32[384]" = torch.ops.aten.mul.Tensor(sum_63, 0.0006377551020408163)
    mul_728: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_729: "f32[384]" = torch.ops.aten.mul.Tensor(mul_727, mul_728);  mul_727 = mul_728 = None
    unsqueeze_626: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_729, 0);  mul_729 = None
    unsqueeze_627: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 2);  unsqueeze_626 = None
    unsqueeze_628: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_627, 3);  unsqueeze_627 = None
    mul_730: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_69);  primals_69 = None
    unsqueeze_629: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_730, 0);  mul_730 = None
    unsqueeze_630: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 2);  unsqueeze_629 = None
    unsqueeze_631: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, 3);  unsqueeze_630 = None
    sub_186: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_622);  convolution_34 = unsqueeze_622 = None
    mul_731: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_186, unsqueeze_628);  sub_186 = unsqueeze_628 = None
    sub_187: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_20, mul_731);  where_20 = mul_731 = None
    sub_188: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_187, unsqueeze_625);  sub_187 = unsqueeze_625 = None
    mul_732: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_631);  sub_188 = unsqueeze_631 = None
    mul_733: "f32[384]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_103);  sum_63 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_732, add_177, primals_165, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_732 = add_177 = primals_165 = None
    getitem_220: "f32[8, 64, 14, 14]" = convolution_backward_30[0]
    getitem_221: "f32[384, 64, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_346: "f32[8, 64, 14, 14]" = torch.ops.aten.add.Tensor(getitem_211, getitem_220);  getitem_211 = getitem_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_632: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
    unsqueeze_633: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 2);  unsqueeze_632 = None
    unsqueeze_634: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_633, 3);  unsqueeze_633 = None
    sum_64: "f32[64]" = torch.ops.aten.sum.dim_IntList(add_346, [0, 2, 3])
    sub_189: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_634)
    mul_734: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(add_346, sub_189);  sub_189 = None
    sum_65: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_734, [0, 2, 3]);  mul_734 = None
    mul_735: "f32[64]" = torch.ops.aten.mul.Tensor(sum_64, 0.0006377551020408163)
    unsqueeze_635: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_735, 0);  mul_735 = None
    unsqueeze_636: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 2);  unsqueeze_635 = None
    unsqueeze_637: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, 3);  unsqueeze_636 = None
    mul_736: "f32[64]" = torch.ops.aten.mul.Tensor(sum_65, 0.0006377551020408163)
    mul_737: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_738: "f32[64]" = torch.ops.aten.mul.Tensor(mul_736, mul_737);  mul_736 = mul_737 = None
    unsqueeze_638: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_738, 0);  mul_738 = None
    unsqueeze_639: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 2);  unsqueeze_638 = None
    unsqueeze_640: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_639, 3);  unsqueeze_639 = None
    mul_739: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_67);  primals_67 = None
    unsqueeze_641: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_739, 0);  mul_739 = None
    unsqueeze_642: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 2);  unsqueeze_641 = None
    unsqueeze_643: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, 3);  unsqueeze_642 = None
    sub_190: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_634);  convolution_33 = unsqueeze_634 = None
    mul_740: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_640);  sub_190 = unsqueeze_640 = None
    sub_191: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(add_346, mul_740);  mul_740 = None
    sub_192: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(sub_191, unsqueeze_637);  sub_191 = unsqueeze_637 = None
    mul_741: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_643);  sub_192 = unsqueeze_643 = None
    mul_742: "f32[64]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_100);  sum_65 = squeeze_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_741, relu_22, primals_164, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_741 = primals_164 = None
    getitem_223: "f32[8, 384, 14, 14]" = convolution_backward_31[0]
    getitem_224: "f32[64, 384, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_108: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(relu_22);  relu_22 = None
    alias_109: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(alias_108);  alias_108 = None
    le_21: "b8[8, 384, 14, 14]" = torch.ops.aten.le.Scalar(alias_109, 0);  alias_109 = None
    scalar_tensor_21: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_21: "f32[8, 384, 14, 14]" = torch.ops.aten.where.self(le_21, scalar_tensor_21, getitem_223);  le_21 = scalar_tensor_21 = getitem_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_644: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
    unsqueeze_645: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 2);  unsqueeze_644 = None
    unsqueeze_646: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_645, 3);  unsqueeze_645 = None
    sum_66: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_193: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_646)
    mul_743: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_21, sub_193);  sub_193 = None
    sum_67: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_743, [0, 2, 3]);  mul_743 = None
    mul_744: "f32[384]" = torch.ops.aten.mul.Tensor(sum_66, 0.0006377551020408163)
    unsqueeze_647: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_744, 0);  mul_744 = None
    unsqueeze_648: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 2);  unsqueeze_647 = None
    unsqueeze_649: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, 3);  unsqueeze_648 = None
    mul_745: "f32[384]" = torch.ops.aten.mul.Tensor(sum_67, 0.0006377551020408163)
    mul_746: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_747: "f32[384]" = torch.ops.aten.mul.Tensor(mul_745, mul_746);  mul_745 = mul_746 = None
    unsqueeze_650: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_747, 0);  mul_747 = None
    unsqueeze_651: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 2);  unsqueeze_650 = None
    unsqueeze_652: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_651, 3);  unsqueeze_651 = None
    mul_748: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_65);  primals_65 = None
    unsqueeze_653: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_748, 0);  mul_748 = None
    unsqueeze_654: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 2);  unsqueeze_653 = None
    unsqueeze_655: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, 3);  unsqueeze_654 = None
    sub_194: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_646);  convolution_32 = unsqueeze_646 = None
    mul_749: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_194, unsqueeze_652);  sub_194 = unsqueeze_652 = None
    sub_195: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_21, mul_749);  where_21 = mul_749 = None
    sub_196: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_195, unsqueeze_649);  sub_195 = unsqueeze_649 = None
    mul_750: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_655);  sub_196 = unsqueeze_655 = None
    mul_751: "f32[384]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_97);  sum_67 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_750, relu_21, primals_163, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 384, [True, True, False]);  mul_750 = primals_163 = None
    getitem_226: "f32[8, 384, 14, 14]" = convolution_backward_32[0]
    getitem_227: "f32[384, 1, 5, 5]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_111: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(relu_21);  relu_21 = None
    alias_112: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(alias_111);  alias_111 = None
    le_22: "b8[8, 384, 14, 14]" = torch.ops.aten.le.Scalar(alias_112, 0);  alias_112 = None
    scalar_tensor_22: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_22: "f32[8, 384, 14, 14]" = torch.ops.aten.where.self(le_22, scalar_tensor_22, getitem_226);  le_22 = scalar_tensor_22 = getitem_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_656: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
    unsqueeze_657: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 2);  unsqueeze_656 = None
    unsqueeze_658: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_657, 3);  unsqueeze_657 = None
    sum_68: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_197: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_658)
    mul_752: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_22, sub_197);  sub_197 = None
    sum_69: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_752, [0, 2, 3]);  mul_752 = None
    mul_753: "f32[384]" = torch.ops.aten.mul.Tensor(sum_68, 0.0006377551020408163)
    unsqueeze_659: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_753, 0);  mul_753 = None
    unsqueeze_660: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 2);  unsqueeze_659 = None
    unsqueeze_661: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, 3);  unsqueeze_660 = None
    mul_754: "f32[384]" = torch.ops.aten.mul.Tensor(sum_69, 0.0006377551020408163)
    mul_755: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_756: "f32[384]" = torch.ops.aten.mul.Tensor(mul_754, mul_755);  mul_754 = mul_755 = None
    unsqueeze_662: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_756, 0);  mul_756 = None
    unsqueeze_663: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 2);  unsqueeze_662 = None
    unsqueeze_664: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_663, 3);  unsqueeze_663 = None
    mul_757: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_63);  primals_63 = None
    unsqueeze_665: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_757, 0);  mul_757 = None
    unsqueeze_666: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 2);  unsqueeze_665 = None
    unsqueeze_667: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, 3);  unsqueeze_666 = None
    sub_198: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_658);  convolution_31 = unsqueeze_658 = None
    mul_758: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_664);  sub_198 = unsqueeze_664 = None
    sub_199: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_22, mul_758);  where_22 = mul_758 = None
    sub_200: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_199, unsqueeze_661);  sub_199 = unsqueeze_661 = None
    mul_759: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_667);  sub_200 = unsqueeze_667 = None
    mul_760: "f32[384]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_94);  sum_69 = squeeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_759, add_161, primals_162, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_759 = add_161 = primals_162 = None
    getitem_229: "f32[8, 64, 14, 14]" = convolution_backward_33[0]
    getitem_230: "f32[384, 64, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_347: "f32[8, 64, 14, 14]" = torch.ops.aten.add.Tensor(add_346, getitem_229);  add_346 = getitem_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_668: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_669: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 2);  unsqueeze_668 = None
    unsqueeze_670: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_669, 3);  unsqueeze_669 = None
    sum_70: "f32[64]" = torch.ops.aten.sum.dim_IntList(add_347, [0, 2, 3])
    sub_201: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_670)
    mul_761: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(add_347, sub_201);  sub_201 = None
    sum_71: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_761, [0, 2, 3]);  mul_761 = None
    mul_762: "f32[64]" = torch.ops.aten.mul.Tensor(sum_70, 0.0006377551020408163)
    unsqueeze_671: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_762, 0);  mul_762 = None
    unsqueeze_672: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 2);  unsqueeze_671 = None
    unsqueeze_673: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, 3);  unsqueeze_672 = None
    mul_763: "f32[64]" = torch.ops.aten.mul.Tensor(sum_71, 0.0006377551020408163)
    mul_764: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_765: "f32[64]" = torch.ops.aten.mul.Tensor(mul_763, mul_764);  mul_763 = mul_764 = None
    unsqueeze_674: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_765, 0);  mul_765 = None
    unsqueeze_675: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 2);  unsqueeze_674 = None
    unsqueeze_676: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_675, 3);  unsqueeze_675 = None
    mul_766: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_61);  primals_61 = None
    unsqueeze_677: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_766, 0);  mul_766 = None
    unsqueeze_678: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 2);  unsqueeze_677 = None
    unsqueeze_679: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, 3);  unsqueeze_678 = None
    sub_202: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_670);  convolution_30 = unsqueeze_670 = None
    mul_767: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_676);  sub_202 = unsqueeze_676 = None
    sub_203: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(add_347, mul_767);  mul_767 = None
    sub_204: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(sub_203, unsqueeze_673);  sub_203 = unsqueeze_673 = None
    mul_768: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_679);  sub_204 = unsqueeze_679 = None
    mul_769: "f32[64]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_91);  sum_71 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_768, relu_20, primals_161, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_768 = primals_161 = None
    getitem_232: "f32[8, 192, 14, 14]" = convolution_backward_34[0]
    getitem_233: "f32[64, 192, 1, 1]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_114: "f32[8, 192, 14, 14]" = torch.ops.aten.alias.default(relu_20);  relu_20 = None
    alias_115: "f32[8, 192, 14, 14]" = torch.ops.aten.alias.default(alias_114);  alias_114 = None
    le_23: "b8[8, 192, 14, 14]" = torch.ops.aten.le.Scalar(alias_115, 0);  alias_115 = None
    scalar_tensor_23: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_23: "f32[8, 192, 14, 14]" = torch.ops.aten.where.self(le_23, scalar_tensor_23, getitem_232);  le_23 = scalar_tensor_23 = getitem_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_680: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    unsqueeze_681: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 2);  unsqueeze_680 = None
    unsqueeze_682: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_681, 3);  unsqueeze_681 = None
    sum_72: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_205: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_682)
    mul_770: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_23, sub_205);  sub_205 = None
    sum_73: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_770, [0, 2, 3]);  mul_770 = None
    mul_771: "f32[192]" = torch.ops.aten.mul.Tensor(sum_72, 0.0006377551020408163)
    unsqueeze_683: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_771, 0);  mul_771 = None
    unsqueeze_684: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 2);  unsqueeze_683 = None
    unsqueeze_685: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, 3);  unsqueeze_684 = None
    mul_772: "f32[192]" = torch.ops.aten.mul.Tensor(sum_73, 0.0006377551020408163)
    mul_773: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_774: "f32[192]" = torch.ops.aten.mul.Tensor(mul_772, mul_773);  mul_772 = mul_773 = None
    unsqueeze_686: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_774, 0);  mul_774 = None
    unsqueeze_687: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 2);  unsqueeze_686 = None
    unsqueeze_688: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_687, 3);  unsqueeze_687 = None
    mul_775: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_59);  primals_59 = None
    unsqueeze_689: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_775, 0);  mul_775 = None
    unsqueeze_690: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 2);  unsqueeze_689 = None
    unsqueeze_691: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, 3);  unsqueeze_690 = None
    sub_206: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_682);  convolution_29 = unsqueeze_682 = None
    mul_776: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_688);  sub_206 = unsqueeze_688 = None
    sub_207: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(where_23, mul_776);  where_23 = mul_776 = None
    sub_208: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(sub_207, unsqueeze_685);  sub_207 = unsqueeze_685 = None
    mul_777: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_691);  sub_208 = unsqueeze_691 = None
    mul_778: "f32[192]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_88);  sum_73 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_777, relu_19, primals_160, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 192, [True, True, False]);  mul_777 = primals_160 = None
    getitem_235: "f32[8, 192, 14, 14]" = convolution_backward_35[0]
    getitem_236: "f32[192, 1, 5, 5]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_117: "f32[8, 192, 14, 14]" = torch.ops.aten.alias.default(relu_19);  relu_19 = None
    alias_118: "f32[8, 192, 14, 14]" = torch.ops.aten.alias.default(alias_117);  alias_117 = None
    le_24: "b8[8, 192, 14, 14]" = torch.ops.aten.le.Scalar(alias_118, 0);  alias_118 = None
    scalar_tensor_24: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_24: "f32[8, 192, 14, 14]" = torch.ops.aten.where.self(le_24, scalar_tensor_24, getitem_235);  le_24 = scalar_tensor_24 = getitem_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_692: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_693: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 2);  unsqueeze_692 = None
    unsqueeze_694: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_693, 3);  unsqueeze_693 = None
    sum_74: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_209: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_694)
    mul_779: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_24, sub_209);  sub_209 = None
    sum_75: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_779, [0, 2, 3]);  mul_779 = None
    mul_780: "f32[192]" = torch.ops.aten.mul.Tensor(sum_74, 0.0006377551020408163)
    unsqueeze_695: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_780, 0);  mul_780 = None
    unsqueeze_696: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 2);  unsqueeze_695 = None
    unsqueeze_697: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, 3);  unsqueeze_696 = None
    mul_781: "f32[192]" = torch.ops.aten.mul.Tensor(sum_75, 0.0006377551020408163)
    mul_782: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_783: "f32[192]" = torch.ops.aten.mul.Tensor(mul_781, mul_782);  mul_781 = mul_782 = None
    unsqueeze_698: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_783, 0);  mul_783 = None
    unsqueeze_699: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 2);  unsqueeze_698 = None
    unsqueeze_700: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_699, 3);  unsqueeze_699 = None
    mul_784: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_57);  primals_57 = None
    unsqueeze_701: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_784, 0);  mul_784 = None
    unsqueeze_702: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 2);  unsqueeze_701 = None
    unsqueeze_703: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, 3);  unsqueeze_702 = None
    sub_210: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_694);  convolution_28 = unsqueeze_694 = None
    mul_785: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_210, unsqueeze_700);  sub_210 = unsqueeze_700 = None
    sub_211: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(where_24, mul_785);  where_24 = mul_785 = None
    sub_212: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(sub_211, unsqueeze_697);  sub_211 = unsqueeze_697 = None
    mul_786: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_703);  sub_212 = unsqueeze_703 = None
    mul_787: "f32[192]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_85);  sum_75 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_786, add_145, primals_159, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_786 = add_145 = primals_159 = None
    getitem_238: "f32[8, 64, 14, 14]" = convolution_backward_36[0]
    getitem_239: "f32[192, 64, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_348: "f32[8, 64, 14, 14]" = torch.ops.aten.add.Tensor(add_347, getitem_238);  add_347 = getitem_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_704: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_705: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 2);  unsqueeze_704 = None
    unsqueeze_706: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_705, 3);  unsqueeze_705 = None
    sum_76: "f32[64]" = torch.ops.aten.sum.dim_IntList(add_348, [0, 2, 3])
    sub_213: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_706)
    mul_788: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(add_348, sub_213);  sub_213 = None
    sum_77: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_788, [0, 2, 3]);  mul_788 = None
    mul_789: "f32[64]" = torch.ops.aten.mul.Tensor(sum_76, 0.0006377551020408163)
    unsqueeze_707: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_789, 0);  mul_789 = None
    unsqueeze_708: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 2);  unsqueeze_707 = None
    unsqueeze_709: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_708, 3);  unsqueeze_708 = None
    mul_790: "f32[64]" = torch.ops.aten.mul.Tensor(sum_77, 0.0006377551020408163)
    mul_791: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_792: "f32[64]" = torch.ops.aten.mul.Tensor(mul_790, mul_791);  mul_790 = mul_791 = None
    unsqueeze_710: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_792, 0);  mul_792 = None
    unsqueeze_711: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 2);  unsqueeze_710 = None
    unsqueeze_712: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_711, 3);  unsqueeze_711 = None
    mul_793: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_55);  primals_55 = None
    unsqueeze_713: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_793, 0);  mul_793 = None
    unsqueeze_714: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 2);  unsqueeze_713 = None
    unsqueeze_715: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, 3);  unsqueeze_714 = None
    sub_214: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_706);  convolution_27 = unsqueeze_706 = None
    mul_794: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_712);  sub_214 = unsqueeze_712 = None
    sub_215: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(add_348, mul_794);  add_348 = mul_794 = None
    sub_216: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(sub_215, unsqueeze_709);  sub_215 = unsqueeze_709 = None
    mul_795: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_715);  sub_216 = unsqueeze_715 = None
    mul_796: "f32[64]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_82);  sum_77 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_795, relu_18, primals_158, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_795 = primals_158 = None
    getitem_241: "f32[8, 192, 14, 14]" = convolution_backward_37[0]
    getitem_242: "f32[64, 192, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_120: "f32[8, 192, 14, 14]" = torch.ops.aten.alias.default(relu_18);  relu_18 = None
    alias_121: "f32[8, 192, 14, 14]" = torch.ops.aten.alias.default(alias_120);  alias_120 = None
    le_25: "b8[8, 192, 14, 14]" = torch.ops.aten.le.Scalar(alias_121, 0);  alias_121 = None
    scalar_tensor_25: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_25: "f32[8, 192, 14, 14]" = torch.ops.aten.where.self(le_25, scalar_tensor_25, getitem_241);  le_25 = scalar_tensor_25 = getitem_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_716: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_717: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 2);  unsqueeze_716 = None
    unsqueeze_718: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_717, 3);  unsqueeze_717 = None
    sum_78: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_217: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_718)
    mul_797: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_25, sub_217);  sub_217 = None
    sum_79: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_797, [0, 2, 3]);  mul_797 = None
    mul_798: "f32[192]" = torch.ops.aten.mul.Tensor(sum_78, 0.0006377551020408163)
    unsqueeze_719: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_798, 0);  mul_798 = None
    unsqueeze_720: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 2);  unsqueeze_719 = None
    unsqueeze_721: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_720, 3);  unsqueeze_720 = None
    mul_799: "f32[192]" = torch.ops.aten.mul.Tensor(sum_79, 0.0006377551020408163)
    mul_800: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_801: "f32[192]" = torch.ops.aten.mul.Tensor(mul_799, mul_800);  mul_799 = mul_800 = None
    unsqueeze_722: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_801, 0);  mul_801 = None
    unsqueeze_723: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 2);  unsqueeze_722 = None
    unsqueeze_724: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_723, 3);  unsqueeze_723 = None
    mul_802: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_53);  primals_53 = None
    unsqueeze_725: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_802, 0);  mul_802 = None
    unsqueeze_726: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 2);  unsqueeze_725 = None
    unsqueeze_727: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, 3);  unsqueeze_726 = None
    sub_218: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_718);  convolution_26 = unsqueeze_718 = None
    mul_803: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_218, unsqueeze_724);  sub_218 = unsqueeze_724 = None
    sub_219: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(where_25, mul_803);  where_25 = mul_803 = None
    sub_220: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(sub_219, unsqueeze_721);  sub_219 = unsqueeze_721 = None
    mul_804: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_727);  sub_220 = unsqueeze_727 = None
    mul_805: "f32[192]" = torch.ops.aten.mul.Tensor(sum_79, squeeze_79);  sum_79 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_804, relu_17, primals_157, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 192, [True, True, False]);  mul_804 = primals_157 = None
    getitem_244: "f32[8, 192, 28, 28]" = convolution_backward_38[0]
    getitem_245: "f32[192, 1, 5, 5]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_123: "f32[8, 192, 28, 28]" = torch.ops.aten.alias.default(relu_17);  relu_17 = None
    alias_124: "f32[8, 192, 28, 28]" = torch.ops.aten.alias.default(alias_123);  alias_123 = None
    le_26: "b8[8, 192, 28, 28]" = torch.ops.aten.le.Scalar(alias_124, 0);  alias_124 = None
    scalar_tensor_26: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_26: "f32[8, 192, 28, 28]" = torch.ops.aten.where.self(le_26, scalar_tensor_26, getitem_244);  le_26 = scalar_tensor_26 = getitem_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_728: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_729: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 2);  unsqueeze_728 = None
    unsqueeze_730: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_729, 3);  unsqueeze_729 = None
    sum_80: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_221: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_730)
    mul_806: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(where_26, sub_221);  sub_221 = None
    sum_81: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_806, [0, 2, 3]);  mul_806 = None
    mul_807: "f32[192]" = torch.ops.aten.mul.Tensor(sum_80, 0.00015943877551020407)
    unsqueeze_731: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_807, 0);  mul_807 = None
    unsqueeze_732: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 2);  unsqueeze_731 = None
    unsqueeze_733: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_732, 3);  unsqueeze_732 = None
    mul_808: "f32[192]" = torch.ops.aten.mul.Tensor(sum_81, 0.00015943877551020407)
    mul_809: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_810: "f32[192]" = torch.ops.aten.mul.Tensor(mul_808, mul_809);  mul_808 = mul_809 = None
    unsqueeze_734: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_810, 0);  mul_810 = None
    unsqueeze_735: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 2);  unsqueeze_734 = None
    unsqueeze_736: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_735, 3);  unsqueeze_735 = None
    mul_811: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_51);  primals_51 = None
    unsqueeze_737: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_811, 0);  mul_811 = None
    unsqueeze_738: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 2);  unsqueeze_737 = None
    unsqueeze_739: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, 3);  unsqueeze_738 = None
    sub_222: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_730);  convolution_25 = unsqueeze_730 = None
    mul_812: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_736);  sub_222 = unsqueeze_736 = None
    sub_223: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(where_26, mul_812);  where_26 = mul_812 = None
    sub_224: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_223, unsqueeze_733);  sub_223 = unsqueeze_733 = None
    mul_813: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_739);  sub_224 = unsqueeze_739 = None
    mul_814: "f32[192]" = torch.ops.aten.mul.Tensor(sum_81, squeeze_76);  sum_81 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_813, add_130, primals_156, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_813 = add_130 = primals_156 = None
    getitem_247: "f32[8, 32, 28, 28]" = convolution_backward_39[0]
    getitem_248: "f32[192, 32, 1, 1]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_740: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_741: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 2);  unsqueeze_740 = None
    unsqueeze_742: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_741, 3);  unsqueeze_741 = None
    sum_82: "f32[32]" = torch.ops.aten.sum.dim_IntList(getitem_247, [0, 2, 3])
    sub_225: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_742)
    mul_815: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_247, sub_225);  sub_225 = None
    sum_83: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_815, [0, 2, 3]);  mul_815 = None
    mul_816: "f32[32]" = torch.ops.aten.mul.Tensor(sum_82, 0.00015943877551020407)
    unsqueeze_743: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_816, 0);  mul_816 = None
    unsqueeze_744: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 2);  unsqueeze_743 = None
    unsqueeze_745: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_744, 3);  unsqueeze_744 = None
    mul_817: "f32[32]" = torch.ops.aten.mul.Tensor(sum_83, 0.00015943877551020407)
    mul_818: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_819: "f32[32]" = torch.ops.aten.mul.Tensor(mul_817, mul_818);  mul_817 = mul_818 = None
    unsqueeze_746: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_819, 0);  mul_819 = None
    unsqueeze_747: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 2);  unsqueeze_746 = None
    unsqueeze_748: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_747, 3);  unsqueeze_747 = None
    mul_820: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_49);  primals_49 = None
    unsqueeze_749: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_820, 0);  mul_820 = None
    unsqueeze_750: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 2);  unsqueeze_749 = None
    unsqueeze_751: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, 3);  unsqueeze_750 = None
    sub_226: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_742);  convolution_24 = unsqueeze_742 = None
    mul_821: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(sub_226, unsqueeze_748);  sub_226 = unsqueeze_748 = None
    sub_227: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_247, mul_821);  mul_821 = None
    sub_228: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(sub_227, unsqueeze_745);  sub_227 = unsqueeze_745 = None
    mul_822: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_751);  sub_228 = unsqueeze_751 = None
    mul_823: "f32[32]" = torch.ops.aten.mul.Tensor(sum_83, squeeze_73);  sum_83 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_822, relu_16, primals_155, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_822 = primals_155 = None
    getitem_250: "f32[8, 192, 28, 28]" = convolution_backward_40[0]
    getitem_251: "f32[32, 192, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_126: "f32[8, 192, 28, 28]" = torch.ops.aten.alias.default(relu_16);  relu_16 = None
    alias_127: "f32[8, 192, 28, 28]" = torch.ops.aten.alias.default(alias_126);  alias_126 = None
    le_27: "b8[8, 192, 28, 28]" = torch.ops.aten.le.Scalar(alias_127, 0);  alias_127 = None
    scalar_tensor_27: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_27: "f32[8, 192, 28, 28]" = torch.ops.aten.where.self(le_27, scalar_tensor_27, getitem_250);  le_27 = scalar_tensor_27 = getitem_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_752: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_753: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 2);  unsqueeze_752 = None
    unsqueeze_754: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_753, 3);  unsqueeze_753 = None
    sum_84: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_229: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_754)
    mul_824: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(where_27, sub_229);  sub_229 = None
    sum_85: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_824, [0, 2, 3]);  mul_824 = None
    mul_825: "f32[192]" = torch.ops.aten.mul.Tensor(sum_84, 0.00015943877551020407)
    unsqueeze_755: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_825, 0);  mul_825 = None
    unsqueeze_756: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 2);  unsqueeze_755 = None
    unsqueeze_757: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_756, 3);  unsqueeze_756 = None
    mul_826: "f32[192]" = torch.ops.aten.mul.Tensor(sum_85, 0.00015943877551020407)
    mul_827: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_828: "f32[192]" = torch.ops.aten.mul.Tensor(mul_826, mul_827);  mul_826 = mul_827 = None
    unsqueeze_758: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_828, 0);  mul_828 = None
    unsqueeze_759: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 2);  unsqueeze_758 = None
    unsqueeze_760: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_759, 3);  unsqueeze_759 = None
    mul_829: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_47);  primals_47 = None
    unsqueeze_761: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_829, 0);  mul_829 = None
    unsqueeze_762: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 2);  unsqueeze_761 = None
    unsqueeze_763: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, 3);  unsqueeze_762 = None
    sub_230: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_754);  convolution_23 = unsqueeze_754 = None
    mul_830: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_760);  sub_230 = unsqueeze_760 = None
    sub_231: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(where_27, mul_830);  where_27 = mul_830 = None
    sub_232: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_231, unsqueeze_757);  sub_231 = unsqueeze_757 = None
    mul_831: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_763);  sub_232 = unsqueeze_763 = None
    mul_832: "f32[192]" = torch.ops.aten.mul.Tensor(sum_85, squeeze_70);  sum_85 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_831, relu_15, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False]);  mul_831 = primals_154 = None
    getitem_253: "f32[8, 192, 28, 28]" = convolution_backward_41[0]
    getitem_254: "f32[192, 1, 3, 3]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_129: "f32[8, 192, 28, 28]" = torch.ops.aten.alias.default(relu_15);  relu_15 = None
    alias_130: "f32[8, 192, 28, 28]" = torch.ops.aten.alias.default(alias_129);  alias_129 = None
    le_28: "b8[8, 192, 28, 28]" = torch.ops.aten.le.Scalar(alias_130, 0);  alias_130 = None
    scalar_tensor_28: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_28: "f32[8, 192, 28, 28]" = torch.ops.aten.where.self(le_28, scalar_tensor_28, getitem_253);  le_28 = scalar_tensor_28 = getitem_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_764: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_765: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 2);  unsqueeze_764 = None
    unsqueeze_766: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_765, 3);  unsqueeze_765 = None
    sum_86: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    sub_233: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_766)
    mul_833: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(where_28, sub_233);  sub_233 = None
    sum_87: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_833, [0, 2, 3]);  mul_833 = None
    mul_834: "f32[192]" = torch.ops.aten.mul.Tensor(sum_86, 0.00015943877551020407)
    unsqueeze_767: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_834, 0);  mul_834 = None
    unsqueeze_768: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 2);  unsqueeze_767 = None
    unsqueeze_769: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_768, 3);  unsqueeze_768 = None
    mul_835: "f32[192]" = torch.ops.aten.mul.Tensor(sum_87, 0.00015943877551020407)
    mul_836: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_837: "f32[192]" = torch.ops.aten.mul.Tensor(mul_835, mul_836);  mul_835 = mul_836 = None
    unsqueeze_770: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_837, 0);  mul_837 = None
    unsqueeze_771: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 2);  unsqueeze_770 = None
    unsqueeze_772: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_771, 3);  unsqueeze_771 = None
    mul_838: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_45);  primals_45 = None
    unsqueeze_773: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_838, 0);  mul_838 = None
    unsqueeze_774: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 2);  unsqueeze_773 = None
    unsqueeze_775: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, 3);  unsqueeze_774 = None
    sub_234: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_766);  convolution_22 = unsqueeze_766 = None
    mul_839: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_234, unsqueeze_772);  sub_234 = unsqueeze_772 = None
    sub_235: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(where_28, mul_839);  where_28 = mul_839 = None
    sub_236: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_235, unsqueeze_769);  sub_235 = unsqueeze_769 = None
    mul_840: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_775);  sub_236 = unsqueeze_775 = None
    mul_841: "f32[192]" = torch.ops.aten.mul.Tensor(sum_87, squeeze_67);  sum_87 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_840, add_114, primals_153, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_840 = add_114 = primals_153 = None
    getitem_256: "f32[8, 32, 28, 28]" = convolution_backward_42[0]
    getitem_257: "f32[192, 32, 1, 1]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_349: "f32[8, 32, 28, 28]" = torch.ops.aten.add.Tensor(getitem_247, getitem_256);  getitem_247 = getitem_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_776: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_777: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 2);  unsqueeze_776 = None
    unsqueeze_778: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_777, 3);  unsqueeze_777 = None
    sum_88: "f32[32]" = torch.ops.aten.sum.dim_IntList(add_349, [0, 2, 3])
    sub_237: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_778)
    mul_842: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(add_349, sub_237);  sub_237 = None
    sum_89: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_842, [0, 2, 3]);  mul_842 = None
    mul_843: "f32[32]" = torch.ops.aten.mul.Tensor(sum_88, 0.00015943877551020407)
    unsqueeze_779: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_843, 0);  mul_843 = None
    unsqueeze_780: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 2);  unsqueeze_779 = None
    unsqueeze_781: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_780, 3);  unsqueeze_780 = None
    mul_844: "f32[32]" = torch.ops.aten.mul.Tensor(sum_89, 0.00015943877551020407)
    mul_845: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_846: "f32[32]" = torch.ops.aten.mul.Tensor(mul_844, mul_845);  mul_844 = mul_845 = None
    unsqueeze_782: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_846, 0);  mul_846 = None
    unsqueeze_783: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 2);  unsqueeze_782 = None
    unsqueeze_784: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_783, 3);  unsqueeze_783 = None
    mul_847: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_43);  primals_43 = None
    unsqueeze_785: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_847, 0);  mul_847 = None
    unsqueeze_786: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 2);  unsqueeze_785 = None
    unsqueeze_787: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, 3);  unsqueeze_786 = None
    sub_238: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_778);  convolution_21 = unsqueeze_778 = None
    mul_848: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(sub_238, unsqueeze_784);  sub_238 = unsqueeze_784 = None
    sub_239: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(add_349, mul_848);  mul_848 = None
    sub_240: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(sub_239, unsqueeze_781);  sub_239 = unsqueeze_781 = None
    mul_849: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_787);  sub_240 = unsqueeze_787 = None
    mul_850: "f32[32]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_64);  sum_89 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_849, relu_14, primals_152, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_849 = primals_152 = None
    getitem_259: "f32[8, 192, 28, 28]" = convolution_backward_43[0]
    getitem_260: "f32[32, 192, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_132: "f32[8, 192, 28, 28]" = torch.ops.aten.alias.default(relu_14);  relu_14 = None
    alias_133: "f32[8, 192, 28, 28]" = torch.ops.aten.alias.default(alias_132);  alias_132 = None
    le_29: "b8[8, 192, 28, 28]" = torch.ops.aten.le.Scalar(alias_133, 0);  alias_133 = None
    scalar_tensor_29: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_29: "f32[8, 192, 28, 28]" = torch.ops.aten.where.self(le_29, scalar_tensor_29, getitem_259);  le_29 = scalar_tensor_29 = getitem_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_788: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_789: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 2);  unsqueeze_788 = None
    unsqueeze_790: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_789, 3);  unsqueeze_789 = None
    sum_90: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    sub_241: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_790)
    mul_851: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(where_29, sub_241);  sub_241 = None
    sum_91: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_851, [0, 2, 3]);  mul_851 = None
    mul_852: "f32[192]" = torch.ops.aten.mul.Tensor(sum_90, 0.00015943877551020407)
    unsqueeze_791: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_852, 0);  mul_852 = None
    unsqueeze_792: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_791, 2);  unsqueeze_791 = None
    unsqueeze_793: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_792, 3);  unsqueeze_792 = None
    mul_853: "f32[192]" = torch.ops.aten.mul.Tensor(sum_91, 0.00015943877551020407)
    mul_854: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_855: "f32[192]" = torch.ops.aten.mul.Tensor(mul_853, mul_854);  mul_853 = mul_854 = None
    unsqueeze_794: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_855, 0);  mul_855 = None
    unsqueeze_795: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 2);  unsqueeze_794 = None
    unsqueeze_796: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_795, 3);  unsqueeze_795 = None
    mul_856: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_41);  primals_41 = None
    unsqueeze_797: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_856, 0);  mul_856 = None
    unsqueeze_798: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 2);  unsqueeze_797 = None
    unsqueeze_799: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, 3);  unsqueeze_798 = None
    sub_242: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_790);  convolution_20 = unsqueeze_790 = None
    mul_857: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_242, unsqueeze_796);  sub_242 = unsqueeze_796 = None
    sub_243: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(where_29, mul_857);  where_29 = mul_857 = None
    sub_244: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_243, unsqueeze_793);  sub_243 = unsqueeze_793 = None
    mul_858: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_799);  sub_244 = unsqueeze_799 = None
    mul_859: "f32[192]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_61);  sum_91 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_858, relu_13, primals_151, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 192, [True, True, False]);  mul_858 = primals_151 = None
    getitem_262: "f32[8, 192, 28, 28]" = convolution_backward_44[0]
    getitem_263: "f32[192, 1, 5, 5]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_135: "f32[8, 192, 28, 28]" = torch.ops.aten.alias.default(relu_13);  relu_13 = None
    alias_136: "f32[8, 192, 28, 28]" = torch.ops.aten.alias.default(alias_135);  alias_135 = None
    le_30: "b8[8, 192, 28, 28]" = torch.ops.aten.le.Scalar(alias_136, 0);  alias_136 = None
    scalar_tensor_30: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_30: "f32[8, 192, 28, 28]" = torch.ops.aten.where.self(le_30, scalar_tensor_30, getitem_262);  le_30 = scalar_tensor_30 = getitem_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_800: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_801: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 2);  unsqueeze_800 = None
    unsqueeze_802: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_801, 3);  unsqueeze_801 = None
    sum_92: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
    sub_245: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_802)
    mul_860: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(where_30, sub_245);  sub_245 = None
    sum_93: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_860, [0, 2, 3]);  mul_860 = None
    mul_861: "f32[192]" = torch.ops.aten.mul.Tensor(sum_92, 0.00015943877551020407)
    unsqueeze_803: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_861, 0);  mul_861 = None
    unsqueeze_804: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_803, 2);  unsqueeze_803 = None
    unsqueeze_805: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_804, 3);  unsqueeze_804 = None
    mul_862: "f32[192]" = torch.ops.aten.mul.Tensor(sum_93, 0.00015943877551020407)
    mul_863: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_864: "f32[192]" = torch.ops.aten.mul.Tensor(mul_862, mul_863);  mul_862 = mul_863 = None
    unsqueeze_806: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_864, 0);  mul_864 = None
    unsqueeze_807: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 2);  unsqueeze_806 = None
    unsqueeze_808: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_807, 3);  unsqueeze_807 = None
    mul_865: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_39);  primals_39 = None
    unsqueeze_809: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_865, 0);  mul_865 = None
    unsqueeze_810: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 2);  unsqueeze_809 = None
    unsqueeze_811: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, 3);  unsqueeze_810 = None
    sub_246: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_802);  convolution_19 = unsqueeze_802 = None
    mul_866: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_246, unsqueeze_808);  sub_246 = unsqueeze_808 = None
    sub_247: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(where_30, mul_866);  where_30 = mul_866 = None
    sub_248: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_247, unsqueeze_805);  sub_247 = unsqueeze_805 = None
    mul_867: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_811);  sub_248 = unsqueeze_811 = None
    mul_868: "f32[192]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_58);  sum_93 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_867, add_98, primals_150, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_867 = add_98 = primals_150 = None
    getitem_265: "f32[8, 32, 28, 28]" = convolution_backward_45[0]
    getitem_266: "f32[192, 32, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_350: "f32[8, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_349, getitem_265);  add_349 = getitem_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_812: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_813: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 2);  unsqueeze_812 = None
    unsqueeze_814: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_813, 3);  unsqueeze_813 = None
    sum_94: "f32[32]" = torch.ops.aten.sum.dim_IntList(add_350, [0, 2, 3])
    sub_249: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_814)
    mul_869: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(add_350, sub_249);  sub_249 = None
    sum_95: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_869, [0, 2, 3]);  mul_869 = None
    mul_870: "f32[32]" = torch.ops.aten.mul.Tensor(sum_94, 0.00015943877551020407)
    unsqueeze_815: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_870, 0);  mul_870 = None
    unsqueeze_816: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_815, 2);  unsqueeze_815 = None
    unsqueeze_817: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_816, 3);  unsqueeze_816 = None
    mul_871: "f32[32]" = torch.ops.aten.mul.Tensor(sum_95, 0.00015943877551020407)
    mul_872: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_873: "f32[32]" = torch.ops.aten.mul.Tensor(mul_871, mul_872);  mul_871 = mul_872 = None
    unsqueeze_818: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_873, 0);  mul_873 = None
    unsqueeze_819: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 2);  unsqueeze_818 = None
    unsqueeze_820: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_819, 3);  unsqueeze_819 = None
    mul_874: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_37);  primals_37 = None
    unsqueeze_821: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_874, 0);  mul_874 = None
    unsqueeze_822: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 2);  unsqueeze_821 = None
    unsqueeze_823: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, 3);  unsqueeze_822 = None
    sub_250: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_814);  convolution_18 = unsqueeze_814 = None
    mul_875: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(sub_250, unsqueeze_820);  sub_250 = unsqueeze_820 = None
    sub_251: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(add_350, mul_875);  mul_875 = None
    sub_252: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(sub_251, unsqueeze_817);  sub_251 = unsqueeze_817 = None
    mul_876: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_823);  sub_252 = unsqueeze_823 = None
    mul_877: "f32[32]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_55);  sum_95 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_876, relu_12, primals_149, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_876 = primals_149 = None
    getitem_268: "f32[8, 96, 28, 28]" = convolution_backward_46[0]
    getitem_269: "f32[32, 96, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_138: "f32[8, 96, 28, 28]" = torch.ops.aten.alias.default(relu_12);  relu_12 = None
    alias_139: "f32[8, 96, 28, 28]" = torch.ops.aten.alias.default(alias_138);  alias_138 = None
    le_31: "b8[8, 96, 28, 28]" = torch.ops.aten.le.Scalar(alias_139, 0);  alias_139 = None
    scalar_tensor_31: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_31: "f32[8, 96, 28, 28]" = torch.ops.aten.where.self(le_31, scalar_tensor_31, getitem_268);  le_31 = scalar_tensor_31 = getitem_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_824: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_825: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 2);  unsqueeze_824 = None
    unsqueeze_826: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_825, 3);  unsqueeze_825 = None
    sum_96: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_253: "f32[8, 96, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_826)
    mul_878: "f32[8, 96, 28, 28]" = torch.ops.aten.mul.Tensor(where_31, sub_253);  sub_253 = None
    sum_97: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_878, [0, 2, 3]);  mul_878 = None
    mul_879: "f32[96]" = torch.ops.aten.mul.Tensor(sum_96, 0.00015943877551020407)
    unsqueeze_827: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_879, 0);  mul_879 = None
    unsqueeze_828: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_827, 2);  unsqueeze_827 = None
    unsqueeze_829: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_828, 3);  unsqueeze_828 = None
    mul_880: "f32[96]" = torch.ops.aten.mul.Tensor(sum_97, 0.00015943877551020407)
    mul_881: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_882: "f32[96]" = torch.ops.aten.mul.Tensor(mul_880, mul_881);  mul_880 = mul_881 = None
    unsqueeze_830: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_882, 0);  mul_882 = None
    unsqueeze_831: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 2);  unsqueeze_830 = None
    unsqueeze_832: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_831, 3);  unsqueeze_831 = None
    mul_883: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_35);  primals_35 = None
    unsqueeze_833: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_883, 0);  mul_883 = None
    unsqueeze_834: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 2);  unsqueeze_833 = None
    unsqueeze_835: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, 3);  unsqueeze_834 = None
    sub_254: "f32[8, 96, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_826);  convolution_17 = unsqueeze_826 = None
    mul_884: "f32[8, 96, 28, 28]" = torch.ops.aten.mul.Tensor(sub_254, unsqueeze_832);  sub_254 = unsqueeze_832 = None
    sub_255: "f32[8, 96, 28, 28]" = torch.ops.aten.sub.Tensor(where_31, mul_884);  where_31 = mul_884 = None
    sub_256: "f32[8, 96, 28, 28]" = torch.ops.aten.sub.Tensor(sub_255, unsqueeze_829);  sub_255 = unsqueeze_829 = None
    mul_885: "f32[8, 96, 28, 28]" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_835);  sub_256 = unsqueeze_835 = None
    mul_886: "f32[96]" = torch.ops.aten.mul.Tensor(sum_97, squeeze_52);  sum_97 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_885, relu_11, primals_148, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 96, [True, True, False]);  mul_885 = primals_148 = None
    getitem_271: "f32[8, 96, 28, 28]" = convolution_backward_47[0]
    getitem_272: "f32[96, 1, 5, 5]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_141: "f32[8, 96, 28, 28]" = torch.ops.aten.alias.default(relu_11);  relu_11 = None
    alias_142: "f32[8, 96, 28, 28]" = torch.ops.aten.alias.default(alias_141);  alias_141 = None
    le_32: "b8[8, 96, 28, 28]" = torch.ops.aten.le.Scalar(alias_142, 0);  alias_142 = None
    scalar_tensor_32: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_32: "f32[8, 96, 28, 28]" = torch.ops.aten.where.self(le_32, scalar_tensor_32, getitem_271);  le_32 = scalar_tensor_32 = getitem_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_836: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_837: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, 2);  unsqueeze_836 = None
    unsqueeze_838: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_837, 3);  unsqueeze_837 = None
    sum_98: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
    sub_257: "f32[8, 96, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_838)
    mul_887: "f32[8, 96, 28, 28]" = torch.ops.aten.mul.Tensor(where_32, sub_257);  sub_257 = None
    sum_99: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_887, [0, 2, 3]);  mul_887 = None
    mul_888: "f32[96]" = torch.ops.aten.mul.Tensor(sum_98, 0.00015943877551020407)
    unsqueeze_839: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_888, 0);  mul_888 = None
    unsqueeze_840: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_839, 2);  unsqueeze_839 = None
    unsqueeze_841: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_840, 3);  unsqueeze_840 = None
    mul_889: "f32[96]" = torch.ops.aten.mul.Tensor(sum_99, 0.00015943877551020407)
    mul_890: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_891: "f32[96]" = torch.ops.aten.mul.Tensor(mul_889, mul_890);  mul_889 = mul_890 = None
    unsqueeze_842: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_891, 0);  mul_891 = None
    unsqueeze_843: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, 2);  unsqueeze_842 = None
    unsqueeze_844: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_843, 3);  unsqueeze_843 = None
    mul_892: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_33);  primals_33 = None
    unsqueeze_845: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_892, 0);  mul_892 = None
    unsqueeze_846: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 2);  unsqueeze_845 = None
    unsqueeze_847: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, 3);  unsqueeze_846 = None
    sub_258: "f32[8, 96, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_838);  convolution_16 = unsqueeze_838 = None
    mul_893: "f32[8, 96, 28, 28]" = torch.ops.aten.mul.Tensor(sub_258, unsqueeze_844);  sub_258 = unsqueeze_844 = None
    sub_259: "f32[8, 96, 28, 28]" = torch.ops.aten.sub.Tensor(where_32, mul_893);  where_32 = mul_893 = None
    sub_260: "f32[8, 96, 28, 28]" = torch.ops.aten.sub.Tensor(sub_259, unsqueeze_841);  sub_259 = unsqueeze_841 = None
    mul_894: "f32[8, 96, 28, 28]" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_847);  sub_260 = unsqueeze_847 = None
    mul_895: "f32[96]" = torch.ops.aten.mul.Tensor(sum_99, squeeze_49);  sum_99 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_894, add_82, primals_147, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_894 = add_82 = primals_147 = None
    getitem_274: "f32[8, 32, 28, 28]" = convolution_backward_48[0]
    getitem_275: "f32[96, 32, 1, 1]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_351: "f32[8, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_350, getitem_274);  add_350 = getitem_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_848: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_849: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, 2);  unsqueeze_848 = None
    unsqueeze_850: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_849, 3);  unsqueeze_849 = None
    sum_100: "f32[32]" = torch.ops.aten.sum.dim_IntList(add_351, [0, 2, 3])
    sub_261: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_850)
    mul_896: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(add_351, sub_261);  sub_261 = None
    sum_101: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_896, [0, 2, 3]);  mul_896 = None
    mul_897: "f32[32]" = torch.ops.aten.mul.Tensor(sum_100, 0.00015943877551020407)
    unsqueeze_851: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_897, 0);  mul_897 = None
    unsqueeze_852: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_851, 2);  unsqueeze_851 = None
    unsqueeze_853: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_852, 3);  unsqueeze_852 = None
    mul_898: "f32[32]" = torch.ops.aten.mul.Tensor(sum_101, 0.00015943877551020407)
    mul_899: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_900: "f32[32]" = torch.ops.aten.mul.Tensor(mul_898, mul_899);  mul_898 = mul_899 = None
    unsqueeze_854: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_900, 0);  mul_900 = None
    unsqueeze_855: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, 2);  unsqueeze_854 = None
    unsqueeze_856: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_855, 3);  unsqueeze_855 = None
    mul_901: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_31);  primals_31 = None
    unsqueeze_857: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_901, 0);  mul_901 = None
    unsqueeze_858: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 2);  unsqueeze_857 = None
    unsqueeze_859: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, 3);  unsqueeze_858 = None
    sub_262: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_850);  convolution_15 = unsqueeze_850 = None
    mul_902: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(sub_262, unsqueeze_856);  sub_262 = unsqueeze_856 = None
    sub_263: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(add_351, mul_902);  add_351 = mul_902 = None
    sub_264: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(sub_263, unsqueeze_853);  sub_263 = unsqueeze_853 = None
    mul_903: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_859);  sub_264 = unsqueeze_859 = None
    mul_904: "f32[32]" = torch.ops.aten.mul.Tensor(sum_101, squeeze_46);  sum_101 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_903, relu_10, primals_146, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_903 = primals_146 = None
    getitem_277: "f32[8, 144, 28, 28]" = convolution_backward_49[0]
    getitem_278: "f32[32, 144, 1, 1]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_144: "f32[8, 144, 28, 28]" = torch.ops.aten.alias.default(relu_10);  relu_10 = None
    alias_145: "f32[8, 144, 28, 28]" = torch.ops.aten.alias.default(alias_144);  alias_144 = None
    le_33: "b8[8, 144, 28, 28]" = torch.ops.aten.le.Scalar(alias_145, 0);  alias_145 = None
    scalar_tensor_33: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_33: "f32[8, 144, 28, 28]" = torch.ops.aten.where.self(le_33, scalar_tensor_33, getitem_277);  le_33 = scalar_tensor_33 = getitem_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_860: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_861: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, 2);  unsqueeze_860 = None
    unsqueeze_862: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_861, 3);  unsqueeze_861 = None
    sum_102: "f32[144]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    sub_265: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_862)
    mul_905: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(where_33, sub_265);  sub_265 = None
    sum_103: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_905, [0, 2, 3]);  mul_905 = None
    mul_906: "f32[144]" = torch.ops.aten.mul.Tensor(sum_102, 0.00015943877551020407)
    unsqueeze_863: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_906, 0);  mul_906 = None
    unsqueeze_864: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_863, 2);  unsqueeze_863 = None
    unsqueeze_865: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_864, 3);  unsqueeze_864 = None
    mul_907: "f32[144]" = torch.ops.aten.mul.Tensor(sum_103, 0.00015943877551020407)
    mul_908: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_909: "f32[144]" = torch.ops.aten.mul.Tensor(mul_907, mul_908);  mul_907 = mul_908 = None
    unsqueeze_866: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_909, 0);  mul_909 = None
    unsqueeze_867: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, 2);  unsqueeze_866 = None
    unsqueeze_868: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_867, 3);  unsqueeze_867 = None
    mul_910: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_29);  primals_29 = None
    unsqueeze_869: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_910, 0);  mul_910 = None
    unsqueeze_870: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 2);  unsqueeze_869 = None
    unsqueeze_871: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, 3);  unsqueeze_870 = None
    sub_266: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_862);  convolution_14 = unsqueeze_862 = None
    mul_911: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_266, unsqueeze_868);  sub_266 = unsqueeze_868 = None
    sub_267: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(where_33, mul_911);  where_33 = mul_911 = None
    sub_268: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(sub_267, unsqueeze_865);  sub_267 = unsqueeze_865 = None
    mul_912: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_871);  sub_268 = unsqueeze_871 = None
    mul_913: "f32[144]" = torch.ops.aten.mul.Tensor(sum_103, squeeze_43);  sum_103 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_912, relu_9, primals_145, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 144, [True, True, False]);  mul_912 = primals_145 = None
    getitem_280: "f32[8, 144, 56, 56]" = convolution_backward_50[0]
    getitem_281: "f32[144, 1, 5, 5]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_147: "f32[8, 144, 56, 56]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
    alias_148: "f32[8, 144, 56, 56]" = torch.ops.aten.alias.default(alias_147);  alias_147 = None
    le_34: "b8[8, 144, 56, 56]" = torch.ops.aten.le.Scalar(alias_148, 0);  alias_148 = None
    scalar_tensor_34: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_34: "f32[8, 144, 56, 56]" = torch.ops.aten.where.self(le_34, scalar_tensor_34, getitem_280);  le_34 = scalar_tensor_34 = getitem_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_872: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_873: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, 2);  unsqueeze_872 = None
    unsqueeze_874: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_873, 3);  unsqueeze_873 = None
    sum_104: "f32[144]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
    sub_269: "f32[8, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_874)
    mul_914: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(where_34, sub_269);  sub_269 = None
    sum_105: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_914, [0, 2, 3]);  mul_914 = None
    mul_915: "f32[144]" = torch.ops.aten.mul.Tensor(sum_104, 3.985969387755102e-05)
    unsqueeze_875: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_915, 0);  mul_915 = None
    unsqueeze_876: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_875, 2);  unsqueeze_875 = None
    unsqueeze_877: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_876, 3);  unsqueeze_876 = None
    mul_916: "f32[144]" = torch.ops.aten.mul.Tensor(sum_105, 3.985969387755102e-05)
    mul_917: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_918: "f32[144]" = torch.ops.aten.mul.Tensor(mul_916, mul_917);  mul_916 = mul_917 = None
    unsqueeze_878: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_918, 0);  mul_918 = None
    unsqueeze_879: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, 2);  unsqueeze_878 = None
    unsqueeze_880: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_879, 3);  unsqueeze_879 = None
    mul_919: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_27);  primals_27 = None
    unsqueeze_881: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_919, 0);  mul_919 = None
    unsqueeze_882: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 2);  unsqueeze_881 = None
    unsqueeze_883: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, 3);  unsqueeze_882 = None
    sub_270: "f32[8, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_874);  convolution_13 = unsqueeze_874 = None
    mul_920: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_270, unsqueeze_880);  sub_270 = unsqueeze_880 = None
    sub_271: "f32[8, 144, 56, 56]" = torch.ops.aten.sub.Tensor(where_34, mul_920);  where_34 = mul_920 = None
    sub_272: "f32[8, 144, 56, 56]" = torch.ops.aten.sub.Tensor(sub_271, unsqueeze_877);  sub_271 = unsqueeze_877 = None
    mul_921: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_883);  sub_272 = unsqueeze_883 = None
    mul_922: "f32[144]" = torch.ops.aten.mul.Tensor(sum_105, squeeze_40);  sum_105 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_921, add_67, primals_144, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_921 = add_67 = primals_144 = None
    getitem_283: "f32[8, 24, 56, 56]" = convolution_backward_51[0]
    getitem_284: "f32[144, 24, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_884: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_885: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, 2);  unsqueeze_884 = None
    unsqueeze_886: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_885, 3);  unsqueeze_885 = None
    sum_106: "f32[24]" = torch.ops.aten.sum.dim_IntList(getitem_283, [0, 2, 3])
    sub_273: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_886)
    mul_923: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_283, sub_273);  sub_273 = None
    sum_107: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_923, [0, 2, 3]);  mul_923 = None
    mul_924: "f32[24]" = torch.ops.aten.mul.Tensor(sum_106, 3.985969387755102e-05)
    unsqueeze_887: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_924, 0);  mul_924 = None
    unsqueeze_888: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_887, 2);  unsqueeze_887 = None
    unsqueeze_889: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_888, 3);  unsqueeze_888 = None
    mul_925: "f32[24]" = torch.ops.aten.mul.Tensor(sum_107, 3.985969387755102e-05)
    mul_926: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_927: "f32[24]" = torch.ops.aten.mul.Tensor(mul_925, mul_926);  mul_925 = mul_926 = None
    unsqueeze_890: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_927, 0);  mul_927 = None
    unsqueeze_891: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, 2);  unsqueeze_890 = None
    unsqueeze_892: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_891, 3);  unsqueeze_891 = None
    mul_928: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_25);  primals_25 = None
    unsqueeze_893: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_928, 0);  mul_928 = None
    unsqueeze_894: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_893, 2);  unsqueeze_893 = None
    unsqueeze_895: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, 3);  unsqueeze_894 = None
    sub_274: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_886);  convolution_12 = unsqueeze_886 = None
    mul_929: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_274, unsqueeze_892);  sub_274 = unsqueeze_892 = None
    sub_275: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(getitem_283, mul_929);  mul_929 = None
    sub_276: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_275, unsqueeze_889);  sub_275 = unsqueeze_889 = None
    mul_930: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_276, unsqueeze_895);  sub_276 = unsqueeze_895 = None
    mul_931: "f32[24]" = torch.ops.aten.mul.Tensor(sum_107, squeeze_37);  sum_107 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_930, relu_8, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_930 = primals_143 = None
    getitem_286: "f32[8, 24, 56, 56]" = convolution_backward_52[0]
    getitem_287: "f32[24, 24, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_150: "f32[8, 24, 56, 56]" = torch.ops.aten.alias.default(relu_8);  relu_8 = None
    alias_151: "f32[8, 24, 56, 56]" = torch.ops.aten.alias.default(alias_150);  alias_150 = None
    le_35: "b8[8, 24, 56, 56]" = torch.ops.aten.le.Scalar(alias_151, 0);  alias_151 = None
    scalar_tensor_35: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_35: "f32[8, 24, 56, 56]" = torch.ops.aten.where.self(le_35, scalar_tensor_35, getitem_286);  le_35 = scalar_tensor_35 = getitem_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_896: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_897: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, 2);  unsqueeze_896 = None
    unsqueeze_898: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_897, 3);  unsqueeze_897 = None
    sum_108: "f32[24]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    sub_277: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_898)
    mul_932: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(where_35, sub_277);  sub_277 = None
    sum_109: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_932, [0, 2, 3]);  mul_932 = None
    mul_933: "f32[24]" = torch.ops.aten.mul.Tensor(sum_108, 3.985969387755102e-05)
    unsqueeze_899: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_933, 0);  mul_933 = None
    unsqueeze_900: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_899, 2);  unsqueeze_899 = None
    unsqueeze_901: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_900, 3);  unsqueeze_900 = None
    mul_934: "f32[24]" = torch.ops.aten.mul.Tensor(sum_109, 3.985969387755102e-05)
    mul_935: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_936: "f32[24]" = torch.ops.aten.mul.Tensor(mul_934, mul_935);  mul_934 = mul_935 = None
    unsqueeze_902: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_936, 0);  mul_936 = None
    unsqueeze_903: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, 2);  unsqueeze_902 = None
    unsqueeze_904: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_903, 3);  unsqueeze_903 = None
    mul_937: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_23);  primals_23 = None
    unsqueeze_905: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_937, 0);  mul_937 = None
    unsqueeze_906: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_905, 2);  unsqueeze_905 = None
    unsqueeze_907: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, 3);  unsqueeze_906 = None
    sub_278: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_898);  convolution_11 = unsqueeze_898 = None
    mul_938: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_278, unsqueeze_904);  sub_278 = unsqueeze_904 = None
    sub_279: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(where_35, mul_938);  where_35 = mul_938 = None
    sub_280: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_279, unsqueeze_901);  sub_279 = unsqueeze_901 = None
    mul_939: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_280, unsqueeze_907);  sub_280 = unsqueeze_907 = None
    mul_940: "f32[24]" = torch.ops.aten.mul.Tensor(sum_109, squeeze_34);  sum_109 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_939, relu_7, primals_142, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 24, [True, True, False]);  mul_939 = primals_142 = None
    getitem_289: "f32[8, 24, 56, 56]" = convolution_backward_53[0]
    getitem_290: "f32[24, 1, 3, 3]" = convolution_backward_53[1];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_153: "f32[8, 24, 56, 56]" = torch.ops.aten.alias.default(relu_7);  relu_7 = None
    alias_154: "f32[8, 24, 56, 56]" = torch.ops.aten.alias.default(alias_153);  alias_153 = None
    le_36: "b8[8, 24, 56, 56]" = torch.ops.aten.le.Scalar(alias_154, 0);  alias_154 = None
    scalar_tensor_36: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_36: "f32[8, 24, 56, 56]" = torch.ops.aten.where.self(le_36, scalar_tensor_36, getitem_289);  le_36 = scalar_tensor_36 = getitem_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_908: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_909: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, 2);  unsqueeze_908 = None
    unsqueeze_910: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_909, 3);  unsqueeze_909 = None
    sum_110: "f32[24]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
    sub_281: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_910)
    mul_941: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(where_36, sub_281);  sub_281 = None
    sum_111: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_941, [0, 2, 3]);  mul_941 = None
    mul_942: "f32[24]" = torch.ops.aten.mul.Tensor(sum_110, 3.985969387755102e-05)
    unsqueeze_911: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_942, 0);  mul_942 = None
    unsqueeze_912: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_911, 2);  unsqueeze_911 = None
    unsqueeze_913: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_912, 3);  unsqueeze_912 = None
    mul_943: "f32[24]" = torch.ops.aten.mul.Tensor(sum_111, 3.985969387755102e-05)
    mul_944: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_945: "f32[24]" = torch.ops.aten.mul.Tensor(mul_943, mul_944);  mul_943 = mul_944 = None
    unsqueeze_914: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_945, 0);  mul_945 = None
    unsqueeze_915: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, 2);  unsqueeze_914 = None
    unsqueeze_916: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_915, 3);  unsqueeze_915 = None
    mul_946: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_21);  primals_21 = None
    unsqueeze_917: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_946, 0);  mul_946 = None
    unsqueeze_918: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_917, 2);  unsqueeze_917 = None
    unsqueeze_919: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, 3);  unsqueeze_918 = None
    sub_282: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_910);  convolution_10 = unsqueeze_910 = None
    mul_947: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_282, unsqueeze_916);  sub_282 = unsqueeze_916 = None
    sub_283: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(where_36, mul_947);  where_36 = mul_947 = None
    sub_284: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_283, unsqueeze_913);  sub_283 = unsqueeze_913 = None
    mul_948: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_919);  sub_284 = unsqueeze_919 = None
    mul_949: "f32[24]" = torch.ops.aten.mul.Tensor(sum_111, squeeze_31);  sum_111 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_948, add_51, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_948 = add_51 = primals_141 = None
    getitem_292: "f32[8, 24, 56, 56]" = convolution_backward_54[0]
    getitem_293: "f32[24, 24, 1, 1]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_352: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(getitem_283, getitem_292);  getitem_283 = getitem_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_920: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_921: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_920, 2);  unsqueeze_920 = None
    unsqueeze_922: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_921, 3);  unsqueeze_921 = None
    sum_112: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_352, [0, 2, 3])
    sub_285: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_922)
    mul_950: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_352, sub_285);  sub_285 = None
    sum_113: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_950, [0, 2, 3]);  mul_950 = None
    mul_951: "f32[24]" = torch.ops.aten.mul.Tensor(sum_112, 3.985969387755102e-05)
    unsqueeze_923: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_951, 0);  mul_951 = None
    unsqueeze_924: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_923, 2);  unsqueeze_923 = None
    unsqueeze_925: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_924, 3);  unsqueeze_924 = None
    mul_952: "f32[24]" = torch.ops.aten.mul.Tensor(sum_113, 3.985969387755102e-05)
    mul_953: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_954: "f32[24]" = torch.ops.aten.mul.Tensor(mul_952, mul_953);  mul_952 = mul_953 = None
    unsqueeze_926: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_954, 0);  mul_954 = None
    unsqueeze_927: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, 2);  unsqueeze_926 = None
    unsqueeze_928: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_927, 3);  unsqueeze_927 = None
    mul_955: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_19);  primals_19 = None
    unsqueeze_929: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_955, 0);  mul_955 = None
    unsqueeze_930: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_929, 2);  unsqueeze_929 = None
    unsqueeze_931: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_930, 3);  unsqueeze_930 = None
    sub_286: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_922);  convolution_9 = unsqueeze_922 = None
    mul_956: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_286, unsqueeze_928);  sub_286 = unsqueeze_928 = None
    sub_287: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(add_352, mul_956);  mul_956 = None
    sub_288: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_287, unsqueeze_925);  sub_287 = unsqueeze_925 = None
    mul_957: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_288, unsqueeze_931);  sub_288 = unsqueeze_931 = None
    mul_958: "f32[24]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_28);  sum_113 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_957, relu_6, primals_140, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_957 = primals_140 = None
    getitem_295: "f32[8, 24, 56, 56]" = convolution_backward_55[0]
    getitem_296: "f32[24, 24, 1, 1]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_156: "f32[8, 24, 56, 56]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_157: "f32[8, 24, 56, 56]" = torch.ops.aten.alias.default(alias_156);  alias_156 = None
    le_37: "b8[8, 24, 56, 56]" = torch.ops.aten.le.Scalar(alias_157, 0);  alias_157 = None
    scalar_tensor_37: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_37: "f32[8, 24, 56, 56]" = torch.ops.aten.where.self(le_37, scalar_tensor_37, getitem_295);  le_37 = scalar_tensor_37 = getitem_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_932: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_933: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_932, 2);  unsqueeze_932 = None
    unsqueeze_934: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_933, 3);  unsqueeze_933 = None
    sum_114: "f32[24]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
    sub_289: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_934)
    mul_959: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(where_37, sub_289);  sub_289 = None
    sum_115: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_959, [0, 2, 3]);  mul_959 = None
    mul_960: "f32[24]" = torch.ops.aten.mul.Tensor(sum_114, 3.985969387755102e-05)
    unsqueeze_935: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_960, 0);  mul_960 = None
    unsqueeze_936: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_935, 2);  unsqueeze_935 = None
    unsqueeze_937: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_936, 3);  unsqueeze_936 = None
    mul_961: "f32[24]" = torch.ops.aten.mul.Tensor(sum_115, 3.985969387755102e-05)
    mul_962: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_963: "f32[24]" = torch.ops.aten.mul.Tensor(mul_961, mul_962);  mul_961 = mul_962 = None
    unsqueeze_938: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_963, 0);  mul_963 = None
    unsqueeze_939: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_938, 2);  unsqueeze_938 = None
    unsqueeze_940: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_939, 3);  unsqueeze_939 = None
    mul_964: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_17);  primals_17 = None
    unsqueeze_941: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_964, 0);  mul_964 = None
    unsqueeze_942: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_941, 2);  unsqueeze_941 = None
    unsqueeze_943: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_942, 3);  unsqueeze_942 = None
    sub_290: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_934);  convolution_8 = unsqueeze_934 = None
    mul_965: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_290, unsqueeze_940);  sub_290 = unsqueeze_940 = None
    sub_291: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(where_37, mul_965);  where_37 = mul_965 = None
    sub_292: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_291, unsqueeze_937);  sub_291 = unsqueeze_937 = None
    mul_966: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_292, unsqueeze_943);  sub_292 = unsqueeze_943 = None
    mul_967: "f32[24]" = torch.ops.aten.mul.Tensor(sum_115, squeeze_25);  sum_115 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_966, relu_5, primals_139, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 24, [True, True, False]);  mul_966 = primals_139 = None
    getitem_298: "f32[8, 24, 56, 56]" = convolution_backward_56[0]
    getitem_299: "f32[24, 1, 3, 3]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_159: "f32[8, 24, 56, 56]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_160: "f32[8, 24, 56, 56]" = torch.ops.aten.alias.default(alias_159);  alias_159 = None
    le_38: "b8[8, 24, 56, 56]" = torch.ops.aten.le.Scalar(alias_160, 0);  alias_160 = None
    scalar_tensor_38: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_38: "f32[8, 24, 56, 56]" = torch.ops.aten.where.self(le_38, scalar_tensor_38, getitem_298);  le_38 = scalar_tensor_38 = getitem_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_944: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_945: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_944, 2);  unsqueeze_944 = None
    unsqueeze_946: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_945, 3);  unsqueeze_945 = None
    sum_116: "f32[24]" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
    sub_293: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_946)
    mul_968: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(where_38, sub_293);  sub_293 = None
    sum_117: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_968, [0, 2, 3]);  mul_968 = None
    mul_969: "f32[24]" = torch.ops.aten.mul.Tensor(sum_116, 3.985969387755102e-05)
    unsqueeze_947: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_969, 0);  mul_969 = None
    unsqueeze_948: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_947, 2);  unsqueeze_947 = None
    unsqueeze_949: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_948, 3);  unsqueeze_948 = None
    mul_970: "f32[24]" = torch.ops.aten.mul.Tensor(sum_117, 3.985969387755102e-05)
    mul_971: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_972: "f32[24]" = torch.ops.aten.mul.Tensor(mul_970, mul_971);  mul_970 = mul_971 = None
    unsqueeze_950: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_972, 0);  mul_972 = None
    unsqueeze_951: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_950, 2);  unsqueeze_950 = None
    unsqueeze_952: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_951, 3);  unsqueeze_951 = None
    mul_973: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_15);  primals_15 = None
    unsqueeze_953: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_973, 0);  mul_973 = None
    unsqueeze_954: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_953, 2);  unsqueeze_953 = None
    unsqueeze_955: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_954, 3);  unsqueeze_954 = None
    sub_294: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_946);  convolution_7 = unsqueeze_946 = None
    mul_974: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_294, unsqueeze_952);  sub_294 = unsqueeze_952 = None
    sub_295: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(where_38, mul_974);  where_38 = mul_974 = None
    sub_296: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_295, unsqueeze_949);  sub_295 = unsqueeze_949 = None
    mul_975: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_296, unsqueeze_955);  sub_296 = unsqueeze_955 = None
    mul_976: "f32[24]" = torch.ops.aten.mul.Tensor(sum_117, squeeze_22);  sum_117 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_975, add_35, primals_138, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_975 = add_35 = primals_138 = None
    getitem_301: "f32[8, 24, 56, 56]" = convolution_backward_57[0]
    getitem_302: "f32[24, 24, 1, 1]" = convolution_backward_57[1];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_353: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(add_352, getitem_301);  add_352 = getitem_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_956: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_957: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_956, 2);  unsqueeze_956 = None
    unsqueeze_958: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_957, 3);  unsqueeze_957 = None
    sum_118: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_353, [0, 2, 3])
    sub_297: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_958)
    mul_977: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_353, sub_297);  sub_297 = None
    sum_119: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_977, [0, 2, 3]);  mul_977 = None
    mul_978: "f32[24]" = torch.ops.aten.mul.Tensor(sum_118, 3.985969387755102e-05)
    unsqueeze_959: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_978, 0);  mul_978 = None
    unsqueeze_960: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_959, 2);  unsqueeze_959 = None
    unsqueeze_961: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_960, 3);  unsqueeze_960 = None
    mul_979: "f32[24]" = torch.ops.aten.mul.Tensor(sum_119, 3.985969387755102e-05)
    mul_980: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_981: "f32[24]" = torch.ops.aten.mul.Tensor(mul_979, mul_980);  mul_979 = mul_980 = None
    unsqueeze_962: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_981, 0);  mul_981 = None
    unsqueeze_963: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_962, 2);  unsqueeze_962 = None
    unsqueeze_964: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_963, 3);  unsqueeze_963 = None
    mul_982: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_13);  primals_13 = None
    unsqueeze_965: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_982, 0);  mul_982 = None
    unsqueeze_966: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_965, 2);  unsqueeze_965 = None
    unsqueeze_967: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_966, 3);  unsqueeze_966 = None
    sub_298: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_958);  convolution_6 = unsqueeze_958 = None
    mul_983: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_298, unsqueeze_964);  sub_298 = unsqueeze_964 = None
    sub_299: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(add_353, mul_983);  add_353 = mul_983 = None
    sub_300: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_299, unsqueeze_961);  sub_299 = unsqueeze_961 = None
    mul_984: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_300, unsqueeze_967);  sub_300 = unsqueeze_967 = None
    mul_985: "f32[24]" = torch.ops.aten.mul.Tensor(sum_119, squeeze_19);  sum_119 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_984, relu_4, primals_137, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_984 = primals_137 = None
    getitem_304: "f32[8, 96, 56, 56]" = convolution_backward_58[0]
    getitem_305: "f32[24, 96, 1, 1]" = convolution_backward_58[1];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_162: "f32[8, 96, 56, 56]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_163: "f32[8, 96, 56, 56]" = torch.ops.aten.alias.default(alias_162);  alias_162 = None
    le_39: "b8[8, 96, 56, 56]" = torch.ops.aten.le.Scalar(alias_163, 0);  alias_163 = None
    scalar_tensor_39: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_39: "f32[8, 96, 56, 56]" = torch.ops.aten.where.self(le_39, scalar_tensor_39, getitem_304);  le_39 = scalar_tensor_39 = getitem_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_968: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_969: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_968, 2);  unsqueeze_968 = None
    unsqueeze_970: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_969, 3);  unsqueeze_969 = None
    sum_120: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
    sub_301: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_970)
    mul_986: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(where_39, sub_301);  sub_301 = None
    sum_121: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_986, [0, 2, 3]);  mul_986 = None
    mul_987: "f32[96]" = torch.ops.aten.mul.Tensor(sum_120, 3.985969387755102e-05)
    unsqueeze_971: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_987, 0);  mul_987 = None
    unsqueeze_972: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_971, 2);  unsqueeze_971 = None
    unsqueeze_973: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_972, 3);  unsqueeze_972 = None
    mul_988: "f32[96]" = torch.ops.aten.mul.Tensor(sum_121, 3.985969387755102e-05)
    mul_989: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_990: "f32[96]" = torch.ops.aten.mul.Tensor(mul_988, mul_989);  mul_988 = mul_989 = None
    unsqueeze_974: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_990, 0);  mul_990 = None
    unsqueeze_975: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_974, 2);  unsqueeze_974 = None
    unsqueeze_976: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_975, 3);  unsqueeze_975 = None
    mul_991: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_11);  primals_11 = None
    unsqueeze_977: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_991, 0);  mul_991 = None
    unsqueeze_978: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_977, 2);  unsqueeze_977 = None
    unsqueeze_979: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_978, 3);  unsqueeze_978 = None
    sub_302: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_970);  convolution_5 = unsqueeze_970 = None
    mul_992: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_302, unsqueeze_976);  sub_302 = unsqueeze_976 = None
    sub_303: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(where_39, mul_992);  where_39 = mul_992 = None
    sub_304: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(sub_303, unsqueeze_973);  sub_303 = unsqueeze_973 = None
    mul_993: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_304, unsqueeze_979);  sub_304 = unsqueeze_979 = None
    mul_994: "f32[96]" = torch.ops.aten.mul.Tensor(sum_121, squeeze_16);  sum_121 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(mul_993, relu_3, primals_136, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 96, [True, True, False]);  mul_993 = primals_136 = None
    getitem_307: "f32[8, 96, 112, 112]" = convolution_backward_59[0]
    getitem_308: "f32[96, 1, 3, 3]" = convolution_backward_59[1];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_165: "f32[8, 96, 112, 112]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_166: "f32[8, 96, 112, 112]" = torch.ops.aten.alias.default(alias_165);  alias_165 = None
    le_40: "b8[8, 96, 112, 112]" = torch.ops.aten.le.Scalar(alias_166, 0);  alias_166 = None
    scalar_tensor_40: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_40: "f32[8, 96, 112, 112]" = torch.ops.aten.where.self(le_40, scalar_tensor_40, getitem_307);  le_40 = scalar_tensor_40 = getitem_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_980: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_981: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_980, 2);  unsqueeze_980 = None
    unsqueeze_982: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_981, 3);  unsqueeze_981 = None
    sum_122: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
    sub_305: "f32[8, 96, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_982)
    mul_995: "f32[8, 96, 112, 112]" = torch.ops.aten.mul.Tensor(where_40, sub_305);  sub_305 = None
    sum_123: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_995, [0, 2, 3]);  mul_995 = None
    mul_996: "f32[96]" = torch.ops.aten.mul.Tensor(sum_122, 9.964923469387754e-06)
    unsqueeze_983: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_996, 0);  mul_996 = None
    unsqueeze_984: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_983, 2);  unsqueeze_983 = None
    unsqueeze_985: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_984, 3);  unsqueeze_984 = None
    mul_997: "f32[96]" = torch.ops.aten.mul.Tensor(sum_123, 9.964923469387754e-06)
    mul_998: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_999: "f32[96]" = torch.ops.aten.mul.Tensor(mul_997, mul_998);  mul_997 = mul_998 = None
    unsqueeze_986: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_999, 0);  mul_999 = None
    unsqueeze_987: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_986, 2);  unsqueeze_986 = None
    unsqueeze_988: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_987, 3);  unsqueeze_987 = None
    mul_1000: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_9);  primals_9 = None
    unsqueeze_989: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1000, 0);  mul_1000 = None
    unsqueeze_990: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_989, 2);  unsqueeze_989 = None
    unsqueeze_991: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_990, 3);  unsqueeze_990 = None
    sub_306: "f32[8, 96, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_982);  convolution_4 = unsqueeze_982 = None
    mul_1001: "f32[8, 96, 112, 112]" = torch.ops.aten.mul.Tensor(sub_306, unsqueeze_988);  sub_306 = unsqueeze_988 = None
    sub_307: "f32[8, 96, 112, 112]" = torch.ops.aten.sub.Tensor(where_40, mul_1001);  where_40 = mul_1001 = None
    sub_308: "f32[8, 96, 112, 112]" = torch.ops.aten.sub.Tensor(sub_307, unsqueeze_985);  sub_307 = unsqueeze_985 = None
    mul_1002: "f32[8, 96, 112, 112]" = torch.ops.aten.mul.Tensor(sub_308, unsqueeze_991);  sub_308 = unsqueeze_991 = None
    mul_1003: "f32[96]" = torch.ops.aten.mul.Tensor(sum_123, squeeze_13);  sum_123 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_1002, add_20, primals_135, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1002 = add_20 = primals_135 = None
    getitem_310: "f32[8, 16, 112, 112]" = convolution_backward_60[0]
    getitem_311: "f32[96, 16, 1, 1]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_992: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_993: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_992, 2);  unsqueeze_992 = None
    unsqueeze_994: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_993, 3);  unsqueeze_993 = None
    sum_124: "f32[16]" = torch.ops.aten.sum.dim_IntList(getitem_310, [0, 2, 3])
    sub_309: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_994)
    mul_1004: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_310, sub_309);  sub_309 = None
    sum_125: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1004, [0, 2, 3]);  mul_1004 = None
    mul_1005: "f32[16]" = torch.ops.aten.mul.Tensor(sum_124, 9.964923469387754e-06)
    unsqueeze_995: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1005, 0);  mul_1005 = None
    unsqueeze_996: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_995, 2);  unsqueeze_995 = None
    unsqueeze_997: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_996, 3);  unsqueeze_996 = None
    mul_1006: "f32[16]" = torch.ops.aten.mul.Tensor(sum_125, 9.964923469387754e-06)
    mul_1007: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_1008: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1006, mul_1007);  mul_1006 = mul_1007 = None
    unsqueeze_998: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1008, 0);  mul_1008 = None
    unsqueeze_999: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_998, 2);  unsqueeze_998 = None
    unsqueeze_1000: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_999, 3);  unsqueeze_999 = None
    mul_1009: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_7);  primals_7 = None
    unsqueeze_1001: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1009, 0);  mul_1009 = None
    unsqueeze_1002: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1001, 2);  unsqueeze_1001 = None
    unsqueeze_1003: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1002, 3);  unsqueeze_1002 = None
    sub_310: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_994);  convolution_3 = unsqueeze_994 = None
    mul_1010: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_310, unsqueeze_1000);  sub_310 = unsqueeze_1000 = None
    sub_311: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(getitem_310, mul_1010);  mul_1010 = None
    sub_312: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_311, unsqueeze_997);  sub_311 = unsqueeze_997 = None
    mul_1011: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_312, unsqueeze_1003);  sub_312 = unsqueeze_1003 = None
    mul_1012: "f32[16]" = torch.ops.aten.mul.Tensor(sum_125, squeeze_10);  sum_125 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_1011, relu_2, primals_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1011 = primals_134 = None
    getitem_313: "f32[8, 16, 112, 112]" = convolution_backward_61[0]
    getitem_314: "f32[16, 16, 1, 1]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_168: "f32[8, 16, 112, 112]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_169: "f32[8, 16, 112, 112]" = torch.ops.aten.alias.default(alias_168);  alias_168 = None
    le_41: "b8[8, 16, 112, 112]" = torch.ops.aten.le.Scalar(alias_169, 0);  alias_169 = None
    scalar_tensor_41: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_41: "f32[8, 16, 112, 112]" = torch.ops.aten.where.self(le_41, scalar_tensor_41, getitem_313);  le_41 = scalar_tensor_41 = getitem_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1004: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_1005: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1004, 2);  unsqueeze_1004 = None
    unsqueeze_1006: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1005, 3);  unsqueeze_1005 = None
    sum_126: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
    sub_313: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_1006)
    mul_1013: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(where_41, sub_313);  sub_313 = None
    sum_127: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1013, [0, 2, 3]);  mul_1013 = None
    mul_1014: "f32[16]" = torch.ops.aten.mul.Tensor(sum_126, 9.964923469387754e-06)
    unsqueeze_1007: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1014, 0);  mul_1014 = None
    unsqueeze_1008: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1007, 2);  unsqueeze_1007 = None
    unsqueeze_1009: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1008, 3);  unsqueeze_1008 = None
    mul_1015: "f32[16]" = torch.ops.aten.mul.Tensor(sum_127, 9.964923469387754e-06)
    mul_1016: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_1017: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1015, mul_1016);  mul_1015 = mul_1016 = None
    unsqueeze_1010: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1017, 0);  mul_1017 = None
    unsqueeze_1011: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1010, 2);  unsqueeze_1010 = None
    unsqueeze_1012: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1011, 3);  unsqueeze_1011 = None
    mul_1018: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_5);  primals_5 = None
    unsqueeze_1013: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1018, 0);  mul_1018 = None
    unsqueeze_1014: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1013, 2);  unsqueeze_1013 = None
    unsqueeze_1015: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1014, 3);  unsqueeze_1014 = None
    sub_314: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_1006);  convolution_2 = unsqueeze_1006 = None
    mul_1019: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_314, unsqueeze_1012);  sub_314 = unsqueeze_1012 = None
    sub_315: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(where_41, mul_1019);  where_41 = mul_1019 = None
    sub_316: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_315, unsqueeze_1009);  sub_315 = unsqueeze_1009 = None
    mul_1020: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_316, unsqueeze_1015);  sub_316 = unsqueeze_1015 = None
    mul_1021: "f32[16]" = torch.ops.aten.mul.Tensor(sum_127, squeeze_7);  sum_127 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_1020, relu_1, primals_133, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False]);  mul_1020 = primals_133 = None
    getitem_316: "f32[8, 16, 112, 112]" = convolution_backward_62[0]
    getitem_317: "f32[16, 1, 3, 3]" = convolution_backward_62[1];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_171: "f32[8, 16, 112, 112]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_172: "f32[8, 16, 112, 112]" = torch.ops.aten.alias.default(alias_171);  alias_171 = None
    le_42: "b8[8, 16, 112, 112]" = torch.ops.aten.le.Scalar(alias_172, 0);  alias_172 = None
    scalar_tensor_42: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_42: "f32[8, 16, 112, 112]" = torch.ops.aten.where.self(le_42, scalar_tensor_42, getitem_316);  le_42 = scalar_tensor_42 = getitem_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1016: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_1017: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1016, 2);  unsqueeze_1016 = None
    unsqueeze_1018: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1017, 3);  unsqueeze_1017 = None
    sum_128: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_42, [0, 2, 3])
    sub_317: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1018)
    mul_1022: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(where_42, sub_317);  sub_317 = None
    sum_129: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1022, [0, 2, 3]);  mul_1022 = None
    mul_1023: "f32[16]" = torch.ops.aten.mul.Tensor(sum_128, 9.964923469387754e-06)
    unsqueeze_1019: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1023, 0);  mul_1023 = None
    unsqueeze_1020: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1019, 2);  unsqueeze_1019 = None
    unsqueeze_1021: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1020, 3);  unsqueeze_1020 = None
    mul_1024: "f32[16]" = torch.ops.aten.mul.Tensor(sum_129, 9.964923469387754e-06)
    mul_1025: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_1026: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1024, mul_1025);  mul_1024 = mul_1025 = None
    unsqueeze_1022: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1026, 0);  mul_1026 = None
    unsqueeze_1023: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1022, 2);  unsqueeze_1022 = None
    unsqueeze_1024: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1023, 3);  unsqueeze_1023 = None
    mul_1027: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_3);  primals_3 = None
    unsqueeze_1025: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1027, 0);  mul_1027 = None
    unsqueeze_1026: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1025, 2);  unsqueeze_1025 = None
    unsqueeze_1027: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1026, 3);  unsqueeze_1026 = None
    sub_318: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1018);  convolution_1 = unsqueeze_1018 = None
    mul_1028: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_318, unsqueeze_1024);  sub_318 = unsqueeze_1024 = None
    sub_319: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(where_42, mul_1028);  where_42 = mul_1028 = None
    sub_320: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_319, unsqueeze_1021);  sub_319 = unsqueeze_1021 = None
    mul_1029: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_320, unsqueeze_1027);  sub_320 = unsqueeze_1027 = None
    mul_1030: "f32[16]" = torch.ops.aten.mul.Tensor(sum_129, squeeze_4);  sum_129 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_1029, relu, primals_132, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1029 = primals_132 = None
    getitem_319: "f32[8, 16, 112, 112]" = convolution_backward_63[0]
    getitem_320: "f32[16, 16, 1, 1]" = convolution_backward_63[1];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_354: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(getitem_310, getitem_319);  getitem_310 = getitem_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_174: "f32[8, 16, 112, 112]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_175: "f32[8, 16, 112, 112]" = torch.ops.aten.alias.default(alias_174);  alias_174 = None
    le_43: "b8[8, 16, 112, 112]" = torch.ops.aten.le.Scalar(alias_175, 0);  alias_175 = None
    scalar_tensor_43: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_43: "f32[8, 16, 112, 112]" = torch.ops.aten.where.self(le_43, scalar_tensor_43, add_354);  le_43 = scalar_tensor_43 = add_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1028: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_1029: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1028, 2);  unsqueeze_1028 = None
    unsqueeze_1030: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1029, 3);  unsqueeze_1029 = None
    sum_130: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
    sub_321: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1030)
    mul_1031: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(where_43, sub_321);  sub_321 = None
    sum_131: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1031, [0, 2, 3]);  mul_1031 = None
    mul_1032: "f32[16]" = torch.ops.aten.mul.Tensor(sum_130, 9.964923469387754e-06)
    unsqueeze_1031: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1032, 0);  mul_1032 = None
    unsqueeze_1032: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1031, 2);  unsqueeze_1031 = None
    unsqueeze_1033: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1032, 3);  unsqueeze_1032 = None
    mul_1033: "f32[16]" = torch.ops.aten.mul.Tensor(sum_131, 9.964923469387754e-06)
    mul_1034: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_1035: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1033, mul_1034);  mul_1033 = mul_1034 = None
    unsqueeze_1034: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1035, 0);  mul_1035 = None
    unsqueeze_1035: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1034, 2);  unsqueeze_1034 = None
    unsqueeze_1036: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1035, 3);  unsqueeze_1035 = None
    mul_1036: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_1);  primals_1 = None
    unsqueeze_1037: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1036, 0);  mul_1036 = None
    unsqueeze_1038: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1037, 2);  unsqueeze_1037 = None
    unsqueeze_1039: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1038, 3);  unsqueeze_1038 = None
    sub_322: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1030);  convolution = unsqueeze_1030 = None
    mul_1037: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_322, unsqueeze_1036);  sub_322 = unsqueeze_1036 = None
    sub_323: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(where_43, mul_1037);  where_43 = mul_1037 = None
    sub_324: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_323, unsqueeze_1033);  sub_323 = unsqueeze_1033 = None
    mul_1038: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_324, unsqueeze_1039);  sub_324 = unsqueeze_1039 = None
    mul_1039: "f32[16]" = torch.ops.aten.mul.Tensor(sum_131, squeeze_1);  sum_131 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:162, code: x = self.conv_stem(x)
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(mul_1038, primals_393, primals_131, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1038 = primals_393 = primals_131 = None
    getitem_323: "f32[16, 3, 3, 3]" = convolution_backward_64[1];  convolution_backward_64 = None
    
    # No stacktrace found for following nodes
    copy_: "i64[]" = torch.ops.aten.copy_.default(primals_198, add);  primals_198 = add = None
    copy__1: "f32[16]" = torch.ops.aten.copy_.default(primals_199, add_2);  primals_199 = add_2 = None
    copy__2: "f32[16]" = torch.ops.aten.copy_.default(primals_200, add_3);  primals_200 = add_3 = None
    copy__3: "i64[]" = torch.ops.aten.copy_.default(primals_201, add_5);  primals_201 = add_5 = None
    copy__4: "f32[16]" = torch.ops.aten.copy_.default(primals_202, add_7);  primals_202 = add_7 = None
    copy__5: "f32[16]" = torch.ops.aten.copy_.default(primals_203, add_8);  primals_203 = add_8 = None
    copy__6: "i64[]" = torch.ops.aten.copy_.default(primals_204, add_10);  primals_204 = add_10 = None
    copy__7: "f32[16]" = torch.ops.aten.copy_.default(primals_205, add_12);  primals_205 = add_12 = None
    copy__8: "f32[16]" = torch.ops.aten.copy_.default(primals_206, add_13);  primals_206 = add_13 = None
    copy__9: "i64[]" = torch.ops.aten.copy_.default(primals_207, add_15);  primals_207 = add_15 = None
    copy__10: "f32[16]" = torch.ops.aten.copy_.default(primals_208, add_17);  primals_208 = add_17 = None
    copy__11: "f32[16]" = torch.ops.aten.copy_.default(primals_209, add_18);  primals_209 = add_18 = None
    copy__12: "i64[]" = torch.ops.aten.copy_.default(primals_210, add_21);  primals_210 = add_21 = None
    copy__13: "f32[96]" = torch.ops.aten.copy_.default(primals_211, add_23);  primals_211 = add_23 = None
    copy__14: "f32[96]" = torch.ops.aten.copy_.default(primals_212, add_24);  primals_212 = add_24 = None
    copy__15: "i64[]" = torch.ops.aten.copy_.default(primals_213, add_26);  primals_213 = add_26 = None
    copy__16: "f32[96]" = torch.ops.aten.copy_.default(primals_214, add_28);  primals_214 = add_28 = None
    copy__17: "f32[96]" = torch.ops.aten.copy_.default(primals_215, add_29);  primals_215 = add_29 = None
    copy__18: "i64[]" = torch.ops.aten.copy_.default(primals_216, add_31);  primals_216 = add_31 = None
    copy__19: "f32[24]" = torch.ops.aten.copy_.default(primals_217, add_33);  primals_217 = add_33 = None
    copy__20: "f32[24]" = torch.ops.aten.copy_.default(primals_218, add_34);  primals_218 = add_34 = None
    copy__21: "i64[]" = torch.ops.aten.copy_.default(primals_219, add_36);  primals_219 = add_36 = None
    copy__22: "f32[24]" = torch.ops.aten.copy_.default(primals_220, add_38);  primals_220 = add_38 = None
    copy__23: "f32[24]" = torch.ops.aten.copy_.default(primals_221, add_39);  primals_221 = add_39 = None
    copy__24: "i64[]" = torch.ops.aten.copy_.default(primals_222, add_41);  primals_222 = add_41 = None
    copy__25: "f32[24]" = torch.ops.aten.copy_.default(primals_223, add_43);  primals_223 = add_43 = None
    copy__26: "f32[24]" = torch.ops.aten.copy_.default(primals_224, add_44);  primals_224 = add_44 = None
    copy__27: "i64[]" = torch.ops.aten.copy_.default(primals_225, add_46);  primals_225 = add_46 = None
    copy__28: "f32[24]" = torch.ops.aten.copy_.default(primals_226, add_48);  primals_226 = add_48 = None
    copy__29: "f32[24]" = torch.ops.aten.copy_.default(primals_227, add_49);  primals_227 = add_49 = None
    copy__30: "i64[]" = torch.ops.aten.copy_.default(primals_228, add_52);  primals_228 = add_52 = None
    copy__31: "f32[24]" = torch.ops.aten.copy_.default(primals_229, add_54);  primals_229 = add_54 = None
    copy__32: "f32[24]" = torch.ops.aten.copy_.default(primals_230, add_55);  primals_230 = add_55 = None
    copy__33: "i64[]" = torch.ops.aten.copy_.default(primals_231, add_57);  primals_231 = add_57 = None
    copy__34: "f32[24]" = torch.ops.aten.copy_.default(primals_232, add_59);  primals_232 = add_59 = None
    copy__35: "f32[24]" = torch.ops.aten.copy_.default(primals_233, add_60);  primals_233 = add_60 = None
    copy__36: "i64[]" = torch.ops.aten.copy_.default(primals_234, add_62);  primals_234 = add_62 = None
    copy__37: "f32[24]" = torch.ops.aten.copy_.default(primals_235, add_64);  primals_235 = add_64 = None
    copy__38: "f32[24]" = torch.ops.aten.copy_.default(primals_236, add_65);  primals_236 = add_65 = None
    copy__39: "i64[]" = torch.ops.aten.copy_.default(primals_237, add_68);  primals_237 = add_68 = None
    copy__40: "f32[144]" = torch.ops.aten.copy_.default(primals_238, add_70);  primals_238 = add_70 = None
    copy__41: "f32[144]" = torch.ops.aten.copy_.default(primals_239, add_71);  primals_239 = add_71 = None
    copy__42: "i64[]" = torch.ops.aten.copy_.default(primals_240, add_73);  primals_240 = add_73 = None
    copy__43: "f32[144]" = torch.ops.aten.copy_.default(primals_241, add_75);  primals_241 = add_75 = None
    copy__44: "f32[144]" = torch.ops.aten.copy_.default(primals_242, add_76);  primals_242 = add_76 = None
    copy__45: "i64[]" = torch.ops.aten.copy_.default(primals_243, add_78);  primals_243 = add_78 = None
    copy__46: "f32[32]" = torch.ops.aten.copy_.default(primals_244, add_80);  primals_244 = add_80 = None
    copy__47: "f32[32]" = torch.ops.aten.copy_.default(primals_245, add_81);  primals_245 = add_81 = None
    copy__48: "i64[]" = torch.ops.aten.copy_.default(primals_246, add_83);  primals_246 = add_83 = None
    copy__49: "f32[96]" = torch.ops.aten.copy_.default(primals_247, add_85);  primals_247 = add_85 = None
    copy__50: "f32[96]" = torch.ops.aten.copy_.default(primals_248, add_86);  primals_248 = add_86 = None
    copy__51: "i64[]" = torch.ops.aten.copy_.default(primals_249, add_88);  primals_249 = add_88 = None
    copy__52: "f32[96]" = torch.ops.aten.copy_.default(primals_250, add_90);  primals_250 = add_90 = None
    copy__53: "f32[96]" = torch.ops.aten.copy_.default(primals_251, add_91);  primals_251 = add_91 = None
    copy__54: "i64[]" = torch.ops.aten.copy_.default(primals_252, add_93);  primals_252 = add_93 = None
    copy__55: "f32[32]" = torch.ops.aten.copy_.default(primals_253, add_95);  primals_253 = add_95 = None
    copy__56: "f32[32]" = torch.ops.aten.copy_.default(primals_254, add_96);  primals_254 = add_96 = None
    copy__57: "i64[]" = torch.ops.aten.copy_.default(primals_255, add_99);  primals_255 = add_99 = None
    copy__58: "f32[192]" = torch.ops.aten.copy_.default(primals_256, add_101);  primals_256 = add_101 = None
    copy__59: "f32[192]" = torch.ops.aten.copy_.default(primals_257, add_102);  primals_257 = add_102 = None
    copy__60: "i64[]" = torch.ops.aten.copy_.default(primals_258, add_104);  primals_258 = add_104 = None
    copy__61: "f32[192]" = torch.ops.aten.copy_.default(primals_259, add_106);  primals_259 = add_106 = None
    copy__62: "f32[192]" = torch.ops.aten.copy_.default(primals_260, add_107);  primals_260 = add_107 = None
    copy__63: "i64[]" = torch.ops.aten.copy_.default(primals_261, add_109);  primals_261 = add_109 = None
    copy__64: "f32[32]" = torch.ops.aten.copy_.default(primals_262, add_111);  primals_262 = add_111 = None
    copy__65: "f32[32]" = torch.ops.aten.copy_.default(primals_263, add_112);  primals_263 = add_112 = None
    copy__66: "i64[]" = torch.ops.aten.copy_.default(primals_264, add_115);  primals_264 = add_115 = None
    copy__67: "f32[192]" = torch.ops.aten.copy_.default(primals_265, add_117);  primals_265 = add_117 = None
    copy__68: "f32[192]" = torch.ops.aten.copy_.default(primals_266, add_118);  primals_266 = add_118 = None
    copy__69: "i64[]" = torch.ops.aten.copy_.default(primals_267, add_120);  primals_267 = add_120 = None
    copy__70: "f32[192]" = torch.ops.aten.copy_.default(primals_268, add_122);  primals_268 = add_122 = None
    copy__71: "f32[192]" = torch.ops.aten.copy_.default(primals_269, add_123);  primals_269 = add_123 = None
    copy__72: "i64[]" = torch.ops.aten.copy_.default(primals_270, add_125);  primals_270 = add_125 = None
    copy__73: "f32[32]" = torch.ops.aten.copy_.default(primals_271, add_127);  primals_271 = add_127 = None
    copy__74: "f32[32]" = torch.ops.aten.copy_.default(primals_272, add_128);  primals_272 = add_128 = None
    copy__75: "i64[]" = torch.ops.aten.copy_.default(primals_273, add_131);  primals_273 = add_131 = None
    copy__76: "f32[192]" = torch.ops.aten.copy_.default(primals_274, add_133);  primals_274 = add_133 = None
    copy__77: "f32[192]" = torch.ops.aten.copy_.default(primals_275, add_134);  primals_275 = add_134 = None
    copy__78: "i64[]" = torch.ops.aten.copy_.default(primals_276, add_136);  primals_276 = add_136 = None
    copy__79: "f32[192]" = torch.ops.aten.copy_.default(primals_277, add_138);  primals_277 = add_138 = None
    copy__80: "f32[192]" = torch.ops.aten.copy_.default(primals_278, add_139);  primals_278 = add_139 = None
    copy__81: "i64[]" = torch.ops.aten.copy_.default(primals_279, add_141);  primals_279 = add_141 = None
    copy__82: "f32[64]" = torch.ops.aten.copy_.default(primals_280, add_143);  primals_280 = add_143 = None
    copy__83: "f32[64]" = torch.ops.aten.copy_.default(primals_281, add_144);  primals_281 = add_144 = None
    copy__84: "i64[]" = torch.ops.aten.copy_.default(primals_282, add_146);  primals_282 = add_146 = None
    copy__85: "f32[192]" = torch.ops.aten.copy_.default(primals_283, add_148);  primals_283 = add_148 = None
    copy__86: "f32[192]" = torch.ops.aten.copy_.default(primals_284, add_149);  primals_284 = add_149 = None
    copy__87: "i64[]" = torch.ops.aten.copy_.default(primals_285, add_151);  primals_285 = add_151 = None
    copy__88: "f32[192]" = torch.ops.aten.copy_.default(primals_286, add_153);  primals_286 = add_153 = None
    copy__89: "f32[192]" = torch.ops.aten.copy_.default(primals_287, add_154);  primals_287 = add_154 = None
    copy__90: "i64[]" = torch.ops.aten.copy_.default(primals_288, add_156);  primals_288 = add_156 = None
    copy__91: "f32[64]" = torch.ops.aten.copy_.default(primals_289, add_158);  primals_289 = add_158 = None
    copy__92: "f32[64]" = torch.ops.aten.copy_.default(primals_290, add_159);  primals_290 = add_159 = None
    copy__93: "i64[]" = torch.ops.aten.copy_.default(primals_291, add_162);  primals_291 = add_162 = None
    copy__94: "f32[384]" = torch.ops.aten.copy_.default(primals_292, add_164);  primals_292 = add_164 = None
    copy__95: "f32[384]" = torch.ops.aten.copy_.default(primals_293, add_165);  primals_293 = add_165 = None
    copy__96: "i64[]" = torch.ops.aten.copy_.default(primals_294, add_167);  primals_294 = add_167 = None
    copy__97: "f32[384]" = torch.ops.aten.copy_.default(primals_295, add_169);  primals_295 = add_169 = None
    copy__98: "f32[384]" = torch.ops.aten.copy_.default(primals_296, add_170);  primals_296 = add_170 = None
    copy__99: "i64[]" = torch.ops.aten.copy_.default(primals_297, add_172);  primals_297 = add_172 = None
    copy__100: "f32[64]" = torch.ops.aten.copy_.default(primals_298, add_174);  primals_298 = add_174 = None
    copy__101: "f32[64]" = torch.ops.aten.copy_.default(primals_299, add_175);  primals_299 = add_175 = None
    copy__102: "i64[]" = torch.ops.aten.copy_.default(primals_300, add_178);  primals_300 = add_178 = None
    copy__103: "f32[384]" = torch.ops.aten.copy_.default(primals_301, add_180);  primals_301 = add_180 = None
    copy__104: "f32[384]" = torch.ops.aten.copy_.default(primals_302, add_181);  primals_302 = add_181 = None
    copy__105: "i64[]" = torch.ops.aten.copy_.default(primals_303, add_183);  primals_303 = add_183 = None
    copy__106: "f32[384]" = torch.ops.aten.copy_.default(primals_304, add_185);  primals_304 = add_185 = None
    copy__107: "f32[384]" = torch.ops.aten.copy_.default(primals_305, add_186);  primals_305 = add_186 = None
    copy__108: "i64[]" = torch.ops.aten.copy_.default(primals_306, add_188);  primals_306 = add_188 = None
    copy__109: "f32[64]" = torch.ops.aten.copy_.default(primals_307, add_190);  primals_307 = add_190 = None
    copy__110: "f32[64]" = torch.ops.aten.copy_.default(primals_308, add_191);  primals_308 = add_191 = None
    copy__111: "i64[]" = torch.ops.aten.copy_.default(primals_309, add_194);  primals_309 = add_194 = None
    copy__112: "f32[384]" = torch.ops.aten.copy_.default(primals_310, add_196);  primals_310 = add_196 = None
    copy__113: "f32[384]" = torch.ops.aten.copy_.default(primals_311, add_197);  primals_311 = add_197 = None
    copy__114: "i64[]" = torch.ops.aten.copy_.default(primals_312, add_199);  primals_312 = add_199 = None
    copy__115: "f32[384]" = torch.ops.aten.copy_.default(primals_313, add_201);  primals_313 = add_201 = None
    copy__116: "f32[384]" = torch.ops.aten.copy_.default(primals_314, add_202);  primals_314 = add_202 = None
    copy__117: "i64[]" = torch.ops.aten.copy_.default(primals_315, add_204);  primals_315 = add_204 = None
    copy__118: "f32[112]" = torch.ops.aten.copy_.default(primals_316, add_206);  primals_316 = add_206 = None
    copy__119: "f32[112]" = torch.ops.aten.copy_.default(primals_317, add_207);  primals_317 = add_207 = None
    copy__120: "i64[]" = torch.ops.aten.copy_.default(primals_318, add_209);  primals_318 = add_209 = None
    copy__121: "f32[672]" = torch.ops.aten.copy_.default(primals_319, add_211);  primals_319 = add_211 = None
    copy__122: "f32[672]" = torch.ops.aten.copy_.default(primals_320, add_212);  primals_320 = add_212 = None
    copy__123: "i64[]" = torch.ops.aten.copy_.default(primals_321, add_214);  primals_321 = add_214 = None
    copy__124: "f32[672]" = torch.ops.aten.copy_.default(primals_322, add_216);  primals_322 = add_216 = None
    copy__125: "f32[672]" = torch.ops.aten.copy_.default(primals_323, add_217);  primals_323 = add_217 = None
    copy__126: "i64[]" = torch.ops.aten.copy_.default(primals_324, add_219);  primals_324 = add_219 = None
    copy__127: "f32[112]" = torch.ops.aten.copy_.default(primals_325, add_221);  primals_325 = add_221 = None
    copy__128: "f32[112]" = torch.ops.aten.copy_.default(primals_326, add_222);  primals_326 = add_222 = None
    copy__129: "i64[]" = torch.ops.aten.copy_.default(primals_327, add_225);  primals_327 = add_225 = None
    copy__130: "f32[672]" = torch.ops.aten.copy_.default(primals_328, add_227);  primals_328 = add_227 = None
    copy__131: "f32[672]" = torch.ops.aten.copy_.default(primals_329, add_228);  primals_329 = add_228 = None
    copy__132: "i64[]" = torch.ops.aten.copy_.default(primals_330, add_230);  primals_330 = add_230 = None
    copy__133: "f32[672]" = torch.ops.aten.copy_.default(primals_331, add_232);  primals_331 = add_232 = None
    copy__134: "f32[672]" = torch.ops.aten.copy_.default(primals_332, add_233);  primals_332 = add_233 = None
    copy__135: "i64[]" = torch.ops.aten.copy_.default(primals_333, add_235);  primals_333 = add_235 = None
    copy__136: "f32[112]" = torch.ops.aten.copy_.default(primals_334, add_237);  primals_334 = add_237 = None
    copy__137: "f32[112]" = torch.ops.aten.copy_.default(primals_335, add_238);  primals_335 = add_238 = None
    copy__138: "i64[]" = torch.ops.aten.copy_.default(primals_336, add_241);  primals_336 = add_241 = None
    copy__139: "f32[336]" = torch.ops.aten.copy_.default(primals_337, add_243);  primals_337 = add_243 = None
    copy__140: "f32[336]" = torch.ops.aten.copy_.default(primals_338, add_244);  primals_338 = add_244 = None
    copy__141: "i64[]" = torch.ops.aten.copy_.default(primals_339, add_246);  primals_339 = add_246 = None
    copy__142: "f32[336]" = torch.ops.aten.copy_.default(primals_340, add_248);  primals_340 = add_248 = None
    copy__143: "f32[336]" = torch.ops.aten.copy_.default(primals_341, add_249);  primals_341 = add_249 = None
    copy__144: "i64[]" = torch.ops.aten.copy_.default(primals_342, add_251);  primals_342 = add_251 = None
    copy__145: "f32[112]" = torch.ops.aten.copy_.default(primals_343, add_253);  primals_343 = add_253 = None
    copy__146: "f32[112]" = torch.ops.aten.copy_.default(primals_344, add_254);  primals_344 = add_254 = None
    copy__147: "i64[]" = torch.ops.aten.copy_.default(primals_345, add_257);  primals_345 = add_257 = None
    copy__148: "f32[672]" = torch.ops.aten.copy_.default(primals_346, add_259);  primals_346 = add_259 = None
    copy__149: "f32[672]" = torch.ops.aten.copy_.default(primals_347, add_260);  primals_347 = add_260 = None
    copy__150: "i64[]" = torch.ops.aten.copy_.default(primals_348, add_262);  primals_348 = add_262 = None
    copy__151: "f32[672]" = torch.ops.aten.copy_.default(primals_349, add_264);  primals_349 = add_264 = None
    copy__152: "f32[672]" = torch.ops.aten.copy_.default(primals_350, add_265);  primals_350 = add_265 = None
    copy__153: "i64[]" = torch.ops.aten.copy_.default(primals_351, add_267);  primals_351 = add_267 = None
    copy__154: "f32[184]" = torch.ops.aten.copy_.default(primals_352, add_269);  primals_352 = add_269 = None
    copy__155: "f32[184]" = torch.ops.aten.copy_.default(primals_353, add_270);  primals_353 = add_270 = None
    copy__156: "i64[]" = torch.ops.aten.copy_.default(primals_354, add_272);  primals_354 = add_272 = None
    copy__157: "f32[1104]" = torch.ops.aten.copy_.default(primals_355, add_274);  primals_355 = add_274 = None
    copy__158: "f32[1104]" = torch.ops.aten.copy_.default(primals_356, add_275);  primals_356 = add_275 = None
    copy__159: "i64[]" = torch.ops.aten.copy_.default(primals_357, add_277);  primals_357 = add_277 = None
    copy__160: "f32[1104]" = torch.ops.aten.copy_.default(primals_358, add_279);  primals_358 = add_279 = None
    copy__161: "f32[1104]" = torch.ops.aten.copy_.default(primals_359, add_280);  primals_359 = add_280 = None
    copy__162: "i64[]" = torch.ops.aten.copy_.default(primals_360, add_282);  primals_360 = add_282 = None
    copy__163: "f32[184]" = torch.ops.aten.copy_.default(primals_361, add_284);  primals_361 = add_284 = None
    copy__164: "f32[184]" = torch.ops.aten.copy_.default(primals_362, add_285);  primals_362 = add_285 = None
    copy__165: "i64[]" = torch.ops.aten.copy_.default(primals_363, add_288);  primals_363 = add_288 = None
    copy__166: "f32[1104]" = torch.ops.aten.copy_.default(primals_364, add_290);  primals_364 = add_290 = None
    copy__167: "f32[1104]" = torch.ops.aten.copy_.default(primals_365, add_291);  primals_365 = add_291 = None
    copy__168: "i64[]" = torch.ops.aten.copy_.default(primals_366, add_293);  primals_366 = add_293 = None
    copy__169: "f32[1104]" = torch.ops.aten.copy_.default(primals_367, add_295);  primals_367 = add_295 = None
    copy__170: "f32[1104]" = torch.ops.aten.copy_.default(primals_368, add_296);  primals_368 = add_296 = None
    copy__171: "i64[]" = torch.ops.aten.copy_.default(primals_369, add_298);  primals_369 = add_298 = None
    copy__172: "f32[184]" = torch.ops.aten.copy_.default(primals_370, add_300);  primals_370 = add_300 = None
    copy__173: "f32[184]" = torch.ops.aten.copy_.default(primals_371, add_301);  primals_371 = add_301 = None
    copy__174: "i64[]" = torch.ops.aten.copy_.default(primals_372, add_304);  primals_372 = add_304 = None
    copy__175: "f32[1104]" = torch.ops.aten.copy_.default(primals_373, add_306);  primals_373 = add_306 = None
    copy__176: "f32[1104]" = torch.ops.aten.copy_.default(primals_374, add_307);  primals_374 = add_307 = None
    copy__177: "i64[]" = torch.ops.aten.copy_.default(primals_375, add_309);  primals_375 = add_309 = None
    copy__178: "f32[1104]" = torch.ops.aten.copy_.default(primals_376, add_311);  primals_376 = add_311 = None
    copy__179: "f32[1104]" = torch.ops.aten.copy_.default(primals_377, add_312);  primals_377 = add_312 = None
    copy__180: "i64[]" = torch.ops.aten.copy_.default(primals_378, add_314);  primals_378 = add_314 = None
    copy__181: "f32[184]" = torch.ops.aten.copy_.default(primals_379, add_316);  primals_379 = add_316 = None
    copy__182: "f32[184]" = torch.ops.aten.copy_.default(primals_380, add_317);  primals_380 = add_317 = None
    copy__183: "i64[]" = torch.ops.aten.copy_.default(primals_381, add_320);  primals_381 = add_320 = None
    copy__184: "f32[1104]" = torch.ops.aten.copy_.default(primals_382, add_322);  primals_382 = add_322 = None
    copy__185: "f32[1104]" = torch.ops.aten.copy_.default(primals_383, add_323);  primals_383 = add_323 = None
    copy__186: "i64[]" = torch.ops.aten.copy_.default(primals_384, add_325);  primals_384 = add_325 = None
    copy__187: "f32[1104]" = torch.ops.aten.copy_.default(primals_385, add_327);  primals_385 = add_327 = None
    copy__188: "f32[1104]" = torch.ops.aten.copy_.default(primals_386, add_328);  primals_386 = add_328 = None
    copy__189: "i64[]" = torch.ops.aten.copy_.default(primals_387, add_330);  primals_387 = add_330 = None
    copy__190: "f32[352]" = torch.ops.aten.copy_.default(primals_388, add_332);  primals_388 = add_332 = None
    copy__191: "f32[352]" = torch.ops.aten.copy_.default(primals_389, add_333);  primals_389 = add_333 = None
    copy__192: "i64[]" = torch.ops.aten.copy_.default(primals_390, add_335);  primals_390 = add_335 = None
    copy__193: "f32[1984]" = torch.ops.aten.copy_.default(primals_391, add_337);  primals_391 = add_337 = None
    copy__194: "f32[1984]" = torch.ops.aten.copy_.default(primals_392, add_338);  primals_392 = add_338 = None
    return pytree.tree_unflatten([addmm, mul_1039, sum_130, mul_1030, sum_128, mul_1021, sum_126, mul_1012, sum_124, mul_1003, sum_122, mul_994, sum_120, mul_985, sum_118, mul_976, sum_116, mul_967, sum_114, mul_958, sum_112, mul_949, sum_110, mul_940, sum_108, mul_931, sum_106, mul_922, sum_104, mul_913, sum_102, mul_904, sum_100, mul_895, sum_98, mul_886, sum_96, mul_877, sum_94, mul_868, sum_92, mul_859, sum_90, mul_850, sum_88, mul_841, sum_86, mul_832, sum_84, mul_823, sum_82, mul_814, sum_80, mul_805, sum_78, mul_796, sum_76, mul_787, sum_74, mul_778, sum_72, mul_769, sum_70, mul_760, sum_68, mul_751, sum_66, mul_742, sum_64, mul_733, sum_62, mul_724, sum_60, mul_715, sum_58, mul_706, sum_56, mul_697, sum_54, mul_688, sum_52, mul_679, sum_50, mul_670, sum_48, mul_661, sum_46, mul_652, sum_44, mul_643, sum_42, mul_634, sum_40, mul_625, sum_38, mul_616, sum_36, mul_607, sum_34, mul_598, sum_32, mul_589, sum_30, mul_580, sum_28, mul_571, sum_26, mul_562, sum_24, mul_553, sum_22, mul_544, sum_20, mul_535, sum_18, mul_526, sum_16, mul_517, sum_14, mul_508, sum_12, mul_499, sum_10, mul_490, sum_8, mul_481, sum_6, mul_472, sum_4, mul_463, sum_2, getitem_323, getitem_320, getitem_317, getitem_314, getitem_311, getitem_308, getitem_305, getitem_302, getitem_299, getitem_296, getitem_293, getitem_290, getitem_287, getitem_284, getitem_281, getitem_278, getitem_275, getitem_272, getitem_269, getitem_266, getitem_263, getitem_260, getitem_257, getitem_254, getitem_251, getitem_248, getitem_245, getitem_242, getitem_239, getitem_236, getitem_233, getitem_230, getitem_227, getitem_224, getitem_221, getitem_218, getitem_215, getitem_212, getitem_209, getitem_206, getitem_203, getitem_200, getitem_197, getitem_194, getitem_191, getitem_188, getitem_185, getitem_182, getitem_179, getitem_176, getitem_173, getitem_170, getitem_167, getitem_164, getitem_161, getitem_158, getitem_155, getitem_152, getitem_149, getitem_146, getitem_143, getitem_140, getitem_137, getitem_134, getitem_131, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    