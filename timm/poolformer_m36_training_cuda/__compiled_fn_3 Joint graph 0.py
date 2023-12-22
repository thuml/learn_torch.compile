from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[96]"; primals_2: "f32[96]"; primals_3: "f32[96]"; primals_4: "f32[96]"; primals_5: "f32[96]"; primals_6: "f32[96]"; primals_7: "f32[96]"; primals_8: "f32[96]"; primals_9: "f32[96]"; primals_10: "f32[96]"; primals_11: "f32[96]"; primals_12: "f32[96]"; primals_13: "f32[96]"; primals_14: "f32[96]"; primals_15: "f32[96]"; primals_16: "f32[96]"; primals_17: "f32[96]"; primals_18: "f32[96]"; primals_19: "f32[96]"; primals_20: "f32[96]"; primals_21: "f32[96]"; primals_22: "f32[96]"; primals_23: "f32[96]"; primals_24: "f32[96]"; primals_25: "f32[96]"; primals_26: "f32[96]"; primals_27: "f32[96]"; primals_28: "f32[96]"; primals_29: "f32[96]"; primals_30: "f32[96]"; primals_31: "f32[96]"; primals_32: "f32[96]"; primals_33: "f32[96]"; primals_34: "f32[96]"; primals_35: "f32[96]"; primals_36: "f32[96]"; primals_37: "f32[192]"; primals_38: "f32[192]"; primals_39: "f32[192]"; primals_40: "f32[192]"; primals_41: "f32[192]"; primals_42: "f32[192]"; primals_43: "f32[192]"; primals_44: "f32[192]"; primals_45: "f32[192]"; primals_46: "f32[192]"; primals_47: "f32[192]"; primals_48: "f32[192]"; primals_49: "f32[192]"; primals_50: "f32[192]"; primals_51: "f32[192]"; primals_52: "f32[192]"; primals_53: "f32[192]"; primals_54: "f32[192]"; primals_55: "f32[192]"; primals_56: "f32[192]"; primals_57: "f32[192]"; primals_58: "f32[192]"; primals_59: "f32[192]"; primals_60: "f32[192]"; primals_61: "f32[192]"; primals_62: "f32[192]"; primals_63: "f32[192]"; primals_64: "f32[192]"; primals_65: "f32[192]"; primals_66: "f32[192]"; primals_67: "f32[192]"; primals_68: "f32[192]"; primals_69: "f32[192]"; primals_70: "f32[192]"; primals_71: "f32[192]"; primals_72: "f32[192]"; primals_73: "f32[384]"; primals_74: "f32[384]"; primals_75: "f32[384]"; primals_76: "f32[384]"; primals_77: "f32[384]"; primals_78: "f32[384]"; primals_79: "f32[384]"; primals_80: "f32[384]"; primals_81: "f32[384]"; primals_82: "f32[384]"; primals_83: "f32[384]"; primals_84: "f32[384]"; primals_85: "f32[384]"; primals_86: "f32[384]"; primals_87: "f32[384]"; primals_88: "f32[384]"; primals_89: "f32[384]"; primals_90: "f32[384]"; primals_91: "f32[384]"; primals_92: "f32[384]"; primals_93: "f32[384]"; primals_94: "f32[384]"; primals_95: "f32[384]"; primals_96: "f32[384]"; primals_97: "f32[384]"; primals_98: "f32[384]"; primals_99: "f32[384]"; primals_100: "f32[384]"; primals_101: "f32[384]"; primals_102: "f32[384]"; primals_103: "f32[384]"; primals_104: "f32[384]"; primals_105: "f32[384]"; primals_106: "f32[384]"; primals_107: "f32[384]"; primals_108: "f32[384]"; primals_109: "f32[384]"; primals_110: "f32[384]"; primals_111: "f32[384]"; primals_112: "f32[384]"; primals_113: "f32[384]"; primals_114: "f32[384]"; primals_115: "f32[384]"; primals_116: "f32[384]"; primals_117: "f32[384]"; primals_118: "f32[384]"; primals_119: "f32[384]"; primals_120: "f32[384]"; primals_121: "f32[384]"; primals_122: "f32[384]"; primals_123: "f32[384]"; primals_124: "f32[384]"; primals_125: "f32[384]"; primals_126: "f32[384]"; primals_127: "f32[384]"; primals_128: "f32[384]"; primals_129: "f32[384]"; primals_130: "f32[384]"; primals_131: "f32[384]"; primals_132: "f32[384]"; primals_133: "f32[384]"; primals_134: "f32[384]"; primals_135: "f32[384]"; primals_136: "f32[384]"; primals_137: "f32[384]"; primals_138: "f32[384]"; primals_139: "f32[384]"; primals_140: "f32[384]"; primals_141: "f32[384]"; primals_142: "f32[384]"; primals_143: "f32[384]"; primals_144: "f32[384]"; primals_145: "f32[384]"; primals_146: "f32[384]"; primals_147: "f32[384]"; primals_148: "f32[384]"; primals_149: "f32[384]"; primals_150: "f32[384]"; primals_151: "f32[384]"; primals_152: "f32[384]"; primals_153: "f32[384]"; primals_154: "f32[384]"; primals_155: "f32[384]"; primals_156: "f32[384]"; primals_157: "f32[384]"; primals_158: "f32[384]"; primals_159: "f32[384]"; primals_160: "f32[384]"; primals_161: "f32[384]"; primals_162: "f32[384]"; primals_163: "f32[384]"; primals_164: "f32[384]"; primals_165: "f32[384]"; primals_166: "f32[384]"; primals_167: "f32[384]"; primals_168: "f32[384]"; primals_169: "f32[384]"; primals_170: "f32[384]"; primals_171: "f32[384]"; primals_172: "f32[384]"; primals_173: "f32[384]"; primals_174: "f32[384]"; primals_175: "f32[384]"; primals_176: "f32[384]"; primals_177: "f32[384]"; primals_178: "f32[384]"; primals_179: "f32[384]"; primals_180: "f32[384]"; primals_181: "f32[768]"; primals_182: "f32[768]"; primals_183: "f32[768]"; primals_184: "f32[768]"; primals_185: "f32[768]"; primals_186: "f32[768]"; primals_187: "f32[768]"; primals_188: "f32[768]"; primals_189: "f32[768]"; primals_190: "f32[768]"; primals_191: "f32[768]"; primals_192: "f32[768]"; primals_193: "f32[768]"; primals_194: "f32[768]"; primals_195: "f32[768]"; primals_196: "f32[768]"; primals_197: "f32[768]"; primals_198: "f32[768]"; primals_199: "f32[768]"; primals_200: "f32[768]"; primals_201: "f32[768]"; primals_202: "f32[768]"; primals_203: "f32[768]"; primals_204: "f32[768]"; primals_205: "f32[768]"; primals_206: "f32[768]"; primals_207: "f32[768]"; primals_208: "f32[768]"; primals_209: "f32[768]"; primals_210: "f32[768]"; primals_211: "f32[768]"; primals_212: "f32[768]"; primals_213: "f32[768]"; primals_214: "f32[768]"; primals_215: "f32[768]"; primals_216: "f32[768]"; primals_217: "f32[768]"; primals_218: "f32[768]"; primals_219: "f32[96, 3, 7, 7]"; primals_220: "f32[96]"; primals_221: "f32[384, 96, 1, 1]"; primals_222: "f32[384]"; primals_223: "f32[96, 384, 1, 1]"; primals_224: "f32[96]"; primals_225: "f32[384, 96, 1, 1]"; primals_226: "f32[384]"; primals_227: "f32[96, 384, 1, 1]"; primals_228: "f32[96]"; primals_229: "f32[384, 96, 1, 1]"; primals_230: "f32[384]"; primals_231: "f32[96, 384, 1, 1]"; primals_232: "f32[96]"; primals_233: "f32[384, 96, 1, 1]"; primals_234: "f32[384]"; primals_235: "f32[96, 384, 1, 1]"; primals_236: "f32[96]"; primals_237: "f32[384, 96, 1, 1]"; primals_238: "f32[384]"; primals_239: "f32[96, 384, 1, 1]"; primals_240: "f32[96]"; primals_241: "f32[384, 96, 1, 1]"; primals_242: "f32[384]"; primals_243: "f32[96, 384, 1, 1]"; primals_244: "f32[96]"; primals_245: "f32[192, 96, 3, 3]"; primals_246: "f32[192]"; primals_247: "f32[768, 192, 1, 1]"; primals_248: "f32[768]"; primals_249: "f32[192, 768, 1, 1]"; primals_250: "f32[192]"; primals_251: "f32[768, 192, 1, 1]"; primals_252: "f32[768]"; primals_253: "f32[192, 768, 1, 1]"; primals_254: "f32[192]"; primals_255: "f32[768, 192, 1, 1]"; primals_256: "f32[768]"; primals_257: "f32[192, 768, 1, 1]"; primals_258: "f32[192]"; primals_259: "f32[768, 192, 1, 1]"; primals_260: "f32[768]"; primals_261: "f32[192, 768, 1, 1]"; primals_262: "f32[192]"; primals_263: "f32[768, 192, 1, 1]"; primals_264: "f32[768]"; primals_265: "f32[192, 768, 1, 1]"; primals_266: "f32[192]"; primals_267: "f32[768, 192, 1, 1]"; primals_268: "f32[768]"; primals_269: "f32[192, 768, 1, 1]"; primals_270: "f32[192]"; primals_271: "f32[384, 192, 3, 3]"; primals_272: "f32[384]"; primals_273: "f32[1536, 384, 1, 1]"; primals_274: "f32[1536]"; primals_275: "f32[384, 1536, 1, 1]"; primals_276: "f32[384]"; primals_277: "f32[1536, 384, 1, 1]"; primals_278: "f32[1536]"; primals_279: "f32[384, 1536, 1, 1]"; primals_280: "f32[384]"; primals_281: "f32[1536, 384, 1, 1]"; primals_282: "f32[1536]"; primals_283: "f32[384, 1536, 1, 1]"; primals_284: "f32[384]"; primals_285: "f32[1536, 384, 1, 1]"; primals_286: "f32[1536]"; primals_287: "f32[384, 1536, 1, 1]"; primals_288: "f32[384]"; primals_289: "f32[1536, 384, 1, 1]"; primals_290: "f32[1536]"; primals_291: "f32[384, 1536, 1, 1]"; primals_292: "f32[384]"; primals_293: "f32[1536, 384, 1, 1]"; primals_294: "f32[1536]"; primals_295: "f32[384, 1536, 1, 1]"; primals_296: "f32[384]"; primals_297: "f32[1536, 384, 1, 1]"; primals_298: "f32[1536]"; primals_299: "f32[384, 1536, 1, 1]"; primals_300: "f32[384]"; primals_301: "f32[1536, 384, 1, 1]"; primals_302: "f32[1536]"; primals_303: "f32[384, 1536, 1, 1]"; primals_304: "f32[384]"; primals_305: "f32[1536, 384, 1, 1]"; primals_306: "f32[1536]"; primals_307: "f32[384, 1536, 1, 1]"; primals_308: "f32[384]"; primals_309: "f32[1536, 384, 1, 1]"; primals_310: "f32[1536]"; primals_311: "f32[384, 1536, 1, 1]"; primals_312: "f32[384]"; primals_313: "f32[1536, 384, 1, 1]"; primals_314: "f32[1536]"; primals_315: "f32[384, 1536, 1, 1]"; primals_316: "f32[384]"; primals_317: "f32[1536, 384, 1, 1]"; primals_318: "f32[1536]"; primals_319: "f32[384, 1536, 1, 1]"; primals_320: "f32[384]"; primals_321: "f32[1536, 384, 1, 1]"; primals_322: "f32[1536]"; primals_323: "f32[384, 1536, 1, 1]"; primals_324: "f32[384]"; primals_325: "f32[1536, 384, 1, 1]"; primals_326: "f32[1536]"; primals_327: "f32[384, 1536, 1, 1]"; primals_328: "f32[384]"; primals_329: "f32[1536, 384, 1, 1]"; primals_330: "f32[1536]"; primals_331: "f32[384, 1536, 1, 1]"; primals_332: "f32[384]"; primals_333: "f32[1536, 384, 1, 1]"; primals_334: "f32[1536]"; primals_335: "f32[384, 1536, 1, 1]"; primals_336: "f32[384]"; primals_337: "f32[1536, 384, 1, 1]"; primals_338: "f32[1536]"; primals_339: "f32[384, 1536, 1, 1]"; primals_340: "f32[384]"; primals_341: "f32[1536, 384, 1, 1]"; primals_342: "f32[1536]"; primals_343: "f32[384, 1536, 1, 1]"; primals_344: "f32[384]"; primals_345: "f32[768, 384, 3, 3]"; primals_346: "f32[768]"; primals_347: "f32[3072, 768, 1, 1]"; primals_348: "f32[3072]"; primals_349: "f32[768, 3072, 1, 1]"; primals_350: "f32[768]"; primals_351: "f32[3072, 768, 1, 1]"; primals_352: "f32[3072]"; primals_353: "f32[768, 3072, 1, 1]"; primals_354: "f32[768]"; primals_355: "f32[3072, 768, 1, 1]"; primals_356: "f32[3072]"; primals_357: "f32[768, 3072, 1, 1]"; primals_358: "f32[768]"; primals_359: "f32[3072, 768, 1, 1]"; primals_360: "f32[3072]"; primals_361: "f32[768, 3072, 1, 1]"; primals_362: "f32[768]"; primals_363: "f32[3072, 768, 1, 1]"; primals_364: "f32[3072]"; primals_365: "f32[768, 3072, 1, 1]"; primals_366: "f32[768]"; primals_367: "f32[3072, 768, 1, 1]"; primals_368: "f32[3072]"; primals_369: "f32[768, 3072, 1, 1]"; primals_370: "f32[768]"; primals_371: "f32[1000, 768]"; primals_372: "f32[1000]"; primals_373: "f32[8, 3, 224, 224]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:72, code: x = self.conv(x)
    convolution: "f32[8, 96, 56, 56]" = torch.ops.aten.convolution.default(primals_373, primals_219, primals_220, [4, 4], [2, 2], [1, 1], False, [0, 0], 1);  primals_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(convolution, [8, 1, 96, 3136])
    var_mean = torch.ops.aten.var_mean.correction(view, [2, 3], correction = 0, keepdim = True)
    getitem: "f32[8, 1, 1, 1]" = var_mean[0]
    getitem_1: "f32[8, 1, 1, 1]" = var_mean[1];  var_mean = None
    add: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
    sub: "f32[8, 1, 96, 3136]" = torch.ops.aten.sub.Tensor(view, getitem_1);  view = None
    mul: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    view_1: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(mul, [8, 96, 56, 56]);  mul = None
    unsqueeze: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_2, 0);  primals_2 = None
    unsqueeze_1: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
    unsqueeze_2: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
    unsqueeze_3: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_1, 0)
    unsqueeze_4: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
    unsqueeze_5: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
    mul_1: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(view_1, unsqueeze_5);  view_1 = unsqueeze_5 = None
    add_1: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_2);  mul_1 = unsqueeze_2 = None
    squeeze: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_1, [2, 3]);  getitem_1 = None
    squeeze_1: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt, [2, 3]);  rsqrt = None
    alias: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze);  squeeze = None
    alias_1: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_1);  squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d: "f32[8, 96, 56, 56]" = torch.ops.aten.avg_pool2d.default(add_1, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_1: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(avg_pool2d, add_1);  avg_pool2d = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_2: "f32[96, 1, 1]" = torch.ops.aten.view.default(primals_3, [96, 1, 1]);  primals_3 = None
    mul_2: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_1, view_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_2: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(convolution, mul_2);  mul_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_3: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_2, [8, 1, 96, 3136])
    var_mean_1 = torch.ops.aten.var_mean.correction(view_3, [2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[8, 1, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[8, 1, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_3: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt_1: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
    sub_2: "f32[8, 1, 96, 3136]" = torch.ops.aten.sub.Tensor(view_3, getitem_3);  view_3 = None
    mul_3: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
    view_4: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(mul_3, [8, 96, 56, 56]);  mul_3 = None
    unsqueeze_6: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_5, 0);  primals_5 = None
    unsqueeze_7: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 2);  unsqueeze_6 = None
    unsqueeze_8: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
    unsqueeze_9: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_4, 0)
    unsqueeze_10: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 2);  unsqueeze_9 = None
    unsqueeze_11: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 3);  unsqueeze_10 = None
    mul_4: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(view_4, unsqueeze_11);  view_4 = unsqueeze_11 = None
    add_4: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_4, unsqueeze_8);  mul_4 = unsqueeze_8 = None
    squeeze_2: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_3, [2, 3]);  getitem_3 = None
    squeeze_3: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_1, [2, 3]);  rsqrt_1 = None
    alias_2: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_2);  squeeze_2 = None
    alias_3: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_3);  squeeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_1: "f32[8, 384, 56, 56]" = torch.ops.aten.convolution.default(add_4, primals_221, primals_222, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_5: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_1, 0.5)
    mul_6: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_1, 0.7071067811865476)
    erf: "f32[8, 384, 56, 56]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
    add_5: "f32[8, 384, 56, 56]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_7: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(mul_5, add_5);  mul_5 = add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone: "f32[8, 384, 56, 56]" = torch.ops.aten.clone.default(mul_7);  mul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_2: "f32[8, 96, 56, 56]" = torch.ops.aten.convolution.default(clone, primals_223, primals_224, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_1: "f32[8, 96, 56, 56]" = torch.ops.aten.clone.default(convolution_2);  convolution_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_5: "f32[96, 1, 1]" = torch.ops.aten.view.default(primals_6, [96, 1, 1]);  primals_6 = None
    mul_8: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(clone_1, view_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_6: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_2, mul_8);  mul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_6: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_6, [8, 1, 96, 3136])
    var_mean_2 = torch.ops.aten.var_mean.correction(view_6, [2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[8, 1, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[8, 1, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_7: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_2: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    sub_3: "f32[8, 1, 96, 3136]" = torch.ops.aten.sub.Tensor(view_6, getitem_5);  view_6 = None
    mul_9: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
    view_7: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(mul_9, [8, 96, 56, 56]);  mul_9 = None
    unsqueeze_12: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_8, 0);  primals_8 = None
    unsqueeze_13: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, 2);  unsqueeze_12 = None
    unsqueeze_14: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 3);  unsqueeze_13 = None
    unsqueeze_15: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_7, 0)
    unsqueeze_16: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_15, 2);  unsqueeze_15 = None
    unsqueeze_17: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, 3);  unsqueeze_16 = None
    mul_10: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(view_7, unsqueeze_17);  view_7 = unsqueeze_17 = None
    add_8: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_10, unsqueeze_14);  mul_10 = unsqueeze_14 = None
    squeeze_4: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_5, [2, 3]);  getitem_5 = None
    squeeze_5: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_2, [2, 3]);  rsqrt_2 = None
    alias_4: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_4);  squeeze_4 = None
    alias_5: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_5);  squeeze_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_1: "f32[8, 96, 56, 56]" = torch.ops.aten.avg_pool2d.default(add_8, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_4: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(avg_pool2d_1, add_8);  avg_pool2d_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_8: "f32[96, 1, 1]" = torch.ops.aten.view.default(primals_9, [96, 1, 1]);  primals_9 = None
    mul_11: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, view_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_9: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_6, mul_11);  mul_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_9: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_9, [8, 1, 96, 3136])
    var_mean_3 = torch.ops.aten.var_mean.correction(view_9, [2, 3], correction = 0, keepdim = True)
    getitem_6: "f32[8, 1, 1, 1]" = var_mean_3[0]
    getitem_7: "f32[8, 1, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_10: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
    rsqrt_3: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
    sub_5: "f32[8, 1, 96, 3136]" = torch.ops.aten.sub.Tensor(view_9, getitem_7);  view_9 = None
    mul_12: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
    view_10: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(mul_12, [8, 96, 56, 56]);  mul_12 = None
    unsqueeze_18: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_11, 0);  primals_11 = None
    unsqueeze_19: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, 2);  unsqueeze_18 = None
    unsqueeze_20: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_19, 3);  unsqueeze_19 = None
    unsqueeze_21: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_10, 0)
    unsqueeze_22: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_21, 2);  unsqueeze_21 = None
    unsqueeze_23: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, 3);  unsqueeze_22 = None
    mul_13: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(view_10, unsqueeze_23);  view_10 = unsqueeze_23 = None
    add_11: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_20);  mul_13 = unsqueeze_20 = None
    squeeze_6: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_7, [2, 3]);  getitem_7 = None
    squeeze_7: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_3, [2, 3]);  rsqrt_3 = None
    alias_6: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_6);  squeeze_6 = None
    alias_7: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_7);  squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_3: "f32[8, 384, 56, 56]" = torch.ops.aten.convolution.default(add_11, primals_225, primals_226, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_14: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_3, 0.5)
    mul_15: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_3, 0.7071067811865476)
    erf_1: "f32[8, 384, 56, 56]" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
    add_12: "f32[8, 384, 56, 56]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_16: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(mul_14, add_12);  mul_14 = add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_2: "f32[8, 384, 56, 56]" = torch.ops.aten.clone.default(mul_16);  mul_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_4: "f32[8, 96, 56, 56]" = torch.ops.aten.convolution.default(clone_2, primals_227, primals_228, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_3: "f32[8, 96, 56, 56]" = torch.ops.aten.clone.default(convolution_4);  convolution_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_11: "f32[96, 1, 1]" = torch.ops.aten.view.default(primals_12, [96, 1, 1]);  primals_12 = None
    mul_17: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(clone_3, view_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_13: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_9, mul_17);  mul_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_12: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_13, [8, 1, 96, 3136])
    var_mean_4 = torch.ops.aten.var_mean.correction(view_12, [2, 3], correction = 0, keepdim = True)
    getitem_8: "f32[8, 1, 1, 1]" = var_mean_4[0]
    getitem_9: "f32[8, 1, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_14: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_4: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    sub_6: "f32[8, 1, 96, 3136]" = torch.ops.aten.sub.Tensor(view_12, getitem_9);  view_12 = None
    mul_18: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
    view_13: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(mul_18, [8, 96, 56, 56]);  mul_18 = None
    unsqueeze_24: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_14, 0);  primals_14 = None
    unsqueeze_25: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, 2);  unsqueeze_24 = None
    unsqueeze_26: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_25, 3);  unsqueeze_25 = None
    unsqueeze_27: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_13, 0)
    unsqueeze_28: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_27, 2);  unsqueeze_27 = None
    unsqueeze_29: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, 3);  unsqueeze_28 = None
    mul_19: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(view_13, unsqueeze_29);  view_13 = unsqueeze_29 = None
    add_15: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_19, unsqueeze_26);  mul_19 = unsqueeze_26 = None
    squeeze_8: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_9, [2, 3]);  getitem_9 = None
    squeeze_9: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_4, [2, 3]);  rsqrt_4 = None
    alias_8: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_8);  squeeze_8 = None
    alias_9: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_9);  squeeze_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_2: "f32[8, 96, 56, 56]" = torch.ops.aten.avg_pool2d.default(add_15, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_7: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(avg_pool2d_2, add_15);  avg_pool2d_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_14: "f32[96, 1, 1]" = torch.ops.aten.view.default(primals_15, [96, 1, 1]);  primals_15 = None
    mul_20: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, view_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_16: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_13, mul_20);  mul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_15: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_16, [8, 1, 96, 3136])
    var_mean_5 = torch.ops.aten.var_mean.correction(view_15, [2, 3], correction = 0, keepdim = True)
    getitem_10: "f32[8, 1, 1, 1]" = var_mean_5[0]
    getitem_11: "f32[8, 1, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_17: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_5: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
    sub_8: "f32[8, 1, 96, 3136]" = torch.ops.aten.sub.Tensor(view_15, getitem_11);  view_15 = None
    mul_21: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = None
    view_16: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(mul_21, [8, 96, 56, 56]);  mul_21 = None
    unsqueeze_30: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_17, 0);  primals_17 = None
    unsqueeze_31: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, 2);  unsqueeze_30 = None
    unsqueeze_32: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_31, 3);  unsqueeze_31 = None
    unsqueeze_33: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_16, 0)
    unsqueeze_34: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_33, 2);  unsqueeze_33 = None
    unsqueeze_35: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, 3);  unsqueeze_34 = None
    mul_22: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(view_16, unsqueeze_35);  view_16 = unsqueeze_35 = None
    add_18: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_22, unsqueeze_32);  mul_22 = unsqueeze_32 = None
    squeeze_10: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_11, [2, 3]);  getitem_11 = None
    squeeze_11: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_5, [2, 3]);  rsqrt_5 = None
    alias_10: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_10);  squeeze_10 = None
    alias_11: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_11);  squeeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_5: "f32[8, 384, 56, 56]" = torch.ops.aten.convolution.default(add_18, primals_229, primals_230, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_23: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_5, 0.5)
    mul_24: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_5, 0.7071067811865476)
    erf_2: "f32[8, 384, 56, 56]" = torch.ops.aten.erf.default(mul_24);  mul_24 = None
    add_19: "f32[8, 384, 56, 56]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_25: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(mul_23, add_19);  mul_23 = add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_4: "f32[8, 384, 56, 56]" = torch.ops.aten.clone.default(mul_25);  mul_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_6: "f32[8, 96, 56, 56]" = torch.ops.aten.convolution.default(clone_4, primals_231, primals_232, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_5: "f32[8, 96, 56, 56]" = torch.ops.aten.clone.default(convolution_6);  convolution_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_17: "f32[96, 1, 1]" = torch.ops.aten.view.default(primals_18, [96, 1, 1]);  primals_18 = None
    mul_26: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(clone_5, view_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_20: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_16, mul_26);  mul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_18: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_20, [8, 1, 96, 3136])
    var_mean_6 = torch.ops.aten.var_mean.correction(view_18, [2, 3], correction = 0, keepdim = True)
    getitem_12: "f32[8, 1, 1, 1]" = var_mean_6[0]
    getitem_13: "f32[8, 1, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_21: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_6: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_9: "f32[8, 1, 96, 3136]" = torch.ops.aten.sub.Tensor(view_18, getitem_13);  view_18 = None
    mul_27: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = None
    view_19: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(mul_27, [8, 96, 56, 56]);  mul_27 = None
    unsqueeze_36: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_20, 0);  primals_20 = None
    unsqueeze_37: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, 2);  unsqueeze_36 = None
    unsqueeze_38: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_37, 3);  unsqueeze_37 = None
    unsqueeze_39: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_19, 0)
    unsqueeze_40: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_39, 2);  unsqueeze_39 = None
    unsqueeze_41: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, 3);  unsqueeze_40 = None
    mul_28: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(view_19, unsqueeze_41);  view_19 = unsqueeze_41 = None
    add_22: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_28, unsqueeze_38);  mul_28 = unsqueeze_38 = None
    squeeze_12: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_13, [2, 3]);  getitem_13 = None
    squeeze_13: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_6, [2, 3]);  rsqrt_6 = None
    alias_12: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_12);  squeeze_12 = None
    alias_13: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_13);  squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_3: "f32[8, 96, 56, 56]" = torch.ops.aten.avg_pool2d.default(add_22, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_10: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(avg_pool2d_3, add_22);  avg_pool2d_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_20: "f32[96, 1, 1]" = torch.ops.aten.view.default(primals_21, [96, 1, 1]);  primals_21 = None
    mul_29: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_10, view_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_23: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_20, mul_29);  mul_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_21: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_23, [8, 1, 96, 3136])
    var_mean_7 = torch.ops.aten.var_mean.correction(view_21, [2, 3], correction = 0, keepdim = True)
    getitem_14: "f32[8, 1, 1, 1]" = var_mean_7[0]
    getitem_15: "f32[8, 1, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_24: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_7: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    sub_11: "f32[8, 1, 96, 3136]" = torch.ops.aten.sub.Tensor(view_21, getitem_15);  view_21 = None
    mul_30: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = None
    view_22: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(mul_30, [8, 96, 56, 56]);  mul_30 = None
    unsqueeze_42: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_23, 0);  primals_23 = None
    unsqueeze_43: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, 2);  unsqueeze_42 = None
    unsqueeze_44: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_43, 3);  unsqueeze_43 = None
    unsqueeze_45: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_22, 0)
    unsqueeze_46: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_45, 2);  unsqueeze_45 = None
    unsqueeze_47: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, 3);  unsqueeze_46 = None
    mul_31: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(view_22, unsqueeze_47);  view_22 = unsqueeze_47 = None
    add_25: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_31, unsqueeze_44);  mul_31 = unsqueeze_44 = None
    squeeze_14: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_15, [2, 3]);  getitem_15 = None
    squeeze_15: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_7, [2, 3]);  rsqrt_7 = None
    alias_14: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_14);  squeeze_14 = None
    alias_15: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_15);  squeeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_7: "f32[8, 384, 56, 56]" = torch.ops.aten.convolution.default(add_25, primals_233, primals_234, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_32: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_7, 0.5)
    mul_33: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_7, 0.7071067811865476)
    erf_3: "f32[8, 384, 56, 56]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
    add_26: "f32[8, 384, 56, 56]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_34: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(mul_32, add_26);  mul_32 = add_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_6: "f32[8, 384, 56, 56]" = torch.ops.aten.clone.default(mul_34);  mul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_8: "f32[8, 96, 56, 56]" = torch.ops.aten.convolution.default(clone_6, primals_235, primals_236, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_7: "f32[8, 96, 56, 56]" = torch.ops.aten.clone.default(convolution_8);  convolution_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_23: "f32[96, 1, 1]" = torch.ops.aten.view.default(primals_24, [96, 1, 1]);  primals_24 = None
    mul_35: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(clone_7, view_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_27: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_23, mul_35);  mul_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_24: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_27, [8, 1, 96, 3136])
    var_mean_8 = torch.ops.aten.var_mean.correction(view_24, [2, 3], correction = 0, keepdim = True)
    getitem_16: "f32[8, 1, 1, 1]" = var_mean_8[0]
    getitem_17: "f32[8, 1, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_28: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_8: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_12: "f32[8, 1, 96, 3136]" = torch.ops.aten.sub.Tensor(view_24, getitem_17);  view_24 = None
    mul_36: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = None
    view_25: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(mul_36, [8, 96, 56, 56]);  mul_36 = None
    unsqueeze_48: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_26, 0);  primals_26 = None
    unsqueeze_49: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, 2);  unsqueeze_48 = None
    unsqueeze_50: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_49, 3);  unsqueeze_49 = None
    unsqueeze_51: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_25, 0)
    unsqueeze_52: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_51, 2);  unsqueeze_51 = None
    unsqueeze_53: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, 3);  unsqueeze_52 = None
    mul_37: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(view_25, unsqueeze_53);  view_25 = unsqueeze_53 = None
    add_29: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_37, unsqueeze_50);  mul_37 = unsqueeze_50 = None
    squeeze_16: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_17, [2, 3]);  getitem_17 = None
    squeeze_17: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_8, [2, 3]);  rsqrt_8 = None
    alias_16: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_16);  squeeze_16 = None
    alias_17: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_17);  squeeze_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_4: "f32[8, 96, 56, 56]" = torch.ops.aten.avg_pool2d.default(add_29, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_13: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(avg_pool2d_4, add_29);  avg_pool2d_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_26: "f32[96, 1, 1]" = torch.ops.aten.view.default(primals_27, [96, 1, 1]);  primals_27 = None
    mul_38: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_13, view_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_30: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_27, mul_38);  mul_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_27: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_30, [8, 1, 96, 3136])
    var_mean_9 = torch.ops.aten.var_mean.correction(view_27, [2, 3], correction = 0, keepdim = True)
    getitem_18: "f32[8, 1, 1, 1]" = var_mean_9[0]
    getitem_19: "f32[8, 1, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_31: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
    rsqrt_9: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_14: "f32[8, 1, 96, 3136]" = torch.ops.aten.sub.Tensor(view_27, getitem_19);  view_27 = None
    mul_39: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = None
    view_28: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(mul_39, [8, 96, 56, 56]);  mul_39 = None
    unsqueeze_54: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_29, 0);  primals_29 = None
    unsqueeze_55: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, 2);  unsqueeze_54 = None
    unsqueeze_56: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_55, 3);  unsqueeze_55 = None
    unsqueeze_57: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_28, 0)
    unsqueeze_58: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_57, 2);  unsqueeze_57 = None
    unsqueeze_59: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, 3);  unsqueeze_58 = None
    mul_40: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(view_28, unsqueeze_59);  view_28 = unsqueeze_59 = None
    add_32: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_40, unsqueeze_56);  mul_40 = unsqueeze_56 = None
    squeeze_18: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_19, [2, 3]);  getitem_19 = None
    squeeze_19: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_9, [2, 3]);  rsqrt_9 = None
    alias_18: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_18);  squeeze_18 = None
    alias_19: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_19);  squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_9: "f32[8, 384, 56, 56]" = torch.ops.aten.convolution.default(add_32, primals_237, primals_238, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_41: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_9, 0.5)
    mul_42: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_9, 0.7071067811865476)
    erf_4: "f32[8, 384, 56, 56]" = torch.ops.aten.erf.default(mul_42);  mul_42 = None
    add_33: "f32[8, 384, 56, 56]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_43: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(mul_41, add_33);  mul_41 = add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_8: "f32[8, 384, 56, 56]" = torch.ops.aten.clone.default(mul_43);  mul_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_10: "f32[8, 96, 56, 56]" = torch.ops.aten.convolution.default(clone_8, primals_239, primals_240, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_9: "f32[8, 96, 56, 56]" = torch.ops.aten.clone.default(convolution_10);  convolution_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_29: "f32[96, 1, 1]" = torch.ops.aten.view.default(primals_30, [96, 1, 1]);  primals_30 = None
    mul_44: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(clone_9, view_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_34: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_30, mul_44);  mul_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_30: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_34, [8, 1, 96, 3136])
    var_mean_10 = torch.ops.aten.var_mean.correction(view_30, [2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[8, 1, 1, 1]" = var_mean_10[0]
    getitem_21: "f32[8, 1, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_35: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
    rsqrt_10: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
    sub_15: "f32[8, 1, 96, 3136]" = torch.ops.aten.sub.Tensor(view_30, getitem_21);  view_30 = None
    mul_45: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = None
    view_31: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(mul_45, [8, 96, 56, 56]);  mul_45 = None
    unsqueeze_60: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_32, 0);  primals_32 = None
    unsqueeze_61: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, 2);  unsqueeze_60 = None
    unsqueeze_62: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_61, 3);  unsqueeze_61 = None
    unsqueeze_63: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_31, 0)
    unsqueeze_64: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_63, 2);  unsqueeze_63 = None
    unsqueeze_65: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, 3);  unsqueeze_64 = None
    mul_46: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(view_31, unsqueeze_65);  view_31 = unsqueeze_65 = None
    add_36: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_46, unsqueeze_62);  mul_46 = unsqueeze_62 = None
    squeeze_20: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_21, [2, 3]);  getitem_21 = None
    squeeze_21: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_10, [2, 3]);  rsqrt_10 = None
    alias_20: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_20);  squeeze_20 = None
    alias_21: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_21);  squeeze_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_5: "f32[8, 96, 56, 56]" = torch.ops.aten.avg_pool2d.default(add_36, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_16: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(avg_pool2d_5, add_36);  avg_pool2d_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_32: "f32[96, 1, 1]" = torch.ops.aten.view.default(primals_33, [96, 1, 1]);  primals_33 = None
    mul_47: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_16, view_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_37: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_34, mul_47);  mul_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_33: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_37, [8, 1, 96, 3136])
    var_mean_11 = torch.ops.aten.var_mean.correction(view_33, [2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[8, 1, 1, 1]" = var_mean_11[0]
    getitem_23: "f32[8, 1, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_38: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_11: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_17: "f32[8, 1, 96, 3136]" = torch.ops.aten.sub.Tensor(view_33, getitem_23);  view_33 = None
    mul_48: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = None
    view_34: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(mul_48, [8, 96, 56, 56]);  mul_48 = None
    unsqueeze_66: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_35, 0);  primals_35 = None
    unsqueeze_67: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, 2);  unsqueeze_66 = None
    unsqueeze_68: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_67, 3);  unsqueeze_67 = None
    unsqueeze_69: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_34, 0)
    unsqueeze_70: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_69, 2);  unsqueeze_69 = None
    unsqueeze_71: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, 3);  unsqueeze_70 = None
    mul_49: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(view_34, unsqueeze_71);  view_34 = unsqueeze_71 = None
    add_39: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_49, unsqueeze_68);  mul_49 = unsqueeze_68 = None
    squeeze_22: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_23, [2, 3]);  getitem_23 = None
    squeeze_23: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_11, [2, 3]);  rsqrt_11 = None
    alias_22: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_22);  squeeze_22 = None
    alias_23: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_23);  squeeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_11: "f32[8, 384, 56, 56]" = torch.ops.aten.convolution.default(add_39, primals_241, primals_242, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_50: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_11, 0.5)
    mul_51: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_11, 0.7071067811865476)
    erf_5: "f32[8, 384, 56, 56]" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
    add_40: "f32[8, 384, 56, 56]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_52: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(mul_50, add_40);  mul_50 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_10: "f32[8, 384, 56, 56]" = torch.ops.aten.clone.default(mul_52);  mul_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_12: "f32[8, 96, 56, 56]" = torch.ops.aten.convolution.default(clone_10, primals_243, primals_244, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_11: "f32[8, 96, 56, 56]" = torch.ops.aten.clone.default(convolution_12);  convolution_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_35: "f32[96, 1, 1]" = torch.ops.aten.view.default(primals_36, [96, 1, 1]);  primals_36 = None
    mul_53: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(clone_11, view_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_41: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_37, mul_53);  mul_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:103, code: x = self.conv(x)
    convolution_13: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(add_41, primals_245, primals_246, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  primals_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_36: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(convolution_13, [8, 1, 192, 784])
    var_mean_12 = torch.ops.aten.var_mean.correction(view_36, [2, 3], correction = 0, keepdim = True)
    getitem_24: "f32[8, 1, 1, 1]" = var_mean_12[0]
    getitem_25: "f32[8, 1, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_42: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_12: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_18: "f32[8, 1, 192, 784]" = torch.ops.aten.sub.Tensor(view_36, getitem_25);  view_36 = None
    mul_54: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = None
    view_37: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(mul_54, [8, 192, 28, 28]);  mul_54 = None
    unsqueeze_72: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_38, 0);  primals_38 = None
    unsqueeze_73: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, 2);  unsqueeze_72 = None
    unsqueeze_74: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_73, 3);  unsqueeze_73 = None
    unsqueeze_75: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_37, 0)
    unsqueeze_76: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_75, 2);  unsqueeze_75 = None
    unsqueeze_77: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, 3);  unsqueeze_76 = None
    mul_55: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(view_37, unsqueeze_77);  view_37 = unsqueeze_77 = None
    add_43: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_74);  mul_55 = unsqueeze_74 = None
    squeeze_24: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_25, [2, 3]);  getitem_25 = None
    squeeze_25: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_12, [2, 3]);  rsqrt_12 = None
    alias_24: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_24);  squeeze_24 = None
    alias_25: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_25);  squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_6: "f32[8, 192, 28, 28]" = torch.ops.aten.avg_pool2d.default(add_43, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_19: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(avg_pool2d_6, add_43);  avg_pool2d_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_38: "f32[192, 1, 1]" = torch.ops.aten.view.default(primals_39, [192, 1, 1]);  primals_39 = None
    mul_56: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_19, view_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_44: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(convolution_13, mul_56);  mul_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_39: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_44, [8, 1, 192, 784])
    var_mean_13 = torch.ops.aten.var_mean.correction(view_39, [2, 3], correction = 0, keepdim = True)
    getitem_26: "f32[8, 1, 1, 1]" = var_mean_13[0]
    getitem_27: "f32[8, 1, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_45: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
    rsqrt_13: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
    sub_20: "f32[8, 1, 192, 784]" = torch.ops.aten.sub.Tensor(view_39, getitem_27);  view_39 = None
    mul_57: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = None
    view_40: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(mul_57, [8, 192, 28, 28]);  mul_57 = None
    unsqueeze_78: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_41, 0);  primals_41 = None
    unsqueeze_79: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, 2);  unsqueeze_78 = None
    unsqueeze_80: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_79, 3);  unsqueeze_79 = None
    unsqueeze_81: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_40, 0)
    unsqueeze_82: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_81, 2);  unsqueeze_81 = None
    unsqueeze_83: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, 3);  unsqueeze_82 = None
    mul_58: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(view_40, unsqueeze_83);  view_40 = unsqueeze_83 = None
    add_46: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_58, unsqueeze_80);  mul_58 = unsqueeze_80 = None
    squeeze_26: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_27, [2, 3]);  getitem_27 = None
    squeeze_27: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_13, [2, 3]);  rsqrt_13 = None
    alias_26: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_26);  squeeze_26 = None
    alias_27: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_27);  squeeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_14: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_46, primals_247, primals_248, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_59: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_14, 0.5)
    mul_60: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_14, 0.7071067811865476)
    erf_6: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_60);  mul_60 = None
    add_47: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_61: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_59, add_47);  mul_59 = add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_12: "f32[8, 768, 28, 28]" = torch.ops.aten.clone.default(mul_61);  mul_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_15: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(clone_12, primals_249, primals_250, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_13: "f32[8, 192, 28, 28]" = torch.ops.aten.clone.default(convolution_15);  convolution_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_41: "f32[192, 1, 1]" = torch.ops.aten.view.default(primals_42, [192, 1, 1]);  primals_42 = None
    mul_62: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(clone_13, view_41)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_48: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_44, mul_62);  mul_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_42: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_48, [8, 1, 192, 784])
    var_mean_14 = torch.ops.aten.var_mean.correction(view_42, [2, 3], correction = 0, keepdim = True)
    getitem_28: "f32[8, 1, 1, 1]" = var_mean_14[0]
    getitem_29: "f32[8, 1, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_49: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_14: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_21: "f32[8, 1, 192, 784]" = torch.ops.aten.sub.Tensor(view_42, getitem_29);  view_42 = None
    mul_63: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = None
    view_43: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(mul_63, [8, 192, 28, 28]);  mul_63 = None
    unsqueeze_84: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_44, 0);  primals_44 = None
    unsqueeze_85: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, 2);  unsqueeze_84 = None
    unsqueeze_86: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_85, 3);  unsqueeze_85 = None
    unsqueeze_87: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_43, 0)
    unsqueeze_88: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_87, 2);  unsqueeze_87 = None
    unsqueeze_89: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, 3);  unsqueeze_88 = None
    mul_64: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(view_43, unsqueeze_89);  view_43 = unsqueeze_89 = None
    add_50: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_64, unsqueeze_86);  mul_64 = unsqueeze_86 = None
    squeeze_28: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_29, [2, 3]);  getitem_29 = None
    squeeze_29: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_14, [2, 3]);  rsqrt_14 = None
    alias_28: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_28);  squeeze_28 = None
    alias_29: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_29);  squeeze_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_7: "f32[8, 192, 28, 28]" = torch.ops.aten.avg_pool2d.default(add_50, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_22: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(avg_pool2d_7, add_50);  avg_pool2d_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_44: "f32[192, 1, 1]" = torch.ops.aten.view.default(primals_45, [192, 1, 1]);  primals_45 = None
    mul_65: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_22, view_44)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_51: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_48, mul_65);  mul_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_45: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_51, [8, 1, 192, 784])
    var_mean_15 = torch.ops.aten.var_mean.correction(view_45, [2, 3], correction = 0, keepdim = True)
    getitem_30: "f32[8, 1, 1, 1]" = var_mean_15[0]
    getitem_31: "f32[8, 1, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_52: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
    rsqrt_15: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_23: "f32[8, 1, 192, 784]" = torch.ops.aten.sub.Tensor(view_45, getitem_31);  view_45 = None
    mul_66: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = None
    view_46: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(mul_66, [8, 192, 28, 28]);  mul_66 = None
    unsqueeze_90: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_47, 0);  primals_47 = None
    unsqueeze_91: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, 2);  unsqueeze_90 = None
    unsqueeze_92: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_91, 3);  unsqueeze_91 = None
    unsqueeze_93: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_46, 0)
    unsqueeze_94: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_93, 2);  unsqueeze_93 = None
    unsqueeze_95: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, 3);  unsqueeze_94 = None
    mul_67: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(view_46, unsqueeze_95);  view_46 = unsqueeze_95 = None
    add_53: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_67, unsqueeze_92);  mul_67 = unsqueeze_92 = None
    squeeze_30: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_31, [2, 3]);  getitem_31 = None
    squeeze_31: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_15, [2, 3]);  rsqrt_15 = None
    alias_30: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_30);  squeeze_30 = None
    alias_31: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_31);  squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_16: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_53, primals_251, primals_252, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_68: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_16, 0.5)
    mul_69: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_16, 0.7071067811865476)
    erf_7: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
    add_54: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_70: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_68, add_54);  mul_68 = add_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_14: "f32[8, 768, 28, 28]" = torch.ops.aten.clone.default(mul_70);  mul_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_17: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(clone_14, primals_253, primals_254, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_15: "f32[8, 192, 28, 28]" = torch.ops.aten.clone.default(convolution_17);  convolution_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_47: "f32[192, 1, 1]" = torch.ops.aten.view.default(primals_48, [192, 1, 1]);  primals_48 = None
    mul_71: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(clone_15, view_47)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_55: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_51, mul_71);  mul_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_48: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_55, [8, 1, 192, 784])
    var_mean_16 = torch.ops.aten.var_mean.correction(view_48, [2, 3], correction = 0, keepdim = True)
    getitem_32: "f32[8, 1, 1, 1]" = var_mean_16[0]
    getitem_33: "f32[8, 1, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_56: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
    rsqrt_16: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_24: "f32[8, 1, 192, 784]" = torch.ops.aten.sub.Tensor(view_48, getitem_33);  view_48 = None
    mul_72: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = None
    view_49: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(mul_72, [8, 192, 28, 28]);  mul_72 = None
    unsqueeze_96: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_50, 0);  primals_50 = None
    unsqueeze_97: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, 2);  unsqueeze_96 = None
    unsqueeze_98: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_97, 3);  unsqueeze_97 = None
    unsqueeze_99: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_49, 0)
    unsqueeze_100: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_99, 2);  unsqueeze_99 = None
    unsqueeze_101: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, 3);  unsqueeze_100 = None
    mul_73: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(view_49, unsqueeze_101);  view_49 = unsqueeze_101 = None
    add_57: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_73, unsqueeze_98);  mul_73 = unsqueeze_98 = None
    squeeze_32: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_33, [2, 3]);  getitem_33 = None
    squeeze_33: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_16, [2, 3]);  rsqrt_16 = None
    alias_32: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_32);  squeeze_32 = None
    alias_33: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_33);  squeeze_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_8: "f32[8, 192, 28, 28]" = torch.ops.aten.avg_pool2d.default(add_57, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_25: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(avg_pool2d_8, add_57);  avg_pool2d_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_50: "f32[192, 1, 1]" = torch.ops.aten.view.default(primals_51, [192, 1, 1]);  primals_51 = None
    mul_74: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_25, view_50)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_58: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_55, mul_74);  mul_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_51: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_58, [8, 1, 192, 784])
    var_mean_17 = torch.ops.aten.var_mean.correction(view_51, [2, 3], correction = 0, keepdim = True)
    getitem_34: "f32[8, 1, 1, 1]" = var_mean_17[0]
    getitem_35: "f32[8, 1, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_59: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
    rsqrt_17: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
    sub_26: "f32[8, 1, 192, 784]" = torch.ops.aten.sub.Tensor(view_51, getitem_35);  view_51 = None
    mul_75: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = None
    view_52: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(mul_75, [8, 192, 28, 28]);  mul_75 = None
    unsqueeze_102: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_53, 0);  primals_53 = None
    unsqueeze_103: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, 2);  unsqueeze_102 = None
    unsqueeze_104: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_103, 3);  unsqueeze_103 = None
    unsqueeze_105: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_52, 0)
    unsqueeze_106: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_105, 2);  unsqueeze_105 = None
    unsqueeze_107: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, 3);  unsqueeze_106 = None
    mul_76: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(view_52, unsqueeze_107);  view_52 = unsqueeze_107 = None
    add_60: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_104);  mul_76 = unsqueeze_104 = None
    squeeze_34: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_35, [2, 3]);  getitem_35 = None
    squeeze_35: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_17, [2, 3]);  rsqrt_17 = None
    alias_34: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_34);  squeeze_34 = None
    alias_35: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_35);  squeeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_18: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_60, primals_255, primals_256, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_77: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_18, 0.5)
    mul_78: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_18, 0.7071067811865476)
    erf_8: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_78);  mul_78 = None
    add_61: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_79: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_77, add_61);  mul_77 = add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_16: "f32[8, 768, 28, 28]" = torch.ops.aten.clone.default(mul_79);  mul_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_19: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(clone_16, primals_257, primals_258, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_17: "f32[8, 192, 28, 28]" = torch.ops.aten.clone.default(convolution_19);  convolution_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_53: "f32[192, 1, 1]" = torch.ops.aten.view.default(primals_54, [192, 1, 1]);  primals_54 = None
    mul_80: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(clone_17, view_53)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_62: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_58, mul_80);  mul_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_54: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_62, [8, 1, 192, 784])
    var_mean_18 = torch.ops.aten.var_mean.correction(view_54, [2, 3], correction = 0, keepdim = True)
    getitem_36: "f32[8, 1, 1, 1]" = var_mean_18[0]
    getitem_37: "f32[8, 1, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_63: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
    rsqrt_18: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_27: "f32[8, 1, 192, 784]" = torch.ops.aten.sub.Tensor(view_54, getitem_37);  view_54 = None
    mul_81: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = None
    view_55: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(mul_81, [8, 192, 28, 28]);  mul_81 = None
    unsqueeze_108: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_56, 0);  primals_56 = None
    unsqueeze_109: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, 2);  unsqueeze_108 = None
    unsqueeze_110: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_109, 3);  unsqueeze_109 = None
    unsqueeze_111: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_55, 0)
    unsqueeze_112: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_111, 2);  unsqueeze_111 = None
    unsqueeze_113: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, 3);  unsqueeze_112 = None
    mul_82: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(view_55, unsqueeze_113);  view_55 = unsqueeze_113 = None
    add_64: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_82, unsqueeze_110);  mul_82 = unsqueeze_110 = None
    squeeze_36: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_37, [2, 3]);  getitem_37 = None
    squeeze_37: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_18, [2, 3]);  rsqrt_18 = None
    alias_36: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_36);  squeeze_36 = None
    alias_37: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_37);  squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_9: "f32[8, 192, 28, 28]" = torch.ops.aten.avg_pool2d.default(add_64, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_28: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(avg_pool2d_9, add_64);  avg_pool2d_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_56: "f32[192, 1, 1]" = torch.ops.aten.view.default(primals_57, [192, 1, 1]);  primals_57 = None
    mul_83: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_28, view_56)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_65: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_62, mul_83);  mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_57: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_65, [8, 1, 192, 784])
    var_mean_19 = torch.ops.aten.var_mean.correction(view_57, [2, 3], correction = 0, keepdim = True)
    getitem_38: "f32[8, 1, 1, 1]" = var_mean_19[0]
    getitem_39: "f32[8, 1, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_66: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
    rsqrt_19: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_29: "f32[8, 1, 192, 784]" = torch.ops.aten.sub.Tensor(view_57, getitem_39);  view_57 = None
    mul_84: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = None
    view_58: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(mul_84, [8, 192, 28, 28]);  mul_84 = None
    unsqueeze_114: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_59, 0);  primals_59 = None
    unsqueeze_115: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, 2);  unsqueeze_114 = None
    unsqueeze_116: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_115, 3);  unsqueeze_115 = None
    unsqueeze_117: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_58, 0)
    unsqueeze_118: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_117, 2);  unsqueeze_117 = None
    unsqueeze_119: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, 3);  unsqueeze_118 = None
    mul_85: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(view_58, unsqueeze_119);  view_58 = unsqueeze_119 = None
    add_67: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_85, unsqueeze_116);  mul_85 = unsqueeze_116 = None
    squeeze_38: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_39, [2, 3]);  getitem_39 = None
    squeeze_39: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_19, [2, 3]);  rsqrt_19 = None
    alias_38: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_38);  squeeze_38 = None
    alias_39: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_39);  squeeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_20: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_67, primals_259, primals_260, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_86: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_20, 0.5)
    mul_87: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_20, 0.7071067811865476)
    erf_9: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
    add_68: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_88: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_86, add_68);  mul_86 = add_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_18: "f32[8, 768, 28, 28]" = torch.ops.aten.clone.default(mul_88);  mul_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_21: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(clone_18, primals_261, primals_262, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_19: "f32[8, 192, 28, 28]" = torch.ops.aten.clone.default(convolution_21);  convolution_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_59: "f32[192, 1, 1]" = torch.ops.aten.view.default(primals_60, [192, 1, 1]);  primals_60 = None
    mul_89: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(clone_19, view_59)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_69: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_65, mul_89);  mul_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_60: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_69, [8, 1, 192, 784])
    var_mean_20 = torch.ops.aten.var_mean.correction(view_60, [2, 3], correction = 0, keepdim = True)
    getitem_40: "f32[8, 1, 1, 1]" = var_mean_20[0]
    getitem_41: "f32[8, 1, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_70: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_20: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    sub_30: "f32[8, 1, 192, 784]" = torch.ops.aten.sub.Tensor(view_60, getitem_41);  view_60 = None
    mul_90: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = None
    view_61: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(mul_90, [8, 192, 28, 28]);  mul_90 = None
    unsqueeze_120: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_62, 0);  primals_62 = None
    unsqueeze_121: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, 2);  unsqueeze_120 = None
    unsqueeze_122: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_121, 3);  unsqueeze_121 = None
    unsqueeze_123: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_61, 0)
    unsqueeze_124: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_123, 2);  unsqueeze_123 = None
    unsqueeze_125: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, 3);  unsqueeze_124 = None
    mul_91: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(view_61, unsqueeze_125);  view_61 = unsqueeze_125 = None
    add_71: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_91, unsqueeze_122);  mul_91 = unsqueeze_122 = None
    squeeze_40: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_41, [2, 3]);  getitem_41 = None
    squeeze_41: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_20, [2, 3]);  rsqrt_20 = None
    alias_40: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_40);  squeeze_40 = None
    alias_41: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_41);  squeeze_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_10: "f32[8, 192, 28, 28]" = torch.ops.aten.avg_pool2d.default(add_71, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_31: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(avg_pool2d_10, add_71);  avg_pool2d_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_62: "f32[192, 1, 1]" = torch.ops.aten.view.default(primals_63, [192, 1, 1]);  primals_63 = None
    mul_92: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_31, view_62)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_72: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_69, mul_92);  mul_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_63: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_72, [8, 1, 192, 784])
    var_mean_21 = torch.ops.aten.var_mean.correction(view_63, [2, 3], correction = 0, keepdim = True)
    getitem_42: "f32[8, 1, 1, 1]" = var_mean_21[0]
    getitem_43: "f32[8, 1, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_73: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
    rsqrt_21: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    sub_32: "f32[8, 1, 192, 784]" = torch.ops.aten.sub.Tensor(view_63, getitem_43);  view_63 = None
    mul_93: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = None
    view_64: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(mul_93, [8, 192, 28, 28]);  mul_93 = None
    unsqueeze_126: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_65, 0);  primals_65 = None
    unsqueeze_127: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, 2);  unsqueeze_126 = None
    unsqueeze_128: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_127, 3);  unsqueeze_127 = None
    unsqueeze_129: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_64, 0)
    unsqueeze_130: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_129, 2);  unsqueeze_129 = None
    unsqueeze_131: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, 3);  unsqueeze_130 = None
    mul_94: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(view_64, unsqueeze_131);  view_64 = unsqueeze_131 = None
    add_74: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_94, unsqueeze_128);  mul_94 = unsqueeze_128 = None
    squeeze_42: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_43, [2, 3]);  getitem_43 = None
    squeeze_43: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_21, [2, 3]);  rsqrt_21 = None
    alias_42: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_42);  squeeze_42 = None
    alias_43: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_43);  squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_22: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_74, primals_263, primals_264, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_95: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_22, 0.5)
    mul_96: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_22, 0.7071067811865476)
    erf_10: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_96);  mul_96 = None
    add_75: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_97: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_95, add_75);  mul_95 = add_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_20: "f32[8, 768, 28, 28]" = torch.ops.aten.clone.default(mul_97);  mul_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_23: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(clone_20, primals_265, primals_266, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_21: "f32[8, 192, 28, 28]" = torch.ops.aten.clone.default(convolution_23);  convolution_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_65: "f32[192, 1, 1]" = torch.ops.aten.view.default(primals_66, [192, 1, 1]);  primals_66 = None
    mul_98: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(clone_21, view_65)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_76: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_72, mul_98);  mul_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_66: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_76, [8, 1, 192, 784])
    var_mean_22 = torch.ops.aten.var_mean.correction(view_66, [2, 3], correction = 0, keepdim = True)
    getitem_44: "f32[8, 1, 1, 1]" = var_mean_22[0]
    getitem_45: "f32[8, 1, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_77: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
    rsqrt_22: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
    sub_33: "f32[8, 1, 192, 784]" = torch.ops.aten.sub.Tensor(view_66, getitem_45);  view_66 = None
    mul_99: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = None
    view_67: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(mul_99, [8, 192, 28, 28]);  mul_99 = None
    unsqueeze_132: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_68, 0);  primals_68 = None
    unsqueeze_133: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, 2);  unsqueeze_132 = None
    unsqueeze_134: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_133, 3);  unsqueeze_133 = None
    unsqueeze_135: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_67, 0)
    unsqueeze_136: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_135, 2);  unsqueeze_135 = None
    unsqueeze_137: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, 3);  unsqueeze_136 = None
    mul_100: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(view_67, unsqueeze_137);  view_67 = unsqueeze_137 = None
    add_78: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_100, unsqueeze_134);  mul_100 = unsqueeze_134 = None
    squeeze_44: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_45, [2, 3]);  getitem_45 = None
    squeeze_45: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_22, [2, 3]);  rsqrt_22 = None
    alias_44: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_44);  squeeze_44 = None
    alias_45: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_45);  squeeze_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_11: "f32[8, 192, 28, 28]" = torch.ops.aten.avg_pool2d.default(add_78, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_34: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(avg_pool2d_11, add_78);  avg_pool2d_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_68: "f32[192, 1, 1]" = torch.ops.aten.view.default(primals_69, [192, 1, 1]);  primals_69 = None
    mul_101: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_34, view_68)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_79: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_76, mul_101);  mul_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_69: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_79, [8, 1, 192, 784])
    var_mean_23 = torch.ops.aten.var_mean.correction(view_69, [2, 3], correction = 0, keepdim = True)
    getitem_46: "f32[8, 1, 1, 1]" = var_mean_23[0]
    getitem_47: "f32[8, 1, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_80: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
    rsqrt_23: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    sub_35: "f32[8, 1, 192, 784]" = torch.ops.aten.sub.Tensor(view_69, getitem_47);  view_69 = None
    mul_102: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = None
    view_70: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(mul_102, [8, 192, 28, 28]);  mul_102 = None
    unsqueeze_138: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_71, 0);  primals_71 = None
    unsqueeze_139: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, 2);  unsqueeze_138 = None
    unsqueeze_140: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_139, 3);  unsqueeze_139 = None
    unsqueeze_141: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_70, 0)
    unsqueeze_142: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_141, 2);  unsqueeze_141 = None
    unsqueeze_143: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, 3);  unsqueeze_142 = None
    mul_103: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(view_70, unsqueeze_143);  view_70 = unsqueeze_143 = None
    add_81: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_103, unsqueeze_140);  mul_103 = unsqueeze_140 = None
    squeeze_46: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_47, [2, 3]);  getitem_47 = None
    squeeze_47: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_23, [2, 3]);  rsqrt_23 = None
    alias_46: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_46);  squeeze_46 = None
    alias_47: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_47);  squeeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_24: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_81, primals_267, primals_268, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_104: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_24, 0.5)
    mul_105: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_24, 0.7071067811865476)
    erf_11: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_105);  mul_105 = None
    add_82: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_106: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_104, add_82);  mul_104 = add_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_22: "f32[8, 768, 28, 28]" = torch.ops.aten.clone.default(mul_106);  mul_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_25: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(clone_22, primals_269, primals_270, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_23: "f32[8, 192, 28, 28]" = torch.ops.aten.clone.default(convolution_25);  convolution_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_71: "f32[192, 1, 1]" = torch.ops.aten.view.default(primals_72, [192, 1, 1]);  primals_72 = None
    mul_107: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(clone_23, view_71)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_83: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_79, mul_107);  mul_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:103, code: x = self.conv(x)
    convolution_26: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(add_83, primals_271, primals_272, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  primals_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_72: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(convolution_26, [8, 1, 384, 196])
    var_mean_24 = torch.ops.aten.var_mean.correction(view_72, [2, 3], correction = 0, keepdim = True)
    getitem_48: "f32[8, 1, 1, 1]" = var_mean_24[0]
    getitem_49: "f32[8, 1, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_84: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
    rsqrt_24: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_36: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_72, getitem_49);  view_72 = None
    mul_108: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = None
    view_73: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_108, [8, 384, 14, 14]);  mul_108 = None
    unsqueeze_144: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_74, 0);  primals_74 = None
    unsqueeze_145: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, 2);  unsqueeze_144 = None
    unsqueeze_146: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_145, 3);  unsqueeze_145 = None
    unsqueeze_147: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_73, 0)
    unsqueeze_148: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_147, 2);  unsqueeze_147 = None
    unsqueeze_149: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, 3);  unsqueeze_148 = None
    mul_109: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_73, unsqueeze_149);  view_73 = unsqueeze_149 = None
    add_85: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_109, unsqueeze_146);  mul_109 = unsqueeze_146 = None
    squeeze_48: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_49, [2, 3]);  getitem_49 = None
    squeeze_49: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_24, [2, 3]);  rsqrt_24 = None
    alias_48: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_48);  squeeze_48 = None
    alias_49: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_49);  squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_12: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_85, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_37: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_12, add_85);  avg_pool2d_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_74: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_75, [384, 1, 1]);  primals_75 = None
    mul_110: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_37, view_74)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_86: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(convolution_26, mul_110);  mul_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_75: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_86, [8, 1, 384, 196])
    var_mean_25 = torch.ops.aten.var_mean.correction(view_75, [2, 3], correction = 0, keepdim = True)
    getitem_50: "f32[8, 1, 1, 1]" = var_mean_25[0]
    getitem_51: "f32[8, 1, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_87: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
    rsqrt_25: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    sub_38: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_75, getitem_51);  view_75 = None
    mul_111: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = None
    view_76: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_111, [8, 384, 14, 14]);  mul_111 = None
    unsqueeze_150: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_77, 0);  primals_77 = None
    unsqueeze_151: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, 2);  unsqueeze_150 = None
    unsqueeze_152: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_151, 3);  unsqueeze_151 = None
    unsqueeze_153: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_76, 0)
    unsqueeze_154: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_153, 2);  unsqueeze_153 = None
    unsqueeze_155: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, 3);  unsqueeze_154 = None
    mul_112: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_76, unsqueeze_155);  view_76 = unsqueeze_155 = None
    add_88: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_112, unsqueeze_152);  mul_112 = unsqueeze_152 = None
    squeeze_50: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_51, [2, 3]);  getitem_51 = None
    squeeze_51: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_25, [2, 3]);  rsqrt_25 = None
    alias_50: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_50);  squeeze_50 = None
    alias_51: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_51);  squeeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_27: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_88, primals_273, primals_274, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_113: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_27, 0.5)
    mul_114: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_27, 0.7071067811865476)
    erf_12: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_114);  mul_114 = None
    add_89: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_115: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_113, add_89);  mul_113 = add_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_24: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_115);  mul_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_28: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_24, primals_275, primals_276, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_25: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_28);  convolution_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_77: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_78, [384, 1, 1]);  primals_78 = None
    mul_116: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_25, view_77)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_90: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_86, mul_116);  mul_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_78: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_90, [8, 1, 384, 196])
    var_mean_26 = torch.ops.aten.var_mean.correction(view_78, [2, 3], correction = 0, keepdim = True)
    getitem_52: "f32[8, 1, 1, 1]" = var_mean_26[0]
    getitem_53: "f32[8, 1, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_91: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
    rsqrt_26: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
    sub_39: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_78, getitem_53);  view_78 = None
    mul_117: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_26);  sub_39 = None
    view_79: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_117, [8, 384, 14, 14]);  mul_117 = None
    unsqueeze_156: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_80, 0);  primals_80 = None
    unsqueeze_157: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, 2);  unsqueeze_156 = None
    unsqueeze_158: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_157, 3);  unsqueeze_157 = None
    unsqueeze_159: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_79, 0)
    unsqueeze_160: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_159, 2);  unsqueeze_159 = None
    unsqueeze_161: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, 3);  unsqueeze_160 = None
    mul_118: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_79, unsqueeze_161);  view_79 = unsqueeze_161 = None
    add_92: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_158);  mul_118 = unsqueeze_158 = None
    squeeze_52: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_53, [2, 3]);  getitem_53 = None
    squeeze_53: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_26, [2, 3]);  rsqrt_26 = None
    alias_52: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_52);  squeeze_52 = None
    alias_53: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_53);  squeeze_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_13: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_92, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_40: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_13, add_92);  avg_pool2d_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_80: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_81, [384, 1, 1]);  primals_81 = None
    mul_119: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_40, view_80)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_93: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_90, mul_119);  mul_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_81: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_93, [8, 1, 384, 196])
    var_mean_27 = torch.ops.aten.var_mean.correction(view_81, [2, 3], correction = 0, keepdim = True)
    getitem_54: "f32[8, 1, 1, 1]" = var_mean_27[0]
    getitem_55: "f32[8, 1, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_94: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
    rsqrt_27: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    sub_41: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_81, getitem_55);  view_81 = None
    mul_120: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_27);  sub_41 = None
    view_82: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_120, [8, 384, 14, 14]);  mul_120 = None
    unsqueeze_162: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_83, 0);  primals_83 = None
    unsqueeze_163: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, 2);  unsqueeze_162 = None
    unsqueeze_164: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_163, 3);  unsqueeze_163 = None
    unsqueeze_165: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_82, 0)
    unsqueeze_166: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_165, 2);  unsqueeze_165 = None
    unsqueeze_167: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, 3);  unsqueeze_166 = None
    mul_121: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_82, unsqueeze_167);  view_82 = unsqueeze_167 = None
    add_95: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_121, unsqueeze_164);  mul_121 = unsqueeze_164 = None
    squeeze_54: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_55, [2, 3]);  getitem_55 = None
    squeeze_55: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_27, [2, 3]);  rsqrt_27 = None
    alias_54: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_54);  squeeze_54 = None
    alias_55: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_55);  squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_29: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_95, primals_277, primals_278, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_122: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_29, 0.5)
    mul_123: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_29, 0.7071067811865476)
    erf_13: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_123);  mul_123 = None
    add_96: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_124: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_122, add_96);  mul_122 = add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_26: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_124);  mul_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_30: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_26, primals_279, primals_280, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_27: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_30);  convolution_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_83: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_84, [384, 1, 1]);  primals_84 = None
    mul_125: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_27, view_83)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_97: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_93, mul_125);  mul_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_84: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_97, [8, 1, 384, 196])
    var_mean_28 = torch.ops.aten.var_mean.correction(view_84, [2, 3], correction = 0, keepdim = True)
    getitem_56: "f32[8, 1, 1, 1]" = var_mean_28[0]
    getitem_57: "f32[8, 1, 1, 1]" = var_mean_28[1];  var_mean_28 = None
    add_98: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
    rsqrt_28: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    sub_42: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_84, getitem_57);  view_84 = None
    mul_126: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_28);  sub_42 = None
    view_85: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_126, [8, 384, 14, 14]);  mul_126 = None
    unsqueeze_168: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_86, 0);  primals_86 = None
    unsqueeze_169: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, 2);  unsqueeze_168 = None
    unsqueeze_170: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_169, 3);  unsqueeze_169 = None
    unsqueeze_171: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_85, 0)
    unsqueeze_172: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_171, 2);  unsqueeze_171 = None
    unsqueeze_173: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, 3);  unsqueeze_172 = None
    mul_127: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_85, unsqueeze_173);  view_85 = unsqueeze_173 = None
    add_99: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_127, unsqueeze_170);  mul_127 = unsqueeze_170 = None
    squeeze_56: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_57, [2, 3]);  getitem_57 = None
    squeeze_57: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_28, [2, 3]);  rsqrt_28 = None
    alias_56: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_56);  squeeze_56 = None
    alias_57: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_57);  squeeze_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_14: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_99, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_43: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_14, add_99);  avg_pool2d_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_86: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_87, [384, 1, 1]);  primals_87 = None
    mul_128: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_43, view_86)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_100: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_97, mul_128);  mul_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_87: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_100, [8, 1, 384, 196])
    var_mean_29 = torch.ops.aten.var_mean.correction(view_87, [2, 3], correction = 0, keepdim = True)
    getitem_58: "f32[8, 1, 1, 1]" = var_mean_29[0]
    getitem_59: "f32[8, 1, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_101: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
    rsqrt_29: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    sub_44: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_87, getitem_59);  view_87 = None
    mul_129: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_29);  sub_44 = None
    view_88: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_129, [8, 384, 14, 14]);  mul_129 = None
    unsqueeze_174: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_89, 0);  primals_89 = None
    unsqueeze_175: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, 2);  unsqueeze_174 = None
    unsqueeze_176: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_175, 3);  unsqueeze_175 = None
    unsqueeze_177: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_88, 0)
    unsqueeze_178: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_177, 2);  unsqueeze_177 = None
    unsqueeze_179: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, 3);  unsqueeze_178 = None
    mul_130: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_88, unsqueeze_179);  view_88 = unsqueeze_179 = None
    add_102: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_130, unsqueeze_176);  mul_130 = unsqueeze_176 = None
    squeeze_58: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_59, [2, 3]);  getitem_59 = None
    squeeze_59: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_29, [2, 3]);  rsqrt_29 = None
    alias_58: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_58);  squeeze_58 = None
    alias_59: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_59);  squeeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_31: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_102, primals_281, primals_282, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_131: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_31, 0.5)
    mul_132: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_31, 0.7071067811865476)
    erf_14: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_132);  mul_132 = None
    add_103: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_133: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_131, add_103);  mul_131 = add_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_28: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_133);  mul_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_32: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_28, primals_283, primals_284, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_29: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_32);  convolution_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_89: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_90, [384, 1, 1]);  primals_90 = None
    mul_134: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_29, view_89)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_104: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_100, mul_134);  mul_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_90: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_104, [8, 1, 384, 196])
    var_mean_30 = torch.ops.aten.var_mean.correction(view_90, [2, 3], correction = 0, keepdim = True)
    getitem_60: "f32[8, 1, 1, 1]" = var_mean_30[0]
    getitem_61: "f32[8, 1, 1, 1]" = var_mean_30[1];  var_mean_30 = None
    add_105: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
    rsqrt_30: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    sub_45: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_90, getitem_61);  view_90 = None
    mul_135: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_30);  sub_45 = None
    view_91: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_135, [8, 384, 14, 14]);  mul_135 = None
    unsqueeze_180: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_92, 0);  primals_92 = None
    unsqueeze_181: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, 2);  unsqueeze_180 = None
    unsqueeze_182: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_181, 3);  unsqueeze_181 = None
    unsqueeze_183: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_91, 0)
    unsqueeze_184: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 2);  unsqueeze_183 = None
    unsqueeze_185: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, 3);  unsqueeze_184 = None
    mul_136: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_91, unsqueeze_185);  view_91 = unsqueeze_185 = None
    add_106: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_136, unsqueeze_182);  mul_136 = unsqueeze_182 = None
    squeeze_60: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_61, [2, 3]);  getitem_61 = None
    squeeze_61: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_30, [2, 3]);  rsqrt_30 = None
    alias_60: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_60);  squeeze_60 = None
    alias_61: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_61);  squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_15: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_106, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_46: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_15, add_106);  avg_pool2d_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_92: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_93, [384, 1, 1]);  primals_93 = None
    mul_137: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_46, view_92)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_107: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_104, mul_137);  mul_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_93: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_107, [8, 1, 384, 196])
    var_mean_31 = torch.ops.aten.var_mean.correction(view_93, [2, 3], correction = 0, keepdim = True)
    getitem_62: "f32[8, 1, 1, 1]" = var_mean_31[0]
    getitem_63: "f32[8, 1, 1, 1]" = var_mean_31[1];  var_mean_31 = None
    add_108: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
    rsqrt_31: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    sub_47: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_93, getitem_63);  view_93 = None
    mul_138: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_31);  sub_47 = None
    view_94: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_138, [8, 384, 14, 14]);  mul_138 = None
    unsqueeze_186: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_95, 0);  primals_95 = None
    unsqueeze_187: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, 2);  unsqueeze_186 = None
    unsqueeze_188: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_187, 3);  unsqueeze_187 = None
    unsqueeze_189: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_94, 0)
    unsqueeze_190: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_189, 2);  unsqueeze_189 = None
    unsqueeze_191: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, 3);  unsqueeze_190 = None
    mul_139: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_94, unsqueeze_191);  view_94 = unsqueeze_191 = None
    add_109: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_139, unsqueeze_188);  mul_139 = unsqueeze_188 = None
    squeeze_62: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_63, [2, 3]);  getitem_63 = None
    squeeze_63: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_31, [2, 3]);  rsqrt_31 = None
    alias_62: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_62);  squeeze_62 = None
    alias_63: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_63);  squeeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_33: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_109, primals_285, primals_286, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_140: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_33, 0.5)
    mul_141: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_33, 0.7071067811865476)
    erf_15: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_141);  mul_141 = None
    add_110: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_142: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_140, add_110);  mul_140 = add_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_30: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_142);  mul_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_34: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_30, primals_287, primals_288, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_31: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_34);  convolution_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_95: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_96, [384, 1, 1]);  primals_96 = None
    mul_143: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_31, view_95)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_111: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_107, mul_143);  mul_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_96: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_111, [8, 1, 384, 196])
    var_mean_32 = torch.ops.aten.var_mean.correction(view_96, [2, 3], correction = 0, keepdim = True)
    getitem_64: "f32[8, 1, 1, 1]" = var_mean_32[0]
    getitem_65: "f32[8, 1, 1, 1]" = var_mean_32[1];  var_mean_32 = None
    add_112: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
    rsqrt_32: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
    sub_48: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_96, getitem_65);  view_96 = None
    mul_144: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_32);  sub_48 = None
    view_97: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_144, [8, 384, 14, 14]);  mul_144 = None
    unsqueeze_192: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_98, 0);  primals_98 = None
    unsqueeze_193: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, 2);  unsqueeze_192 = None
    unsqueeze_194: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_193, 3);  unsqueeze_193 = None
    unsqueeze_195: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_97, 0)
    unsqueeze_196: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_195, 2);  unsqueeze_195 = None
    unsqueeze_197: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, 3);  unsqueeze_196 = None
    mul_145: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_97, unsqueeze_197);  view_97 = unsqueeze_197 = None
    add_113: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_145, unsqueeze_194);  mul_145 = unsqueeze_194 = None
    squeeze_64: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_65, [2, 3]);  getitem_65 = None
    squeeze_65: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_32, [2, 3]);  rsqrt_32 = None
    alias_64: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_64);  squeeze_64 = None
    alias_65: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_65);  squeeze_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_16: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_113, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_49: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_16, add_113);  avg_pool2d_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_98: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_99, [384, 1, 1]);  primals_99 = None
    mul_146: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_49, view_98)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_114: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_111, mul_146);  mul_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_99: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_114, [8, 1, 384, 196])
    var_mean_33 = torch.ops.aten.var_mean.correction(view_99, [2, 3], correction = 0, keepdim = True)
    getitem_66: "f32[8, 1, 1, 1]" = var_mean_33[0]
    getitem_67: "f32[8, 1, 1, 1]" = var_mean_33[1];  var_mean_33 = None
    add_115: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
    rsqrt_33: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    sub_50: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_99, getitem_67);  view_99 = None
    mul_147: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_33);  sub_50 = None
    view_100: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_147, [8, 384, 14, 14]);  mul_147 = None
    unsqueeze_198: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_101, 0);  primals_101 = None
    unsqueeze_199: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, 2);  unsqueeze_198 = None
    unsqueeze_200: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_199, 3);  unsqueeze_199 = None
    unsqueeze_201: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_100, 0)
    unsqueeze_202: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_201, 2);  unsqueeze_201 = None
    unsqueeze_203: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, 3);  unsqueeze_202 = None
    mul_148: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_100, unsqueeze_203);  view_100 = unsqueeze_203 = None
    add_116: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_148, unsqueeze_200);  mul_148 = unsqueeze_200 = None
    squeeze_66: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_67, [2, 3]);  getitem_67 = None
    squeeze_67: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_33, [2, 3]);  rsqrt_33 = None
    alias_66: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_66);  squeeze_66 = None
    alias_67: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_67);  squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_35: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_116, primals_289, primals_290, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_149: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_35, 0.5)
    mul_150: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_35, 0.7071067811865476)
    erf_16: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_150);  mul_150 = None
    add_117: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_151: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_149, add_117);  mul_149 = add_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_32: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_151);  mul_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_36: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_32, primals_291, primals_292, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_33: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_36);  convolution_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_101: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_102, [384, 1, 1]);  primals_102 = None
    mul_152: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_33, view_101)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_118: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_114, mul_152);  mul_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_102: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_118, [8, 1, 384, 196])
    var_mean_34 = torch.ops.aten.var_mean.correction(view_102, [2, 3], correction = 0, keepdim = True)
    getitem_68: "f32[8, 1, 1, 1]" = var_mean_34[0]
    getitem_69: "f32[8, 1, 1, 1]" = var_mean_34[1];  var_mean_34 = None
    add_119: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
    rsqrt_34: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
    sub_51: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_102, getitem_69);  view_102 = None
    mul_153: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_34);  sub_51 = None
    view_103: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_153, [8, 384, 14, 14]);  mul_153 = None
    unsqueeze_204: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_104, 0);  primals_104 = None
    unsqueeze_205: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, 2);  unsqueeze_204 = None
    unsqueeze_206: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 3);  unsqueeze_205 = None
    unsqueeze_207: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_103, 0)
    unsqueeze_208: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_207, 2);  unsqueeze_207 = None
    unsqueeze_209: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, 3);  unsqueeze_208 = None
    mul_154: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_103, unsqueeze_209);  view_103 = unsqueeze_209 = None
    add_120: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_154, unsqueeze_206);  mul_154 = unsqueeze_206 = None
    squeeze_68: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_69, [2, 3]);  getitem_69 = None
    squeeze_69: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_34, [2, 3]);  rsqrt_34 = None
    alias_68: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_68);  squeeze_68 = None
    alias_69: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_69);  squeeze_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_17: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_120, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_52: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_17, add_120);  avg_pool2d_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_104: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_105, [384, 1, 1]);  primals_105 = None
    mul_155: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_52, view_104)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_121: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_118, mul_155);  mul_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_105: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_121, [8, 1, 384, 196])
    var_mean_35 = torch.ops.aten.var_mean.correction(view_105, [2, 3], correction = 0, keepdim = True)
    getitem_70: "f32[8, 1, 1, 1]" = var_mean_35[0]
    getitem_71: "f32[8, 1, 1, 1]" = var_mean_35[1];  var_mean_35 = None
    add_122: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
    rsqrt_35: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    sub_53: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_105, getitem_71);  view_105 = None
    mul_156: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_35);  sub_53 = None
    view_106: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_156, [8, 384, 14, 14]);  mul_156 = None
    unsqueeze_210: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_107, 0);  primals_107 = None
    unsqueeze_211: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, 2);  unsqueeze_210 = None
    unsqueeze_212: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 3);  unsqueeze_211 = None
    unsqueeze_213: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_106, 0)
    unsqueeze_214: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 2);  unsqueeze_213 = None
    unsqueeze_215: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, 3);  unsqueeze_214 = None
    mul_157: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_106, unsqueeze_215);  view_106 = unsqueeze_215 = None
    add_123: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_157, unsqueeze_212);  mul_157 = unsqueeze_212 = None
    squeeze_70: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_71, [2, 3]);  getitem_71 = None
    squeeze_71: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_35, [2, 3]);  rsqrt_35 = None
    alias_70: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_70);  squeeze_70 = None
    alias_71: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_71);  squeeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_37: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_123, primals_293, primals_294, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_158: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_37, 0.5)
    mul_159: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_37, 0.7071067811865476)
    erf_17: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_159);  mul_159 = None
    add_124: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_160: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_158, add_124);  mul_158 = add_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_34: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_160);  mul_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_38: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_34, primals_295, primals_296, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_35: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_38);  convolution_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_107: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_108, [384, 1, 1]);  primals_108 = None
    mul_161: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_35, view_107)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_125: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_121, mul_161);  mul_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_108: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_125, [8, 1, 384, 196])
    var_mean_36 = torch.ops.aten.var_mean.correction(view_108, [2, 3], correction = 0, keepdim = True)
    getitem_72: "f32[8, 1, 1, 1]" = var_mean_36[0]
    getitem_73: "f32[8, 1, 1, 1]" = var_mean_36[1];  var_mean_36 = None
    add_126: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
    rsqrt_36: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    sub_54: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_108, getitem_73);  view_108 = None
    mul_162: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_36);  sub_54 = None
    view_109: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_162, [8, 384, 14, 14]);  mul_162 = None
    unsqueeze_216: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_110, 0);  primals_110 = None
    unsqueeze_217: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, 2);  unsqueeze_216 = None
    unsqueeze_218: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 3);  unsqueeze_217 = None
    unsqueeze_219: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_109, 0)
    unsqueeze_220: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 2);  unsqueeze_219 = None
    unsqueeze_221: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, 3);  unsqueeze_220 = None
    mul_163: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_109, unsqueeze_221);  view_109 = unsqueeze_221 = None
    add_127: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_163, unsqueeze_218);  mul_163 = unsqueeze_218 = None
    squeeze_72: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_73, [2, 3]);  getitem_73 = None
    squeeze_73: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_36, [2, 3]);  rsqrt_36 = None
    alias_72: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_72);  squeeze_72 = None
    alias_73: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_73);  squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_18: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_127, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_55: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_18, add_127);  avg_pool2d_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_110: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_111, [384, 1, 1]);  primals_111 = None
    mul_164: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_55, view_110)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_128: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_125, mul_164);  mul_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_111: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_128, [8, 1, 384, 196])
    var_mean_37 = torch.ops.aten.var_mean.correction(view_111, [2, 3], correction = 0, keepdim = True)
    getitem_74: "f32[8, 1, 1, 1]" = var_mean_37[0]
    getitem_75: "f32[8, 1, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_129: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
    rsqrt_37: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
    sub_56: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_111, getitem_75);  view_111 = None
    mul_165: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_37);  sub_56 = None
    view_112: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_165, [8, 384, 14, 14]);  mul_165 = None
    unsqueeze_222: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_113, 0);  primals_113 = None
    unsqueeze_223: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, 2);  unsqueeze_222 = None
    unsqueeze_224: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 3);  unsqueeze_223 = None
    unsqueeze_225: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_112, 0)
    unsqueeze_226: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 2);  unsqueeze_225 = None
    unsqueeze_227: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 3);  unsqueeze_226 = None
    mul_166: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_112, unsqueeze_227);  view_112 = unsqueeze_227 = None
    add_130: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_166, unsqueeze_224);  mul_166 = unsqueeze_224 = None
    squeeze_74: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_75, [2, 3]);  getitem_75 = None
    squeeze_75: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_37, [2, 3]);  rsqrt_37 = None
    alias_74: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_74);  squeeze_74 = None
    alias_75: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_75);  squeeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_39: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_130, primals_297, primals_298, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_167: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_39, 0.5)
    mul_168: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_39, 0.7071067811865476)
    erf_18: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_168);  mul_168 = None
    add_131: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_169: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_167, add_131);  mul_167 = add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_36: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_169);  mul_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_40: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_36, primals_299, primals_300, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_37: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_40);  convolution_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_113: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_114, [384, 1, 1]);  primals_114 = None
    mul_170: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_37, view_113)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_132: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_128, mul_170);  mul_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_114: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_132, [8, 1, 384, 196])
    var_mean_38 = torch.ops.aten.var_mean.correction(view_114, [2, 3], correction = 0, keepdim = True)
    getitem_76: "f32[8, 1, 1, 1]" = var_mean_38[0]
    getitem_77: "f32[8, 1, 1, 1]" = var_mean_38[1];  var_mean_38 = None
    add_133: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
    rsqrt_38: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
    sub_57: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_114, getitem_77);  view_114 = None
    mul_171: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_38);  sub_57 = None
    view_115: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_171, [8, 384, 14, 14]);  mul_171 = None
    unsqueeze_228: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_116, 0);  primals_116 = None
    unsqueeze_229: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, 2);  unsqueeze_228 = None
    unsqueeze_230: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 3);  unsqueeze_229 = None
    unsqueeze_231: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_115, 0)
    unsqueeze_232: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 2);  unsqueeze_231 = None
    unsqueeze_233: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 3);  unsqueeze_232 = None
    mul_172: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_115, unsqueeze_233);  view_115 = unsqueeze_233 = None
    add_134: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_172, unsqueeze_230);  mul_172 = unsqueeze_230 = None
    squeeze_76: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_77, [2, 3]);  getitem_77 = None
    squeeze_77: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_38, [2, 3]);  rsqrt_38 = None
    alias_76: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_76);  squeeze_76 = None
    alias_77: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_77);  squeeze_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_19: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_134, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_58: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_19, add_134);  avg_pool2d_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_116: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_117, [384, 1, 1]);  primals_117 = None
    mul_173: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_58, view_116)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_135: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_132, mul_173);  mul_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_117: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_135, [8, 1, 384, 196])
    var_mean_39 = torch.ops.aten.var_mean.correction(view_117, [2, 3], correction = 0, keepdim = True)
    getitem_78: "f32[8, 1, 1, 1]" = var_mean_39[0]
    getitem_79: "f32[8, 1, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_136: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
    rsqrt_39: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
    sub_59: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_117, getitem_79);  view_117 = None
    mul_174: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_39);  sub_59 = None
    view_118: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_174, [8, 384, 14, 14]);  mul_174 = None
    unsqueeze_234: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_119, 0);  primals_119 = None
    unsqueeze_235: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, 2);  unsqueeze_234 = None
    unsqueeze_236: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 3);  unsqueeze_235 = None
    unsqueeze_237: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_118, 0)
    unsqueeze_238: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_237, 2);  unsqueeze_237 = None
    unsqueeze_239: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 3);  unsqueeze_238 = None
    mul_175: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_118, unsqueeze_239);  view_118 = unsqueeze_239 = None
    add_137: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_175, unsqueeze_236);  mul_175 = unsqueeze_236 = None
    squeeze_78: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_79, [2, 3]);  getitem_79 = None
    squeeze_79: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_39, [2, 3]);  rsqrt_39 = None
    alias_78: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_78);  squeeze_78 = None
    alias_79: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_79);  squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_41: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_137, primals_301, primals_302, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_176: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_41, 0.5)
    mul_177: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_41, 0.7071067811865476)
    erf_19: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_177);  mul_177 = None
    add_138: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_178: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_176, add_138);  mul_176 = add_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_38: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_178);  mul_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_42: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_38, primals_303, primals_304, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_39: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_42);  convolution_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_119: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_120, [384, 1, 1]);  primals_120 = None
    mul_179: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_39, view_119)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_139: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_135, mul_179);  mul_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_120: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_139, [8, 1, 384, 196])
    var_mean_40 = torch.ops.aten.var_mean.correction(view_120, [2, 3], correction = 0, keepdim = True)
    getitem_80: "f32[8, 1, 1, 1]" = var_mean_40[0]
    getitem_81: "f32[8, 1, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_140: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
    rsqrt_40: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    sub_60: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_120, getitem_81);  view_120 = None
    mul_180: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_40);  sub_60 = None
    view_121: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_180, [8, 384, 14, 14]);  mul_180 = None
    unsqueeze_240: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_122, 0);  primals_122 = None
    unsqueeze_241: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, 2);  unsqueeze_240 = None
    unsqueeze_242: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 3);  unsqueeze_241 = None
    unsqueeze_243: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_121, 0)
    unsqueeze_244: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_243, 2);  unsqueeze_243 = None
    unsqueeze_245: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 3);  unsqueeze_244 = None
    mul_181: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_121, unsqueeze_245);  view_121 = unsqueeze_245 = None
    add_141: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_242);  mul_181 = unsqueeze_242 = None
    squeeze_80: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_81, [2, 3]);  getitem_81 = None
    squeeze_81: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_40, [2, 3]);  rsqrt_40 = None
    alias_80: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_80);  squeeze_80 = None
    alias_81: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_81);  squeeze_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_20: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_141, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_61: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_20, add_141);  avg_pool2d_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_122: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_123, [384, 1, 1]);  primals_123 = None
    mul_182: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_61, view_122)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_142: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_139, mul_182);  mul_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_123: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_142, [8, 1, 384, 196])
    var_mean_41 = torch.ops.aten.var_mean.correction(view_123, [2, 3], correction = 0, keepdim = True)
    getitem_82: "f32[8, 1, 1, 1]" = var_mean_41[0]
    getitem_83: "f32[8, 1, 1, 1]" = var_mean_41[1];  var_mean_41 = None
    add_143: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
    rsqrt_41: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
    sub_62: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_123, getitem_83);  view_123 = None
    mul_183: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_41);  sub_62 = None
    view_124: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_183, [8, 384, 14, 14]);  mul_183 = None
    unsqueeze_246: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_125, 0);  primals_125 = None
    unsqueeze_247: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, 2);  unsqueeze_246 = None
    unsqueeze_248: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 3);  unsqueeze_247 = None
    unsqueeze_249: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_124, 0)
    unsqueeze_250: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 2);  unsqueeze_249 = None
    unsqueeze_251: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 3);  unsqueeze_250 = None
    mul_184: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_124, unsqueeze_251);  view_124 = unsqueeze_251 = None
    add_144: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_184, unsqueeze_248);  mul_184 = unsqueeze_248 = None
    squeeze_82: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_83, [2, 3]);  getitem_83 = None
    squeeze_83: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_41, [2, 3]);  rsqrt_41 = None
    alias_82: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_82);  squeeze_82 = None
    alias_83: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_83);  squeeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_43: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_144, primals_305, primals_306, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_185: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_43, 0.5)
    mul_186: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_43, 0.7071067811865476)
    erf_20: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_186);  mul_186 = None
    add_145: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_187: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_185, add_145);  mul_185 = add_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_40: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_187);  mul_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_44: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_40, primals_307, primals_308, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_41: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_44);  convolution_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_125: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_126, [384, 1, 1]);  primals_126 = None
    mul_188: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_41, view_125)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_146: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_142, mul_188);  mul_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_126: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_146, [8, 1, 384, 196])
    var_mean_42 = torch.ops.aten.var_mean.correction(view_126, [2, 3], correction = 0, keepdim = True)
    getitem_84: "f32[8, 1, 1, 1]" = var_mean_42[0]
    getitem_85: "f32[8, 1, 1, 1]" = var_mean_42[1];  var_mean_42 = None
    add_147: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
    rsqrt_42: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
    sub_63: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_126, getitem_85);  view_126 = None
    mul_189: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_42);  sub_63 = None
    view_127: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_189, [8, 384, 14, 14]);  mul_189 = None
    unsqueeze_252: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_128, 0);  primals_128 = None
    unsqueeze_253: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, 2);  unsqueeze_252 = None
    unsqueeze_254: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 3);  unsqueeze_253 = None
    unsqueeze_255: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_127, 0)
    unsqueeze_256: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 2);  unsqueeze_255 = None
    unsqueeze_257: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 3);  unsqueeze_256 = None
    mul_190: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_127, unsqueeze_257);  view_127 = unsqueeze_257 = None
    add_148: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_190, unsqueeze_254);  mul_190 = unsqueeze_254 = None
    squeeze_84: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_85, [2, 3]);  getitem_85 = None
    squeeze_85: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_42, [2, 3]);  rsqrt_42 = None
    alias_84: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_84);  squeeze_84 = None
    alias_85: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_85);  squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_21: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_148, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_64: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_21, add_148);  avg_pool2d_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_128: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_129, [384, 1, 1]);  primals_129 = None
    mul_191: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_64, view_128)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_149: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_146, mul_191);  mul_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_129: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_149, [8, 1, 384, 196])
    var_mean_43 = torch.ops.aten.var_mean.correction(view_129, [2, 3], correction = 0, keepdim = True)
    getitem_86: "f32[8, 1, 1, 1]" = var_mean_43[0]
    getitem_87: "f32[8, 1, 1, 1]" = var_mean_43[1];  var_mean_43 = None
    add_150: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
    rsqrt_43: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
    sub_65: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_129, getitem_87);  view_129 = None
    mul_192: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_43);  sub_65 = None
    view_130: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_192, [8, 384, 14, 14]);  mul_192 = None
    unsqueeze_258: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_131, 0);  primals_131 = None
    unsqueeze_259: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, 2);  unsqueeze_258 = None
    unsqueeze_260: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 3);  unsqueeze_259 = None
    unsqueeze_261: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_130, 0)
    unsqueeze_262: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 2);  unsqueeze_261 = None
    unsqueeze_263: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 3);  unsqueeze_262 = None
    mul_193: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_130, unsqueeze_263);  view_130 = unsqueeze_263 = None
    add_151: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_193, unsqueeze_260);  mul_193 = unsqueeze_260 = None
    squeeze_86: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_87, [2, 3]);  getitem_87 = None
    squeeze_87: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_43, [2, 3]);  rsqrt_43 = None
    alias_86: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_86);  squeeze_86 = None
    alias_87: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_87);  squeeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_45: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_151, primals_309, primals_310, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_194: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_45, 0.5)
    mul_195: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_45, 0.7071067811865476)
    erf_21: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_195);  mul_195 = None
    add_152: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_196: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_194, add_152);  mul_194 = add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_42: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_196);  mul_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_46: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_42, primals_311, primals_312, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_43: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_46);  convolution_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_131: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_132, [384, 1, 1]);  primals_132 = None
    mul_197: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_43, view_131)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_153: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_149, mul_197);  mul_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_132: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_153, [8, 1, 384, 196])
    var_mean_44 = torch.ops.aten.var_mean.correction(view_132, [2, 3], correction = 0, keepdim = True)
    getitem_88: "f32[8, 1, 1, 1]" = var_mean_44[0]
    getitem_89: "f32[8, 1, 1, 1]" = var_mean_44[1];  var_mean_44 = None
    add_154: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
    rsqrt_44: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
    sub_66: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_132, getitem_89);  view_132 = None
    mul_198: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_44);  sub_66 = None
    view_133: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_198, [8, 384, 14, 14]);  mul_198 = None
    unsqueeze_264: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_134, 0);  primals_134 = None
    unsqueeze_265: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 2);  unsqueeze_264 = None
    unsqueeze_266: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 3);  unsqueeze_265 = None
    unsqueeze_267: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_133, 0)
    unsqueeze_268: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 2);  unsqueeze_267 = None
    unsqueeze_269: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 3);  unsqueeze_268 = None
    mul_199: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_133, unsqueeze_269);  view_133 = unsqueeze_269 = None
    add_155: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_199, unsqueeze_266);  mul_199 = unsqueeze_266 = None
    squeeze_88: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_89, [2, 3]);  getitem_89 = None
    squeeze_89: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_44, [2, 3]);  rsqrt_44 = None
    alias_88: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_88);  squeeze_88 = None
    alias_89: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_89);  squeeze_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_22: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_155, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_67: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_22, add_155);  avg_pool2d_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_134: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_135, [384, 1, 1]);  primals_135 = None
    mul_200: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_67, view_134)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_156: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_153, mul_200);  mul_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_135: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_156, [8, 1, 384, 196])
    var_mean_45 = torch.ops.aten.var_mean.correction(view_135, [2, 3], correction = 0, keepdim = True)
    getitem_90: "f32[8, 1, 1, 1]" = var_mean_45[0]
    getitem_91: "f32[8, 1, 1, 1]" = var_mean_45[1];  var_mean_45 = None
    add_157: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
    rsqrt_45: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
    sub_68: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_135, getitem_91);  view_135 = None
    mul_201: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_45);  sub_68 = None
    view_136: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_201, [8, 384, 14, 14]);  mul_201 = None
    unsqueeze_270: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_137, 0);  primals_137 = None
    unsqueeze_271: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 2);  unsqueeze_270 = None
    unsqueeze_272: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 3);  unsqueeze_271 = None
    unsqueeze_273: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_136, 0)
    unsqueeze_274: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 2);  unsqueeze_273 = None
    unsqueeze_275: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 3);  unsqueeze_274 = None
    mul_202: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_136, unsqueeze_275);  view_136 = unsqueeze_275 = None
    add_158: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_272);  mul_202 = unsqueeze_272 = None
    squeeze_90: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_91, [2, 3]);  getitem_91 = None
    squeeze_91: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_45, [2, 3]);  rsqrt_45 = None
    alias_90: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_90);  squeeze_90 = None
    alias_91: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_91);  squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_47: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_158, primals_313, primals_314, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_203: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_47, 0.5)
    mul_204: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_47, 0.7071067811865476)
    erf_22: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_204);  mul_204 = None
    add_159: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_205: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_203, add_159);  mul_203 = add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_44: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_205);  mul_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_48: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_44, primals_315, primals_316, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_45: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_48);  convolution_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_137: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_138, [384, 1, 1]);  primals_138 = None
    mul_206: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_45, view_137)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_160: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_156, mul_206);  mul_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_138: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_160, [8, 1, 384, 196])
    var_mean_46 = torch.ops.aten.var_mean.correction(view_138, [2, 3], correction = 0, keepdim = True)
    getitem_92: "f32[8, 1, 1, 1]" = var_mean_46[0]
    getitem_93: "f32[8, 1, 1, 1]" = var_mean_46[1];  var_mean_46 = None
    add_161: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
    rsqrt_46: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    sub_69: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_138, getitem_93);  view_138 = None
    mul_207: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_46);  sub_69 = None
    view_139: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_207, [8, 384, 14, 14]);  mul_207 = None
    unsqueeze_276: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_140, 0);  primals_140 = None
    unsqueeze_277: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 2);  unsqueeze_276 = None
    unsqueeze_278: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 3);  unsqueeze_277 = None
    unsqueeze_279: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_139, 0)
    unsqueeze_280: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 2);  unsqueeze_279 = None
    unsqueeze_281: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 3);  unsqueeze_280 = None
    mul_208: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_139, unsqueeze_281);  view_139 = unsqueeze_281 = None
    add_162: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_208, unsqueeze_278);  mul_208 = unsqueeze_278 = None
    squeeze_92: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_93, [2, 3]);  getitem_93 = None
    squeeze_93: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_46, [2, 3]);  rsqrt_46 = None
    alias_92: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_92);  squeeze_92 = None
    alias_93: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_93);  squeeze_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_23: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_162, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_70: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_23, add_162);  avg_pool2d_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_140: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_141, [384, 1, 1]);  primals_141 = None
    mul_209: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_70, view_140)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_163: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_160, mul_209);  mul_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_141: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_163, [8, 1, 384, 196])
    var_mean_47 = torch.ops.aten.var_mean.correction(view_141, [2, 3], correction = 0, keepdim = True)
    getitem_94: "f32[8, 1, 1, 1]" = var_mean_47[0]
    getitem_95: "f32[8, 1, 1, 1]" = var_mean_47[1];  var_mean_47 = None
    add_164: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
    rsqrt_47: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
    sub_71: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_141, getitem_95);  view_141 = None
    mul_210: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_47);  sub_71 = None
    view_142: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_210, [8, 384, 14, 14]);  mul_210 = None
    unsqueeze_282: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_143, 0);  primals_143 = None
    unsqueeze_283: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 2);  unsqueeze_282 = None
    unsqueeze_284: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 3);  unsqueeze_283 = None
    unsqueeze_285: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_142, 0)
    unsqueeze_286: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 2);  unsqueeze_285 = None
    unsqueeze_287: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 3);  unsqueeze_286 = None
    mul_211: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_142, unsqueeze_287);  view_142 = unsqueeze_287 = None
    add_165: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_211, unsqueeze_284);  mul_211 = unsqueeze_284 = None
    squeeze_94: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_95, [2, 3]);  getitem_95 = None
    squeeze_95: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_47, [2, 3]);  rsqrt_47 = None
    alias_94: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_94);  squeeze_94 = None
    alias_95: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_95);  squeeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_49: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_165, primals_317, primals_318, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_212: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_49, 0.5)
    mul_213: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_49, 0.7071067811865476)
    erf_23: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_213);  mul_213 = None
    add_166: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_214: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_212, add_166);  mul_212 = add_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_46: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_214);  mul_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_50: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_46, primals_319, primals_320, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_47: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_50);  convolution_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_143: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_144, [384, 1, 1]);  primals_144 = None
    mul_215: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_47, view_143)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_167: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_163, mul_215);  mul_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_144: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_167, [8, 1, 384, 196])
    var_mean_48 = torch.ops.aten.var_mean.correction(view_144, [2, 3], correction = 0, keepdim = True)
    getitem_96: "f32[8, 1, 1, 1]" = var_mean_48[0]
    getitem_97: "f32[8, 1, 1, 1]" = var_mean_48[1];  var_mean_48 = None
    add_168: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
    rsqrt_48: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    sub_72: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_144, getitem_97);  view_144 = None
    mul_216: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_48);  sub_72 = None
    view_145: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_216, [8, 384, 14, 14]);  mul_216 = None
    unsqueeze_288: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_146, 0);  primals_146 = None
    unsqueeze_289: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 2);  unsqueeze_288 = None
    unsqueeze_290: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 3);  unsqueeze_289 = None
    unsqueeze_291: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_145, 0)
    unsqueeze_292: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 2);  unsqueeze_291 = None
    unsqueeze_293: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 3);  unsqueeze_292 = None
    mul_217: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_145, unsqueeze_293);  view_145 = unsqueeze_293 = None
    add_169: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_217, unsqueeze_290);  mul_217 = unsqueeze_290 = None
    squeeze_96: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_97, [2, 3]);  getitem_97 = None
    squeeze_97: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_48, [2, 3]);  rsqrt_48 = None
    alias_96: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_96);  squeeze_96 = None
    alias_97: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_97);  squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_24: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_169, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_73: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_24, add_169);  avg_pool2d_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_146: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_147, [384, 1, 1]);  primals_147 = None
    mul_218: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_73, view_146)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_170: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_167, mul_218);  mul_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_147: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_170, [8, 1, 384, 196])
    var_mean_49 = torch.ops.aten.var_mean.correction(view_147, [2, 3], correction = 0, keepdim = True)
    getitem_98: "f32[8, 1, 1, 1]" = var_mean_49[0]
    getitem_99: "f32[8, 1, 1, 1]" = var_mean_49[1];  var_mean_49 = None
    add_171: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05);  getitem_98 = None
    rsqrt_49: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
    sub_74: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_147, getitem_99);  view_147 = None
    mul_219: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_49);  sub_74 = None
    view_148: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_219, [8, 384, 14, 14]);  mul_219 = None
    unsqueeze_294: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_149, 0);  primals_149 = None
    unsqueeze_295: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 2);  unsqueeze_294 = None
    unsqueeze_296: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 3);  unsqueeze_295 = None
    unsqueeze_297: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_148, 0)
    unsqueeze_298: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 2);  unsqueeze_297 = None
    unsqueeze_299: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 3);  unsqueeze_298 = None
    mul_220: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_148, unsqueeze_299);  view_148 = unsqueeze_299 = None
    add_172: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_220, unsqueeze_296);  mul_220 = unsqueeze_296 = None
    squeeze_98: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_99, [2, 3]);  getitem_99 = None
    squeeze_99: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_49, [2, 3]);  rsqrt_49 = None
    alias_98: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_98);  squeeze_98 = None
    alias_99: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_99);  squeeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_51: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_172, primals_321, primals_322, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_221: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_51, 0.5)
    mul_222: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_51, 0.7071067811865476)
    erf_24: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_222);  mul_222 = None
    add_173: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
    mul_223: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_221, add_173);  mul_221 = add_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_48: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_223);  mul_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_52: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_48, primals_323, primals_324, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_49: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_52);  convolution_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_149: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_150, [384, 1, 1]);  primals_150 = None
    mul_224: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_49, view_149)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_174: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_170, mul_224);  mul_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_150: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_174, [8, 1, 384, 196])
    var_mean_50 = torch.ops.aten.var_mean.correction(view_150, [2, 3], correction = 0, keepdim = True)
    getitem_100: "f32[8, 1, 1, 1]" = var_mean_50[0]
    getitem_101: "f32[8, 1, 1, 1]" = var_mean_50[1];  var_mean_50 = None
    add_175: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05);  getitem_100 = None
    rsqrt_50: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
    sub_75: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_150, getitem_101);  view_150 = None
    mul_225: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_50);  sub_75 = None
    view_151: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_225, [8, 384, 14, 14]);  mul_225 = None
    unsqueeze_300: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_152, 0);  primals_152 = None
    unsqueeze_301: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 2);  unsqueeze_300 = None
    unsqueeze_302: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 3);  unsqueeze_301 = None
    unsqueeze_303: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_151, 0)
    unsqueeze_304: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 2);  unsqueeze_303 = None
    unsqueeze_305: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 3);  unsqueeze_304 = None
    mul_226: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_151, unsqueeze_305);  view_151 = unsqueeze_305 = None
    add_176: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_226, unsqueeze_302);  mul_226 = unsqueeze_302 = None
    squeeze_100: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_101, [2, 3]);  getitem_101 = None
    squeeze_101: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_50, [2, 3]);  rsqrt_50 = None
    alias_100: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_100);  squeeze_100 = None
    alias_101: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_101);  squeeze_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_25: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_176, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_76: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_25, add_176);  avg_pool2d_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_152: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_153, [384, 1, 1]);  primals_153 = None
    mul_227: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_76, view_152)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_177: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_174, mul_227);  mul_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_153: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_177, [8, 1, 384, 196])
    var_mean_51 = torch.ops.aten.var_mean.correction(view_153, [2, 3], correction = 0, keepdim = True)
    getitem_102: "f32[8, 1, 1, 1]" = var_mean_51[0]
    getitem_103: "f32[8, 1, 1, 1]" = var_mean_51[1];  var_mean_51 = None
    add_178: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
    rsqrt_51: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
    sub_77: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_153, getitem_103);  view_153 = None
    mul_228: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_51);  sub_77 = None
    view_154: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_228, [8, 384, 14, 14]);  mul_228 = None
    unsqueeze_306: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_155, 0);  primals_155 = None
    unsqueeze_307: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, 2);  unsqueeze_306 = None
    unsqueeze_308: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 3);  unsqueeze_307 = None
    unsqueeze_309: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_154, 0)
    unsqueeze_310: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 2);  unsqueeze_309 = None
    unsqueeze_311: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 3);  unsqueeze_310 = None
    mul_229: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_154, unsqueeze_311);  view_154 = unsqueeze_311 = None
    add_179: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_229, unsqueeze_308);  mul_229 = unsqueeze_308 = None
    squeeze_102: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_103, [2, 3]);  getitem_103 = None
    squeeze_103: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_51, [2, 3]);  rsqrt_51 = None
    alias_102: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_102);  squeeze_102 = None
    alias_103: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_103);  squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_53: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_179, primals_325, primals_326, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_230: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_53, 0.5)
    mul_231: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_53, 0.7071067811865476)
    erf_25: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_231);  mul_231 = None
    add_180: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
    mul_232: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_230, add_180);  mul_230 = add_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_50: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_232);  mul_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_54: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_50, primals_327, primals_328, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_51: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_54);  convolution_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_155: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_156, [384, 1, 1]);  primals_156 = None
    mul_233: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_51, view_155)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_181: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_177, mul_233);  mul_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_156: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_181, [8, 1, 384, 196])
    var_mean_52 = torch.ops.aten.var_mean.correction(view_156, [2, 3], correction = 0, keepdim = True)
    getitem_104: "f32[8, 1, 1, 1]" = var_mean_52[0]
    getitem_105: "f32[8, 1, 1, 1]" = var_mean_52[1];  var_mean_52 = None
    add_182: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
    rsqrt_52: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
    sub_78: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_156, getitem_105);  view_156 = None
    mul_234: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_52);  sub_78 = None
    view_157: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_234, [8, 384, 14, 14]);  mul_234 = None
    unsqueeze_312: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_158, 0);  primals_158 = None
    unsqueeze_313: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 2);  unsqueeze_312 = None
    unsqueeze_314: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 3);  unsqueeze_313 = None
    unsqueeze_315: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_157, 0)
    unsqueeze_316: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 2);  unsqueeze_315 = None
    unsqueeze_317: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, 3);  unsqueeze_316 = None
    mul_235: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_157, unsqueeze_317);  view_157 = unsqueeze_317 = None
    add_183: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_235, unsqueeze_314);  mul_235 = unsqueeze_314 = None
    squeeze_104: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_105, [2, 3]);  getitem_105 = None
    squeeze_105: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_52, [2, 3]);  rsqrt_52 = None
    alias_104: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_104);  squeeze_104 = None
    alias_105: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_105);  squeeze_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_26: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_183, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_79: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_26, add_183);  avg_pool2d_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_158: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_159, [384, 1, 1]);  primals_159 = None
    mul_236: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_79, view_158)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_184: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_181, mul_236);  mul_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_159: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_184, [8, 1, 384, 196])
    var_mean_53 = torch.ops.aten.var_mean.correction(view_159, [2, 3], correction = 0, keepdim = True)
    getitem_106: "f32[8, 1, 1, 1]" = var_mean_53[0]
    getitem_107: "f32[8, 1, 1, 1]" = var_mean_53[1];  var_mean_53 = None
    add_185: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05);  getitem_106 = None
    rsqrt_53: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
    sub_80: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_159, getitem_107);  view_159 = None
    mul_237: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_53);  sub_80 = None
    view_160: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_237, [8, 384, 14, 14]);  mul_237 = None
    unsqueeze_318: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_161, 0);  primals_161 = None
    unsqueeze_319: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, 2);  unsqueeze_318 = None
    unsqueeze_320: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 3);  unsqueeze_319 = None
    unsqueeze_321: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_160, 0)
    unsqueeze_322: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 2);  unsqueeze_321 = None
    unsqueeze_323: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 3);  unsqueeze_322 = None
    mul_238: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_160, unsqueeze_323);  view_160 = unsqueeze_323 = None
    add_186: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_238, unsqueeze_320);  mul_238 = unsqueeze_320 = None
    squeeze_106: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_107, [2, 3]);  getitem_107 = None
    squeeze_107: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_53, [2, 3]);  rsqrt_53 = None
    alias_106: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_106);  squeeze_106 = None
    alias_107: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_107);  squeeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_55: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_186, primals_329, primals_330, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_239: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_55, 0.5)
    mul_240: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_55, 0.7071067811865476)
    erf_26: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_240);  mul_240 = None
    add_187: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
    mul_241: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_239, add_187);  mul_239 = add_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_52: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_241);  mul_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_56: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_52, primals_331, primals_332, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_53: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_56);  convolution_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_161: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_162, [384, 1, 1]);  primals_162 = None
    mul_242: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_53, view_161)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_188: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_184, mul_242);  mul_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_162: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_188, [8, 1, 384, 196])
    var_mean_54 = torch.ops.aten.var_mean.correction(view_162, [2, 3], correction = 0, keepdim = True)
    getitem_108: "f32[8, 1, 1, 1]" = var_mean_54[0]
    getitem_109: "f32[8, 1, 1, 1]" = var_mean_54[1];  var_mean_54 = None
    add_189: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
    rsqrt_54: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_189);  add_189 = None
    sub_81: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_162, getitem_109);  view_162 = None
    mul_243: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_54);  sub_81 = None
    view_163: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_243, [8, 384, 14, 14]);  mul_243 = None
    unsqueeze_324: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_164, 0);  primals_164 = None
    unsqueeze_325: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, 2);  unsqueeze_324 = None
    unsqueeze_326: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 3);  unsqueeze_325 = None
    unsqueeze_327: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_163, 0)
    unsqueeze_328: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 2);  unsqueeze_327 = None
    unsqueeze_329: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 3);  unsqueeze_328 = None
    mul_244: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_163, unsqueeze_329);  view_163 = unsqueeze_329 = None
    add_190: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_244, unsqueeze_326);  mul_244 = unsqueeze_326 = None
    squeeze_108: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_109, [2, 3]);  getitem_109 = None
    squeeze_109: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_54, [2, 3]);  rsqrt_54 = None
    alias_108: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_108);  squeeze_108 = None
    alias_109: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_109);  squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_27: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_190, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_82: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_27, add_190);  avg_pool2d_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_164: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_165, [384, 1, 1]);  primals_165 = None
    mul_245: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_82, view_164)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_191: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_188, mul_245);  mul_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_165: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_191, [8, 1, 384, 196])
    var_mean_55 = torch.ops.aten.var_mean.correction(view_165, [2, 3], correction = 0, keepdim = True)
    getitem_110: "f32[8, 1, 1, 1]" = var_mean_55[0]
    getitem_111: "f32[8, 1, 1, 1]" = var_mean_55[1];  var_mean_55 = None
    add_192: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
    rsqrt_55: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
    sub_83: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_165, getitem_111);  view_165 = None
    mul_246: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_55);  sub_83 = None
    view_166: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_246, [8, 384, 14, 14]);  mul_246 = None
    unsqueeze_330: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_167, 0);  primals_167 = None
    unsqueeze_331: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 2);  unsqueeze_330 = None
    unsqueeze_332: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 3);  unsqueeze_331 = None
    unsqueeze_333: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_166, 0)
    unsqueeze_334: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 2);  unsqueeze_333 = None
    unsqueeze_335: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 3);  unsqueeze_334 = None
    mul_247: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_166, unsqueeze_335);  view_166 = unsqueeze_335 = None
    add_193: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_247, unsqueeze_332);  mul_247 = unsqueeze_332 = None
    squeeze_110: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_111, [2, 3]);  getitem_111 = None
    squeeze_111: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_55, [2, 3]);  rsqrt_55 = None
    alias_110: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_110);  squeeze_110 = None
    alias_111: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_111);  squeeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_57: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_193, primals_333, primals_334, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_248: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_57, 0.5)
    mul_249: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_57, 0.7071067811865476)
    erf_27: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_249);  mul_249 = None
    add_194: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
    mul_250: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_248, add_194);  mul_248 = add_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_54: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_250);  mul_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_58: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_54, primals_335, primals_336, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_55: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_58);  convolution_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_167: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_168, [384, 1, 1]);  primals_168 = None
    mul_251: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_55, view_167)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_195: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_191, mul_251);  mul_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_168: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_195, [8, 1, 384, 196])
    var_mean_56 = torch.ops.aten.var_mean.correction(view_168, [2, 3], correction = 0, keepdim = True)
    getitem_112: "f32[8, 1, 1, 1]" = var_mean_56[0]
    getitem_113: "f32[8, 1, 1, 1]" = var_mean_56[1];  var_mean_56 = None
    add_196: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
    rsqrt_56: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
    sub_84: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_168, getitem_113);  view_168 = None
    mul_252: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_56);  sub_84 = None
    view_169: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_252, [8, 384, 14, 14]);  mul_252 = None
    unsqueeze_336: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_170, 0);  primals_170 = None
    unsqueeze_337: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 2);  unsqueeze_336 = None
    unsqueeze_338: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 3);  unsqueeze_337 = None
    unsqueeze_339: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_169, 0)
    unsqueeze_340: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 2);  unsqueeze_339 = None
    unsqueeze_341: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 3);  unsqueeze_340 = None
    mul_253: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_169, unsqueeze_341);  view_169 = unsqueeze_341 = None
    add_197: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_253, unsqueeze_338);  mul_253 = unsqueeze_338 = None
    squeeze_112: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_113, [2, 3]);  getitem_113 = None
    squeeze_113: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_56, [2, 3]);  rsqrt_56 = None
    alias_112: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_112);  squeeze_112 = None
    alias_113: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_113);  squeeze_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_28: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_197, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_85: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_28, add_197);  avg_pool2d_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_170: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_171, [384, 1, 1]);  primals_171 = None
    mul_254: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_85, view_170)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_198: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_195, mul_254);  mul_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_171: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_198, [8, 1, 384, 196])
    var_mean_57 = torch.ops.aten.var_mean.correction(view_171, [2, 3], correction = 0, keepdim = True)
    getitem_114: "f32[8, 1, 1, 1]" = var_mean_57[0]
    getitem_115: "f32[8, 1, 1, 1]" = var_mean_57[1];  var_mean_57 = None
    add_199: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-05);  getitem_114 = None
    rsqrt_57: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
    sub_86: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_171, getitem_115);  view_171 = None
    mul_255: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_57);  sub_86 = None
    view_172: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_255, [8, 384, 14, 14]);  mul_255 = None
    unsqueeze_342: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_173, 0);  primals_173 = None
    unsqueeze_343: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 2);  unsqueeze_342 = None
    unsqueeze_344: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 3);  unsqueeze_343 = None
    unsqueeze_345: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_172, 0)
    unsqueeze_346: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 2);  unsqueeze_345 = None
    unsqueeze_347: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 3);  unsqueeze_346 = None
    mul_256: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_172, unsqueeze_347);  view_172 = unsqueeze_347 = None
    add_200: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_256, unsqueeze_344);  mul_256 = unsqueeze_344 = None
    squeeze_114: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_115, [2, 3]);  getitem_115 = None
    squeeze_115: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_57, [2, 3]);  rsqrt_57 = None
    alias_114: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_114);  squeeze_114 = None
    alias_115: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_115);  squeeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_59: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_200, primals_337, primals_338, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_257: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_59, 0.5)
    mul_258: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_59, 0.7071067811865476)
    erf_28: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_258);  mul_258 = None
    add_201: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
    mul_259: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_257, add_201);  mul_257 = add_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_56: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_259);  mul_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_60: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_56, primals_339, primals_340, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_57: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_60);  convolution_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_173: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_174, [384, 1, 1]);  primals_174 = None
    mul_260: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_57, view_173)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_202: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_198, mul_260);  mul_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_174: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_202, [8, 1, 384, 196])
    var_mean_58 = torch.ops.aten.var_mean.correction(view_174, [2, 3], correction = 0, keepdim = True)
    getitem_116: "f32[8, 1, 1, 1]" = var_mean_58[0]
    getitem_117: "f32[8, 1, 1, 1]" = var_mean_58[1];  var_mean_58 = None
    add_203: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-05);  getitem_116 = None
    rsqrt_58: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
    sub_87: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_174, getitem_117);  view_174 = None
    mul_261: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_58);  sub_87 = None
    view_175: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_261, [8, 384, 14, 14]);  mul_261 = None
    unsqueeze_348: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_176, 0);  primals_176 = None
    unsqueeze_349: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 2);  unsqueeze_348 = None
    unsqueeze_350: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 3);  unsqueeze_349 = None
    unsqueeze_351: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_175, 0)
    unsqueeze_352: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 2);  unsqueeze_351 = None
    unsqueeze_353: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 3);  unsqueeze_352 = None
    mul_262: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_175, unsqueeze_353);  view_175 = unsqueeze_353 = None
    add_204: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_262, unsqueeze_350);  mul_262 = unsqueeze_350 = None
    squeeze_116: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_117, [2, 3]);  getitem_117 = None
    squeeze_117: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_58, [2, 3]);  rsqrt_58 = None
    alias_116: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_116);  squeeze_116 = None
    alias_117: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_117);  squeeze_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_29: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d.default(add_204, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_88: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_29, add_204);  avg_pool2d_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_176: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_177, [384, 1, 1]);  primals_177 = None
    mul_263: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_88, view_176)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_205: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_202, mul_263);  mul_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_177: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_205, [8, 1, 384, 196])
    var_mean_59 = torch.ops.aten.var_mean.correction(view_177, [2, 3], correction = 0, keepdim = True)
    getitem_118: "f32[8, 1, 1, 1]" = var_mean_59[0]
    getitem_119: "f32[8, 1, 1, 1]" = var_mean_59[1];  var_mean_59 = None
    add_206: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05);  getitem_118 = None
    rsqrt_59: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_206);  add_206 = None
    sub_89: "f32[8, 1, 384, 196]" = torch.ops.aten.sub.Tensor(view_177, getitem_119);  view_177 = None
    mul_264: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_59);  sub_89 = None
    view_178: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(mul_264, [8, 384, 14, 14]);  mul_264 = None
    unsqueeze_354: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_179, 0);  primals_179 = None
    unsqueeze_355: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, 2);  unsqueeze_354 = None
    unsqueeze_356: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 3);  unsqueeze_355 = None
    unsqueeze_357: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_178, 0)
    unsqueeze_358: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 2);  unsqueeze_357 = None
    unsqueeze_359: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 3);  unsqueeze_358 = None
    mul_265: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(view_178, unsqueeze_359);  view_178 = unsqueeze_359 = None
    add_207: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_265, unsqueeze_356);  mul_265 = unsqueeze_356 = None
    squeeze_118: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_119, [2, 3]);  getitem_119 = None
    squeeze_119: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_59, [2, 3]);  rsqrt_59 = None
    alias_118: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_118);  squeeze_118 = None
    alias_119: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_119);  squeeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_61: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_207, primals_341, primals_342, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_266: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_61, 0.5)
    mul_267: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_61, 0.7071067811865476)
    erf_29: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_267);  mul_267 = None
    add_208: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
    mul_268: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_266, add_208);  mul_266 = add_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_58: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_268);  mul_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_62: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_58, primals_343, primals_344, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_59: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_62);  convolution_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_179: "f32[384, 1, 1]" = torch.ops.aten.view.default(primals_180, [384, 1, 1]);  primals_180 = None
    mul_269: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(clone_59, view_179)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_209: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_205, mul_269);  mul_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:103, code: x = self.conv(x)
    convolution_63: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(add_209, primals_345, primals_346, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  primals_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_180: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(convolution_63, [8, 1, 768, 49])
    var_mean_60 = torch.ops.aten.var_mean.correction(view_180, [2, 3], correction = 0, keepdim = True)
    getitem_120: "f32[8, 1, 1, 1]" = var_mean_60[0]
    getitem_121: "f32[8, 1, 1, 1]" = var_mean_60[1];  var_mean_60 = None
    add_210: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-05);  getitem_120 = None
    rsqrt_60: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
    sub_90: "f32[8, 1, 768, 49]" = torch.ops.aten.sub.Tensor(view_180, getitem_121);  view_180 = None
    mul_270: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_60);  sub_90 = None
    view_181: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(mul_270, [8, 768, 7, 7]);  mul_270 = None
    unsqueeze_360: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_182, 0);  primals_182 = None
    unsqueeze_361: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, 2);  unsqueeze_360 = None
    unsqueeze_362: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 3);  unsqueeze_361 = None
    unsqueeze_363: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_181, 0)
    unsqueeze_364: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 2);  unsqueeze_363 = None
    unsqueeze_365: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 3);  unsqueeze_364 = None
    mul_271: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(view_181, unsqueeze_365);  view_181 = unsqueeze_365 = None
    add_211: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_271, unsqueeze_362);  mul_271 = unsqueeze_362 = None
    squeeze_120: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_121, [2, 3]);  getitem_121 = None
    squeeze_121: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_60, [2, 3]);  rsqrt_60 = None
    alias_120: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_120);  squeeze_120 = None
    alias_121: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_121);  squeeze_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_30: "f32[8, 768, 7, 7]" = torch.ops.aten.avg_pool2d.default(add_211, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_91: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(avg_pool2d_30, add_211);  avg_pool2d_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_182: "f32[768, 1, 1]" = torch.ops.aten.view.default(primals_183, [768, 1, 1]);  primals_183 = None
    mul_272: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_91, view_182)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_212: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(convolution_63, mul_272);  mul_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_183: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_212, [8, 1, 768, 49])
    var_mean_61 = torch.ops.aten.var_mean.correction(view_183, [2, 3], correction = 0, keepdim = True)
    getitem_122: "f32[8, 1, 1, 1]" = var_mean_61[0]
    getitem_123: "f32[8, 1, 1, 1]" = var_mean_61[1];  var_mean_61 = None
    add_213: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-05);  getitem_122 = None
    rsqrt_61: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
    sub_92: "f32[8, 1, 768, 49]" = torch.ops.aten.sub.Tensor(view_183, getitem_123);  view_183 = None
    mul_273: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_61);  sub_92 = None
    view_184: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(mul_273, [8, 768, 7, 7]);  mul_273 = None
    unsqueeze_366: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_185, 0);  primals_185 = None
    unsqueeze_367: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, 2);  unsqueeze_366 = None
    unsqueeze_368: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 3);  unsqueeze_367 = None
    unsqueeze_369: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_184, 0)
    unsqueeze_370: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 2);  unsqueeze_369 = None
    unsqueeze_371: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 3);  unsqueeze_370 = None
    mul_274: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(view_184, unsqueeze_371);  view_184 = unsqueeze_371 = None
    add_214: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_274, unsqueeze_368);  mul_274 = unsqueeze_368 = None
    squeeze_122: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_123, [2, 3]);  getitem_123 = None
    squeeze_123: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_61, [2, 3]);  rsqrt_61 = None
    alias_122: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_122);  squeeze_122 = None
    alias_123: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_123);  squeeze_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_64: "f32[8, 3072, 7, 7]" = torch.ops.aten.convolution.default(add_214, primals_347, primals_348, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_275: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_64, 0.5)
    mul_276: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_64, 0.7071067811865476)
    erf_30: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_276);  mul_276 = None
    add_215: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
    mul_277: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_275, add_215);  mul_275 = add_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_60: "f32[8, 3072, 7, 7]" = torch.ops.aten.clone.default(mul_277);  mul_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_65: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(clone_60, primals_349, primals_350, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_61: "f32[8, 768, 7, 7]" = torch.ops.aten.clone.default(convolution_65);  convolution_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_185: "f32[768, 1, 1]" = torch.ops.aten.view.default(primals_186, [768, 1, 1]);  primals_186 = None
    mul_278: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(clone_61, view_185)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_216: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_212, mul_278);  mul_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_186: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_216, [8, 1, 768, 49])
    var_mean_62 = torch.ops.aten.var_mean.correction(view_186, [2, 3], correction = 0, keepdim = True)
    getitem_124: "f32[8, 1, 1, 1]" = var_mean_62[0]
    getitem_125: "f32[8, 1, 1, 1]" = var_mean_62[1];  var_mean_62 = None
    add_217: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-05);  getitem_124 = None
    rsqrt_62: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_217);  add_217 = None
    sub_93: "f32[8, 1, 768, 49]" = torch.ops.aten.sub.Tensor(view_186, getitem_125);  view_186 = None
    mul_279: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_62);  sub_93 = None
    view_187: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(mul_279, [8, 768, 7, 7]);  mul_279 = None
    unsqueeze_372: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_188, 0);  primals_188 = None
    unsqueeze_373: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, 2);  unsqueeze_372 = None
    unsqueeze_374: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 3);  unsqueeze_373 = None
    unsqueeze_375: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_187, 0)
    unsqueeze_376: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 2);  unsqueeze_375 = None
    unsqueeze_377: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 3);  unsqueeze_376 = None
    mul_280: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(view_187, unsqueeze_377);  view_187 = unsqueeze_377 = None
    add_218: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_280, unsqueeze_374);  mul_280 = unsqueeze_374 = None
    squeeze_124: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_125, [2, 3]);  getitem_125 = None
    squeeze_125: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_62, [2, 3]);  rsqrt_62 = None
    alias_124: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_124);  squeeze_124 = None
    alias_125: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_125);  squeeze_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_31: "f32[8, 768, 7, 7]" = torch.ops.aten.avg_pool2d.default(add_218, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_94: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(avg_pool2d_31, add_218);  avg_pool2d_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_188: "f32[768, 1, 1]" = torch.ops.aten.view.default(primals_189, [768, 1, 1]);  primals_189 = None
    mul_281: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_94, view_188)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_219: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_216, mul_281);  mul_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_189: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_219, [8, 1, 768, 49])
    var_mean_63 = torch.ops.aten.var_mean.correction(view_189, [2, 3], correction = 0, keepdim = True)
    getitem_126: "f32[8, 1, 1, 1]" = var_mean_63[0]
    getitem_127: "f32[8, 1, 1, 1]" = var_mean_63[1];  var_mean_63 = None
    add_220: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-05);  getitem_126 = None
    rsqrt_63: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_220);  add_220 = None
    sub_95: "f32[8, 1, 768, 49]" = torch.ops.aten.sub.Tensor(view_189, getitem_127);  view_189 = None
    mul_282: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_63);  sub_95 = None
    view_190: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(mul_282, [8, 768, 7, 7]);  mul_282 = None
    unsqueeze_378: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_191, 0);  primals_191 = None
    unsqueeze_379: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 2);  unsqueeze_378 = None
    unsqueeze_380: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 3);  unsqueeze_379 = None
    unsqueeze_381: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_190, 0)
    unsqueeze_382: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 2);  unsqueeze_381 = None
    unsqueeze_383: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 3);  unsqueeze_382 = None
    mul_283: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(view_190, unsqueeze_383);  view_190 = unsqueeze_383 = None
    add_221: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_283, unsqueeze_380);  mul_283 = unsqueeze_380 = None
    squeeze_126: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_127, [2, 3]);  getitem_127 = None
    squeeze_127: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_63, [2, 3]);  rsqrt_63 = None
    alias_126: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_126);  squeeze_126 = None
    alias_127: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_127);  squeeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_66: "f32[8, 3072, 7, 7]" = torch.ops.aten.convolution.default(add_221, primals_351, primals_352, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_284: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_66, 0.5)
    mul_285: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_66, 0.7071067811865476)
    erf_31: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_285);  mul_285 = None
    add_222: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
    mul_286: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_284, add_222);  mul_284 = add_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_62: "f32[8, 3072, 7, 7]" = torch.ops.aten.clone.default(mul_286);  mul_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_67: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(clone_62, primals_353, primals_354, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_63: "f32[8, 768, 7, 7]" = torch.ops.aten.clone.default(convolution_67);  convolution_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_191: "f32[768, 1, 1]" = torch.ops.aten.view.default(primals_192, [768, 1, 1]);  primals_192 = None
    mul_287: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(clone_63, view_191)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_223: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_219, mul_287);  mul_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_192: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_223, [8, 1, 768, 49])
    var_mean_64 = torch.ops.aten.var_mean.correction(view_192, [2, 3], correction = 0, keepdim = True)
    getitem_128: "f32[8, 1, 1, 1]" = var_mean_64[0]
    getitem_129: "f32[8, 1, 1, 1]" = var_mean_64[1];  var_mean_64 = None
    add_224: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-05);  getitem_128 = None
    rsqrt_64: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_224);  add_224 = None
    sub_96: "f32[8, 1, 768, 49]" = torch.ops.aten.sub.Tensor(view_192, getitem_129);  view_192 = None
    mul_288: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_64);  sub_96 = None
    view_193: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(mul_288, [8, 768, 7, 7]);  mul_288 = None
    unsqueeze_384: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_194, 0);  primals_194 = None
    unsqueeze_385: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, 2);  unsqueeze_384 = None
    unsqueeze_386: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 3);  unsqueeze_385 = None
    unsqueeze_387: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_193, 0)
    unsqueeze_388: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 2);  unsqueeze_387 = None
    unsqueeze_389: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 3);  unsqueeze_388 = None
    mul_289: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(view_193, unsqueeze_389);  view_193 = unsqueeze_389 = None
    add_225: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_289, unsqueeze_386);  mul_289 = unsqueeze_386 = None
    squeeze_128: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_129, [2, 3]);  getitem_129 = None
    squeeze_129: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_64, [2, 3]);  rsqrt_64 = None
    alias_128: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_128);  squeeze_128 = None
    alias_129: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_129);  squeeze_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_32: "f32[8, 768, 7, 7]" = torch.ops.aten.avg_pool2d.default(add_225, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_97: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(avg_pool2d_32, add_225);  avg_pool2d_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_194: "f32[768, 1, 1]" = torch.ops.aten.view.default(primals_195, [768, 1, 1]);  primals_195 = None
    mul_290: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_97, view_194)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_226: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_223, mul_290);  mul_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_195: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_226, [8, 1, 768, 49])
    var_mean_65 = torch.ops.aten.var_mean.correction(view_195, [2, 3], correction = 0, keepdim = True)
    getitem_130: "f32[8, 1, 1, 1]" = var_mean_65[0]
    getitem_131: "f32[8, 1, 1, 1]" = var_mean_65[1];  var_mean_65 = None
    add_227: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
    rsqrt_65: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_227);  add_227 = None
    sub_98: "f32[8, 1, 768, 49]" = torch.ops.aten.sub.Tensor(view_195, getitem_131);  view_195 = None
    mul_291: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(sub_98, rsqrt_65);  sub_98 = None
    view_196: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(mul_291, [8, 768, 7, 7]);  mul_291 = None
    unsqueeze_390: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_197, 0);  primals_197 = None
    unsqueeze_391: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, 2);  unsqueeze_390 = None
    unsqueeze_392: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 3);  unsqueeze_391 = None
    unsqueeze_393: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_196, 0)
    unsqueeze_394: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 2);  unsqueeze_393 = None
    unsqueeze_395: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 3);  unsqueeze_394 = None
    mul_292: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(view_196, unsqueeze_395);  view_196 = unsqueeze_395 = None
    add_228: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_292, unsqueeze_392);  mul_292 = unsqueeze_392 = None
    squeeze_130: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_131, [2, 3]);  getitem_131 = None
    squeeze_131: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_65, [2, 3]);  rsqrt_65 = None
    alias_130: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_130);  squeeze_130 = None
    alias_131: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_131);  squeeze_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_68: "f32[8, 3072, 7, 7]" = torch.ops.aten.convolution.default(add_228, primals_355, primals_356, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_293: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_68, 0.5)
    mul_294: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_68, 0.7071067811865476)
    erf_32: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_294);  mul_294 = None
    add_229: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
    mul_295: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_293, add_229);  mul_293 = add_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_64: "f32[8, 3072, 7, 7]" = torch.ops.aten.clone.default(mul_295);  mul_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_69: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(clone_64, primals_357, primals_358, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_65: "f32[8, 768, 7, 7]" = torch.ops.aten.clone.default(convolution_69);  convolution_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_197: "f32[768, 1, 1]" = torch.ops.aten.view.default(primals_198, [768, 1, 1]);  primals_198 = None
    mul_296: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(clone_65, view_197)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_230: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_226, mul_296);  mul_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_198: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_230, [8, 1, 768, 49])
    var_mean_66 = torch.ops.aten.var_mean.correction(view_198, [2, 3], correction = 0, keepdim = True)
    getitem_132: "f32[8, 1, 1, 1]" = var_mean_66[0]
    getitem_133: "f32[8, 1, 1, 1]" = var_mean_66[1];  var_mean_66 = None
    add_231: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
    rsqrt_66: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_231);  add_231 = None
    sub_99: "f32[8, 1, 768, 49]" = torch.ops.aten.sub.Tensor(view_198, getitem_133);  view_198 = None
    mul_297: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(sub_99, rsqrt_66);  sub_99 = None
    view_199: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(mul_297, [8, 768, 7, 7]);  mul_297 = None
    unsqueeze_396: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_200, 0);  primals_200 = None
    unsqueeze_397: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, 2);  unsqueeze_396 = None
    unsqueeze_398: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 3);  unsqueeze_397 = None
    unsqueeze_399: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_199, 0)
    unsqueeze_400: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 2);  unsqueeze_399 = None
    unsqueeze_401: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 3);  unsqueeze_400 = None
    mul_298: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(view_199, unsqueeze_401);  view_199 = unsqueeze_401 = None
    add_232: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_298, unsqueeze_398);  mul_298 = unsqueeze_398 = None
    squeeze_132: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_133, [2, 3]);  getitem_133 = None
    squeeze_133: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_66, [2, 3]);  rsqrt_66 = None
    alias_132: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_132);  squeeze_132 = None
    alias_133: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_133);  squeeze_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_33: "f32[8, 768, 7, 7]" = torch.ops.aten.avg_pool2d.default(add_232, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_100: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(avg_pool2d_33, add_232);  avg_pool2d_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_200: "f32[768, 1, 1]" = torch.ops.aten.view.default(primals_201, [768, 1, 1]);  primals_201 = None
    mul_299: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_100, view_200)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_233: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_230, mul_299);  mul_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_201: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_233, [8, 1, 768, 49])
    var_mean_67 = torch.ops.aten.var_mean.correction(view_201, [2, 3], correction = 0, keepdim = True)
    getitem_134: "f32[8, 1, 1, 1]" = var_mean_67[0]
    getitem_135: "f32[8, 1, 1, 1]" = var_mean_67[1];  var_mean_67 = None
    add_234: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_134, 1e-05);  getitem_134 = None
    rsqrt_67: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_234);  add_234 = None
    sub_101: "f32[8, 1, 768, 49]" = torch.ops.aten.sub.Tensor(view_201, getitem_135);  view_201 = None
    mul_300: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(sub_101, rsqrt_67);  sub_101 = None
    view_202: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(mul_300, [8, 768, 7, 7]);  mul_300 = None
    unsqueeze_402: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_203, 0);  primals_203 = None
    unsqueeze_403: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 2);  unsqueeze_402 = None
    unsqueeze_404: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 3);  unsqueeze_403 = None
    unsqueeze_405: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_202, 0)
    unsqueeze_406: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 2);  unsqueeze_405 = None
    unsqueeze_407: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 3);  unsqueeze_406 = None
    mul_301: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(view_202, unsqueeze_407);  view_202 = unsqueeze_407 = None
    add_235: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_301, unsqueeze_404);  mul_301 = unsqueeze_404 = None
    squeeze_134: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_135, [2, 3]);  getitem_135 = None
    squeeze_135: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_67, [2, 3]);  rsqrt_67 = None
    alias_134: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_134);  squeeze_134 = None
    alias_135: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_135);  squeeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_70: "f32[8, 3072, 7, 7]" = torch.ops.aten.convolution.default(add_235, primals_359, primals_360, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_302: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_70, 0.5)
    mul_303: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_70, 0.7071067811865476)
    erf_33: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_303);  mul_303 = None
    add_236: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
    mul_304: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_302, add_236);  mul_302 = add_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_66: "f32[8, 3072, 7, 7]" = torch.ops.aten.clone.default(mul_304);  mul_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_71: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(clone_66, primals_361, primals_362, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_67: "f32[8, 768, 7, 7]" = torch.ops.aten.clone.default(convolution_71);  convolution_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_203: "f32[768, 1, 1]" = torch.ops.aten.view.default(primals_204, [768, 1, 1]);  primals_204 = None
    mul_305: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(clone_67, view_203)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_237: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_233, mul_305);  mul_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_204: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_237, [8, 1, 768, 49])
    var_mean_68 = torch.ops.aten.var_mean.correction(view_204, [2, 3], correction = 0, keepdim = True)
    getitem_136: "f32[8, 1, 1, 1]" = var_mean_68[0]
    getitem_137: "f32[8, 1, 1, 1]" = var_mean_68[1];  var_mean_68 = None
    add_238: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_136, 1e-05);  getitem_136 = None
    rsqrt_68: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_238);  add_238 = None
    sub_102: "f32[8, 1, 768, 49]" = torch.ops.aten.sub.Tensor(view_204, getitem_137);  view_204 = None
    mul_306: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(sub_102, rsqrt_68);  sub_102 = None
    view_205: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(mul_306, [8, 768, 7, 7]);  mul_306 = None
    unsqueeze_408: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_206, 0);  primals_206 = None
    unsqueeze_409: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, 2);  unsqueeze_408 = None
    unsqueeze_410: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 3);  unsqueeze_409 = None
    unsqueeze_411: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_205, 0)
    unsqueeze_412: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 2);  unsqueeze_411 = None
    unsqueeze_413: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 3);  unsqueeze_412 = None
    mul_307: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(view_205, unsqueeze_413);  view_205 = unsqueeze_413 = None
    add_239: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_307, unsqueeze_410);  mul_307 = unsqueeze_410 = None
    squeeze_136: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_137, [2, 3]);  getitem_137 = None
    squeeze_137: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_68, [2, 3]);  rsqrt_68 = None
    alias_136: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_136);  squeeze_136 = None
    alias_137: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_137);  squeeze_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_34: "f32[8, 768, 7, 7]" = torch.ops.aten.avg_pool2d.default(add_239, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_103: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(avg_pool2d_34, add_239);  avg_pool2d_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_206: "f32[768, 1, 1]" = torch.ops.aten.view.default(primals_207, [768, 1, 1]);  primals_207 = None
    mul_308: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_103, view_206)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_240: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_237, mul_308);  mul_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_207: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_240, [8, 1, 768, 49])
    var_mean_69 = torch.ops.aten.var_mean.correction(view_207, [2, 3], correction = 0, keepdim = True)
    getitem_138: "f32[8, 1, 1, 1]" = var_mean_69[0]
    getitem_139: "f32[8, 1, 1, 1]" = var_mean_69[1];  var_mean_69 = None
    add_241: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_138, 1e-05);  getitem_138 = None
    rsqrt_69: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_241);  add_241 = None
    sub_104: "f32[8, 1, 768, 49]" = torch.ops.aten.sub.Tensor(view_207, getitem_139);  view_207 = None
    mul_309: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(sub_104, rsqrt_69);  sub_104 = None
    view_208: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(mul_309, [8, 768, 7, 7]);  mul_309 = None
    unsqueeze_414: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_209, 0);  primals_209 = None
    unsqueeze_415: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 2);  unsqueeze_414 = None
    unsqueeze_416: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 3);  unsqueeze_415 = None
    unsqueeze_417: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_208, 0)
    unsqueeze_418: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 2);  unsqueeze_417 = None
    unsqueeze_419: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 3);  unsqueeze_418 = None
    mul_310: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(view_208, unsqueeze_419);  view_208 = unsqueeze_419 = None
    add_242: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_310, unsqueeze_416);  mul_310 = unsqueeze_416 = None
    squeeze_138: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_139, [2, 3]);  getitem_139 = None
    squeeze_139: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_69, [2, 3]);  rsqrt_69 = None
    alias_138: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_138);  squeeze_138 = None
    alias_139: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_139);  squeeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_72: "f32[8, 3072, 7, 7]" = torch.ops.aten.convolution.default(add_242, primals_363, primals_364, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_311: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_72, 0.5)
    mul_312: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_72, 0.7071067811865476)
    erf_34: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_312);  mul_312 = None
    add_243: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
    mul_313: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_311, add_243);  mul_311 = add_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_68: "f32[8, 3072, 7, 7]" = torch.ops.aten.clone.default(mul_313);  mul_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_73: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(clone_68, primals_365, primals_366, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_69: "f32[8, 768, 7, 7]" = torch.ops.aten.clone.default(convolution_73);  convolution_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_209: "f32[768, 1, 1]" = torch.ops.aten.view.default(primals_210, [768, 1, 1]);  primals_210 = None
    mul_314: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(clone_69, view_209)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_244: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_240, mul_314);  mul_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_210: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_244, [8, 1, 768, 49])
    var_mean_70 = torch.ops.aten.var_mean.correction(view_210, [2, 3], correction = 0, keepdim = True)
    getitem_140: "f32[8, 1, 1, 1]" = var_mean_70[0]
    getitem_141: "f32[8, 1, 1, 1]" = var_mean_70[1];  var_mean_70 = None
    add_245: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_140, 1e-05);  getitem_140 = None
    rsqrt_70: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_245);  add_245 = None
    sub_105: "f32[8, 1, 768, 49]" = torch.ops.aten.sub.Tensor(view_210, getitem_141);  view_210 = None
    mul_315: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(sub_105, rsqrt_70);  sub_105 = None
    view_211: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(mul_315, [8, 768, 7, 7]);  mul_315 = None
    unsqueeze_420: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_212, 0);  primals_212 = None
    unsqueeze_421: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 2);  unsqueeze_420 = None
    unsqueeze_422: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 3);  unsqueeze_421 = None
    unsqueeze_423: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_211, 0)
    unsqueeze_424: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 2);  unsqueeze_423 = None
    unsqueeze_425: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 3);  unsqueeze_424 = None
    mul_316: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(view_211, unsqueeze_425);  view_211 = unsqueeze_425 = None
    add_246: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_316, unsqueeze_422);  mul_316 = unsqueeze_422 = None
    squeeze_140: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_141, [2, 3]);  getitem_141 = None
    squeeze_141: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_70, [2, 3]);  rsqrt_70 = None
    alias_140: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_140);  squeeze_140 = None
    alias_141: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_141);  squeeze_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_35: "f32[8, 768, 7, 7]" = torch.ops.aten.avg_pool2d.default(add_246, [3, 3], [1, 1], [1, 1], False, False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_106: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(avg_pool2d_35, add_246);  avg_pool2d_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_212: "f32[768, 1, 1]" = torch.ops.aten.view.default(primals_213, [768, 1, 1]);  primals_213 = None
    mul_317: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_106, view_212)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    add_247: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_244, mul_317);  mul_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    view_213: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_247, [8, 1, 768, 49])
    var_mean_71 = torch.ops.aten.var_mean.correction(view_213, [2, 3], correction = 0, keepdim = True)
    getitem_142: "f32[8, 1, 1, 1]" = var_mean_71[0]
    getitem_143: "f32[8, 1, 1, 1]" = var_mean_71[1];  var_mean_71 = None
    add_248: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-05);  getitem_142 = None
    rsqrt_71: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_248);  add_248 = None
    sub_107: "f32[8, 1, 768, 49]" = torch.ops.aten.sub.Tensor(view_213, getitem_143);  view_213 = None
    mul_318: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(sub_107, rsqrt_71);  sub_107 = None
    view_214: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(mul_318, [8, 768, 7, 7]);  mul_318 = None
    unsqueeze_426: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_215, 0);  primals_215 = None
    unsqueeze_427: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 2);  unsqueeze_426 = None
    unsqueeze_428: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 3);  unsqueeze_427 = None
    unsqueeze_429: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_214, 0)
    unsqueeze_430: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 2);  unsqueeze_429 = None
    unsqueeze_431: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 3);  unsqueeze_430 = None
    mul_319: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(view_214, unsqueeze_431);  view_214 = unsqueeze_431 = None
    add_249: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_319, unsqueeze_428);  mul_319 = unsqueeze_428 = None
    squeeze_142: "f32[8, 1]" = torch.ops.aten.squeeze.dims(getitem_143, [2, 3]);  getitem_143 = None
    squeeze_143: "f32[8, 1]" = torch.ops.aten.squeeze.dims(rsqrt_71, [2, 3]);  rsqrt_71 = None
    alias_142: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_142);  squeeze_142 = None
    alias_143: "f32[8, 1]" = torch.ops.aten.alias.default(squeeze_143);  squeeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    convolution_74: "f32[8, 3072, 7, 7]" = torch.ops.aten.convolution.default(add_249, primals_367, primals_368, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_320: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_74, 0.5)
    mul_321: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_74, 0.7071067811865476)
    erf_35: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_321);  mul_321 = None
    add_250: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
    mul_322: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_320, add_250);  mul_320 = add_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_70: "f32[8, 3072, 7, 7]" = torch.ops.aten.clone.default(mul_322);  mul_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    convolution_75: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(clone_70, primals_369, primals_370, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_71: "f32[8, 768, 7, 7]" = torch.ops.aten.clone.default(convolution_75);  convolution_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    view_215: "f32[768, 1, 1]" = torch.ops.aten.view.default(primals_216, [768, 1, 1]);  primals_216 = None
    mul_323: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(clone_71, view_215)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    add_251: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_247, mul_323);  mul_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean: "f32[8, 768, 1, 1]" = torch.ops.aten.mean.dim(add_251, [-1, -2], True);  add_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    permute: "f32[8, 1, 1, 768]" = torch.ops.aten.permute.default(mean, [0, 2, 3, 1]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_72 = torch.ops.aten.var_mean.correction(permute, [3], correction = 0, keepdim = True)
    getitem_144: "f32[8, 1, 1, 1]" = var_mean_72[0]
    getitem_145: "f32[8, 1, 1, 1]" = var_mean_72[1];  var_mean_72 = None
    add_252: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-06);  getitem_144 = None
    rsqrt_72: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_252);  add_252 = None
    sub_108: "f32[8, 1, 1, 768]" = torch.ops.aten.sub.Tensor(permute, getitem_145)
    mul_324: "f32[8, 1, 1, 768]" = torch.ops.aten.mul.Tensor(sub_108, rsqrt_72);  sub_108 = None
    mul_325: "f32[8, 1, 1, 768]" = torch.ops.aten.mul.Tensor(mul_324, primals_217);  mul_324 = None
    add_253: "f32[8, 1, 1, 768]" = torch.ops.aten.add.Tensor(mul_325, primals_218);  mul_325 = primals_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    permute_1: "f32[8, 768, 1, 1]" = torch.ops.aten.permute.default(add_253, [0, 3, 1, 2]);  add_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:600, code: x = self.head.flatten(x)
    view_216: "f32[8, 768]" = torch.ops.aten.view.default(permute_1, [8, 768]);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:602, code: return x if pre_logits else self.head.fc(x)
    permute_2: "f32[768, 1000]" = torch.ops.aten.permute.default(primals_371, [1, 0]);  primals_371 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_372, view_216, permute_2);  primals_372 = None
    permute_3: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    mm: "f32[8, 768]" = torch.ops.aten.mm.default(tangents_1, permute_3);  permute_3 = None
    permute_4: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 768]" = torch.ops.aten.mm.default(permute_4, view_216);  permute_4 = view_216 = None
    permute_5: "f32[768, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_217: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_6: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:600, code: x = self.head.flatten(x)
    view_218: "f32[8, 768, 1, 1]" = torch.ops.aten.view.default(mm, [8, 768, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    permute_7: "f32[8, 1, 1, 768]" = torch.ops.aten.permute.default(view_218, [0, 2, 3, 1]);  view_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_109: "f32[8, 1, 1, 768]" = torch.ops.aten.sub.Tensor(permute, getitem_145);  permute = getitem_145 = None
    mul_326: "f32[8, 1, 1, 768]" = torch.ops.aten.mul.Tensor(sub_109, rsqrt_72);  sub_109 = None
    mul_327: "f32[8, 1, 1, 768]" = torch.ops.aten.mul.Tensor(permute_7, primals_217);  primals_217 = None
    mul_328: "f32[8, 1, 1, 768]" = torch.ops.aten.mul.Tensor(mul_327, 768)
    sum_2: "f32[8, 1, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_327, [3], True)
    mul_329: "f32[8, 1, 1, 768]" = torch.ops.aten.mul.Tensor(mul_327, mul_326);  mul_327 = None
    sum_3: "f32[8, 1, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_329, [3], True);  mul_329 = None
    mul_330: "f32[8, 1, 1, 768]" = torch.ops.aten.mul.Tensor(mul_326, sum_3);  sum_3 = None
    sub_110: "f32[8, 1, 1, 768]" = torch.ops.aten.sub.Tensor(mul_328, sum_2);  mul_328 = sum_2 = None
    sub_111: "f32[8, 1, 1, 768]" = torch.ops.aten.sub.Tensor(sub_110, mul_330);  sub_110 = mul_330 = None
    div: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_72, 768);  rsqrt_72 = None
    mul_331: "f32[8, 1, 1, 768]" = torch.ops.aten.mul.Tensor(div, sub_111);  div = sub_111 = None
    mul_332: "f32[8, 1, 1, 768]" = torch.ops.aten.mul.Tensor(permute_7, mul_326);  mul_326 = None
    sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_332, [0, 1, 2]);  mul_332 = None
    sum_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_7, [0, 1, 2]);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    permute_8: "f32[8, 768, 1, 1]" = torch.ops.aten.permute.default(mul_331, [0, 3, 1, 2]);  mul_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 768, 7, 7]" = torch.ops.aten.expand.default(permute_8, [8, 768, 7, 7]);  permute_8 = None
    div_1: "f32[8, 768, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_333: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(div_1, clone_71);  clone_71 = None
    mul_334: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(div_1, view_215);  view_215 = None
    sum_6: "f32[1, 768, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_333, [0, 2, 3], True);  mul_333 = None
    view_219: "f32[768, 1, 1]" = torch.ops.aten.view.default(sum_6, [768, 1, 1]);  sum_6 = None
    view_220: "f32[768]" = torch.ops.aten.view.default(view_219, [768]);  view_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_7: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_334, [0, 2, 3])
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_334, clone_70, primals_369, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_334 = clone_70 = primals_369 = None
    getitem_146: "f32[8, 3072, 7, 7]" = convolution_backward[0]
    getitem_147: "f32[768, 3072, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_335: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_74, 0.7071067811865476)
    erf_36: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_335);  mul_335 = None
    add_254: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
    mul_336: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(add_254, 0.5);  add_254 = None
    mul_337: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_74, convolution_74)
    mul_338: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_337, -0.5);  mul_337 = None
    exp: "f32[8, 3072, 7, 7]" = torch.ops.aten.exp.default(mul_338);  mul_338 = None
    mul_339: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
    mul_340: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_74, mul_339);  convolution_74 = mul_339 = None
    add_255: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(mul_336, mul_340);  mul_336 = mul_340 = None
    mul_341: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_146, add_255);  getitem_146 = add_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_8: "f32[3072]" = torch.ops.aten.sum.dim_IntList(mul_341, [0, 2, 3])
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_341, add_249, primals_367, [3072], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_341 = add_249 = primals_367 = None
    getitem_149: "f32[8, 768, 7, 7]" = convolution_backward_1[0]
    getitem_150: "f32[3072, 768, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_144: "f32[8, 1]" = torch.ops.aten.alias.default(alias_142);  alias_142 = None
    alias_145: "f32[8, 1]" = torch.ops.aten.alias.default(alias_143);  alias_143 = None
    mul_342: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_149, add_247)
    view_221: "f32[8, 768, 49]" = torch.ops.aten.view.default(mul_342, [8, 768, 49]);  mul_342 = None
    sum_9: "f32[8, 768]" = torch.ops.aten.sum.dim_IntList(view_221, [2]);  view_221 = None
    view_222: "f32[8, 768, 49]" = torch.ops.aten.view.default(getitem_149, [8, 768, 49])
    sum_10: "f32[8, 768]" = torch.ops.aten.sum.dim_IntList(view_222, [2]);  view_222 = None
    unsqueeze_432: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_214, 0)
    mul_343: "f32[8, 768]" = torch.ops.aten.mul.Tensor(sum_9, unsqueeze_432);  unsqueeze_432 = None
    view_223: "f32[8, 1, 768]" = torch.ops.aten.view.default(mul_343, [8, 1, 768]);  mul_343 = None
    sum_11: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_223, [2]);  view_223 = None
    unsqueeze_433: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_214, 0)
    mul_344: "f32[8, 768]" = torch.ops.aten.mul.Tensor(sum_10, unsqueeze_433);  unsqueeze_433 = None
    view_224: "f32[8, 1, 768]" = torch.ops.aten.view.default(mul_344, [8, 1, 768]);  mul_344 = None
    sum_12: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_224, [2]);  view_224 = None
    unsqueeze_434: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_145, -1)
    view_225: "f32[1, 1, 768]" = torch.ops.aten.view.default(primals_214, [1, 1, 768]);  primals_214 = None
    mul_345: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(unsqueeze_434, view_225);  unsqueeze_434 = view_225 = None
    mul_346: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_12, alias_144)
    sub_112: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_346, sum_11);  mul_346 = sum_11 = None
    mul_347: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_112, alias_145);  sub_112 = None
    mul_348: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_347, alias_145);  mul_347 = None
    mul_349: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_348, alias_145);  mul_348 = None
    mul_350: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_349, 2.657312925170068e-05);  mul_349 = None
    neg: "f32[8, 1]" = torch.ops.aten.neg.default(mul_350)
    mul_351: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg, alias_144);  neg = None
    mul_352: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_12, alias_145);  sum_12 = None
    mul_353: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_352, 2.657312925170068e-05);  mul_352 = None
    sub_113: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_351, mul_353);  mul_351 = mul_353 = None
    unsqueeze_435: "f32[8, 1, 768, 1]" = torch.ops.aten.unsqueeze.default(mul_345, -1);  mul_345 = None
    unsqueeze_436: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_350, -1);  mul_350 = None
    unsqueeze_437: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
    unsqueeze_438: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_113, -1);  sub_113 = None
    unsqueeze_439: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
    view_226: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(getitem_149, [8, 1, 768, 49]);  getitem_149 = None
    mul_354: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(view_226, unsqueeze_435);  view_226 = unsqueeze_435 = None
    view_227: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_247, [8, 1, 768, 49]);  add_247 = None
    mul_355: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(view_227, unsqueeze_437);  view_227 = unsqueeze_437 = None
    add_256: "f32[8, 1, 768, 49]" = torch.ops.aten.add.Tensor(mul_354, mul_355);  mul_354 = mul_355 = None
    add_257: "f32[8, 1, 768, 49]" = torch.ops.aten.add.Tensor(add_256, unsqueeze_439);  add_256 = unsqueeze_439 = None
    view_228: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(add_257, [8, 768, 7, 7]);  add_257 = None
    view_229: "f32[8, 1, 768]" = torch.ops.aten.view.default(sum_9, [8, 1, 768]);  sum_9 = None
    view_230: "f32[8, 1, 768]" = torch.ops.aten.view.default(sum_10, [8, 1, 768])
    unsqueeze_440: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_144, -1);  alias_144 = None
    mul_356: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(view_230, unsqueeze_440);  view_230 = unsqueeze_440 = None
    sub_114: "f32[8, 1, 768]" = torch.ops.aten.sub.Tensor(view_229, mul_356);  view_229 = mul_356 = None
    unsqueeze_441: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_145, -1);  alias_145 = None
    mul_357: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_441);  sub_114 = unsqueeze_441 = None
    sum_13: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(mul_357, [0]);  mul_357 = None
    view_231: "f32[768]" = torch.ops.aten.view.default(sum_13, [768]);  sum_13 = None
    sum_14: "f32[768]" = torch.ops.aten.sum.dim_IntList(sum_10, [0]);  sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_258: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(div_1, view_228);  div_1 = view_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_358: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_258, sub_106);  sub_106 = None
    mul_359: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_258, view_212);  view_212 = None
    sum_15: "f32[1, 768, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_358, [0, 2, 3], True);  mul_358 = None
    view_232: "f32[768, 1, 1]" = torch.ops.aten.view.default(sum_15, [768, 1, 1]);  sum_15 = None
    view_233: "f32[768]" = torch.ops.aten.view.default(view_232, [768]);  view_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_1: "f32[8, 768, 7, 7]" = torch.ops.aten.neg.default(mul_359)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward: "f32[8, 768, 7, 7]" = torch.ops.aten.avg_pool2d_backward.default(mul_359, add_246, [3, 3], [1, 1], [1, 1], False, False, None);  mul_359 = add_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_259: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(neg_1, avg_pool2d_backward);  neg_1 = avg_pool2d_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_146: "f32[8, 1]" = torch.ops.aten.alias.default(alias_140);  alias_140 = None
    alias_147: "f32[8, 1]" = torch.ops.aten.alias.default(alias_141);  alias_141 = None
    mul_360: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_259, add_244)
    view_234: "f32[8, 768, 49]" = torch.ops.aten.view.default(mul_360, [8, 768, 49]);  mul_360 = None
    sum_16: "f32[8, 768]" = torch.ops.aten.sum.dim_IntList(view_234, [2]);  view_234 = None
    view_235: "f32[8, 768, 49]" = torch.ops.aten.view.default(add_259, [8, 768, 49])
    sum_17: "f32[8, 768]" = torch.ops.aten.sum.dim_IntList(view_235, [2]);  view_235 = None
    unsqueeze_442: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_211, 0)
    mul_361: "f32[8, 768]" = torch.ops.aten.mul.Tensor(sum_16, unsqueeze_442);  unsqueeze_442 = None
    view_236: "f32[8, 1, 768]" = torch.ops.aten.view.default(mul_361, [8, 1, 768]);  mul_361 = None
    sum_18: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_236, [2]);  view_236 = None
    unsqueeze_443: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_211, 0)
    mul_362: "f32[8, 768]" = torch.ops.aten.mul.Tensor(sum_17, unsqueeze_443);  unsqueeze_443 = None
    view_237: "f32[8, 1, 768]" = torch.ops.aten.view.default(mul_362, [8, 1, 768]);  mul_362 = None
    sum_19: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_237, [2]);  view_237 = None
    unsqueeze_444: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_147, -1)
    view_238: "f32[1, 1, 768]" = torch.ops.aten.view.default(primals_211, [1, 1, 768]);  primals_211 = None
    mul_363: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(unsqueeze_444, view_238);  unsqueeze_444 = view_238 = None
    mul_364: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_19, alias_146)
    sub_115: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_364, sum_18);  mul_364 = sum_18 = None
    mul_365: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_115, alias_147);  sub_115 = None
    mul_366: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_365, alias_147);  mul_365 = None
    mul_367: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_366, alias_147);  mul_366 = None
    mul_368: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_367, 2.657312925170068e-05);  mul_367 = None
    neg_2: "f32[8, 1]" = torch.ops.aten.neg.default(mul_368)
    mul_369: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_2, alias_146);  neg_2 = None
    mul_370: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_19, alias_147);  sum_19 = None
    mul_371: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_370, 2.657312925170068e-05);  mul_370 = None
    sub_116: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_369, mul_371);  mul_369 = mul_371 = None
    unsqueeze_445: "f32[8, 1, 768, 1]" = torch.ops.aten.unsqueeze.default(mul_363, -1);  mul_363 = None
    unsqueeze_446: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_368, -1);  mul_368 = None
    unsqueeze_447: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
    unsqueeze_448: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_116, -1);  sub_116 = None
    unsqueeze_449: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, -1);  unsqueeze_448 = None
    view_239: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_259, [8, 1, 768, 49]);  add_259 = None
    mul_372: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(view_239, unsqueeze_445);  view_239 = unsqueeze_445 = None
    view_240: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_244, [8, 1, 768, 49]);  add_244 = None
    mul_373: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(view_240, unsqueeze_447);  view_240 = unsqueeze_447 = None
    add_260: "f32[8, 1, 768, 49]" = torch.ops.aten.add.Tensor(mul_372, mul_373);  mul_372 = mul_373 = None
    add_261: "f32[8, 1, 768, 49]" = torch.ops.aten.add.Tensor(add_260, unsqueeze_449);  add_260 = unsqueeze_449 = None
    view_241: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(add_261, [8, 768, 7, 7]);  add_261 = None
    view_242: "f32[8, 1, 768]" = torch.ops.aten.view.default(sum_16, [8, 1, 768]);  sum_16 = None
    view_243: "f32[8, 1, 768]" = torch.ops.aten.view.default(sum_17, [8, 1, 768])
    unsqueeze_450: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_146, -1);  alias_146 = None
    mul_374: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(view_243, unsqueeze_450);  view_243 = unsqueeze_450 = None
    sub_117: "f32[8, 1, 768]" = torch.ops.aten.sub.Tensor(view_242, mul_374);  view_242 = mul_374 = None
    unsqueeze_451: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_147, -1);  alias_147 = None
    mul_375: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_451);  sub_117 = unsqueeze_451 = None
    sum_20: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(mul_375, [0]);  mul_375 = None
    view_244: "f32[768]" = torch.ops.aten.view.default(sum_20, [768]);  sum_20 = None
    sum_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(sum_17, [0]);  sum_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_262: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_258, view_241);  add_258 = view_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_376: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_262, clone_69);  clone_69 = None
    mul_377: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_262, view_209);  view_209 = None
    sum_22: "f32[1, 768, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_376, [0, 2, 3], True);  mul_376 = None
    view_245: "f32[768, 1, 1]" = torch.ops.aten.view.default(sum_22, [768, 1, 1]);  sum_22 = None
    view_246: "f32[768]" = torch.ops.aten.view.default(view_245, [768]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_23: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_377, [0, 2, 3])
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_377, clone_68, primals_365, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_377 = clone_68 = primals_365 = None
    getitem_152: "f32[8, 3072, 7, 7]" = convolution_backward_2[0]
    getitem_153: "f32[768, 3072, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_378: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_72, 0.7071067811865476)
    erf_37: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_378);  mul_378 = None
    add_263: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
    mul_379: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(add_263, 0.5);  add_263 = None
    mul_380: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_72, convolution_72)
    mul_381: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_380, -0.5);  mul_380 = None
    exp_1: "f32[8, 3072, 7, 7]" = torch.ops.aten.exp.default(mul_381);  mul_381 = None
    mul_382: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
    mul_383: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_72, mul_382);  convolution_72 = mul_382 = None
    add_264: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(mul_379, mul_383);  mul_379 = mul_383 = None
    mul_384: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_152, add_264);  getitem_152 = add_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_24: "f32[3072]" = torch.ops.aten.sum.dim_IntList(mul_384, [0, 2, 3])
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_384, add_242, primals_363, [3072], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_384 = add_242 = primals_363 = None
    getitem_155: "f32[8, 768, 7, 7]" = convolution_backward_3[0]
    getitem_156: "f32[3072, 768, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_148: "f32[8, 1]" = torch.ops.aten.alias.default(alias_138);  alias_138 = None
    alias_149: "f32[8, 1]" = torch.ops.aten.alias.default(alias_139);  alias_139 = None
    mul_385: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_155, add_240)
    view_247: "f32[8, 768, 49]" = torch.ops.aten.view.default(mul_385, [8, 768, 49]);  mul_385 = None
    sum_25: "f32[8, 768]" = torch.ops.aten.sum.dim_IntList(view_247, [2]);  view_247 = None
    view_248: "f32[8, 768, 49]" = torch.ops.aten.view.default(getitem_155, [8, 768, 49])
    sum_26: "f32[8, 768]" = torch.ops.aten.sum.dim_IntList(view_248, [2]);  view_248 = None
    unsqueeze_452: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_208, 0)
    mul_386: "f32[8, 768]" = torch.ops.aten.mul.Tensor(sum_25, unsqueeze_452);  unsqueeze_452 = None
    view_249: "f32[8, 1, 768]" = torch.ops.aten.view.default(mul_386, [8, 1, 768]);  mul_386 = None
    sum_27: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_249, [2]);  view_249 = None
    unsqueeze_453: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_208, 0)
    mul_387: "f32[8, 768]" = torch.ops.aten.mul.Tensor(sum_26, unsqueeze_453);  unsqueeze_453 = None
    view_250: "f32[8, 1, 768]" = torch.ops.aten.view.default(mul_387, [8, 1, 768]);  mul_387 = None
    sum_28: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_250, [2]);  view_250 = None
    unsqueeze_454: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_149, -1)
    view_251: "f32[1, 1, 768]" = torch.ops.aten.view.default(primals_208, [1, 1, 768]);  primals_208 = None
    mul_388: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(unsqueeze_454, view_251);  unsqueeze_454 = view_251 = None
    mul_389: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_28, alias_148)
    sub_118: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_389, sum_27);  mul_389 = sum_27 = None
    mul_390: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_118, alias_149);  sub_118 = None
    mul_391: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_390, alias_149);  mul_390 = None
    mul_392: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_391, alias_149);  mul_391 = None
    mul_393: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_392, 2.657312925170068e-05);  mul_392 = None
    neg_3: "f32[8, 1]" = torch.ops.aten.neg.default(mul_393)
    mul_394: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_3, alias_148);  neg_3 = None
    mul_395: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_28, alias_149);  sum_28 = None
    mul_396: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_395, 2.657312925170068e-05);  mul_395 = None
    sub_119: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_394, mul_396);  mul_394 = mul_396 = None
    unsqueeze_455: "f32[8, 1, 768, 1]" = torch.ops.aten.unsqueeze.default(mul_388, -1);  mul_388 = None
    unsqueeze_456: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_393, -1);  mul_393 = None
    unsqueeze_457: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, -1);  unsqueeze_456 = None
    unsqueeze_458: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_119, -1);  sub_119 = None
    unsqueeze_459: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, -1);  unsqueeze_458 = None
    view_252: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(getitem_155, [8, 1, 768, 49]);  getitem_155 = None
    mul_397: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(view_252, unsqueeze_455);  view_252 = unsqueeze_455 = None
    view_253: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_240, [8, 1, 768, 49]);  add_240 = None
    mul_398: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(view_253, unsqueeze_457);  view_253 = unsqueeze_457 = None
    add_265: "f32[8, 1, 768, 49]" = torch.ops.aten.add.Tensor(mul_397, mul_398);  mul_397 = mul_398 = None
    add_266: "f32[8, 1, 768, 49]" = torch.ops.aten.add.Tensor(add_265, unsqueeze_459);  add_265 = unsqueeze_459 = None
    view_254: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(add_266, [8, 768, 7, 7]);  add_266 = None
    view_255: "f32[8, 1, 768]" = torch.ops.aten.view.default(sum_25, [8, 1, 768]);  sum_25 = None
    view_256: "f32[8, 1, 768]" = torch.ops.aten.view.default(sum_26, [8, 1, 768])
    unsqueeze_460: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_148, -1);  alias_148 = None
    mul_399: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(view_256, unsqueeze_460);  view_256 = unsqueeze_460 = None
    sub_120: "f32[8, 1, 768]" = torch.ops.aten.sub.Tensor(view_255, mul_399);  view_255 = mul_399 = None
    unsqueeze_461: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_149, -1);  alias_149 = None
    mul_400: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_461);  sub_120 = unsqueeze_461 = None
    sum_29: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(mul_400, [0]);  mul_400 = None
    view_257: "f32[768]" = torch.ops.aten.view.default(sum_29, [768]);  sum_29 = None
    sum_30: "f32[768]" = torch.ops.aten.sum.dim_IntList(sum_26, [0]);  sum_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_267: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_262, view_254);  add_262 = view_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_401: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_267, sub_103);  sub_103 = None
    mul_402: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_267, view_206);  view_206 = None
    sum_31: "f32[1, 768, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_401, [0, 2, 3], True);  mul_401 = None
    view_258: "f32[768, 1, 1]" = torch.ops.aten.view.default(sum_31, [768, 1, 1]);  sum_31 = None
    view_259: "f32[768]" = torch.ops.aten.view.default(view_258, [768]);  view_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_4: "f32[8, 768, 7, 7]" = torch.ops.aten.neg.default(mul_402)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_1: "f32[8, 768, 7, 7]" = torch.ops.aten.avg_pool2d_backward.default(mul_402, add_239, [3, 3], [1, 1], [1, 1], False, False, None);  mul_402 = add_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_268: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(neg_4, avg_pool2d_backward_1);  neg_4 = avg_pool2d_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_150: "f32[8, 1]" = torch.ops.aten.alias.default(alias_136);  alias_136 = None
    alias_151: "f32[8, 1]" = torch.ops.aten.alias.default(alias_137);  alias_137 = None
    mul_403: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_268, add_237)
    view_260: "f32[8, 768, 49]" = torch.ops.aten.view.default(mul_403, [8, 768, 49]);  mul_403 = None
    sum_32: "f32[8, 768]" = torch.ops.aten.sum.dim_IntList(view_260, [2]);  view_260 = None
    view_261: "f32[8, 768, 49]" = torch.ops.aten.view.default(add_268, [8, 768, 49])
    sum_33: "f32[8, 768]" = torch.ops.aten.sum.dim_IntList(view_261, [2]);  view_261 = None
    unsqueeze_462: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_205, 0)
    mul_404: "f32[8, 768]" = torch.ops.aten.mul.Tensor(sum_32, unsqueeze_462);  unsqueeze_462 = None
    view_262: "f32[8, 1, 768]" = torch.ops.aten.view.default(mul_404, [8, 1, 768]);  mul_404 = None
    sum_34: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_262, [2]);  view_262 = None
    unsqueeze_463: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_205, 0)
    mul_405: "f32[8, 768]" = torch.ops.aten.mul.Tensor(sum_33, unsqueeze_463);  unsqueeze_463 = None
    view_263: "f32[8, 1, 768]" = torch.ops.aten.view.default(mul_405, [8, 1, 768]);  mul_405 = None
    sum_35: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_263, [2]);  view_263 = None
    unsqueeze_464: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_151, -1)
    view_264: "f32[1, 1, 768]" = torch.ops.aten.view.default(primals_205, [1, 1, 768]);  primals_205 = None
    mul_406: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(unsqueeze_464, view_264);  unsqueeze_464 = view_264 = None
    mul_407: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_35, alias_150)
    sub_121: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_407, sum_34);  mul_407 = sum_34 = None
    mul_408: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_121, alias_151);  sub_121 = None
    mul_409: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_408, alias_151);  mul_408 = None
    mul_410: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_409, alias_151);  mul_409 = None
    mul_411: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_410, 2.657312925170068e-05);  mul_410 = None
    neg_5: "f32[8, 1]" = torch.ops.aten.neg.default(mul_411)
    mul_412: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_5, alias_150);  neg_5 = None
    mul_413: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_35, alias_151);  sum_35 = None
    mul_414: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_413, 2.657312925170068e-05);  mul_413 = None
    sub_122: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_412, mul_414);  mul_412 = mul_414 = None
    unsqueeze_465: "f32[8, 1, 768, 1]" = torch.ops.aten.unsqueeze.default(mul_406, -1);  mul_406 = None
    unsqueeze_466: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_411, -1);  mul_411 = None
    unsqueeze_467: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
    unsqueeze_468: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_122, -1);  sub_122 = None
    unsqueeze_469: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
    view_265: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_268, [8, 1, 768, 49]);  add_268 = None
    mul_415: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(view_265, unsqueeze_465);  view_265 = unsqueeze_465 = None
    view_266: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_237, [8, 1, 768, 49]);  add_237 = None
    mul_416: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(view_266, unsqueeze_467);  view_266 = unsqueeze_467 = None
    add_269: "f32[8, 1, 768, 49]" = torch.ops.aten.add.Tensor(mul_415, mul_416);  mul_415 = mul_416 = None
    add_270: "f32[8, 1, 768, 49]" = torch.ops.aten.add.Tensor(add_269, unsqueeze_469);  add_269 = unsqueeze_469 = None
    view_267: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(add_270, [8, 768, 7, 7]);  add_270 = None
    view_268: "f32[8, 1, 768]" = torch.ops.aten.view.default(sum_32, [8, 1, 768]);  sum_32 = None
    view_269: "f32[8, 1, 768]" = torch.ops.aten.view.default(sum_33, [8, 1, 768])
    unsqueeze_470: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_150, -1);  alias_150 = None
    mul_417: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(view_269, unsqueeze_470);  view_269 = unsqueeze_470 = None
    sub_123: "f32[8, 1, 768]" = torch.ops.aten.sub.Tensor(view_268, mul_417);  view_268 = mul_417 = None
    unsqueeze_471: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_151, -1);  alias_151 = None
    mul_418: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_471);  sub_123 = unsqueeze_471 = None
    sum_36: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(mul_418, [0]);  mul_418 = None
    view_270: "f32[768]" = torch.ops.aten.view.default(sum_36, [768]);  sum_36 = None
    sum_37: "f32[768]" = torch.ops.aten.sum.dim_IntList(sum_33, [0]);  sum_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_271: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_267, view_267);  add_267 = view_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_419: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_271, clone_67);  clone_67 = None
    mul_420: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_271, view_203);  view_203 = None
    sum_38: "f32[1, 768, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_419, [0, 2, 3], True);  mul_419 = None
    view_271: "f32[768, 1, 1]" = torch.ops.aten.view.default(sum_38, [768, 1, 1]);  sum_38 = None
    view_272: "f32[768]" = torch.ops.aten.view.default(view_271, [768]);  view_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_39: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_420, [0, 2, 3])
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_420, clone_66, primals_361, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_420 = clone_66 = primals_361 = None
    getitem_158: "f32[8, 3072, 7, 7]" = convolution_backward_4[0]
    getitem_159: "f32[768, 3072, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_421: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_70, 0.7071067811865476)
    erf_38: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_421);  mul_421 = None
    add_272: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
    mul_422: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(add_272, 0.5);  add_272 = None
    mul_423: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_70, convolution_70)
    mul_424: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_423, -0.5);  mul_423 = None
    exp_2: "f32[8, 3072, 7, 7]" = torch.ops.aten.exp.default(mul_424);  mul_424 = None
    mul_425: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
    mul_426: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_70, mul_425);  convolution_70 = mul_425 = None
    add_273: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(mul_422, mul_426);  mul_422 = mul_426 = None
    mul_427: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_158, add_273);  getitem_158 = add_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_40: "f32[3072]" = torch.ops.aten.sum.dim_IntList(mul_427, [0, 2, 3])
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_427, add_235, primals_359, [3072], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_427 = add_235 = primals_359 = None
    getitem_161: "f32[8, 768, 7, 7]" = convolution_backward_5[0]
    getitem_162: "f32[3072, 768, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_152: "f32[8, 1]" = torch.ops.aten.alias.default(alias_134);  alias_134 = None
    alias_153: "f32[8, 1]" = torch.ops.aten.alias.default(alias_135);  alias_135 = None
    mul_428: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_161, add_233)
    view_273: "f32[8, 768, 49]" = torch.ops.aten.view.default(mul_428, [8, 768, 49]);  mul_428 = None
    sum_41: "f32[8, 768]" = torch.ops.aten.sum.dim_IntList(view_273, [2]);  view_273 = None
    view_274: "f32[8, 768, 49]" = torch.ops.aten.view.default(getitem_161, [8, 768, 49])
    sum_42: "f32[8, 768]" = torch.ops.aten.sum.dim_IntList(view_274, [2]);  view_274 = None
    unsqueeze_472: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_202, 0)
    mul_429: "f32[8, 768]" = torch.ops.aten.mul.Tensor(sum_41, unsqueeze_472);  unsqueeze_472 = None
    view_275: "f32[8, 1, 768]" = torch.ops.aten.view.default(mul_429, [8, 1, 768]);  mul_429 = None
    sum_43: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_275, [2]);  view_275 = None
    unsqueeze_473: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_202, 0)
    mul_430: "f32[8, 768]" = torch.ops.aten.mul.Tensor(sum_42, unsqueeze_473);  unsqueeze_473 = None
    view_276: "f32[8, 1, 768]" = torch.ops.aten.view.default(mul_430, [8, 1, 768]);  mul_430 = None
    sum_44: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_276, [2]);  view_276 = None
    unsqueeze_474: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_153, -1)
    view_277: "f32[1, 1, 768]" = torch.ops.aten.view.default(primals_202, [1, 1, 768]);  primals_202 = None
    mul_431: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(unsqueeze_474, view_277);  unsqueeze_474 = view_277 = None
    mul_432: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_44, alias_152)
    sub_124: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_432, sum_43);  mul_432 = sum_43 = None
    mul_433: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_124, alias_153);  sub_124 = None
    mul_434: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_433, alias_153);  mul_433 = None
    mul_435: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_434, alias_153);  mul_434 = None
    mul_436: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_435, 2.657312925170068e-05);  mul_435 = None
    neg_6: "f32[8, 1]" = torch.ops.aten.neg.default(mul_436)
    mul_437: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_6, alias_152);  neg_6 = None
    mul_438: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_44, alias_153);  sum_44 = None
    mul_439: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_438, 2.657312925170068e-05);  mul_438 = None
    sub_125: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_437, mul_439);  mul_437 = mul_439 = None
    unsqueeze_475: "f32[8, 1, 768, 1]" = torch.ops.aten.unsqueeze.default(mul_431, -1);  mul_431 = None
    unsqueeze_476: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_436, -1);  mul_436 = None
    unsqueeze_477: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
    unsqueeze_478: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_125, -1);  sub_125 = None
    unsqueeze_479: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
    view_278: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(getitem_161, [8, 1, 768, 49]);  getitem_161 = None
    mul_440: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(view_278, unsqueeze_475);  view_278 = unsqueeze_475 = None
    view_279: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_233, [8, 1, 768, 49]);  add_233 = None
    mul_441: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(view_279, unsqueeze_477);  view_279 = unsqueeze_477 = None
    add_274: "f32[8, 1, 768, 49]" = torch.ops.aten.add.Tensor(mul_440, mul_441);  mul_440 = mul_441 = None
    add_275: "f32[8, 1, 768, 49]" = torch.ops.aten.add.Tensor(add_274, unsqueeze_479);  add_274 = unsqueeze_479 = None
    view_280: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(add_275, [8, 768, 7, 7]);  add_275 = None
    view_281: "f32[8, 1, 768]" = torch.ops.aten.view.default(sum_41, [8, 1, 768]);  sum_41 = None
    view_282: "f32[8, 1, 768]" = torch.ops.aten.view.default(sum_42, [8, 1, 768])
    unsqueeze_480: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_152, -1);  alias_152 = None
    mul_442: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(view_282, unsqueeze_480);  view_282 = unsqueeze_480 = None
    sub_126: "f32[8, 1, 768]" = torch.ops.aten.sub.Tensor(view_281, mul_442);  view_281 = mul_442 = None
    unsqueeze_481: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_153, -1);  alias_153 = None
    mul_443: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_481);  sub_126 = unsqueeze_481 = None
    sum_45: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(mul_443, [0]);  mul_443 = None
    view_283: "f32[768]" = torch.ops.aten.view.default(sum_45, [768]);  sum_45 = None
    sum_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(sum_42, [0]);  sum_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_276: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_271, view_280);  add_271 = view_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_444: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_276, sub_100);  sub_100 = None
    mul_445: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_276, view_200);  view_200 = None
    sum_47: "f32[1, 768, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_444, [0, 2, 3], True);  mul_444 = None
    view_284: "f32[768, 1, 1]" = torch.ops.aten.view.default(sum_47, [768, 1, 1]);  sum_47 = None
    view_285: "f32[768]" = torch.ops.aten.view.default(view_284, [768]);  view_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_7: "f32[8, 768, 7, 7]" = torch.ops.aten.neg.default(mul_445)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_2: "f32[8, 768, 7, 7]" = torch.ops.aten.avg_pool2d_backward.default(mul_445, add_232, [3, 3], [1, 1], [1, 1], False, False, None);  mul_445 = add_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_277: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(neg_7, avg_pool2d_backward_2);  neg_7 = avg_pool2d_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_154: "f32[8, 1]" = torch.ops.aten.alias.default(alias_132);  alias_132 = None
    alias_155: "f32[8, 1]" = torch.ops.aten.alias.default(alias_133);  alias_133 = None
    mul_446: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_277, add_230)
    view_286: "f32[8, 768, 49]" = torch.ops.aten.view.default(mul_446, [8, 768, 49]);  mul_446 = None
    sum_48: "f32[8, 768]" = torch.ops.aten.sum.dim_IntList(view_286, [2]);  view_286 = None
    view_287: "f32[8, 768, 49]" = torch.ops.aten.view.default(add_277, [8, 768, 49])
    sum_49: "f32[8, 768]" = torch.ops.aten.sum.dim_IntList(view_287, [2]);  view_287 = None
    unsqueeze_482: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_199, 0)
    mul_447: "f32[8, 768]" = torch.ops.aten.mul.Tensor(sum_48, unsqueeze_482);  unsqueeze_482 = None
    view_288: "f32[8, 1, 768]" = torch.ops.aten.view.default(mul_447, [8, 1, 768]);  mul_447 = None
    sum_50: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_288, [2]);  view_288 = None
    unsqueeze_483: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_199, 0)
    mul_448: "f32[8, 768]" = torch.ops.aten.mul.Tensor(sum_49, unsqueeze_483);  unsqueeze_483 = None
    view_289: "f32[8, 1, 768]" = torch.ops.aten.view.default(mul_448, [8, 1, 768]);  mul_448 = None
    sum_51: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_289, [2]);  view_289 = None
    unsqueeze_484: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_155, -1)
    view_290: "f32[1, 1, 768]" = torch.ops.aten.view.default(primals_199, [1, 1, 768]);  primals_199 = None
    mul_449: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(unsqueeze_484, view_290);  unsqueeze_484 = view_290 = None
    mul_450: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_51, alias_154)
    sub_127: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_450, sum_50);  mul_450 = sum_50 = None
    mul_451: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_127, alias_155);  sub_127 = None
    mul_452: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_451, alias_155);  mul_451 = None
    mul_453: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_452, alias_155);  mul_452 = None
    mul_454: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_453, 2.657312925170068e-05);  mul_453 = None
    neg_8: "f32[8, 1]" = torch.ops.aten.neg.default(mul_454)
    mul_455: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_8, alias_154);  neg_8 = None
    mul_456: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_51, alias_155);  sum_51 = None
    mul_457: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_456, 2.657312925170068e-05);  mul_456 = None
    sub_128: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_455, mul_457);  mul_455 = mul_457 = None
    unsqueeze_485: "f32[8, 1, 768, 1]" = torch.ops.aten.unsqueeze.default(mul_449, -1);  mul_449 = None
    unsqueeze_486: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_454, -1);  mul_454 = None
    unsqueeze_487: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
    unsqueeze_488: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_128, -1);  sub_128 = None
    unsqueeze_489: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
    view_291: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_277, [8, 1, 768, 49]);  add_277 = None
    mul_458: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(view_291, unsqueeze_485);  view_291 = unsqueeze_485 = None
    view_292: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_230, [8, 1, 768, 49]);  add_230 = None
    mul_459: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(view_292, unsqueeze_487);  view_292 = unsqueeze_487 = None
    add_278: "f32[8, 1, 768, 49]" = torch.ops.aten.add.Tensor(mul_458, mul_459);  mul_458 = mul_459 = None
    add_279: "f32[8, 1, 768, 49]" = torch.ops.aten.add.Tensor(add_278, unsqueeze_489);  add_278 = unsqueeze_489 = None
    view_293: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(add_279, [8, 768, 7, 7]);  add_279 = None
    view_294: "f32[8, 1, 768]" = torch.ops.aten.view.default(sum_48, [8, 1, 768]);  sum_48 = None
    view_295: "f32[8, 1, 768]" = torch.ops.aten.view.default(sum_49, [8, 1, 768])
    unsqueeze_490: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_154, -1);  alias_154 = None
    mul_460: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(view_295, unsqueeze_490);  view_295 = unsqueeze_490 = None
    sub_129: "f32[8, 1, 768]" = torch.ops.aten.sub.Tensor(view_294, mul_460);  view_294 = mul_460 = None
    unsqueeze_491: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_155, -1);  alias_155 = None
    mul_461: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_491);  sub_129 = unsqueeze_491 = None
    sum_52: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(mul_461, [0]);  mul_461 = None
    view_296: "f32[768]" = torch.ops.aten.view.default(sum_52, [768]);  sum_52 = None
    sum_53: "f32[768]" = torch.ops.aten.sum.dim_IntList(sum_49, [0]);  sum_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_280: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_276, view_293);  add_276 = view_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_462: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_280, clone_65);  clone_65 = None
    mul_463: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_280, view_197);  view_197 = None
    sum_54: "f32[1, 768, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_462, [0, 2, 3], True);  mul_462 = None
    view_297: "f32[768, 1, 1]" = torch.ops.aten.view.default(sum_54, [768, 1, 1]);  sum_54 = None
    view_298: "f32[768]" = torch.ops.aten.view.default(view_297, [768]);  view_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_463, [0, 2, 3])
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_463, clone_64, primals_357, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_463 = clone_64 = primals_357 = None
    getitem_164: "f32[8, 3072, 7, 7]" = convolution_backward_6[0]
    getitem_165: "f32[768, 3072, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_464: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_68, 0.7071067811865476)
    erf_39: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_464);  mul_464 = None
    add_281: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
    mul_465: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(add_281, 0.5);  add_281 = None
    mul_466: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_68, convolution_68)
    mul_467: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_466, -0.5);  mul_466 = None
    exp_3: "f32[8, 3072, 7, 7]" = torch.ops.aten.exp.default(mul_467);  mul_467 = None
    mul_468: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(exp_3, 0.3989422804014327);  exp_3 = None
    mul_469: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_68, mul_468);  convolution_68 = mul_468 = None
    add_282: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(mul_465, mul_469);  mul_465 = mul_469 = None
    mul_470: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_164, add_282);  getitem_164 = add_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_56: "f32[3072]" = torch.ops.aten.sum.dim_IntList(mul_470, [0, 2, 3])
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_470, add_228, primals_355, [3072], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_470 = add_228 = primals_355 = None
    getitem_167: "f32[8, 768, 7, 7]" = convolution_backward_7[0]
    getitem_168: "f32[3072, 768, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_156: "f32[8, 1]" = torch.ops.aten.alias.default(alias_130);  alias_130 = None
    alias_157: "f32[8, 1]" = torch.ops.aten.alias.default(alias_131);  alias_131 = None
    mul_471: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_167, add_226)
    view_299: "f32[8, 768, 49]" = torch.ops.aten.view.default(mul_471, [8, 768, 49]);  mul_471 = None
    sum_57: "f32[8, 768]" = torch.ops.aten.sum.dim_IntList(view_299, [2]);  view_299 = None
    view_300: "f32[8, 768, 49]" = torch.ops.aten.view.default(getitem_167, [8, 768, 49])
    sum_58: "f32[8, 768]" = torch.ops.aten.sum.dim_IntList(view_300, [2]);  view_300 = None
    unsqueeze_492: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_196, 0)
    mul_472: "f32[8, 768]" = torch.ops.aten.mul.Tensor(sum_57, unsqueeze_492);  unsqueeze_492 = None
    view_301: "f32[8, 1, 768]" = torch.ops.aten.view.default(mul_472, [8, 1, 768]);  mul_472 = None
    sum_59: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_301, [2]);  view_301 = None
    unsqueeze_493: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_196, 0)
    mul_473: "f32[8, 768]" = torch.ops.aten.mul.Tensor(sum_58, unsqueeze_493);  unsqueeze_493 = None
    view_302: "f32[8, 1, 768]" = torch.ops.aten.view.default(mul_473, [8, 1, 768]);  mul_473 = None
    sum_60: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_302, [2]);  view_302 = None
    unsqueeze_494: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_157, -1)
    view_303: "f32[1, 1, 768]" = torch.ops.aten.view.default(primals_196, [1, 1, 768]);  primals_196 = None
    mul_474: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(unsqueeze_494, view_303);  unsqueeze_494 = view_303 = None
    mul_475: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_60, alias_156)
    sub_130: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_475, sum_59);  mul_475 = sum_59 = None
    mul_476: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_130, alias_157);  sub_130 = None
    mul_477: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_476, alias_157);  mul_476 = None
    mul_478: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_477, alias_157);  mul_477 = None
    mul_479: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_478, 2.657312925170068e-05);  mul_478 = None
    neg_9: "f32[8, 1]" = torch.ops.aten.neg.default(mul_479)
    mul_480: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_9, alias_156);  neg_9 = None
    mul_481: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_60, alias_157);  sum_60 = None
    mul_482: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_481, 2.657312925170068e-05);  mul_481 = None
    sub_131: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_480, mul_482);  mul_480 = mul_482 = None
    unsqueeze_495: "f32[8, 1, 768, 1]" = torch.ops.aten.unsqueeze.default(mul_474, -1);  mul_474 = None
    unsqueeze_496: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_479, -1);  mul_479 = None
    unsqueeze_497: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, -1);  unsqueeze_496 = None
    unsqueeze_498: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_131, -1);  sub_131 = None
    unsqueeze_499: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, -1);  unsqueeze_498 = None
    view_304: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(getitem_167, [8, 1, 768, 49]);  getitem_167 = None
    mul_483: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(view_304, unsqueeze_495);  view_304 = unsqueeze_495 = None
    view_305: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_226, [8, 1, 768, 49]);  add_226 = None
    mul_484: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(view_305, unsqueeze_497);  view_305 = unsqueeze_497 = None
    add_283: "f32[8, 1, 768, 49]" = torch.ops.aten.add.Tensor(mul_483, mul_484);  mul_483 = mul_484 = None
    add_284: "f32[8, 1, 768, 49]" = torch.ops.aten.add.Tensor(add_283, unsqueeze_499);  add_283 = unsqueeze_499 = None
    view_306: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(add_284, [8, 768, 7, 7]);  add_284 = None
    view_307: "f32[8, 1, 768]" = torch.ops.aten.view.default(sum_57, [8, 1, 768]);  sum_57 = None
    view_308: "f32[8, 1, 768]" = torch.ops.aten.view.default(sum_58, [8, 1, 768])
    unsqueeze_500: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_156, -1);  alias_156 = None
    mul_485: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(view_308, unsqueeze_500);  view_308 = unsqueeze_500 = None
    sub_132: "f32[8, 1, 768]" = torch.ops.aten.sub.Tensor(view_307, mul_485);  view_307 = mul_485 = None
    unsqueeze_501: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_157, -1);  alias_157 = None
    mul_486: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_501);  sub_132 = unsqueeze_501 = None
    sum_61: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(mul_486, [0]);  mul_486 = None
    view_309: "f32[768]" = torch.ops.aten.view.default(sum_61, [768]);  sum_61 = None
    sum_62: "f32[768]" = torch.ops.aten.sum.dim_IntList(sum_58, [0]);  sum_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_285: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_280, view_306);  add_280 = view_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_487: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_285, sub_97);  sub_97 = None
    mul_488: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_285, view_194);  view_194 = None
    sum_63: "f32[1, 768, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_487, [0, 2, 3], True);  mul_487 = None
    view_310: "f32[768, 1, 1]" = torch.ops.aten.view.default(sum_63, [768, 1, 1]);  sum_63 = None
    view_311: "f32[768]" = torch.ops.aten.view.default(view_310, [768]);  view_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_10: "f32[8, 768, 7, 7]" = torch.ops.aten.neg.default(mul_488)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_3: "f32[8, 768, 7, 7]" = torch.ops.aten.avg_pool2d_backward.default(mul_488, add_225, [3, 3], [1, 1], [1, 1], False, False, None);  mul_488 = add_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_286: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(neg_10, avg_pool2d_backward_3);  neg_10 = avg_pool2d_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_158: "f32[8, 1]" = torch.ops.aten.alias.default(alias_128);  alias_128 = None
    alias_159: "f32[8, 1]" = torch.ops.aten.alias.default(alias_129);  alias_129 = None
    mul_489: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_286, add_223)
    view_312: "f32[8, 768, 49]" = torch.ops.aten.view.default(mul_489, [8, 768, 49]);  mul_489 = None
    sum_64: "f32[8, 768]" = torch.ops.aten.sum.dim_IntList(view_312, [2]);  view_312 = None
    view_313: "f32[8, 768, 49]" = torch.ops.aten.view.default(add_286, [8, 768, 49])
    sum_65: "f32[8, 768]" = torch.ops.aten.sum.dim_IntList(view_313, [2]);  view_313 = None
    unsqueeze_502: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_193, 0)
    mul_490: "f32[8, 768]" = torch.ops.aten.mul.Tensor(sum_64, unsqueeze_502);  unsqueeze_502 = None
    view_314: "f32[8, 1, 768]" = torch.ops.aten.view.default(mul_490, [8, 1, 768]);  mul_490 = None
    sum_66: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_314, [2]);  view_314 = None
    unsqueeze_503: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_193, 0)
    mul_491: "f32[8, 768]" = torch.ops.aten.mul.Tensor(sum_65, unsqueeze_503);  unsqueeze_503 = None
    view_315: "f32[8, 1, 768]" = torch.ops.aten.view.default(mul_491, [8, 1, 768]);  mul_491 = None
    sum_67: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_315, [2]);  view_315 = None
    unsqueeze_504: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_159, -1)
    view_316: "f32[1, 1, 768]" = torch.ops.aten.view.default(primals_193, [1, 1, 768]);  primals_193 = None
    mul_492: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(unsqueeze_504, view_316);  unsqueeze_504 = view_316 = None
    mul_493: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_67, alias_158)
    sub_133: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_493, sum_66);  mul_493 = sum_66 = None
    mul_494: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_133, alias_159);  sub_133 = None
    mul_495: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_494, alias_159);  mul_494 = None
    mul_496: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_495, alias_159);  mul_495 = None
    mul_497: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_496, 2.657312925170068e-05);  mul_496 = None
    neg_11: "f32[8, 1]" = torch.ops.aten.neg.default(mul_497)
    mul_498: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_11, alias_158);  neg_11 = None
    mul_499: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_67, alias_159);  sum_67 = None
    mul_500: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_499, 2.657312925170068e-05);  mul_499 = None
    sub_134: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_498, mul_500);  mul_498 = mul_500 = None
    unsqueeze_505: "f32[8, 1, 768, 1]" = torch.ops.aten.unsqueeze.default(mul_492, -1);  mul_492 = None
    unsqueeze_506: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_497, -1);  mul_497 = None
    unsqueeze_507: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, -1);  unsqueeze_506 = None
    unsqueeze_508: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_134, -1);  sub_134 = None
    unsqueeze_509: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, -1);  unsqueeze_508 = None
    view_317: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_286, [8, 1, 768, 49]);  add_286 = None
    mul_501: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(view_317, unsqueeze_505);  view_317 = unsqueeze_505 = None
    view_318: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_223, [8, 1, 768, 49]);  add_223 = None
    mul_502: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(view_318, unsqueeze_507);  view_318 = unsqueeze_507 = None
    add_287: "f32[8, 1, 768, 49]" = torch.ops.aten.add.Tensor(mul_501, mul_502);  mul_501 = mul_502 = None
    add_288: "f32[8, 1, 768, 49]" = torch.ops.aten.add.Tensor(add_287, unsqueeze_509);  add_287 = unsqueeze_509 = None
    view_319: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(add_288, [8, 768, 7, 7]);  add_288 = None
    view_320: "f32[8, 1, 768]" = torch.ops.aten.view.default(sum_64, [8, 1, 768]);  sum_64 = None
    view_321: "f32[8, 1, 768]" = torch.ops.aten.view.default(sum_65, [8, 1, 768])
    unsqueeze_510: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_158, -1);  alias_158 = None
    mul_503: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(view_321, unsqueeze_510);  view_321 = unsqueeze_510 = None
    sub_135: "f32[8, 1, 768]" = torch.ops.aten.sub.Tensor(view_320, mul_503);  view_320 = mul_503 = None
    unsqueeze_511: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_159, -1);  alias_159 = None
    mul_504: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_511);  sub_135 = unsqueeze_511 = None
    sum_68: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(mul_504, [0]);  mul_504 = None
    view_322: "f32[768]" = torch.ops.aten.view.default(sum_68, [768]);  sum_68 = None
    sum_69: "f32[768]" = torch.ops.aten.sum.dim_IntList(sum_65, [0]);  sum_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_289: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_285, view_319);  add_285 = view_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_505: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_289, clone_63);  clone_63 = None
    mul_506: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_289, view_191);  view_191 = None
    sum_70: "f32[1, 768, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_505, [0, 2, 3], True);  mul_505 = None
    view_323: "f32[768, 1, 1]" = torch.ops.aten.view.default(sum_70, [768, 1, 1]);  sum_70 = None
    view_324: "f32[768]" = torch.ops.aten.view.default(view_323, [768]);  view_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_71: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_506, [0, 2, 3])
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_506, clone_62, primals_353, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_506 = clone_62 = primals_353 = None
    getitem_170: "f32[8, 3072, 7, 7]" = convolution_backward_8[0]
    getitem_171: "f32[768, 3072, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_507: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_66, 0.7071067811865476)
    erf_40: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_507);  mul_507 = None
    add_290: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
    mul_508: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(add_290, 0.5);  add_290 = None
    mul_509: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_66, convolution_66)
    mul_510: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_509, -0.5);  mul_509 = None
    exp_4: "f32[8, 3072, 7, 7]" = torch.ops.aten.exp.default(mul_510);  mul_510 = None
    mul_511: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
    mul_512: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_66, mul_511);  convolution_66 = mul_511 = None
    add_291: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(mul_508, mul_512);  mul_508 = mul_512 = None
    mul_513: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_170, add_291);  getitem_170 = add_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_72: "f32[3072]" = torch.ops.aten.sum.dim_IntList(mul_513, [0, 2, 3])
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_513, add_221, primals_351, [3072], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_513 = add_221 = primals_351 = None
    getitem_173: "f32[8, 768, 7, 7]" = convolution_backward_9[0]
    getitem_174: "f32[3072, 768, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_160: "f32[8, 1]" = torch.ops.aten.alias.default(alias_126);  alias_126 = None
    alias_161: "f32[8, 1]" = torch.ops.aten.alias.default(alias_127);  alias_127 = None
    mul_514: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_173, add_219)
    view_325: "f32[8, 768, 49]" = torch.ops.aten.view.default(mul_514, [8, 768, 49]);  mul_514 = None
    sum_73: "f32[8, 768]" = torch.ops.aten.sum.dim_IntList(view_325, [2]);  view_325 = None
    view_326: "f32[8, 768, 49]" = torch.ops.aten.view.default(getitem_173, [8, 768, 49])
    sum_74: "f32[8, 768]" = torch.ops.aten.sum.dim_IntList(view_326, [2]);  view_326 = None
    unsqueeze_512: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_190, 0)
    mul_515: "f32[8, 768]" = torch.ops.aten.mul.Tensor(sum_73, unsqueeze_512);  unsqueeze_512 = None
    view_327: "f32[8, 1, 768]" = torch.ops.aten.view.default(mul_515, [8, 1, 768]);  mul_515 = None
    sum_75: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_327, [2]);  view_327 = None
    unsqueeze_513: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_190, 0)
    mul_516: "f32[8, 768]" = torch.ops.aten.mul.Tensor(sum_74, unsqueeze_513);  unsqueeze_513 = None
    view_328: "f32[8, 1, 768]" = torch.ops.aten.view.default(mul_516, [8, 1, 768]);  mul_516 = None
    sum_76: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_328, [2]);  view_328 = None
    unsqueeze_514: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_161, -1)
    view_329: "f32[1, 1, 768]" = torch.ops.aten.view.default(primals_190, [1, 1, 768]);  primals_190 = None
    mul_517: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(unsqueeze_514, view_329);  unsqueeze_514 = view_329 = None
    mul_518: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_76, alias_160)
    sub_136: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_518, sum_75);  mul_518 = sum_75 = None
    mul_519: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_136, alias_161);  sub_136 = None
    mul_520: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_519, alias_161);  mul_519 = None
    mul_521: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_520, alias_161);  mul_520 = None
    mul_522: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_521, 2.657312925170068e-05);  mul_521 = None
    neg_12: "f32[8, 1]" = torch.ops.aten.neg.default(mul_522)
    mul_523: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_12, alias_160);  neg_12 = None
    mul_524: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_76, alias_161);  sum_76 = None
    mul_525: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_524, 2.657312925170068e-05);  mul_524 = None
    sub_137: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_523, mul_525);  mul_523 = mul_525 = None
    unsqueeze_515: "f32[8, 1, 768, 1]" = torch.ops.aten.unsqueeze.default(mul_517, -1);  mul_517 = None
    unsqueeze_516: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_522, -1);  mul_522 = None
    unsqueeze_517: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, -1);  unsqueeze_516 = None
    unsqueeze_518: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_137, -1);  sub_137 = None
    unsqueeze_519: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, -1);  unsqueeze_518 = None
    view_330: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(getitem_173, [8, 1, 768, 49]);  getitem_173 = None
    mul_526: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(view_330, unsqueeze_515);  view_330 = unsqueeze_515 = None
    view_331: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_219, [8, 1, 768, 49]);  add_219 = None
    mul_527: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(view_331, unsqueeze_517);  view_331 = unsqueeze_517 = None
    add_292: "f32[8, 1, 768, 49]" = torch.ops.aten.add.Tensor(mul_526, mul_527);  mul_526 = mul_527 = None
    add_293: "f32[8, 1, 768, 49]" = torch.ops.aten.add.Tensor(add_292, unsqueeze_519);  add_292 = unsqueeze_519 = None
    view_332: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(add_293, [8, 768, 7, 7]);  add_293 = None
    view_333: "f32[8, 1, 768]" = torch.ops.aten.view.default(sum_73, [8, 1, 768]);  sum_73 = None
    view_334: "f32[8, 1, 768]" = torch.ops.aten.view.default(sum_74, [8, 1, 768])
    unsqueeze_520: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_160, -1);  alias_160 = None
    mul_528: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(view_334, unsqueeze_520);  view_334 = unsqueeze_520 = None
    sub_138: "f32[8, 1, 768]" = torch.ops.aten.sub.Tensor(view_333, mul_528);  view_333 = mul_528 = None
    unsqueeze_521: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_161, -1);  alias_161 = None
    mul_529: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_521);  sub_138 = unsqueeze_521 = None
    sum_77: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(mul_529, [0]);  mul_529 = None
    view_335: "f32[768]" = torch.ops.aten.view.default(sum_77, [768]);  sum_77 = None
    sum_78: "f32[768]" = torch.ops.aten.sum.dim_IntList(sum_74, [0]);  sum_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_294: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_289, view_332);  add_289 = view_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_530: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_294, sub_94);  sub_94 = None
    mul_531: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_294, view_188);  view_188 = None
    sum_79: "f32[1, 768, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_530, [0, 2, 3], True);  mul_530 = None
    view_336: "f32[768, 1, 1]" = torch.ops.aten.view.default(sum_79, [768, 1, 1]);  sum_79 = None
    view_337: "f32[768]" = torch.ops.aten.view.default(view_336, [768]);  view_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_13: "f32[8, 768, 7, 7]" = torch.ops.aten.neg.default(mul_531)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_4: "f32[8, 768, 7, 7]" = torch.ops.aten.avg_pool2d_backward.default(mul_531, add_218, [3, 3], [1, 1], [1, 1], False, False, None);  mul_531 = add_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_295: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(neg_13, avg_pool2d_backward_4);  neg_13 = avg_pool2d_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_162: "f32[8, 1]" = torch.ops.aten.alias.default(alias_124);  alias_124 = None
    alias_163: "f32[8, 1]" = torch.ops.aten.alias.default(alias_125);  alias_125 = None
    mul_532: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_295, add_216)
    view_338: "f32[8, 768, 49]" = torch.ops.aten.view.default(mul_532, [8, 768, 49]);  mul_532 = None
    sum_80: "f32[8, 768]" = torch.ops.aten.sum.dim_IntList(view_338, [2]);  view_338 = None
    view_339: "f32[8, 768, 49]" = torch.ops.aten.view.default(add_295, [8, 768, 49])
    sum_81: "f32[8, 768]" = torch.ops.aten.sum.dim_IntList(view_339, [2]);  view_339 = None
    unsqueeze_522: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_187, 0)
    mul_533: "f32[8, 768]" = torch.ops.aten.mul.Tensor(sum_80, unsqueeze_522);  unsqueeze_522 = None
    view_340: "f32[8, 1, 768]" = torch.ops.aten.view.default(mul_533, [8, 1, 768]);  mul_533 = None
    sum_82: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_340, [2]);  view_340 = None
    unsqueeze_523: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_187, 0)
    mul_534: "f32[8, 768]" = torch.ops.aten.mul.Tensor(sum_81, unsqueeze_523);  unsqueeze_523 = None
    view_341: "f32[8, 1, 768]" = torch.ops.aten.view.default(mul_534, [8, 1, 768]);  mul_534 = None
    sum_83: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_341, [2]);  view_341 = None
    unsqueeze_524: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_163, -1)
    view_342: "f32[1, 1, 768]" = torch.ops.aten.view.default(primals_187, [1, 1, 768]);  primals_187 = None
    mul_535: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(unsqueeze_524, view_342);  unsqueeze_524 = view_342 = None
    mul_536: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_83, alias_162)
    sub_139: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_536, sum_82);  mul_536 = sum_82 = None
    mul_537: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_139, alias_163);  sub_139 = None
    mul_538: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_537, alias_163);  mul_537 = None
    mul_539: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_538, alias_163);  mul_538 = None
    mul_540: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_539, 2.657312925170068e-05);  mul_539 = None
    neg_14: "f32[8, 1]" = torch.ops.aten.neg.default(mul_540)
    mul_541: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_14, alias_162);  neg_14 = None
    mul_542: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_83, alias_163);  sum_83 = None
    mul_543: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_542, 2.657312925170068e-05);  mul_542 = None
    sub_140: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_541, mul_543);  mul_541 = mul_543 = None
    unsqueeze_525: "f32[8, 1, 768, 1]" = torch.ops.aten.unsqueeze.default(mul_535, -1);  mul_535 = None
    unsqueeze_526: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_540, -1);  mul_540 = None
    unsqueeze_527: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, -1);  unsqueeze_526 = None
    unsqueeze_528: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_140, -1);  sub_140 = None
    unsqueeze_529: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, -1);  unsqueeze_528 = None
    view_343: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_295, [8, 1, 768, 49]);  add_295 = None
    mul_544: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(view_343, unsqueeze_525);  view_343 = unsqueeze_525 = None
    view_344: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_216, [8, 1, 768, 49]);  add_216 = None
    mul_545: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(view_344, unsqueeze_527);  view_344 = unsqueeze_527 = None
    add_296: "f32[8, 1, 768, 49]" = torch.ops.aten.add.Tensor(mul_544, mul_545);  mul_544 = mul_545 = None
    add_297: "f32[8, 1, 768, 49]" = torch.ops.aten.add.Tensor(add_296, unsqueeze_529);  add_296 = unsqueeze_529 = None
    view_345: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(add_297, [8, 768, 7, 7]);  add_297 = None
    view_346: "f32[8, 1, 768]" = torch.ops.aten.view.default(sum_80, [8, 1, 768]);  sum_80 = None
    view_347: "f32[8, 1, 768]" = torch.ops.aten.view.default(sum_81, [8, 1, 768])
    unsqueeze_530: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_162, -1);  alias_162 = None
    mul_546: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(view_347, unsqueeze_530);  view_347 = unsqueeze_530 = None
    sub_141: "f32[8, 1, 768]" = torch.ops.aten.sub.Tensor(view_346, mul_546);  view_346 = mul_546 = None
    unsqueeze_531: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_163, -1);  alias_163 = None
    mul_547: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_531);  sub_141 = unsqueeze_531 = None
    sum_84: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(mul_547, [0]);  mul_547 = None
    view_348: "f32[768]" = torch.ops.aten.view.default(sum_84, [768]);  sum_84 = None
    sum_85: "f32[768]" = torch.ops.aten.sum.dim_IntList(sum_81, [0]);  sum_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_298: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_294, view_345);  add_294 = view_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_548: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_298, clone_61);  clone_61 = None
    mul_549: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_298, view_185);  view_185 = None
    sum_86: "f32[1, 768, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_548, [0, 2, 3], True);  mul_548 = None
    view_349: "f32[768, 1, 1]" = torch.ops.aten.view.default(sum_86, [768, 1, 1]);  sum_86 = None
    view_350: "f32[768]" = torch.ops.aten.view.default(view_349, [768]);  view_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_87: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_549, [0, 2, 3])
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_549, clone_60, primals_349, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_549 = clone_60 = primals_349 = None
    getitem_176: "f32[8, 3072, 7, 7]" = convolution_backward_10[0]
    getitem_177: "f32[768, 3072, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_550: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_64, 0.7071067811865476)
    erf_41: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_550);  mul_550 = None
    add_299: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
    mul_551: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(add_299, 0.5);  add_299 = None
    mul_552: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_64, convolution_64)
    mul_553: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_552, -0.5);  mul_552 = None
    exp_5: "f32[8, 3072, 7, 7]" = torch.ops.aten.exp.default(mul_553);  mul_553 = None
    mul_554: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(exp_5, 0.3989422804014327);  exp_5 = None
    mul_555: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_64, mul_554);  convolution_64 = mul_554 = None
    add_300: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(mul_551, mul_555);  mul_551 = mul_555 = None
    mul_556: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_176, add_300);  getitem_176 = add_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_88: "f32[3072]" = torch.ops.aten.sum.dim_IntList(mul_556, [0, 2, 3])
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_556, add_214, primals_347, [3072], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_556 = add_214 = primals_347 = None
    getitem_179: "f32[8, 768, 7, 7]" = convolution_backward_11[0]
    getitem_180: "f32[3072, 768, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_164: "f32[8, 1]" = torch.ops.aten.alias.default(alias_122);  alias_122 = None
    alias_165: "f32[8, 1]" = torch.ops.aten.alias.default(alias_123);  alias_123 = None
    mul_557: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_179, add_212)
    view_351: "f32[8, 768, 49]" = torch.ops.aten.view.default(mul_557, [8, 768, 49]);  mul_557 = None
    sum_89: "f32[8, 768]" = torch.ops.aten.sum.dim_IntList(view_351, [2]);  view_351 = None
    view_352: "f32[8, 768, 49]" = torch.ops.aten.view.default(getitem_179, [8, 768, 49])
    sum_90: "f32[8, 768]" = torch.ops.aten.sum.dim_IntList(view_352, [2]);  view_352 = None
    unsqueeze_532: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_184, 0)
    mul_558: "f32[8, 768]" = torch.ops.aten.mul.Tensor(sum_89, unsqueeze_532);  unsqueeze_532 = None
    view_353: "f32[8, 1, 768]" = torch.ops.aten.view.default(mul_558, [8, 1, 768]);  mul_558 = None
    sum_91: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_353, [2]);  view_353 = None
    unsqueeze_533: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_184, 0)
    mul_559: "f32[8, 768]" = torch.ops.aten.mul.Tensor(sum_90, unsqueeze_533);  unsqueeze_533 = None
    view_354: "f32[8, 1, 768]" = torch.ops.aten.view.default(mul_559, [8, 1, 768]);  mul_559 = None
    sum_92: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_354, [2]);  view_354 = None
    unsqueeze_534: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_165, -1)
    view_355: "f32[1, 1, 768]" = torch.ops.aten.view.default(primals_184, [1, 1, 768]);  primals_184 = None
    mul_560: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(unsqueeze_534, view_355);  unsqueeze_534 = view_355 = None
    mul_561: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_92, alias_164)
    sub_142: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_561, sum_91);  mul_561 = sum_91 = None
    mul_562: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_142, alias_165);  sub_142 = None
    mul_563: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_562, alias_165);  mul_562 = None
    mul_564: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_563, alias_165);  mul_563 = None
    mul_565: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_564, 2.657312925170068e-05);  mul_564 = None
    neg_15: "f32[8, 1]" = torch.ops.aten.neg.default(mul_565)
    mul_566: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_15, alias_164);  neg_15 = None
    mul_567: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_92, alias_165);  sum_92 = None
    mul_568: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_567, 2.657312925170068e-05);  mul_567 = None
    sub_143: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_566, mul_568);  mul_566 = mul_568 = None
    unsqueeze_535: "f32[8, 1, 768, 1]" = torch.ops.aten.unsqueeze.default(mul_560, -1);  mul_560 = None
    unsqueeze_536: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_565, -1);  mul_565 = None
    unsqueeze_537: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, -1);  unsqueeze_536 = None
    unsqueeze_538: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_143, -1);  sub_143 = None
    unsqueeze_539: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, -1);  unsqueeze_538 = None
    view_356: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(getitem_179, [8, 1, 768, 49]);  getitem_179 = None
    mul_569: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(view_356, unsqueeze_535);  view_356 = unsqueeze_535 = None
    view_357: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_212, [8, 1, 768, 49]);  add_212 = None
    mul_570: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(view_357, unsqueeze_537);  view_357 = unsqueeze_537 = None
    add_301: "f32[8, 1, 768, 49]" = torch.ops.aten.add.Tensor(mul_569, mul_570);  mul_569 = mul_570 = None
    add_302: "f32[8, 1, 768, 49]" = torch.ops.aten.add.Tensor(add_301, unsqueeze_539);  add_301 = unsqueeze_539 = None
    view_358: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(add_302, [8, 768, 7, 7]);  add_302 = None
    view_359: "f32[8, 1, 768]" = torch.ops.aten.view.default(sum_89, [8, 1, 768]);  sum_89 = None
    view_360: "f32[8, 1, 768]" = torch.ops.aten.view.default(sum_90, [8, 1, 768])
    unsqueeze_540: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_164, -1);  alias_164 = None
    mul_571: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(view_360, unsqueeze_540);  view_360 = unsqueeze_540 = None
    sub_144: "f32[8, 1, 768]" = torch.ops.aten.sub.Tensor(view_359, mul_571);  view_359 = mul_571 = None
    unsqueeze_541: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_165, -1);  alias_165 = None
    mul_572: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_541);  sub_144 = unsqueeze_541 = None
    sum_93: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(mul_572, [0]);  mul_572 = None
    view_361: "f32[768]" = torch.ops.aten.view.default(sum_93, [768]);  sum_93 = None
    sum_94: "f32[768]" = torch.ops.aten.sum.dim_IntList(sum_90, [0]);  sum_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_303: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_298, view_358);  add_298 = view_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_573: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_303, sub_91);  sub_91 = None
    mul_574: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_303, view_182);  view_182 = None
    sum_95: "f32[1, 768, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_573, [0, 2, 3], True);  mul_573 = None
    view_362: "f32[768, 1, 1]" = torch.ops.aten.view.default(sum_95, [768, 1, 1]);  sum_95 = None
    view_363: "f32[768]" = torch.ops.aten.view.default(view_362, [768]);  view_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_16: "f32[8, 768, 7, 7]" = torch.ops.aten.neg.default(mul_574)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_5: "f32[8, 768, 7, 7]" = torch.ops.aten.avg_pool2d_backward.default(mul_574, add_211, [3, 3], [1, 1], [1, 1], False, False, None);  mul_574 = add_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_304: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(neg_16, avg_pool2d_backward_5);  neg_16 = avg_pool2d_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_166: "f32[8, 1]" = torch.ops.aten.alias.default(alias_120);  alias_120 = None
    alias_167: "f32[8, 1]" = torch.ops.aten.alias.default(alias_121);  alias_121 = None
    mul_575: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_304, convolution_63)
    view_364: "f32[8, 768, 49]" = torch.ops.aten.view.default(mul_575, [8, 768, 49]);  mul_575 = None
    sum_96: "f32[8, 768]" = torch.ops.aten.sum.dim_IntList(view_364, [2]);  view_364 = None
    view_365: "f32[8, 768, 49]" = torch.ops.aten.view.default(add_304, [8, 768, 49])
    sum_97: "f32[8, 768]" = torch.ops.aten.sum.dim_IntList(view_365, [2]);  view_365 = None
    unsqueeze_542: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_181, 0)
    mul_576: "f32[8, 768]" = torch.ops.aten.mul.Tensor(sum_96, unsqueeze_542);  unsqueeze_542 = None
    view_366: "f32[8, 1, 768]" = torch.ops.aten.view.default(mul_576, [8, 1, 768]);  mul_576 = None
    sum_98: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_366, [2]);  view_366 = None
    unsqueeze_543: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_181, 0)
    mul_577: "f32[8, 768]" = torch.ops.aten.mul.Tensor(sum_97, unsqueeze_543);  unsqueeze_543 = None
    view_367: "f32[8, 1, 768]" = torch.ops.aten.view.default(mul_577, [8, 1, 768]);  mul_577 = None
    sum_99: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_367, [2]);  view_367 = None
    unsqueeze_544: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_167, -1)
    view_368: "f32[1, 1, 768]" = torch.ops.aten.view.default(primals_181, [1, 1, 768]);  primals_181 = None
    mul_578: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(unsqueeze_544, view_368);  unsqueeze_544 = view_368 = None
    mul_579: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_99, alias_166)
    sub_145: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_579, sum_98);  mul_579 = sum_98 = None
    mul_580: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_145, alias_167);  sub_145 = None
    mul_581: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_580, alias_167);  mul_580 = None
    mul_582: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_581, alias_167);  mul_581 = None
    mul_583: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_582, 2.657312925170068e-05);  mul_582 = None
    neg_17: "f32[8, 1]" = torch.ops.aten.neg.default(mul_583)
    mul_584: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_17, alias_166);  neg_17 = None
    mul_585: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_99, alias_167);  sum_99 = None
    mul_586: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_585, 2.657312925170068e-05);  mul_585 = None
    sub_146: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_584, mul_586);  mul_584 = mul_586 = None
    unsqueeze_545: "f32[8, 1, 768, 1]" = torch.ops.aten.unsqueeze.default(mul_578, -1);  mul_578 = None
    unsqueeze_546: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_583, -1);  mul_583 = None
    unsqueeze_547: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, -1);  unsqueeze_546 = None
    unsqueeze_548: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_146, -1);  sub_146 = None
    unsqueeze_549: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, -1);  unsqueeze_548 = None
    view_369: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(add_304, [8, 1, 768, 49]);  add_304 = None
    mul_587: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(view_369, unsqueeze_545);  view_369 = unsqueeze_545 = None
    view_370: "f32[8, 1, 768, 49]" = torch.ops.aten.view.default(convolution_63, [8, 1, 768, 49]);  convolution_63 = None
    mul_588: "f32[8, 1, 768, 49]" = torch.ops.aten.mul.Tensor(view_370, unsqueeze_547);  view_370 = unsqueeze_547 = None
    add_305: "f32[8, 1, 768, 49]" = torch.ops.aten.add.Tensor(mul_587, mul_588);  mul_587 = mul_588 = None
    add_306: "f32[8, 1, 768, 49]" = torch.ops.aten.add.Tensor(add_305, unsqueeze_549);  add_305 = unsqueeze_549 = None
    view_371: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(add_306, [8, 768, 7, 7]);  add_306 = None
    view_372: "f32[8, 1, 768]" = torch.ops.aten.view.default(sum_96, [8, 1, 768]);  sum_96 = None
    view_373: "f32[8, 1, 768]" = torch.ops.aten.view.default(sum_97, [8, 1, 768])
    unsqueeze_550: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_166, -1);  alias_166 = None
    mul_589: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(view_373, unsqueeze_550);  view_373 = unsqueeze_550 = None
    sub_147: "f32[8, 1, 768]" = torch.ops.aten.sub.Tensor(view_372, mul_589);  view_372 = mul_589 = None
    unsqueeze_551: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_167, -1);  alias_167 = None
    mul_590: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_551);  sub_147 = unsqueeze_551 = None
    sum_100: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(mul_590, [0]);  mul_590 = None
    view_374: "f32[768]" = torch.ops.aten.view.default(sum_100, [768]);  sum_100 = None
    sum_101: "f32[768]" = torch.ops.aten.sum.dim_IntList(sum_97, [0]);  sum_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_307: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_303, view_371);  add_303 = view_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:103, code: x = self.conv(x)
    sum_102: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_307, [0, 2, 3])
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(add_307, add_209, primals_345, [768], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_307 = add_209 = primals_345 = None
    getitem_182: "f32[8, 384, 14, 14]" = convolution_backward_12[0]
    getitem_183: "f32[768, 384, 3, 3]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_591: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_182, clone_59);  clone_59 = None
    mul_592: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_182, view_179);  view_179 = None
    sum_103: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_591, [0, 2, 3], True);  mul_591 = None
    view_375: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_103, [384, 1, 1]);  sum_103 = None
    view_376: "f32[384]" = torch.ops.aten.view.default(view_375, [384]);  view_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_104: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_592, [0, 2, 3])
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_592, clone_58, primals_343, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_592 = clone_58 = primals_343 = None
    getitem_185: "f32[8, 1536, 14, 14]" = convolution_backward_13[0]
    getitem_186: "f32[384, 1536, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_593: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_61, 0.7071067811865476)
    erf_42: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_593);  mul_593 = None
    add_308: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
    mul_594: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_308, 0.5);  add_308 = None
    mul_595: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_61, convolution_61)
    mul_596: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_595, -0.5);  mul_595 = None
    exp_6: "f32[8, 1536, 14, 14]" = torch.ops.aten.exp.default(mul_596);  mul_596 = None
    mul_597: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
    mul_598: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_61, mul_597);  convolution_61 = mul_597 = None
    add_309: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_594, mul_598);  mul_594 = mul_598 = None
    mul_599: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_185, add_309);  getitem_185 = add_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_105: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_599, [0, 2, 3])
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_599, add_207, primals_341, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_599 = add_207 = primals_341 = None
    getitem_188: "f32[8, 384, 14, 14]" = convolution_backward_14[0]
    getitem_189: "f32[1536, 384, 1, 1]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_168: "f32[8, 1]" = torch.ops.aten.alias.default(alias_118);  alias_118 = None
    alias_169: "f32[8, 1]" = torch.ops.aten.alias.default(alias_119);  alias_119 = None
    mul_600: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_188, add_205)
    view_377: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_600, [8, 384, 196]);  mul_600 = None
    sum_106: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_377, [2]);  view_377 = None
    view_378: "f32[8, 384, 196]" = torch.ops.aten.view.default(getitem_188, [8, 384, 196])
    sum_107: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_378, [2]);  view_378 = None
    unsqueeze_552: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_178, 0)
    mul_601: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_106, unsqueeze_552);  unsqueeze_552 = None
    view_379: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_601, [8, 1, 384]);  mul_601 = None
    sum_108: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_379, [2]);  view_379 = None
    unsqueeze_553: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_178, 0)
    mul_602: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_107, unsqueeze_553);  unsqueeze_553 = None
    view_380: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_602, [8, 1, 384]);  mul_602 = None
    sum_109: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_380, [2]);  view_380 = None
    unsqueeze_554: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_169, -1)
    view_381: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_178, [1, 1, 384]);  primals_178 = None
    mul_603: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_554, view_381);  unsqueeze_554 = view_381 = None
    mul_604: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_109, alias_168)
    sub_148: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_604, sum_108);  mul_604 = sum_108 = None
    mul_605: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_148, alias_169);  sub_148 = None
    mul_606: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_605, alias_169);  mul_605 = None
    mul_607: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_606, alias_169);  mul_606 = None
    mul_608: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_607, 1.328656462585034e-05);  mul_607 = None
    neg_18: "f32[8, 1]" = torch.ops.aten.neg.default(mul_608)
    mul_609: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_18, alias_168);  neg_18 = None
    mul_610: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_109, alias_169);  sum_109 = None
    mul_611: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_610, 1.328656462585034e-05);  mul_610 = None
    sub_149: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_609, mul_611);  mul_609 = mul_611 = None
    unsqueeze_555: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_603, -1);  mul_603 = None
    unsqueeze_556: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_608, -1);  mul_608 = None
    unsqueeze_557: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, -1);  unsqueeze_556 = None
    unsqueeze_558: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_149, -1);  sub_149 = None
    unsqueeze_559: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, -1);  unsqueeze_558 = None
    view_382: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(getitem_188, [8, 1, 384, 196]);  getitem_188 = None
    mul_612: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_382, unsqueeze_555);  view_382 = unsqueeze_555 = None
    view_383: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_205, [8, 1, 384, 196]);  add_205 = None
    mul_613: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_383, unsqueeze_557);  view_383 = unsqueeze_557 = None
    add_310: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_612, mul_613);  mul_612 = mul_613 = None
    add_311: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_310, unsqueeze_559);  add_310 = unsqueeze_559 = None
    view_384: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_311, [8, 384, 14, 14]);  add_311 = None
    view_385: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_106, [8, 1, 384]);  sum_106 = None
    view_386: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_107, [8, 1, 384])
    unsqueeze_560: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_168, -1);  alias_168 = None
    mul_614: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_386, unsqueeze_560);  view_386 = unsqueeze_560 = None
    sub_150: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_385, mul_614);  view_385 = mul_614 = None
    unsqueeze_561: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_169, -1);  alias_169 = None
    mul_615: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_561);  sub_150 = unsqueeze_561 = None
    sum_110: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_615, [0]);  mul_615 = None
    view_387: "f32[384]" = torch.ops.aten.view.default(sum_110, [384]);  sum_110 = None
    sum_111: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_107, [0]);  sum_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_312: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(getitem_182, view_384);  getitem_182 = view_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_616: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_312, sub_88);  sub_88 = None
    mul_617: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_312, view_176);  view_176 = None
    sum_112: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_616, [0, 2, 3], True);  mul_616 = None
    view_388: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_112, [384, 1, 1]);  sum_112 = None
    view_389: "f32[384]" = torch.ops.aten.view.default(view_388, [384]);  view_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_19: "f32[8, 384, 14, 14]" = torch.ops.aten.neg.default(mul_617)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_6: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d_backward.default(mul_617, add_204, [3, 3], [1, 1], [1, 1], False, False, None);  mul_617 = add_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_313: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(neg_19, avg_pool2d_backward_6);  neg_19 = avg_pool2d_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_170: "f32[8, 1]" = torch.ops.aten.alias.default(alias_116);  alias_116 = None
    alias_171: "f32[8, 1]" = torch.ops.aten.alias.default(alias_117);  alias_117 = None
    mul_618: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_313, add_202)
    view_390: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_618, [8, 384, 196]);  mul_618 = None
    sum_113: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_390, [2]);  view_390 = None
    view_391: "f32[8, 384, 196]" = torch.ops.aten.view.default(add_313, [8, 384, 196])
    sum_114: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_391, [2]);  view_391 = None
    unsqueeze_562: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_175, 0)
    mul_619: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_113, unsqueeze_562);  unsqueeze_562 = None
    view_392: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_619, [8, 1, 384]);  mul_619 = None
    sum_115: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_392, [2]);  view_392 = None
    unsqueeze_563: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_175, 0)
    mul_620: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_114, unsqueeze_563);  unsqueeze_563 = None
    view_393: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_620, [8, 1, 384]);  mul_620 = None
    sum_116: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_393, [2]);  view_393 = None
    unsqueeze_564: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_171, -1)
    view_394: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_175, [1, 1, 384]);  primals_175 = None
    mul_621: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_564, view_394);  unsqueeze_564 = view_394 = None
    mul_622: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_116, alias_170)
    sub_151: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_622, sum_115);  mul_622 = sum_115 = None
    mul_623: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_151, alias_171);  sub_151 = None
    mul_624: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_623, alias_171);  mul_623 = None
    mul_625: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_624, alias_171);  mul_624 = None
    mul_626: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_625, 1.328656462585034e-05);  mul_625 = None
    neg_20: "f32[8, 1]" = torch.ops.aten.neg.default(mul_626)
    mul_627: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_20, alias_170);  neg_20 = None
    mul_628: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_116, alias_171);  sum_116 = None
    mul_629: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_628, 1.328656462585034e-05);  mul_628 = None
    sub_152: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_627, mul_629);  mul_627 = mul_629 = None
    unsqueeze_565: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_621, -1);  mul_621 = None
    unsqueeze_566: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_626, -1);  mul_626 = None
    unsqueeze_567: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, -1);  unsqueeze_566 = None
    unsqueeze_568: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_152, -1);  sub_152 = None
    unsqueeze_569: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, -1);  unsqueeze_568 = None
    view_395: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_313, [8, 1, 384, 196]);  add_313 = None
    mul_630: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_395, unsqueeze_565);  view_395 = unsqueeze_565 = None
    view_396: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_202, [8, 1, 384, 196]);  add_202 = None
    mul_631: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_396, unsqueeze_567);  view_396 = unsqueeze_567 = None
    add_314: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_630, mul_631);  mul_630 = mul_631 = None
    add_315: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_314, unsqueeze_569);  add_314 = unsqueeze_569 = None
    view_397: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_315, [8, 384, 14, 14]);  add_315 = None
    view_398: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_113, [8, 1, 384]);  sum_113 = None
    view_399: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_114, [8, 1, 384])
    unsqueeze_570: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_170, -1);  alias_170 = None
    mul_632: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_399, unsqueeze_570);  view_399 = unsqueeze_570 = None
    sub_153: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_398, mul_632);  view_398 = mul_632 = None
    unsqueeze_571: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_171, -1);  alias_171 = None
    mul_633: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_571);  sub_153 = unsqueeze_571 = None
    sum_117: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_633, [0]);  mul_633 = None
    view_400: "f32[384]" = torch.ops.aten.view.default(sum_117, [384]);  sum_117 = None
    sum_118: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_114, [0]);  sum_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_316: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_312, view_397);  add_312 = view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_634: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_316, clone_57);  clone_57 = None
    mul_635: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_316, view_173);  view_173 = None
    sum_119: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_634, [0, 2, 3], True);  mul_634 = None
    view_401: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_119, [384, 1, 1]);  sum_119 = None
    view_402: "f32[384]" = torch.ops.aten.view.default(view_401, [384]);  view_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_120: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_635, [0, 2, 3])
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_635, clone_56, primals_339, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_635 = clone_56 = primals_339 = None
    getitem_191: "f32[8, 1536, 14, 14]" = convolution_backward_15[0]
    getitem_192: "f32[384, 1536, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_636: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_59, 0.7071067811865476)
    erf_43: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_636);  mul_636 = None
    add_317: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
    mul_637: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_317, 0.5);  add_317 = None
    mul_638: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_59, convolution_59)
    mul_639: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_638, -0.5);  mul_638 = None
    exp_7: "f32[8, 1536, 14, 14]" = torch.ops.aten.exp.default(mul_639);  mul_639 = None
    mul_640: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
    mul_641: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_59, mul_640);  convolution_59 = mul_640 = None
    add_318: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_637, mul_641);  mul_637 = mul_641 = None
    mul_642: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_191, add_318);  getitem_191 = add_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_121: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_642, [0, 2, 3])
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_642, add_200, primals_337, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_642 = add_200 = primals_337 = None
    getitem_194: "f32[8, 384, 14, 14]" = convolution_backward_16[0]
    getitem_195: "f32[1536, 384, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_172: "f32[8, 1]" = torch.ops.aten.alias.default(alias_114);  alias_114 = None
    alias_173: "f32[8, 1]" = torch.ops.aten.alias.default(alias_115);  alias_115 = None
    mul_643: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_194, add_198)
    view_403: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_643, [8, 384, 196]);  mul_643 = None
    sum_122: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_403, [2]);  view_403 = None
    view_404: "f32[8, 384, 196]" = torch.ops.aten.view.default(getitem_194, [8, 384, 196])
    sum_123: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_404, [2]);  view_404 = None
    unsqueeze_572: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_172, 0)
    mul_644: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_122, unsqueeze_572);  unsqueeze_572 = None
    view_405: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_644, [8, 1, 384]);  mul_644 = None
    sum_124: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_405, [2]);  view_405 = None
    unsqueeze_573: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_172, 0)
    mul_645: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_123, unsqueeze_573);  unsqueeze_573 = None
    view_406: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_645, [8, 1, 384]);  mul_645 = None
    sum_125: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_406, [2]);  view_406 = None
    unsqueeze_574: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_173, -1)
    view_407: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_172, [1, 1, 384]);  primals_172 = None
    mul_646: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_574, view_407);  unsqueeze_574 = view_407 = None
    mul_647: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_125, alias_172)
    sub_154: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_647, sum_124);  mul_647 = sum_124 = None
    mul_648: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_154, alias_173);  sub_154 = None
    mul_649: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_648, alias_173);  mul_648 = None
    mul_650: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_649, alias_173);  mul_649 = None
    mul_651: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_650, 1.328656462585034e-05);  mul_650 = None
    neg_21: "f32[8, 1]" = torch.ops.aten.neg.default(mul_651)
    mul_652: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_21, alias_172);  neg_21 = None
    mul_653: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_125, alias_173);  sum_125 = None
    mul_654: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_653, 1.328656462585034e-05);  mul_653 = None
    sub_155: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_652, mul_654);  mul_652 = mul_654 = None
    unsqueeze_575: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_646, -1);  mul_646 = None
    unsqueeze_576: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_651, -1);  mul_651 = None
    unsqueeze_577: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, -1);  unsqueeze_576 = None
    unsqueeze_578: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_155, -1);  sub_155 = None
    unsqueeze_579: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, -1);  unsqueeze_578 = None
    view_408: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(getitem_194, [8, 1, 384, 196]);  getitem_194 = None
    mul_655: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_408, unsqueeze_575);  view_408 = unsqueeze_575 = None
    view_409: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_198, [8, 1, 384, 196]);  add_198 = None
    mul_656: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_409, unsqueeze_577);  view_409 = unsqueeze_577 = None
    add_319: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_655, mul_656);  mul_655 = mul_656 = None
    add_320: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_319, unsqueeze_579);  add_319 = unsqueeze_579 = None
    view_410: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_320, [8, 384, 14, 14]);  add_320 = None
    view_411: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_122, [8, 1, 384]);  sum_122 = None
    view_412: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_123, [8, 1, 384])
    unsqueeze_580: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_172, -1);  alias_172 = None
    mul_657: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_412, unsqueeze_580);  view_412 = unsqueeze_580 = None
    sub_156: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_411, mul_657);  view_411 = mul_657 = None
    unsqueeze_581: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_173, -1);  alias_173 = None
    mul_658: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_581);  sub_156 = unsqueeze_581 = None
    sum_126: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_658, [0]);  mul_658 = None
    view_413: "f32[384]" = torch.ops.aten.view.default(sum_126, [384]);  sum_126 = None
    sum_127: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_123, [0]);  sum_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_321: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_316, view_410);  add_316 = view_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_659: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_321, sub_85);  sub_85 = None
    mul_660: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_321, view_170);  view_170 = None
    sum_128: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_659, [0, 2, 3], True);  mul_659 = None
    view_414: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_128, [384, 1, 1]);  sum_128 = None
    view_415: "f32[384]" = torch.ops.aten.view.default(view_414, [384]);  view_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_22: "f32[8, 384, 14, 14]" = torch.ops.aten.neg.default(mul_660)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_7: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d_backward.default(mul_660, add_197, [3, 3], [1, 1], [1, 1], False, False, None);  mul_660 = add_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_322: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(neg_22, avg_pool2d_backward_7);  neg_22 = avg_pool2d_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_174: "f32[8, 1]" = torch.ops.aten.alias.default(alias_112);  alias_112 = None
    alias_175: "f32[8, 1]" = torch.ops.aten.alias.default(alias_113);  alias_113 = None
    mul_661: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_322, add_195)
    view_416: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_661, [8, 384, 196]);  mul_661 = None
    sum_129: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_416, [2]);  view_416 = None
    view_417: "f32[8, 384, 196]" = torch.ops.aten.view.default(add_322, [8, 384, 196])
    sum_130: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_417, [2]);  view_417 = None
    unsqueeze_582: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_169, 0)
    mul_662: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_129, unsqueeze_582);  unsqueeze_582 = None
    view_418: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_662, [8, 1, 384]);  mul_662 = None
    sum_131: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_418, [2]);  view_418 = None
    unsqueeze_583: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_169, 0)
    mul_663: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_130, unsqueeze_583);  unsqueeze_583 = None
    view_419: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_663, [8, 1, 384]);  mul_663 = None
    sum_132: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_419, [2]);  view_419 = None
    unsqueeze_584: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_175, -1)
    view_420: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_169, [1, 1, 384]);  primals_169 = None
    mul_664: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_584, view_420);  unsqueeze_584 = view_420 = None
    mul_665: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_132, alias_174)
    sub_157: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_665, sum_131);  mul_665 = sum_131 = None
    mul_666: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_157, alias_175);  sub_157 = None
    mul_667: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_666, alias_175);  mul_666 = None
    mul_668: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_667, alias_175);  mul_667 = None
    mul_669: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_668, 1.328656462585034e-05);  mul_668 = None
    neg_23: "f32[8, 1]" = torch.ops.aten.neg.default(mul_669)
    mul_670: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_23, alias_174);  neg_23 = None
    mul_671: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_132, alias_175);  sum_132 = None
    mul_672: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_671, 1.328656462585034e-05);  mul_671 = None
    sub_158: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_670, mul_672);  mul_670 = mul_672 = None
    unsqueeze_585: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_664, -1);  mul_664 = None
    unsqueeze_586: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_669, -1);  mul_669 = None
    unsqueeze_587: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, -1);  unsqueeze_586 = None
    unsqueeze_588: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_158, -1);  sub_158 = None
    unsqueeze_589: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, -1);  unsqueeze_588 = None
    view_421: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_322, [8, 1, 384, 196]);  add_322 = None
    mul_673: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_421, unsqueeze_585);  view_421 = unsqueeze_585 = None
    view_422: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_195, [8, 1, 384, 196]);  add_195 = None
    mul_674: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_422, unsqueeze_587);  view_422 = unsqueeze_587 = None
    add_323: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_673, mul_674);  mul_673 = mul_674 = None
    add_324: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_323, unsqueeze_589);  add_323 = unsqueeze_589 = None
    view_423: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_324, [8, 384, 14, 14]);  add_324 = None
    view_424: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_129, [8, 1, 384]);  sum_129 = None
    view_425: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_130, [8, 1, 384])
    unsqueeze_590: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_174, -1);  alias_174 = None
    mul_675: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_425, unsqueeze_590);  view_425 = unsqueeze_590 = None
    sub_159: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_424, mul_675);  view_424 = mul_675 = None
    unsqueeze_591: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_175, -1);  alias_175 = None
    mul_676: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_591);  sub_159 = unsqueeze_591 = None
    sum_133: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_676, [0]);  mul_676 = None
    view_426: "f32[384]" = torch.ops.aten.view.default(sum_133, [384]);  sum_133 = None
    sum_134: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_130, [0]);  sum_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_325: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_321, view_423);  add_321 = view_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_677: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_325, clone_55);  clone_55 = None
    mul_678: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_325, view_167);  view_167 = None
    sum_135: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_677, [0, 2, 3], True);  mul_677 = None
    view_427: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_135, [384, 1, 1]);  sum_135 = None
    view_428: "f32[384]" = torch.ops.aten.view.default(view_427, [384]);  view_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_136: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_678, [0, 2, 3])
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_678, clone_54, primals_335, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_678 = clone_54 = primals_335 = None
    getitem_197: "f32[8, 1536, 14, 14]" = convolution_backward_17[0]
    getitem_198: "f32[384, 1536, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_679: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_57, 0.7071067811865476)
    erf_44: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_679);  mul_679 = None
    add_326: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_44, 1);  erf_44 = None
    mul_680: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_326, 0.5);  add_326 = None
    mul_681: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_57, convolution_57)
    mul_682: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_681, -0.5);  mul_681 = None
    exp_8: "f32[8, 1536, 14, 14]" = torch.ops.aten.exp.default(mul_682);  mul_682 = None
    mul_683: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
    mul_684: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_57, mul_683);  convolution_57 = mul_683 = None
    add_327: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_680, mul_684);  mul_680 = mul_684 = None
    mul_685: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_197, add_327);  getitem_197 = add_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_137: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_685, [0, 2, 3])
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_685, add_193, primals_333, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_685 = add_193 = primals_333 = None
    getitem_200: "f32[8, 384, 14, 14]" = convolution_backward_18[0]
    getitem_201: "f32[1536, 384, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_176: "f32[8, 1]" = torch.ops.aten.alias.default(alias_110);  alias_110 = None
    alias_177: "f32[8, 1]" = torch.ops.aten.alias.default(alias_111);  alias_111 = None
    mul_686: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_200, add_191)
    view_429: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_686, [8, 384, 196]);  mul_686 = None
    sum_138: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_429, [2]);  view_429 = None
    view_430: "f32[8, 384, 196]" = torch.ops.aten.view.default(getitem_200, [8, 384, 196])
    sum_139: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_430, [2]);  view_430 = None
    unsqueeze_592: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_166, 0)
    mul_687: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_138, unsqueeze_592);  unsqueeze_592 = None
    view_431: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_687, [8, 1, 384]);  mul_687 = None
    sum_140: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_431, [2]);  view_431 = None
    unsqueeze_593: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_166, 0)
    mul_688: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_139, unsqueeze_593);  unsqueeze_593 = None
    view_432: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_688, [8, 1, 384]);  mul_688 = None
    sum_141: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_432, [2]);  view_432 = None
    unsqueeze_594: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_177, -1)
    view_433: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_166, [1, 1, 384]);  primals_166 = None
    mul_689: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_594, view_433);  unsqueeze_594 = view_433 = None
    mul_690: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_141, alias_176)
    sub_160: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_690, sum_140);  mul_690 = sum_140 = None
    mul_691: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_160, alias_177);  sub_160 = None
    mul_692: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_691, alias_177);  mul_691 = None
    mul_693: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_692, alias_177);  mul_692 = None
    mul_694: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_693, 1.328656462585034e-05);  mul_693 = None
    neg_24: "f32[8, 1]" = torch.ops.aten.neg.default(mul_694)
    mul_695: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_24, alias_176);  neg_24 = None
    mul_696: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_141, alias_177);  sum_141 = None
    mul_697: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_696, 1.328656462585034e-05);  mul_696 = None
    sub_161: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_695, mul_697);  mul_695 = mul_697 = None
    unsqueeze_595: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_689, -1);  mul_689 = None
    unsqueeze_596: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_694, -1);  mul_694 = None
    unsqueeze_597: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, -1);  unsqueeze_596 = None
    unsqueeze_598: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_161, -1);  sub_161 = None
    unsqueeze_599: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, -1);  unsqueeze_598 = None
    view_434: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(getitem_200, [8, 1, 384, 196]);  getitem_200 = None
    mul_698: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_434, unsqueeze_595);  view_434 = unsqueeze_595 = None
    view_435: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_191, [8, 1, 384, 196]);  add_191 = None
    mul_699: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_435, unsqueeze_597);  view_435 = unsqueeze_597 = None
    add_328: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_698, mul_699);  mul_698 = mul_699 = None
    add_329: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_328, unsqueeze_599);  add_328 = unsqueeze_599 = None
    view_436: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_329, [8, 384, 14, 14]);  add_329 = None
    view_437: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_138, [8, 1, 384]);  sum_138 = None
    view_438: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_139, [8, 1, 384])
    unsqueeze_600: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_176, -1);  alias_176 = None
    mul_700: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_438, unsqueeze_600);  view_438 = unsqueeze_600 = None
    sub_162: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_437, mul_700);  view_437 = mul_700 = None
    unsqueeze_601: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_177, -1);  alias_177 = None
    mul_701: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_601);  sub_162 = unsqueeze_601 = None
    sum_142: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_701, [0]);  mul_701 = None
    view_439: "f32[384]" = torch.ops.aten.view.default(sum_142, [384]);  sum_142 = None
    sum_143: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_139, [0]);  sum_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_330: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_325, view_436);  add_325 = view_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_702: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_330, sub_82);  sub_82 = None
    mul_703: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_330, view_164);  view_164 = None
    sum_144: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_702, [0, 2, 3], True);  mul_702 = None
    view_440: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_144, [384, 1, 1]);  sum_144 = None
    view_441: "f32[384]" = torch.ops.aten.view.default(view_440, [384]);  view_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_25: "f32[8, 384, 14, 14]" = torch.ops.aten.neg.default(mul_703)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_8: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d_backward.default(mul_703, add_190, [3, 3], [1, 1], [1, 1], False, False, None);  mul_703 = add_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_331: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(neg_25, avg_pool2d_backward_8);  neg_25 = avg_pool2d_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_178: "f32[8, 1]" = torch.ops.aten.alias.default(alias_108);  alias_108 = None
    alias_179: "f32[8, 1]" = torch.ops.aten.alias.default(alias_109);  alias_109 = None
    mul_704: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_331, add_188)
    view_442: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_704, [8, 384, 196]);  mul_704 = None
    sum_145: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_442, [2]);  view_442 = None
    view_443: "f32[8, 384, 196]" = torch.ops.aten.view.default(add_331, [8, 384, 196])
    sum_146: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_443, [2]);  view_443 = None
    unsqueeze_602: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_163, 0)
    mul_705: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_145, unsqueeze_602);  unsqueeze_602 = None
    view_444: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_705, [8, 1, 384]);  mul_705 = None
    sum_147: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_444, [2]);  view_444 = None
    unsqueeze_603: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_163, 0)
    mul_706: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_146, unsqueeze_603);  unsqueeze_603 = None
    view_445: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_706, [8, 1, 384]);  mul_706 = None
    sum_148: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_445, [2]);  view_445 = None
    unsqueeze_604: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_179, -1)
    view_446: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_163, [1, 1, 384]);  primals_163 = None
    mul_707: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_604, view_446);  unsqueeze_604 = view_446 = None
    mul_708: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_148, alias_178)
    sub_163: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_708, sum_147);  mul_708 = sum_147 = None
    mul_709: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_163, alias_179);  sub_163 = None
    mul_710: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_709, alias_179);  mul_709 = None
    mul_711: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_710, alias_179);  mul_710 = None
    mul_712: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_711, 1.328656462585034e-05);  mul_711 = None
    neg_26: "f32[8, 1]" = torch.ops.aten.neg.default(mul_712)
    mul_713: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_26, alias_178);  neg_26 = None
    mul_714: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_148, alias_179);  sum_148 = None
    mul_715: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_714, 1.328656462585034e-05);  mul_714 = None
    sub_164: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_713, mul_715);  mul_713 = mul_715 = None
    unsqueeze_605: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_707, -1);  mul_707 = None
    unsqueeze_606: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_712, -1);  mul_712 = None
    unsqueeze_607: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, -1);  unsqueeze_606 = None
    unsqueeze_608: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_164, -1);  sub_164 = None
    unsqueeze_609: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, -1);  unsqueeze_608 = None
    view_447: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_331, [8, 1, 384, 196]);  add_331 = None
    mul_716: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_447, unsqueeze_605);  view_447 = unsqueeze_605 = None
    view_448: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_188, [8, 1, 384, 196]);  add_188 = None
    mul_717: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_448, unsqueeze_607);  view_448 = unsqueeze_607 = None
    add_332: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_716, mul_717);  mul_716 = mul_717 = None
    add_333: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_332, unsqueeze_609);  add_332 = unsqueeze_609 = None
    view_449: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_333, [8, 384, 14, 14]);  add_333 = None
    view_450: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_145, [8, 1, 384]);  sum_145 = None
    view_451: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_146, [8, 1, 384])
    unsqueeze_610: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_178, -1);  alias_178 = None
    mul_718: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_451, unsqueeze_610);  view_451 = unsqueeze_610 = None
    sub_165: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_450, mul_718);  view_450 = mul_718 = None
    unsqueeze_611: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_179, -1);  alias_179 = None
    mul_719: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_611);  sub_165 = unsqueeze_611 = None
    sum_149: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_719, [0]);  mul_719 = None
    view_452: "f32[384]" = torch.ops.aten.view.default(sum_149, [384]);  sum_149 = None
    sum_150: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_146, [0]);  sum_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_334: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_330, view_449);  add_330 = view_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_720: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_334, clone_53);  clone_53 = None
    mul_721: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_334, view_161);  view_161 = None
    sum_151: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_720, [0, 2, 3], True);  mul_720 = None
    view_453: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_151, [384, 1, 1]);  sum_151 = None
    view_454: "f32[384]" = torch.ops.aten.view.default(view_453, [384]);  view_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_152: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_721, [0, 2, 3])
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_721, clone_52, primals_331, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_721 = clone_52 = primals_331 = None
    getitem_203: "f32[8, 1536, 14, 14]" = convolution_backward_19[0]
    getitem_204: "f32[384, 1536, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_722: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_55, 0.7071067811865476)
    erf_45: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_722);  mul_722 = None
    add_335: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_45, 1);  erf_45 = None
    mul_723: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_335, 0.5);  add_335 = None
    mul_724: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_55, convolution_55)
    mul_725: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_724, -0.5);  mul_724 = None
    exp_9: "f32[8, 1536, 14, 14]" = torch.ops.aten.exp.default(mul_725);  mul_725 = None
    mul_726: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
    mul_727: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_55, mul_726);  convolution_55 = mul_726 = None
    add_336: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_723, mul_727);  mul_723 = mul_727 = None
    mul_728: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_203, add_336);  getitem_203 = add_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_153: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_728, [0, 2, 3])
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_728, add_186, primals_329, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_728 = add_186 = primals_329 = None
    getitem_206: "f32[8, 384, 14, 14]" = convolution_backward_20[0]
    getitem_207: "f32[1536, 384, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_180: "f32[8, 1]" = torch.ops.aten.alias.default(alias_106);  alias_106 = None
    alias_181: "f32[8, 1]" = torch.ops.aten.alias.default(alias_107);  alias_107 = None
    mul_729: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_206, add_184)
    view_455: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_729, [8, 384, 196]);  mul_729 = None
    sum_154: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_455, [2]);  view_455 = None
    view_456: "f32[8, 384, 196]" = torch.ops.aten.view.default(getitem_206, [8, 384, 196])
    sum_155: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_456, [2]);  view_456 = None
    unsqueeze_612: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_160, 0)
    mul_730: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_154, unsqueeze_612);  unsqueeze_612 = None
    view_457: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_730, [8, 1, 384]);  mul_730 = None
    sum_156: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_457, [2]);  view_457 = None
    unsqueeze_613: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_160, 0)
    mul_731: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_155, unsqueeze_613);  unsqueeze_613 = None
    view_458: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_731, [8, 1, 384]);  mul_731 = None
    sum_157: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_458, [2]);  view_458 = None
    unsqueeze_614: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_181, -1)
    view_459: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_160, [1, 1, 384]);  primals_160 = None
    mul_732: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_614, view_459);  unsqueeze_614 = view_459 = None
    mul_733: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_157, alias_180)
    sub_166: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_733, sum_156);  mul_733 = sum_156 = None
    mul_734: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_166, alias_181);  sub_166 = None
    mul_735: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_734, alias_181);  mul_734 = None
    mul_736: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_735, alias_181);  mul_735 = None
    mul_737: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_736, 1.328656462585034e-05);  mul_736 = None
    neg_27: "f32[8, 1]" = torch.ops.aten.neg.default(mul_737)
    mul_738: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_27, alias_180);  neg_27 = None
    mul_739: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_157, alias_181);  sum_157 = None
    mul_740: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_739, 1.328656462585034e-05);  mul_739 = None
    sub_167: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_738, mul_740);  mul_738 = mul_740 = None
    unsqueeze_615: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_732, -1);  mul_732 = None
    unsqueeze_616: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_737, -1);  mul_737 = None
    unsqueeze_617: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, -1);  unsqueeze_616 = None
    unsqueeze_618: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_167, -1);  sub_167 = None
    unsqueeze_619: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, -1);  unsqueeze_618 = None
    view_460: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(getitem_206, [8, 1, 384, 196]);  getitem_206 = None
    mul_741: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_460, unsqueeze_615);  view_460 = unsqueeze_615 = None
    view_461: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_184, [8, 1, 384, 196]);  add_184 = None
    mul_742: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_461, unsqueeze_617);  view_461 = unsqueeze_617 = None
    add_337: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_741, mul_742);  mul_741 = mul_742 = None
    add_338: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_337, unsqueeze_619);  add_337 = unsqueeze_619 = None
    view_462: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_338, [8, 384, 14, 14]);  add_338 = None
    view_463: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_154, [8, 1, 384]);  sum_154 = None
    view_464: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_155, [8, 1, 384])
    unsqueeze_620: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_180, -1);  alias_180 = None
    mul_743: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_464, unsqueeze_620);  view_464 = unsqueeze_620 = None
    sub_168: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_463, mul_743);  view_463 = mul_743 = None
    unsqueeze_621: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_181, -1);  alias_181 = None
    mul_744: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_621);  sub_168 = unsqueeze_621 = None
    sum_158: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_744, [0]);  mul_744 = None
    view_465: "f32[384]" = torch.ops.aten.view.default(sum_158, [384]);  sum_158 = None
    sum_159: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_155, [0]);  sum_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_339: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_334, view_462);  add_334 = view_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_745: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_339, sub_79);  sub_79 = None
    mul_746: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_339, view_158);  view_158 = None
    sum_160: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_745, [0, 2, 3], True);  mul_745 = None
    view_466: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_160, [384, 1, 1]);  sum_160 = None
    view_467: "f32[384]" = torch.ops.aten.view.default(view_466, [384]);  view_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_28: "f32[8, 384, 14, 14]" = torch.ops.aten.neg.default(mul_746)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_9: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d_backward.default(mul_746, add_183, [3, 3], [1, 1], [1, 1], False, False, None);  mul_746 = add_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_340: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(neg_28, avg_pool2d_backward_9);  neg_28 = avg_pool2d_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_182: "f32[8, 1]" = torch.ops.aten.alias.default(alias_104);  alias_104 = None
    alias_183: "f32[8, 1]" = torch.ops.aten.alias.default(alias_105);  alias_105 = None
    mul_747: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_340, add_181)
    view_468: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_747, [8, 384, 196]);  mul_747 = None
    sum_161: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_468, [2]);  view_468 = None
    view_469: "f32[8, 384, 196]" = torch.ops.aten.view.default(add_340, [8, 384, 196])
    sum_162: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_469, [2]);  view_469 = None
    unsqueeze_622: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_157, 0)
    mul_748: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_161, unsqueeze_622);  unsqueeze_622 = None
    view_470: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_748, [8, 1, 384]);  mul_748 = None
    sum_163: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_470, [2]);  view_470 = None
    unsqueeze_623: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_157, 0)
    mul_749: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_162, unsqueeze_623);  unsqueeze_623 = None
    view_471: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_749, [8, 1, 384]);  mul_749 = None
    sum_164: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_471, [2]);  view_471 = None
    unsqueeze_624: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_183, -1)
    view_472: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_157, [1, 1, 384]);  primals_157 = None
    mul_750: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_624, view_472);  unsqueeze_624 = view_472 = None
    mul_751: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_164, alias_182)
    sub_169: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_751, sum_163);  mul_751 = sum_163 = None
    mul_752: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_169, alias_183);  sub_169 = None
    mul_753: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_752, alias_183);  mul_752 = None
    mul_754: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_753, alias_183);  mul_753 = None
    mul_755: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_754, 1.328656462585034e-05);  mul_754 = None
    neg_29: "f32[8, 1]" = torch.ops.aten.neg.default(mul_755)
    mul_756: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_29, alias_182);  neg_29 = None
    mul_757: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_164, alias_183);  sum_164 = None
    mul_758: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_757, 1.328656462585034e-05);  mul_757 = None
    sub_170: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_756, mul_758);  mul_756 = mul_758 = None
    unsqueeze_625: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_750, -1);  mul_750 = None
    unsqueeze_626: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_755, -1);  mul_755 = None
    unsqueeze_627: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, -1);  unsqueeze_626 = None
    unsqueeze_628: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_170, -1);  sub_170 = None
    unsqueeze_629: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, -1);  unsqueeze_628 = None
    view_473: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_340, [8, 1, 384, 196]);  add_340 = None
    mul_759: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_473, unsqueeze_625);  view_473 = unsqueeze_625 = None
    view_474: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_181, [8, 1, 384, 196]);  add_181 = None
    mul_760: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_474, unsqueeze_627);  view_474 = unsqueeze_627 = None
    add_341: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_759, mul_760);  mul_759 = mul_760 = None
    add_342: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_341, unsqueeze_629);  add_341 = unsqueeze_629 = None
    view_475: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_342, [8, 384, 14, 14]);  add_342 = None
    view_476: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_161, [8, 1, 384]);  sum_161 = None
    view_477: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_162, [8, 1, 384])
    unsqueeze_630: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_182, -1);  alias_182 = None
    mul_761: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_477, unsqueeze_630);  view_477 = unsqueeze_630 = None
    sub_171: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_476, mul_761);  view_476 = mul_761 = None
    unsqueeze_631: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_183, -1);  alias_183 = None
    mul_762: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_631);  sub_171 = unsqueeze_631 = None
    sum_165: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_762, [0]);  mul_762 = None
    view_478: "f32[384]" = torch.ops.aten.view.default(sum_165, [384]);  sum_165 = None
    sum_166: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_162, [0]);  sum_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_343: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_339, view_475);  add_339 = view_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_763: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_343, clone_51);  clone_51 = None
    mul_764: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_343, view_155);  view_155 = None
    sum_167: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_763, [0, 2, 3], True);  mul_763 = None
    view_479: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_167, [384, 1, 1]);  sum_167 = None
    view_480: "f32[384]" = torch.ops.aten.view.default(view_479, [384]);  view_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_168: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_764, [0, 2, 3])
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_764, clone_50, primals_327, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_764 = clone_50 = primals_327 = None
    getitem_209: "f32[8, 1536, 14, 14]" = convolution_backward_21[0]
    getitem_210: "f32[384, 1536, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_765: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_53, 0.7071067811865476)
    erf_46: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_765);  mul_765 = None
    add_344: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_46, 1);  erf_46 = None
    mul_766: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_344, 0.5);  add_344 = None
    mul_767: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_53, convolution_53)
    mul_768: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_767, -0.5);  mul_767 = None
    exp_10: "f32[8, 1536, 14, 14]" = torch.ops.aten.exp.default(mul_768);  mul_768 = None
    mul_769: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
    mul_770: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_53, mul_769);  convolution_53 = mul_769 = None
    add_345: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_766, mul_770);  mul_766 = mul_770 = None
    mul_771: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_209, add_345);  getitem_209 = add_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_169: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_771, [0, 2, 3])
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_771, add_179, primals_325, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_771 = add_179 = primals_325 = None
    getitem_212: "f32[8, 384, 14, 14]" = convolution_backward_22[0]
    getitem_213: "f32[1536, 384, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_184: "f32[8, 1]" = torch.ops.aten.alias.default(alias_102);  alias_102 = None
    alias_185: "f32[8, 1]" = torch.ops.aten.alias.default(alias_103);  alias_103 = None
    mul_772: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_212, add_177)
    view_481: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_772, [8, 384, 196]);  mul_772 = None
    sum_170: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_481, [2]);  view_481 = None
    view_482: "f32[8, 384, 196]" = torch.ops.aten.view.default(getitem_212, [8, 384, 196])
    sum_171: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_482, [2]);  view_482 = None
    unsqueeze_632: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_154, 0)
    mul_773: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_170, unsqueeze_632);  unsqueeze_632 = None
    view_483: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_773, [8, 1, 384]);  mul_773 = None
    sum_172: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_483, [2]);  view_483 = None
    unsqueeze_633: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_154, 0)
    mul_774: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_171, unsqueeze_633);  unsqueeze_633 = None
    view_484: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_774, [8, 1, 384]);  mul_774 = None
    sum_173: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_484, [2]);  view_484 = None
    unsqueeze_634: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_185, -1)
    view_485: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_154, [1, 1, 384]);  primals_154 = None
    mul_775: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_634, view_485);  unsqueeze_634 = view_485 = None
    mul_776: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_173, alias_184)
    sub_172: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_776, sum_172);  mul_776 = sum_172 = None
    mul_777: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_172, alias_185);  sub_172 = None
    mul_778: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_777, alias_185);  mul_777 = None
    mul_779: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_778, alias_185);  mul_778 = None
    mul_780: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_779, 1.328656462585034e-05);  mul_779 = None
    neg_30: "f32[8, 1]" = torch.ops.aten.neg.default(mul_780)
    mul_781: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_30, alias_184);  neg_30 = None
    mul_782: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_173, alias_185);  sum_173 = None
    mul_783: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_782, 1.328656462585034e-05);  mul_782 = None
    sub_173: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_781, mul_783);  mul_781 = mul_783 = None
    unsqueeze_635: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_775, -1);  mul_775 = None
    unsqueeze_636: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_780, -1);  mul_780 = None
    unsqueeze_637: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, -1);  unsqueeze_636 = None
    unsqueeze_638: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_173, -1);  sub_173 = None
    unsqueeze_639: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, -1);  unsqueeze_638 = None
    view_486: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(getitem_212, [8, 1, 384, 196]);  getitem_212 = None
    mul_784: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_486, unsqueeze_635);  view_486 = unsqueeze_635 = None
    view_487: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_177, [8, 1, 384, 196]);  add_177 = None
    mul_785: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_487, unsqueeze_637);  view_487 = unsqueeze_637 = None
    add_346: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_784, mul_785);  mul_784 = mul_785 = None
    add_347: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_346, unsqueeze_639);  add_346 = unsqueeze_639 = None
    view_488: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_347, [8, 384, 14, 14]);  add_347 = None
    view_489: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_170, [8, 1, 384]);  sum_170 = None
    view_490: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_171, [8, 1, 384])
    unsqueeze_640: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_184, -1);  alias_184 = None
    mul_786: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_490, unsqueeze_640);  view_490 = unsqueeze_640 = None
    sub_174: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_489, mul_786);  view_489 = mul_786 = None
    unsqueeze_641: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_185, -1);  alias_185 = None
    mul_787: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_641);  sub_174 = unsqueeze_641 = None
    sum_174: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_787, [0]);  mul_787 = None
    view_491: "f32[384]" = torch.ops.aten.view.default(sum_174, [384]);  sum_174 = None
    sum_175: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_171, [0]);  sum_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_348: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_343, view_488);  add_343 = view_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_788: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_348, sub_76);  sub_76 = None
    mul_789: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_348, view_152);  view_152 = None
    sum_176: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_788, [0, 2, 3], True);  mul_788 = None
    view_492: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_176, [384, 1, 1]);  sum_176 = None
    view_493: "f32[384]" = torch.ops.aten.view.default(view_492, [384]);  view_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_31: "f32[8, 384, 14, 14]" = torch.ops.aten.neg.default(mul_789)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_10: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d_backward.default(mul_789, add_176, [3, 3], [1, 1], [1, 1], False, False, None);  mul_789 = add_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_349: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(neg_31, avg_pool2d_backward_10);  neg_31 = avg_pool2d_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_186: "f32[8, 1]" = torch.ops.aten.alias.default(alias_100);  alias_100 = None
    alias_187: "f32[8, 1]" = torch.ops.aten.alias.default(alias_101);  alias_101 = None
    mul_790: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_349, add_174)
    view_494: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_790, [8, 384, 196]);  mul_790 = None
    sum_177: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_494, [2]);  view_494 = None
    view_495: "f32[8, 384, 196]" = torch.ops.aten.view.default(add_349, [8, 384, 196])
    sum_178: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_495, [2]);  view_495 = None
    unsqueeze_642: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_151, 0)
    mul_791: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_177, unsqueeze_642);  unsqueeze_642 = None
    view_496: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_791, [8, 1, 384]);  mul_791 = None
    sum_179: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_496, [2]);  view_496 = None
    unsqueeze_643: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_151, 0)
    mul_792: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_178, unsqueeze_643);  unsqueeze_643 = None
    view_497: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_792, [8, 1, 384]);  mul_792 = None
    sum_180: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_497, [2]);  view_497 = None
    unsqueeze_644: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_187, -1)
    view_498: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_151, [1, 1, 384]);  primals_151 = None
    mul_793: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_644, view_498);  unsqueeze_644 = view_498 = None
    mul_794: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_180, alias_186)
    sub_175: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_794, sum_179);  mul_794 = sum_179 = None
    mul_795: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_175, alias_187);  sub_175 = None
    mul_796: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_795, alias_187);  mul_795 = None
    mul_797: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_796, alias_187);  mul_796 = None
    mul_798: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_797, 1.328656462585034e-05);  mul_797 = None
    neg_32: "f32[8, 1]" = torch.ops.aten.neg.default(mul_798)
    mul_799: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_32, alias_186);  neg_32 = None
    mul_800: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_180, alias_187);  sum_180 = None
    mul_801: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_800, 1.328656462585034e-05);  mul_800 = None
    sub_176: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_799, mul_801);  mul_799 = mul_801 = None
    unsqueeze_645: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_793, -1);  mul_793 = None
    unsqueeze_646: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_798, -1);  mul_798 = None
    unsqueeze_647: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, -1);  unsqueeze_646 = None
    unsqueeze_648: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_176, -1);  sub_176 = None
    unsqueeze_649: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, -1);  unsqueeze_648 = None
    view_499: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_349, [8, 1, 384, 196]);  add_349 = None
    mul_802: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_499, unsqueeze_645);  view_499 = unsqueeze_645 = None
    view_500: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_174, [8, 1, 384, 196]);  add_174 = None
    mul_803: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_500, unsqueeze_647);  view_500 = unsqueeze_647 = None
    add_350: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_802, mul_803);  mul_802 = mul_803 = None
    add_351: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_350, unsqueeze_649);  add_350 = unsqueeze_649 = None
    view_501: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_351, [8, 384, 14, 14]);  add_351 = None
    view_502: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_177, [8, 1, 384]);  sum_177 = None
    view_503: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_178, [8, 1, 384])
    unsqueeze_650: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_186, -1);  alias_186 = None
    mul_804: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_503, unsqueeze_650);  view_503 = unsqueeze_650 = None
    sub_177: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_502, mul_804);  view_502 = mul_804 = None
    unsqueeze_651: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_187, -1);  alias_187 = None
    mul_805: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_651);  sub_177 = unsqueeze_651 = None
    sum_181: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_805, [0]);  mul_805 = None
    view_504: "f32[384]" = torch.ops.aten.view.default(sum_181, [384]);  sum_181 = None
    sum_182: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_178, [0]);  sum_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_352: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_348, view_501);  add_348 = view_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_806: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_352, clone_49);  clone_49 = None
    mul_807: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_352, view_149);  view_149 = None
    sum_183: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_806, [0, 2, 3], True);  mul_806 = None
    view_505: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_183, [384, 1, 1]);  sum_183 = None
    view_506: "f32[384]" = torch.ops.aten.view.default(view_505, [384]);  view_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_184: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_807, [0, 2, 3])
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_807, clone_48, primals_323, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_807 = clone_48 = primals_323 = None
    getitem_215: "f32[8, 1536, 14, 14]" = convolution_backward_23[0]
    getitem_216: "f32[384, 1536, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_808: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_51, 0.7071067811865476)
    erf_47: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_808);  mul_808 = None
    add_353: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_47, 1);  erf_47 = None
    mul_809: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_353, 0.5);  add_353 = None
    mul_810: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_51, convolution_51)
    mul_811: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_810, -0.5);  mul_810 = None
    exp_11: "f32[8, 1536, 14, 14]" = torch.ops.aten.exp.default(mul_811);  mul_811 = None
    mul_812: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
    mul_813: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_51, mul_812);  convolution_51 = mul_812 = None
    add_354: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_809, mul_813);  mul_809 = mul_813 = None
    mul_814: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_215, add_354);  getitem_215 = add_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_185: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_814, [0, 2, 3])
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_814, add_172, primals_321, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_814 = add_172 = primals_321 = None
    getitem_218: "f32[8, 384, 14, 14]" = convolution_backward_24[0]
    getitem_219: "f32[1536, 384, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_188: "f32[8, 1]" = torch.ops.aten.alias.default(alias_98);  alias_98 = None
    alias_189: "f32[8, 1]" = torch.ops.aten.alias.default(alias_99);  alias_99 = None
    mul_815: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_218, add_170)
    view_507: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_815, [8, 384, 196]);  mul_815 = None
    sum_186: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_507, [2]);  view_507 = None
    view_508: "f32[8, 384, 196]" = torch.ops.aten.view.default(getitem_218, [8, 384, 196])
    sum_187: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_508, [2]);  view_508 = None
    unsqueeze_652: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_148, 0)
    mul_816: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_186, unsqueeze_652);  unsqueeze_652 = None
    view_509: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_816, [8, 1, 384]);  mul_816 = None
    sum_188: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_509, [2]);  view_509 = None
    unsqueeze_653: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_148, 0)
    mul_817: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_187, unsqueeze_653);  unsqueeze_653 = None
    view_510: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_817, [8, 1, 384]);  mul_817 = None
    sum_189: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_510, [2]);  view_510 = None
    unsqueeze_654: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_189, -1)
    view_511: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_148, [1, 1, 384]);  primals_148 = None
    mul_818: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_654, view_511);  unsqueeze_654 = view_511 = None
    mul_819: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_189, alias_188)
    sub_178: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_819, sum_188);  mul_819 = sum_188 = None
    mul_820: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_178, alias_189);  sub_178 = None
    mul_821: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_820, alias_189);  mul_820 = None
    mul_822: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_821, alias_189);  mul_821 = None
    mul_823: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_822, 1.328656462585034e-05);  mul_822 = None
    neg_33: "f32[8, 1]" = torch.ops.aten.neg.default(mul_823)
    mul_824: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_33, alias_188);  neg_33 = None
    mul_825: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_189, alias_189);  sum_189 = None
    mul_826: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_825, 1.328656462585034e-05);  mul_825 = None
    sub_179: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_824, mul_826);  mul_824 = mul_826 = None
    unsqueeze_655: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_818, -1);  mul_818 = None
    unsqueeze_656: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_823, -1);  mul_823 = None
    unsqueeze_657: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, -1);  unsqueeze_656 = None
    unsqueeze_658: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_179, -1);  sub_179 = None
    unsqueeze_659: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, -1);  unsqueeze_658 = None
    view_512: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(getitem_218, [8, 1, 384, 196]);  getitem_218 = None
    mul_827: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_512, unsqueeze_655);  view_512 = unsqueeze_655 = None
    view_513: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_170, [8, 1, 384, 196]);  add_170 = None
    mul_828: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_513, unsqueeze_657);  view_513 = unsqueeze_657 = None
    add_355: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_827, mul_828);  mul_827 = mul_828 = None
    add_356: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_355, unsqueeze_659);  add_355 = unsqueeze_659 = None
    view_514: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_356, [8, 384, 14, 14]);  add_356 = None
    view_515: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_186, [8, 1, 384]);  sum_186 = None
    view_516: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_187, [8, 1, 384])
    unsqueeze_660: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_188, -1);  alias_188 = None
    mul_829: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_516, unsqueeze_660);  view_516 = unsqueeze_660 = None
    sub_180: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_515, mul_829);  view_515 = mul_829 = None
    unsqueeze_661: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_189, -1);  alias_189 = None
    mul_830: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_661);  sub_180 = unsqueeze_661 = None
    sum_190: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_830, [0]);  mul_830 = None
    view_517: "f32[384]" = torch.ops.aten.view.default(sum_190, [384]);  sum_190 = None
    sum_191: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_187, [0]);  sum_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_357: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_352, view_514);  add_352 = view_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_831: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_357, sub_73);  sub_73 = None
    mul_832: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_357, view_146);  view_146 = None
    sum_192: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_831, [0, 2, 3], True);  mul_831 = None
    view_518: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_192, [384, 1, 1]);  sum_192 = None
    view_519: "f32[384]" = torch.ops.aten.view.default(view_518, [384]);  view_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_34: "f32[8, 384, 14, 14]" = torch.ops.aten.neg.default(mul_832)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_11: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d_backward.default(mul_832, add_169, [3, 3], [1, 1], [1, 1], False, False, None);  mul_832 = add_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_358: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(neg_34, avg_pool2d_backward_11);  neg_34 = avg_pool2d_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_190: "f32[8, 1]" = torch.ops.aten.alias.default(alias_96);  alias_96 = None
    alias_191: "f32[8, 1]" = torch.ops.aten.alias.default(alias_97);  alias_97 = None
    mul_833: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_358, add_167)
    view_520: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_833, [8, 384, 196]);  mul_833 = None
    sum_193: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_520, [2]);  view_520 = None
    view_521: "f32[8, 384, 196]" = torch.ops.aten.view.default(add_358, [8, 384, 196])
    sum_194: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_521, [2]);  view_521 = None
    unsqueeze_662: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_145, 0)
    mul_834: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_193, unsqueeze_662);  unsqueeze_662 = None
    view_522: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_834, [8, 1, 384]);  mul_834 = None
    sum_195: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_522, [2]);  view_522 = None
    unsqueeze_663: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_145, 0)
    mul_835: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_194, unsqueeze_663);  unsqueeze_663 = None
    view_523: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_835, [8, 1, 384]);  mul_835 = None
    sum_196: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_523, [2]);  view_523 = None
    unsqueeze_664: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_191, -1)
    view_524: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_145, [1, 1, 384]);  primals_145 = None
    mul_836: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_664, view_524);  unsqueeze_664 = view_524 = None
    mul_837: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_196, alias_190)
    sub_181: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_837, sum_195);  mul_837 = sum_195 = None
    mul_838: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_181, alias_191);  sub_181 = None
    mul_839: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_838, alias_191);  mul_838 = None
    mul_840: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_839, alias_191);  mul_839 = None
    mul_841: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_840, 1.328656462585034e-05);  mul_840 = None
    neg_35: "f32[8, 1]" = torch.ops.aten.neg.default(mul_841)
    mul_842: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_35, alias_190);  neg_35 = None
    mul_843: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_196, alias_191);  sum_196 = None
    mul_844: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_843, 1.328656462585034e-05);  mul_843 = None
    sub_182: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_842, mul_844);  mul_842 = mul_844 = None
    unsqueeze_665: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_836, -1);  mul_836 = None
    unsqueeze_666: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_841, -1);  mul_841 = None
    unsqueeze_667: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, -1);  unsqueeze_666 = None
    unsqueeze_668: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_182, -1);  sub_182 = None
    unsqueeze_669: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, -1);  unsqueeze_668 = None
    view_525: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_358, [8, 1, 384, 196]);  add_358 = None
    mul_845: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_525, unsqueeze_665);  view_525 = unsqueeze_665 = None
    view_526: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_167, [8, 1, 384, 196]);  add_167 = None
    mul_846: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_526, unsqueeze_667);  view_526 = unsqueeze_667 = None
    add_359: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_845, mul_846);  mul_845 = mul_846 = None
    add_360: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_359, unsqueeze_669);  add_359 = unsqueeze_669 = None
    view_527: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_360, [8, 384, 14, 14]);  add_360 = None
    view_528: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_193, [8, 1, 384]);  sum_193 = None
    view_529: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_194, [8, 1, 384])
    unsqueeze_670: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_190, -1);  alias_190 = None
    mul_847: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_529, unsqueeze_670);  view_529 = unsqueeze_670 = None
    sub_183: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_528, mul_847);  view_528 = mul_847 = None
    unsqueeze_671: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_191, -1);  alias_191 = None
    mul_848: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_671);  sub_183 = unsqueeze_671 = None
    sum_197: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_848, [0]);  mul_848 = None
    view_530: "f32[384]" = torch.ops.aten.view.default(sum_197, [384]);  sum_197 = None
    sum_198: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_194, [0]);  sum_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_361: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_357, view_527);  add_357 = view_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_849: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_361, clone_47);  clone_47 = None
    mul_850: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_361, view_143);  view_143 = None
    sum_199: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_849, [0, 2, 3], True);  mul_849 = None
    view_531: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_199, [384, 1, 1]);  sum_199 = None
    view_532: "f32[384]" = torch.ops.aten.view.default(view_531, [384]);  view_531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_200: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_850, [0, 2, 3])
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_850, clone_46, primals_319, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_850 = clone_46 = primals_319 = None
    getitem_221: "f32[8, 1536, 14, 14]" = convolution_backward_25[0]
    getitem_222: "f32[384, 1536, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_851: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_49, 0.7071067811865476)
    erf_48: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_851);  mul_851 = None
    add_362: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_48, 1);  erf_48 = None
    mul_852: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_362, 0.5);  add_362 = None
    mul_853: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_49, convolution_49)
    mul_854: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_853, -0.5);  mul_853 = None
    exp_12: "f32[8, 1536, 14, 14]" = torch.ops.aten.exp.default(mul_854);  mul_854 = None
    mul_855: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
    mul_856: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_49, mul_855);  convolution_49 = mul_855 = None
    add_363: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_852, mul_856);  mul_852 = mul_856 = None
    mul_857: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_221, add_363);  getitem_221 = add_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_201: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_857, [0, 2, 3])
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_857, add_165, primals_317, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_857 = add_165 = primals_317 = None
    getitem_224: "f32[8, 384, 14, 14]" = convolution_backward_26[0]
    getitem_225: "f32[1536, 384, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_192: "f32[8, 1]" = torch.ops.aten.alias.default(alias_94);  alias_94 = None
    alias_193: "f32[8, 1]" = torch.ops.aten.alias.default(alias_95);  alias_95 = None
    mul_858: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_224, add_163)
    view_533: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_858, [8, 384, 196]);  mul_858 = None
    sum_202: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_533, [2]);  view_533 = None
    view_534: "f32[8, 384, 196]" = torch.ops.aten.view.default(getitem_224, [8, 384, 196])
    sum_203: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_534, [2]);  view_534 = None
    unsqueeze_672: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_142, 0)
    mul_859: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_202, unsqueeze_672);  unsqueeze_672 = None
    view_535: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_859, [8, 1, 384]);  mul_859 = None
    sum_204: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_535, [2]);  view_535 = None
    unsqueeze_673: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_142, 0)
    mul_860: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_203, unsqueeze_673);  unsqueeze_673 = None
    view_536: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_860, [8, 1, 384]);  mul_860 = None
    sum_205: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_536, [2]);  view_536 = None
    unsqueeze_674: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_193, -1)
    view_537: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_142, [1, 1, 384]);  primals_142 = None
    mul_861: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_674, view_537);  unsqueeze_674 = view_537 = None
    mul_862: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_205, alias_192)
    sub_184: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_862, sum_204);  mul_862 = sum_204 = None
    mul_863: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_184, alias_193);  sub_184 = None
    mul_864: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_863, alias_193);  mul_863 = None
    mul_865: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_864, alias_193);  mul_864 = None
    mul_866: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_865, 1.328656462585034e-05);  mul_865 = None
    neg_36: "f32[8, 1]" = torch.ops.aten.neg.default(mul_866)
    mul_867: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_36, alias_192);  neg_36 = None
    mul_868: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_205, alias_193);  sum_205 = None
    mul_869: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_868, 1.328656462585034e-05);  mul_868 = None
    sub_185: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_867, mul_869);  mul_867 = mul_869 = None
    unsqueeze_675: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_861, -1);  mul_861 = None
    unsqueeze_676: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_866, -1);  mul_866 = None
    unsqueeze_677: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, -1);  unsqueeze_676 = None
    unsqueeze_678: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_185, -1);  sub_185 = None
    unsqueeze_679: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, -1);  unsqueeze_678 = None
    view_538: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(getitem_224, [8, 1, 384, 196]);  getitem_224 = None
    mul_870: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_538, unsqueeze_675);  view_538 = unsqueeze_675 = None
    view_539: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_163, [8, 1, 384, 196]);  add_163 = None
    mul_871: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_539, unsqueeze_677);  view_539 = unsqueeze_677 = None
    add_364: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_870, mul_871);  mul_870 = mul_871 = None
    add_365: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_364, unsqueeze_679);  add_364 = unsqueeze_679 = None
    view_540: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_365, [8, 384, 14, 14]);  add_365 = None
    view_541: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_202, [8, 1, 384]);  sum_202 = None
    view_542: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_203, [8, 1, 384])
    unsqueeze_680: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_192, -1);  alias_192 = None
    mul_872: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_542, unsqueeze_680);  view_542 = unsqueeze_680 = None
    sub_186: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_541, mul_872);  view_541 = mul_872 = None
    unsqueeze_681: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_193, -1);  alias_193 = None
    mul_873: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_186, unsqueeze_681);  sub_186 = unsqueeze_681 = None
    sum_206: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_873, [0]);  mul_873 = None
    view_543: "f32[384]" = torch.ops.aten.view.default(sum_206, [384]);  sum_206 = None
    sum_207: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_203, [0]);  sum_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_366: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_361, view_540);  add_361 = view_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_874: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_366, sub_70);  sub_70 = None
    mul_875: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_366, view_140);  view_140 = None
    sum_208: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_874, [0, 2, 3], True);  mul_874 = None
    view_544: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_208, [384, 1, 1]);  sum_208 = None
    view_545: "f32[384]" = torch.ops.aten.view.default(view_544, [384]);  view_544 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_37: "f32[8, 384, 14, 14]" = torch.ops.aten.neg.default(mul_875)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_12: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d_backward.default(mul_875, add_162, [3, 3], [1, 1], [1, 1], False, False, None);  mul_875 = add_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_367: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(neg_37, avg_pool2d_backward_12);  neg_37 = avg_pool2d_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_194: "f32[8, 1]" = torch.ops.aten.alias.default(alias_92);  alias_92 = None
    alias_195: "f32[8, 1]" = torch.ops.aten.alias.default(alias_93);  alias_93 = None
    mul_876: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_367, add_160)
    view_546: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_876, [8, 384, 196]);  mul_876 = None
    sum_209: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_546, [2]);  view_546 = None
    view_547: "f32[8, 384, 196]" = torch.ops.aten.view.default(add_367, [8, 384, 196])
    sum_210: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_547, [2]);  view_547 = None
    unsqueeze_682: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_139, 0)
    mul_877: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_209, unsqueeze_682);  unsqueeze_682 = None
    view_548: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_877, [8, 1, 384]);  mul_877 = None
    sum_211: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_548, [2]);  view_548 = None
    unsqueeze_683: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_139, 0)
    mul_878: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_210, unsqueeze_683);  unsqueeze_683 = None
    view_549: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_878, [8, 1, 384]);  mul_878 = None
    sum_212: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_549, [2]);  view_549 = None
    unsqueeze_684: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_195, -1)
    view_550: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_139, [1, 1, 384]);  primals_139 = None
    mul_879: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_684, view_550);  unsqueeze_684 = view_550 = None
    mul_880: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_212, alias_194)
    sub_187: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_880, sum_211);  mul_880 = sum_211 = None
    mul_881: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_187, alias_195);  sub_187 = None
    mul_882: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_881, alias_195);  mul_881 = None
    mul_883: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_882, alias_195);  mul_882 = None
    mul_884: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_883, 1.328656462585034e-05);  mul_883 = None
    neg_38: "f32[8, 1]" = torch.ops.aten.neg.default(mul_884)
    mul_885: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_38, alias_194);  neg_38 = None
    mul_886: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_212, alias_195);  sum_212 = None
    mul_887: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_886, 1.328656462585034e-05);  mul_886 = None
    sub_188: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_885, mul_887);  mul_885 = mul_887 = None
    unsqueeze_685: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_879, -1);  mul_879 = None
    unsqueeze_686: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_884, -1);  mul_884 = None
    unsqueeze_687: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, -1);  unsqueeze_686 = None
    unsqueeze_688: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_188, -1);  sub_188 = None
    unsqueeze_689: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, -1);  unsqueeze_688 = None
    view_551: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_367, [8, 1, 384, 196]);  add_367 = None
    mul_888: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_551, unsqueeze_685);  view_551 = unsqueeze_685 = None
    view_552: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_160, [8, 1, 384, 196]);  add_160 = None
    mul_889: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_552, unsqueeze_687);  view_552 = unsqueeze_687 = None
    add_368: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_888, mul_889);  mul_888 = mul_889 = None
    add_369: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_368, unsqueeze_689);  add_368 = unsqueeze_689 = None
    view_553: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_369, [8, 384, 14, 14]);  add_369 = None
    view_554: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_209, [8, 1, 384]);  sum_209 = None
    view_555: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_210, [8, 1, 384])
    unsqueeze_690: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_194, -1);  alias_194 = None
    mul_890: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_555, unsqueeze_690);  view_555 = unsqueeze_690 = None
    sub_189: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_554, mul_890);  view_554 = mul_890 = None
    unsqueeze_691: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_195, -1);  alias_195 = None
    mul_891: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_691);  sub_189 = unsqueeze_691 = None
    sum_213: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_891, [0]);  mul_891 = None
    view_556: "f32[384]" = torch.ops.aten.view.default(sum_213, [384]);  sum_213 = None
    sum_214: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_210, [0]);  sum_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_370: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_366, view_553);  add_366 = view_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_892: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_370, clone_45);  clone_45 = None
    mul_893: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_370, view_137);  view_137 = None
    sum_215: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_892, [0, 2, 3], True);  mul_892 = None
    view_557: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_215, [384, 1, 1]);  sum_215 = None
    view_558: "f32[384]" = torch.ops.aten.view.default(view_557, [384]);  view_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_216: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_893, [0, 2, 3])
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_893, clone_44, primals_315, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_893 = clone_44 = primals_315 = None
    getitem_227: "f32[8, 1536, 14, 14]" = convolution_backward_27[0]
    getitem_228: "f32[384, 1536, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_894: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_47, 0.7071067811865476)
    erf_49: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_894);  mul_894 = None
    add_371: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_49, 1);  erf_49 = None
    mul_895: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_371, 0.5);  add_371 = None
    mul_896: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_47, convolution_47)
    mul_897: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_896, -0.5);  mul_896 = None
    exp_13: "f32[8, 1536, 14, 14]" = torch.ops.aten.exp.default(mul_897);  mul_897 = None
    mul_898: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
    mul_899: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_47, mul_898);  convolution_47 = mul_898 = None
    add_372: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_895, mul_899);  mul_895 = mul_899 = None
    mul_900: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_227, add_372);  getitem_227 = add_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_217: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_900, [0, 2, 3])
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_900, add_158, primals_313, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_900 = add_158 = primals_313 = None
    getitem_230: "f32[8, 384, 14, 14]" = convolution_backward_28[0]
    getitem_231: "f32[1536, 384, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_196: "f32[8, 1]" = torch.ops.aten.alias.default(alias_90);  alias_90 = None
    alias_197: "f32[8, 1]" = torch.ops.aten.alias.default(alias_91);  alias_91 = None
    mul_901: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_230, add_156)
    view_559: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_901, [8, 384, 196]);  mul_901 = None
    sum_218: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_559, [2]);  view_559 = None
    view_560: "f32[8, 384, 196]" = torch.ops.aten.view.default(getitem_230, [8, 384, 196])
    sum_219: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_560, [2]);  view_560 = None
    unsqueeze_692: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_136, 0)
    mul_902: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_218, unsqueeze_692);  unsqueeze_692 = None
    view_561: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_902, [8, 1, 384]);  mul_902 = None
    sum_220: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_561, [2]);  view_561 = None
    unsqueeze_693: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_136, 0)
    mul_903: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_219, unsqueeze_693);  unsqueeze_693 = None
    view_562: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_903, [8, 1, 384]);  mul_903 = None
    sum_221: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_562, [2]);  view_562 = None
    unsqueeze_694: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_197, -1)
    view_563: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_136, [1, 1, 384]);  primals_136 = None
    mul_904: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_694, view_563);  unsqueeze_694 = view_563 = None
    mul_905: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_221, alias_196)
    sub_190: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_905, sum_220);  mul_905 = sum_220 = None
    mul_906: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_190, alias_197);  sub_190 = None
    mul_907: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_906, alias_197);  mul_906 = None
    mul_908: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_907, alias_197);  mul_907 = None
    mul_909: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_908, 1.328656462585034e-05);  mul_908 = None
    neg_39: "f32[8, 1]" = torch.ops.aten.neg.default(mul_909)
    mul_910: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_39, alias_196);  neg_39 = None
    mul_911: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_221, alias_197);  sum_221 = None
    mul_912: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_911, 1.328656462585034e-05);  mul_911 = None
    sub_191: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_910, mul_912);  mul_910 = mul_912 = None
    unsqueeze_695: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_904, -1);  mul_904 = None
    unsqueeze_696: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_909, -1);  mul_909 = None
    unsqueeze_697: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, -1);  unsqueeze_696 = None
    unsqueeze_698: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_191, -1);  sub_191 = None
    unsqueeze_699: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, -1);  unsqueeze_698 = None
    view_564: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(getitem_230, [8, 1, 384, 196]);  getitem_230 = None
    mul_913: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_564, unsqueeze_695);  view_564 = unsqueeze_695 = None
    view_565: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_156, [8, 1, 384, 196]);  add_156 = None
    mul_914: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_565, unsqueeze_697);  view_565 = unsqueeze_697 = None
    add_373: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_913, mul_914);  mul_913 = mul_914 = None
    add_374: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_373, unsqueeze_699);  add_373 = unsqueeze_699 = None
    view_566: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_374, [8, 384, 14, 14]);  add_374 = None
    view_567: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_218, [8, 1, 384]);  sum_218 = None
    view_568: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_219, [8, 1, 384])
    unsqueeze_700: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_196, -1);  alias_196 = None
    mul_915: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_568, unsqueeze_700);  view_568 = unsqueeze_700 = None
    sub_192: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_567, mul_915);  view_567 = mul_915 = None
    unsqueeze_701: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_197, -1);  alias_197 = None
    mul_916: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_701);  sub_192 = unsqueeze_701 = None
    sum_222: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_916, [0]);  mul_916 = None
    view_569: "f32[384]" = torch.ops.aten.view.default(sum_222, [384]);  sum_222 = None
    sum_223: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_219, [0]);  sum_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_375: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_370, view_566);  add_370 = view_566 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_917: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_375, sub_67);  sub_67 = None
    mul_918: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_375, view_134);  view_134 = None
    sum_224: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_917, [0, 2, 3], True);  mul_917 = None
    view_570: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_224, [384, 1, 1]);  sum_224 = None
    view_571: "f32[384]" = torch.ops.aten.view.default(view_570, [384]);  view_570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_40: "f32[8, 384, 14, 14]" = torch.ops.aten.neg.default(mul_918)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_13: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d_backward.default(mul_918, add_155, [3, 3], [1, 1], [1, 1], False, False, None);  mul_918 = add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_376: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(neg_40, avg_pool2d_backward_13);  neg_40 = avg_pool2d_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_198: "f32[8, 1]" = torch.ops.aten.alias.default(alias_88);  alias_88 = None
    alias_199: "f32[8, 1]" = torch.ops.aten.alias.default(alias_89);  alias_89 = None
    mul_919: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_376, add_153)
    view_572: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_919, [8, 384, 196]);  mul_919 = None
    sum_225: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_572, [2]);  view_572 = None
    view_573: "f32[8, 384, 196]" = torch.ops.aten.view.default(add_376, [8, 384, 196])
    sum_226: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_573, [2]);  view_573 = None
    unsqueeze_702: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_133, 0)
    mul_920: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_225, unsqueeze_702);  unsqueeze_702 = None
    view_574: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_920, [8, 1, 384]);  mul_920 = None
    sum_227: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_574, [2]);  view_574 = None
    unsqueeze_703: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_133, 0)
    mul_921: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_226, unsqueeze_703);  unsqueeze_703 = None
    view_575: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_921, [8, 1, 384]);  mul_921 = None
    sum_228: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_575, [2]);  view_575 = None
    unsqueeze_704: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_199, -1)
    view_576: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_133, [1, 1, 384]);  primals_133 = None
    mul_922: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_704, view_576);  unsqueeze_704 = view_576 = None
    mul_923: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_228, alias_198)
    sub_193: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_923, sum_227);  mul_923 = sum_227 = None
    mul_924: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_193, alias_199);  sub_193 = None
    mul_925: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_924, alias_199);  mul_924 = None
    mul_926: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_925, alias_199);  mul_925 = None
    mul_927: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_926, 1.328656462585034e-05);  mul_926 = None
    neg_41: "f32[8, 1]" = torch.ops.aten.neg.default(mul_927)
    mul_928: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_41, alias_198);  neg_41 = None
    mul_929: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_228, alias_199);  sum_228 = None
    mul_930: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_929, 1.328656462585034e-05);  mul_929 = None
    sub_194: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_928, mul_930);  mul_928 = mul_930 = None
    unsqueeze_705: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_922, -1);  mul_922 = None
    unsqueeze_706: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_927, -1);  mul_927 = None
    unsqueeze_707: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, -1);  unsqueeze_706 = None
    unsqueeze_708: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_194, -1);  sub_194 = None
    unsqueeze_709: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_708, -1);  unsqueeze_708 = None
    view_577: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_376, [8, 1, 384, 196]);  add_376 = None
    mul_931: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_577, unsqueeze_705);  view_577 = unsqueeze_705 = None
    view_578: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_153, [8, 1, 384, 196]);  add_153 = None
    mul_932: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_578, unsqueeze_707);  view_578 = unsqueeze_707 = None
    add_377: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_931, mul_932);  mul_931 = mul_932 = None
    add_378: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_377, unsqueeze_709);  add_377 = unsqueeze_709 = None
    view_579: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_378, [8, 384, 14, 14]);  add_378 = None
    view_580: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_225, [8, 1, 384]);  sum_225 = None
    view_581: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_226, [8, 1, 384])
    unsqueeze_710: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_198, -1);  alias_198 = None
    mul_933: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_581, unsqueeze_710);  view_581 = unsqueeze_710 = None
    sub_195: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_580, mul_933);  view_580 = mul_933 = None
    unsqueeze_711: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_199, -1);  alias_199 = None
    mul_934: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_195, unsqueeze_711);  sub_195 = unsqueeze_711 = None
    sum_229: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_934, [0]);  mul_934 = None
    view_582: "f32[384]" = torch.ops.aten.view.default(sum_229, [384]);  sum_229 = None
    sum_230: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_226, [0]);  sum_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_379: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_375, view_579);  add_375 = view_579 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_935: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_379, clone_43);  clone_43 = None
    mul_936: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_379, view_131);  view_131 = None
    sum_231: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_935, [0, 2, 3], True);  mul_935 = None
    view_583: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_231, [384, 1, 1]);  sum_231 = None
    view_584: "f32[384]" = torch.ops.aten.view.default(view_583, [384]);  view_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_232: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_936, [0, 2, 3])
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_936, clone_42, primals_311, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_936 = clone_42 = primals_311 = None
    getitem_233: "f32[8, 1536, 14, 14]" = convolution_backward_29[0]
    getitem_234: "f32[384, 1536, 1, 1]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_937: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_45, 0.7071067811865476)
    erf_50: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_937);  mul_937 = None
    add_380: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_50, 1);  erf_50 = None
    mul_938: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_380, 0.5);  add_380 = None
    mul_939: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_45, convolution_45)
    mul_940: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_939, -0.5);  mul_939 = None
    exp_14: "f32[8, 1536, 14, 14]" = torch.ops.aten.exp.default(mul_940);  mul_940 = None
    mul_941: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
    mul_942: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_45, mul_941);  convolution_45 = mul_941 = None
    add_381: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_938, mul_942);  mul_938 = mul_942 = None
    mul_943: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_233, add_381);  getitem_233 = add_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_233: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_943, [0, 2, 3])
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_943, add_151, primals_309, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_943 = add_151 = primals_309 = None
    getitem_236: "f32[8, 384, 14, 14]" = convolution_backward_30[0]
    getitem_237: "f32[1536, 384, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_200: "f32[8, 1]" = torch.ops.aten.alias.default(alias_86);  alias_86 = None
    alias_201: "f32[8, 1]" = torch.ops.aten.alias.default(alias_87);  alias_87 = None
    mul_944: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_236, add_149)
    view_585: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_944, [8, 384, 196]);  mul_944 = None
    sum_234: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_585, [2]);  view_585 = None
    view_586: "f32[8, 384, 196]" = torch.ops.aten.view.default(getitem_236, [8, 384, 196])
    sum_235: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_586, [2]);  view_586 = None
    unsqueeze_712: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_130, 0)
    mul_945: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_234, unsqueeze_712);  unsqueeze_712 = None
    view_587: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_945, [8, 1, 384]);  mul_945 = None
    sum_236: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_587, [2]);  view_587 = None
    unsqueeze_713: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_130, 0)
    mul_946: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_235, unsqueeze_713);  unsqueeze_713 = None
    view_588: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_946, [8, 1, 384]);  mul_946 = None
    sum_237: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_588, [2]);  view_588 = None
    unsqueeze_714: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_201, -1)
    view_589: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_130, [1, 1, 384]);  primals_130 = None
    mul_947: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_714, view_589);  unsqueeze_714 = view_589 = None
    mul_948: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_237, alias_200)
    sub_196: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_948, sum_236);  mul_948 = sum_236 = None
    mul_949: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_196, alias_201);  sub_196 = None
    mul_950: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_949, alias_201);  mul_949 = None
    mul_951: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_950, alias_201);  mul_950 = None
    mul_952: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_951, 1.328656462585034e-05);  mul_951 = None
    neg_42: "f32[8, 1]" = torch.ops.aten.neg.default(mul_952)
    mul_953: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_42, alias_200);  neg_42 = None
    mul_954: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_237, alias_201);  sum_237 = None
    mul_955: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_954, 1.328656462585034e-05);  mul_954 = None
    sub_197: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_953, mul_955);  mul_953 = mul_955 = None
    unsqueeze_715: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_947, -1);  mul_947 = None
    unsqueeze_716: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_952, -1);  mul_952 = None
    unsqueeze_717: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, -1);  unsqueeze_716 = None
    unsqueeze_718: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_197, -1);  sub_197 = None
    unsqueeze_719: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, -1);  unsqueeze_718 = None
    view_590: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(getitem_236, [8, 1, 384, 196]);  getitem_236 = None
    mul_956: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_590, unsqueeze_715);  view_590 = unsqueeze_715 = None
    view_591: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_149, [8, 1, 384, 196]);  add_149 = None
    mul_957: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_591, unsqueeze_717);  view_591 = unsqueeze_717 = None
    add_382: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_956, mul_957);  mul_956 = mul_957 = None
    add_383: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_382, unsqueeze_719);  add_382 = unsqueeze_719 = None
    view_592: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_383, [8, 384, 14, 14]);  add_383 = None
    view_593: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_234, [8, 1, 384]);  sum_234 = None
    view_594: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_235, [8, 1, 384])
    unsqueeze_720: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_200, -1);  alias_200 = None
    mul_958: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_594, unsqueeze_720);  view_594 = unsqueeze_720 = None
    sub_198: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_593, mul_958);  view_593 = mul_958 = None
    unsqueeze_721: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_201, -1);  alias_201 = None
    mul_959: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_721);  sub_198 = unsqueeze_721 = None
    sum_238: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_959, [0]);  mul_959 = None
    view_595: "f32[384]" = torch.ops.aten.view.default(sum_238, [384]);  sum_238 = None
    sum_239: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_235, [0]);  sum_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_384: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_379, view_592);  add_379 = view_592 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_960: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_384, sub_64);  sub_64 = None
    mul_961: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_384, view_128);  view_128 = None
    sum_240: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_960, [0, 2, 3], True);  mul_960 = None
    view_596: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_240, [384, 1, 1]);  sum_240 = None
    view_597: "f32[384]" = torch.ops.aten.view.default(view_596, [384]);  view_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_43: "f32[8, 384, 14, 14]" = torch.ops.aten.neg.default(mul_961)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_14: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d_backward.default(mul_961, add_148, [3, 3], [1, 1], [1, 1], False, False, None);  mul_961 = add_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_385: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(neg_43, avg_pool2d_backward_14);  neg_43 = avg_pool2d_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_202: "f32[8, 1]" = torch.ops.aten.alias.default(alias_84);  alias_84 = None
    alias_203: "f32[8, 1]" = torch.ops.aten.alias.default(alias_85);  alias_85 = None
    mul_962: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_385, add_146)
    view_598: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_962, [8, 384, 196]);  mul_962 = None
    sum_241: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_598, [2]);  view_598 = None
    view_599: "f32[8, 384, 196]" = torch.ops.aten.view.default(add_385, [8, 384, 196])
    sum_242: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_599, [2]);  view_599 = None
    unsqueeze_722: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_127, 0)
    mul_963: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_241, unsqueeze_722);  unsqueeze_722 = None
    view_600: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_963, [8, 1, 384]);  mul_963 = None
    sum_243: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_600, [2]);  view_600 = None
    unsqueeze_723: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_127, 0)
    mul_964: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_242, unsqueeze_723);  unsqueeze_723 = None
    view_601: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_964, [8, 1, 384]);  mul_964 = None
    sum_244: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_601, [2]);  view_601 = None
    unsqueeze_724: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_203, -1)
    view_602: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_127, [1, 1, 384]);  primals_127 = None
    mul_965: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_724, view_602);  unsqueeze_724 = view_602 = None
    mul_966: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_244, alias_202)
    sub_199: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_966, sum_243);  mul_966 = sum_243 = None
    mul_967: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_199, alias_203);  sub_199 = None
    mul_968: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_967, alias_203);  mul_967 = None
    mul_969: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_968, alias_203);  mul_968 = None
    mul_970: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_969, 1.328656462585034e-05);  mul_969 = None
    neg_44: "f32[8, 1]" = torch.ops.aten.neg.default(mul_970)
    mul_971: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_44, alias_202);  neg_44 = None
    mul_972: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_244, alias_203);  sum_244 = None
    mul_973: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_972, 1.328656462585034e-05);  mul_972 = None
    sub_200: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_971, mul_973);  mul_971 = mul_973 = None
    unsqueeze_725: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_965, -1);  mul_965 = None
    unsqueeze_726: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_970, -1);  mul_970 = None
    unsqueeze_727: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, -1);  unsqueeze_726 = None
    unsqueeze_728: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_200, -1);  sub_200 = None
    unsqueeze_729: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, -1);  unsqueeze_728 = None
    view_603: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_385, [8, 1, 384, 196]);  add_385 = None
    mul_974: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_603, unsqueeze_725);  view_603 = unsqueeze_725 = None
    view_604: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_146, [8, 1, 384, 196]);  add_146 = None
    mul_975: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_604, unsqueeze_727);  view_604 = unsqueeze_727 = None
    add_386: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_974, mul_975);  mul_974 = mul_975 = None
    add_387: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_386, unsqueeze_729);  add_386 = unsqueeze_729 = None
    view_605: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_387, [8, 384, 14, 14]);  add_387 = None
    view_606: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_241, [8, 1, 384]);  sum_241 = None
    view_607: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_242, [8, 1, 384])
    unsqueeze_730: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_202, -1);  alias_202 = None
    mul_976: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_607, unsqueeze_730);  view_607 = unsqueeze_730 = None
    sub_201: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_606, mul_976);  view_606 = mul_976 = None
    unsqueeze_731: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_203, -1);  alias_203 = None
    mul_977: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_731);  sub_201 = unsqueeze_731 = None
    sum_245: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_977, [0]);  mul_977 = None
    view_608: "f32[384]" = torch.ops.aten.view.default(sum_245, [384]);  sum_245 = None
    sum_246: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_242, [0]);  sum_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_388: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_384, view_605);  add_384 = view_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_978: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_388, clone_41);  clone_41 = None
    mul_979: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_388, view_125);  view_125 = None
    sum_247: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_978, [0, 2, 3], True);  mul_978 = None
    view_609: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_247, [384, 1, 1]);  sum_247 = None
    view_610: "f32[384]" = torch.ops.aten.view.default(view_609, [384]);  view_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_248: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_979, [0, 2, 3])
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_979, clone_40, primals_307, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_979 = clone_40 = primals_307 = None
    getitem_239: "f32[8, 1536, 14, 14]" = convolution_backward_31[0]
    getitem_240: "f32[384, 1536, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_980: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_43, 0.7071067811865476)
    erf_51: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_980);  mul_980 = None
    add_389: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_51, 1);  erf_51 = None
    mul_981: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_389, 0.5);  add_389 = None
    mul_982: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_43, convolution_43)
    mul_983: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_982, -0.5);  mul_982 = None
    exp_15: "f32[8, 1536, 14, 14]" = torch.ops.aten.exp.default(mul_983);  mul_983 = None
    mul_984: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
    mul_985: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_43, mul_984);  convolution_43 = mul_984 = None
    add_390: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_981, mul_985);  mul_981 = mul_985 = None
    mul_986: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_239, add_390);  getitem_239 = add_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_249: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_986, [0, 2, 3])
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_986, add_144, primals_305, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_986 = add_144 = primals_305 = None
    getitem_242: "f32[8, 384, 14, 14]" = convolution_backward_32[0]
    getitem_243: "f32[1536, 384, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_204: "f32[8, 1]" = torch.ops.aten.alias.default(alias_82);  alias_82 = None
    alias_205: "f32[8, 1]" = torch.ops.aten.alias.default(alias_83);  alias_83 = None
    mul_987: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_242, add_142)
    view_611: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_987, [8, 384, 196]);  mul_987 = None
    sum_250: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_611, [2]);  view_611 = None
    view_612: "f32[8, 384, 196]" = torch.ops.aten.view.default(getitem_242, [8, 384, 196])
    sum_251: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_612, [2]);  view_612 = None
    unsqueeze_732: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_124, 0)
    mul_988: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_250, unsqueeze_732);  unsqueeze_732 = None
    view_613: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_988, [8, 1, 384]);  mul_988 = None
    sum_252: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_613, [2]);  view_613 = None
    unsqueeze_733: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_124, 0)
    mul_989: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_251, unsqueeze_733);  unsqueeze_733 = None
    view_614: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_989, [8, 1, 384]);  mul_989 = None
    sum_253: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_614, [2]);  view_614 = None
    unsqueeze_734: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_205, -1)
    view_615: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_124, [1, 1, 384]);  primals_124 = None
    mul_990: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_734, view_615);  unsqueeze_734 = view_615 = None
    mul_991: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_253, alias_204)
    sub_202: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_991, sum_252);  mul_991 = sum_252 = None
    mul_992: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_202, alias_205);  sub_202 = None
    mul_993: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_992, alias_205);  mul_992 = None
    mul_994: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_993, alias_205);  mul_993 = None
    mul_995: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_994, 1.328656462585034e-05);  mul_994 = None
    neg_45: "f32[8, 1]" = torch.ops.aten.neg.default(mul_995)
    mul_996: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_45, alias_204);  neg_45 = None
    mul_997: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_253, alias_205);  sum_253 = None
    mul_998: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_997, 1.328656462585034e-05);  mul_997 = None
    sub_203: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_996, mul_998);  mul_996 = mul_998 = None
    unsqueeze_735: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_990, -1);  mul_990 = None
    unsqueeze_736: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_995, -1);  mul_995 = None
    unsqueeze_737: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, -1);  unsqueeze_736 = None
    unsqueeze_738: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_203, -1);  sub_203 = None
    unsqueeze_739: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, -1);  unsqueeze_738 = None
    view_616: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(getitem_242, [8, 1, 384, 196]);  getitem_242 = None
    mul_999: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_616, unsqueeze_735);  view_616 = unsqueeze_735 = None
    view_617: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_142, [8, 1, 384, 196]);  add_142 = None
    mul_1000: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_617, unsqueeze_737);  view_617 = unsqueeze_737 = None
    add_391: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_999, mul_1000);  mul_999 = mul_1000 = None
    add_392: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_391, unsqueeze_739);  add_391 = unsqueeze_739 = None
    view_618: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_392, [8, 384, 14, 14]);  add_392 = None
    view_619: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_250, [8, 1, 384]);  sum_250 = None
    view_620: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_251, [8, 1, 384])
    unsqueeze_740: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_204, -1);  alias_204 = None
    mul_1001: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_620, unsqueeze_740);  view_620 = unsqueeze_740 = None
    sub_204: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_619, mul_1001);  view_619 = mul_1001 = None
    unsqueeze_741: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_205, -1);  alias_205 = None
    mul_1002: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_741);  sub_204 = unsqueeze_741 = None
    sum_254: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_1002, [0]);  mul_1002 = None
    view_621: "f32[384]" = torch.ops.aten.view.default(sum_254, [384]);  sum_254 = None
    sum_255: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_251, [0]);  sum_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_393: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_388, view_618);  add_388 = view_618 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1003: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_393, sub_61);  sub_61 = None
    mul_1004: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_393, view_122);  view_122 = None
    sum_256: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1003, [0, 2, 3], True);  mul_1003 = None
    view_622: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_256, [384, 1, 1]);  sum_256 = None
    view_623: "f32[384]" = torch.ops.aten.view.default(view_622, [384]);  view_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_46: "f32[8, 384, 14, 14]" = torch.ops.aten.neg.default(mul_1004)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_15: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d_backward.default(mul_1004, add_141, [3, 3], [1, 1], [1, 1], False, False, None);  mul_1004 = add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_394: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(neg_46, avg_pool2d_backward_15);  neg_46 = avg_pool2d_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_206: "f32[8, 1]" = torch.ops.aten.alias.default(alias_80);  alias_80 = None
    alias_207: "f32[8, 1]" = torch.ops.aten.alias.default(alias_81);  alias_81 = None
    mul_1005: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_394, add_139)
    view_624: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_1005, [8, 384, 196]);  mul_1005 = None
    sum_257: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_624, [2]);  view_624 = None
    view_625: "f32[8, 384, 196]" = torch.ops.aten.view.default(add_394, [8, 384, 196])
    sum_258: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_625, [2]);  view_625 = None
    unsqueeze_742: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_121, 0)
    mul_1006: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_257, unsqueeze_742);  unsqueeze_742 = None
    view_626: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1006, [8, 1, 384]);  mul_1006 = None
    sum_259: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_626, [2]);  view_626 = None
    unsqueeze_743: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_121, 0)
    mul_1007: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_258, unsqueeze_743);  unsqueeze_743 = None
    view_627: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1007, [8, 1, 384]);  mul_1007 = None
    sum_260: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_627, [2]);  view_627 = None
    unsqueeze_744: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_207, -1)
    view_628: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_121, [1, 1, 384]);  primals_121 = None
    mul_1008: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_744, view_628);  unsqueeze_744 = view_628 = None
    mul_1009: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_260, alias_206)
    sub_205: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1009, sum_259);  mul_1009 = sum_259 = None
    mul_1010: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_205, alias_207);  sub_205 = None
    mul_1011: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1010, alias_207);  mul_1010 = None
    mul_1012: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1011, alias_207);  mul_1011 = None
    mul_1013: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1012, 1.328656462585034e-05);  mul_1012 = None
    neg_47: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1013)
    mul_1014: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_47, alias_206);  neg_47 = None
    mul_1015: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_260, alias_207);  sum_260 = None
    mul_1016: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1015, 1.328656462585034e-05);  mul_1015 = None
    sub_206: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1014, mul_1016);  mul_1014 = mul_1016 = None
    unsqueeze_745: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_1008, -1);  mul_1008 = None
    unsqueeze_746: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1013, -1);  mul_1013 = None
    unsqueeze_747: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, -1);  unsqueeze_746 = None
    unsqueeze_748: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_206, -1);  sub_206 = None
    unsqueeze_749: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, -1);  unsqueeze_748 = None
    view_629: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_394, [8, 1, 384, 196]);  add_394 = None
    mul_1017: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_629, unsqueeze_745);  view_629 = unsqueeze_745 = None
    view_630: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_139, [8, 1, 384, 196]);  add_139 = None
    mul_1018: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_630, unsqueeze_747);  view_630 = unsqueeze_747 = None
    add_395: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_1017, mul_1018);  mul_1017 = mul_1018 = None
    add_396: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_395, unsqueeze_749);  add_395 = unsqueeze_749 = None
    view_631: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_396, [8, 384, 14, 14]);  add_396 = None
    view_632: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_257, [8, 1, 384]);  sum_257 = None
    view_633: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_258, [8, 1, 384])
    unsqueeze_750: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_206, -1);  alias_206 = None
    mul_1019: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_633, unsqueeze_750);  view_633 = unsqueeze_750 = None
    sub_207: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_632, mul_1019);  view_632 = mul_1019 = None
    unsqueeze_751: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_207, -1);  alias_207 = None
    mul_1020: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_751);  sub_207 = unsqueeze_751 = None
    sum_261: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_1020, [0]);  mul_1020 = None
    view_634: "f32[384]" = torch.ops.aten.view.default(sum_261, [384]);  sum_261 = None
    sum_262: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_258, [0]);  sum_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_397: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_393, view_631);  add_393 = view_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1021: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_397, clone_39);  clone_39 = None
    mul_1022: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_397, view_119);  view_119 = None
    sum_263: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1021, [0, 2, 3], True);  mul_1021 = None
    view_635: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_263, [384, 1, 1]);  sum_263 = None
    view_636: "f32[384]" = torch.ops.aten.view.default(view_635, [384]);  view_635 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_264: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_1022, [0, 2, 3])
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_1022, clone_38, primals_303, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1022 = clone_38 = primals_303 = None
    getitem_245: "f32[8, 1536, 14, 14]" = convolution_backward_33[0]
    getitem_246: "f32[384, 1536, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1023: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_41, 0.7071067811865476)
    erf_52: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_1023);  mul_1023 = None
    add_398: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_52, 1);  erf_52 = None
    mul_1024: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_398, 0.5);  add_398 = None
    mul_1025: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_41, convolution_41)
    mul_1026: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_1025, -0.5);  mul_1025 = None
    exp_16: "f32[8, 1536, 14, 14]" = torch.ops.aten.exp.default(mul_1026);  mul_1026 = None
    mul_1027: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
    mul_1028: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_41, mul_1027);  convolution_41 = mul_1027 = None
    add_399: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_1024, mul_1028);  mul_1024 = mul_1028 = None
    mul_1029: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_245, add_399);  getitem_245 = add_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_265: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_1029, [0, 2, 3])
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_1029, add_137, primals_301, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1029 = add_137 = primals_301 = None
    getitem_248: "f32[8, 384, 14, 14]" = convolution_backward_34[0]
    getitem_249: "f32[1536, 384, 1, 1]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_208: "f32[8, 1]" = torch.ops.aten.alias.default(alias_78);  alias_78 = None
    alias_209: "f32[8, 1]" = torch.ops.aten.alias.default(alias_79);  alias_79 = None
    mul_1030: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_248, add_135)
    view_637: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_1030, [8, 384, 196]);  mul_1030 = None
    sum_266: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_637, [2]);  view_637 = None
    view_638: "f32[8, 384, 196]" = torch.ops.aten.view.default(getitem_248, [8, 384, 196])
    sum_267: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_638, [2]);  view_638 = None
    unsqueeze_752: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_118, 0)
    mul_1031: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_266, unsqueeze_752);  unsqueeze_752 = None
    view_639: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1031, [8, 1, 384]);  mul_1031 = None
    sum_268: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_639, [2]);  view_639 = None
    unsqueeze_753: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_118, 0)
    mul_1032: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_267, unsqueeze_753);  unsqueeze_753 = None
    view_640: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1032, [8, 1, 384]);  mul_1032 = None
    sum_269: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_640, [2]);  view_640 = None
    unsqueeze_754: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_209, -1)
    view_641: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_118, [1, 1, 384]);  primals_118 = None
    mul_1033: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_754, view_641);  unsqueeze_754 = view_641 = None
    mul_1034: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_269, alias_208)
    sub_208: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1034, sum_268);  mul_1034 = sum_268 = None
    mul_1035: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_208, alias_209);  sub_208 = None
    mul_1036: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1035, alias_209);  mul_1035 = None
    mul_1037: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1036, alias_209);  mul_1036 = None
    mul_1038: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1037, 1.328656462585034e-05);  mul_1037 = None
    neg_48: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1038)
    mul_1039: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_48, alias_208);  neg_48 = None
    mul_1040: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_269, alias_209);  sum_269 = None
    mul_1041: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1040, 1.328656462585034e-05);  mul_1040 = None
    sub_209: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1039, mul_1041);  mul_1039 = mul_1041 = None
    unsqueeze_755: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_1033, -1);  mul_1033 = None
    unsqueeze_756: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1038, -1);  mul_1038 = None
    unsqueeze_757: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_756, -1);  unsqueeze_756 = None
    unsqueeze_758: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_209, -1);  sub_209 = None
    unsqueeze_759: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, -1);  unsqueeze_758 = None
    view_642: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(getitem_248, [8, 1, 384, 196]);  getitem_248 = None
    mul_1042: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_642, unsqueeze_755);  view_642 = unsqueeze_755 = None
    view_643: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_135, [8, 1, 384, 196]);  add_135 = None
    mul_1043: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_643, unsqueeze_757);  view_643 = unsqueeze_757 = None
    add_400: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_1042, mul_1043);  mul_1042 = mul_1043 = None
    add_401: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_400, unsqueeze_759);  add_400 = unsqueeze_759 = None
    view_644: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_401, [8, 384, 14, 14]);  add_401 = None
    view_645: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_266, [8, 1, 384]);  sum_266 = None
    view_646: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_267, [8, 1, 384])
    unsqueeze_760: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_208, -1);  alias_208 = None
    mul_1044: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_646, unsqueeze_760);  view_646 = unsqueeze_760 = None
    sub_210: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_645, mul_1044);  view_645 = mul_1044 = None
    unsqueeze_761: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_209, -1);  alias_209 = None
    mul_1045: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_210, unsqueeze_761);  sub_210 = unsqueeze_761 = None
    sum_270: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_1045, [0]);  mul_1045 = None
    view_647: "f32[384]" = torch.ops.aten.view.default(sum_270, [384]);  sum_270 = None
    sum_271: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_267, [0]);  sum_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_402: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_397, view_644);  add_397 = view_644 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1046: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_402, sub_58);  sub_58 = None
    mul_1047: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_402, view_116);  view_116 = None
    sum_272: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1046, [0, 2, 3], True);  mul_1046 = None
    view_648: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_272, [384, 1, 1]);  sum_272 = None
    view_649: "f32[384]" = torch.ops.aten.view.default(view_648, [384]);  view_648 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_49: "f32[8, 384, 14, 14]" = torch.ops.aten.neg.default(mul_1047)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_16: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d_backward.default(mul_1047, add_134, [3, 3], [1, 1], [1, 1], False, False, None);  mul_1047 = add_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_403: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(neg_49, avg_pool2d_backward_16);  neg_49 = avg_pool2d_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_210: "f32[8, 1]" = torch.ops.aten.alias.default(alias_76);  alias_76 = None
    alias_211: "f32[8, 1]" = torch.ops.aten.alias.default(alias_77);  alias_77 = None
    mul_1048: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_403, add_132)
    view_650: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_1048, [8, 384, 196]);  mul_1048 = None
    sum_273: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_650, [2]);  view_650 = None
    view_651: "f32[8, 384, 196]" = torch.ops.aten.view.default(add_403, [8, 384, 196])
    sum_274: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_651, [2]);  view_651 = None
    unsqueeze_762: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_115, 0)
    mul_1049: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_273, unsqueeze_762);  unsqueeze_762 = None
    view_652: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1049, [8, 1, 384]);  mul_1049 = None
    sum_275: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_652, [2]);  view_652 = None
    unsqueeze_763: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_115, 0)
    mul_1050: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_274, unsqueeze_763);  unsqueeze_763 = None
    view_653: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1050, [8, 1, 384]);  mul_1050 = None
    sum_276: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_653, [2]);  view_653 = None
    unsqueeze_764: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_211, -1)
    view_654: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_115, [1, 1, 384]);  primals_115 = None
    mul_1051: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_764, view_654);  unsqueeze_764 = view_654 = None
    mul_1052: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_276, alias_210)
    sub_211: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1052, sum_275);  mul_1052 = sum_275 = None
    mul_1053: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_211, alias_211);  sub_211 = None
    mul_1054: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1053, alias_211);  mul_1053 = None
    mul_1055: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1054, alias_211);  mul_1054 = None
    mul_1056: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1055, 1.328656462585034e-05);  mul_1055 = None
    neg_50: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1056)
    mul_1057: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_50, alias_210);  neg_50 = None
    mul_1058: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_276, alias_211);  sum_276 = None
    mul_1059: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1058, 1.328656462585034e-05);  mul_1058 = None
    sub_212: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1057, mul_1059);  mul_1057 = mul_1059 = None
    unsqueeze_765: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_1051, -1);  mul_1051 = None
    unsqueeze_766: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1056, -1);  mul_1056 = None
    unsqueeze_767: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, -1);  unsqueeze_766 = None
    unsqueeze_768: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_212, -1);  sub_212 = None
    unsqueeze_769: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_768, -1);  unsqueeze_768 = None
    view_655: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_403, [8, 1, 384, 196]);  add_403 = None
    mul_1060: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_655, unsqueeze_765);  view_655 = unsqueeze_765 = None
    view_656: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_132, [8, 1, 384, 196]);  add_132 = None
    mul_1061: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_656, unsqueeze_767);  view_656 = unsqueeze_767 = None
    add_404: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_1060, mul_1061);  mul_1060 = mul_1061 = None
    add_405: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_404, unsqueeze_769);  add_404 = unsqueeze_769 = None
    view_657: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_405, [8, 384, 14, 14]);  add_405 = None
    view_658: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_273, [8, 1, 384]);  sum_273 = None
    view_659: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_274, [8, 1, 384])
    unsqueeze_770: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_210, -1);  alias_210 = None
    mul_1062: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_659, unsqueeze_770);  view_659 = unsqueeze_770 = None
    sub_213: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_658, mul_1062);  view_658 = mul_1062 = None
    unsqueeze_771: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_211, -1);  alias_211 = None
    mul_1063: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_771);  sub_213 = unsqueeze_771 = None
    sum_277: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_1063, [0]);  mul_1063 = None
    view_660: "f32[384]" = torch.ops.aten.view.default(sum_277, [384]);  sum_277 = None
    sum_278: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_274, [0]);  sum_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_406: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_402, view_657);  add_402 = view_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1064: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_406, clone_37);  clone_37 = None
    mul_1065: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_406, view_113);  view_113 = None
    sum_279: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1064, [0, 2, 3], True);  mul_1064 = None
    view_661: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_279, [384, 1, 1]);  sum_279 = None
    view_662: "f32[384]" = torch.ops.aten.view.default(view_661, [384]);  view_661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_280: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_1065, [0, 2, 3])
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_1065, clone_36, primals_299, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1065 = clone_36 = primals_299 = None
    getitem_251: "f32[8, 1536, 14, 14]" = convolution_backward_35[0]
    getitem_252: "f32[384, 1536, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1066: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_39, 0.7071067811865476)
    erf_53: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_1066);  mul_1066 = None
    add_407: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_53, 1);  erf_53 = None
    mul_1067: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_407, 0.5);  add_407 = None
    mul_1068: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_39, convolution_39)
    mul_1069: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_1068, -0.5);  mul_1068 = None
    exp_17: "f32[8, 1536, 14, 14]" = torch.ops.aten.exp.default(mul_1069);  mul_1069 = None
    mul_1070: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
    mul_1071: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_39, mul_1070);  convolution_39 = mul_1070 = None
    add_408: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_1067, mul_1071);  mul_1067 = mul_1071 = None
    mul_1072: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_251, add_408);  getitem_251 = add_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_281: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_1072, [0, 2, 3])
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_1072, add_130, primals_297, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1072 = add_130 = primals_297 = None
    getitem_254: "f32[8, 384, 14, 14]" = convolution_backward_36[0]
    getitem_255: "f32[1536, 384, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_212: "f32[8, 1]" = torch.ops.aten.alias.default(alias_74);  alias_74 = None
    alias_213: "f32[8, 1]" = torch.ops.aten.alias.default(alias_75);  alias_75 = None
    mul_1073: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_254, add_128)
    view_663: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_1073, [8, 384, 196]);  mul_1073 = None
    sum_282: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_663, [2]);  view_663 = None
    view_664: "f32[8, 384, 196]" = torch.ops.aten.view.default(getitem_254, [8, 384, 196])
    sum_283: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_664, [2]);  view_664 = None
    unsqueeze_772: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_112, 0)
    mul_1074: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_282, unsqueeze_772);  unsqueeze_772 = None
    view_665: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1074, [8, 1, 384]);  mul_1074 = None
    sum_284: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_665, [2]);  view_665 = None
    unsqueeze_773: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_112, 0)
    mul_1075: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_283, unsqueeze_773);  unsqueeze_773 = None
    view_666: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1075, [8, 1, 384]);  mul_1075 = None
    sum_285: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_666, [2]);  view_666 = None
    unsqueeze_774: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_213, -1)
    view_667: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_112, [1, 1, 384]);  primals_112 = None
    mul_1076: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_774, view_667);  unsqueeze_774 = view_667 = None
    mul_1077: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_285, alias_212)
    sub_214: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1077, sum_284);  mul_1077 = sum_284 = None
    mul_1078: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_214, alias_213);  sub_214 = None
    mul_1079: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1078, alias_213);  mul_1078 = None
    mul_1080: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1079, alias_213);  mul_1079 = None
    mul_1081: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1080, 1.328656462585034e-05);  mul_1080 = None
    neg_51: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1081)
    mul_1082: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_51, alias_212);  neg_51 = None
    mul_1083: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_285, alias_213);  sum_285 = None
    mul_1084: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1083, 1.328656462585034e-05);  mul_1083 = None
    sub_215: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1082, mul_1084);  mul_1082 = mul_1084 = None
    unsqueeze_775: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_1076, -1);  mul_1076 = None
    unsqueeze_776: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1081, -1);  mul_1081 = None
    unsqueeze_777: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, -1);  unsqueeze_776 = None
    unsqueeze_778: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_215, -1);  sub_215 = None
    unsqueeze_779: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, -1);  unsqueeze_778 = None
    view_668: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(getitem_254, [8, 1, 384, 196]);  getitem_254 = None
    mul_1085: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_668, unsqueeze_775);  view_668 = unsqueeze_775 = None
    view_669: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_128, [8, 1, 384, 196]);  add_128 = None
    mul_1086: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_669, unsqueeze_777);  view_669 = unsqueeze_777 = None
    add_409: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_1085, mul_1086);  mul_1085 = mul_1086 = None
    add_410: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_409, unsqueeze_779);  add_409 = unsqueeze_779 = None
    view_670: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_410, [8, 384, 14, 14]);  add_410 = None
    view_671: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_282, [8, 1, 384]);  sum_282 = None
    view_672: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_283, [8, 1, 384])
    unsqueeze_780: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_212, -1);  alias_212 = None
    mul_1087: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_672, unsqueeze_780);  view_672 = unsqueeze_780 = None
    sub_216: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_671, mul_1087);  view_671 = mul_1087 = None
    unsqueeze_781: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_213, -1);  alias_213 = None
    mul_1088: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_781);  sub_216 = unsqueeze_781 = None
    sum_286: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_1088, [0]);  mul_1088 = None
    view_673: "f32[384]" = torch.ops.aten.view.default(sum_286, [384]);  sum_286 = None
    sum_287: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_283, [0]);  sum_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_411: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_406, view_670);  add_406 = view_670 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1089: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_411, sub_55);  sub_55 = None
    mul_1090: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_411, view_110);  view_110 = None
    sum_288: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1089, [0, 2, 3], True);  mul_1089 = None
    view_674: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_288, [384, 1, 1]);  sum_288 = None
    view_675: "f32[384]" = torch.ops.aten.view.default(view_674, [384]);  view_674 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_52: "f32[8, 384, 14, 14]" = torch.ops.aten.neg.default(mul_1090)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_17: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d_backward.default(mul_1090, add_127, [3, 3], [1, 1], [1, 1], False, False, None);  mul_1090 = add_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_412: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(neg_52, avg_pool2d_backward_17);  neg_52 = avg_pool2d_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_214: "f32[8, 1]" = torch.ops.aten.alias.default(alias_72);  alias_72 = None
    alias_215: "f32[8, 1]" = torch.ops.aten.alias.default(alias_73);  alias_73 = None
    mul_1091: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_412, add_125)
    view_676: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_1091, [8, 384, 196]);  mul_1091 = None
    sum_289: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_676, [2]);  view_676 = None
    view_677: "f32[8, 384, 196]" = torch.ops.aten.view.default(add_412, [8, 384, 196])
    sum_290: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_677, [2]);  view_677 = None
    unsqueeze_782: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_109, 0)
    mul_1092: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_289, unsqueeze_782);  unsqueeze_782 = None
    view_678: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1092, [8, 1, 384]);  mul_1092 = None
    sum_291: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_678, [2]);  view_678 = None
    unsqueeze_783: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_109, 0)
    mul_1093: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_290, unsqueeze_783);  unsqueeze_783 = None
    view_679: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1093, [8, 1, 384]);  mul_1093 = None
    sum_292: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_679, [2]);  view_679 = None
    unsqueeze_784: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_215, -1)
    view_680: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_109, [1, 1, 384]);  primals_109 = None
    mul_1094: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_784, view_680);  unsqueeze_784 = view_680 = None
    mul_1095: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_292, alias_214)
    sub_217: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1095, sum_291);  mul_1095 = sum_291 = None
    mul_1096: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_217, alias_215);  sub_217 = None
    mul_1097: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1096, alias_215);  mul_1096 = None
    mul_1098: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1097, alias_215);  mul_1097 = None
    mul_1099: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1098, 1.328656462585034e-05);  mul_1098 = None
    neg_53: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1099)
    mul_1100: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_53, alias_214);  neg_53 = None
    mul_1101: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_292, alias_215);  sum_292 = None
    mul_1102: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1101, 1.328656462585034e-05);  mul_1101 = None
    sub_218: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1100, mul_1102);  mul_1100 = mul_1102 = None
    unsqueeze_785: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_1094, -1);  mul_1094 = None
    unsqueeze_786: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1099, -1);  mul_1099 = None
    unsqueeze_787: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, -1);  unsqueeze_786 = None
    unsqueeze_788: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_218, -1);  sub_218 = None
    unsqueeze_789: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, -1);  unsqueeze_788 = None
    view_681: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_412, [8, 1, 384, 196]);  add_412 = None
    mul_1103: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_681, unsqueeze_785);  view_681 = unsqueeze_785 = None
    view_682: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_125, [8, 1, 384, 196]);  add_125 = None
    mul_1104: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_682, unsqueeze_787);  view_682 = unsqueeze_787 = None
    add_413: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_1103, mul_1104);  mul_1103 = mul_1104 = None
    add_414: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_413, unsqueeze_789);  add_413 = unsqueeze_789 = None
    view_683: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_414, [8, 384, 14, 14]);  add_414 = None
    view_684: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_289, [8, 1, 384]);  sum_289 = None
    view_685: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_290, [8, 1, 384])
    unsqueeze_790: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_214, -1);  alias_214 = None
    mul_1105: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_685, unsqueeze_790);  view_685 = unsqueeze_790 = None
    sub_219: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_684, mul_1105);  view_684 = mul_1105 = None
    unsqueeze_791: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_215, -1);  alias_215 = None
    mul_1106: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_791);  sub_219 = unsqueeze_791 = None
    sum_293: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_1106, [0]);  mul_1106 = None
    view_686: "f32[384]" = torch.ops.aten.view.default(sum_293, [384]);  sum_293 = None
    sum_294: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_290, [0]);  sum_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_415: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_411, view_683);  add_411 = view_683 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1107: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_415, clone_35);  clone_35 = None
    mul_1108: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_415, view_107);  view_107 = None
    sum_295: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1107, [0, 2, 3], True);  mul_1107 = None
    view_687: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_295, [384, 1, 1]);  sum_295 = None
    view_688: "f32[384]" = torch.ops.aten.view.default(view_687, [384]);  view_687 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_296: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_1108, [0, 2, 3])
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_1108, clone_34, primals_295, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1108 = clone_34 = primals_295 = None
    getitem_257: "f32[8, 1536, 14, 14]" = convolution_backward_37[0]
    getitem_258: "f32[384, 1536, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1109: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_37, 0.7071067811865476)
    erf_54: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_1109);  mul_1109 = None
    add_416: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_54, 1);  erf_54 = None
    mul_1110: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_416, 0.5);  add_416 = None
    mul_1111: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_37, convolution_37)
    mul_1112: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_1111, -0.5);  mul_1111 = None
    exp_18: "f32[8, 1536, 14, 14]" = torch.ops.aten.exp.default(mul_1112);  mul_1112 = None
    mul_1113: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_1114: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_37, mul_1113);  convolution_37 = mul_1113 = None
    add_417: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_1110, mul_1114);  mul_1110 = mul_1114 = None
    mul_1115: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_257, add_417);  getitem_257 = add_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_297: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_1115, [0, 2, 3])
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_1115, add_123, primals_293, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1115 = add_123 = primals_293 = None
    getitem_260: "f32[8, 384, 14, 14]" = convolution_backward_38[0]
    getitem_261: "f32[1536, 384, 1, 1]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_216: "f32[8, 1]" = torch.ops.aten.alias.default(alias_70);  alias_70 = None
    alias_217: "f32[8, 1]" = torch.ops.aten.alias.default(alias_71);  alias_71 = None
    mul_1116: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_260, add_121)
    view_689: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_1116, [8, 384, 196]);  mul_1116 = None
    sum_298: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_689, [2]);  view_689 = None
    view_690: "f32[8, 384, 196]" = torch.ops.aten.view.default(getitem_260, [8, 384, 196])
    sum_299: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_690, [2]);  view_690 = None
    unsqueeze_792: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_106, 0)
    mul_1117: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_298, unsqueeze_792);  unsqueeze_792 = None
    view_691: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1117, [8, 1, 384]);  mul_1117 = None
    sum_300: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_691, [2]);  view_691 = None
    unsqueeze_793: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_106, 0)
    mul_1118: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_299, unsqueeze_793);  unsqueeze_793 = None
    view_692: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1118, [8, 1, 384]);  mul_1118 = None
    sum_301: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_692, [2]);  view_692 = None
    unsqueeze_794: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_217, -1)
    view_693: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_106, [1, 1, 384]);  primals_106 = None
    mul_1119: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_794, view_693);  unsqueeze_794 = view_693 = None
    mul_1120: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_301, alias_216)
    sub_220: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1120, sum_300);  mul_1120 = sum_300 = None
    mul_1121: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_220, alias_217);  sub_220 = None
    mul_1122: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1121, alias_217);  mul_1121 = None
    mul_1123: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1122, alias_217);  mul_1122 = None
    mul_1124: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1123, 1.328656462585034e-05);  mul_1123 = None
    neg_54: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1124)
    mul_1125: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_54, alias_216);  neg_54 = None
    mul_1126: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_301, alias_217);  sum_301 = None
    mul_1127: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1126, 1.328656462585034e-05);  mul_1126 = None
    sub_221: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1125, mul_1127);  mul_1125 = mul_1127 = None
    unsqueeze_795: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_1119, -1);  mul_1119 = None
    unsqueeze_796: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1124, -1);  mul_1124 = None
    unsqueeze_797: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, -1);  unsqueeze_796 = None
    unsqueeze_798: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_221, -1);  sub_221 = None
    unsqueeze_799: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, -1);  unsqueeze_798 = None
    view_694: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(getitem_260, [8, 1, 384, 196]);  getitem_260 = None
    mul_1128: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_694, unsqueeze_795);  view_694 = unsqueeze_795 = None
    view_695: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_121, [8, 1, 384, 196]);  add_121 = None
    mul_1129: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_695, unsqueeze_797);  view_695 = unsqueeze_797 = None
    add_418: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_1128, mul_1129);  mul_1128 = mul_1129 = None
    add_419: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_418, unsqueeze_799);  add_418 = unsqueeze_799 = None
    view_696: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_419, [8, 384, 14, 14]);  add_419 = None
    view_697: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_298, [8, 1, 384]);  sum_298 = None
    view_698: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_299, [8, 1, 384])
    unsqueeze_800: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_216, -1);  alias_216 = None
    mul_1130: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_698, unsqueeze_800);  view_698 = unsqueeze_800 = None
    sub_222: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_697, mul_1130);  view_697 = mul_1130 = None
    unsqueeze_801: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_217, -1);  alias_217 = None
    mul_1131: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_801);  sub_222 = unsqueeze_801 = None
    sum_302: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_1131, [0]);  mul_1131 = None
    view_699: "f32[384]" = torch.ops.aten.view.default(sum_302, [384]);  sum_302 = None
    sum_303: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_299, [0]);  sum_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_420: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_415, view_696);  add_415 = view_696 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1132: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_420, sub_52);  sub_52 = None
    mul_1133: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_420, view_104);  view_104 = None
    sum_304: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1132, [0, 2, 3], True);  mul_1132 = None
    view_700: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_304, [384, 1, 1]);  sum_304 = None
    view_701: "f32[384]" = torch.ops.aten.view.default(view_700, [384]);  view_700 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_55: "f32[8, 384, 14, 14]" = torch.ops.aten.neg.default(mul_1133)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_18: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d_backward.default(mul_1133, add_120, [3, 3], [1, 1], [1, 1], False, False, None);  mul_1133 = add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_421: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(neg_55, avg_pool2d_backward_18);  neg_55 = avg_pool2d_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_218: "f32[8, 1]" = torch.ops.aten.alias.default(alias_68);  alias_68 = None
    alias_219: "f32[8, 1]" = torch.ops.aten.alias.default(alias_69);  alias_69 = None
    mul_1134: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_421, add_118)
    view_702: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_1134, [8, 384, 196]);  mul_1134 = None
    sum_305: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_702, [2]);  view_702 = None
    view_703: "f32[8, 384, 196]" = torch.ops.aten.view.default(add_421, [8, 384, 196])
    sum_306: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_703, [2]);  view_703 = None
    unsqueeze_802: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_103, 0)
    mul_1135: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_305, unsqueeze_802);  unsqueeze_802 = None
    view_704: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1135, [8, 1, 384]);  mul_1135 = None
    sum_307: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_704, [2]);  view_704 = None
    unsqueeze_803: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_103, 0)
    mul_1136: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_306, unsqueeze_803);  unsqueeze_803 = None
    view_705: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1136, [8, 1, 384]);  mul_1136 = None
    sum_308: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_705, [2]);  view_705 = None
    unsqueeze_804: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_219, -1)
    view_706: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_103, [1, 1, 384]);  primals_103 = None
    mul_1137: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_804, view_706);  unsqueeze_804 = view_706 = None
    mul_1138: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_308, alias_218)
    sub_223: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1138, sum_307);  mul_1138 = sum_307 = None
    mul_1139: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_223, alias_219);  sub_223 = None
    mul_1140: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1139, alias_219);  mul_1139 = None
    mul_1141: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1140, alias_219);  mul_1140 = None
    mul_1142: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1141, 1.328656462585034e-05);  mul_1141 = None
    neg_56: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1142)
    mul_1143: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_56, alias_218);  neg_56 = None
    mul_1144: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_308, alias_219);  sum_308 = None
    mul_1145: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1144, 1.328656462585034e-05);  mul_1144 = None
    sub_224: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1143, mul_1145);  mul_1143 = mul_1145 = None
    unsqueeze_805: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_1137, -1);  mul_1137 = None
    unsqueeze_806: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1142, -1);  mul_1142 = None
    unsqueeze_807: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, -1);  unsqueeze_806 = None
    unsqueeze_808: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_224, -1);  sub_224 = None
    unsqueeze_809: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, -1);  unsqueeze_808 = None
    view_707: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_421, [8, 1, 384, 196]);  add_421 = None
    mul_1146: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_707, unsqueeze_805);  view_707 = unsqueeze_805 = None
    view_708: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_118, [8, 1, 384, 196]);  add_118 = None
    mul_1147: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_708, unsqueeze_807);  view_708 = unsqueeze_807 = None
    add_422: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_1146, mul_1147);  mul_1146 = mul_1147 = None
    add_423: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_422, unsqueeze_809);  add_422 = unsqueeze_809 = None
    view_709: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_423, [8, 384, 14, 14]);  add_423 = None
    view_710: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_305, [8, 1, 384]);  sum_305 = None
    view_711: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_306, [8, 1, 384])
    unsqueeze_810: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_218, -1);  alias_218 = None
    mul_1148: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_711, unsqueeze_810);  view_711 = unsqueeze_810 = None
    sub_225: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_710, mul_1148);  view_710 = mul_1148 = None
    unsqueeze_811: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_219, -1);  alias_219 = None
    mul_1149: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_811);  sub_225 = unsqueeze_811 = None
    sum_309: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_1149, [0]);  mul_1149 = None
    view_712: "f32[384]" = torch.ops.aten.view.default(sum_309, [384]);  sum_309 = None
    sum_310: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_306, [0]);  sum_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_424: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_420, view_709);  add_420 = view_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1150: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_424, clone_33);  clone_33 = None
    mul_1151: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_424, view_101);  view_101 = None
    sum_311: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1150, [0, 2, 3], True);  mul_1150 = None
    view_713: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_311, [384, 1, 1]);  sum_311 = None
    view_714: "f32[384]" = torch.ops.aten.view.default(view_713, [384]);  view_713 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_312: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_1151, [0, 2, 3])
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_1151, clone_32, primals_291, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1151 = clone_32 = primals_291 = None
    getitem_263: "f32[8, 1536, 14, 14]" = convolution_backward_39[0]
    getitem_264: "f32[384, 1536, 1, 1]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1152: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_35, 0.7071067811865476)
    erf_55: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_1152);  mul_1152 = None
    add_425: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_55, 1);  erf_55 = None
    mul_1153: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_425, 0.5);  add_425 = None
    mul_1154: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_35, convolution_35)
    mul_1155: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_1154, -0.5);  mul_1154 = None
    exp_19: "f32[8, 1536, 14, 14]" = torch.ops.aten.exp.default(mul_1155);  mul_1155 = None
    mul_1156: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_1157: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_35, mul_1156);  convolution_35 = mul_1156 = None
    add_426: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_1153, mul_1157);  mul_1153 = mul_1157 = None
    mul_1158: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_263, add_426);  getitem_263 = add_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_313: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_1158, [0, 2, 3])
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_1158, add_116, primals_289, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1158 = add_116 = primals_289 = None
    getitem_266: "f32[8, 384, 14, 14]" = convolution_backward_40[0]
    getitem_267: "f32[1536, 384, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_220: "f32[8, 1]" = torch.ops.aten.alias.default(alias_66);  alias_66 = None
    alias_221: "f32[8, 1]" = torch.ops.aten.alias.default(alias_67);  alias_67 = None
    mul_1159: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_266, add_114)
    view_715: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_1159, [8, 384, 196]);  mul_1159 = None
    sum_314: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_715, [2]);  view_715 = None
    view_716: "f32[8, 384, 196]" = torch.ops.aten.view.default(getitem_266, [8, 384, 196])
    sum_315: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_716, [2]);  view_716 = None
    unsqueeze_812: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_100, 0)
    mul_1160: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_314, unsqueeze_812);  unsqueeze_812 = None
    view_717: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1160, [8, 1, 384]);  mul_1160 = None
    sum_316: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_717, [2]);  view_717 = None
    unsqueeze_813: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_100, 0)
    mul_1161: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_315, unsqueeze_813);  unsqueeze_813 = None
    view_718: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1161, [8, 1, 384]);  mul_1161 = None
    sum_317: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_718, [2]);  view_718 = None
    unsqueeze_814: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_221, -1)
    view_719: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_100, [1, 1, 384]);  primals_100 = None
    mul_1162: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_814, view_719);  unsqueeze_814 = view_719 = None
    mul_1163: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_317, alias_220)
    sub_226: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1163, sum_316);  mul_1163 = sum_316 = None
    mul_1164: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_226, alias_221);  sub_226 = None
    mul_1165: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1164, alias_221);  mul_1164 = None
    mul_1166: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1165, alias_221);  mul_1165 = None
    mul_1167: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1166, 1.328656462585034e-05);  mul_1166 = None
    neg_57: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1167)
    mul_1168: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_57, alias_220);  neg_57 = None
    mul_1169: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_317, alias_221);  sum_317 = None
    mul_1170: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1169, 1.328656462585034e-05);  mul_1169 = None
    sub_227: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1168, mul_1170);  mul_1168 = mul_1170 = None
    unsqueeze_815: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_1162, -1);  mul_1162 = None
    unsqueeze_816: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1167, -1);  mul_1167 = None
    unsqueeze_817: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_816, -1);  unsqueeze_816 = None
    unsqueeze_818: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_227, -1);  sub_227 = None
    unsqueeze_819: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, -1);  unsqueeze_818 = None
    view_720: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(getitem_266, [8, 1, 384, 196]);  getitem_266 = None
    mul_1171: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_720, unsqueeze_815);  view_720 = unsqueeze_815 = None
    view_721: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_114, [8, 1, 384, 196]);  add_114 = None
    mul_1172: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_721, unsqueeze_817);  view_721 = unsqueeze_817 = None
    add_427: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_1171, mul_1172);  mul_1171 = mul_1172 = None
    add_428: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_427, unsqueeze_819);  add_427 = unsqueeze_819 = None
    view_722: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_428, [8, 384, 14, 14]);  add_428 = None
    view_723: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_314, [8, 1, 384]);  sum_314 = None
    view_724: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_315, [8, 1, 384])
    unsqueeze_820: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_220, -1);  alias_220 = None
    mul_1173: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_724, unsqueeze_820);  view_724 = unsqueeze_820 = None
    sub_228: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_723, mul_1173);  view_723 = mul_1173 = None
    unsqueeze_821: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_221, -1);  alias_221 = None
    mul_1174: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_821);  sub_228 = unsqueeze_821 = None
    sum_318: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_1174, [0]);  mul_1174 = None
    view_725: "f32[384]" = torch.ops.aten.view.default(sum_318, [384]);  sum_318 = None
    sum_319: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_315, [0]);  sum_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_429: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_424, view_722);  add_424 = view_722 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1175: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_429, sub_49);  sub_49 = None
    mul_1176: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_429, view_98);  view_98 = None
    sum_320: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1175, [0, 2, 3], True);  mul_1175 = None
    view_726: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_320, [384, 1, 1]);  sum_320 = None
    view_727: "f32[384]" = torch.ops.aten.view.default(view_726, [384]);  view_726 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_58: "f32[8, 384, 14, 14]" = torch.ops.aten.neg.default(mul_1176)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_19: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d_backward.default(mul_1176, add_113, [3, 3], [1, 1], [1, 1], False, False, None);  mul_1176 = add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_430: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(neg_58, avg_pool2d_backward_19);  neg_58 = avg_pool2d_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_222: "f32[8, 1]" = torch.ops.aten.alias.default(alias_64);  alias_64 = None
    alias_223: "f32[8, 1]" = torch.ops.aten.alias.default(alias_65);  alias_65 = None
    mul_1177: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_430, add_111)
    view_728: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_1177, [8, 384, 196]);  mul_1177 = None
    sum_321: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_728, [2]);  view_728 = None
    view_729: "f32[8, 384, 196]" = torch.ops.aten.view.default(add_430, [8, 384, 196])
    sum_322: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_729, [2]);  view_729 = None
    unsqueeze_822: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_97, 0)
    mul_1178: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_321, unsqueeze_822);  unsqueeze_822 = None
    view_730: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1178, [8, 1, 384]);  mul_1178 = None
    sum_323: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_730, [2]);  view_730 = None
    unsqueeze_823: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_97, 0)
    mul_1179: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_322, unsqueeze_823);  unsqueeze_823 = None
    view_731: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1179, [8, 1, 384]);  mul_1179 = None
    sum_324: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_731, [2]);  view_731 = None
    unsqueeze_824: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_223, -1)
    view_732: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_97, [1, 1, 384]);  primals_97 = None
    mul_1180: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_824, view_732);  unsqueeze_824 = view_732 = None
    mul_1181: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_324, alias_222)
    sub_229: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1181, sum_323);  mul_1181 = sum_323 = None
    mul_1182: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_229, alias_223);  sub_229 = None
    mul_1183: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1182, alias_223);  mul_1182 = None
    mul_1184: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1183, alias_223);  mul_1183 = None
    mul_1185: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1184, 1.328656462585034e-05);  mul_1184 = None
    neg_59: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1185)
    mul_1186: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_59, alias_222);  neg_59 = None
    mul_1187: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_324, alias_223);  sum_324 = None
    mul_1188: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1187, 1.328656462585034e-05);  mul_1187 = None
    sub_230: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1186, mul_1188);  mul_1186 = mul_1188 = None
    unsqueeze_825: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_1180, -1);  mul_1180 = None
    unsqueeze_826: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1185, -1);  mul_1185 = None
    unsqueeze_827: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, -1);  unsqueeze_826 = None
    unsqueeze_828: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_230, -1);  sub_230 = None
    unsqueeze_829: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_828, -1);  unsqueeze_828 = None
    view_733: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_430, [8, 1, 384, 196]);  add_430 = None
    mul_1189: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_733, unsqueeze_825);  view_733 = unsqueeze_825 = None
    view_734: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_111, [8, 1, 384, 196]);  add_111 = None
    mul_1190: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_734, unsqueeze_827);  view_734 = unsqueeze_827 = None
    add_431: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_1189, mul_1190);  mul_1189 = mul_1190 = None
    add_432: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_431, unsqueeze_829);  add_431 = unsqueeze_829 = None
    view_735: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_432, [8, 384, 14, 14]);  add_432 = None
    view_736: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_321, [8, 1, 384]);  sum_321 = None
    view_737: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_322, [8, 1, 384])
    unsqueeze_830: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_222, -1);  alias_222 = None
    mul_1191: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_737, unsqueeze_830);  view_737 = unsqueeze_830 = None
    sub_231: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_736, mul_1191);  view_736 = mul_1191 = None
    unsqueeze_831: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_223, -1);  alias_223 = None
    mul_1192: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_231, unsqueeze_831);  sub_231 = unsqueeze_831 = None
    sum_325: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_1192, [0]);  mul_1192 = None
    view_738: "f32[384]" = torch.ops.aten.view.default(sum_325, [384]);  sum_325 = None
    sum_326: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_322, [0]);  sum_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_433: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_429, view_735);  add_429 = view_735 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1193: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_433, clone_31);  clone_31 = None
    mul_1194: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_433, view_95);  view_95 = None
    sum_327: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1193, [0, 2, 3], True);  mul_1193 = None
    view_739: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_327, [384, 1, 1]);  sum_327 = None
    view_740: "f32[384]" = torch.ops.aten.view.default(view_739, [384]);  view_739 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_328: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_1194, [0, 2, 3])
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_1194, clone_30, primals_287, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1194 = clone_30 = primals_287 = None
    getitem_269: "f32[8, 1536, 14, 14]" = convolution_backward_41[0]
    getitem_270: "f32[384, 1536, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1195: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_33, 0.7071067811865476)
    erf_56: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_1195);  mul_1195 = None
    add_434: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_56, 1);  erf_56 = None
    mul_1196: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_434, 0.5);  add_434 = None
    mul_1197: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_33, convolution_33)
    mul_1198: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_1197, -0.5);  mul_1197 = None
    exp_20: "f32[8, 1536, 14, 14]" = torch.ops.aten.exp.default(mul_1198);  mul_1198 = None
    mul_1199: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_1200: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_33, mul_1199);  convolution_33 = mul_1199 = None
    add_435: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_1196, mul_1200);  mul_1196 = mul_1200 = None
    mul_1201: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_269, add_435);  getitem_269 = add_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_329: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_1201, [0, 2, 3])
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_1201, add_109, primals_285, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1201 = add_109 = primals_285 = None
    getitem_272: "f32[8, 384, 14, 14]" = convolution_backward_42[0]
    getitem_273: "f32[1536, 384, 1, 1]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_224: "f32[8, 1]" = torch.ops.aten.alias.default(alias_62);  alias_62 = None
    alias_225: "f32[8, 1]" = torch.ops.aten.alias.default(alias_63);  alias_63 = None
    mul_1202: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_272, add_107)
    view_741: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_1202, [8, 384, 196]);  mul_1202 = None
    sum_330: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_741, [2]);  view_741 = None
    view_742: "f32[8, 384, 196]" = torch.ops.aten.view.default(getitem_272, [8, 384, 196])
    sum_331: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_742, [2]);  view_742 = None
    unsqueeze_832: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_94, 0)
    mul_1203: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_330, unsqueeze_832);  unsqueeze_832 = None
    view_743: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1203, [8, 1, 384]);  mul_1203 = None
    sum_332: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_743, [2]);  view_743 = None
    unsqueeze_833: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_94, 0)
    mul_1204: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_331, unsqueeze_833);  unsqueeze_833 = None
    view_744: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1204, [8, 1, 384]);  mul_1204 = None
    sum_333: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_744, [2]);  view_744 = None
    unsqueeze_834: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_225, -1)
    view_745: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_94, [1, 1, 384]);  primals_94 = None
    mul_1205: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_834, view_745);  unsqueeze_834 = view_745 = None
    mul_1206: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_333, alias_224)
    sub_232: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1206, sum_332);  mul_1206 = sum_332 = None
    mul_1207: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_232, alias_225);  sub_232 = None
    mul_1208: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1207, alias_225);  mul_1207 = None
    mul_1209: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1208, alias_225);  mul_1208 = None
    mul_1210: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1209, 1.328656462585034e-05);  mul_1209 = None
    neg_60: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1210)
    mul_1211: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_60, alias_224);  neg_60 = None
    mul_1212: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_333, alias_225);  sum_333 = None
    mul_1213: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1212, 1.328656462585034e-05);  mul_1212 = None
    sub_233: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1211, mul_1213);  mul_1211 = mul_1213 = None
    unsqueeze_835: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_1205, -1);  mul_1205 = None
    unsqueeze_836: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1210, -1);  mul_1210 = None
    unsqueeze_837: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, -1);  unsqueeze_836 = None
    unsqueeze_838: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_233, -1);  sub_233 = None
    unsqueeze_839: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_838, -1);  unsqueeze_838 = None
    view_746: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(getitem_272, [8, 1, 384, 196]);  getitem_272 = None
    mul_1214: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_746, unsqueeze_835);  view_746 = unsqueeze_835 = None
    view_747: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_107, [8, 1, 384, 196]);  add_107 = None
    mul_1215: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_747, unsqueeze_837);  view_747 = unsqueeze_837 = None
    add_436: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_1214, mul_1215);  mul_1214 = mul_1215 = None
    add_437: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_436, unsqueeze_839);  add_436 = unsqueeze_839 = None
    view_748: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_437, [8, 384, 14, 14]);  add_437 = None
    view_749: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_330, [8, 1, 384]);  sum_330 = None
    view_750: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_331, [8, 1, 384])
    unsqueeze_840: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_224, -1);  alias_224 = None
    mul_1216: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_750, unsqueeze_840);  view_750 = unsqueeze_840 = None
    sub_234: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_749, mul_1216);  view_749 = mul_1216 = None
    unsqueeze_841: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_225, -1);  alias_225 = None
    mul_1217: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_234, unsqueeze_841);  sub_234 = unsqueeze_841 = None
    sum_334: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_1217, [0]);  mul_1217 = None
    view_751: "f32[384]" = torch.ops.aten.view.default(sum_334, [384]);  sum_334 = None
    sum_335: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_331, [0]);  sum_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_438: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_433, view_748);  add_433 = view_748 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1218: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_438, sub_46);  sub_46 = None
    mul_1219: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_438, view_92);  view_92 = None
    sum_336: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1218, [0, 2, 3], True);  mul_1218 = None
    view_752: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_336, [384, 1, 1]);  sum_336 = None
    view_753: "f32[384]" = torch.ops.aten.view.default(view_752, [384]);  view_752 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_61: "f32[8, 384, 14, 14]" = torch.ops.aten.neg.default(mul_1219)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_20: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d_backward.default(mul_1219, add_106, [3, 3], [1, 1], [1, 1], False, False, None);  mul_1219 = add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_439: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(neg_61, avg_pool2d_backward_20);  neg_61 = avg_pool2d_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_226: "f32[8, 1]" = torch.ops.aten.alias.default(alias_60);  alias_60 = None
    alias_227: "f32[8, 1]" = torch.ops.aten.alias.default(alias_61);  alias_61 = None
    mul_1220: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_439, add_104)
    view_754: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_1220, [8, 384, 196]);  mul_1220 = None
    sum_337: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_754, [2]);  view_754 = None
    view_755: "f32[8, 384, 196]" = torch.ops.aten.view.default(add_439, [8, 384, 196])
    sum_338: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_755, [2]);  view_755 = None
    unsqueeze_842: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_91, 0)
    mul_1221: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_337, unsqueeze_842);  unsqueeze_842 = None
    view_756: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1221, [8, 1, 384]);  mul_1221 = None
    sum_339: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_756, [2]);  view_756 = None
    unsqueeze_843: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_91, 0)
    mul_1222: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_338, unsqueeze_843);  unsqueeze_843 = None
    view_757: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1222, [8, 1, 384]);  mul_1222 = None
    sum_340: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_757, [2]);  view_757 = None
    unsqueeze_844: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_227, -1)
    view_758: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_91, [1, 1, 384]);  primals_91 = None
    mul_1223: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_844, view_758);  unsqueeze_844 = view_758 = None
    mul_1224: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_340, alias_226)
    sub_235: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1224, sum_339);  mul_1224 = sum_339 = None
    mul_1225: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_235, alias_227);  sub_235 = None
    mul_1226: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1225, alias_227);  mul_1225 = None
    mul_1227: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1226, alias_227);  mul_1226 = None
    mul_1228: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1227, 1.328656462585034e-05);  mul_1227 = None
    neg_62: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1228)
    mul_1229: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_62, alias_226);  neg_62 = None
    mul_1230: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_340, alias_227);  sum_340 = None
    mul_1231: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1230, 1.328656462585034e-05);  mul_1230 = None
    sub_236: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1229, mul_1231);  mul_1229 = mul_1231 = None
    unsqueeze_845: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_1223, -1);  mul_1223 = None
    unsqueeze_846: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1228, -1);  mul_1228 = None
    unsqueeze_847: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, -1);  unsqueeze_846 = None
    unsqueeze_848: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_236, -1);  sub_236 = None
    unsqueeze_849: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, -1);  unsqueeze_848 = None
    view_759: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_439, [8, 1, 384, 196]);  add_439 = None
    mul_1232: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_759, unsqueeze_845);  view_759 = unsqueeze_845 = None
    view_760: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_104, [8, 1, 384, 196]);  add_104 = None
    mul_1233: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_760, unsqueeze_847);  view_760 = unsqueeze_847 = None
    add_440: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_1232, mul_1233);  mul_1232 = mul_1233 = None
    add_441: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_440, unsqueeze_849);  add_440 = unsqueeze_849 = None
    view_761: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_441, [8, 384, 14, 14]);  add_441 = None
    view_762: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_337, [8, 1, 384]);  sum_337 = None
    view_763: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_338, [8, 1, 384])
    unsqueeze_850: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_226, -1);  alias_226 = None
    mul_1234: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_763, unsqueeze_850);  view_763 = unsqueeze_850 = None
    sub_237: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_762, mul_1234);  view_762 = mul_1234 = None
    unsqueeze_851: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_227, -1);  alias_227 = None
    mul_1235: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_851);  sub_237 = unsqueeze_851 = None
    sum_341: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_1235, [0]);  mul_1235 = None
    view_764: "f32[384]" = torch.ops.aten.view.default(sum_341, [384]);  sum_341 = None
    sum_342: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_338, [0]);  sum_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_442: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_438, view_761);  add_438 = view_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1236: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_442, clone_29);  clone_29 = None
    mul_1237: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_442, view_89);  view_89 = None
    sum_343: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1236, [0, 2, 3], True);  mul_1236 = None
    view_765: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_343, [384, 1, 1]);  sum_343 = None
    view_766: "f32[384]" = torch.ops.aten.view.default(view_765, [384]);  view_765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_344: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_1237, [0, 2, 3])
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_1237, clone_28, primals_283, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1237 = clone_28 = primals_283 = None
    getitem_275: "f32[8, 1536, 14, 14]" = convolution_backward_43[0]
    getitem_276: "f32[384, 1536, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1238: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_31, 0.7071067811865476)
    erf_57: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_1238);  mul_1238 = None
    add_443: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_57, 1);  erf_57 = None
    mul_1239: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_443, 0.5);  add_443 = None
    mul_1240: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_31, convolution_31)
    mul_1241: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_1240, -0.5);  mul_1240 = None
    exp_21: "f32[8, 1536, 14, 14]" = torch.ops.aten.exp.default(mul_1241);  mul_1241 = None
    mul_1242: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_1243: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_31, mul_1242);  convolution_31 = mul_1242 = None
    add_444: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_1239, mul_1243);  mul_1239 = mul_1243 = None
    mul_1244: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_275, add_444);  getitem_275 = add_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_345: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_1244, [0, 2, 3])
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_1244, add_102, primals_281, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1244 = add_102 = primals_281 = None
    getitem_278: "f32[8, 384, 14, 14]" = convolution_backward_44[0]
    getitem_279: "f32[1536, 384, 1, 1]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_228: "f32[8, 1]" = torch.ops.aten.alias.default(alias_58);  alias_58 = None
    alias_229: "f32[8, 1]" = torch.ops.aten.alias.default(alias_59);  alias_59 = None
    mul_1245: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_278, add_100)
    view_767: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_1245, [8, 384, 196]);  mul_1245 = None
    sum_346: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_767, [2]);  view_767 = None
    view_768: "f32[8, 384, 196]" = torch.ops.aten.view.default(getitem_278, [8, 384, 196])
    sum_347: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_768, [2]);  view_768 = None
    unsqueeze_852: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_88, 0)
    mul_1246: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_346, unsqueeze_852);  unsqueeze_852 = None
    view_769: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1246, [8, 1, 384]);  mul_1246 = None
    sum_348: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_769, [2]);  view_769 = None
    unsqueeze_853: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_88, 0)
    mul_1247: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_347, unsqueeze_853);  unsqueeze_853 = None
    view_770: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1247, [8, 1, 384]);  mul_1247 = None
    sum_349: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_770, [2]);  view_770 = None
    unsqueeze_854: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_229, -1)
    view_771: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_88, [1, 1, 384]);  primals_88 = None
    mul_1248: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_854, view_771);  unsqueeze_854 = view_771 = None
    mul_1249: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_349, alias_228)
    sub_238: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1249, sum_348);  mul_1249 = sum_348 = None
    mul_1250: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_238, alias_229);  sub_238 = None
    mul_1251: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1250, alias_229);  mul_1250 = None
    mul_1252: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1251, alias_229);  mul_1251 = None
    mul_1253: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1252, 1.328656462585034e-05);  mul_1252 = None
    neg_63: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1253)
    mul_1254: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_63, alias_228);  neg_63 = None
    mul_1255: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_349, alias_229);  sum_349 = None
    mul_1256: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1255, 1.328656462585034e-05);  mul_1255 = None
    sub_239: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1254, mul_1256);  mul_1254 = mul_1256 = None
    unsqueeze_855: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_1248, -1);  mul_1248 = None
    unsqueeze_856: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1253, -1);  mul_1253 = None
    unsqueeze_857: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, -1);  unsqueeze_856 = None
    unsqueeze_858: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_239, -1);  sub_239 = None
    unsqueeze_859: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, -1);  unsqueeze_858 = None
    view_772: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(getitem_278, [8, 1, 384, 196]);  getitem_278 = None
    mul_1257: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_772, unsqueeze_855);  view_772 = unsqueeze_855 = None
    view_773: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_100, [8, 1, 384, 196]);  add_100 = None
    mul_1258: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_773, unsqueeze_857);  view_773 = unsqueeze_857 = None
    add_445: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_1257, mul_1258);  mul_1257 = mul_1258 = None
    add_446: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_445, unsqueeze_859);  add_445 = unsqueeze_859 = None
    view_774: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_446, [8, 384, 14, 14]);  add_446 = None
    view_775: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_346, [8, 1, 384]);  sum_346 = None
    view_776: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_347, [8, 1, 384])
    unsqueeze_860: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_228, -1);  alias_228 = None
    mul_1259: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_776, unsqueeze_860);  view_776 = unsqueeze_860 = None
    sub_240: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_775, mul_1259);  view_775 = mul_1259 = None
    unsqueeze_861: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_229, -1);  alias_229 = None
    mul_1260: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_861);  sub_240 = unsqueeze_861 = None
    sum_350: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_1260, [0]);  mul_1260 = None
    view_777: "f32[384]" = torch.ops.aten.view.default(sum_350, [384]);  sum_350 = None
    sum_351: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_347, [0]);  sum_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_447: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_442, view_774);  add_442 = view_774 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1261: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_447, sub_43);  sub_43 = None
    mul_1262: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_447, view_86);  view_86 = None
    sum_352: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1261, [0, 2, 3], True);  mul_1261 = None
    view_778: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_352, [384, 1, 1]);  sum_352 = None
    view_779: "f32[384]" = torch.ops.aten.view.default(view_778, [384]);  view_778 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_64: "f32[8, 384, 14, 14]" = torch.ops.aten.neg.default(mul_1262)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_21: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d_backward.default(mul_1262, add_99, [3, 3], [1, 1], [1, 1], False, False, None);  mul_1262 = add_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_448: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(neg_64, avg_pool2d_backward_21);  neg_64 = avg_pool2d_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_230: "f32[8, 1]" = torch.ops.aten.alias.default(alias_56);  alias_56 = None
    alias_231: "f32[8, 1]" = torch.ops.aten.alias.default(alias_57);  alias_57 = None
    mul_1263: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_448, add_97)
    view_780: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_1263, [8, 384, 196]);  mul_1263 = None
    sum_353: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_780, [2]);  view_780 = None
    view_781: "f32[8, 384, 196]" = torch.ops.aten.view.default(add_448, [8, 384, 196])
    sum_354: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_781, [2]);  view_781 = None
    unsqueeze_862: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_85, 0)
    mul_1264: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_353, unsqueeze_862);  unsqueeze_862 = None
    view_782: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1264, [8, 1, 384]);  mul_1264 = None
    sum_355: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_782, [2]);  view_782 = None
    unsqueeze_863: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_85, 0)
    mul_1265: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_354, unsqueeze_863);  unsqueeze_863 = None
    view_783: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1265, [8, 1, 384]);  mul_1265 = None
    sum_356: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_783, [2]);  view_783 = None
    unsqueeze_864: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_231, -1)
    view_784: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_85, [1, 1, 384]);  primals_85 = None
    mul_1266: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_864, view_784);  unsqueeze_864 = view_784 = None
    mul_1267: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_356, alias_230)
    sub_241: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1267, sum_355);  mul_1267 = sum_355 = None
    mul_1268: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_241, alias_231);  sub_241 = None
    mul_1269: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1268, alias_231);  mul_1268 = None
    mul_1270: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1269, alias_231);  mul_1269 = None
    mul_1271: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1270, 1.328656462585034e-05);  mul_1270 = None
    neg_65: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1271)
    mul_1272: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_65, alias_230);  neg_65 = None
    mul_1273: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_356, alias_231);  sum_356 = None
    mul_1274: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1273, 1.328656462585034e-05);  mul_1273 = None
    sub_242: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1272, mul_1274);  mul_1272 = mul_1274 = None
    unsqueeze_865: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_1266, -1);  mul_1266 = None
    unsqueeze_866: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1271, -1);  mul_1271 = None
    unsqueeze_867: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, -1);  unsqueeze_866 = None
    unsqueeze_868: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_242, -1);  sub_242 = None
    unsqueeze_869: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, -1);  unsqueeze_868 = None
    view_785: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_448, [8, 1, 384, 196]);  add_448 = None
    mul_1275: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_785, unsqueeze_865);  view_785 = unsqueeze_865 = None
    view_786: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_97, [8, 1, 384, 196]);  add_97 = None
    mul_1276: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_786, unsqueeze_867);  view_786 = unsqueeze_867 = None
    add_449: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_1275, mul_1276);  mul_1275 = mul_1276 = None
    add_450: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_449, unsqueeze_869);  add_449 = unsqueeze_869 = None
    view_787: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_450, [8, 384, 14, 14]);  add_450 = None
    view_788: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_353, [8, 1, 384]);  sum_353 = None
    view_789: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_354, [8, 1, 384])
    unsqueeze_870: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_230, -1);  alias_230 = None
    mul_1277: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_789, unsqueeze_870);  view_789 = unsqueeze_870 = None
    sub_243: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_788, mul_1277);  view_788 = mul_1277 = None
    unsqueeze_871: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_231, -1);  alias_231 = None
    mul_1278: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_871);  sub_243 = unsqueeze_871 = None
    sum_357: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_1278, [0]);  mul_1278 = None
    view_790: "f32[384]" = torch.ops.aten.view.default(sum_357, [384]);  sum_357 = None
    sum_358: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_354, [0]);  sum_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_451: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_447, view_787);  add_447 = view_787 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1279: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_451, clone_27);  clone_27 = None
    mul_1280: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_451, view_83);  view_83 = None
    sum_359: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1279, [0, 2, 3], True);  mul_1279 = None
    view_791: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_359, [384, 1, 1]);  sum_359 = None
    view_792: "f32[384]" = torch.ops.aten.view.default(view_791, [384]);  view_791 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_360: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_1280, [0, 2, 3])
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_1280, clone_26, primals_279, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1280 = clone_26 = primals_279 = None
    getitem_281: "f32[8, 1536, 14, 14]" = convolution_backward_45[0]
    getitem_282: "f32[384, 1536, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1281: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_29, 0.7071067811865476)
    erf_58: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_1281);  mul_1281 = None
    add_452: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_58, 1);  erf_58 = None
    mul_1282: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_452, 0.5);  add_452 = None
    mul_1283: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_29, convolution_29)
    mul_1284: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_1283, -0.5);  mul_1283 = None
    exp_22: "f32[8, 1536, 14, 14]" = torch.ops.aten.exp.default(mul_1284);  mul_1284 = None
    mul_1285: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_1286: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_29, mul_1285);  convolution_29 = mul_1285 = None
    add_453: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_1282, mul_1286);  mul_1282 = mul_1286 = None
    mul_1287: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_281, add_453);  getitem_281 = add_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_361: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_1287, [0, 2, 3])
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_1287, add_95, primals_277, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1287 = add_95 = primals_277 = None
    getitem_284: "f32[8, 384, 14, 14]" = convolution_backward_46[0]
    getitem_285: "f32[1536, 384, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_232: "f32[8, 1]" = torch.ops.aten.alias.default(alias_54);  alias_54 = None
    alias_233: "f32[8, 1]" = torch.ops.aten.alias.default(alias_55);  alias_55 = None
    mul_1288: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_284, add_93)
    view_793: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_1288, [8, 384, 196]);  mul_1288 = None
    sum_362: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_793, [2]);  view_793 = None
    view_794: "f32[8, 384, 196]" = torch.ops.aten.view.default(getitem_284, [8, 384, 196])
    sum_363: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_794, [2]);  view_794 = None
    unsqueeze_872: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_82, 0)
    mul_1289: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_362, unsqueeze_872);  unsqueeze_872 = None
    view_795: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1289, [8, 1, 384]);  mul_1289 = None
    sum_364: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_795, [2]);  view_795 = None
    unsqueeze_873: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_82, 0)
    mul_1290: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_363, unsqueeze_873);  unsqueeze_873 = None
    view_796: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1290, [8, 1, 384]);  mul_1290 = None
    sum_365: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_796, [2]);  view_796 = None
    unsqueeze_874: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_233, -1)
    view_797: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_82, [1, 1, 384]);  primals_82 = None
    mul_1291: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_874, view_797);  unsqueeze_874 = view_797 = None
    mul_1292: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_365, alias_232)
    sub_244: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1292, sum_364);  mul_1292 = sum_364 = None
    mul_1293: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_244, alias_233);  sub_244 = None
    mul_1294: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1293, alias_233);  mul_1293 = None
    mul_1295: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1294, alias_233);  mul_1294 = None
    mul_1296: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1295, 1.328656462585034e-05);  mul_1295 = None
    neg_66: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1296)
    mul_1297: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_66, alias_232);  neg_66 = None
    mul_1298: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_365, alias_233);  sum_365 = None
    mul_1299: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1298, 1.328656462585034e-05);  mul_1298 = None
    sub_245: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1297, mul_1299);  mul_1297 = mul_1299 = None
    unsqueeze_875: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_1291, -1);  mul_1291 = None
    unsqueeze_876: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1296, -1);  mul_1296 = None
    unsqueeze_877: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_876, -1);  unsqueeze_876 = None
    unsqueeze_878: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_245, -1);  sub_245 = None
    unsqueeze_879: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, -1);  unsqueeze_878 = None
    view_798: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(getitem_284, [8, 1, 384, 196]);  getitem_284 = None
    mul_1300: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_798, unsqueeze_875);  view_798 = unsqueeze_875 = None
    view_799: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_93, [8, 1, 384, 196]);  add_93 = None
    mul_1301: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_799, unsqueeze_877);  view_799 = unsqueeze_877 = None
    add_454: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_1300, mul_1301);  mul_1300 = mul_1301 = None
    add_455: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_454, unsqueeze_879);  add_454 = unsqueeze_879 = None
    view_800: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_455, [8, 384, 14, 14]);  add_455 = None
    view_801: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_362, [8, 1, 384]);  sum_362 = None
    view_802: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_363, [8, 1, 384])
    unsqueeze_880: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_232, -1);  alias_232 = None
    mul_1302: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_802, unsqueeze_880);  view_802 = unsqueeze_880 = None
    sub_246: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_801, mul_1302);  view_801 = mul_1302 = None
    unsqueeze_881: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_233, -1);  alias_233 = None
    mul_1303: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_246, unsqueeze_881);  sub_246 = unsqueeze_881 = None
    sum_366: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_1303, [0]);  mul_1303 = None
    view_803: "f32[384]" = torch.ops.aten.view.default(sum_366, [384]);  sum_366 = None
    sum_367: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_363, [0]);  sum_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_456: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_451, view_800);  add_451 = view_800 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1304: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_456, sub_40);  sub_40 = None
    mul_1305: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_456, view_80);  view_80 = None
    sum_368: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1304, [0, 2, 3], True);  mul_1304 = None
    view_804: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_368, [384, 1, 1]);  sum_368 = None
    view_805: "f32[384]" = torch.ops.aten.view.default(view_804, [384]);  view_804 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_67: "f32[8, 384, 14, 14]" = torch.ops.aten.neg.default(mul_1305)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_22: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d_backward.default(mul_1305, add_92, [3, 3], [1, 1], [1, 1], False, False, None);  mul_1305 = add_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_457: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(neg_67, avg_pool2d_backward_22);  neg_67 = avg_pool2d_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_234: "f32[8, 1]" = torch.ops.aten.alias.default(alias_52);  alias_52 = None
    alias_235: "f32[8, 1]" = torch.ops.aten.alias.default(alias_53);  alias_53 = None
    mul_1306: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_457, add_90)
    view_806: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_1306, [8, 384, 196]);  mul_1306 = None
    sum_369: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_806, [2]);  view_806 = None
    view_807: "f32[8, 384, 196]" = torch.ops.aten.view.default(add_457, [8, 384, 196])
    sum_370: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_807, [2]);  view_807 = None
    unsqueeze_882: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_79, 0)
    mul_1307: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_369, unsqueeze_882);  unsqueeze_882 = None
    view_808: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1307, [8, 1, 384]);  mul_1307 = None
    sum_371: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_808, [2]);  view_808 = None
    unsqueeze_883: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_79, 0)
    mul_1308: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_370, unsqueeze_883);  unsqueeze_883 = None
    view_809: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1308, [8, 1, 384]);  mul_1308 = None
    sum_372: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_809, [2]);  view_809 = None
    unsqueeze_884: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_235, -1)
    view_810: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_79, [1, 1, 384]);  primals_79 = None
    mul_1309: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_884, view_810);  unsqueeze_884 = view_810 = None
    mul_1310: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_372, alias_234)
    sub_247: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1310, sum_371);  mul_1310 = sum_371 = None
    mul_1311: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_247, alias_235);  sub_247 = None
    mul_1312: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1311, alias_235);  mul_1311 = None
    mul_1313: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1312, alias_235);  mul_1312 = None
    mul_1314: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1313, 1.328656462585034e-05);  mul_1313 = None
    neg_68: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1314)
    mul_1315: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_68, alias_234);  neg_68 = None
    mul_1316: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_372, alias_235);  sum_372 = None
    mul_1317: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1316, 1.328656462585034e-05);  mul_1316 = None
    sub_248: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1315, mul_1317);  mul_1315 = mul_1317 = None
    unsqueeze_885: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_1309, -1);  mul_1309 = None
    unsqueeze_886: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1314, -1);  mul_1314 = None
    unsqueeze_887: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_886, -1);  unsqueeze_886 = None
    unsqueeze_888: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_248, -1);  sub_248 = None
    unsqueeze_889: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_888, -1);  unsqueeze_888 = None
    view_811: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_457, [8, 1, 384, 196]);  add_457 = None
    mul_1318: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_811, unsqueeze_885);  view_811 = unsqueeze_885 = None
    view_812: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_90, [8, 1, 384, 196]);  add_90 = None
    mul_1319: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_812, unsqueeze_887);  view_812 = unsqueeze_887 = None
    add_458: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_1318, mul_1319);  mul_1318 = mul_1319 = None
    add_459: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_458, unsqueeze_889);  add_458 = unsqueeze_889 = None
    view_813: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_459, [8, 384, 14, 14]);  add_459 = None
    view_814: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_369, [8, 1, 384]);  sum_369 = None
    view_815: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_370, [8, 1, 384])
    unsqueeze_890: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_234, -1);  alias_234 = None
    mul_1320: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_815, unsqueeze_890);  view_815 = unsqueeze_890 = None
    sub_249: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_814, mul_1320);  view_814 = mul_1320 = None
    unsqueeze_891: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_235, -1);  alias_235 = None
    mul_1321: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_249, unsqueeze_891);  sub_249 = unsqueeze_891 = None
    sum_373: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_1321, [0]);  mul_1321 = None
    view_816: "f32[384]" = torch.ops.aten.view.default(sum_373, [384]);  sum_373 = None
    sum_374: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_370, [0]);  sum_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_460: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_456, view_813);  add_456 = view_813 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1322: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_460, clone_25);  clone_25 = None
    mul_1323: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_460, view_77);  view_77 = None
    sum_375: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1322, [0, 2, 3], True);  mul_1322 = None
    view_817: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_375, [384, 1, 1]);  sum_375 = None
    view_818: "f32[384]" = torch.ops.aten.view.default(view_817, [384]);  view_817 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_376: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_1323, [0, 2, 3])
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_1323, clone_24, primals_275, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1323 = clone_24 = primals_275 = None
    getitem_287: "f32[8, 1536, 14, 14]" = convolution_backward_47[0]
    getitem_288: "f32[384, 1536, 1, 1]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1324: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_27, 0.7071067811865476)
    erf_59: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_1324);  mul_1324 = None
    add_461: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_59, 1);  erf_59 = None
    mul_1325: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_461, 0.5);  add_461 = None
    mul_1326: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_27, convolution_27)
    mul_1327: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_1326, -0.5);  mul_1326 = None
    exp_23: "f32[8, 1536, 14, 14]" = torch.ops.aten.exp.default(mul_1327);  mul_1327 = None
    mul_1328: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_1329: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_27, mul_1328);  convolution_27 = mul_1328 = None
    add_462: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_1325, mul_1329);  mul_1325 = mul_1329 = None
    mul_1330: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_287, add_462);  getitem_287 = add_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_377: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_1330, [0, 2, 3])
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_1330, add_88, primals_273, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1330 = add_88 = primals_273 = None
    getitem_290: "f32[8, 384, 14, 14]" = convolution_backward_48[0]
    getitem_291: "f32[1536, 384, 1, 1]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_236: "f32[8, 1]" = torch.ops.aten.alias.default(alias_50);  alias_50 = None
    alias_237: "f32[8, 1]" = torch.ops.aten.alias.default(alias_51);  alias_51 = None
    mul_1331: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_290, add_86)
    view_819: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_1331, [8, 384, 196]);  mul_1331 = None
    sum_378: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_819, [2]);  view_819 = None
    view_820: "f32[8, 384, 196]" = torch.ops.aten.view.default(getitem_290, [8, 384, 196])
    sum_379: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_820, [2]);  view_820 = None
    unsqueeze_892: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_76, 0)
    mul_1332: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_378, unsqueeze_892);  unsqueeze_892 = None
    view_821: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1332, [8, 1, 384]);  mul_1332 = None
    sum_380: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_821, [2]);  view_821 = None
    unsqueeze_893: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_76, 0)
    mul_1333: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_379, unsqueeze_893);  unsqueeze_893 = None
    view_822: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1333, [8, 1, 384]);  mul_1333 = None
    sum_381: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_822, [2]);  view_822 = None
    unsqueeze_894: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_237, -1)
    view_823: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_76, [1, 1, 384]);  primals_76 = None
    mul_1334: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_894, view_823);  unsqueeze_894 = view_823 = None
    mul_1335: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_381, alias_236)
    sub_250: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1335, sum_380);  mul_1335 = sum_380 = None
    mul_1336: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_250, alias_237);  sub_250 = None
    mul_1337: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1336, alias_237);  mul_1336 = None
    mul_1338: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1337, alias_237);  mul_1337 = None
    mul_1339: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1338, 1.328656462585034e-05);  mul_1338 = None
    neg_69: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1339)
    mul_1340: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_69, alias_236);  neg_69 = None
    mul_1341: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_381, alias_237);  sum_381 = None
    mul_1342: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1341, 1.328656462585034e-05);  mul_1341 = None
    sub_251: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1340, mul_1342);  mul_1340 = mul_1342 = None
    unsqueeze_895: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_1334, -1);  mul_1334 = None
    unsqueeze_896: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1339, -1);  mul_1339 = None
    unsqueeze_897: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, -1);  unsqueeze_896 = None
    unsqueeze_898: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_251, -1);  sub_251 = None
    unsqueeze_899: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_898, -1);  unsqueeze_898 = None
    view_824: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(getitem_290, [8, 1, 384, 196]);  getitem_290 = None
    mul_1343: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_824, unsqueeze_895);  view_824 = unsqueeze_895 = None
    view_825: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_86, [8, 1, 384, 196]);  add_86 = None
    mul_1344: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_825, unsqueeze_897);  view_825 = unsqueeze_897 = None
    add_463: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_1343, mul_1344);  mul_1343 = mul_1344 = None
    add_464: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_463, unsqueeze_899);  add_463 = unsqueeze_899 = None
    view_826: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_464, [8, 384, 14, 14]);  add_464 = None
    view_827: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_378, [8, 1, 384]);  sum_378 = None
    view_828: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_379, [8, 1, 384])
    unsqueeze_900: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_236, -1);  alias_236 = None
    mul_1345: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_828, unsqueeze_900);  view_828 = unsqueeze_900 = None
    sub_252: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_827, mul_1345);  view_827 = mul_1345 = None
    unsqueeze_901: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_237, -1);  alias_237 = None
    mul_1346: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_901);  sub_252 = unsqueeze_901 = None
    sum_382: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_1346, [0]);  mul_1346 = None
    view_829: "f32[384]" = torch.ops.aten.view.default(sum_382, [384]);  sum_382 = None
    sum_383: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_379, [0]);  sum_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_465: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_460, view_826);  add_460 = view_826 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1347: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_465, sub_37);  sub_37 = None
    mul_1348: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_465, view_74);  view_74 = None
    sum_384: "f32[1, 384, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1347, [0, 2, 3], True);  mul_1347 = None
    view_830: "f32[384, 1, 1]" = torch.ops.aten.view.default(sum_384, [384, 1, 1]);  sum_384 = None
    view_831: "f32[384]" = torch.ops.aten.view.default(view_830, [384]);  view_830 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_70: "f32[8, 384, 14, 14]" = torch.ops.aten.neg.default(mul_1348)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_23: "f32[8, 384, 14, 14]" = torch.ops.aten.avg_pool2d_backward.default(mul_1348, add_85, [3, 3], [1, 1], [1, 1], False, False, None);  mul_1348 = add_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_466: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(neg_70, avg_pool2d_backward_23);  neg_70 = avg_pool2d_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_238: "f32[8, 1]" = torch.ops.aten.alias.default(alias_48);  alias_48 = None
    alias_239: "f32[8, 1]" = torch.ops.aten.alias.default(alias_49);  alias_49 = None
    mul_1349: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_466, convolution_26)
    view_832: "f32[8, 384, 196]" = torch.ops.aten.view.default(mul_1349, [8, 384, 196]);  mul_1349 = None
    sum_385: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_832, [2]);  view_832 = None
    view_833: "f32[8, 384, 196]" = torch.ops.aten.view.default(add_466, [8, 384, 196])
    sum_386: "f32[8, 384]" = torch.ops.aten.sum.dim_IntList(view_833, [2]);  view_833 = None
    unsqueeze_902: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_73, 0)
    mul_1350: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_385, unsqueeze_902);  unsqueeze_902 = None
    view_834: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1350, [8, 1, 384]);  mul_1350 = None
    sum_387: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_834, [2]);  view_834 = None
    unsqueeze_903: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_73, 0)
    mul_1351: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sum_386, unsqueeze_903);  unsqueeze_903 = None
    view_835: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_1351, [8, 1, 384]);  mul_1351 = None
    sum_388: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_835, [2]);  view_835 = None
    unsqueeze_904: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_239, -1)
    view_836: "f32[1, 1, 384]" = torch.ops.aten.view.default(primals_73, [1, 1, 384]);  primals_73 = None
    mul_1352: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(unsqueeze_904, view_836);  unsqueeze_904 = view_836 = None
    mul_1353: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_388, alias_238)
    sub_253: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1353, sum_387);  mul_1353 = sum_387 = None
    mul_1354: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_253, alias_239);  sub_253 = None
    mul_1355: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1354, alias_239);  mul_1354 = None
    mul_1356: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1355, alias_239);  mul_1355 = None
    mul_1357: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1356, 1.328656462585034e-05);  mul_1356 = None
    neg_71: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1357)
    mul_1358: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_71, alias_238);  neg_71 = None
    mul_1359: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_388, alias_239);  sum_388 = None
    mul_1360: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1359, 1.328656462585034e-05);  mul_1359 = None
    sub_254: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1358, mul_1360);  mul_1358 = mul_1360 = None
    unsqueeze_905: "f32[8, 1, 384, 1]" = torch.ops.aten.unsqueeze.default(mul_1352, -1);  mul_1352 = None
    unsqueeze_906: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1357, -1);  mul_1357 = None
    unsqueeze_907: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, -1);  unsqueeze_906 = None
    unsqueeze_908: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_254, -1);  sub_254 = None
    unsqueeze_909: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, -1);  unsqueeze_908 = None
    view_837: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(add_466, [8, 1, 384, 196]);  add_466 = None
    mul_1361: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_837, unsqueeze_905);  view_837 = unsqueeze_905 = None
    view_838: "f32[8, 1, 384, 196]" = torch.ops.aten.view.default(convolution_26, [8, 1, 384, 196]);  convolution_26 = None
    mul_1362: "f32[8, 1, 384, 196]" = torch.ops.aten.mul.Tensor(view_838, unsqueeze_907);  view_838 = unsqueeze_907 = None
    add_467: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(mul_1361, mul_1362);  mul_1361 = mul_1362 = None
    add_468: "f32[8, 1, 384, 196]" = torch.ops.aten.add.Tensor(add_467, unsqueeze_909);  add_467 = unsqueeze_909 = None
    view_839: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(add_468, [8, 384, 14, 14]);  add_468 = None
    view_840: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_385, [8, 1, 384]);  sum_385 = None
    view_841: "f32[8, 1, 384]" = torch.ops.aten.view.default(sum_386, [8, 1, 384])
    unsqueeze_910: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_238, -1);  alias_238 = None
    mul_1363: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_841, unsqueeze_910);  view_841 = unsqueeze_910 = None
    sub_255: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(view_840, mul_1363);  view_840 = mul_1363 = None
    unsqueeze_911: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_239, -1);  alias_239 = None
    mul_1364: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_911);  sub_255 = unsqueeze_911 = None
    sum_389: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(mul_1364, [0]);  mul_1364 = None
    view_842: "f32[384]" = torch.ops.aten.view.default(sum_389, [384]);  sum_389 = None
    sum_390: "f32[384]" = torch.ops.aten.sum.dim_IntList(sum_386, [0]);  sum_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_469: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_465, view_839);  add_465 = view_839 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:103, code: x = self.conv(x)
    sum_391: "f32[384]" = torch.ops.aten.sum.dim_IntList(add_469, [0, 2, 3])
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(add_469, add_83, primals_271, [384], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_469 = add_83 = primals_271 = None
    getitem_293: "f32[8, 192, 28, 28]" = convolution_backward_49[0]
    getitem_294: "f32[384, 192, 3, 3]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1365: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_293, clone_23);  clone_23 = None
    mul_1366: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_293, view_71);  view_71 = None
    sum_392: "f32[1, 192, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1365, [0, 2, 3], True);  mul_1365 = None
    view_843: "f32[192, 1, 1]" = torch.ops.aten.view.default(sum_392, [192, 1, 1]);  sum_392 = None
    view_844: "f32[192]" = torch.ops.aten.view.default(view_843, [192]);  view_843 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_393: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1366, [0, 2, 3])
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_1366, clone_22, primals_269, [192], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1366 = clone_22 = primals_269 = None
    getitem_296: "f32[8, 768, 28, 28]" = convolution_backward_50[0]
    getitem_297: "f32[192, 768, 1, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1367: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_24, 0.7071067811865476)
    erf_60: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_1367);  mul_1367 = None
    add_470: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_60, 1);  erf_60 = None
    mul_1368: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(add_470, 0.5);  add_470 = None
    mul_1369: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_24, convolution_24)
    mul_1370: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1369, -0.5);  mul_1369 = None
    exp_24: "f32[8, 768, 28, 28]" = torch.ops.aten.exp.default(mul_1370);  mul_1370 = None
    mul_1371: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_1372: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_24, mul_1371);  convolution_24 = mul_1371 = None
    add_471: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_1368, mul_1372);  mul_1368 = mul_1372 = None
    mul_1373: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_296, add_471);  getitem_296 = add_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_394: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1373, [0, 2, 3])
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_1373, add_81, primals_267, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1373 = add_81 = primals_267 = None
    getitem_299: "f32[8, 192, 28, 28]" = convolution_backward_51[0]
    getitem_300: "f32[768, 192, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_240: "f32[8, 1]" = torch.ops.aten.alias.default(alias_46);  alias_46 = None
    alias_241: "f32[8, 1]" = torch.ops.aten.alias.default(alias_47);  alias_47 = None
    mul_1374: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_299, add_79)
    view_845: "f32[8, 192, 784]" = torch.ops.aten.view.default(mul_1374, [8, 192, 784]);  mul_1374 = None
    sum_395: "f32[8, 192]" = torch.ops.aten.sum.dim_IntList(view_845, [2]);  view_845 = None
    view_846: "f32[8, 192, 784]" = torch.ops.aten.view.default(getitem_299, [8, 192, 784])
    sum_396: "f32[8, 192]" = torch.ops.aten.sum.dim_IntList(view_846, [2]);  view_846 = None
    unsqueeze_912: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_70, 0)
    mul_1375: "f32[8, 192]" = torch.ops.aten.mul.Tensor(sum_395, unsqueeze_912);  unsqueeze_912 = None
    view_847: "f32[8, 1, 192]" = torch.ops.aten.view.default(mul_1375, [8, 1, 192]);  mul_1375 = None
    sum_397: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_847, [2]);  view_847 = None
    unsqueeze_913: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_70, 0)
    mul_1376: "f32[8, 192]" = torch.ops.aten.mul.Tensor(sum_396, unsqueeze_913);  unsqueeze_913 = None
    view_848: "f32[8, 1, 192]" = torch.ops.aten.view.default(mul_1376, [8, 1, 192]);  mul_1376 = None
    sum_398: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_848, [2]);  view_848 = None
    unsqueeze_914: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_241, -1)
    view_849: "f32[1, 1, 192]" = torch.ops.aten.view.default(primals_70, [1, 1, 192]);  primals_70 = None
    mul_1377: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(unsqueeze_914, view_849);  unsqueeze_914 = view_849 = None
    mul_1378: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_398, alias_240)
    sub_256: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1378, sum_397);  mul_1378 = sum_397 = None
    mul_1379: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_256, alias_241);  sub_256 = None
    mul_1380: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1379, alias_241);  mul_1379 = None
    mul_1381: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1380, alias_241);  mul_1380 = None
    mul_1382: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1381, 6.64328231292517e-06);  mul_1381 = None
    neg_72: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1382)
    mul_1383: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_72, alias_240);  neg_72 = None
    mul_1384: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_398, alias_241);  sum_398 = None
    mul_1385: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1384, 6.64328231292517e-06);  mul_1384 = None
    sub_257: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1383, mul_1385);  mul_1383 = mul_1385 = None
    unsqueeze_915: "f32[8, 1, 192, 1]" = torch.ops.aten.unsqueeze.default(mul_1377, -1);  mul_1377 = None
    unsqueeze_916: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1382, -1);  mul_1382 = None
    unsqueeze_917: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_916, -1);  unsqueeze_916 = None
    unsqueeze_918: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_257, -1);  sub_257 = None
    unsqueeze_919: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, -1);  unsqueeze_918 = None
    view_850: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(getitem_299, [8, 1, 192, 784]);  getitem_299 = None
    mul_1386: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(view_850, unsqueeze_915);  view_850 = unsqueeze_915 = None
    view_851: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_79, [8, 1, 192, 784]);  add_79 = None
    mul_1387: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(view_851, unsqueeze_917);  view_851 = unsqueeze_917 = None
    add_472: "f32[8, 1, 192, 784]" = torch.ops.aten.add.Tensor(mul_1386, mul_1387);  mul_1386 = mul_1387 = None
    add_473: "f32[8, 1, 192, 784]" = torch.ops.aten.add.Tensor(add_472, unsqueeze_919);  add_472 = unsqueeze_919 = None
    view_852: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(add_473, [8, 192, 28, 28]);  add_473 = None
    view_853: "f32[8, 1, 192]" = torch.ops.aten.view.default(sum_395, [8, 1, 192]);  sum_395 = None
    view_854: "f32[8, 1, 192]" = torch.ops.aten.view.default(sum_396, [8, 1, 192])
    unsqueeze_920: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_240, -1);  alias_240 = None
    mul_1388: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(view_854, unsqueeze_920);  view_854 = unsqueeze_920 = None
    sub_258: "f32[8, 1, 192]" = torch.ops.aten.sub.Tensor(view_853, mul_1388);  view_853 = mul_1388 = None
    unsqueeze_921: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_241, -1);  alias_241 = None
    mul_1389: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(sub_258, unsqueeze_921);  sub_258 = unsqueeze_921 = None
    sum_399: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(mul_1389, [0]);  mul_1389 = None
    view_855: "f32[192]" = torch.ops.aten.view.default(sum_399, [192]);  sum_399 = None
    sum_400: "f32[192]" = torch.ops.aten.sum.dim_IntList(sum_396, [0]);  sum_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_474: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(getitem_293, view_852);  getitem_293 = view_852 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1390: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(add_474, sub_34);  sub_34 = None
    mul_1391: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(add_474, view_68);  view_68 = None
    sum_401: "f32[1, 192, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1390, [0, 2, 3], True);  mul_1390 = None
    view_856: "f32[192, 1, 1]" = torch.ops.aten.view.default(sum_401, [192, 1, 1]);  sum_401 = None
    view_857: "f32[192]" = torch.ops.aten.view.default(view_856, [192]);  view_856 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_73: "f32[8, 192, 28, 28]" = torch.ops.aten.neg.default(mul_1391)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_24: "f32[8, 192, 28, 28]" = torch.ops.aten.avg_pool2d_backward.default(mul_1391, add_78, [3, 3], [1, 1], [1, 1], False, False, None);  mul_1391 = add_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_475: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(neg_73, avg_pool2d_backward_24);  neg_73 = avg_pool2d_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_242: "f32[8, 1]" = torch.ops.aten.alias.default(alias_44);  alias_44 = None
    alias_243: "f32[8, 1]" = torch.ops.aten.alias.default(alias_45);  alias_45 = None
    mul_1392: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(add_475, add_76)
    view_858: "f32[8, 192, 784]" = torch.ops.aten.view.default(mul_1392, [8, 192, 784]);  mul_1392 = None
    sum_402: "f32[8, 192]" = torch.ops.aten.sum.dim_IntList(view_858, [2]);  view_858 = None
    view_859: "f32[8, 192, 784]" = torch.ops.aten.view.default(add_475, [8, 192, 784])
    sum_403: "f32[8, 192]" = torch.ops.aten.sum.dim_IntList(view_859, [2]);  view_859 = None
    unsqueeze_922: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_67, 0)
    mul_1393: "f32[8, 192]" = torch.ops.aten.mul.Tensor(sum_402, unsqueeze_922);  unsqueeze_922 = None
    view_860: "f32[8, 1, 192]" = torch.ops.aten.view.default(mul_1393, [8, 1, 192]);  mul_1393 = None
    sum_404: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_860, [2]);  view_860 = None
    unsqueeze_923: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_67, 0)
    mul_1394: "f32[8, 192]" = torch.ops.aten.mul.Tensor(sum_403, unsqueeze_923);  unsqueeze_923 = None
    view_861: "f32[8, 1, 192]" = torch.ops.aten.view.default(mul_1394, [8, 1, 192]);  mul_1394 = None
    sum_405: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_861, [2]);  view_861 = None
    unsqueeze_924: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_243, -1)
    view_862: "f32[1, 1, 192]" = torch.ops.aten.view.default(primals_67, [1, 1, 192]);  primals_67 = None
    mul_1395: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(unsqueeze_924, view_862);  unsqueeze_924 = view_862 = None
    mul_1396: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_405, alias_242)
    sub_259: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1396, sum_404);  mul_1396 = sum_404 = None
    mul_1397: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_259, alias_243);  sub_259 = None
    mul_1398: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1397, alias_243);  mul_1397 = None
    mul_1399: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1398, alias_243);  mul_1398 = None
    mul_1400: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1399, 6.64328231292517e-06);  mul_1399 = None
    neg_74: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1400)
    mul_1401: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_74, alias_242);  neg_74 = None
    mul_1402: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_405, alias_243);  sum_405 = None
    mul_1403: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1402, 6.64328231292517e-06);  mul_1402 = None
    sub_260: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1401, mul_1403);  mul_1401 = mul_1403 = None
    unsqueeze_925: "f32[8, 1, 192, 1]" = torch.ops.aten.unsqueeze.default(mul_1395, -1);  mul_1395 = None
    unsqueeze_926: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1400, -1);  mul_1400 = None
    unsqueeze_927: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, -1);  unsqueeze_926 = None
    unsqueeze_928: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_260, -1);  sub_260 = None
    unsqueeze_929: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_928, -1);  unsqueeze_928 = None
    view_863: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_475, [8, 1, 192, 784]);  add_475 = None
    mul_1404: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(view_863, unsqueeze_925);  view_863 = unsqueeze_925 = None
    view_864: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_76, [8, 1, 192, 784]);  add_76 = None
    mul_1405: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(view_864, unsqueeze_927);  view_864 = unsqueeze_927 = None
    add_476: "f32[8, 1, 192, 784]" = torch.ops.aten.add.Tensor(mul_1404, mul_1405);  mul_1404 = mul_1405 = None
    add_477: "f32[8, 1, 192, 784]" = torch.ops.aten.add.Tensor(add_476, unsqueeze_929);  add_476 = unsqueeze_929 = None
    view_865: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(add_477, [8, 192, 28, 28]);  add_477 = None
    view_866: "f32[8, 1, 192]" = torch.ops.aten.view.default(sum_402, [8, 1, 192]);  sum_402 = None
    view_867: "f32[8, 1, 192]" = torch.ops.aten.view.default(sum_403, [8, 1, 192])
    unsqueeze_930: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_242, -1);  alias_242 = None
    mul_1406: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(view_867, unsqueeze_930);  view_867 = unsqueeze_930 = None
    sub_261: "f32[8, 1, 192]" = torch.ops.aten.sub.Tensor(view_866, mul_1406);  view_866 = mul_1406 = None
    unsqueeze_931: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_243, -1);  alias_243 = None
    mul_1407: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(sub_261, unsqueeze_931);  sub_261 = unsqueeze_931 = None
    sum_406: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(mul_1407, [0]);  mul_1407 = None
    view_868: "f32[192]" = torch.ops.aten.view.default(sum_406, [192]);  sum_406 = None
    sum_407: "f32[192]" = torch.ops.aten.sum.dim_IntList(sum_403, [0]);  sum_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_478: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_474, view_865);  add_474 = view_865 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1408: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(add_478, clone_21);  clone_21 = None
    mul_1409: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(add_478, view_65);  view_65 = None
    sum_408: "f32[1, 192, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1408, [0, 2, 3], True);  mul_1408 = None
    view_869: "f32[192, 1, 1]" = torch.ops.aten.view.default(sum_408, [192, 1, 1]);  sum_408 = None
    view_870: "f32[192]" = torch.ops.aten.view.default(view_869, [192]);  view_869 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_409: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1409, [0, 2, 3])
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_1409, clone_20, primals_265, [192], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1409 = clone_20 = primals_265 = None
    getitem_302: "f32[8, 768, 28, 28]" = convolution_backward_52[0]
    getitem_303: "f32[192, 768, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1410: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_22, 0.7071067811865476)
    erf_61: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_1410);  mul_1410 = None
    add_479: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_61, 1);  erf_61 = None
    mul_1411: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(add_479, 0.5);  add_479 = None
    mul_1412: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_22, convolution_22)
    mul_1413: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1412, -0.5);  mul_1412 = None
    exp_25: "f32[8, 768, 28, 28]" = torch.ops.aten.exp.default(mul_1413);  mul_1413 = None
    mul_1414: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_1415: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_22, mul_1414);  convolution_22 = mul_1414 = None
    add_480: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_1411, mul_1415);  mul_1411 = mul_1415 = None
    mul_1416: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_302, add_480);  getitem_302 = add_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_410: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1416, [0, 2, 3])
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_1416, add_74, primals_263, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1416 = add_74 = primals_263 = None
    getitem_305: "f32[8, 192, 28, 28]" = convolution_backward_53[0]
    getitem_306: "f32[768, 192, 1, 1]" = convolution_backward_53[1];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_244: "f32[8, 1]" = torch.ops.aten.alias.default(alias_42);  alias_42 = None
    alias_245: "f32[8, 1]" = torch.ops.aten.alias.default(alias_43);  alias_43 = None
    mul_1417: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_305, add_72)
    view_871: "f32[8, 192, 784]" = torch.ops.aten.view.default(mul_1417, [8, 192, 784]);  mul_1417 = None
    sum_411: "f32[8, 192]" = torch.ops.aten.sum.dim_IntList(view_871, [2]);  view_871 = None
    view_872: "f32[8, 192, 784]" = torch.ops.aten.view.default(getitem_305, [8, 192, 784])
    sum_412: "f32[8, 192]" = torch.ops.aten.sum.dim_IntList(view_872, [2]);  view_872 = None
    unsqueeze_932: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_64, 0)
    mul_1418: "f32[8, 192]" = torch.ops.aten.mul.Tensor(sum_411, unsqueeze_932);  unsqueeze_932 = None
    view_873: "f32[8, 1, 192]" = torch.ops.aten.view.default(mul_1418, [8, 1, 192]);  mul_1418 = None
    sum_413: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_873, [2]);  view_873 = None
    unsqueeze_933: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_64, 0)
    mul_1419: "f32[8, 192]" = torch.ops.aten.mul.Tensor(sum_412, unsqueeze_933);  unsqueeze_933 = None
    view_874: "f32[8, 1, 192]" = torch.ops.aten.view.default(mul_1419, [8, 1, 192]);  mul_1419 = None
    sum_414: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_874, [2]);  view_874 = None
    unsqueeze_934: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_245, -1)
    view_875: "f32[1, 1, 192]" = torch.ops.aten.view.default(primals_64, [1, 1, 192]);  primals_64 = None
    mul_1420: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(unsqueeze_934, view_875);  unsqueeze_934 = view_875 = None
    mul_1421: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_414, alias_244)
    sub_262: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1421, sum_413);  mul_1421 = sum_413 = None
    mul_1422: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_262, alias_245);  sub_262 = None
    mul_1423: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1422, alias_245);  mul_1422 = None
    mul_1424: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1423, alias_245);  mul_1423 = None
    mul_1425: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1424, 6.64328231292517e-06);  mul_1424 = None
    neg_75: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1425)
    mul_1426: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_75, alias_244);  neg_75 = None
    mul_1427: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_414, alias_245);  sum_414 = None
    mul_1428: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1427, 6.64328231292517e-06);  mul_1427 = None
    sub_263: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1426, mul_1428);  mul_1426 = mul_1428 = None
    unsqueeze_935: "f32[8, 1, 192, 1]" = torch.ops.aten.unsqueeze.default(mul_1420, -1);  mul_1420 = None
    unsqueeze_936: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1425, -1);  mul_1425 = None
    unsqueeze_937: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_936, -1);  unsqueeze_936 = None
    unsqueeze_938: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_263, -1);  sub_263 = None
    unsqueeze_939: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_938, -1);  unsqueeze_938 = None
    view_876: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(getitem_305, [8, 1, 192, 784]);  getitem_305 = None
    mul_1429: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(view_876, unsqueeze_935);  view_876 = unsqueeze_935 = None
    view_877: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_72, [8, 1, 192, 784]);  add_72 = None
    mul_1430: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(view_877, unsqueeze_937);  view_877 = unsqueeze_937 = None
    add_481: "f32[8, 1, 192, 784]" = torch.ops.aten.add.Tensor(mul_1429, mul_1430);  mul_1429 = mul_1430 = None
    add_482: "f32[8, 1, 192, 784]" = torch.ops.aten.add.Tensor(add_481, unsqueeze_939);  add_481 = unsqueeze_939 = None
    view_878: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(add_482, [8, 192, 28, 28]);  add_482 = None
    view_879: "f32[8, 1, 192]" = torch.ops.aten.view.default(sum_411, [8, 1, 192]);  sum_411 = None
    view_880: "f32[8, 1, 192]" = torch.ops.aten.view.default(sum_412, [8, 1, 192])
    unsqueeze_940: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_244, -1);  alias_244 = None
    mul_1431: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(view_880, unsqueeze_940);  view_880 = unsqueeze_940 = None
    sub_264: "f32[8, 1, 192]" = torch.ops.aten.sub.Tensor(view_879, mul_1431);  view_879 = mul_1431 = None
    unsqueeze_941: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_245, -1);  alias_245 = None
    mul_1432: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_941);  sub_264 = unsqueeze_941 = None
    sum_415: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(mul_1432, [0]);  mul_1432 = None
    view_881: "f32[192]" = torch.ops.aten.view.default(sum_415, [192]);  sum_415 = None
    sum_416: "f32[192]" = torch.ops.aten.sum.dim_IntList(sum_412, [0]);  sum_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_483: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_478, view_878);  add_478 = view_878 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1433: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(add_483, sub_31);  sub_31 = None
    mul_1434: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(add_483, view_62);  view_62 = None
    sum_417: "f32[1, 192, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1433, [0, 2, 3], True);  mul_1433 = None
    view_882: "f32[192, 1, 1]" = torch.ops.aten.view.default(sum_417, [192, 1, 1]);  sum_417 = None
    view_883: "f32[192]" = torch.ops.aten.view.default(view_882, [192]);  view_882 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_76: "f32[8, 192, 28, 28]" = torch.ops.aten.neg.default(mul_1434)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_25: "f32[8, 192, 28, 28]" = torch.ops.aten.avg_pool2d_backward.default(mul_1434, add_71, [3, 3], [1, 1], [1, 1], False, False, None);  mul_1434 = add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_484: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(neg_76, avg_pool2d_backward_25);  neg_76 = avg_pool2d_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_246: "f32[8, 1]" = torch.ops.aten.alias.default(alias_40);  alias_40 = None
    alias_247: "f32[8, 1]" = torch.ops.aten.alias.default(alias_41);  alias_41 = None
    mul_1435: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(add_484, add_69)
    view_884: "f32[8, 192, 784]" = torch.ops.aten.view.default(mul_1435, [8, 192, 784]);  mul_1435 = None
    sum_418: "f32[8, 192]" = torch.ops.aten.sum.dim_IntList(view_884, [2]);  view_884 = None
    view_885: "f32[8, 192, 784]" = torch.ops.aten.view.default(add_484, [8, 192, 784])
    sum_419: "f32[8, 192]" = torch.ops.aten.sum.dim_IntList(view_885, [2]);  view_885 = None
    unsqueeze_942: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_61, 0)
    mul_1436: "f32[8, 192]" = torch.ops.aten.mul.Tensor(sum_418, unsqueeze_942);  unsqueeze_942 = None
    view_886: "f32[8, 1, 192]" = torch.ops.aten.view.default(mul_1436, [8, 1, 192]);  mul_1436 = None
    sum_420: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_886, [2]);  view_886 = None
    unsqueeze_943: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_61, 0)
    mul_1437: "f32[8, 192]" = torch.ops.aten.mul.Tensor(sum_419, unsqueeze_943);  unsqueeze_943 = None
    view_887: "f32[8, 1, 192]" = torch.ops.aten.view.default(mul_1437, [8, 1, 192]);  mul_1437 = None
    sum_421: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_887, [2]);  view_887 = None
    unsqueeze_944: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_247, -1)
    view_888: "f32[1, 1, 192]" = torch.ops.aten.view.default(primals_61, [1, 1, 192]);  primals_61 = None
    mul_1438: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(unsqueeze_944, view_888);  unsqueeze_944 = view_888 = None
    mul_1439: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_421, alias_246)
    sub_265: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1439, sum_420);  mul_1439 = sum_420 = None
    mul_1440: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_265, alias_247);  sub_265 = None
    mul_1441: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1440, alias_247);  mul_1440 = None
    mul_1442: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1441, alias_247);  mul_1441 = None
    mul_1443: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1442, 6.64328231292517e-06);  mul_1442 = None
    neg_77: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1443)
    mul_1444: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_77, alias_246);  neg_77 = None
    mul_1445: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_421, alias_247);  sum_421 = None
    mul_1446: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1445, 6.64328231292517e-06);  mul_1445 = None
    sub_266: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1444, mul_1446);  mul_1444 = mul_1446 = None
    unsqueeze_945: "f32[8, 1, 192, 1]" = torch.ops.aten.unsqueeze.default(mul_1438, -1);  mul_1438 = None
    unsqueeze_946: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1443, -1);  mul_1443 = None
    unsqueeze_947: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_946, -1);  unsqueeze_946 = None
    unsqueeze_948: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_266, -1);  sub_266 = None
    unsqueeze_949: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_948, -1);  unsqueeze_948 = None
    view_889: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_484, [8, 1, 192, 784]);  add_484 = None
    mul_1447: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(view_889, unsqueeze_945);  view_889 = unsqueeze_945 = None
    view_890: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_69, [8, 1, 192, 784]);  add_69 = None
    mul_1448: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(view_890, unsqueeze_947);  view_890 = unsqueeze_947 = None
    add_485: "f32[8, 1, 192, 784]" = torch.ops.aten.add.Tensor(mul_1447, mul_1448);  mul_1447 = mul_1448 = None
    add_486: "f32[8, 1, 192, 784]" = torch.ops.aten.add.Tensor(add_485, unsqueeze_949);  add_485 = unsqueeze_949 = None
    view_891: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(add_486, [8, 192, 28, 28]);  add_486 = None
    view_892: "f32[8, 1, 192]" = torch.ops.aten.view.default(sum_418, [8, 1, 192]);  sum_418 = None
    view_893: "f32[8, 1, 192]" = torch.ops.aten.view.default(sum_419, [8, 1, 192])
    unsqueeze_950: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_246, -1);  alias_246 = None
    mul_1449: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(view_893, unsqueeze_950);  view_893 = unsqueeze_950 = None
    sub_267: "f32[8, 1, 192]" = torch.ops.aten.sub.Tensor(view_892, mul_1449);  view_892 = mul_1449 = None
    unsqueeze_951: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_247, -1);  alias_247 = None
    mul_1450: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(sub_267, unsqueeze_951);  sub_267 = unsqueeze_951 = None
    sum_422: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(mul_1450, [0]);  mul_1450 = None
    view_894: "f32[192]" = torch.ops.aten.view.default(sum_422, [192]);  sum_422 = None
    sum_423: "f32[192]" = torch.ops.aten.sum.dim_IntList(sum_419, [0]);  sum_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_487: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_483, view_891);  add_483 = view_891 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1451: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(add_487, clone_19);  clone_19 = None
    mul_1452: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(add_487, view_59);  view_59 = None
    sum_424: "f32[1, 192, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1451, [0, 2, 3], True);  mul_1451 = None
    view_895: "f32[192, 1, 1]" = torch.ops.aten.view.default(sum_424, [192, 1, 1]);  sum_424 = None
    view_896: "f32[192]" = torch.ops.aten.view.default(view_895, [192]);  view_895 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_425: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1452, [0, 2, 3])
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_1452, clone_18, primals_261, [192], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1452 = clone_18 = primals_261 = None
    getitem_308: "f32[8, 768, 28, 28]" = convolution_backward_54[0]
    getitem_309: "f32[192, 768, 1, 1]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1453: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_20, 0.7071067811865476)
    erf_62: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_1453);  mul_1453 = None
    add_488: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_62, 1);  erf_62 = None
    mul_1454: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(add_488, 0.5);  add_488 = None
    mul_1455: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_20, convolution_20)
    mul_1456: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1455, -0.5);  mul_1455 = None
    exp_26: "f32[8, 768, 28, 28]" = torch.ops.aten.exp.default(mul_1456);  mul_1456 = None
    mul_1457: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_1458: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_20, mul_1457);  convolution_20 = mul_1457 = None
    add_489: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_1454, mul_1458);  mul_1454 = mul_1458 = None
    mul_1459: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_308, add_489);  getitem_308 = add_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_426: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1459, [0, 2, 3])
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_1459, add_67, primals_259, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1459 = add_67 = primals_259 = None
    getitem_311: "f32[8, 192, 28, 28]" = convolution_backward_55[0]
    getitem_312: "f32[768, 192, 1, 1]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_248: "f32[8, 1]" = torch.ops.aten.alias.default(alias_38);  alias_38 = None
    alias_249: "f32[8, 1]" = torch.ops.aten.alias.default(alias_39);  alias_39 = None
    mul_1460: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_311, add_65)
    view_897: "f32[8, 192, 784]" = torch.ops.aten.view.default(mul_1460, [8, 192, 784]);  mul_1460 = None
    sum_427: "f32[8, 192]" = torch.ops.aten.sum.dim_IntList(view_897, [2]);  view_897 = None
    view_898: "f32[8, 192, 784]" = torch.ops.aten.view.default(getitem_311, [8, 192, 784])
    sum_428: "f32[8, 192]" = torch.ops.aten.sum.dim_IntList(view_898, [2]);  view_898 = None
    unsqueeze_952: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_58, 0)
    mul_1461: "f32[8, 192]" = torch.ops.aten.mul.Tensor(sum_427, unsqueeze_952);  unsqueeze_952 = None
    view_899: "f32[8, 1, 192]" = torch.ops.aten.view.default(mul_1461, [8, 1, 192]);  mul_1461 = None
    sum_429: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_899, [2]);  view_899 = None
    unsqueeze_953: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_58, 0)
    mul_1462: "f32[8, 192]" = torch.ops.aten.mul.Tensor(sum_428, unsqueeze_953);  unsqueeze_953 = None
    view_900: "f32[8, 1, 192]" = torch.ops.aten.view.default(mul_1462, [8, 1, 192]);  mul_1462 = None
    sum_430: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_900, [2]);  view_900 = None
    unsqueeze_954: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_249, -1)
    view_901: "f32[1, 1, 192]" = torch.ops.aten.view.default(primals_58, [1, 1, 192]);  primals_58 = None
    mul_1463: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(unsqueeze_954, view_901);  unsqueeze_954 = view_901 = None
    mul_1464: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_430, alias_248)
    sub_268: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1464, sum_429);  mul_1464 = sum_429 = None
    mul_1465: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_268, alias_249);  sub_268 = None
    mul_1466: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1465, alias_249);  mul_1465 = None
    mul_1467: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1466, alias_249);  mul_1466 = None
    mul_1468: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1467, 6.64328231292517e-06);  mul_1467 = None
    neg_78: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1468)
    mul_1469: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_78, alias_248);  neg_78 = None
    mul_1470: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_430, alias_249);  sum_430 = None
    mul_1471: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1470, 6.64328231292517e-06);  mul_1470 = None
    sub_269: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1469, mul_1471);  mul_1469 = mul_1471 = None
    unsqueeze_955: "f32[8, 1, 192, 1]" = torch.ops.aten.unsqueeze.default(mul_1463, -1);  mul_1463 = None
    unsqueeze_956: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1468, -1);  mul_1468 = None
    unsqueeze_957: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_956, -1);  unsqueeze_956 = None
    unsqueeze_958: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_269, -1);  sub_269 = None
    unsqueeze_959: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_958, -1);  unsqueeze_958 = None
    view_902: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(getitem_311, [8, 1, 192, 784]);  getitem_311 = None
    mul_1472: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(view_902, unsqueeze_955);  view_902 = unsqueeze_955 = None
    view_903: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_65, [8, 1, 192, 784]);  add_65 = None
    mul_1473: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(view_903, unsqueeze_957);  view_903 = unsqueeze_957 = None
    add_490: "f32[8, 1, 192, 784]" = torch.ops.aten.add.Tensor(mul_1472, mul_1473);  mul_1472 = mul_1473 = None
    add_491: "f32[8, 1, 192, 784]" = torch.ops.aten.add.Tensor(add_490, unsqueeze_959);  add_490 = unsqueeze_959 = None
    view_904: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(add_491, [8, 192, 28, 28]);  add_491 = None
    view_905: "f32[8, 1, 192]" = torch.ops.aten.view.default(sum_427, [8, 1, 192]);  sum_427 = None
    view_906: "f32[8, 1, 192]" = torch.ops.aten.view.default(sum_428, [8, 1, 192])
    unsqueeze_960: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_248, -1);  alias_248 = None
    mul_1474: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(view_906, unsqueeze_960);  view_906 = unsqueeze_960 = None
    sub_270: "f32[8, 1, 192]" = torch.ops.aten.sub.Tensor(view_905, mul_1474);  view_905 = mul_1474 = None
    unsqueeze_961: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_249, -1);  alias_249 = None
    mul_1475: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(sub_270, unsqueeze_961);  sub_270 = unsqueeze_961 = None
    sum_431: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(mul_1475, [0]);  mul_1475 = None
    view_907: "f32[192]" = torch.ops.aten.view.default(sum_431, [192]);  sum_431 = None
    sum_432: "f32[192]" = torch.ops.aten.sum.dim_IntList(sum_428, [0]);  sum_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_492: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_487, view_904);  add_487 = view_904 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1476: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(add_492, sub_28);  sub_28 = None
    mul_1477: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(add_492, view_56);  view_56 = None
    sum_433: "f32[1, 192, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1476, [0, 2, 3], True);  mul_1476 = None
    view_908: "f32[192, 1, 1]" = torch.ops.aten.view.default(sum_433, [192, 1, 1]);  sum_433 = None
    view_909: "f32[192]" = torch.ops.aten.view.default(view_908, [192]);  view_908 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_79: "f32[8, 192, 28, 28]" = torch.ops.aten.neg.default(mul_1477)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_26: "f32[8, 192, 28, 28]" = torch.ops.aten.avg_pool2d_backward.default(mul_1477, add_64, [3, 3], [1, 1], [1, 1], False, False, None);  mul_1477 = add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_493: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(neg_79, avg_pool2d_backward_26);  neg_79 = avg_pool2d_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_250: "f32[8, 1]" = torch.ops.aten.alias.default(alias_36);  alias_36 = None
    alias_251: "f32[8, 1]" = torch.ops.aten.alias.default(alias_37);  alias_37 = None
    mul_1478: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(add_493, add_62)
    view_910: "f32[8, 192, 784]" = torch.ops.aten.view.default(mul_1478, [8, 192, 784]);  mul_1478 = None
    sum_434: "f32[8, 192]" = torch.ops.aten.sum.dim_IntList(view_910, [2]);  view_910 = None
    view_911: "f32[8, 192, 784]" = torch.ops.aten.view.default(add_493, [8, 192, 784])
    sum_435: "f32[8, 192]" = torch.ops.aten.sum.dim_IntList(view_911, [2]);  view_911 = None
    unsqueeze_962: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_55, 0)
    mul_1479: "f32[8, 192]" = torch.ops.aten.mul.Tensor(sum_434, unsqueeze_962);  unsqueeze_962 = None
    view_912: "f32[8, 1, 192]" = torch.ops.aten.view.default(mul_1479, [8, 1, 192]);  mul_1479 = None
    sum_436: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_912, [2]);  view_912 = None
    unsqueeze_963: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_55, 0)
    mul_1480: "f32[8, 192]" = torch.ops.aten.mul.Tensor(sum_435, unsqueeze_963);  unsqueeze_963 = None
    view_913: "f32[8, 1, 192]" = torch.ops.aten.view.default(mul_1480, [8, 1, 192]);  mul_1480 = None
    sum_437: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_913, [2]);  view_913 = None
    unsqueeze_964: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_251, -1)
    view_914: "f32[1, 1, 192]" = torch.ops.aten.view.default(primals_55, [1, 1, 192]);  primals_55 = None
    mul_1481: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(unsqueeze_964, view_914);  unsqueeze_964 = view_914 = None
    mul_1482: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_437, alias_250)
    sub_271: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1482, sum_436);  mul_1482 = sum_436 = None
    mul_1483: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_271, alias_251);  sub_271 = None
    mul_1484: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1483, alias_251);  mul_1483 = None
    mul_1485: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1484, alias_251);  mul_1484 = None
    mul_1486: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1485, 6.64328231292517e-06);  mul_1485 = None
    neg_80: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1486)
    mul_1487: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_80, alias_250);  neg_80 = None
    mul_1488: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_437, alias_251);  sum_437 = None
    mul_1489: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1488, 6.64328231292517e-06);  mul_1488 = None
    sub_272: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1487, mul_1489);  mul_1487 = mul_1489 = None
    unsqueeze_965: "f32[8, 1, 192, 1]" = torch.ops.aten.unsqueeze.default(mul_1481, -1);  mul_1481 = None
    unsqueeze_966: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1486, -1);  mul_1486 = None
    unsqueeze_967: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_966, -1);  unsqueeze_966 = None
    unsqueeze_968: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_272, -1);  sub_272 = None
    unsqueeze_969: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_968, -1);  unsqueeze_968 = None
    view_915: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_493, [8, 1, 192, 784]);  add_493 = None
    mul_1490: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(view_915, unsqueeze_965);  view_915 = unsqueeze_965 = None
    view_916: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_62, [8, 1, 192, 784]);  add_62 = None
    mul_1491: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(view_916, unsqueeze_967);  view_916 = unsqueeze_967 = None
    add_494: "f32[8, 1, 192, 784]" = torch.ops.aten.add.Tensor(mul_1490, mul_1491);  mul_1490 = mul_1491 = None
    add_495: "f32[8, 1, 192, 784]" = torch.ops.aten.add.Tensor(add_494, unsqueeze_969);  add_494 = unsqueeze_969 = None
    view_917: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(add_495, [8, 192, 28, 28]);  add_495 = None
    view_918: "f32[8, 1, 192]" = torch.ops.aten.view.default(sum_434, [8, 1, 192]);  sum_434 = None
    view_919: "f32[8, 1, 192]" = torch.ops.aten.view.default(sum_435, [8, 1, 192])
    unsqueeze_970: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_250, -1);  alias_250 = None
    mul_1492: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(view_919, unsqueeze_970);  view_919 = unsqueeze_970 = None
    sub_273: "f32[8, 1, 192]" = torch.ops.aten.sub.Tensor(view_918, mul_1492);  view_918 = mul_1492 = None
    unsqueeze_971: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_251, -1);  alias_251 = None
    mul_1493: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(sub_273, unsqueeze_971);  sub_273 = unsqueeze_971 = None
    sum_438: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(mul_1493, [0]);  mul_1493 = None
    view_920: "f32[192]" = torch.ops.aten.view.default(sum_438, [192]);  sum_438 = None
    sum_439: "f32[192]" = torch.ops.aten.sum.dim_IntList(sum_435, [0]);  sum_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_496: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_492, view_917);  add_492 = view_917 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1494: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(add_496, clone_17);  clone_17 = None
    mul_1495: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(add_496, view_53);  view_53 = None
    sum_440: "f32[1, 192, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1494, [0, 2, 3], True);  mul_1494 = None
    view_921: "f32[192, 1, 1]" = torch.ops.aten.view.default(sum_440, [192, 1, 1]);  sum_440 = None
    view_922: "f32[192]" = torch.ops.aten.view.default(view_921, [192]);  view_921 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_441: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1495, [0, 2, 3])
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_1495, clone_16, primals_257, [192], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1495 = clone_16 = primals_257 = None
    getitem_314: "f32[8, 768, 28, 28]" = convolution_backward_56[0]
    getitem_315: "f32[192, 768, 1, 1]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1496: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_18, 0.7071067811865476)
    erf_63: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_1496);  mul_1496 = None
    add_497: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_63, 1);  erf_63 = None
    mul_1497: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(add_497, 0.5);  add_497 = None
    mul_1498: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_18, convolution_18)
    mul_1499: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1498, -0.5);  mul_1498 = None
    exp_27: "f32[8, 768, 28, 28]" = torch.ops.aten.exp.default(mul_1499);  mul_1499 = None
    mul_1500: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_1501: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_18, mul_1500);  convolution_18 = mul_1500 = None
    add_498: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_1497, mul_1501);  mul_1497 = mul_1501 = None
    mul_1502: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_314, add_498);  getitem_314 = add_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_442: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1502, [0, 2, 3])
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_1502, add_60, primals_255, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1502 = add_60 = primals_255 = None
    getitem_317: "f32[8, 192, 28, 28]" = convolution_backward_57[0]
    getitem_318: "f32[768, 192, 1, 1]" = convolution_backward_57[1];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_252: "f32[8, 1]" = torch.ops.aten.alias.default(alias_34);  alias_34 = None
    alias_253: "f32[8, 1]" = torch.ops.aten.alias.default(alias_35);  alias_35 = None
    mul_1503: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_317, add_58)
    view_923: "f32[8, 192, 784]" = torch.ops.aten.view.default(mul_1503, [8, 192, 784]);  mul_1503 = None
    sum_443: "f32[8, 192]" = torch.ops.aten.sum.dim_IntList(view_923, [2]);  view_923 = None
    view_924: "f32[8, 192, 784]" = torch.ops.aten.view.default(getitem_317, [8, 192, 784])
    sum_444: "f32[8, 192]" = torch.ops.aten.sum.dim_IntList(view_924, [2]);  view_924 = None
    unsqueeze_972: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_52, 0)
    mul_1504: "f32[8, 192]" = torch.ops.aten.mul.Tensor(sum_443, unsqueeze_972);  unsqueeze_972 = None
    view_925: "f32[8, 1, 192]" = torch.ops.aten.view.default(mul_1504, [8, 1, 192]);  mul_1504 = None
    sum_445: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_925, [2]);  view_925 = None
    unsqueeze_973: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_52, 0)
    mul_1505: "f32[8, 192]" = torch.ops.aten.mul.Tensor(sum_444, unsqueeze_973);  unsqueeze_973 = None
    view_926: "f32[8, 1, 192]" = torch.ops.aten.view.default(mul_1505, [8, 1, 192]);  mul_1505 = None
    sum_446: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_926, [2]);  view_926 = None
    unsqueeze_974: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_253, -1)
    view_927: "f32[1, 1, 192]" = torch.ops.aten.view.default(primals_52, [1, 1, 192]);  primals_52 = None
    mul_1506: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(unsqueeze_974, view_927);  unsqueeze_974 = view_927 = None
    mul_1507: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_446, alias_252)
    sub_274: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1507, sum_445);  mul_1507 = sum_445 = None
    mul_1508: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_274, alias_253);  sub_274 = None
    mul_1509: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1508, alias_253);  mul_1508 = None
    mul_1510: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1509, alias_253);  mul_1509 = None
    mul_1511: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1510, 6.64328231292517e-06);  mul_1510 = None
    neg_81: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1511)
    mul_1512: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_81, alias_252);  neg_81 = None
    mul_1513: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_446, alias_253);  sum_446 = None
    mul_1514: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1513, 6.64328231292517e-06);  mul_1513 = None
    sub_275: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1512, mul_1514);  mul_1512 = mul_1514 = None
    unsqueeze_975: "f32[8, 1, 192, 1]" = torch.ops.aten.unsqueeze.default(mul_1506, -1);  mul_1506 = None
    unsqueeze_976: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1511, -1);  mul_1511 = None
    unsqueeze_977: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_976, -1);  unsqueeze_976 = None
    unsqueeze_978: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_275, -1);  sub_275 = None
    unsqueeze_979: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_978, -1);  unsqueeze_978 = None
    view_928: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(getitem_317, [8, 1, 192, 784]);  getitem_317 = None
    mul_1515: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(view_928, unsqueeze_975);  view_928 = unsqueeze_975 = None
    view_929: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_58, [8, 1, 192, 784]);  add_58 = None
    mul_1516: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(view_929, unsqueeze_977);  view_929 = unsqueeze_977 = None
    add_499: "f32[8, 1, 192, 784]" = torch.ops.aten.add.Tensor(mul_1515, mul_1516);  mul_1515 = mul_1516 = None
    add_500: "f32[8, 1, 192, 784]" = torch.ops.aten.add.Tensor(add_499, unsqueeze_979);  add_499 = unsqueeze_979 = None
    view_930: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(add_500, [8, 192, 28, 28]);  add_500 = None
    view_931: "f32[8, 1, 192]" = torch.ops.aten.view.default(sum_443, [8, 1, 192]);  sum_443 = None
    view_932: "f32[8, 1, 192]" = torch.ops.aten.view.default(sum_444, [8, 1, 192])
    unsqueeze_980: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_252, -1);  alias_252 = None
    mul_1517: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(view_932, unsqueeze_980);  view_932 = unsqueeze_980 = None
    sub_276: "f32[8, 1, 192]" = torch.ops.aten.sub.Tensor(view_931, mul_1517);  view_931 = mul_1517 = None
    unsqueeze_981: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_253, -1);  alias_253 = None
    mul_1518: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(sub_276, unsqueeze_981);  sub_276 = unsqueeze_981 = None
    sum_447: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(mul_1518, [0]);  mul_1518 = None
    view_933: "f32[192]" = torch.ops.aten.view.default(sum_447, [192]);  sum_447 = None
    sum_448: "f32[192]" = torch.ops.aten.sum.dim_IntList(sum_444, [0]);  sum_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_501: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_496, view_930);  add_496 = view_930 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1519: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(add_501, sub_25);  sub_25 = None
    mul_1520: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(add_501, view_50);  view_50 = None
    sum_449: "f32[1, 192, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1519, [0, 2, 3], True);  mul_1519 = None
    view_934: "f32[192, 1, 1]" = torch.ops.aten.view.default(sum_449, [192, 1, 1]);  sum_449 = None
    view_935: "f32[192]" = torch.ops.aten.view.default(view_934, [192]);  view_934 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_82: "f32[8, 192, 28, 28]" = torch.ops.aten.neg.default(mul_1520)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_27: "f32[8, 192, 28, 28]" = torch.ops.aten.avg_pool2d_backward.default(mul_1520, add_57, [3, 3], [1, 1], [1, 1], False, False, None);  mul_1520 = add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_502: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(neg_82, avg_pool2d_backward_27);  neg_82 = avg_pool2d_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_254: "f32[8, 1]" = torch.ops.aten.alias.default(alias_32);  alias_32 = None
    alias_255: "f32[8, 1]" = torch.ops.aten.alias.default(alias_33);  alias_33 = None
    mul_1521: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(add_502, add_55)
    view_936: "f32[8, 192, 784]" = torch.ops.aten.view.default(mul_1521, [8, 192, 784]);  mul_1521 = None
    sum_450: "f32[8, 192]" = torch.ops.aten.sum.dim_IntList(view_936, [2]);  view_936 = None
    view_937: "f32[8, 192, 784]" = torch.ops.aten.view.default(add_502, [8, 192, 784])
    sum_451: "f32[8, 192]" = torch.ops.aten.sum.dim_IntList(view_937, [2]);  view_937 = None
    unsqueeze_982: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_49, 0)
    mul_1522: "f32[8, 192]" = torch.ops.aten.mul.Tensor(sum_450, unsqueeze_982);  unsqueeze_982 = None
    view_938: "f32[8, 1, 192]" = torch.ops.aten.view.default(mul_1522, [8, 1, 192]);  mul_1522 = None
    sum_452: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_938, [2]);  view_938 = None
    unsqueeze_983: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_49, 0)
    mul_1523: "f32[8, 192]" = torch.ops.aten.mul.Tensor(sum_451, unsqueeze_983);  unsqueeze_983 = None
    view_939: "f32[8, 1, 192]" = torch.ops.aten.view.default(mul_1523, [8, 1, 192]);  mul_1523 = None
    sum_453: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_939, [2]);  view_939 = None
    unsqueeze_984: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_255, -1)
    view_940: "f32[1, 1, 192]" = torch.ops.aten.view.default(primals_49, [1, 1, 192]);  primals_49 = None
    mul_1524: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(unsqueeze_984, view_940);  unsqueeze_984 = view_940 = None
    mul_1525: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_453, alias_254)
    sub_277: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1525, sum_452);  mul_1525 = sum_452 = None
    mul_1526: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_277, alias_255);  sub_277 = None
    mul_1527: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1526, alias_255);  mul_1526 = None
    mul_1528: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1527, alias_255);  mul_1527 = None
    mul_1529: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1528, 6.64328231292517e-06);  mul_1528 = None
    neg_83: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1529)
    mul_1530: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_83, alias_254);  neg_83 = None
    mul_1531: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_453, alias_255);  sum_453 = None
    mul_1532: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1531, 6.64328231292517e-06);  mul_1531 = None
    sub_278: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1530, mul_1532);  mul_1530 = mul_1532 = None
    unsqueeze_985: "f32[8, 1, 192, 1]" = torch.ops.aten.unsqueeze.default(mul_1524, -1);  mul_1524 = None
    unsqueeze_986: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1529, -1);  mul_1529 = None
    unsqueeze_987: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_986, -1);  unsqueeze_986 = None
    unsqueeze_988: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_278, -1);  sub_278 = None
    unsqueeze_989: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_988, -1);  unsqueeze_988 = None
    view_941: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_502, [8, 1, 192, 784]);  add_502 = None
    mul_1533: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(view_941, unsqueeze_985);  view_941 = unsqueeze_985 = None
    view_942: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_55, [8, 1, 192, 784]);  add_55 = None
    mul_1534: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(view_942, unsqueeze_987);  view_942 = unsqueeze_987 = None
    add_503: "f32[8, 1, 192, 784]" = torch.ops.aten.add.Tensor(mul_1533, mul_1534);  mul_1533 = mul_1534 = None
    add_504: "f32[8, 1, 192, 784]" = torch.ops.aten.add.Tensor(add_503, unsqueeze_989);  add_503 = unsqueeze_989 = None
    view_943: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(add_504, [8, 192, 28, 28]);  add_504 = None
    view_944: "f32[8, 1, 192]" = torch.ops.aten.view.default(sum_450, [8, 1, 192]);  sum_450 = None
    view_945: "f32[8, 1, 192]" = torch.ops.aten.view.default(sum_451, [8, 1, 192])
    unsqueeze_990: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_254, -1);  alias_254 = None
    mul_1535: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(view_945, unsqueeze_990);  view_945 = unsqueeze_990 = None
    sub_279: "f32[8, 1, 192]" = torch.ops.aten.sub.Tensor(view_944, mul_1535);  view_944 = mul_1535 = None
    unsqueeze_991: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_255, -1);  alias_255 = None
    mul_1536: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(sub_279, unsqueeze_991);  sub_279 = unsqueeze_991 = None
    sum_454: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(mul_1536, [0]);  mul_1536 = None
    view_946: "f32[192]" = torch.ops.aten.view.default(sum_454, [192]);  sum_454 = None
    sum_455: "f32[192]" = torch.ops.aten.sum.dim_IntList(sum_451, [0]);  sum_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_505: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_501, view_943);  add_501 = view_943 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1537: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(add_505, clone_15);  clone_15 = None
    mul_1538: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(add_505, view_47);  view_47 = None
    sum_456: "f32[1, 192, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1537, [0, 2, 3], True);  mul_1537 = None
    view_947: "f32[192, 1, 1]" = torch.ops.aten.view.default(sum_456, [192, 1, 1]);  sum_456 = None
    view_948: "f32[192]" = torch.ops.aten.view.default(view_947, [192]);  view_947 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_457: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1538, [0, 2, 3])
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_1538, clone_14, primals_253, [192], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1538 = clone_14 = primals_253 = None
    getitem_320: "f32[8, 768, 28, 28]" = convolution_backward_58[0]
    getitem_321: "f32[192, 768, 1, 1]" = convolution_backward_58[1];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1539: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_16, 0.7071067811865476)
    erf_64: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_1539);  mul_1539 = None
    add_506: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_64, 1);  erf_64 = None
    mul_1540: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(add_506, 0.5);  add_506 = None
    mul_1541: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_16, convolution_16)
    mul_1542: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1541, -0.5);  mul_1541 = None
    exp_28: "f32[8, 768, 28, 28]" = torch.ops.aten.exp.default(mul_1542);  mul_1542 = None
    mul_1543: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
    mul_1544: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_16, mul_1543);  convolution_16 = mul_1543 = None
    add_507: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_1540, mul_1544);  mul_1540 = mul_1544 = None
    mul_1545: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_320, add_507);  getitem_320 = add_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_458: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1545, [0, 2, 3])
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(mul_1545, add_53, primals_251, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1545 = add_53 = primals_251 = None
    getitem_323: "f32[8, 192, 28, 28]" = convolution_backward_59[0]
    getitem_324: "f32[768, 192, 1, 1]" = convolution_backward_59[1];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_256: "f32[8, 1]" = torch.ops.aten.alias.default(alias_30);  alias_30 = None
    alias_257: "f32[8, 1]" = torch.ops.aten.alias.default(alias_31);  alias_31 = None
    mul_1546: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_323, add_51)
    view_949: "f32[8, 192, 784]" = torch.ops.aten.view.default(mul_1546, [8, 192, 784]);  mul_1546 = None
    sum_459: "f32[8, 192]" = torch.ops.aten.sum.dim_IntList(view_949, [2]);  view_949 = None
    view_950: "f32[8, 192, 784]" = torch.ops.aten.view.default(getitem_323, [8, 192, 784])
    sum_460: "f32[8, 192]" = torch.ops.aten.sum.dim_IntList(view_950, [2]);  view_950 = None
    unsqueeze_992: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_46, 0)
    mul_1547: "f32[8, 192]" = torch.ops.aten.mul.Tensor(sum_459, unsqueeze_992);  unsqueeze_992 = None
    view_951: "f32[8, 1, 192]" = torch.ops.aten.view.default(mul_1547, [8, 1, 192]);  mul_1547 = None
    sum_461: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_951, [2]);  view_951 = None
    unsqueeze_993: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_46, 0)
    mul_1548: "f32[8, 192]" = torch.ops.aten.mul.Tensor(sum_460, unsqueeze_993);  unsqueeze_993 = None
    view_952: "f32[8, 1, 192]" = torch.ops.aten.view.default(mul_1548, [8, 1, 192]);  mul_1548 = None
    sum_462: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_952, [2]);  view_952 = None
    unsqueeze_994: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_257, -1)
    view_953: "f32[1, 1, 192]" = torch.ops.aten.view.default(primals_46, [1, 1, 192]);  primals_46 = None
    mul_1549: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(unsqueeze_994, view_953);  unsqueeze_994 = view_953 = None
    mul_1550: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_462, alias_256)
    sub_280: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1550, sum_461);  mul_1550 = sum_461 = None
    mul_1551: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_280, alias_257);  sub_280 = None
    mul_1552: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1551, alias_257);  mul_1551 = None
    mul_1553: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1552, alias_257);  mul_1552 = None
    mul_1554: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1553, 6.64328231292517e-06);  mul_1553 = None
    neg_84: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1554)
    mul_1555: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_84, alias_256);  neg_84 = None
    mul_1556: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_462, alias_257);  sum_462 = None
    mul_1557: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1556, 6.64328231292517e-06);  mul_1556 = None
    sub_281: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1555, mul_1557);  mul_1555 = mul_1557 = None
    unsqueeze_995: "f32[8, 1, 192, 1]" = torch.ops.aten.unsqueeze.default(mul_1549, -1);  mul_1549 = None
    unsqueeze_996: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1554, -1);  mul_1554 = None
    unsqueeze_997: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_996, -1);  unsqueeze_996 = None
    unsqueeze_998: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_281, -1);  sub_281 = None
    unsqueeze_999: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_998, -1);  unsqueeze_998 = None
    view_954: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(getitem_323, [8, 1, 192, 784]);  getitem_323 = None
    mul_1558: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(view_954, unsqueeze_995);  view_954 = unsqueeze_995 = None
    view_955: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_51, [8, 1, 192, 784]);  add_51 = None
    mul_1559: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(view_955, unsqueeze_997);  view_955 = unsqueeze_997 = None
    add_508: "f32[8, 1, 192, 784]" = torch.ops.aten.add.Tensor(mul_1558, mul_1559);  mul_1558 = mul_1559 = None
    add_509: "f32[8, 1, 192, 784]" = torch.ops.aten.add.Tensor(add_508, unsqueeze_999);  add_508 = unsqueeze_999 = None
    view_956: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(add_509, [8, 192, 28, 28]);  add_509 = None
    view_957: "f32[8, 1, 192]" = torch.ops.aten.view.default(sum_459, [8, 1, 192]);  sum_459 = None
    view_958: "f32[8, 1, 192]" = torch.ops.aten.view.default(sum_460, [8, 1, 192])
    unsqueeze_1000: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_256, -1);  alias_256 = None
    mul_1560: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(view_958, unsqueeze_1000);  view_958 = unsqueeze_1000 = None
    sub_282: "f32[8, 1, 192]" = torch.ops.aten.sub.Tensor(view_957, mul_1560);  view_957 = mul_1560 = None
    unsqueeze_1001: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_257, -1);  alias_257 = None
    mul_1561: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(sub_282, unsqueeze_1001);  sub_282 = unsqueeze_1001 = None
    sum_463: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(mul_1561, [0]);  mul_1561 = None
    view_959: "f32[192]" = torch.ops.aten.view.default(sum_463, [192]);  sum_463 = None
    sum_464: "f32[192]" = torch.ops.aten.sum.dim_IntList(sum_460, [0]);  sum_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_510: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_505, view_956);  add_505 = view_956 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1562: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(add_510, sub_22);  sub_22 = None
    mul_1563: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(add_510, view_44);  view_44 = None
    sum_465: "f32[1, 192, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1562, [0, 2, 3], True);  mul_1562 = None
    view_960: "f32[192, 1, 1]" = torch.ops.aten.view.default(sum_465, [192, 1, 1]);  sum_465 = None
    view_961: "f32[192]" = torch.ops.aten.view.default(view_960, [192]);  view_960 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_85: "f32[8, 192, 28, 28]" = torch.ops.aten.neg.default(mul_1563)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_28: "f32[8, 192, 28, 28]" = torch.ops.aten.avg_pool2d_backward.default(mul_1563, add_50, [3, 3], [1, 1], [1, 1], False, False, None);  mul_1563 = add_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_511: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(neg_85, avg_pool2d_backward_28);  neg_85 = avg_pool2d_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_258: "f32[8, 1]" = torch.ops.aten.alias.default(alias_28);  alias_28 = None
    alias_259: "f32[8, 1]" = torch.ops.aten.alias.default(alias_29);  alias_29 = None
    mul_1564: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(add_511, add_48)
    view_962: "f32[8, 192, 784]" = torch.ops.aten.view.default(mul_1564, [8, 192, 784]);  mul_1564 = None
    sum_466: "f32[8, 192]" = torch.ops.aten.sum.dim_IntList(view_962, [2]);  view_962 = None
    view_963: "f32[8, 192, 784]" = torch.ops.aten.view.default(add_511, [8, 192, 784])
    sum_467: "f32[8, 192]" = torch.ops.aten.sum.dim_IntList(view_963, [2]);  view_963 = None
    unsqueeze_1002: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_43, 0)
    mul_1565: "f32[8, 192]" = torch.ops.aten.mul.Tensor(sum_466, unsqueeze_1002);  unsqueeze_1002 = None
    view_964: "f32[8, 1, 192]" = torch.ops.aten.view.default(mul_1565, [8, 1, 192]);  mul_1565 = None
    sum_468: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_964, [2]);  view_964 = None
    unsqueeze_1003: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_43, 0)
    mul_1566: "f32[8, 192]" = torch.ops.aten.mul.Tensor(sum_467, unsqueeze_1003);  unsqueeze_1003 = None
    view_965: "f32[8, 1, 192]" = torch.ops.aten.view.default(mul_1566, [8, 1, 192]);  mul_1566 = None
    sum_469: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_965, [2]);  view_965 = None
    unsqueeze_1004: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_259, -1)
    view_966: "f32[1, 1, 192]" = torch.ops.aten.view.default(primals_43, [1, 1, 192]);  primals_43 = None
    mul_1567: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(unsqueeze_1004, view_966);  unsqueeze_1004 = view_966 = None
    mul_1568: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_469, alias_258)
    sub_283: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1568, sum_468);  mul_1568 = sum_468 = None
    mul_1569: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_283, alias_259);  sub_283 = None
    mul_1570: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1569, alias_259);  mul_1569 = None
    mul_1571: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1570, alias_259);  mul_1570 = None
    mul_1572: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1571, 6.64328231292517e-06);  mul_1571 = None
    neg_86: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1572)
    mul_1573: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_86, alias_258);  neg_86 = None
    mul_1574: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_469, alias_259);  sum_469 = None
    mul_1575: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1574, 6.64328231292517e-06);  mul_1574 = None
    sub_284: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1573, mul_1575);  mul_1573 = mul_1575 = None
    unsqueeze_1005: "f32[8, 1, 192, 1]" = torch.ops.aten.unsqueeze.default(mul_1567, -1);  mul_1567 = None
    unsqueeze_1006: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1572, -1);  mul_1572 = None
    unsqueeze_1007: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1006, -1);  unsqueeze_1006 = None
    unsqueeze_1008: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_284, -1);  sub_284 = None
    unsqueeze_1009: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1008, -1);  unsqueeze_1008 = None
    view_967: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_511, [8, 1, 192, 784]);  add_511 = None
    mul_1576: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(view_967, unsqueeze_1005);  view_967 = unsqueeze_1005 = None
    view_968: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_48, [8, 1, 192, 784]);  add_48 = None
    mul_1577: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(view_968, unsqueeze_1007);  view_968 = unsqueeze_1007 = None
    add_512: "f32[8, 1, 192, 784]" = torch.ops.aten.add.Tensor(mul_1576, mul_1577);  mul_1576 = mul_1577 = None
    add_513: "f32[8, 1, 192, 784]" = torch.ops.aten.add.Tensor(add_512, unsqueeze_1009);  add_512 = unsqueeze_1009 = None
    view_969: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(add_513, [8, 192, 28, 28]);  add_513 = None
    view_970: "f32[8, 1, 192]" = torch.ops.aten.view.default(sum_466, [8, 1, 192]);  sum_466 = None
    view_971: "f32[8, 1, 192]" = torch.ops.aten.view.default(sum_467, [8, 1, 192])
    unsqueeze_1010: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_258, -1);  alias_258 = None
    mul_1578: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(view_971, unsqueeze_1010);  view_971 = unsqueeze_1010 = None
    sub_285: "f32[8, 1, 192]" = torch.ops.aten.sub.Tensor(view_970, mul_1578);  view_970 = mul_1578 = None
    unsqueeze_1011: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_259, -1);  alias_259 = None
    mul_1579: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(sub_285, unsqueeze_1011);  sub_285 = unsqueeze_1011 = None
    sum_470: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(mul_1579, [0]);  mul_1579 = None
    view_972: "f32[192]" = torch.ops.aten.view.default(sum_470, [192]);  sum_470 = None
    sum_471: "f32[192]" = torch.ops.aten.sum.dim_IntList(sum_467, [0]);  sum_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_514: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_510, view_969);  add_510 = view_969 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1580: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(add_514, clone_13);  clone_13 = None
    mul_1581: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(add_514, view_41);  view_41 = None
    sum_472: "f32[1, 192, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1580, [0, 2, 3], True);  mul_1580 = None
    view_973: "f32[192, 1, 1]" = torch.ops.aten.view.default(sum_472, [192, 1, 1]);  sum_472 = None
    view_974: "f32[192]" = torch.ops.aten.view.default(view_973, [192]);  view_973 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_473: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1581, [0, 2, 3])
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_1581, clone_12, primals_249, [192], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1581 = clone_12 = primals_249 = None
    getitem_326: "f32[8, 768, 28, 28]" = convolution_backward_60[0]
    getitem_327: "f32[192, 768, 1, 1]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1582: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_14, 0.7071067811865476)
    erf_65: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_1582);  mul_1582 = None
    add_515: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_65, 1);  erf_65 = None
    mul_1583: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(add_515, 0.5);  add_515 = None
    mul_1584: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_14, convolution_14)
    mul_1585: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1584, -0.5);  mul_1584 = None
    exp_29: "f32[8, 768, 28, 28]" = torch.ops.aten.exp.default(mul_1585);  mul_1585 = None
    mul_1586: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
    mul_1587: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_14, mul_1586);  convolution_14 = mul_1586 = None
    add_516: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_1583, mul_1587);  mul_1583 = mul_1587 = None
    mul_1588: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_326, add_516);  getitem_326 = add_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_474: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1588, [0, 2, 3])
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_1588, add_46, primals_247, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1588 = add_46 = primals_247 = None
    getitem_329: "f32[8, 192, 28, 28]" = convolution_backward_61[0]
    getitem_330: "f32[768, 192, 1, 1]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_260: "f32[8, 1]" = torch.ops.aten.alias.default(alias_26);  alias_26 = None
    alias_261: "f32[8, 1]" = torch.ops.aten.alias.default(alias_27);  alias_27 = None
    mul_1589: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_329, add_44)
    view_975: "f32[8, 192, 784]" = torch.ops.aten.view.default(mul_1589, [8, 192, 784]);  mul_1589 = None
    sum_475: "f32[8, 192]" = torch.ops.aten.sum.dim_IntList(view_975, [2]);  view_975 = None
    view_976: "f32[8, 192, 784]" = torch.ops.aten.view.default(getitem_329, [8, 192, 784])
    sum_476: "f32[8, 192]" = torch.ops.aten.sum.dim_IntList(view_976, [2]);  view_976 = None
    unsqueeze_1012: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_40, 0)
    mul_1590: "f32[8, 192]" = torch.ops.aten.mul.Tensor(sum_475, unsqueeze_1012);  unsqueeze_1012 = None
    view_977: "f32[8, 1, 192]" = torch.ops.aten.view.default(mul_1590, [8, 1, 192]);  mul_1590 = None
    sum_477: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_977, [2]);  view_977 = None
    unsqueeze_1013: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_40, 0)
    mul_1591: "f32[8, 192]" = torch.ops.aten.mul.Tensor(sum_476, unsqueeze_1013);  unsqueeze_1013 = None
    view_978: "f32[8, 1, 192]" = torch.ops.aten.view.default(mul_1591, [8, 1, 192]);  mul_1591 = None
    sum_478: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_978, [2]);  view_978 = None
    unsqueeze_1014: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_261, -1)
    view_979: "f32[1, 1, 192]" = torch.ops.aten.view.default(primals_40, [1, 1, 192]);  primals_40 = None
    mul_1592: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(unsqueeze_1014, view_979);  unsqueeze_1014 = view_979 = None
    mul_1593: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_478, alias_260)
    sub_286: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1593, sum_477);  mul_1593 = sum_477 = None
    mul_1594: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_286, alias_261);  sub_286 = None
    mul_1595: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1594, alias_261);  mul_1594 = None
    mul_1596: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1595, alias_261);  mul_1595 = None
    mul_1597: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1596, 6.64328231292517e-06);  mul_1596 = None
    neg_87: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1597)
    mul_1598: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_87, alias_260);  neg_87 = None
    mul_1599: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_478, alias_261);  sum_478 = None
    mul_1600: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1599, 6.64328231292517e-06);  mul_1599 = None
    sub_287: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1598, mul_1600);  mul_1598 = mul_1600 = None
    unsqueeze_1015: "f32[8, 1, 192, 1]" = torch.ops.aten.unsqueeze.default(mul_1592, -1);  mul_1592 = None
    unsqueeze_1016: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1597, -1);  mul_1597 = None
    unsqueeze_1017: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1016, -1);  unsqueeze_1016 = None
    unsqueeze_1018: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_287, -1);  sub_287 = None
    unsqueeze_1019: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1018, -1);  unsqueeze_1018 = None
    view_980: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(getitem_329, [8, 1, 192, 784]);  getitem_329 = None
    mul_1601: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(view_980, unsqueeze_1015);  view_980 = unsqueeze_1015 = None
    view_981: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_44, [8, 1, 192, 784]);  add_44 = None
    mul_1602: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(view_981, unsqueeze_1017);  view_981 = unsqueeze_1017 = None
    add_517: "f32[8, 1, 192, 784]" = torch.ops.aten.add.Tensor(mul_1601, mul_1602);  mul_1601 = mul_1602 = None
    add_518: "f32[8, 1, 192, 784]" = torch.ops.aten.add.Tensor(add_517, unsqueeze_1019);  add_517 = unsqueeze_1019 = None
    view_982: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(add_518, [8, 192, 28, 28]);  add_518 = None
    view_983: "f32[8, 1, 192]" = torch.ops.aten.view.default(sum_475, [8, 1, 192]);  sum_475 = None
    view_984: "f32[8, 1, 192]" = torch.ops.aten.view.default(sum_476, [8, 1, 192])
    unsqueeze_1020: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_260, -1);  alias_260 = None
    mul_1603: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(view_984, unsqueeze_1020);  view_984 = unsqueeze_1020 = None
    sub_288: "f32[8, 1, 192]" = torch.ops.aten.sub.Tensor(view_983, mul_1603);  view_983 = mul_1603 = None
    unsqueeze_1021: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_261, -1);  alias_261 = None
    mul_1604: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(sub_288, unsqueeze_1021);  sub_288 = unsqueeze_1021 = None
    sum_479: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(mul_1604, [0]);  mul_1604 = None
    view_985: "f32[192]" = torch.ops.aten.view.default(sum_479, [192]);  sum_479 = None
    sum_480: "f32[192]" = torch.ops.aten.sum.dim_IntList(sum_476, [0]);  sum_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_519: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_514, view_982);  add_514 = view_982 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1605: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(add_519, sub_19);  sub_19 = None
    mul_1606: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(add_519, view_38);  view_38 = None
    sum_481: "f32[1, 192, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1605, [0, 2, 3], True);  mul_1605 = None
    view_986: "f32[192, 1, 1]" = torch.ops.aten.view.default(sum_481, [192, 1, 1]);  sum_481 = None
    view_987: "f32[192]" = torch.ops.aten.view.default(view_986, [192]);  view_986 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_88: "f32[8, 192, 28, 28]" = torch.ops.aten.neg.default(mul_1606)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_29: "f32[8, 192, 28, 28]" = torch.ops.aten.avg_pool2d_backward.default(mul_1606, add_43, [3, 3], [1, 1], [1, 1], False, False, None);  mul_1606 = add_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_520: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(neg_88, avg_pool2d_backward_29);  neg_88 = avg_pool2d_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_262: "f32[8, 1]" = torch.ops.aten.alias.default(alias_24);  alias_24 = None
    alias_263: "f32[8, 1]" = torch.ops.aten.alias.default(alias_25);  alias_25 = None
    mul_1607: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(add_520, convolution_13)
    view_988: "f32[8, 192, 784]" = torch.ops.aten.view.default(mul_1607, [8, 192, 784]);  mul_1607 = None
    sum_482: "f32[8, 192]" = torch.ops.aten.sum.dim_IntList(view_988, [2]);  view_988 = None
    view_989: "f32[8, 192, 784]" = torch.ops.aten.view.default(add_520, [8, 192, 784])
    sum_483: "f32[8, 192]" = torch.ops.aten.sum.dim_IntList(view_989, [2]);  view_989 = None
    unsqueeze_1022: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_37, 0)
    mul_1608: "f32[8, 192]" = torch.ops.aten.mul.Tensor(sum_482, unsqueeze_1022);  unsqueeze_1022 = None
    view_990: "f32[8, 1, 192]" = torch.ops.aten.view.default(mul_1608, [8, 1, 192]);  mul_1608 = None
    sum_484: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_990, [2]);  view_990 = None
    unsqueeze_1023: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_37, 0)
    mul_1609: "f32[8, 192]" = torch.ops.aten.mul.Tensor(sum_483, unsqueeze_1023);  unsqueeze_1023 = None
    view_991: "f32[8, 1, 192]" = torch.ops.aten.view.default(mul_1609, [8, 1, 192]);  mul_1609 = None
    sum_485: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_991, [2]);  view_991 = None
    unsqueeze_1024: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_263, -1)
    view_992: "f32[1, 1, 192]" = torch.ops.aten.view.default(primals_37, [1, 1, 192]);  primals_37 = None
    mul_1610: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(unsqueeze_1024, view_992);  unsqueeze_1024 = view_992 = None
    mul_1611: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_485, alias_262)
    sub_289: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1611, sum_484);  mul_1611 = sum_484 = None
    mul_1612: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_289, alias_263);  sub_289 = None
    mul_1613: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1612, alias_263);  mul_1612 = None
    mul_1614: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1613, alias_263);  mul_1613 = None
    mul_1615: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1614, 6.64328231292517e-06);  mul_1614 = None
    neg_89: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1615)
    mul_1616: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_89, alias_262);  neg_89 = None
    mul_1617: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_485, alias_263);  sum_485 = None
    mul_1618: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1617, 6.64328231292517e-06);  mul_1617 = None
    sub_290: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1616, mul_1618);  mul_1616 = mul_1618 = None
    unsqueeze_1025: "f32[8, 1, 192, 1]" = torch.ops.aten.unsqueeze.default(mul_1610, -1);  mul_1610 = None
    unsqueeze_1026: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1615, -1);  mul_1615 = None
    unsqueeze_1027: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1026, -1);  unsqueeze_1026 = None
    unsqueeze_1028: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_290, -1);  sub_290 = None
    unsqueeze_1029: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1028, -1);  unsqueeze_1028 = None
    view_993: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(add_520, [8, 1, 192, 784]);  add_520 = None
    mul_1619: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(view_993, unsqueeze_1025);  view_993 = unsqueeze_1025 = None
    view_994: "f32[8, 1, 192, 784]" = torch.ops.aten.view.default(convolution_13, [8, 1, 192, 784]);  convolution_13 = None
    mul_1620: "f32[8, 1, 192, 784]" = torch.ops.aten.mul.Tensor(view_994, unsqueeze_1027);  view_994 = unsqueeze_1027 = None
    add_521: "f32[8, 1, 192, 784]" = torch.ops.aten.add.Tensor(mul_1619, mul_1620);  mul_1619 = mul_1620 = None
    add_522: "f32[8, 1, 192, 784]" = torch.ops.aten.add.Tensor(add_521, unsqueeze_1029);  add_521 = unsqueeze_1029 = None
    view_995: "f32[8, 192, 28, 28]" = torch.ops.aten.view.default(add_522, [8, 192, 28, 28]);  add_522 = None
    view_996: "f32[8, 1, 192]" = torch.ops.aten.view.default(sum_482, [8, 1, 192]);  sum_482 = None
    view_997: "f32[8, 1, 192]" = torch.ops.aten.view.default(sum_483, [8, 1, 192])
    unsqueeze_1030: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_262, -1);  alias_262 = None
    mul_1621: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(view_997, unsqueeze_1030);  view_997 = unsqueeze_1030 = None
    sub_291: "f32[8, 1, 192]" = torch.ops.aten.sub.Tensor(view_996, mul_1621);  view_996 = mul_1621 = None
    unsqueeze_1031: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_263, -1);  alias_263 = None
    mul_1622: "f32[8, 1, 192]" = torch.ops.aten.mul.Tensor(sub_291, unsqueeze_1031);  sub_291 = unsqueeze_1031 = None
    sum_486: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(mul_1622, [0]);  mul_1622 = None
    view_998: "f32[192]" = torch.ops.aten.view.default(sum_486, [192]);  sum_486 = None
    sum_487: "f32[192]" = torch.ops.aten.sum.dim_IntList(sum_483, [0]);  sum_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_523: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_519, view_995);  add_519 = view_995 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:103, code: x = self.conv(x)
    sum_488: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_523, [0, 2, 3])
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(add_523, add_41, primals_245, [192], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_523 = add_41 = primals_245 = None
    getitem_332: "f32[8, 96, 56, 56]" = convolution_backward_62[0]
    getitem_333: "f32[192, 96, 3, 3]" = convolution_backward_62[1];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1623: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_332, clone_11);  clone_11 = None
    mul_1624: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_332, view_35);  view_35 = None
    sum_489: "f32[1, 96, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1623, [0, 2, 3], True);  mul_1623 = None
    view_999: "f32[96, 1, 1]" = torch.ops.aten.view.default(sum_489, [96, 1, 1]);  sum_489 = None
    view_1000: "f32[96]" = torch.ops.aten.view.default(view_999, [96]);  view_999 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_490: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1624, [0, 2, 3])
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_1624, clone_10, primals_243, [96], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1624 = clone_10 = primals_243 = None
    getitem_335: "f32[8, 384, 56, 56]" = convolution_backward_63[0]
    getitem_336: "f32[96, 384, 1, 1]" = convolution_backward_63[1];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1625: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_11, 0.7071067811865476)
    erf_66: "f32[8, 384, 56, 56]" = torch.ops.aten.erf.default(mul_1625);  mul_1625 = None
    add_524: "f32[8, 384, 56, 56]" = torch.ops.aten.add.Tensor(erf_66, 1);  erf_66 = None
    mul_1626: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(add_524, 0.5);  add_524 = None
    mul_1627: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_11, convolution_11)
    mul_1628: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(mul_1627, -0.5);  mul_1627 = None
    exp_30: "f32[8, 384, 56, 56]" = torch.ops.aten.exp.default(mul_1628);  mul_1628 = None
    mul_1629: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(exp_30, 0.3989422804014327);  exp_30 = None
    mul_1630: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_11, mul_1629);  convolution_11 = mul_1629 = None
    add_525: "f32[8, 384, 56, 56]" = torch.ops.aten.add.Tensor(mul_1626, mul_1630);  mul_1626 = mul_1630 = None
    mul_1631: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_335, add_525);  getitem_335 = add_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_491: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_1631, [0, 2, 3])
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(mul_1631, add_39, primals_241, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1631 = add_39 = primals_241 = None
    getitem_338: "f32[8, 96, 56, 56]" = convolution_backward_64[0]
    getitem_339: "f32[384, 96, 1, 1]" = convolution_backward_64[1];  convolution_backward_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_264: "f32[8, 1]" = torch.ops.aten.alias.default(alias_22);  alias_22 = None
    alias_265: "f32[8, 1]" = torch.ops.aten.alias.default(alias_23);  alias_23 = None
    mul_1632: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_338, add_37)
    view_1001: "f32[8, 96, 3136]" = torch.ops.aten.view.default(mul_1632, [8, 96, 3136]);  mul_1632 = None
    sum_492: "f32[8, 96]" = torch.ops.aten.sum.dim_IntList(view_1001, [2]);  view_1001 = None
    view_1002: "f32[8, 96, 3136]" = torch.ops.aten.view.default(getitem_338, [8, 96, 3136])
    sum_493: "f32[8, 96]" = torch.ops.aten.sum.dim_IntList(view_1002, [2]);  view_1002 = None
    unsqueeze_1032: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_34, 0)
    mul_1633: "f32[8, 96]" = torch.ops.aten.mul.Tensor(sum_492, unsqueeze_1032);  unsqueeze_1032 = None
    view_1003: "f32[8, 1, 96]" = torch.ops.aten.view.default(mul_1633, [8, 1, 96]);  mul_1633 = None
    sum_494: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_1003, [2]);  view_1003 = None
    unsqueeze_1033: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_34, 0)
    mul_1634: "f32[8, 96]" = torch.ops.aten.mul.Tensor(sum_493, unsqueeze_1033);  unsqueeze_1033 = None
    view_1004: "f32[8, 1, 96]" = torch.ops.aten.view.default(mul_1634, [8, 1, 96]);  mul_1634 = None
    sum_495: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_1004, [2]);  view_1004 = None
    unsqueeze_1034: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_265, -1)
    view_1005: "f32[1, 1, 96]" = torch.ops.aten.view.default(primals_34, [1, 1, 96]);  primals_34 = None
    mul_1635: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(unsqueeze_1034, view_1005);  unsqueeze_1034 = view_1005 = None
    mul_1636: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_495, alias_264)
    sub_292: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1636, sum_494);  mul_1636 = sum_494 = None
    mul_1637: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_292, alias_265);  sub_292 = None
    mul_1638: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1637, alias_265);  mul_1637 = None
    mul_1639: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1638, alias_265);  mul_1638 = None
    mul_1640: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1639, 3.321641156462585e-06);  mul_1639 = None
    neg_90: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1640)
    mul_1641: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_90, alias_264);  neg_90 = None
    mul_1642: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_495, alias_265);  sum_495 = None
    mul_1643: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1642, 3.321641156462585e-06);  mul_1642 = None
    sub_293: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1641, mul_1643);  mul_1641 = mul_1643 = None
    unsqueeze_1035: "f32[8, 1, 96, 1]" = torch.ops.aten.unsqueeze.default(mul_1635, -1);  mul_1635 = None
    unsqueeze_1036: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1640, -1);  mul_1640 = None
    unsqueeze_1037: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1036, -1);  unsqueeze_1036 = None
    unsqueeze_1038: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_293, -1);  sub_293 = None
    unsqueeze_1039: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1038, -1);  unsqueeze_1038 = None
    view_1006: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(getitem_338, [8, 1, 96, 3136]);  getitem_338 = None
    mul_1644: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(view_1006, unsqueeze_1035);  view_1006 = unsqueeze_1035 = None
    view_1007: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_37, [8, 1, 96, 3136]);  add_37 = None
    mul_1645: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(view_1007, unsqueeze_1037);  view_1007 = unsqueeze_1037 = None
    add_526: "f32[8, 1, 96, 3136]" = torch.ops.aten.add.Tensor(mul_1644, mul_1645);  mul_1644 = mul_1645 = None
    add_527: "f32[8, 1, 96, 3136]" = torch.ops.aten.add.Tensor(add_526, unsqueeze_1039);  add_526 = unsqueeze_1039 = None
    view_1008: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(add_527, [8, 96, 56, 56]);  add_527 = None
    view_1009: "f32[8, 1, 96]" = torch.ops.aten.view.default(sum_492, [8, 1, 96]);  sum_492 = None
    view_1010: "f32[8, 1, 96]" = torch.ops.aten.view.default(sum_493, [8, 1, 96])
    unsqueeze_1040: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_264, -1);  alias_264 = None
    mul_1646: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(view_1010, unsqueeze_1040);  view_1010 = unsqueeze_1040 = None
    sub_294: "f32[8, 1, 96]" = torch.ops.aten.sub.Tensor(view_1009, mul_1646);  view_1009 = mul_1646 = None
    unsqueeze_1041: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_265, -1);  alias_265 = None
    mul_1647: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(sub_294, unsqueeze_1041);  sub_294 = unsqueeze_1041 = None
    sum_496: "f32[1, 96]" = torch.ops.aten.sum.dim_IntList(mul_1647, [0]);  mul_1647 = None
    view_1011: "f32[96]" = torch.ops.aten.view.default(sum_496, [96]);  sum_496 = None
    sum_497: "f32[96]" = torch.ops.aten.sum.dim_IntList(sum_493, [0]);  sum_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_528: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(getitem_332, view_1008);  getitem_332 = view_1008 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1648: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_528, sub_16);  sub_16 = None
    mul_1649: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_528, view_32);  view_32 = None
    sum_498: "f32[1, 96, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1648, [0, 2, 3], True);  mul_1648 = None
    view_1012: "f32[96, 1, 1]" = torch.ops.aten.view.default(sum_498, [96, 1, 1]);  sum_498 = None
    view_1013: "f32[96]" = torch.ops.aten.view.default(view_1012, [96]);  view_1012 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_91: "f32[8, 96, 56, 56]" = torch.ops.aten.neg.default(mul_1649)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_30: "f32[8, 96, 56, 56]" = torch.ops.aten.avg_pool2d_backward.default(mul_1649, add_36, [3, 3], [1, 1], [1, 1], False, False, None);  mul_1649 = add_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_529: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(neg_91, avg_pool2d_backward_30);  neg_91 = avg_pool2d_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_266: "f32[8, 1]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
    alias_267: "f32[8, 1]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    mul_1650: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_529, add_34)
    view_1014: "f32[8, 96, 3136]" = torch.ops.aten.view.default(mul_1650, [8, 96, 3136]);  mul_1650 = None
    sum_499: "f32[8, 96]" = torch.ops.aten.sum.dim_IntList(view_1014, [2]);  view_1014 = None
    view_1015: "f32[8, 96, 3136]" = torch.ops.aten.view.default(add_529, [8, 96, 3136])
    sum_500: "f32[8, 96]" = torch.ops.aten.sum.dim_IntList(view_1015, [2]);  view_1015 = None
    unsqueeze_1042: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_31, 0)
    mul_1651: "f32[8, 96]" = torch.ops.aten.mul.Tensor(sum_499, unsqueeze_1042);  unsqueeze_1042 = None
    view_1016: "f32[8, 1, 96]" = torch.ops.aten.view.default(mul_1651, [8, 1, 96]);  mul_1651 = None
    sum_501: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_1016, [2]);  view_1016 = None
    unsqueeze_1043: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_31, 0)
    mul_1652: "f32[8, 96]" = torch.ops.aten.mul.Tensor(sum_500, unsqueeze_1043);  unsqueeze_1043 = None
    view_1017: "f32[8, 1, 96]" = torch.ops.aten.view.default(mul_1652, [8, 1, 96]);  mul_1652 = None
    sum_502: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_1017, [2]);  view_1017 = None
    unsqueeze_1044: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_267, -1)
    view_1018: "f32[1, 1, 96]" = torch.ops.aten.view.default(primals_31, [1, 1, 96]);  primals_31 = None
    mul_1653: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(unsqueeze_1044, view_1018);  unsqueeze_1044 = view_1018 = None
    mul_1654: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_502, alias_266)
    sub_295: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1654, sum_501);  mul_1654 = sum_501 = None
    mul_1655: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_295, alias_267);  sub_295 = None
    mul_1656: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1655, alias_267);  mul_1655 = None
    mul_1657: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1656, alias_267);  mul_1656 = None
    mul_1658: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1657, 3.321641156462585e-06);  mul_1657 = None
    neg_92: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1658)
    mul_1659: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_92, alias_266);  neg_92 = None
    mul_1660: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_502, alias_267);  sum_502 = None
    mul_1661: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1660, 3.321641156462585e-06);  mul_1660 = None
    sub_296: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1659, mul_1661);  mul_1659 = mul_1661 = None
    unsqueeze_1045: "f32[8, 1, 96, 1]" = torch.ops.aten.unsqueeze.default(mul_1653, -1);  mul_1653 = None
    unsqueeze_1046: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1658, -1);  mul_1658 = None
    unsqueeze_1047: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1046, -1);  unsqueeze_1046 = None
    unsqueeze_1048: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_296, -1);  sub_296 = None
    unsqueeze_1049: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1048, -1);  unsqueeze_1048 = None
    view_1019: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_529, [8, 1, 96, 3136]);  add_529 = None
    mul_1662: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(view_1019, unsqueeze_1045);  view_1019 = unsqueeze_1045 = None
    view_1020: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_34, [8, 1, 96, 3136]);  add_34 = None
    mul_1663: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(view_1020, unsqueeze_1047);  view_1020 = unsqueeze_1047 = None
    add_530: "f32[8, 1, 96, 3136]" = torch.ops.aten.add.Tensor(mul_1662, mul_1663);  mul_1662 = mul_1663 = None
    add_531: "f32[8, 1, 96, 3136]" = torch.ops.aten.add.Tensor(add_530, unsqueeze_1049);  add_530 = unsqueeze_1049 = None
    view_1021: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(add_531, [8, 96, 56, 56]);  add_531 = None
    view_1022: "f32[8, 1, 96]" = torch.ops.aten.view.default(sum_499, [8, 1, 96]);  sum_499 = None
    view_1023: "f32[8, 1, 96]" = torch.ops.aten.view.default(sum_500, [8, 1, 96])
    unsqueeze_1050: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_266, -1);  alias_266 = None
    mul_1664: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(view_1023, unsqueeze_1050);  view_1023 = unsqueeze_1050 = None
    sub_297: "f32[8, 1, 96]" = torch.ops.aten.sub.Tensor(view_1022, mul_1664);  view_1022 = mul_1664 = None
    unsqueeze_1051: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_267, -1);  alias_267 = None
    mul_1665: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(sub_297, unsqueeze_1051);  sub_297 = unsqueeze_1051 = None
    sum_503: "f32[1, 96]" = torch.ops.aten.sum.dim_IntList(mul_1665, [0]);  mul_1665 = None
    view_1024: "f32[96]" = torch.ops.aten.view.default(sum_503, [96]);  sum_503 = None
    sum_504: "f32[96]" = torch.ops.aten.sum.dim_IntList(sum_500, [0]);  sum_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_532: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_528, view_1021);  add_528 = view_1021 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1666: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_532, clone_9);  clone_9 = None
    mul_1667: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_532, view_29);  view_29 = None
    sum_505: "f32[1, 96, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1666, [0, 2, 3], True);  mul_1666 = None
    view_1025: "f32[96, 1, 1]" = torch.ops.aten.view.default(sum_505, [96, 1, 1]);  sum_505 = None
    view_1026: "f32[96]" = torch.ops.aten.view.default(view_1025, [96]);  view_1025 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_506: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1667, [0, 2, 3])
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(mul_1667, clone_8, primals_239, [96], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1667 = clone_8 = primals_239 = None
    getitem_341: "f32[8, 384, 56, 56]" = convolution_backward_65[0]
    getitem_342: "f32[96, 384, 1, 1]" = convolution_backward_65[1];  convolution_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1668: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_9, 0.7071067811865476)
    erf_67: "f32[8, 384, 56, 56]" = torch.ops.aten.erf.default(mul_1668);  mul_1668 = None
    add_533: "f32[8, 384, 56, 56]" = torch.ops.aten.add.Tensor(erf_67, 1);  erf_67 = None
    mul_1669: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(add_533, 0.5);  add_533 = None
    mul_1670: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_9, convolution_9)
    mul_1671: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(mul_1670, -0.5);  mul_1670 = None
    exp_31: "f32[8, 384, 56, 56]" = torch.ops.aten.exp.default(mul_1671);  mul_1671 = None
    mul_1672: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(exp_31, 0.3989422804014327);  exp_31 = None
    mul_1673: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_9, mul_1672);  convolution_9 = mul_1672 = None
    add_534: "f32[8, 384, 56, 56]" = torch.ops.aten.add.Tensor(mul_1669, mul_1673);  mul_1669 = mul_1673 = None
    mul_1674: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_341, add_534);  getitem_341 = add_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_507: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_1674, [0, 2, 3])
    convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_1674, add_32, primals_237, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1674 = add_32 = primals_237 = None
    getitem_344: "f32[8, 96, 56, 56]" = convolution_backward_66[0]
    getitem_345: "f32[384, 96, 1, 1]" = convolution_backward_66[1];  convolution_backward_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_268: "f32[8, 1]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
    alias_269: "f32[8, 1]" = torch.ops.aten.alias.default(alias_19);  alias_19 = None
    mul_1675: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_344, add_30)
    view_1027: "f32[8, 96, 3136]" = torch.ops.aten.view.default(mul_1675, [8, 96, 3136]);  mul_1675 = None
    sum_508: "f32[8, 96]" = torch.ops.aten.sum.dim_IntList(view_1027, [2]);  view_1027 = None
    view_1028: "f32[8, 96, 3136]" = torch.ops.aten.view.default(getitem_344, [8, 96, 3136])
    sum_509: "f32[8, 96]" = torch.ops.aten.sum.dim_IntList(view_1028, [2]);  view_1028 = None
    unsqueeze_1052: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_28, 0)
    mul_1676: "f32[8, 96]" = torch.ops.aten.mul.Tensor(sum_508, unsqueeze_1052);  unsqueeze_1052 = None
    view_1029: "f32[8, 1, 96]" = torch.ops.aten.view.default(mul_1676, [8, 1, 96]);  mul_1676 = None
    sum_510: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_1029, [2]);  view_1029 = None
    unsqueeze_1053: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_28, 0)
    mul_1677: "f32[8, 96]" = torch.ops.aten.mul.Tensor(sum_509, unsqueeze_1053);  unsqueeze_1053 = None
    view_1030: "f32[8, 1, 96]" = torch.ops.aten.view.default(mul_1677, [8, 1, 96]);  mul_1677 = None
    sum_511: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_1030, [2]);  view_1030 = None
    unsqueeze_1054: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_269, -1)
    view_1031: "f32[1, 1, 96]" = torch.ops.aten.view.default(primals_28, [1, 1, 96]);  primals_28 = None
    mul_1678: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(unsqueeze_1054, view_1031);  unsqueeze_1054 = view_1031 = None
    mul_1679: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_511, alias_268)
    sub_298: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1679, sum_510);  mul_1679 = sum_510 = None
    mul_1680: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_298, alias_269);  sub_298 = None
    mul_1681: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1680, alias_269);  mul_1680 = None
    mul_1682: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1681, alias_269);  mul_1681 = None
    mul_1683: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1682, 3.321641156462585e-06);  mul_1682 = None
    neg_93: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1683)
    mul_1684: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_93, alias_268);  neg_93 = None
    mul_1685: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_511, alias_269);  sum_511 = None
    mul_1686: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1685, 3.321641156462585e-06);  mul_1685 = None
    sub_299: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1684, mul_1686);  mul_1684 = mul_1686 = None
    unsqueeze_1055: "f32[8, 1, 96, 1]" = torch.ops.aten.unsqueeze.default(mul_1678, -1);  mul_1678 = None
    unsqueeze_1056: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1683, -1);  mul_1683 = None
    unsqueeze_1057: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1056, -1);  unsqueeze_1056 = None
    unsqueeze_1058: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_299, -1);  sub_299 = None
    unsqueeze_1059: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1058, -1);  unsqueeze_1058 = None
    view_1032: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(getitem_344, [8, 1, 96, 3136]);  getitem_344 = None
    mul_1687: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(view_1032, unsqueeze_1055);  view_1032 = unsqueeze_1055 = None
    view_1033: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_30, [8, 1, 96, 3136]);  add_30 = None
    mul_1688: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(view_1033, unsqueeze_1057);  view_1033 = unsqueeze_1057 = None
    add_535: "f32[8, 1, 96, 3136]" = torch.ops.aten.add.Tensor(mul_1687, mul_1688);  mul_1687 = mul_1688 = None
    add_536: "f32[8, 1, 96, 3136]" = torch.ops.aten.add.Tensor(add_535, unsqueeze_1059);  add_535 = unsqueeze_1059 = None
    view_1034: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(add_536, [8, 96, 56, 56]);  add_536 = None
    view_1035: "f32[8, 1, 96]" = torch.ops.aten.view.default(sum_508, [8, 1, 96]);  sum_508 = None
    view_1036: "f32[8, 1, 96]" = torch.ops.aten.view.default(sum_509, [8, 1, 96])
    unsqueeze_1060: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_268, -1);  alias_268 = None
    mul_1689: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(view_1036, unsqueeze_1060);  view_1036 = unsqueeze_1060 = None
    sub_300: "f32[8, 1, 96]" = torch.ops.aten.sub.Tensor(view_1035, mul_1689);  view_1035 = mul_1689 = None
    unsqueeze_1061: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_269, -1);  alias_269 = None
    mul_1690: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(sub_300, unsqueeze_1061);  sub_300 = unsqueeze_1061 = None
    sum_512: "f32[1, 96]" = torch.ops.aten.sum.dim_IntList(mul_1690, [0]);  mul_1690 = None
    view_1037: "f32[96]" = torch.ops.aten.view.default(sum_512, [96]);  sum_512 = None
    sum_513: "f32[96]" = torch.ops.aten.sum.dim_IntList(sum_509, [0]);  sum_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_537: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_532, view_1034);  add_532 = view_1034 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1691: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_537, sub_13);  sub_13 = None
    mul_1692: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_537, view_26);  view_26 = None
    sum_514: "f32[1, 96, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1691, [0, 2, 3], True);  mul_1691 = None
    view_1038: "f32[96, 1, 1]" = torch.ops.aten.view.default(sum_514, [96, 1, 1]);  sum_514 = None
    view_1039: "f32[96]" = torch.ops.aten.view.default(view_1038, [96]);  view_1038 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_94: "f32[8, 96, 56, 56]" = torch.ops.aten.neg.default(mul_1692)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_31: "f32[8, 96, 56, 56]" = torch.ops.aten.avg_pool2d_backward.default(mul_1692, add_29, [3, 3], [1, 1], [1, 1], False, False, None);  mul_1692 = add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_538: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(neg_94, avg_pool2d_backward_31);  neg_94 = avg_pool2d_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_270: "f32[8, 1]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
    alias_271: "f32[8, 1]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    mul_1693: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_538, add_27)
    view_1040: "f32[8, 96, 3136]" = torch.ops.aten.view.default(mul_1693, [8, 96, 3136]);  mul_1693 = None
    sum_515: "f32[8, 96]" = torch.ops.aten.sum.dim_IntList(view_1040, [2]);  view_1040 = None
    view_1041: "f32[8, 96, 3136]" = torch.ops.aten.view.default(add_538, [8, 96, 3136])
    sum_516: "f32[8, 96]" = torch.ops.aten.sum.dim_IntList(view_1041, [2]);  view_1041 = None
    unsqueeze_1062: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_25, 0)
    mul_1694: "f32[8, 96]" = torch.ops.aten.mul.Tensor(sum_515, unsqueeze_1062);  unsqueeze_1062 = None
    view_1042: "f32[8, 1, 96]" = torch.ops.aten.view.default(mul_1694, [8, 1, 96]);  mul_1694 = None
    sum_517: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_1042, [2]);  view_1042 = None
    unsqueeze_1063: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_25, 0)
    mul_1695: "f32[8, 96]" = torch.ops.aten.mul.Tensor(sum_516, unsqueeze_1063);  unsqueeze_1063 = None
    view_1043: "f32[8, 1, 96]" = torch.ops.aten.view.default(mul_1695, [8, 1, 96]);  mul_1695 = None
    sum_518: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_1043, [2]);  view_1043 = None
    unsqueeze_1064: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_271, -1)
    view_1044: "f32[1, 1, 96]" = torch.ops.aten.view.default(primals_25, [1, 1, 96]);  primals_25 = None
    mul_1696: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(unsqueeze_1064, view_1044);  unsqueeze_1064 = view_1044 = None
    mul_1697: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_518, alias_270)
    sub_301: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1697, sum_517);  mul_1697 = sum_517 = None
    mul_1698: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_301, alias_271);  sub_301 = None
    mul_1699: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1698, alias_271);  mul_1698 = None
    mul_1700: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1699, alias_271);  mul_1699 = None
    mul_1701: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1700, 3.321641156462585e-06);  mul_1700 = None
    neg_95: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1701)
    mul_1702: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_95, alias_270);  neg_95 = None
    mul_1703: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_518, alias_271);  sum_518 = None
    mul_1704: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1703, 3.321641156462585e-06);  mul_1703 = None
    sub_302: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1702, mul_1704);  mul_1702 = mul_1704 = None
    unsqueeze_1065: "f32[8, 1, 96, 1]" = torch.ops.aten.unsqueeze.default(mul_1696, -1);  mul_1696 = None
    unsqueeze_1066: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1701, -1);  mul_1701 = None
    unsqueeze_1067: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1066, -1);  unsqueeze_1066 = None
    unsqueeze_1068: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_302, -1);  sub_302 = None
    unsqueeze_1069: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1068, -1);  unsqueeze_1068 = None
    view_1045: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_538, [8, 1, 96, 3136]);  add_538 = None
    mul_1705: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(view_1045, unsqueeze_1065);  view_1045 = unsqueeze_1065 = None
    view_1046: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_27, [8, 1, 96, 3136]);  add_27 = None
    mul_1706: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(view_1046, unsqueeze_1067);  view_1046 = unsqueeze_1067 = None
    add_539: "f32[8, 1, 96, 3136]" = torch.ops.aten.add.Tensor(mul_1705, mul_1706);  mul_1705 = mul_1706 = None
    add_540: "f32[8, 1, 96, 3136]" = torch.ops.aten.add.Tensor(add_539, unsqueeze_1069);  add_539 = unsqueeze_1069 = None
    view_1047: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(add_540, [8, 96, 56, 56]);  add_540 = None
    view_1048: "f32[8, 1, 96]" = torch.ops.aten.view.default(sum_515, [8, 1, 96]);  sum_515 = None
    view_1049: "f32[8, 1, 96]" = torch.ops.aten.view.default(sum_516, [8, 1, 96])
    unsqueeze_1070: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_270, -1);  alias_270 = None
    mul_1707: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(view_1049, unsqueeze_1070);  view_1049 = unsqueeze_1070 = None
    sub_303: "f32[8, 1, 96]" = torch.ops.aten.sub.Tensor(view_1048, mul_1707);  view_1048 = mul_1707 = None
    unsqueeze_1071: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_271, -1);  alias_271 = None
    mul_1708: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(sub_303, unsqueeze_1071);  sub_303 = unsqueeze_1071 = None
    sum_519: "f32[1, 96]" = torch.ops.aten.sum.dim_IntList(mul_1708, [0]);  mul_1708 = None
    view_1050: "f32[96]" = torch.ops.aten.view.default(sum_519, [96]);  sum_519 = None
    sum_520: "f32[96]" = torch.ops.aten.sum.dim_IntList(sum_516, [0]);  sum_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_541: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_537, view_1047);  add_537 = view_1047 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1709: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_541, clone_7);  clone_7 = None
    mul_1710: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_541, view_23);  view_23 = None
    sum_521: "f32[1, 96, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1709, [0, 2, 3], True);  mul_1709 = None
    view_1051: "f32[96, 1, 1]" = torch.ops.aten.view.default(sum_521, [96, 1, 1]);  sum_521 = None
    view_1052: "f32[96]" = torch.ops.aten.view.default(view_1051, [96]);  view_1051 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_522: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1710, [0, 2, 3])
    convolution_backward_67 = torch.ops.aten.convolution_backward.default(mul_1710, clone_6, primals_235, [96], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1710 = clone_6 = primals_235 = None
    getitem_347: "f32[8, 384, 56, 56]" = convolution_backward_67[0]
    getitem_348: "f32[96, 384, 1, 1]" = convolution_backward_67[1];  convolution_backward_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1711: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_7, 0.7071067811865476)
    erf_68: "f32[8, 384, 56, 56]" = torch.ops.aten.erf.default(mul_1711);  mul_1711 = None
    add_542: "f32[8, 384, 56, 56]" = torch.ops.aten.add.Tensor(erf_68, 1);  erf_68 = None
    mul_1712: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(add_542, 0.5);  add_542 = None
    mul_1713: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_7, convolution_7)
    mul_1714: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(mul_1713, -0.5);  mul_1713 = None
    exp_32: "f32[8, 384, 56, 56]" = torch.ops.aten.exp.default(mul_1714);  mul_1714 = None
    mul_1715: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(exp_32, 0.3989422804014327);  exp_32 = None
    mul_1716: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_7, mul_1715);  convolution_7 = mul_1715 = None
    add_543: "f32[8, 384, 56, 56]" = torch.ops.aten.add.Tensor(mul_1712, mul_1716);  mul_1712 = mul_1716 = None
    mul_1717: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_347, add_543);  getitem_347 = add_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_523: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_1717, [0, 2, 3])
    convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_1717, add_25, primals_233, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1717 = add_25 = primals_233 = None
    getitem_350: "f32[8, 96, 56, 56]" = convolution_backward_68[0]
    getitem_351: "f32[384, 96, 1, 1]" = convolution_backward_68[1];  convolution_backward_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_272: "f32[8, 1]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    alias_273: "f32[8, 1]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    mul_1718: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_350, add_23)
    view_1053: "f32[8, 96, 3136]" = torch.ops.aten.view.default(mul_1718, [8, 96, 3136]);  mul_1718 = None
    sum_524: "f32[8, 96]" = torch.ops.aten.sum.dim_IntList(view_1053, [2]);  view_1053 = None
    view_1054: "f32[8, 96, 3136]" = torch.ops.aten.view.default(getitem_350, [8, 96, 3136])
    sum_525: "f32[8, 96]" = torch.ops.aten.sum.dim_IntList(view_1054, [2]);  view_1054 = None
    unsqueeze_1072: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_22, 0)
    mul_1719: "f32[8, 96]" = torch.ops.aten.mul.Tensor(sum_524, unsqueeze_1072);  unsqueeze_1072 = None
    view_1055: "f32[8, 1, 96]" = torch.ops.aten.view.default(mul_1719, [8, 1, 96]);  mul_1719 = None
    sum_526: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_1055, [2]);  view_1055 = None
    unsqueeze_1073: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_22, 0)
    mul_1720: "f32[8, 96]" = torch.ops.aten.mul.Tensor(sum_525, unsqueeze_1073);  unsqueeze_1073 = None
    view_1056: "f32[8, 1, 96]" = torch.ops.aten.view.default(mul_1720, [8, 1, 96]);  mul_1720 = None
    sum_527: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_1056, [2]);  view_1056 = None
    unsqueeze_1074: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_273, -1)
    view_1057: "f32[1, 1, 96]" = torch.ops.aten.view.default(primals_22, [1, 1, 96]);  primals_22 = None
    mul_1721: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(unsqueeze_1074, view_1057);  unsqueeze_1074 = view_1057 = None
    mul_1722: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_527, alias_272)
    sub_304: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1722, sum_526);  mul_1722 = sum_526 = None
    mul_1723: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_304, alias_273);  sub_304 = None
    mul_1724: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1723, alias_273);  mul_1723 = None
    mul_1725: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1724, alias_273);  mul_1724 = None
    mul_1726: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1725, 3.321641156462585e-06);  mul_1725 = None
    neg_96: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1726)
    mul_1727: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_96, alias_272);  neg_96 = None
    mul_1728: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_527, alias_273);  sum_527 = None
    mul_1729: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1728, 3.321641156462585e-06);  mul_1728 = None
    sub_305: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1727, mul_1729);  mul_1727 = mul_1729 = None
    unsqueeze_1075: "f32[8, 1, 96, 1]" = torch.ops.aten.unsqueeze.default(mul_1721, -1);  mul_1721 = None
    unsqueeze_1076: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1726, -1);  mul_1726 = None
    unsqueeze_1077: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1076, -1);  unsqueeze_1076 = None
    unsqueeze_1078: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_305, -1);  sub_305 = None
    unsqueeze_1079: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1078, -1);  unsqueeze_1078 = None
    view_1058: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(getitem_350, [8, 1, 96, 3136]);  getitem_350 = None
    mul_1730: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(view_1058, unsqueeze_1075);  view_1058 = unsqueeze_1075 = None
    view_1059: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_23, [8, 1, 96, 3136]);  add_23 = None
    mul_1731: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(view_1059, unsqueeze_1077);  view_1059 = unsqueeze_1077 = None
    add_544: "f32[8, 1, 96, 3136]" = torch.ops.aten.add.Tensor(mul_1730, mul_1731);  mul_1730 = mul_1731 = None
    add_545: "f32[8, 1, 96, 3136]" = torch.ops.aten.add.Tensor(add_544, unsqueeze_1079);  add_544 = unsqueeze_1079 = None
    view_1060: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(add_545, [8, 96, 56, 56]);  add_545 = None
    view_1061: "f32[8, 1, 96]" = torch.ops.aten.view.default(sum_524, [8, 1, 96]);  sum_524 = None
    view_1062: "f32[8, 1, 96]" = torch.ops.aten.view.default(sum_525, [8, 1, 96])
    unsqueeze_1080: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_272, -1);  alias_272 = None
    mul_1732: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(view_1062, unsqueeze_1080);  view_1062 = unsqueeze_1080 = None
    sub_306: "f32[8, 1, 96]" = torch.ops.aten.sub.Tensor(view_1061, mul_1732);  view_1061 = mul_1732 = None
    unsqueeze_1081: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_273, -1);  alias_273 = None
    mul_1733: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(sub_306, unsqueeze_1081);  sub_306 = unsqueeze_1081 = None
    sum_528: "f32[1, 96]" = torch.ops.aten.sum.dim_IntList(mul_1733, [0]);  mul_1733 = None
    view_1063: "f32[96]" = torch.ops.aten.view.default(sum_528, [96]);  sum_528 = None
    sum_529: "f32[96]" = torch.ops.aten.sum.dim_IntList(sum_525, [0]);  sum_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_546: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_541, view_1060);  add_541 = view_1060 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1734: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_546, sub_10);  sub_10 = None
    mul_1735: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_546, view_20);  view_20 = None
    sum_530: "f32[1, 96, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1734, [0, 2, 3], True);  mul_1734 = None
    view_1064: "f32[96, 1, 1]" = torch.ops.aten.view.default(sum_530, [96, 1, 1]);  sum_530 = None
    view_1065: "f32[96]" = torch.ops.aten.view.default(view_1064, [96]);  view_1064 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_97: "f32[8, 96, 56, 56]" = torch.ops.aten.neg.default(mul_1735)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_32: "f32[8, 96, 56, 56]" = torch.ops.aten.avg_pool2d_backward.default(mul_1735, add_22, [3, 3], [1, 1], [1, 1], False, False, None);  mul_1735 = add_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_547: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(neg_97, avg_pool2d_backward_32);  neg_97 = avg_pool2d_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_274: "f32[8, 1]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    alias_275: "f32[8, 1]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    mul_1736: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_547, add_20)
    view_1066: "f32[8, 96, 3136]" = torch.ops.aten.view.default(mul_1736, [8, 96, 3136]);  mul_1736 = None
    sum_531: "f32[8, 96]" = torch.ops.aten.sum.dim_IntList(view_1066, [2]);  view_1066 = None
    view_1067: "f32[8, 96, 3136]" = torch.ops.aten.view.default(add_547, [8, 96, 3136])
    sum_532: "f32[8, 96]" = torch.ops.aten.sum.dim_IntList(view_1067, [2]);  view_1067 = None
    unsqueeze_1082: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_19, 0)
    mul_1737: "f32[8, 96]" = torch.ops.aten.mul.Tensor(sum_531, unsqueeze_1082);  unsqueeze_1082 = None
    view_1068: "f32[8, 1, 96]" = torch.ops.aten.view.default(mul_1737, [8, 1, 96]);  mul_1737 = None
    sum_533: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_1068, [2]);  view_1068 = None
    unsqueeze_1083: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_19, 0)
    mul_1738: "f32[8, 96]" = torch.ops.aten.mul.Tensor(sum_532, unsqueeze_1083);  unsqueeze_1083 = None
    view_1069: "f32[8, 1, 96]" = torch.ops.aten.view.default(mul_1738, [8, 1, 96]);  mul_1738 = None
    sum_534: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_1069, [2]);  view_1069 = None
    unsqueeze_1084: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_275, -1)
    view_1070: "f32[1, 1, 96]" = torch.ops.aten.view.default(primals_19, [1, 1, 96]);  primals_19 = None
    mul_1739: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(unsqueeze_1084, view_1070);  unsqueeze_1084 = view_1070 = None
    mul_1740: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_534, alias_274)
    sub_307: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1740, sum_533);  mul_1740 = sum_533 = None
    mul_1741: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_307, alias_275);  sub_307 = None
    mul_1742: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1741, alias_275);  mul_1741 = None
    mul_1743: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1742, alias_275);  mul_1742 = None
    mul_1744: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1743, 3.321641156462585e-06);  mul_1743 = None
    neg_98: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1744)
    mul_1745: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_98, alias_274);  neg_98 = None
    mul_1746: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_534, alias_275);  sum_534 = None
    mul_1747: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1746, 3.321641156462585e-06);  mul_1746 = None
    sub_308: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1745, mul_1747);  mul_1745 = mul_1747 = None
    unsqueeze_1085: "f32[8, 1, 96, 1]" = torch.ops.aten.unsqueeze.default(mul_1739, -1);  mul_1739 = None
    unsqueeze_1086: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1744, -1);  mul_1744 = None
    unsqueeze_1087: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1086, -1);  unsqueeze_1086 = None
    unsqueeze_1088: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_308, -1);  sub_308 = None
    unsqueeze_1089: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1088, -1);  unsqueeze_1088 = None
    view_1071: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_547, [8, 1, 96, 3136]);  add_547 = None
    mul_1748: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(view_1071, unsqueeze_1085);  view_1071 = unsqueeze_1085 = None
    view_1072: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_20, [8, 1, 96, 3136]);  add_20 = None
    mul_1749: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(view_1072, unsqueeze_1087);  view_1072 = unsqueeze_1087 = None
    add_548: "f32[8, 1, 96, 3136]" = torch.ops.aten.add.Tensor(mul_1748, mul_1749);  mul_1748 = mul_1749 = None
    add_549: "f32[8, 1, 96, 3136]" = torch.ops.aten.add.Tensor(add_548, unsqueeze_1089);  add_548 = unsqueeze_1089 = None
    view_1073: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(add_549, [8, 96, 56, 56]);  add_549 = None
    view_1074: "f32[8, 1, 96]" = torch.ops.aten.view.default(sum_531, [8, 1, 96]);  sum_531 = None
    view_1075: "f32[8, 1, 96]" = torch.ops.aten.view.default(sum_532, [8, 1, 96])
    unsqueeze_1090: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_274, -1);  alias_274 = None
    mul_1750: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(view_1075, unsqueeze_1090);  view_1075 = unsqueeze_1090 = None
    sub_309: "f32[8, 1, 96]" = torch.ops.aten.sub.Tensor(view_1074, mul_1750);  view_1074 = mul_1750 = None
    unsqueeze_1091: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_275, -1);  alias_275 = None
    mul_1751: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(sub_309, unsqueeze_1091);  sub_309 = unsqueeze_1091 = None
    sum_535: "f32[1, 96]" = torch.ops.aten.sum.dim_IntList(mul_1751, [0]);  mul_1751 = None
    view_1076: "f32[96]" = torch.ops.aten.view.default(sum_535, [96]);  sum_535 = None
    sum_536: "f32[96]" = torch.ops.aten.sum.dim_IntList(sum_532, [0]);  sum_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_550: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_546, view_1073);  add_546 = view_1073 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1752: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_550, clone_5);  clone_5 = None
    mul_1753: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_550, view_17);  view_17 = None
    sum_537: "f32[1, 96, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1752, [0, 2, 3], True);  mul_1752 = None
    view_1077: "f32[96, 1, 1]" = torch.ops.aten.view.default(sum_537, [96, 1, 1]);  sum_537 = None
    view_1078: "f32[96]" = torch.ops.aten.view.default(view_1077, [96]);  view_1077 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_538: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1753, [0, 2, 3])
    convolution_backward_69 = torch.ops.aten.convolution_backward.default(mul_1753, clone_4, primals_231, [96], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1753 = clone_4 = primals_231 = None
    getitem_353: "f32[8, 384, 56, 56]" = convolution_backward_69[0]
    getitem_354: "f32[96, 384, 1, 1]" = convolution_backward_69[1];  convolution_backward_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1754: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_5, 0.7071067811865476)
    erf_69: "f32[8, 384, 56, 56]" = torch.ops.aten.erf.default(mul_1754);  mul_1754 = None
    add_551: "f32[8, 384, 56, 56]" = torch.ops.aten.add.Tensor(erf_69, 1);  erf_69 = None
    mul_1755: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(add_551, 0.5);  add_551 = None
    mul_1756: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_5, convolution_5)
    mul_1757: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(mul_1756, -0.5);  mul_1756 = None
    exp_33: "f32[8, 384, 56, 56]" = torch.ops.aten.exp.default(mul_1757);  mul_1757 = None
    mul_1758: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(exp_33, 0.3989422804014327);  exp_33 = None
    mul_1759: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_5, mul_1758);  convolution_5 = mul_1758 = None
    add_552: "f32[8, 384, 56, 56]" = torch.ops.aten.add.Tensor(mul_1755, mul_1759);  mul_1755 = mul_1759 = None
    mul_1760: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_353, add_552);  getitem_353 = add_552 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_539: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_1760, [0, 2, 3])
    convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_1760, add_18, primals_229, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1760 = add_18 = primals_229 = None
    getitem_356: "f32[8, 96, 56, 56]" = convolution_backward_70[0]
    getitem_357: "f32[384, 96, 1, 1]" = convolution_backward_70[1];  convolution_backward_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_276: "f32[8, 1]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    alias_277: "f32[8, 1]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    mul_1761: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_356, add_16)
    view_1079: "f32[8, 96, 3136]" = torch.ops.aten.view.default(mul_1761, [8, 96, 3136]);  mul_1761 = None
    sum_540: "f32[8, 96]" = torch.ops.aten.sum.dim_IntList(view_1079, [2]);  view_1079 = None
    view_1080: "f32[8, 96, 3136]" = torch.ops.aten.view.default(getitem_356, [8, 96, 3136])
    sum_541: "f32[8, 96]" = torch.ops.aten.sum.dim_IntList(view_1080, [2]);  view_1080 = None
    unsqueeze_1092: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_16, 0)
    mul_1762: "f32[8, 96]" = torch.ops.aten.mul.Tensor(sum_540, unsqueeze_1092);  unsqueeze_1092 = None
    view_1081: "f32[8, 1, 96]" = torch.ops.aten.view.default(mul_1762, [8, 1, 96]);  mul_1762 = None
    sum_542: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_1081, [2]);  view_1081 = None
    unsqueeze_1093: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_16, 0)
    mul_1763: "f32[8, 96]" = torch.ops.aten.mul.Tensor(sum_541, unsqueeze_1093);  unsqueeze_1093 = None
    view_1082: "f32[8, 1, 96]" = torch.ops.aten.view.default(mul_1763, [8, 1, 96]);  mul_1763 = None
    sum_543: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_1082, [2]);  view_1082 = None
    unsqueeze_1094: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_277, -1)
    view_1083: "f32[1, 1, 96]" = torch.ops.aten.view.default(primals_16, [1, 1, 96]);  primals_16 = None
    mul_1764: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(unsqueeze_1094, view_1083);  unsqueeze_1094 = view_1083 = None
    mul_1765: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_543, alias_276)
    sub_310: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1765, sum_542);  mul_1765 = sum_542 = None
    mul_1766: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_310, alias_277);  sub_310 = None
    mul_1767: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1766, alias_277);  mul_1766 = None
    mul_1768: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1767, alias_277);  mul_1767 = None
    mul_1769: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1768, 3.321641156462585e-06);  mul_1768 = None
    neg_99: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1769)
    mul_1770: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_99, alias_276);  neg_99 = None
    mul_1771: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_543, alias_277);  sum_543 = None
    mul_1772: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1771, 3.321641156462585e-06);  mul_1771 = None
    sub_311: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1770, mul_1772);  mul_1770 = mul_1772 = None
    unsqueeze_1095: "f32[8, 1, 96, 1]" = torch.ops.aten.unsqueeze.default(mul_1764, -1);  mul_1764 = None
    unsqueeze_1096: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1769, -1);  mul_1769 = None
    unsqueeze_1097: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1096, -1);  unsqueeze_1096 = None
    unsqueeze_1098: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_311, -1);  sub_311 = None
    unsqueeze_1099: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1098, -1);  unsqueeze_1098 = None
    view_1084: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(getitem_356, [8, 1, 96, 3136]);  getitem_356 = None
    mul_1773: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(view_1084, unsqueeze_1095);  view_1084 = unsqueeze_1095 = None
    view_1085: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_16, [8, 1, 96, 3136]);  add_16 = None
    mul_1774: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(view_1085, unsqueeze_1097);  view_1085 = unsqueeze_1097 = None
    add_553: "f32[8, 1, 96, 3136]" = torch.ops.aten.add.Tensor(mul_1773, mul_1774);  mul_1773 = mul_1774 = None
    add_554: "f32[8, 1, 96, 3136]" = torch.ops.aten.add.Tensor(add_553, unsqueeze_1099);  add_553 = unsqueeze_1099 = None
    view_1086: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(add_554, [8, 96, 56, 56]);  add_554 = None
    view_1087: "f32[8, 1, 96]" = torch.ops.aten.view.default(sum_540, [8, 1, 96]);  sum_540 = None
    view_1088: "f32[8, 1, 96]" = torch.ops.aten.view.default(sum_541, [8, 1, 96])
    unsqueeze_1100: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_276, -1);  alias_276 = None
    mul_1775: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(view_1088, unsqueeze_1100);  view_1088 = unsqueeze_1100 = None
    sub_312: "f32[8, 1, 96]" = torch.ops.aten.sub.Tensor(view_1087, mul_1775);  view_1087 = mul_1775 = None
    unsqueeze_1101: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_277, -1);  alias_277 = None
    mul_1776: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(sub_312, unsqueeze_1101);  sub_312 = unsqueeze_1101 = None
    sum_544: "f32[1, 96]" = torch.ops.aten.sum.dim_IntList(mul_1776, [0]);  mul_1776 = None
    view_1089: "f32[96]" = torch.ops.aten.view.default(sum_544, [96]);  sum_544 = None
    sum_545: "f32[96]" = torch.ops.aten.sum.dim_IntList(sum_541, [0]);  sum_541 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_555: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_550, view_1086);  add_550 = view_1086 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1777: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_555, sub_7);  sub_7 = None
    mul_1778: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_555, view_14);  view_14 = None
    sum_546: "f32[1, 96, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1777, [0, 2, 3], True);  mul_1777 = None
    view_1090: "f32[96, 1, 1]" = torch.ops.aten.view.default(sum_546, [96, 1, 1]);  sum_546 = None
    view_1091: "f32[96]" = torch.ops.aten.view.default(view_1090, [96]);  view_1090 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_100: "f32[8, 96, 56, 56]" = torch.ops.aten.neg.default(mul_1778)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_33: "f32[8, 96, 56, 56]" = torch.ops.aten.avg_pool2d_backward.default(mul_1778, add_15, [3, 3], [1, 1], [1, 1], False, False, None);  mul_1778 = add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_556: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(neg_100, avg_pool2d_backward_33);  neg_100 = avg_pool2d_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_278: "f32[8, 1]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    alias_279: "f32[8, 1]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    mul_1779: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_556, add_13)
    view_1092: "f32[8, 96, 3136]" = torch.ops.aten.view.default(mul_1779, [8, 96, 3136]);  mul_1779 = None
    sum_547: "f32[8, 96]" = torch.ops.aten.sum.dim_IntList(view_1092, [2]);  view_1092 = None
    view_1093: "f32[8, 96, 3136]" = torch.ops.aten.view.default(add_556, [8, 96, 3136])
    sum_548: "f32[8, 96]" = torch.ops.aten.sum.dim_IntList(view_1093, [2]);  view_1093 = None
    unsqueeze_1102: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_13, 0)
    mul_1780: "f32[8, 96]" = torch.ops.aten.mul.Tensor(sum_547, unsqueeze_1102);  unsqueeze_1102 = None
    view_1094: "f32[8, 1, 96]" = torch.ops.aten.view.default(mul_1780, [8, 1, 96]);  mul_1780 = None
    sum_549: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_1094, [2]);  view_1094 = None
    unsqueeze_1103: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_13, 0)
    mul_1781: "f32[8, 96]" = torch.ops.aten.mul.Tensor(sum_548, unsqueeze_1103);  unsqueeze_1103 = None
    view_1095: "f32[8, 1, 96]" = torch.ops.aten.view.default(mul_1781, [8, 1, 96]);  mul_1781 = None
    sum_550: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_1095, [2]);  view_1095 = None
    unsqueeze_1104: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_279, -1)
    view_1096: "f32[1, 1, 96]" = torch.ops.aten.view.default(primals_13, [1, 1, 96]);  primals_13 = None
    mul_1782: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(unsqueeze_1104, view_1096);  unsqueeze_1104 = view_1096 = None
    mul_1783: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_550, alias_278)
    sub_313: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1783, sum_549);  mul_1783 = sum_549 = None
    mul_1784: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_313, alias_279);  sub_313 = None
    mul_1785: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1784, alias_279);  mul_1784 = None
    mul_1786: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1785, alias_279);  mul_1785 = None
    mul_1787: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1786, 3.321641156462585e-06);  mul_1786 = None
    neg_101: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1787)
    mul_1788: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_101, alias_278);  neg_101 = None
    mul_1789: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_550, alias_279);  sum_550 = None
    mul_1790: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1789, 3.321641156462585e-06);  mul_1789 = None
    sub_314: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1788, mul_1790);  mul_1788 = mul_1790 = None
    unsqueeze_1105: "f32[8, 1, 96, 1]" = torch.ops.aten.unsqueeze.default(mul_1782, -1);  mul_1782 = None
    unsqueeze_1106: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1787, -1);  mul_1787 = None
    unsqueeze_1107: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1106, -1);  unsqueeze_1106 = None
    unsqueeze_1108: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_314, -1);  sub_314 = None
    unsqueeze_1109: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1108, -1);  unsqueeze_1108 = None
    view_1097: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_556, [8, 1, 96, 3136]);  add_556 = None
    mul_1791: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(view_1097, unsqueeze_1105);  view_1097 = unsqueeze_1105 = None
    view_1098: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_13, [8, 1, 96, 3136]);  add_13 = None
    mul_1792: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(view_1098, unsqueeze_1107);  view_1098 = unsqueeze_1107 = None
    add_557: "f32[8, 1, 96, 3136]" = torch.ops.aten.add.Tensor(mul_1791, mul_1792);  mul_1791 = mul_1792 = None
    add_558: "f32[8, 1, 96, 3136]" = torch.ops.aten.add.Tensor(add_557, unsqueeze_1109);  add_557 = unsqueeze_1109 = None
    view_1099: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(add_558, [8, 96, 56, 56]);  add_558 = None
    view_1100: "f32[8, 1, 96]" = torch.ops.aten.view.default(sum_547, [8, 1, 96]);  sum_547 = None
    view_1101: "f32[8, 1, 96]" = torch.ops.aten.view.default(sum_548, [8, 1, 96])
    unsqueeze_1110: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_278, -1);  alias_278 = None
    mul_1793: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(view_1101, unsqueeze_1110);  view_1101 = unsqueeze_1110 = None
    sub_315: "f32[8, 1, 96]" = torch.ops.aten.sub.Tensor(view_1100, mul_1793);  view_1100 = mul_1793 = None
    unsqueeze_1111: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_279, -1);  alias_279 = None
    mul_1794: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(sub_315, unsqueeze_1111);  sub_315 = unsqueeze_1111 = None
    sum_551: "f32[1, 96]" = torch.ops.aten.sum.dim_IntList(mul_1794, [0]);  mul_1794 = None
    view_1102: "f32[96]" = torch.ops.aten.view.default(sum_551, [96]);  sum_551 = None
    sum_552: "f32[96]" = torch.ops.aten.sum.dim_IntList(sum_548, [0]);  sum_548 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_559: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_555, view_1099);  add_555 = view_1099 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1795: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_559, clone_3);  clone_3 = None
    mul_1796: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_559, view_11);  view_11 = None
    sum_553: "f32[1, 96, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1795, [0, 2, 3], True);  mul_1795 = None
    view_1103: "f32[96, 1, 1]" = torch.ops.aten.view.default(sum_553, [96, 1, 1]);  sum_553 = None
    view_1104: "f32[96]" = torch.ops.aten.view.default(view_1103, [96]);  view_1103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_554: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1796, [0, 2, 3])
    convolution_backward_71 = torch.ops.aten.convolution_backward.default(mul_1796, clone_2, primals_227, [96], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1796 = clone_2 = primals_227 = None
    getitem_359: "f32[8, 384, 56, 56]" = convolution_backward_71[0]
    getitem_360: "f32[96, 384, 1, 1]" = convolution_backward_71[1];  convolution_backward_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1797: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_3, 0.7071067811865476)
    erf_70: "f32[8, 384, 56, 56]" = torch.ops.aten.erf.default(mul_1797);  mul_1797 = None
    add_560: "f32[8, 384, 56, 56]" = torch.ops.aten.add.Tensor(erf_70, 1);  erf_70 = None
    mul_1798: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(add_560, 0.5);  add_560 = None
    mul_1799: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_3, convolution_3)
    mul_1800: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(mul_1799, -0.5);  mul_1799 = None
    exp_34: "f32[8, 384, 56, 56]" = torch.ops.aten.exp.default(mul_1800);  mul_1800 = None
    mul_1801: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(exp_34, 0.3989422804014327);  exp_34 = None
    mul_1802: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_3, mul_1801);  convolution_3 = mul_1801 = None
    add_561: "f32[8, 384, 56, 56]" = torch.ops.aten.add.Tensor(mul_1798, mul_1802);  mul_1798 = mul_1802 = None
    mul_1803: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_359, add_561);  getitem_359 = add_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_555: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_1803, [0, 2, 3])
    convolution_backward_72 = torch.ops.aten.convolution_backward.default(mul_1803, add_11, primals_225, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1803 = add_11 = primals_225 = None
    getitem_362: "f32[8, 96, 56, 56]" = convolution_backward_72[0]
    getitem_363: "f32[384, 96, 1, 1]" = convolution_backward_72[1];  convolution_backward_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_280: "f32[8, 1]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    alias_281: "f32[8, 1]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    mul_1804: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_362, add_9)
    view_1105: "f32[8, 96, 3136]" = torch.ops.aten.view.default(mul_1804, [8, 96, 3136]);  mul_1804 = None
    sum_556: "f32[8, 96]" = torch.ops.aten.sum.dim_IntList(view_1105, [2]);  view_1105 = None
    view_1106: "f32[8, 96, 3136]" = torch.ops.aten.view.default(getitem_362, [8, 96, 3136])
    sum_557: "f32[8, 96]" = torch.ops.aten.sum.dim_IntList(view_1106, [2]);  view_1106 = None
    unsqueeze_1112: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_10, 0)
    mul_1805: "f32[8, 96]" = torch.ops.aten.mul.Tensor(sum_556, unsqueeze_1112);  unsqueeze_1112 = None
    view_1107: "f32[8, 1, 96]" = torch.ops.aten.view.default(mul_1805, [8, 1, 96]);  mul_1805 = None
    sum_558: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_1107, [2]);  view_1107 = None
    unsqueeze_1113: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_10, 0)
    mul_1806: "f32[8, 96]" = torch.ops.aten.mul.Tensor(sum_557, unsqueeze_1113);  unsqueeze_1113 = None
    view_1108: "f32[8, 1, 96]" = torch.ops.aten.view.default(mul_1806, [8, 1, 96]);  mul_1806 = None
    sum_559: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_1108, [2]);  view_1108 = None
    unsqueeze_1114: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_281, -1)
    view_1109: "f32[1, 1, 96]" = torch.ops.aten.view.default(primals_10, [1, 1, 96]);  primals_10 = None
    mul_1807: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(unsqueeze_1114, view_1109);  unsqueeze_1114 = view_1109 = None
    mul_1808: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_559, alias_280)
    sub_316: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1808, sum_558);  mul_1808 = sum_558 = None
    mul_1809: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_316, alias_281);  sub_316 = None
    mul_1810: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1809, alias_281);  mul_1809 = None
    mul_1811: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1810, alias_281);  mul_1810 = None
    mul_1812: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1811, 3.321641156462585e-06);  mul_1811 = None
    neg_102: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1812)
    mul_1813: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_102, alias_280);  neg_102 = None
    mul_1814: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_559, alias_281);  sum_559 = None
    mul_1815: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1814, 3.321641156462585e-06);  mul_1814 = None
    sub_317: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1813, mul_1815);  mul_1813 = mul_1815 = None
    unsqueeze_1115: "f32[8, 1, 96, 1]" = torch.ops.aten.unsqueeze.default(mul_1807, -1);  mul_1807 = None
    unsqueeze_1116: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1812, -1);  mul_1812 = None
    unsqueeze_1117: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1116, -1);  unsqueeze_1116 = None
    unsqueeze_1118: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_317, -1);  sub_317 = None
    unsqueeze_1119: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1118, -1);  unsqueeze_1118 = None
    view_1110: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(getitem_362, [8, 1, 96, 3136]);  getitem_362 = None
    mul_1816: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(view_1110, unsqueeze_1115);  view_1110 = unsqueeze_1115 = None
    view_1111: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_9, [8, 1, 96, 3136]);  add_9 = None
    mul_1817: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(view_1111, unsqueeze_1117);  view_1111 = unsqueeze_1117 = None
    add_562: "f32[8, 1, 96, 3136]" = torch.ops.aten.add.Tensor(mul_1816, mul_1817);  mul_1816 = mul_1817 = None
    add_563: "f32[8, 1, 96, 3136]" = torch.ops.aten.add.Tensor(add_562, unsqueeze_1119);  add_562 = unsqueeze_1119 = None
    view_1112: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(add_563, [8, 96, 56, 56]);  add_563 = None
    view_1113: "f32[8, 1, 96]" = torch.ops.aten.view.default(sum_556, [8, 1, 96]);  sum_556 = None
    view_1114: "f32[8, 1, 96]" = torch.ops.aten.view.default(sum_557, [8, 1, 96])
    unsqueeze_1120: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_280, -1);  alias_280 = None
    mul_1818: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(view_1114, unsqueeze_1120);  view_1114 = unsqueeze_1120 = None
    sub_318: "f32[8, 1, 96]" = torch.ops.aten.sub.Tensor(view_1113, mul_1818);  view_1113 = mul_1818 = None
    unsqueeze_1121: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_281, -1);  alias_281 = None
    mul_1819: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(sub_318, unsqueeze_1121);  sub_318 = unsqueeze_1121 = None
    sum_560: "f32[1, 96]" = torch.ops.aten.sum.dim_IntList(mul_1819, [0]);  mul_1819 = None
    view_1115: "f32[96]" = torch.ops.aten.view.default(sum_560, [96]);  sum_560 = None
    sum_561: "f32[96]" = torch.ops.aten.sum.dim_IntList(sum_557, [0]);  sum_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_564: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_559, view_1112);  add_559 = view_1112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1820: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_564, sub_4);  sub_4 = None
    mul_1821: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_564, view_8);  view_8 = None
    sum_562: "f32[1, 96, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1820, [0, 2, 3], True);  mul_1820 = None
    view_1116: "f32[96, 1, 1]" = torch.ops.aten.view.default(sum_562, [96, 1, 1]);  sum_562 = None
    view_1117: "f32[96]" = torch.ops.aten.view.default(view_1116, [96]);  view_1116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_103: "f32[8, 96, 56, 56]" = torch.ops.aten.neg.default(mul_1821)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_34: "f32[8, 96, 56, 56]" = torch.ops.aten.avg_pool2d_backward.default(mul_1821, add_8, [3, 3], [1, 1], [1, 1], False, False, None);  mul_1821 = add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_565: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(neg_103, avg_pool2d_backward_34);  neg_103 = avg_pool2d_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_282: "f32[8, 1]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    alias_283: "f32[8, 1]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul_1822: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_565, add_6)
    view_1118: "f32[8, 96, 3136]" = torch.ops.aten.view.default(mul_1822, [8, 96, 3136]);  mul_1822 = None
    sum_563: "f32[8, 96]" = torch.ops.aten.sum.dim_IntList(view_1118, [2]);  view_1118 = None
    view_1119: "f32[8, 96, 3136]" = torch.ops.aten.view.default(add_565, [8, 96, 3136])
    sum_564: "f32[8, 96]" = torch.ops.aten.sum.dim_IntList(view_1119, [2]);  view_1119 = None
    unsqueeze_1122: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_7, 0)
    mul_1823: "f32[8, 96]" = torch.ops.aten.mul.Tensor(sum_563, unsqueeze_1122);  unsqueeze_1122 = None
    view_1120: "f32[8, 1, 96]" = torch.ops.aten.view.default(mul_1823, [8, 1, 96]);  mul_1823 = None
    sum_565: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_1120, [2]);  view_1120 = None
    unsqueeze_1123: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_7, 0)
    mul_1824: "f32[8, 96]" = torch.ops.aten.mul.Tensor(sum_564, unsqueeze_1123);  unsqueeze_1123 = None
    view_1121: "f32[8, 1, 96]" = torch.ops.aten.view.default(mul_1824, [8, 1, 96]);  mul_1824 = None
    sum_566: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_1121, [2]);  view_1121 = None
    unsqueeze_1124: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_283, -1)
    view_1122: "f32[1, 1, 96]" = torch.ops.aten.view.default(primals_7, [1, 1, 96]);  primals_7 = None
    mul_1825: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(unsqueeze_1124, view_1122);  unsqueeze_1124 = view_1122 = None
    mul_1826: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_566, alias_282)
    sub_319: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1826, sum_565);  mul_1826 = sum_565 = None
    mul_1827: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_319, alias_283);  sub_319 = None
    mul_1828: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1827, alias_283);  mul_1827 = None
    mul_1829: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1828, alias_283);  mul_1828 = None
    mul_1830: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1829, 3.321641156462585e-06);  mul_1829 = None
    neg_104: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1830)
    mul_1831: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_104, alias_282);  neg_104 = None
    mul_1832: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_566, alias_283);  sum_566 = None
    mul_1833: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1832, 3.321641156462585e-06);  mul_1832 = None
    sub_320: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1831, mul_1833);  mul_1831 = mul_1833 = None
    unsqueeze_1125: "f32[8, 1, 96, 1]" = torch.ops.aten.unsqueeze.default(mul_1825, -1);  mul_1825 = None
    unsqueeze_1126: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1830, -1);  mul_1830 = None
    unsqueeze_1127: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1126, -1);  unsqueeze_1126 = None
    unsqueeze_1128: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_320, -1);  sub_320 = None
    unsqueeze_1129: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1128, -1);  unsqueeze_1128 = None
    view_1123: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_565, [8, 1, 96, 3136]);  add_565 = None
    mul_1834: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(view_1123, unsqueeze_1125);  view_1123 = unsqueeze_1125 = None
    view_1124: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_6, [8, 1, 96, 3136]);  add_6 = None
    mul_1835: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(view_1124, unsqueeze_1127);  view_1124 = unsqueeze_1127 = None
    add_566: "f32[8, 1, 96, 3136]" = torch.ops.aten.add.Tensor(mul_1834, mul_1835);  mul_1834 = mul_1835 = None
    add_567: "f32[8, 1, 96, 3136]" = torch.ops.aten.add.Tensor(add_566, unsqueeze_1129);  add_566 = unsqueeze_1129 = None
    view_1125: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(add_567, [8, 96, 56, 56]);  add_567 = None
    view_1126: "f32[8, 1, 96]" = torch.ops.aten.view.default(sum_563, [8, 1, 96]);  sum_563 = None
    view_1127: "f32[8, 1, 96]" = torch.ops.aten.view.default(sum_564, [8, 1, 96])
    unsqueeze_1130: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_282, -1);  alias_282 = None
    mul_1836: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(view_1127, unsqueeze_1130);  view_1127 = unsqueeze_1130 = None
    sub_321: "f32[8, 1, 96]" = torch.ops.aten.sub.Tensor(view_1126, mul_1836);  view_1126 = mul_1836 = None
    unsqueeze_1131: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_283, -1);  alias_283 = None
    mul_1837: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(sub_321, unsqueeze_1131);  sub_321 = unsqueeze_1131 = None
    sum_567: "f32[1, 96]" = torch.ops.aten.sum.dim_IntList(mul_1837, [0]);  mul_1837 = None
    view_1128: "f32[96]" = torch.ops.aten.view.default(sum_567, [96]);  sum_567 = None
    sum_568: "f32[96]" = torch.ops.aten.sum.dim_IntList(sum_564, [0]);  sum_564 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_568: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_564, view_1125);  add_564 = view_1125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1838: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_568, clone_1);  clone_1 = None
    mul_1839: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_568, view_5);  view_5 = None
    sum_569: "f32[1, 96, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1838, [0, 2, 3], True);  mul_1838 = None
    view_1129: "f32[96, 1, 1]" = torch.ops.aten.view.default(sum_569, [96, 1, 1]);  sum_569 = None
    view_1130: "f32[96]" = torch.ops.aten.view.default(view_1129, [96]);  view_1129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    sum_570: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1839, [0, 2, 3])
    convolution_backward_73 = torch.ops.aten.convolution_backward.default(mul_1839, clone, primals_223, [96], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1839 = clone = primals_223 = None
    getitem_365: "f32[8, 384, 56, 56]" = convolution_backward_73[0]
    getitem_366: "f32[96, 384, 1, 1]" = convolution_backward_73[1];  convolution_backward_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1840: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_1, 0.7071067811865476)
    erf_71: "f32[8, 384, 56, 56]" = torch.ops.aten.erf.default(mul_1840);  mul_1840 = None
    add_569: "f32[8, 384, 56, 56]" = torch.ops.aten.add.Tensor(erf_71, 1);  erf_71 = None
    mul_1841: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(add_569, 0.5);  add_569 = None
    mul_1842: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_1, convolution_1)
    mul_1843: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(mul_1842, -0.5);  mul_1842 = None
    exp_35: "f32[8, 384, 56, 56]" = torch.ops.aten.exp.default(mul_1843);  mul_1843 = None
    mul_1844: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(exp_35, 0.3989422804014327);  exp_35 = None
    mul_1845: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_1, mul_1844);  convolution_1 = mul_1844 = None
    add_570: "f32[8, 384, 56, 56]" = torch.ops.aten.add.Tensor(mul_1841, mul_1845);  mul_1841 = mul_1845 = None
    mul_1846: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_365, add_570);  getitem_365 = add_570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_571: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_1846, [0, 2, 3])
    convolution_backward_74 = torch.ops.aten.convolution_backward.default(mul_1846, add_4, primals_221, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1846 = add_4 = primals_221 = None
    getitem_368: "f32[8, 96, 56, 56]" = convolution_backward_74[0]
    getitem_369: "f32[384, 96, 1, 1]" = convolution_backward_74[1];  convolution_backward_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_284: "f32[8, 1]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    alias_285: "f32[8, 1]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    mul_1847: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_368, add_2)
    view_1131: "f32[8, 96, 3136]" = torch.ops.aten.view.default(mul_1847, [8, 96, 3136]);  mul_1847 = None
    sum_572: "f32[8, 96]" = torch.ops.aten.sum.dim_IntList(view_1131, [2]);  view_1131 = None
    view_1132: "f32[8, 96, 3136]" = torch.ops.aten.view.default(getitem_368, [8, 96, 3136])
    sum_573: "f32[8, 96]" = torch.ops.aten.sum.dim_IntList(view_1132, [2]);  view_1132 = None
    unsqueeze_1132: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_4, 0)
    mul_1848: "f32[8, 96]" = torch.ops.aten.mul.Tensor(sum_572, unsqueeze_1132);  unsqueeze_1132 = None
    view_1133: "f32[8, 1, 96]" = torch.ops.aten.view.default(mul_1848, [8, 1, 96]);  mul_1848 = None
    sum_574: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_1133, [2]);  view_1133 = None
    unsqueeze_1133: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_4, 0)
    mul_1849: "f32[8, 96]" = torch.ops.aten.mul.Tensor(sum_573, unsqueeze_1133);  unsqueeze_1133 = None
    view_1134: "f32[8, 1, 96]" = torch.ops.aten.view.default(mul_1849, [8, 1, 96]);  mul_1849 = None
    sum_575: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_1134, [2]);  view_1134 = None
    unsqueeze_1134: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_285, -1)
    view_1135: "f32[1, 1, 96]" = torch.ops.aten.view.default(primals_4, [1, 1, 96]);  primals_4 = None
    mul_1850: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(unsqueeze_1134, view_1135);  unsqueeze_1134 = view_1135 = None
    mul_1851: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_575, alias_284)
    sub_322: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1851, sum_574);  mul_1851 = sum_574 = None
    mul_1852: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_322, alias_285);  sub_322 = None
    mul_1853: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1852, alias_285);  mul_1852 = None
    mul_1854: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1853, alias_285);  mul_1853 = None
    mul_1855: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1854, 3.321641156462585e-06);  mul_1854 = None
    neg_105: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1855)
    mul_1856: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_105, alias_284);  neg_105 = None
    mul_1857: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_575, alias_285);  sum_575 = None
    mul_1858: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1857, 3.321641156462585e-06);  mul_1857 = None
    sub_323: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1856, mul_1858);  mul_1856 = mul_1858 = None
    unsqueeze_1135: "f32[8, 1, 96, 1]" = torch.ops.aten.unsqueeze.default(mul_1850, -1);  mul_1850 = None
    unsqueeze_1136: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1855, -1);  mul_1855 = None
    unsqueeze_1137: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1136, -1);  unsqueeze_1136 = None
    unsqueeze_1138: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_323, -1);  sub_323 = None
    unsqueeze_1139: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1138, -1);  unsqueeze_1138 = None
    view_1136: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(getitem_368, [8, 1, 96, 3136]);  getitem_368 = None
    mul_1859: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(view_1136, unsqueeze_1135);  view_1136 = unsqueeze_1135 = None
    view_1137: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_2, [8, 1, 96, 3136]);  add_2 = None
    mul_1860: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(view_1137, unsqueeze_1137);  view_1137 = unsqueeze_1137 = None
    add_571: "f32[8, 1, 96, 3136]" = torch.ops.aten.add.Tensor(mul_1859, mul_1860);  mul_1859 = mul_1860 = None
    add_572: "f32[8, 1, 96, 3136]" = torch.ops.aten.add.Tensor(add_571, unsqueeze_1139);  add_571 = unsqueeze_1139 = None
    view_1138: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(add_572, [8, 96, 56, 56]);  add_572 = None
    view_1139: "f32[8, 1, 96]" = torch.ops.aten.view.default(sum_572, [8, 1, 96]);  sum_572 = None
    view_1140: "f32[8, 1, 96]" = torch.ops.aten.view.default(sum_573, [8, 1, 96])
    unsqueeze_1140: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_284, -1);  alias_284 = None
    mul_1861: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(view_1140, unsqueeze_1140);  view_1140 = unsqueeze_1140 = None
    sub_324: "f32[8, 1, 96]" = torch.ops.aten.sub.Tensor(view_1139, mul_1861);  view_1139 = mul_1861 = None
    unsqueeze_1141: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_285, -1);  alias_285 = None
    mul_1862: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(sub_324, unsqueeze_1141);  sub_324 = unsqueeze_1141 = None
    sum_576: "f32[1, 96]" = torch.ops.aten.sum.dim_IntList(mul_1862, [0]);  mul_1862 = None
    view_1141: "f32[96]" = torch.ops.aten.view.default(sum_576, [96]);  sum_576 = None
    sum_577: "f32[96]" = torch.ops.aten.sum.dim_IntList(sum_573, [0]);  sum_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_573: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_568, view_1138);  add_568 = view_1138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    mul_1863: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_573, sub_1);  sub_1 = None
    mul_1864: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_573, view_2);  view_2 = None
    sum_578: "f32[1, 96, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1863, [0, 2, 3], True);  mul_1863 = None
    view_1142: "f32[96, 1, 1]" = torch.ops.aten.view.default(sum_578, [96, 1, 1]);  sum_578 = None
    view_1143: "f32[96]" = torch.ops.aten.view.default(view_1142, [96]);  view_1142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    neg_106: "f32[8, 96, 56, 56]" = torch.ops.aten.neg.default(mul_1864)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    avg_pool2d_backward_35: "f32[8, 96, 56, 56]" = torch.ops.aten.avg_pool2d_backward.default(mul_1864, add_1, [3, 3], [1, 1], [1, 1], False, False, None);  mul_1864 = add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    add_574: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(neg_106, avg_pool2d_backward_35);  neg_106 = avg_pool2d_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    alias_286: "f32[8, 1]" = torch.ops.aten.alias.default(alias);  alias = None
    alias_287: "f32[8, 1]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    mul_1865: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_574, convolution)
    view_1144: "f32[8, 96, 3136]" = torch.ops.aten.view.default(mul_1865, [8, 96, 3136]);  mul_1865 = None
    sum_579: "f32[8, 96]" = torch.ops.aten.sum.dim_IntList(view_1144, [2]);  view_1144 = None
    view_1145: "f32[8, 96, 3136]" = torch.ops.aten.view.default(add_574, [8, 96, 3136])
    sum_580: "f32[8, 96]" = torch.ops.aten.sum.dim_IntList(view_1145, [2]);  view_1145 = None
    unsqueeze_1142: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_1, 0)
    mul_1866: "f32[8, 96]" = torch.ops.aten.mul.Tensor(sum_579, unsqueeze_1142);  unsqueeze_1142 = None
    view_1146: "f32[8, 1, 96]" = torch.ops.aten.view.default(mul_1866, [8, 1, 96]);  mul_1866 = None
    sum_581: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_1146, [2]);  view_1146 = None
    unsqueeze_1143: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_1, 0)
    mul_1867: "f32[8, 96]" = torch.ops.aten.mul.Tensor(sum_580, unsqueeze_1143);  unsqueeze_1143 = None
    view_1147: "f32[8, 1, 96]" = torch.ops.aten.view.default(mul_1867, [8, 1, 96]);  mul_1867 = None
    sum_582: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(view_1147, [2]);  view_1147 = None
    unsqueeze_1144: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_287, -1)
    view_1148: "f32[1, 1, 96]" = torch.ops.aten.view.default(primals_1, [1, 1, 96]);  primals_1 = None
    mul_1868: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(unsqueeze_1144, view_1148);  unsqueeze_1144 = view_1148 = None
    mul_1869: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_582, alias_286)
    sub_325: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1869, sum_581);  mul_1869 = sum_581 = None
    mul_1870: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sub_325, alias_287);  sub_325 = None
    mul_1871: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1870, alias_287);  mul_1870 = None
    mul_1872: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1871, alias_287);  mul_1871 = None
    mul_1873: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1872, 3.321641156462585e-06);  mul_1872 = None
    neg_107: "f32[8, 1]" = torch.ops.aten.neg.default(mul_1873)
    mul_1874: "f32[8, 1]" = torch.ops.aten.mul.Tensor(neg_107, alias_286);  neg_107 = None
    mul_1875: "f32[8, 1]" = torch.ops.aten.mul.Tensor(sum_582, alias_287);  sum_582 = None
    mul_1876: "f32[8, 1]" = torch.ops.aten.mul.Tensor(mul_1875, 3.321641156462585e-06);  mul_1875 = None
    sub_326: "f32[8, 1]" = torch.ops.aten.sub.Tensor(mul_1874, mul_1876);  mul_1874 = mul_1876 = None
    unsqueeze_1145: "f32[8, 1, 96, 1]" = torch.ops.aten.unsqueeze.default(mul_1868, -1);  mul_1868 = None
    unsqueeze_1146: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_1873, -1);  mul_1873 = None
    unsqueeze_1147: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1146, -1);  unsqueeze_1146 = None
    unsqueeze_1148: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(sub_326, -1);  sub_326 = None
    unsqueeze_1149: "f32[8, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1148, -1);  unsqueeze_1148 = None
    view_1149: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(add_574, [8, 1, 96, 3136]);  add_574 = None
    mul_1877: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(view_1149, unsqueeze_1145);  view_1149 = unsqueeze_1145 = None
    view_1150: "f32[8, 1, 96, 3136]" = torch.ops.aten.view.default(convolution, [8, 1, 96, 3136]);  convolution = None
    mul_1878: "f32[8, 1, 96, 3136]" = torch.ops.aten.mul.Tensor(view_1150, unsqueeze_1147);  view_1150 = unsqueeze_1147 = None
    add_575: "f32[8, 1, 96, 3136]" = torch.ops.aten.add.Tensor(mul_1877, mul_1878);  mul_1877 = mul_1878 = None
    add_576: "f32[8, 1, 96, 3136]" = torch.ops.aten.add.Tensor(add_575, unsqueeze_1149);  add_575 = unsqueeze_1149 = None
    view_1151: "f32[8, 96, 56, 56]" = torch.ops.aten.view.default(add_576, [8, 96, 56, 56]);  add_576 = None
    view_1152: "f32[8, 1, 96]" = torch.ops.aten.view.default(sum_579, [8, 1, 96]);  sum_579 = None
    view_1153: "f32[8, 1, 96]" = torch.ops.aten.view.default(sum_580, [8, 1, 96])
    unsqueeze_1150: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_286, -1);  alias_286 = None
    mul_1879: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(view_1153, unsqueeze_1150);  view_1153 = unsqueeze_1150 = None
    sub_327: "f32[8, 1, 96]" = torch.ops.aten.sub.Tensor(view_1152, mul_1879);  view_1152 = mul_1879 = None
    unsqueeze_1151: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(alias_287, -1);  alias_287 = None
    mul_1880: "f32[8, 1, 96]" = torch.ops.aten.mul.Tensor(sub_327, unsqueeze_1151);  sub_327 = unsqueeze_1151 = None
    sum_583: "f32[1, 96]" = torch.ops.aten.sum.dim_IntList(mul_1880, [0]);  mul_1880 = None
    view_1154: "f32[96]" = torch.ops.aten.view.default(sum_583, [96]);  sum_583 = None
    sum_584: "f32[96]" = torch.ops.aten.sum.dim_IntList(sum_580, [0]);  sum_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    add_577: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_573, view_1151);  add_573 = view_1151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:72, code: x = self.conv(x)
    sum_585: "f32[96]" = torch.ops.aten.sum.dim_IntList(add_577, [0, 2, 3])
    convolution_backward_75 = torch.ops.aten.convolution_backward.default(add_577, primals_373, primals_219, [96], [4, 4], [2, 2], [1, 1], False, [0, 0], 1, [False, True, False]);  add_577 = primals_373 = primals_219 = None
    getitem_372: "f32[96, 3, 7, 7]" = convolution_backward_75[1];  convolution_backward_75 = None
    return pytree.tree_unflatten([addmm, view_1154, sum_584, view_1143, view_1141, sum_577, view_1130, view_1128, sum_568, view_1117, view_1115, sum_561, view_1104, view_1102, sum_552, view_1091, view_1089, sum_545, view_1078, view_1076, sum_536, view_1065, view_1063, sum_529, view_1052, view_1050, sum_520, view_1039, view_1037, sum_513, view_1026, view_1024, sum_504, view_1013, view_1011, sum_497, view_1000, view_998, sum_487, view_987, view_985, sum_480, view_974, view_972, sum_471, view_961, view_959, sum_464, view_948, view_946, sum_455, view_935, view_933, sum_448, view_922, view_920, sum_439, view_909, view_907, sum_432, view_896, view_894, sum_423, view_883, view_881, sum_416, view_870, view_868, sum_407, view_857, view_855, sum_400, view_844, view_842, sum_390, view_831, view_829, sum_383, view_818, view_816, sum_374, view_805, view_803, sum_367, view_792, view_790, sum_358, view_779, view_777, sum_351, view_766, view_764, sum_342, view_753, view_751, sum_335, view_740, view_738, sum_326, view_727, view_725, sum_319, view_714, view_712, sum_310, view_701, view_699, sum_303, view_688, view_686, sum_294, view_675, view_673, sum_287, view_662, view_660, sum_278, view_649, view_647, sum_271, view_636, view_634, sum_262, view_623, view_621, sum_255, view_610, view_608, sum_246, view_597, view_595, sum_239, view_584, view_582, sum_230, view_571, view_569, sum_223, view_558, view_556, sum_214, view_545, view_543, sum_207, view_532, view_530, sum_198, view_519, view_517, sum_191, view_506, view_504, sum_182, view_493, view_491, sum_175, view_480, view_478, sum_166, view_467, view_465, sum_159, view_454, view_452, sum_150, view_441, view_439, sum_143, view_428, view_426, sum_134, view_415, view_413, sum_127, view_402, view_400, sum_118, view_389, view_387, sum_111, view_376, view_374, sum_101, view_363, view_361, sum_94, view_350, view_348, sum_85, view_337, view_335, sum_78, view_324, view_322, sum_69, view_311, view_309, sum_62, view_298, view_296, sum_53, view_285, view_283, sum_46, view_272, view_270, sum_37, view_259, view_257, sum_30, view_246, view_244, sum_21, view_233, view_231, sum_14, view_220, sum_4, sum_5, getitem_372, sum_585, getitem_369, sum_571, getitem_366, sum_570, getitem_363, sum_555, getitem_360, sum_554, getitem_357, sum_539, getitem_354, sum_538, getitem_351, sum_523, getitem_348, sum_522, getitem_345, sum_507, getitem_342, sum_506, getitem_339, sum_491, getitem_336, sum_490, getitem_333, sum_488, getitem_330, sum_474, getitem_327, sum_473, getitem_324, sum_458, getitem_321, sum_457, getitem_318, sum_442, getitem_315, sum_441, getitem_312, sum_426, getitem_309, sum_425, getitem_306, sum_410, getitem_303, sum_409, getitem_300, sum_394, getitem_297, sum_393, getitem_294, sum_391, getitem_291, sum_377, getitem_288, sum_376, getitem_285, sum_361, getitem_282, sum_360, getitem_279, sum_345, getitem_276, sum_344, getitem_273, sum_329, getitem_270, sum_328, getitem_267, sum_313, getitem_264, sum_312, getitem_261, sum_297, getitem_258, sum_296, getitem_255, sum_281, getitem_252, sum_280, getitem_249, sum_265, getitem_246, sum_264, getitem_243, sum_249, getitem_240, sum_248, getitem_237, sum_233, getitem_234, sum_232, getitem_231, sum_217, getitem_228, sum_216, getitem_225, sum_201, getitem_222, sum_200, getitem_219, sum_185, getitem_216, sum_184, getitem_213, sum_169, getitem_210, sum_168, getitem_207, sum_153, getitem_204, sum_152, getitem_201, sum_137, getitem_198, sum_136, getitem_195, sum_121, getitem_192, sum_120, getitem_189, sum_105, getitem_186, sum_104, getitem_183, sum_102, getitem_180, sum_88, getitem_177, sum_87, getitem_174, sum_72, getitem_171, sum_71, getitem_168, sum_56, getitem_165, sum_55, getitem_162, sum_40, getitem_159, sum_39, getitem_156, sum_24, getitem_153, sum_23, getitem_150, sum_8, getitem_147, sum_7, permute_6, view_217, None], self._out_spec)
    