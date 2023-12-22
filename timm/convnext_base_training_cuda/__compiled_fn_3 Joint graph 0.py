from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[128]"; primals_2: "f32[128]"; primals_3: "f32[128]"; primals_4: "f32[128]"; primals_5: "f32[128]"; primals_6: "f32[128]"; primals_7: "f32[128]"; primals_8: "f32[128]"; primals_9: "f32[128]"; primals_10: "f32[128]"; primals_11: "f32[128]"; primals_12: "f32[128]"; primals_13: "f32[128]"; primals_14: "f32[256]"; primals_15: "f32[256]"; primals_16: "f32[256]"; primals_17: "f32[256]"; primals_18: "f32[256]"; primals_19: "f32[256]"; primals_20: "f32[256]"; primals_21: "f32[256]"; primals_22: "f32[256]"; primals_23: "f32[256]"; primals_24: "f32[256]"; primals_25: "f32[512]"; primals_26: "f32[512]"; primals_27: "f32[512]"; primals_28: "f32[512]"; primals_29: "f32[512]"; primals_30: "f32[512]"; primals_31: "f32[512]"; primals_32: "f32[512]"; primals_33: "f32[512]"; primals_34: "f32[512]"; primals_35: "f32[512]"; primals_36: "f32[512]"; primals_37: "f32[512]"; primals_38: "f32[512]"; primals_39: "f32[512]"; primals_40: "f32[512]"; primals_41: "f32[512]"; primals_42: "f32[512]"; primals_43: "f32[512]"; primals_44: "f32[512]"; primals_45: "f32[512]"; primals_46: "f32[512]"; primals_47: "f32[512]"; primals_48: "f32[512]"; primals_49: "f32[512]"; primals_50: "f32[512]"; primals_51: "f32[512]"; primals_52: "f32[512]"; primals_53: "f32[512]"; primals_54: "f32[512]"; primals_55: "f32[512]"; primals_56: "f32[512]"; primals_57: "f32[512]"; primals_58: "f32[512]"; primals_59: "f32[512]"; primals_60: "f32[512]"; primals_61: "f32[512]"; primals_62: "f32[512]"; primals_63: "f32[512]"; primals_64: "f32[512]"; primals_65: "f32[512]"; primals_66: "f32[512]"; primals_67: "f32[512]"; primals_68: "f32[512]"; primals_69: "f32[512]"; primals_70: "f32[512]"; primals_71: "f32[512]"; primals_72: "f32[512]"; primals_73: "f32[512]"; primals_74: "f32[512]"; primals_75: "f32[512]"; primals_76: "f32[512]"; primals_77: "f32[512]"; primals_78: "f32[512]"; primals_79: "f32[512]"; primals_80: "f32[512]"; primals_81: "f32[512]"; primals_82: "f32[512]"; primals_83: "f32[512]"; primals_84: "f32[512]"; primals_85: "f32[512]"; primals_86: "f32[512]"; primals_87: "f32[512]"; primals_88: "f32[512]"; primals_89: "f32[512]"; primals_90: "f32[512]"; primals_91: "f32[512]"; primals_92: "f32[512]"; primals_93: "f32[512]"; primals_94: "f32[512]"; primals_95: "f32[512]"; primals_96: "f32[512]"; primals_97: "f32[512]"; primals_98: "f32[512]"; primals_99: "f32[512]"; primals_100: "f32[512]"; primals_101: "f32[512]"; primals_102: "f32[512]"; primals_103: "f32[512]"; primals_104: "f32[512]"; primals_105: "f32[512]"; primals_106: "f32[512]"; primals_107: "f32[512]"; primals_108: "f32[1024]"; primals_109: "f32[1024]"; primals_110: "f32[1024]"; primals_111: "f32[1024]"; primals_112: "f32[1024]"; primals_113: "f32[1024]"; primals_114: "f32[1024]"; primals_115: "f32[1024]"; primals_116: "f32[1024]"; primals_117: "f32[1024]"; primals_118: "f32[1024]"; primals_119: "f32[128, 3, 4, 4]"; primals_120: "f32[128]"; primals_121: "f32[128, 1, 7, 7]"; primals_122: "f32[128]"; primals_123: "f32[512, 128]"; primals_124: "f32[512]"; primals_125: "f32[128, 512]"; primals_126: "f32[128]"; primals_127: "f32[128, 1, 7, 7]"; primals_128: "f32[128]"; primals_129: "f32[512, 128]"; primals_130: "f32[512]"; primals_131: "f32[128, 512]"; primals_132: "f32[128]"; primals_133: "f32[128, 1, 7, 7]"; primals_134: "f32[128]"; primals_135: "f32[512, 128]"; primals_136: "f32[512]"; primals_137: "f32[128, 512]"; primals_138: "f32[128]"; primals_139: "f32[256, 128, 2, 2]"; primals_140: "f32[256]"; primals_141: "f32[256, 1, 7, 7]"; primals_142: "f32[256]"; primals_143: "f32[1024, 256]"; primals_144: "f32[1024]"; primals_145: "f32[256, 1024]"; primals_146: "f32[256]"; primals_147: "f32[256, 1, 7, 7]"; primals_148: "f32[256]"; primals_149: "f32[1024, 256]"; primals_150: "f32[1024]"; primals_151: "f32[256, 1024]"; primals_152: "f32[256]"; primals_153: "f32[256, 1, 7, 7]"; primals_154: "f32[256]"; primals_155: "f32[1024, 256]"; primals_156: "f32[1024]"; primals_157: "f32[256, 1024]"; primals_158: "f32[256]"; primals_159: "f32[512, 256, 2, 2]"; primals_160: "f32[512]"; primals_161: "f32[512, 1, 7, 7]"; primals_162: "f32[512]"; primals_163: "f32[2048, 512]"; primals_164: "f32[2048]"; primals_165: "f32[512, 2048]"; primals_166: "f32[512]"; primals_167: "f32[512, 1, 7, 7]"; primals_168: "f32[512]"; primals_169: "f32[2048, 512]"; primals_170: "f32[2048]"; primals_171: "f32[512, 2048]"; primals_172: "f32[512]"; primals_173: "f32[512, 1, 7, 7]"; primals_174: "f32[512]"; primals_175: "f32[2048, 512]"; primals_176: "f32[2048]"; primals_177: "f32[512, 2048]"; primals_178: "f32[512]"; primals_179: "f32[512, 1, 7, 7]"; primals_180: "f32[512]"; primals_181: "f32[2048, 512]"; primals_182: "f32[2048]"; primals_183: "f32[512, 2048]"; primals_184: "f32[512]"; primals_185: "f32[512, 1, 7, 7]"; primals_186: "f32[512]"; primals_187: "f32[2048, 512]"; primals_188: "f32[2048]"; primals_189: "f32[512, 2048]"; primals_190: "f32[512]"; primals_191: "f32[512, 1, 7, 7]"; primals_192: "f32[512]"; primals_193: "f32[2048, 512]"; primals_194: "f32[2048]"; primals_195: "f32[512, 2048]"; primals_196: "f32[512]"; primals_197: "f32[512, 1, 7, 7]"; primals_198: "f32[512]"; primals_199: "f32[2048, 512]"; primals_200: "f32[2048]"; primals_201: "f32[512, 2048]"; primals_202: "f32[512]"; primals_203: "f32[512, 1, 7, 7]"; primals_204: "f32[512]"; primals_205: "f32[2048, 512]"; primals_206: "f32[2048]"; primals_207: "f32[512, 2048]"; primals_208: "f32[512]"; primals_209: "f32[512, 1, 7, 7]"; primals_210: "f32[512]"; primals_211: "f32[2048, 512]"; primals_212: "f32[2048]"; primals_213: "f32[512, 2048]"; primals_214: "f32[512]"; primals_215: "f32[512, 1, 7, 7]"; primals_216: "f32[512]"; primals_217: "f32[2048, 512]"; primals_218: "f32[2048]"; primals_219: "f32[512, 2048]"; primals_220: "f32[512]"; primals_221: "f32[512, 1, 7, 7]"; primals_222: "f32[512]"; primals_223: "f32[2048, 512]"; primals_224: "f32[2048]"; primals_225: "f32[512, 2048]"; primals_226: "f32[512]"; primals_227: "f32[512, 1, 7, 7]"; primals_228: "f32[512]"; primals_229: "f32[2048, 512]"; primals_230: "f32[2048]"; primals_231: "f32[512, 2048]"; primals_232: "f32[512]"; primals_233: "f32[512, 1, 7, 7]"; primals_234: "f32[512]"; primals_235: "f32[2048, 512]"; primals_236: "f32[2048]"; primals_237: "f32[512, 2048]"; primals_238: "f32[512]"; primals_239: "f32[512, 1, 7, 7]"; primals_240: "f32[512]"; primals_241: "f32[2048, 512]"; primals_242: "f32[2048]"; primals_243: "f32[512, 2048]"; primals_244: "f32[512]"; primals_245: "f32[512, 1, 7, 7]"; primals_246: "f32[512]"; primals_247: "f32[2048, 512]"; primals_248: "f32[2048]"; primals_249: "f32[512, 2048]"; primals_250: "f32[512]"; primals_251: "f32[512, 1, 7, 7]"; primals_252: "f32[512]"; primals_253: "f32[2048, 512]"; primals_254: "f32[2048]"; primals_255: "f32[512, 2048]"; primals_256: "f32[512]"; primals_257: "f32[512, 1, 7, 7]"; primals_258: "f32[512]"; primals_259: "f32[2048, 512]"; primals_260: "f32[2048]"; primals_261: "f32[512, 2048]"; primals_262: "f32[512]"; primals_263: "f32[512, 1, 7, 7]"; primals_264: "f32[512]"; primals_265: "f32[2048, 512]"; primals_266: "f32[2048]"; primals_267: "f32[512, 2048]"; primals_268: "f32[512]"; primals_269: "f32[512, 1, 7, 7]"; primals_270: "f32[512]"; primals_271: "f32[2048, 512]"; primals_272: "f32[2048]"; primals_273: "f32[512, 2048]"; primals_274: "f32[512]"; primals_275: "f32[512, 1, 7, 7]"; primals_276: "f32[512]"; primals_277: "f32[2048, 512]"; primals_278: "f32[2048]"; primals_279: "f32[512, 2048]"; primals_280: "f32[512]"; primals_281: "f32[512, 1, 7, 7]"; primals_282: "f32[512]"; primals_283: "f32[2048, 512]"; primals_284: "f32[2048]"; primals_285: "f32[512, 2048]"; primals_286: "f32[512]"; primals_287: "f32[512, 1, 7, 7]"; primals_288: "f32[512]"; primals_289: "f32[2048, 512]"; primals_290: "f32[2048]"; primals_291: "f32[512, 2048]"; primals_292: "f32[512]"; primals_293: "f32[512, 1, 7, 7]"; primals_294: "f32[512]"; primals_295: "f32[2048, 512]"; primals_296: "f32[2048]"; primals_297: "f32[512, 2048]"; primals_298: "f32[512]"; primals_299: "f32[512, 1, 7, 7]"; primals_300: "f32[512]"; primals_301: "f32[2048, 512]"; primals_302: "f32[2048]"; primals_303: "f32[512, 2048]"; primals_304: "f32[512]"; primals_305: "f32[512, 1, 7, 7]"; primals_306: "f32[512]"; primals_307: "f32[2048, 512]"; primals_308: "f32[2048]"; primals_309: "f32[512, 2048]"; primals_310: "f32[512]"; primals_311: "f32[512, 1, 7, 7]"; primals_312: "f32[512]"; primals_313: "f32[2048, 512]"; primals_314: "f32[2048]"; primals_315: "f32[512, 2048]"; primals_316: "f32[512]"; primals_317: "f32[512, 1, 7, 7]"; primals_318: "f32[512]"; primals_319: "f32[2048, 512]"; primals_320: "f32[2048]"; primals_321: "f32[512, 2048]"; primals_322: "f32[512]"; primals_323: "f32[1024, 512, 2, 2]"; primals_324: "f32[1024]"; primals_325: "f32[1024, 1, 7, 7]"; primals_326: "f32[1024]"; primals_327: "f32[4096, 1024]"; primals_328: "f32[4096]"; primals_329: "f32[1024, 4096]"; primals_330: "f32[1024]"; primals_331: "f32[1024, 1, 7, 7]"; primals_332: "f32[1024]"; primals_333: "f32[4096, 1024]"; primals_334: "f32[4096]"; primals_335: "f32[1024, 4096]"; primals_336: "f32[1024]"; primals_337: "f32[1024, 1, 7, 7]"; primals_338: "f32[1024]"; primals_339: "f32[4096, 1024]"; primals_340: "f32[4096]"; primals_341: "f32[1024, 4096]"; primals_342: "f32[1024]"; primals_343: "f32[1000, 1024]"; primals_344: "f32[1000]"; primals_345: "f32[8, 3, 224, 224]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:411, code: x = self.stem(x)
    convolution: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(primals_345, primals_119, primals_120, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  primals_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    permute: "f32[8, 56, 56, 128]" = torch.ops.aten.permute.default(convolution, [0, 2, 3, 1]);  convolution = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone: "f32[8, 56, 56, 128]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format)
    var_mean = torch.ops.aten.var_mean.correction(clone, [3], correction = 0, keepdim = True)
    getitem: "f32[8, 56, 56, 1]" = var_mean[0]
    getitem_1: "f32[8, 56, 56, 1]" = var_mean[1];  var_mean = None
    add: "f32[8, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
    rsqrt: "f32[8, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
    sub: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(clone, getitem_1);  clone = None
    mul: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul, primals_1);  mul = None
    add_1: "f32[8, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_1, primals_2);  mul_1 = primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    permute_1: "f32[8, 128, 56, 56]" = torch.ops.aten.permute.default(add_1, [0, 3, 1, 2]);  add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_1: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(permute_1, primals_121, primals_122, [1, 1], [3, 3], [1, 1], False, [0, 0], 128);  primals_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_2: "f32[8, 56, 56, 128]" = torch.ops.aten.permute.default(convolution_1, [0, 2, 3, 1]);  convolution_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_1: "f32[8, 56, 56, 128]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format)
    var_mean_1 = torch.ops.aten.var_mean.correction(clone_1, [3], correction = 0, keepdim = True)
    getitem_2: "f32[8, 56, 56, 1]" = var_mean_1[0]
    getitem_3: "f32[8, 56, 56, 1]" = var_mean_1[1];  var_mean_1 = None
    add_2: "f32[8, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-06);  getitem_2 = None
    rsqrt_1: "f32[8, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
    sub_1: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(clone_1, getitem_3);  clone_1 = None
    mul_2: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    mul_3: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_2, primals_3);  mul_2 = None
    add_3: "f32[8, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_3, primals_4);  mul_3 = primals_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view: "f32[25088, 128]" = torch.ops.aten.view.default(add_3, [25088, 128]);  add_3 = None
    permute_3: "f32[128, 512]" = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
    addmm: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_124, view, permute_3);  primals_124 = None
    view_1: "f32[8, 56, 56, 512]" = torch.ops.aten.view.default(addmm, [8, 56, 56, 512]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_4: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_1, 0.5)
    mul_5: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_1, 0.7071067811865476)
    erf: "f32[8, 56, 56, 512]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
    add_4: "f32[8, 56, 56, 512]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_6: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(mul_4, add_4);  mul_4 = add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_2: "f32[8, 56, 56, 512]" = torch.ops.aten.clone.default(mul_6);  mul_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_2: "f32[25088, 512]" = torch.ops.aten.view.default(clone_2, [25088, 512]);  clone_2 = None
    permute_4: "f32[512, 128]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
    addmm_1: "f32[25088, 128]" = torch.ops.aten.addmm.default(primals_126, view_2, permute_4);  primals_126 = None
    view_3: "f32[8, 56, 56, 128]" = torch.ops.aten.view.default(addmm_1, [8, 56, 56, 128]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_3: "f32[8, 56, 56, 128]" = torch.ops.aten.clone.default(view_3);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_5: "f32[8, 128, 56, 56]" = torch.ops.aten.permute.default(clone_3, [0, 3, 1, 2]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_4: "f32[1, 128, 1, 1]" = torch.ops.aten.view.default(primals_5, [1, -1, 1, 1]);  primals_5 = None
    mul_7: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(permute_5, view_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_5: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_7, permute_1);  mul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_2: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(add_5, primals_127, primals_128, [1, 1], [3, 3], [1, 1], False, [0, 0], 128);  primals_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_6: "f32[8, 56, 56, 128]" = torch.ops.aten.permute.default(convolution_2, [0, 2, 3, 1]);  convolution_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_4: "f32[8, 56, 56, 128]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format)
    var_mean_2 = torch.ops.aten.var_mean.correction(clone_4, [3], correction = 0, keepdim = True)
    getitem_4: "f32[8, 56, 56, 1]" = var_mean_2[0]
    getitem_5: "f32[8, 56, 56, 1]" = var_mean_2[1];  var_mean_2 = None
    add_6: "f32[8, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-06);  getitem_4 = None
    rsqrt_2: "f32[8, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_2: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(clone_4, getitem_5);  clone_4 = None
    mul_8: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    mul_9: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_8, primals_6);  mul_8 = None
    add_7: "f32[8, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_9, primals_7);  mul_9 = primals_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_5: "f32[25088, 128]" = torch.ops.aten.view.default(add_7, [25088, 128]);  add_7 = None
    permute_7: "f32[128, 512]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
    addmm_2: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_130, view_5, permute_7);  primals_130 = None
    view_6: "f32[8, 56, 56, 512]" = torch.ops.aten.view.default(addmm_2, [8, 56, 56, 512]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_10: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_6, 0.5)
    mul_11: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_6, 0.7071067811865476)
    erf_1: "f32[8, 56, 56, 512]" = torch.ops.aten.erf.default(mul_11);  mul_11 = None
    add_8: "f32[8, 56, 56, 512]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_12: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(mul_10, add_8);  mul_10 = add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_5: "f32[8, 56, 56, 512]" = torch.ops.aten.clone.default(mul_12);  mul_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_7: "f32[25088, 512]" = torch.ops.aten.view.default(clone_5, [25088, 512]);  clone_5 = None
    permute_8: "f32[512, 128]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
    addmm_3: "f32[25088, 128]" = torch.ops.aten.addmm.default(primals_132, view_7, permute_8);  primals_132 = None
    view_8: "f32[8, 56, 56, 128]" = torch.ops.aten.view.default(addmm_3, [8, 56, 56, 128]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_6: "f32[8, 56, 56, 128]" = torch.ops.aten.clone.default(view_8);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_9: "f32[8, 128, 56, 56]" = torch.ops.aten.permute.default(clone_6, [0, 3, 1, 2]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_9: "f32[1, 128, 1, 1]" = torch.ops.aten.view.default(primals_8, [1, -1, 1, 1]);  primals_8 = None
    mul_13: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(permute_9, view_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_9: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_13, add_5);  mul_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_3: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(add_9, primals_133, primals_134, [1, 1], [3, 3], [1, 1], False, [0, 0], 128);  primals_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_10: "f32[8, 56, 56, 128]" = torch.ops.aten.permute.default(convolution_3, [0, 2, 3, 1]);  convolution_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_7: "f32[8, 56, 56, 128]" = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format)
    var_mean_3 = torch.ops.aten.var_mean.correction(clone_7, [3], correction = 0, keepdim = True)
    getitem_6: "f32[8, 56, 56, 1]" = var_mean_3[0]
    getitem_7: "f32[8, 56, 56, 1]" = var_mean_3[1];  var_mean_3 = None
    add_10: "f32[8, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-06);  getitem_6 = None
    rsqrt_3: "f32[8, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
    sub_3: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(clone_7, getitem_7);  clone_7 = None
    mul_14: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    mul_15: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_14, primals_9);  mul_14 = None
    add_11: "f32[8, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_15, primals_10);  mul_15 = primals_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_10: "f32[25088, 128]" = torch.ops.aten.view.default(add_11, [25088, 128]);  add_11 = None
    permute_11: "f32[128, 512]" = torch.ops.aten.permute.default(primals_135, [1, 0]);  primals_135 = None
    addmm_4: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_136, view_10, permute_11);  primals_136 = None
    view_11: "f32[8, 56, 56, 512]" = torch.ops.aten.view.default(addmm_4, [8, 56, 56, 512]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_16: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_11, 0.5)
    mul_17: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_11, 0.7071067811865476)
    erf_2: "f32[8, 56, 56, 512]" = torch.ops.aten.erf.default(mul_17);  mul_17 = None
    add_12: "f32[8, 56, 56, 512]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_18: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(mul_16, add_12);  mul_16 = add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_8: "f32[8, 56, 56, 512]" = torch.ops.aten.clone.default(mul_18);  mul_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_12: "f32[25088, 512]" = torch.ops.aten.view.default(clone_8, [25088, 512]);  clone_8 = None
    permute_12: "f32[512, 128]" = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
    addmm_5: "f32[25088, 128]" = torch.ops.aten.addmm.default(primals_138, view_12, permute_12);  primals_138 = None
    view_13: "f32[8, 56, 56, 128]" = torch.ops.aten.view.default(addmm_5, [8, 56, 56, 128]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_9: "f32[8, 56, 56, 128]" = torch.ops.aten.clone.default(view_13);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_13: "f32[8, 128, 56, 56]" = torch.ops.aten.permute.default(clone_9, [0, 3, 1, 2]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_14: "f32[1, 128, 1, 1]" = torch.ops.aten.view.default(primals_11, [1, -1, 1, 1]);  primals_11 = None
    mul_19: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(permute_13, view_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_13: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_19, add_9);  mul_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    permute_14: "f32[8, 56, 56, 128]" = torch.ops.aten.permute.default(add_13, [0, 2, 3, 1]);  add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_4 = torch.ops.aten.var_mean.correction(permute_14, [3], correction = 0, keepdim = True)
    getitem_8: "f32[8, 56, 56, 1]" = var_mean_4[0]
    getitem_9: "f32[8, 56, 56, 1]" = var_mean_4[1];  var_mean_4 = None
    add_14: "f32[8, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-06);  getitem_8 = None
    rsqrt_4: "f32[8, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    sub_4: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(permute_14, getitem_9)
    mul_20: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    mul_21: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_20, primals_12);  mul_20 = None
    add_15: "f32[8, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_21, primals_13);  mul_21 = primals_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    permute_15: "f32[8, 128, 56, 56]" = torch.ops.aten.permute.default(add_15, [0, 3, 1, 2]);  add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:229, code: x = self.downsample(x)
    convolution_4: "f32[8, 256, 28, 28]" = torch.ops.aten.convolution.default(permute_15, primals_139, primals_140, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_5: "f32[8, 256, 28, 28]" = torch.ops.aten.convolution.default(convolution_4, primals_141, primals_142, [1, 1], [3, 3], [1, 1], False, [0, 0], 256);  primals_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_16: "f32[8, 28, 28, 256]" = torch.ops.aten.permute.default(convolution_5, [0, 2, 3, 1]);  convolution_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_10: "f32[8, 28, 28, 256]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format)
    var_mean_5 = torch.ops.aten.var_mean.correction(clone_10, [3], correction = 0, keepdim = True)
    getitem_10: "f32[8, 28, 28, 1]" = var_mean_5[0]
    getitem_11: "f32[8, 28, 28, 1]" = var_mean_5[1];  var_mean_5 = None
    add_16: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-06);  getitem_10 = None
    rsqrt_5: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_5: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(clone_10, getitem_11);  clone_10 = None
    mul_22: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    mul_23: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_22, primals_14);  mul_22 = None
    add_17: "f32[8, 28, 28, 256]" = torch.ops.aten.add.Tensor(mul_23, primals_15);  mul_23 = primals_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_15: "f32[6272, 256]" = torch.ops.aten.view.default(add_17, [6272, 256]);  add_17 = None
    permute_17: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
    addmm_6: "f32[6272, 1024]" = torch.ops.aten.addmm.default(primals_144, view_15, permute_17);  primals_144 = None
    view_16: "f32[8, 28, 28, 1024]" = torch.ops.aten.view.default(addmm_6, [8, 28, 28, 1024]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_24: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_16, 0.5)
    mul_25: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_16, 0.7071067811865476)
    erf_3: "f32[8, 28, 28, 1024]" = torch.ops.aten.erf.default(mul_25);  mul_25 = None
    add_18: "f32[8, 28, 28, 1024]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_26: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(mul_24, add_18);  mul_24 = add_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_11: "f32[8, 28, 28, 1024]" = torch.ops.aten.clone.default(mul_26);  mul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_17: "f32[6272, 1024]" = torch.ops.aten.view.default(clone_11, [6272, 1024]);  clone_11 = None
    permute_18: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
    addmm_7: "f32[6272, 256]" = torch.ops.aten.addmm.default(primals_146, view_17, permute_18);  primals_146 = None
    view_18: "f32[8, 28, 28, 256]" = torch.ops.aten.view.default(addmm_7, [8, 28, 28, 256]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_12: "f32[8, 28, 28, 256]" = torch.ops.aten.clone.default(view_18);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_19: "f32[8, 256, 28, 28]" = torch.ops.aten.permute.default(clone_12, [0, 3, 1, 2]);  clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_19: "f32[1, 256, 1, 1]" = torch.ops.aten.view.default(primals_16, [1, -1, 1, 1]);  primals_16 = None
    mul_27: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(permute_19, view_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_19: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(mul_27, convolution_4);  mul_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_6: "f32[8, 256, 28, 28]" = torch.ops.aten.convolution.default(add_19, primals_147, primals_148, [1, 1], [3, 3], [1, 1], False, [0, 0], 256);  primals_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_20: "f32[8, 28, 28, 256]" = torch.ops.aten.permute.default(convolution_6, [0, 2, 3, 1]);  convolution_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_13: "f32[8, 28, 28, 256]" = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format)
    var_mean_6 = torch.ops.aten.var_mean.correction(clone_13, [3], correction = 0, keepdim = True)
    getitem_12: "f32[8, 28, 28, 1]" = var_mean_6[0]
    getitem_13: "f32[8, 28, 28, 1]" = var_mean_6[1];  var_mean_6 = None
    add_20: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
    rsqrt_6: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
    sub_6: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(clone_13, getitem_13);  clone_13 = None
    mul_28: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    mul_29: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_28, primals_17);  mul_28 = None
    add_21: "f32[8, 28, 28, 256]" = torch.ops.aten.add.Tensor(mul_29, primals_18);  mul_29 = primals_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_20: "f32[6272, 256]" = torch.ops.aten.view.default(add_21, [6272, 256]);  add_21 = None
    permute_21: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_149, [1, 0]);  primals_149 = None
    addmm_8: "f32[6272, 1024]" = torch.ops.aten.addmm.default(primals_150, view_20, permute_21);  primals_150 = None
    view_21: "f32[8, 28, 28, 1024]" = torch.ops.aten.view.default(addmm_8, [8, 28, 28, 1024]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_30: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_21, 0.5)
    mul_31: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_21, 0.7071067811865476)
    erf_4: "f32[8, 28, 28, 1024]" = torch.ops.aten.erf.default(mul_31);  mul_31 = None
    add_22: "f32[8, 28, 28, 1024]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_32: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(mul_30, add_22);  mul_30 = add_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_14: "f32[8, 28, 28, 1024]" = torch.ops.aten.clone.default(mul_32);  mul_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_22: "f32[6272, 1024]" = torch.ops.aten.view.default(clone_14, [6272, 1024]);  clone_14 = None
    permute_22: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_151, [1, 0]);  primals_151 = None
    addmm_9: "f32[6272, 256]" = torch.ops.aten.addmm.default(primals_152, view_22, permute_22);  primals_152 = None
    view_23: "f32[8, 28, 28, 256]" = torch.ops.aten.view.default(addmm_9, [8, 28, 28, 256]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_15: "f32[8, 28, 28, 256]" = torch.ops.aten.clone.default(view_23);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_23: "f32[8, 256, 28, 28]" = torch.ops.aten.permute.default(clone_15, [0, 3, 1, 2]);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_24: "f32[1, 256, 1, 1]" = torch.ops.aten.view.default(primals_19, [1, -1, 1, 1]);  primals_19 = None
    mul_33: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(permute_23, view_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_23: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(mul_33, add_19);  mul_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_7: "f32[8, 256, 28, 28]" = torch.ops.aten.convolution.default(add_23, primals_153, primals_154, [1, 1], [3, 3], [1, 1], False, [0, 0], 256);  primals_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_24: "f32[8, 28, 28, 256]" = torch.ops.aten.permute.default(convolution_7, [0, 2, 3, 1]);  convolution_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_16: "f32[8, 28, 28, 256]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format)
    var_mean_7 = torch.ops.aten.var_mean.correction(clone_16, [3], correction = 0, keepdim = True)
    getitem_14: "f32[8, 28, 28, 1]" = var_mean_7[0]
    getitem_15: "f32[8, 28, 28, 1]" = var_mean_7[1];  var_mean_7 = None
    add_24: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
    rsqrt_7: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    sub_7: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(clone_16, getitem_15);  clone_16 = None
    mul_34: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    mul_35: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_34, primals_20);  mul_34 = None
    add_25: "f32[8, 28, 28, 256]" = torch.ops.aten.add.Tensor(mul_35, primals_21);  mul_35 = primals_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_25: "f32[6272, 256]" = torch.ops.aten.view.default(add_25, [6272, 256]);  add_25 = None
    permute_25: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
    addmm_10: "f32[6272, 1024]" = torch.ops.aten.addmm.default(primals_156, view_25, permute_25);  primals_156 = None
    view_26: "f32[8, 28, 28, 1024]" = torch.ops.aten.view.default(addmm_10, [8, 28, 28, 1024]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_36: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_26, 0.5)
    mul_37: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_26, 0.7071067811865476)
    erf_5: "f32[8, 28, 28, 1024]" = torch.ops.aten.erf.default(mul_37);  mul_37 = None
    add_26: "f32[8, 28, 28, 1024]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_38: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(mul_36, add_26);  mul_36 = add_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_17: "f32[8, 28, 28, 1024]" = torch.ops.aten.clone.default(mul_38);  mul_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_27: "f32[6272, 1024]" = torch.ops.aten.view.default(clone_17, [6272, 1024]);  clone_17 = None
    permute_26: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_157, [1, 0]);  primals_157 = None
    addmm_11: "f32[6272, 256]" = torch.ops.aten.addmm.default(primals_158, view_27, permute_26);  primals_158 = None
    view_28: "f32[8, 28, 28, 256]" = torch.ops.aten.view.default(addmm_11, [8, 28, 28, 256]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_18: "f32[8, 28, 28, 256]" = torch.ops.aten.clone.default(view_28);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_27: "f32[8, 256, 28, 28]" = torch.ops.aten.permute.default(clone_18, [0, 3, 1, 2]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_29: "f32[1, 256, 1, 1]" = torch.ops.aten.view.default(primals_22, [1, -1, 1, 1]);  primals_22 = None
    mul_39: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(permute_27, view_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_27: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(mul_39, add_23);  mul_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    permute_28: "f32[8, 28, 28, 256]" = torch.ops.aten.permute.default(add_27, [0, 2, 3, 1]);  add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_8 = torch.ops.aten.var_mean.correction(permute_28, [3], correction = 0, keepdim = True)
    getitem_16: "f32[8, 28, 28, 1]" = var_mean_8[0]
    getitem_17: "f32[8, 28, 28, 1]" = var_mean_8[1];  var_mean_8 = None
    add_28: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-06);  getitem_16 = None
    rsqrt_8: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_8: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(permute_28, getitem_17)
    mul_40: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    mul_41: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_40, primals_23);  mul_40 = None
    add_29: "f32[8, 28, 28, 256]" = torch.ops.aten.add.Tensor(mul_41, primals_24);  mul_41 = primals_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    permute_29: "f32[8, 256, 28, 28]" = torch.ops.aten.permute.default(add_29, [0, 3, 1, 2]);  add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:229, code: x = self.downsample(x)
    convolution_8: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(permute_29, primals_159, primals_160, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_9: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(convolution_8, primals_161, primals_162, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_30: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_9, [0, 2, 3, 1]);  convolution_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_19: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format)
    var_mean_9 = torch.ops.aten.var_mean.correction(clone_19, [3], correction = 0, keepdim = True)
    getitem_18: "f32[8, 14, 14, 1]" = var_mean_9[0]
    getitem_19: "f32[8, 14, 14, 1]" = var_mean_9[1];  var_mean_9 = None
    add_30: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-06);  getitem_18 = None
    rsqrt_9: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
    sub_9: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_19, getitem_19);  clone_19 = None
    mul_42: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    mul_43: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_42, primals_25);  mul_42 = None
    add_31: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_43, primals_26);  mul_43 = primals_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_30: "f32[1568, 512]" = torch.ops.aten.view.default(add_31, [1568, 512]);  add_31 = None
    permute_31: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_163, [1, 0]);  primals_163 = None
    addmm_12: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_164, view_30, permute_31);  primals_164 = None
    view_31: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_12, [8, 14, 14, 2048]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_44: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_31, 0.5)
    mul_45: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_31, 0.7071067811865476)
    erf_6: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_45);  mul_45 = None
    add_32: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_46: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_44, add_32);  mul_44 = add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_20: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_46);  mul_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_32: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_20, [1568, 2048]);  clone_20 = None
    permute_32: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_165, [1, 0]);  primals_165 = None
    addmm_13: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_166, view_32, permute_32);  primals_166 = None
    view_33: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_13, [8, 14, 14, 512]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_21: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_33);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_33: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_21, [0, 3, 1, 2]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_34: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_27, [1, -1, 1, 1]);  primals_27 = None
    mul_47: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_33, view_34)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_33: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_47, convolution_8);  mul_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_10: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_33, primals_167, primals_168, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_34: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_10, [0, 2, 3, 1]);  convolution_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_22: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format)
    var_mean_10 = torch.ops.aten.var_mean.correction(clone_22, [3], correction = 0, keepdim = True)
    getitem_20: "f32[8, 14, 14, 1]" = var_mean_10[0]
    getitem_21: "f32[8, 14, 14, 1]" = var_mean_10[1];  var_mean_10 = None
    add_34: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
    rsqrt_10: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
    sub_10: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_22, getitem_21);  clone_22 = None
    mul_48: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    mul_49: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_48, primals_28);  mul_48 = None
    add_35: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_49, primals_29);  mul_49 = primals_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_35: "f32[1568, 512]" = torch.ops.aten.view.default(add_35, [1568, 512]);  add_35 = None
    permute_35: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_169, [1, 0]);  primals_169 = None
    addmm_14: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_170, view_35, permute_35);  primals_170 = None
    view_36: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_14, [8, 14, 14, 2048]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_50: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_36, 0.5)
    mul_51: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_36, 0.7071067811865476)
    erf_7: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
    add_36: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_52: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_50, add_36);  mul_50 = add_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_23: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_52);  mul_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_37: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_23, [1568, 2048]);  clone_23 = None
    permute_36: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
    addmm_15: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_172, view_37, permute_36);  primals_172 = None
    view_38: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_15, [8, 14, 14, 512]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_24: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_38);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_37: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_24, [0, 3, 1, 2]);  clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_39: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_30, [1, -1, 1, 1]);  primals_30 = None
    mul_53: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_37, view_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_37: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_53, add_33);  mul_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_11: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_37, primals_173, primals_174, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_38: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_11, [0, 2, 3, 1]);  convolution_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_25: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format)
    var_mean_11 = torch.ops.aten.var_mean.correction(clone_25, [3], correction = 0, keepdim = True)
    getitem_22: "f32[8, 14, 14, 1]" = var_mean_11[0]
    getitem_23: "f32[8, 14, 14, 1]" = var_mean_11[1];  var_mean_11 = None
    add_38: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-06);  getitem_22 = None
    rsqrt_11: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_11: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_25, getitem_23);  clone_25 = None
    mul_54: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    mul_55: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_54, primals_31);  mul_54 = None
    add_39: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_55, primals_32);  mul_55 = primals_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_40: "f32[1568, 512]" = torch.ops.aten.view.default(add_39, [1568, 512]);  add_39 = None
    permute_39: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_175, [1, 0]);  primals_175 = None
    addmm_16: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_176, view_40, permute_39);  primals_176 = None
    view_41: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_16, [8, 14, 14, 2048]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_56: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_41, 0.5)
    mul_57: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476)
    erf_8: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_57);  mul_57 = None
    add_40: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_58: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_56, add_40);  mul_56 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_26: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_58);  mul_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_42: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_26, [1568, 2048]);  clone_26 = None
    permute_40: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_177, [1, 0]);  primals_177 = None
    addmm_17: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_178, view_42, permute_40);  primals_178 = None
    view_43: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_17, [8, 14, 14, 512]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_27: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_43);  view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_41: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_27, [0, 3, 1, 2]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_44: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_33, [1, -1, 1, 1]);  primals_33 = None
    mul_59: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_41, view_44)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_41: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_59, add_37);  mul_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_12: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_41, primals_179, primals_180, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_42: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_12, [0, 2, 3, 1]);  convolution_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_28: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_42, memory_format = torch.contiguous_format)
    var_mean_12 = torch.ops.aten.var_mean.correction(clone_28, [3], correction = 0, keepdim = True)
    getitem_24: "f32[8, 14, 14, 1]" = var_mean_12[0]
    getitem_25: "f32[8, 14, 14, 1]" = var_mean_12[1];  var_mean_12 = None
    add_42: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-06);  getitem_24 = None
    rsqrt_12: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_12: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_28, getitem_25);  clone_28 = None
    mul_60: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    mul_61: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_60, primals_34);  mul_60 = None
    add_43: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_61, primals_35);  mul_61 = primals_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_45: "f32[1568, 512]" = torch.ops.aten.view.default(add_43, [1568, 512]);  add_43 = None
    permute_43: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_181, [1, 0]);  primals_181 = None
    addmm_18: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_182, view_45, permute_43);  primals_182 = None
    view_46: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_18, [8, 14, 14, 2048]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_62: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_46, 0.5)
    mul_63: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_46, 0.7071067811865476)
    erf_9: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
    add_44: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_64: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_62, add_44);  mul_62 = add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_29: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_64);  mul_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_47: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_29, [1568, 2048]);  clone_29 = None
    permute_44: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_183, [1, 0]);  primals_183 = None
    addmm_19: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_184, view_47, permute_44);  primals_184 = None
    view_48: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_19, [8, 14, 14, 512]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_30: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_48);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_45: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_30, [0, 3, 1, 2]);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_49: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_36, [1, -1, 1, 1]);  primals_36 = None
    mul_65: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_45, view_49)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_45: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_65, add_41);  mul_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_13: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_45, primals_185, primals_186, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_46: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_13, [0, 2, 3, 1]);  convolution_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_31: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format)
    var_mean_13 = torch.ops.aten.var_mean.correction(clone_31, [3], correction = 0, keepdim = True)
    getitem_26: "f32[8, 14, 14, 1]" = var_mean_13[0]
    getitem_27: "f32[8, 14, 14, 1]" = var_mean_13[1];  var_mean_13 = None
    add_46: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-06);  getitem_26 = None
    rsqrt_13: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    sub_13: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_31, getitem_27);  clone_31 = None
    mul_66: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    mul_67: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_66, primals_37);  mul_66 = None
    add_47: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_67, primals_38);  mul_67 = primals_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_50: "f32[1568, 512]" = torch.ops.aten.view.default(add_47, [1568, 512]);  add_47 = None
    permute_47: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_187, [1, 0]);  primals_187 = None
    addmm_20: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_188, view_50, permute_47);  primals_188 = None
    view_51: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_20, [8, 14, 14, 2048]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_68: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_51, 0.5)
    mul_69: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_51, 0.7071067811865476)
    erf_10: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
    add_48: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_70: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_68, add_48);  mul_68 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_32: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_70);  mul_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_52: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_32, [1568, 2048]);  clone_32 = None
    permute_48: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_189, [1, 0]);  primals_189 = None
    addmm_21: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_190, view_52, permute_48);  primals_190 = None
    view_53: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_21, [8, 14, 14, 512]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_33: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_53);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_49: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_33, [0, 3, 1, 2]);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_54: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_39, [1, -1, 1, 1]);  primals_39 = None
    mul_71: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_49, view_54)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_49: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_71, add_45);  mul_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_14: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_49, primals_191, primals_192, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_50: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_14, [0, 2, 3, 1]);  convolution_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_34: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_50, memory_format = torch.contiguous_format)
    var_mean_14 = torch.ops.aten.var_mean.correction(clone_34, [3], correction = 0, keepdim = True)
    getitem_28: "f32[8, 14, 14, 1]" = var_mean_14[0]
    getitem_29: "f32[8, 14, 14, 1]" = var_mean_14[1];  var_mean_14 = None
    add_50: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-06);  getitem_28 = None
    rsqrt_14: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    sub_14: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_34, getitem_29);  clone_34 = None
    mul_72: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    mul_73: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_72, primals_40);  mul_72 = None
    add_51: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_73, primals_41);  mul_73 = primals_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_55: "f32[1568, 512]" = torch.ops.aten.view.default(add_51, [1568, 512]);  add_51 = None
    permute_51: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_193, [1, 0]);  primals_193 = None
    addmm_22: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_194, view_55, permute_51);  primals_194 = None
    view_56: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_22, [8, 14, 14, 2048]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_74: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_56, 0.5)
    mul_75: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_56, 0.7071067811865476)
    erf_11: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
    add_52: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_76: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_74, add_52);  mul_74 = add_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_35: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_76);  mul_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_57: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_35, [1568, 2048]);  clone_35 = None
    permute_52: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_195, [1, 0]);  primals_195 = None
    addmm_23: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_196, view_57, permute_52);  primals_196 = None
    view_58: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_23, [8, 14, 14, 512]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_36: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_58);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_53: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_36, [0, 3, 1, 2]);  clone_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_59: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_42, [1, -1, 1, 1]);  primals_42 = None
    mul_77: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_53, view_59)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_53: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_77, add_49);  mul_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_15: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_53, primals_197, primals_198, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_54: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_15, [0, 2, 3, 1]);  convolution_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_37: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_54, memory_format = torch.contiguous_format)
    var_mean_15 = torch.ops.aten.var_mean.correction(clone_37, [3], correction = 0, keepdim = True)
    getitem_30: "f32[8, 14, 14, 1]" = var_mean_15[0]
    getitem_31: "f32[8, 14, 14, 1]" = var_mean_15[1];  var_mean_15 = None
    add_54: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
    rsqrt_15: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
    sub_15: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_37, getitem_31);  clone_37 = None
    mul_78: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    mul_79: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_78, primals_43);  mul_78 = None
    add_55: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_79, primals_44);  mul_79 = primals_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_60: "f32[1568, 512]" = torch.ops.aten.view.default(add_55, [1568, 512]);  add_55 = None
    permute_55: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_199, [1, 0]);  primals_199 = None
    addmm_24: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_200, view_60, permute_55);  primals_200 = None
    view_61: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_24, [8, 14, 14, 2048]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_80: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_61, 0.5)
    mul_81: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_61, 0.7071067811865476)
    erf_12: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_81);  mul_81 = None
    add_56: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_82: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_80, add_56);  mul_80 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_38: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_82);  mul_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_62: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_38, [1568, 2048]);  clone_38 = None
    permute_56: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_201, [1, 0]);  primals_201 = None
    addmm_25: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_202, view_62, permute_56);  primals_202 = None
    view_63: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_25, [8, 14, 14, 512]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_39: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_63);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_57: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_39, [0, 3, 1, 2]);  clone_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_64: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_45, [1, -1, 1, 1]);  primals_45 = None
    mul_83: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_57, view_64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_57: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_83, add_53);  mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_16: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_57, primals_203, primals_204, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_58: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_16, [0, 2, 3, 1]);  convolution_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_40: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_58, memory_format = torch.contiguous_format)
    var_mean_16 = torch.ops.aten.var_mean.correction(clone_40, [3], correction = 0, keepdim = True)
    getitem_32: "f32[8, 14, 14, 1]" = var_mean_16[0]
    getitem_33: "f32[8, 14, 14, 1]" = var_mean_16[1];  var_mean_16 = None
    add_58: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
    rsqrt_16: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_16: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_40, getitem_33);  clone_40 = None
    mul_84: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    mul_85: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_84, primals_46);  mul_84 = None
    add_59: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_85, primals_47);  mul_85 = primals_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_65: "f32[1568, 512]" = torch.ops.aten.view.default(add_59, [1568, 512]);  add_59 = None
    permute_59: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_205, [1, 0]);  primals_205 = None
    addmm_26: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_206, view_65, permute_59);  primals_206 = None
    view_66: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_26, [8, 14, 14, 2048]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_86: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_66, 0.5)
    mul_87: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_66, 0.7071067811865476)
    erf_13: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
    add_60: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_88: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_86, add_60);  mul_86 = add_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_41: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_88);  mul_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_67: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_41, [1568, 2048]);  clone_41 = None
    permute_60: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_207, [1, 0]);  primals_207 = None
    addmm_27: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_208, view_67, permute_60);  primals_208 = None
    view_68: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_27, [8, 14, 14, 512]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_42: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_68);  view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_61: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_42, [0, 3, 1, 2]);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_69: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_48, [1, -1, 1, 1]);  primals_48 = None
    mul_89: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_61, view_69)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_61: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_89, add_57);  mul_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_17: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_61, primals_209, primals_210, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_62: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_17, [0, 2, 3, 1]);  convolution_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_43: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format)
    var_mean_17 = torch.ops.aten.var_mean.correction(clone_43, [3], correction = 0, keepdim = True)
    getitem_34: "f32[8, 14, 14, 1]" = var_mean_17[0]
    getitem_35: "f32[8, 14, 14, 1]" = var_mean_17[1];  var_mean_17 = None
    add_62: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-06);  getitem_34 = None
    rsqrt_17: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    sub_17: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_43, getitem_35);  clone_43 = None
    mul_90: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    mul_91: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_90, primals_49);  mul_90 = None
    add_63: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_91, primals_50);  mul_91 = primals_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_70: "f32[1568, 512]" = torch.ops.aten.view.default(add_63, [1568, 512]);  add_63 = None
    permute_63: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_211, [1, 0]);  primals_211 = None
    addmm_28: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_212, view_70, permute_63);  primals_212 = None
    view_71: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_28, [8, 14, 14, 2048]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_92: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_71, 0.5)
    mul_93: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_71, 0.7071067811865476)
    erf_14: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_93);  mul_93 = None
    add_64: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_94: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_92, add_64);  mul_92 = add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_44: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_94);  mul_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_72: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_44, [1568, 2048]);  clone_44 = None
    permute_64: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_213, [1, 0]);  primals_213 = None
    addmm_29: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_214, view_72, permute_64);  primals_214 = None
    view_73: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_29, [8, 14, 14, 512]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_45: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_73);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_65: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_45, [0, 3, 1, 2]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_74: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_51, [1, -1, 1, 1]);  primals_51 = None
    mul_95: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_65, view_74)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_65: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_95, add_61);  mul_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_18: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_65, primals_215, primals_216, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_66: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_18, [0, 2, 3, 1]);  convolution_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_46: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_66, memory_format = torch.contiguous_format)
    var_mean_18 = torch.ops.aten.var_mean.correction(clone_46, [3], correction = 0, keepdim = True)
    getitem_36: "f32[8, 14, 14, 1]" = var_mean_18[0]
    getitem_37: "f32[8, 14, 14, 1]" = var_mean_18[1];  var_mean_18 = None
    add_66: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
    rsqrt_18: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_18: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_46, getitem_37);  clone_46 = None
    mul_96: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    mul_97: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_96, primals_52);  mul_96 = None
    add_67: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_97, primals_53);  mul_97 = primals_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_75: "f32[1568, 512]" = torch.ops.aten.view.default(add_67, [1568, 512]);  add_67 = None
    permute_67: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_217, [1, 0]);  primals_217 = None
    addmm_30: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_218, view_75, permute_67);  primals_218 = None
    view_76: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_30, [8, 14, 14, 2048]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_98: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_76, 0.5)
    mul_99: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_76, 0.7071067811865476)
    erf_15: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_99);  mul_99 = None
    add_68: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_100: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_98, add_68);  mul_98 = add_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_47: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_100);  mul_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_77: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_47, [1568, 2048]);  clone_47 = None
    permute_68: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_219, [1, 0]);  primals_219 = None
    addmm_31: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_220, view_77, permute_68);  primals_220 = None
    view_78: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_31, [8, 14, 14, 512]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_48: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_78);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_69: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_48, [0, 3, 1, 2]);  clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_79: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_54, [1, -1, 1, 1]);  primals_54 = None
    mul_101: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_69, view_79)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_69: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_101, add_65);  mul_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_19: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_69, primals_221, primals_222, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_70: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_19, [0, 2, 3, 1]);  convolution_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_49: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format)
    var_mean_19 = torch.ops.aten.var_mean.correction(clone_49, [3], correction = 0, keepdim = True)
    getitem_38: "f32[8, 14, 14, 1]" = var_mean_19[0]
    getitem_39: "f32[8, 14, 14, 1]" = var_mean_19[1];  var_mean_19 = None
    add_70: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-06);  getitem_38 = None
    rsqrt_19: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    sub_19: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_49, getitem_39);  clone_49 = None
    mul_102: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    mul_103: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_102, primals_55);  mul_102 = None
    add_71: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_103, primals_56);  mul_103 = primals_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_80: "f32[1568, 512]" = torch.ops.aten.view.default(add_71, [1568, 512]);  add_71 = None
    permute_71: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_223, [1, 0]);  primals_223 = None
    addmm_32: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_224, view_80, permute_71);  primals_224 = None
    view_81: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_32, [8, 14, 14, 2048]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_104: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_81, 0.5)
    mul_105: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_81, 0.7071067811865476)
    erf_16: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_105);  mul_105 = None
    add_72: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_106: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_104, add_72);  mul_104 = add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_50: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_106);  mul_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_82: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_50, [1568, 2048]);  clone_50 = None
    permute_72: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_225, [1, 0]);  primals_225 = None
    addmm_33: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_226, view_82, permute_72);  primals_226 = None
    view_83: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_33, [8, 14, 14, 512]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_51: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_83);  view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_73: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_51, [0, 3, 1, 2]);  clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_84: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_57, [1, -1, 1, 1]);  primals_57 = None
    mul_107: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_73, view_84)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_73: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_107, add_69);  mul_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_20: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_73, primals_227, primals_228, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_74: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_20, [0, 2, 3, 1]);  convolution_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_52: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format)
    var_mean_20 = torch.ops.aten.var_mean.correction(clone_52, [3], correction = 0, keepdim = True)
    getitem_40: "f32[8, 14, 14, 1]" = var_mean_20[0]
    getitem_41: "f32[8, 14, 14, 1]" = var_mean_20[1];  var_mean_20 = None
    add_74: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-06);  getitem_40 = None
    rsqrt_20: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_20: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_52, getitem_41);  clone_52 = None
    mul_108: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    mul_109: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_108, primals_58);  mul_108 = None
    add_75: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_109, primals_59);  mul_109 = primals_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_85: "f32[1568, 512]" = torch.ops.aten.view.default(add_75, [1568, 512]);  add_75 = None
    permute_75: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_229, [1, 0]);  primals_229 = None
    addmm_34: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_230, view_85, permute_75);  primals_230 = None
    view_86: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_34, [8, 14, 14, 2048]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_110: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_86, 0.5)
    mul_111: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_86, 0.7071067811865476)
    erf_17: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_111);  mul_111 = None
    add_76: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_112: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_110, add_76);  mul_110 = add_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_53: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_112);  mul_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_87: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_53, [1568, 2048]);  clone_53 = None
    permute_76: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_231, [1, 0]);  primals_231 = None
    addmm_35: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_232, view_87, permute_76);  primals_232 = None
    view_88: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_35, [8, 14, 14, 512]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_54: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_88);  view_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_77: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_54, [0, 3, 1, 2]);  clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_89: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_60, [1, -1, 1, 1]);  primals_60 = None
    mul_113: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_77, view_89)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_77: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_113, add_73);  mul_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_21: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_77, primals_233, primals_234, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_78: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_21, [0, 2, 3, 1]);  convolution_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_55: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_78, memory_format = torch.contiguous_format)
    var_mean_21 = torch.ops.aten.var_mean.correction(clone_55, [3], correction = 0, keepdim = True)
    getitem_42: "f32[8, 14, 14, 1]" = var_mean_21[0]
    getitem_43: "f32[8, 14, 14, 1]" = var_mean_21[1];  var_mean_21 = None
    add_78: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-06);  getitem_42 = None
    rsqrt_21: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_21: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_55, getitem_43);  clone_55 = None
    mul_114: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    mul_115: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_114, primals_61);  mul_114 = None
    add_79: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_115, primals_62);  mul_115 = primals_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_90: "f32[1568, 512]" = torch.ops.aten.view.default(add_79, [1568, 512]);  add_79 = None
    permute_79: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_235, [1, 0]);  primals_235 = None
    addmm_36: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_236, view_90, permute_79);  primals_236 = None
    view_91: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_36, [8, 14, 14, 2048]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_116: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_91, 0.5)
    mul_117: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_91, 0.7071067811865476)
    erf_18: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_117);  mul_117 = None
    add_80: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_118: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_116, add_80);  mul_116 = add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_56: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_118);  mul_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_92: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_56, [1568, 2048]);  clone_56 = None
    permute_80: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_237, [1, 0]);  primals_237 = None
    addmm_37: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_238, view_92, permute_80);  primals_238 = None
    view_93: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_37, [8, 14, 14, 512]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_57: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_93);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_81: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_57, [0, 3, 1, 2]);  clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_94: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_63, [1, -1, 1, 1]);  primals_63 = None
    mul_119: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_81, view_94)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_81: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_119, add_77);  mul_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_22: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_81, primals_239, primals_240, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_82: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_22, [0, 2, 3, 1]);  convolution_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_58: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format)
    var_mean_22 = torch.ops.aten.var_mean.correction(clone_58, [3], correction = 0, keepdim = True)
    getitem_44: "f32[8, 14, 14, 1]" = var_mean_22[0]
    getitem_45: "f32[8, 14, 14, 1]" = var_mean_22[1];  var_mean_22 = None
    add_82: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-06);  getitem_44 = None
    rsqrt_22: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    sub_22: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_58, getitem_45);  clone_58 = None
    mul_120: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    mul_121: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_120, primals_64);  mul_120 = None
    add_83: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_121, primals_65);  mul_121 = primals_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_95: "f32[1568, 512]" = torch.ops.aten.view.default(add_83, [1568, 512]);  add_83 = None
    permute_83: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_241, [1, 0]);  primals_241 = None
    addmm_38: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_242, view_95, permute_83);  primals_242 = None
    view_96: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_38, [8, 14, 14, 2048]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_122: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_96, 0.5)
    mul_123: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_96, 0.7071067811865476)
    erf_19: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_123);  mul_123 = None
    add_84: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_124: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_122, add_84);  mul_122 = add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_59: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_124);  mul_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_97: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_59, [1568, 2048]);  clone_59 = None
    permute_84: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_243, [1, 0]);  primals_243 = None
    addmm_39: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_244, view_97, permute_84);  primals_244 = None
    view_98: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_39, [8, 14, 14, 512]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_60: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_98);  view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_85: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_60, [0, 3, 1, 2]);  clone_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_99: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_66, [1, -1, 1, 1]);  primals_66 = None
    mul_125: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_85, view_99)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_85: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_125, add_81);  mul_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_23: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_85, primals_245, primals_246, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_86: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_23, [0, 2, 3, 1]);  convolution_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_61: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_86, memory_format = torch.contiguous_format)
    var_mean_23 = torch.ops.aten.var_mean.correction(clone_61, [3], correction = 0, keepdim = True)
    getitem_46: "f32[8, 14, 14, 1]" = var_mean_23[0]
    getitem_47: "f32[8, 14, 14, 1]" = var_mean_23[1];  var_mean_23 = None
    add_86: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-06);  getitem_46 = None
    rsqrt_23: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    sub_23: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_61, getitem_47);  clone_61 = None
    mul_126: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    mul_127: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_126, primals_67);  mul_126 = None
    add_87: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_127, primals_68);  mul_127 = primals_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_100: "f32[1568, 512]" = torch.ops.aten.view.default(add_87, [1568, 512]);  add_87 = None
    permute_87: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_247, [1, 0]);  primals_247 = None
    addmm_40: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_248, view_100, permute_87);  primals_248 = None
    view_101: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_40, [8, 14, 14, 2048]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_128: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_101, 0.5)
    mul_129: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_101, 0.7071067811865476)
    erf_20: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_129);  mul_129 = None
    add_88: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_130: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_128, add_88);  mul_128 = add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_62: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_130);  mul_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_102: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_62, [1568, 2048]);  clone_62 = None
    permute_88: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_249, [1, 0]);  primals_249 = None
    addmm_41: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_250, view_102, permute_88);  primals_250 = None
    view_103: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_41, [8, 14, 14, 512]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_63: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_103);  view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_89: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_63, [0, 3, 1, 2]);  clone_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_104: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_69, [1, -1, 1, 1]);  primals_69 = None
    mul_131: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_89, view_104)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_89: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_131, add_85);  mul_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_24: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_89, primals_251, primals_252, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_90: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_24, [0, 2, 3, 1]);  convolution_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_64: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format)
    var_mean_24 = torch.ops.aten.var_mean.correction(clone_64, [3], correction = 0, keepdim = True)
    getitem_48: "f32[8, 14, 14, 1]" = var_mean_24[0]
    getitem_49: "f32[8, 14, 14, 1]" = var_mean_24[1];  var_mean_24 = None
    add_90: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-06);  getitem_48 = None
    rsqrt_24: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
    sub_24: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_64, getitem_49);  clone_64 = None
    mul_132: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    mul_133: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_132, primals_70);  mul_132 = None
    add_91: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_133, primals_71);  mul_133 = primals_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_105: "f32[1568, 512]" = torch.ops.aten.view.default(add_91, [1568, 512]);  add_91 = None
    permute_91: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_253, [1, 0]);  primals_253 = None
    addmm_42: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_254, view_105, permute_91);  primals_254 = None
    view_106: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_42, [8, 14, 14, 2048]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_134: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_106, 0.5)
    mul_135: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_106, 0.7071067811865476)
    erf_21: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_135);  mul_135 = None
    add_92: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_136: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_134, add_92);  mul_134 = add_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_65: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_136);  mul_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_107: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_65, [1568, 2048]);  clone_65 = None
    permute_92: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_255, [1, 0]);  primals_255 = None
    addmm_43: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_256, view_107, permute_92);  primals_256 = None
    view_108: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_43, [8, 14, 14, 512]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_66: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_108);  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_93: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_66, [0, 3, 1, 2]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_109: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_72, [1, -1, 1, 1]);  primals_72 = None
    mul_137: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_93, view_109)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_93: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_137, add_89);  mul_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_25: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_93, primals_257, primals_258, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_94: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_25, [0, 2, 3, 1]);  convolution_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_67: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_94, memory_format = torch.contiguous_format)
    var_mean_25 = torch.ops.aten.var_mean.correction(clone_67, [3], correction = 0, keepdim = True)
    getitem_50: "f32[8, 14, 14, 1]" = var_mean_25[0]
    getitem_51: "f32[8, 14, 14, 1]" = var_mean_25[1];  var_mean_25 = None
    add_94: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-06);  getitem_50 = None
    rsqrt_25: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    sub_25: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_67, getitem_51);  clone_67 = None
    mul_138: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    mul_139: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_138, primals_73);  mul_138 = None
    add_95: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_139, primals_74);  mul_139 = primals_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_110: "f32[1568, 512]" = torch.ops.aten.view.default(add_95, [1568, 512]);  add_95 = None
    permute_95: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_259, [1, 0]);  primals_259 = None
    addmm_44: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_260, view_110, permute_95);  primals_260 = None
    view_111: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_44, [8, 14, 14, 2048]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_140: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_111, 0.5)
    mul_141: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_111, 0.7071067811865476)
    erf_22: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_141);  mul_141 = None
    add_96: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_142: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_140, add_96);  mul_140 = add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_68: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_142);  mul_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_112: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_68, [1568, 2048]);  clone_68 = None
    permute_96: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_261, [1, 0]);  primals_261 = None
    addmm_45: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_262, view_112, permute_96);  primals_262 = None
    view_113: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_45, [8, 14, 14, 512]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_69: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_113);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_97: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_69, [0, 3, 1, 2]);  clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_114: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_75, [1, -1, 1, 1]);  primals_75 = None
    mul_143: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_97, view_114)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_97: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_143, add_93);  mul_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_26: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_97, primals_263, primals_264, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_98: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_26, [0, 2, 3, 1]);  convolution_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_70: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_98, memory_format = torch.contiguous_format)
    var_mean_26 = torch.ops.aten.var_mean.correction(clone_70, [3], correction = 0, keepdim = True)
    getitem_52: "f32[8, 14, 14, 1]" = var_mean_26[0]
    getitem_53: "f32[8, 14, 14, 1]" = var_mean_26[1];  var_mean_26 = None
    add_98: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-06);  getitem_52 = None
    rsqrt_26: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    sub_26: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_70, getitem_53);  clone_70 = None
    mul_144: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    mul_145: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_144, primals_76);  mul_144 = None
    add_99: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_145, primals_77);  mul_145 = primals_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_115: "f32[1568, 512]" = torch.ops.aten.view.default(add_99, [1568, 512]);  add_99 = None
    permute_99: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_265, [1, 0]);  primals_265 = None
    addmm_46: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_266, view_115, permute_99);  primals_266 = None
    view_116: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_46, [8, 14, 14, 2048]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_146: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_116, 0.5)
    mul_147: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_116, 0.7071067811865476)
    erf_23: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_147);  mul_147 = None
    add_100: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_148: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_146, add_100);  mul_146 = add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_71: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_148);  mul_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_117: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_71, [1568, 2048]);  clone_71 = None
    permute_100: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_267, [1, 0]);  primals_267 = None
    addmm_47: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_268, view_117, permute_100);  primals_268 = None
    view_118: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_47, [8, 14, 14, 512]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_72: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_118);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_101: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_72, [0, 3, 1, 2]);  clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_119: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_78, [1, -1, 1, 1]);  primals_78 = None
    mul_149: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_101, view_119)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_101: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_149, add_97);  mul_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_27: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_101, primals_269, primals_270, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_102: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_27, [0, 2, 3, 1]);  convolution_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_73: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format)
    var_mean_27 = torch.ops.aten.var_mean.correction(clone_73, [3], correction = 0, keepdim = True)
    getitem_54: "f32[8, 14, 14, 1]" = var_mean_27[0]
    getitem_55: "f32[8, 14, 14, 1]" = var_mean_27[1];  var_mean_27 = None
    add_102: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-06);  getitem_54 = None
    rsqrt_27: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
    sub_27: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_73, getitem_55);  clone_73 = None
    mul_150: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    mul_151: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_150, primals_79);  mul_150 = None
    add_103: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_151, primals_80);  mul_151 = primals_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_120: "f32[1568, 512]" = torch.ops.aten.view.default(add_103, [1568, 512]);  add_103 = None
    permute_103: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_271, [1, 0]);  primals_271 = None
    addmm_48: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_272, view_120, permute_103);  primals_272 = None
    view_121: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_48, [8, 14, 14, 2048]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_152: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_121, 0.5)
    mul_153: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_121, 0.7071067811865476)
    erf_24: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_153);  mul_153 = None
    add_104: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
    mul_154: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_152, add_104);  mul_152 = add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_74: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_154);  mul_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_122: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_74, [1568, 2048]);  clone_74 = None
    permute_104: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_273, [1, 0]);  primals_273 = None
    addmm_49: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_274, view_122, permute_104);  primals_274 = None
    view_123: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_49, [8, 14, 14, 512]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_75: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_123);  view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_105: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_75, [0, 3, 1, 2]);  clone_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_124: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_81, [1, -1, 1, 1]);  primals_81 = None
    mul_155: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_105, view_124)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_105: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_155, add_101);  mul_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_28: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_105, primals_275, primals_276, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_106: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_28, [0, 2, 3, 1]);  convolution_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_76: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format)
    var_mean_28 = torch.ops.aten.var_mean.correction(clone_76, [3], correction = 0, keepdim = True)
    getitem_56: "f32[8, 14, 14, 1]" = var_mean_28[0]
    getitem_57: "f32[8, 14, 14, 1]" = var_mean_28[1];  var_mean_28 = None
    add_106: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-06);  getitem_56 = None
    rsqrt_28: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
    sub_28: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_76, getitem_57);  clone_76 = None
    mul_156: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    mul_157: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_156, primals_82);  mul_156 = None
    add_107: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_157, primals_83);  mul_157 = primals_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_125: "f32[1568, 512]" = torch.ops.aten.view.default(add_107, [1568, 512]);  add_107 = None
    permute_107: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_277, [1, 0]);  primals_277 = None
    addmm_50: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_278, view_125, permute_107);  primals_278 = None
    view_126: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_50, [8, 14, 14, 2048]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_158: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_126, 0.5)
    mul_159: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_126, 0.7071067811865476)
    erf_25: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_159);  mul_159 = None
    add_108: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
    mul_160: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_158, add_108);  mul_158 = add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_77: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_160);  mul_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_127: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_77, [1568, 2048]);  clone_77 = None
    permute_108: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_279, [1, 0]);  primals_279 = None
    addmm_51: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_280, view_127, permute_108);  primals_280 = None
    view_128: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_51, [8, 14, 14, 512]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_78: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_128);  view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_109: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_78, [0, 3, 1, 2]);  clone_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_129: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_84, [1, -1, 1, 1]);  primals_84 = None
    mul_161: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_109, view_129)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_109: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_161, add_105);  mul_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_29: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_109, primals_281, primals_282, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_110: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_29, [0, 2, 3, 1]);  convolution_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_79: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format)
    var_mean_29 = torch.ops.aten.var_mean.correction(clone_79, [3], correction = 0, keepdim = True)
    getitem_58: "f32[8, 14, 14, 1]" = var_mean_29[0]
    getitem_59: "f32[8, 14, 14, 1]" = var_mean_29[1];  var_mean_29 = None
    add_110: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-06);  getitem_58 = None
    rsqrt_29: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
    sub_29: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_79, getitem_59);  clone_79 = None
    mul_162: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    mul_163: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_162, primals_85);  mul_162 = None
    add_111: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_163, primals_86);  mul_163 = primals_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_130: "f32[1568, 512]" = torch.ops.aten.view.default(add_111, [1568, 512]);  add_111 = None
    permute_111: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_283, [1, 0]);  primals_283 = None
    addmm_52: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_284, view_130, permute_111);  primals_284 = None
    view_131: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_52, [8, 14, 14, 2048]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_164: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_131, 0.5)
    mul_165: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_131, 0.7071067811865476)
    erf_26: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_165);  mul_165 = None
    add_112: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
    mul_166: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_164, add_112);  mul_164 = add_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_80: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_166);  mul_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_132: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_80, [1568, 2048]);  clone_80 = None
    permute_112: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_285, [1, 0]);  primals_285 = None
    addmm_53: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_286, view_132, permute_112);  primals_286 = None
    view_133: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_53, [8, 14, 14, 512]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_81: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_133);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_113: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_81, [0, 3, 1, 2]);  clone_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_134: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_87, [1, -1, 1, 1]);  primals_87 = None
    mul_167: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_113, view_134)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_113: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_167, add_109);  mul_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_30: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_113, primals_287, primals_288, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_114: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_30, [0, 2, 3, 1]);  convolution_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_82: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format)
    var_mean_30 = torch.ops.aten.var_mean.correction(clone_82, [3], correction = 0, keepdim = True)
    getitem_60: "f32[8, 14, 14, 1]" = var_mean_30[0]
    getitem_61: "f32[8, 14, 14, 1]" = var_mean_30[1];  var_mean_30 = None
    add_114: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-06);  getitem_60 = None
    rsqrt_30: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
    sub_30: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_82, getitem_61);  clone_82 = None
    mul_168: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    mul_169: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_168, primals_88);  mul_168 = None
    add_115: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_169, primals_89);  mul_169 = primals_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_135: "f32[1568, 512]" = torch.ops.aten.view.default(add_115, [1568, 512]);  add_115 = None
    permute_115: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_289, [1, 0]);  primals_289 = None
    addmm_54: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_290, view_135, permute_115);  primals_290 = None
    view_136: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_54, [8, 14, 14, 2048]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_170: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_136, 0.5)
    mul_171: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_136, 0.7071067811865476)
    erf_27: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_171);  mul_171 = None
    add_116: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
    mul_172: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_170, add_116);  mul_170 = add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_83: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_172);  mul_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_137: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_83, [1568, 2048]);  clone_83 = None
    permute_116: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_291, [1, 0]);  primals_291 = None
    addmm_55: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_292, view_137, permute_116);  primals_292 = None
    view_138: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_55, [8, 14, 14, 512]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_84: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_138);  view_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_117: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_84, [0, 3, 1, 2]);  clone_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_139: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_90, [1, -1, 1, 1]);  primals_90 = None
    mul_173: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_117, view_139)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_117: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_173, add_113);  mul_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_31: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_117, primals_293, primals_294, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_118: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_31, [0, 2, 3, 1]);  convolution_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_85: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_118, memory_format = torch.contiguous_format)
    var_mean_31 = torch.ops.aten.var_mean.correction(clone_85, [3], correction = 0, keepdim = True)
    getitem_62: "f32[8, 14, 14, 1]" = var_mean_31[0]
    getitem_63: "f32[8, 14, 14, 1]" = var_mean_31[1];  var_mean_31 = None
    add_118: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-06);  getitem_62 = None
    rsqrt_31: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    sub_31: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_85, getitem_63);  clone_85 = None
    mul_174: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    mul_175: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_174, primals_91);  mul_174 = None
    add_119: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_175, primals_92);  mul_175 = primals_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_140: "f32[1568, 512]" = torch.ops.aten.view.default(add_119, [1568, 512]);  add_119 = None
    permute_119: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_295, [1, 0]);  primals_295 = None
    addmm_56: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_296, view_140, permute_119);  primals_296 = None
    view_141: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_56, [8, 14, 14, 2048]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_176: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_141, 0.5)
    mul_177: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_141, 0.7071067811865476)
    erf_28: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_177);  mul_177 = None
    add_120: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
    mul_178: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_176, add_120);  mul_176 = add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_86: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_178);  mul_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_142: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_86, [1568, 2048]);  clone_86 = None
    permute_120: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_297, [1, 0]);  primals_297 = None
    addmm_57: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_298, view_142, permute_120);  primals_298 = None
    view_143: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_57, [8, 14, 14, 512]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_87: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_143);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_121: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_87, [0, 3, 1, 2]);  clone_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_144: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_93, [1, -1, 1, 1]);  primals_93 = None
    mul_179: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_121, view_144)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_121: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_179, add_117);  mul_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_32: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_121, primals_299, primals_300, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_122: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_32, [0, 2, 3, 1]);  convolution_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_88: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format)
    var_mean_32 = torch.ops.aten.var_mean.correction(clone_88, [3], correction = 0, keepdim = True)
    getitem_64: "f32[8, 14, 14, 1]" = var_mean_32[0]
    getitem_65: "f32[8, 14, 14, 1]" = var_mean_32[1];  var_mean_32 = None
    add_122: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-06);  getitem_64 = None
    rsqrt_32: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    sub_32: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_88, getitem_65);  clone_88 = None
    mul_180: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    mul_181: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_180, primals_94);  mul_180 = None
    add_123: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_181, primals_95);  mul_181 = primals_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_145: "f32[1568, 512]" = torch.ops.aten.view.default(add_123, [1568, 512]);  add_123 = None
    permute_123: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_301, [1, 0]);  primals_301 = None
    addmm_58: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_302, view_145, permute_123);  primals_302 = None
    view_146: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_58, [8, 14, 14, 2048]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_182: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_146, 0.5)
    mul_183: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_146, 0.7071067811865476)
    erf_29: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_183);  mul_183 = None
    add_124: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
    mul_184: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_182, add_124);  mul_182 = add_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_89: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_184);  mul_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_147: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_89, [1568, 2048]);  clone_89 = None
    permute_124: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_303, [1, 0]);  primals_303 = None
    addmm_59: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_304, view_147, permute_124);  primals_304 = None
    view_148: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_59, [8, 14, 14, 512]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_90: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_148);  view_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_125: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_90, [0, 3, 1, 2]);  clone_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_149: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_96, [1, -1, 1, 1]);  primals_96 = None
    mul_185: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_125, view_149)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_125: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_185, add_121);  mul_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_33: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_125, primals_305, primals_306, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_126: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_33, [0, 2, 3, 1]);  convolution_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_91: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format)
    var_mean_33 = torch.ops.aten.var_mean.correction(clone_91, [3], correction = 0, keepdim = True)
    getitem_66: "f32[8, 14, 14, 1]" = var_mean_33[0]
    getitem_67: "f32[8, 14, 14, 1]" = var_mean_33[1];  var_mean_33 = None
    add_126: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-06);  getitem_66 = None
    rsqrt_33: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    sub_33: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_91, getitem_67);  clone_91 = None
    mul_186: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    mul_187: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_186, primals_97);  mul_186 = None
    add_127: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_187, primals_98);  mul_187 = primals_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_150: "f32[1568, 512]" = torch.ops.aten.view.default(add_127, [1568, 512]);  add_127 = None
    permute_127: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_307, [1, 0]);  primals_307 = None
    addmm_60: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_308, view_150, permute_127);  primals_308 = None
    view_151: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_60, [8, 14, 14, 2048]);  addmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_188: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_151, 0.5)
    mul_189: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476)
    erf_30: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_189);  mul_189 = None
    add_128: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
    mul_190: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_188, add_128);  mul_188 = add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_92: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_190);  mul_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_152: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_92, [1568, 2048]);  clone_92 = None
    permute_128: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_309, [1, 0]);  primals_309 = None
    addmm_61: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_310, view_152, permute_128);  primals_310 = None
    view_153: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_61, [8, 14, 14, 512]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_93: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_153);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_129: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_93, [0, 3, 1, 2]);  clone_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_154: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_99, [1, -1, 1, 1]);  primals_99 = None
    mul_191: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_129, view_154)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_129: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_191, add_125);  mul_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_34: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_129, primals_311, primals_312, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_130: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_34, [0, 2, 3, 1]);  convolution_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_94: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_130, memory_format = torch.contiguous_format)
    var_mean_34 = torch.ops.aten.var_mean.correction(clone_94, [3], correction = 0, keepdim = True)
    getitem_68: "f32[8, 14, 14, 1]" = var_mean_34[0]
    getitem_69: "f32[8, 14, 14, 1]" = var_mean_34[1];  var_mean_34 = None
    add_130: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-06);  getitem_68 = None
    rsqrt_34: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
    sub_34: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_94, getitem_69);  clone_94 = None
    mul_192: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    mul_193: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_192, primals_100);  mul_192 = None
    add_131: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_193, primals_101);  mul_193 = primals_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_155: "f32[1568, 512]" = torch.ops.aten.view.default(add_131, [1568, 512]);  add_131 = None
    permute_131: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_313, [1, 0]);  primals_313 = None
    addmm_62: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_314, view_155, permute_131);  primals_314 = None
    view_156: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_62, [8, 14, 14, 2048]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_194: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_156, 0.5)
    mul_195: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_156, 0.7071067811865476)
    erf_31: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_195);  mul_195 = None
    add_132: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
    mul_196: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_194, add_132);  mul_194 = add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_95: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_196);  mul_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_157: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_95, [1568, 2048]);  clone_95 = None
    permute_132: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_315, [1, 0]);  primals_315 = None
    addmm_63: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_316, view_157, permute_132);  primals_316 = None
    view_158: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_63, [8, 14, 14, 512]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_96: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_158);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_133: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_96, [0, 3, 1, 2]);  clone_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_159: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_102, [1, -1, 1, 1]);  primals_102 = None
    mul_197: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_133, view_159)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_133: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_197, add_129);  mul_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_35: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(add_133, primals_317, primals_318, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  primals_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_134: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_35, [0, 2, 3, 1]);  convolution_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_97: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_134, memory_format = torch.contiguous_format)
    var_mean_35 = torch.ops.aten.var_mean.correction(clone_97, [3], correction = 0, keepdim = True)
    getitem_70: "f32[8, 14, 14, 1]" = var_mean_35[0]
    getitem_71: "f32[8, 14, 14, 1]" = var_mean_35[1];  var_mean_35 = None
    add_134: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-06);  getitem_70 = None
    rsqrt_35: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
    sub_35: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_97, getitem_71);  clone_97 = None
    mul_198: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    mul_199: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_198, primals_103);  mul_198 = None
    add_135: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_199, primals_104);  mul_199 = primals_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_160: "f32[1568, 512]" = torch.ops.aten.view.default(add_135, [1568, 512]);  add_135 = None
    permute_135: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_319, [1, 0]);  primals_319 = None
    addmm_64: "f32[1568, 2048]" = torch.ops.aten.addmm.default(primals_320, view_160, permute_135);  primals_320 = None
    view_161: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(addmm_64, [8, 14, 14, 2048]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_200: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_161, 0.5)
    mul_201: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_161, 0.7071067811865476)
    erf_32: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_201);  mul_201 = None
    add_136: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
    mul_202: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_200, add_136);  mul_200 = add_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_98: "f32[8, 14, 14, 2048]" = torch.ops.aten.clone.default(mul_202);  mul_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_162: "f32[1568, 2048]" = torch.ops.aten.view.default(clone_98, [1568, 2048]);  clone_98 = None
    permute_136: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_321, [1, 0]);  primals_321 = None
    addmm_65: "f32[1568, 512]" = torch.ops.aten.addmm.default(primals_322, view_162, permute_136);  primals_322 = None
    view_163: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(addmm_65, [8, 14, 14, 512]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_99: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(view_163);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_137: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(clone_99, [0, 3, 1, 2]);  clone_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_164: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(primals_105, [1, -1, 1, 1]);  primals_105 = None
    mul_203: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_137, view_164)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_137: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_203, add_133);  mul_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    permute_138: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(add_137, [0, 2, 3, 1]);  add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_36 = torch.ops.aten.var_mean.correction(permute_138, [3], correction = 0, keepdim = True)
    getitem_72: "f32[8, 14, 14, 1]" = var_mean_36[0]
    getitem_73: "f32[8, 14, 14, 1]" = var_mean_36[1];  var_mean_36 = None
    add_138: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-06);  getitem_72 = None
    rsqrt_36: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
    sub_36: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_138, getitem_73)
    mul_204: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    mul_205: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_204, primals_106);  mul_204 = None
    add_139: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_205, primals_107);  mul_205 = primals_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    permute_139: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(add_139, [0, 3, 1, 2]);  add_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:229, code: x = self.downsample(x)
    convolution_36: "f32[8, 1024, 7, 7]" = torch.ops.aten.convolution.default(permute_139, primals_323, primals_324, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_37: "f32[8, 1024, 7, 7]" = torch.ops.aten.convolution.default(convolution_36, primals_325, primals_326, [1, 1], [3, 3], [1, 1], False, [0, 0], 1024);  primals_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_140: "f32[8, 7, 7, 1024]" = torch.ops.aten.permute.default(convolution_37, [0, 2, 3, 1]);  convolution_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_100: "f32[8, 7, 7, 1024]" = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format)
    var_mean_37 = torch.ops.aten.var_mean.correction(clone_100, [3], correction = 0, keepdim = True)
    getitem_74: "f32[8, 7, 7, 1]" = var_mean_37[0]
    getitem_75: "f32[8, 7, 7, 1]" = var_mean_37[1];  var_mean_37 = None
    add_140: "f32[8, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-06);  getitem_74 = None
    rsqrt_37: "f32[8, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    sub_37: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(clone_100, getitem_75);  clone_100 = None
    mul_206: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    mul_207: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_206, primals_108);  mul_206 = None
    add_141: "f32[8, 7, 7, 1024]" = torch.ops.aten.add.Tensor(mul_207, primals_109);  mul_207 = primals_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_165: "f32[392, 1024]" = torch.ops.aten.view.default(add_141, [392, 1024]);  add_141 = None
    permute_141: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_327, [1, 0]);  primals_327 = None
    addmm_66: "f32[392, 4096]" = torch.ops.aten.addmm.default(primals_328, view_165, permute_141);  primals_328 = None
    view_166: "f32[8, 7, 7, 4096]" = torch.ops.aten.view.default(addmm_66, [8, 7, 7, 4096]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_208: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_166, 0.5)
    mul_209: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_166, 0.7071067811865476)
    erf_33: "f32[8, 7, 7, 4096]" = torch.ops.aten.erf.default(mul_209);  mul_209 = None
    add_142: "f32[8, 7, 7, 4096]" = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
    mul_210: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(mul_208, add_142);  mul_208 = add_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_101: "f32[8, 7, 7, 4096]" = torch.ops.aten.clone.default(mul_210);  mul_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_167: "f32[392, 4096]" = torch.ops.aten.view.default(clone_101, [392, 4096]);  clone_101 = None
    permute_142: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_329, [1, 0]);  primals_329 = None
    addmm_67: "f32[392, 1024]" = torch.ops.aten.addmm.default(primals_330, view_167, permute_142);  primals_330 = None
    view_168: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(addmm_67, [8, 7, 7, 1024]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_102: "f32[8, 7, 7, 1024]" = torch.ops.aten.clone.default(view_168);  view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_143: "f32[8, 1024, 7, 7]" = torch.ops.aten.permute.default(clone_102, [0, 3, 1, 2]);  clone_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_169: "f32[1, 1024, 1, 1]" = torch.ops.aten.view.default(primals_110, [1, -1, 1, 1]);  primals_110 = None
    mul_211: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(permute_143, view_169)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_143: "f32[8, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_211, convolution_36);  mul_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_38: "f32[8, 1024, 7, 7]" = torch.ops.aten.convolution.default(add_143, primals_331, primals_332, [1, 1], [3, 3], [1, 1], False, [0, 0], 1024);  primals_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_144: "f32[8, 7, 7, 1024]" = torch.ops.aten.permute.default(convolution_38, [0, 2, 3, 1]);  convolution_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_103: "f32[8, 7, 7, 1024]" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format)
    var_mean_38 = torch.ops.aten.var_mean.correction(clone_103, [3], correction = 0, keepdim = True)
    getitem_76: "f32[8, 7, 7, 1]" = var_mean_38[0]
    getitem_77: "f32[8, 7, 7, 1]" = var_mean_38[1];  var_mean_38 = None
    add_144: "f32[8, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-06);  getitem_76 = None
    rsqrt_38: "f32[8, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    sub_38: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(clone_103, getitem_77);  clone_103 = None
    mul_212: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    mul_213: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_212, primals_111);  mul_212 = None
    add_145: "f32[8, 7, 7, 1024]" = torch.ops.aten.add.Tensor(mul_213, primals_112);  mul_213 = primals_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_170: "f32[392, 1024]" = torch.ops.aten.view.default(add_145, [392, 1024]);  add_145 = None
    permute_145: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_333, [1, 0]);  primals_333 = None
    addmm_68: "f32[392, 4096]" = torch.ops.aten.addmm.default(primals_334, view_170, permute_145);  primals_334 = None
    view_171: "f32[8, 7, 7, 4096]" = torch.ops.aten.view.default(addmm_68, [8, 7, 7, 4096]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_214: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_171, 0.5)
    mul_215: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_171, 0.7071067811865476)
    erf_34: "f32[8, 7, 7, 4096]" = torch.ops.aten.erf.default(mul_215);  mul_215 = None
    add_146: "f32[8, 7, 7, 4096]" = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
    mul_216: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(mul_214, add_146);  mul_214 = add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_104: "f32[8, 7, 7, 4096]" = torch.ops.aten.clone.default(mul_216);  mul_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_172: "f32[392, 4096]" = torch.ops.aten.view.default(clone_104, [392, 4096]);  clone_104 = None
    permute_146: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_335, [1, 0]);  primals_335 = None
    addmm_69: "f32[392, 1024]" = torch.ops.aten.addmm.default(primals_336, view_172, permute_146);  primals_336 = None
    view_173: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(addmm_69, [8, 7, 7, 1024]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_105: "f32[8, 7, 7, 1024]" = torch.ops.aten.clone.default(view_173);  view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_147: "f32[8, 1024, 7, 7]" = torch.ops.aten.permute.default(clone_105, [0, 3, 1, 2]);  clone_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_174: "f32[1, 1024, 1, 1]" = torch.ops.aten.view.default(primals_113, [1, -1, 1, 1]);  primals_113 = None
    mul_217: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(permute_147, view_174)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_147: "f32[8, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_217, add_143);  mul_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_39: "f32[8, 1024, 7, 7]" = torch.ops.aten.convolution.default(add_147, primals_337, primals_338, [1, 1], [3, 3], [1, 1], False, [0, 0], 1024);  primals_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_148: "f32[8, 7, 7, 1024]" = torch.ops.aten.permute.default(convolution_39, [0, 2, 3, 1]);  convolution_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_106: "f32[8, 7, 7, 1024]" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format)
    var_mean_39 = torch.ops.aten.var_mean.correction(clone_106, [3], correction = 0, keepdim = True)
    getitem_78: "f32[8, 7, 7, 1]" = var_mean_39[0]
    getitem_79: "f32[8, 7, 7, 1]" = var_mean_39[1];  var_mean_39 = None
    add_148: "f32[8, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-06);  getitem_78 = None
    rsqrt_39: "f32[8, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
    sub_39: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(clone_106, getitem_79);  clone_106 = None
    mul_218: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    mul_219: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_218, primals_114);  mul_218 = None
    add_149: "f32[8, 7, 7, 1024]" = torch.ops.aten.add.Tensor(mul_219, primals_115);  mul_219 = primals_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_175: "f32[392, 1024]" = torch.ops.aten.view.default(add_149, [392, 1024]);  add_149 = None
    permute_149: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_339, [1, 0]);  primals_339 = None
    addmm_70: "f32[392, 4096]" = torch.ops.aten.addmm.default(primals_340, view_175, permute_149);  primals_340 = None
    view_176: "f32[8, 7, 7, 4096]" = torch.ops.aten.view.default(addmm_70, [8, 7, 7, 4096]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_220: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_176, 0.5)
    mul_221: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_176, 0.7071067811865476)
    erf_35: "f32[8, 7, 7, 4096]" = torch.ops.aten.erf.default(mul_221);  mul_221 = None
    add_150: "f32[8, 7, 7, 4096]" = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
    mul_222: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(mul_220, add_150);  mul_220 = add_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_107: "f32[8, 7, 7, 4096]" = torch.ops.aten.clone.default(mul_222);  mul_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_177: "f32[392, 4096]" = torch.ops.aten.view.default(clone_107, [392, 4096]);  clone_107 = None
    permute_150: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_341, [1, 0]);  primals_341 = None
    addmm_71: "f32[392, 1024]" = torch.ops.aten.addmm.default(primals_342, view_177, permute_150);  primals_342 = None
    view_178: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(addmm_71, [8, 7, 7, 1024]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_108: "f32[8, 7, 7, 1024]" = torch.ops.aten.clone.default(view_178);  view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_151: "f32[8, 1024, 7, 7]" = torch.ops.aten.permute.default(clone_108, [0, 3, 1, 2]);  clone_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_179: "f32[1, 1024, 1, 1]" = torch.ops.aten.view.default(primals_116, [1, -1, 1, 1]);  primals_116 = None
    mul_223: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(permute_151, view_179)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_151: "f32[8, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_223, add_147);  mul_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean: "f32[8, 1024, 1, 1]" = torch.ops.aten.mean.dim(add_151, [-1, -2], True);  add_151 = None
    as_strided: "f32[8, 1024, 1, 1]" = torch.ops.aten.as_strided.default(mean, [8, 1024, 1, 1], [1024, 1, 1024, 1024]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    permute_152: "f32[8, 1, 1, 1024]" = torch.ops.aten.permute.default(as_strided, [0, 2, 3, 1]);  as_strided = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_40 = torch.ops.aten.var_mean.correction(permute_152, [3], correction = 0, keepdim = True)
    getitem_80: "f32[8, 1, 1, 1]" = var_mean_40[0]
    getitem_81: "f32[8, 1, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_152: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-06);  getitem_80 = None
    rsqrt_40: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
    sub_40: "f32[8, 1, 1, 1024]" = torch.ops.aten.sub.Tensor(permute_152, getitem_81)
    mul_224: "f32[8, 1, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    mul_225: "f32[8, 1, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_224, primals_117);  mul_224 = None
    add_153: "f32[8, 1, 1, 1024]" = torch.ops.aten.add.Tensor(mul_225, primals_118);  mul_225 = primals_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    permute_153: "f32[8, 1024, 1, 1]" = torch.ops.aten.permute.default(add_153, [0, 3, 1, 2]);  add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:202, code: x = self.flatten(x)
    view_180: "f32[8, 1024]" = torch.ops.aten.view.default(permute_153, [8, 1024]);  permute_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:204, code: x = self.drop(x)
    clone_109: "f32[8, 1024]" = torch.ops.aten.clone.default(view_180);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:207, code: x = self.fc(x)
    permute_154: "f32[1024, 1000]" = torch.ops.aten.permute.default(primals_343, [1, 0]);  primals_343 = None
    addmm_72: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_344, clone_109, permute_154);  primals_344 = None
    permute_155: "f32[1000, 1024]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
    mm: "f32[8, 1024]" = torch.ops.aten.mm.default(tangents_1, permute_155);  permute_155 = None
    permute_156: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 1024]" = torch.ops.aten.mm.default(permute_156, clone_109);  permute_156 = clone_109 = None
    permute_157: "f32[1024, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_181: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_158: "f32[1000, 1024]" = torch.ops.aten.permute.default(permute_157, [1, 0]);  permute_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:202, code: x = self.flatten(x)
    view_182: "f32[8, 1024, 1, 1]" = torch.ops.aten.view.default(mm, [8, 1024, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    permute_159: "f32[8, 1, 1, 1024]" = torch.ops.aten.permute.default(view_182, [0, 2, 3, 1]);  view_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_41: "f32[8, 1, 1, 1024]" = torch.ops.aten.sub.Tensor(permute_152, getitem_81);  permute_152 = getitem_81 = None
    mul_226: "f32[8, 1, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_40);  sub_41 = None
    mul_227: "f32[8, 1, 1, 1024]" = torch.ops.aten.mul.Tensor(permute_159, primals_117);  primals_117 = None
    mul_228: "f32[8, 1, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_227, 1024)
    sum_2: "f32[8, 1, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_227, [3], True)
    mul_229: "f32[8, 1, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_227, mul_226);  mul_227 = None
    sum_3: "f32[8, 1, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_229, [3], True);  mul_229 = None
    mul_230: "f32[8, 1, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_226, sum_3);  sum_3 = None
    sub_42: "f32[8, 1, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_228, sum_2);  mul_228 = sum_2 = None
    sub_43: "f32[8, 1, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_42, mul_230);  sub_42 = mul_230 = None
    div: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 1024);  rsqrt_40 = None
    mul_231: "f32[8, 1, 1, 1024]" = torch.ops.aten.mul.Tensor(div, sub_43);  div = sub_43 = None
    mul_232: "f32[8, 1, 1, 1024]" = torch.ops.aten.mul.Tensor(permute_159, mul_226);  mul_226 = None
    sum_4: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_232, [0, 1, 2]);  mul_232 = None
    sum_5: "f32[1024]" = torch.ops.aten.sum.dim_IntList(permute_159, [0, 1, 2]);  permute_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    permute_160: "f32[8, 1024, 1, 1]" = torch.ops.aten.permute.default(mul_231, [0, 3, 1, 2]);  mul_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    squeeze: "f32[8, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_160, 3);  permute_160 = None
    squeeze_1: "f32[8, 1024]" = torch.ops.aten.squeeze.dim(squeeze, 2);  squeeze = None
    full: "f32[8192]" = torch.ops.aten.full.default([8192], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    as_strided_1: "f32[8, 1024]" = torch.ops.aten.as_strided.default(full, [8, 1024], [1024, 1], 0)
    copy: "f32[8, 1024]" = torch.ops.aten.copy.default(as_strided_1, squeeze_1);  as_strided_1 = squeeze_1 = None
    as_strided_scatter: "f32[8192]" = torch.ops.aten.as_strided_scatter.default(full, copy, [8, 1024], [1024, 1], 0);  full = copy = None
    as_strided_4: "f32[8, 1024, 1, 1]" = torch.ops.aten.as_strided.default(as_strided_scatter, [8, 1024, 1, 1], [1024, 1, 1, 1], 0);  as_strided_scatter = None
    expand_1: "f32[8, 1024, 7, 7]" = torch.ops.aten.expand.default(as_strided_4, [8, 1024, 7, 7]);  as_strided_4 = None
    div_1: "f32[8, 1024, 7, 7]" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_233: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(div_1, permute_151);  permute_151 = None
    mul_234: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(div_1, view_179);  view_179 = None
    sum_6: "f32[1, 1024, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_233, [0, 2, 3], True);  mul_233 = None
    view_183: "f32[1024]" = torch.ops.aten.view.default(sum_6, [1024]);  sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_161: "f32[8, 7, 7, 1024]" = torch.ops.aten.permute.default(mul_234, [0, 2, 3, 1]);  mul_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_110: "f32[8, 7, 7, 1024]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
    view_184: "f32[392, 1024]" = torch.ops.aten.view.default(clone_110, [392, 1024]);  clone_110 = None
    permute_162: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_150, [1, 0]);  permute_150 = None
    mm_2: "f32[392, 4096]" = torch.ops.aten.mm.default(view_184, permute_162);  permute_162 = None
    permute_163: "f32[1024, 392]" = torch.ops.aten.permute.default(view_184, [1, 0])
    mm_3: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_163, view_177);  permute_163 = view_177 = None
    permute_164: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_7: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_184, [0], True);  view_184 = None
    view_185: "f32[1024]" = torch.ops.aten.view.default(sum_7, [1024]);  sum_7 = None
    permute_165: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
    view_186: "f32[8, 7, 7, 4096]" = torch.ops.aten.view.default(mm_2, [8, 7, 7, 4096]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_235: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_176, 0.7071067811865476)
    erf_36: "f32[8, 7, 7, 4096]" = torch.ops.aten.erf.default(mul_235);  mul_235 = None
    add_154: "f32[8, 7, 7, 4096]" = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
    mul_236: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(add_154, 0.5);  add_154 = None
    mul_237: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_176, view_176)
    mul_238: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(mul_237, -0.5);  mul_237 = None
    exp: "f32[8, 7, 7, 4096]" = torch.ops.aten.exp.default(mul_238);  mul_238 = None
    mul_239: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
    mul_240: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_176, mul_239);  view_176 = mul_239 = None
    add_155: "f32[8, 7, 7, 4096]" = torch.ops.aten.add.Tensor(mul_236, mul_240);  mul_236 = mul_240 = None
    mul_241: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_186, add_155);  view_186 = add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_187: "f32[392, 4096]" = torch.ops.aten.view.default(mul_241, [392, 4096]);  mul_241 = None
    permute_166: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_149, [1, 0]);  permute_149 = None
    mm_4: "f32[392, 1024]" = torch.ops.aten.mm.default(view_187, permute_166);  permute_166 = None
    permute_167: "f32[4096, 392]" = torch.ops.aten.permute.default(view_187, [1, 0])
    mm_5: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_167, view_175);  permute_167 = view_175 = None
    permute_168: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_8: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_187, [0], True);  view_187 = None
    view_188: "f32[4096]" = torch.ops.aten.view.default(sum_8, [4096]);  sum_8 = None
    permute_169: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_168, [1, 0]);  permute_168 = None
    view_189: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(mm_4, [8, 7, 7, 1024]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_111: "f32[8, 7, 7, 1024]" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
    sub_44: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(clone_111, getitem_79);  clone_111 = getitem_79 = None
    mul_242: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_39);  sub_44 = None
    mul_243: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_189, primals_114);  primals_114 = None
    mul_244: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_243, 1024)
    sum_9: "f32[8, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_243, [3], True)
    mul_245: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_243, mul_242);  mul_243 = None
    sum_10: "f32[8, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_245, [3], True);  mul_245 = None
    mul_246: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_242, sum_10);  sum_10 = None
    sub_45: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(mul_244, sum_9);  mul_244 = sum_9 = None
    sub_46: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(sub_45, mul_246);  sub_45 = mul_246 = None
    div_2: "f32[8, 7, 7, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 1024);  rsqrt_39 = None
    mul_247: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(div_2, sub_46);  div_2 = sub_46 = None
    mul_248: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_189, mul_242);  mul_242 = None
    sum_11: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_248, [0, 1, 2]);  mul_248 = None
    sum_12: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_189, [0, 1, 2]);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_170: "f32[8, 1024, 7, 7]" = torch.ops.aten.permute.default(mul_247, [0, 3, 1, 2]);  mul_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_13: "f32[1024]" = torch.ops.aten.sum.dim_IntList(permute_170, [0, 2, 3])
    convolution_backward = torch.ops.aten.convolution_backward.default(permute_170, add_147, primals_337, [1024], [1, 1], [3, 3], [1, 1], False, [0, 0], 1024, [True, True, False]);  permute_170 = add_147 = primals_337 = None
    getitem_82: "f32[8, 1024, 7, 7]" = convolution_backward[0]
    getitem_83: "f32[1024, 1, 7, 7]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_156: "f32[8, 1024, 7, 7]" = torch.ops.aten.add.Tensor(div_1, getitem_82);  div_1 = getitem_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_249: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(add_156, permute_147);  permute_147 = None
    mul_250: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(add_156, view_174);  view_174 = None
    sum_14: "f32[1, 1024, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_249, [0, 2, 3], True);  mul_249 = None
    view_190: "f32[1024]" = torch.ops.aten.view.default(sum_14, [1024]);  sum_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_171: "f32[8, 7, 7, 1024]" = torch.ops.aten.permute.default(mul_250, [0, 2, 3, 1]);  mul_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_112: "f32[8, 7, 7, 1024]" = torch.ops.aten.clone.default(permute_171, memory_format = torch.contiguous_format);  permute_171 = None
    view_191: "f32[392, 1024]" = torch.ops.aten.view.default(clone_112, [392, 1024]);  clone_112 = None
    permute_172: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_146, [1, 0]);  permute_146 = None
    mm_6: "f32[392, 4096]" = torch.ops.aten.mm.default(view_191, permute_172);  permute_172 = None
    permute_173: "f32[1024, 392]" = torch.ops.aten.permute.default(view_191, [1, 0])
    mm_7: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_173, view_172);  permute_173 = view_172 = None
    permute_174: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_15: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_191, [0], True);  view_191 = None
    view_192: "f32[1024]" = torch.ops.aten.view.default(sum_15, [1024]);  sum_15 = None
    permute_175: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
    view_193: "f32[8, 7, 7, 4096]" = torch.ops.aten.view.default(mm_6, [8, 7, 7, 4096]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_251: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_171, 0.7071067811865476)
    erf_37: "f32[8, 7, 7, 4096]" = torch.ops.aten.erf.default(mul_251);  mul_251 = None
    add_157: "f32[8, 7, 7, 4096]" = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
    mul_252: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(add_157, 0.5);  add_157 = None
    mul_253: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_171, view_171)
    mul_254: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(mul_253, -0.5);  mul_253 = None
    exp_1: "f32[8, 7, 7, 4096]" = torch.ops.aten.exp.default(mul_254);  mul_254 = None
    mul_255: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
    mul_256: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_171, mul_255);  view_171 = mul_255 = None
    add_158: "f32[8, 7, 7, 4096]" = torch.ops.aten.add.Tensor(mul_252, mul_256);  mul_252 = mul_256 = None
    mul_257: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_193, add_158);  view_193 = add_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_194: "f32[392, 4096]" = torch.ops.aten.view.default(mul_257, [392, 4096]);  mul_257 = None
    permute_176: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
    mm_8: "f32[392, 1024]" = torch.ops.aten.mm.default(view_194, permute_176);  permute_176 = None
    permute_177: "f32[4096, 392]" = torch.ops.aten.permute.default(view_194, [1, 0])
    mm_9: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_177, view_170);  permute_177 = view_170 = None
    permute_178: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_16: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_194, [0], True);  view_194 = None
    view_195: "f32[4096]" = torch.ops.aten.view.default(sum_16, [4096]);  sum_16 = None
    permute_179: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
    view_196: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(mm_8, [8, 7, 7, 1024]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_113: "f32[8, 7, 7, 1024]" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
    sub_47: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(clone_113, getitem_77);  clone_113 = getitem_77 = None
    mul_258: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_38);  sub_47 = None
    mul_259: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_196, primals_111);  primals_111 = None
    mul_260: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_259, 1024)
    sum_17: "f32[8, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_259, [3], True)
    mul_261: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_259, mul_258);  mul_259 = None
    sum_18: "f32[8, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_261, [3], True);  mul_261 = None
    mul_262: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_258, sum_18);  sum_18 = None
    sub_48: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(mul_260, sum_17);  mul_260 = sum_17 = None
    sub_49: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(sub_48, mul_262);  sub_48 = mul_262 = None
    div_3: "f32[8, 7, 7, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 1024);  rsqrt_38 = None
    mul_263: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(div_3, sub_49);  div_3 = sub_49 = None
    mul_264: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_196, mul_258);  mul_258 = None
    sum_19: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_264, [0, 1, 2]);  mul_264 = None
    sum_20: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_196, [0, 1, 2]);  view_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_180: "f32[8, 1024, 7, 7]" = torch.ops.aten.permute.default(mul_263, [0, 3, 1, 2]);  mul_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_21: "f32[1024]" = torch.ops.aten.sum.dim_IntList(permute_180, [0, 2, 3])
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(permute_180, add_143, primals_331, [1024], [1, 1], [3, 3], [1, 1], False, [0, 0], 1024, [True, True, False]);  permute_180 = add_143 = primals_331 = None
    getitem_85: "f32[8, 1024, 7, 7]" = convolution_backward_1[0]
    getitem_86: "f32[1024, 1, 7, 7]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_159: "f32[8, 1024, 7, 7]" = torch.ops.aten.add.Tensor(add_156, getitem_85);  add_156 = getitem_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_265: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(add_159, permute_143);  permute_143 = None
    mul_266: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(add_159, view_169);  view_169 = None
    sum_22: "f32[1, 1024, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_265, [0, 2, 3], True);  mul_265 = None
    view_197: "f32[1024]" = torch.ops.aten.view.default(sum_22, [1024]);  sum_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_181: "f32[8, 7, 7, 1024]" = torch.ops.aten.permute.default(mul_266, [0, 2, 3, 1]);  mul_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_114: "f32[8, 7, 7, 1024]" = torch.ops.aten.clone.default(permute_181, memory_format = torch.contiguous_format);  permute_181 = None
    view_198: "f32[392, 1024]" = torch.ops.aten.view.default(clone_114, [392, 1024]);  clone_114 = None
    permute_182: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
    mm_10: "f32[392, 4096]" = torch.ops.aten.mm.default(view_198, permute_182);  permute_182 = None
    permute_183: "f32[1024, 392]" = torch.ops.aten.permute.default(view_198, [1, 0])
    mm_11: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_183, view_167);  permute_183 = view_167 = None
    permute_184: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_23: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_198, [0], True);  view_198 = None
    view_199: "f32[1024]" = torch.ops.aten.view.default(sum_23, [1024]);  sum_23 = None
    permute_185: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
    view_200: "f32[8, 7, 7, 4096]" = torch.ops.aten.view.default(mm_10, [8, 7, 7, 4096]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_267: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_166, 0.7071067811865476)
    erf_38: "f32[8, 7, 7, 4096]" = torch.ops.aten.erf.default(mul_267);  mul_267 = None
    add_160: "f32[8, 7, 7, 4096]" = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
    mul_268: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(add_160, 0.5);  add_160 = None
    mul_269: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_166, view_166)
    mul_270: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(mul_269, -0.5);  mul_269 = None
    exp_2: "f32[8, 7, 7, 4096]" = torch.ops.aten.exp.default(mul_270);  mul_270 = None
    mul_271: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
    mul_272: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_166, mul_271);  view_166 = mul_271 = None
    add_161: "f32[8, 7, 7, 4096]" = torch.ops.aten.add.Tensor(mul_268, mul_272);  mul_268 = mul_272 = None
    mul_273: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_200, add_161);  view_200 = add_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_201: "f32[392, 4096]" = torch.ops.aten.view.default(mul_273, [392, 4096]);  mul_273 = None
    permute_186: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
    mm_12: "f32[392, 1024]" = torch.ops.aten.mm.default(view_201, permute_186);  permute_186 = None
    permute_187: "f32[4096, 392]" = torch.ops.aten.permute.default(view_201, [1, 0])
    mm_13: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_187, view_165);  permute_187 = view_165 = None
    permute_188: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_24: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_201, [0], True);  view_201 = None
    view_202: "f32[4096]" = torch.ops.aten.view.default(sum_24, [4096]);  sum_24 = None
    permute_189: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_188, [1, 0]);  permute_188 = None
    view_203: "f32[8, 7, 7, 1024]" = torch.ops.aten.view.default(mm_12, [8, 7, 7, 1024]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_115: "f32[8, 7, 7, 1024]" = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
    sub_50: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(clone_115, getitem_75);  clone_115 = getitem_75 = None
    mul_274: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_37);  sub_50 = None
    mul_275: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_203, primals_108);  primals_108 = None
    mul_276: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_275, 1024)
    sum_25: "f32[8, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_275, [3], True)
    mul_277: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_275, mul_274);  mul_275 = None
    sum_26: "f32[8, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_277, [3], True);  mul_277 = None
    mul_278: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_274, sum_26);  sum_26 = None
    sub_51: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(mul_276, sum_25);  mul_276 = sum_25 = None
    sub_52: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(sub_51, mul_278);  sub_51 = mul_278 = None
    div_4: "f32[8, 7, 7, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 1024);  rsqrt_37 = None
    mul_279: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(div_4, sub_52);  div_4 = sub_52 = None
    mul_280: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_203, mul_274);  mul_274 = None
    sum_27: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_280, [0, 1, 2]);  mul_280 = None
    sum_28: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_203, [0, 1, 2]);  view_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_190: "f32[8, 1024, 7, 7]" = torch.ops.aten.permute.default(mul_279, [0, 3, 1, 2]);  mul_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_29: "f32[1024]" = torch.ops.aten.sum.dim_IntList(permute_190, [0, 2, 3])
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(permute_190, convolution_36, primals_325, [1024], [1, 1], [3, 3], [1, 1], False, [0, 0], 1024, [True, True, False]);  permute_190 = convolution_36 = primals_325 = None
    getitem_88: "f32[8, 1024, 7, 7]" = convolution_backward_2[0]
    getitem_89: "f32[1024, 1, 7, 7]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_162: "f32[8, 1024, 7, 7]" = torch.ops.aten.add.Tensor(add_159, getitem_88);  add_159 = getitem_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:229, code: x = self.downsample(x)
    sum_30: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_162, [0, 2, 3])
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(add_162, permute_139, primals_323, [1024], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_162 = permute_139 = primals_323 = None
    getitem_91: "f32[8, 512, 14, 14]" = convolution_backward_3[0]
    getitem_92: "f32[1024, 512, 2, 2]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    permute_191: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(getitem_91, [0, 2, 3, 1]);  getitem_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_116: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_191, memory_format = torch.contiguous_format);  permute_191 = None
    sub_53: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_138, getitem_73);  permute_138 = getitem_73 = None
    mul_281: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_36);  sub_53 = None
    mul_282: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(clone_116, primals_106);  primals_106 = None
    mul_283: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_282, 512)
    sum_31: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_282, [3], True)
    mul_284: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_282, mul_281);  mul_282 = None
    sum_32: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_284, [3], True);  mul_284 = None
    mul_285: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_281, sum_32);  sum_32 = None
    sub_54: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_283, sum_31);  mul_283 = sum_31 = None
    sub_55: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_54, mul_285);  sub_54 = mul_285 = None
    div_5: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 512);  rsqrt_36 = None
    mul_286: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_5, sub_55);  div_5 = sub_55 = None
    mul_287: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(clone_116, mul_281);  mul_281 = None
    sum_33: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_287, [0, 1, 2]);  mul_287 = None
    sum_34: "f32[512]" = torch.ops.aten.sum.dim_IntList(clone_116, [0, 1, 2]);  clone_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    permute_192: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_286, [0, 3, 1, 2]);  mul_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_288: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_192, permute_137);  permute_137 = None
    mul_289: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_192, view_164);  view_164 = None
    sum_35: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_288, [0, 2, 3], True);  mul_288 = None
    view_204: "f32[512]" = torch.ops.aten.view.default(sum_35, [512]);  sum_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_193: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_289, [0, 2, 3, 1]);  mul_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_205: "f32[1568, 512]" = torch.ops.aten.view.default(permute_193, [1568, 512]);  permute_193 = None
    permute_194: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
    mm_14: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_205, permute_194);  permute_194 = None
    permute_195: "f32[512, 1568]" = torch.ops.aten.permute.default(view_205, [1, 0])
    mm_15: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_195, view_162);  permute_195 = view_162 = None
    permute_196: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_36: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_205, [0], True);  view_205 = None
    view_206: "f32[512]" = torch.ops.aten.view.default(sum_36, [512]);  sum_36 = None
    permute_197: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
    view_207: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(mm_14, [8, 14, 14, 2048]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_290: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_161, 0.7071067811865476)
    erf_39: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_290);  mul_290 = None
    add_163: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
    mul_291: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_163, 0.5);  add_163 = None
    mul_292: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_161, view_161)
    mul_293: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_292, -0.5);  mul_292 = None
    exp_3: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_293);  mul_293 = None
    mul_294: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_3, 0.3989422804014327);  exp_3 = None
    mul_295: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_161, mul_294);  view_161 = mul_294 = None
    add_164: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_291, mul_295);  mul_291 = mul_295 = None
    mul_296: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_207, add_164);  view_207 = add_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_208: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_296, [1568, 2048]);  mul_296 = None
    permute_198: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_135, [1, 0]);  permute_135 = None
    mm_16: "f32[1568, 512]" = torch.ops.aten.mm.default(view_208, permute_198);  permute_198 = None
    permute_199: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_208, [1, 0])
    mm_17: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_199, view_160);  permute_199 = view_160 = None
    permute_200: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_37: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_208, [0], True);  view_208 = None
    view_209: "f32[2048]" = torch.ops.aten.view.default(sum_37, [2048]);  sum_37 = None
    permute_201: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
    view_210: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(mm_16, [8, 14, 14, 512]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_117: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_134, memory_format = torch.contiguous_format);  permute_134 = None
    sub_56: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_117, getitem_71);  clone_117 = getitem_71 = None
    mul_297: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_35);  sub_56 = None
    mul_298: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_210, primals_103);  primals_103 = None
    mul_299: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_298, 512)
    sum_38: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_298, [3], True)
    mul_300: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_298, mul_297);  mul_298 = None
    sum_39: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_300, [3], True);  mul_300 = None
    mul_301: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_297, sum_39);  sum_39 = None
    sub_57: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_299, sum_38);  mul_299 = sum_38 = None
    sub_58: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_57, mul_301);  sub_57 = mul_301 = None
    div_6: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 512);  rsqrt_35 = None
    mul_302: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_6, sub_58);  div_6 = sub_58 = None
    mul_303: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_210, mul_297);  mul_297 = None
    sum_40: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_303, [0, 1, 2]);  mul_303 = None
    sum_41: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_210, [0, 1, 2]);  view_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_202: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_302, [0, 3, 1, 2]);  mul_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_42: "f32[512]" = torch.ops.aten.sum.dim_IntList(permute_202, [0, 2, 3])
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(permute_202, add_133, primals_317, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False]);  permute_202 = add_133 = primals_317 = None
    getitem_94: "f32[8, 512, 14, 14]" = convolution_backward_4[0]
    getitem_95: "f32[512, 1, 7, 7]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_165: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(permute_192, getitem_94);  permute_192 = getitem_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_304: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_165, permute_133);  permute_133 = None
    mul_305: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_165, view_159);  view_159 = None
    sum_43: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_304, [0, 2, 3], True);  mul_304 = None
    view_211: "f32[512]" = torch.ops.aten.view.default(sum_43, [512]);  sum_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_203: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_305, [0, 2, 3, 1]);  mul_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_212: "f32[1568, 512]" = torch.ops.aten.view.default(permute_203, [1568, 512]);  permute_203 = None
    permute_204: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    mm_18: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_212, permute_204);  permute_204 = None
    permute_205: "f32[512, 1568]" = torch.ops.aten.permute.default(view_212, [1, 0])
    mm_19: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_205, view_157);  permute_205 = view_157 = None
    permute_206: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_44: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_212, [0], True);  view_212 = None
    view_213: "f32[512]" = torch.ops.aten.view.default(sum_44, [512]);  sum_44 = None
    permute_207: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
    view_214: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(mm_18, [8, 14, 14, 2048]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_306: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_156, 0.7071067811865476)
    erf_40: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_306);  mul_306 = None
    add_166: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
    mul_307: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_166, 0.5);  add_166 = None
    mul_308: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_156, view_156)
    mul_309: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_308, -0.5);  mul_308 = None
    exp_4: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_309);  mul_309 = None
    mul_310: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
    mul_311: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_156, mul_310);  view_156 = mul_310 = None
    add_167: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_307, mul_311);  mul_307 = mul_311 = None
    mul_312: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_214, add_167);  view_214 = add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_215: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_312, [1568, 2048]);  mul_312 = None
    permute_208: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    mm_20: "f32[1568, 512]" = torch.ops.aten.mm.default(view_215, permute_208);  permute_208 = None
    permute_209: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_215, [1, 0])
    mm_21: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_209, view_155);  permute_209 = view_155 = None
    permute_210: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_45: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_215, [0], True);  view_215 = None
    view_216: "f32[2048]" = torch.ops.aten.view.default(sum_45, [2048]);  sum_45 = None
    permute_211: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
    view_217: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(mm_20, [8, 14, 14, 512]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_118: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_130, memory_format = torch.contiguous_format);  permute_130 = None
    sub_59: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_118, getitem_69);  clone_118 = getitem_69 = None
    mul_313: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_34);  sub_59 = None
    mul_314: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_217, primals_100);  primals_100 = None
    mul_315: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_314, 512)
    sum_46: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_314, [3], True)
    mul_316: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_314, mul_313);  mul_314 = None
    sum_47: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_316, [3], True);  mul_316 = None
    mul_317: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_313, sum_47);  sum_47 = None
    sub_60: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_315, sum_46);  mul_315 = sum_46 = None
    sub_61: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_60, mul_317);  sub_60 = mul_317 = None
    div_7: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 512);  rsqrt_34 = None
    mul_318: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_7, sub_61);  div_7 = sub_61 = None
    mul_319: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_217, mul_313);  mul_313 = None
    sum_48: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_319, [0, 1, 2]);  mul_319 = None
    sum_49: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_217, [0, 1, 2]);  view_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_212: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_318, [0, 3, 1, 2]);  mul_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_50: "f32[512]" = torch.ops.aten.sum.dim_IntList(permute_212, [0, 2, 3])
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(permute_212, add_129, primals_311, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False]);  permute_212 = add_129 = primals_311 = None
    getitem_97: "f32[8, 512, 14, 14]" = convolution_backward_5[0]
    getitem_98: "f32[512, 1, 7, 7]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_168: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_165, getitem_97);  add_165 = getitem_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_320: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_168, permute_129);  permute_129 = None
    mul_321: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_168, view_154);  view_154 = None
    sum_51: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_320, [0, 2, 3], True);  mul_320 = None
    view_218: "f32[512]" = torch.ops.aten.view.default(sum_51, [512]);  sum_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_213: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_321, [0, 2, 3, 1]);  mul_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_219: "f32[1568, 512]" = torch.ops.aten.view.default(permute_213, [1568, 512]);  permute_213 = None
    permute_214: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
    mm_22: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_219, permute_214);  permute_214 = None
    permute_215: "f32[512, 1568]" = torch.ops.aten.permute.default(view_219, [1, 0])
    mm_23: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_215, view_152);  permute_215 = view_152 = None
    permute_216: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_52: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_219, [0], True);  view_219 = None
    view_220: "f32[512]" = torch.ops.aten.view.default(sum_52, [512]);  sum_52 = None
    permute_217: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_216, [1, 0]);  permute_216 = None
    view_221: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(mm_22, [8, 14, 14, 2048]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_322: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476)
    erf_41: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_322);  mul_322 = None
    add_169: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
    mul_323: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_169, 0.5);  add_169 = None
    mul_324: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_151, view_151)
    mul_325: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_324, -0.5);  mul_324 = None
    exp_5: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_325);  mul_325 = None
    mul_326: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_5, 0.3989422804014327);  exp_5 = None
    mul_327: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_151, mul_326);  view_151 = mul_326 = None
    add_170: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_323, mul_327);  mul_323 = mul_327 = None
    mul_328: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_221, add_170);  view_221 = add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_222: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_328, [1568, 2048]);  mul_328 = None
    permute_218: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_127, [1, 0]);  permute_127 = None
    mm_24: "f32[1568, 512]" = torch.ops.aten.mm.default(view_222, permute_218);  permute_218 = None
    permute_219: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_222, [1, 0])
    mm_25: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_219, view_150);  permute_219 = view_150 = None
    permute_220: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_53: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_222, [0], True);  view_222 = None
    view_223: "f32[2048]" = torch.ops.aten.view.default(sum_53, [2048]);  sum_53 = None
    permute_221: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
    view_224: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(mm_24, [8, 14, 14, 512]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_119: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
    sub_62: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_119, getitem_67);  clone_119 = getitem_67 = None
    mul_329: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_33);  sub_62 = None
    mul_330: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_224, primals_97);  primals_97 = None
    mul_331: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_330, 512)
    sum_54: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_330, [3], True)
    mul_332: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_330, mul_329);  mul_330 = None
    sum_55: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_332, [3], True);  mul_332 = None
    mul_333: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_329, sum_55);  sum_55 = None
    sub_63: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_331, sum_54);  mul_331 = sum_54 = None
    sub_64: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_63, mul_333);  sub_63 = mul_333 = None
    div_8: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 512);  rsqrt_33 = None
    mul_334: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_8, sub_64);  div_8 = sub_64 = None
    mul_335: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_224, mul_329);  mul_329 = None
    sum_56: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_335, [0, 1, 2]);  mul_335 = None
    sum_57: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_224, [0, 1, 2]);  view_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_222: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_334, [0, 3, 1, 2]);  mul_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_58: "f32[512]" = torch.ops.aten.sum.dim_IntList(permute_222, [0, 2, 3])
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(permute_222, add_125, primals_305, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False]);  permute_222 = add_125 = primals_305 = None
    getitem_100: "f32[8, 512, 14, 14]" = convolution_backward_6[0]
    getitem_101: "f32[512, 1, 7, 7]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_171: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_168, getitem_100);  add_168 = getitem_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_336: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_171, permute_125);  permute_125 = None
    mul_337: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_171, view_149);  view_149 = None
    sum_59: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_336, [0, 2, 3], True);  mul_336 = None
    view_225: "f32[512]" = torch.ops.aten.view.default(sum_59, [512]);  sum_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_223: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_337, [0, 2, 3, 1]);  mul_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_226: "f32[1568, 512]" = torch.ops.aten.view.default(permute_223, [1568, 512]);  permute_223 = None
    permute_224: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
    mm_26: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_226, permute_224);  permute_224 = None
    permute_225: "f32[512, 1568]" = torch.ops.aten.permute.default(view_226, [1, 0])
    mm_27: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_225, view_147);  permute_225 = view_147 = None
    permute_226: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_60: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_226, [0], True);  view_226 = None
    view_227: "f32[512]" = torch.ops.aten.view.default(sum_60, [512]);  sum_60 = None
    permute_227: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_226, [1, 0]);  permute_226 = None
    view_228: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(mm_26, [8, 14, 14, 2048]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_338: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_146, 0.7071067811865476)
    erf_42: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_338);  mul_338 = None
    add_172: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
    mul_339: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_172, 0.5);  add_172 = None
    mul_340: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_146, view_146)
    mul_341: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_340, -0.5);  mul_340 = None
    exp_6: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_341);  mul_341 = None
    mul_342: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
    mul_343: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_146, mul_342);  view_146 = mul_342 = None
    add_173: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_339, mul_343);  mul_339 = mul_343 = None
    mul_344: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_228, add_173);  view_228 = add_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_229: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_344, [1568, 2048]);  mul_344 = None
    permute_228: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
    mm_28: "f32[1568, 512]" = torch.ops.aten.mm.default(view_229, permute_228);  permute_228 = None
    permute_229: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_229, [1, 0])
    mm_29: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_229, view_145);  permute_229 = view_145 = None
    permute_230: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_61: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_229, [0], True);  view_229 = None
    view_230: "f32[2048]" = torch.ops.aten.view.default(sum_61, [2048]);  sum_61 = None
    permute_231: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None
    view_231: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(mm_28, [8, 14, 14, 512]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_120: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
    sub_65: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_120, getitem_65);  clone_120 = getitem_65 = None
    mul_345: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_32);  sub_65 = None
    mul_346: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_231, primals_94);  primals_94 = None
    mul_347: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_346, 512)
    sum_62: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_346, [3], True)
    mul_348: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_346, mul_345);  mul_346 = None
    sum_63: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_348, [3], True);  mul_348 = None
    mul_349: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_345, sum_63);  sum_63 = None
    sub_66: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_347, sum_62);  mul_347 = sum_62 = None
    sub_67: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_66, mul_349);  sub_66 = mul_349 = None
    div_9: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 512);  rsqrt_32 = None
    mul_350: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_9, sub_67);  div_9 = sub_67 = None
    mul_351: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_231, mul_345);  mul_345 = None
    sum_64: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_351, [0, 1, 2]);  mul_351 = None
    sum_65: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_231, [0, 1, 2]);  view_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_232: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_350, [0, 3, 1, 2]);  mul_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_66: "f32[512]" = torch.ops.aten.sum.dim_IntList(permute_232, [0, 2, 3])
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(permute_232, add_121, primals_299, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False]);  permute_232 = add_121 = primals_299 = None
    getitem_103: "f32[8, 512, 14, 14]" = convolution_backward_7[0]
    getitem_104: "f32[512, 1, 7, 7]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_174: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_171, getitem_103);  add_171 = getitem_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_352: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_174, permute_121);  permute_121 = None
    mul_353: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_174, view_144);  view_144 = None
    sum_67: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_352, [0, 2, 3], True);  mul_352 = None
    view_232: "f32[512]" = torch.ops.aten.view.default(sum_67, [512]);  sum_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_233: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_353, [0, 2, 3, 1]);  mul_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_233: "f32[1568, 512]" = torch.ops.aten.view.default(permute_233, [1568, 512]);  permute_233 = None
    permute_234: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    mm_30: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_233, permute_234);  permute_234 = None
    permute_235: "f32[512, 1568]" = torch.ops.aten.permute.default(view_233, [1, 0])
    mm_31: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_235, view_142);  permute_235 = view_142 = None
    permute_236: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_68: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_233, [0], True);  view_233 = None
    view_234: "f32[512]" = torch.ops.aten.view.default(sum_68, [512]);  sum_68 = None
    permute_237: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_236, [1, 0]);  permute_236 = None
    view_235: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(mm_30, [8, 14, 14, 2048]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_354: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_141, 0.7071067811865476)
    erf_43: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_354);  mul_354 = None
    add_175: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
    mul_355: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_175, 0.5);  add_175 = None
    mul_356: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_141, view_141)
    mul_357: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_356, -0.5);  mul_356 = None
    exp_7: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_357);  mul_357 = None
    mul_358: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
    mul_359: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_141, mul_358);  view_141 = mul_358 = None
    add_176: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_355, mul_359);  mul_355 = mul_359 = None
    mul_360: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_235, add_176);  view_235 = add_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_236: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_360, [1568, 2048]);  mul_360 = None
    permute_238: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    mm_32: "f32[1568, 512]" = torch.ops.aten.mm.default(view_236, permute_238);  permute_238 = None
    permute_239: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_236, [1, 0])
    mm_33: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_239, view_140);  permute_239 = view_140 = None
    permute_240: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_69: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_236, [0], True);  view_236 = None
    view_237: "f32[2048]" = torch.ops.aten.view.default(sum_69, [2048]);  sum_69 = None
    permute_241: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
    view_238: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(mm_32, [8, 14, 14, 512]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_121: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_118, memory_format = torch.contiguous_format);  permute_118 = None
    sub_68: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_121, getitem_63);  clone_121 = getitem_63 = None
    mul_361: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_31);  sub_68 = None
    mul_362: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_238, primals_91);  primals_91 = None
    mul_363: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_362, 512)
    sum_70: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_362, [3], True)
    mul_364: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_362, mul_361);  mul_362 = None
    sum_71: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_364, [3], True);  mul_364 = None
    mul_365: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_361, sum_71);  sum_71 = None
    sub_69: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_363, sum_70);  mul_363 = sum_70 = None
    sub_70: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_69, mul_365);  sub_69 = mul_365 = None
    div_10: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 512);  rsqrt_31 = None
    mul_366: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_10, sub_70);  div_10 = sub_70 = None
    mul_367: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_238, mul_361);  mul_361 = None
    sum_72: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_367, [0, 1, 2]);  mul_367 = None
    sum_73: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_238, [0, 1, 2]);  view_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_242: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_366, [0, 3, 1, 2]);  mul_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_74: "f32[512]" = torch.ops.aten.sum.dim_IntList(permute_242, [0, 2, 3])
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(permute_242, add_117, primals_293, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False]);  permute_242 = add_117 = primals_293 = None
    getitem_106: "f32[8, 512, 14, 14]" = convolution_backward_8[0]
    getitem_107: "f32[512, 1, 7, 7]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_177: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_174, getitem_106);  add_174 = getitem_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_368: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_177, permute_117);  permute_117 = None
    mul_369: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_177, view_139);  view_139 = None
    sum_75: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_368, [0, 2, 3], True);  mul_368 = None
    view_239: "f32[512]" = torch.ops.aten.view.default(sum_75, [512]);  sum_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_243: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_369, [0, 2, 3, 1]);  mul_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_240: "f32[1568, 512]" = torch.ops.aten.view.default(permute_243, [1568, 512]);  permute_243 = None
    permute_244: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
    mm_34: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_240, permute_244);  permute_244 = None
    permute_245: "f32[512, 1568]" = torch.ops.aten.permute.default(view_240, [1, 0])
    mm_35: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_245, view_137);  permute_245 = view_137 = None
    permute_246: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_76: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_240, [0], True);  view_240 = None
    view_241: "f32[512]" = torch.ops.aten.view.default(sum_76, [512]);  sum_76 = None
    permute_247: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_246, [1, 0]);  permute_246 = None
    view_242: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(mm_34, [8, 14, 14, 2048]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_370: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_136, 0.7071067811865476)
    erf_44: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_370);  mul_370 = None
    add_178: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_44, 1);  erf_44 = None
    mul_371: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_178, 0.5);  add_178 = None
    mul_372: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_136, view_136)
    mul_373: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_372, -0.5);  mul_372 = None
    exp_8: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_373);  mul_373 = None
    mul_374: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
    mul_375: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_136, mul_374);  view_136 = mul_374 = None
    add_179: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_371, mul_375);  mul_371 = mul_375 = None
    mul_376: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_242, add_179);  view_242 = add_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_243: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_376, [1568, 2048]);  mul_376 = None
    permute_248: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None
    mm_36: "f32[1568, 512]" = torch.ops.aten.mm.default(view_243, permute_248);  permute_248 = None
    permute_249: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_243, [1, 0])
    mm_37: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_249, view_135);  permute_249 = view_135 = None
    permute_250: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_77: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_243, [0], True);  view_243 = None
    view_244: "f32[2048]" = torch.ops.aten.view.default(sum_77, [2048]);  sum_77 = None
    permute_251: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_250, [1, 0]);  permute_250 = None
    view_245: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(mm_36, [8, 14, 14, 512]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_122: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
    sub_71: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_122, getitem_61);  clone_122 = getitem_61 = None
    mul_377: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_30);  sub_71 = None
    mul_378: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_245, primals_88);  primals_88 = None
    mul_379: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_378, 512)
    sum_78: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_378, [3], True)
    mul_380: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_378, mul_377);  mul_378 = None
    sum_79: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_380, [3], True);  mul_380 = None
    mul_381: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_377, sum_79);  sum_79 = None
    sub_72: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_379, sum_78);  mul_379 = sum_78 = None
    sub_73: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_72, mul_381);  sub_72 = mul_381 = None
    div_11: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 512);  rsqrt_30 = None
    mul_382: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_11, sub_73);  div_11 = sub_73 = None
    mul_383: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_245, mul_377);  mul_377 = None
    sum_80: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_383, [0, 1, 2]);  mul_383 = None
    sum_81: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_245, [0, 1, 2]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_252: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_382, [0, 3, 1, 2]);  mul_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_82: "f32[512]" = torch.ops.aten.sum.dim_IntList(permute_252, [0, 2, 3])
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(permute_252, add_113, primals_287, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False]);  permute_252 = add_113 = primals_287 = None
    getitem_109: "f32[8, 512, 14, 14]" = convolution_backward_9[0]
    getitem_110: "f32[512, 1, 7, 7]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_180: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_177, getitem_109);  add_177 = getitem_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_384: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_180, permute_113);  permute_113 = None
    mul_385: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_180, view_134);  view_134 = None
    sum_83: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_384, [0, 2, 3], True);  mul_384 = None
    view_246: "f32[512]" = torch.ops.aten.view.default(sum_83, [512]);  sum_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_253: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_385, [0, 2, 3, 1]);  mul_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_247: "f32[1568, 512]" = torch.ops.aten.view.default(permute_253, [1568, 512]);  permute_253 = None
    permute_254: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
    mm_38: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_247, permute_254);  permute_254 = None
    permute_255: "f32[512, 1568]" = torch.ops.aten.permute.default(view_247, [1, 0])
    mm_39: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_255, view_132);  permute_255 = view_132 = None
    permute_256: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_84: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_247, [0], True);  view_247 = None
    view_248: "f32[512]" = torch.ops.aten.view.default(sum_84, [512]);  sum_84 = None
    permute_257: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_256, [1, 0]);  permute_256 = None
    view_249: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(mm_38, [8, 14, 14, 2048]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_386: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_131, 0.7071067811865476)
    erf_45: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_386);  mul_386 = None
    add_181: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_45, 1);  erf_45 = None
    mul_387: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_181, 0.5);  add_181 = None
    mul_388: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_131, view_131)
    mul_389: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_388, -0.5);  mul_388 = None
    exp_9: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_389);  mul_389 = None
    mul_390: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
    mul_391: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_131, mul_390);  view_131 = mul_390 = None
    add_182: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_387, mul_391);  mul_387 = mul_391 = None
    mul_392: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_249, add_182);  view_249 = add_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_250: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_392, [1568, 2048]);  mul_392 = None
    permute_258: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    mm_40: "f32[1568, 512]" = torch.ops.aten.mm.default(view_250, permute_258);  permute_258 = None
    permute_259: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_250, [1, 0])
    mm_41: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_259, view_130);  permute_259 = view_130 = None
    permute_260: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_85: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_250, [0], True);  view_250 = None
    view_251: "f32[2048]" = torch.ops.aten.view.default(sum_85, [2048]);  sum_85 = None
    permute_261: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_260, [1, 0]);  permute_260 = None
    view_252: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(mm_40, [8, 14, 14, 512]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_123: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None
    sub_74: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_123, getitem_59);  clone_123 = getitem_59 = None
    mul_393: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_29);  sub_74 = None
    mul_394: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_252, primals_85);  primals_85 = None
    mul_395: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_394, 512)
    sum_86: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_394, [3], True)
    mul_396: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_394, mul_393);  mul_394 = None
    sum_87: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_396, [3], True);  mul_396 = None
    mul_397: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_393, sum_87);  sum_87 = None
    sub_75: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_395, sum_86);  mul_395 = sum_86 = None
    sub_76: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_75, mul_397);  sub_75 = mul_397 = None
    div_12: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 512);  rsqrt_29 = None
    mul_398: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_12, sub_76);  div_12 = sub_76 = None
    mul_399: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_252, mul_393);  mul_393 = None
    sum_88: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_399, [0, 1, 2]);  mul_399 = None
    sum_89: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_252, [0, 1, 2]);  view_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_262: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_398, [0, 3, 1, 2]);  mul_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_90: "f32[512]" = torch.ops.aten.sum.dim_IntList(permute_262, [0, 2, 3])
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(permute_262, add_109, primals_281, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False]);  permute_262 = add_109 = primals_281 = None
    getitem_112: "f32[8, 512, 14, 14]" = convolution_backward_10[0]
    getitem_113: "f32[512, 1, 7, 7]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_183: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_180, getitem_112);  add_180 = getitem_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_400: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_183, permute_109);  permute_109 = None
    mul_401: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_183, view_129);  view_129 = None
    sum_91: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_400, [0, 2, 3], True);  mul_400 = None
    view_253: "f32[512]" = torch.ops.aten.view.default(sum_91, [512]);  sum_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_263: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_401, [0, 2, 3, 1]);  mul_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_254: "f32[1568, 512]" = torch.ops.aten.view.default(permute_263, [1568, 512]);  permute_263 = None
    permute_264: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    mm_42: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_254, permute_264);  permute_264 = None
    permute_265: "f32[512, 1568]" = torch.ops.aten.permute.default(view_254, [1, 0])
    mm_43: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_265, view_127);  permute_265 = view_127 = None
    permute_266: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_92: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_254, [0], True);  view_254 = None
    view_255: "f32[512]" = torch.ops.aten.view.default(sum_92, [512]);  sum_92 = None
    permute_267: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_266, [1, 0]);  permute_266 = None
    view_256: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(mm_42, [8, 14, 14, 2048]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_402: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_126, 0.7071067811865476)
    erf_46: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_402);  mul_402 = None
    add_184: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_46, 1);  erf_46 = None
    mul_403: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_184, 0.5);  add_184 = None
    mul_404: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_126, view_126)
    mul_405: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_404, -0.5);  mul_404 = None
    exp_10: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_405);  mul_405 = None
    mul_406: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
    mul_407: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_126, mul_406);  view_126 = mul_406 = None
    add_185: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_403, mul_407);  mul_403 = mul_407 = None
    mul_408: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_256, add_185);  view_256 = add_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_257: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_408, [1568, 2048]);  mul_408 = None
    permute_268: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    mm_44: "f32[1568, 512]" = torch.ops.aten.mm.default(view_257, permute_268);  permute_268 = None
    permute_269: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_257, [1, 0])
    mm_45: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_269, view_125);  permute_269 = view_125 = None
    permute_270: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_93: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_257, [0], True);  view_257 = None
    view_258: "f32[2048]" = torch.ops.aten.view.default(sum_93, [2048]);  sum_93 = None
    permute_271: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_270, [1, 0]);  permute_270 = None
    view_259: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(mm_44, [8, 14, 14, 512]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_124: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    sub_77: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_124, getitem_57);  clone_124 = getitem_57 = None
    mul_409: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_28);  sub_77 = None
    mul_410: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_259, primals_82);  primals_82 = None
    mul_411: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_410, 512)
    sum_94: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_410, [3], True)
    mul_412: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_410, mul_409);  mul_410 = None
    sum_95: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_412, [3], True);  mul_412 = None
    mul_413: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_409, sum_95);  sum_95 = None
    sub_78: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_411, sum_94);  mul_411 = sum_94 = None
    sub_79: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_78, mul_413);  sub_78 = mul_413 = None
    div_13: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 512);  rsqrt_28 = None
    mul_414: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_13, sub_79);  div_13 = sub_79 = None
    mul_415: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_259, mul_409);  mul_409 = None
    sum_96: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_415, [0, 1, 2]);  mul_415 = None
    sum_97: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_259, [0, 1, 2]);  view_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_272: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_414, [0, 3, 1, 2]);  mul_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_98: "f32[512]" = torch.ops.aten.sum.dim_IntList(permute_272, [0, 2, 3])
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(permute_272, add_105, primals_275, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False]);  permute_272 = add_105 = primals_275 = None
    getitem_115: "f32[8, 512, 14, 14]" = convolution_backward_11[0]
    getitem_116: "f32[512, 1, 7, 7]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_186: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_183, getitem_115);  add_183 = getitem_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_416: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_186, permute_105);  permute_105 = None
    mul_417: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_186, view_124);  view_124 = None
    sum_99: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_416, [0, 2, 3], True);  mul_416 = None
    view_260: "f32[512]" = torch.ops.aten.view.default(sum_99, [512]);  sum_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_273: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_417, [0, 2, 3, 1]);  mul_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_261: "f32[1568, 512]" = torch.ops.aten.view.default(permute_273, [1568, 512]);  permute_273 = None
    permute_274: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_104, [1, 0]);  permute_104 = None
    mm_46: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_261, permute_274);  permute_274 = None
    permute_275: "f32[512, 1568]" = torch.ops.aten.permute.default(view_261, [1, 0])
    mm_47: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_275, view_122);  permute_275 = view_122 = None
    permute_276: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_100: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_261, [0], True);  view_261 = None
    view_262: "f32[512]" = torch.ops.aten.view.default(sum_100, [512]);  sum_100 = None
    permute_277: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_276, [1, 0]);  permute_276 = None
    view_263: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(mm_46, [8, 14, 14, 2048]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_418: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_121, 0.7071067811865476)
    erf_47: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_418);  mul_418 = None
    add_187: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_47, 1);  erf_47 = None
    mul_419: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_187, 0.5);  add_187 = None
    mul_420: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_121, view_121)
    mul_421: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_420, -0.5);  mul_420 = None
    exp_11: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_421);  mul_421 = None
    mul_422: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
    mul_423: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_121, mul_422);  view_121 = mul_422 = None
    add_188: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_419, mul_423);  mul_419 = mul_423 = None
    mul_424: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_263, add_188);  view_263 = add_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_264: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_424, [1568, 2048]);  mul_424 = None
    permute_278: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
    mm_48: "f32[1568, 512]" = torch.ops.aten.mm.default(view_264, permute_278);  permute_278 = None
    permute_279: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_264, [1, 0])
    mm_49: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_279, view_120);  permute_279 = view_120 = None
    permute_280: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_101: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_264, [0], True);  view_264 = None
    view_265: "f32[2048]" = torch.ops.aten.view.default(sum_101, [2048]);  sum_101 = None
    permute_281: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_280, [1, 0]);  permute_280 = None
    view_266: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(mm_48, [8, 14, 14, 512]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_125: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
    sub_80: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_125, getitem_55);  clone_125 = getitem_55 = None
    mul_425: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_27);  sub_80 = None
    mul_426: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_266, primals_79);  primals_79 = None
    mul_427: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_426, 512)
    sum_102: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_426, [3], True)
    mul_428: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_426, mul_425);  mul_426 = None
    sum_103: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_428, [3], True);  mul_428 = None
    mul_429: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_425, sum_103);  sum_103 = None
    sub_81: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_427, sum_102);  mul_427 = sum_102 = None
    sub_82: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_81, mul_429);  sub_81 = mul_429 = None
    div_14: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 512);  rsqrt_27 = None
    mul_430: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_14, sub_82);  div_14 = sub_82 = None
    mul_431: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_266, mul_425);  mul_425 = None
    sum_104: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_431, [0, 1, 2]);  mul_431 = None
    sum_105: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_266, [0, 1, 2]);  view_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_282: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_430, [0, 3, 1, 2]);  mul_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_106: "f32[512]" = torch.ops.aten.sum.dim_IntList(permute_282, [0, 2, 3])
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(permute_282, add_101, primals_269, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False]);  permute_282 = add_101 = primals_269 = None
    getitem_118: "f32[8, 512, 14, 14]" = convolution_backward_12[0]
    getitem_119: "f32[512, 1, 7, 7]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_189: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_186, getitem_118);  add_186 = getitem_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_432: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_189, permute_101);  permute_101 = None
    mul_433: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_189, view_119);  view_119 = None
    sum_107: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_432, [0, 2, 3], True);  mul_432 = None
    view_267: "f32[512]" = torch.ops.aten.view.default(sum_107, [512]);  sum_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_283: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_433, [0, 2, 3, 1]);  mul_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_268: "f32[1568, 512]" = torch.ops.aten.view.default(permute_283, [1568, 512]);  permute_283 = None
    permute_284: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
    mm_50: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_268, permute_284);  permute_284 = None
    permute_285: "f32[512, 1568]" = torch.ops.aten.permute.default(view_268, [1, 0])
    mm_51: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_285, view_117);  permute_285 = view_117 = None
    permute_286: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_108: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_268, [0], True);  view_268 = None
    view_269: "f32[512]" = torch.ops.aten.view.default(sum_108, [512]);  sum_108 = None
    permute_287: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_286, [1, 0]);  permute_286 = None
    view_270: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(mm_50, [8, 14, 14, 2048]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_434: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_116, 0.7071067811865476)
    erf_48: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_434);  mul_434 = None
    add_190: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_48, 1);  erf_48 = None
    mul_435: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_190, 0.5);  add_190 = None
    mul_436: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_116, view_116)
    mul_437: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_436, -0.5);  mul_436 = None
    exp_12: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_437);  mul_437 = None
    mul_438: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
    mul_439: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_116, mul_438);  view_116 = mul_438 = None
    add_191: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_435, mul_439);  mul_435 = mul_439 = None
    mul_440: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_270, add_191);  view_270 = add_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_271: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_440, [1568, 2048]);  mul_440 = None
    permute_288: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    mm_52: "f32[1568, 512]" = torch.ops.aten.mm.default(view_271, permute_288);  permute_288 = None
    permute_289: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_271, [1, 0])
    mm_53: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_289, view_115);  permute_289 = view_115 = None
    permute_290: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_109: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_271, [0], True);  view_271 = None
    view_272: "f32[2048]" = torch.ops.aten.view.default(sum_109, [2048]);  sum_109 = None
    permute_291: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_290, [1, 0]);  permute_290 = None
    view_273: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(mm_52, [8, 14, 14, 512]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_126: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_98, memory_format = torch.contiguous_format);  permute_98 = None
    sub_83: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_126, getitem_53);  clone_126 = getitem_53 = None
    mul_441: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_26);  sub_83 = None
    mul_442: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_273, primals_76);  primals_76 = None
    mul_443: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_442, 512)
    sum_110: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_442, [3], True)
    mul_444: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_442, mul_441);  mul_442 = None
    sum_111: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_444, [3], True);  mul_444 = None
    mul_445: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_441, sum_111);  sum_111 = None
    sub_84: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_443, sum_110);  mul_443 = sum_110 = None
    sub_85: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_84, mul_445);  sub_84 = mul_445 = None
    div_15: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 512);  rsqrt_26 = None
    mul_446: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_15, sub_85);  div_15 = sub_85 = None
    mul_447: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_273, mul_441);  mul_441 = None
    sum_112: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_447, [0, 1, 2]);  mul_447 = None
    sum_113: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_273, [0, 1, 2]);  view_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_292: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_446, [0, 3, 1, 2]);  mul_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_114: "f32[512]" = torch.ops.aten.sum.dim_IntList(permute_292, [0, 2, 3])
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(permute_292, add_97, primals_263, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False]);  permute_292 = add_97 = primals_263 = None
    getitem_121: "f32[8, 512, 14, 14]" = convolution_backward_13[0]
    getitem_122: "f32[512, 1, 7, 7]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_192: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_189, getitem_121);  add_189 = getitem_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_448: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_192, permute_97);  permute_97 = None
    mul_449: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_192, view_114);  view_114 = None
    sum_115: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_448, [0, 2, 3], True);  mul_448 = None
    view_274: "f32[512]" = torch.ops.aten.view.default(sum_115, [512]);  sum_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_293: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_449, [0, 2, 3, 1]);  mul_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_275: "f32[1568, 512]" = torch.ops.aten.view.default(permute_293, [1568, 512]);  permute_293 = None
    permute_294: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    mm_54: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_275, permute_294);  permute_294 = None
    permute_295: "f32[512, 1568]" = torch.ops.aten.permute.default(view_275, [1, 0])
    mm_55: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_295, view_112);  permute_295 = view_112 = None
    permute_296: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_116: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_275, [0], True);  view_275 = None
    view_276: "f32[512]" = torch.ops.aten.view.default(sum_116, [512]);  sum_116 = None
    permute_297: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_296, [1, 0]);  permute_296 = None
    view_277: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(mm_54, [8, 14, 14, 2048]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_450: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_111, 0.7071067811865476)
    erf_49: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_450);  mul_450 = None
    add_193: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_49, 1);  erf_49 = None
    mul_451: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_193, 0.5);  add_193 = None
    mul_452: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_111, view_111)
    mul_453: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_452, -0.5);  mul_452 = None
    exp_13: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_453);  mul_453 = None
    mul_454: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
    mul_455: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_111, mul_454);  view_111 = mul_454 = None
    add_194: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_451, mul_455);  mul_451 = mul_455 = None
    mul_456: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_277, add_194);  view_277 = add_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_278: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_456, [1568, 2048]);  mul_456 = None
    permute_298: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_95, [1, 0]);  permute_95 = None
    mm_56: "f32[1568, 512]" = torch.ops.aten.mm.default(view_278, permute_298);  permute_298 = None
    permute_299: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_278, [1, 0])
    mm_57: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_299, view_110);  permute_299 = view_110 = None
    permute_300: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_117: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_278, [0], True);  view_278 = None
    view_279: "f32[2048]" = torch.ops.aten.view.default(sum_117, [2048]);  sum_117 = None
    permute_301: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_300, [1, 0]);  permute_300 = None
    view_280: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(mm_56, [8, 14, 14, 512]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_127: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_94, memory_format = torch.contiguous_format);  permute_94 = None
    sub_86: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_127, getitem_51);  clone_127 = getitem_51 = None
    mul_457: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_25);  sub_86 = None
    mul_458: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_280, primals_73);  primals_73 = None
    mul_459: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_458, 512)
    sum_118: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_458, [3], True)
    mul_460: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_458, mul_457);  mul_458 = None
    sum_119: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_460, [3], True);  mul_460 = None
    mul_461: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_457, sum_119);  sum_119 = None
    sub_87: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_459, sum_118);  mul_459 = sum_118 = None
    sub_88: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_87, mul_461);  sub_87 = mul_461 = None
    div_16: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 512);  rsqrt_25 = None
    mul_462: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_16, sub_88);  div_16 = sub_88 = None
    mul_463: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_280, mul_457);  mul_457 = None
    sum_120: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_463, [0, 1, 2]);  mul_463 = None
    sum_121: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_280, [0, 1, 2]);  view_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_302: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_462, [0, 3, 1, 2]);  mul_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_122: "f32[512]" = torch.ops.aten.sum.dim_IntList(permute_302, [0, 2, 3])
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(permute_302, add_93, primals_257, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False]);  permute_302 = add_93 = primals_257 = None
    getitem_124: "f32[8, 512, 14, 14]" = convolution_backward_14[0]
    getitem_125: "f32[512, 1, 7, 7]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_195: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_192, getitem_124);  add_192 = getitem_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_464: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_195, permute_93);  permute_93 = None
    mul_465: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_195, view_109);  view_109 = None
    sum_123: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_464, [0, 2, 3], True);  mul_464 = None
    view_281: "f32[512]" = torch.ops.aten.view.default(sum_123, [512]);  sum_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_303: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_465, [0, 2, 3, 1]);  mul_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_282: "f32[1568, 512]" = torch.ops.aten.view.default(permute_303, [1568, 512]);  permute_303 = None
    permute_304: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
    mm_58: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_282, permute_304);  permute_304 = None
    permute_305: "f32[512, 1568]" = torch.ops.aten.permute.default(view_282, [1, 0])
    mm_59: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_305, view_107);  permute_305 = view_107 = None
    permute_306: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_124: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_282, [0], True);  view_282 = None
    view_283: "f32[512]" = torch.ops.aten.view.default(sum_124, [512]);  sum_124 = None
    permute_307: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_306, [1, 0]);  permute_306 = None
    view_284: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(mm_58, [8, 14, 14, 2048]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_466: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_106, 0.7071067811865476)
    erf_50: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_466);  mul_466 = None
    add_196: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_50, 1);  erf_50 = None
    mul_467: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_196, 0.5);  add_196 = None
    mul_468: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_106, view_106)
    mul_469: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_468, -0.5);  mul_468 = None
    exp_14: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_469);  mul_469 = None
    mul_470: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
    mul_471: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_106, mul_470);  view_106 = mul_470 = None
    add_197: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_467, mul_471);  mul_467 = mul_471 = None
    mul_472: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_284, add_197);  view_284 = add_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_285: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_472, [1568, 2048]);  mul_472 = None
    permute_308: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
    mm_60: "f32[1568, 512]" = torch.ops.aten.mm.default(view_285, permute_308);  permute_308 = None
    permute_309: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_285, [1, 0])
    mm_61: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_309, view_105);  permute_309 = view_105 = None
    permute_310: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_125: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_285, [0], True);  view_285 = None
    view_286: "f32[2048]" = torch.ops.aten.view.default(sum_125, [2048]);  sum_125 = None
    permute_311: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_310, [1, 0]);  permute_310 = None
    view_287: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(mm_60, [8, 14, 14, 512]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_128: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
    sub_89: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_128, getitem_49);  clone_128 = getitem_49 = None
    mul_473: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_24);  sub_89 = None
    mul_474: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_287, primals_70);  primals_70 = None
    mul_475: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_474, 512)
    sum_126: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_474, [3], True)
    mul_476: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_474, mul_473);  mul_474 = None
    sum_127: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_476, [3], True);  mul_476 = None
    mul_477: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_473, sum_127);  sum_127 = None
    sub_90: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_475, sum_126);  mul_475 = sum_126 = None
    sub_91: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_90, mul_477);  sub_90 = mul_477 = None
    div_17: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 512);  rsqrt_24 = None
    mul_478: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_17, sub_91);  div_17 = sub_91 = None
    mul_479: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_287, mul_473);  mul_473 = None
    sum_128: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_479, [0, 1, 2]);  mul_479 = None
    sum_129: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_287, [0, 1, 2]);  view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_312: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_478, [0, 3, 1, 2]);  mul_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_130: "f32[512]" = torch.ops.aten.sum.dim_IntList(permute_312, [0, 2, 3])
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(permute_312, add_89, primals_251, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False]);  permute_312 = add_89 = primals_251 = None
    getitem_127: "f32[8, 512, 14, 14]" = convolution_backward_15[0]
    getitem_128: "f32[512, 1, 7, 7]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_198: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_195, getitem_127);  add_195 = getitem_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_480: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_198, permute_89);  permute_89 = None
    mul_481: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_198, view_104);  view_104 = None
    sum_131: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_480, [0, 2, 3], True);  mul_480 = None
    view_288: "f32[512]" = torch.ops.aten.view.default(sum_131, [512]);  sum_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_313: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_481, [0, 2, 3, 1]);  mul_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_289: "f32[1568, 512]" = torch.ops.aten.view.default(permute_313, [1568, 512]);  permute_313 = None
    permute_314: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    mm_62: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_289, permute_314);  permute_314 = None
    permute_315: "f32[512, 1568]" = torch.ops.aten.permute.default(view_289, [1, 0])
    mm_63: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_315, view_102);  permute_315 = view_102 = None
    permute_316: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_132: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_289, [0], True);  view_289 = None
    view_290: "f32[512]" = torch.ops.aten.view.default(sum_132, [512]);  sum_132 = None
    permute_317: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_316, [1, 0]);  permute_316 = None
    view_291: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(mm_62, [8, 14, 14, 2048]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_482: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_101, 0.7071067811865476)
    erf_51: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_482);  mul_482 = None
    add_199: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_51, 1);  erf_51 = None
    mul_483: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_199, 0.5);  add_199 = None
    mul_484: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_101, view_101)
    mul_485: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_484, -0.5);  mul_484 = None
    exp_15: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_485);  mul_485 = None
    mul_486: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
    mul_487: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_101, mul_486);  view_101 = mul_486 = None
    add_200: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_483, mul_487);  mul_483 = mul_487 = None
    mul_488: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_291, add_200);  view_291 = add_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_292: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_488, [1568, 2048]);  mul_488 = None
    permute_318: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    mm_64: "f32[1568, 512]" = torch.ops.aten.mm.default(view_292, permute_318);  permute_318 = None
    permute_319: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_292, [1, 0])
    mm_65: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_319, view_100);  permute_319 = view_100 = None
    permute_320: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_133: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_292, [0], True);  view_292 = None
    view_293: "f32[2048]" = torch.ops.aten.view.default(sum_133, [2048]);  sum_133 = None
    permute_321: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_320, [1, 0]);  permute_320 = None
    view_294: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(mm_64, [8, 14, 14, 512]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_129: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_86, memory_format = torch.contiguous_format);  permute_86 = None
    sub_92: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_129, getitem_47);  clone_129 = getitem_47 = None
    mul_489: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_23);  sub_92 = None
    mul_490: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_294, primals_67);  primals_67 = None
    mul_491: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_490, 512)
    sum_134: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_490, [3], True)
    mul_492: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_490, mul_489);  mul_490 = None
    sum_135: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_492, [3], True);  mul_492 = None
    mul_493: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_489, sum_135);  sum_135 = None
    sub_93: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_491, sum_134);  mul_491 = sum_134 = None
    sub_94: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_93, mul_493);  sub_93 = mul_493 = None
    div_18: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 512);  rsqrt_23 = None
    mul_494: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_18, sub_94);  div_18 = sub_94 = None
    mul_495: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_294, mul_489);  mul_489 = None
    sum_136: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_495, [0, 1, 2]);  mul_495 = None
    sum_137: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_294, [0, 1, 2]);  view_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_322: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_494, [0, 3, 1, 2]);  mul_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_138: "f32[512]" = torch.ops.aten.sum.dim_IntList(permute_322, [0, 2, 3])
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(permute_322, add_85, primals_245, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False]);  permute_322 = add_85 = primals_245 = None
    getitem_130: "f32[8, 512, 14, 14]" = convolution_backward_16[0]
    getitem_131: "f32[512, 1, 7, 7]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_201: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_198, getitem_130);  add_198 = getitem_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_496: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_201, permute_85);  permute_85 = None
    mul_497: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_201, view_99);  view_99 = None
    sum_139: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_496, [0, 2, 3], True);  mul_496 = None
    view_295: "f32[512]" = torch.ops.aten.view.default(sum_139, [512]);  sum_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_323: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_497, [0, 2, 3, 1]);  mul_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_296: "f32[1568, 512]" = torch.ops.aten.view.default(permute_323, [1568, 512]);  permute_323 = None
    permute_324: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_84, [1, 0]);  permute_84 = None
    mm_66: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_296, permute_324);  permute_324 = None
    permute_325: "f32[512, 1568]" = torch.ops.aten.permute.default(view_296, [1, 0])
    mm_67: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_325, view_97);  permute_325 = view_97 = None
    permute_326: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_140: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_296, [0], True);  view_296 = None
    view_297: "f32[512]" = torch.ops.aten.view.default(sum_140, [512]);  sum_140 = None
    permute_327: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_326, [1, 0]);  permute_326 = None
    view_298: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(mm_66, [8, 14, 14, 2048]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_498: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_96, 0.7071067811865476)
    erf_52: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_498);  mul_498 = None
    add_202: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_52, 1);  erf_52 = None
    mul_499: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_202, 0.5);  add_202 = None
    mul_500: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_96, view_96)
    mul_501: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_500, -0.5);  mul_500 = None
    exp_16: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_501);  mul_501 = None
    mul_502: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
    mul_503: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_96, mul_502);  view_96 = mul_502 = None
    add_203: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_499, mul_503);  mul_499 = mul_503 = None
    mul_504: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_298, add_203);  view_298 = add_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_299: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_504, [1568, 2048]);  mul_504 = None
    permute_328: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
    mm_68: "f32[1568, 512]" = torch.ops.aten.mm.default(view_299, permute_328);  permute_328 = None
    permute_329: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_299, [1, 0])
    mm_69: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_329, view_95);  permute_329 = view_95 = None
    permute_330: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_141: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_299, [0], True);  view_299 = None
    view_300: "f32[2048]" = torch.ops.aten.view.default(sum_141, [2048]);  sum_141 = None
    permute_331: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_330, [1, 0]);  permute_330 = None
    view_301: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(mm_68, [8, 14, 14, 512]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_130: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    sub_95: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_130, getitem_45);  clone_130 = getitem_45 = None
    mul_505: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_22);  sub_95 = None
    mul_506: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_301, primals_64);  primals_64 = None
    mul_507: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_506, 512)
    sum_142: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_506, [3], True)
    mul_508: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_506, mul_505);  mul_506 = None
    sum_143: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_508, [3], True);  mul_508 = None
    mul_509: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_505, sum_143);  sum_143 = None
    sub_96: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_507, sum_142);  mul_507 = sum_142 = None
    sub_97: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_96, mul_509);  sub_96 = mul_509 = None
    div_19: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 512);  rsqrt_22 = None
    mul_510: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_19, sub_97);  div_19 = sub_97 = None
    mul_511: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_301, mul_505);  mul_505 = None
    sum_144: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_511, [0, 1, 2]);  mul_511 = None
    sum_145: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_301, [0, 1, 2]);  view_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_332: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_510, [0, 3, 1, 2]);  mul_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_146: "f32[512]" = torch.ops.aten.sum.dim_IntList(permute_332, [0, 2, 3])
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(permute_332, add_81, primals_239, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False]);  permute_332 = add_81 = primals_239 = None
    getitem_133: "f32[8, 512, 14, 14]" = convolution_backward_17[0]
    getitem_134: "f32[512, 1, 7, 7]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_204: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_201, getitem_133);  add_201 = getitem_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_512: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_204, permute_81);  permute_81 = None
    mul_513: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_204, view_94);  view_94 = None
    sum_147: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_512, [0, 2, 3], True);  mul_512 = None
    view_302: "f32[512]" = torch.ops.aten.view.default(sum_147, [512]);  sum_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_333: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_513, [0, 2, 3, 1]);  mul_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_303: "f32[1568, 512]" = torch.ops.aten.view.default(permute_333, [1568, 512]);  permute_333 = None
    permute_334: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
    mm_70: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_303, permute_334);  permute_334 = None
    permute_335: "f32[512, 1568]" = torch.ops.aten.permute.default(view_303, [1, 0])
    mm_71: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_335, view_92);  permute_335 = view_92 = None
    permute_336: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_148: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_303, [0], True);  view_303 = None
    view_304: "f32[512]" = torch.ops.aten.view.default(sum_148, [512]);  sum_148 = None
    permute_337: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_336, [1, 0]);  permute_336 = None
    view_305: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(mm_70, [8, 14, 14, 2048]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_514: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_91, 0.7071067811865476)
    erf_53: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_514);  mul_514 = None
    add_205: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_53, 1);  erf_53 = None
    mul_515: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_205, 0.5);  add_205 = None
    mul_516: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_91, view_91)
    mul_517: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_516, -0.5);  mul_516 = None
    exp_17: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_517);  mul_517 = None
    mul_518: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
    mul_519: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_91, mul_518);  view_91 = mul_518 = None
    add_206: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_515, mul_519);  mul_515 = mul_519 = None
    mul_520: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_305, add_206);  view_305 = add_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_306: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_520, [1568, 2048]);  mul_520 = None
    permute_338: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
    mm_72: "f32[1568, 512]" = torch.ops.aten.mm.default(view_306, permute_338);  permute_338 = None
    permute_339: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_306, [1, 0])
    mm_73: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_339, view_90);  permute_339 = view_90 = None
    permute_340: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_149: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_306, [0], True);  view_306 = None
    view_307: "f32[2048]" = torch.ops.aten.view.default(sum_149, [2048]);  sum_149 = None
    permute_341: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_340, [1, 0]);  permute_340 = None
    view_308: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(mm_72, [8, 14, 14, 512]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_131: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_78, memory_format = torch.contiguous_format);  permute_78 = None
    sub_98: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_131, getitem_43);  clone_131 = getitem_43 = None
    mul_521: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_98, rsqrt_21);  sub_98 = None
    mul_522: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_308, primals_61);  primals_61 = None
    mul_523: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_522, 512)
    sum_150: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_522, [3], True)
    mul_524: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_522, mul_521);  mul_522 = None
    sum_151: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_524, [3], True);  mul_524 = None
    mul_525: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_521, sum_151);  sum_151 = None
    sub_99: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_523, sum_150);  mul_523 = sum_150 = None
    sub_100: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_99, mul_525);  sub_99 = mul_525 = None
    div_20: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 512);  rsqrt_21 = None
    mul_526: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_20, sub_100);  div_20 = sub_100 = None
    mul_527: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_308, mul_521);  mul_521 = None
    sum_152: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_527, [0, 1, 2]);  mul_527 = None
    sum_153: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_308, [0, 1, 2]);  view_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_342: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_526, [0, 3, 1, 2]);  mul_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_154: "f32[512]" = torch.ops.aten.sum.dim_IntList(permute_342, [0, 2, 3])
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(permute_342, add_77, primals_233, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False]);  permute_342 = add_77 = primals_233 = None
    getitem_136: "f32[8, 512, 14, 14]" = convolution_backward_18[0]
    getitem_137: "f32[512, 1, 7, 7]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_207: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_204, getitem_136);  add_204 = getitem_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_528: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_207, permute_77);  permute_77 = None
    mul_529: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_207, view_89);  view_89 = None
    sum_155: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_528, [0, 2, 3], True);  mul_528 = None
    view_309: "f32[512]" = torch.ops.aten.view.default(sum_155, [512]);  sum_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_343: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_529, [0, 2, 3, 1]);  mul_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_310: "f32[1568, 512]" = torch.ops.aten.view.default(permute_343, [1568, 512]);  permute_343 = None
    permute_344: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    mm_74: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_310, permute_344);  permute_344 = None
    permute_345: "f32[512, 1568]" = torch.ops.aten.permute.default(view_310, [1, 0])
    mm_75: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_345, view_87);  permute_345 = view_87 = None
    permute_346: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_156: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_310, [0], True);  view_310 = None
    view_311: "f32[512]" = torch.ops.aten.view.default(sum_156, [512]);  sum_156 = None
    permute_347: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_346, [1, 0]);  permute_346 = None
    view_312: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(mm_74, [8, 14, 14, 2048]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_530: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_86, 0.7071067811865476)
    erf_54: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_530);  mul_530 = None
    add_208: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_54, 1);  erf_54 = None
    mul_531: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_208, 0.5);  add_208 = None
    mul_532: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_86, view_86)
    mul_533: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_532, -0.5);  mul_532 = None
    exp_18: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_533);  mul_533 = None
    mul_534: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_535: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_86, mul_534);  view_86 = mul_534 = None
    add_209: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_531, mul_535);  mul_531 = mul_535 = None
    mul_536: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_312, add_209);  view_312 = add_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_313: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_536, [1568, 2048]);  mul_536 = None
    permute_348: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    mm_76: "f32[1568, 512]" = torch.ops.aten.mm.default(view_313, permute_348);  permute_348 = None
    permute_349: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_313, [1, 0])
    mm_77: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_349, view_85);  permute_349 = view_85 = None
    permute_350: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_157: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_313, [0], True);  view_313 = None
    view_314: "f32[2048]" = torch.ops.aten.view.default(sum_157, [2048]);  sum_157 = None
    permute_351: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_350, [1, 0]);  permute_350 = None
    view_315: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(mm_76, [8, 14, 14, 512]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_132: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None
    sub_101: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_132, getitem_41);  clone_132 = getitem_41 = None
    mul_537: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_101, rsqrt_20);  sub_101 = None
    mul_538: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_315, primals_58);  primals_58 = None
    mul_539: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_538, 512)
    sum_158: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_538, [3], True)
    mul_540: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_538, mul_537);  mul_538 = None
    sum_159: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_540, [3], True);  mul_540 = None
    mul_541: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_537, sum_159);  sum_159 = None
    sub_102: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_539, sum_158);  mul_539 = sum_158 = None
    sub_103: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_102, mul_541);  sub_102 = mul_541 = None
    div_21: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 512);  rsqrt_20 = None
    mul_542: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_21, sub_103);  div_21 = sub_103 = None
    mul_543: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_315, mul_537);  mul_537 = None
    sum_160: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_543, [0, 1, 2]);  mul_543 = None
    sum_161: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_315, [0, 1, 2]);  view_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_352: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_542, [0, 3, 1, 2]);  mul_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_162: "f32[512]" = torch.ops.aten.sum.dim_IntList(permute_352, [0, 2, 3])
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(permute_352, add_73, primals_227, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False]);  permute_352 = add_73 = primals_227 = None
    getitem_139: "f32[8, 512, 14, 14]" = convolution_backward_19[0]
    getitem_140: "f32[512, 1, 7, 7]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_210: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_207, getitem_139);  add_207 = getitem_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_544: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_210, permute_73);  permute_73 = None
    mul_545: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_210, view_84);  view_84 = None
    sum_163: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_544, [0, 2, 3], True);  mul_544 = None
    view_316: "f32[512]" = torch.ops.aten.view.default(sum_163, [512]);  sum_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_353: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_545, [0, 2, 3, 1]);  mul_545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_317: "f32[1568, 512]" = torch.ops.aten.view.default(permute_353, [1568, 512]);  permute_353 = None
    permute_354: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
    mm_78: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_317, permute_354);  permute_354 = None
    permute_355: "f32[512, 1568]" = torch.ops.aten.permute.default(view_317, [1, 0])
    mm_79: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_355, view_82);  permute_355 = view_82 = None
    permute_356: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_164: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_317, [0], True);  view_317 = None
    view_318: "f32[512]" = torch.ops.aten.view.default(sum_164, [512]);  sum_164 = None
    permute_357: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_356, [1, 0]);  permute_356 = None
    view_319: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(mm_78, [8, 14, 14, 2048]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_546: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_81, 0.7071067811865476)
    erf_55: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_546);  mul_546 = None
    add_211: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_55, 1);  erf_55 = None
    mul_547: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_211, 0.5);  add_211 = None
    mul_548: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_81, view_81)
    mul_549: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_548, -0.5);  mul_548 = None
    exp_19: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_549);  mul_549 = None
    mul_550: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_551: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_81, mul_550);  view_81 = mul_550 = None
    add_212: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_547, mul_551);  mul_547 = mul_551 = None
    mul_552: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_319, add_212);  view_319 = add_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_320: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_552, [1568, 2048]);  mul_552 = None
    permute_358: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
    mm_80: "f32[1568, 512]" = torch.ops.aten.mm.default(view_320, permute_358);  permute_358 = None
    permute_359: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_320, [1, 0])
    mm_81: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_359, view_80);  permute_359 = view_80 = None
    permute_360: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_165: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_320, [0], True);  view_320 = None
    view_321: "f32[2048]" = torch.ops.aten.view.default(sum_165, [2048]);  sum_165 = None
    permute_361: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_360, [1, 0]);  permute_360 = None
    view_322: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(mm_80, [8, 14, 14, 512]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_133: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
    sub_104: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_133, getitem_39);  clone_133 = getitem_39 = None
    mul_553: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_104, rsqrt_19);  sub_104 = None
    mul_554: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_322, primals_55);  primals_55 = None
    mul_555: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_554, 512)
    sum_166: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_554, [3], True)
    mul_556: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_554, mul_553);  mul_554 = None
    sum_167: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_556, [3], True);  mul_556 = None
    mul_557: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_553, sum_167);  sum_167 = None
    sub_105: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_555, sum_166);  mul_555 = sum_166 = None
    sub_106: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_105, mul_557);  sub_105 = mul_557 = None
    div_22: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 512);  rsqrt_19 = None
    mul_558: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_22, sub_106);  div_22 = sub_106 = None
    mul_559: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_322, mul_553);  mul_553 = None
    sum_168: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_559, [0, 1, 2]);  mul_559 = None
    sum_169: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_322, [0, 1, 2]);  view_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_362: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_558, [0, 3, 1, 2]);  mul_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_170: "f32[512]" = torch.ops.aten.sum.dim_IntList(permute_362, [0, 2, 3])
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(permute_362, add_69, primals_221, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False]);  permute_362 = add_69 = primals_221 = None
    getitem_142: "f32[8, 512, 14, 14]" = convolution_backward_20[0]
    getitem_143: "f32[512, 1, 7, 7]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_213: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_210, getitem_142);  add_210 = getitem_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_560: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_213, permute_69);  permute_69 = None
    mul_561: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_213, view_79);  view_79 = None
    sum_171: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_560, [0, 2, 3], True);  mul_560 = None
    view_323: "f32[512]" = torch.ops.aten.view.default(sum_171, [512]);  sum_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_363: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_561, [0, 2, 3, 1]);  mul_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_324: "f32[1568, 512]" = torch.ops.aten.view.default(permute_363, [1568, 512]);  permute_363 = None
    permute_364: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
    mm_82: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_324, permute_364);  permute_364 = None
    permute_365: "f32[512, 1568]" = torch.ops.aten.permute.default(view_324, [1, 0])
    mm_83: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_365, view_77);  permute_365 = view_77 = None
    permute_366: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_172: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_324, [0], True);  view_324 = None
    view_325: "f32[512]" = torch.ops.aten.view.default(sum_172, [512]);  sum_172 = None
    permute_367: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_366, [1, 0]);  permute_366 = None
    view_326: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(mm_82, [8, 14, 14, 2048]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_562: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_76, 0.7071067811865476)
    erf_56: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_562);  mul_562 = None
    add_214: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_56, 1);  erf_56 = None
    mul_563: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_214, 0.5);  add_214 = None
    mul_564: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_76, view_76)
    mul_565: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_564, -0.5);  mul_564 = None
    exp_20: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_565);  mul_565 = None
    mul_566: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_567: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_76, mul_566);  view_76 = mul_566 = None
    add_215: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_563, mul_567);  mul_563 = mul_567 = None
    mul_568: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_326, add_215);  view_326 = add_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_327: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_568, [1568, 2048]);  mul_568 = None
    permute_368: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    mm_84: "f32[1568, 512]" = torch.ops.aten.mm.default(view_327, permute_368);  permute_368 = None
    permute_369: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_327, [1, 0])
    mm_85: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_369, view_75);  permute_369 = view_75 = None
    permute_370: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_173: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_327, [0], True);  view_327 = None
    view_328: "f32[2048]" = torch.ops.aten.view.default(sum_173, [2048]);  sum_173 = None
    permute_371: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_370, [1, 0]);  permute_370 = None
    view_329: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(mm_84, [8, 14, 14, 512]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_134: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_66, memory_format = torch.contiguous_format);  permute_66 = None
    sub_107: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_134, getitem_37);  clone_134 = getitem_37 = None
    mul_569: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_107, rsqrt_18);  sub_107 = None
    mul_570: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_329, primals_52);  primals_52 = None
    mul_571: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_570, 512)
    sum_174: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_570, [3], True)
    mul_572: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_570, mul_569);  mul_570 = None
    sum_175: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_572, [3], True);  mul_572 = None
    mul_573: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_569, sum_175);  sum_175 = None
    sub_108: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_571, sum_174);  mul_571 = sum_174 = None
    sub_109: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_108, mul_573);  sub_108 = mul_573 = None
    div_23: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 512);  rsqrt_18 = None
    mul_574: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_23, sub_109);  div_23 = sub_109 = None
    mul_575: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_329, mul_569);  mul_569 = None
    sum_176: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_575, [0, 1, 2]);  mul_575 = None
    sum_177: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_329, [0, 1, 2]);  view_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_372: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_574, [0, 3, 1, 2]);  mul_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_178: "f32[512]" = torch.ops.aten.sum.dim_IntList(permute_372, [0, 2, 3])
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(permute_372, add_65, primals_215, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False]);  permute_372 = add_65 = primals_215 = None
    getitem_145: "f32[8, 512, 14, 14]" = convolution_backward_21[0]
    getitem_146: "f32[512, 1, 7, 7]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_216: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_213, getitem_145);  add_213 = getitem_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_576: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_216, permute_65);  permute_65 = None
    mul_577: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_216, view_74);  view_74 = None
    sum_179: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_576, [0, 2, 3], True);  mul_576 = None
    view_330: "f32[512]" = torch.ops.aten.view.default(sum_179, [512]);  sum_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_373: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_577, [0, 2, 3, 1]);  mul_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_331: "f32[1568, 512]" = torch.ops.aten.view.default(permute_373, [1568, 512]);  permute_373 = None
    permute_374: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    mm_86: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_331, permute_374);  permute_374 = None
    permute_375: "f32[512, 1568]" = torch.ops.aten.permute.default(view_331, [1, 0])
    mm_87: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_375, view_72);  permute_375 = view_72 = None
    permute_376: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_180: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_331, [0], True);  view_331 = None
    view_332: "f32[512]" = torch.ops.aten.view.default(sum_180, [512]);  sum_180 = None
    permute_377: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_376, [1, 0]);  permute_376 = None
    view_333: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(mm_86, [8, 14, 14, 2048]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_578: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_71, 0.7071067811865476)
    erf_57: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_578);  mul_578 = None
    add_217: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_57, 1);  erf_57 = None
    mul_579: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_217, 0.5);  add_217 = None
    mul_580: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_71, view_71)
    mul_581: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_580, -0.5);  mul_580 = None
    exp_21: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_581);  mul_581 = None
    mul_582: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_583: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_71, mul_582);  view_71 = mul_582 = None
    add_218: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_579, mul_583);  mul_579 = mul_583 = None
    mul_584: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_333, add_218);  view_333 = add_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_334: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_584, [1568, 2048]);  mul_584 = None
    permute_378: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    mm_88: "f32[1568, 512]" = torch.ops.aten.mm.default(view_334, permute_378);  permute_378 = None
    permute_379: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_334, [1, 0])
    mm_89: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_379, view_70);  permute_379 = view_70 = None
    permute_380: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_181: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_334, [0], True);  view_334 = None
    view_335: "f32[2048]" = torch.ops.aten.view.default(sum_181, [2048]);  sum_181 = None
    permute_381: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_380, [1, 0]);  permute_380 = None
    view_336: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(mm_88, [8, 14, 14, 512]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_135: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    sub_110: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_135, getitem_35);  clone_135 = getitem_35 = None
    mul_585: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_110, rsqrt_17);  sub_110 = None
    mul_586: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_336, primals_49);  primals_49 = None
    mul_587: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_586, 512)
    sum_182: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_586, [3], True)
    mul_588: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_586, mul_585);  mul_586 = None
    sum_183: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_588, [3], True);  mul_588 = None
    mul_589: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_585, sum_183);  sum_183 = None
    sub_111: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_587, sum_182);  mul_587 = sum_182 = None
    sub_112: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_111, mul_589);  sub_111 = mul_589 = None
    div_24: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 512);  rsqrt_17 = None
    mul_590: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_24, sub_112);  div_24 = sub_112 = None
    mul_591: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_336, mul_585);  mul_585 = None
    sum_184: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_591, [0, 1, 2]);  mul_591 = None
    sum_185: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_336, [0, 1, 2]);  view_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_382: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_590, [0, 3, 1, 2]);  mul_590 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_186: "f32[512]" = torch.ops.aten.sum.dim_IntList(permute_382, [0, 2, 3])
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(permute_382, add_61, primals_209, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False]);  permute_382 = add_61 = primals_209 = None
    getitem_148: "f32[8, 512, 14, 14]" = convolution_backward_22[0]
    getitem_149: "f32[512, 1, 7, 7]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_219: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_216, getitem_148);  add_216 = getitem_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_592: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_219, permute_61);  permute_61 = None
    mul_593: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_219, view_69);  view_69 = None
    sum_187: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_592, [0, 2, 3], True);  mul_592 = None
    view_337: "f32[512]" = torch.ops.aten.view.default(sum_187, [512]);  sum_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_383: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_593, [0, 2, 3, 1]);  mul_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_338: "f32[1568, 512]" = torch.ops.aten.view.default(permute_383, [1568, 512]);  permute_383 = None
    permute_384: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
    mm_90: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_338, permute_384);  permute_384 = None
    permute_385: "f32[512, 1568]" = torch.ops.aten.permute.default(view_338, [1, 0])
    mm_91: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_385, view_67);  permute_385 = view_67 = None
    permute_386: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_188: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_338, [0], True);  view_338 = None
    view_339: "f32[512]" = torch.ops.aten.view.default(sum_188, [512]);  sum_188 = None
    permute_387: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_386, [1, 0]);  permute_386 = None
    view_340: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(mm_90, [8, 14, 14, 2048]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_594: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_66, 0.7071067811865476)
    erf_58: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_594);  mul_594 = None
    add_220: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_58, 1);  erf_58 = None
    mul_595: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_220, 0.5);  add_220 = None
    mul_596: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_66, view_66)
    mul_597: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_596, -0.5);  mul_596 = None
    exp_22: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_597);  mul_597 = None
    mul_598: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_599: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_66, mul_598);  view_66 = mul_598 = None
    add_221: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_595, mul_599);  mul_595 = mul_599 = None
    mul_600: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_340, add_221);  view_340 = add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_341: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_600, [1568, 2048]);  mul_600 = None
    permute_388: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
    mm_92: "f32[1568, 512]" = torch.ops.aten.mm.default(view_341, permute_388);  permute_388 = None
    permute_389: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_341, [1, 0])
    mm_93: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_389, view_65);  permute_389 = view_65 = None
    permute_390: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_189: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_341, [0], True);  view_341 = None
    view_342: "f32[2048]" = torch.ops.aten.view.default(sum_189, [2048]);  sum_189 = None
    permute_391: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_390, [1, 0]);  permute_390 = None
    view_343: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(mm_92, [8, 14, 14, 512]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_136: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_58, memory_format = torch.contiguous_format);  permute_58 = None
    sub_113: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_136, getitem_33);  clone_136 = getitem_33 = None
    mul_601: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_113, rsqrt_16);  sub_113 = None
    mul_602: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_343, primals_46);  primals_46 = None
    mul_603: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_602, 512)
    sum_190: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_602, [3], True)
    mul_604: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_602, mul_601);  mul_602 = None
    sum_191: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_604, [3], True);  mul_604 = None
    mul_605: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_601, sum_191);  sum_191 = None
    sub_114: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_603, sum_190);  mul_603 = sum_190 = None
    sub_115: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_114, mul_605);  sub_114 = mul_605 = None
    div_25: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 512);  rsqrt_16 = None
    mul_606: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_25, sub_115);  div_25 = sub_115 = None
    mul_607: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_343, mul_601);  mul_601 = None
    sum_192: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_607, [0, 1, 2]);  mul_607 = None
    sum_193: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_343, [0, 1, 2]);  view_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_392: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_606, [0, 3, 1, 2]);  mul_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_194: "f32[512]" = torch.ops.aten.sum.dim_IntList(permute_392, [0, 2, 3])
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(permute_392, add_57, primals_203, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False]);  permute_392 = add_57 = primals_203 = None
    getitem_151: "f32[8, 512, 14, 14]" = convolution_backward_23[0]
    getitem_152: "f32[512, 1, 7, 7]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_222: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_219, getitem_151);  add_219 = getitem_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_608: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_222, permute_57);  permute_57 = None
    mul_609: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_222, view_64);  view_64 = None
    sum_195: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_608, [0, 2, 3], True);  mul_608 = None
    view_344: "f32[512]" = torch.ops.aten.view.default(sum_195, [512]);  sum_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_393: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_609, [0, 2, 3, 1]);  mul_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_345: "f32[1568, 512]" = torch.ops.aten.view.default(permute_393, [1568, 512]);  permute_393 = None
    permute_394: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    mm_94: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_345, permute_394);  permute_394 = None
    permute_395: "f32[512, 1568]" = torch.ops.aten.permute.default(view_345, [1, 0])
    mm_95: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_395, view_62);  permute_395 = view_62 = None
    permute_396: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_196: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_345, [0], True);  view_345 = None
    view_346: "f32[512]" = torch.ops.aten.view.default(sum_196, [512]);  sum_196 = None
    permute_397: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_396, [1, 0]);  permute_396 = None
    view_347: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(mm_94, [8, 14, 14, 2048]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_610: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_61, 0.7071067811865476)
    erf_59: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_610);  mul_610 = None
    add_223: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_59, 1);  erf_59 = None
    mul_611: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_223, 0.5);  add_223 = None
    mul_612: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_61, view_61)
    mul_613: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_612, -0.5);  mul_612 = None
    exp_23: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_613);  mul_613 = None
    mul_614: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_615: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_61, mul_614);  view_61 = mul_614 = None
    add_224: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_611, mul_615);  mul_611 = mul_615 = None
    mul_616: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_347, add_224);  view_347 = add_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_348: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_616, [1568, 2048]);  mul_616 = None
    permute_398: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    mm_96: "f32[1568, 512]" = torch.ops.aten.mm.default(view_348, permute_398);  permute_398 = None
    permute_399: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_348, [1, 0])
    mm_97: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_399, view_60);  permute_399 = view_60 = None
    permute_400: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_197: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_348, [0], True);  view_348 = None
    view_349: "f32[2048]" = torch.ops.aten.view.default(sum_197, [2048]);  sum_197 = None
    permute_401: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_400, [1, 0]);  permute_400 = None
    view_350: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(mm_96, [8, 14, 14, 512]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_137: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_54, memory_format = torch.contiguous_format);  permute_54 = None
    sub_116: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_137, getitem_31);  clone_137 = getitem_31 = None
    mul_617: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_116, rsqrt_15);  sub_116 = None
    mul_618: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_350, primals_43);  primals_43 = None
    mul_619: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_618, 512)
    sum_198: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_618, [3], True)
    mul_620: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_618, mul_617);  mul_618 = None
    sum_199: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_620, [3], True);  mul_620 = None
    mul_621: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_617, sum_199);  sum_199 = None
    sub_117: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_619, sum_198);  mul_619 = sum_198 = None
    sub_118: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_117, mul_621);  sub_117 = mul_621 = None
    div_26: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 512);  rsqrt_15 = None
    mul_622: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_26, sub_118);  div_26 = sub_118 = None
    mul_623: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_350, mul_617);  mul_617 = None
    sum_200: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_623, [0, 1, 2]);  mul_623 = None
    sum_201: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_350, [0, 1, 2]);  view_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_402: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_622, [0, 3, 1, 2]);  mul_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_202: "f32[512]" = torch.ops.aten.sum.dim_IntList(permute_402, [0, 2, 3])
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(permute_402, add_53, primals_197, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False]);  permute_402 = add_53 = primals_197 = None
    getitem_154: "f32[8, 512, 14, 14]" = convolution_backward_24[0]
    getitem_155: "f32[512, 1, 7, 7]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_225: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_222, getitem_154);  add_222 = getitem_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_624: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_225, permute_53);  permute_53 = None
    mul_625: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_225, view_59);  view_59 = None
    sum_203: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_624, [0, 2, 3], True);  mul_624 = None
    view_351: "f32[512]" = torch.ops.aten.view.default(sum_203, [512]);  sum_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_403: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_625, [0, 2, 3, 1]);  mul_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_352: "f32[1568, 512]" = torch.ops.aten.view.default(permute_403, [1568, 512]);  permute_403 = None
    permute_404: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    mm_98: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_352, permute_404);  permute_404 = None
    permute_405: "f32[512, 1568]" = torch.ops.aten.permute.default(view_352, [1, 0])
    mm_99: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_405, view_57);  permute_405 = view_57 = None
    permute_406: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_204: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_352, [0], True);  view_352 = None
    view_353: "f32[512]" = torch.ops.aten.view.default(sum_204, [512]);  sum_204 = None
    permute_407: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_406, [1, 0]);  permute_406 = None
    view_354: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(mm_98, [8, 14, 14, 2048]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_626: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_56, 0.7071067811865476)
    erf_60: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_626);  mul_626 = None
    add_226: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_60, 1);  erf_60 = None
    mul_627: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_226, 0.5);  add_226 = None
    mul_628: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_56, view_56)
    mul_629: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_628, -0.5);  mul_628 = None
    exp_24: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_629);  mul_629 = None
    mul_630: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_631: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_56, mul_630);  view_56 = mul_630 = None
    add_227: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_627, mul_631);  mul_627 = mul_631 = None
    mul_632: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_354, add_227);  view_354 = add_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_355: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_632, [1568, 2048]);  mul_632 = None
    permute_408: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
    mm_100: "f32[1568, 512]" = torch.ops.aten.mm.default(view_355, permute_408);  permute_408 = None
    permute_409: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_355, [1, 0])
    mm_101: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_409, view_55);  permute_409 = view_55 = None
    permute_410: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_205: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_355, [0], True);  view_355 = None
    view_356: "f32[2048]" = torch.ops.aten.view.default(sum_205, [2048]);  sum_205 = None
    permute_411: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_410, [1, 0]);  permute_410 = None
    view_357: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(mm_100, [8, 14, 14, 512]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_138: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_50, memory_format = torch.contiguous_format);  permute_50 = None
    sub_119: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_138, getitem_29);  clone_138 = getitem_29 = None
    mul_633: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_119, rsqrt_14);  sub_119 = None
    mul_634: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_357, primals_40);  primals_40 = None
    mul_635: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_634, 512)
    sum_206: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_634, [3], True)
    mul_636: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_634, mul_633);  mul_634 = None
    sum_207: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_636, [3], True);  mul_636 = None
    mul_637: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_633, sum_207);  sum_207 = None
    sub_120: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_635, sum_206);  mul_635 = sum_206 = None
    sub_121: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_120, mul_637);  sub_120 = mul_637 = None
    div_27: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 512);  rsqrt_14 = None
    mul_638: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_27, sub_121);  div_27 = sub_121 = None
    mul_639: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_357, mul_633);  mul_633 = None
    sum_208: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_639, [0, 1, 2]);  mul_639 = None
    sum_209: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_357, [0, 1, 2]);  view_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_412: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_638, [0, 3, 1, 2]);  mul_638 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_210: "f32[512]" = torch.ops.aten.sum.dim_IntList(permute_412, [0, 2, 3])
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(permute_412, add_49, primals_191, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False]);  permute_412 = add_49 = primals_191 = None
    getitem_157: "f32[8, 512, 14, 14]" = convolution_backward_25[0]
    getitem_158: "f32[512, 1, 7, 7]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_228: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_225, getitem_157);  add_225 = getitem_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_640: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_228, permute_49);  permute_49 = None
    mul_641: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_228, view_54);  view_54 = None
    sum_211: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_640, [0, 2, 3], True);  mul_640 = None
    view_358: "f32[512]" = torch.ops.aten.view.default(sum_211, [512]);  sum_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_413: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_641, [0, 2, 3, 1]);  mul_641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_359: "f32[1568, 512]" = torch.ops.aten.view.default(permute_413, [1568, 512]);  permute_413 = None
    permute_414: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
    mm_102: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_359, permute_414);  permute_414 = None
    permute_415: "f32[512, 1568]" = torch.ops.aten.permute.default(view_359, [1, 0])
    mm_103: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_415, view_52);  permute_415 = view_52 = None
    permute_416: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_212: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_359, [0], True);  view_359 = None
    view_360: "f32[512]" = torch.ops.aten.view.default(sum_212, [512]);  sum_212 = None
    permute_417: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_416, [1, 0]);  permute_416 = None
    view_361: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(mm_102, [8, 14, 14, 2048]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_642: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_51, 0.7071067811865476)
    erf_61: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_642);  mul_642 = None
    add_229: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_61, 1);  erf_61 = None
    mul_643: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_229, 0.5);  add_229 = None
    mul_644: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_51, view_51)
    mul_645: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_644, -0.5);  mul_644 = None
    exp_25: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_645);  mul_645 = None
    mul_646: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_647: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_51, mul_646);  view_51 = mul_646 = None
    add_230: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_643, mul_647);  mul_643 = mul_647 = None
    mul_648: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_361, add_230);  view_361 = add_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_362: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_648, [1568, 2048]);  mul_648 = None
    permute_418: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    mm_104: "f32[1568, 512]" = torch.ops.aten.mm.default(view_362, permute_418);  permute_418 = None
    permute_419: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_362, [1, 0])
    mm_105: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_419, view_50);  permute_419 = view_50 = None
    permute_420: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_213: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_362, [0], True);  view_362 = None
    view_363: "f32[2048]" = torch.ops.aten.view.default(sum_213, [2048]);  sum_213 = None
    permute_421: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_420, [1, 0]);  permute_420 = None
    view_364: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(mm_104, [8, 14, 14, 512]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_139: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
    sub_122: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_139, getitem_27);  clone_139 = getitem_27 = None
    mul_649: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_122, rsqrt_13);  sub_122 = None
    mul_650: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_364, primals_37);  primals_37 = None
    mul_651: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_650, 512)
    sum_214: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_650, [3], True)
    mul_652: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_650, mul_649);  mul_650 = None
    sum_215: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_652, [3], True);  mul_652 = None
    mul_653: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_649, sum_215);  sum_215 = None
    sub_123: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_651, sum_214);  mul_651 = sum_214 = None
    sub_124: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_123, mul_653);  sub_123 = mul_653 = None
    div_28: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 512);  rsqrt_13 = None
    mul_654: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_28, sub_124);  div_28 = sub_124 = None
    mul_655: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_364, mul_649);  mul_649 = None
    sum_216: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_655, [0, 1, 2]);  mul_655 = None
    sum_217: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_364, [0, 1, 2]);  view_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_422: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_654, [0, 3, 1, 2]);  mul_654 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_218: "f32[512]" = torch.ops.aten.sum.dim_IntList(permute_422, [0, 2, 3])
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(permute_422, add_45, primals_185, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False]);  permute_422 = add_45 = primals_185 = None
    getitem_160: "f32[8, 512, 14, 14]" = convolution_backward_26[0]
    getitem_161: "f32[512, 1, 7, 7]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_231: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_228, getitem_160);  add_228 = getitem_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_656: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_231, permute_45);  permute_45 = None
    mul_657: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_231, view_49);  view_49 = None
    sum_219: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_656, [0, 2, 3], True);  mul_656 = None
    view_365: "f32[512]" = torch.ops.aten.view.default(sum_219, [512]);  sum_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_423: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_657, [0, 2, 3, 1]);  mul_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_366: "f32[1568, 512]" = torch.ops.aten.view.default(permute_423, [1568, 512]);  permute_423 = None
    permute_424: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    mm_106: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_366, permute_424);  permute_424 = None
    permute_425: "f32[512, 1568]" = torch.ops.aten.permute.default(view_366, [1, 0])
    mm_107: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_425, view_47);  permute_425 = view_47 = None
    permute_426: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_220: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_366, [0], True);  view_366 = None
    view_367: "f32[512]" = torch.ops.aten.view.default(sum_220, [512]);  sum_220 = None
    permute_427: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_426, [1, 0]);  permute_426 = None
    view_368: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(mm_106, [8, 14, 14, 2048]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_658: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_46, 0.7071067811865476)
    erf_62: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_658);  mul_658 = None
    add_232: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_62, 1);  erf_62 = None
    mul_659: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_232, 0.5);  add_232 = None
    mul_660: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_46, view_46)
    mul_661: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_660, -0.5);  mul_660 = None
    exp_26: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_661);  mul_661 = None
    mul_662: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_663: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_46, mul_662);  view_46 = mul_662 = None
    add_233: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_659, mul_663);  mul_659 = mul_663 = None
    mul_664: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_368, add_233);  view_368 = add_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_369: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_664, [1568, 2048]);  mul_664 = None
    permute_428: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    mm_108: "f32[1568, 512]" = torch.ops.aten.mm.default(view_369, permute_428);  permute_428 = None
    permute_429: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_369, [1, 0])
    mm_109: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_429, view_45);  permute_429 = view_45 = None
    permute_430: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_221: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_369, [0], True);  view_369 = None
    view_370: "f32[2048]" = torch.ops.aten.view.default(sum_221, [2048]);  sum_221 = None
    permute_431: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_430, [1, 0]);  permute_430 = None
    view_371: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(mm_108, [8, 14, 14, 512]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_140: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_42, memory_format = torch.contiguous_format);  permute_42 = None
    sub_125: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_140, getitem_25);  clone_140 = getitem_25 = None
    mul_665: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_125, rsqrt_12);  sub_125 = None
    mul_666: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_371, primals_34);  primals_34 = None
    mul_667: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_666, 512)
    sum_222: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_666, [3], True)
    mul_668: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_666, mul_665);  mul_666 = None
    sum_223: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_668, [3], True);  mul_668 = None
    mul_669: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_665, sum_223);  sum_223 = None
    sub_126: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_667, sum_222);  mul_667 = sum_222 = None
    sub_127: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_126, mul_669);  sub_126 = mul_669 = None
    div_29: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 512);  rsqrt_12 = None
    mul_670: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_29, sub_127);  div_29 = sub_127 = None
    mul_671: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_371, mul_665);  mul_665 = None
    sum_224: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_671, [0, 1, 2]);  mul_671 = None
    sum_225: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_371, [0, 1, 2]);  view_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_432: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_670, [0, 3, 1, 2]);  mul_670 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_226: "f32[512]" = torch.ops.aten.sum.dim_IntList(permute_432, [0, 2, 3])
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(permute_432, add_41, primals_179, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False]);  permute_432 = add_41 = primals_179 = None
    getitem_163: "f32[8, 512, 14, 14]" = convolution_backward_27[0]
    getitem_164: "f32[512, 1, 7, 7]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_234: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_231, getitem_163);  add_231 = getitem_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_672: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_234, permute_41);  permute_41 = None
    mul_673: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_234, view_44);  view_44 = None
    sum_227: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_672, [0, 2, 3], True);  mul_672 = None
    view_372: "f32[512]" = torch.ops.aten.view.default(sum_227, [512]);  sum_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_433: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_673, [0, 2, 3, 1]);  mul_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_373: "f32[1568, 512]" = torch.ops.aten.view.default(permute_433, [1568, 512]);  permute_433 = None
    permute_434: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
    mm_110: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_373, permute_434);  permute_434 = None
    permute_435: "f32[512, 1568]" = torch.ops.aten.permute.default(view_373, [1, 0])
    mm_111: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_435, view_42);  permute_435 = view_42 = None
    permute_436: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_228: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_373, [0], True);  view_373 = None
    view_374: "f32[512]" = torch.ops.aten.view.default(sum_228, [512]);  sum_228 = None
    permute_437: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_436, [1, 0]);  permute_436 = None
    view_375: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(mm_110, [8, 14, 14, 2048]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_674: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476)
    erf_63: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_674);  mul_674 = None
    add_235: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_63, 1);  erf_63 = None
    mul_675: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_235, 0.5);  add_235 = None
    mul_676: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_41, view_41)
    mul_677: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_676, -0.5);  mul_676 = None
    exp_27: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_677);  mul_677 = None
    mul_678: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_679: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_41, mul_678);  view_41 = mul_678 = None
    add_236: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_675, mul_679);  mul_675 = mul_679 = None
    mul_680: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_375, add_236);  view_375 = add_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_376: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_680, [1568, 2048]);  mul_680 = None
    permute_438: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
    mm_112: "f32[1568, 512]" = torch.ops.aten.mm.default(view_376, permute_438);  permute_438 = None
    permute_439: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_376, [1, 0])
    mm_113: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_439, view_40);  permute_439 = view_40 = None
    permute_440: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    sum_229: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_376, [0], True);  view_376 = None
    view_377: "f32[2048]" = torch.ops.aten.view.default(sum_229, [2048]);  sum_229 = None
    permute_441: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_440, [1, 0]);  permute_440 = None
    view_378: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(mm_112, [8, 14, 14, 512]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_141: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    sub_128: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_141, getitem_23);  clone_141 = getitem_23 = None
    mul_681: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_128, rsqrt_11);  sub_128 = None
    mul_682: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_378, primals_31);  primals_31 = None
    mul_683: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_682, 512)
    sum_230: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_682, [3], True)
    mul_684: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_682, mul_681);  mul_682 = None
    sum_231: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_684, [3], True);  mul_684 = None
    mul_685: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_681, sum_231);  sum_231 = None
    sub_129: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_683, sum_230);  mul_683 = sum_230 = None
    sub_130: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_129, mul_685);  sub_129 = mul_685 = None
    div_30: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 512);  rsqrt_11 = None
    mul_686: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_30, sub_130);  div_30 = sub_130 = None
    mul_687: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_378, mul_681);  mul_681 = None
    sum_232: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_687, [0, 1, 2]);  mul_687 = None
    sum_233: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_378, [0, 1, 2]);  view_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_442: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_686, [0, 3, 1, 2]);  mul_686 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_234: "f32[512]" = torch.ops.aten.sum.dim_IntList(permute_442, [0, 2, 3])
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(permute_442, add_37, primals_173, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False]);  permute_442 = add_37 = primals_173 = None
    getitem_166: "f32[8, 512, 14, 14]" = convolution_backward_28[0]
    getitem_167: "f32[512, 1, 7, 7]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_237: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_234, getitem_166);  add_234 = getitem_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_688: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_237, permute_37);  permute_37 = None
    mul_689: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_237, view_39);  view_39 = None
    sum_235: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_688, [0, 2, 3], True);  mul_688 = None
    view_379: "f32[512]" = torch.ops.aten.view.default(sum_235, [512]);  sum_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_443: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_689, [0, 2, 3, 1]);  mul_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_380: "f32[1568, 512]" = torch.ops.aten.view.default(permute_443, [1568, 512]);  permute_443 = None
    permute_444: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    mm_114: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_380, permute_444);  permute_444 = None
    permute_445: "f32[512, 1568]" = torch.ops.aten.permute.default(view_380, [1, 0])
    mm_115: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_445, view_37);  permute_445 = view_37 = None
    permute_446: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_236: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_380, [0], True);  view_380 = None
    view_381: "f32[512]" = torch.ops.aten.view.default(sum_236, [512]);  sum_236 = None
    permute_447: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_446, [1, 0]);  permute_446 = None
    view_382: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(mm_114, [8, 14, 14, 2048]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_690: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_36, 0.7071067811865476)
    erf_64: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_690);  mul_690 = None
    add_238: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_64, 1);  erf_64 = None
    mul_691: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_238, 0.5);  add_238 = None
    mul_692: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_36, view_36)
    mul_693: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_692, -0.5);  mul_692 = None
    exp_28: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_693);  mul_693 = None
    mul_694: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
    mul_695: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_36, mul_694);  view_36 = mul_694 = None
    add_239: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_691, mul_695);  mul_691 = mul_695 = None
    mul_696: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_382, add_239);  view_382 = add_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_383: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_696, [1568, 2048]);  mul_696 = None
    permute_448: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    mm_116: "f32[1568, 512]" = torch.ops.aten.mm.default(view_383, permute_448);  permute_448 = None
    permute_449: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_383, [1, 0])
    mm_117: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_449, view_35);  permute_449 = view_35 = None
    permute_450: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_237: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_383, [0], True);  view_383 = None
    view_384: "f32[2048]" = torch.ops.aten.view.default(sum_237, [2048]);  sum_237 = None
    permute_451: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_450, [1, 0]);  permute_450 = None
    view_385: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(mm_116, [8, 14, 14, 512]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_142: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
    sub_131: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_142, getitem_21);  clone_142 = getitem_21 = None
    mul_697: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_131, rsqrt_10);  sub_131 = None
    mul_698: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_385, primals_28);  primals_28 = None
    mul_699: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_698, 512)
    sum_238: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_698, [3], True)
    mul_700: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_698, mul_697);  mul_698 = None
    sum_239: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_700, [3], True);  mul_700 = None
    mul_701: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_697, sum_239);  sum_239 = None
    sub_132: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_699, sum_238);  mul_699 = sum_238 = None
    sub_133: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_132, mul_701);  sub_132 = mul_701 = None
    div_31: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 512);  rsqrt_10 = None
    mul_702: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_31, sub_133);  div_31 = sub_133 = None
    mul_703: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_385, mul_697);  mul_697 = None
    sum_240: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_703, [0, 1, 2]);  mul_703 = None
    sum_241: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_385, [0, 1, 2]);  view_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_452: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_702, [0, 3, 1, 2]);  mul_702 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_242: "f32[512]" = torch.ops.aten.sum.dim_IntList(permute_452, [0, 2, 3])
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(permute_452, add_33, primals_167, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False]);  permute_452 = add_33 = primals_167 = None
    getitem_169: "f32[8, 512, 14, 14]" = convolution_backward_29[0]
    getitem_170: "f32[512, 1, 7, 7]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_240: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_237, getitem_169);  add_237 = getitem_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_704: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_240, permute_33);  permute_33 = None
    mul_705: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_240, view_34);  view_34 = None
    sum_243: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_704, [0, 2, 3], True);  mul_704 = None
    view_386: "f32[512]" = torch.ops.aten.view.default(sum_243, [512]);  sum_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_453: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_705, [0, 2, 3, 1]);  mul_705 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_387: "f32[1568, 512]" = torch.ops.aten.view.default(permute_453, [1568, 512]);  permute_453 = None
    permute_454: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    mm_118: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_387, permute_454);  permute_454 = None
    permute_455: "f32[512, 1568]" = torch.ops.aten.permute.default(view_387, [1, 0])
    mm_119: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_455, view_32);  permute_455 = view_32 = None
    permute_456: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_244: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_387, [0], True);  view_387 = None
    view_388: "f32[512]" = torch.ops.aten.view.default(sum_244, [512]);  sum_244 = None
    permute_457: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_456, [1, 0]);  permute_456 = None
    view_389: "f32[8, 14, 14, 2048]" = torch.ops.aten.view.default(mm_118, [8, 14, 14, 2048]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_706: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_31, 0.7071067811865476)
    erf_65: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_706);  mul_706 = None
    add_241: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_65, 1);  erf_65 = None
    mul_707: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_241, 0.5);  add_241 = None
    mul_708: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_31, view_31)
    mul_709: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_708, -0.5);  mul_708 = None
    exp_29: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_709);  mul_709 = None
    mul_710: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
    mul_711: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_31, mul_710);  view_31 = mul_710 = None
    add_242: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_707, mul_711);  mul_707 = mul_711 = None
    mul_712: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_389, add_242);  view_389 = add_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_390: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_712, [1568, 2048]);  mul_712 = None
    permute_458: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    mm_120: "f32[1568, 512]" = torch.ops.aten.mm.default(view_390, permute_458);  permute_458 = None
    permute_459: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_390, [1, 0])
    mm_121: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_459, view_30);  permute_459 = view_30 = None
    permute_460: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    sum_245: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_390, [0], True);  view_390 = None
    view_391: "f32[2048]" = torch.ops.aten.view.default(sum_245, [2048]);  sum_245 = None
    permute_461: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_460, [1, 0]);  permute_460 = None
    view_392: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(mm_120, [8, 14, 14, 512]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_143: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
    sub_134: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(clone_143, getitem_19);  clone_143 = getitem_19 = None
    mul_713: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_134, rsqrt_9);  sub_134 = None
    mul_714: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_392, primals_25);  primals_25 = None
    mul_715: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_714, 512)
    sum_246: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_714, [3], True)
    mul_716: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_714, mul_713);  mul_714 = None
    sum_247: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_716, [3], True);  mul_716 = None
    mul_717: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_713, sum_247);  sum_247 = None
    sub_135: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_715, sum_246);  mul_715 = sum_246 = None
    sub_136: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_135, mul_717);  sub_135 = mul_717 = None
    div_32: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 512);  rsqrt_9 = None
    mul_718: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_32, sub_136);  div_32 = sub_136 = None
    mul_719: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_392, mul_713);  mul_713 = None
    sum_248: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_719, [0, 1, 2]);  mul_719 = None
    sum_249: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_392, [0, 1, 2]);  view_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_462: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_718, [0, 3, 1, 2]);  mul_718 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_250: "f32[512]" = torch.ops.aten.sum.dim_IntList(permute_462, [0, 2, 3])
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(permute_462, convolution_8, primals_161, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, False]);  permute_462 = convolution_8 = primals_161 = None
    getitem_172: "f32[8, 512, 14, 14]" = convolution_backward_30[0]
    getitem_173: "f32[512, 1, 7, 7]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_243: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_240, getitem_172);  add_240 = getitem_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:229, code: x = self.downsample(x)
    sum_251: "f32[512]" = torch.ops.aten.sum.dim_IntList(add_243, [0, 2, 3])
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(add_243, permute_29, primals_159, [512], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_243 = permute_29 = primals_159 = None
    getitem_175: "f32[8, 256, 28, 28]" = convolution_backward_31[0]
    getitem_176: "f32[512, 256, 2, 2]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    permute_463: "f32[8, 28, 28, 256]" = torch.ops.aten.permute.default(getitem_175, [0, 2, 3, 1]);  getitem_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_144: "f32[8, 28, 28, 256]" = torch.ops.aten.clone.default(permute_463, memory_format = torch.contiguous_format);  permute_463 = None
    sub_137: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(permute_28, getitem_17);  permute_28 = getitem_17 = None
    mul_720: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(sub_137, rsqrt_8);  sub_137 = None
    mul_721: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(clone_144, primals_23);  primals_23 = None
    mul_722: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_721, 256)
    sum_252: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_721, [3], True)
    mul_723: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_721, mul_720);  mul_721 = None
    sum_253: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_723, [3], True);  mul_723 = None
    mul_724: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_720, sum_253);  sum_253 = None
    sub_138: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(mul_722, sum_252);  mul_722 = sum_252 = None
    sub_139: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(sub_138, mul_724);  sub_138 = mul_724 = None
    div_33: "f32[8, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 256);  rsqrt_8 = None
    mul_725: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(div_33, sub_139);  div_33 = sub_139 = None
    mul_726: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(clone_144, mul_720);  mul_720 = None
    sum_254: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_726, [0, 1, 2]);  mul_726 = None
    sum_255: "f32[256]" = torch.ops.aten.sum.dim_IntList(clone_144, [0, 1, 2]);  clone_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    permute_464: "f32[8, 256, 28, 28]" = torch.ops.aten.permute.default(mul_725, [0, 3, 1, 2]);  mul_725 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_727: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(permute_464, permute_27);  permute_27 = None
    mul_728: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(permute_464, view_29);  view_29 = None
    sum_256: "f32[1, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_727, [0, 2, 3], True);  mul_727 = None
    view_393: "f32[256]" = torch.ops.aten.view.default(sum_256, [256]);  sum_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_465: "f32[8, 28, 28, 256]" = torch.ops.aten.permute.default(mul_728, [0, 2, 3, 1]);  mul_728 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_394: "f32[6272, 256]" = torch.ops.aten.view.default(permute_465, [6272, 256]);  permute_465 = None
    permute_466: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
    mm_122: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_394, permute_466);  permute_466 = None
    permute_467: "f32[256, 6272]" = torch.ops.aten.permute.default(view_394, [1, 0])
    mm_123: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_467, view_27);  permute_467 = view_27 = None
    permute_468: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_257: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_394, [0], True);  view_394 = None
    view_395: "f32[256]" = torch.ops.aten.view.default(sum_257, [256]);  sum_257 = None
    permute_469: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_468, [1, 0]);  permute_468 = None
    view_396: "f32[8, 28, 28, 1024]" = torch.ops.aten.view.default(mm_122, [8, 28, 28, 1024]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_729: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_26, 0.7071067811865476)
    erf_66: "f32[8, 28, 28, 1024]" = torch.ops.aten.erf.default(mul_729);  mul_729 = None
    add_244: "f32[8, 28, 28, 1024]" = torch.ops.aten.add.Tensor(erf_66, 1);  erf_66 = None
    mul_730: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(add_244, 0.5);  add_244 = None
    mul_731: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_26, view_26)
    mul_732: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(mul_731, -0.5);  mul_731 = None
    exp_30: "f32[8, 28, 28, 1024]" = torch.ops.aten.exp.default(mul_732);  mul_732 = None
    mul_733: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(exp_30, 0.3989422804014327);  exp_30 = None
    mul_734: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_26, mul_733);  view_26 = mul_733 = None
    add_245: "f32[8, 28, 28, 1024]" = torch.ops.aten.add.Tensor(mul_730, mul_734);  mul_730 = mul_734 = None
    mul_735: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_396, add_245);  view_396 = add_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_397: "f32[6272, 1024]" = torch.ops.aten.view.default(mul_735, [6272, 1024]);  mul_735 = None
    permute_470: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    mm_124: "f32[6272, 256]" = torch.ops.aten.mm.default(view_397, permute_470);  permute_470 = None
    permute_471: "f32[1024, 6272]" = torch.ops.aten.permute.default(view_397, [1, 0])
    mm_125: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_471, view_25);  permute_471 = view_25 = None
    permute_472: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    sum_258: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_397, [0], True);  view_397 = None
    view_398: "f32[1024]" = torch.ops.aten.view.default(sum_258, [1024]);  sum_258 = None
    permute_473: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_472, [1, 0]);  permute_472 = None
    view_399: "f32[8, 28, 28, 256]" = torch.ops.aten.view.default(mm_124, [8, 28, 28, 256]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_145: "f32[8, 28, 28, 256]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    sub_140: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(clone_145, getitem_15);  clone_145 = getitem_15 = None
    mul_736: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(sub_140, rsqrt_7);  sub_140 = None
    mul_737: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(view_399, primals_20);  primals_20 = None
    mul_738: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_737, 256)
    sum_259: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_737, [3], True)
    mul_739: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_737, mul_736);  mul_737 = None
    sum_260: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_739, [3], True);  mul_739 = None
    mul_740: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_736, sum_260);  sum_260 = None
    sub_141: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(mul_738, sum_259);  mul_738 = sum_259 = None
    sub_142: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(sub_141, mul_740);  sub_141 = mul_740 = None
    div_34: "f32[8, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 256);  rsqrt_7 = None
    mul_741: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(div_34, sub_142);  div_34 = sub_142 = None
    mul_742: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(view_399, mul_736);  mul_736 = None
    sum_261: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_742, [0, 1, 2]);  mul_742 = None
    sum_262: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_399, [0, 1, 2]);  view_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_474: "f32[8, 256, 28, 28]" = torch.ops.aten.permute.default(mul_741, [0, 3, 1, 2]);  mul_741 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_263: "f32[256]" = torch.ops.aten.sum.dim_IntList(permute_474, [0, 2, 3])
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(permute_474, add_23, primals_153, [256], [1, 1], [3, 3], [1, 1], False, [0, 0], 256, [True, True, False]);  permute_474 = add_23 = primals_153 = None
    getitem_178: "f32[8, 256, 28, 28]" = convolution_backward_32[0]
    getitem_179: "f32[256, 1, 7, 7]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_246: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(permute_464, getitem_178);  permute_464 = getitem_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_743: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(add_246, permute_23);  permute_23 = None
    mul_744: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(add_246, view_24);  view_24 = None
    sum_264: "f32[1, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_743, [0, 2, 3], True);  mul_743 = None
    view_400: "f32[256]" = torch.ops.aten.view.default(sum_264, [256]);  sum_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_475: "f32[8, 28, 28, 256]" = torch.ops.aten.permute.default(mul_744, [0, 2, 3, 1]);  mul_744 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_401: "f32[6272, 256]" = torch.ops.aten.view.default(permute_475, [6272, 256]);  permute_475 = None
    permute_476: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    mm_126: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_401, permute_476);  permute_476 = None
    permute_477: "f32[256, 6272]" = torch.ops.aten.permute.default(view_401, [1, 0])
    mm_127: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_477, view_22);  permute_477 = view_22 = None
    permute_478: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_265: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_401, [0], True);  view_401 = None
    view_402: "f32[256]" = torch.ops.aten.view.default(sum_265, [256]);  sum_265 = None
    permute_479: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_478, [1, 0]);  permute_478 = None
    view_403: "f32[8, 28, 28, 1024]" = torch.ops.aten.view.default(mm_126, [8, 28, 28, 1024]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_745: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_21, 0.7071067811865476)
    erf_67: "f32[8, 28, 28, 1024]" = torch.ops.aten.erf.default(mul_745);  mul_745 = None
    add_247: "f32[8, 28, 28, 1024]" = torch.ops.aten.add.Tensor(erf_67, 1);  erf_67 = None
    mul_746: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(add_247, 0.5);  add_247 = None
    mul_747: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_21, view_21)
    mul_748: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(mul_747, -0.5);  mul_747 = None
    exp_31: "f32[8, 28, 28, 1024]" = torch.ops.aten.exp.default(mul_748);  mul_748 = None
    mul_749: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(exp_31, 0.3989422804014327);  exp_31 = None
    mul_750: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_21, mul_749);  view_21 = mul_749 = None
    add_248: "f32[8, 28, 28, 1024]" = torch.ops.aten.add.Tensor(mul_746, mul_750);  mul_746 = mul_750 = None
    mul_751: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_403, add_248);  view_403 = add_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_404: "f32[6272, 1024]" = torch.ops.aten.view.default(mul_751, [6272, 1024]);  mul_751 = None
    permute_480: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    mm_128: "f32[6272, 256]" = torch.ops.aten.mm.default(view_404, permute_480);  permute_480 = None
    permute_481: "f32[1024, 6272]" = torch.ops.aten.permute.default(view_404, [1, 0])
    mm_129: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_481, view_20);  permute_481 = view_20 = None
    permute_482: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_266: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_404, [0], True);  view_404 = None
    view_405: "f32[1024]" = torch.ops.aten.view.default(sum_266, [1024]);  sum_266 = None
    permute_483: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_482, [1, 0]);  permute_482 = None
    view_406: "f32[8, 28, 28, 256]" = torch.ops.aten.view.default(mm_128, [8, 28, 28, 256]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_146: "f32[8, 28, 28, 256]" = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format);  permute_20 = None
    sub_143: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(clone_146, getitem_13);  clone_146 = getitem_13 = None
    mul_752: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(sub_143, rsqrt_6);  sub_143 = None
    mul_753: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(view_406, primals_17);  primals_17 = None
    mul_754: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_753, 256)
    sum_267: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_753, [3], True)
    mul_755: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_753, mul_752);  mul_753 = None
    sum_268: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_755, [3], True);  mul_755 = None
    mul_756: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_752, sum_268);  sum_268 = None
    sub_144: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(mul_754, sum_267);  mul_754 = sum_267 = None
    sub_145: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(sub_144, mul_756);  sub_144 = mul_756 = None
    div_35: "f32[8, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 256);  rsqrt_6 = None
    mul_757: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(div_35, sub_145);  div_35 = sub_145 = None
    mul_758: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(view_406, mul_752);  mul_752 = None
    sum_269: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_758, [0, 1, 2]);  mul_758 = None
    sum_270: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_406, [0, 1, 2]);  view_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_484: "f32[8, 256, 28, 28]" = torch.ops.aten.permute.default(mul_757, [0, 3, 1, 2]);  mul_757 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_271: "f32[256]" = torch.ops.aten.sum.dim_IntList(permute_484, [0, 2, 3])
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(permute_484, add_19, primals_147, [256], [1, 1], [3, 3], [1, 1], False, [0, 0], 256, [True, True, False]);  permute_484 = add_19 = primals_147 = None
    getitem_181: "f32[8, 256, 28, 28]" = convolution_backward_33[0]
    getitem_182: "f32[256, 1, 7, 7]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_249: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(add_246, getitem_181);  add_246 = getitem_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_759: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(add_249, permute_19);  permute_19 = None
    mul_760: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(add_249, view_19);  view_19 = None
    sum_272: "f32[1, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_759, [0, 2, 3], True);  mul_759 = None
    view_407: "f32[256]" = torch.ops.aten.view.default(sum_272, [256]);  sum_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_485: "f32[8, 28, 28, 256]" = torch.ops.aten.permute.default(mul_760, [0, 2, 3, 1]);  mul_760 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_408: "f32[6272, 256]" = torch.ops.aten.view.default(permute_485, [6272, 256]);  permute_485 = None
    permute_486: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
    mm_130: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_408, permute_486);  permute_486 = None
    permute_487: "f32[256, 6272]" = torch.ops.aten.permute.default(view_408, [1, 0])
    mm_131: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_487, view_17);  permute_487 = view_17 = None
    permute_488: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    sum_273: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_408, [0], True);  view_408 = None
    view_409: "f32[256]" = torch.ops.aten.view.default(sum_273, [256]);  sum_273 = None
    permute_489: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_488, [1, 0]);  permute_488 = None
    view_410: "f32[8, 28, 28, 1024]" = torch.ops.aten.view.default(mm_130, [8, 28, 28, 1024]);  mm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_761: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_16, 0.7071067811865476)
    erf_68: "f32[8, 28, 28, 1024]" = torch.ops.aten.erf.default(mul_761);  mul_761 = None
    add_250: "f32[8, 28, 28, 1024]" = torch.ops.aten.add.Tensor(erf_68, 1);  erf_68 = None
    mul_762: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(add_250, 0.5);  add_250 = None
    mul_763: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_16, view_16)
    mul_764: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(mul_763, -0.5);  mul_763 = None
    exp_32: "f32[8, 28, 28, 1024]" = torch.ops.aten.exp.default(mul_764);  mul_764 = None
    mul_765: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(exp_32, 0.3989422804014327);  exp_32 = None
    mul_766: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_16, mul_765);  view_16 = mul_765 = None
    add_251: "f32[8, 28, 28, 1024]" = torch.ops.aten.add.Tensor(mul_762, mul_766);  mul_762 = mul_766 = None
    mul_767: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_410, add_251);  view_410 = add_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_411: "f32[6272, 1024]" = torch.ops.aten.view.default(mul_767, [6272, 1024]);  mul_767 = None
    permute_490: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    mm_132: "f32[6272, 256]" = torch.ops.aten.mm.default(view_411, permute_490);  permute_490 = None
    permute_491: "f32[1024, 6272]" = torch.ops.aten.permute.default(view_411, [1, 0])
    mm_133: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_491, view_15);  permute_491 = view_15 = None
    permute_492: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_274: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_411, [0], True);  view_411 = None
    view_412: "f32[1024]" = torch.ops.aten.view.default(sum_274, [1024]);  sum_274 = None
    permute_493: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_492, [1, 0]);  permute_492 = None
    view_413: "f32[8, 28, 28, 256]" = torch.ops.aten.view.default(mm_132, [8, 28, 28, 256]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_147: "f32[8, 28, 28, 256]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    sub_146: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(clone_147, getitem_11);  clone_147 = getitem_11 = None
    mul_768: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(sub_146, rsqrt_5);  sub_146 = None
    mul_769: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(view_413, primals_14);  primals_14 = None
    mul_770: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_769, 256)
    sum_275: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_769, [3], True)
    mul_771: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_769, mul_768);  mul_769 = None
    sum_276: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_771, [3], True);  mul_771 = None
    mul_772: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_768, sum_276);  sum_276 = None
    sub_147: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(mul_770, sum_275);  mul_770 = sum_275 = None
    sub_148: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(sub_147, mul_772);  sub_147 = mul_772 = None
    div_36: "f32[8, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 256);  rsqrt_5 = None
    mul_773: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(div_36, sub_148);  div_36 = sub_148 = None
    mul_774: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(view_413, mul_768);  mul_768 = None
    sum_277: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_774, [0, 1, 2]);  mul_774 = None
    sum_278: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_413, [0, 1, 2]);  view_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_494: "f32[8, 256, 28, 28]" = torch.ops.aten.permute.default(mul_773, [0, 3, 1, 2]);  mul_773 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_279: "f32[256]" = torch.ops.aten.sum.dim_IntList(permute_494, [0, 2, 3])
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(permute_494, convolution_4, primals_141, [256], [1, 1], [3, 3], [1, 1], False, [0, 0], 256, [True, True, False]);  permute_494 = convolution_4 = primals_141 = None
    getitem_184: "f32[8, 256, 28, 28]" = convolution_backward_34[0]
    getitem_185: "f32[256, 1, 7, 7]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_252: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(add_249, getitem_184);  add_249 = getitem_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:229, code: x = self.downsample(x)
    sum_280: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_252, [0, 2, 3])
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(add_252, permute_15, primals_139, [256], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_252 = permute_15 = primals_139 = None
    getitem_187: "f32[8, 128, 56, 56]" = convolution_backward_35[0]
    getitem_188: "f32[256, 128, 2, 2]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    permute_495: "f32[8, 56, 56, 128]" = torch.ops.aten.permute.default(getitem_187, [0, 2, 3, 1]);  getitem_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_148: "f32[8, 56, 56, 128]" = torch.ops.aten.clone.default(permute_495, memory_format = torch.contiguous_format);  permute_495 = None
    sub_149: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(permute_14, getitem_9);  permute_14 = getitem_9 = None
    mul_775: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_149, rsqrt_4);  sub_149 = None
    mul_776: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(clone_148, primals_12);  primals_12 = None
    mul_777: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_776, 128)
    sum_281: "f32[8, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_776, [3], True)
    mul_778: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_776, mul_775);  mul_776 = None
    sum_282: "f32[8, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_778, [3], True);  mul_778 = None
    mul_779: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_775, sum_282);  sum_282 = None
    sub_150: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(mul_777, sum_281);  mul_777 = sum_281 = None
    sub_151: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(sub_150, mul_779);  sub_150 = mul_779 = None
    div_37: "f32[8, 56, 56, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 128);  rsqrt_4 = None
    mul_780: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(div_37, sub_151);  div_37 = sub_151 = None
    mul_781: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(clone_148, mul_775);  mul_775 = None
    sum_283: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_781, [0, 1, 2]);  mul_781 = None
    sum_284: "f32[128]" = torch.ops.aten.sum.dim_IntList(clone_148, [0, 1, 2]);  clone_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    permute_496: "f32[8, 128, 56, 56]" = torch.ops.aten.permute.default(mul_780, [0, 3, 1, 2]);  mul_780 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_782: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(permute_496, permute_13);  permute_13 = None
    mul_783: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(permute_496, view_14);  view_14 = None
    sum_285: "f32[1, 128, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_782, [0, 2, 3], True);  mul_782 = None
    view_414: "f32[128]" = torch.ops.aten.view.default(sum_285, [128]);  sum_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_497: "f32[8, 56, 56, 128]" = torch.ops.aten.permute.default(mul_783, [0, 2, 3, 1]);  mul_783 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_415: "f32[25088, 128]" = torch.ops.aten.view.default(permute_497, [25088, 128]);  permute_497 = None
    permute_498: "f32[128, 512]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    mm_134: "f32[25088, 512]" = torch.ops.aten.mm.default(view_415, permute_498);  permute_498 = None
    permute_499: "f32[128, 25088]" = torch.ops.aten.permute.default(view_415, [1, 0])
    mm_135: "f32[128, 512]" = torch.ops.aten.mm.default(permute_499, view_12);  permute_499 = view_12 = None
    permute_500: "f32[512, 128]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_286: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_415, [0], True);  view_415 = None
    view_416: "f32[128]" = torch.ops.aten.view.default(sum_286, [128]);  sum_286 = None
    permute_501: "f32[128, 512]" = torch.ops.aten.permute.default(permute_500, [1, 0]);  permute_500 = None
    view_417: "f32[8, 56, 56, 512]" = torch.ops.aten.view.default(mm_134, [8, 56, 56, 512]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_784: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_11, 0.7071067811865476)
    erf_69: "f32[8, 56, 56, 512]" = torch.ops.aten.erf.default(mul_784);  mul_784 = None
    add_253: "f32[8, 56, 56, 512]" = torch.ops.aten.add.Tensor(erf_69, 1);  erf_69 = None
    mul_785: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(add_253, 0.5);  add_253 = None
    mul_786: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_11, view_11)
    mul_787: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(mul_786, -0.5);  mul_786 = None
    exp_33: "f32[8, 56, 56, 512]" = torch.ops.aten.exp.default(mul_787);  mul_787 = None
    mul_788: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(exp_33, 0.3989422804014327);  exp_33 = None
    mul_789: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_11, mul_788);  view_11 = mul_788 = None
    add_254: "f32[8, 56, 56, 512]" = torch.ops.aten.add.Tensor(mul_785, mul_789);  mul_785 = mul_789 = None
    mul_790: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_417, add_254);  view_417 = add_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_418: "f32[25088, 512]" = torch.ops.aten.view.default(mul_790, [25088, 512]);  mul_790 = None
    permute_502: "f32[512, 128]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    mm_136: "f32[25088, 128]" = torch.ops.aten.mm.default(view_418, permute_502);  permute_502 = None
    permute_503: "f32[512, 25088]" = torch.ops.aten.permute.default(view_418, [1, 0])
    mm_137: "f32[512, 128]" = torch.ops.aten.mm.default(permute_503, view_10);  permute_503 = view_10 = None
    permute_504: "f32[128, 512]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    sum_287: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_418, [0], True);  view_418 = None
    view_419: "f32[512]" = torch.ops.aten.view.default(sum_287, [512]);  sum_287 = None
    permute_505: "f32[512, 128]" = torch.ops.aten.permute.default(permute_504, [1, 0]);  permute_504 = None
    view_420: "f32[8, 56, 56, 128]" = torch.ops.aten.view.default(mm_136, [8, 56, 56, 128]);  mm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_149: "f32[8, 56, 56, 128]" = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None
    sub_152: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(clone_149, getitem_7);  clone_149 = getitem_7 = None
    mul_791: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_152, rsqrt_3);  sub_152 = None
    mul_792: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(view_420, primals_9);  primals_9 = None
    mul_793: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_792, 128)
    sum_288: "f32[8, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_792, [3], True)
    mul_794: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_792, mul_791);  mul_792 = None
    sum_289: "f32[8, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_794, [3], True);  mul_794 = None
    mul_795: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_791, sum_289);  sum_289 = None
    sub_153: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(mul_793, sum_288);  mul_793 = sum_288 = None
    sub_154: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(sub_153, mul_795);  sub_153 = mul_795 = None
    div_38: "f32[8, 56, 56, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 128);  rsqrt_3 = None
    mul_796: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(div_38, sub_154);  div_38 = sub_154 = None
    mul_797: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(view_420, mul_791);  mul_791 = None
    sum_290: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_797, [0, 1, 2]);  mul_797 = None
    sum_291: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_420, [0, 1, 2]);  view_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_506: "f32[8, 128, 56, 56]" = torch.ops.aten.permute.default(mul_796, [0, 3, 1, 2]);  mul_796 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_292: "f32[128]" = torch.ops.aten.sum.dim_IntList(permute_506, [0, 2, 3])
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(permute_506, add_9, primals_133, [128], [1, 1], [3, 3], [1, 1], False, [0, 0], 128, [True, True, False]);  permute_506 = add_9 = primals_133 = None
    getitem_190: "f32[8, 128, 56, 56]" = convolution_backward_36[0]
    getitem_191: "f32[128, 1, 7, 7]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_255: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(permute_496, getitem_190);  permute_496 = getitem_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_798: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(add_255, permute_9);  permute_9 = None
    mul_799: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(add_255, view_9);  view_9 = None
    sum_293: "f32[1, 128, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_798, [0, 2, 3], True);  mul_798 = None
    view_421: "f32[128]" = torch.ops.aten.view.default(sum_293, [128]);  sum_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_507: "f32[8, 56, 56, 128]" = torch.ops.aten.permute.default(mul_799, [0, 2, 3, 1]);  mul_799 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_422: "f32[25088, 128]" = torch.ops.aten.view.default(permute_507, [25088, 128]);  permute_507 = None
    permute_508: "f32[128, 512]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    mm_138: "f32[25088, 512]" = torch.ops.aten.mm.default(view_422, permute_508);  permute_508 = None
    permute_509: "f32[128, 25088]" = torch.ops.aten.permute.default(view_422, [1, 0])
    mm_139: "f32[128, 512]" = torch.ops.aten.mm.default(permute_509, view_7);  permute_509 = view_7 = None
    permute_510: "f32[512, 128]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_294: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_422, [0], True);  view_422 = None
    view_423: "f32[128]" = torch.ops.aten.view.default(sum_294, [128]);  sum_294 = None
    permute_511: "f32[128, 512]" = torch.ops.aten.permute.default(permute_510, [1, 0]);  permute_510 = None
    view_424: "f32[8, 56, 56, 512]" = torch.ops.aten.view.default(mm_138, [8, 56, 56, 512]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_800: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_6, 0.7071067811865476)
    erf_70: "f32[8, 56, 56, 512]" = torch.ops.aten.erf.default(mul_800);  mul_800 = None
    add_256: "f32[8, 56, 56, 512]" = torch.ops.aten.add.Tensor(erf_70, 1);  erf_70 = None
    mul_801: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(add_256, 0.5);  add_256 = None
    mul_802: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_6, view_6)
    mul_803: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(mul_802, -0.5);  mul_802 = None
    exp_34: "f32[8, 56, 56, 512]" = torch.ops.aten.exp.default(mul_803);  mul_803 = None
    mul_804: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(exp_34, 0.3989422804014327);  exp_34 = None
    mul_805: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_6, mul_804);  view_6 = mul_804 = None
    add_257: "f32[8, 56, 56, 512]" = torch.ops.aten.add.Tensor(mul_801, mul_805);  mul_801 = mul_805 = None
    mul_806: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_424, add_257);  view_424 = add_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_425: "f32[25088, 512]" = torch.ops.aten.view.default(mul_806, [25088, 512]);  mul_806 = None
    permute_512: "f32[512, 128]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
    mm_140: "f32[25088, 128]" = torch.ops.aten.mm.default(view_425, permute_512);  permute_512 = None
    permute_513: "f32[512, 25088]" = torch.ops.aten.permute.default(view_425, [1, 0])
    mm_141: "f32[512, 128]" = torch.ops.aten.mm.default(permute_513, view_5);  permute_513 = view_5 = None
    permute_514: "f32[128, 512]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_295: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_425, [0], True);  view_425 = None
    view_426: "f32[512]" = torch.ops.aten.view.default(sum_295, [512]);  sum_295 = None
    permute_515: "f32[512, 128]" = torch.ops.aten.permute.default(permute_514, [1, 0]);  permute_514 = None
    view_427: "f32[8, 56, 56, 128]" = torch.ops.aten.view.default(mm_140, [8, 56, 56, 128]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_150: "f32[8, 56, 56, 128]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
    sub_155: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(clone_150, getitem_5);  clone_150 = getitem_5 = None
    mul_807: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_155, rsqrt_2);  sub_155 = None
    mul_808: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(view_427, primals_6);  primals_6 = None
    mul_809: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_808, 128)
    sum_296: "f32[8, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_808, [3], True)
    mul_810: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_808, mul_807);  mul_808 = None
    sum_297: "f32[8, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_810, [3], True);  mul_810 = None
    mul_811: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_807, sum_297);  sum_297 = None
    sub_156: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(mul_809, sum_296);  mul_809 = sum_296 = None
    sub_157: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(sub_156, mul_811);  sub_156 = mul_811 = None
    div_39: "f32[8, 56, 56, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 128);  rsqrt_2 = None
    mul_812: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(div_39, sub_157);  div_39 = sub_157 = None
    mul_813: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(view_427, mul_807);  mul_807 = None
    sum_298: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_813, [0, 1, 2]);  mul_813 = None
    sum_299: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_427, [0, 1, 2]);  view_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_516: "f32[8, 128, 56, 56]" = torch.ops.aten.permute.default(mul_812, [0, 3, 1, 2]);  mul_812 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_300: "f32[128]" = torch.ops.aten.sum.dim_IntList(permute_516, [0, 2, 3])
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(permute_516, add_5, primals_127, [128], [1, 1], [3, 3], [1, 1], False, [0, 0], 128, [True, True, False]);  permute_516 = add_5 = primals_127 = None
    getitem_193: "f32[8, 128, 56, 56]" = convolution_backward_37[0]
    getitem_194: "f32[128, 1, 7, 7]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_258: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(add_255, getitem_193);  add_255 = getitem_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_814: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(add_258, permute_5);  permute_5 = None
    mul_815: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(add_258, view_4);  view_4 = None
    sum_301: "f32[1, 128, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_814, [0, 2, 3], True);  mul_814 = None
    view_428: "f32[128]" = torch.ops.aten.view.default(sum_301, [128]);  sum_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_517: "f32[8, 56, 56, 128]" = torch.ops.aten.permute.default(mul_815, [0, 2, 3, 1]);  mul_815 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_429: "f32[25088, 128]" = torch.ops.aten.view.default(permute_517, [25088, 128]);  permute_517 = None
    permute_518: "f32[128, 512]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
    mm_142: "f32[25088, 512]" = torch.ops.aten.mm.default(view_429, permute_518);  permute_518 = None
    permute_519: "f32[128, 25088]" = torch.ops.aten.permute.default(view_429, [1, 0])
    mm_143: "f32[128, 512]" = torch.ops.aten.mm.default(permute_519, view_2);  permute_519 = view_2 = None
    permute_520: "f32[512, 128]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_302: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_429, [0], True);  view_429 = None
    view_430: "f32[128]" = torch.ops.aten.view.default(sum_302, [128]);  sum_302 = None
    permute_521: "f32[128, 512]" = torch.ops.aten.permute.default(permute_520, [1, 0]);  permute_520 = None
    view_431: "f32[8, 56, 56, 512]" = torch.ops.aten.view.default(mm_142, [8, 56, 56, 512]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_816: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_1, 0.7071067811865476)
    erf_71: "f32[8, 56, 56, 512]" = torch.ops.aten.erf.default(mul_816);  mul_816 = None
    add_259: "f32[8, 56, 56, 512]" = torch.ops.aten.add.Tensor(erf_71, 1);  erf_71 = None
    mul_817: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(add_259, 0.5);  add_259 = None
    mul_818: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_1, view_1)
    mul_819: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(mul_818, -0.5);  mul_818 = None
    exp_35: "f32[8, 56, 56, 512]" = torch.ops.aten.exp.default(mul_819);  mul_819 = None
    mul_820: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(exp_35, 0.3989422804014327);  exp_35 = None
    mul_821: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_1, mul_820);  view_1 = mul_820 = None
    add_260: "f32[8, 56, 56, 512]" = torch.ops.aten.add.Tensor(mul_817, mul_821);  mul_817 = mul_821 = None
    mul_822: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_431, add_260);  view_431 = add_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_432: "f32[25088, 512]" = torch.ops.aten.view.default(mul_822, [25088, 512]);  mul_822 = None
    permute_522: "f32[512, 128]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    mm_144: "f32[25088, 128]" = torch.ops.aten.mm.default(view_432, permute_522);  permute_522 = None
    permute_523: "f32[512, 25088]" = torch.ops.aten.permute.default(view_432, [1, 0])
    mm_145: "f32[512, 128]" = torch.ops.aten.mm.default(permute_523, view);  permute_523 = view = None
    permute_524: "f32[128, 512]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    sum_303: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_432, [0], True);  view_432 = None
    view_433: "f32[512]" = torch.ops.aten.view.default(sum_303, [512]);  sum_303 = None
    permute_525: "f32[512, 128]" = torch.ops.aten.permute.default(permute_524, [1, 0]);  permute_524 = None
    view_434: "f32[8, 56, 56, 128]" = torch.ops.aten.view.default(mm_144, [8, 56, 56, 128]);  mm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_151: "f32[8, 56, 56, 128]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    sub_158: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(clone_151, getitem_3);  clone_151 = getitem_3 = None
    mul_823: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_158, rsqrt_1);  sub_158 = None
    mul_824: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(view_434, primals_3);  primals_3 = None
    mul_825: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_824, 128)
    sum_304: "f32[8, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_824, [3], True)
    mul_826: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_824, mul_823);  mul_824 = None
    sum_305: "f32[8, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_826, [3], True);  mul_826 = None
    mul_827: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_823, sum_305);  sum_305 = None
    sub_159: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(mul_825, sum_304);  mul_825 = sum_304 = None
    sub_160: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(sub_159, mul_827);  sub_159 = mul_827 = None
    div_40: "f32[8, 56, 56, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 128);  rsqrt_1 = None
    mul_828: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(div_40, sub_160);  div_40 = sub_160 = None
    mul_829: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(view_434, mul_823);  mul_823 = None
    sum_306: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_829, [0, 1, 2]);  mul_829 = None
    sum_307: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_434, [0, 1, 2]);  view_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_526: "f32[8, 128, 56, 56]" = torch.ops.aten.permute.default(mul_828, [0, 3, 1, 2]);  mul_828 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    sum_308: "f32[128]" = torch.ops.aten.sum.dim_IntList(permute_526, [0, 2, 3])
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(permute_526, permute_1, primals_121, [128], [1, 1], [3, 3], [1, 1], False, [0, 0], 128, [True, True, False]);  permute_526 = permute_1 = primals_121 = None
    getitem_196: "f32[8, 128, 56, 56]" = convolution_backward_38[0]
    getitem_197: "f32[128, 1, 7, 7]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_261: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(add_258, getitem_196);  add_258 = getitem_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    permute_527: "f32[8, 56, 56, 128]" = torch.ops.aten.permute.default(add_261, [0, 2, 3, 1]);  add_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_152: "f32[8, 56, 56, 128]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
    sub_161: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(clone_152, getitem_1);  clone_152 = getitem_1 = None
    mul_830: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_161, rsqrt);  sub_161 = None
    mul_831: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(permute_527, primals_1);  primals_1 = None
    mul_832: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_831, 128)
    sum_309: "f32[8, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_831, [3], True)
    mul_833: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_831, mul_830);  mul_831 = None
    sum_310: "f32[8, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_833, [3], True);  mul_833 = None
    mul_834: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_830, sum_310);  sum_310 = None
    sub_162: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(mul_832, sum_309);  mul_832 = sum_309 = None
    sub_163: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(sub_162, mul_834);  sub_162 = mul_834 = None
    div_41: "f32[8, 56, 56, 1]" = torch.ops.aten.div.Tensor(rsqrt, 128);  rsqrt = None
    mul_835: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(div_41, sub_163);  div_41 = sub_163 = None
    mul_836: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(permute_527, mul_830);  mul_830 = None
    sum_311: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_836, [0, 1, 2]);  mul_836 = None
    sum_312: "f32[128]" = torch.ops.aten.sum.dim_IntList(permute_527, [0, 1, 2]);  permute_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    permute_528: "f32[8, 128, 56, 56]" = torch.ops.aten.permute.default(mul_835, [0, 3, 1, 2]);  mul_835 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:411, code: x = self.stem(x)
    sum_313: "f32[128]" = torch.ops.aten.sum.dim_IntList(permute_528, [0, 2, 3])
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(permute_528, primals_345, primals_119, [128], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  permute_528 = primals_345 = primals_119 = None
    getitem_200: "f32[128, 3, 4, 4]" = convolution_backward_39[1];  convolution_backward_39 = None
    return pytree.tree_unflatten([addmm_72, sum_311, sum_312, sum_306, sum_307, view_428, sum_298, sum_299, view_421, sum_290, sum_291, view_414, sum_283, sum_284, sum_277, sum_278, view_407, sum_269, sum_270, view_400, sum_261, sum_262, view_393, sum_254, sum_255, sum_248, sum_249, view_386, sum_240, sum_241, view_379, sum_232, sum_233, view_372, sum_224, sum_225, view_365, sum_216, sum_217, view_358, sum_208, sum_209, view_351, sum_200, sum_201, view_344, sum_192, sum_193, view_337, sum_184, sum_185, view_330, sum_176, sum_177, view_323, sum_168, sum_169, view_316, sum_160, sum_161, view_309, sum_152, sum_153, view_302, sum_144, sum_145, view_295, sum_136, sum_137, view_288, sum_128, sum_129, view_281, sum_120, sum_121, view_274, sum_112, sum_113, view_267, sum_104, sum_105, view_260, sum_96, sum_97, view_253, sum_88, sum_89, view_246, sum_80, sum_81, view_239, sum_72, sum_73, view_232, sum_64, sum_65, view_225, sum_56, sum_57, view_218, sum_48, sum_49, view_211, sum_40, sum_41, view_204, sum_33, sum_34, sum_27, sum_28, view_197, sum_19, sum_20, view_190, sum_11, sum_12, view_183, sum_4, sum_5, getitem_200, sum_313, getitem_197, sum_308, permute_525, view_433, permute_521, view_430, getitem_194, sum_300, permute_515, view_426, permute_511, view_423, getitem_191, sum_292, permute_505, view_419, permute_501, view_416, getitem_188, sum_280, getitem_185, sum_279, permute_493, view_412, permute_489, view_409, getitem_182, sum_271, permute_483, view_405, permute_479, view_402, getitem_179, sum_263, permute_473, view_398, permute_469, view_395, getitem_176, sum_251, getitem_173, sum_250, permute_461, view_391, permute_457, view_388, getitem_170, sum_242, permute_451, view_384, permute_447, view_381, getitem_167, sum_234, permute_441, view_377, permute_437, view_374, getitem_164, sum_226, permute_431, view_370, permute_427, view_367, getitem_161, sum_218, permute_421, view_363, permute_417, view_360, getitem_158, sum_210, permute_411, view_356, permute_407, view_353, getitem_155, sum_202, permute_401, view_349, permute_397, view_346, getitem_152, sum_194, permute_391, view_342, permute_387, view_339, getitem_149, sum_186, permute_381, view_335, permute_377, view_332, getitem_146, sum_178, permute_371, view_328, permute_367, view_325, getitem_143, sum_170, permute_361, view_321, permute_357, view_318, getitem_140, sum_162, permute_351, view_314, permute_347, view_311, getitem_137, sum_154, permute_341, view_307, permute_337, view_304, getitem_134, sum_146, permute_331, view_300, permute_327, view_297, getitem_131, sum_138, permute_321, view_293, permute_317, view_290, getitem_128, sum_130, permute_311, view_286, permute_307, view_283, getitem_125, sum_122, permute_301, view_279, permute_297, view_276, getitem_122, sum_114, permute_291, view_272, permute_287, view_269, getitem_119, sum_106, permute_281, view_265, permute_277, view_262, getitem_116, sum_98, permute_271, view_258, permute_267, view_255, getitem_113, sum_90, permute_261, view_251, permute_257, view_248, getitem_110, sum_82, permute_251, view_244, permute_247, view_241, getitem_107, sum_74, permute_241, view_237, permute_237, view_234, getitem_104, sum_66, permute_231, view_230, permute_227, view_227, getitem_101, sum_58, permute_221, view_223, permute_217, view_220, getitem_98, sum_50, permute_211, view_216, permute_207, view_213, getitem_95, sum_42, permute_201, view_209, permute_197, view_206, getitem_92, sum_30, getitem_89, sum_29, permute_189, view_202, permute_185, view_199, getitem_86, sum_21, permute_179, view_195, permute_175, view_192, getitem_83, sum_13, permute_169, view_188, permute_165, view_185, permute_158, view_181, None], self._out_spec)
    