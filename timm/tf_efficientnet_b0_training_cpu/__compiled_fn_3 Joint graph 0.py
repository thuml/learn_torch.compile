from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[32, 3, 3, 3]"; primals_2: "f32[32]"; primals_3: "f32[32]"; primals_4: "f32[32]"; primals_5: "f32[32]"; primals_6: "f32[16]"; primals_7: "f32[16]"; primals_8: "f32[96]"; primals_9: "f32[96]"; primals_10: "f32[96, 1, 3, 3]"; primals_11: "f32[96]"; primals_12: "f32[96]"; primals_13: "f32[24]"; primals_14: "f32[24]"; primals_15: "f32[144]"; primals_16: "f32[144]"; primals_17: "f32[144]"; primals_18: "f32[144]"; primals_19: "f32[24]"; primals_20: "f32[24]"; primals_21: "f32[144]"; primals_22: "f32[144]"; primals_23: "f32[144, 1, 5, 5]"; primals_24: "f32[144]"; primals_25: "f32[144]"; primals_26: "f32[40]"; primals_27: "f32[40]"; primals_28: "f32[240]"; primals_29: "f32[240]"; primals_30: "f32[240]"; primals_31: "f32[240]"; primals_32: "f32[40]"; primals_33: "f32[40]"; primals_34: "f32[240]"; primals_35: "f32[240]"; primals_36: "f32[240, 1, 3, 3]"; primals_37: "f32[240]"; primals_38: "f32[240]"; primals_39: "f32[80]"; primals_40: "f32[80]"; primals_41: "f32[480]"; primals_42: "f32[480]"; primals_43: "f32[480]"; primals_44: "f32[480]"; primals_45: "f32[80]"; primals_46: "f32[80]"; primals_47: "f32[480]"; primals_48: "f32[480]"; primals_49: "f32[480]"; primals_50: "f32[480]"; primals_51: "f32[80]"; primals_52: "f32[80]"; primals_53: "f32[480]"; primals_54: "f32[480]"; primals_55: "f32[480]"; primals_56: "f32[480]"; primals_57: "f32[112]"; primals_58: "f32[112]"; primals_59: "f32[672]"; primals_60: "f32[672]"; primals_61: "f32[672]"; primals_62: "f32[672]"; primals_63: "f32[112]"; primals_64: "f32[112]"; primals_65: "f32[672]"; primals_66: "f32[672]"; primals_67: "f32[672]"; primals_68: "f32[672]"; primals_69: "f32[112]"; primals_70: "f32[112]"; primals_71: "f32[672]"; primals_72: "f32[672]"; primals_73: "f32[672, 1, 5, 5]"; primals_74: "f32[672]"; primals_75: "f32[672]"; primals_76: "f32[192]"; primals_77: "f32[192]"; primals_78: "f32[1152]"; primals_79: "f32[1152]"; primals_80: "f32[1152]"; primals_81: "f32[1152]"; primals_82: "f32[192]"; primals_83: "f32[192]"; primals_84: "f32[1152]"; primals_85: "f32[1152]"; primals_86: "f32[1152]"; primals_87: "f32[1152]"; primals_88: "f32[192]"; primals_89: "f32[192]"; primals_90: "f32[1152]"; primals_91: "f32[1152]"; primals_92: "f32[1152]"; primals_93: "f32[1152]"; primals_94: "f32[192]"; primals_95: "f32[192]"; primals_96: "f32[1152]"; primals_97: "f32[1152]"; primals_98: "f32[1152]"; primals_99: "f32[1152]"; primals_100: "f32[320]"; primals_101: "f32[320]"; primals_102: "f32[1280]"; primals_103: "f32[1280]"; primals_104: "f32[32, 1, 3, 3]"; primals_105: "f32[8, 32, 1, 1]"; primals_106: "f32[8]"; primals_107: "f32[32, 8, 1, 1]"; primals_108: "f32[32]"; primals_109: "f32[16, 32, 1, 1]"; primals_110: "f32[96, 16, 1, 1]"; primals_111: "f32[4, 96, 1, 1]"; primals_112: "f32[4]"; primals_113: "f32[96, 4, 1, 1]"; primals_114: "f32[96]"; primals_115: "f32[24, 96, 1, 1]"; primals_116: "f32[144, 24, 1, 1]"; primals_117: "f32[144, 1, 3, 3]"; primals_118: "f32[6, 144, 1, 1]"; primals_119: "f32[6]"; primals_120: "f32[144, 6, 1, 1]"; primals_121: "f32[144]"; primals_122: "f32[24, 144, 1, 1]"; primals_123: "f32[144, 24, 1, 1]"; primals_124: "f32[6, 144, 1, 1]"; primals_125: "f32[6]"; primals_126: "f32[144, 6, 1, 1]"; primals_127: "f32[144]"; primals_128: "f32[40, 144, 1, 1]"; primals_129: "f32[240, 40, 1, 1]"; primals_130: "f32[240, 1, 5, 5]"; primals_131: "f32[10, 240, 1, 1]"; primals_132: "f32[10]"; primals_133: "f32[240, 10, 1, 1]"; primals_134: "f32[240]"; primals_135: "f32[40, 240, 1, 1]"; primals_136: "f32[240, 40, 1, 1]"; primals_137: "f32[10, 240, 1, 1]"; primals_138: "f32[10]"; primals_139: "f32[240, 10, 1, 1]"; primals_140: "f32[240]"; primals_141: "f32[80, 240, 1, 1]"; primals_142: "f32[480, 80, 1, 1]"; primals_143: "f32[480, 1, 3, 3]"; primals_144: "f32[20, 480, 1, 1]"; primals_145: "f32[20]"; primals_146: "f32[480, 20, 1, 1]"; primals_147: "f32[480]"; primals_148: "f32[80, 480, 1, 1]"; primals_149: "f32[480, 80, 1, 1]"; primals_150: "f32[480, 1, 3, 3]"; primals_151: "f32[20, 480, 1, 1]"; primals_152: "f32[20]"; primals_153: "f32[480, 20, 1, 1]"; primals_154: "f32[480]"; primals_155: "f32[80, 480, 1, 1]"; primals_156: "f32[480, 80, 1, 1]"; primals_157: "f32[480, 1, 5, 5]"; primals_158: "f32[20, 480, 1, 1]"; primals_159: "f32[20]"; primals_160: "f32[480, 20, 1, 1]"; primals_161: "f32[480]"; primals_162: "f32[112, 480, 1, 1]"; primals_163: "f32[672, 112, 1, 1]"; primals_164: "f32[672, 1, 5, 5]"; primals_165: "f32[28, 672, 1, 1]"; primals_166: "f32[28]"; primals_167: "f32[672, 28, 1, 1]"; primals_168: "f32[672]"; primals_169: "f32[112, 672, 1, 1]"; primals_170: "f32[672, 112, 1, 1]"; primals_171: "f32[672, 1, 5, 5]"; primals_172: "f32[28, 672, 1, 1]"; primals_173: "f32[28]"; primals_174: "f32[672, 28, 1, 1]"; primals_175: "f32[672]"; primals_176: "f32[112, 672, 1, 1]"; primals_177: "f32[672, 112, 1, 1]"; primals_178: "f32[28, 672, 1, 1]"; primals_179: "f32[28]"; primals_180: "f32[672, 28, 1, 1]"; primals_181: "f32[672]"; primals_182: "f32[192, 672, 1, 1]"; primals_183: "f32[1152, 192, 1, 1]"; primals_184: "f32[1152, 1, 5, 5]"; primals_185: "f32[48, 1152, 1, 1]"; primals_186: "f32[48]"; primals_187: "f32[1152, 48, 1, 1]"; primals_188: "f32[1152]"; primals_189: "f32[192, 1152, 1, 1]"; primals_190: "f32[1152, 192, 1, 1]"; primals_191: "f32[1152, 1, 5, 5]"; primals_192: "f32[48, 1152, 1, 1]"; primals_193: "f32[48]"; primals_194: "f32[1152, 48, 1, 1]"; primals_195: "f32[1152]"; primals_196: "f32[192, 1152, 1, 1]"; primals_197: "f32[1152, 192, 1, 1]"; primals_198: "f32[1152, 1, 5, 5]"; primals_199: "f32[48, 1152, 1, 1]"; primals_200: "f32[48]"; primals_201: "f32[1152, 48, 1, 1]"; primals_202: "f32[1152]"; primals_203: "f32[192, 1152, 1, 1]"; primals_204: "f32[1152, 192, 1, 1]"; primals_205: "f32[1152, 1, 3, 3]"; primals_206: "f32[48, 1152, 1, 1]"; primals_207: "f32[48]"; primals_208: "f32[1152, 48, 1, 1]"; primals_209: "f32[1152]"; primals_210: "f32[320, 1152, 1, 1]"; primals_211: "f32[1280, 320, 1, 1]"; primals_212: "f32[1000, 1280]"; primals_213: "f32[1000]"; primals_214: "i64[]"; primals_215: "f32[32]"; primals_216: "f32[32]"; primals_217: "i64[]"; primals_218: "f32[32]"; primals_219: "f32[32]"; primals_220: "i64[]"; primals_221: "f32[16]"; primals_222: "f32[16]"; primals_223: "i64[]"; primals_224: "f32[96]"; primals_225: "f32[96]"; primals_226: "i64[]"; primals_227: "f32[96]"; primals_228: "f32[96]"; primals_229: "i64[]"; primals_230: "f32[24]"; primals_231: "f32[24]"; primals_232: "i64[]"; primals_233: "f32[144]"; primals_234: "f32[144]"; primals_235: "i64[]"; primals_236: "f32[144]"; primals_237: "f32[144]"; primals_238: "i64[]"; primals_239: "f32[24]"; primals_240: "f32[24]"; primals_241: "i64[]"; primals_242: "f32[144]"; primals_243: "f32[144]"; primals_244: "i64[]"; primals_245: "f32[144]"; primals_246: "f32[144]"; primals_247: "i64[]"; primals_248: "f32[40]"; primals_249: "f32[40]"; primals_250: "i64[]"; primals_251: "f32[240]"; primals_252: "f32[240]"; primals_253: "i64[]"; primals_254: "f32[240]"; primals_255: "f32[240]"; primals_256: "i64[]"; primals_257: "f32[40]"; primals_258: "f32[40]"; primals_259: "i64[]"; primals_260: "f32[240]"; primals_261: "f32[240]"; primals_262: "i64[]"; primals_263: "f32[240]"; primals_264: "f32[240]"; primals_265: "i64[]"; primals_266: "f32[80]"; primals_267: "f32[80]"; primals_268: "i64[]"; primals_269: "f32[480]"; primals_270: "f32[480]"; primals_271: "i64[]"; primals_272: "f32[480]"; primals_273: "f32[480]"; primals_274: "i64[]"; primals_275: "f32[80]"; primals_276: "f32[80]"; primals_277: "i64[]"; primals_278: "f32[480]"; primals_279: "f32[480]"; primals_280: "i64[]"; primals_281: "f32[480]"; primals_282: "f32[480]"; primals_283: "i64[]"; primals_284: "f32[80]"; primals_285: "f32[80]"; primals_286: "i64[]"; primals_287: "f32[480]"; primals_288: "f32[480]"; primals_289: "i64[]"; primals_290: "f32[480]"; primals_291: "f32[480]"; primals_292: "i64[]"; primals_293: "f32[112]"; primals_294: "f32[112]"; primals_295: "i64[]"; primals_296: "f32[672]"; primals_297: "f32[672]"; primals_298: "i64[]"; primals_299: "f32[672]"; primals_300: "f32[672]"; primals_301: "i64[]"; primals_302: "f32[112]"; primals_303: "f32[112]"; primals_304: "i64[]"; primals_305: "f32[672]"; primals_306: "f32[672]"; primals_307: "i64[]"; primals_308: "f32[672]"; primals_309: "f32[672]"; primals_310: "i64[]"; primals_311: "f32[112]"; primals_312: "f32[112]"; primals_313: "i64[]"; primals_314: "f32[672]"; primals_315: "f32[672]"; primals_316: "i64[]"; primals_317: "f32[672]"; primals_318: "f32[672]"; primals_319: "i64[]"; primals_320: "f32[192]"; primals_321: "f32[192]"; primals_322: "i64[]"; primals_323: "f32[1152]"; primals_324: "f32[1152]"; primals_325: "i64[]"; primals_326: "f32[1152]"; primals_327: "f32[1152]"; primals_328: "i64[]"; primals_329: "f32[192]"; primals_330: "f32[192]"; primals_331: "i64[]"; primals_332: "f32[1152]"; primals_333: "f32[1152]"; primals_334: "i64[]"; primals_335: "f32[1152]"; primals_336: "f32[1152]"; primals_337: "i64[]"; primals_338: "f32[192]"; primals_339: "f32[192]"; primals_340: "i64[]"; primals_341: "f32[1152]"; primals_342: "f32[1152]"; primals_343: "i64[]"; primals_344: "f32[1152]"; primals_345: "f32[1152]"; primals_346: "i64[]"; primals_347: "f32[192]"; primals_348: "f32[192]"; primals_349: "i64[]"; primals_350: "f32[1152]"; primals_351: "f32[1152]"; primals_352: "i64[]"; primals_353: "f32[1152]"; primals_354: "f32[1152]"; primals_355: "i64[]"; primals_356: "f32[320]"; primals_357: "f32[320]"; primals_358: "i64[]"; primals_359: "f32[1280]"; primals_360: "f32[1280]"; primals_361: "f32[8, 3, 224, 224]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd: "f32[8, 3, 225, 225]" = torch.ops.aten.constant_pad_nd.default(primals_361, [0, 1, 0, 1], 0.0);  primals_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution: "f32[8, 32, 112, 112]" = torch.ops.aten.convolution.default(constant_pad_nd, primals_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_214, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
    getitem: "f32[1, 32, 1, 1]" = var_mean[0]
    getitem_1: "f32[1, 32, 1, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 0.001)
    rsqrt: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
    mul: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    squeeze: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
    squeeze_1: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
    mul_1: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
    mul_2: "f32[32]" = torch.ops.aten.mul.Tensor(primals_215, 0.9)
    add_2: "f32[32]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[32]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.00000996502277);  squeeze_2 = None
    mul_4: "f32[32]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[32]" = torch.ops.aten.mul.Tensor(primals_216, 0.9)
    add_3: "f32[32]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1)
    unsqueeze_1: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1);  primals_3 = None
    unsqueeze_3: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone: "f32[8, 32, 112, 112]" = torch.ops.aten.clone.default(add_4)
    sigmoid: "f32[8, 32, 112, 112]" = torch.ops.aten.sigmoid.default(add_4)
    mul_7: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(add_4, sigmoid);  add_4 = sigmoid = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_1: "f32[8, 32, 112, 112]" = torch.ops.aten.convolution.default(mul_7, primals_104, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_217, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[1, 32, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 32, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 0.001)
    rsqrt_1: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_1: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
    mul_8: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
    squeeze_4: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_9: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_10: "f32[32]" = torch.ops.aten.mul.Tensor(primals_218, 0.9)
    add_7: "f32[32]" = torch.ops.aten.add.Tensor(mul_9, mul_10);  mul_9 = mul_10 = None
    squeeze_5: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_11: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.00000996502277);  squeeze_5 = None
    mul_12: "f32[32]" = torch.ops.aten.mul.Tensor(mul_11, 0.1);  mul_11 = None
    mul_13: "f32[32]" = torch.ops.aten.mul.Tensor(primals_219, 0.9)
    add_8: "f32[32]" = torch.ops.aten.add.Tensor(mul_12, mul_13);  mul_12 = mul_13 = None
    unsqueeze_4: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_4, -1)
    unsqueeze_5: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_14: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_8, unsqueeze_5);  mul_8 = unsqueeze_5 = None
    unsqueeze_6: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1);  primals_5 = None
    unsqueeze_7: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_9: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_7);  mul_14 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_1: "f32[8, 32, 112, 112]" = torch.ops.aten.clone.default(add_9)
    sigmoid_1: "f32[8, 32, 112, 112]" = torch.ops.aten.sigmoid.default(add_9)
    mul_15: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(add_9, sigmoid_1);  add_9 = sigmoid_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean: "f32[8, 32, 1, 1]" = torch.ops.aten.mean.dim(mul_15, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_2: "f32[8, 8, 1, 1]" = torch.ops.aten.convolution.default(mean, primals_105, primals_106, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_2: "f32[8, 8, 1, 1]" = torch.ops.aten.clone.default(convolution_2)
    sigmoid_2: "f32[8, 8, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_2)
    mul_16: "f32[8, 8, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_2, sigmoid_2);  convolution_2 = sigmoid_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_3: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mul_16, primals_107, primals_108, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_3: "f32[8, 32, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_3);  convolution_3 = None
    alias: "f32[8, 32, 1, 1]" = torch.ops.aten.alias.default(sigmoid_3)
    mul_17: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_15, sigmoid_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_4: "f32[8, 16, 112, 112]" = torch.ops.aten.convolution.default(mul_17, primals_109, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_220, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 16, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 16, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 0.001)
    rsqrt_2: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_2: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_5)
    mul_18: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_19: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_20: "f32[16]" = torch.ops.aten.mul.Tensor(primals_221, 0.9)
    add_12: "f32[16]" = torch.ops.aten.add.Tensor(mul_19, mul_20);  mul_19 = mul_20 = None
    squeeze_8: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_21: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.00000996502277);  squeeze_8 = None
    mul_22: "f32[16]" = torch.ops.aten.mul.Tensor(mul_21, 0.1);  mul_21 = None
    mul_23: "f32[16]" = torch.ops.aten.mul.Tensor(primals_222, 0.9)
    add_13: "f32[16]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
    unsqueeze_8: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1)
    unsqueeze_9: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_24: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_18, unsqueeze_9);  mul_18 = unsqueeze_9 = None
    unsqueeze_10: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1);  primals_7 = None
    unsqueeze_11: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_14: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_24, unsqueeze_11);  mul_24 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_5: "f32[8, 96, 112, 112]" = torch.ops.aten.convolution.default(add_14, primals_110, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_15: "i64[]" = torch.ops.aten.add.Tensor(primals_223, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_3 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
    getitem_6: "f32[1, 96, 1, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 96, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_16: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 0.001)
    rsqrt_3: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_3: "f32[8, 96, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_7)
    mul_25: "f32[8, 96, 112, 112]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
    squeeze_10: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_26: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_27: "f32[96]" = torch.ops.aten.mul.Tensor(primals_224, 0.9)
    add_17: "f32[96]" = torch.ops.aten.add.Tensor(mul_26, mul_27);  mul_26 = mul_27 = None
    squeeze_11: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
    mul_28: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.00000996502277);  squeeze_11 = None
    mul_29: "f32[96]" = torch.ops.aten.mul.Tensor(mul_28, 0.1);  mul_28 = None
    mul_30: "f32[96]" = torch.ops.aten.mul.Tensor(primals_225, 0.9)
    add_18: "f32[96]" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
    unsqueeze_12: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1)
    unsqueeze_13: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_31: "f32[8, 96, 112, 112]" = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_13);  mul_25 = unsqueeze_13 = None
    unsqueeze_14: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1);  primals_9 = None
    unsqueeze_15: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_19: "f32[8, 96, 112, 112]" = torch.ops.aten.add.Tensor(mul_31, unsqueeze_15);  mul_31 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_3: "f32[8, 96, 112, 112]" = torch.ops.aten.clone.default(add_19)
    sigmoid_4: "f32[8, 96, 112, 112]" = torch.ops.aten.sigmoid.default(add_19)
    mul_32: "f32[8, 96, 112, 112]" = torch.ops.aten.mul.Tensor(add_19, sigmoid_4);  add_19 = sigmoid_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_1: "f32[8, 96, 113, 113]" = torch.ops.aten.constant_pad_nd.default(mul_32, [0, 1, 0, 1], 0.0);  mul_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_6: "f32[8, 96, 56, 56]" = torch.ops.aten.convolution.default(constant_pad_nd_1, primals_10, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 96)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_20: "i64[]" = torch.ops.aten.add.Tensor(primals_226, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_4 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
    getitem_8: "f32[1, 96, 1, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 96, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_21: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 0.001)
    rsqrt_4: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_4: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_9)
    mul_33: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
    squeeze_13: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
    mul_34: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_35: "f32[96]" = torch.ops.aten.mul.Tensor(primals_227, 0.9)
    add_22: "f32[96]" = torch.ops.aten.add.Tensor(mul_34, mul_35);  mul_34 = mul_35 = None
    squeeze_14: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
    mul_36: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000398612827361);  squeeze_14 = None
    mul_37: "f32[96]" = torch.ops.aten.mul.Tensor(mul_36, 0.1);  mul_36 = None
    mul_38: "f32[96]" = torch.ops.aten.mul.Tensor(primals_228, 0.9)
    add_23: "f32[96]" = torch.ops.aten.add.Tensor(mul_37, mul_38);  mul_37 = mul_38 = None
    unsqueeze_16: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_17: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    mul_39: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(mul_33, unsqueeze_17);  mul_33 = unsqueeze_17 = None
    unsqueeze_18: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_19: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    add_24: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_39, unsqueeze_19);  mul_39 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_4: "f32[8, 96, 56, 56]" = torch.ops.aten.clone.default(add_24)
    sigmoid_5: "f32[8, 96, 56, 56]" = torch.ops.aten.sigmoid.default(add_24)
    mul_40: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_24, sigmoid_5);  add_24 = sigmoid_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_1: "f32[8, 96, 1, 1]" = torch.ops.aten.mean.dim(mul_40, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_7: "f32[8, 4, 1, 1]" = torch.ops.aten.convolution.default(mean_1, primals_111, primals_112, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_5: "f32[8, 4, 1, 1]" = torch.ops.aten.clone.default(convolution_7)
    sigmoid_6: "f32[8, 4, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_7)
    mul_41: "f32[8, 4, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_7, sigmoid_6);  convolution_7 = sigmoid_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_8: "f32[8, 96, 1, 1]" = torch.ops.aten.convolution.default(mul_41, primals_113, primals_114, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_7: "f32[8, 96, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_8);  convolution_8 = None
    alias_1: "f32[8, 96, 1, 1]" = torch.ops.aten.alias.default(sigmoid_7)
    mul_42: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(mul_40, sigmoid_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_9: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(mul_42, primals_115, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_25: "i64[]" = torch.ops.aten.add.Tensor(primals_229, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_5 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
    getitem_10: "f32[1, 24, 1, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 24, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_26: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 0.001)
    rsqrt_5: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_5: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_11)
    mul_43: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_15: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
    squeeze_16: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    mul_44: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_45: "f32[24]" = torch.ops.aten.mul.Tensor(primals_230, 0.9)
    add_27: "f32[24]" = torch.ops.aten.add.Tensor(mul_44, mul_45);  mul_44 = mul_45 = None
    squeeze_17: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
    mul_46: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000398612827361);  squeeze_17 = None
    mul_47: "f32[24]" = torch.ops.aten.mul.Tensor(mul_46, 0.1);  mul_46 = None
    mul_48: "f32[24]" = torch.ops.aten.mul.Tensor(primals_231, 0.9)
    add_28: "f32[24]" = torch.ops.aten.add.Tensor(mul_47, mul_48);  mul_47 = mul_48 = None
    unsqueeze_20: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1)
    unsqueeze_21: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_49: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_21);  mul_43 = unsqueeze_21 = None
    unsqueeze_22: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1);  primals_14 = None
    unsqueeze_23: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_29: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_49, unsqueeze_23);  mul_49 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_10: "f32[8, 144, 56, 56]" = torch.ops.aten.convolution.default(add_29, primals_116, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_30: "i64[]" = torch.ops.aten.add.Tensor(primals_232, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_6 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
    getitem_12: "f32[1, 144, 1, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 144, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_31: "f32[1, 144, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 0.001)
    rsqrt_6: "f32[1, 144, 1, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_6: "f32[8, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_13)
    mul_50: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_18: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
    squeeze_19: "f32[144]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
    mul_51: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_52: "f32[144]" = torch.ops.aten.mul.Tensor(primals_233, 0.9)
    add_32: "f32[144]" = torch.ops.aten.add.Tensor(mul_51, mul_52);  mul_51 = mul_52 = None
    squeeze_20: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
    mul_53: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0000398612827361);  squeeze_20 = None
    mul_54: "f32[144]" = torch.ops.aten.mul.Tensor(mul_53, 0.1);  mul_53 = None
    mul_55: "f32[144]" = torch.ops.aten.mul.Tensor(primals_234, 0.9)
    add_33: "f32[144]" = torch.ops.aten.add.Tensor(mul_54, mul_55);  mul_54 = mul_55 = None
    unsqueeze_24: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1)
    unsqueeze_25: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    mul_56: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_50, unsqueeze_25);  mul_50 = unsqueeze_25 = None
    unsqueeze_26: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1);  primals_16 = None
    unsqueeze_27: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    add_34: "f32[8, 144, 56, 56]" = torch.ops.aten.add.Tensor(mul_56, unsqueeze_27);  mul_56 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_6: "f32[8, 144, 56, 56]" = torch.ops.aten.clone.default(add_34)
    sigmoid_8: "f32[8, 144, 56, 56]" = torch.ops.aten.sigmoid.default(add_34)
    mul_57: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(add_34, sigmoid_8);  add_34 = sigmoid_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_11: "f32[8, 144, 56, 56]" = torch.ops.aten.convolution.default(mul_57, primals_117, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 144)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_35: "i64[]" = torch.ops.aten.add.Tensor(primals_235, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_7 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
    getitem_14: "f32[1, 144, 1, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 144, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_36: "f32[1, 144, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 0.001)
    rsqrt_7: "f32[1, 144, 1, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_7: "f32[8, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_15)
    mul_58: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_21: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
    squeeze_22: "f32[144]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
    mul_59: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_60: "f32[144]" = torch.ops.aten.mul.Tensor(primals_236, 0.9)
    add_37: "f32[144]" = torch.ops.aten.add.Tensor(mul_59, mul_60);  mul_59 = mul_60 = None
    squeeze_23: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
    mul_61: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0000398612827361);  squeeze_23 = None
    mul_62: "f32[144]" = torch.ops.aten.mul.Tensor(mul_61, 0.1);  mul_61 = None
    mul_63: "f32[144]" = torch.ops.aten.mul.Tensor(primals_237, 0.9)
    add_38: "f32[144]" = torch.ops.aten.add.Tensor(mul_62, mul_63);  mul_62 = mul_63 = None
    unsqueeze_28: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_29: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_64: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_29);  mul_58 = unsqueeze_29 = None
    unsqueeze_30: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_31: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_39: "f32[8, 144, 56, 56]" = torch.ops.aten.add.Tensor(mul_64, unsqueeze_31);  mul_64 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_7: "f32[8, 144, 56, 56]" = torch.ops.aten.clone.default(add_39)
    sigmoid_9: "f32[8, 144, 56, 56]" = torch.ops.aten.sigmoid.default(add_39)
    mul_65: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(add_39, sigmoid_9);  add_39 = sigmoid_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_2: "f32[8, 144, 1, 1]" = torch.ops.aten.mean.dim(mul_65, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_12: "f32[8, 6, 1, 1]" = torch.ops.aten.convolution.default(mean_2, primals_118, primals_119, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_8: "f32[8, 6, 1, 1]" = torch.ops.aten.clone.default(convolution_12)
    sigmoid_10: "f32[8, 6, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_12)
    mul_66: "f32[8, 6, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_12, sigmoid_10);  convolution_12 = sigmoid_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_13: "f32[8, 144, 1, 1]" = torch.ops.aten.convolution.default(mul_66, primals_120, primals_121, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_11: "f32[8, 144, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_13);  convolution_13 = None
    alias_2: "f32[8, 144, 1, 1]" = torch.ops.aten.alias.default(sigmoid_11)
    mul_67: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_65, sigmoid_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_14: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(mul_67, primals_122, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_40: "i64[]" = torch.ops.aten.add.Tensor(primals_238, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_8 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
    getitem_16: "f32[1, 24, 1, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 24, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_41: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 0.001)
    rsqrt_8: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
    sub_8: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_17)
    mul_68: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_24: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
    squeeze_25: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
    mul_69: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_70: "f32[24]" = torch.ops.aten.mul.Tensor(primals_239, 0.9)
    add_42: "f32[24]" = torch.ops.aten.add.Tensor(mul_69, mul_70);  mul_69 = mul_70 = None
    squeeze_26: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
    mul_71: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0000398612827361);  squeeze_26 = None
    mul_72: "f32[24]" = torch.ops.aten.mul.Tensor(mul_71, 0.1);  mul_71 = None
    mul_73: "f32[24]" = torch.ops.aten.mul.Tensor(primals_240, 0.9)
    add_43: "f32[24]" = torch.ops.aten.add.Tensor(mul_72, mul_73);  mul_72 = mul_73 = None
    unsqueeze_32: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
    unsqueeze_33: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    mul_74: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_68, unsqueeze_33);  mul_68 = unsqueeze_33 = None
    unsqueeze_34: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
    unsqueeze_35: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    add_44: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_74, unsqueeze_35);  mul_74 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_45: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(add_44, add_29);  add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_15: "f32[8, 144, 56, 56]" = torch.ops.aten.convolution.default(add_45, primals_123, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_46: "i64[]" = torch.ops.aten.add.Tensor(primals_241, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_9 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
    getitem_18: "f32[1, 144, 1, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 144, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_47: "f32[1, 144, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 0.001)
    rsqrt_9: "f32[1, 144, 1, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_9: "f32[8, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_19)
    mul_75: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_27: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
    squeeze_28: "f32[144]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_76: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_77: "f32[144]" = torch.ops.aten.mul.Tensor(primals_242, 0.9)
    add_48: "f32[144]" = torch.ops.aten.add.Tensor(mul_76, mul_77);  mul_76 = mul_77 = None
    squeeze_29: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
    mul_78: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0000398612827361);  squeeze_29 = None
    mul_79: "f32[144]" = torch.ops.aten.mul.Tensor(mul_78, 0.1);  mul_78 = None
    mul_80: "f32[144]" = torch.ops.aten.mul.Tensor(primals_243, 0.9)
    add_49: "f32[144]" = torch.ops.aten.add.Tensor(mul_79, mul_80);  mul_79 = mul_80 = None
    unsqueeze_36: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1)
    unsqueeze_37: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_81: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_75, unsqueeze_37);  mul_75 = unsqueeze_37 = None
    unsqueeze_38: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1);  primals_22 = None
    unsqueeze_39: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_50: "f32[8, 144, 56, 56]" = torch.ops.aten.add.Tensor(mul_81, unsqueeze_39);  mul_81 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_9: "f32[8, 144, 56, 56]" = torch.ops.aten.clone.default(add_50)
    sigmoid_12: "f32[8, 144, 56, 56]" = torch.ops.aten.sigmoid.default(add_50)
    mul_82: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(add_50, sigmoid_12);  add_50 = sigmoid_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_2: "f32[8, 144, 59, 59]" = torch.ops.aten.constant_pad_nd.default(mul_82, [1, 2, 1, 2], 0.0);  mul_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_16: "f32[8, 144, 28, 28]" = torch.ops.aten.convolution.default(constant_pad_nd_2, primals_23, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 144)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_51: "i64[]" = torch.ops.aten.add.Tensor(primals_244, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_10 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[1, 144, 1, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 144, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_52: "f32[1, 144, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 0.001)
    rsqrt_10: "f32[1, 144, 1, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_10: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_21)
    mul_83: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_30: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
    squeeze_31: "f32[144]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
    mul_84: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_85: "f32[144]" = torch.ops.aten.mul.Tensor(primals_245, 0.9)
    add_53: "f32[144]" = torch.ops.aten.add.Tensor(mul_84, mul_85);  mul_84 = mul_85 = None
    squeeze_32: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
    mul_86: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0001594642002871);  squeeze_32 = None
    mul_87: "f32[144]" = torch.ops.aten.mul.Tensor(mul_86, 0.1);  mul_86 = None
    mul_88: "f32[144]" = torch.ops.aten.mul.Tensor(primals_246, 0.9)
    add_54: "f32[144]" = torch.ops.aten.add.Tensor(mul_87, mul_88);  mul_87 = mul_88 = None
    unsqueeze_40: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1)
    unsqueeze_41: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    mul_89: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(mul_83, unsqueeze_41);  mul_83 = unsqueeze_41 = None
    unsqueeze_42: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1);  primals_25 = None
    unsqueeze_43: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    add_55: "f32[8, 144, 28, 28]" = torch.ops.aten.add.Tensor(mul_89, unsqueeze_43);  mul_89 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_10: "f32[8, 144, 28, 28]" = torch.ops.aten.clone.default(add_55)
    sigmoid_13: "f32[8, 144, 28, 28]" = torch.ops.aten.sigmoid.default(add_55)
    mul_90: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(add_55, sigmoid_13);  add_55 = sigmoid_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_3: "f32[8, 144, 1, 1]" = torch.ops.aten.mean.dim(mul_90, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_17: "f32[8, 6, 1, 1]" = torch.ops.aten.convolution.default(mean_3, primals_124, primals_125, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_11: "f32[8, 6, 1, 1]" = torch.ops.aten.clone.default(convolution_17)
    sigmoid_14: "f32[8, 6, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_17)
    mul_91: "f32[8, 6, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_17, sigmoid_14);  convolution_17 = sigmoid_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_18: "f32[8, 144, 1, 1]" = torch.ops.aten.convolution.default(mul_91, primals_126, primals_127, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_15: "f32[8, 144, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_18);  convolution_18 = None
    alias_3: "f32[8, 144, 1, 1]" = torch.ops.aten.alias.default(sigmoid_15)
    mul_92: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(mul_90, sigmoid_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_19: "f32[8, 40, 28, 28]" = torch.ops.aten.convolution.default(mul_92, primals_128, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_56: "i64[]" = torch.ops.aten.add.Tensor(primals_247, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_11 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[1, 40, 1, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 40, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_57: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 0.001)
    rsqrt_11: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    sub_11: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_23)
    mul_93: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    squeeze_33: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
    squeeze_34: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
    mul_94: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_95: "f32[40]" = torch.ops.aten.mul.Tensor(primals_248, 0.9)
    add_58: "f32[40]" = torch.ops.aten.add.Tensor(mul_94, mul_95);  mul_94 = mul_95 = None
    squeeze_35: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
    mul_96: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0001594642002871);  squeeze_35 = None
    mul_97: "f32[40]" = torch.ops.aten.mul.Tensor(mul_96, 0.1);  mul_96 = None
    mul_98: "f32[40]" = torch.ops.aten.mul.Tensor(primals_249, 0.9)
    add_59: "f32[40]" = torch.ops.aten.add.Tensor(mul_97, mul_98);  mul_97 = mul_98 = None
    unsqueeze_44: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1)
    unsqueeze_45: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_99: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_93, unsqueeze_45);  mul_93 = unsqueeze_45 = None
    unsqueeze_46: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1);  primals_27 = None
    unsqueeze_47: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_60: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_99, unsqueeze_47);  mul_99 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_20: "f32[8, 240, 28, 28]" = torch.ops.aten.convolution.default(add_60, primals_129, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_61: "i64[]" = torch.ops.aten.add.Tensor(primals_250, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_12 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
    getitem_24: "f32[1, 240, 1, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 240, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_62: "f32[1, 240, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 0.001)
    rsqrt_12: "f32[1, 240, 1, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    sub_12: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_25)
    mul_100: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_36: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
    squeeze_37: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
    mul_101: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_102: "f32[240]" = torch.ops.aten.mul.Tensor(primals_251, 0.9)
    add_63: "f32[240]" = torch.ops.aten.add.Tensor(mul_101, mul_102);  mul_101 = mul_102 = None
    squeeze_38: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
    mul_103: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0001594642002871);  squeeze_38 = None
    mul_104: "f32[240]" = torch.ops.aten.mul.Tensor(mul_103, 0.1);  mul_103 = None
    mul_105: "f32[240]" = torch.ops.aten.mul.Tensor(primals_252, 0.9)
    add_64: "f32[240]" = torch.ops.aten.add.Tensor(mul_104, mul_105);  mul_104 = mul_105 = None
    unsqueeze_48: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1)
    unsqueeze_49: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    mul_106: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_100, unsqueeze_49);  mul_100 = unsqueeze_49 = None
    unsqueeze_50: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1);  primals_29 = None
    unsqueeze_51: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    add_65: "f32[8, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_106, unsqueeze_51);  mul_106 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_12: "f32[8, 240, 28, 28]" = torch.ops.aten.clone.default(add_65)
    sigmoid_16: "f32[8, 240, 28, 28]" = torch.ops.aten.sigmoid.default(add_65)
    mul_107: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_65, sigmoid_16);  add_65 = sigmoid_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_21: "f32[8, 240, 28, 28]" = torch.ops.aten.convolution.default(mul_107, primals_130, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 240)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_66: "i64[]" = torch.ops.aten.add.Tensor(primals_253, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_13 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
    getitem_26: "f32[1, 240, 1, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 240, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_67: "f32[1, 240, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 0.001)
    rsqrt_13: "f32[1, 240, 1, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    sub_13: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_27)
    mul_108: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_39: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
    squeeze_40: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_109: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_110: "f32[240]" = torch.ops.aten.mul.Tensor(primals_254, 0.9)
    add_68: "f32[240]" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
    squeeze_41: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
    mul_111: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0001594642002871);  squeeze_41 = None
    mul_112: "f32[240]" = torch.ops.aten.mul.Tensor(mul_111, 0.1);  mul_111 = None
    mul_113: "f32[240]" = torch.ops.aten.mul.Tensor(primals_255, 0.9)
    add_69: "f32[240]" = torch.ops.aten.add.Tensor(mul_112, mul_113);  mul_112 = mul_113 = None
    unsqueeze_52: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1)
    unsqueeze_53: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_114: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_108, unsqueeze_53);  mul_108 = unsqueeze_53 = None
    unsqueeze_54: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1);  primals_31 = None
    unsqueeze_55: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_70: "f32[8, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_114, unsqueeze_55);  mul_114 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_13: "f32[8, 240, 28, 28]" = torch.ops.aten.clone.default(add_70)
    sigmoid_17: "f32[8, 240, 28, 28]" = torch.ops.aten.sigmoid.default(add_70)
    mul_115: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_70, sigmoid_17);  add_70 = sigmoid_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_4: "f32[8, 240, 1, 1]" = torch.ops.aten.mean.dim(mul_115, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_22: "f32[8, 10, 1, 1]" = torch.ops.aten.convolution.default(mean_4, primals_131, primals_132, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_14: "f32[8, 10, 1, 1]" = torch.ops.aten.clone.default(convolution_22)
    sigmoid_18: "f32[8, 10, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_22)
    mul_116: "f32[8, 10, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_22, sigmoid_18);  convolution_22 = sigmoid_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_23: "f32[8, 240, 1, 1]" = torch.ops.aten.convolution.default(mul_116, primals_133, primals_134, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_19: "f32[8, 240, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_23);  convolution_23 = None
    alias_4: "f32[8, 240, 1, 1]" = torch.ops.aten.alias.default(sigmoid_19)
    mul_117: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_115, sigmoid_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_24: "f32[8, 40, 28, 28]" = torch.ops.aten.convolution.default(mul_117, primals_135, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_71: "i64[]" = torch.ops.aten.add.Tensor(primals_256, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_14 = torch.ops.aten.var_mean.correction(convolution_24, [0, 2, 3], correction = 0, keepdim = True)
    getitem_28: "f32[1, 40, 1, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 40, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_72: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 0.001)
    rsqrt_14: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
    sub_14: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, getitem_29)
    mul_118: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_42: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
    squeeze_43: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
    mul_119: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_120: "f32[40]" = torch.ops.aten.mul.Tensor(primals_257, 0.9)
    add_73: "f32[40]" = torch.ops.aten.add.Tensor(mul_119, mul_120);  mul_119 = mul_120 = None
    squeeze_44: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
    mul_121: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0001594642002871);  squeeze_44 = None
    mul_122: "f32[40]" = torch.ops.aten.mul.Tensor(mul_121, 0.1);  mul_121 = None
    mul_123: "f32[40]" = torch.ops.aten.mul.Tensor(primals_258, 0.9)
    add_74: "f32[40]" = torch.ops.aten.add.Tensor(mul_122, mul_123);  mul_122 = mul_123 = None
    unsqueeze_56: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1)
    unsqueeze_57: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    mul_124: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_118, unsqueeze_57);  mul_118 = unsqueeze_57 = None
    unsqueeze_58: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1);  primals_33 = None
    unsqueeze_59: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    add_75: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_124, unsqueeze_59);  mul_124 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_76: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_75, add_60);  add_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_25: "f32[8, 240, 28, 28]" = torch.ops.aten.convolution.default(add_76, primals_136, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_77: "i64[]" = torch.ops.aten.add.Tensor(primals_259, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_15 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
    getitem_30: "f32[1, 240, 1, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 240, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_78: "f32[1, 240, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 0.001)
    rsqrt_15: "f32[1, 240, 1, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_15: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_31)
    mul_125: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    squeeze_45: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
    squeeze_46: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
    mul_126: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_127: "f32[240]" = torch.ops.aten.mul.Tensor(primals_260, 0.9)
    add_79: "f32[240]" = torch.ops.aten.add.Tensor(mul_126, mul_127);  mul_126 = mul_127 = None
    squeeze_47: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
    mul_128: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0001594642002871);  squeeze_47 = None
    mul_129: "f32[240]" = torch.ops.aten.mul.Tensor(mul_128, 0.1);  mul_128 = None
    mul_130: "f32[240]" = torch.ops.aten.mul.Tensor(primals_261, 0.9)
    add_80: "f32[240]" = torch.ops.aten.add.Tensor(mul_129, mul_130);  mul_129 = mul_130 = None
    unsqueeze_60: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1)
    unsqueeze_61: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_131: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_125, unsqueeze_61);  mul_125 = unsqueeze_61 = None
    unsqueeze_62: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1);  primals_35 = None
    unsqueeze_63: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_81: "f32[8, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_131, unsqueeze_63);  mul_131 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_15: "f32[8, 240, 28, 28]" = torch.ops.aten.clone.default(add_81)
    sigmoid_20: "f32[8, 240, 28, 28]" = torch.ops.aten.sigmoid.default(add_81)
    mul_132: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_81, sigmoid_20);  add_81 = sigmoid_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_3: "f32[8, 240, 29, 29]" = torch.ops.aten.constant_pad_nd.default(mul_132, [0, 1, 0, 1], 0.0);  mul_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_26: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(constant_pad_nd_3, primals_36, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 240)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_82: "i64[]" = torch.ops.aten.add.Tensor(primals_262, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_16 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
    getitem_32: "f32[1, 240, 1, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 240, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_83: "f32[1, 240, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 0.001)
    rsqrt_16: "f32[1, 240, 1, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
    sub_16: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_33)
    mul_133: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_48: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
    squeeze_49: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
    mul_134: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_135: "f32[240]" = torch.ops.aten.mul.Tensor(primals_263, 0.9)
    add_84: "f32[240]" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
    squeeze_50: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
    mul_136: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0006381620931717);  squeeze_50 = None
    mul_137: "f32[240]" = torch.ops.aten.mul.Tensor(mul_136, 0.1);  mul_136 = None
    mul_138: "f32[240]" = torch.ops.aten.mul.Tensor(primals_264, 0.9)
    add_85: "f32[240]" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
    unsqueeze_64: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1)
    unsqueeze_65: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    mul_139: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_65);  mul_133 = unsqueeze_65 = None
    unsqueeze_66: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1);  primals_38 = None
    unsqueeze_67: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    add_86: "f32[8, 240, 14, 14]" = torch.ops.aten.add.Tensor(mul_139, unsqueeze_67);  mul_139 = unsqueeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_16: "f32[8, 240, 14, 14]" = torch.ops.aten.clone.default(add_86)
    sigmoid_21: "f32[8, 240, 14, 14]" = torch.ops.aten.sigmoid.default(add_86)
    mul_140: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(add_86, sigmoid_21);  add_86 = sigmoid_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_5: "f32[8, 240, 1, 1]" = torch.ops.aten.mean.dim(mul_140, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_27: "f32[8, 10, 1, 1]" = torch.ops.aten.convolution.default(mean_5, primals_137, primals_138, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_17: "f32[8, 10, 1, 1]" = torch.ops.aten.clone.default(convolution_27)
    sigmoid_22: "f32[8, 10, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_27)
    mul_141: "f32[8, 10, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_27, sigmoid_22);  convolution_27 = sigmoid_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_28: "f32[8, 240, 1, 1]" = torch.ops.aten.convolution.default(mul_141, primals_139, primals_140, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_23: "f32[8, 240, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_28);  convolution_28 = None
    alias_5: "f32[8, 240, 1, 1]" = torch.ops.aten.alias.default(sigmoid_23)
    mul_142: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_140, sigmoid_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_29: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(mul_142, primals_141, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_87: "i64[]" = torch.ops.aten.add.Tensor(primals_265, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_17 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
    getitem_34: "f32[1, 80, 1, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 80, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_88: "f32[1, 80, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 0.001)
    rsqrt_17: "f32[1, 80, 1, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
    sub_17: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, getitem_35)
    mul_143: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    squeeze_51: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
    squeeze_52: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
    mul_144: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_145: "f32[80]" = torch.ops.aten.mul.Tensor(primals_266, 0.9)
    add_89: "f32[80]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
    squeeze_53: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
    mul_146: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0006381620931717);  squeeze_53 = None
    mul_147: "f32[80]" = torch.ops.aten.mul.Tensor(mul_146, 0.1);  mul_146 = None
    mul_148: "f32[80]" = torch.ops.aten.mul.Tensor(primals_267, 0.9)
    add_90: "f32[80]" = torch.ops.aten.add.Tensor(mul_147, mul_148);  mul_147 = mul_148 = None
    unsqueeze_68: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1)
    unsqueeze_69: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_149: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_143, unsqueeze_69);  mul_143 = unsqueeze_69 = None
    unsqueeze_70: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1);  primals_40 = None
    unsqueeze_71: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_91: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_71);  mul_149 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_30: "f32[8, 480, 14, 14]" = torch.ops.aten.convolution.default(add_91, primals_142, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_92: "i64[]" = torch.ops.aten.add.Tensor(primals_268, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_18 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
    getitem_36: "f32[1, 480, 1, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 480, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_93: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 0.001)
    rsqrt_18: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
    sub_18: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, getitem_37)
    mul_150: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    squeeze_54: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
    squeeze_55: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
    mul_151: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_152: "f32[480]" = torch.ops.aten.mul.Tensor(primals_269, 0.9)
    add_94: "f32[480]" = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
    squeeze_56: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
    mul_153: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0006381620931717);  squeeze_56 = None
    mul_154: "f32[480]" = torch.ops.aten.mul.Tensor(mul_153, 0.1);  mul_153 = None
    mul_155: "f32[480]" = torch.ops.aten.mul.Tensor(primals_270, 0.9)
    add_95: "f32[480]" = torch.ops.aten.add.Tensor(mul_154, mul_155);  mul_154 = mul_155 = None
    unsqueeze_72: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_73: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    mul_156: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_150, unsqueeze_73);  mul_150 = unsqueeze_73 = None
    unsqueeze_74: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_75: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    add_96: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_156, unsqueeze_75);  mul_156 = unsqueeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_18: "f32[8, 480, 14, 14]" = torch.ops.aten.clone.default(add_96)
    sigmoid_24: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_96)
    mul_157: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_96, sigmoid_24);  add_96 = sigmoid_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_31: "f32[8, 480, 14, 14]" = torch.ops.aten.convolution.default(mul_157, primals_143, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_97: "i64[]" = torch.ops.aten.add.Tensor(primals_271, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_19 = torch.ops.aten.var_mean.correction(convolution_31, [0, 2, 3], correction = 0, keepdim = True)
    getitem_38: "f32[1, 480, 1, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 480, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_98: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 0.001)
    rsqrt_19: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    sub_19: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, getitem_39)
    mul_158: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    squeeze_57: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
    squeeze_58: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
    mul_159: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_160: "f32[480]" = torch.ops.aten.mul.Tensor(primals_272, 0.9)
    add_99: "f32[480]" = torch.ops.aten.add.Tensor(mul_159, mul_160);  mul_159 = mul_160 = None
    squeeze_59: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
    mul_161: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0006381620931717);  squeeze_59 = None
    mul_162: "f32[480]" = torch.ops.aten.mul.Tensor(mul_161, 0.1);  mul_161 = None
    mul_163: "f32[480]" = torch.ops.aten.mul.Tensor(primals_273, 0.9)
    add_100: "f32[480]" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
    unsqueeze_76: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1)
    unsqueeze_77: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_164: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_158, unsqueeze_77);  mul_158 = unsqueeze_77 = None
    unsqueeze_78: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1);  primals_44 = None
    unsqueeze_79: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_101: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_164, unsqueeze_79);  mul_164 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_19: "f32[8, 480, 14, 14]" = torch.ops.aten.clone.default(add_101)
    sigmoid_25: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_101)
    mul_165: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_101, sigmoid_25);  add_101 = sigmoid_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_6: "f32[8, 480, 1, 1]" = torch.ops.aten.mean.dim(mul_165, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_32: "f32[8, 20, 1, 1]" = torch.ops.aten.convolution.default(mean_6, primals_144, primals_145, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_20: "f32[8, 20, 1, 1]" = torch.ops.aten.clone.default(convolution_32)
    sigmoid_26: "f32[8, 20, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_32)
    mul_166: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_32, sigmoid_26);  convolution_32 = sigmoid_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_33: "f32[8, 480, 1, 1]" = torch.ops.aten.convolution.default(mul_166, primals_146, primals_147, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_27: "f32[8, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_33);  convolution_33 = None
    alias_6: "f32[8, 480, 1, 1]" = torch.ops.aten.alias.default(sigmoid_27)
    mul_167: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_165, sigmoid_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_34: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(mul_167, primals_148, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_102: "i64[]" = torch.ops.aten.add.Tensor(primals_274, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_20 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
    getitem_40: "f32[1, 80, 1, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 80, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_103: "f32[1, 80, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 0.001)
    rsqrt_20: "f32[1, 80, 1, 1]" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
    sub_20: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_41)
    mul_168: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    squeeze_60: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
    squeeze_61: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
    mul_169: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_170: "f32[80]" = torch.ops.aten.mul.Tensor(primals_275, 0.9)
    add_104: "f32[80]" = torch.ops.aten.add.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
    squeeze_62: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
    mul_171: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0006381620931717);  squeeze_62 = None
    mul_172: "f32[80]" = torch.ops.aten.mul.Tensor(mul_171, 0.1);  mul_171 = None
    mul_173: "f32[80]" = torch.ops.aten.mul.Tensor(primals_276, 0.9)
    add_105: "f32[80]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
    unsqueeze_80: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1)
    unsqueeze_81: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    mul_174: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_168, unsqueeze_81);  mul_168 = unsqueeze_81 = None
    unsqueeze_82: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1);  primals_46 = None
    unsqueeze_83: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    add_106: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_174, unsqueeze_83);  mul_174 = unsqueeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_107: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_106, add_91);  add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_35: "f32[8, 480, 14, 14]" = torch.ops.aten.convolution.default(add_107, primals_149, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_108: "i64[]" = torch.ops.aten.add.Tensor(primals_277, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_21 = torch.ops.aten.var_mean.correction(convolution_35, [0, 2, 3], correction = 0, keepdim = True)
    getitem_42: "f32[1, 480, 1, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 480, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_109: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 0.001)
    rsqrt_21: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
    sub_21: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, getitem_43)
    mul_175: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    squeeze_63: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
    squeeze_64: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
    mul_176: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_177: "f32[480]" = torch.ops.aten.mul.Tensor(primals_278, 0.9)
    add_110: "f32[480]" = torch.ops.aten.add.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
    squeeze_65: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
    mul_178: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0006381620931717);  squeeze_65 = None
    mul_179: "f32[480]" = torch.ops.aten.mul.Tensor(mul_178, 0.1);  mul_178 = None
    mul_180: "f32[480]" = torch.ops.aten.mul.Tensor(primals_279, 0.9)
    add_111: "f32[480]" = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
    unsqueeze_84: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_85: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_181: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_85);  mul_175 = unsqueeze_85 = None
    unsqueeze_86: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_87: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_112: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_87);  mul_181 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_21: "f32[8, 480, 14, 14]" = torch.ops.aten.clone.default(add_112)
    sigmoid_28: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_112)
    mul_182: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_112, sigmoid_28);  add_112 = sigmoid_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_36: "f32[8, 480, 14, 14]" = torch.ops.aten.convolution.default(mul_182, primals_150, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_113: "i64[]" = torch.ops.aten.add.Tensor(primals_280, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_22 = torch.ops.aten.var_mean.correction(convolution_36, [0, 2, 3], correction = 0, keepdim = True)
    getitem_44: "f32[1, 480, 1, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 480, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_114: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 0.001)
    rsqrt_22: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
    sub_22: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, getitem_45)
    mul_183: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    squeeze_66: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
    squeeze_67: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
    mul_184: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_185: "f32[480]" = torch.ops.aten.mul.Tensor(primals_281, 0.9)
    add_115: "f32[480]" = torch.ops.aten.add.Tensor(mul_184, mul_185);  mul_184 = mul_185 = None
    squeeze_68: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
    mul_186: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0006381620931717);  squeeze_68 = None
    mul_187: "f32[480]" = torch.ops.aten.mul.Tensor(mul_186, 0.1);  mul_186 = None
    mul_188: "f32[480]" = torch.ops.aten.mul.Tensor(primals_282, 0.9)
    add_116: "f32[480]" = torch.ops.aten.add.Tensor(mul_187, mul_188);  mul_187 = mul_188 = None
    unsqueeze_88: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1)
    unsqueeze_89: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    mul_189: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_183, unsqueeze_89);  mul_183 = unsqueeze_89 = None
    unsqueeze_90: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1);  primals_50 = None
    unsqueeze_91: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    add_117: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_189, unsqueeze_91);  mul_189 = unsqueeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_22: "f32[8, 480, 14, 14]" = torch.ops.aten.clone.default(add_117)
    sigmoid_29: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_117)
    mul_190: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_117, sigmoid_29);  add_117 = sigmoid_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_7: "f32[8, 480, 1, 1]" = torch.ops.aten.mean.dim(mul_190, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_37: "f32[8, 20, 1, 1]" = torch.ops.aten.convolution.default(mean_7, primals_151, primals_152, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_23: "f32[8, 20, 1, 1]" = torch.ops.aten.clone.default(convolution_37)
    sigmoid_30: "f32[8, 20, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_37)
    mul_191: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_37, sigmoid_30);  convolution_37 = sigmoid_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_38: "f32[8, 480, 1, 1]" = torch.ops.aten.convolution.default(mul_191, primals_153, primals_154, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_31: "f32[8, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_38);  convolution_38 = None
    alias_7: "f32[8, 480, 1, 1]" = torch.ops.aten.alias.default(sigmoid_31)
    mul_192: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_190, sigmoid_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_39: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(mul_192, primals_155, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_118: "i64[]" = torch.ops.aten.add.Tensor(primals_283, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_23 = torch.ops.aten.var_mean.correction(convolution_39, [0, 2, 3], correction = 0, keepdim = True)
    getitem_46: "f32[1, 80, 1, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 80, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_119: "f32[1, 80, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 0.001)
    rsqrt_23: "f32[1, 80, 1, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
    sub_23: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, getitem_47)
    mul_193: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    squeeze_69: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
    squeeze_70: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
    mul_194: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
    mul_195: "f32[80]" = torch.ops.aten.mul.Tensor(primals_284, 0.9)
    add_120: "f32[80]" = torch.ops.aten.add.Tensor(mul_194, mul_195);  mul_194 = mul_195 = None
    squeeze_71: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
    mul_196: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0006381620931717);  squeeze_71 = None
    mul_197: "f32[80]" = torch.ops.aten.mul.Tensor(mul_196, 0.1);  mul_196 = None
    mul_198: "f32[80]" = torch.ops.aten.mul.Tensor(primals_285, 0.9)
    add_121: "f32[80]" = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
    unsqueeze_92: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1)
    unsqueeze_93: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_199: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_193, unsqueeze_93);  mul_193 = unsqueeze_93 = None
    unsqueeze_94: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1);  primals_52 = None
    unsqueeze_95: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_122: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_199, unsqueeze_95);  mul_199 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_123: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_122, add_107);  add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_40: "f32[8, 480, 14, 14]" = torch.ops.aten.convolution.default(add_123, primals_156, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_124: "i64[]" = torch.ops.aten.add.Tensor(primals_286, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_24 = torch.ops.aten.var_mean.correction(convolution_40, [0, 2, 3], correction = 0, keepdim = True)
    getitem_48: "f32[1, 480, 1, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 480, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_125: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_48, 0.001)
    rsqrt_24: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_125);  add_125 = None
    sub_24: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_49)
    mul_200: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    squeeze_72: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
    squeeze_73: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
    mul_201: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_202: "f32[480]" = torch.ops.aten.mul.Tensor(primals_287, 0.9)
    add_126: "f32[480]" = torch.ops.aten.add.Tensor(mul_201, mul_202);  mul_201 = mul_202 = None
    squeeze_74: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
    mul_203: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0006381620931717);  squeeze_74 = None
    mul_204: "f32[480]" = torch.ops.aten.mul.Tensor(mul_203, 0.1);  mul_203 = None
    mul_205: "f32[480]" = torch.ops.aten.mul.Tensor(primals_288, 0.9)
    add_127: "f32[480]" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
    unsqueeze_96: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_97: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    mul_206: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_200, unsqueeze_97);  mul_200 = unsqueeze_97 = None
    unsqueeze_98: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_99: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    add_128: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_206, unsqueeze_99);  mul_206 = unsqueeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_24: "f32[8, 480, 14, 14]" = torch.ops.aten.clone.default(add_128)
    sigmoid_32: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_128)
    mul_207: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_128, sigmoid_32);  add_128 = sigmoid_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_41: "f32[8, 480, 14, 14]" = torch.ops.aten.convolution.default(mul_207, primals_157, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 480)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_129: "i64[]" = torch.ops.aten.add.Tensor(primals_289, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_25 = torch.ops.aten.var_mean.correction(convolution_41, [0, 2, 3], correction = 0, keepdim = True)
    getitem_50: "f32[1, 480, 1, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 480, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_130: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 0.001)
    rsqrt_25: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
    sub_25: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, getitem_51)
    mul_208: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    squeeze_75: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
    squeeze_76: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
    mul_209: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_210: "f32[480]" = torch.ops.aten.mul.Tensor(primals_290, 0.9)
    add_131: "f32[480]" = torch.ops.aten.add.Tensor(mul_209, mul_210);  mul_209 = mul_210 = None
    squeeze_77: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
    mul_211: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0006381620931717);  squeeze_77 = None
    mul_212: "f32[480]" = torch.ops.aten.mul.Tensor(mul_211, 0.1);  mul_211 = None
    mul_213: "f32[480]" = torch.ops.aten.mul.Tensor(primals_291, 0.9)
    add_132: "f32[480]" = torch.ops.aten.add.Tensor(mul_212, mul_213);  mul_212 = mul_213 = None
    unsqueeze_100: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1)
    unsqueeze_101: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_214: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_208, unsqueeze_101);  mul_208 = unsqueeze_101 = None
    unsqueeze_102: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1);  primals_56 = None
    unsqueeze_103: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_133: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_214, unsqueeze_103);  mul_214 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_25: "f32[8, 480, 14, 14]" = torch.ops.aten.clone.default(add_133)
    sigmoid_33: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_133)
    mul_215: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_133, sigmoid_33);  add_133 = sigmoid_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_8: "f32[8, 480, 1, 1]" = torch.ops.aten.mean.dim(mul_215, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_42: "f32[8, 20, 1, 1]" = torch.ops.aten.convolution.default(mean_8, primals_158, primals_159, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_26: "f32[8, 20, 1, 1]" = torch.ops.aten.clone.default(convolution_42)
    sigmoid_34: "f32[8, 20, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_42)
    mul_216: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_42, sigmoid_34);  convolution_42 = sigmoid_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_43: "f32[8, 480, 1, 1]" = torch.ops.aten.convolution.default(mul_216, primals_160, primals_161, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_35: "f32[8, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_43);  convolution_43 = None
    alias_8: "f32[8, 480, 1, 1]" = torch.ops.aten.alias.default(sigmoid_35)
    mul_217: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_215, sigmoid_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_44: "f32[8, 112, 14, 14]" = torch.ops.aten.convolution.default(mul_217, primals_162, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_134: "i64[]" = torch.ops.aten.add.Tensor(primals_292, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_26 = torch.ops.aten.var_mean.correction(convolution_44, [0, 2, 3], correction = 0, keepdim = True)
    getitem_52: "f32[1, 112, 1, 1]" = var_mean_26[0]
    getitem_53: "f32[1, 112, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_135: "f32[1, 112, 1, 1]" = torch.ops.aten.add.Tensor(getitem_52, 0.001)
    rsqrt_26: "f32[1, 112, 1, 1]" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
    sub_26: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, getitem_53)
    mul_218: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    squeeze_78: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
    squeeze_79: "f32[112]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
    mul_219: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_220: "f32[112]" = torch.ops.aten.mul.Tensor(primals_293, 0.9)
    add_136: "f32[112]" = torch.ops.aten.add.Tensor(mul_219, mul_220);  mul_219 = mul_220 = None
    squeeze_80: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
    mul_221: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0006381620931717);  squeeze_80 = None
    mul_222: "f32[112]" = torch.ops.aten.mul.Tensor(mul_221, 0.1);  mul_221 = None
    mul_223: "f32[112]" = torch.ops.aten.mul.Tensor(primals_294, 0.9)
    add_137: "f32[112]" = torch.ops.aten.add.Tensor(mul_222, mul_223);  mul_222 = mul_223 = None
    unsqueeze_104: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1)
    unsqueeze_105: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    mul_224: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(mul_218, unsqueeze_105);  mul_218 = unsqueeze_105 = None
    unsqueeze_106: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_58, -1);  primals_58 = None
    unsqueeze_107: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    add_138: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(mul_224, unsqueeze_107);  mul_224 = unsqueeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_45: "f32[8, 672, 14, 14]" = torch.ops.aten.convolution.default(add_138, primals_163, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_139: "i64[]" = torch.ops.aten.add.Tensor(primals_295, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_27 = torch.ops.aten.var_mean.correction(convolution_45, [0, 2, 3], correction = 0, keepdim = True)
    getitem_54: "f32[1, 672, 1, 1]" = var_mean_27[0]
    getitem_55: "f32[1, 672, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_140: "f32[1, 672, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 0.001)
    rsqrt_27: "f32[1, 672, 1, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    sub_27: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, getitem_55)
    mul_225: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    squeeze_81: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
    squeeze_82: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
    mul_226: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
    mul_227: "f32[672]" = torch.ops.aten.mul.Tensor(primals_296, 0.9)
    add_141: "f32[672]" = torch.ops.aten.add.Tensor(mul_226, mul_227);  mul_226 = mul_227 = None
    squeeze_83: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
    mul_228: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0006381620931717);  squeeze_83 = None
    mul_229: "f32[672]" = torch.ops.aten.mul.Tensor(mul_228, 0.1);  mul_228 = None
    mul_230: "f32[672]" = torch.ops.aten.mul.Tensor(primals_297, 0.9)
    add_142: "f32[672]" = torch.ops.aten.add.Tensor(mul_229, mul_230);  mul_229 = mul_230 = None
    unsqueeze_108: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_109: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_231: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_225, unsqueeze_109);  mul_225 = unsqueeze_109 = None
    unsqueeze_110: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_111: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_143: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_231, unsqueeze_111);  mul_231 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_27: "f32[8, 672, 14, 14]" = torch.ops.aten.clone.default(add_143)
    sigmoid_36: "f32[8, 672, 14, 14]" = torch.ops.aten.sigmoid.default(add_143)
    mul_232: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_143, sigmoid_36);  add_143 = sigmoid_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_46: "f32[8, 672, 14, 14]" = torch.ops.aten.convolution.default(mul_232, primals_164, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 672)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_144: "i64[]" = torch.ops.aten.add.Tensor(primals_298, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_28 = torch.ops.aten.var_mean.correction(convolution_46, [0, 2, 3], correction = 0, keepdim = True)
    getitem_56: "f32[1, 672, 1, 1]" = var_mean_28[0]
    getitem_57: "f32[1, 672, 1, 1]" = var_mean_28[1];  var_mean_28 = None
    add_145: "f32[1, 672, 1, 1]" = torch.ops.aten.add.Tensor(getitem_56, 0.001)
    rsqrt_28: "f32[1, 672, 1, 1]" = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
    sub_28: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, getitem_57)
    mul_233: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    squeeze_84: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
    squeeze_85: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
    mul_234: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
    mul_235: "f32[672]" = torch.ops.aten.mul.Tensor(primals_299, 0.9)
    add_146: "f32[672]" = torch.ops.aten.add.Tensor(mul_234, mul_235);  mul_234 = mul_235 = None
    squeeze_86: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
    mul_236: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0006381620931717);  squeeze_86 = None
    mul_237: "f32[672]" = torch.ops.aten.mul.Tensor(mul_236, 0.1);  mul_236 = None
    mul_238: "f32[672]" = torch.ops.aten.mul.Tensor(primals_300, 0.9)
    add_147: "f32[672]" = torch.ops.aten.add.Tensor(mul_237, mul_238);  mul_237 = mul_238 = None
    unsqueeze_112: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_61, -1)
    unsqueeze_113: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    mul_239: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_233, unsqueeze_113);  mul_233 = unsqueeze_113 = None
    unsqueeze_114: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1);  primals_62 = None
    unsqueeze_115: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    add_148: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_239, unsqueeze_115);  mul_239 = unsqueeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_28: "f32[8, 672, 14, 14]" = torch.ops.aten.clone.default(add_148)
    sigmoid_37: "f32[8, 672, 14, 14]" = torch.ops.aten.sigmoid.default(add_148)
    mul_240: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_148, sigmoid_37);  add_148 = sigmoid_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_9: "f32[8, 672, 1, 1]" = torch.ops.aten.mean.dim(mul_240, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_47: "f32[8, 28, 1, 1]" = torch.ops.aten.convolution.default(mean_9, primals_165, primals_166, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_29: "f32[8, 28, 1, 1]" = torch.ops.aten.clone.default(convolution_47)
    sigmoid_38: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_47)
    mul_241: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_47, sigmoid_38);  convolution_47 = sigmoid_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_48: "f32[8, 672, 1, 1]" = torch.ops.aten.convolution.default(mul_241, primals_167, primals_168, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_39: "f32[8, 672, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_48);  convolution_48 = None
    alias_9: "f32[8, 672, 1, 1]" = torch.ops.aten.alias.default(sigmoid_39)
    mul_242: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_240, sigmoid_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_49: "f32[8, 112, 14, 14]" = torch.ops.aten.convolution.default(mul_242, primals_169, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_149: "i64[]" = torch.ops.aten.add.Tensor(primals_301, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_29 = torch.ops.aten.var_mean.correction(convolution_49, [0, 2, 3], correction = 0, keepdim = True)
    getitem_58: "f32[1, 112, 1, 1]" = var_mean_29[0]
    getitem_59: "f32[1, 112, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_150: "f32[1, 112, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 0.001)
    rsqrt_29: "f32[1, 112, 1, 1]" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
    sub_29: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, getitem_59)
    mul_243: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    squeeze_87: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
    squeeze_88: "f32[112]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
    mul_244: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
    mul_245: "f32[112]" = torch.ops.aten.mul.Tensor(primals_302, 0.9)
    add_151: "f32[112]" = torch.ops.aten.add.Tensor(mul_244, mul_245);  mul_244 = mul_245 = None
    squeeze_89: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
    mul_246: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0006381620931717);  squeeze_89 = None
    mul_247: "f32[112]" = torch.ops.aten.mul.Tensor(mul_246, 0.1);  mul_246 = None
    mul_248: "f32[112]" = torch.ops.aten.mul.Tensor(primals_303, 0.9)
    add_152: "f32[112]" = torch.ops.aten.add.Tensor(mul_247, mul_248);  mul_247 = mul_248 = None
    unsqueeze_116: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1)
    unsqueeze_117: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_249: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(mul_243, unsqueeze_117);  mul_243 = unsqueeze_117 = None
    unsqueeze_118: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_64, -1);  primals_64 = None
    unsqueeze_119: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_153: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(mul_249, unsqueeze_119);  mul_249 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_154: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(add_153, add_138);  add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_50: "f32[8, 672, 14, 14]" = torch.ops.aten.convolution.default(add_154, primals_170, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_155: "i64[]" = torch.ops.aten.add.Tensor(primals_304, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_30 = torch.ops.aten.var_mean.correction(convolution_50, [0, 2, 3], correction = 0, keepdim = True)
    getitem_60: "f32[1, 672, 1, 1]" = var_mean_30[0]
    getitem_61: "f32[1, 672, 1, 1]" = var_mean_30[1];  var_mean_30 = None
    add_156: "f32[1, 672, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 0.001)
    rsqrt_30: "f32[1, 672, 1, 1]" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
    sub_30: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, getitem_61)
    mul_250: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    squeeze_90: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
    squeeze_91: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
    mul_251: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
    mul_252: "f32[672]" = torch.ops.aten.mul.Tensor(primals_305, 0.9)
    add_157: "f32[672]" = torch.ops.aten.add.Tensor(mul_251, mul_252);  mul_251 = mul_252 = None
    squeeze_92: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
    mul_253: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0006381620931717);  squeeze_92 = None
    mul_254: "f32[672]" = torch.ops.aten.mul.Tensor(mul_253, 0.1);  mul_253 = None
    mul_255: "f32[672]" = torch.ops.aten.mul.Tensor(primals_306, 0.9)
    add_158: "f32[672]" = torch.ops.aten.add.Tensor(mul_254, mul_255);  mul_254 = mul_255 = None
    unsqueeze_120: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
    unsqueeze_121: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    mul_256: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_250, unsqueeze_121);  mul_250 = unsqueeze_121 = None
    unsqueeze_122: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
    unsqueeze_123: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    add_159: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_256, unsqueeze_123);  mul_256 = unsqueeze_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_30: "f32[8, 672, 14, 14]" = torch.ops.aten.clone.default(add_159)
    sigmoid_40: "f32[8, 672, 14, 14]" = torch.ops.aten.sigmoid.default(add_159)
    mul_257: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_159, sigmoid_40);  add_159 = sigmoid_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_51: "f32[8, 672, 14, 14]" = torch.ops.aten.convolution.default(mul_257, primals_171, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 672)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_160: "i64[]" = torch.ops.aten.add.Tensor(primals_307, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_31 = torch.ops.aten.var_mean.correction(convolution_51, [0, 2, 3], correction = 0, keepdim = True)
    getitem_62: "f32[1, 672, 1, 1]" = var_mean_31[0]
    getitem_63: "f32[1, 672, 1, 1]" = var_mean_31[1];  var_mean_31 = None
    add_161: "f32[1, 672, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 0.001)
    rsqrt_31: "f32[1, 672, 1, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    sub_31: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, getitem_63)
    mul_258: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    squeeze_93: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
    squeeze_94: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
    mul_259: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
    mul_260: "f32[672]" = torch.ops.aten.mul.Tensor(primals_308, 0.9)
    add_162: "f32[672]" = torch.ops.aten.add.Tensor(mul_259, mul_260);  mul_259 = mul_260 = None
    squeeze_95: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
    mul_261: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0006381620931717);  squeeze_95 = None
    mul_262: "f32[672]" = torch.ops.aten.mul.Tensor(mul_261, 0.1);  mul_261 = None
    mul_263: "f32[672]" = torch.ops.aten.mul.Tensor(primals_309, 0.9)
    add_163: "f32[672]" = torch.ops.aten.add.Tensor(mul_262, mul_263);  mul_262 = mul_263 = None
    unsqueeze_124: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_67, -1)
    unsqueeze_125: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_264: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_258, unsqueeze_125);  mul_258 = unsqueeze_125 = None
    unsqueeze_126: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1);  primals_68 = None
    unsqueeze_127: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_164: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_264, unsqueeze_127);  mul_264 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_31: "f32[8, 672, 14, 14]" = torch.ops.aten.clone.default(add_164)
    sigmoid_41: "f32[8, 672, 14, 14]" = torch.ops.aten.sigmoid.default(add_164)
    mul_265: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_164, sigmoid_41);  add_164 = sigmoid_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_10: "f32[8, 672, 1, 1]" = torch.ops.aten.mean.dim(mul_265, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_52: "f32[8, 28, 1, 1]" = torch.ops.aten.convolution.default(mean_10, primals_172, primals_173, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_32: "f32[8, 28, 1, 1]" = torch.ops.aten.clone.default(convolution_52)
    sigmoid_42: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_52)
    mul_266: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_52, sigmoid_42);  convolution_52 = sigmoid_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_53: "f32[8, 672, 1, 1]" = torch.ops.aten.convolution.default(mul_266, primals_174, primals_175, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_43: "f32[8, 672, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_53);  convolution_53 = None
    alias_10: "f32[8, 672, 1, 1]" = torch.ops.aten.alias.default(sigmoid_43)
    mul_267: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_265, sigmoid_43)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_54: "f32[8, 112, 14, 14]" = torch.ops.aten.convolution.default(mul_267, primals_176, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_165: "i64[]" = torch.ops.aten.add.Tensor(primals_310, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_32 = torch.ops.aten.var_mean.correction(convolution_54, [0, 2, 3], correction = 0, keepdim = True)
    getitem_64: "f32[1, 112, 1, 1]" = var_mean_32[0]
    getitem_65: "f32[1, 112, 1, 1]" = var_mean_32[1];  var_mean_32 = None
    add_166: "f32[1, 112, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 0.001)
    rsqrt_32: "f32[1, 112, 1, 1]" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
    sub_32: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, getitem_65)
    mul_268: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    squeeze_96: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
    squeeze_97: "f32[112]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
    mul_269: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
    mul_270: "f32[112]" = torch.ops.aten.mul.Tensor(primals_311, 0.9)
    add_167: "f32[112]" = torch.ops.aten.add.Tensor(mul_269, mul_270);  mul_269 = mul_270 = None
    squeeze_98: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
    mul_271: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0006381620931717);  squeeze_98 = None
    mul_272: "f32[112]" = torch.ops.aten.mul.Tensor(mul_271, 0.1);  mul_271 = None
    mul_273: "f32[112]" = torch.ops.aten.mul.Tensor(primals_312, 0.9)
    add_168: "f32[112]" = torch.ops.aten.add.Tensor(mul_272, mul_273);  mul_272 = mul_273 = None
    unsqueeze_128: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1)
    unsqueeze_129: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    mul_274: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(mul_268, unsqueeze_129);  mul_268 = unsqueeze_129 = None
    unsqueeze_130: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_70, -1);  primals_70 = None
    unsqueeze_131: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    add_169: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(mul_274, unsqueeze_131);  mul_274 = unsqueeze_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_170: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(add_169, add_154);  add_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_55: "f32[8, 672, 14, 14]" = torch.ops.aten.convolution.default(add_170, primals_177, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_171: "i64[]" = torch.ops.aten.add.Tensor(primals_313, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_33 = torch.ops.aten.var_mean.correction(convolution_55, [0, 2, 3], correction = 0, keepdim = True)
    getitem_66: "f32[1, 672, 1, 1]" = var_mean_33[0]
    getitem_67: "f32[1, 672, 1, 1]" = var_mean_33[1];  var_mean_33 = None
    add_172: "f32[1, 672, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 0.001)
    rsqrt_33: "f32[1, 672, 1, 1]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
    sub_33: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, getitem_67)
    mul_275: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    squeeze_99: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
    squeeze_100: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
    mul_276: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
    mul_277: "f32[672]" = torch.ops.aten.mul.Tensor(primals_314, 0.9)
    add_173: "f32[672]" = torch.ops.aten.add.Tensor(mul_276, mul_277);  mul_276 = mul_277 = None
    squeeze_101: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
    mul_278: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0006381620931717);  squeeze_101 = None
    mul_279: "f32[672]" = torch.ops.aten.mul.Tensor(mul_278, 0.1);  mul_278 = None
    mul_280: "f32[672]" = torch.ops.aten.mul.Tensor(primals_315, 0.9)
    add_174: "f32[672]" = torch.ops.aten.add.Tensor(mul_279, mul_280);  mul_279 = mul_280 = None
    unsqueeze_132: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_133: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_281: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_275, unsqueeze_133);  mul_275 = unsqueeze_133 = None
    unsqueeze_134: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_135: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_175: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_281, unsqueeze_135);  mul_281 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_33: "f32[8, 672, 14, 14]" = torch.ops.aten.clone.default(add_175)
    sigmoid_44: "f32[8, 672, 14, 14]" = torch.ops.aten.sigmoid.default(add_175)
    mul_282: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_175, sigmoid_44);  add_175 = sigmoid_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_4: "f32[8, 672, 17, 17]" = torch.ops.aten.constant_pad_nd.default(mul_282, [1, 2, 1, 2], 0.0);  mul_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_56: "f32[8, 672, 7, 7]" = torch.ops.aten.convolution.default(constant_pad_nd_4, primals_73, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 672)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_176: "i64[]" = torch.ops.aten.add.Tensor(primals_316, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_34 = torch.ops.aten.var_mean.correction(convolution_56, [0, 2, 3], correction = 0, keepdim = True)
    getitem_68: "f32[1, 672, 1, 1]" = var_mean_34[0]
    getitem_69: "f32[1, 672, 1, 1]" = var_mean_34[1];  var_mean_34 = None
    add_177: "f32[1, 672, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 0.001)
    rsqrt_34: "f32[1, 672, 1, 1]" = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
    sub_34: "f32[8, 672, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_56, getitem_69)
    mul_283: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    squeeze_102: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
    squeeze_103: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
    mul_284: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
    mul_285: "f32[672]" = torch.ops.aten.mul.Tensor(primals_317, 0.9)
    add_178: "f32[672]" = torch.ops.aten.add.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
    squeeze_104: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
    mul_286: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0025575447570332);  squeeze_104 = None
    mul_287: "f32[672]" = torch.ops.aten.mul.Tensor(mul_286, 0.1);  mul_286 = None
    mul_288: "f32[672]" = torch.ops.aten.mul.Tensor(primals_318, 0.9)
    add_179: "f32[672]" = torch.ops.aten.add.Tensor(mul_287, mul_288);  mul_287 = mul_288 = None
    unsqueeze_136: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1)
    unsqueeze_137: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    mul_289: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(mul_283, unsqueeze_137);  mul_283 = unsqueeze_137 = None
    unsqueeze_138: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1);  primals_75 = None
    unsqueeze_139: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    add_180: "f32[8, 672, 7, 7]" = torch.ops.aten.add.Tensor(mul_289, unsqueeze_139);  mul_289 = unsqueeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_34: "f32[8, 672, 7, 7]" = torch.ops.aten.clone.default(add_180)
    sigmoid_45: "f32[8, 672, 7, 7]" = torch.ops.aten.sigmoid.default(add_180)
    mul_290: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(add_180, sigmoid_45);  add_180 = sigmoid_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_11: "f32[8, 672, 1, 1]" = torch.ops.aten.mean.dim(mul_290, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_57: "f32[8, 28, 1, 1]" = torch.ops.aten.convolution.default(mean_11, primals_178, primals_179, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_35: "f32[8, 28, 1, 1]" = torch.ops.aten.clone.default(convolution_57)
    sigmoid_46: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_57)
    mul_291: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_57, sigmoid_46);  convolution_57 = sigmoid_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_58: "f32[8, 672, 1, 1]" = torch.ops.aten.convolution.default(mul_291, primals_180, primals_181, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_47: "f32[8, 672, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_58);  convolution_58 = None
    alias_11: "f32[8, 672, 1, 1]" = torch.ops.aten.alias.default(sigmoid_47)
    mul_292: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(mul_290, sigmoid_47)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_59: "f32[8, 192, 7, 7]" = torch.ops.aten.convolution.default(mul_292, primals_182, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_181: "i64[]" = torch.ops.aten.add.Tensor(primals_319, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_35 = torch.ops.aten.var_mean.correction(convolution_59, [0, 2, 3], correction = 0, keepdim = True)
    getitem_70: "f32[1, 192, 1, 1]" = var_mean_35[0]
    getitem_71: "f32[1, 192, 1, 1]" = var_mean_35[1];  var_mean_35 = None
    add_182: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_70, 0.001)
    rsqrt_35: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
    sub_35: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_59, getitem_71)
    mul_293: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    squeeze_105: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
    squeeze_106: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
    mul_294: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
    mul_295: "f32[192]" = torch.ops.aten.mul.Tensor(primals_320, 0.9)
    add_183: "f32[192]" = torch.ops.aten.add.Tensor(mul_294, mul_295);  mul_294 = mul_295 = None
    squeeze_107: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
    mul_296: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0025575447570332);  squeeze_107 = None
    mul_297: "f32[192]" = torch.ops.aten.mul.Tensor(mul_296, 0.1);  mul_296 = None
    mul_298: "f32[192]" = torch.ops.aten.mul.Tensor(primals_321, 0.9)
    add_184: "f32[192]" = torch.ops.aten.add.Tensor(mul_297, mul_298);  mul_297 = mul_298 = None
    unsqueeze_140: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_76, -1)
    unsqueeze_141: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_299: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(mul_293, unsqueeze_141);  mul_293 = unsqueeze_141 = None
    unsqueeze_142: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1);  primals_77 = None
    unsqueeze_143: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_185: "f32[8, 192, 7, 7]" = torch.ops.aten.add.Tensor(mul_299, unsqueeze_143);  mul_299 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_60: "f32[8, 1152, 7, 7]" = torch.ops.aten.convolution.default(add_185, primals_183, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_186: "i64[]" = torch.ops.aten.add.Tensor(primals_322, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_36 = torch.ops.aten.var_mean.correction(convolution_60, [0, 2, 3], correction = 0, keepdim = True)
    getitem_72: "f32[1, 1152, 1, 1]" = var_mean_36[0]
    getitem_73: "f32[1, 1152, 1, 1]" = var_mean_36[1];  var_mean_36 = None
    add_187: "f32[1, 1152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_72, 0.001)
    rsqrt_36: "f32[1, 1152, 1, 1]" = torch.ops.aten.rsqrt.default(add_187);  add_187 = None
    sub_36: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_60, getitem_73)
    mul_300: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    squeeze_108: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
    squeeze_109: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
    mul_301: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
    mul_302: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_323, 0.9)
    add_188: "f32[1152]" = torch.ops.aten.add.Tensor(mul_301, mul_302);  mul_301 = mul_302 = None
    squeeze_110: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
    mul_303: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0025575447570332);  squeeze_110 = None
    mul_304: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_303, 0.1);  mul_303 = None
    mul_305: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_324, 0.9)
    add_189: "f32[1152]" = torch.ops.aten.add.Tensor(mul_304, mul_305);  mul_304 = mul_305 = None
    unsqueeze_144: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1)
    unsqueeze_145: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    mul_306: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_300, unsqueeze_145);  mul_300 = unsqueeze_145 = None
    unsqueeze_146: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_79, -1);  primals_79 = None
    unsqueeze_147: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    add_190: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_306, unsqueeze_147);  mul_306 = unsqueeze_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_36: "f32[8, 1152, 7, 7]" = torch.ops.aten.clone.default(add_190)
    sigmoid_48: "f32[8, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_190)
    mul_307: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_190, sigmoid_48);  add_190 = sigmoid_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_61: "f32[8, 1152, 7, 7]" = torch.ops.aten.convolution.default(mul_307, primals_184, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_191: "i64[]" = torch.ops.aten.add.Tensor(primals_325, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_37 = torch.ops.aten.var_mean.correction(convolution_61, [0, 2, 3], correction = 0, keepdim = True)
    getitem_74: "f32[1, 1152, 1, 1]" = var_mean_37[0]
    getitem_75: "f32[1, 1152, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_192: "f32[1, 1152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_74, 0.001)
    rsqrt_37: "f32[1, 1152, 1, 1]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
    sub_37: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_61, getitem_75)
    mul_308: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    squeeze_111: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
    squeeze_112: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
    mul_309: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
    mul_310: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_326, 0.9)
    add_193: "f32[1152]" = torch.ops.aten.add.Tensor(mul_309, mul_310);  mul_309 = mul_310 = None
    squeeze_113: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
    mul_311: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0025575447570332);  squeeze_113 = None
    mul_312: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_311, 0.1);  mul_311 = None
    mul_313: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_327, 0.9)
    add_194: "f32[1152]" = torch.ops.aten.add.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None
    unsqueeze_148: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1)
    unsqueeze_149: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_314: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_308, unsqueeze_149);  mul_308 = unsqueeze_149 = None
    unsqueeze_150: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1);  primals_81 = None
    unsqueeze_151: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_195: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_314, unsqueeze_151);  mul_314 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_37: "f32[8, 1152, 7, 7]" = torch.ops.aten.clone.default(add_195)
    sigmoid_49: "f32[8, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_195)
    mul_315: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_195, sigmoid_49);  add_195 = sigmoid_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_12: "f32[8, 1152, 1, 1]" = torch.ops.aten.mean.dim(mul_315, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_62: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_12, primals_185, primals_186, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_38: "f32[8, 48, 1, 1]" = torch.ops.aten.clone.default(convolution_62)
    sigmoid_50: "f32[8, 48, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_62)
    mul_316: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_62, sigmoid_50);  convolution_62 = sigmoid_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_63: "f32[8, 1152, 1, 1]" = torch.ops.aten.convolution.default(mul_316, primals_187, primals_188, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_51: "f32[8, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_63);  convolution_63 = None
    alias_12: "f32[8, 1152, 1, 1]" = torch.ops.aten.alias.default(sigmoid_51)
    mul_317: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_315, sigmoid_51)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_64: "f32[8, 192, 7, 7]" = torch.ops.aten.convolution.default(mul_317, primals_189, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_196: "i64[]" = torch.ops.aten.add.Tensor(primals_328, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_38 = torch.ops.aten.var_mean.correction(convolution_64, [0, 2, 3], correction = 0, keepdim = True)
    getitem_76: "f32[1, 192, 1, 1]" = var_mean_38[0]
    getitem_77: "f32[1, 192, 1, 1]" = var_mean_38[1];  var_mean_38 = None
    add_197: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_76, 0.001)
    rsqrt_38: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
    sub_38: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_64, getitem_77)
    mul_318: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    squeeze_114: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
    squeeze_115: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
    mul_319: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
    mul_320: "f32[192]" = torch.ops.aten.mul.Tensor(primals_329, 0.9)
    add_198: "f32[192]" = torch.ops.aten.add.Tensor(mul_319, mul_320);  mul_319 = mul_320 = None
    squeeze_116: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
    mul_321: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0025575447570332);  squeeze_116 = None
    mul_322: "f32[192]" = torch.ops.aten.mul.Tensor(mul_321, 0.1);  mul_321 = None
    mul_323: "f32[192]" = torch.ops.aten.mul.Tensor(primals_330, 0.9)
    add_199: "f32[192]" = torch.ops.aten.add.Tensor(mul_322, mul_323);  mul_322 = mul_323 = None
    unsqueeze_152: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_82, -1)
    unsqueeze_153: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    mul_324: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(mul_318, unsqueeze_153);  mul_318 = unsqueeze_153 = None
    unsqueeze_154: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1);  primals_83 = None
    unsqueeze_155: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    add_200: "f32[8, 192, 7, 7]" = torch.ops.aten.add.Tensor(mul_324, unsqueeze_155);  mul_324 = unsqueeze_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_201: "f32[8, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_200, add_185);  add_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_65: "f32[8, 1152, 7, 7]" = torch.ops.aten.convolution.default(add_201, primals_190, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_202: "i64[]" = torch.ops.aten.add.Tensor(primals_331, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_39 = torch.ops.aten.var_mean.correction(convolution_65, [0, 2, 3], correction = 0, keepdim = True)
    getitem_78: "f32[1, 1152, 1, 1]" = var_mean_39[0]
    getitem_79: "f32[1, 1152, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_203: "f32[1, 1152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 0.001)
    rsqrt_39: "f32[1, 1152, 1, 1]" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
    sub_39: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_65, getitem_79)
    mul_325: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    squeeze_117: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
    squeeze_118: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
    mul_326: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
    mul_327: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_332, 0.9)
    add_204: "f32[1152]" = torch.ops.aten.add.Tensor(mul_326, mul_327);  mul_326 = mul_327 = None
    squeeze_119: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
    mul_328: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0025575447570332);  squeeze_119 = None
    mul_329: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_328, 0.1);  mul_328 = None
    mul_330: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_333, 0.9)
    add_205: "f32[1152]" = torch.ops.aten.add.Tensor(mul_329, mul_330);  mul_329 = mul_330 = None
    unsqueeze_156: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1)
    unsqueeze_157: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_331: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_325, unsqueeze_157);  mul_325 = unsqueeze_157 = None
    unsqueeze_158: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_85, -1);  primals_85 = None
    unsqueeze_159: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_206: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_331, unsqueeze_159);  mul_331 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_39: "f32[8, 1152, 7, 7]" = torch.ops.aten.clone.default(add_206)
    sigmoid_52: "f32[8, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_206)
    mul_332: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_206, sigmoid_52);  add_206 = sigmoid_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_66: "f32[8, 1152, 7, 7]" = torch.ops.aten.convolution.default(mul_332, primals_191, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_207: "i64[]" = torch.ops.aten.add.Tensor(primals_334, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_40 = torch.ops.aten.var_mean.correction(convolution_66, [0, 2, 3], correction = 0, keepdim = True)
    getitem_80: "f32[1, 1152, 1, 1]" = var_mean_40[0]
    getitem_81: "f32[1, 1152, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_208: "f32[1, 1152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_80, 0.001)
    rsqrt_40: "f32[1, 1152, 1, 1]" = torch.ops.aten.rsqrt.default(add_208);  add_208 = None
    sub_40: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_66, getitem_81)
    mul_333: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    squeeze_120: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
    squeeze_121: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
    mul_334: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
    mul_335: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_335, 0.9)
    add_209: "f32[1152]" = torch.ops.aten.add.Tensor(mul_334, mul_335);  mul_334 = mul_335 = None
    squeeze_122: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
    mul_336: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0025575447570332);  squeeze_122 = None
    mul_337: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_336, 0.1);  mul_336 = None
    mul_338: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_336, 0.9)
    add_210: "f32[1152]" = torch.ops.aten.add.Tensor(mul_337, mul_338);  mul_337 = mul_338 = None
    unsqueeze_160: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1)
    unsqueeze_161: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    mul_339: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_333, unsqueeze_161);  mul_333 = unsqueeze_161 = None
    unsqueeze_162: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1);  primals_87 = None
    unsqueeze_163: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    add_211: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_339, unsqueeze_163);  mul_339 = unsqueeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_40: "f32[8, 1152, 7, 7]" = torch.ops.aten.clone.default(add_211)
    sigmoid_53: "f32[8, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_211)
    mul_340: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_211, sigmoid_53);  add_211 = sigmoid_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_13: "f32[8, 1152, 1, 1]" = torch.ops.aten.mean.dim(mul_340, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_67: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_13, primals_192, primals_193, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_41: "f32[8, 48, 1, 1]" = torch.ops.aten.clone.default(convolution_67)
    sigmoid_54: "f32[8, 48, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_67)
    mul_341: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_67, sigmoid_54);  convolution_67 = sigmoid_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_68: "f32[8, 1152, 1, 1]" = torch.ops.aten.convolution.default(mul_341, primals_194, primals_195, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_55: "f32[8, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_68);  convolution_68 = None
    alias_13: "f32[8, 1152, 1, 1]" = torch.ops.aten.alias.default(sigmoid_55)
    mul_342: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_340, sigmoid_55)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_69: "f32[8, 192, 7, 7]" = torch.ops.aten.convolution.default(mul_342, primals_196, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_212: "i64[]" = torch.ops.aten.add.Tensor(primals_337, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_41 = torch.ops.aten.var_mean.correction(convolution_69, [0, 2, 3], correction = 0, keepdim = True)
    getitem_82: "f32[1, 192, 1, 1]" = var_mean_41[0]
    getitem_83: "f32[1, 192, 1, 1]" = var_mean_41[1];  var_mean_41 = None
    add_213: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_82, 0.001)
    rsqrt_41: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
    sub_41: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_69, getitem_83)
    mul_343: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
    squeeze_123: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
    squeeze_124: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
    mul_344: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
    mul_345: "f32[192]" = torch.ops.aten.mul.Tensor(primals_338, 0.9)
    add_214: "f32[192]" = torch.ops.aten.add.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
    squeeze_125: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
    mul_346: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0025575447570332);  squeeze_125 = None
    mul_347: "f32[192]" = torch.ops.aten.mul.Tensor(mul_346, 0.1);  mul_346 = None
    mul_348: "f32[192]" = torch.ops.aten.mul.Tensor(primals_339, 0.9)
    add_215: "f32[192]" = torch.ops.aten.add.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
    unsqueeze_164: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_88, -1)
    unsqueeze_165: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_349: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_165);  mul_343 = unsqueeze_165 = None
    unsqueeze_166: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1);  primals_89 = None
    unsqueeze_167: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_216: "f32[8, 192, 7, 7]" = torch.ops.aten.add.Tensor(mul_349, unsqueeze_167);  mul_349 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_217: "f32[8, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_216, add_201);  add_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_70: "f32[8, 1152, 7, 7]" = torch.ops.aten.convolution.default(add_217, primals_197, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_218: "i64[]" = torch.ops.aten.add.Tensor(primals_340, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_42 = torch.ops.aten.var_mean.correction(convolution_70, [0, 2, 3], correction = 0, keepdim = True)
    getitem_84: "f32[1, 1152, 1, 1]" = var_mean_42[0]
    getitem_85: "f32[1, 1152, 1, 1]" = var_mean_42[1];  var_mean_42 = None
    add_219: "f32[1, 1152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_84, 0.001)
    rsqrt_42: "f32[1, 1152, 1, 1]" = torch.ops.aten.rsqrt.default(add_219);  add_219 = None
    sub_42: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_70, getitem_85)
    mul_350: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
    squeeze_126: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
    squeeze_127: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
    mul_351: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
    mul_352: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_341, 0.9)
    add_220: "f32[1152]" = torch.ops.aten.add.Tensor(mul_351, mul_352);  mul_351 = mul_352 = None
    squeeze_128: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
    mul_353: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0025575447570332);  squeeze_128 = None
    mul_354: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_353, 0.1);  mul_353 = None
    mul_355: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_342, 0.9)
    add_221: "f32[1152]" = torch.ops.aten.add.Tensor(mul_354, mul_355);  mul_354 = mul_355 = None
    unsqueeze_168: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1)
    unsqueeze_169: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    mul_356: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_350, unsqueeze_169);  mul_350 = unsqueeze_169 = None
    unsqueeze_170: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_91, -1);  primals_91 = None
    unsqueeze_171: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    add_222: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_356, unsqueeze_171);  mul_356 = unsqueeze_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_42: "f32[8, 1152, 7, 7]" = torch.ops.aten.clone.default(add_222)
    sigmoid_56: "f32[8, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_222)
    mul_357: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_222, sigmoid_56);  add_222 = sigmoid_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_71: "f32[8, 1152, 7, 7]" = torch.ops.aten.convolution.default(mul_357, primals_198, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_223: "i64[]" = torch.ops.aten.add.Tensor(primals_343, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_43 = torch.ops.aten.var_mean.correction(convolution_71, [0, 2, 3], correction = 0, keepdim = True)
    getitem_86: "f32[1, 1152, 1, 1]" = var_mean_43[0]
    getitem_87: "f32[1, 1152, 1, 1]" = var_mean_43[1];  var_mean_43 = None
    add_224: "f32[1, 1152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_86, 0.001)
    rsqrt_43: "f32[1, 1152, 1, 1]" = torch.ops.aten.rsqrt.default(add_224);  add_224 = None
    sub_43: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_71, getitem_87)
    mul_358: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
    squeeze_129: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
    squeeze_130: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
    mul_359: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
    mul_360: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_344, 0.9)
    add_225: "f32[1152]" = torch.ops.aten.add.Tensor(mul_359, mul_360);  mul_359 = mul_360 = None
    squeeze_131: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
    mul_361: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0025575447570332);  squeeze_131 = None
    mul_362: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_361, 0.1);  mul_361 = None
    mul_363: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_345, 0.9)
    add_226: "f32[1152]" = torch.ops.aten.add.Tensor(mul_362, mul_363);  mul_362 = mul_363 = None
    unsqueeze_172: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1)
    unsqueeze_173: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_364: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_358, unsqueeze_173);  mul_358 = unsqueeze_173 = None
    unsqueeze_174: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_93, -1);  primals_93 = None
    unsqueeze_175: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_227: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_364, unsqueeze_175);  mul_364 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_43: "f32[8, 1152, 7, 7]" = torch.ops.aten.clone.default(add_227)
    sigmoid_57: "f32[8, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_227)
    mul_365: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_227, sigmoid_57);  add_227 = sigmoid_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_14: "f32[8, 1152, 1, 1]" = torch.ops.aten.mean.dim(mul_365, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_72: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_14, primals_199, primals_200, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_44: "f32[8, 48, 1, 1]" = torch.ops.aten.clone.default(convolution_72)
    sigmoid_58: "f32[8, 48, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_72)
    mul_366: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_72, sigmoid_58);  convolution_72 = sigmoid_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_73: "f32[8, 1152, 1, 1]" = torch.ops.aten.convolution.default(mul_366, primals_201, primals_202, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_59: "f32[8, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_73);  convolution_73 = None
    alias_14: "f32[8, 1152, 1, 1]" = torch.ops.aten.alias.default(sigmoid_59)
    mul_367: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_365, sigmoid_59)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_74: "f32[8, 192, 7, 7]" = torch.ops.aten.convolution.default(mul_367, primals_203, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_228: "i64[]" = torch.ops.aten.add.Tensor(primals_346, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_44 = torch.ops.aten.var_mean.correction(convolution_74, [0, 2, 3], correction = 0, keepdim = True)
    getitem_88: "f32[1, 192, 1, 1]" = var_mean_44[0]
    getitem_89: "f32[1, 192, 1, 1]" = var_mean_44[1];  var_mean_44 = None
    add_229: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_88, 0.001)
    rsqrt_44: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
    sub_44: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_74, getitem_89)
    mul_368: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
    squeeze_132: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
    squeeze_133: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
    mul_369: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1)
    mul_370: "f32[192]" = torch.ops.aten.mul.Tensor(primals_347, 0.9)
    add_230: "f32[192]" = torch.ops.aten.add.Tensor(mul_369, mul_370);  mul_369 = mul_370 = None
    squeeze_134: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
    mul_371: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_134, 1.0025575447570332);  squeeze_134 = None
    mul_372: "f32[192]" = torch.ops.aten.mul.Tensor(mul_371, 0.1);  mul_371 = None
    mul_373: "f32[192]" = torch.ops.aten.mul.Tensor(primals_348, 0.9)
    add_231: "f32[192]" = torch.ops.aten.add.Tensor(mul_372, mul_373);  mul_372 = mul_373 = None
    unsqueeze_176: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_94, -1)
    unsqueeze_177: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    mul_374: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(mul_368, unsqueeze_177);  mul_368 = unsqueeze_177 = None
    unsqueeze_178: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1);  primals_95 = None
    unsqueeze_179: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    add_232: "f32[8, 192, 7, 7]" = torch.ops.aten.add.Tensor(mul_374, unsqueeze_179);  mul_374 = unsqueeze_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_233: "f32[8, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_232, add_217);  add_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_75: "f32[8, 1152, 7, 7]" = torch.ops.aten.convolution.default(add_233, primals_204, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_234: "i64[]" = torch.ops.aten.add.Tensor(primals_349, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_45 = torch.ops.aten.var_mean.correction(convolution_75, [0, 2, 3], correction = 0, keepdim = True)
    getitem_90: "f32[1, 1152, 1, 1]" = var_mean_45[0]
    getitem_91: "f32[1, 1152, 1, 1]" = var_mean_45[1];  var_mean_45 = None
    add_235: "f32[1, 1152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_90, 0.001)
    rsqrt_45: "f32[1, 1152, 1, 1]" = torch.ops.aten.rsqrt.default(add_235);  add_235 = None
    sub_45: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_75, getitem_91)
    mul_375: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
    squeeze_135: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
    squeeze_136: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
    mul_376: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1)
    mul_377: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_350, 0.9)
    add_236: "f32[1152]" = torch.ops.aten.add.Tensor(mul_376, mul_377);  mul_376 = mul_377 = None
    squeeze_137: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
    mul_378: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_137, 1.0025575447570332);  squeeze_137 = None
    mul_379: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_378, 0.1);  mul_378 = None
    mul_380: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_351, 0.9)
    add_237: "f32[1152]" = torch.ops.aten.add.Tensor(mul_379, mul_380);  mul_379 = mul_380 = None
    unsqueeze_180: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1)
    unsqueeze_181: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_381: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_375, unsqueeze_181);  mul_375 = unsqueeze_181 = None
    unsqueeze_182: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_97, -1);  primals_97 = None
    unsqueeze_183: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_238: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_381, unsqueeze_183);  mul_381 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_45: "f32[8, 1152, 7, 7]" = torch.ops.aten.clone.default(add_238)
    sigmoid_60: "f32[8, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_238)
    mul_382: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_238, sigmoid_60);  add_238 = sigmoid_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_76: "f32[8, 1152, 7, 7]" = torch.ops.aten.convolution.default(mul_382, primals_205, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1152)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_239: "i64[]" = torch.ops.aten.add.Tensor(primals_352, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_46 = torch.ops.aten.var_mean.correction(convolution_76, [0, 2, 3], correction = 0, keepdim = True)
    getitem_92: "f32[1, 1152, 1, 1]" = var_mean_46[0]
    getitem_93: "f32[1, 1152, 1, 1]" = var_mean_46[1];  var_mean_46 = None
    add_240: "f32[1, 1152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_92, 0.001)
    rsqrt_46: "f32[1, 1152, 1, 1]" = torch.ops.aten.rsqrt.default(add_240);  add_240 = None
    sub_46: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_76, getitem_93)
    mul_383: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
    squeeze_138: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
    squeeze_139: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
    mul_384: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_138, 0.1)
    mul_385: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_353, 0.9)
    add_241: "f32[1152]" = torch.ops.aten.add.Tensor(mul_384, mul_385);  mul_384 = mul_385 = None
    squeeze_140: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_92, [0, 2, 3]);  getitem_92 = None
    mul_386: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_140, 1.0025575447570332);  squeeze_140 = None
    mul_387: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_386, 0.1);  mul_386 = None
    mul_388: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_354, 0.9)
    add_242: "f32[1152]" = torch.ops.aten.add.Tensor(mul_387, mul_388);  mul_387 = mul_388 = None
    unsqueeze_184: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1)
    unsqueeze_185: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    mul_389: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_383, unsqueeze_185);  mul_383 = unsqueeze_185 = None
    unsqueeze_186: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1);  primals_99 = None
    unsqueeze_187: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    add_243: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_389, unsqueeze_187);  mul_389 = unsqueeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_46: "f32[8, 1152, 7, 7]" = torch.ops.aten.clone.default(add_243)
    sigmoid_61: "f32[8, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_243)
    mul_390: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_243, sigmoid_61);  add_243 = sigmoid_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_15: "f32[8, 1152, 1, 1]" = torch.ops.aten.mean.dim(mul_390, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_77: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_15, primals_206, primals_207, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_47: "f32[8, 48, 1, 1]" = torch.ops.aten.clone.default(convolution_77)
    sigmoid_62: "f32[8, 48, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_77)
    mul_391: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_77, sigmoid_62);  convolution_77 = sigmoid_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_78: "f32[8, 1152, 1, 1]" = torch.ops.aten.convolution.default(mul_391, primals_208, primals_209, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_63: "f32[8, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_78);  convolution_78 = None
    alias_15: "f32[8, 1152, 1, 1]" = torch.ops.aten.alias.default(sigmoid_63)
    mul_392: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_390, sigmoid_63)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_79: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(mul_392, primals_210, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_244: "i64[]" = torch.ops.aten.add.Tensor(primals_355, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_47 = torch.ops.aten.var_mean.correction(convolution_79, [0, 2, 3], correction = 0, keepdim = True)
    getitem_94: "f32[1, 320, 1, 1]" = var_mean_47[0]
    getitem_95: "f32[1, 320, 1, 1]" = var_mean_47[1];  var_mean_47 = None
    add_245: "f32[1, 320, 1, 1]" = torch.ops.aten.add.Tensor(getitem_94, 0.001)
    rsqrt_47: "f32[1, 320, 1, 1]" = torch.ops.aten.rsqrt.default(add_245);  add_245 = None
    sub_47: "f32[8, 320, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_79, getitem_95)
    mul_393: "f32[8, 320, 7, 7]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
    squeeze_141: "f32[320]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
    squeeze_142: "f32[320]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
    mul_394: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_141, 0.1)
    mul_395: "f32[320]" = torch.ops.aten.mul.Tensor(primals_356, 0.9)
    add_246: "f32[320]" = torch.ops.aten.add.Tensor(mul_394, mul_395);  mul_394 = mul_395 = None
    squeeze_143: "f32[320]" = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
    mul_396: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_143, 1.0025575447570332);  squeeze_143 = None
    mul_397: "f32[320]" = torch.ops.aten.mul.Tensor(mul_396, 0.1);  mul_396 = None
    mul_398: "f32[320]" = torch.ops.aten.mul.Tensor(primals_357, 0.9)
    add_247: "f32[320]" = torch.ops.aten.add.Tensor(mul_397, mul_398);  mul_397 = mul_398 = None
    unsqueeze_188: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_100, -1)
    unsqueeze_189: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_399: "f32[8, 320, 7, 7]" = torch.ops.aten.mul.Tensor(mul_393, unsqueeze_189);  mul_393 = unsqueeze_189 = None
    unsqueeze_190: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_101, -1);  primals_101 = None
    unsqueeze_191: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_248: "f32[8, 320, 7, 7]" = torch.ops.aten.add.Tensor(mul_399, unsqueeze_191);  mul_399 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:168, code: x = self.conv_head(x)
    convolution_80: "f32[8, 1280, 7, 7]" = torch.ops.aten.convolution.default(add_248, primals_211, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_249: "i64[]" = torch.ops.aten.add.Tensor(primals_358, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_48 = torch.ops.aten.var_mean.correction(convolution_80, [0, 2, 3], correction = 0, keepdim = True)
    getitem_96: "f32[1, 1280, 1, 1]" = var_mean_48[0]
    getitem_97: "f32[1, 1280, 1, 1]" = var_mean_48[1];  var_mean_48 = None
    add_250: "f32[1, 1280, 1, 1]" = torch.ops.aten.add.Tensor(getitem_96, 0.001)
    rsqrt_48: "f32[1, 1280, 1, 1]" = torch.ops.aten.rsqrt.default(add_250);  add_250 = None
    sub_48: "f32[8, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_80, getitem_97)
    mul_400: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
    squeeze_144: "f32[1280]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
    squeeze_145: "f32[1280]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
    mul_401: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_144, 0.1)
    mul_402: "f32[1280]" = torch.ops.aten.mul.Tensor(primals_359, 0.9)
    add_251: "f32[1280]" = torch.ops.aten.add.Tensor(mul_401, mul_402);  mul_401 = mul_402 = None
    squeeze_146: "f32[1280]" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
    mul_403: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_146, 1.0025575447570332);  squeeze_146 = None
    mul_404: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_403, 0.1);  mul_403 = None
    mul_405: "f32[1280]" = torch.ops.aten.mul.Tensor(primals_360, 0.9)
    add_252: "f32[1280]" = torch.ops.aten.add.Tensor(mul_404, mul_405);  mul_404 = mul_405 = None
    unsqueeze_192: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1)
    unsqueeze_193: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    mul_406: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_400, unsqueeze_193);  mul_400 = unsqueeze_193 = None
    unsqueeze_194: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_103, -1);  primals_103 = None
    unsqueeze_195: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    add_253: "f32[8, 1280, 7, 7]" = torch.ops.aten.add.Tensor(mul_406, unsqueeze_195);  mul_406 = unsqueeze_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_48: "f32[8, 1280, 7, 7]" = torch.ops.aten.clone.default(add_253)
    sigmoid_64: "f32[8, 1280, 7, 7]" = torch.ops.aten.sigmoid.default(add_253)
    mul_407: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(add_253, sigmoid_64);  add_253 = sigmoid_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean_16: "f32[8, 1280, 1, 1]" = torch.ops.aten.mean.dim(mul_407, [-1, -2], True);  mul_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view: "f32[8, 1280]" = torch.ops.aten.view.default(mean_16, [8, 1280]);  mean_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:176, code: return x if pre_logits else self.classifier(x)
    permute: "f32[1280, 1000]" = torch.ops.aten.permute.default(primals_212, [1, 0]);  primals_212 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_213, view, permute);  primals_213 = None
    permute_1: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm: "f32[8, 1280]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 1280]" = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
    permute_3: "f32[1280, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_2: "f32[8, 1280, 1, 1]" = torch.ops.aten.view.default(mm, [8, 1280, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 1280, 7, 7]" = torch.ops.aten.expand.default(view_2, [8, 1280, 7, 7]);  view_2 = None
    div: "f32[8, 1280, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_65: "f32[8, 1280, 7, 7]" = torch.ops.aten.sigmoid.default(clone_48)
    full: "f32[8, 1280, 7, 7]" = torch.ops.aten.full.default([8, 1280, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_49: "f32[8, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(full, sigmoid_65);  full = None
    mul_408: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(clone_48, sub_49);  clone_48 = sub_49 = None
    add_254: "f32[8, 1280, 7, 7]" = torch.ops.aten.add.Scalar(mul_408, 1);  mul_408 = None
    mul_409: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_65, add_254);  sigmoid_65 = add_254 = None
    mul_410: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(div, mul_409);  div = mul_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_196: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
    unsqueeze_197: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, 2);  unsqueeze_196 = None
    unsqueeze_198: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 3);  unsqueeze_197 = None
    sum_2: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_410, [0, 2, 3])
    sub_50: "f32[8, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_198)
    mul_411: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_410, sub_50);  sub_50 = None
    sum_3: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_411, [0, 2, 3]);  mul_411 = None
    mul_412: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_2, 0.002551020408163265)
    unsqueeze_199: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_412, 0);  mul_412 = None
    unsqueeze_200: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_199, 2);  unsqueeze_199 = None
    unsqueeze_201: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, 3);  unsqueeze_200 = None
    mul_413: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_3, 0.002551020408163265)
    mul_414: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
    mul_415: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_413, mul_414);  mul_413 = mul_414 = None
    unsqueeze_202: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_415, 0);  mul_415 = None
    unsqueeze_203: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, 2);  unsqueeze_202 = None
    unsqueeze_204: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 3);  unsqueeze_203 = None
    mul_416: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_102);  primals_102 = None
    unsqueeze_205: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_416, 0);  mul_416 = None
    unsqueeze_206: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 2);  unsqueeze_205 = None
    unsqueeze_207: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 3);  unsqueeze_206 = None
    sub_51: "f32[8, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_198);  convolution_80 = unsqueeze_198 = None
    mul_417: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_204);  sub_51 = unsqueeze_204 = None
    sub_52: "f32[8, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(mul_410, mul_417);  mul_410 = mul_417 = None
    sub_53: "f32[8, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(sub_52, unsqueeze_201);  sub_52 = unsqueeze_201 = None
    mul_418: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_207);  sub_53 = unsqueeze_207 = None
    mul_419: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_145);  sum_3 = squeeze_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:168, code: x = self.conv_head(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_418, add_248, primals_211, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_418 = add_248 = primals_211 = None
    getitem_98: "f32[8, 320, 7, 7]" = convolution_backward[0]
    getitem_99: "f32[1280, 320, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_208: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
    unsqueeze_209: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, 2);  unsqueeze_208 = None
    unsqueeze_210: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 3);  unsqueeze_209 = None
    sum_4: "f32[320]" = torch.ops.aten.sum.dim_IntList(getitem_98, [0, 2, 3])
    sub_54: "f32[8, 320, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_210)
    mul_420: "f32[8, 320, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_98, sub_54);  sub_54 = None
    sum_5: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_420, [0, 2, 3]);  mul_420 = None
    mul_421: "f32[320]" = torch.ops.aten.mul.Tensor(sum_4, 0.002551020408163265)
    unsqueeze_211: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_421, 0);  mul_421 = None
    unsqueeze_212: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 2);  unsqueeze_211 = None
    unsqueeze_213: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 3);  unsqueeze_212 = None
    mul_422: "f32[320]" = torch.ops.aten.mul.Tensor(sum_5, 0.002551020408163265)
    mul_423: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
    mul_424: "f32[320]" = torch.ops.aten.mul.Tensor(mul_422, mul_423);  mul_422 = mul_423 = None
    unsqueeze_214: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_424, 0);  mul_424 = None
    unsqueeze_215: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, 2);  unsqueeze_214 = None
    unsqueeze_216: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 3);  unsqueeze_215 = None
    mul_425: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_100);  primals_100 = None
    unsqueeze_217: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_425, 0);  mul_425 = None
    unsqueeze_218: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 2);  unsqueeze_217 = None
    unsqueeze_219: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 3);  unsqueeze_218 = None
    sub_55: "f32[8, 320, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_210);  convolution_79 = unsqueeze_210 = None
    mul_426: "f32[8, 320, 7, 7]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_216);  sub_55 = unsqueeze_216 = None
    sub_56: "f32[8, 320, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_98, mul_426);  getitem_98 = mul_426 = None
    sub_57: "f32[8, 320, 7, 7]" = torch.ops.aten.sub.Tensor(sub_56, unsqueeze_213);  sub_56 = unsqueeze_213 = None
    mul_427: "f32[8, 320, 7, 7]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_219);  sub_57 = unsqueeze_219 = None
    mul_428: "f32[320]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_142);  sum_5 = squeeze_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_427, mul_392, primals_210, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_427 = mul_392 = primals_210 = None
    getitem_101: "f32[8, 1152, 7, 7]" = convolution_backward_1[0]
    getitem_102: "f32[320, 1152, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_429: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_101, mul_390);  mul_390 = None
    mul_430: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_101, sigmoid_63);  getitem_101 = sigmoid_63 = None
    sum_6: "f32[8, 1152, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_429, [2, 3], True);  mul_429 = None
    alias_16: "f32[8, 1152, 1, 1]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    sub_58: "f32[8, 1152, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_16)
    mul_431: "f32[8, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(alias_16, sub_58);  alias_16 = sub_58 = None
    mul_432: "f32[8, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(sum_6, mul_431);  sum_6 = mul_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_432, mul_391, primals_208, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_432 = mul_391 = primals_208 = None
    getitem_104: "f32[8, 48, 1, 1]" = convolution_backward_2[0]
    getitem_105: "f32[1152, 48, 1, 1]" = convolution_backward_2[1]
    getitem_106: "f32[1152]" = convolution_backward_2[2];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_66: "f32[8, 48, 1, 1]" = torch.ops.aten.sigmoid.default(clone_47)
    full_1: "f32[8, 48, 1, 1]" = torch.ops.aten.full.default([8, 48, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_59: "f32[8, 48, 1, 1]" = torch.ops.aten.sub.Tensor(full_1, sigmoid_66);  full_1 = None
    mul_433: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(clone_47, sub_59);  clone_47 = sub_59 = None
    add_255: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Scalar(mul_433, 1);  mul_433 = None
    mul_434: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_66, add_255);  sigmoid_66 = add_255 = None
    mul_435: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_104, mul_434);  getitem_104 = mul_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_435, mean_15, primals_206, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_435 = mean_15 = primals_206 = None
    getitem_107: "f32[8, 1152, 1, 1]" = convolution_backward_3[0]
    getitem_108: "f32[48, 1152, 1, 1]" = convolution_backward_3[1]
    getitem_109: "f32[48]" = convolution_backward_3[2];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_1: "f32[8, 1152, 7, 7]" = torch.ops.aten.expand.default(getitem_107, [8, 1152, 7, 7]);  getitem_107 = None
    div_1: "f32[8, 1152, 7, 7]" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_256: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_430, div_1);  mul_430 = div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_67: "f32[8, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(clone_46)
    full_2: "f32[8, 1152, 7, 7]" = torch.ops.aten.full.default([8, 1152, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_60: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(full_2, sigmoid_67);  full_2 = None
    mul_436: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(clone_46, sub_60);  clone_46 = sub_60 = None
    add_257: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Scalar(mul_436, 1);  mul_436 = None
    mul_437: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_67, add_257);  sigmoid_67 = add_257 = None
    mul_438: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_256, mul_437);  add_256 = mul_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_220: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
    unsqueeze_221: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, 2);  unsqueeze_220 = None
    unsqueeze_222: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 3);  unsqueeze_221 = None
    sum_7: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_438, [0, 2, 3])
    sub_61: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_222)
    mul_439: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_438, sub_61);  sub_61 = None
    sum_8: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_439, [0, 2, 3]);  mul_439 = None
    mul_440: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_7, 0.002551020408163265)
    unsqueeze_223: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_440, 0);  mul_440 = None
    unsqueeze_224: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 2);  unsqueeze_223 = None
    unsqueeze_225: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 3);  unsqueeze_224 = None
    mul_441: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_8, 0.002551020408163265)
    mul_442: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
    mul_443: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_441, mul_442);  mul_441 = mul_442 = None
    unsqueeze_226: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_443, 0);  mul_443 = None
    unsqueeze_227: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 2);  unsqueeze_226 = None
    unsqueeze_228: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 3);  unsqueeze_227 = None
    mul_444: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_98);  primals_98 = None
    unsqueeze_229: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_444, 0);  mul_444 = None
    unsqueeze_230: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 2);  unsqueeze_229 = None
    unsqueeze_231: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 3);  unsqueeze_230 = None
    sub_62: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_222);  convolution_76 = unsqueeze_222 = None
    mul_445: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_228);  sub_62 = unsqueeze_228 = None
    sub_63: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(mul_438, mul_445);  mul_438 = mul_445 = None
    sub_64: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_63, unsqueeze_225);  sub_63 = unsqueeze_225 = None
    mul_446: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_231);  sub_64 = unsqueeze_231 = None
    mul_447: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_8, squeeze_139);  sum_8 = squeeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_446, mul_382, primals_205, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_446 = mul_382 = primals_205 = None
    getitem_110: "f32[8, 1152, 7, 7]" = convolution_backward_4[0]
    getitem_111: "f32[1152, 1, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_68: "f32[8, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(clone_45)
    full_3: "f32[8, 1152, 7, 7]" = torch.ops.aten.full.default([8, 1152, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_65: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(full_3, sigmoid_68);  full_3 = None
    mul_448: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(clone_45, sub_65);  clone_45 = sub_65 = None
    add_258: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Scalar(mul_448, 1);  mul_448 = None
    mul_449: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_68, add_258);  sigmoid_68 = add_258 = None
    mul_450: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_110, mul_449);  getitem_110 = mul_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_232: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
    unsqueeze_233: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 2);  unsqueeze_232 = None
    unsqueeze_234: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 3);  unsqueeze_233 = None
    sum_9: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_450, [0, 2, 3])
    sub_66: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_234)
    mul_451: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_450, sub_66);  sub_66 = None
    sum_10: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_451, [0, 2, 3]);  mul_451 = None
    mul_452: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_9, 0.002551020408163265)
    unsqueeze_235: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_452, 0);  mul_452 = None
    unsqueeze_236: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 2);  unsqueeze_235 = None
    unsqueeze_237: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 3);  unsqueeze_236 = None
    mul_453: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_10, 0.002551020408163265)
    mul_454: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
    mul_455: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_453, mul_454);  mul_453 = mul_454 = None
    unsqueeze_238: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_455, 0);  mul_455 = None
    unsqueeze_239: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 2);  unsqueeze_238 = None
    unsqueeze_240: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 3);  unsqueeze_239 = None
    mul_456: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_96);  primals_96 = None
    unsqueeze_241: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_456, 0);  mul_456 = None
    unsqueeze_242: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 2);  unsqueeze_241 = None
    unsqueeze_243: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 3);  unsqueeze_242 = None
    sub_67: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_234);  convolution_75 = unsqueeze_234 = None
    mul_457: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_240);  sub_67 = unsqueeze_240 = None
    sub_68: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(mul_450, mul_457);  mul_450 = mul_457 = None
    sub_69: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_68, unsqueeze_237);  sub_68 = unsqueeze_237 = None
    mul_458: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_243);  sub_69 = unsqueeze_243 = None
    mul_459: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_10, squeeze_136);  sum_10 = squeeze_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_458, add_233, primals_204, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_458 = add_233 = primals_204 = None
    getitem_113: "f32[8, 192, 7, 7]" = convolution_backward_5[0]
    getitem_114: "f32[1152, 192, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_244: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
    unsqueeze_245: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 2);  unsqueeze_244 = None
    unsqueeze_246: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 3);  unsqueeze_245 = None
    sum_11: "f32[192]" = torch.ops.aten.sum.dim_IntList(getitem_113, [0, 2, 3])
    sub_70: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_246)
    mul_460: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_113, sub_70);  sub_70 = None
    sum_12: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_460, [0, 2, 3]);  mul_460 = None
    mul_461: "f32[192]" = torch.ops.aten.mul.Tensor(sum_11, 0.002551020408163265)
    unsqueeze_247: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_461, 0);  mul_461 = None
    unsqueeze_248: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 2);  unsqueeze_247 = None
    unsqueeze_249: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 3);  unsqueeze_248 = None
    mul_462: "f32[192]" = torch.ops.aten.mul.Tensor(sum_12, 0.002551020408163265)
    mul_463: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
    mul_464: "f32[192]" = torch.ops.aten.mul.Tensor(mul_462, mul_463);  mul_462 = mul_463 = None
    unsqueeze_250: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_464, 0);  mul_464 = None
    unsqueeze_251: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 2);  unsqueeze_250 = None
    unsqueeze_252: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 3);  unsqueeze_251 = None
    mul_465: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_94);  primals_94 = None
    unsqueeze_253: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_465, 0);  mul_465 = None
    unsqueeze_254: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None
    unsqueeze_255: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 3);  unsqueeze_254 = None
    sub_71: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_246);  convolution_74 = unsqueeze_246 = None
    mul_466: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_252);  sub_71 = unsqueeze_252 = None
    sub_72: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_113, mul_466);  mul_466 = None
    sub_73: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(sub_72, unsqueeze_249);  sub_72 = unsqueeze_249 = None
    mul_467: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_255);  sub_73 = unsqueeze_255 = None
    mul_468: "f32[192]" = torch.ops.aten.mul.Tensor(sum_12, squeeze_133);  sum_12 = squeeze_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_467, mul_367, primals_203, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_467 = mul_367 = primals_203 = None
    getitem_116: "f32[8, 1152, 7, 7]" = convolution_backward_6[0]
    getitem_117: "f32[192, 1152, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_469: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_116, mul_365);  mul_365 = None
    mul_470: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_116, sigmoid_59);  getitem_116 = sigmoid_59 = None
    sum_13: "f32[8, 1152, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_469, [2, 3], True);  mul_469 = None
    alias_17: "f32[8, 1152, 1, 1]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    sub_74: "f32[8, 1152, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_17)
    mul_471: "f32[8, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(alias_17, sub_74);  alias_17 = sub_74 = None
    mul_472: "f32[8, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(sum_13, mul_471);  sum_13 = mul_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_472, mul_366, primals_201, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_472 = mul_366 = primals_201 = None
    getitem_119: "f32[8, 48, 1, 1]" = convolution_backward_7[0]
    getitem_120: "f32[1152, 48, 1, 1]" = convolution_backward_7[1]
    getitem_121: "f32[1152]" = convolution_backward_7[2];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_69: "f32[8, 48, 1, 1]" = torch.ops.aten.sigmoid.default(clone_44)
    full_4: "f32[8, 48, 1, 1]" = torch.ops.aten.full.default([8, 48, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_75: "f32[8, 48, 1, 1]" = torch.ops.aten.sub.Tensor(full_4, sigmoid_69);  full_4 = None
    mul_473: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(clone_44, sub_75);  clone_44 = sub_75 = None
    add_259: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Scalar(mul_473, 1);  mul_473 = None
    mul_474: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_69, add_259);  sigmoid_69 = add_259 = None
    mul_475: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_119, mul_474);  getitem_119 = mul_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_475, mean_14, primals_199, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_475 = mean_14 = primals_199 = None
    getitem_122: "f32[8, 1152, 1, 1]" = convolution_backward_8[0]
    getitem_123: "f32[48, 1152, 1, 1]" = convolution_backward_8[1]
    getitem_124: "f32[48]" = convolution_backward_8[2];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_2: "f32[8, 1152, 7, 7]" = torch.ops.aten.expand.default(getitem_122, [8, 1152, 7, 7]);  getitem_122 = None
    div_2: "f32[8, 1152, 7, 7]" = torch.ops.aten.div.Scalar(expand_2, 49);  expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_260: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_470, div_2);  mul_470 = div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_70: "f32[8, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(clone_43)
    full_5: "f32[8, 1152, 7, 7]" = torch.ops.aten.full.default([8, 1152, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_76: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(full_5, sigmoid_70);  full_5 = None
    mul_476: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(clone_43, sub_76);  clone_43 = sub_76 = None
    add_261: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Scalar(mul_476, 1);  mul_476 = None
    mul_477: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_70, add_261);  sigmoid_70 = add_261 = None
    mul_478: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_260, mul_477);  add_260 = mul_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_256: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
    unsqueeze_257: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 2);  unsqueeze_256 = None
    unsqueeze_258: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 3);  unsqueeze_257 = None
    sum_14: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_478, [0, 2, 3])
    sub_77: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_258)
    mul_479: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_478, sub_77);  sub_77 = None
    sum_15: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_479, [0, 2, 3]);  mul_479 = None
    mul_480: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_14, 0.002551020408163265)
    unsqueeze_259: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_480, 0);  mul_480 = None
    unsqueeze_260: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 2);  unsqueeze_259 = None
    unsqueeze_261: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 3);  unsqueeze_260 = None
    mul_481: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_15, 0.002551020408163265)
    mul_482: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
    mul_483: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_481, mul_482);  mul_481 = mul_482 = None
    unsqueeze_262: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_483, 0);  mul_483 = None
    unsqueeze_263: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 2);  unsqueeze_262 = None
    unsqueeze_264: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 3);  unsqueeze_263 = None
    mul_484: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_92);  primals_92 = None
    unsqueeze_265: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_484, 0);  mul_484 = None
    unsqueeze_266: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
    unsqueeze_267: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 3);  unsqueeze_266 = None
    sub_78: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_258);  convolution_71 = unsqueeze_258 = None
    mul_485: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_264);  sub_78 = unsqueeze_264 = None
    sub_79: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(mul_478, mul_485);  mul_478 = mul_485 = None
    sub_80: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_79, unsqueeze_261);  sub_79 = unsqueeze_261 = None
    mul_486: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_267);  sub_80 = unsqueeze_267 = None
    mul_487: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_130);  sum_15 = squeeze_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_486, mul_357, primals_198, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_486 = mul_357 = primals_198 = None
    getitem_125: "f32[8, 1152, 7, 7]" = convolution_backward_9[0]
    getitem_126: "f32[1152, 1, 5, 5]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_71: "f32[8, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(clone_42)
    full_6: "f32[8, 1152, 7, 7]" = torch.ops.aten.full.default([8, 1152, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_81: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(full_6, sigmoid_71);  full_6 = None
    mul_488: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(clone_42, sub_81);  clone_42 = sub_81 = None
    add_262: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Scalar(mul_488, 1);  mul_488 = None
    mul_489: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_71, add_262);  sigmoid_71 = add_262 = None
    mul_490: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_125, mul_489);  getitem_125 = mul_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_268: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
    unsqueeze_269: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 2);  unsqueeze_268 = None
    unsqueeze_270: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 3);  unsqueeze_269 = None
    sum_16: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_490, [0, 2, 3])
    sub_82: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_270)
    mul_491: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_490, sub_82);  sub_82 = None
    sum_17: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_491, [0, 2, 3]);  mul_491 = None
    mul_492: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_16, 0.002551020408163265)
    unsqueeze_271: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_492, 0);  mul_492 = None
    unsqueeze_272: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 2);  unsqueeze_271 = None
    unsqueeze_273: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 3);  unsqueeze_272 = None
    mul_493: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_17, 0.002551020408163265)
    mul_494: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
    mul_495: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_493, mul_494);  mul_493 = mul_494 = None
    unsqueeze_274: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_495, 0);  mul_495 = None
    unsqueeze_275: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 2);  unsqueeze_274 = None
    unsqueeze_276: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 3);  unsqueeze_275 = None
    mul_496: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_90);  primals_90 = None
    unsqueeze_277: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_496, 0);  mul_496 = None
    unsqueeze_278: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None
    unsqueeze_279: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 3);  unsqueeze_278 = None
    sub_83: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_270);  convolution_70 = unsqueeze_270 = None
    mul_497: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_276);  sub_83 = unsqueeze_276 = None
    sub_84: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(mul_490, mul_497);  mul_490 = mul_497 = None
    sub_85: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_84, unsqueeze_273);  sub_84 = unsqueeze_273 = None
    mul_498: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_279);  sub_85 = unsqueeze_279 = None
    mul_499: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_127);  sum_17 = squeeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_498, add_217, primals_197, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_498 = add_217 = primals_197 = None
    getitem_128: "f32[8, 192, 7, 7]" = convolution_backward_10[0]
    getitem_129: "f32[1152, 192, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_263: "f32[8, 192, 7, 7]" = torch.ops.aten.add.Tensor(getitem_113, getitem_128);  getitem_113 = getitem_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_280: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
    unsqueeze_281: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 2);  unsqueeze_280 = None
    unsqueeze_282: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 3);  unsqueeze_281 = None
    sum_18: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_263, [0, 2, 3])
    sub_86: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_282)
    mul_500: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_263, sub_86);  sub_86 = None
    sum_19: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_500, [0, 2, 3]);  mul_500 = None
    mul_501: "f32[192]" = torch.ops.aten.mul.Tensor(sum_18, 0.002551020408163265)
    unsqueeze_283: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_501, 0);  mul_501 = None
    unsqueeze_284: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 2);  unsqueeze_283 = None
    unsqueeze_285: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 3);  unsqueeze_284 = None
    mul_502: "f32[192]" = torch.ops.aten.mul.Tensor(sum_19, 0.002551020408163265)
    mul_503: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
    mul_504: "f32[192]" = torch.ops.aten.mul.Tensor(mul_502, mul_503);  mul_502 = mul_503 = None
    unsqueeze_286: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_504, 0);  mul_504 = None
    unsqueeze_287: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 2);  unsqueeze_286 = None
    unsqueeze_288: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 3);  unsqueeze_287 = None
    mul_505: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_88);  primals_88 = None
    unsqueeze_289: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_505, 0);  mul_505 = None
    unsqueeze_290: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 2);  unsqueeze_289 = None
    unsqueeze_291: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 3);  unsqueeze_290 = None
    sub_87: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_282);  convolution_69 = unsqueeze_282 = None
    mul_506: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_288);  sub_87 = unsqueeze_288 = None
    sub_88: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(add_263, mul_506);  mul_506 = None
    sub_89: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(sub_88, unsqueeze_285);  sub_88 = unsqueeze_285 = None
    mul_507: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_291);  sub_89 = unsqueeze_291 = None
    mul_508: "f32[192]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_124);  sum_19 = squeeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_507, mul_342, primals_196, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_507 = mul_342 = primals_196 = None
    getitem_131: "f32[8, 1152, 7, 7]" = convolution_backward_11[0]
    getitem_132: "f32[192, 1152, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_509: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_131, mul_340);  mul_340 = None
    mul_510: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_131, sigmoid_55);  getitem_131 = sigmoid_55 = None
    sum_20: "f32[8, 1152, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_509, [2, 3], True);  mul_509 = None
    alias_18: "f32[8, 1152, 1, 1]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    sub_90: "f32[8, 1152, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_18)
    mul_511: "f32[8, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(alias_18, sub_90);  alias_18 = sub_90 = None
    mul_512: "f32[8, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(sum_20, mul_511);  sum_20 = mul_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_512, mul_341, primals_194, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_512 = mul_341 = primals_194 = None
    getitem_134: "f32[8, 48, 1, 1]" = convolution_backward_12[0]
    getitem_135: "f32[1152, 48, 1, 1]" = convolution_backward_12[1]
    getitem_136: "f32[1152]" = convolution_backward_12[2];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_72: "f32[8, 48, 1, 1]" = torch.ops.aten.sigmoid.default(clone_41)
    full_7: "f32[8, 48, 1, 1]" = torch.ops.aten.full.default([8, 48, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_91: "f32[8, 48, 1, 1]" = torch.ops.aten.sub.Tensor(full_7, sigmoid_72);  full_7 = None
    mul_513: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(clone_41, sub_91);  clone_41 = sub_91 = None
    add_264: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Scalar(mul_513, 1);  mul_513 = None
    mul_514: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_72, add_264);  sigmoid_72 = add_264 = None
    mul_515: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_134, mul_514);  getitem_134 = mul_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_515, mean_13, primals_192, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_515 = mean_13 = primals_192 = None
    getitem_137: "f32[8, 1152, 1, 1]" = convolution_backward_13[0]
    getitem_138: "f32[48, 1152, 1, 1]" = convolution_backward_13[1]
    getitem_139: "f32[48]" = convolution_backward_13[2];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_3: "f32[8, 1152, 7, 7]" = torch.ops.aten.expand.default(getitem_137, [8, 1152, 7, 7]);  getitem_137 = None
    div_3: "f32[8, 1152, 7, 7]" = torch.ops.aten.div.Scalar(expand_3, 49);  expand_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_265: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_510, div_3);  mul_510 = div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_73: "f32[8, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(clone_40)
    full_8: "f32[8, 1152, 7, 7]" = torch.ops.aten.full.default([8, 1152, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_92: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(full_8, sigmoid_73);  full_8 = None
    mul_516: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(clone_40, sub_92);  clone_40 = sub_92 = None
    add_266: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Scalar(mul_516, 1);  mul_516 = None
    mul_517: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_73, add_266);  sigmoid_73 = add_266 = None
    mul_518: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_265, mul_517);  add_265 = mul_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_292: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
    unsqueeze_293: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 2);  unsqueeze_292 = None
    unsqueeze_294: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 3);  unsqueeze_293 = None
    sum_21: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_518, [0, 2, 3])
    sub_93: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_294)
    mul_519: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_518, sub_93);  sub_93 = None
    sum_22: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_519, [0, 2, 3]);  mul_519 = None
    mul_520: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_21, 0.002551020408163265)
    unsqueeze_295: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_520, 0);  mul_520 = None
    unsqueeze_296: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 2);  unsqueeze_295 = None
    unsqueeze_297: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 3);  unsqueeze_296 = None
    mul_521: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_22, 0.002551020408163265)
    mul_522: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_523: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_521, mul_522);  mul_521 = mul_522 = None
    unsqueeze_298: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_523, 0);  mul_523 = None
    unsqueeze_299: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 2);  unsqueeze_298 = None
    unsqueeze_300: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 3);  unsqueeze_299 = None
    mul_524: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_86);  primals_86 = None
    unsqueeze_301: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_524, 0);  mul_524 = None
    unsqueeze_302: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None
    unsqueeze_303: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 3);  unsqueeze_302 = None
    sub_94: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_294);  convolution_66 = unsqueeze_294 = None
    mul_525: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_300);  sub_94 = unsqueeze_300 = None
    sub_95: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(mul_518, mul_525);  mul_518 = mul_525 = None
    sub_96: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_95, unsqueeze_297);  sub_95 = unsqueeze_297 = None
    mul_526: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_303);  sub_96 = unsqueeze_303 = None
    mul_527: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_22, squeeze_121);  sum_22 = squeeze_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_526, mul_332, primals_191, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_526 = mul_332 = primals_191 = None
    getitem_140: "f32[8, 1152, 7, 7]" = convolution_backward_14[0]
    getitem_141: "f32[1152, 1, 5, 5]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_74: "f32[8, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(clone_39)
    full_9: "f32[8, 1152, 7, 7]" = torch.ops.aten.full.default([8, 1152, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_97: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(full_9, sigmoid_74);  full_9 = None
    mul_528: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(clone_39, sub_97);  clone_39 = sub_97 = None
    add_267: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Scalar(mul_528, 1);  mul_528 = None
    mul_529: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_74, add_267);  sigmoid_74 = add_267 = None
    mul_530: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_140, mul_529);  getitem_140 = mul_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_304: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
    unsqueeze_305: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 2);  unsqueeze_304 = None
    unsqueeze_306: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 3);  unsqueeze_305 = None
    sum_23: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_530, [0, 2, 3])
    sub_98: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_306)
    mul_531: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_530, sub_98);  sub_98 = None
    sum_24: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_531, [0, 2, 3]);  mul_531 = None
    mul_532: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_23, 0.002551020408163265)
    unsqueeze_307: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_532, 0);  mul_532 = None
    unsqueeze_308: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 2);  unsqueeze_307 = None
    unsqueeze_309: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 3);  unsqueeze_308 = None
    mul_533: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_24, 0.002551020408163265)
    mul_534: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_535: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_533, mul_534);  mul_533 = mul_534 = None
    unsqueeze_310: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_535, 0);  mul_535 = None
    unsqueeze_311: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 2);  unsqueeze_310 = None
    unsqueeze_312: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 3);  unsqueeze_311 = None
    mul_536: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_84);  primals_84 = None
    unsqueeze_313: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_536, 0);  mul_536 = None
    unsqueeze_314: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 2);  unsqueeze_313 = None
    unsqueeze_315: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 3);  unsqueeze_314 = None
    sub_99: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_306);  convolution_65 = unsqueeze_306 = None
    mul_537: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_312);  sub_99 = unsqueeze_312 = None
    sub_100: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(mul_530, mul_537);  mul_530 = mul_537 = None
    sub_101: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_100, unsqueeze_309);  sub_100 = unsqueeze_309 = None
    mul_538: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_315);  sub_101 = unsqueeze_315 = None
    mul_539: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_24, squeeze_118);  sum_24 = squeeze_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_538, add_201, primals_190, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_538 = add_201 = primals_190 = None
    getitem_143: "f32[8, 192, 7, 7]" = convolution_backward_15[0]
    getitem_144: "f32[1152, 192, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_268: "f32[8, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_263, getitem_143);  add_263 = getitem_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_316: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
    unsqueeze_317: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, 2);  unsqueeze_316 = None
    unsqueeze_318: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 3);  unsqueeze_317 = None
    sum_25: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_268, [0, 2, 3])
    sub_102: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_318)
    mul_540: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_268, sub_102);  sub_102 = None
    sum_26: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_540, [0, 2, 3]);  mul_540 = None
    mul_541: "f32[192]" = torch.ops.aten.mul.Tensor(sum_25, 0.002551020408163265)
    unsqueeze_319: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_541, 0);  mul_541 = None
    unsqueeze_320: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 2);  unsqueeze_319 = None
    unsqueeze_321: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 3);  unsqueeze_320 = None
    mul_542: "f32[192]" = torch.ops.aten.mul.Tensor(sum_26, 0.002551020408163265)
    mul_543: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_544: "f32[192]" = torch.ops.aten.mul.Tensor(mul_542, mul_543);  mul_542 = mul_543 = None
    unsqueeze_322: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_544, 0);  mul_544 = None
    unsqueeze_323: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 2);  unsqueeze_322 = None
    unsqueeze_324: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 3);  unsqueeze_323 = None
    mul_545: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_82);  primals_82 = None
    unsqueeze_325: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_545, 0);  mul_545 = None
    unsqueeze_326: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None
    unsqueeze_327: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 3);  unsqueeze_326 = None
    sub_103: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_318);  convolution_64 = unsqueeze_318 = None
    mul_546: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_324);  sub_103 = unsqueeze_324 = None
    sub_104: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(add_268, mul_546);  mul_546 = None
    sub_105: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(sub_104, unsqueeze_321);  sub_104 = unsqueeze_321 = None
    mul_547: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_327);  sub_105 = unsqueeze_327 = None
    mul_548: "f32[192]" = torch.ops.aten.mul.Tensor(sum_26, squeeze_115);  sum_26 = squeeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_547, mul_317, primals_189, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_547 = mul_317 = primals_189 = None
    getitem_146: "f32[8, 1152, 7, 7]" = convolution_backward_16[0]
    getitem_147: "f32[192, 1152, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_549: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_146, mul_315);  mul_315 = None
    mul_550: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_146, sigmoid_51);  getitem_146 = sigmoid_51 = None
    sum_27: "f32[8, 1152, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_549, [2, 3], True);  mul_549 = None
    alias_19: "f32[8, 1152, 1, 1]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    sub_106: "f32[8, 1152, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_19)
    mul_551: "f32[8, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(alias_19, sub_106);  alias_19 = sub_106 = None
    mul_552: "f32[8, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(sum_27, mul_551);  sum_27 = mul_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_552, mul_316, primals_187, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_552 = mul_316 = primals_187 = None
    getitem_149: "f32[8, 48, 1, 1]" = convolution_backward_17[0]
    getitem_150: "f32[1152, 48, 1, 1]" = convolution_backward_17[1]
    getitem_151: "f32[1152]" = convolution_backward_17[2];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_75: "f32[8, 48, 1, 1]" = torch.ops.aten.sigmoid.default(clone_38)
    full_10: "f32[8, 48, 1, 1]" = torch.ops.aten.full.default([8, 48, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_107: "f32[8, 48, 1, 1]" = torch.ops.aten.sub.Tensor(full_10, sigmoid_75);  full_10 = None
    mul_553: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(clone_38, sub_107);  clone_38 = sub_107 = None
    add_269: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Scalar(mul_553, 1);  mul_553 = None
    mul_554: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_75, add_269);  sigmoid_75 = add_269 = None
    mul_555: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_149, mul_554);  getitem_149 = mul_554 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_555, mean_12, primals_185, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_555 = mean_12 = primals_185 = None
    getitem_152: "f32[8, 1152, 1, 1]" = convolution_backward_18[0]
    getitem_153: "f32[48, 1152, 1, 1]" = convolution_backward_18[1]
    getitem_154: "f32[48]" = convolution_backward_18[2];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_4: "f32[8, 1152, 7, 7]" = torch.ops.aten.expand.default(getitem_152, [8, 1152, 7, 7]);  getitem_152 = None
    div_4: "f32[8, 1152, 7, 7]" = torch.ops.aten.div.Scalar(expand_4, 49);  expand_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_270: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_550, div_4);  mul_550 = div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_76: "f32[8, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(clone_37)
    full_11: "f32[8, 1152, 7, 7]" = torch.ops.aten.full.default([8, 1152, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_108: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(full_11, sigmoid_76);  full_11 = None
    mul_556: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(clone_37, sub_108);  clone_37 = sub_108 = None
    add_271: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Scalar(mul_556, 1);  mul_556 = None
    mul_557: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_76, add_271);  sigmoid_76 = add_271 = None
    mul_558: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_270, mul_557);  add_270 = mul_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_328: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
    unsqueeze_329: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 2);  unsqueeze_328 = None
    unsqueeze_330: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 3);  unsqueeze_329 = None
    sum_28: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_558, [0, 2, 3])
    sub_109: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_330)
    mul_559: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_558, sub_109);  sub_109 = None
    sum_29: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_559, [0, 2, 3]);  mul_559 = None
    mul_560: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_28, 0.002551020408163265)
    unsqueeze_331: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_560, 0);  mul_560 = None
    unsqueeze_332: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 2);  unsqueeze_331 = None
    unsqueeze_333: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 3);  unsqueeze_332 = None
    mul_561: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_29, 0.002551020408163265)
    mul_562: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_563: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_561, mul_562);  mul_561 = mul_562 = None
    unsqueeze_334: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_563, 0);  mul_563 = None
    unsqueeze_335: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 2);  unsqueeze_334 = None
    unsqueeze_336: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 3);  unsqueeze_335 = None
    mul_564: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_80);  primals_80 = None
    unsqueeze_337: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_564, 0);  mul_564 = None
    unsqueeze_338: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 2);  unsqueeze_337 = None
    unsqueeze_339: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 3);  unsqueeze_338 = None
    sub_110: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_330);  convolution_61 = unsqueeze_330 = None
    mul_565: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_336);  sub_110 = unsqueeze_336 = None
    sub_111: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(mul_558, mul_565);  mul_558 = mul_565 = None
    sub_112: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_111, unsqueeze_333);  sub_111 = unsqueeze_333 = None
    mul_566: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_339);  sub_112 = unsqueeze_339 = None
    mul_567: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_112);  sum_29 = squeeze_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_566, mul_307, primals_184, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_566 = mul_307 = primals_184 = None
    getitem_155: "f32[8, 1152, 7, 7]" = convolution_backward_19[0]
    getitem_156: "f32[1152, 1, 5, 5]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_77: "f32[8, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(clone_36)
    full_12: "f32[8, 1152, 7, 7]" = torch.ops.aten.full.default([8, 1152, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_113: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(full_12, sigmoid_77);  full_12 = None
    mul_568: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(clone_36, sub_113);  clone_36 = sub_113 = None
    add_272: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Scalar(mul_568, 1);  mul_568 = None
    mul_569: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_77, add_272);  sigmoid_77 = add_272 = None
    mul_570: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_155, mul_569);  getitem_155 = mul_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_340: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
    unsqueeze_341: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 2);  unsqueeze_340 = None
    unsqueeze_342: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 3);  unsqueeze_341 = None
    sum_30: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_570, [0, 2, 3])
    sub_114: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_342)
    mul_571: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_570, sub_114);  sub_114 = None
    sum_31: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_571, [0, 2, 3]);  mul_571 = None
    mul_572: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_30, 0.002551020408163265)
    unsqueeze_343: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_572, 0);  mul_572 = None
    unsqueeze_344: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 2);  unsqueeze_343 = None
    unsqueeze_345: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 3);  unsqueeze_344 = None
    mul_573: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_31, 0.002551020408163265)
    mul_574: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_575: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_573, mul_574);  mul_573 = mul_574 = None
    unsqueeze_346: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_575, 0);  mul_575 = None
    unsqueeze_347: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 2);  unsqueeze_346 = None
    unsqueeze_348: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 3);  unsqueeze_347 = None
    mul_576: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_78);  primals_78 = None
    unsqueeze_349: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_576, 0);  mul_576 = None
    unsqueeze_350: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None
    unsqueeze_351: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 3);  unsqueeze_350 = None
    sub_115: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_342);  convolution_60 = unsqueeze_342 = None
    mul_577: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_348);  sub_115 = unsqueeze_348 = None
    sub_116: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(mul_570, mul_577);  mul_570 = mul_577 = None
    sub_117: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_116, unsqueeze_345);  sub_116 = unsqueeze_345 = None
    mul_578: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_351);  sub_117 = unsqueeze_351 = None
    mul_579: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_109);  sum_31 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_578, add_185, primals_183, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_578 = add_185 = primals_183 = None
    getitem_158: "f32[8, 192, 7, 7]" = convolution_backward_20[0]
    getitem_159: "f32[1152, 192, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_273: "f32[8, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_268, getitem_158);  add_268 = getitem_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_352: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
    unsqueeze_353: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 2);  unsqueeze_352 = None
    unsqueeze_354: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 3);  unsqueeze_353 = None
    sum_32: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_273, [0, 2, 3])
    sub_118: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_354)
    mul_580: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_273, sub_118);  sub_118 = None
    sum_33: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_580, [0, 2, 3]);  mul_580 = None
    mul_581: "f32[192]" = torch.ops.aten.mul.Tensor(sum_32, 0.002551020408163265)
    unsqueeze_355: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_581, 0);  mul_581 = None
    unsqueeze_356: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 2);  unsqueeze_355 = None
    unsqueeze_357: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 3);  unsqueeze_356 = None
    mul_582: "f32[192]" = torch.ops.aten.mul.Tensor(sum_33, 0.002551020408163265)
    mul_583: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_584: "f32[192]" = torch.ops.aten.mul.Tensor(mul_582, mul_583);  mul_582 = mul_583 = None
    unsqueeze_358: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_584, 0);  mul_584 = None
    unsqueeze_359: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 2);  unsqueeze_358 = None
    unsqueeze_360: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 3);  unsqueeze_359 = None
    mul_585: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_76);  primals_76 = None
    unsqueeze_361: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_585, 0);  mul_585 = None
    unsqueeze_362: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
    unsqueeze_363: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 3);  unsqueeze_362 = None
    sub_119: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_354);  convolution_59 = unsqueeze_354 = None
    mul_586: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_360);  sub_119 = unsqueeze_360 = None
    sub_120: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(add_273, mul_586);  add_273 = mul_586 = None
    sub_121: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(sub_120, unsqueeze_357);  sub_120 = unsqueeze_357 = None
    mul_587: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_363);  sub_121 = unsqueeze_363 = None
    mul_588: "f32[192]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_106);  sum_33 = squeeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_587, mul_292, primals_182, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_587 = mul_292 = primals_182 = None
    getitem_161: "f32[8, 672, 7, 7]" = convolution_backward_21[0]
    getitem_162: "f32[192, 672, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_589: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_161, mul_290);  mul_290 = None
    mul_590: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_161, sigmoid_47);  getitem_161 = sigmoid_47 = None
    sum_34: "f32[8, 672, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_589, [2, 3], True);  mul_589 = None
    alias_20: "f32[8, 672, 1, 1]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    sub_122: "f32[8, 672, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_20)
    mul_591: "f32[8, 672, 1, 1]" = torch.ops.aten.mul.Tensor(alias_20, sub_122);  alias_20 = sub_122 = None
    mul_592: "f32[8, 672, 1, 1]" = torch.ops.aten.mul.Tensor(sum_34, mul_591);  sum_34 = mul_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_592, mul_291, primals_180, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_592 = mul_291 = primals_180 = None
    getitem_164: "f32[8, 28, 1, 1]" = convolution_backward_22[0]
    getitem_165: "f32[672, 28, 1, 1]" = convolution_backward_22[1]
    getitem_166: "f32[672]" = convolution_backward_22[2];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_78: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(clone_35)
    full_13: "f32[8, 28, 1, 1]" = torch.ops.aten.full.default([8, 28, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_123: "f32[8, 28, 1, 1]" = torch.ops.aten.sub.Tensor(full_13, sigmoid_78);  full_13 = None
    mul_593: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(clone_35, sub_123);  clone_35 = sub_123 = None
    add_274: "f32[8, 28, 1, 1]" = torch.ops.aten.add.Scalar(mul_593, 1);  mul_593 = None
    mul_594: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_78, add_274);  sigmoid_78 = add_274 = None
    mul_595: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_164, mul_594);  getitem_164 = mul_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_595, mean_11, primals_178, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_595 = mean_11 = primals_178 = None
    getitem_167: "f32[8, 672, 1, 1]" = convolution_backward_23[0]
    getitem_168: "f32[28, 672, 1, 1]" = convolution_backward_23[1]
    getitem_169: "f32[28]" = convolution_backward_23[2];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_5: "f32[8, 672, 7, 7]" = torch.ops.aten.expand.default(getitem_167, [8, 672, 7, 7]);  getitem_167 = None
    div_5: "f32[8, 672, 7, 7]" = torch.ops.aten.div.Scalar(expand_5, 49);  expand_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_275: "f32[8, 672, 7, 7]" = torch.ops.aten.add.Tensor(mul_590, div_5);  mul_590 = div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_79: "f32[8, 672, 7, 7]" = torch.ops.aten.sigmoid.default(clone_34)
    full_14: "f32[8, 672, 7, 7]" = torch.ops.aten.full.default([8, 672, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_124: "f32[8, 672, 7, 7]" = torch.ops.aten.sub.Tensor(full_14, sigmoid_79);  full_14 = None
    mul_596: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(clone_34, sub_124);  clone_34 = sub_124 = None
    add_276: "f32[8, 672, 7, 7]" = torch.ops.aten.add.Scalar(mul_596, 1);  mul_596 = None
    mul_597: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_79, add_276);  sigmoid_79 = add_276 = None
    mul_598: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(add_275, mul_597);  add_275 = mul_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_364: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
    unsqueeze_365: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 2);  unsqueeze_364 = None
    unsqueeze_366: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 3);  unsqueeze_365 = None
    sum_35: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_598, [0, 2, 3])
    sub_125: "f32[8, 672, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_366)
    mul_599: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(mul_598, sub_125);  sub_125 = None
    sum_36: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_599, [0, 2, 3]);  mul_599 = None
    mul_600: "f32[672]" = torch.ops.aten.mul.Tensor(sum_35, 0.002551020408163265)
    unsqueeze_367: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_600, 0);  mul_600 = None
    unsqueeze_368: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
    unsqueeze_369: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 3);  unsqueeze_368 = None
    mul_601: "f32[672]" = torch.ops.aten.mul.Tensor(sum_36, 0.002551020408163265)
    mul_602: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_603: "f32[672]" = torch.ops.aten.mul.Tensor(mul_601, mul_602);  mul_601 = mul_602 = None
    unsqueeze_370: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_603, 0);  mul_603 = None
    unsqueeze_371: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 2);  unsqueeze_370 = None
    unsqueeze_372: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 3);  unsqueeze_371 = None
    mul_604: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_74);  primals_74 = None
    unsqueeze_373: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_604, 0);  mul_604 = None
    unsqueeze_374: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
    unsqueeze_375: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 3);  unsqueeze_374 = None
    sub_126: "f32[8, 672, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_366);  convolution_56 = unsqueeze_366 = None
    mul_605: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_372);  sub_126 = unsqueeze_372 = None
    sub_127: "f32[8, 672, 7, 7]" = torch.ops.aten.sub.Tensor(mul_598, mul_605);  mul_598 = mul_605 = None
    sub_128: "f32[8, 672, 7, 7]" = torch.ops.aten.sub.Tensor(sub_127, unsqueeze_369);  sub_127 = unsqueeze_369 = None
    mul_606: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_375);  sub_128 = unsqueeze_375 = None
    mul_607: "f32[672]" = torch.ops.aten.mul.Tensor(sum_36, squeeze_103);  sum_36 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_606, constant_pad_nd_4, primals_73, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 672, [True, True, False]);  mul_606 = constant_pad_nd_4 = primals_73 = None
    getitem_170: "f32[8, 672, 17, 17]" = convolution_backward_24[0]
    getitem_171: "f32[672, 1, 5, 5]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_5: "f32[8, 672, 14, 14]" = torch.ops.aten.constant_pad_nd.default(getitem_170, [-1, -2, -1, -2]);  getitem_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_80: "f32[8, 672, 14, 14]" = torch.ops.aten.sigmoid.default(clone_33)
    full_15: "f32[8, 672, 14, 14]" = torch.ops.aten.full.default([8, 672, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_129: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(full_15, sigmoid_80);  full_15 = None
    mul_608: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(clone_33, sub_129);  clone_33 = sub_129 = None
    add_277: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Scalar(mul_608, 1);  mul_608 = None
    mul_609: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_80, add_277);  sigmoid_80 = add_277 = None
    mul_610: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(constant_pad_nd_5, mul_609);  constant_pad_nd_5 = mul_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_376: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
    unsqueeze_377: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 2);  unsqueeze_376 = None
    unsqueeze_378: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 3);  unsqueeze_377 = None
    sum_37: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_610, [0, 2, 3])
    sub_130: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_378)
    mul_611: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_610, sub_130);  sub_130 = None
    sum_38: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_611, [0, 2, 3]);  mul_611 = None
    mul_612: "f32[672]" = torch.ops.aten.mul.Tensor(sum_37, 0.0006377551020408163)
    unsqueeze_379: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_612, 0);  mul_612 = None
    unsqueeze_380: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
    unsqueeze_381: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 3);  unsqueeze_380 = None
    mul_613: "f32[672]" = torch.ops.aten.mul.Tensor(sum_38, 0.0006377551020408163)
    mul_614: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_615: "f32[672]" = torch.ops.aten.mul.Tensor(mul_613, mul_614);  mul_613 = mul_614 = None
    unsqueeze_382: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_615, 0);  mul_615 = None
    unsqueeze_383: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 2);  unsqueeze_382 = None
    unsqueeze_384: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 3);  unsqueeze_383 = None
    mul_616: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_71);  primals_71 = None
    unsqueeze_385: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_616, 0);  mul_616 = None
    unsqueeze_386: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
    unsqueeze_387: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
    sub_131: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_378);  convolution_55 = unsqueeze_378 = None
    mul_617: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_384);  sub_131 = unsqueeze_384 = None
    sub_132: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(mul_610, mul_617);  mul_610 = mul_617 = None
    sub_133: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(sub_132, unsqueeze_381);  sub_132 = unsqueeze_381 = None
    mul_618: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_387);  sub_133 = unsqueeze_387 = None
    mul_619: "f32[672]" = torch.ops.aten.mul.Tensor(sum_38, squeeze_100);  sum_38 = squeeze_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_618, add_170, primals_177, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_618 = add_170 = primals_177 = None
    getitem_173: "f32[8, 112, 14, 14]" = convolution_backward_25[0]
    getitem_174: "f32[672, 112, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_388: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
    unsqueeze_389: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 2);  unsqueeze_388 = None
    unsqueeze_390: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 3);  unsqueeze_389 = None
    sum_39: "f32[112]" = torch.ops.aten.sum.dim_IntList(getitem_173, [0, 2, 3])
    sub_134: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_390)
    mul_620: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_173, sub_134);  sub_134 = None
    sum_40: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_620, [0, 2, 3]);  mul_620 = None
    mul_621: "f32[112]" = torch.ops.aten.mul.Tensor(sum_39, 0.0006377551020408163)
    unsqueeze_391: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_621, 0);  mul_621 = None
    unsqueeze_392: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
    unsqueeze_393: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 3);  unsqueeze_392 = None
    mul_622: "f32[112]" = torch.ops.aten.mul.Tensor(sum_40, 0.0006377551020408163)
    mul_623: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_624: "f32[112]" = torch.ops.aten.mul.Tensor(mul_622, mul_623);  mul_622 = mul_623 = None
    unsqueeze_394: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_624, 0);  mul_624 = None
    unsqueeze_395: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 2);  unsqueeze_394 = None
    unsqueeze_396: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 3);  unsqueeze_395 = None
    mul_625: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_69);  primals_69 = None
    unsqueeze_397: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_625, 0);  mul_625 = None
    unsqueeze_398: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
    unsqueeze_399: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 3);  unsqueeze_398 = None
    sub_135: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_390);  convolution_54 = unsqueeze_390 = None
    mul_626: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_396);  sub_135 = unsqueeze_396 = None
    sub_136: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_173, mul_626);  mul_626 = None
    sub_137: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(sub_136, unsqueeze_393);  sub_136 = unsqueeze_393 = None
    mul_627: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_399);  sub_137 = unsqueeze_399 = None
    mul_628: "f32[112]" = torch.ops.aten.mul.Tensor(sum_40, squeeze_97);  sum_40 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_627, mul_267, primals_176, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_627 = mul_267 = primals_176 = None
    getitem_176: "f32[8, 672, 14, 14]" = convolution_backward_26[0]
    getitem_177: "f32[112, 672, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_629: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_176, mul_265);  mul_265 = None
    mul_630: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_176, sigmoid_43);  getitem_176 = sigmoid_43 = None
    sum_41: "f32[8, 672, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_629, [2, 3], True);  mul_629 = None
    alias_21: "f32[8, 672, 1, 1]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    sub_138: "f32[8, 672, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_21)
    mul_631: "f32[8, 672, 1, 1]" = torch.ops.aten.mul.Tensor(alias_21, sub_138);  alias_21 = sub_138 = None
    mul_632: "f32[8, 672, 1, 1]" = torch.ops.aten.mul.Tensor(sum_41, mul_631);  sum_41 = mul_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_632, mul_266, primals_174, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_632 = mul_266 = primals_174 = None
    getitem_179: "f32[8, 28, 1, 1]" = convolution_backward_27[0]
    getitem_180: "f32[672, 28, 1, 1]" = convolution_backward_27[1]
    getitem_181: "f32[672]" = convolution_backward_27[2];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_81: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(clone_32)
    full_16: "f32[8, 28, 1, 1]" = torch.ops.aten.full.default([8, 28, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_139: "f32[8, 28, 1, 1]" = torch.ops.aten.sub.Tensor(full_16, sigmoid_81);  full_16 = None
    mul_633: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(clone_32, sub_139);  clone_32 = sub_139 = None
    add_278: "f32[8, 28, 1, 1]" = torch.ops.aten.add.Scalar(mul_633, 1);  mul_633 = None
    mul_634: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_81, add_278);  sigmoid_81 = add_278 = None
    mul_635: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_179, mul_634);  getitem_179 = mul_634 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_635, mean_10, primals_172, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_635 = mean_10 = primals_172 = None
    getitem_182: "f32[8, 672, 1, 1]" = convolution_backward_28[0]
    getitem_183: "f32[28, 672, 1, 1]" = convolution_backward_28[1]
    getitem_184: "f32[28]" = convolution_backward_28[2];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_6: "f32[8, 672, 14, 14]" = torch.ops.aten.expand.default(getitem_182, [8, 672, 14, 14]);  getitem_182 = None
    div_6: "f32[8, 672, 14, 14]" = torch.ops.aten.div.Scalar(expand_6, 196);  expand_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_279: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_630, div_6);  mul_630 = div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_82: "f32[8, 672, 14, 14]" = torch.ops.aten.sigmoid.default(clone_31)
    full_17: "f32[8, 672, 14, 14]" = torch.ops.aten.full.default([8, 672, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_140: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(full_17, sigmoid_82);  full_17 = None
    mul_636: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(clone_31, sub_140);  clone_31 = sub_140 = None
    add_280: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Scalar(mul_636, 1);  mul_636 = None
    mul_637: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_82, add_280);  sigmoid_82 = add_280 = None
    mul_638: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_279, mul_637);  add_279 = mul_637 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_400: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
    unsqueeze_401: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 2);  unsqueeze_400 = None
    unsqueeze_402: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 3);  unsqueeze_401 = None
    sum_42: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_638, [0, 2, 3])
    sub_141: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_402)
    mul_639: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_638, sub_141);  sub_141 = None
    sum_43: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_639, [0, 2, 3]);  mul_639 = None
    mul_640: "f32[672]" = torch.ops.aten.mul.Tensor(sum_42, 0.0006377551020408163)
    unsqueeze_403: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_640, 0);  mul_640 = None
    unsqueeze_404: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 2);  unsqueeze_403 = None
    unsqueeze_405: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 3);  unsqueeze_404 = None
    mul_641: "f32[672]" = torch.ops.aten.mul.Tensor(sum_43, 0.0006377551020408163)
    mul_642: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_643: "f32[672]" = torch.ops.aten.mul.Tensor(mul_641, mul_642);  mul_641 = mul_642 = None
    unsqueeze_406: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_643, 0);  mul_643 = None
    unsqueeze_407: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 2);  unsqueeze_406 = None
    unsqueeze_408: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 3);  unsqueeze_407 = None
    mul_644: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_67);  primals_67 = None
    unsqueeze_409: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_644, 0);  mul_644 = None
    unsqueeze_410: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 2);  unsqueeze_409 = None
    unsqueeze_411: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 3);  unsqueeze_410 = None
    sub_142: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_402);  convolution_51 = unsqueeze_402 = None
    mul_645: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_408);  sub_142 = unsqueeze_408 = None
    sub_143: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(mul_638, mul_645);  mul_638 = mul_645 = None
    sub_144: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(sub_143, unsqueeze_405);  sub_143 = unsqueeze_405 = None
    mul_646: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_411);  sub_144 = unsqueeze_411 = None
    mul_647: "f32[672]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_94);  sum_43 = squeeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_646, mul_257, primals_171, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False]);  mul_646 = mul_257 = primals_171 = None
    getitem_185: "f32[8, 672, 14, 14]" = convolution_backward_29[0]
    getitem_186: "f32[672, 1, 5, 5]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_83: "f32[8, 672, 14, 14]" = torch.ops.aten.sigmoid.default(clone_30)
    full_18: "f32[8, 672, 14, 14]" = torch.ops.aten.full.default([8, 672, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_145: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(full_18, sigmoid_83);  full_18 = None
    mul_648: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(clone_30, sub_145);  clone_30 = sub_145 = None
    add_281: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Scalar(mul_648, 1);  mul_648 = None
    mul_649: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_83, add_281);  sigmoid_83 = add_281 = None
    mul_650: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_185, mul_649);  getitem_185 = mul_649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_412: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_413: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 2);  unsqueeze_412 = None
    unsqueeze_414: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 3);  unsqueeze_413 = None
    sum_44: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_650, [0, 2, 3])
    sub_146: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_414)
    mul_651: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_650, sub_146);  sub_146 = None
    sum_45: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_651, [0, 2, 3]);  mul_651 = None
    mul_652: "f32[672]" = torch.ops.aten.mul.Tensor(sum_44, 0.0006377551020408163)
    unsqueeze_415: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_652, 0);  mul_652 = None
    unsqueeze_416: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 2);  unsqueeze_415 = None
    unsqueeze_417: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 3);  unsqueeze_416 = None
    mul_653: "f32[672]" = torch.ops.aten.mul.Tensor(sum_45, 0.0006377551020408163)
    mul_654: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_655: "f32[672]" = torch.ops.aten.mul.Tensor(mul_653, mul_654);  mul_653 = mul_654 = None
    unsqueeze_418: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_655, 0);  mul_655 = None
    unsqueeze_419: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 2);  unsqueeze_418 = None
    unsqueeze_420: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 3);  unsqueeze_419 = None
    mul_656: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_65);  primals_65 = None
    unsqueeze_421: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_656, 0);  mul_656 = None
    unsqueeze_422: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None
    unsqueeze_423: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 3);  unsqueeze_422 = None
    sub_147: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_414);  convolution_50 = unsqueeze_414 = None
    mul_657: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_420);  sub_147 = unsqueeze_420 = None
    sub_148: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(mul_650, mul_657);  mul_650 = mul_657 = None
    sub_149: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(sub_148, unsqueeze_417);  sub_148 = unsqueeze_417 = None
    mul_658: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_423);  sub_149 = unsqueeze_423 = None
    mul_659: "f32[672]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_91);  sum_45 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_658, add_154, primals_170, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_658 = add_154 = primals_170 = None
    getitem_188: "f32[8, 112, 14, 14]" = convolution_backward_30[0]
    getitem_189: "f32[672, 112, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_282: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(getitem_173, getitem_188);  getitem_173 = getitem_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_424: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    unsqueeze_425: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 2);  unsqueeze_424 = None
    unsqueeze_426: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 3);  unsqueeze_425 = None
    sum_46: "f32[112]" = torch.ops.aten.sum.dim_IntList(add_282, [0, 2, 3])
    sub_150: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_426)
    mul_660: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(add_282, sub_150);  sub_150 = None
    sum_47: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_660, [0, 2, 3]);  mul_660 = None
    mul_661: "f32[112]" = torch.ops.aten.mul.Tensor(sum_46, 0.0006377551020408163)
    unsqueeze_427: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_661, 0);  mul_661 = None
    unsqueeze_428: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 2);  unsqueeze_427 = None
    unsqueeze_429: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 3);  unsqueeze_428 = None
    mul_662: "f32[112]" = torch.ops.aten.mul.Tensor(sum_47, 0.0006377551020408163)
    mul_663: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_664: "f32[112]" = torch.ops.aten.mul.Tensor(mul_662, mul_663);  mul_662 = mul_663 = None
    unsqueeze_430: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_664, 0);  mul_664 = None
    unsqueeze_431: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 2);  unsqueeze_430 = None
    unsqueeze_432: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 3);  unsqueeze_431 = None
    mul_665: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_63);  primals_63 = None
    unsqueeze_433: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_665, 0);  mul_665 = None
    unsqueeze_434: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
    unsqueeze_435: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 3);  unsqueeze_434 = None
    sub_151: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_426);  convolution_49 = unsqueeze_426 = None
    mul_666: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_432);  sub_151 = unsqueeze_432 = None
    sub_152: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(add_282, mul_666);  mul_666 = None
    sub_153: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(sub_152, unsqueeze_429);  sub_152 = unsqueeze_429 = None
    mul_667: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_435);  sub_153 = unsqueeze_435 = None
    mul_668: "f32[112]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_88);  sum_47 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_667, mul_242, primals_169, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_667 = mul_242 = primals_169 = None
    getitem_191: "f32[8, 672, 14, 14]" = convolution_backward_31[0]
    getitem_192: "f32[112, 672, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_669: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_191, mul_240);  mul_240 = None
    mul_670: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_191, sigmoid_39);  getitem_191 = sigmoid_39 = None
    sum_48: "f32[8, 672, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_669, [2, 3], True);  mul_669 = None
    alias_22: "f32[8, 672, 1, 1]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    sub_154: "f32[8, 672, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_22)
    mul_671: "f32[8, 672, 1, 1]" = torch.ops.aten.mul.Tensor(alias_22, sub_154);  alias_22 = sub_154 = None
    mul_672: "f32[8, 672, 1, 1]" = torch.ops.aten.mul.Tensor(sum_48, mul_671);  sum_48 = mul_671 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_672, mul_241, primals_167, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_672 = mul_241 = primals_167 = None
    getitem_194: "f32[8, 28, 1, 1]" = convolution_backward_32[0]
    getitem_195: "f32[672, 28, 1, 1]" = convolution_backward_32[1]
    getitem_196: "f32[672]" = convolution_backward_32[2];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_84: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(clone_29)
    full_19: "f32[8, 28, 1, 1]" = torch.ops.aten.full.default([8, 28, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_155: "f32[8, 28, 1, 1]" = torch.ops.aten.sub.Tensor(full_19, sigmoid_84);  full_19 = None
    mul_673: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(clone_29, sub_155);  clone_29 = sub_155 = None
    add_283: "f32[8, 28, 1, 1]" = torch.ops.aten.add.Scalar(mul_673, 1);  mul_673 = None
    mul_674: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_84, add_283);  sigmoid_84 = add_283 = None
    mul_675: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_194, mul_674);  getitem_194 = mul_674 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_675, mean_9, primals_165, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_675 = mean_9 = primals_165 = None
    getitem_197: "f32[8, 672, 1, 1]" = convolution_backward_33[0]
    getitem_198: "f32[28, 672, 1, 1]" = convolution_backward_33[1]
    getitem_199: "f32[28]" = convolution_backward_33[2];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_7: "f32[8, 672, 14, 14]" = torch.ops.aten.expand.default(getitem_197, [8, 672, 14, 14]);  getitem_197 = None
    div_7: "f32[8, 672, 14, 14]" = torch.ops.aten.div.Scalar(expand_7, 196);  expand_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_284: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_670, div_7);  mul_670 = div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_85: "f32[8, 672, 14, 14]" = torch.ops.aten.sigmoid.default(clone_28)
    full_20: "f32[8, 672, 14, 14]" = torch.ops.aten.full.default([8, 672, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_156: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(full_20, sigmoid_85);  full_20 = None
    mul_676: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(clone_28, sub_156);  clone_28 = sub_156 = None
    add_285: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Scalar(mul_676, 1);  mul_676 = None
    mul_677: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_85, add_285);  sigmoid_85 = add_285 = None
    mul_678: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_284, mul_677);  add_284 = mul_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_436: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_437: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 2);  unsqueeze_436 = None
    unsqueeze_438: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 3);  unsqueeze_437 = None
    sum_49: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_678, [0, 2, 3])
    sub_157: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_438)
    mul_679: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_678, sub_157);  sub_157 = None
    sum_50: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_679, [0, 2, 3]);  mul_679 = None
    mul_680: "f32[672]" = torch.ops.aten.mul.Tensor(sum_49, 0.0006377551020408163)
    unsqueeze_439: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_680, 0);  mul_680 = None
    unsqueeze_440: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 2);  unsqueeze_439 = None
    unsqueeze_441: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 3);  unsqueeze_440 = None
    mul_681: "f32[672]" = torch.ops.aten.mul.Tensor(sum_50, 0.0006377551020408163)
    mul_682: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_683: "f32[672]" = torch.ops.aten.mul.Tensor(mul_681, mul_682);  mul_681 = mul_682 = None
    unsqueeze_442: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_683, 0);  mul_683 = None
    unsqueeze_443: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 2);  unsqueeze_442 = None
    unsqueeze_444: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 3);  unsqueeze_443 = None
    mul_684: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_61);  primals_61 = None
    unsqueeze_445: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_684, 0);  mul_684 = None
    unsqueeze_446: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
    unsqueeze_447: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 3);  unsqueeze_446 = None
    sub_158: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_438);  convolution_46 = unsqueeze_438 = None
    mul_685: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_444);  sub_158 = unsqueeze_444 = None
    sub_159: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(mul_678, mul_685);  mul_678 = mul_685 = None
    sub_160: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(sub_159, unsqueeze_441);  sub_159 = unsqueeze_441 = None
    mul_686: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_447);  sub_160 = unsqueeze_447 = None
    mul_687: "f32[672]" = torch.ops.aten.mul.Tensor(sum_50, squeeze_85);  sum_50 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_686, mul_232, primals_164, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False]);  mul_686 = mul_232 = primals_164 = None
    getitem_200: "f32[8, 672, 14, 14]" = convolution_backward_34[0]
    getitem_201: "f32[672, 1, 5, 5]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_86: "f32[8, 672, 14, 14]" = torch.ops.aten.sigmoid.default(clone_27)
    full_21: "f32[8, 672, 14, 14]" = torch.ops.aten.full.default([8, 672, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_161: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(full_21, sigmoid_86);  full_21 = None
    mul_688: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(clone_27, sub_161);  clone_27 = sub_161 = None
    add_286: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Scalar(mul_688, 1);  mul_688 = None
    mul_689: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_86, add_286);  sigmoid_86 = add_286 = None
    mul_690: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_200, mul_689);  getitem_200 = mul_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_448: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_449: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, 2);  unsqueeze_448 = None
    unsqueeze_450: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 3);  unsqueeze_449 = None
    sum_51: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_690, [0, 2, 3])
    sub_162: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_450)
    mul_691: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_690, sub_162);  sub_162 = None
    sum_52: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_691, [0, 2, 3]);  mul_691 = None
    mul_692: "f32[672]" = torch.ops.aten.mul.Tensor(sum_51, 0.0006377551020408163)
    unsqueeze_451: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_692, 0);  mul_692 = None
    unsqueeze_452: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 2);  unsqueeze_451 = None
    unsqueeze_453: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 3);  unsqueeze_452 = None
    mul_693: "f32[672]" = torch.ops.aten.mul.Tensor(sum_52, 0.0006377551020408163)
    mul_694: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_695: "f32[672]" = torch.ops.aten.mul.Tensor(mul_693, mul_694);  mul_693 = mul_694 = None
    unsqueeze_454: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_695, 0);  mul_695 = None
    unsqueeze_455: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 2);  unsqueeze_454 = None
    unsqueeze_456: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 3);  unsqueeze_455 = None
    mul_696: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_59);  primals_59 = None
    unsqueeze_457: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_696, 0);  mul_696 = None
    unsqueeze_458: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 2);  unsqueeze_457 = None
    unsqueeze_459: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 3);  unsqueeze_458 = None
    sub_163: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_450);  convolution_45 = unsqueeze_450 = None
    mul_697: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_456);  sub_163 = unsqueeze_456 = None
    sub_164: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(mul_690, mul_697);  mul_690 = mul_697 = None
    sub_165: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(sub_164, unsqueeze_453);  sub_164 = unsqueeze_453 = None
    mul_698: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_459);  sub_165 = unsqueeze_459 = None
    mul_699: "f32[672]" = torch.ops.aten.mul.Tensor(sum_52, squeeze_82);  sum_52 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_698, add_138, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_698 = add_138 = primals_163 = None
    getitem_203: "f32[8, 112, 14, 14]" = convolution_backward_35[0]
    getitem_204: "f32[672, 112, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_287: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(add_282, getitem_203);  add_282 = getitem_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_460: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_461: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, 2);  unsqueeze_460 = None
    unsqueeze_462: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 3);  unsqueeze_461 = None
    sum_53: "f32[112]" = torch.ops.aten.sum.dim_IntList(add_287, [0, 2, 3])
    sub_166: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_462)
    mul_700: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(add_287, sub_166);  sub_166 = None
    sum_54: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_700, [0, 2, 3]);  mul_700 = None
    mul_701: "f32[112]" = torch.ops.aten.mul.Tensor(sum_53, 0.0006377551020408163)
    unsqueeze_463: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_701, 0);  mul_701 = None
    unsqueeze_464: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 2);  unsqueeze_463 = None
    unsqueeze_465: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 3);  unsqueeze_464 = None
    mul_702: "f32[112]" = torch.ops.aten.mul.Tensor(sum_54, 0.0006377551020408163)
    mul_703: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_704: "f32[112]" = torch.ops.aten.mul.Tensor(mul_702, mul_703);  mul_702 = mul_703 = None
    unsqueeze_466: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_704, 0);  mul_704 = None
    unsqueeze_467: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 2);  unsqueeze_466 = None
    unsqueeze_468: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 3);  unsqueeze_467 = None
    mul_705: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_57);  primals_57 = None
    unsqueeze_469: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_705, 0);  mul_705 = None
    unsqueeze_470: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None
    unsqueeze_471: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 3);  unsqueeze_470 = None
    sub_167: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_462);  convolution_44 = unsqueeze_462 = None
    mul_706: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_468);  sub_167 = unsqueeze_468 = None
    sub_168: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(add_287, mul_706);  add_287 = mul_706 = None
    sub_169: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(sub_168, unsqueeze_465);  sub_168 = unsqueeze_465 = None
    mul_707: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_471);  sub_169 = unsqueeze_471 = None
    mul_708: "f32[112]" = torch.ops.aten.mul.Tensor(sum_54, squeeze_79);  sum_54 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_707, mul_217, primals_162, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_707 = mul_217 = primals_162 = None
    getitem_206: "f32[8, 480, 14, 14]" = convolution_backward_36[0]
    getitem_207: "f32[112, 480, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_709: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_206, mul_215);  mul_215 = None
    mul_710: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_206, sigmoid_35);  getitem_206 = sigmoid_35 = None
    sum_55: "f32[8, 480, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_709, [2, 3], True);  mul_709 = None
    alias_23: "f32[8, 480, 1, 1]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    sub_170: "f32[8, 480, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_23)
    mul_711: "f32[8, 480, 1, 1]" = torch.ops.aten.mul.Tensor(alias_23, sub_170);  alias_23 = sub_170 = None
    mul_712: "f32[8, 480, 1, 1]" = torch.ops.aten.mul.Tensor(sum_55, mul_711);  sum_55 = mul_711 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_712, mul_216, primals_160, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_712 = mul_216 = primals_160 = None
    getitem_209: "f32[8, 20, 1, 1]" = convolution_backward_37[0]
    getitem_210: "f32[480, 20, 1, 1]" = convolution_backward_37[1]
    getitem_211: "f32[480]" = convolution_backward_37[2];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_87: "f32[8, 20, 1, 1]" = torch.ops.aten.sigmoid.default(clone_26)
    full_22: "f32[8, 20, 1, 1]" = torch.ops.aten.full.default([8, 20, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_171: "f32[8, 20, 1, 1]" = torch.ops.aten.sub.Tensor(full_22, sigmoid_87);  full_22 = None
    mul_713: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(clone_26, sub_171);  clone_26 = sub_171 = None
    add_288: "f32[8, 20, 1, 1]" = torch.ops.aten.add.Scalar(mul_713, 1);  mul_713 = None
    mul_714: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_87, add_288);  sigmoid_87 = add_288 = None
    mul_715: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_209, mul_714);  getitem_209 = mul_714 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_715, mean_8, primals_158, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_715 = mean_8 = primals_158 = None
    getitem_212: "f32[8, 480, 1, 1]" = convolution_backward_38[0]
    getitem_213: "f32[20, 480, 1, 1]" = convolution_backward_38[1]
    getitem_214: "f32[20]" = convolution_backward_38[2];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_8: "f32[8, 480, 14, 14]" = torch.ops.aten.expand.default(getitem_212, [8, 480, 14, 14]);  getitem_212 = None
    div_8: "f32[8, 480, 14, 14]" = torch.ops.aten.div.Scalar(expand_8, 196);  expand_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_289: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_710, div_8);  mul_710 = div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_88: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(clone_25)
    full_23: "f32[8, 480, 14, 14]" = torch.ops.aten.full.default([8, 480, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_172: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(full_23, sigmoid_88);  full_23 = None
    mul_716: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(clone_25, sub_172);  clone_25 = sub_172 = None
    add_290: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Scalar(mul_716, 1);  mul_716 = None
    mul_717: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_88, add_290);  sigmoid_88 = add_290 = None
    mul_718: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_289, mul_717);  add_289 = mul_717 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_472: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_473: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, 2);  unsqueeze_472 = None
    unsqueeze_474: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 3);  unsqueeze_473 = None
    sum_56: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_718, [0, 2, 3])
    sub_173: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_474)
    mul_719: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_718, sub_173);  sub_173 = None
    sum_57: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_719, [0, 2, 3]);  mul_719 = None
    mul_720: "f32[480]" = torch.ops.aten.mul.Tensor(sum_56, 0.0006377551020408163)
    unsqueeze_475: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_720, 0);  mul_720 = None
    unsqueeze_476: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 2);  unsqueeze_475 = None
    unsqueeze_477: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 3);  unsqueeze_476 = None
    mul_721: "f32[480]" = torch.ops.aten.mul.Tensor(sum_57, 0.0006377551020408163)
    mul_722: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_723: "f32[480]" = torch.ops.aten.mul.Tensor(mul_721, mul_722);  mul_721 = mul_722 = None
    unsqueeze_478: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_723, 0);  mul_723 = None
    unsqueeze_479: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 2);  unsqueeze_478 = None
    unsqueeze_480: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 3);  unsqueeze_479 = None
    mul_724: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_55);  primals_55 = None
    unsqueeze_481: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_724, 0);  mul_724 = None
    unsqueeze_482: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 2);  unsqueeze_481 = None
    unsqueeze_483: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 3);  unsqueeze_482 = None
    sub_174: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_474);  convolution_41 = unsqueeze_474 = None
    mul_725: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_480);  sub_174 = unsqueeze_480 = None
    sub_175: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(mul_718, mul_725);  mul_718 = mul_725 = None
    sub_176: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_175, unsqueeze_477);  sub_175 = unsqueeze_477 = None
    mul_726: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_483);  sub_176 = unsqueeze_483 = None
    mul_727: "f32[480]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_76);  sum_57 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_726, mul_207, primals_157, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_726 = mul_207 = primals_157 = None
    getitem_215: "f32[8, 480, 14, 14]" = convolution_backward_39[0]
    getitem_216: "f32[480, 1, 5, 5]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_89: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(clone_24)
    full_24: "f32[8, 480, 14, 14]" = torch.ops.aten.full.default([8, 480, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_177: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(full_24, sigmoid_89);  full_24 = None
    mul_728: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(clone_24, sub_177);  clone_24 = sub_177 = None
    add_291: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Scalar(mul_728, 1);  mul_728 = None
    mul_729: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_89, add_291);  sigmoid_89 = add_291 = None
    mul_730: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_215, mul_729);  getitem_215 = mul_729 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_484: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_485: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 2);  unsqueeze_484 = None
    unsqueeze_486: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 3);  unsqueeze_485 = None
    sum_58: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_730, [0, 2, 3])
    sub_178: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_486)
    mul_731: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_730, sub_178);  sub_178 = None
    sum_59: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_731, [0, 2, 3]);  mul_731 = None
    mul_732: "f32[480]" = torch.ops.aten.mul.Tensor(sum_58, 0.0006377551020408163)
    unsqueeze_487: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_732, 0);  mul_732 = None
    unsqueeze_488: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
    unsqueeze_489: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 3);  unsqueeze_488 = None
    mul_733: "f32[480]" = torch.ops.aten.mul.Tensor(sum_59, 0.0006377551020408163)
    mul_734: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_735: "f32[480]" = torch.ops.aten.mul.Tensor(mul_733, mul_734);  mul_733 = mul_734 = None
    unsqueeze_490: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_735, 0);  mul_735 = None
    unsqueeze_491: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 2);  unsqueeze_490 = None
    unsqueeze_492: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 3);  unsqueeze_491 = None
    mul_736: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_53);  primals_53 = None
    unsqueeze_493: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_736, 0);  mul_736 = None
    unsqueeze_494: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
    unsqueeze_495: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 3);  unsqueeze_494 = None
    sub_179: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_486);  convolution_40 = unsqueeze_486 = None
    mul_737: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_492);  sub_179 = unsqueeze_492 = None
    sub_180: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(mul_730, mul_737);  mul_730 = mul_737 = None
    sub_181: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_180, unsqueeze_489);  sub_180 = unsqueeze_489 = None
    mul_738: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_495);  sub_181 = unsqueeze_495 = None
    mul_739: "f32[480]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_73);  sum_59 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_738, add_123, primals_156, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_738 = add_123 = primals_156 = None
    getitem_218: "f32[8, 80, 14, 14]" = convolution_backward_40[0]
    getitem_219: "f32[480, 80, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_496: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_497: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 2);  unsqueeze_496 = None
    unsqueeze_498: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 3);  unsqueeze_497 = None
    sum_60: "f32[80]" = torch.ops.aten.sum.dim_IntList(getitem_218, [0, 2, 3])
    sub_182: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_498)
    mul_740: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_218, sub_182);  sub_182 = None
    sum_61: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_740, [0, 2, 3]);  mul_740 = None
    mul_741: "f32[80]" = torch.ops.aten.mul.Tensor(sum_60, 0.0006377551020408163)
    unsqueeze_499: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_741, 0);  mul_741 = None
    unsqueeze_500: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 2);  unsqueeze_499 = None
    unsqueeze_501: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 3);  unsqueeze_500 = None
    mul_742: "f32[80]" = torch.ops.aten.mul.Tensor(sum_61, 0.0006377551020408163)
    mul_743: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_744: "f32[80]" = torch.ops.aten.mul.Tensor(mul_742, mul_743);  mul_742 = mul_743 = None
    unsqueeze_502: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_744, 0);  mul_744 = None
    unsqueeze_503: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 2);  unsqueeze_502 = None
    unsqueeze_504: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 3);  unsqueeze_503 = None
    mul_745: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_51);  primals_51 = None
    unsqueeze_505: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_745, 0);  mul_745 = None
    unsqueeze_506: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 2);  unsqueeze_505 = None
    unsqueeze_507: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 3);  unsqueeze_506 = None
    sub_183: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_498);  convolution_39 = unsqueeze_498 = None
    mul_746: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_504);  sub_183 = unsqueeze_504 = None
    sub_184: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_218, mul_746);  mul_746 = None
    sub_185: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(sub_184, unsqueeze_501);  sub_184 = unsqueeze_501 = None
    mul_747: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_507);  sub_185 = unsqueeze_507 = None
    mul_748: "f32[80]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_70);  sum_61 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_747, mul_192, primals_155, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_747 = mul_192 = primals_155 = None
    getitem_221: "f32[8, 480, 14, 14]" = convolution_backward_41[0]
    getitem_222: "f32[80, 480, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_749: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_221, mul_190);  mul_190 = None
    mul_750: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_221, sigmoid_31);  getitem_221 = sigmoid_31 = None
    sum_62: "f32[8, 480, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_749, [2, 3], True);  mul_749 = None
    alias_24: "f32[8, 480, 1, 1]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    sub_186: "f32[8, 480, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_24)
    mul_751: "f32[8, 480, 1, 1]" = torch.ops.aten.mul.Tensor(alias_24, sub_186);  alias_24 = sub_186 = None
    mul_752: "f32[8, 480, 1, 1]" = torch.ops.aten.mul.Tensor(sum_62, mul_751);  sum_62 = mul_751 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_752, mul_191, primals_153, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_752 = mul_191 = primals_153 = None
    getitem_224: "f32[8, 20, 1, 1]" = convolution_backward_42[0]
    getitem_225: "f32[480, 20, 1, 1]" = convolution_backward_42[1]
    getitem_226: "f32[480]" = convolution_backward_42[2];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_90: "f32[8, 20, 1, 1]" = torch.ops.aten.sigmoid.default(clone_23)
    full_25: "f32[8, 20, 1, 1]" = torch.ops.aten.full.default([8, 20, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_187: "f32[8, 20, 1, 1]" = torch.ops.aten.sub.Tensor(full_25, sigmoid_90);  full_25 = None
    mul_753: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(clone_23, sub_187);  clone_23 = sub_187 = None
    add_292: "f32[8, 20, 1, 1]" = torch.ops.aten.add.Scalar(mul_753, 1);  mul_753 = None
    mul_754: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_90, add_292);  sigmoid_90 = add_292 = None
    mul_755: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_224, mul_754);  getitem_224 = mul_754 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_755, mean_7, primals_151, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_755 = mean_7 = primals_151 = None
    getitem_227: "f32[8, 480, 1, 1]" = convolution_backward_43[0]
    getitem_228: "f32[20, 480, 1, 1]" = convolution_backward_43[1]
    getitem_229: "f32[20]" = convolution_backward_43[2];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_9: "f32[8, 480, 14, 14]" = torch.ops.aten.expand.default(getitem_227, [8, 480, 14, 14]);  getitem_227 = None
    div_9: "f32[8, 480, 14, 14]" = torch.ops.aten.div.Scalar(expand_9, 196);  expand_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_293: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_750, div_9);  mul_750 = div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_91: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(clone_22)
    full_26: "f32[8, 480, 14, 14]" = torch.ops.aten.full.default([8, 480, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_188: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(full_26, sigmoid_91);  full_26 = None
    mul_756: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(clone_22, sub_188);  clone_22 = sub_188 = None
    add_294: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Scalar(mul_756, 1);  mul_756 = None
    mul_757: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_91, add_294);  sigmoid_91 = add_294 = None
    mul_758: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_293, mul_757);  add_293 = mul_757 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_508: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_509: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, 2);  unsqueeze_508 = None
    unsqueeze_510: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 3);  unsqueeze_509 = None
    sum_63: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_758, [0, 2, 3])
    sub_189: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_510)
    mul_759: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_758, sub_189);  sub_189 = None
    sum_64: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_759, [0, 2, 3]);  mul_759 = None
    mul_760: "f32[480]" = torch.ops.aten.mul.Tensor(sum_63, 0.0006377551020408163)
    unsqueeze_511: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_760, 0);  mul_760 = None
    unsqueeze_512: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 2);  unsqueeze_511 = None
    unsqueeze_513: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 3);  unsqueeze_512 = None
    mul_761: "f32[480]" = torch.ops.aten.mul.Tensor(sum_64, 0.0006377551020408163)
    mul_762: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_763: "f32[480]" = torch.ops.aten.mul.Tensor(mul_761, mul_762);  mul_761 = mul_762 = None
    unsqueeze_514: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_763, 0);  mul_763 = None
    unsqueeze_515: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 2);  unsqueeze_514 = None
    unsqueeze_516: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 3);  unsqueeze_515 = None
    mul_764: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_49);  primals_49 = None
    unsqueeze_517: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_764, 0);  mul_764 = None
    unsqueeze_518: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 2);  unsqueeze_517 = None
    unsqueeze_519: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 3);  unsqueeze_518 = None
    sub_190: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_510);  convolution_36 = unsqueeze_510 = None
    mul_765: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_516);  sub_190 = unsqueeze_516 = None
    sub_191: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(mul_758, mul_765);  mul_758 = mul_765 = None
    sub_192: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_191, unsqueeze_513);  sub_191 = unsqueeze_513 = None
    mul_766: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_519);  sub_192 = unsqueeze_519 = None
    mul_767: "f32[480]" = torch.ops.aten.mul.Tensor(sum_64, squeeze_67);  sum_64 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_766, mul_182, primals_150, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_766 = mul_182 = primals_150 = None
    getitem_230: "f32[8, 480, 14, 14]" = convolution_backward_44[0]
    getitem_231: "f32[480, 1, 3, 3]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_92: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(clone_21)
    full_27: "f32[8, 480, 14, 14]" = torch.ops.aten.full.default([8, 480, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_193: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(full_27, sigmoid_92);  full_27 = None
    mul_768: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(clone_21, sub_193);  clone_21 = sub_193 = None
    add_295: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Scalar(mul_768, 1);  mul_768 = None
    mul_769: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_92, add_295);  sigmoid_92 = add_295 = None
    mul_770: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_230, mul_769);  getitem_230 = mul_769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_520: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_521: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 2);  unsqueeze_520 = None
    unsqueeze_522: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 3);  unsqueeze_521 = None
    sum_65: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_770, [0, 2, 3])
    sub_194: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_522)
    mul_771: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_770, sub_194);  sub_194 = None
    sum_66: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_771, [0, 2, 3]);  mul_771 = None
    mul_772: "f32[480]" = torch.ops.aten.mul.Tensor(sum_65, 0.0006377551020408163)
    unsqueeze_523: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_772, 0);  mul_772 = None
    unsqueeze_524: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 2);  unsqueeze_523 = None
    unsqueeze_525: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 3);  unsqueeze_524 = None
    mul_773: "f32[480]" = torch.ops.aten.mul.Tensor(sum_66, 0.0006377551020408163)
    mul_774: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_775: "f32[480]" = torch.ops.aten.mul.Tensor(mul_773, mul_774);  mul_773 = mul_774 = None
    unsqueeze_526: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_775, 0);  mul_775 = None
    unsqueeze_527: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 2);  unsqueeze_526 = None
    unsqueeze_528: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 3);  unsqueeze_527 = None
    mul_776: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_47);  primals_47 = None
    unsqueeze_529: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_776, 0);  mul_776 = None
    unsqueeze_530: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 2);  unsqueeze_529 = None
    unsqueeze_531: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 3);  unsqueeze_530 = None
    sub_195: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_522);  convolution_35 = unsqueeze_522 = None
    mul_777: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_195, unsqueeze_528);  sub_195 = unsqueeze_528 = None
    sub_196: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(mul_770, mul_777);  mul_770 = mul_777 = None
    sub_197: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_196, unsqueeze_525);  sub_196 = unsqueeze_525 = None
    mul_778: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_531);  sub_197 = unsqueeze_531 = None
    mul_779: "f32[480]" = torch.ops.aten.mul.Tensor(sum_66, squeeze_64);  sum_66 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_778, add_107, primals_149, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_778 = add_107 = primals_149 = None
    getitem_233: "f32[8, 80, 14, 14]" = convolution_backward_45[0]
    getitem_234: "f32[480, 80, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_296: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(getitem_218, getitem_233);  getitem_218 = getitem_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_532: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_533: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 2);  unsqueeze_532 = None
    unsqueeze_534: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 3);  unsqueeze_533 = None
    sum_67: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_296, [0, 2, 3])
    sub_198: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_534)
    mul_780: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_296, sub_198);  sub_198 = None
    sum_68: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_780, [0, 2, 3]);  mul_780 = None
    mul_781: "f32[80]" = torch.ops.aten.mul.Tensor(sum_67, 0.0006377551020408163)
    unsqueeze_535: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_781, 0);  mul_781 = None
    unsqueeze_536: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 2);  unsqueeze_535 = None
    unsqueeze_537: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 3);  unsqueeze_536 = None
    mul_782: "f32[80]" = torch.ops.aten.mul.Tensor(sum_68, 0.0006377551020408163)
    mul_783: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_784: "f32[80]" = torch.ops.aten.mul.Tensor(mul_782, mul_783);  mul_782 = mul_783 = None
    unsqueeze_538: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_784, 0);  mul_784 = None
    unsqueeze_539: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 2);  unsqueeze_538 = None
    unsqueeze_540: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 3);  unsqueeze_539 = None
    mul_785: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_45);  primals_45 = None
    unsqueeze_541: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_785, 0);  mul_785 = None
    unsqueeze_542: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 2);  unsqueeze_541 = None
    unsqueeze_543: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 3);  unsqueeze_542 = None
    sub_199: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_534);  convolution_34 = unsqueeze_534 = None
    mul_786: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_540);  sub_199 = unsqueeze_540 = None
    sub_200: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(add_296, mul_786);  mul_786 = None
    sub_201: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(sub_200, unsqueeze_537);  sub_200 = unsqueeze_537 = None
    mul_787: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_543);  sub_201 = unsqueeze_543 = None
    mul_788: "f32[80]" = torch.ops.aten.mul.Tensor(sum_68, squeeze_61);  sum_68 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_787, mul_167, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_787 = mul_167 = primals_148 = None
    getitem_236: "f32[8, 480, 14, 14]" = convolution_backward_46[0]
    getitem_237: "f32[80, 480, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_789: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_236, mul_165);  mul_165 = None
    mul_790: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_236, sigmoid_27);  getitem_236 = sigmoid_27 = None
    sum_69: "f32[8, 480, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_789, [2, 3], True);  mul_789 = None
    alias_25: "f32[8, 480, 1, 1]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    sub_202: "f32[8, 480, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_25)
    mul_791: "f32[8, 480, 1, 1]" = torch.ops.aten.mul.Tensor(alias_25, sub_202);  alias_25 = sub_202 = None
    mul_792: "f32[8, 480, 1, 1]" = torch.ops.aten.mul.Tensor(sum_69, mul_791);  sum_69 = mul_791 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_792, mul_166, primals_146, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_792 = mul_166 = primals_146 = None
    getitem_239: "f32[8, 20, 1, 1]" = convolution_backward_47[0]
    getitem_240: "f32[480, 20, 1, 1]" = convolution_backward_47[1]
    getitem_241: "f32[480]" = convolution_backward_47[2];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_93: "f32[8, 20, 1, 1]" = torch.ops.aten.sigmoid.default(clone_20)
    full_28: "f32[8, 20, 1, 1]" = torch.ops.aten.full.default([8, 20, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_203: "f32[8, 20, 1, 1]" = torch.ops.aten.sub.Tensor(full_28, sigmoid_93);  full_28 = None
    mul_793: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(clone_20, sub_203);  clone_20 = sub_203 = None
    add_297: "f32[8, 20, 1, 1]" = torch.ops.aten.add.Scalar(mul_793, 1);  mul_793 = None
    mul_794: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_93, add_297);  sigmoid_93 = add_297 = None
    mul_795: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_239, mul_794);  getitem_239 = mul_794 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_795, mean_6, primals_144, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_795 = mean_6 = primals_144 = None
    getitem_242: "f32[8, 480, 1, 1]" = convolution_backward_48[0]
    getitem_243: "f32[20, 480, 1, 1]" = convolution_backward_48[1]
    getitem_244: "f32[20]" = convolution_backward_48[2];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_10: "f32[8, 480, 14, 14]" = torch.ops.aten.expand.default(getitem_242, [8, 480, 14, 14]);  getitem_242 = None
    div_10: "f32[8, 480, 14, 14]" = torch.ops.aten.div.Scalar(expand_10, 196);  expand_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_298: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_790, div_10);  mul_790 = div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_94: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(clone_19)
    full_29: "f32[8, 480, 14, 14]" = torch.ops.aten.full.default([8, 480, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_204: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(full_29, sigmoid_94);  full_29 = None
    mul_796: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(clone_19, sub_204);  clone_19 = sub_204 = None
    add_299: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Scalar(mul_796, 1);  mul_796 = None
    mul_797: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_94, add_299);  sigmoid_94 = add_299 = None
    mul_798: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_298, mul_797);  add_298 = mul_797 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_544: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_545: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, 2);  unsqueeze_544 = None
    unsqueeze_546: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 3);  unsqueeze_545 = None
    sum_70: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_798, [0, 2, 3])
    sub_205: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_546)
    mul_799: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_798, sub_205);  sub_205 = None
    sum_71: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_799, [0, 2, 3]);  mul_799 = None
    mul_800: "f32[480]" = torch.ops.aten.mul.Tensor(sum_70, 0.0006377551020408163)
    unsqueeze_547: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_800, 0);  mul_800 = None
    unsqueeze_548: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_547, 2);  unsqueeze_547 = None
    unsqueeze_549: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 3);  unsqueeze_548 = None
    mul_801: "f32[480]" = torch.ops.aten.mul.Tensor(sum_71, 0.0006377551020408163)
    mul_802: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_803: "f32[480]" = torch.ops.aten.mul.Tensor(mul_801, mul_802);  mul_801 = mul_802 = None
    unsqueeze_550: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_803, 0);  mul_803 = None
    unsqueeze_551: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 2);  unsqueeze_550 = None
    unsqueeze_552: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 3);  unsqueeze_551 = None
    mul_804: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_43);  primals_43 = None
    unsqueeze_553: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_804, 0);  mul_804 = None
    unsqueeze_554: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 2);  unsqueeze_553 = None
    unsqueeze_555: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 3);  unsqueeze_554 = None
    sub_206: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_546);  convolution_31 = unsqueeze_546 = None
    mul_805: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_552);  sub_206 = unsqueeze_552 = None
    sub_207: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(mul_798, mul_805);  mul_798 = mul_805 = None
    sub_208: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_207, unsqueeze_549);  sub_207 = unsqueeze_549 = None
    mul_806: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_555);  sub_208 = unsqueeze_555 = None
    mul_807: "f32[480]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_58);  sum_71 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_806, mul_157, primals_143, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_806 = mul_157 = primals_143 = None
    getitem_245: "f32[8, 480, 14, 14]" = convolution_backward_49[0]
    getitem_246: "f32[480, 1, 3, 3]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_95: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(clone_18)
    full_30: "f32[8, 480, 14, 14]" = torch.ops.aten.full.default([8, 480, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_209: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(full_30, sigmoid_95);  full_30 = None
    mul_808: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(clone_18, sub_209);  clone_18 = sub_209 = None
    add_300: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Scalar(mul_808, 1);  mul_808 = None
    mul_809: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_95, add_300);  sigmoid_95 = add_300 = None
    mul_810: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_245, mul_809);  getitem_245 = mul_809 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_556: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_557: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 2);  unsqueeze_556 = None
    unsqueeze_558: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 3);  unsqueeze_557 = None
    sum_72: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_810, [0, 2, 3])
    sub_210: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_558)
    mul_811: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_810, sub_210);  sub_210 = None
    sum_73: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_811, [0, 2, 3]);  mul_811 = None
    mul_812: "f32[480]" = torch.ops.aten.mul.Tensor(sum_72, 0.0006377551020408163)
    unsqueeze_559: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_812, 0);  mul_812 = None
    unsqueeze_560: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 2);  unsqueeze_559 = None
    unsqueeze_561: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 3);  unsqueeze_560 = None
    mul_813: "f32[480]" = torch.ops.aten.mul.Tensor(sum_73, 0.0006377551020408163)
    mul_814: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_815: "f32[480]" = torch.ops.aten.mul.Tensor(mul_813, mul_814);  mul_813 = mul_814 = None
    unsqueeze_562: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_815, 0);  mul_815 = None
    unsqueeze_563: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 2);  unsqueeze_562 = None
    unsqueeze_564: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 3);  unsqueeze_563 = None
    mul_816: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_41);  primals_41 = None
    unsqueeze_565: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_816, 0);  mul_816 = None
    unsqueeze_566: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 2);  unsqueeze_565 = None
    unsqueeze_567: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 3);  unsqueeze_566 = None
    sub_211: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_558);  convolution_30 = unsqueeze_558 = None
    mul_817: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_564);  sub_211 = unsqueeze_564 = None
    sub_212: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(mul_810, mul_817);  mul_810 = mul_817 = None
    sub_213: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_212, unsqueeze_561);  sub_212 = unsqueeze_561 = None
    mul_818: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_567);  sub_213 = unsqueeze_567 = None
    mul_819: "f32[480]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_55);  sum_73 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_818, add_91, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_818 = add_91 = primals_142 = None
    getitem_248: "f32[8, 80, 14, 14]" = convolution_backward_50[0]
    getitem_249: "f32[480, 80, 1, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_301: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_296, getitem_248);  add_296 = getitem_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_568: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_569: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, 2);  unsqueeze_568 = None
    unsqueeze_570: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 3);  unsqueeze_569 = None
    sum_74: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_301, [0, 2, 3])
    sub_214: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_570)
    mul_820: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_301, sub_214);  sub_214 = None
    sum_75: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_820, [0, 2, 3]);  mul_820 = None
    mul_821: "f32[80]" = torch.ops.aten.mul.Tensor(sum_74, 0.0006377551020408163)
    unsqueeze_571: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_821, 0);  mul_821 = None
    unsqueeze_572: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_571, 2);  unsqueeze_571 = None
    unsqueeze_573: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 3);  unsqueeze_572 = None
    mul_822: "f32[80]" = torch.ops.aten.mul.Tensor(sum_75, 0.0006377551020408163)
    mul_823: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_824: "f32[80]" = torch.ops.aten.mul.Tensor(mul_822, mul_823);  mul_822 = mul_823 = None
    unsqueeze_574: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_824, 0);  mul_824 = None
    unsqueeze_575: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, 2);  unsqueeze_574 = None
    unsqueeze_576: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 3);  unsqueeze_575 = None
    mul_825: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_39);  primals_39 = None
    unsqueeze_577: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_825, 0);  mul_825 = None
    unsqueeze_578: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 2);  unsqueeze_577 = None
    unsqueeze_579: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 3);  unsqueeze_578 = None
    sub_215: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_570);  convolution_29 = unsqueeze_570 = None
    mul_826: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_215, unsqueeze_576);  sub_215 = unsqueeze_576 = None
    sub_216: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(add_301, mul_826);  add_301 = mul_826 = None
    sub_217: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(sub_216, unsqueeze_573);  sub_216 = unsqueeze_573 = None
    mul_827: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_579);  sub_217 = unsqueeze_579 = None
    mul_828: "f32[80]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_52);  sum_75 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_827, mul_142, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_827 = mul_142 = primals_141 = None
    getitem_251: "f32[8, 240, 14, 14]" = convolution_backward_51[0]
    getitem_252: "f32[80, 240, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_829: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_251, mul_140);  mul_140 = None
    mul_830: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_251, sigmoid_23);  getitem_251 = sigmoid_23 = None
    sum_76: "f32[8, 240, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_829, [2, 3], True);  mul_829 = None
    alias_26: "f32[8, 240, 1, 1]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    sub_218: "f32[8, 240, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_26)
    mul_831: "f32[8, 240, 1, 1]" = torch.ops.aten.mul.Tensor(alias_26, sub_218);  alias_26 = sub_218 = None
    mul_832: "f32[8, 240, 1, 1]" = torch.ops.aten.mul.Tensor(sum_76, mul_831);  sum_76 = mul_831 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_832, mul_141, primals_139, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_832 = mul_141 = primals_139 = None
    getitem_254: "f32[8, 10, 1, 1]" = convolution_backward_52[0]
    getitem_255: "f32[240, 10, 1, 1]" = convolution_backward_52[1]
    getitem_256: "f32[240]" = convolution_backward_52[2];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_96: "f32[8, 10, 1, 1]" = torch.ops.aten.sigmoid.default(clone_17)
    full_31: "f32[8, 10, 1, 1]" = torch.ops.aten.full.default([8, 10, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_219: "f32[8, 10, 1, 1]" = torch.ops.aten.sub.Tensor(full_31, sigmoid_96);  full_31 = None
    mul_833: "f32[8, 10, 1, 1]" = torch.ops.aten.mul.Tensor(clone_17, sub_219);  clone_17 = sub_219 = None
    add_302: "f32[8, 10, 1, 1]" = torch.ops.aten.add.Scalar(mul_833, 1);  mul_833 = None
    mul_834: "f32[8, 10, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_96, add_302);  sigmoid_96 = add_302 = None
    mul_835: "f32[8, 10, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_254, mul_834);  getitem_254 = mul_834 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_835, mean_5, primals_137, [10], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_835 = mean_5 = primals_137 = None
    getitem_257: "f32[8, 240, 1, 1]" = convolution_backward_53[0]
    getitem_258: "f32[10, 240, 1, 1]" = convolution_backward_53[1]
    getitem_259: "f32[10]" = convolution_backward_53[2];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_11: "f32[8, 240, 14, 14]" = torch.ops.aten.expand.default(getitem_257, [8, 240, 14, 14]);  getitem_257 = None
    div_11: "f32[8, 240, 14, 14]" = torch.ops.aten.div.Scalar(expand_11, 196);  expand_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_303: "f32[8, 240, 14, 14]" = torch.ops.aten.add.Tensor(mul_830, div_11);  mul_830 = div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_97: "f32[8, 240, 14, 14]" = torch.ops.aten.sigmoid.default(clone_16)
    full_32: "f32[8, 240, 14, 14]" = torch.ops.aten.full.default([8, 240, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_220: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(full_32, sigmoid_97);  full_32 = None
    mul_836: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(clone_16, sub_220);  clone_16 = sub_220 = None
    add_304: "f32[8, 240, 14, 14]" = torch.ops.aten.add.Scalar(mul_836, 1);  mul_836 = None
    mul_837: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_97, add_304);  sigmoid_97 = add_304 = None
    mul_838: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(add_303, mul_837);  add_303 = mul_837 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_580: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_581: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 2);  unsqueeze_580 = None
    unsqueeze_582: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 3);  unsqueeze_581 = None
    sum_77: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_838, [0, 2, 3])
    sub_221: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_582)
    mul_839: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_838, sub_221);  sub_221 = None
    sum_78: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_839, [0, 2, 3]);  mul_839 = None
    mul_840: "f32[240]" = torch.ops.aten.mul.Tensor(sum_77, 0.0006377551020408163)
    unsqueeze_583: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_840, 0);  mul_840 = None
    unsqueeze_584: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_583, 2);  unsqueeze_583 = None
    unsqueeze_585: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 3);  unsqueeze_584 = None
    mul_841: "f32[240]" = torch.ops.aten.mul.Tensor(sum_78, 0.0006377551020408163)
    mul_842: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_843: "f32[240]" = torch.ops.aten.mul.Tensor(mul_841, mul_842);  mul_841 = mul_842 = None
    unsqueeze_586: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_843, 0);  mul_843 = None
    unsqueeze_587: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 2);  unsqueeze_586 = None
    unsqueeze_588: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 3);  unsqueeze_587 = None
    mul_844: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_37);  primals_37 = None
    unsqueeze_589: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_844, 0);  mul_844 = None
    unsqueeze_590: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 2);  unsqueeze_589 = None
    unsqueeze_591: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 3);  unsqueeze_590 = None
    sub_222: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_582);  convolution_26 = unsqueeze_582 = None
    mul_845: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_588);  sub_222 = unsqueeze_588 = None
    sub_223: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(mul_838, mul_845);  mul_838 = mul_845 = None
    sub_224: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(sub_223, unsqueeze_585);  sub_223 = unsqueeze_585 = None
    mul_846: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_591);  sub_224 = unsqueeze_591 = None
    mul_847: "f32[240]" = torch.ops.aten.mul.Tensor(sum_78, squeeze_49);  sum_78 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_846, constant_pad_nd_3, primals_36, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 240, [True, True, False]);  mul_846 = constant_pad_nd_3 = primals_36 = None
    getitem_260: "f32[8, 240, 29, 29]" = convolution_backward_54[0]
    getitem_261: "f32[240, 1, 3, 3]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_6: "f32[8, 240, 28, 28]" = torch.ops.aten.constant_pad_nd.default(getitem_260, [0, -1, 0, -1]);  getitem_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_98: "f32[8, 240, 28, 28]" = torch.ops.aten.sigmoid.default(clone_15)
    full_33: "f32[8, 240, 28, 28]" = torch.ops.aten.full.default([8, 240, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_225: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(full_33, sigmoid_98);  full_33 = None
    mul_848: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(clone_15, sub_225);  clone_15 = sub_225 = None
    add_305: "f32[8, 240, 28, 28]" = torch.ops.aten.add.Scalar(mul_848, 1);  mul_848 = None
    mul_849: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_98, add_305);  sigmoid_98 = add_305 = None
    mul_850: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(constant_pad_nd_6, mul_849);  constant_pad_nd_6 = mul_849 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_592: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_593: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 2);  unsqueeze_592 = None
    unsqueeze_594: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 3);  unsqueeze_593 = None
    sum_79: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_850, [0, 2, 3])
    sub_226: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_594)
    mul_851: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_850, sub_226);  sub_226 = None
    sum_80: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_851, [0, 2, 3]);  mul_851 = None
    mul_852: "f32[240]" = torch.ops.aten.mul.Tensor(sum_79, 0.00015943877551020407)
    unsqueeze_595: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_852, 0);  mul_852 = None
    unsqueeze_596: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 2);  unsqueeze_595 = None
    unsqueeze_597: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 3);  unsqueeze_596 = None
    mul_853: "f32[240]" = torch.ops.aten.mul.Tensor(sum_80, 0.00015943877551020407)
    mul_854: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_855: "f32[240]" = torch.ops.aten.mul.Tensor(mul_853, mul_854);  mul_853 = mul_854 = None
    unsqueeze_598: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_855, 0);  mul_855 = None
    unsqueeze_599: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 2);  unsqueeze_598 = None
    unsqueeze_600: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 3);  unsqueeze_599 = None
    mul_856: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_34);  primals_34 = None
    unsqueeze_601: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_856, 0);  mul_856 = None
    unsqueeze_602: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 2);  unsqueeze_601 = None
    unsqueeze_603: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 3);  unsqueeze_602 = None
    sub_227: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_594);  convolution_25 = unsqueeze_594 = None
    mul_857: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_600);  sub_227 = unsqueeze_600 = None
    sub_228: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(mul_850, mul_857);  mul_850 = mul_857 = None
    sub_229: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(sub_228, unsqueeze_597);  sub_228 = unsqueeze_597 = None
    mul_858: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_603);  sub_229 = unsqueeze_603 = None
    mul_859: "f32[240]" = torch.ops.aten.mul.Tensor(sum_80, squeeze_46);  sum_80 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_858, add_76, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_858 = add_76 = primals_136 = None
    getitem_263: "f32[8, 40, 28, 28]" = convolution_backward_55[0]
    getitem_264: "f32[240, 40, 1, 1]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_604: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_605: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, 2);  unsqueeze_604 = None
    unsqueeze_606: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 3);  unsqueeze_605 = None
    sum_81: "f32[40]" = torch.ops.aten.sum.dim_IntList(getitem_263, [0, 2, 3])
    sub_230: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_606)
    mul_860: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_263, sub_230);  sub_230 = None
    sum_82: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_860, [0, 2, 3]);  mul_860 = None
    mul_861: "f32[40]" = torch.ops.aten.mul.Tensor(sum_81, 0.00015943877551020407)
    unsqueeze_607: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_861, 0);  mul_861 = None
    unsqueeze_608: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_607, 2);  unsqueeze_607 = None
    unsqueeze_609: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 3);  unsqueeze_608 = None
    mul_862: "f32[40]" = torch.ops.aten.mul.Tensor(sum_82, 0.00015943877551020407)
    mul_863: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_864: "f32[40]" = torch.ops.aten.mul.Tensor(mul_862, mul_863);  mul_862 = mul_863 = None
    unsqueeze_610: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_864, 0);  mul_864 = None
    unsqueeze_611: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 2);  unsqueeze_610 = None
    unsqueeze_612: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 3);  unsqueeze_611 = None
    mul_865: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_32);  primals_32 = None
    unsqueeze_613: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_865, 0);  mul_865 = None
    unsqueeze_614: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 2);  unsqueeze_613 = None
    unsqueeze_615: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 3);  unsqueeze_614 = None
    sub_231: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_606);  convolution_24 = unsqueeze_606 = None
    mul_866: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_231, unsqueeze_612);  sub_231 = unsqueeze_612 = None
    sub_232: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_263, mul_866);  mul_866 = None
    sub_233: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_232, unsqueeze_609);  sub_232 = unsqueeze_609 = None
    mul_867: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_233, unsqueeze_615);  sub_233 = unsqueeze_615 = None
    mul_868: "f32[40]" = torch.ops.aten.mul.Tensor(sum_82, squeeze_43);  sum_82 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_867, mul_117, primals_135, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_867 = mul_117 = primals_135 = None
    getitem_266: "f32[8, 240, 28, 28]" = convolution_backward_56[0]
    getitem_267: "f32[40, 240, 1, 1]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_869: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_266, mul_115);  mul_115 = None
    mul_870: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_266, sigmoid_19);  getitem_266 = sigmoid_19 = None
    sum_83: "f32[8, 240, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_869, [2, 3], True);  mul_869 = None
    alias_27: "f32[8, 240, 1, 1]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    sub_234: "f32[8, 240, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_27)
    mul_871: "f32[8, 240, 1, 1]" = torch.ops.aten.mul.Tensor(alias_27, sub_234);  alias_27 = sub_234 = None
    mul_872: "f32[8, 240, 1, 1]" = torch.ops.aten.mul.Tensor(sum_83, mul_871);  sum_83 = mul_871 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_872, mul_116, primals_133, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_872 = mul_116 = primals_133 = None
    getitem_269: "f32[8, 10, 1, 1]" = convolution_backward_57[0]
    getitem_270: "f32[240, 10, 1, 1]" = convolution_backward_57[1]
    getitem_271: "f32[240]" = convolution_backward_57[2];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_99: "f32[8, 10, 1, 1]" = torch.ops.aten.sigmoid.default(clone_14)
    full_34: "f32[8, 10, 1, 1]" = torch.ops.aten.full.default([8, 10, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_235: "f32[8, 10, 1, 1]" = torch.ops.aten.sub.Tensor(full_34, sigmoid_99);  full_34 = None
    mul_873: "f32[8, 10, 1, 1]" = torch.ops.aten.mul.Tensor(clone_14, sub_235);  clone_14 = sub_235 = None
    add_306: "f32[8, 10, 1, 1]" = torch.ops.aten.add.Scalar(mul_873, 1);  mul_873 = None
    mul_874: "f32[8, 10, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_99, add_306);  sigmoid_99 = add_306 = None
    mul_875: "f32[8, 10, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_269, mul_874);  getitem_269 = mul_874 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_875, mean_4, primals_131, [10], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_875 = mean_4 = primals_131 = None
    getitem_272: "f32[8, 240, 1, 1]" = convolution_backward_58[0]
    getitem_273: "f32[10, 240, 1, 1]" = convolution_backward_58[1]
    getitem_274: "f32[10]" = convolution_backward_58[2];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_12: "f32[8, 240, 28, 28]" = torch.ops.aten.expand.default(getitem_272, [8, 240, 28, 28]);  getitem_272 = None
    div_12: "f32[8, 240, 28, 28]" = torch.ops.aten.div.Scalar(expand_12, 784);  expand_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_307: "f32[8, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_870, div_12);  mul_870 = div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_100: "f32[8, 240, 28, 28]" = torch.ops.aten.sigmoid.default(clone_13)
    full_35: "f32[8, 240, 28, 28]" = torch.ops.aten.full.default([8, 240, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_236: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(full_35, sigmoid_100);  full_35 = None
    mul_876: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(clone_13, sub_236);  clone_13 = sub_236 = None
    add_308: "f32[8, 240, 28, 28]" = torch.ops.aten.add.Scalar(mul_876, 1);  mul_876 = None
    mul_877: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_100, add_308);  sigmoid_100 = add_308 = None
    mul_878: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_307, mul_877);  add_307 = mul_877 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_616: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_617: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 2);  unsqueeze_616 = None
    unsqueeze_618: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 3);  unsqueeze_617 = None
    sum_84: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_878, [0, 2, 3])
    sub_237: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_618)
    mul_879: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_878, sub_237);  sub_237 = None
    sum_85: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_879, [0, 2, 3]);  mul_879 = None
    mul_880: "f32[240]" = torch.ops.aten.mul.Tensor(sum_84, 0.00015943877551020407)
    unsqueeze_619: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_880, 0);  mul_880 = None
    unsqueeze_620: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_619, 2);  unsqueeze_619 = None
    unsqueeze_621: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 3);  unsqueeze_620 = None
    mul_881: "f32[240]" = torch.ops.aten.mul.Tensor(sum_85, 0.00015943877551020407)
    mul_882: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_883: "f32[240]" = torch.ops.aten.mul.Tensor(mul_881, mul_882);  mul_881 = mul_882 = None
    unsqueeze_622: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_883, 0);  mul_883 = None
    unsqueeze_623: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 2);  unsqueeze_622 = None
    unsqueeze_624: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 3);  unsqueeze_623 = None
    mul_884: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_30);  primals_30 = None
    unsqueeze_625: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_884, 0);  mul_884 = None
    unsqueeze_626: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 2);  unsqueeze_625 = None
    unsqueeze_627: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 3);  unsqueeze_626 = None
    sub_238: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_618);  convolution_21 = unsqueeze_618 = None
    mul_885: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_238, unsqueeze_624);  sub_238 = unsqueeze_624 = None
    sub_239: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(mul_878, mul_885);  mul_878 = mul_885 = None
    sub_240: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(sub_239, unsqueeze_621);  sub_239 = unsqueeze_621 = None
    mul_886: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_627);  sub_240 = unsqueeze_627 = None
    mul_887: "f32[240]" = torch.ops.aten.mul.Tensor(sum_85, squeeze_40);  sum_85 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(mul_886, mul_107, primals_130, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 240, [True, True, False]);  mul_886 = mul_107 = primals_130 = None
    getitem_275: "f32[8, 240, 28, 28]" = convolution_backward_59[0]
    getitem_276: "f32[240, 1, 5, 5]" = convolution_backward_59[1];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_101: "f32[8, 240, 28, 28]" = torch.ops.aten.sigmoid.default(clone_12)
    full_36: "f32[8, 240, 28, 28]" = torch.ops.aten.full.default([8, 240, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_241: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(full_36, sigmoid_101);  full_36 = None
    mul_888: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(clone_12, sub_241);  clone_12 = sub_241 = None
    add_309: "f32[8, 240, 28, 28]" = torch.ops.aten.add.Scalar(mul_888, 1);  mul_888 = None
    mul_889: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_101, add_309);  sigmoid_101 = add_309 = None
    mul_890: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_275, mul_889);  getitem_275 = mul_889 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_628: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_629: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, 2);  unsqueeze_628 = None
    unsqueeze_630: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 3);  unsqueeze_629 = None
    sum_86: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_890, [0, 2, 3])
    sub_242: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_630)
    mul_891: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_890, sub_242);  sub_242 = None
    sum_87: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_891, [0, 2, 3]);  mul_891 = None
    mul_892: "f32[240]" = torch.ops.aten.mul.Tensor(sum_86, 0.00015943877551020407)
    unsqueeze_631: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_892, 0);  mul_892 = None
    unsqueeze_632: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_631, 2);  unsqueeze_631 = None
    unsqueeze_633: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 3);  unsqueeze_632 = None
    mul_893: "f32[240]" = torch.ops.aten.mul.Tensor(sum_87, 0.00015943877551020407)
    mul_894: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_895: "f32[240]" = torch.ops.aten.mul.Tensor(mul_893, mul_894);  mul_893 = mul_894 = None
    unsqueeze_634: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_895, 0);  mul_895 = None
    unsqueeze_635: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, 2);  unsqueeze_634 = None
    unsqueeze_636: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 3);  unsqueeze_635 = None
    mul_896: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_28);  primals_28 = None
    unsqueeze_637: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_896, 0);  mul_896 = None
    unsqueeze_638: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 2);  unsqueeze_637 = None
    unsqueeze_639: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 3);  unsqueeze_638 = None
    sub_243: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_630);  convolution_20 = unsqueeze_630 = None
    mul_897: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_636);  sub_243 = unsqueeze_636 = None
    sub_244: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(mul_890, mul_897);  mul_890 = mul_897 = None
    sub_245: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(sub_244, unsqueeze_633);  sub_244 = unsqueeze_633 = None
    mul_898: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_639);  sub_245 = unsqueeze_639 = None
    mul_899: "f32[240]" = torch.ops.aten.mul.Tensor(sum_87, squeeze_37);  sum_87 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_898, add_60, primals_129, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_898 = add_60 = primals_129 = None
    getitem_278: "f32[8, 40, 28, 28]" = convolution_backward_60[0]
    getitem_279: "f32[240, 40, 1, 1]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_310: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(getitem_263, getitem_278);  getitem_263 = getitem_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_640: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_641: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, 2);  unsqueeze_640 = None
    unsqueeze_642: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 3);  unsqueeze_641 = None
    sum_88: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_310, [0, 2, 3])
    sub_246: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_642)
    mul_900: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_310, sub_246);  sub_246 = None
    sum_89: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_900, [0, 2, 3]);  mul_900 = None
    mul_901: "f32[40]" = torch.ops.aten.mul.Tensor(sum_88, 0.00015943877551020407)
    unsqueeze_643: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_901, 0);  mul_901 = None
    unsqueeze_644: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_643, 2);  unsqueeze_643 = None
    unsqueeze_645: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 3);  unsqueeze_644 = None
    mul_902: "f32[40]" = torch.ops.aten.mul.Tensor(sum_89, 0.00015943877551020407)
    mul_903: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_904: "f32[40]" = torch.ops.aten.mul.Tensor(mul_902, mul_903);  mul_902 = mul_903 = None
    unsqueeze_646: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_904, 0);  mul_904 = None
    unsqueeze_647: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, 2);  unsqueeze_646 = None
    unsqueeze_648: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 3);  unsqueeze_647 = None
    mul_905: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_26);  primals_26 = None
    unsqueeze_649: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_905, 0);  mul_905 = None
    unsqueeze_650: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 2);  unsqueeze_649 = None
    unsqueeze_651: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 3);  unsqueeze_650 = None
    sub_247: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_642);  convolution_19 = unsqueeze_642 = None
    mul_906: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_247, unsqueeze_648);  sub_247 = unsqueeze_648 = None
    sub_248: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(add_310, mul_906);  add_310 = mul_906 = None
    sub_249: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_248, unsqueeze_645);  sub_248 = unsqueeze_645 = None
    mul_907: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_249, unsqueeze_651);  sub_249 = unsqueeze_651 = None
    mul_908: "f32[40]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_34);  sum_89 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_907, mul_92, primals_128, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_907 = mul_92 = primals_128 = None
    getitem_281: "f32[8, 144, 28, 28]" = convolution_backward_61[0]
    getitem_282: "f32[40, 144, 1, 1]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_909: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_281, mul_90);  mul_90 = None
    mul_910: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_281, sigmoid_15);  getitem_281 = sigmoid_15 = None
    sum_90: "f32[8, 144, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_909, [2, 3], True);  mul_909 = None
    alias_28: "f32[8, 144, 1, 1]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    sub_250: "f32[8, 144, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_28)
    mul_911: "f32[8, 144, 1, 1]" = torch.ops.aten.mul.Tensor(alias_28, sub_250);  alias_28 = sub_250 = None
    mul_912: "f32[8, 144, 1, 1]" = torch.ops.aten.mul.Tensor(sum_90, mul_911);  sum_90 = mul_911 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_912, mul_91, primals_126, [144], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_912 = mul_91 = primals_126 = None
    getitem_284: "f32[8, 6, 1, 1]" = convolution_backward_62[0]
    getitem_285: "f32[144, 6, 1, 1]" = convolution_backward_62[1]
    getitem_286: "f32[144]" = convolution_backward_62[2];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_102: "f32[8, 6, 1, 1]" = torch.ops.aten.sigmoid.default(clone_11)
    full_37: "f32[8, 6, 1, 1]" = torch.ops.aten.full.default([8, 6, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_251: "f32[8, 6, 1, 1]" = torch.ops.aten.sub.Tensor(full_37, sigmoid_102);  full_37 = None
    mul_913: "f32[8, 6, 1, 1]" = torch.ops.aten.mul.Tensor(clone_11, sub_251);  clone_11 = sub_251 = None
    add_311: "f32[8, 6, 1, 1]" = torch.ops.aten.add.Scalar(mul_913, 1);  mul_913 = None
    mul_914: "f32[8, 6, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_102, add_311);  sigmoid_102 = add_311 = None
    mul_915: "f32[8, 6, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_284, mul_914);  getitem_284 = mul_914 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_915, mean_3, primals_124, [6], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_915 = mean_3 = primals_124 = None
    getitem_287: "f32[8, 144, 1, 1]" = convolution_backward_63[0]
    getitem_288: "f32[6, 144, 1, 1]" = convolution_backward_63[1]
    getitem_289: "f32[6]" = convolution_backward_63[2];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_13: "f32[8, 144, 28, 28]" = torch.ops.aten.expand.default(getitem_287, [8, 144, 28, 28]);  getitem_287 = None
    div_13: "f32[8, 144, 28, 28]" = torch.ops.aten.div.Scalar(expand_13, 784);  expand_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_312: "f32[8, 144, 28, 28]" = torch.ops.aten.add.Tensor(mul_910, div_13);  mul_910 = div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_103: "f32[8, 144, 28, 28]" = torch.ops.aten.sigmoid.default(clone_10)
    full_38: "f32[8, 144, 28, 28]" = torch.ops.aten.full.default([8, 144, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_252: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(full_38, sigmoid_103);  full_38 = None
    mul_916: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(clone_10, sub_252);  clone_10 = sub_252 = None
    add_313: "f32[8, 144, 28, 28]" = torch.ops.aten.add.Scalar(mul_916, 1);  mul_916 = None
    mul_917: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_103, add_313);  sigmoid_103 = add_313 = None
    mul_918: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(add_312, mul_917);  add_312 = mul_917 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_652: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_653: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, 2);  unsqueeze_652 = None
    unsqueeze_654: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 3);  unsqueeze_653 = None
    sum_91: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_918, [0, 2, 3])
    sub_253: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_654)
    mul_919: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(mul_918, sub_253);  sub_253 = None
    sum_92: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_919, [0, 2, 3]);  mul_919 = None
    mul_920: "f32[144]" = torch.ops.aten.mul.Tensor(sum_91, 0.00015943877551020407)
    unsqueeze_655: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_920, 0);  mul_920 = None
    unsqueeze_656: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_655, 2);  unsqueeze_655 = None
    unsqueeze_657: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 3);  unsqueeze_656 = None
    mul_921: "f32[144]" = torch.ops.aten.mul.Tensor(sum_92, 0.00015943877551020407)
    mul_922: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_923: "f32[144]" = torch.ops.aten.mul.Tensor(mul_921, mul_922);  mul_921 = mul_922 = None
    unsqueeze_658: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_923, 0);  mul_923 = None
    unsqueeze_659: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, 2);  unsqueeze_658 = None
    unsqueeze_660: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 3);  unsqueeze_659 = None
    mul_924: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_24);  primals_24 = None
    unsqueeze_661: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_924, 0);  mul_924 = None
    unsqueeze_662: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 2);  unsqueeze_661 = None
    unsqueeze_663: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 3);  unsqueeze_662 = None
    sub_254: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_654);  convolution_16 = unsqueeze_654 = None
    mul_925: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_254, unsqueeze_660);  sub_254 = unsqueeze_660 = None
    sub_255: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(mul_918, mul_925);  mul_918 = mul_925 = None
    sub_256: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(sub_255, unsqueeze_657);  sub_255 = unsqueeze_657 = None
    mul_926: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_663);  sub_256 = unsqueeze_663 = None
    mul_927: "f32[144]" = torch.ops.aten.mul.Tensor(sum_92, squeeze_31);  sum_92 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(mul_926, constant_pad_nd_2, primals_23, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 144, [True, True, False]);  mul_926 = constant_pad_nd_2 = primals_23 = None
    getitem_290: "f32[8, 144, 59, 59]" = convolution_backward_64[0]
    getitem_291: "f32[144, 1, 5, 5]" = convolution_backward_64[1];  convolution_backward_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_7: "f32[8, 144, 56, 56]" = torch.ops.aten.constant_pad_nd.default(getitem_290, [-1, -2, -1, -2]);  getitem_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_104: "f32[8, 144, 56, 56]" = torch.ops.aten.sigmoid.default(clone_9)
    full_39: "f32[8, 144, 56, 56]" = torch.ops.aten.full.default([8, 144, 56, 56], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_257: "f32[8, 144, 56, 56]" = torch.ops.aten.sub.Tensor(full_39, sigmoid_104);  full_39 = None
    mul_928: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(clone_9, sub_257);  clone_9 = sub_257 = None
    add_314: "f32[8, 144, 56, 56]" = torch.ops.aten.add.Scalar(mul_928, 1);  mul_928 = None
    mul_929: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sigmoid_104, add_314);  sigmoid_104 = add_314 = None
    mul_930: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(constant_pad_nd_7, mul_929);  constant_pad_nd_7 = mul_929 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_664: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_665: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, 2);  unsqueeze_664 = None
    unsqueeze_666: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 3);  unsqueeze_665 = None
    sum_93: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_930, [0, 2, 3])
    sub_258: "f32[8, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_666)
    mul_931: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_930, sub_258);  sub_258 = None
    sum_94: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_931, [0, 2, 3]);  mul_931 = None
    mul_932: "f32[144]" = torch.ops.aten.mul.Tensor(sum_93, 3.985969387755102e-05)
    unsqueeze_667: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_932, 0);  mul_932 = None
    unsqueeze_668: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_667, 2);  unsqueeze_667 = None
    unsqueeze_669: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 3);  unsqueeze_668 = None
    mul_933: "f32[144]" = torch.ops.aten.mul.Tensor(sum_94, 3.985969387755102e-05)
    mul_934: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_935: "f32[144]" = torch.ops.aten.mul.Tensor(mul_933, mul_934);  mul_933 = mul_934 = None
    unsqueeze_670: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_935, 0);  mul_935 = None
    unsqueeze_671: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, 2);  unsqueeze_670 = None
    unsqueeze_672: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 3);  unsqueeze_671 = None
    mul_936: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_21);  primals_21 = None
    unsqueeze_673: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_936, 0);  mul_936 = None
    unsqueeze_674: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 2);  unsqueeze_673 = None
    unsqueeze_675: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 3);  unsqueeze_674 = None
    sub_259: "f32[8, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_666);  convolution_15 = unsqueeze_666 = None
    mul_937: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_259, unsqueeze_672);  sub_259 = unsqueeze_672 = None
    sub_260: "f32[8, 144, 56, 56]" = torch.ops.aten.sub.Tensor(mul_930, mul_937);  mul_930 = mul_937 = None
    sub_261: "f32[8, 144, 56, 56]" = torch.ops.aten.sub.Tensor(sub_260, unsqueeze_669);  sub_260 = unsqueeze_669 = None
    mul_938: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_261, unsqueeze_675);  sub_261 = unsqueeze_675 = None
    mul_939: "f32[144]" = torch.ops.aten.mul.Tensor(sum_94, squeeze_28);  sum_94 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(mul_938, add_45, primals_123, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_938 = add_45 = primals_123 = None
    getitem_293: "f32[8, 24, 56, 56]" = convolution_backward_65[0]
    getitem_294: "f32[144, 24, 1, 1]" = convolution_backward_65[1];  convolution_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_676: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_677: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, 2);  unsqueeze_676 = None
    unsqueeze_678: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 3);  unsqueeze_677 = None
    sum_95: "f32[24]" = torch.ops.aten.sum.dim_IntList(getitem_293, [0, 2, 3])
    sub_262: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_678)
    mul_940: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_293, sub_262);  sub_262 = None
    sum_96: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_940, [0, 2, 3]);  mul_940 = None
    mul_941: "f32[24]" = torch.ops.aten.mul.Tensor(sum_95, 3.985969387755102e-05)
    unsqueeze_679: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_941, 0);  mul_941 = None
    unsqueeze_680: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_679, 2);  unsqueeze_679 = None
    unsqueeze_681: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 3);  unsqueeze_680 = None
    mul_942: "f32[24]" = torch.ops.aten.mul.Tensor(sum_96, 3.985969387755102e-05)
    mul_943: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_944: "f32[24]" = torch.ops.aten.mul.Tensor(mul_942, mul_943);  mul_942 = mul_943 = None
    unsqueeze_682: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_944, 0);  mul_944 = None
    unsqueeze_683: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, 2);  unsqueeze_682 = None
    unsqueeze_684: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 3);  unsqueeze_683 = None
    mul_945: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_19);  primals_19 = None
    unsqueeze_685: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_945, 0);  mul_945 = None
    unsqueeze_686: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 2);  unsqueeze_685 = None
    unsqueeze_687: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 3);  unsqueeze_686 = None
    sub_263: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_678);  convolution_14 = unsqueeze_678 = None
    mul_946: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_263, unsqueeze_684);  sub_263 = unsqueeze_684 = None
    sub_264: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(getitem_293, mul_946);  mul_946 = None
    sub_265: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_264, unsqueeze_681);  sub_264 = unsqueeze_681 = None
    mul_947: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_265, unsqueeze_687);  sub_265 = unsqueeze_687 = None
    mul_948: "f32[24]" = torch.ops.aten.mul.Tensor(sum_96, squeeze_25);  sum_96 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_947, mul_67, primals_122, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_947 = mul_67 = primals_122 = None
    getitem_296: "f32[8, 144, 56, 56]" = convolution_backward_66[0]
    getitem_297: "f32[24, 144, 1, 1]" = convolution_backward_66[1];  convolution_backward_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_949: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_296, mul_65);  mul_65 = None
    mul_950: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_296, sigmoid_11);  getitem_296 = sigmoid_11 = None
    sum_97: "f32[8, 144, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_949, [2, 3], True);  mul_949 = None
    alias_29: "f32[8, 144, 1, 1]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    sub_266: "f32[8, 144, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_29)
    mul_951: "f32[8, 144, 1, 1]" = torch.ops.aten.mul.Tensor(alias_29, sub_266);  alias_29 = sub_266 = None
    mul_952: "f32[8, 144, 1, 1]" = torch.ops.aten.mul.Tensor(sum_97, mul_951);  sum_97 = mul_951 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_67 = torch.ops.aten.convolution_backward.default(mul_952, mul_66, primals_120, [144], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_952 = mul_66 = primals_120 = None
    getitem_299: "f32[8, 6, 1, 1]" = convolution_backward_67[0]
    getitem_300: "f32[144, 6, 1, 1]" = convolution_backward_67[1]
    getitem_301: "f32[144]" = convolution_backward_67[2];  convolution_backward_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_105: "f32[8, 6, 1, 1]" = torch.ops.aten.sigmoid.default(clone_8)
    full_40: "f32[8, 6, 1, 1]" = torch.ops.aten.full.default([8, 6, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_267: "f32[8, 6, 1, 1]" = torch.ops.aten.sub.Tensor(full_40, sigmoid_105);  full_40 = None
    mul_953: "f32[8, 6, 1, 1]" = torch.ops.aten.mul.Tensor(clone_8, sub_267);  clone_8 = sub_267 = None
    add_315: "f32[8, 6, 1, 1]" = torch.ops.aten.add.Scalar(mul_953, 1);  mul_953 = None
    mul_954: "f32[8, 6, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_105, add_315);  sigmoid_105 = add_315 = None
    mul_955: "f32[8, 6, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_299, mul_954);  getitem_299 = mul_954 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_955, mean_2, primals_118, [6], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_955 = mean_2 = primals_118 = None
    getitem_302: "f32[8, 144, 1, 1]" = convolution_backward_68[0]
    getitem_303: "f32[6, 144, 1, 1]" = convolution_backward_68[1]
    getitem_304: "f32[6]" = convolution_backward_68[2];  convolution_backward_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_14: "f32[8, 144, 56, 56]" = torch.ops.aten.expand.default(getitem_302, [8, 144, 56, 56]);  getitem_302 = None
    div_14: "f32[8, 144, 56, 56]" = torch.ops.aten.div.Scalar(expand_14, 3136);  expand_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_316: "f32[8, 144, 56, 56]" = torch.ops.aten.add.Tensor(mul_950, div_14);  mul_950 = div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_106: "f32[8, 144, 56, 56]" = torch.ops.aten.sigmoid.default(clone_7)
    full_41: "f32[8, 144, 56, 56]" = torch.ops.aten.full.default([8, 144, 56, 56], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_268: "f32[8, 144, 56, 56]" = torch.ops.aten.sub.Tensor(full_41, sigmoid_106);  full_41 = None
    mul_956: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(clone_7, sub_268);  clone_7 = sub_268 = None
    add_317: "f32[8, 144, 56, 56]" = torch.ops.aten.add.Scalar(mul_956, 1);  mul_956 = None
    mul_957: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sigmoid_106, add_317);  sigmoid_106 = add_317 = None
    mul_958: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(add_316, mul_957);  add_316 = mul_957 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_688: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_689: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, 2);  unsqueeze_688 = None
    unsqueeze_690: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 3);  unsqueeze_689 = None
    sum_98: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_958, [0, 2, 3])
    sub_269: "f32[8, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_690)
    mul_959: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_958, sub_269);  sub_269 = None
    sum_99: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_959, [0, 2, 3]);  mul_959 = None
    mul_960: "f32[144]" = torch.ops.aten.mul.Tensor(sum_98, 3.985969387755102e-05)
    unsqueeze_691: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_960, 0);  mul_960 = None
    unsqueeze_692: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_691, 2);  unsqueeze_691 = None
    unsqueeze_693: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 3);  unsqueeze_692 = None
    mul_961: "f32[144]" = torch.ops.aten.mul.Tensor(sum_99, 3.985969387755102e-05)
    mul_962: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_963: "f32[144]" = torch.ops.aten.mul.Tensor(mul_961, mul_962);  mul_961 = mul_962 = None
    unsqueeze_694: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_963, 0);  mul_963 = None
    unsqueeze_695: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 2);  unsqueeze_694 = None
    unsqueeze_696: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 3);  unsqueeze_695 = None
    mul_964: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_17);  primals_17 = None
    unsqueeze_697: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_964, 0);  mul_964 = None
    unsqueeze_698: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 2);  unsqueeze_697 = None
    unsqueeze_699: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 3);  unsqueeze_698 = None
    sub_270: "f32[8, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_690);  convolution_11 = unsqueeze_690 = None
    mul_965: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_270, unsqueeze_696);  sub_270 = unsqueeze_696 = None
    sub_271: "f32[8, 144, 56, 56]" = torch.ops.aten.sub.Tensor(mul_958, mul_965);  mul_958 = mul_965 = None
    sub_272: "f32[8, 144, 56, 56]" = torch.ops.aten.sub.Tensor(sub_271, unsqueeze_693);  sub_271 = unsqueeze_693 = None
    mul_966: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_699);  sub_272 = unsqueeze_699 = None
    mul_967: "f32[144]" = torch.ops.aten.mul.Tensor(sum_99, squeeze_22);  sum_99 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_69 = torch.ops.aten.convolution_backward.default(mul_966, mul_57, primals_117, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 144, [True, True, False]);  mul_966 = mul_57 = primals_117 = None
    getitem_305: "f32[8, 144, 56, 56]" = convolution_backward_69[0]
    getitem_306: "f32[144, 1, 3, 3]" = convolution_backward_69[1];  convolution_backward_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_107: "f32[8, 144, 56, 56]" = torch.ops.aten.sigmoid.default(clone_6)
    full_42: "f32[8, 144, 56, 56]" = torch.ops.aten.full.default([8, 144, 56, 56], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_273: "f32[8, 144, 56, 56]" = torch.ops.aten.sub.Tensor(full_42, sigmoid_107);  full_42 = None
    mul_968: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(clone_6, sub_273);  clone_6 = sub_273 = None
    add_318: "f32[8, 144, 56, 56]" = torch.ops.aten.add.Scalar(mul_968, 1);  mul_968 = None
    mul_969: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sigmoid_107, add_318);  sigmoid_107 = add_318 = None
    mul_970: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_305, mul_969);  getitem_305 = mul_969 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_700: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_701: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, 2);  unsqueeze_700 = None
    unsqueeze_702: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 3);  unsqueeze_701 = None
    sum_100: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_970, [0, 2, 3])
    sub_274: "f32[8, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_702)
    mul_971: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_970, sub_274);  sub_274 = None
    sum_101: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_971, [0, 2, 3]);  mul_971 = None
    mul_972: "f32[144]" = torch.ops.aten.mul.Tensor(sum_100, 3.985969387755102e-05)
    unsqueeze_703: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_972, 0);  mul_972 = None
    unsqueeze_704: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_703, 2);  unsqueeze_703 = None
    unsqueeze_705: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 3);  unsqueeze_704 = None
    mul_973: "f32[144]" = torch.ops.aten.mul.Tensor(sum_101, 3.985969387755102e-05)
    mul_974: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_975: "f32[144]" = torch.ops.aten.mul.Tensor(mul_973, mul_974);  mul_973 = mul_974 = None
    unsqueeze_706: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_975, 0);  mul_975 = None
    unsqueeze_707: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 2);  unsqueeze_706 = None
    unsqueeze_708: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 3);  unsqueeze_707 = None
    mul_976: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_15);  primals_15 = None
    unsqueeze_709: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_976, 0);  mul_976 = None
    unsqueeze_710: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 2);  unsqueeze_709 = None
    unsqueeze_711: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 3);  unsqueeze_710 = None
    sub_275: "f32[8, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_702);  convolution_10 = unsqueeze_702 = None
    mul_977: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_275, unsqueeze_708);  sub_275 = unsqueeze_708 = None
    sub_276: "f32[8, 144, 56, 56]" = torch.ops.aten.sub.Tensor(mul_970, mul_977);  mul_970 = mul_977 = None
    sub_277: "f32[8, 144, 56, 56]" = torch.ops.aten.sub.Tensor(sub_276, unsqueeze_705);  sub_276 = unsqueeze_705 = None
    mul_978: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_277, unsqueeze_711);  sub_277 = unsqueeze_711 = None
    mul_979: "f32[144]" = torch.ops.aten.mul.Tensor(sum_101, squeeze_19);  sum_101 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_978, add_29, primals_116, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_978 = add_29 = primals_116 = None
    getitem_308: "f32[8, 24, 56, 56]" = convolution_backward_70[0]
    getitem_309: "f32[144, 24, 1, 1]" = convolution_backward_70[1];  convolution_backward_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_319: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(getitem_293, getitem_308);  getitem_293 = getitem_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_712: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_713: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, 2);  unsqueeze_712 = None
    unsqueeze_714: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 3);  unsqueeze_713 = None
    sum_102: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_319, [0, 2, 3])
    sub_278: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_714)
    mul_980: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_319, sub_278);  sub_278 = None
    sum_103: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_980, [0, 2, 3]);  mul_980 = None
    mul_981: "f32[24]" = torch.ops.aten.mul.Tensor(sum_102, 3.985969387755102e-05)
    unsqueeze_715: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_981, 0);  mul_981 = None
    unsqueeze_716: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_715, 2);  unsqueeze_715 = None
    unsqueeze_717: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 3);  unsqueeze_716 = None
    mul_982: "f32[24]" = torch.ops.aten.mul.Tensor(sum_103, 3.985969387755102e-05)
    mul_983: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_984: "f32[24]" = torch.ops.aten.mul.Tensor(mul_982, mul_983);  mul_982 = mul_983 = None
    unsqueeze_718: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_984, 0);  mul_984 = None
    unsqueeze_719: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 2);  unsqueeze_718 = None
    unsqueeze_720: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 3);  unsqueeze_719 = None
    mul_985: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_13);  primals_13 = None
    unsqueeze_721: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_985, 0);  mul_985 = None
    unsqueeze_722: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 2);  unsqueeze_721 = None
    unsqueeze_723: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 3);  unsqueeze_722 = None
    sub_279: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_714);  convolution_9 = unsqueeze_714 = None
    mul_986: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_279, unsqueeze_720);  sub_279 = unsqueeze_720 = None
    sub_280: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(add_319, mul_986);  add_319 = mul_986 = None
    sub_281: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_280, unsqueeze_717);  sub_280 = unsqueeze_717 = None
    mul_987: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_281, unsqueeze_723);  sub_281 = unsqueeze_723 = None
    mul_988: "f32[24]" = torch.ops.aten.mul.Tensor(sum_103, squeeze_16);  sum_103 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_71 = torch.ops.aten.convolution_backward.default(mul_987, mul_42, primals_115, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_987 = mul_42 = primals_115 = None
    getitem_311: "f32[8, 96, 56, 56]" = convolution_backward_71[0]
    getitem_312: "f32[24, 96, 1, 1]" = convolution_backward_71[1];  convolution_backward_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_989: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_311, mul_40);  mul_40 = None
    mul_990: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_311, sigmoid_7);  getitem_311 = sigmoid_7 = None
    sum_104: "f32[8, 96, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_989, [2, 3], True);  mul_989 = None
    alias_30: "f32[8, 96, 1, 1]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    sub_282: "f32[8, 96, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_30)
    mul_991: "f32[8, 96, 1, 1]" = torch.ops.aten.mul.Tensor(alias_30, sub_282);  alias_30 = sub_282 = None
    mul_992: "f32[8, 96, 1, 1]" = torch.ops.aten.mul.Tensor(sum_104, mul_991);  sum_104 = mul_991 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_72 = torch.ops.aten.convolution_backward.default(mul_992, mul_41, primals_113, [96], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_992 = mul_41 = primals_113 = None
    getitem_314: "f32[8, 4, 1, 1]" = convolution_backward_72[0]
    getitem_315: "f32[96, 4, 1, 1]" = convolution_backward_72[1]
    getitem_316: "f32[96]" = convolution_backward_72[2];  convolution_backward_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_108: "f32[8, 4, 1, 1]" = torch.ops.aten.sigmoid.default(clone_5)
    full_43: "f32[8, 4, 1, 1]" = torch.ops.aten.full.default([8, 4, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_283: "f32[8, 4, 1, 1]" = torch.ops.aten.sub.Tensor(full_43, sigmoid_108);  full_43 = None
    mul_993: "f32[8, 4, 1, 1]" = torch.ops.aten.mul.Tensor(clone_5, sub_283);  clone_5 = sub_283 = None
    add_320: "f32[8, 4, 1, 1]" = torch.ops.aten.add.Scalar(mul_993, 1);  mul_993 = None
    mul_994: "f32[8, 4, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_108, add_320);  sigmoid_108 = add_320 = None
    mul_995: "f32[8, 4, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_314, mul_994);  getitem_314 = mul_994 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_73 = torch.ops.aten.convolution_backward.default(mul_995, mean_1, primals_111, [4], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_995 = mean_1 = primals_111 = None
    getitem_317: "f32[8, 96, 1, 1]" = convolution_backward_73[0]
    getitem_318: "f32[4, 96, 1, 1]" = convolution_backward_73[1]
    getitem_319: "f32[4]" = convolution_backward_73[2];  convolution_backward_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_15: "f32[8, 96, 56, 56]" = torch.ops.aten.expand.default(getitem_317, [8, 96, 56, 56]);  getitem_317 = None
    div_15: "f32[8, 96, 56, 56]" = torch.ops.aten.div.Scalar(expand_15, 3136);  expand_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_321: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_990, div_15);  mul_990 = div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_109: "f32[8, 96, 56, 56]" = torch.ops.aten.sigmoid.default(clone_4)
    full_44: "f32[8, 96, 56, 56]" = torch.ops.aten.full.default([8, 96, 56, 56], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_284: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(full_44, sigmoid_109);  full_44 = None
    mul_996: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(clone_4, sub_284);  clone_4 = sub_284 = None
    add_322: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Scalar(mul_996, 1);  mul_996 = None
    mul_997: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sigmoid_109, add_322);  sigmoid_109 = add_322 = None
    mul_998: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_321, mul_997);  add_321 = mul_997 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_724: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_725: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, 2);  unsqueeze_724 = None
    unsqueeze_726: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 3);  unsqueeze_725 = None
    sum_105: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_998, [0, 2, 3])
    sub_285: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_726)
    mul_999: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(mul_998, sub_285);  sub_285 = None
    sum_106: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_999, [0, 2, 3]);  mul_999 = None
    mul_1000: "f32[96]" = torch.ops.aten.mul.Tensor(sum_105, 3.985969387755102e-05)
    unsqueeze_727: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1000, 0);  mul_1000 = None
    unsqueeze_728: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_727, 2);  unsqueeze_727 = None
    unsqueeze_729: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 3);  unsqueeze_728 = None
    mul_1001: "f32[96]" = torch.ops.aten.mul.Tensor(sum_106, 3.985969387755102e-05)
    mul_1002: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_1003: "f32[96]" = torch.ops.aten.mul.Tensor(mul_1001, mul_1002);  mul_1001 = mul_1002 = None
    unsqueeze_730: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1003, 0);  mul_1003 = None
    unsqueeze_731: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, 2);  unsqueeze_730 = None
    unsqueeze_732: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 3);  unsqueeze_731 = None
    mul_1004: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_11);  primals_11 = None
    unsqueeze_733: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1004, 0);  mul_1004 = None
    unsqueeze_734: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 2);  unsqueeze_733 = None
    unsqueeze_735: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 3);  unsqueeze_734 = None
    sub_286: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_726);  convolution_6 = unsqueeze_726 = None
    mul_1005: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_286, unsqueeze_732);  sub_286 = unsqueeze_732 = None
    sub_287: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(mul_998, mul_1005);  mul_998 = mul_1005 = None
    sub_288: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(sub_287, unsqueeze_729);  sub_287 = unsqueeze_729 = None
    mul_1006: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_288, unsqueeze_735);  sub_288 = unsqueeze_735 = None
    mul_1007: "f32[96]" = torch.ops.aten.mul.Tensor(sum_106, squeeze_13);  sum_106 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_74 = torch.ops.aten.convolution_backward.default(mul_1006, constant_pad_nd_1, primals_10, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 96, [True, True, False]);  mul_1006 = constant_pad_nd_1 = primals_10 = None
    getitem_320: "f32[8, 96, 113, 113]" = convolution_backward_74[0]
    getitem_321: "f32[96, 1, 3, 3]" = convolution_backward_74[1];  convolution_backward_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_8: "f32[8, 96, 112, 112]" = torch.ops.aten.constant_pad_nd.default(getitem_320, [0, -1, 0, -1]);  getitem_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_110: "f32[8, 96, 112, 112]" = torch.ops.aten.sigmoid.default(clone_3)
    full_45: "f32[8, 96, 112, 112]" = torch.ops.aten.full.default([8, 96, 112, 112], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_289: "f32[8, 96, 112, 112]" = torch.ops.aten.sub.Tensor(full_45, sigmoid_110);  full_45 = None
    mul_1008: "f32[8, 96, 112, 112]" = torch.ops.aten.mul.Tensor(clone_3, sub_289);  clone_3 = sub_289 = None
    add_323: "f32[8, 96, 112, 112]" = torch.ops.aten.add.Scalar(mul_1008, 1);  mul_1008 = None
    mul_1009: "f32[8, 96, 112, 112]" = torch.ops.aten.mul.Tensor(sigmoid_110, add_323);  sigmoid_110 = add_323 = None
    mul_1010: "f32[8, 96, 112, 112]" = torch.ops.aten.mul.Tensor(constant_pad_nd_8, mul_1009);  constant_pad_nd_8 = mul_1009 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_736: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_737: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, 2);  unsqueeze_736 = None
    unsqueeze_738: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 3);  unsqueeze_737 = None
    sum_107: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1010, [0, 2, 3])
    sub_290: "f32[8, 96, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_738)
    mul_1011: "f32[8, 96, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1010, sub_290);  sub_290 = None
    sum_108: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1011, [0, 2, 3]);  mul_1011 = None
    mul_1012: "f32[96]" = torch.ops.aten.mul.Tensor(sum_107, 9.964923469387754e-06)
    unsqueeze_739: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1012, 0);  mul_1012 = None
    unsqueeze_740: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_739, 2);  unsqueeze_739 = None
    unsqueeze_741: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 3);  unsqueeze_740 = None
    mul_1013: "f32[96]" = torch.ops.aten.mul.Tensor(sum_108, 9.964923469387754e-06)
    mul_1014: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_1015: "f32[96]" = torch.ops.aten.mul.Tensor(mul_1013, mul_1014);  mul_1013 = mul_1014 = None
    unsqueeze_742: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1015, 0);  mul_1015 = None
    unsqueeze_743: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, 2);  unsqueeze_742 = None
    unsqueeze_744: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 3);  unsqueeze_743 = None
    mul_1016: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_8);  primals_8 = None
    unsqueeze_745: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1016, 0);  mul_1016 = None
    unsqueeze_746: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_745, 2);  unsqueeze_745 = None
    unsqueeze_747: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 3);  unsqueeze_746 = None
    sub_291: "f32[8, 96, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_738);  convolution_5 = unsqueeze_738 = None
    mul_1017: "f32[8, 96, 112, 112]" = torch.ops.aten.mul.Tensor(sub_291, unsqueeze_744);  sub_291 = unsqueeze_744 = None
    sub_292: "f32[8, 96, 112, 112]" = torch.ops.aten.sub.Tensor(mul_1010, mul_1017);  mul_1010 = mul_1017 = None
    sub_293: "f32[8, 96, 112, 112]" = torch.ops.aten.sub.Tensor(sub_292, unsqueeze_741);  sub_292 = unsqueeze_741 = None
    mul_1018: "f32[8, 96, 112, 112]" = torch.ops.aten.mul.Tensor(sub_293, unsqueeze_747);  sub_293 = unsqueeze_747 = None
    mul_1019: "f32[96]" = torch.ops.aten.mul.Tensor(sum_108, squeeze_10);  sum_108 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_75 = torch.ops.aten.convolution_backward.default(mul_1018, add_14, primals_110, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1018 = add_14 = primals_110 = None
    getitem_323: "f32[8, 16, 112, 112]" = convolution_backward_75[0]
    getitem_324: "f32[96, 16, 1, 1]" = convolution_backward_75[1];  convolution_backward_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_748: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_749: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, 2);  unsqueeze_748 = None
    unsqueeze_750: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 3);  unsqueeze_749 = None
    sum_109: "f32[16]" = torch.ops.aten.sum.dim_IntList(getitem_323, [0, 2, 3])
    sub_294: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_750)
    mul_1020: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_323, sub_294);  sub_294 = None
    sum_110: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1020, [0, 2, 3]);  mul_1020 = None
    mul_1021: "f32[16]" = torch.ops.aten.mul.Tensor(sum_109, 9.964923469387754e-06)
    unsqueeze_751: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1021, 0);  mul_1021 = None
    unsqueeze_752: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_751, 2);  unsqueeze_751 = None
    unsqueeze_753: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 3);  unsqueeze_752 = None
    mul_1022: "f32[16]" = torch.ops.aten.mul.Tensor(sum_110, 9.964923469387754e-06)
    mul_1023: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_1024: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1022, mul_1023);  mul_1022 = mul_1023 = None
    unsqueeze_754: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1024, 0);  mul_1024 = None
    unsqueeze_755: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, 2);  unsqueeze_754 = None
    unsqueeze_756: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 3);  unsqueeze_755 = None
    mul_1025: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_6);  primals_6 = None
    unsqueeze_757: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1025, 0);  mul_1025 = None
    unsqueeze_758: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_757, 2);  unsqueeze_757 = None
    unsqueeze_759: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 3);  unsqueeze_758 = None
    sub_295: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_750);  convolution_4 = unsqueeze_750 = None
    mul_1026: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_295, unsqueeze_756);  sub_295 = unsqueeze_756 = None
    sub_296: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(getitem_323, mul_1026);  getitem_323 = mul_1026 = None
    sub_297: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_296, unsqueeze_753);  sub_296 = unsqueeze_753 = None
    mul_1027: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_297, unsqueeze_759);  sub_297 = unsqueeze_759 = None
    mul_1028: "f32[16]" = torch.ops.aten.mul.Tensor(sum_110, squeeze_7);  sum_110 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_76 = torch.ops.aten.convolution_backward.default(mul_1027, mul_17, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1027 = mul_17 = primals_109 = None
    getitem_326: "f32[8, 32, 112, 112]" = convolution_backward_76[0]
    getitem_327: "f32[16, 32, 1, 1]" = convolution_backward_76[1];  convolution_backward_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_1029: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_326, mul_15);  mul_15 = None
    mul_1030: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_326, sigmoid_3);  getitem_326 = sigmoid_3 = None
    sum_111: "f32[8, 32, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1029, [2, 3], True);  mul_1029 = None
    alias_31: "f32[8, 32, 1, 1]" = torch.ops.aten.alias.default(alias);  alias = None
    sub_298: "f32[8, 32, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_31)
    mul_1031: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(alias_31, sub_298);  alias_31 = sub_298 = None
    mul_1032: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(sum_111, mul_1031);  sum_111 = mul_1031 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_77 = torch.ops.aten.convolution_backward.default(mul_1032, mul_16, primals_107, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1032 = mul_16 = primals_107 = None
    getitem_329: "f32[8, 8, 1, 1]" = convolution_backward_77[0]
    getitem_330: "f32[32, 8, 1, 1]" = convolution_backward_77[1]
    getitem_331: "f32[32]" = convolution_backward_77[2];  convolution_backward_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_111: "f32[8, 8, 1, 1]" = torch.ops.aten.sigmoid.default(clone_2)
    full_46: "f32[8, 8, 1, 1]" = torch.ops.aten.full.default([8, 8, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_299: "f32[8, 8, 1, 1]" = torch.ops.aten.sub.Tensor(full_46, sigmoid_111);  full_46 = None
    mul_1033: "f32[8, 8, 1, 1]" = torch.ops.aten.mul.Tensor(clone_2, sub_299);  clone_2 = sub_299 = None
    add_324: "f32[8, 8, 1, 1]" = torch.ops.aten.add.Scalar(mul_1033, 1);  mul_1033 = None
    mul_1034: "f32[8, 8, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_111, add_324);  sigmoid_111 = add_324 = None
    mul_1035: "f32[8, 8, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_329, mul_1034);  getitem_329 = mul_1034 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_78 = torch.ops.aten.convolution_backward.default(mul_1035, mean, primals_105, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1035 = mean = primals_105 = None
    getitem_332: "f32[8, 32, 1, 1]" = convolution_backward_78[0]
    getitem_333: "f32[8, 32, 1, 1]" = convolution_backward_78[1]
    getitem_334: "f32[8]" = convolution_backward_78[2];  convolution_backward_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_16: "f32[8, 32, 112, 112]" = torch.ops.aten.expand.default(getitem_332, [8, 32, 112, 112]);  getitem_332 = None
    div_16: "f32[8, 32, 112, 112]" = torch.ops.aten.div.Scalar(expand_16, 12544);  expand_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_325: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_1030, div_16);  mul_1030 = div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_112: "f32[8, 32, 112, 112]" = torch.ops.aten.sigmoid.default(clone_1)
    full_47: "f32[8, 32, 112, 112]" = torch.ops.aten.full.default([8, 32, 112, 112], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_300: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(full_47, sigmoid_112);  full_47 = None
    mul_1036: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(clone_1, sub_300);  clone_1 = sub_300 = None
    add_326: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Scalar(mul_1036, 1);  mul_1036 = None
    mul_1037: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sigmoid_112, add_326);  sigmoid_112 = add_326 = None
    mul_1038: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(add_325, mul_1037);  add_325 = mul_1037 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_760: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_761: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, 2);  unsqueeze_760 = None
    unsqueeze_762: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 3);  unsqueeze_761 = None
    sum_112: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1038, [0, 2, 3])
    sub_301: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_762)
    mul_1039: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1038, sub_301);  sub_301 = None
    sum_113: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1039, [0, 2, 3]);  mul_1039 = None
    mul_1040: "f32[32]" = torch.ops.aten.mul.Tensor(sum_112, 9.964923469387754e-06)
    unsqueeze_763: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1040, 0);  mul_1040 = None
    unsqueeze_764: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_763, 2);  unsqueeze_763 = None
    unsqueeze_765: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 3);  unsqueeze_764 = None
    mul_1041: "f32[32]" = torch.ops.aten.mul.Tensor(sum_113, 9.964923469387754e-06)
    mul_1042: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_1043: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1041, mul_1042);  mul_1041 = mul_1042 = None
    unsqueeze_766: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1043, 0);  mul_1043 = None
    unsqueeze_767: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, 2);  unsqueeze_766 = None
    unsqueeze_768: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 3);  unsqueeze_767 = None
    mul_1044: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_4);  primals_4 = None
    unsqueeze_769: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1044, 0);  mul_1044 = None
    unsqueeze_770: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_769, 2);  unsqueeze_769 = None
    unsqueeze_771: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 3);  unsqueeze_770 = None
    sub_302: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_762);  convolution_1 = unsqueeze_762 = None
    mul_1045: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_302, unsqueeze_768);  sub_302 = unsqueeze_768 = None
    sub_303: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(mul_1038, mul_1045);  mul_1038 = mul_1045 = None
    sub_304: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_303, unsqueeze_765);  sub_303 = unsqueeze_765 = None
    mul_1046: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_304, unsqueeze_771);  sub_304 = unsqueeze_771 = None
    mul_1047: "f32[32]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_4);  sum_113 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_79 = torch.ops.aten.convolution_backward.default(mul_1046, mul_7, primals_104, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1046 = mul_7 = primals_104 = None
    getitem_335: "f32[8, 32, 112, 112]" = convolution_backward_79[0]
    getitem_336: "f32[32, 1, 3, 3]" = convolution_backward_79[1];  convolution_backward_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_113: "f32[8, 32, 112, 112]" = torch.ops.aten.sigmoid.default(clone)
    full_48: "f32[8, 32, 112, 112]" = torch.ops.aten.full.default([8, 32, 112, 112], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_305: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(full_48, sigmoid_113);  full_48 = None
    mul_1048: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(clone, sub_305);  clone = sub_305 = None
    add_327: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Scalar(mul_1048, 1);  mul_1048 = None
    mul_1049: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sigmoid_113, add_327);  sigmoid_113 = add_327 = None
    mul_1050: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_335, mul_1049);  getitem_335 = mul_1049 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_772: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_773: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, 2);  unsqueeze_772 = None
    unsqueeze_774: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 3);  unsqueeze_773 = None
    sum_114: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1050, [0, 2, 3])
    sub_306: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_774)
    mul_1051: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1050, sub_306);  sub_306 = None
    sum_115: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1051, [0, 2, 3]);  mul_1051 = None
    mul_1052: "f32[32]" = torch.ops.aten.mul.Tensor(sum_114, 9.964923469387754e-06)
    unsqueeze_775: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1052, 0);  mul_1052 = None
    unsqueeze_776: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_775, 2);  unsqueeze_775 = None
    unsqueeze_777: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 3);  unsqueeze_776 = None
    mul_1053: "f32[32]" = torch.ops.aten.mul.Tensor(sum_115, 9.964923469387754e-06)
    mul_1054: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_1055: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1053, mul_1054);  mul_1053 = mul_1054 = None
    unsqueeze_778: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1055, 0);  mul_1055 = None
    unsqueeze_779: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, 2);  unsqueeze_778 = None
    unsqueeze_780: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 3);  unsqueeze_779 = None
    mul_1056: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_2);  primals_2 = None
    unsqueeze_781: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1056, 0);  mul_1056 = None
    unsqueeze_782: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_781, 2);  unsqueeze_781 = None
    unsqueeze_783: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 3);  unsqueeze_782 = None
    sub_307: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_774);  convolution = unsqueeze_774 = None
    mul_1057: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_307, unsqueeze_780);  sub_307 = unsqueeze_780 = None
    sub_308: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(mul_1050, mul_1057);  mul_1050 = mul_1057 = None
    sub_309: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_308, unsqueeze_777);  sub_308 = unsqueeze_777 = None
    mul_1058: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_309, unsqueeze_783);  sub_309 = unsqueeze_783 = None
    mul_1059: "f32[32]" = torch.ops.aten.mul.Tensor(sum_115, squeeze_1);  sum_115 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_80 = torch.ops.aten.convolution_backward.default(mul_1058, constant_pad_nd, primals_1, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1058 = constant_pad_nd = primals_1 = None
    getitem_339: "f32[32, 3, 3, 3]" = convolution_backward_80[1];  convolution_backward_80 = None
    
    # No stacktrace found for following nodes
    copy_: "i64[]" = torch.ops.aten.copy_.default(primals_214, add);  primals_214 = add = None
    copy__1: "f32[32]" = torch.ops.aten.copy_.default(primals_215, add_2);  primals_215 = add_2 = None
    copy__2: "f32[32]" = torch.ops.aten.copy_.default(primals_216, add_3);  primals_216 = add_3 = None
    copy__3: "i64[]" = torch.ops.aten.copy_.default(primals_217, add_5);  primals_217 = add_5 = None
    copy__4: "f32[32]" = torch.ops.aten.copy_.default(primals_218, add_7);  primals_218 = add_7 = None
    copy__5: "f32[32]" = torch.ops.aten.copy_.default(primals_219, add_8);  primals_219 = add_8 = None
    copy__6: "i64[]" = torch.ops.aten.copy_.default(primals_220, add_10);  primals_220 = add_10 = None
    copy__7: "f32[16]" = torch.ops.aten.copy_.default(primals_221, add_12);  primals_221 = add_12 = None
    copy__8: "f32[16]" = torch.ops.aten.copy_.default(primals_222, add_13);  primals_222 = add_13 = None
    copy__9: "i64[]" = torch.ops.aten.copy_.default(primals_223, add_15);  primals_223 = add_15 = None
    copy__10: "f32[96]" = torch.ops.aten.copy_.default(primals_224, add_17);  primals_224 = add_17 = None
    copy__11: "f32[96]" = torch.ops.aten.copy_.default(primals_225, add_18);  primals_225 = add_18 = None
    copy__12: "i64[]" = torch.ops.aten.copy_.default(primals_226, add_20);  primals_226 = add_20 = None
    copy__13: "f32[96]" = torch.ops.aten.copy_.default(primals_227, add_22);  primals_227 = add_22 = None
    copy__14: "f32[96]" = torch.ops.aten.copy_.default(primals_228, add_23);  primals_228 = add_23 = None
    copy__15: "i64[]" = torch.ops.aten.copy_.default(primals_229, add_25);  primals_229 = add_25 = None
    copy__16: "f32[24]" = torch.ops.aten.copy_.default(primals_230, add_27);  primals_230 = add_27 = None
    copy__17: "f32[24]" = torch.ops.aten.copy_.default(primals_231, add_28);  primals_231 = add_28 = None
    copy__18: "i64[]" = torch.ops.aten.copy_.default(primals_232, add_30);  primals_232 = add_30 = None
    copy__19: "f32[144]" = torch.ops.aten.copy_.default(primals_233, add_32);  primals_233 = add_32 = None
    copy__20: "f32[144]" = torch.ops.aten.copy_.default(primals_234, add_33);  primals_234 = add_33 = None
    copy__21: "i64[]" = torch.ops.aten.copy_.default(primals_235, add_35);  primals_235 = add_35 = None
    copy__22: "f32[144]" = torch.ops.aten.copy_.default(primals_236, add_37);  primals_236 = add_37 = None
    copy__23: "f32[144]" = torch.ops.aten.copy_.default(primals_237, add_38);  primals_237 = add_38 = None
    copy__24: "i64[]" = torch.ops.aten.copy_.default(primals_238, add_40);  primals_238 = add_40 = None
    copy__25: "f32[24]" = torch.ops.aten.copy_.default(primals_239, add_42);  primals_239 = add_42 = None
    copy__26: "f32[24]" = torch.ops.aten.copy_.default(primals_240, add_43);  primals_240 = add_43 = None
    copy__27: "i64[]" = torch.ops.aten.copy_.default(primals_241, add_46);  primals_241 = add_46 = None
    copy__28: "f32[144]" = torch.ops.aten.copy_.default(primals_242, add_48);  primals_242 = add_48 = None
    copy__29: "f32[144]" = torch.ops.aten.copy_.default(primals_243, add_49);  primals_243 = add_49 = None
    copy__30: "i64[]" = torch.ops.aten.copy_.default(primals_244, add_51);  primals_244 = add_51 = None
    copy__31: "f32[144]" = torch.ops.aten.copy_.default(primals_245, add_53);  primals_245 = add_53 = None
    copy__32: "f32[144]" = torch.ops.aten.copy_.default(primals_246, add_54);  primals_246 = add_54 = None
    copy__33: "i64[]" = torch.ops.aten.copy_.default(primals_247, add_56);  primals_247 = add_56 = None
    copy__34: "f32[40]" = torch.ops.aten.copy_.default(primals_248, add_58);  primals_248 = add_58 = None
    copy__35: "f32[40]" = torch.ops.aten.copy_.default(primals_249, add_59);  primals_249 = add_59 = None
    copy__36: "i64[]" = torch.ops.aten.copy_.default(primals_250, add_61);  primals_250 = add_61 = None
    copy__37: "f32[240]" = torch.ops.aten.copy_.default(primals_251, add_63);  primals_251 = add_63 = None
    copy__38: "f32[240]" = torch.ops.aten.copy_.default(primals_252, add_64);  primals_252 = add_64 = None
    copy__39: "i64[]" = torch.ops.aten.copy_.default(primals_253, add_66);  primals_253 = add_66 = None
    copy__40: "f32[240]" = torch.ops.aten.copy_.default(primals_254, add_68);  primals_254 = add_68 = None
    copy__41: "f32[240]" = torch.ops.aten.copy_.default(primals_255, add_69);  primals_255 = add_69 = None
    copy__42: "i64[]" = torch.ops.aten.copy_.default(primals_256, add_71);  primals_256 = add_71 = None
    copy__43: "f32[40]" = torch.ops.aten.copy_.default(primals_257, add_73);  primals_257 = add_73 = None
    copy__44: "f32[40]" = torch.ops.aten.copy_.default(primals_258, add_74);  primals_258 = add_74 = None
    copy__45: "i64[]" = torch.ops.aten.copy_.default(primals_259, add_77);  primals_259 = add_77 = None
    copy__46: "f32[240]" = torch.ops.aten.copy_.default(primals_260, add_79);  primals_260 = add_79 = None
    copy__47: "f32[240]" = torch.ops.aten.copy_.default(primals_261, add_80);  primals_261 = add_80 = None
    copy__48: "i64[]" = torch.ops.aten.copy_.default(primals_262, add_82);  primals_262 = add_82 = None
    copy__49: "f32[240]" = torch.ops.aten.copy_.default(primals_263, add_84);  primals_263 = add_84 = None
    copy__50: "f32[240]" = torch.ops.aten.copy_.default(primals_264, add_85);  primals_264 = add_85 = None
    copy__51: "i64[]" = torch.ops.aten.copy_.default(primals_265, add_87);  primals_265 = add_87 = None
    copy__52: "f32[80]" = torch.ops.aten.copy_.default(primals_266, add_89);  primals_266 = add_89 = None
    copy__53: "f32[80]" = torch.ops.aten.copy_.default(primals_267, add_90);  primals_267 = add_90 = None
    copy__54: "i64[]" = torch.ops.aten.copy_.default(primals_268, add_92);  primals_268 = add_92 = None
    copy__55: "f32[480]" = torch.ops.aten.copy_.default(primals_269, add_94);  primals_269 = add_94 = None
    copy__56: "f32[480]" = torch.ops.aten.copy_.default(primals_270, add_95);  primals_270 = add_95 = None
    copy__57: "i64[]" = torch.ops.aten.copy_.default(primals_271, add_97);  primals_271 = add_97 = None
    copy__58: "f32[480]" = torch.ops.aten.copy_.default(primals_272, add_99);  primals_272 = add_99 = None
    copy__59: "f32[480]" = torch.ops.aten.copy_.default(primals_273, add_100);  primals_273 = add_100 = None
    copy__60: "i64[]" = torch.ops.aten.copy_.default(primals_274, add_102);  primals_274 = add_102 = None
    copy__61: "f32[80]" = torch.ops.aten.copy_.default(primals_275, add_104);  primals_275 = add_104 = None
    copy__62: "f32[80]" = torch.ops.aten.copy_.default(primals_276, add_105);  primals_276 = add_105 = None
    copy__63: "i64[]" = torch.ops.aten.copy_.default(primals_277, add_108);  primals_277 = add_108 = None
    copy__64: "f32[480]" = torch.ops.aten.copy_.default(primals_278, add_110);  primals_278 = add_110 = None
    copy__65: "f32[480]" = torch.ops.aten.copy_.default(primals_279, add_111);  primals_279 = add_111 = None
    copy__66: "i64[]" = torch.ops.aten.copy_.default(primals_280, add_113);  primals_280 = add_113 = None
    copy__67: "f32[480]" = torch.ops.aten.copy_.default(primals_281, add_115);  primals_281 = add_115 = None
    copy__68: "f32[480]" = torch.ops.aten.copy_.default(primals_282, add_116);  primals_282 = add_116 = None
    copy__69: "i64[]" = torch.ops.aten.copy_.default(primals_283, add_118);  primals_283 = add_118 = None
    copy__70: "f32[80]" = torch.ops.aten.copy_.default(primals_284, add_120);  primals_284 = add_120 = None
    copy__71: "f32[80]" = torch.ops.aten.copy_.default(primals_285, add_121);  primals_285 = add_121 = None
    copy__72: "i64[]" = torch.ops.aten.copy_.default(primals_286, add_124);  primals_286 = add_124 = None
    copy__73: "f32[480]" = torch.ops.aten.copy_.default(primals_287, add_126);  primals_287 = add_126 = None
    copy__74: "f32[480]" = torch.ops.aten.copy_.default(primals_288, add_127);  primals_288 = add_127 = None
    copy__75: "i64[]" = torch.ops.aten.copy_.default(primals_289, add_129);  primals_289 = add_129 = None
    copy__76: "f32[480]" = torch.ops.aten.copy_.default(primals_290, add_131);  primals_290 = add_131 = None
    copy__77: "f32[480]" = torch.ops.aten.copy_.default(primals_291, add_132);  primals_291 = add_132 = None
    copy__78: "i64[]" = torch.ops.aten.copy_.default(primals_292, add_134);  primals_292 = add_134 = None
    copy__79: "f32[112]" = torch.ops.aten.copy_.default(primals_293, add_136);  primals_293 = add_136 = None
    copy__80: "f32[112]" = torch.ops.aten.copy_.default(primals_294, add_137);  primals_294 = add_137 = None
    copy__81: "i64[]" = torch.ops.aten.copy_.default(primals_295, add_139);  primals_295 = add_139 = None
    copy__82: "f32[672]" = torch.ops.aten.copy_.default(primals_296, add_141);  primals_296 = add_141 = None
    copy__83: "f32[672]" = torch.ops.aten.copy_.default(primals_297, add_142);  primals_297 = add_142 = None
    copy__84: "i64[]" = torch.ops.aten.copy_.default(primals_298, add_144);  primals_298 = add_144 = None
    copy__85: "f32[672]" = torch.ops.aten.copy_.default(primals_299, add_146);  primals_299 = add_146 = None
    copy__86: "f32[672]" = torch.ops.aten.copy_.default(primals_300, add_147);  primals_300 = add_147 = None
    copy__87: "i64[]" = torch.ops.aten.copy_.default(primals_301, add_149);  primals_301 = add_149 = None
    copy__88: "f32[112]" = torch.ops.aten.copy_.default(primals_302, add_151);  primals_302 = add_151 = None
    copy__89: "f32[112]" = torch.ops.aten.copy_.default(primals_303, add_152);  primals_303 = add_152 = None
    copy__90: "i64[]" = torch.ops.aten.copy_.default(primals_304, add_155);  primals_304 = add_155 = None
    copy__91: "f32[672]" = torch.ops.aten.copy_.default(primals_305, add_157);  primals_305 = add_157 = None
    copy__92: "f32[672]" = torch.ops.aten.copy_.default(primals_306, add_158);  primals_306 = add_158 = None
    copy__93: "i64[]" = torch.ops.aten.copy_.default(primals_307, add_160);  primals_307 = add_160 = None
    copy__94: "f32[672]" = torch.ops.aten.copy_.default(primals_308, add_162);  primals_308 = add_162 = None
    copy__95: "f32[672]" = torch.ops.aten.copy_.default(primals_309, add_163);  primals_309 = add_163 = None
    copy__96: "i64[]" = torch.ops.aten.copy_.default(primals_310, add_165);  primals_310 = add_165 = None
    copy__97: "f32[112]" = torch.ops.aten.copy_.default(primals_311, add_167);  primals_311 = add_167 = None
    copy__98: "f32[112]" = torch.ops.aten.copy_.default(primals_312, add_168);  primals_312 = add_168 = None
    copy__99: "i64[]" = torch.ops.aten.copy_.default(primals_313, add_171);  primals_313 = add_171 = None
    copy__100: "f32[672]" = torch.ops.aten.copy_.default(primals_314, add_173);  primals_314 = add_173 = None
    copy__101: "f32[672]" = torch.ops.aten.copy_.default(primals_315, add_174);  primals_315 = add_174 = None
    copy__102: "i64[]" = torch.ops.aten.copy_.default(primals_316, add_176);  primals_316 = add_176 = None
    copy__103: "f32[672]" = torch.ops.aten.copy_.default(primals_317, add_178);  primals_317 = add_178 = None
    copy__104: "f32[672]" = torch.ops.aten.copy_.default(primals_318, add_179);  primals_318 = add_179 = None
    copy__105: "i64[]" = torch.ops.aten.copy_.default(primals_319, add_181);  primals_319 = add_181 = None
    copy__106: "f32[192]" = torch.ops.aten.copy_.default(primals_320, add_183);  primals_320 = add_183 = None
    copy__107: "f32[192]" = torch.ops.aten.copy_.default(primals_321, add_184);  primals_321 = add_184 = None
    copy__108: "i64[]" = torch.ops.aten.copy_.default(primals_322, add_186);  primals_322 = add_186 = None
    copy__109: "f32[1152]" = torch.ops.aten.copy_.default(primals_323, add_188);  primals_323 = add_188 = None
    copy__110: "f32[1152]" = torch.ops.aten.copy_.default(primals_324, add_189);  primals_324 = add_189 = None
    copy__111: "i64[]" = torch.ops.aten.copy_.default(primals_325, add_191);  primals_325 = add_191 = None
    copy__112: "f32[1152]" = torch.ops.aten.copy_.default(primals_326, add_193);  primals_326 = add_193 = None
    copy__113: "f32[1152]" = torch.ops.aten.copy_.default(primals_327, add_194);  primals_327 = add_194 = None
    copy__114: "i64[]" = torch.ops.aten.copy_.default(primals_328, add_196);  primals_328 = add_196 = None
    copy__115: "f32[192]" = torch.ops.aten.copy_.default(primals_329, add_198);  primals_329 = add_198 = None
    copy__116: "f32[192]" = torch.ops.aten.copy_.default(primals_330, add_199);  primals_330 = add_199 = None
    copy__117: "i64[]" = torch.ops.aten.copy_.default(primals_331, add_202);  primals_331 = add_202 = None
    copy__118: "f32[1152]" = torch.ops.aten.copy_.default(primals_332, add_204);  primals_332 = add_204 = None
    copy__119: "f32[1152]" = torch.ops.aten.copy_.default(primals_333, add_205);  primals_333 = add_205 = None
    copy__120: "i64[]" = torch.ops.aten.copy_.default(primals_334, add_207);  primals_334 = add_207 = None
    copy__121: "f32[1152]" = torch.ops.aten.copy_.default(primals_335, add_209);  primals_335 = add_209 = None
    copy__122: "f32[1152]" = torch.ops.aten.copy_.default(primals_336, add_210);  primals_336 = add_210 = None
    copy__123: "i64[]" = torch.ops.aten.copy_.default(primals_337, add_212);  primals_337 = add_212 = None
    copy__124: "f32[192]" = torch.ops.aten.copy_.default(primals_338, add_214);  primals_338 = add_214 = None
    copy__125: "f32[192]" = torch.ops.aten.copy_.default(primals_339, add_215);  primals_339 = add_215 = None
    copy__126: "i64[]" = torch.ops.aten.copy_.default(primals_340, add_218);  primals_340 = add_218 = None
    copy__127: "f32[1152]" = torch.ops.aten.copy_.default(primals_341, add_220);  primals_341 = add_220 = None
    copy__128: "f32[1152]" = torch.ops.aten.copy_.default(primals_342, add_221);  primals_342 = add_221 = None
    copy__129: "i64[]" = torch.ops.aten.copy_.default(primals_343, add_223);  primals_343 = add_223 = None
    copy__130: "f32[1152]" = torch.ops.aten.copy_.default(primals_344, add_225);  primals_344 = add_225 = None
    copy__131: "f32[1152]" = torch.ops.aten.copy_.default(primals_345, add_226);  primals_345 = add_226 = None
    copy__132: "i64[]" = torch.ops.aten.copy_.default(primals_346, add_228);  primals_346 = add_228 = None
    copy__133: "f32[192]" = torch.ops.aten.copy_.default(primals_347, add_230);  primals_347 = add_230 = None
    copy__134: "f32[192]" = torch.ops.aten.copy_.default(primals_348, add_231);  primals_348 = add_231 = None
    copy__135: "i64[]" = torch.ops.aten.copy_.default(primals_349, add_234);  primals_349 = add_234 = None
    copy__136: "f32[1152]" = torch.ops.aten.copy_.default(primals_350, add_236);  primals_350 = add_236 = None
    copy__137: "f32[1152]" = torch.ops.aten.copy_.default(primals_351, add_237);  primals_351 = add_237 = None
    copy__138: "i64[]" = torch.ops.aten.copy_.default(primals_352, add_239);  primals_352 = add_239 = None
    copy__139: "f32[1152]" = torch.ops.aten.copy_.default(primals_353, add_241);  primals_353 = add_241 = None
    copy__140: "f32[1152]" = torch.ops.aten.copy_.default(primals_354, add_242);  primals_354 = add_242 = None
    copy__141: "i64[]" = torch.ops.aten.copy_.default(primals_355, add_244);  primals_355 = add_244 = None
    copy__142: "f32[320]" = torch.ops.aten.copy_.default(primals_356, add_246);  primals_356 = add_246 = None
    copy__143: "f32[320]" = torch.ops.aten.copy_.default(primals_357, add_247);  primals_357 = add_247 = None
    copy__144: "i64[]" = torch.ops.aten.copy_.default(primals_358, add_249);  primals_358 = add_249 = None
    copy__145: "f32[1280]" = torch.ops.aten.copy_.default(primals_359, add_251);  primals_359 = add_251 = None
    copy__146: "f32[1280]" = torch.ops.aten.copy_.default(primals_360, add_252);  primals_360 = add_252 = None
    return pytree.tree_unflatten([addmm, getitem_339, mul_1059, sum_114, mul_1047, sum_112, mul_1028, sum_109, mul_1019, sum_107, getitem_321, mul_1007, sum_105, mul_988, sum_102, mul_979, sum_100, mul_967, sum_98, mul_948, sum_95, mul_939, sum_93, getitem_291, mul_927, sum_91, mul_908, sum_88, mul_899, sum_86, mul_887, sum_84, mul_868, sum_81, mul_859, sum_79, getitem_261, mul_847, sum_77, mul_828, sum_74, mul_819, sum_72, mul_807, sum_70, mul_788, sum_67, mul_779, sum_65, mul_767, sum_63, mul_748, sum_60, mul_739, sum_58, mul_727, sum_56, mul_708, sum_53, mul_699, sum_51, mul_687, sum_49, mul_668, sum_46, mul_659, sum_44, mul_647, sum_42, mul_628, sum_39, mul_619, sum_37, getitem_171, mul_607, sum_35, mul_588, sum_32, mul_579, sum_30, mul_567, sum_28, mul_548, sum_25, mul_539, sum_23, mul_527, sum_21, mul_508, sum_18, mul_499, sum_16, mul_487, sum_14, mul_468, sum_11, mul_459, sum_9, mul_447, sum_7, mul_428, sum_4, mul_419, sum_2, getitem_336, getitem_333, getitem_334, getitem_330, getitem_331, getitem_327, getitem_324, getitem_318, getitem_319, getitem_315, getitem_316, getitem_312, getitem_309, getitem_306, getitem_303, getitem_304, getitem_300, getitem_301, getitem_297, getitem_294, getitem_288, getitem_289, getitem_285, getitem_286, getitem_282, getitem_279, getitem_276, getitem_273, getitem_274, getitem_270, getitem_271, getitem_267, getitem_264, getitem_258, getitem_259, getitem_255, getitem_256, getitem_252, getitem_249, getitem_246, getitem_243, getitem_244, getitem_240, getitem_241, getitem_237, getitem_234, getitem_231, getitem_228, getitem_229, getitem_225, getitem_226, getitem_222, getitem_219, getitem_216, getitem_213, getitem_214, getitem_210, getitem_211, getitem_207, getitem_204, getitem_201, getitem_198, getitem_199, getitem_195, getitem_196, getitem_192, getitem_189, getitem_186, getitem_183, getitem_184, getitem_180, getitem_181, getitem_177, getitem_174, getitem_168, getitem_169, getitem_165, getitem_166, getitem_162, getitem_159, getitem_156, getitem_153, getitem_154, getitem_150, getitem_151, getitem_147, getitem_144, getitem_141, getitem_138, getitem_139, getitem_135, getitem_136, getitem_132, getitem_129, getitem_126, getitem_123, getitem_124, getitem_120, getitem_121, getitem_117, getitem_114, getitem_111, getitem_108, getitem_109, getitem_105, getitem_106, getitem_102, getitem_99, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    