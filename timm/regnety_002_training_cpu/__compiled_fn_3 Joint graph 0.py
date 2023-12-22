from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[32]"; primals_2: "f32[32]"; primals_3: "f32[24]"; primals_4: "f32[24]"; primals_5: "f32[24]"; primals_6: "f32[24]"; primals_7: "f32[24]"; primals_8: "f32[24]"; primals_9: "f32[24]"; primals_10: "f32[24]"; primals_11: "f32[56]"; primals_12: "f32[56]"; primals_13: "f32[56]"; primals_14: "f32[56]"; primals_15: "f32[56]"; primals_16: "f32[56]"; primals_17: "f32[56]"; primals_18: "f32[56]"; primals_19: "f32[152]"; primals_20: "f32[152]"; primals_21: "f32[152]"; primals_22: "f32[152]"; primals_23: "f32[152]"; primals_24: "f32[152]"; primals_25: "f32[152]"; primals_26: "f32[152]"; primals_27: "f32[152]"; primals_28: "f32[152]"; primals_29: "f32[152]"; primals_30: "f32[152]"; primals_31: "f32[152]"; primals_32: "f32[152]"; primals_33: "f32[152]"; primals_34: "f32[152]"; primals_35: "f32[152]"; primals_36: "f32[152]"; primals_37: "f32[152]"; primals_38: "f32[152]"; primals_39: "f32[152]"; primals_40: "f32[152]"; primals_41: "f32[152]"; primals_42: "f32[152]"; primals_43: "f32[152]"; primals_44: "f32[152]"; primals_45: "f32[368]"; primals_46: "f32[368]"; primals_47: "f32[368]"; primals_48: "f32[368]"; primals_49: "f32[368]"; primals_50: "f32[368]"; primals_51: "f32[368]"; primals_52: "f32[368]"; primals_53: "f32[368]"; primals_54: "f32[368]"; primals_55: "f32[368]"; primals_56: "f32[368]"; primals_57: "f32[368]"; primals_58: "f32[368]"; primals_59: "f32[368]"; primals_60: "f32[368]"; primals_61: "f32[368]"; primals_62: "f32[368]"; primals_63: "f32[368]"; primals_64: "f32[368]"; primals_65: "f32[368]"; primals_66: "f32[368]"; primals_67: "f32[368]"; primals_68: "f32[368]"; primals_69: "f32[368]"; primals_70: "f32[368]"; primals_71: "f32[368]"; primals_72: "f32[368]"; primals_73: "f32[368]"; primals_74: "f32[368]"; primals_75: "f32[368]"; primals_76: "f32[368]"; primals_77: "f32[368]"; primals_78: "f32[368]"; primals_79: "f32[368]"; primals_80: "f32[368]"; primals_81: "f32[368]"; primals_82: "f32[368]"; primals_83: "f32[368]"; primals_84: "f32[368]"; primals_85: "f32[368]"; primals_86: "f32[368]"; primals_87: "f32[368]"; primals_88: "f32[368]"; primals_89: "f32[32, 3, 3, 3]"; primals_90: "f32[24, 32, 1, 1]"; primals_91: "f32[24, 8, 3, 3]"; primals_92: "f32[8, 24, 1, 1]"; primals_93: "f32[8]"; primals_94: "f32[24, 8, 1, 1]"; primals_95: "f32[24]"; primals_96: "f32[24, 24, 1, 1]"; primals_97: "f32[24, 32, 1, 1]"; primals_98: "f32[56, 24, 1, 1]"; primals_99: "f32[56, 8, 3, 3]"; primals_100: "f32[6, 56, 1, 1]"; primals_101: "f32[6]"; primals_102: "f32[56, 6, 1, 1]"; primals_103: "f32[56]"; primals_104: "f32[56, 56, 1, 1]"; primals_105: "f32[56, 24, 1, 1]"; primals_106: "f32[152, 56, 1, 1]"; primals_107: "f32[152, 8, 3, 3]"; primals_108: "f32[14, 152, 1, 1]"; primals_109: "f32[14]"; primals_110: "f32[152, 14, 1, 1]"; primals_111: "f32[152]"; primals_112: "f32[152, 152, 1, 1]"; primals_113: "f32[152, 56, 1, 1]"; primals_114: "f32[152, 152, 1, 1]"; primals_115: "f32[152, 8, 3, 3]"; primals_116: "f32[38, 152, 1, 1]"; primals_117: "f32[38]"; primals_118: "f32[152, 38, 1, 1]"; primals_119: "f32[152]"; primals_120: "f32[152, 152, 1, 1]"; primals_121: "f32[152, 152, 1, 1]"; primals_122: "f32[152, 8, 3, 3]"; primals_123: "f32[38, 152, 1, 1]"; primals_124: "f32[38]"; primals_125: "f32[152, 38, 1, 1]"; primals_126: "f32[152]"; primals_127: "f32[152, 152, 1, 1]"; primals_128: "f32[152, 152, 1, 1]"; primals_129: "f32[152, 8, 3, 3]"; primals_130: "f32[38, 152, 1, 1]"; primals_131: "f32[38]"; primals_132: "f32[152, 38, 1, 1]"; primals_133: "f32[152]"; primals_134: "f32[152, 152, 1, 1]"; primals_135: "f32[368, 152, 1, 1]"; primals_136: "f32[368, 8, 3, 3]"; primals_137: "f32[38, 368, 1, 1]"; primals_138: "f32[38]"; primals_139: "f32[368, 38, 1, 1]"; primals_140: "f32[368]"; primals_141: "f32[368, 368, 1, 1]"; primals_142: "f32[368, 152, 1, 1]"; primals_143: "f32[368, 368, 1, 1]"; primals_144: "f32[368, 8, 3, 3]"; primals_145: "f32[92, 368, 1, 1]"; primals_146: "f32[92]"; primals_147: "f32[368, 92, 1, 1]"; primals_148: "f32[368]"; primals_149: "f32[368, 368, 1, 1]"; primals_150: "f32[368, 368, 1, 1]"; primals_151: "f32[368, 8, 3, 3]"; primals_152: "f32[92, 368, 1, 1]"; primals_153: "f32[92]"; primals_154: "f32[368, 92, 1, 1]"; primals_155: "f32[368]"; primals_156: "f32[368, 368, 1, 1]"; primals_157: "f32[368, 368, 1, 1]"; primals_158: "f32[368, 8, 3, 3]"; primals_159: "f32[92, 368, 1, 1]"; primals_160: "f32[92]"; primals_161: "f32[368, 92, 1, 1]"; primals_162: "f32[368]"; primals_163: "f32[368, 368, 1, 1]"; primals_164: "f32[368, 368, 1, 1]"; primals_165: "f32[368, 8, 3, 3]"; primals_166: "f32[92, 368, 1, 1]"; primals_167: "f32[92]"; primals_168: "f32[368, 92, 1, 1]"; primals_169: "f32[368]"; primals_170: "f32[368, 368, 1, 1]"; primals_171: "f32[368, 368, 1, 1]"; primals_172: "f32[368, 8, 3, 3]"; primals_173: "f32[92, 368, 1, 1]"; primals_174: "f32[92]"; primals_175: "f32[368, 92, 1, 1]"; primals_176: "f32[368]"; primals_177: "f32[368, 368, 1, 1]"; primals_178: "f32[368, 368, 1, 1]"; primals_179: "f32[368, 8, 3, 3]"; primals_180: "f32[92, 368, 1, 1]"; primals_181: "f32[92]"; primals_182: "f32[368, 92, 1, 1]"; primals_183: "f32[368]"; primals_184: "f32[368, 368, 1, 1]"; primals_185: "f32[1000, 368]"; primals_186: "f32[1000]"; primals_187: "i64[]"; primals_188: "f32[32]"; primals_189: "f32[32]"; primals_190: "i64[]"; primals_191: "f32[24]"; primals_192: "f32[24]"; primals_193: "i64[]"; primals_194: "f32[24]"; primals_195: "f32[24]"; primals_196: "i64[]"; primals_197: "f32[24]"; primals_198: "f32[24]"; primals_199: "i64[]"; primals_200: "f32[24]"; primals_201: "f32[24]"; primals_202: "i64[]"; primals_203: "f32[56]"; primals_204: "f32[56]"; primals_205: "i64[]"; primals_206: "f32[56]"; primals_207: "f32[56]"; primals_208: "i64[]"; primals_209: "f32[56]"; primals_210: "f32[56]"; primals_211: "i64[]"; primals_212: "f32[56]"; primals_213: "f32[56]"; primals_214: "i64[]"; primals_215: "f32[152]"; primals_216: "f32[152]"; primals_217: "i64[]"; primals_218: "f32[152]"; primals_219: "f32[152]"; primals_220: "i64[]"; primals_221: "f32[152]"; primals_222: "f32[152]"; primals_223: "i64[]"; primals_224: "f32[152]"; primals_225: "f32[152]"; primals_226: "i64[]"; primals_227: "f32[152]"; primals_228: "f32[152]"; primals_229: "i64[]"; primals_230: "f32[152]"; primals_231: "f32[152]"; primals_232: "i64[]"; primals_233: "f32[152]"; primals_234: "f32[152]"; primals_235: "i64[]"; primals_236: "f32[152]"; primals_237: "f32[152]"; primals_238: "i64[]"; primals_239: "f32[152]"; primals_240: "f32[152]"; primals_241: "i64[]"; primals_242: "f32[152]"; primals_243: "f32[152]"; primals_244: "i64[]"; primals_245: "f32[152]"; primals_246: "f32[152]"; primals_247: "i64[]"; primals_248: "f32[152]"; primals_249: "f32[152]"; primals_250: "i64[]"; primals_251: "f32[152]"; primals_252: "f32[152]"; primals_253: "i64[]"; primals_254: "f32[368]"; primals_255: "f32[368]"; primals_256: "i64[]"; primals_257: "f32[368]"; primals_258: "f32[368]"; primals_259: "i64[]"; primals_260: "f32[368]"; primals_261: "f32[368]"; primals_262: "i64[]"; primals_263: "f32[368]"; primals_264: "f32[368]"; primals_265: "i64[]"; primals_266: "f32[368]"; primals_267: "f32[368]"; primals_268: "i64[]"; primals_269: "f32[368]"; primals_270: "f32[368]"; primals_271: "i64[]"; primals_272: "f32[368]"; primals_273: "f32[368]"; primals_274: "i64[]"; primals_275: "f32[368]"; primals_276: "f32[368]"; primals_277: "i64[]"; primals_278: "f32[368]"; primals_279: "f32[368]"; primals_280: "i64[]"; primals_281: "f32[368]"; primals_282: "f32[368]"; primals_283: "i64[]"; primals_284: "f32[368]"; primals_285: "f32[368]"; primals_286: "i64[]"; primals_287: "f32[368]"; primals_288: "f32[368]"; primals_289: "i64[]"; primals_290: "f32[368]"; primals_291: "f32[368]"; primals_292: "i64[]"; primals_293: "f32[368]"; primals_294: "f32[368]"; primals_295: "i64[]"; primals_296: "f32[368]"; primals_297: "f32[368]"; primals_298: "i64[]"; primals_299: "f32[368]"; primals_300: "f32[368]"; primals_301: "i64[]"; primals_302: "f32[368]"; primals_303: "f32[368]"; primals_304: "i64[]"; primals_305: "f32[368]"; primals_306: "f32[368]"; primals_307: "i64[]"; primals_308: "f32[368]"; primals_309: "f32[368]"; primals_310: "i64[]"; primals_311: "f32[368]"; primals_312: "f32[368]"; primals_313: "i64[]"; primals_314: "f32[368]"; primals_315: "f32[368]"; primals_316: "i64[]"; primals_317: "f32[368]"; primals_318: "f32[368]"; primals_319: "f32[8, 3, 224, 224]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution: "f32[8, 32, 112, 112]" = torch.ops.aten.convolution.default(primals_319, primals_89, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_187, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
    getitem: "f32[1, 32, 1, 1]" = var_mean[0]
    getitem_1: "f32[1, 32, 1, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
    rsqrt: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
    mul: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    squeeze: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
    squeeze_1: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
    mul_1: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
    mul_2: "f32[32]" = torch.ops.aten.mul.Tensor(primals_188, 0.9)
    add_2: "f32[32]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[32]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.00000996502277);  squeeze_2 = None
    mul_4: "f32[32]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[32]" = torch.ops.aten.mul.Tensor(primals_189, 0.9)
    add_3: "f32[32]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_1, -1)
    unsqueeze_1: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1);  primals_2 = None
    unsqueeze_3: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu: "f32[8, 32, 112, 112]" = torch.ops.aten.relu.default(add_4);  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_1: "f32[8, 24, 112, 112]" = torch.ops.aten.convolution.default(relu, primals_90, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_190, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[1, 24, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 24, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
    rsqrt_1: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_1: "f32[8, 24, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
    mul_7: "f32[8, 24, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
    squeeze_4: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_8: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_9: "f32[24]" = torch.ops.aten.mul.Tensor(primals_191, 0.9)
    add_7: "f32[24]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
    squeeze_5: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_10: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.00000996502277);  squeeze_5 = None
    mul_11: "f32[24]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
    mul_12: "f32[24]" = torch.ops.aten.mul.Tensor(primals_192, 0.9)
    add_8: "f32[24]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
    unsqueeze_4: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1)
    unsqueeze_5: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_13: "f32[8, 24, 112, 112]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
    unsqueeze_6: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_4, -1);  primals_4 = None
    unsqueeze_7: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_9: "f32[8, 24, 112, 112]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_1: "f32[8, 24, 112, 112]" = torch.ops.aten.relu.default(add_9);  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_2: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(relu_1, primals_91, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_193, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 24, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 24, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
    rsqrt_2: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_2: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
    mul_14: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_15: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_16: "f32[24]" = torch.ops.aten.mul.Tensor(primals_194, 0.9)
    add_12: "f32[24]" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
    squeeze_8: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_17: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0000398612827361);  squeeze_8 = None
    mul_18: "f32[24]" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
    mul_19: "f32[24]" = torch.ops.aten.mul.Tensor(primals_195, 0.9)
    add_13: "f32[24]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
    unsqueeze_8: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_9: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_20: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
    unsqueeze_10: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_11: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_14: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_2: "f32[8, 24, 56, 56]" = torch.ops.aten.relu.default(add_14);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean: "f32[8, 24, 1, 1]" = torch.ops.aten.mean.dim(relu_2, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_3: "f32[8, 8, 1, 1]" = torch.ops.aten.convolution.default(mean, primals_92, primals_93, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_3: "f32[8, 8, 1, 1]" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_4: "f32[8, 24, 1, 1]" = torch.ops.aten.convolution.default(relu_3, primals_94, primals_95, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid: "f32[8, 24, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_4);  convolution_4 = None
    alias_4: "f32[8, 24, 1, 1]" = torch.ops.aten.alias.default(sigmoid)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_21: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(relu_2, sigmoid)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_5: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(mul_21, primals_96, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_15: "i64[]" = torch.ops.aten.add.Tensor(primals_196, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_3 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
    getitem_6: "f32[1, 24, 1, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 24, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_16: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
    rsqrt_3: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_3: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_7)
    mul_22: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
    squeeze_10: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_23: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_24: "f32[24]" = torch.ops.aten.mul.Tensor(primals_197, 0.9)
    add_17: "f32[24]" = torch.ops.aten.add.Tensor(mul_23, mul_24);  mul_23 = mul_24 = None
    squeeze_11: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
    mul_25: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0000398612827361);  squeeze_11 = None
    mul_26: "f32[24]" = torch.ops.aten.mul.Tensor(mul_25, 0.1);  mul_25 = None
    mul_27: "f32[24]" = torch.ops.aten.mul.Tensor(primals_198, 0.9)
    add_18: "f32[24]" = torch.ops.aten.add.Tensor(mul_26, mul_27);  mul_26 = mul_27 = None
    unsqueeze_12: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1)
    unsqueeze_13: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_28: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_13);  mul_22 = unsqueeze_13 = None
    unsqueeze_14: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1);  primals_8 = None
    unsqueeze_15: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_19: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_28, unsqueeze_15);  mul_28 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_6: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(relu, primals_97, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_20: "i64[]" = torch.ops.aten.add.Tensor(primals_199, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_4 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
    getitem_8: "f32[1, 24, 1, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 24, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_21: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
    rsqrt_4: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_4: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_9)
    mul_29: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
    squeeze_13: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
    mul_30: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_31: "f32[24]" = torch.ops.aten.mul.Tensor(primals_200, 0.9)
    add_22: "f32[24]" = torch.ops.aten.add.Tensor(mul_30, mul_31);  mul_30 = mul_31 = None
    squeeze_14: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
    mul_32: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000398612827361);  squeeze_14 = None
    mul_33: "f32[24]" = torch.ops.aten.mul.Tensor(mul_32, 0.1);  mul_32 = None
    mul_34: "f32[24]" = torch.ops.aten.mul.Tensor(primals_201, 0.9)
    add_23: "f32[24]" = torch.ops.aten.add.Tensor(mul_33, mul_34);  mul_33 = mul_34 = None
    unsqueeze_16: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1)
    unsqueeze_17: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    mul_35: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_29, unsqueeze_17);  mul_29 = unsqueeze_17 = None
    unsqueeze_18: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_10, -1);  primals_10 = None
    unsqueeze_19: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    add_24: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_19);  mul_35 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_25: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(add_19, add_24);  add_19 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_4: "f32[8, 24, 56, 56]" = torch.ops.aten.relu.default(add_25);  add_25 = None
    alias_5: "f32[8, 24, 56, 56]" = torch.ops.aten.alias.default(relu_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_7: "f32[8, 56, 56, 56]" = torch.ops.aten.convolution.default(relu_4, primals_98, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_26: "i64[]" = torch.ops.aten.add.Tensor(primals_202, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_5 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
    getitem_10: "f32[1, 56, 1, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 56, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_27: "f32[1, 56, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
    rsqrt_5: "f32[1, 56, 1, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    sub_5: "f32[8, 56, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_11)
    mul_36: "f32[8, 56, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_15: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
    squeeze_16: "f32[56]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    mul_37: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_38: "f32[56]" = torch.ops.aten.mul.Tensor(primals_203, 0.9)
    add_28: "f32[56]" = torch.ops.aten.add.Tensor(mul_37, mul_38);  mul_37 = mul_38 = None
    squeeze_17: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
    mul_39: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000398612827361);  squeeze_17 = None
    mul_40: "f32[56]" = torch.ops.aten.mul.Tensor(mul_39, 0.1);  mul_39 = None
    mul_41: "f32[56]" = torch.ops.aten.mul.Tensor(primals_204, 0.9)
    add_29: "f32[56]" = torch.ops.aten.add.Tensor(mul_40, mul_41);  mul_40 = mul_41 = None
    unsqueeze_20: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_21: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_42: "f32[8, 56, 56, 56]" = torch.ops.aten.mul.Tensor(mul_36, unsqueeze_21);  mul_36 = unsqueeze_21 = None
    unsqueeze_22: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_23: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_30: "f32[8, 56, 56, 56]" = torch.ops.aten.add.Tensor(mul_42, unsqueeze_23);  mul_42 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_5: "f32[8, 56, 56, 56]" = torch.ops.aten.relu.default(add_30);  add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_8: "f32[8, 56, 28, 28]" = torch.ops.aten.convolution.default(relu_5, primals_99, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_31: "i64[]" = torch.ops.aten.add.Tensor(primals_205, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_6 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
    getitem_12: "f32[1, 56, 1, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 56, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_32: "f32[1, 56, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
    rsqrt_6: "f32[1, 56, 1, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_6: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_13)
    mul_43: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_18: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
    squeeze_19: "f32[56]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
    mul_44: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_45: "f32[56]" = torch.ops.aten.mul.Tensor(primals_206, 0.9)
    add_33: "f32[56]" = torch.ops.aten.add.Tensor(mul_44, mul_45);  mul_44 = mul_45 = None
    squeeze_20: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
    mul_46: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0001594642002871);  squeeze_20 = None
    mul_47: "f32[56]" = torch.ops.aten.mul.Tensor(mul_46, 0.1);  mul_46 = None
    mul_48: "f32[56]" = torch.ops.aten.mul.Tensor(primals_207, 0.9)
    add_34: "f32[56]" = torch.ops.aten.add.Tensor(mul_47, mul_48);  mul_47 = mul_48 = None
    unsqueeze_24: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1)
    unsqueeze_25: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    mul_49: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_25);  mul_43 = unsqueeze_25 = None
    unsqueeze_26: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1);  primals_14 = None
    unsqueeze_27: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    add_35: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(mul_49, unsqueeze_27);  mul_49 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_6: "f32[8, 56, 28, 28]" = torch.ops.aten.relu.default(add_35);  add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_1: "f32[8, 56, 1, 1]" = torch.ops.aten.mean.dim(relu_6, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_9: "f32[8, 6, 1, 1]" = torch.ops.aten.convolution.default(mean_1, primals_100, primals_101, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_7: "f32[8, 6, 1, 1]" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_10: "f32[8, 56, 1, 1]" = torch.ops.aten.convolution.default(relu_7, primals_102, primals_103, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_1: "f32[8, 56, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_10);  convolution_10 = None
    alias_9: "f32[8, 56, 1, 1]" = torch.ops.aten.alias.default(sigmoid_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_50: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(relu_6, sigmoid_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_11: "f32[8, 56, 28, 28]" = torch.ops.aten.convolution.default(mul_50, primals_104, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_36: "i64[]" = torch.ops.aten.add.Tensor(primals_208, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_7 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
    getitem_14: "f32[1, 56, 1, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 56, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_37: "f32[1, 56, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
    rsqrt_7: "f32[1, 56, 1, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_7: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_15)
    mul_51: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_21: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
    squeeze_22: "f32[56]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
    mul_52: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_53: "f32[56]" = torch.ops.aten.mul.Tensor(primals_209, 0.9)
    add_38: "f32[56]" = torch.ops.aten.add.Tensor(mul_52, mul_53);  mul_52 = mul_53 = None
    squeeze_23: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
    mul_54: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0001594642002871);  squeeze_23 = None
    mul_55: "f32[56]" = torch.ops.aten.mul.Tensor(mul_54, 0.1);  mul_54 = None
    mul_56: "f32[56]" = torch.ops.aten.mul.Tensor(primals_210, 0.9)
    add_39: "f32[56]" = torch.ops.aten.add.Tensor(mul_55, mul_56);  mul_55 = mul_56 = None
    unsqueeze_28: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1)
    unsqueeze_29: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_57: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(mul_51, unsqueeze_29);  mul_51 = unsqueeze_29 = None
    unsqueeze_30: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1);  primals_16 = None
    unsqueeze_31: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_40: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(mul_57, unsqueeze_31);  mul_57 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_12: "f32[8, 56, 28, 28]" = torch.ops.aten.convolution.default(relu_4, primals_105, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_41: "i64[]" = torch.ops.aten.add.Tensor(primals_211, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_8 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
    getitem_16: "f32[1, 56, 1, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 56, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_42: "f32[1, 56, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
    rsqrt_8: "f32[1, 56, 1, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_8: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_17)
    mul_58: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_24: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
    squeeze_25: "f32[56]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
    mul_59: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_60: "f32[56]" = torch.ops.aten.mul.Tensor(primals_212, 0.9)
    add_43: "f32[56]" = torch.ops.aten.add.Tensor(mul_59, mul_60);  mul_59 = mul_60 = None
    squeeze_26: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
    mul_61: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0001594642002871);  squeeze_26 = None
    mul_62: "f32[56]" = torch.ops.aten.mul.Tensor(mul_61, 0.1);  mul_61 = None
    mul_63: "f32[56]" = torch.ops.aten.mul.Tensor(primals_213, 0.9)
    add_44: "f32[56]" = torch.ops.aten.add.Tensor(mul_62, mul_63);  mul_62 = mul_63 = None
    unsqueeze_32: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_33: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    mul_64: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_33);  mul_58 = unsqueeze_33 = None
    unsqueeze_34: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_35: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    add_45: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(mul_64, unsqueeze_35);  mul_64 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_46: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(add_40, add_45);  add_40 = add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_8: "f32[8, 56, 28, 28]" = torch.ops.aten.relu.default(add_46);  add_46 = None
    alias_10: "f32[8, 56, 28, 28]" = torch.ops.aten.alias.default(relu_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_13: "f32[8, 152, 28, 28]" = torch.ops.aten.convolution.default(relu_8, primals_106, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_47: "i64[]" = torch.ops.aten.add.Tensor(primals_214, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_9 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
    getitem_18: "f32[1, 152, 1, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 152, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_48: "f32[1, 152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
    rsqrt_9: "f32[1, 152, 1, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
    sub_9: "f32[8, 152, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_19)
    mul_65: "f32[8, 152, 28, 28]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_27: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
    squeeze_28: "f32[152]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_66: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_67: "f32[152]" = torch.ops.aten.mul.Tensor(primals_215, 0.9)
    add_49: "f32[152]" = torch.ops.aten.add.Tensor(mul_66, mul_67);  mul_66 = mul_67 = None
    squeeze_29: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
    mul_68: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0001594642002871);  squeeze_29 = None
    mul_69: "f32[152]" = torch.ops.aten.mul.Tensor(mul_68, 0.1);  mul_68 = None
    mul_70: "f32[152]" = torch.ops.aten.mul.Tensor(primals_216, 0.9)
    add_50: "f32[152]" = torch.ops.aten.add.Tensor(mul_69, mul_70);  mul_69 = mul_70 = None
    unsqueeze_36: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
    unsqueeze_37: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_71: "f32[8, 152, 28, 28]" = torch.ops.aten.mul.Tensor(mul_65, unsqueeze_37);  mul_65 = unsqueeze_37 = None
    unsqueeze_38: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
    unsqueeze_39: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_51: "f32[8, 152, 28, 28]" = torch.ops.aten.add.Tensor(mul_71, unsqueeze_39);  mul_71 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_9: "f32[8, 152, 28, 28]" = torch.ops.aten.relu.default(add_51);  add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_14: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(relu_9, primals_107, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_52: "i64[]" = torch.ops.aten.add.Tensor(primals_217, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_10 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[1, 152, 1, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 152, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_53: "f32[1, 152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
    rsqrt_10: "f32[1, 152, 1, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    sub_10: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_21)
    mul_72: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_30: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
    squeeze_31: "f32[152]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
    mul_73: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_74: "f32[152]" = torch.ops.aten.mul.Tensor(primals_218, 0.9)
    add_54: "f32[152]" = torch.ops.aten.add.Tensor(mul_73, mul_74);  mul_73 = mul_74 = None
    squeeze_32: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
    mul_75: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0006381620931717);  squeeze_32 = None
    mul_76: "f32[152]" = torch.ops.aten.mul.Tensor(mul_75, 0.1);  mul_75 = None
    mul_77: "f32[152]" = torch.ops.aten.mul.Tensor(primals_219, 0.9)
    add_55: "f32[152]" = torch.ops.aten.add.Tensor(mul_76, mul_77);  mul_76 = mul_77 = None
    unsqueeze_40: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1)
    unsqueeze_41: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    mul_78: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_72, unsqueeze_41);  mul_72 = unsqueeze_41 = None
    unsqueeze_42: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1);  primals_22 = None
    unsqueeze_43: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    add_56: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_78, unsqueeze_43);  mul_78 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_10: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_56);  add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_2: "f32[8, 152, 1, 1]" = torch.ops.aten.mean.dim(relu_10, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_15: "f32[8, 14, 1, 1]" = torch.ops.aten.convolution.default(mean_2, primals_108, primals_109, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_11: "f32[8, 14, 1, 1]" = torch.ops.aten.relu.default(convolution_15);  convolution_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_16: "f32[8, 152, 1, 1]" = torch.ops.aten.convolution.default(relu_11, primals_110, primals_111, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_2: "f32[8, 152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_16);  convolution_16 = None
    alias_14: "f32[8, 152, 1, 1]" = torch.ops.aten.alias.default(sigmoid_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_79: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(relu_10, sigmoid_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_17: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(mul_79, primals_112, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_57: "i64[]" = torch.ops.aten.add.Tensor(primals_220, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_11 = torch.ops.aten.var_mean.correction(convolution_17, [0, 2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[1, 152, 1, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 152, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_58: "f32[1, 152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
    rsqrt_11: "f32[1, 152, 1, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_11: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_17, getitem_23)
    mul_80: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    squeeze_33: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
    squeeze_34: "f32[152]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
    mul_81: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_82: "f32[152]" = torch.ops.aten.mul.Tensor(primals_221, 0.9)
    add_59: "f32[152]" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
    squeeze_35: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
    mul_83: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0006381620931717);  squeeze_35 = None
    mul_84: "f32[152]" = torch.ops.aten.mul.Tensor(mul_83, 0.1);  mul_83 = None
    mul_85: "f32[152]" = torch.ops.aten.mul.Tensor(primals_222, 0.9)
    add_60: "f32[152]" = torch.ops.aten.add.Tensor(mul_84, mul_85);  mul_84 = mul_85 = None
    unsqueeze_44: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_45: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_86: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_80, unsqueeze_45);  mul_80 = unsqueeze_45 = None
    unsqueeze_46: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_47: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_61: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_86, unsqueeze_47);  mul_86 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_18: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(relu_8, primals_113, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_62: "i64[]" = torch.ops.aten.add.Tensor(primals_223, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_12 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
    getitem_24: "f32[1, 152, 1, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 152, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_63: "f32[1, 152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
    rsqrt_12: "f32[1, 152, 1, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_12: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_25)
    mul_87: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_36: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
    squeeze_37: "f32[152]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
    mul_88: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_89: "f32[152]" = torch.ops.aten.mul.Tensor(primals_224, 0.9)
    add_64: "f32[152]" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
    squeeze_38: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
    mul_90: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0006381620931717);  squeeze_38 = None
    mul_91: "f32[152]" = torch.ops.aten.mul.Tensor(mul_90, 0.1);  mul_90 = None
    mul_92: "f32[152]" = torch.ops.aten.mul.Tensor(primals_225, 0.9)
    add_65: "f32[152]" = torch.ops.aten.add.Tensor(mul_91, mul_92);  mul_91 = mul_92 = None
    unsqueeze_48: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1)
    unsqueeze_49: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    mul_93: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_87, unsqueeze_49);  mul_87 = unsqueeze_49 = None
    unsqueeze_50: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1);  primals_26 = None
    unsqueeze_51: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    add_66: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_93, unsqueeze_51);  mul_93 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_67: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(add_61, add_66);  add_61 = add_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_12: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_67);  add_67 = None
    alias_15: "f32[8, 152, 14, 14]" = torch.ops.aten.alias.default(relu_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_19: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(relu_12, primals_114, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_68: "i64[]" = torch.ops.aten.add.Tensor(primals_226, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_13 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
    getitem_26: "f32[1, 152, 1, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 152, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_69: "f32[1, 152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
    rsqrt_13: "f32[1, 152, 1, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
    sub_13: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_27)
    mul_94: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_39: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
    squeeze_40: "f32[152]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_95: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_96: "f32[152]" = torch.ops.aten.mul.Tensor(primals_227, 0.9)
    add_70: "f32[152]" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
    squeeze_41: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
    mul_97: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0006381620931717);  squeeze_41 = None
    mul_98: "f32[152]" = torch.ops.aten.mul.Tensor(mul_97, 0.1);  mul_97 = None
    mul_99: "f32[152]" = torch.ops.aten.mul.Tensor(primals_228, 0.9)
    add_71: "f32[152]" = torch.ops.aten.add.Tensor(mul_98, mul_99);  mul_98 = mul_99 = None
    unsqueeze_52: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1)
    unsqueeze_53: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_100: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_94, unsqueeze_53);  mul_94 = unsqueeze_53 = None
    unsqueeze_54: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1);  primals_28 = None
    unsqueeze_55: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_72: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_100, unsqueeze_55);  mul_100 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_13: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_72);  add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_20: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(relu_13, primals_115, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_73: "i64[]" = torch.ops.aten.add.Tensor(primals_229, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_14 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
    getitem_28: "f32[1, 152, 1, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 152, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_74: "f32[1, 152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
    rsqrt_14: "f32[1, 152, 1, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_14: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_29)
    mul_101: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_42: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
    squeeze_43: "f32[152]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
    mul_102: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_103: "f32[152]" = torch.ops.aten.mul.Tensor(primals_230, 0.9)
    add_75: "f32[152]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
    squeeze_44: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
    mul_104: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0006381620931717);  squeeze_44 = None
    mul_105: "f32[152]" = torch.ops.aten.mul.Tensor(mul_104, 0.1);  mul_104 = None
    mul_106: "f32[152]" = torch.ops.aten.mul.Tensor(primals_231, 0.9)
    add_76: "f32[152]" = torch.ops.aten.add.Tensor(mul_105, mul_106);  mul_105 = mul_106 = None
    unsqueeze_56: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_57: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    mul_107: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_101, unsqueeze_57);  mul_101 = unsqueeze_57 = None
    unsqueeze_58: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_59: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    add_77: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_107, unsqueeze_59);  mul_107 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_14: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_77);  add_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_3: "f32[8, 152, 1, 1]" = torch.ops.aten.mean.dim(relu_14, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_21: "f32[8, 38, 1, 1]" = torch.ops.aten.convolution.default(mean_3, primals_116, primals_117, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_15: "f32[8, 38, 1, 1]" = torch.ops.aten.relu.default(convolution_21);  convolution_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_22: "f32[8, 152, 1, 1]" = torch.ops.aten.convolution.default(relu_15, primals_118, primals_119, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_3: "f32[8, 152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_22);  convolution_22 = None
    alias_19: "f32[8, 152, 1, 1]" = torch.ops.aten.alias.default(sigmoid_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_108: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(relu_14, sigmoid_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_23: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(mul_108, primals_120, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_78: "i64[]" = torch.ops.aten.add.Tensor(primals_232, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_15 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
    getitem_30: "f32[1, 152, 1, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 152, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_79: "f32[1, 152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
    rsqrt_15: "f32[1, 152, 1, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
    sub_15: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_31)
    mul_109: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    squeeze_45: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
    squeeze_46: "f32[152]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
    mul_110: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_111: "f32[152]" = torch.ops.aten.mul.Tensor(primals_233, 0.9)
    add_80: "f32[152]" = torch.ops.aten.add.Tensor(mul_110, mul_111);  mul_110 = mul_111 = None
    squeeze_47: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
    mul_112: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0006381620931717);  squeeze_47 = None
    mul_113: "f32[152]" = torch.ops.aten.mul.Tensor(mul_112, 0.1);  mul_112 = None
    mul_114: "f32[152]" = torch.ops.aten.mul.Tensor(primals_234, 0.9)
    add_81: "f32[152]" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
    unsqueeze_60: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1)
    unsqueeze_61: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_115: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_109, unsqueeze_61);  mul_109 = unsqueeze_61 = None
    unsqueeze_62: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1);  primals_32 = None
    unsqueeze_63: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_82: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_115, unsqueeze_63);  mul_115 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_83: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(add_82, relu_12);  add_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_16: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_83);  add_83 = None
    alias_20: "f32[8, 152, 14, 14]" = torch.ops.aten.alias.default(relu_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_24: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(relu_16, primals_121, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_84: "i64[]" = torch.ops.aten.add.Tensor(primals_235, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_16 = torch.ops.aten.var_mean.correction(convolution_24, [0, 2, 3], correction = 0, keepdim = True)
    getitem_32: "f32[1, 152, 1, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 152, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_85: "f32[1, 152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
    rsqrt_16: "f32[1, 152, 1, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
    sub_16: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_24, getitem_33)
    mul_116: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_48: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
    squeeze_49: "f32[152]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
    mul_117: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_118: "f32[152]" = torch.ops.aten.mul.Tensor(primals_236, 0.9)
    add_86: "f32[152]" = torch.ops.aten.add.Tensor(mul_117, mul_118);  mul_117 = mul_118 = None
    squeeze_50: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
    mul_119: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0006381620931717);  squeeze_50 = None
    mul_120: "f32[152]" = torch.ops.aten.mul.Tensor(mul_119, 0.1);  mul_119 = None
    mul_121: "f32[152]" = torch.ops.aten.mul.Tensor(primals_237, 0.9)
    add_87: "f32[152]" = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
    unsqueeze_64: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1)
    unsqueeze_65: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    mul_122: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_116, unsqueeze_65);  mul_116 = unsqueeze_65 = None
    unsqueeze_66: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1);  primals_34 = None
    unsqueeze_67: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    add_88: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_122, unsqueeze_67);  mul_122 = unsqueeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_17: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_88);  add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_25: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(relu_17, primals_122, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_89: "i64[]" = torch.ops.aten.add.Tensor(primals_238, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_17 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
    getitem_34: "f32[1, 152, 1, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 152, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_90: "f32[1, 152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
    rsqrt_17: "f32[1, 152, 1, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
    sub_17: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_35)
    mul_123: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    squeeze_51: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
    squeeze_52: "f32[152]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
    mul_124: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_125: "f32[152]" = torch.ops.aten.mul.Tensor(primals_239, 0.9)
    add_91: "f32[152]" = torch.ops.aten.add.Tensor(mul_124, mul_125);  mul_124 = mul_125 = None
    squeeze_53: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
    mul_126: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0006381620931717);  squeeze_53 = None
    mul_127: "f32[152]" = torch.ops.aten.mul.Tensor(mul_126, 0.1);  mul_126 = None
    mul_128: "f32[152]" = torch.ops.aten.mul.Tensor(primals_240, 0.9)
    add_92: "f32[152]" = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
    unsqueeze_68: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_69: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_129: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_123, unsqueeze_69);  mul_123 = unsqueeze_69 = None
    unsqueeze_70: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_71: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_93: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_129, unsqueeze_71);  mul_129 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_18: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_93);  add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_4: "f32[8, 152, 1, 1]" = torch.ops.aten.mean.dim(relu_18, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_26: "f32[8, 38, 1, 1]" = torch.ops.aten.convolution.default(mean_4, primals_123, primals_124, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_19: "f32[8, 38, 1, 1]" = torch.ops.aten.relu.default(convolution_26);  convolution_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_27: "f32[8, 152, 1, 1]" = torch.ops.aten.convolution.default(relu_19, primals_125, primals_126, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_4: "f32[8, 152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_27);  convolution_27 = None
    alias_24: "f32[8, 152, 1, 1]" = torch.ops.aten.alias.default(sigmoid_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_130: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(relu_18, sigmoid_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_28: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(mul_130, primals_127, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_94: "i64[]" = torch.ops.aten.add.Tensor(primals_241, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_18 = torch.ops.aten.var_mean.correction(convolution_28, [0, 2, 3], correction = 0, keepdim = True)
    getitem_36: "f32[1, 152, 1, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 152, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_95: "f32[1, 152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
    rsqrt_18: "f32[1, 152, 1, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
    sub_18: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, getitem_37)
    mul_131: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    squeeze_54: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
    squeeze_55: "f32[152]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
    mul_132: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_133: "f32[152]" = torch.ops.aten.mul.Tensor(primals_242, 0.9)
    add_96: "f32[152]" = torch.ops.aten.add.Tensor(mul_132, mul_133);  mul_132 = mul_133 = None
    squeeze_56: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
    mul_134: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0006381620931717);  squeeze_56 = None
    mul_135: "f32[152]" = torch.ops.aten.mul.Tensor(mul_134, 0.1);  mul_134 = None
    mul_136: "f32[152]" = torch.ops.aten.mul.Tensor(primals_243, 0.9)
    add_97: "f32[152]" = torch.ops.aten.add.Tensor(mul_135, mul_136);  mul_135 = mul_136 = None
    unsqueeze_72: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1)
    unsqueeze_73: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    mul_137: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_131, unsqueeze_73);  mul_131 = unsqueeze_73 = None
    unsqueeze_74: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1);  primals_38 = None
    unsqueeze_75: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    add_98: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_137, unsqueeze_75);  mul_137 = unsqueeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_99: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(add_98, relu_16);  add_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_20: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_99);  add_99 = None
    alias_25: "f32[8, 152, 14, 14]" = torch.ops.aten.alias.default(relu_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_29: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(relu_20, primals_128, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_100: "i64[]" = torch.ops.aten.add.Tensor(primals_244, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_19 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
    getitem_38: "f32[1, 152, 1, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 152, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_101: "f32[1, 152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
    rsqrt_19: "f32[1, 152, 1, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    sub_19: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, getitem_39)
    mul_138: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    squeeze_57: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
    squeeze_58: "f32[152]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
    mul_139: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_140: "f32[152]" = torch.ops.aten.mul.Tensor(primals_245, 0.9)
    add_102: "f32[152]" = torch.ops.aten.add.Tensor(mul_139, mul_140);  mul_139 = mul_140 = None
    squeeze_59: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
    mul_141: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0006381620931717);  squeeze_59 = None
    mul_142: "f32[152]" = torch.ops.aten.mul.Tensor(mul_141, 0.1);  mul_141 = None
    mul_143: "f32[152]" = torch.ops.aten.mul.Tensor(primals_246, 0.9)
    add_103: "f32[152]" = torch.ops.aten.add.Tensor(mul_142, mul_143);  mul_142 = mul_143 = None
    unsqueeze_76: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1)
    unsqueeze_77: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_144: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_138, unsqueeze_77);  mul_138 = unsqueeze_77 = None
    unsqueeze_78: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1);  primals_40 = None
    unsqueeze_79: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_104: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_144, unsqueeze_79);  mul_144 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_21: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_104);  add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_30: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(relu_21, primals_129, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_105: "i64[]" = torch.ops.aten.add.Tensor(primals_247, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_20 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
    getitem_40: "f32[1, 152, 1, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 152, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_106: "f32[1, 152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
    rsqrt_20: "f32[1, 152, 1, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
    sub_20: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, getitem_41)
    mul_145: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    squeeze_60: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
    squeeze_61: "f32[152]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
    mul_146: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_147: "f32[152]" = torch.ops.aten.mul.Tensor(primals_248, 0.9)
    add_107: "f32[152]" = torch.ops.aten.add.Tensor(mul_146, mul_147);  mul_146 = mul_147 = None
    squeeze_62: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
    mul_148: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0006381620931717);  squeeze_62 = None
    mul_149: "f32[152]" = torch.ops.aten.mul.Tensor(mul_148, 0.1);  mul_148 = None
    mul_150: "f32[152]" = torch.ops.aten.mul.Tensor(primals_249, 0.9)
    add_108: "f32[152]" = torch.ops.aten.add.Tensor(mul_149, mul_150);  mul_149 = mul_150 = None
    unsqueeze_80: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_81: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    mul_151: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_145, unsqueeze_81);  mul_145 = unsqueeze_81 = None
    unsqueeze_82: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_83: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    add_109: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_151, unsqueeze_83);  mul_151 = unsqueeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_22: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_109);  add_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_5: "f32[8, 152, 1, 1]" = torch.ops.aten.mean.dim(relu_22, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_31: "f32[8, 38, 1, 1]" = torch.ops.aten.convolution.default(mean_5, primals_130, primals_131, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_23: "f32[8, 38, 1, 1]" = torch.ops.aten.relu.default(convolution_31);  convolution_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_32: "f32[8, 152, 1, 1]" = torch.ops.aten.convolution.default(relu_23, primals_132, primals_133, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_5: "f32[8, 152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_32);  convolution_32 = None
    alias_29: "f32[8, 152, 1, 1]" = torch.ops.aten.alias.default(sigmoid_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_152: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(relu_22, sigmoid_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_33: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(mul_152, primals_134, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_110: "i64[]" = torch.ops.aten.add.Tensor(primals_250, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_21 = torch.ops.aten.var_mean.correction(convolution_33, [0, 2, 3], correction = 0, keepdim = True)
    getitem_42: "f32[1, 152, 1, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 152, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_111: "f32[1, 152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
    rsqrt_21: "f32[1, 152, 1, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
    sub_21: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, getitem_43)
    mul_153: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    squeeze_63: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
    squeeze_64: "f32[152]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
    mul_154: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_155: "f32[152]" = torch.ops.aten.mul.Tensor(primals_251, 0.9)
    add_112: "f32[152]" = torch.ops.aten.add.Tensor(mul_154, mul_155);  mul_154 = mul_155 = None
    squeeze_65: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
    mul_156: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0006381620931717);  squeeze_65 = None
    mul_157: "f32[152]" = torch.ops.aten.mul.Tensor(mul_156, 0.1);  mul_156 = None
    mul_158: "f32[152]" = torch.ops.aten.mul.Tensor(primals_252, 0.9)
    add_113: "f32[152]" = torch.ops.aten.add.Tensor(mul_157, mul_158);  mul_157 = mul_158 = None
    unsqueeze_84: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1)
    unsqueeze_85: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_159: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_153, unsqueeze_85);  mul_153 = unsqueeze_85 = None
    unsqueeze_86: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1);  primals_44 = None
    unsqueeze_87: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_114: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_159, unsqueeze_87);  mul_159 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_115: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(add_114, relu_20);  add_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_24: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_115);  add_115 = None
    alias_30: "f32[8, 152, 14, 14]" = torch.ops.aten.alias.default(relu_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_34: "f32[8, 368, 14, 14]" = torch.ops.aten.convolution.default(relu_24, primals_135, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_116: "i64[]" = torch.ops.aten.add.Tensor(primals_253, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_22 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
    getitem_44: "f32[1, 368, 1, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 368, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_117: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05)
    rsqrt_22: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
    sub_22: "f32[8, 368, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_45)
    mul_160: "f32[8, 368, 14, 14]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    squeeze_66: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
    squeeze_67: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
    mul_161: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_162: "f32[368]" = torch.ops.aten.mul.Tensor(primals_254, 0.9)
    add_118: "f32[368]" = torch.ops.aten.add.Tensor(mul_161, mul_162);  mul_161 = mul_162 = None
    squeeze_68: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
    mul_163: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0006381620931717);  squeeze_68 = None
    mul_164: "f32[368]" = torch.ops.aten.mul.Tensor(mul_163, 0.1);  mul_163 = None
    mul_165: "f32[368]" = torch.ops.aten.mul.Tensor(primals_255, 0.9)
    add_119: "f32[368]" = torch.ops.aten.add.Tensor(mul_164, mul_165);  mul_164 = mul_165 = None
    unsqueeze_88: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1)
    unsqueeze_89: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    mul_166: "f32[8, 368, 14, 14]" = torch.ops.aten.mul.Tensor(mul_160, unsqueeze_89);  mul_160 = unsqueeze_89 = None
    unsqueeze_90: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1);  primals_46 = None
    unsqueeze_91: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    add_120: "f32[8, 368, 14, 14]" = torch.ops.aten.add.Tensor(mul_166, unsqueeze_91);  mul_166 = unsqueeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_25: "f32[8, 368, 14, 14]" = torch.ops.aten.relu.default(add_120);  add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_35: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_25, primals_136, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_121: "i64[]" = torch.ops.aten.add.Tensor(primals_256, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_23 = torch.ops.aten.var_mean.correction(convolution_35, [0, 2, 3], correction = 0, keepdim = True)
    getitem_46: "f32[1, 368, 1, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 368, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_122: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
    rsqrt_23: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    sub_23: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_35, getitem_47)
    mul_167: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    squeeze_69: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
    squeeze_70: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
    mul_168: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
    mul_169: "f32[368]" = torch.ops.aten.mul.Tensor(primals_257, 0.9)
    add_123: "f32[368]" = torch.ops.aten.add.Tensor(mul_168, mul_169);  mul_168 = mul_169 = None
    squeeze_71: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
    mul_170: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0025575447570332);  squeeze_71 = None
    mul_171: "f32[368]" = torch.ops.aten.mul.Tensor(mul_170, 0.1);  mul_170 = None
    mul_172: "f32[368]" = torch.ops.aten.mul.Tensor(primals_258, 0.9)
    add_124: "f32[368]" = torch.ops.aten.add.Tensor(mul_171, mul_172);  mul_171 = mul_172 = None
    unsqueeze_92: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_93: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_173: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_167, unsqueeze_93);  mul_167 = unsqueeze_93 = None
    unsqueeze_94: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_95: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_125: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_173, unsqueeze_95);  mul_173 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_26: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_125);  add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_6: "f32[8, 368, 1, 1]" = torch.ops.aten.mean.dim(relu_26, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_36: "f32[8, 38, 1, 1]" = torch.ops.aten.convolution.default(mean_6, primals_137, primals_138, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_27: "f32[8, 38, 1, 1]" = torch.ops.aten.relu.default(convolution_36);  convolution_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_37: "f32[8, 368, 1, 1]" = torch.ops.aten.convolution.default(relu_27, primals_139, primals_140, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_6: "f32[8, 368, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_37);  convolution_37 = None
    alias_34: "f32[8, 368, 1, 1]" = torch.ops.aten.alias.default(sigmoid_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_174: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(relu_26, sigmoid_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_38: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(mul_174, primals_141, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_126: "i64[]" = torch.ops.aten.add.Tensor(primals_259, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_24 = torch.ops.aten.var_mean.correction(convolution_38, [0, 2, 3], correction = 0, keepdim = True)
    getitem_48: "f32[1, 368, 1, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 368, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_127: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05)
    rsqrt_24: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
    sub_24: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_38, getitem_49)
    mul_175: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    squeeze_72: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
    squeeze_73: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
    mul_176: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_177: "f32[368]" = torch.ops.aten.mul.Tensor(primals_260, 0.9)
    add_128: "f32[368]" = torch.ops.aten.add.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
    squeeze_74: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
    mul_178: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0025575447570332);  squeeze_74 = None
    mul_179: "f32[368]" = torch.ops.aten.mul.Tensor(mul_178, 0.1);  mul_178 = None
    mul_180: "f32[368]" = torch.ops.aten.mul.Tensor(primals_261, 0.9)
    add_129: "f32[368]" = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
    unsqueeze_96: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1)
    unsqueeze_97: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    mul_181: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_97);  mul_175 = unsqueeze_97 = None
    unsqueeze_98: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1);  primals_50 = None
    unsqueeze_99: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    add_130: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_99);  mul_181 = unsqueeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_39: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_24, primals_142, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_131: "i64[]" = torch.ops.aten.add.Tensor(primals_262, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_25 = torch.ops.aten.var_mean.correction(convolution_39, [0, 2, 3], correction = 0, keepdim = True)
    getitem_50: "f32[1, 368, 1, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 368, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_132: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
    rsqrt_25: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
    sub_25: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_39, getitem_51)
    mul_182: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    squeeze_75: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
    squeeze_76: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
    mul_183: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_184: "f32[368]" = torch.ops.aten.mul.Tensor(primals_263, 0.9)
    add_133: "f32[368]" = torch.ops.aten.add.Tensor(mul_183, mul_184);  mul_183 = mul_184 = None
    squeeze_77: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
    mul_185: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0025575447570332);  squeeze_77 = None
    mul_186: "f32[368]" = torch.ops.aten.mul.Tensor(mul_185, 0.1);  mul_185 = None
    mul_187: "f32[368]" = torch.ops.aten.mul.Tensor(primals_264, 0.9)
    add_134: "f32[368]" = torch.ops.aten.add.Tensor(mul_186, mul_187);  mul_186 = mul_187 = None
    unsqueeze_100: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1)
    unsqueeze_101: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_188: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_101);  mul_182 = unsqueeze_101 = None
    unsqueeze_102: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1);  primals_52 = None
    unsqueeze_103: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_135: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_103);  mul_188 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_136: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(add_130, add_135);  add_130 = add_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_28: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_136);  add_136 = None
    alias_35: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(relu_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_40: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_28, primals_143, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_137: "i64[]" = torch.ops.aten.add.Tensor(primals_265, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_26 = torch.ops.aten.var_mean.correction(convolution_40, [0, 2, 3], correction = 0, keepdim = True)
    getitem_52: "f32[1, 368, 1, 1]" = var_mean_26[0]
    getitem_53: "f32[1, 368, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_138: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05)
    rsqrt_26: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
    sub_26: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_53)
    mul_189: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    squeeze_78: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
    squeeze_79: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
    mul_190: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_191: "f32[368]" = torch.ops.aten.mul.Tensor(primals_266, 0.9)
    add_139: "f32[368]" = torch.ops.aten.add.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
    squeeze_80: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
    mul_192: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0025575447570332);  squeeze_80 = None
    mul_193: "f32[368]" = torch.ops.aten.mul.Tensor(mul_192, 0.1);  mul_192 = None
    mul_194: "f32[368]" = torch.ops.aten.mul.Tensor(primals_267, 0.9)
    add_140: "f32[368]" = torch.ops.aten.add.Tensor(mul_193, mul_194);  mul_193 = mul_194 = None
    unsqueeze_104: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_105: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    mul_195: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_189, unsqueeze_105);  mul_189 = unsqueeze_105 = None
    unsqueeze_106: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_107: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    add_141: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_195, unsqueeze_107);  mul_195 = unsqueeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_29: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_141);  add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_41: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_29, primals_144, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_142: "i64[]" = torch.ops.aten.add.Tensor(primals_268, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_27 = torch.ops.aten.var_mean.correction(convolution_41, [0, 2, 3], correction = 0, keepdim = True)
    getitem_54: "f32[1, 368, 1, 1]" = var_mean_27[0]
    getitem_55: "f32[1, 368, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_143: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05)
    rsqrt_27: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
    sub_27: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_41, getitem_55)
    mul_196: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    squeeze_81: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
    squeeze_82: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
    mul_197: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
    mul_198: "f32[368]" = torch.ops.aten.mul.Tensor(primals_269, 0.9)
    add_144: "f32[368]" = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
    squeeze_83: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
    mul_199: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0025575447570332);  squeeze_83 = None
    mul_200: "f32[368]" = torch.ops.aten.mul.Tensor(mul_199, 0.1);  mul_199 = None
    mul_201: "f32[368]" = torch.ops.aten.mul.Tensor(primals_270, 0.9)
    add_145: "f32[368]" = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
    unsqueeze_108: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1)
    unsqueeze_109: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_202: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_109);  mul_196 = unsqueeze_109 = None
    unsqueeze_110: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1);  primals_56 = None
    unsqueeze_111: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_146: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_111);  mul_202 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_30: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_146);  add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_7: "f32[8, 368, 1, 1]" = torch.ops.aten.mean.dim(relu_30, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_42: "f32[8, 92, 1, 1]" = torch.ops.aten.convolution.default(mean_7, primals_145, primals_146, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_31: "f32[8, 92, 1, 1]" = torch.ops.aten.relu.default(convolution_42);  convolution_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_43: "f32[8, 368, 1, 1]" = torch.ops.aten.convolution.default(relu_31, primals_147, primals_148, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_7: "f32[8, 368, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_43);  convolution_43 = None
    alias_39: "f32[8, 368, 1, 1]" = torch.ops.aten.alias.default(sigmoid_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_203: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(relu_30, sigmoid_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_44: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(mul_203, primals_149, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_147: "i64[]" = torch.ops.aten.add.Tensor(primals_271, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_28 = torch.ops.aten.var_mean.correction(convolution_44, [0, 2, 3], correction = 0, keepdim = True)
    getitem_56: "f32[1, 368, 1, 1]" = var_mean_28[0]
    getitem_57: "f32[1, 368, 1, 1]" = var_mean_28[1];  var_mean_28 = None
    add_148: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05)
    rsqrt_28: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
    sub_28: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_44, getitem_57)
    mul_204: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    squeeze_84: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
    squeeze_85: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
    mul_205: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
    mul_206: "f32[368]" = torch.ops.aten.mul.Tensor(primals_272, 0.9)
    add_149: "f32[368]" = torch.ops.aten.add.Tensor(mul_205, mul_206);  mul_205 = mul_206 = None
    squeeze_86: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
    mul_207: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0025575447570332);  squeeze_86 = None
    mul_208: "f32[368]" = torch.ops.aten.mul.Tensor(mul_207, 0.1);  mul_207 = None
    mul_209: "f32[368]" = torch.ops.aten.mul.Tensor(primals_273, 0.9)
    add_150: "f32[368]" = torch.ops.aten.add.Tensor(mul_208, mul_209);  mul_208 = mul_209 = None
    unsqueeze_112: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1)
    unsqueeze_113: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    mul_210: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_204, unsqueeze_113);  mul_204 = unsqueeze_113 = None
    unsqueeze_114: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_58, -1);  primals_58 = None
    unsqueeze_115: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    add_151: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_210, unsqueeze_115);  mul_210 = unsqueeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_152: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(add_151, relu_28);  add_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_32: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_152);  add_152 = None
    alias_40: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(relu_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_45: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_32, primals_150, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_153: "i64[]" = torch.ops.aten.add.Tensor(primals_274, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_29 = torch.ops.aten.var_mean.correction(convolution_45, [0, 2, 3], correction = 0, keepdim = True)
    getitem_58: "f32[1, 368, 1, 1]" = var_mean_29[0]
    getitem_59: "f32[1, 368, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_154: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
    rsqrt_29: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
    sub_29: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_45, getitem_59)
    mul_211: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    squeeze_87: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
    squeeze_88: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
    mul_212: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
    mul_213: "f32[368]" = torch.ops.aten.mul.Tensor(primals_275, 0.9)
    add_155: "f32[368]" = torch.ops.aten.add.Tensor(mul_212, mul_213);  mul_212 = mul_213 = None
    squeeze_89: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
    mul_214: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0025575447570332);  squeeze_89 = None
    mul_215: "f32[368]" = torch.ops.aten.mul.Tensor(mul_214, 0.1);  mul_214 = None
    mul_216: "f32[368]" = torch.ops.aten.mul.Tensor(primals_276, 0.9)
    add_156: "f32[368]" = torch.ops.aten.add.Tensor(mul_215, mul_216);  mul_215 = mul_216 = None
    unsqueeze_116: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_117: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_217: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_211, unsqueeze_117);  mul_211 = unsqueeze_117 = None
    unsqueeze_118: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_119: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_157: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_217, unsqueeze_119);  mul_217 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_33: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_157);  add_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_46: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_33, primals_151, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_158: "i64[]" = torch.ops.aten.add.Tensor(primals_277, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_30 = torch.ops.aten.var_mean.correction(convolution_46, [0, 2, 3], correction = 0, keepdim = True)
    getitem_60: "f32[1, 368, 1, 1]" = var_mean_30[0]
    getitem_61: "f32[1, 368, 1, 1]" = var_mean_30[1];  var_mean_30 = None
    add_159: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
    rsqrt_30: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
    sub_30: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_46, getitem_61)
    mul_218: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    squeeze_90: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
    squeeze_91: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
    mul_219: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
    mul_220: "f32[368]" = torch.ops.aten.mul.Tensor(primals_278, 0.9)
    add_160: "f32[368]" = torch.ops.aten.add.Tensor(mul_219, mul_220);  mul_219 = mul_220 = None
    squeeze_92: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
    mul_221: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0025575447570332);  squeeze_92 = None
    mul_222: "f32[368]" = torch.ops.aten.mul.Tensor(mul_221, 0.1);  mul_221 = None
    mul_223: "f32[368]" = torch.ops.aten.mul.Tensor(primals_279, 0.9)
    add_161: "f32[368]" = torch.ops.aten.add.Tensor(mul_222, mul_223);  mul_222 = mul_223 = None
    unsqueeze_120: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_61, -1)
    unsqueeze_121: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    mul_224: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_218, unsqueeze_121);  mul_218 = unsqueeze_121 = None
    unsqueeze_122: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1);  primals_62 = None
    unsqueeze_123: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    add_162: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_224, unsqueeze_123);  mul_224 = unsqueeze_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_34: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_162);  add_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_8: "f32[8, 368, 1, 1]" = torch.ops.aten.mean.dim(relu_34, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_47: "f32[8, 92, 1, 1]" = torch.ops.aten.convolution.default(mean_8, primals_152, primals_153, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_35: "f32[8, 92, 1, 1]" = torch.ops.aten.relu.default(convolution_47);  convolution_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_48: "f32[8, 368, 1, 1]" = torch.ops.aten.convolution.default(relu_35, primals_154, primals_155, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_8: "f32[8, 368, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_48);  convolution_48 = None
    alias_44: "f32[8, 368, 1, 1]" = torch.ops.aten.alias.default(sigmoid_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_225: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(relu_34, sigmoid_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_49: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(mul_225, primals_156, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_163: "i64[]" = torch.ops.aten.add.Tensor(primals_280, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_31 = torch.ops.aten.var_mean.correction(convolution_49, [0, 2, 3], correction = 0, keepdim = True)
    getitem_62: "f32[1, 368, 1, 1]" = var_mean_31[0]
    getitem_63: "f32[1, 368, 1, 1]" = var_mean_31[1];  var_mean_31 = None
    add_164: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
    rsqrt_31: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
    sub_31: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_49, getitem_63)
    mul_226: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    squeeze_93: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
    squeeze_94: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
    mul_227: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
    mul_228: "f32[368]" = torch.ops.aten.mul.Tensor(primals_281, 0.9)
    add_165: "f32[368]" = torch.ops.aten.add.Tensor(mul_227, mul_228);  mul_227 = mul_228 = None
    squeeze_95: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
    mul_229: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0025575447570332);  squeeze_95 = None
    mul_230: "f32[368]" = torch.ops.aten.mul.Tensor(mul_229, 0.1);  mul_229 = None
    mul_231: "f32[368]" = torch.ops.aten.mul.Tensor(primals_282, 0.9)
    add_166: "f32[368]" = torch.ops.aten.add.Tensor(mul_230, mul_231);  mul_230 = mul_231 = None
    unsqueeze_124: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1)
    unsqueeze_125: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_232: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_226, unsqueeze_125);  mul_226 = unsqueeze_125 = None
    unsqueeze_126: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_64, -1);  primals_64 = None
    unsqueeze_127: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_167: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_232, unsqueeze_127);  mul_232 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_168: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(add_167, relu_32);  add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_36: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_168);  add_168 = None
    alias_45: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(relu_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_50: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_36, primals_157, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_169: "i64[]" = torch.ops.aten.add.Tensor(primals_283, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_32 = torch.ops.aten.var_mean.correction(convolution_50, [0, 2, 3], correction = 0, keepdim = True)
    getitem_64: "f32[1, 368, 1, 1]" = var_mean_32[0]
    getitem_65: "f32[1, 368, 1, 1]" = var_mean_32[1];  var_mean_32 = None
    add_170: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
    rsqrt_32: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
    sub_32: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, getitem_65)
    mul_233: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    squeeze_96: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
    squeeze_97: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
    mul_234: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
    mul_235: "f32[368]" = torch.ops.aten.mul.Tensor(primals_284, 0.9)
    add_171: "f32[368]" = torch.ops.aten.add.Tensor(mul_234, mul_235);  mul_234 = mul_235 = None
    squeeze_98: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
    mul_236: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0025575447570332);  squeeze_98 = None
    mul_237: "f32[368]" = torch.ops.aten.mul.Tensor(mul_236, 0.1);  mul_236 = None
    mul_238: "f32[368]" = torch.ops.aten.mul.Tensor(primals_285, 0.9)
    add_172: "f32[368]" = torch.ops.aten.add.Tensor(mul_237, mul_238);  mul_237 = mul_238 = None
    unsqueeze_128: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
    unsqueeze_129: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    mul_239: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_233, unsqueeze_129);  mul_233 = unsqueeze_129 = None
    unsqueeze_130: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
    unsqueeze_131: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    add_173: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_239, unsqueeze_131);  mul_239 = unsqueeze_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_37: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_173);  add_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_51: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_37, primals_158, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_174: "i64[]" = torch.ops.aten.add.Tensor(primals_286, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_33 = torch.ops.aten.var_mean.correction(convolution_51, [0, 2, 3], correction = 0, keepdim = True)
    getitem_66: "f32[1, 368, 1, 1]" = var_mean_33[0]
    getitem_67: "f32[1, 368, 1, 1]" = var_mean_33[1];  var_mean_33 = None
    add_175: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
    rsqrt_33: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
    sub_33: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, getitem_67)
    mul_240: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    squeeze_99: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
    squeeze_100: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
    mul_241: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
    mul_242: "f32[368]" = torch.ops.aten.mul.Tensor(primals_287, 0.9)
    add_176: "f32[368]" = torch.ops.aten.add.Tensor(mul_241, mul_242);  mul_241 = mul_242 = None
    squeeze_101: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
    mul_243: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0025575447570332);  squeeze_101 = None
    mul_244: "f32[368]" = torch.ops.aten.mul.Tensor(mul_243, 0.1);  mul_243 = None
    mul_245: "f32[368]" = torch.ops.aten.mul.Tensor(primals_288, 0.9)
    add_177: "f32[368]" = torch.ops.aten.add.Tensor(mul_244, mul_245);  mul_244 = mul_245 = None
    unsqueeze_132: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_67, -1)
    unsqueeze_133: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_246: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_240, unsqueeze_133);  mul_240 = unsqueeze_133 = None
    unsqueeze_134: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1);  primals_68 = None
    unsqueeze_135: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_178: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_246, unsqueeze_135);  mul_246 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_38: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_178);  add_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_9: "f32[8, 368, 1, 1]" = torch.ops.aten.mean.dim(relu_38, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_52: "f32[8, 92, 1, 1]" = torch.ops.aten.convolution.default(mean_9, primals_159, primals_160, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_39: "f32[8, 92, 1, 1]" = torch.ops.aten.relu.default(convolution_52);  convolution_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_53: "f32[8, 368, 1, 1]" = torch.ops.aten.convolution.default(relu_39, primals_161, primals_162, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_9: "f32[8, 368, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_53);  convolution_53 = None
    alias_49: "f32[8, 368, 1, 1]" = torch.ops.aten.alias.default(sigmoid_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_247: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(relu_38, sigmoid_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_54: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(mul_247, primals_163, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_179: "i64[]" = torch.ops.aten.add.Tensor(primals_289, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_34 = torch.ops.aten.var_mean.correction(convolution_54, [0, 2, 3], correction = 0, keepdim = True)
    getitem_68: "f32[1, 368, 1, 1]" = var_mean_34[0]
    getitem_69: "f32[1, 368, 1, 1]" = var_mean_34[1];  var_mean_34 = None
    add_180: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05)
    rsqrt_34: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_180);  add_180 = None
    sub_34: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_54, getitem_69)
    mul_248: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    squeeze_102: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
    squeeze_103: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
    mul_249: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
    mul_250: "f32[368]" = torch.ops.aten.mul.Tensor(primals_290, 0.9)
    add_181: "f32[368]" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
    squeeze_104: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
    mul_251: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0025575447570332);  squeeze_104 = None
    mul_252: "f32[368]" = torch.ops.aten.mul.Tensor(mul_251, 0.1);  mul_251 = None
    mul_253: "f32[368]" = torch.ops.aten.mul.Tensor(primals_291, 0.9)
    add_182: "f32[368]" = torch.ops.aten.add.Tensor(mul_252, mul_253);  mul_252 = mul_253 = None
    unsqueeze_136: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1)
    unsqueeze_137: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    mul_254: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_248, unsqueeze_137);  mul_248 = unsqueeze_137 = None
    unsqueeze_138: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_70, -1);  primals_70 = None
    unsqueeze_139: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    add_183: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_254, unsqueeze_139);  mul_254 = unsqueeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_184: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(add_183, relu_36);  add_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_40: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_184);  add_184 = None
    alias_50: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(relu_40)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_55: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_40, primals_164, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_185: "i64[]" = torch.ops.aten.add.Tensor(primals_292, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_35 = torch.ops.aten.var_mean.correction(convolution_55, [0, 2, 3], correction = 0, keepdim = True)
    getitem_70: "f32[1, 368, 1, 1]" = var_mean_35[0]
    getitem_71: "f32[1, 368, 1, 1]" = var_mean_35[1];  var_mean_35 = None
    add_186: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05)
    rsqrt_35: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
    sub_35: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_55, getitem_71)
    mul_255: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    squeeze_105: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
    squeeze_106: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
    mul_256: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
    mul_257: "f32[368]" = torch.ops.aten.mul.Tensor(primals_293, 0.9)
    add_187: "f32[368]" = torch.ops.aten.add.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
    squeeze_107: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
    mul_258: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0025575447570332);  squeeze_107 = None
    mul_259: "f32[368]" = torch.ops.aten.mul.Tensor(mul_258, 0.1);  mul_258 = None
    mul_260: "f32[368]" = torch.ops.aten.mul.Tensor(primals_294, 0.9)
    add_188: "f32[368]" = torch.ops.aten.add.Tensor(mul_259, mul_260);  mul_259 = mul_260 = None
    unsqueeze_140: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_141: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_261: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_255, unsqueeze_141);  mul_255 = unsqueeze_141 = None
    unsqueeze_142: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_143: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_189: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_261, unsqueeze_143);  mul_261 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_41: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_189);  add_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_56: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_41, primals_165, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_190: "i64[]" = torch.ops.aten.add.Tensor(primals_295, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_36 = torch.ops.aten.var_mean.correction(convolution_56, [0, 2, 3], correction = 0, keepdim = True)
    getitem_72: "f32[1, 368, 1, 1]" = var_mean_36[0]
    getitem_73: "f32[1, 368, 1, 1]" = var_mean_36[1];  var_mean_36 = None
    add_191: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05)
    rsqrt_36: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
    sub_36: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_56, getitem_73)
    mul_262: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    squeeze_108: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
    squeeze_109: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
    mul_263: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
    mul_264: "f32[368]" = torch.ops.aten.mul.Tensor(primals_296, 0.9)
    add_192: "f32[368]" = torch.ops.aten.add.Tensor(mul_263, mul_264);  mul_263 = mul_264 = None
    squeeze_110: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
    mul_265: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0025575447570332);  squeeze_110 = None
    mul_266: "f32[368]" = torch.ops.aten.mul.Tensor(mul_265, 0.1);  mul_265 = None
    mul_267: "f32[368]" = torch.ops.aten.mul.Tensor(primals_297, 0.9)
    add_193: "f32[368]" = torch.ops.aten.add.Tensor(mul_266, mul_267);  mul_266 = mul_267 = None
    unsqueeze_144: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_73, -1)
    unsqueeze_145: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    mul_268: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_262, unsqueeze_145);  mul_262 = unsqueeze_145 = None
    unsqueeze_146: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1);  primals_74 = None
    unsqueeze_147: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    add_194: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_268, unsqueeze_147);  mul_268 = unsqueeze_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_42: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_194);  add_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_10: "f32[8, 368, 1, 1]" = torch.ops.aten.mean.dim(relu_42, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_57: "f32[8, 92, 1, 1]" = torch.ops.aten.convolution.default(mean_10, primals_166, primals_167, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_43: "f32[8, 92, 1, 1]" = torch.ops.aten.relu.default(convolution_57);  convolution_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_58: "f32[8, 368, 1, 1]" = torch.ops.aten.convolution.default(relu_43, primals_168, primals_169, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_10: "f32[8, 368, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_58);  convolution_58 = None
    alias_54: "f32[8, 368, 1, 1]" = torch.ops.aten.alias.default(sigmoid_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_269: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(relu_42, sigmoid_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_59: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(mul_269, primals_170, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_195: "i64[]" = torch.ops.aten.add.Tensor(primals_298, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_37 = torch.ops.aten.var_mean.correction(convolution_59, [0, 2, 3], correction = 0, keepdim = True)
    getitem_74: "f32[1, 368, 1, 1]" = var_mean_37[0]
    getitem_75: "f32[1, 368, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_196: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05)
    rsqrt_37: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
    sub_37: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_59, getitem_75)
    mul_270: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    squeeze_111: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
    squeeze_112: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
    mul_271: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
    mul_272: "f32[368]" = torch.ops.aten.mul.Tensor(primals_299, 0.9)
    add_197: "f32[368]" = torch.ops.aten.add.Tensor(mul_271, mul_272);  mul_271 = mul_272 = None
    squeeze_113: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
    mul_273: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0025575447570332);  squeeze_113 = None
    mul_274: "f32[368]" = torch.ops.aten.mul.Tensor(mul_273, 0.1);  mul_273 = None
    mul_275: "f32[368]" = torch.ops.aten.mul.Tensor(primals_300, 0.9)
    add_198: "f32[368]" = torch.ops.aten.add.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None
    unsqueeze_148: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1)
    unsqueeze_149: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_276: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_270, unsqueeze_149);  mul_270 = unsqueeze_149 = None
    unsqueeze_150: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_76, -1);  primals_76 = None
    unsqueeze_151: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_199: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_276, unsqueeze_151);  mul_276 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_200: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(add_199, relu_40);  add_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_44: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_200);  add_200 = None
    alias_55: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(relu_44)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_60: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_44, primals_171, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_201: "i64[]" = torch.ops.aten.add.Tensor(primals_301, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_38 = torch.ops.aten.var_mean.correction(convolution_60, [0, 2, 3], correction = 0, keepdim = True)
    getitem_76: "f32[1, 368, 1, 1]" = var_mean_38[0]
    getitem_77: "f32[1, 368, 1, 1]" = var_mean_38[1];  var_mean_38 = None
    add_202: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
    rsqrt_38: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_202);  add_202 = None
    sub_38: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_60, getitem_77)
    mul_277: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    squeeze_114: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
    squeeze_115: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
    mul_278: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
    mul_279: "f32[368]" = torch.ops.aten.mul.Tensor(primals_302, 0.9)
    add_203: "f32[368]" = torch.ops.aten.add.Tensor(mul_278, mul_279);  mul_278 = mul_279 = None
    squeeze_116: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
    mul_280: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0025575447570332);  squeeze_116 = None
    mul_281: "f32[368]" = torch.ops.aten.mul.Tensor(mul_280, 0.1);  mul_280 = None
    mul_282: "f32[368]" = torch.ops.aten.mul.Tensor(primals_303, 0.9)
    add_204: "f32[368]" = torch.ops.aten.add.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
    unsqueeze_152: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
    unsqueeze_153: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    mul_283: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_277, unsqueeze_153);  mul_277 = unsqueeze_153 = None
    unsqueeze_154: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
    unsqueeze_155: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    add_205: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_283, unsqueeze_155);  mul_283 = unsqueeze_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_45: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_205);  add_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_61: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_45, primals_172, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_206: "i64[]" = torch.ops.aten.add.Tensor(primals_304, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_39 = torch.ops.aten.var_mean.correction(convolution_61, [0, 2, 3], correction = 0, keepdim = True)
    getitem_78: "f32[1, 368, 1, 1]" = var_mean_39[0]
    getitem_79: "f32[1, 368, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_207: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
    rsqrt_39: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_207);  add_207 = None
    sub_39: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_61, getitem_79)
    mul_284: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    squeeze_117: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
    squeeze_118: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
    mul_285: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
    mul_286: "f32[368]" = torch.ops.aten.mul.Tensor(primals_305, 0.9)
    add_208: "f32[368]" = torch.ops.aten.add.Tensor(mul_285, mul_286);  mul_285 = mul_286 = None
    squeeze_119: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
    mul_287: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0025575447570332);  squeeze_119 = None
    mul_288: "f32[368]" = torch.ops.aten.mul.Tensor(mul_287, 0.1);  mul_287 = None
    mul_289: "f32[368]" = torch.ops.aten.mul.Tensor(primals_306, 0.9)
    add_209: "f32[368]" = torch.ops.aten.add.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
    unsqueeze_156: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_79, -1)
    unsqueeze_157: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_290: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_284, unsqueeze_157);  mul_284 = unsqueeze_157 = None
    unsqueeze_158: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1);  primals_80 = None
    unsqueeze_159: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_210: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_290, unsqueeze_159);  mul_290 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_46: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_210);  add_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_11: "f32[8, 368, 1, 1]" = torch.ops.aten.mean.dim(relu_46, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_62: "f32[8, 92, 1, 1]" = torch.ops.aten.convolution.default(mean_11, primals_173, primals_174, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_47: "f32[8, 92, 1, 1]" = torch.ops.aten.relu.default(convolution_62);  convolution_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_63: "f32[8, 368, 1, 1]" = torch.ops.aten.convolution.default(relu_47, primals_175, primals_176, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_11: "f32[8, 368, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_63);  convolution_63 = None
    alias_59: "f32[8, 368, 1, 1]" = torch.ops.aten.alias.default(sigmoid_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_291: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(relu_46, sigmoid_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_64: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(mul_291, primals_177, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_211: "i64[]" = torch.ops.aten.add.Tensor(primals_307, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_40 = torch.ops.aten.var_mean.correction(convolution_64, [0, 2, 3], correction = 0, keepdim = True)
    getitem_80: "f32[1, 368, 1, 1]" = var_mean_40[0]
    getitem_81: "f32[1, 368, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_212: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05)
    rsqrt_40: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_212);  add_212 = None
    sub_40: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_64, getitem_81)
    mul_292: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    squeeze_120: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
    squeeze_121: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
    mul_293: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
    mul_294: "f32[368]" = torch.ops.aten.mul.Tensor(primals_308, 0.9)
    add_213: "f32[368]" = torch.ops.aten.add.Tensor(mul_293, mul_294);  mul_293 = mul_294 = None
    squeeze_122: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
    mul_295: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0025575447570332);  squeeze_122 = None
    mul_296: "f32[368]" = torch.ops.aten.mul.Tensor(mul_295, 0.1);  mul_295 = None
    mul_297: "f32[368]" = torch.ops.aten.mul.Tensor(primals_309, 0.9)
    add_214: "f32[368]" = torch.ops.aten.add.Tensor(mul_296, mul_297);  mul_296 = mul_297 = None
    unsqueeze_160: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1)
    unsqueeze_161: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    mul_298: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_292, unsqueeze_161);  mul_292 = unsqueeze_161 = None
    unsqueeze_162: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_82, -1);  primals_82 = None
    unsqueeze_163: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    add_215: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_298, unsqueeze_163);  mul_298 = unsqueeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_216: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(add_215, relu_44);  add_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_48: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_216);  add_216 = None
    alias_60: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(relu_48)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_65: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_48, primals_178, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_217: "i64[]" = torch.ops.aten.add.Tensor(primals_310, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_41 = torch.ops.aten.var_mean.correction(convolution_65, [0, 2, 3], correction = 0, keepdim = True)
    getitem_82: "f32[1, 368, 1, 1]" = var_mean_41[0]
    getitem_83: "f32[1, 368, 1, 1]" = var_mean_41[1];  var_mean_41 = None
    add_218: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05)
    rsqrt_41: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
    sub_41: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_65, getitem_83)
    mul_299: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
    squeeze_123: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
    squeeze_124: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
    mul_300: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
    mul_301: "f32[368]" = torch.ops.aten.mul.Tensor(primals_311, 0.9)
    add_219: "f32[368]" = torch.ops.aten.add.Tensor(mul_300, mul_301);  mul_300 = mul_301 = None
    squeeze_125: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
    mul_302: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0025575447570332);  squeeze_125 = None
    mul_303: "f32[368]" = torch.ops.aten.mul.Tensor(mul_302, 0.1);  mul_302 = None
    mul_304: "f32[368]" = torch.ops.aten.mul.Tensor(primals_312, 0.9)
    add_220: "f32[368]" = torch.ops.aten.add.Tensor(mul_303, mul_304);  mul_303 = mul_304 = None
    unsqueeze_164: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1)
    unsqueeze_165: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_305: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_299, unsqueeze_165);  mul_299 = unsqueeze_165 = None
    unsqueeze_166: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
    unsqueeze_167: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_221: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_305, unsqueeze_167);  mul_305 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_49: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_221);  add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_66: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_49, primals_179, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_222: "i64[]" = torch.ops.aten.add.Tensor(primals_313, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_42 = torch.ops.aten.var_mean.correction(convolution_66, [0, 2, 3], correction = 0, keepdim = True)
    getitem_84: "f32[1, 368, 1, 1]" = var_mean_42[0]
    getitem_85: "f32[1, 368, 1, 1]" = var_mean_42[1];  var_mean_42 = None
    add_223: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05)
    rsqrt_42: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_223);  add_223 = None
    sub_42: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_66, getitem_85)
    mul_306: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
    squeeze_126: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
    squeeze_127: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
    mul_307: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
    mul_308: "f32[368]" = torch.ops.aten.mul.Tensor(primals_314, 0.9)
    add_224: "f32[368]" = torch.ops.aten.add.Tensor(mul_307, mul_308);  mul_307 = mul_308 = None
    squeeze_128: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
    mul_309: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0025575447570332);  squeeze_128 = None
    mul_310: "f32[368]" = torch.ops.aten.mul.Tensor(mul_309, 0.1);  mul_309 = None
    mul_311: "f32[368]" = torch.ops.aten.mul.Tensor(primals_315, 0.9)
    add_225: "f32[368]" = torch.ops.aten.add.Tensor(mul_310, mul_311);  mul_310 = mul_311 = None
    unsqueeze_168: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_85, -1)
    unsqueeze_169: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    mul_312: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_306, unsqueeze_169);  mul_306 = unsqueeze_169 = None
    unsqueeze_170: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1);  primals_86 = None
    unsqueeze_171: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    add_226: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_312, unsqueeze_171);  mul_312 = unsqueeze_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_50: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_226);  add_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_12: "f32[8, 368, 1, 1]" = torch.ops.aten.mean.dim(relu_50, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_67: "f32[8, 92, 1, 1]" = torch.ops.aten.convolution.default(mean_12, primals_180, primals_181, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_51: "f32[8, 92, 1, 1]" = torch.ops.aten.relu.default(convolution_67);  convolution_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_68: "f32[8, 368, 1, 1]" = torch.ops.aten.convolution.default(relu_51, primals_182, primals_183, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_12: "f32[8, 368, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_68);  convolution_68 = None
    alias_64: "f32[8, 368, 1, 1]" = torch.ops.aten.alias.default(sigmoid_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_313: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(relu_50, sigmoid_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_69: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(mul_313, primals_184, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_227: "i64[]" = torch.ops.aten.add.Tensor(primals_316, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_43 = torch.ops.aten.var_mean.correction(convolution_69, [0, 2, 3], correction = 0, keepdim = True)
    getitem_86: "f32[1, 368, 1, 1]" = var_mean_43[0]
    getitem_87: "f32[1, 368, 1, 1]" = var_mean_43[1];  var_mean_43 = None
    add_228: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05)
    rsqrt_43: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_228);  add_228 = None
    sub_43: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_69, getitem_87)
    mul_314: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
    squeeze_129: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
    squeeze_130: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
    mul_315: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
    mul_316: "f32[368]" = torch.ops.aten.mul.Tensor(primals_317, 0.9)
    add_229: "f32[368]" = torch.ops.aten.add.Tensor(mul_315, mul_316);  mul_315 = mul_316 = None
    squeeze_131: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
    mul_317: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0025575447570332);  squeeze_131 = None
    mul_318: "f32[368]" = torch.ops.aten.mul.Tensor(mul_317, 0.1);  mul_317 = None
    mul_319: "f32[368]" = torch.ops.aten.mul.Tensor(primals_318, 0.9)
    add_230: "f32[368]" = torch.ops.aten.add.Tensor(mul_318, mul_319);  mul_318 = mul_319 = None
    unsqueeze_172: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1)
    unsqueeze_173: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_320: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_314, unsqueeze_173);  mul_314 = unsqueeze_173 = None
    unsqueeze_174: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_88, -1);  primals_88 = None
    unsqueeze_175: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_231: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_320, unsqueeze_175);  mul_320 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_232: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(add_231, relu_48);  add_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_52: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_232);  add_232 = None
    alias_65: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(relu_52)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean_13: "f32[8, 368, 1, 1]" = torch.ops.aten.mean.dim(relu_52, [-1, -2], True);  relu_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view: "f32[8, 368]" = torch.ops.aten.view.default(mean_13, [8, 368]);  mean_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:131, code: x = self.drop(x)
    clone: "f32[8, 368]" = torch.ops.aten.clone.default(view);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    permute: "f32[368, 1000]" = torch.ops.aten.permute.default(primals_185, [1, 0]);  primals_185 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_186, clone, permute);  primals_186 = None
    permute_1: "f32[1000, 368]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm: "f32[8, 368]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 368]" = torch.ops.aten.mm.default(permute_2, clone);  permute_2 = clone = None
    permute_3: "f32[368, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 368]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_2: "f32[8, 368, 1, 1]" = torch.ops.aten.view.default(mm, [8, 368, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 368, 7, 7]" = torch.ops.aten.expand.default(view_2, [8, 368, 7, 7]);  view_2 = None
    div: "f32[8, 368, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    alias_66: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(alias_65);  alias_65 = None
    le: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(alias_66, 0);  alias_66 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le, scalar_tensor, div);  le = scalar_tensor = div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_176: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
    unsqueeze_177: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, 2);  unsqueeze_176 = None
    unsqueeze_178: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_177, 3);  unsqueeze_177 = None
    sum_2: "f32[368]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_44: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_178)
    mul_321: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_44);  sub_44 = None
    sum_3: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_321, [0, 2, 3]);  mul_321 = None
    mul_322: "f32[368]" = torch.ops.aten.mul.Tensor(sum_2, 0.002551020408163265)
    unsqueeze_179: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_322, 0);  mul_322 = None
    unsqueeze_180: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_179, 2);  unsqueeze_179 = None
    unsqueeze_181: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, 3);  unsqueeze_180 = None
    mul_323: "f32[368]" = torch.ops.aten.mul.Tensor(sum_3, 0.002551020408163265)
    mul_324: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
    mul_325: "f32[368]" = torch.ops.aten.mul.Tensor(mul_323, mul_324);  mul_323 = mul_324 = None
    unsqueeze_182: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_325, 0);  mul_325 = None
    unsqueeze_183: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, 2);  unsqueeze_182 = None
    unsqueeze_184: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 3);  unsqueeze_183 = None
    mul_326: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_87);  primals_87 = None
    unsqueeze_185: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_326, 0);  mul_326 = None
    unsqueeze_186: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 2);  unsqueeze_185 = None
    unsqueeze_187: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, 3);  unsqueeze_186 = None
    sub_45: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_178);  convolution_69 = unsqueeze_178 = None
    mul_327: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_184);  sub_45 = unsqueeze_184 = None
    sub_46: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where, mul_327);  mul_327 = None
    sub_47: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_46, unsqueeze_181);  sub_46 = unsqueeze_181 = None
    mul_328: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_187);  sub_47 = unsqueeze_187 = None
    mul_329: "f32[368]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_130);  sum_3 = squeeze_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_328, mul_313, primals_184, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_328 = mul_313 = primals_184 = None
    getitem_88: "f32[8, 368, 7, 7]" = convolution_backward[0]
    getitem_89: "f32[368, 368, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_330: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_88, relu_50)
    mul_331: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_88, sigmoid_12);  getitem_88 = sigmoid_12 = None
    sum_4: "f32[8, 368, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_330, [2, 3], True);  mul_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_67: "f32[8, 368, 1, 1]" = torch.ops.aten.alias.default(alias_64);  alias_64 = None
    sub_48: "f32[8, 368, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_67)
    mul_332: "f32[8, 368, 1, 1]" = torch.ops.aten.mul.Tensor(alias_67, sub_48);  alias_67 = sub_48 = None
    mul_333: "f32[8, 368, 1, 1]" = torch.ops.aten.mul.Tensor(sum_4, mul_332);  sum_4 = mul_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_333, relu_51, primals_182, [368], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_333 = primals_182 = None
    getitem_91: "f32[8, 92, 1, 1]" = convolution_backward_1[0]
    getitem_92: "f32[368, 92, 1, 1]" = convolution_backward_1[1]
    getitem_93: "f32[368]" = convolution_backward_1[2];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_69: "f32[8, 92, 1, 1]" = torch.ops.aten.alias.default(relu_51);  relu_51 = None
    alias_70: "f32[8, 92, 1, 1]" = torch.ops.aten.alias.default(alias_69);  alias_69 = None
    le_1: "b8[8, 92, 1, 1]" = torch.ops.aten.le.Scalar(alias_70, 0);  alias_70 = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_1: "f32[8, 92, 1, 1]" = torch.ops.aten.where.self(le_1, scalar_tensor_1, getitem_91);  le_1 = scalar_tensor_1 = getitem_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(where_1, mean_12, primals_180, [92], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_1 = mean_12 = primals_180 = None
    getitem_94: "f32[8, 368, 1, 1]" = convolution_backward_2[0]
    getitem_95: "f32[92, 368, 1, 1]" = convolution_backward_2[1]
    getitem_96: "f32[92]" = convolution_backward_2[2];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_1: "f32[8, 368, 7, 7]" = torch.ops.aten.expand.default(getitem_94, [8, 368, 7, 7]);  getitem_94 = None
    div_1: "f32[8, 368, 7, 7]" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_233: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_331, div_1);  mul_331 = div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_72: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(relu_50);  relu_50 = None
    alias_73: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(alias_72);  alias_72 = None
    le_2: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(alias_73, 0);  alias_73 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_2: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_2, scalar_tensor_2, add_233);  le_2 = scalar_tensor_2 = add_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_188: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
    unsqueeze_189: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, 2);  unsqueeze_188 = None
    unsqueeze_190: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_189, 3);  unsqueeze_189 = None
    sum_5: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_49: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_190)
    mul_334: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_49);  sub_49 = None
    sum_6: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_334, [0, 2, 3]);  mul_334 = None
    mul_335: "f32[368]" = torch.ops.aten.mul.Tensor(sum_5, 0.002551020408163265)
    unsqueeze_191: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_335, 0);  mul_335 = None
    unsqueeze_192: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_191, 2);  unsqueeze_191 = None
    unsqueeze_193: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, 3);  unsqueeze_192 = None
    mul_336: "f32[368]" = torch.ops.aten.mul.Tensor(sum_6, 0.002551020408163265)
    mul_337: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
    mul_338: "f32[368]" = torch.ops.aten.mul.Tensor(mul_336, mul_337);  mul_336 = mul_337 = None
    unsqueeze_194: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_338, 0);  mul_338 = None
    unsqueeze_195: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, 2);  unsqueeze_194 = None
    unsqueeze_196: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_195, 3);  unsqueeze_195 = None
    mul_339: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_85);  primals_85 = None
    unsqueeze_197: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_339, 0);  mul_339 = None
    unsqueeze_198: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 2);  unsqueeze_197 = None
    unsqueeze_199: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, 3);  unsqueeze_198 = None
    sub_50: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_190);  convolution_66 = unsqueeze_190 = None
    mul_340: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_196);  sub_50 = unsqueeze_196 = None
    sub_51: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_2, mul_340);  where_2 = mul_340 = None
    sub_52: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_51, unsqueeze_193);  sub_51 = unsqueeze_193 = None
    mul_341: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_199);  sub_52 = unsqueeze_199 = None
    mul_342: "f32[368]" = torch.ops.aten.mul.Tensor(sum_6, squeeze_127);  sum_6 = squeeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_341, relu_49, primals_179, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 46, [True, True, False]);  mul_341 = primals_179 = None
    getitem_97: "f32[8, 368, 7, 7]" = convolution_backward_3[0]
    getitem_98: "f32[368, 8, 3, 3]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_75: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(relu_49);  relu_49 = None
    alias_76: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(alias_75);  alias_75 = None
    le_3: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(alias_76, 0);  alias_76 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_3: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_3, scalar_tensor_3, getitem_97);  le_3 = scalar_tensor_3 = getitem_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_200: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
    unsqueeze_201: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, 2);  unsqueeze_200 = None
    unsqueeze_202: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_201, 3);  unsqueeze_201 = None
    sum_7: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_53: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_202)
    mul_343: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_53);  sub_53 = None
    sum_8: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_343, [0, 2, 3]);  mul_343 = None
    mul_344: "f32[368]" = torch.ops.aten.mul.Tensor(sum_7, 0.002551020408163265)
    unsqueeze_203: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_344, 0);  mul_344 = None
    unsqueeze_204: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 2);  unsqueeze_203 = None
    unsqueeze_205: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, 3);  unsqueeze_204 = None
    mul_345: "f32[368]" = torch.ops.aten.mul.Tensor(sum_8, 0.002551020408163265)
    mul_346: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
    mul_347: "f32[368]" = torch.ops.aten.mul.Tensor(mul_345, mul_346);  mul_345 = mul_346 = None
    unsqueeze_206: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_347, 0);  mul_347 = None
    unsqueeze_207: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 2);  unsqueeze_206 = None
    unsqueeze_208: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_207, 3);  unsqueeze_207 = None
    mul_348: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_83);  primals_83 = None
    unsqueeze_209: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_348, 0);  mul_348 = None
    unsqueeze_210: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 2);  unsqueeze_209 = None
    unsqueeze_211: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, 3);  unsqueeze_210 = None
    sub_54: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_202);  convolution_65 = unsqueeze_202 = None
    mul_349: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_208);  sub_54 = unsqueeze_208 = None
    sub_55: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_3, mul_349);  where_3 = mul_349 = None
    sub_56: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_55, unsqueeze_205);  sub_55 = unsqueeze_205 = None
    mul_350: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_211);  sub_56 = unsqueeze_211 = None
    mul_351: "f32[368]" = torch.ops.aten.mul.Tensor(sum_8, squeeze_124);  sum_8 = squeeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_350, relu_48, primals_178, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_350 = relu_48 = primals_178 = None
    getitem_100: "f32[8, 368, 7, 7]" = convolution_backward_4[0]
    getitem_101: "f32[368, 368, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_234: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(where, getitem_100);  where = getitem_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    alias_77: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(alias_60);  alias_60 = None
    le_4: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(alias_77, 0);  alias_77 = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_4: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_4, scalar_tensor_4, add_234);  le_4 = scalar_tensor_4 = add_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_212: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
    unsqueeze_213: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 2);  unsqueeze_212 = None
    unsqueeze_214: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 3);  unsqueeze_213 = None
    sum_9: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_57: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_214)
    mul_352: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, sub_57);  sub_57 = None
    sum_10: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_352, [0, 2, 3]);  mul_352 = None
    mul_353: "f32[368]" = torch.ops.aten.mul.Tensor(sum_9, 0.002551020408163265)
    unsqueeze_215: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_353, 0);  mul_353 = None
    unsqueeze_216: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 2);  unsqueeze_215 = None
    unsqueeze_217: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, 3);  unsqueeze_216 = None
    mul_354: "f32[368]" = torch.ops.aten.mul.Tensor(sum_10, 0.002551020408163265)
    mul_355: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_356: "f32[368]" = torch.ops.aten.mul.Tensor(mul_354, mul_355);  mul_354 = mul_355 = None
    unsqueeze_218: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_356, 0);  mul_356 = None
    unsqueeze_219: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 2);  unsqueeze_218 = None
    unsqueeze_220: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 3);  unsqueeze_219 = None
    mul_357: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_81);  primals_81 = None
    unsqueeze_221: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_357, 0);  mul_357 = None
    unsqueeze_222: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 2);  unsqueeze_221 = None
    unsqueeze_223: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, 3);  unsqueeze_222 = None
    sub_58: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_214);  convolution_64 = unsqueeze_214 = None
    mul_358: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_220);  sub_58 = unsqueeze_220 = None
    sub_59: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_4, mul_358);  mul_358 = None
    sub_60: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_59, unsqueeze_217);  sub_59 = unsqueeze_217 = None
    mul_359: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_223);  sub_60 = unsqueeze_223 = None
    mul_360: "f32[368]" = torch.ops.aten.mul.Tensor(sum_10, squeeze_121);  sum_10 = squeeze_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_359, mul_291, primals_177, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_359 = mul_291 = primals_177 = None
    getitem_103: "f32[8, 368, 7, 7]" = convolution_backward_5[0]
    getitem_104: "f32[368, 368, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_361: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_103, relu_46)
    mul_362: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_103, sigmoid_11);  getitem_103 = sigmoid_11 = None
    sum_11: "f32[8, 368, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_361, [2, 3], True);  mul_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_78: "f32[8, 368, 1, 1]" = torch.ops.aten.alias.default(alias_59);  alias_59 = None
    sub_61: "f32[8, 368, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_78)
    mul_363: "f32[8, 368, 1, 1]" = torch.ops.aten.mul.Tensor(alias_78, sub_61);  alias_78 = sub_61 = None
    mul_364: "f32[8, 368, 1, 1]" = torch.ops.aten.mul.Tensor(sum_11, mul_363);  sum_11 = mul_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_364, relu_47, primals_175, [368], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_364 = primals_175 = None
    getitem_106: "f32[8, 92, 1, 1]" = convolution_backward_6[0]
    getitem_107: "f32[368, 92, 1, 1]" = convolution_backward_6[1]
    getitem_108: "f32[368]" = convolution_backward_6[2];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_80: "f32[8, 92, 1, 1]" = torch.ops.aten.alias.default(relu_47);  relu_47 = None
    alias_81: "f32[8, 92, 1, 1]" = torch.ops.aten.alias.default(alias_80);  alias_80 = None
    le_5: "b8[8, 92, 1, 1]" = torch.ops.aten.le.Scalar(alias_81, 0);  alias_81 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_5: "f32[8, 92, 1, 1]" = torch.ops.aten.where.self(le_5, scalar_tensor_5, getitem_106);  le_5 = scalar_tensor_5 = getitem_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(where_5, mean_11, primals_173, [92], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_5 = mean_11 = primals_173 = None
    getitem_109: "f32[8, 368, 1, 1]" = convolution_backward_7[0]
    getitem_110: "f32[92, 368, 1, 1]" = convolution_backward_7[1]
    getitem_111: "f32[92]" = convolution_backward_7[2];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_2: "f32[8, 368, 7, 7]" = torch.ops.aten.expand.default(getitem_109, [8, 368, 7, 7]);  getitem_109 = None
    div_2: "f32[8, 368, 7, 7]" = torch.ops.aten.div.Scalar(expand_2, 49);  expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_235: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_362, div_2);  mul_362 = div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_83: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(relu_46);  relu_46 = None
    alias_84: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(alias_83);  alias_83 = None
    le_6: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(alias_84, 0);  alias_84 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_6: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_6, scalar_tensor_6, add_235);  le_6 = scalar_tensor_6 = add_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_224: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
    unsqueeze_225: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 2);  unsqueeze_224 = None
    unsqueeze_226: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 3);  unsqueeze_225 = None
    sum_12: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_62: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_226)
    mul_365: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, sub_62);  sub_62 = None
    sum_13: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_365, [0, 2, 3]);  mul_365 = None
    mul_366: "f32[368]" = torch.ops.aten.mul.Tensor(sum_12, 0.002551020408163265)
    unsqueeze_227: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_366, 0);  mul_366 = None
    unsqueeze_228: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 2);  unsqueeze_227 = None
    unsqueeze_229: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, 3);  unsqueeze_228 = None
    mul_367: "f32[368]" = torch.ops.aten.mul.Tensor(sum_13, 0.002551020408163265)
    mul_368: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_369: "f32[368]" = torch.ops.aten.mul.Tensor(mul_367, mul_368);  mul_367 = mul_368 = None
    unsqueeze_230: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_369, 0);  mul_369 = None
    unsqueeze_231: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 2);  unsqueeze_230 = None
    unsqueeze_232: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 3);  unsqueeze_231 = None
    mul_370: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_79);  primals_79 = None
    unsqueeze_233: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_370, 0);  mul_370 = None
    unsqueeze_234: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 2);  unsqueeze_233 = None
    unsqueeze_235: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, 3);  unsqueeze_234 = None
    sub_63: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_226);  convolution_61 = unsqueeze_226 = None
    mul_371: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_232);  sub_63 = unsqueeze_232 = None
    sub_64: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_6, mul_371);  where_6 = mul_371 = None
    sub_65: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_64, unsqueeze_229);  sub_64 = unsqueeze_229 = None
    mul_372: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_235);  sub_65 = unsqueeze_235 = None
    mul_373: "f32[368]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_118);  sum_13 = squeeze_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_372, relu_45, primals_172, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 46, [True, True, False]);  mul_372 = primals_172 = None
    getitem_112: "f32[8, 368, 7, 7]" = convolution_backward_8[0]
    getitem_113: "f32[368, 8, 3, 3]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_86: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(relu_45);  relu_45 = None
    alias_87: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(alias_86);  alias_86 = None
    le_7: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(alias_87, 0);  alias_87 = None
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_7: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_7, scalar_tensor_7, getitem_112);  le_7 = scalar_tensor_7 = getitem_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_236: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
    unsqueeze_237: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 2);  unsqueeze_236 = None
    unsqueeze_238: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_237, 3);  unsqueeze_237 = None
    sum_14: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_66: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_238)
    mul_374: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, sub_66);  sub_66 = None
    sum_15: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_374, [0, 2, 3]);  mul_374 = None
    mul_375: "f32[368]" = torch.ops.aten.mul.Tensor(sum_14, 0.002551020408163265)
    unsqueeze_239: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_375, 0);  mul_375 = None
    unsqueeze_240: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 2);  unsqueeze_239 = None
    unsqueeze_241: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, 3);  unsqueeze_240 = None
    mul_376: "f32[368]" = torch.ops.aten.mul.Tensor(sum_15, 0.002551020408163265)
    mul_377: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_378: "f32[368]" = torch.ops.aten.mul.Tensor(mul_376, mul_377);  mul_376 = mul_377 = None
    unsqueeze_242: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_378, 0);  mul_378 = None
    unsqueeze_243: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 2);  unsqueeze_242 = None
    unsqueeze_244: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_243, 3);  unsqueeze_243 = None
    mul_379: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_77);  primals_77 = None
    unsqueeze_245: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_379, 0);  mul_379 = None
    unsqueeze_246: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 2);  unsqueeze_245 = None
    unsqueeze_247: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, 3);  unsqueeze_246 = None
    sub_67: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_238);  convolution_60 = unsqueeze_238 = None
    mul_380: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_244);  sub_67 = unsqueeze_244 = None
    sub_68: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_7, mul_380);  where_7 = mul_380 = None
    sub_69: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_68, unsqueeze_241);  sub_68 = unsqueeze_241 = None
    mul_381: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_247);  sub_69 = unsqueeze_247 = None
    mul_382: "f32[368]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_115);  sum_15 = squeeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_381, relu_44, primals_171, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_381 = relu_44 = primals_171 = None
    getitem_115: "f32[8, 368, 7, 7]" = convolution_backward_9[0]
    getitem_116: "f32[368, 368, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_236: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(where_4, getitem_115);  where_4 = getitem_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    alias_88: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(alias_55);  alias_55 = None
    le_8: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(alias_88, 0);  alias_88 = None
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_8: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_8, scalar_tensor_8, add_236);  le_8 = scalar_tensor_8 = add_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_248: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
    unsqueeze_249: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 2);  unsqueeze_248 = None
    unsqueeze_250: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 3);  unsqueeze_249 = None
    sum_16: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_70: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_250)
    mul_383: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_8, sub_70);  sub_70 = None
    sum_17: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_383, [0, 2, 3]);  mul_383 = None
    mul_384: "f32[368]" = torch.ops.aten.mul.Tensor(sum_16, 0.002551020408163265)
    unsqueeze_251: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_384, 0);  mul_384 = None
    unsqueeze_252: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 2);  unsqueeze_251 = None
    unsqueeze_253: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, 3);  unsqueeze_252 = None
    mul_385: "f32[368]" = torch.ops.aten.mul.Tensor(sum_17, 0.002551020408163265)
    mul_386: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_387: "f32[368]" = torch.ops.aten.mul.Tensor(mul_385, mul_386);  mul_385 = mul_386 = None
    unsqueeze_254: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_387, 0);  mul_387 = None
    unsqueeze_255: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 2);  unsqueeze_254 = None
    unsqueeze_256: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 3);  unsqueeze_255 = None
    mul_388: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_75);  primals_75 = None
    unsqueeze_257: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_388, 0);  mul_388 = None
    unsqueeze_258: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 2);  unsqueeze_257 = None
    unsqueeze_259: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, 3);  unsqueeze_258 = None
    sub_71: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_250);  convolution_59 = unsqueeze_250 = None
    mul_389: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_256);  sub_71 = unsqueeze_256 = None
    sub_72: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_8, mul_389);  mul_389 = None
    sub_73: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_72, unsqueeze_253);  sub_72 = unsqueeze_253 = None
    mul_390: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_259);  sub_73 = unsqueeze_259 = None
    mul_391: "f32[368]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_112);  sum_17 = squeeze_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_390, mul_269, primals_170, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_390 = mul_269 = primals_170 = None
    getitem_118: "f32[8, 368, 7, 7]" = convolution_backward_10[0]
    getitem_119: "f32[368, 368, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_392: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_118, relu_42)
    mul_393: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_118, sigmoid_10);  getitem_118 = sigmoid_10 = None
    sum_18: "f32[8, 368, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_392, [2, 3], True);  mul_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_89: "f32[8, 368, 1, 1]" = torch.ops.aten.alias.default(alias_54);  alias_54 = None
    sub_74: "f32[8, 368, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_89)
    mul_394: "f32[8, 368, 1, 1]" = torch.ops.aten.mul.Tensor(alias_89, sub_74);  alias_89 = sub_74 = None
    mul_395: "f32[8, 368, 1, 1]" = torch.ops.aten.mul.Tensor(sum_18, mul_394);  sum_18 = mul_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_395, relu_43, primals_168, [368], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_395 = primals_168 = None
    getitem_121: "f32[8, 92, 1, 1]" = convolution_backward_11[0]
    getitem_122: "f32[368, 92, 1, 1]" = convolution_backward_11[1]
    getitem_123: "f32[368]" = convolution_backward_11[2];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_91: "f32[8, 92, 1, 1]" = torch.ops.aten.alias.default(relu_43);  relu_43 = None
    alias_92: "f32[8, 92, 1, 1]" = torch.ops.aten.alias.default(alias_91);  alias_91 = None
    le_9: "b8[8, 92, 1, 1]" = torch.ops.aten.le.Scalar(alias_92, 0);  alias_92 = None
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_9: "f32[8, 92, 1, 1]" = torch.ops.aten.where.self(le_9, scalar_tensor_9, getitem_121);  le_9 = scalar_tensor_9 = getitem_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(where_9, mean_10, primals_166, [92], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_9 = mean_10 = primals_166 = None
    getitem_124: "f32[8, 368, 1, 1]" = convolution_backward_12[0]
    getitem_125: "f32[92, 368, 1, 1]" = convolution_backward_12[1]
    getitem_126: "f32[92]" = convolution_backward_12[2];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_3: "f32[8, 368, 7, 7]" = torch.ops.aten.expand.default(getitem_124, [8, 368, 7, 7]);  getitem_124 = None
    div_3: "f32[8, 368, 7, 7]" = torch.ops.aten.div.Scalar(expand_3, 49);  expand_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_237: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_393, div_3);  mul_393 = div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_94: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(relu_42);  relu_42 = None
    alias_95: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(alias_94);  alias_94 = None
    le_10: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(alias_95, 0);  alias_95 = None
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_10: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_10, scalar_tensor_10, add_237);  le_10 = scalar_tensor_10 = add_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_260: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
    unsqueeze_261: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 2);  unsqueeze_260 = None
    unsqueeze_262: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 3);  unsqueeze_261 = None
    sum_19: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_75: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_262)
    mul_396: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_10, sub_75);  sub_75 = None
    sum_20: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_396, [0, 2, 3]);  mul_396 = None
    mul_397: "f32[368]" = torch.ops.aten.mul.Tensor(sum_19, 0.002551020408163265)
    unsqueeze_263: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_397, 0);  mul_397 = None
    unsqueeze_264: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 2);  unsqueeze_263 = None
    unsqueeze_265: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 3);  unsqueeze_264 = None
    mul_398: "f32[368]" = torch.ops.aten.mul.Tensor(sum_20, 0.002551020408163265)
    mul_399: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_400: "f32[368]" = torch.ops.aten.mul.Tensor(mul_398, mul_399);  mul_398 = mul_399 = None
    unsqueeze_266: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_400, 0);  mul_400 = None
    unsqueeze_267: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 2);  unsqueeze_266 = None
    unsqueeze_268: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 3);  unsqueeze_267 = None
    mul_401: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_73);  primals_73 = None
    unsqueeze_269: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_401, 0);  mul_401 = None
    unsqueeze_270: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 2);  unsqueeze_269 = None
    unsqueeze_271: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 3);  unsqueeze_270 = None
    sub_76: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_262);  convolution_56 = unsqueeze_262 = None
    mul_402: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_268);  sub_76 = unsqueeze_268 = None
    sub_77: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_10, mul_402);  where_10 = mul_402 = None
    sub_78: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_77, unsqueeze_265);  sub_77 = unsqueeze_265 = None
    mul_403: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_271);  sub_78 = unsqueeze_271 = None
    mul_404: "f32[368]" = torch.ops.aten.mul.Tensor(sum_20, squeeze_109);  sum_20 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_403, relu_41, primals_165, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 46, [True, True, False]);  mul_403 = primals_165 = None
    getitem_127: "f32[8, 368, 7, 7]" = convolution_backward_13[0]
    getitem_128: "f32[368, 8, 3, 3]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_97: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(relu_41);  relu_41 = None
    alias_98: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(alias_97);  alias_97 = None
    le_11: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(alias_98, 0);  alias_98 = None
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_11: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_11, scalar_tensor_11, getitem_127);  le_11 = scalar_tensor_11 = getitem_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_272: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
    unsqueeze_273: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 2);  unsqueeze_272 = None
    unsqueeze_274: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 3);  unsqueeze_273 = None
    sum_21: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_79: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_274)
    mul_405: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_11, sub_79);  sub_79 = None
    sum_22: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_405, [0, 2, 3]);  mul_405 = None
    mul_406: "f32[368]" = torch.ops.aten.mul.Tensor(sum_21, 0.002551020408163265)
    unsqueeze_275: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_406, 0);  mul_406 = None
    unsqueeze_276: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 2);  unsqueeze_275 = None
    unsqueeze_277: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 3);  unsqueeze_276 = None
    mul_407: "f32[368]" = torch.ops.aten.mul.Tensor(sum_22, 0.002551020408163265)
    mul_408: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_409: "f32[368]" = torch.ops.aten.mul.Tensor(mul_407, mul_408);  mul_407 = mul_408 = None
    unsqueeze_278: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_409, 0);  mul_409 = None
    unsqueeze_279: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 2);  unsqueeze_278 = None
    unsqueeze_280: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 3);  unsqueeze_279 = None
    mul_410: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_71);  primals_71 = None
    unsqueeze_281: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_410, 0);  mul_410 = None
    unsqueeze_282: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 2);  unsqueeze_281 = None
    unsqueeze_283: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 3);  unsqueeze_282 = None
    sub_80: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_274);  convolution_55 = unsqueeze_274 = None
    mul_411: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_280);  sub_80 = unsqueeze_280 = None
    sub_81: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_11, mul_411);  where_11 = mul_411 = None
    sub_82: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_81, unsqueeze_277);  sub_81 = unsqueeze_277 = None
    mul_412: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_283);  sub_82 = unsqueeze_283 = None
    mul_413: "f32[368]" = torch.ops.aten.mul.Tensor(sum_22, squeeze_106);  sum_22 = squeeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_412, relu_40, primals_164, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_412 = relu_40 = primals_164 = None
    getitem_130: "f32[8, 368, 7, 7]" = convolution_backward_14[0]
    getitem_131: "f32[368, 368, 1, 1]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_238: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(where_8, getitem_130);  where_8 = getitem_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    alias_99: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(alias_50);  alias_50 = None
    le_12: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(alias_99, 0);  alias_99 = None
    scalar_tensor_12: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_12: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_12, scalar_tensor_12, add_238);  le_12 = scalar_tensor_12 = add_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_284: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
    unsqueeze_285: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 2);  unsqueeze_284 = None
    unsqueeze_286: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 3);  unsqueeze_285 = None
    sum_23: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_83: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_286)
    mul_414: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_12, sub_83);  sub_83 = None
    sum_24: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_414, [0, 2, 3]);  mul_414 = None
    mul_415: "f32[368]" = torch.ops.aten.mul.Tensor(sum_23, 0.002551020408163265)
    unsqueeze_287: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_415, 0);  mul_415 = None
    unsqueeze_288: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 2);  unsqueeze_287 = None
    unsqueeze_289: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 3);  unsqueeze_288 = None
    mul_416: "f32[368]" = torch.ops.aten.mul.Tensor(sum_24, 0.002551020408163265)
    mul_417: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_418: "f32[368]" = torch.ops.aten.mul.Tensor(mul_416, mul_417);  mul_416 = mul_417 = None
    unsqueeze_290: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_418, 0);  mul_418 = None
    unsqueeze_291: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 2);  unsqueeze_290 = None
    unsqueeze_292: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 3);  unsqueeze_291 = None
    mul_419: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_69);  primals_69 = None
    unsqueeze_293: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_419, 0);  mul_419 = None
    unsqueeze_294: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 2);  unsqueeze_293 = None
    unsqueeze_295: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 3);  unsqueeze_294 = None
    sub_84: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_286);  convolution_54 = unsqueeze_286 = None
    mul_420: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_292);  sub_84 = unsqueeze_292 = None
    sub_85: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_12, mul_420);  mul_420 = None
    sub_86: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_85, unsqueeze_289);  sub_85 = unsqueeze_289 = None
    mul_421: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_295);  sub_86 = unsqueeze_295 = None
    mul_422: "f32[368]" = torch.ops.aten.mul.Tensor(sum_24, squeeze_103);  sum_24 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_421, mul_247, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_421 = mul_247 = primals_163 = None
    getitem_133: "f32[8, 368, 7, 7]" = convolution_backward_15[0]
    getitem_134: "f32[368, 368, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_423: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_133, relu_38)
    mul_424: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_133, sigmoid_9);  getitem_133 = sigmoid_9 = None
    sum_25: "f32[8, 368, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_423, [2, 3], True);  mul_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_100: "f32[8, 368, 1, 1]" = torch.ops.aten.alias.default(alias_49);  alias_49 = None
    sub_87: "f32[8, 368, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_100)
    mul_425: "f32[8, 368, 1, 1]" = torch.ops.aten.mul.Tensor(alias_100, sub_87);  alias_100 = sub_87 = None
    mul_426: "f32[8, 368, 1, 1]" = torch.ops.aten.mul.Tensor(sum_25, mul_425);  sum_25 = mul_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_426, relu_39, primals_161, [368], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_426 = primals_161 = None
    getitem_136: "f32[8, 92, 1, 1]" = convolution_backward_16[0]
    getitem_137: "f32[368, 92, 1, 1]" = convolution_backward_16[1]
    getitem_138: "f32[368]" = convolution_backward_16[2];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_102: "f32[8, 92, 1, 1]" = torch.ops.aten.alias.default(relu_39);  relu_39 = None
    alias_103: "f32[8, 92, 1, 1]" = torch.ops.aten.alias.default(alias_102);  alias_102 = None
    le_13: "b8[8, 92, 1, 1]" = torch.ops.aten.le.Scalar(alias_103, 0);  alias_103 = None
    scalar_tensor_13: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_13: "f32[8, 92, 1, 1]" = torch.ops.aten.where.self(le_13, scalar_tensor_13, getitem_136);  le_13 = scalar_tensor_13 = getitem_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(where_13, mean_9, primals_159, [92], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_13 = mean_9 = primals_159 = None
    getitem_139: "f32[8, 368, 1, 1]" = convolution_backward_17[0]
    getitem_140: "f32[92, 368, 1, 1]" = convolution_backward_17[1]
    getitem_141: "f32[92]" = convolution_backward_17[2];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_4: "f32[8, 368, 7, 7]" = torch.ops.aten.expand.default(getitem_139, [8, 368, 7, 7]);  getitem_139 = None
    div_4: "f32[8, 368, 7, 7]" = torch.ops.aten.div.Scalar(expand_4, 49);  expand_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_239: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_424, div_4);  mul_424 = div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_105: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(relu_38);  relu_38 = None
    alias_106: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(alias_105);  alias_105 = None
    le_14: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(alias_106, 0);  alias_106 = None
    scalar_tensor_14: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_14: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_14, scalar_tensor_14, add_239);  le_14 = scalar_tensor_14 = add_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_296: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
    unsqueeze_297: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 2);  unsqueeze_296 = None
    unsqueeze_298: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 3);  unsqueeze_297 = None
    sum_26: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_88: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_298)
    mul_427: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_14, sub_88);  sub_88 = None
    sum_27: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_427, [0, 2, 3]);  mul_427 = None
    mul_428: "f32[368]" = torch.ops.aten.mul.Tensor(sum_26, 0.002551020408163265)
    unsqueeze_299: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_428, 0);  mul_428 = None
    unsqueeze_300: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 2);  unsqueeze_299 = None
    unsqueeze_301: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 3);  unsqueeze_300 = None
    mul_429: "f32[368]" = torch.ops.aten.mul.Tensor(sum_27, 0.002551020408163265)
    mul_430: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_431: "f32[368]" = torch.ops.aten.mul.Tensor(mul_429, mul_430);  mul_429 = mul_430 = None
    unsqueeze_302: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_431, 0);  mul_431 = None
    unsqueeze_303: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 2);  unsqueeze_302 = None
    unsqueeze_304: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 3);  unsqueeze_303 = None
    mul_432: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_67);  primals_67 = None
    unsqueeze_305: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_432, 0);  mul_432 = None
    unsqueeze_306: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 2);  unsqueeze_305 = None
    unsqueeze_307: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, 3);  unsqueeze_306 = None
    sub_89: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_298);  convolution_51 = unsqueeze_298 = None
    mul_433: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_304);  sub_89 = unsqueeze_304 = None
    sub_90: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_14, mul_433);  where_14 = mul_433 = None
    sub_91: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_90, unsqueeze_301);  sub_90 = unsqueeze_301 = None
    mul_434: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_307);  sub_91 = unsqueeze_307 = None
    mul_435: "f32[368]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_100);  sum_27 = squeeze_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_434, relu_37, primals_158, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 46, [True, True, False]);  mul_434 = primals_158 = None
    getitem_142: "f32[8, 368, 7, 7]" = convolution_backward_18[0]
    getitem_143: "f32[368, 8, 3, 3]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_108: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(relu_37);  relu_37 = None
    alias_109: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(alias_108);  alias_108 = None
    le_15: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(alias_109, 0);  alias_109 = None
    scalar_tensor_15: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_15: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_15, scalar_tensor_15, getitem_142);  le_15 = scalar_tensor_15 = getitem_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_308: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
    unsqueeze_309: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 2);  unsqueeze_308 = None
    unsqueeze_310: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 3);  unsqueeze_309 = None
    sum_28: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_92: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_310)
    mul_436: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_15, sub_92);  sub_92 = None
    sum_29: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_436, [0, 2, 3]);  mul_436 = None
    mul_437: "f32[368]" = torch.ops.aten.mul.Tensor(sum_28, 0.002551020408163265)
    unsqueeze_311: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_437, 0);  mul_437 = None
    unsqueeze_312: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 2);  unsqueeze_311 = None
    unsqueeze_313: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 3);  unsqueeze_312 = None
    mul_438: "f32[368]" = torch.ops.aten.mul.Tensor(sum_29, 0.002551020408163265)
    mul_439: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_440: "f32[368]" = torch.ops.aten.mul.Tensor(mul_438, mul_439);  mul_438 = mul_439 = None
    unsqueeze_314: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_440, 0);  mul_440 = None
    unsqueeze_315: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 2);  unsqueeze_314 = None
    unsqueeze_316: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 3);  unsqueeze_315 = None
    mul_441: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_65);  primals_65 = None
    unsqueeze_317: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_441, 0);  mul_441 = None
    unsqueeze_318: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None
    unsqueeze_319: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, 3);  unsqueeze_318 = None
    sub_93: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_310);  convolution_50 = unsqueeze_310 = None
    mul_442: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_316);  sub_93 = unsqueeze_316 = None
    sub_94: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_15, mul_442);  where_15 = mul_442 = None
    sub_95: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_94, unsqueeze_313);  sub_94 = unsqueeze_313 = None
    mul_443: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_319);  sub_95 = unsqueeze_319 = None
    mul_444: "f32[368]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_97);  sum_29 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_443, relu_36, primals_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_443 = relu_36 = primals_157 = None
    getitem_145: "f32[8, 368, 7, 7]" = convolution_backward_19[0]
    getitem_146: "f32[368, 368, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_240: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(where_12, getitem_145);  where_12 = getitem_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    alias_110: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(alias_45);  alias_45 = None
    le_16: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(alias_110, 0);  alias_110 = None
    scalar_tensor_16: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_16: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_16, scalar_tensor_16, add_240);  le_16 = scalar_tensor_16 = add_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_320: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
    unsqueeze_321: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 2);  unsqueeze_320 = None
    unsqueeze_322: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 3);  unsqueeze_321 = None
    sum_30: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_96: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_322)
    mul_445: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_16, sub_96);  sub_96 = None
    sum_31: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_445, [0, 2, 3]);  mul_445 = None
    mul_446: "f32[368]" = torch.ops.aten.mul.Tensor(sum_30, 0.002551020408163265)
    unsqueeze_323: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_446, 0);  mul_446 = None
    unsqueeze_324: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 2);  unsqueeze_323 = None
    unsqueeze_325: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, 3);  unsqueeze_324 = None
    mul_447: "f32[368]" = torch.ops.aten.mul.Tensor(sum_31, 0.002551020408163265)
    mul_448: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_449: "f32[368]" = torch.ops.aten.mul.Tensor(mul_447, mul_448);  mul_447 = mul_448 = None
    unsqueeze_326: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_449, 0);  mul_449 = None
    unsqueeze_327: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 2);  unsqueeze_326 = None
    unsqueeze_328: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 3);  unsqueeze_327 = None
    mul_450: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_63);  primals_63 = None
    unsqueeze_329: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_450, 0);  mul_450 = None
    unsqueeze_330: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 2);  unsqueeze_329 = None
    unsqueeze_331: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 3);  unsqueeze_330 = None
    sub_97: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_322);  convolution_49 = unsqueeze_322 = None
    mul_451: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_328);  sub_97 = unsqueeze_328 = None
    sub_98: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_16, mul_451);  mul_451 = None
    sub_99: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_98, unsqueeze_325);  sub_98 = unsqueeze_325 = None
    mul_452: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_331);  sub_99 = unsqueeze_331 = None
    mul_453: "f32[368]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_94);  sum_31 = squeeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_452, mul_225, primals_156, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_452 = mul_225 = primals_156 = None
    getitem_148: "f32[8, 368, 7, 7]" = convolution_backward_20[0]
    getitem_149: "f32[368, 368, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_454: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_148, relu_34)
    mul_455: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_148, sigmoid_8);  getitem_148 = sigmoid_8 = None
    sum_32: "f32[8, 368, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_454, [2, 3], True);  mul_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_111: "f32[8, 368, 1, 1]" = torch.ops.aten.alias.default(alias_44);  alias_44 = None
    sub_100: "f32[8, 368, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_111)
    mul_456: "f32[8, 368, 1, 1]" = torch.ops.aten.mul.Tensor(alias_111, sub_100);  alias_111 = sub_100 = None
    mul_457: "f32[8, 368, 1, 1]" = torch.ops.aten.mul.Tensor(sum_32, mul_456);  sum_32 = mul_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_457, relu_35, primals_154, [368], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_457 = primals_154 = None
    getitem_151: "f32[8, 92, 1, 1]" = convolution_backward_21[0]
    getitem_152: "f32[368, 92, 1, 1]" = convolution_backward_21[1]
    getitem_153: "f32[368]" = convolution_backward_21[2];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_113: "f32[8, 92, 1, 1]" = torch.ops.aten.alias.default(relu_35);  relu_35 = None
    alias_114: "f32[8, 92, 1, 1]" = torch.ops.aten.alias.default(alias_113);  alias_113 = None
    le_17: "b8[8, 92, 1, 1]" = torch.ops.aten.le.Scalar(alias_114, 0);  alias_114 = None
    scalar_tensor_17: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_17: "f32[8, 92, 1, 1]" = torch.ops.aten.where.self(le_17, scalar_tensor_17, getitem_151);  le_17 = scalar_tensor_17 = getitem_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(where_17, mean_8, primals_152, [92], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_17 = mean_8 = primals_152 = None
    getitem_154: "f32[8, 368, 1, 1]" = convolution_backward_22[0]
    getitem_155: "f32[92, 368, 1, 1]" = convolution_backward_22[1]
    getitem_156: "f32[92]" = convolution_backward_22[2];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_5: "f32[8, 368, 7, 7]" = torch.ops.aten.expand.default(getitem_154, [8, 368, 7, 7]);  getitem_154 = None
    div_5: "f32[8, 368, 7, 7]" = torch.ops.aten.div.Scalar(expand_5, 49);  expand_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_241: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_455, div_5);  mul_455 = div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_116: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(relu_34);  relu_34 = None
    alias_117: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(alias_116);  alias_116 = None
    le_18: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(alias_117, 0);  alias_117 = None
    scalar_tensor_18: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_18: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_18, scalar_tensor_18, add_241);  le_18 = scalar_tensor_18 = add_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_332: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_333: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 2);  unsqueeze_332 = None
    unsqueeze_334: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 3);  unsqueeze_333 = None
    sum_33: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_101: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_334)
    mul_458: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_18, sub_101);  sub_101 = None
    sum_34: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_458, [0, 2, 3]);  mul_458 = None
    mul_459: "f32[368]" = torch.ops.aten.mul.Tensor(sum_33, 0.002551020408163265)
    unsqueeze_335: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_459, 0);  mul_459 = None
    unsqueeze_336: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 2);  unsqueeze_335 = None
    unsqueeze_337: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 3);  unsqueeze_336 = None
    mul_460: "f32[368]" = torch.ops.aten.mul.Tensor(sum_34, 0.002551020408163265)
    mul_461: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_462: "f32[368]" = torch.ops.aten.mul.Tensor(mul_460, mul_461);  mul_460 = mul_461 = None
    unsqueeze_338: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_462, 0);  mul_462 = None
    unsqueeze_339: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 2);  unsqueeze_338 = None
    unsqueeze_340: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 3);  unsqueeze_339 = None
    mul_463: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_61);  primals_61 = None
    unsqueeze_341: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_463, 0);  mul_463 = None
    unsqueeze_342: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 2);  unsqueeze_341 = None
    unsqueeze_343: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 3);  unsqueeze_342 = None
    sub_102: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_334);  convolution_46 = unsqueeze_334 = None
    mul_464: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_340);  sub_102 = unsqueeze_340 = None
    sub_103: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_18, mul_464);  where_18 = mul_464 = None
    sub_104: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_103, unsqueeze_337);  sub_103 = unsqueeze_337 = None
    mul_465: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_343);  sub_104 = unsqueeze_343 = None
    mul_466: "f32[368]" = torch.ops.aten.mul.Tensor(sum_34, squeeze_91);  sum_34 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_465, relu_33, primals_151, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 46, [True, True, False]);  mul_465 = primals_151 = None
    getitem_157: "f32[8, 368, 7, 7]" = convolution_backward_23[0]
    getitem_158: "f32[368, 8, 3, 3]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_119: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(relu_33);  relu_33 = None
    alias_120: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(alias_119);  alias_119 = None
    le_19: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(alias_120, 0);  alias_120 = None
    scalar_tensor_19: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_19: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_19, scalar_tensor_19, getitem_157);  le_19 = scalar_tensor_19 = getitem_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_344: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    unsqueeze_345: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 2);  unsqueeze_344 = None
    unsqueeze_346: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 3);  unsqueeze_345 = None
    sum_35: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_105: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_346)
    mul_467: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_19, sub_105);  sub_105 = None
    sum_36: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_467, [0, 2, 3]);  mul_467 = None
    mul_468: "f32[368]" = torch.ops.aten.mul.Tensor(sum_35, 0.002551020408163265)
    unsqueeze_347: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_468, 0);  mul_468 = None
    unsqueeze_348: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 2);  unsqueeze_347 = None
    unsqueeze_349: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 3);  unsqueeze_348 = None
    mul_469: "f32[368]" = torch.ops.aten.mul.Tensor(sum_36, 0.002551020408163265)
    mul_470: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_471: "f32[368]" = torch.ops.aten.mul.Tensor(mul_469, mul_470);  mul_469 = mul_470 = None
    unsqueeze_350: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_471, 0);  mul_471 = None
    unsqueeze_351: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 2);  unsqueeze_350 = None
    unsqueeze_352: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 3);  unsqueeze_351 = None
    mul_472: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_59);  primals_59 = None
    unsqueeze_353: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_472, 0);  mul_472 = None
    unsqueeze_354: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 2);  unsqueeze_353 = None
    unsqueeze_355: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, 3);  unsqueeze_354 = None
    sub_106: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_346);  convolution_45 = unsqueeze_346 = None
    mul_473: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_352);  sub_106 = unsqueeze_352 = None
    sub_107: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_19, mul_473);  where_19 = mul_473 = None
    sub_108: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_107, unsqueeze_349);  sub_107 = unsqueeze_349 = None
    mul_474: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_355);  sub_108 = unsqueeze_355 = None
    mul_475: "f32[368]" = torch.ops.aten.mul.Tensor(sum_36, squeeze_88);  sum_36 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_474, relu_32, primals_150, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_474 = relu_32 = primals_150 = None
    getitem_160: "f32[8, 368, 7, 7]" = convolution_backward_24[0]
    getitem_161: "f32[368, 368, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_242: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(where_16, getitem_160);  where_16 = getitem_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    alias_121: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(alias_40);  alias_40 = None
    le_20: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(alias_121, 0);  alias_121 = None
    scalar_tensor_20: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_20: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_20, scalar_tensor_20, add_242);  le_20 = scalar_tensor_20 = add_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_356: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_357: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 2);  unsqueeze_356 = None
    unsqueeze_358: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 3);  unsqueeze_357 = None
    sum_37: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_109: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_358)
    mul_476: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_20, sub_109);  sub_109 = None
    sum_38: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_476, [0, 2, 3]);  mul_476 = None
    mul_477: "f32[368]" = torch.ops.aten.mul.Tensor(sum_37, 0.002551020408163265)
    unsqueeze_359: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_477, 0);  mul_477 = None
    unsqueeze_360: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 2);  unsqueeze_359 = None
    unsqueeze_361: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, 3);  unsqueeze_360 = None
    mul_478: "f32[368]" = torch.ops.aten.mul.Tensor(sum_38, 0.002551020408163265)
    mul_479: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_480: "f32[368]" = torch.ops.aten.mul.Tensor(mul_478, mul_479);  mul_478 = mul_479 = None
    unsqueeze_362: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_480, 0);  mul_480 = None
    unsqueeze_363: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 2);  unsqueeze_362 = None
    unsqueeze_364: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 3);  unsqueeze_363 = None
    mul_481: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_57);  primals_57 = None
    unsqueeze_365: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_481, 0);  mul_481 = None
    unsqueeze_366: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 2);  unsqueeze_365 = None
    unsqueeze_367: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, 3);  unsqueeze_366 = None
    sub_110: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_358);  convolution_44 = unsqueeze_358 = None
    mul_482: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_364);  sub_110 = unsqueeze_364 = None
    sub_111: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_20, mul_482);  mul_482 = None
    sub_112: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_111, unsqueeze_361);  sub_111 = unsqueeze_361 = None
    mul_483: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_367);  sub_112 = unsqueeze_367 = None
    mul_484: "f32[368]" = torch.ops.aten.mul.Tensor(sum_38, squeeze_85);  sum_38 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_483, mul_203, primals_149, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_483 = mul_203 = primals_149 = None
    getitem_163: "f32[8, 368, 7, 7]" = convolution_backward_25[0]
    getitem_164: "f32[368, 368, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_485: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_163, relu_30)
    mul_486: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_163, sigmoid_7);  getitem_163 = sigmoid_7 = None
    sum_39: "f32[8, 368, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_485, [2, 3], True);  mul_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_122: "f32[8, 368, 1, 1]" = torch.ops.aten.alias.default(alias_39);  alias_39 = None
    sub_113: "f32[8, 368, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_122)
    mul_487: "f32[8, 368, 1, 1]" = torch.ops.aten.mul.Tensor(alias_122, sub_113);  alias_122 = sub_113 = None
    mul_488: "f32[8, 368, 1, 1]" = torch.ops.aten.mul.Tensor(sum_39, mul_487);  sum_39 = mul_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_488, relu_31, primals_147, [368], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_488 = primals_147 = None
    getitem_166: "f32[8, 92, 1, 1]" = convolution_backward_26[0]
    getitem_167: "f32[368, 92, 1, 1]" = convolution_backward_26[1]
    getitem_168: "f32[368]" = convolution_backward_26[2];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_124: "f32[8, 92, 1, 1]" = torch.ops.aten.alias.default(relu_31);  relu_31 = None
    alias_125: "f32[8, 92, 1, 1]" = torch.ops.aten.alias.default(alias_124);  alias_124 = None
    le_21: "b8[8, 92, 1, 1]" = torch.ops.aten.le.Scalar(alias_125, 0);  alias_125 = None
    scalar_tensor_21: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_21: "f32[8, 92, 1, 1]" = torch.ops.aten.where.self(le_21, scalar_tensor_21, getitem_166);  le_21 = scalar_tensor_21 = getitem_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(where_21, mean_7, primals_145, [92], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_21 = mean_7 = primals_145 = None
    getitem_169: "f32[8, 368, 1, 1]" = convolution_backward_27[0]
    getitem_170: "f32[92, 368, 1, 1]" = convolution_backward_27[1]
    getitem_171: "f32[92]" = convolution_backward_27[2];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_6: "f32[8, 368, 7, 7]" = torch.ops.aten.expand.default(getitem_169, [8, 368, 7, 7]);  getitem_169 = None
    div_6: "f32[8, 368, 7, 7]" = torch.ops.aten.div.Scalar(expand_6, 49);  expand_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_243: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_486, div_6);  mul_486 = div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_127: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(relu_30);  relu_30 = None
    alias_128: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(alias_127);  alias_127 = None
    le_22: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(alias_128, 0);  alias_128 = None
    scalar_tensor_22: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_22: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_22, scalar_tensor_22, add_243);  le_22 = scalar_tensor_22 = add_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_368: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_369: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 2);  unsqueeze_368 = None
    unsqueeze_370: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 3);  unsqueeze_369 = None
    sum_40: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_114: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_370)
    mul_489: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_22, sub_114);  sub_114 = None
    sum_41: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_489, [0, 2, 3]);  mul_489 = None
    mul_490: "f32[368]" = torch.ops.aten.mul.Tensor(sum_40, 0.002551020408163265)
    unsqueeze_371: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_490, 0);  mul_490 = None
    unsqueeze_372: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 2);  unsqueeze_371 = None
    unsqueeze_373: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, 3);  unsqueeze_372 = None
    mul_491: "f32[368]" = torch.ops.aten.mul.Tensor(sum_41, 0.002551020408163265)
    mul_492: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_493: "f32[368]" = torch.ops.aten.mul.Tensor(mul_491, mul_492);  mul_491 = mul_492 = None
    unsqueeze_374: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_493, 0);  mul_493 = None
    unsqueeze_375: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 2);  unsqueeze_374 = None
    unsqueeze_376: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 3);  unsqueeze_375 = None
    mul_494: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_55);  primals_55 = None
    unsqueeze_377: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_494, 0);  mul_494 = None
    unsqueeze_378: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 2);  unsqueeze_377 = None
    unsqueeze_379: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 3);  unsqueeze_378 = None
    sub_115: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_370);  convolution_41 = unsqueeze_370 = None
    mul_495: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_376);  sub_115 = unsqueeze_376 = None
    sub_116: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_22, mul_495);  where_22 = mul_495 = None
    sub_117: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_116, unsqueeze_373);  sub_116 = unsqueeze_373 = None
    mul_496: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_379);  sub_117 = unsqueeze_379 = None
    mul_497: "f32[368]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_82);  sum_41 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_496, relu_29, primals_144, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 46, [True, True, False]);  mul_496 = primals_144 = None
    getitem_172: "f32[8, 368, 7, 7]" = convolution_backward_28[0]
    getitem_173: "f32[368, 8, 3, 3]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_130: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(relu_29);  relu_29 = None
    alias_131: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(alias_130);  alias_130 = None
    le_23: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(alias_131, 0);  alias_131 = None
    scalar_tensor_23: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_23: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_23, scalar_tensor_23, getitem_172);  le_23 = scalar_tensor_23 = getitem_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_380: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_381: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 2);  unsqueeze_380 = None
    unsqueeze_382: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 3);  unsqueeze_381 = None
    sum_42: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_118: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_382)
    mul_498: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_23, sub_118);  sub_118 = None
    sum_43: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_498, [0, 2, 3]);  mul_498 = None
    mul_499: "f32[368]" = torch.ops.aten.mul.Tensor(sum_42, 0.002551020408163265)
    unsqueeze_383: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_499, 0);  mul_499 = None
    unsqueeze_384: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 2);  unsqueeze_383 = None
    unsqueeze_385: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, 3);  unsqueeze_384 = None
    mul_500: "f32[368]" = torch.ops.aten.mul.Tensor(sum_43, 0.002551020408163265)
    mul_501: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_502: "f32[368]" = torch.ops.aten.mul.Tensor(mul_500, mul_501);  mul_500 = mul_501 = None
    unsqueeze_386: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_502, 0);  mul_502 = None
    unsqueeze_387: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 2);  unsqueeze_386 = None
    unsqueeze_388: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 3);  unsqueeze_387 = None
    mul_503: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_53);  primals_53 = None
    unsqueeze_389: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_503, 0);  mul_503 = None
    unsqueeze_390: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 2);  unsqueeze_389 = None
    unsqueeze_391: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, 3);  unsqueeze_390 = None
    sub_119: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_382);  convolution_40 = unsqueeze_382 = None
    mul_504: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_388);  sub_119 = unsqueeze_388 = None
    sub_120: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_23, mul_504);  where_23 = mul_504 = None
    sub_121: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_120, unsqueeze_385);  sub_120 = unsqueeze_385 = None
    mul_505: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_391);  sub_121 = unsqueeze_391 = None
    mul_506: "f32[368]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_79);  sum_43 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_505, relu_28, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_505 = relu_28 = primals_143 = None
    getitem_175: "f32[8, 368, 7, 7]" = convolution_backward_29[0]
    getitem_176: "f32[368, 368, 1, 1]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_244: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(where_20, getitem_175);  where_20 = getitem_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    alias_132: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(alias_35);  alias_35 = None
    le_24: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(alias_132, 0);  alias_132 = None
    scalar_tensor_24: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_24: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_24, scalar_tensor_24, add_244);  le_24 = scalar_tensor_24 = add_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_392: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_393: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 2);  unsqueeze_392 = None
    unsqueeze_394: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 3);  unsqueeze_393 = None
    sum_44: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_122: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_394)
    mul_507: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_24, sub_122);  sub_122 = None
    sum_45: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_507, [0, 2, 3]);  mul_507 = None
    mul_508: "f32[368]" = torch.ops.aten.mul.Tensor(sum_44, 0.002551020408163265)
    unsqueeze_395: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_508, 0);  mul_508 = None
    unsqueeze_396: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 2);  unsqueeze_395 = None
    unsqueeze_397: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, 3);  unsqueeze_396 = None
    mul_509: "f32[368]" = torch.ops.aten.mul.Tensor(sum_45, 0.002551020408163265)
    mul_510: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_511: "f32[368]" = torch.ops.aten.mul.Tensor(mul_509, mul_510);  mul_509 = mul_510 = None
    unsqueeze_398: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_511, 0);  mul_511 = None
    unsqueeze_399: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 2);  unsqueeze_398 = None
    unsqueeze_400: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 3);  unsqueeze_399 = None
    mul_512: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_51);  primals_51 = None
    unsqueeze_401: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_512, 0);  mul_512 = None
    unsqueeze_402: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 2);  unsqueeze_401 = None
    unsqueeze_403: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 3);  unsqueeze_402 = None
    sub_123: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_394);  convolution_39 = unsqueeze_394 = None
    mul_513: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_400);  sub_123 = unsqueeze_400 = None
    sub_124: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_24, mul_513);  mul_513 = None
    sub_125: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_124, unsqueeze_397);  sub_124 = unsqueeze_397 = None
    mul_514: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_403);  sub_125 = unsqueeze_403 = None
    mul_515: "f32[368]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_76);  sum_45 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_514, relu_24, primals_142, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_514 = primals_142 = None
    getitem_178: "f32[8, 152, 14, 14]" = convolution_backward_30[0]
    getitem_179: "f32[368, 152, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_404: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_405: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 2);  unsqueeze_404 = None
    unsqueeze_406: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 3);  unsqueeze_405 = None
    sum_46: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_126: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_406)
    mul_516: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_24, sub_126);  sub_126 = None
    sum_47: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_516, [0, 2, 3]);  mul_516 = None
    mul_517: "f32[368]" = torch.ops.aten.mul.Tensor(sum_46, 0.002551020408163265)
    unsqueeze_407: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_517, 0);  mul_517 = None
    unsqueeze_408: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 2);  unsqueeze_407 = None
    unsqueeze_409: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, 3);  unsqueeze_408 = None
    mul_518: "f32[368]" = torch.ops.aten.mul.Tensor(sum_47, 0.002551020408163265)
    mul_519: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_520: "f32[368]" = torch.ops.aten.mul.Tensor(mul_518, mul_519);  mul_518 = mul_519 = None
    unsqueeze_410: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_520, 0);  mul_520 = None
    unsqueeze_411: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 2);  unsqueeze_410 = None
    unsqueeze_412: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 3);  unsqueeze_411 = None
    mul_521: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_49);  primals_49 = None
    unsqueeze_413: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_521, 0);  mul_521 = None
    unsqueeze_414: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 2);  unsqueeze_413 = None
    unsqueeze_415: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 3);  unsqueeze_414 = None
    sub_127: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_406);  convolution_38 = unsqueeze_406 = None
    mul_522: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_412);  sub_127 = unsqueeze_412 = None
    sub_128: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_24, mul_522);  where_24 = mul_522 = None
    sub_129: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_128, unsqueeze_409);  sub_128 = unsqueeze_409 = None
    mul_523: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_415);  sub_129 = unsqueeze_415 = None
    mul_524: "f32[368]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_73);  sum_47 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_523, mul_174, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_523 = mul_174 = primals_141 = None
    getitem_181: "f32[8, 368, 7, 7]" = convolution_backward_31[0]
    getitem_182: "f32[368, 368, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_525: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_181, relu_26)
    mul_526: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_181, sigmoid_6);  getitem_181 = sigmoid_6 = None
    sum_48: "f32[8, 368, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_525, [2, 3], True);  mul_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_133: "f32[8, 368, 1, 1]" = torch.ops.aten.alias.default(alias_34);  alias_34 = None
    sub_130: "f32[8, 368, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_133)
    mul_527: "f32[8, 368, 1, 1]" = torch.ops.aten.mul.Tensor(alias_133, sub_130);  alias_133 = sub_130 = None
    mul_528: "f32[8, 368, 1, 1]" = torch.ops.aten.mul.Tensor(sum_48, mul_527);  sum_48 = mul_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_528, relu_27, primals_139, [368], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_528 = primals_139 = None
    getitem_184: "f32[8, 38, 1, 1]" = convolution_backward_32[0]
    getitem_185: "f32[368, 38, 1, 1]" = convolution_backward_32[1]
    getitem_186: "f32[368]" = convolution_backward_32[2];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_135: "f32[8, 38, 1, 1]" = torch.ops.aten.alias.default(relu_27);  relu_27 = None
    alias_136: "f32[8, 38, 1, 1]" = torch.ops.aten.alias.default(alias_135);  alias_135 = None
    le_25: "b8[8, 38, 1, 1]" = torch.ops.aten.le.Scalar(alias_136, 0);  alias_136 = None
    scalar_tensor_25: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_25: "f32[8, 38, 1, 1]" = torch.ops.aten.where.self(le_25, scalar_tensor_25, getitem_184);  le_25 = scalar_tensor_25 = getitem_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(where_25, mean_6, primals_137, [38], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_25 = mean_6 = primals_137 = None
    getitem_187: "f32[8, 368, 1, 1]" = convolution_backward_33[0]
    getitem_188: "f32[38, 368, 1, 1]" = convolution_backward_33[1]
    getitem_189: "f32[38]" = convolution_backward_33[2];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_7: "f32[8, 368, 7, 7]" = torch.ops.aten.expand.default(getitem_187, [8, 368, 7, 7]);  getitem_187 = None
    div_7: "f32[8, 368, 7, 7]" = torch.ops.aten.div.Scalar(expand_7, 49);  expand_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_245: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_526, div_7);  mul_526 = div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_138: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(relu_26);  relu_26 = None
    alias_139: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(alias_138);  alias_138 = None
    le_26: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(alias_139, 0);  alias_139 = None
    scalar_tensor_26: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_26: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_26, scalar_tensor_26, add_245);  le_26 = scalar_tensor_26 = add_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_416: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_417: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 2);  unsqueeze_416 = None
    unsqueeze_418: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 3);  unsqueeze_417 = None
    sum_49: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_131: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_418)
    mul_529: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_26, sub_131);  sub_131 = None
    sum_50: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_529, [0, 2, 3]);  mul_529 = None
    mul_530: "f32[368]" = torch.ops.aten.mul.Tensor(sum_49, 0.002551020408163265)
    unsqueeze_419: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_530, 0);  mul_530 = None
    unsqueeze_420: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 2);  unsqueeze_419 = None
    unsqueeze_421: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 3);  unsqueeze_420 = None
    mul_531: "f32[368]" = torch.ops.aten.mul.Tensor(sum_50, 0.002551020408163265)
    mul_532: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_533: "f32[368]" = torch.ops.aten.mul.Tensor(mul_531, mul_532);  mul_531 = mul_532 = None
    unsqueeze_422: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_533, 0);  mul_533 = None
    unsqueeze_423: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 2);  unsqueeze_422 = None
    unsqueeze_424: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 3);  unsqueeze_423 = None
    mul_534: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_47);  primals_47 = None
    unsqueeze_425: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_534, 0);  mul_534 = None
    unsqueeze_426: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 2);  unsqueeze_425 = None
    unsqueeze_427: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 3);  unsqueeze_426 = None
    sub_132: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_418);  convolution_35 = unsqueeze_418 = None
    mul_535: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_424);  sub_132 = unsqueeze_424 = None
    sub_133: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_26, mul_535);  where_26 = mul_535 = None
    sub_134: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_133, unsqueeze_421);  sub_133 = unsqueeze_421 = None
    mul_536: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_427);  sub_134 = unsqueeze_427 = None
    mul_537: "f32[368]" = torch.ops.aten.mul.Tensor(sum_50, squeeze_70);  sum_50 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_536, relu_25, primals_136, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 46, [True, True, False]);  mul_536 = primals_136 = None
    getitem_190: "f32[8, 368, 14, 14]" = convolution_backward_34[0]
    getitem_191: "f32[368, 8, 3, 3]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_141: "f32[8, 368, 14, 14]" = torch.ops.aten.alias.default(relu_25);  relu_25 = None
    alias_142: "f32[8, 368, 14, 14]" = torch.ops.aten.alias.default(alias_141);  alias_141 = None
    le_27: "b8[8, 368, 14, 14]" = torch.ops.aten.le.Scalar(alias_142, 0);  alias_142 = None
    scalar_tensor_27: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_27: "f32[8, 368, 14, 14]" = torch.ops.aten.where.self(le_27, scalar_tensor_27, getitem_190);  le_27 = scalar_tensor_27 = getitem_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_428: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_429: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 2);  unsqueeze_428 = None
    unsqueeze_430: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 3);  unsqueeze_429 = None
    sum_51: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_135: "f32[8, 368, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_430)
    mul_538: "f32[8, 368, 14, 14]" = torch.ops.aten.mul.Tensor(where_27, sub_135);  sub_135 = None
    sum_52: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_538, [0, 2, 3]);  mul_538 = None
    mul_539: "f32[368]" = torch.ops.aten.mul.Tensor(sum_51, 0.0006377551020408163)
    unsqueeze_431: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_539, 0);  mul_539 = None
    unsqueeze_432: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 2);  unsqueeze_431 = None
    unsqueeze_433: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 3);  unsqueeze_432 = None
    mul_540: "f32[368]" = torch.ops.aten.mul.Tensor(sum_52, 0.0006377551020408163)
    mul_541: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_542: "f32[368]" = torch.ops.aten.mul.Tensor(mul_540, mul_541);  mul_540 = mul_541 = None
    unsqueeze_434: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_542, 0);  mul_542 = None
    unsqueeze_435: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 2);  unsqueeze_434 = None
    unsqueeze_436: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 3);  unsqueeze_435 = None
    mul_543: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_45);  primals_45 = None
    unsqueeze_437: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_543, 0);  mul_543 = None
    unsqueeze_438: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 2);  unsqueeze_437 = None
    unsqueeze_439: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 3);  unsqueeze_438 = None
    sub_136: "f32[8, 368, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_430);  convolution_34 = unsqueeze_430 = None
    mul_544: "f32[8, 368, 14, 14]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_436);  sub_136 = unsqueeze_436 = None
    sub_137: "f32[8, 368, 14, 14]" = torch.ops.aten.sub.Tensor(where_27, mul_544);  where_27 = mul_544 = None
    sub_138: "f32[8, 368, 14, 14]" = torch.ops.aten.sub.Tensor(sub_137, unsqueeze_433);  sub_137 = unsqueeze_433 = None
    mul_545: "f32[8, 368, 14, 14]" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_439);  sub_138 = unsqueeze_439 = None
    mul_546: "f32[368]" = torch.ops.aten.mul.Tensor(sum_52, squeeze_67);  sum_52 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_545, relu_24, primals_135, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_545 = relu_24 = primals_135 = None
    getitem_193: "f32[8, 152, 14, 14]" = convolution_backward_35[0]
    getitem_194: "f32[368, 152, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_246: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(getitem_178, getitem_193);  getitem_178 = getitem_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    alias_143: "f32[8, 152, 14, 14]" = torch.ops.aten.alias.default(alias_30);  alias_30 = None
    le_28: "b8[8, 152, 14, 14]" = torch.ops.aten.le.Scalar(alias_143, 0);  alias_143 = None
    scalar_tensor_28: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_28: "f32[8, 152, 14, 14]" = torch.ops.aten.where.self(le_28, scalar_tensor_28, add_246);  le_28 = scalar_tensor_28 = add_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_440: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_441: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 2);  unsqueeze_440 = None
    unsqueeze_442: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 3);  unsqueeze_441 = None
    sum_53: "f32[152]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    sub_139: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_442)
    mul_547: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(where_28, sub_139);  sub_139 = None
    sum_54: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_547, [0, 2, 3]);  mul_547 = None
    mul_548: "f32[152]" = torch.ops.aten.mul.Tensor(sum_53, 0.0006377551020408163)
    unsqueeze_443: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_548, 0);  mul_548 = None
    unsqueeze_444: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 2);  unsqueeze_443 = None
    unsqueeze_445: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, 3);  unsqueeze_444 = None
    mul_549: "f32[152]" = torch.ops.aten.mul.Tensor(sum_54, 0.0006377551020408163)
    mul_550: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_551: "f32[152]" = torch.ops.aten.mul.Tensor(mul_549, mul_550);  mul_549 = mul_550 = None
    unsqueeze_446: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_551, 0);  mul_551 = None
    unsqueeze_447: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 2);  unsqueeze_446 = None
    unsqueeze_448: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 3);  unsqueeze_447 = None
    mul_552: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_43);  primals_43 = None
    unsqueeze_449: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_552, 0);  mul_552 = None
    unsqueeze_450: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
    unsqueeze_451: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 3);  unsqueeze_450 = None
    sub_140: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_442);  convolution_33 = unsqueeze_442 = None
    mul_553: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_448);  sub_140 = unsqueeze_448 = None
    sub_141: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(where_28, mul_553);  mul_553 = None
    sub_142: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(sub_141, unsqueeze_445);  sub_141 = unsqueeze_445 = None
    mul_554: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_451);  sub_142 = unsqueeze_451 = None
    mul_555: "f32[152]" = torch.ops.aten.mul.Tensor(sum_54, squeeze_64);  sum_54 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_554, mul_152, primals_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_554 = mul_152 = primals_134 = None
    getitem_196: "f32[8, 152, 14, 14]" = convolution_backward_36[0]
    getitem_197: "f32[152, 152, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_556: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_196, relu_22)
    mul_557: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_196, sigmoid_5);  getitem_196 = sigmoid_5 = None
    sum_55: "f32[8, 152, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_556, [2, 3], True);  mul_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_144: "f32[8, 152, 1, 1]" = torch.ops.aten.alias.default(alias_29);  alias_29 = None
    sub_143: "f32[8, 152, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_144)
    mul_558: "f32[8, 152, 1, 1]" = torch.ops.aten.mul.Tensor(alias_144, sub_143);  alias_144 = sub_143 = None
    mul_559: "f32[8, 152, 1, 1]" = torch.ops.aten.mul.Tensor(sum_55, mul_558);  sum_55 = mul_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_559, relu_23, primals_132, [152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_559 = primals_132 = None
    getitem_199: "f32[8, 38, 1, 1]" = convolution_backward_37[0]
    getitem_200: "f32[152, 38, 1, 1]" = convolution_backward_37[1]
    getitem_201: "f32[152]" = convolution_backward_37[2];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_146: "f32[8, 38, 1, 1]" = torch.ops.aten.alias.default(relu_23);  relu_23 = None
    alias_147: "f32[8, 38, 1, 1]" = torch.ops.aten.alias.default(alias_146);  alias_146 = None
    le_29: "b8[8, 38, 1, 1]" = torch.ops.aten.le.Scalar(alias_147, 0);  alias_147 = None
    scalar_tensor_29: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_29: "f32[8, 38, 1, 1]" = torch.ops.aten.where.self(le_29, scalar_tensor_29, getitem_199);  le_29 = scalar_tensor_29 = getitem_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(where_29, mean_5, primals_130, [38], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_29 = mean_5 = primals_130 = None
    getitem_202: "f32[8, 152, 1, 1]" = convolution_backward_38[0]
    getitem_203: "f32[38, 152, 1, 1]" = convolution_backward_38[1]
    getitem_204: "f32[38]" = convolution_backward_38[2];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_8: "f32[8, 152, 14, 14]" = torch.ops.aten.expand.default(getitem_202, [8, 152, 14, 14]);  getitem_202 = None
    div_8: "f32[8, 152, 14, 14]" = torch.ops.aten.div.Scalar(expand_8, 196);  expand_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_247: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_557, div_8);  mul_557 = div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_149: "f32[8, 152, 14, 14]" = torch.ops.aten.alias.default(relu_22);  relu_22 = None
    alias_150: "f32[8, 152, 14, 14]" = torch.ops.aten.alias.default(alias_149);  alias_149 = None
    le_30: "b8[8, 152, 14, 14]" = torch.ops.aten.le.Scalar(alias_150, 0);  alias_150 = None
    scalar_tensor_30: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_30: "f32[8, 152, 14, 14]" = torch.ops.aten.where.self(le_30, scalar_tensor_30, add_247);  le_30 = scalar_tensor_30 = add_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_452: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_453: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 2);  unsqueeze_452 = None
    unsqueeze_454: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 3);  unsqueeze_453 = None
    sum_56: "f32[152]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
    sub_144: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_454)
    mul_560: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(where_30, sub_144);  sub_144 = None
    sum_57: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_560, [0, 2, 3]);  mul_560 = None
    mul_561: "f32[152]" = torch.ops.aten.mul.Tensor(sum_56, 0.0006377551020408163)
    unsqueeze_455: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_561, 0);  mul_561 = None
    unsqueeze_456: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 2);  unsqueeze_455 = None
    unsqueeze_457: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, 3);  unsqueeze_456 = None
    mul_562: "f32[152]" = torch.ops.aten.mul.Tensor(sum_57, 0.0006377551020408163)
    mul_563: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_564: "f32[152]" = torch.ops.aten.mul.Tensor(mul_562, mul_563);  mul_562 = mul_563 = None
    unsqueeze_458: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_564, 0);  mul_564 = None
    unsqueeze_459: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 2);  unsqueeze_458 = None
    unsqueeze_460: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 3);  unsqueeze_459 = None
    mul_565: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_41);  primals_41 = None
    unsqueeze_461: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_565, 0);  mul_565 = None
    unsqueeze_462: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
    unsqueeze_463: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 3);  unsqueeze_462 = None
    sub_145: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_454);  convolution_30 = unsqueeze_454 = None
    mul_566: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_460);  sub_145 = unsqueeze_460 = None
    sub_146: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(where_30, mul_566);  where_30 = mul_566 = None
    sub_147: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(sub_146, unsqueeze_457);  sub_146 = unsqueeze_457 = None
    mul_567: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_463);  sub_147 = unsqueeze_463 = None
    mul_568: "f32[152]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_61);  sum_57 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_567, relu_21, primals_129, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 19, [True, True, False]);  mul_567 = primals_129 = None
    getitem_205: "f32[8, 152, 14, 14]" = convolution_backward_39[0]
    getitem_206: "f32[152, 8, 3, 3]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_152: "f32[8, 152, 14, 14]" = torch.ops.aten.alias.default(relu_21);  relu_21 = None
    alias_153: "f32[8, 152, 14, 14]" = torch.ops.aten.alias.default(alias_152);  alias_152 = None
    le_31: "b8[8, 152, 14, 14]" = torch.ops.aten.le.Scalar(alias_153, 0);  alias_153 = None
    scalar_tensor_31: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_31: "f32[8, 152, 14, 14]" = torch.ops.aten.where.self(le_31, scalar_tensor_31, getitem_205);  le_31 = scalar_tensor_31 = getitem_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_464: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_465: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 2);  unsqueeze_464 = None
    unsqueeze_466: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 3);  unsqueeze_465 = None
    sum_58: "f32[152]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_148: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_466)
    mul_569: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(where_31, sub_148);  sub_148 = None
    sum_59: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_569, [0, 2, 3]);  mul_569 = None
    mul_570: "f32[152]" = torch.ops.aten.mul.Tensor(sum_58, 0.0006377551020408163)
    unsqueeze_467: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_570, 0);  mul_570 = None
    unsqueeze_468: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 2);  unsqueeze_467 = None
    unsqueeze_469: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, 3);  unsqueeze_468 = None
    mul_571: "f32[152]" = torch.ops.aten.mul.Tensor(sum_59, 0.0006377551020408163)
    mul_572: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_573: "f32[152]" = torch.ops.aten.mul.Tensor(mul_571, mul_572);  mul_571 = mul_572 = None
    unsqueeze_470: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_573, 0);  mul_573 = None
    unsqueeze_471: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 2);  unsqueeze_470 = None
    unsqueeze_472: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 3);  unsqueeze_471 = None
    mul_574: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_39);  primals_39 = None
    unsqueeze_473: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_574, 0);  mul_574 = None
    unsqueeze_474: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
    unsqueeze_475: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 3);  unsqueeze_474 = None
    sub_149: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_466);  convolution_29 = unsqueeze_466 = None
    mul_575: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_472);  sub_149 = unsqueeze_472 = None
    sub_150: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(where_31, mul_575);  where_31 = mul_575 = None
    sub_151: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(sub_150, unsqueeze_469);  sub_150 = unsqueeze_469 = None
    mul_576: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_475);  sub_151 = unsqueeze_475 = None
    mul_577: "f32[152]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_58);  sum_59 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_576, relu_20, primals_128, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_576 = relu_20 = primals_128 = None
    getitem_208: "f32[8, 152, 14, 14]" = convolution_backward_40[0]
    getitem_209: "f32[152, 152, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_248: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(where_28, getitem_208);  where_28 = getitem_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    alias_154: "f32[8, 152, 14, 14]" = torch.ops.aten.alias.default(alias_25);  alias_25 = None
    le_32: "b8[8, 152, 14, 14]" = torch.ops.aten.le.Scalar(alias_154, 0);  alias_154 = None
    scalar_tensor_32: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_32: "f32[8, 152, 14, 14]" = torch.ops.aten.where.self(le_32, scalar_tensor_32, add_248);  le_32 = scalar_tensor_32 = add_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_476: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_477: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 2);  unsqueeze_476 = None
    unsqueeze_478: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 3);  unsqueeze_477 = None
    sum_60: "f32[152]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
    sub_152: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_478)
    mul_578: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(where_32, sub_152);  sub_152 = None
    sum_61: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_578, [0, 2, 3]);  mul_578 = None
    mul_579: "f32[152]" = torch.ops.aten.mul.Tensor(sum_60, 0.0006377551020408163)
    unsqueeze_479: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_579, 0);  mul_579 = None
    unsqueeze_480: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 2);  unsqueeze_479 = None
    unsqueeze_481: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 3);  unsqueeze_480 = None
    mul_580: "f32[152]" = torch.ops.aten.mul.Tensor(sum_61, 0.0006377551020408163)
    mul_581: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_582: "f32[152]" = torch.ops.aten.mul.Tensor(mul_580, mul_581);  mul_580 = mul_581 = None
    unsqueeze_482: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_582, 0);  mul_582 = None
    unsqueeze_483: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 2);  unsqueeze_482 = None
    unsqueeze_484: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 3);  unsqueeze_483 = None
    mul_583: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_37);  primals_37 = None
    unsqueeze_485: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_583, 0);  mul_583 = None
    unsqueeze_486: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
    unsqueeze_487: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 3);  unsqueeze_486 = None
    sub_153: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_478);  convolution_28 = unsqueeze_478 = None
    mul_584: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_484);  sub_153 = unsqueeze_484 = None
    sub_154: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(where_32, mul_584);  mul_584 = None
    sub_155: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(sub_154, unsqueeze_481);  sub_154 = unsqueeze_481 = None
    mul_585: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_487);  sub_155 = unsqueeze_487 = None
    mul_586: "f32[152]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_55);  sum_61 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_585, mul_130, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_585 = mul_130 = primals_127 = None
    getitem_211: "f32[8, 152, 14, 14]" = convolution_backward_41[0]
    getitem_212: "f32[152, 152, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_587: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_211, relu_18)
    mul_588: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_211, sigmoid_4);  getitem_211 = sigmoid_4 = None
    sum_62: "f32[8, 152, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_587, [2, 3], True);  mul_587 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_155: "f32[8, 152, 1, 1]" = torch.ops.aten.alias.default(alias_24);  alias_24 = None
    sub_156: "f32[8, 152, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_155)
    mul_589: "f32[8, 152, 1, 1]" = torch.ops.aten.mul.Tensor(alias_155, sub_156);  alias_155 = sub_156 = None
    mul_590: "f32[8, 152, 1, 1]" = torch.ops.aten.mul.Tensor(sum_62, mul_589);  sum_62 = mul_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_590, relu_19, primals_125, [152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_590 = primals_125 = None
    getitem_214: "f32[8, 38, 1, 1]" = convolution_backward_42[0]
    getitem_215: "f32[152, 38, 1, 1]" = convolution_backward_42[1]
    getitem_216: "f32[152]" = convolution_backward_42[2];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_157: "f32[8, 38, 1, 1]" = torch.ops.aten.alias.default(relu_19);  relu_19 = None
    alias_158: "f32[8, 38, 1, 1]" = torch.ops.aten.alias.default(alias_157);  alias_157 = None
    le_33: "b8[8, 38, 1, 1]" = torch.ops.aten.le.Scalar(alias_158, 0);  alias_158 = None
    scalar_tensor_33: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_33: "f32[8, 38, 1, 1]" = torch.ops.aten.where.self(le_33, scalar_tensor_33, getitem_214);  le_33 = scalar_tensor_33 = getitem_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(where_33, mean_4, primals_123, [38], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_33 = mean_4 = primals_123 = None
    getitem_217: "f32[8, 152, 1, 1]" = convolution_backward_43[0]
    getitem_218: "f32[38, 152, 1, 1]" = convolution_backward_43[1]
    getitem_219: "f32[38]" = convolution_backward_43[2];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_9: "f32[8, 152, 14, 14]" = torch.ops.aten.expand.default(getitem_217, [8, 152, 14, 14]);  getitem_217 = None
    div_9: "f32[8, 152, 14, 14]" = torch.ops.aten.div.Scalar(expand_9, 196);  expand_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_249: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_588, div_9);  mul_588 = div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_160: "f32[8, 152, 14, 14]" = torch.ops.aten.alias.default(relu_18);  relu_18 = None
    alias_161: "f32[8, 152, 14, 14]" = torch.ops.aten.alias.default(alias_160);  alias_160 = None
    le_34: "b8[8, 152, 14, 14]" = torch.ops.aten.le.Scalar(alias_161, 0);  alias_161 = None
    scalar_tensor_34: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_34: "f32[8, 152, 14, 14]" = torch.ops.aten.where.self(le_34, scalar_tensor_34, add_249);  le_34 = scalar_tensor_34 = add_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_488: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_489: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 2);  unsqueeze_488 = None
    unsqueeze_490: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 3);  unsqueeze_489 = None
    sum_63: "f32[152]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
    sub_157: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_490)
    mul_591: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(where_34, sub_157);  sub_157 = None
    sum_64: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_591, [0, 2, 3]);  mul_591 = None
    mul_592: "f32[152]" = torch.ops.aten.mul.Tensor(sum_63, 0.0006377551020408163)
    unsqueeze_491: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_592, 0);  mul_592 = None
    unsqueeze_492: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 2);  unsqueeze_491 = None
    unsqueeze_493: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, 3);  unsqueeze_492 = None
    mul_593: "f32[152]" = torch.ops.aten.mul.Tensor(sum_64, 0.0006377551020408163)
    mul_594: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_595: "f32[152]" = torch.ops.aten.mul.Tensor(mul_593, mul_594);  mul_593 = mul_594 = None
    unsqueeze_494: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_595, 0);  mul_595 = None
    unsqueeze_495: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 2);  unsqueeze_494 = None
    unsqueeze_496: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 3);  unsqueeze_495 = None
    mul_596: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_35);  primals_35 = None
    unsqueeze_497: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_596, 0);  mul_596 = None
    unsqueeze_498: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
    unsqueeze_499: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 3);  unsqueeze_498 = None
    sub_158: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_490);  convolution_25 = unsqueeze_490 = None
    mul_597: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_496);  sub_158 = unsqueeze_496 = None
    sub_159: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(where_34, mul_597);  where_34 = mul_597 = None
    sub_160: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(sub_159, unsqueeze_493);  sub_159 = unsqueeze_493 = None
    mul_598: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_499);  sub_160 = unsqueeze_499 = None
    mul_599: "f32[152]" = torch.ops.aten.mul.Tensor(sum_64, squeeze_52);  sum_64 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_598, relu_17, primals_122, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 19, [True, True, False]);  mul_598 = primals_122 = None
    getitem_220: "f32[8, 152, 14, 14]" = convolution_backward_44[0]
    getitem_221: "f32[152, 8, 3, 3]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_163: "f32[8, 152, 14, 14]" = torch.ops.aten.alias.default(relu_17);  relu_17 = None
    alias_164: "f32[8, 152, 14, 14]" = torch.ops.aten.alias.default(alias_163);  alias_163 = None
    le_35: "b8[8, 152, 14, 14]" = torch.ops.aten.le.Scalar(alias_164, 0);  alias_164 = None
    scalar_tensor_35: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_35: "f32[8, 152, 14, 14]" = torch.ops.aten.where.self(le_35, scalar_tensor_35, getitem_220);  le_35 = scalar_tensor_35 = getitem_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_500: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_501: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 2);  unsqueeze_500 = None
    unsqueeze_502: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 3);  unsqueeze_501 = None
    sum_65: "f32[152]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    sub_161: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_502)
    mul_600: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(where_35, sub_161);  sub_161 = None
    sum_66: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_600, [0, 2, 3]);  mul_600 = None
    mul_601: "f32[152]" = torch.ops.aten.mul.Tensor(sum_65, 0.0006377551020408163)
    unsqueeze_503: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_601, 0);  mul_601 = None
    unsqueeze_504: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 2);  unsqueeze_503 = None
    unsqueeze_505: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, 3);  unsqueeze_504 = None
    mul_602: "f32[152]" = torch.ops.aten.mul.Tensor(sum_66, 0.0006377551020408163)
    mul_603: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_604: "f32[152]" = torch.ops.aten.mul.Tensor(mul_602, mul_603);  mul_602 = mul_603 = None
    unsqueeze_506: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_604, 0);  mul_604 = None
    unsqueeze_507: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 2);  unsqueeze_506 = None
    unsqueeze_508: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 3);  unsqueeze_507 = None
    mul_605: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_33);  primals_33 = None
    unsqueeze_509: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_605, 0);  mul_605 = None
    unsqueeze_510: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 2);  unsqueeze_509 = None
    unsqueeze_511: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 3);  unsqueeze_510 = None
    sub_162: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_502);  convolution_24 = unsqueeze_502 = None
    mul_606: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_508);  sub_162 = unsqueeze_508 = None
    sub_163: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(where_35, mul_606);  where_35 = mul_606 = None
    sub_164: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(sub_163, unsqueeze_505);  sub_163 = unsqueeze_505 = None
    mul_607: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_511);  sub_164 = unsqueeze_511 = None
    mul_608: "f32[152]" = torch.ops.aten.mul.Tensor(sum_66, squeeze_49);  sum_66 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_607, relu_16, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_607 = relu_16 = primals_121 = None
    getitem_223: "f32[8, 152, 14, 14]" = convolution_backward_45[0]
    getitem_224: "f32[152, 152, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_250: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(where_32, getitem_223);  where_32 = getitem_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    alias_165: "f32[8, 152, 14, 14]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
    le_36: "b8[8, 152, 14, 14]" = torch.ops.aten.le.Scalar(alias_165, 0);  alias_165 = None
    scalar_tensor_36: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_36: "f32[8, 152, 14, 14]" = torch.ops.aten.where.self(le_36, scalar_tensor_36, add_250);  le_36 = scalar_tensor_36 = add_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_512: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_513: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 2);  unsqueeze_512 = None
    unsqueeze_514: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 3);  unsqueeze_513 = None
    sum_67: "f32[152]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
    sub_165: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_514)
    mul_609: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(where_36, sub_165);  sub_165 = None
    sum_68: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_609, [0, 2, 3]);  mul_609 = None
    mul_610: "f32[152]" = torch.ops.aten.mul.Tensor(sum_67, 0.0006377551020408163)
    unsqueeze_515: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_610, 0);  mul_610 = None
    unsqueeze_516: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 2);  unsqueeze_515 = None
    unsqueeze_517: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, 3);  unsqueeze_516 = None
    mul_611: "f32[152]" = torch.ops.aten.mul.Tensor(sum_68, 0.0006377551020408163)
    mul_612: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_613: "f32[152]" = torch.ops.aten.mul.Tensor(mul_611, mul_612);  mul_611 = mul_612 = None
    unsqueeze_518: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_613, 0);  mul_613 = None
    unsqueeze_519: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 2);  unsqueeze_518 = None
    unsqueeze_520: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_519, 3);  unsqueeze_519 = None
    mul_614: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_31);  primals_31 = None
    unsqueeze_521: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_614, 0);  mul_614 = None
    unsqueeze_522: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 2);  unsqueeze_521 = None
    unsqueeze_523: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, 3);  unsqueeze_522 = None
    sub_166: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_514);  convolution_23 = unsqueeze_514 = None
    mul_615: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_520);  sub_166 = unsqueeze_520 = None
    sub_167: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(where_36, mul_615);  mul_615 = None
    sub_168: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(sub_167, unsqueeze_517);  sub_167 = unsqueeze_517 = None
    mul_616: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_523);  sub_168 = unsqueeze_523 = None
    mul_617: "f32[152]" = torch.ops.aten.mul.Tensor(sum_68, squeeze_46);  sum_68 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_616, mul_108, primals_120, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_616 = mul_108 = primals_120 = None
    getitem_226: "f32[8, 152, 14, 14]" = convolution_backward_46[0]
    getitem_227: "f32[152, 152, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_618: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_226, relu_14)
    mul_619: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_226, sigmoid_3);  getitem_226 = sigmoid_3 = None
    sum_69: "f32[8, 152, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_618, [2, 3], True);  mul_618 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_166: "f32[8, 152, 1, 1]" = torch.ops.aten.alias.default(alias_19);  alias_19 = None
    sub_169: "f32[8, 152, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_166)
    mul_620: "f32[8, 152, 1, 1]" = torch.ops.aten.mul.Tensor(alias_166, sub_169);  alias_166 = sub_169 = None
    mul_621: "f32[8, 152, 1, 1]" = torch.ops.aten.mul.Tensor(sum_69, mul_620);  sum_69 = mul_620 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_621, relu_15, primals_118, [152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_621 = primals_118 = None
    getitem_229: "f32[8, 38, 1, 1]" = convolution_backward_47[0]
    getitem_230: "f32[152, 38, 1, 1]" = convolution_backward_47[1]
    getitem_231: "f32[152]" = convolution_backward_47[2];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_168: "f32[8, 38, 1, 1]" = torch.ops.aten.alias.default(relu_15);  relu_15 = None
    alias_169: "f32[8, 38, 1, 1]" = torch.ops.aten.alias.default(alias_168);  alias_168 = None
    le_37: "b8[8, 38, 1, 1]" = torch.ops.aten.le.Scalar(alias_169, 0);  alias_169 = None
    scalar_tensor_37: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_37: "f32[8, 38, 1, 1]" = torch.ops.aten.where.self(le_37, scalar_tensor_37, getitem_229);  le_37 = scalar_tensor_37 = getitem_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(where_37, mean_3, primals_116, [38], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_37 = mean_3 = primals_116 = None
    getitem_232: "f32[8, 152, 1, 1]" = convolution_backward_48[0]
    getitem_233: "f32[38, 152, 1, 1]" = convolution_backward_48[1]
    getitem_234: "f32[38]" = convolution_backward_48[2];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_10: "f32[8, 152, 14, 14]" = torch.ops.aten.expand.default(getitem_232, [8, 152, 14, 14]);  getitem_232 = None
    div_10: "f32[8, 152, 14, 14]" = torch.ops.aten.div.Scalar(expand_10, 196);  expand_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_251: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_619, div_10);  mul_619 = div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_171: "f32[8, 152, 14, 14]" = torch.ops.aten.alias.default(relu_14);  relu_14 = None
    alias_172: "f32[8, 152, 14, 14]" = torch.ops.aten.alias.default(alias_171);  alias_171 = None
    le_38: "b8[8, 152, 14, 14]" = torch.ops.aten.le.Scalar(alias_172, 0);  alias_172 = None
    scalar_tensor_38: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_38: "f32[8, 152, 14, 14]" = torch.ops.aten.where.self(le_38, scalar_tensor_38, add_251);  le_38 = scalar_tensor_38 = add_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_524: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_525: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 2);  unsqueeze_524 = None
    unsqueeze_526: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_525, 3);  unsqueeze_525 = None
    sum_70: "f32[152]" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
    sub_170: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_526)
    mul_622: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(where_38, sub_170);  sub_170 = None
    sum_71: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_622, [0, 2, 3]);  mul_622 = None
    mul_623: "f32[152]" = torch.ops.aten.mul.Tensor(sum_70, 0.0006377551020408163)
    unsqueeze_527: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_623, 0);  mul_623 = None
    unsqueeze_528: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 2);  unsqueeze_527 = None
    unsqueeze_529: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, 3);  unsqueeze_528 = None
    mul_624: "f32[152]" = torch.ops.aten.mul.Tensor(sum_71, 0.0006377551020408163)
    mul_625: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_626: "f32[152]" = torch.ops.aten.mul.Tensor(mul_624, mul_625);  mul_624 = mul_625 = None
    unsqueeze_530: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_626, 0);  mul_626 = None
    unsqueeze_531: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 2);  unsqueeze_530 = None
    unsqueeze_532: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_531, 3);  unsqueeze_531 = None
    mul_627: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_29);  primals_29 = None
    unsqueeze_533: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_627, 0);  mul_627 = None
    unsqueeze_534: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 2);  unsqueeze_533 = None
    unsqueeze_535: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 3);  unsqueeze_534 = None
    sub_171: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_526);  convolution_20 = unsqueeze_526 = None
    mul_628: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_532);  sub_171 = unsqueeze_532 = None
    sub_172: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(where_38, mul_628);  where_38 = mul_628 = None
    sub_173: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(sub_172, unsqueeze_529);  sub_172 = unsqueeze_529 = None
    mul_629: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_535);  sub_173 = unsqueeze_535 = None
    mul_630: "f32[152]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_43);  sum_71 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_629, relu_13, primals_115, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 19, [True, True, False]);  mul_629 = primals_115 = None
    getitem_235: "f32[8, 152, 14, 14]" = convolution_backward_49[0]
    getitem_236: "f32[152, 8, 3, 3]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_174: "f32[8, 152, 14, 14]" = torch.ops.aten.alias.default(relu_13);  relu_13 = None
    alias_175: "f32[8, 152, 14, 14]" = torch.ops.aten.alias.default(alias_174);  alias_174 = None
    le_39: "b8[8, 152, 14, 14]" = torch.ops.aten.le.Scalar(alias_175, 0);  alias_175 = None
    scalar_tensor_39: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_39: "f32[8, 152, 14, 14]" = torch.ops.aten.where.self(le_39, scalar_tensor_39, getitem_235);  le_39 = scalar_tensor_39 = getitem_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_536: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_537: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 2);  unsqueeze_536 = None
    unsqueeze_538: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_537, 3);  unsqueeze_537 = None
    sum_72: "f32[152]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
    sub_174: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_538)
    mul_631: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(where_39, sub_174);  sub_174 = None
    sum_73: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_631, [0, 2, 3]);  mul_631 = None
    mul_632: "f32[152]" = torch.ops.aten.mul.Tensor(sum_72, 0.0006377551020408163)
    unsqueeze_539: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_632, 0);  mul_632 = None
    unsqueeze_540: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 2);  unsqueeze_539 = None
    unsqueeze_541: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, 3);  unsqueeze_540 = None
    mul_633: "f32[152]" = torch.ops.aten.mul.Tensor(sum_73, 0.0006377551020408163)
    mul_634: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_635: "f32[152]" = torch.ops.aten.mul.Tensor(mul_633, mul_634);  mul_633 = mul_634 = None
    unsqueeze_542: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_635, 0);  mul_635 = None
    unsqueeze_543: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 2);  unsqueeze_542 = None
    unsqueeze_544: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_543, 3);  unsqueeze_543 = None
    mul_636: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_27);  primals_27 = None
    unsqueeze_545: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_636, 0);  mul_636 = None
    unsqueeze_546: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 2);  unsqueeze_545 = None
    unsqueeze_547: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, 3);  unsqueeze_546 = None
    sub_175: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_538);  convolution_19 = unsqueeze_538 = None
    mul_637: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_544);  sub_175 = unsqueeze_544 = None
    sub_176: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(where_39, mul_637);  where_39 = mul_637 = None
    sub_177: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(sub_176, unsqueeze_541);  sub_176 = unsqueeze_541 = None
    mul_638: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_547);  sub_177 = unsqueeze_547 = None
    mul_639: "f32[152]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_40);  sum_73 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_638, relu_12, primals_114, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_638 = relu_12 = primals_114 = None
    getitem_238: "f32[8, 152, 14, 14]" = convolution_backward_50[0]
    getitem_239: "f32[152, 152, 1, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_252: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(where_36, getitem_238);  where_36 = getitem_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    alias_176: "f32[8, 152, 14, 14]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    le_40: "b8[8, 152, 14, 14]" = torch.ops.aten.le.Scalar(alias_176, 0);  alias_176 = None
    scalar_tensor_40: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_40: "f32[8, 152, 14, 14]" = torch.ops.aten.where.self(le_40, scalar_tensor_40, add_252);  le_40 = scalar_tensor_40 = add_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_548: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_549: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 2);  unsqueeze_548 = None
    unsqueeze_550: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_549, 3);  unsqueeze_549 = None
    sum_74: "f32[152]" = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
    sub_178: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_550)
    mul_640: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(where_40, sub_178);  sub_178 = None
    sum_75: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_640, [0, 2, 3]);  mul_640 = None
    mul_641: "f32[152]" = torch.ops.aten.mul.Tensor(sum_74, 0.0006377551020408163)
    unsqueeze_551: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_641, 0);  mul_641 = None
    unsqueeze_552: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 2);  unsqueeze_551 = None
    unsqueeze_553: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, 3);  unsqueeze_552 = None
    mul_642: "f32[152]" = torch.ops.aten.mul.Tensor(sum_75, 0.0006377551020408163)
    mul_643: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_644: "f32[152]" = torch.ops.aten.mul.Tensor(mul_642, mul_643);  mul_642 = mul_643 = None
    unsqueeze_554: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_644, 0);  mul_644 = None
    unsqueeze_555: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 2);  unsqueeze_554 = None
    unsqueeze_556: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_555, 3);  unsqueeze_555 = None
    mul_645: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_25);  primals_25 = None
    unsqueeze_557: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_645, 0);  mul_645 = None
    unsqueeze_558: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 2);  unsqueeze_557 = None
    unsqueeze_559: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, 3);  unsqueeze_558 = None
    sub_179: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_550);  convolution_18 = unsqueeze_550 = None
    mul_646: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_556);  sub_179 = unsqueeze_556 = None
    sub_180: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(where_40, mul_646);  mul_646 = None
    sub_181: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(sub_180, unsqueeze_553);  sub_180 = unsqueeze_553 = None
    mul_647: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_559);  sub_181 = unsqueeze_559 = None
    mul_648: "f32[152]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_37);  sum_75 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_647, relu_8, primals_113, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_647 = primals_113 = None
    getitem_241: "f32[8, 56, 28, 28]" = convolution_backward_51[0]
    getitem_242: "f32[152, 56, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_560: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_561: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 2);  unsqueeze_560 = None
    unsqueeze_562: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_561, 3);  unsqueeze_561 = None
    sum_76: "f32[152]" = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
    sub_182: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_562)
    mul_649: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(where_40, sub_182);  sub_182 = None
    sum_77: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_649, [0, 2, 3]);  mul_649 = None
    mul_650: "f32[152]" = torch.ops.aten.mul.Tensor(sum_76, 0.0006377551020408163)
    unsqueeze_563: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_650, 0);  mul_650 = None
    unsqueeze_564: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 2);  unsqueeze_563 = None
    unsqueeze_565: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, 3);  unsqueeze_564 = None
    mul_651: "f32[152]" = torch.ops.aten.mul.Tensor(sum_77, 0.0006377551020408163)
    mul_652: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_653: "f32[152]" = torch.ops.aten.mul.Tensor(mul_651, mul_652);  mul_651 = mul_652 = None
    unsqueeze_566: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_653, 0);  mul_653 = None
    unsqueeze_567: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 2);  unsqueeze_566 = None
    unsqueeze_568: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_567, 3);  unsqueeze_567 = None
    mul_654: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_23);  primals_23 = None
    unsqueeze_569: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_654, 0);  mul_654 = None
    unsqueeze_570: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 2);  unsqueeze_569 = None
    unsqueeze_571: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, 3);  unsqueeze_570 = None
    sub_183: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_562);  convolution_17 = unsqueeze_562 = None
    mul_655: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_568);  sub_183 = unsqueeze_568 = None
    sub_184: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(where_40, mul_655);  where_40 = mul_655 = None
    sub_185: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(sub_184, unsqueeze_565);  sub_184 = unsqueeze_565 = None
    mul_656: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_571);  sub_185 = unsqueeze_571 = None
    mul_657: "f32[152]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_34);  sum_77 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_656, mul_79, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_656 = mul_79 = primals_112 = None
    getitem_244: "f32[8, 152, 14, 14]" = convolution_backward_52[0]
    getitem_245: "f32[152, 152, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_658: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_244, relu_10)
    mul_659: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_244, sigmoid_2);  getitem_244 = sigmoid_2 = None
    sum_78: "f32[8, 152, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_658, [2, 3], True);  mul_658 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_177: "f32[8, 152, 1, 1]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    sub_186: "f32[8, 152, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_177)
    mul_660: "f32[8, 152, 1, 1]" = torch.ops.aten.mul.Tensor(alias_177, sub_186);  alias_177 = sub_186 = None
    mul_661: "f32[8, 152, 1, 1]" = torch.ops.aten.mul.Tensor(sum_78, mul_660);  sum_78 = mul_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_661, relu_11, primals_110, [152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_661 = primals_110 = None
    getitem_247: "f32[8, 14, 1, 1]" = convolution_backward_53[0]
    getitem_248: "f32[152, 14, 1, 1]" = convolution_backward_53[1]
    getitem_249: "f32[152]" = convolution_backward_53[2];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_179: "f32[8, 14, 1, 1]" = torch.ops.aten.alias.default(relu_11);  relu_11 = None
    alias_180: "f32[8, 14, 1, 1]" = torch.ops.aten.alias.default(alias_179);  alias_179 = None
    le_41: "b8[8, 14, 1, 1]" = torch.ops.aten.le.Scalar(alias_180, 0);  alias_180 = None
    scalar_tensor_41: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_41: "f32[8, 14, 1, 1]" = torch.ops.aten.where.self(le_41, scalar_tensor_41, getitem_247);  le_41 = scalar_tensor_41 = getitem_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(where_41, mean_2, primals_108, [14], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_41 = mean_2 = primals_108 = None
    getitem_250: "f32[8, 152, 1, 1]" = convolution_backward_54[0]
    getitem_251: "f32[14, 152, 1, 1]" = convolution_backward_54[1]
    getitem_252: "f32[14]" = convolution_backward_54[2];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_11: "f32[8, 152, 14, 14]" = torch.ops.aten.expand.default(getitem_250, [8, 152, 14, 14]);  getitem_250 = None
    div_11: "f32[8, 152, 14, 14]" = torch.ops.aten.div.Scalar(expand_11, 196);  expand_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_253: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_659, div_11);  mul_659 = div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_182: "f32[8, 152, 14, 14]" = torch.ops.aten.alias.default(relu_10);  relu_10 = None
    alias_183: "f32[8, 152, 14, 14]" = torch.ops.aten.alias.default(alias_182);  alias_182 = None
    le_42: "b8[8, 152, 14, 14]" = torch.ops.aten.le.Scalar(alias_183, 0);  alias_183 = None
    scalar_tensor_42: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_42: "f32[8, 152, 14, 14]" = torch.ops.aten.where.self(le_42, scalar_tensor_42, add_253);  le_42 = scalar_tensor_42 = add_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_572: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_573: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 2);  unsqueeze_572 = None
    unsqueeze_574: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_573, 3);  unsqueeze_573 = None
    sum_79: "f32[152]" = torch.ops.aten.sum.dim_IntList(where_42, [0, 2, 3])
    sub_187: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_574)
    mul_662: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(where_42, sub_187);  sub_187 = None
    sum_80: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_662, [0, 2, 3]);  mul_662 = None
    mul_663: "f32[152]" = torch.ops.aten.mul.Tensor(sum_79, 0.0006377551020408163)
    unsqueeze_575: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_663, 0);  mul_663 = None
    unsqueeze_576: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 2);  unsqueeze_575 = None
    unsqueeze_577: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, 3);  unsqueeze_576 = None
    mul_664: "f32[152]" = torch.ops.aten.mul.Tensor(sum_80, 0.0006377551020408163)
    mul_665: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_666: "f32[152]" = torch.ops.aten.mul.Tensor(mul_664, mul_665);  mul_664 = mul_665 = None
    unsqueeze_578: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_666, 0);  mul_666 = None
    unsqueeze_579: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 2);  unsqueeze_578 = None
    unsqueeze_580: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_579, 3);  unsqueeze_579 = None
    mul_667: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_21);  primals_21 = None
    unsqueeze_581: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_667, 0);  mul_667 = None
    unsqueeze_582: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 2);  unsqueeze_581 = None
    unsqueeze_583: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, 3);  unsqueeze_582 = None
    sub_188: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_574);  convolution_14 = unsqueeze_574 = None
    mul_668: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_580);  sub_188 = unsqueeze_580 = None
    sub_189: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(where_42, mul_668);  where_42 = mul_668 = None
    sub_190: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(sub_189, unsqueeze_577);  sub_189 = unsqueeze_577 = None
    mul_669: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_583);  sub_190 = unsqueeze_583 = None
    mul_670: "f32[152]" = torch.ops.aten.mul.Tensor(sum_80, squeeze_31);  sum_80 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_669, relu_9, primals_107, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 19, [True, True, False]);  mul_669 = primals_107 = None
    getitem_253: "f32[8, 152, 28, 28]" = convolution_backward_55[0]
    getitem_254: "f32[152, 8, 3, 3]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_185: "f32[8, 152, 28, 28]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
    alias_186: "f32[8, 152, 28, 28]" = torch.ops.aten.alias.default(alias_185);  alias_185 = None
    le_43: "b8[8, 152, 28, 28]" = torch.ops.aten.le.Scalar(alias_186, 0);  alias_186 = None
    scalar_tensor_43: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_43: "f32[8, 152, 28, 28]" = torch.ops.aten.where.self(le_43, scalar_tensor_43, getitem_253);  le_43 = scalar_tensor_43 = getitem_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_584: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_585: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 2);  unsqueeze_584 = None
    unsqueeze_586: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_585, 3);  unsqueeze_585 = None
    sum_81: "f32[152]" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
    sub_191: "f32[8, 152, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_586)
    mul_671: "f32[8, 152, 28, 28]" = torch.ops.aten.mul.Tensor(where_43, sub_191);  sub_191 = None
    sum_82: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_671, [0, 2, 3]);  mul_671 = None
    mul_672: "f32[152]" = torch.ops.aten.mul.Tensor(sum_81, 0.00015943877551020407)
    unsqueeze_587: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_672, 0);  mul_672 = None
    unsqueeze_588: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 2);  unsqueeze_587 = None
    unsqueeze_589: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, 3);  unsqueeze_588 = None
    mul_673: "f32[152]" = torch.ops.aten.mul.Tensor(sum_82, 0.00015943877551020407)
    mul_674: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_675: "f32[152]" = torch.ops.aten.mul.Tensor(mul_673, mul_674);  mul_673 = mul_674 = None
    unsqueeze_590: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_675, 0);  mul_675 = None
    unsqueeze_591: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 2);  unsqueeze_590 = None
    unsqueeze_592: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_591, 3);  unsqueeze_591 = None
    mul_676: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_19);  primals_19 = None
    unsqueeze_593: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_676, 0);  mul_676 = None
    unsqueeze_594: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 2);  unsqueeze_593 = None
    unsqueeze_595: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, 3);  unsqueeze_594 = None
    sub_192: "f32[8, 152, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_586);  convolution_13 = unsqueeze_586 = None
    mul_677: "f32[8, 152, 28, 28]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_592);  sub_192 = unsqueeze_592 = None
    sub_193: "f32[8, 152, 28, 28]" = torch.ops.aten.sub.Tensor(where_43, mul_677);  where_43 = mul_677 = None
    sub_194: "f32[8, 152, 28, 28]" = torch.ops.aten.sub.Tensor(sub_193, unsqueeze_589);  sub_193 = unsqueeze_589 = None
    mul_678: "f32[8, 152, 28, 28]" = torch.ops.aten.mul.Tensor(sub_194, unsqueeze_595);  sub_194 = unsqueeze_595 = None
    mul_679: "f32[152]" = torch.ops.aten.mul.Tensor(sum_82, squeeze_28);  sum_82 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_678, relu_8, primals_106, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_678 = relu_8 = primals_106 = None
    getitem_256: "f32[8, 56, 28, 28]" = convolution_backward_56[0]
    getitem_257: "f32[152, 56, 1, 1]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_254: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(getitem_241, getitem_256);  getitem_241 = getitem_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    alias_187: "f32[8, 56, 28, 28]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    le_44: "b8[8, 56, 28, 28]" = torch.ops.aten.le.Scalar(alias_187, 0);  alias_187 = None
    scalar_tensor_44: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_44: "f32[8, 56, 28, 28]" = torch.ops.aten.where.self(le_44, scalar_tensor_44, add_254);  le_44 = scalar_tensor_44 = add_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_596: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_597: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 2);  unsqueeze_596 = None
    unsqueeze_598: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_597, 3);  unsqueeze_597 = None
    sum_83: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_44, [0, 2, 3])
    sub_195: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_598)
    mul_680: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(where_44, sub_195);  sub_195 = None
    sum_84: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_680, [0, 2, 3]);  mul_680 = None
    mul_681: "f32[56]" = torch.ops.aten.mul.Tensor(sum_83, 0.00015943877551020407)
    unsqueeze_599: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_681, 0);  mul_681 = None
    unsqueeze_600: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 2);  unsqueeze_599 = None
    unsqueeze_601: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, 3);  unsqueeze_600 = None
    mul_682: "f32[56]" = torch.ops.aten.mul.Tensor(sum_84, 0.00015943877551020407)
    mul_683: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_684: "f32[56]" = torch.ops.aten.mul.Tensor(mul_682, mul_683);  mul_682 = mul_683 = None
    unsqueeze_602: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_684, 0);  mul_684 = None
    unsqueeze_603: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 2);  unsqueeze_602 = None
    unsqueeze_604: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_603, 3);  unsqueeze_603 = None
    mul_685: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_17);  primals_17 = None
    unsqueeze_605: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_685, 0);  mul_685 = None
    unsqueeze_606: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 2);  unsqueeze_605 = None
    unsqueeze_607: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, 3);  unsqueeze_606 = None
    sub_196: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_598);  convolution_12 = unsqueeze_598 = None
    mul_686: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_604);  sub_196 = unsqueeze_604 = None
    sub_197: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(where_44, mul_686);  mul_686 = None
    sub_198: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(sub_197, unsqueeze_601);  sub_197 = unsqueeze_601 = None
    mul_687: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_607);  sub_198 = unsqueeze_607 = None
    mul_688: "f32[56]" = torch.ops.aten.mul.Tensor(sum_84, squeeze_25);  sum_84 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_687, relu_4, primals_105, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_687 = primals_105 = None
    getitem_259: "f32[8, 24, 56, 56]" = convolution_backward_57[0]
    getitem_260: "f32[56, 24, 1, 1]" = convolution_backward_57[1];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_608: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_609: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 2);  unsqueeze_608 = None
    unsqueeze_610: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_609, 3);  unsqueeze_609 = None
    sum_85: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_44, [0, 2, 3])
    sub_199: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_610)
    mul_689: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(where_44, sub_199);  sub_199 = None
    sum_86: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_689, [0, 2, 3]);  mul_689 = None
    mul_690: "f32[56]" = torch.ops.aten.mul.Tensor(sum_85, 0.00015943877551020407)
    unsqueeze_611: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_690, 0);  mul_690 = None
    unsqueeze_612: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 2);  unsqueeze_611 = None
    unsqueeze_613: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, 3);  unsqueeze_612 = None
    mul_691: "f32[56]" = torch.ops.aten.mul.Tensor(sum_86, 0.00015943877551020407)
    mul_692: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_693: "f32[56]" = torch.ops.aten.mul.Tensor(mul_691, mul_692);  mul_691 = mul_692 = None
    unsqueeze_614: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_693, 0);  mul_693 = None
    unsqueeze_615: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 2);  unsqueeze_614 = None
    unsqueeze_616: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_615, 3);  unsqueeze_615 = None
    mul_694: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_15);  primals_15 = None
    unsqueeze_617: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_694, 0);  mul_694 = None
    unsqueeze_618: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 2);  unsqueeze_617 = None
    unsqueeze_619: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, 3);  unsqueeze_618 = None
    sub_200: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_610);  convolution_11 = unsqueeze_610 = None
    mul_695: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_616);  sub_200 = unsqueeze_616 = None
    sub_201: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(where_44, mul_695);  where_44 = mul_695 = None
    sub_202: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(sub_201, unsqueeze_613);  sub_201 = unsqueeze_613 = None
    mul_696: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_619);  sub_202 = unsqueeze_619 = None
    mul_697: "f32[56]" = torch.ops.aten.mul.Tensor(sum_86, squeeze_22);  sum_86 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_696, mul_50, primals_104, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_696 = mul_50 = primals_104 = None
    getitem_262: "f32[8, 56, 28, 28]" = convolution_backward_58[0]
    getitem_263: "f32[56, 56, 1, 1]" = convolution_backward_58[1];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_698: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_262, relu_6)
    mul_699: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_262, sigmoid_1);  getitem_262 = sigmoid_1 = None
    sum_87: "f32[8, 56, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_698, [2, 3], True);  mul_698 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_188: "f32[8, 56, 1, 1]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    sub_203: "f32[8, 56, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_188)
    mul_700: "f32[8, 56, 1, 1]" = torch.ops.aten.mul.Tensor(alias_188, sub_203);  alias_188 = sub_203 = None
    mul_701: "f32[8, 56, 1, 1]" = torch.ops.aten.mul.Tensor(sum_87, mul_700);  sum_87 = mul_700 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(mul_701, relu_7, primals_102, [56], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_701 = primals_102 = None
    getitem_265: "f32[8, 6, 1, 1]" = convolution_backward_59[0]
    getitem_266: "f32[56, 6, 1, 1]" = convolution_backward_59[1]
    getitem_267: "f32[56]" = convolution_backward_59[2];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_190: "f32[8, 6, 1, 1]" = torch.ops.aten.alias.default(relu_7);  relu_7 = None
    alias_191: "f32[8, 6, 1, 1]" = torch.ops.aten.alias.default(alias_190);  alias_190 = None
    le_45: "b8[8, 6, 1, 1]" = torch.ops.aten.le.Scalar(alias_191, 0);  alias_191 = None
    scalar_tensor_45: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_45: "f32[8, 6, 1, 1]" = torch.ops.aten.where.self(le_45, scalar_tensor_45, getitem_265);  le_45 = scalar_tensor_45 = getitem_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(where_45, mean_1, primals_100, [6], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_45 = mean_1 = primals_100 = None
    getitem_268: "f32[8, 56, 1, 1]" = convolution_backward_60[0]
    getitem_269: "f32[6, 56, 1, 1]" = convolution_backward_60[1]
    getitem_270: "f32[6]" = convolution_backward_60[2];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_12: "f32[8, 56, 28, 28]" = torch.ops.aten.expand.default(getitem_268, [8, 56, 28, 28]);  getitem_268 = None
    div_12: "f32[8, 56, 28, 28]" = torch.ops.aten.div.Scalar(expand_12, 784);  expand_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_255: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(mul_699, div_12);  mul_699 = div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_193: "f32[8, 56, 28, 28]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_194: "f32[8, 56, 28, 28]" = torch.ops.aten.alias.default(alias_193);  alias_193 = None
    le_46: "b8[8, 56, 28, 28]" = torch.ops.aten.le.Scalar(alias_194, 0);  alias_194 = None
    scalar_tensor_46: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_46: "f32[8, 56, 28, 28]" = torch.ops.aten.where.self(le_46, scalar_tensor_46, add_255);  le_46 = scalar_tensor_46 = add_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_620: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_621: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 2);  unsqueeze_620 = None
    unsqueeze_622: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_621, 3);  unsqueeze_621 = None
    sum_88: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_46, [0, 2, 3])
    sub_204: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_622)
    mul_702: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(where_46, sub_204);  sub_204 = None
    sum_89: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_702, [0, 2, 3]);  mul_702 = None
    mul_703: "f32[56]" = torch.ops.aten.mul.Tensor(sum_88, 0.00015943877551020407)
    unsqueeze_623: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_703, 0);  mul_703 = None
    unsqueeze_624: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 2);  unsqueeze_623 = None
    unsqueeze_625: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, 3);  unsqueeze_624 = None
    mul_704: "f32[56]" = torch.ops.aten.mul.Tensor(sum_89, 0.00015943877551020407)
    mul_705: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_706: "f32[56]" = torch.ops.aten.mul.Tensor(mul_704, mul_705);  mul_704 = mul_705 = None
    unsqueeze_626: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_706, 0);  mul_706 = None
    unsqueeze_627: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 2);  unsqueeze_626 = None
    unsqueeze_628: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_627, 3);  unsqueeze_627 = None
    mul_707: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_13);  primals_13 = None
    unsqueeze_629: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_707, 0);  mul_707 = None
    unsqueeze_630: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 2);  unsqueeze_629 = None
    unsqueeze_631: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, 3);  unsqueeze_630 = None
    sub_205: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_622);  convolution_8 = unsqueeze_622 = None
    mul_708: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_628);  sub_205 = unsqueeze_628 = None
    sub_206: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(where_46, mul_708);  where_46 = mul_708 = None
    sub_207: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(sub_206, unsqueeze_625);  sub_206 = unsqueeze_625 = None
    mul_709: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_631);  sub_207 = unsqueeze_631 = None
    mul_710: "f32[56]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_19);  sum_89 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_709, relu_5, primals_99, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 7, [True, True, False]);  mul_709 = primals_99 = None
    getitem_271: "f32[8, 56, 56, 56]" = convolution_backward_61[0]
    getitem_272: "f32[56, 8, 3, 3]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_196: "f32[8, 56, 56, 56]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_197: "f32[8, 56, 56, 56]" = torch.ops.aten.alias.default(alias_196);  alias_196 = None
    le_47: "b8[8, 56, 56, 56]" = torch.ops.aten.le.Scalar(alias_197, 0);  alias_197 = None
    scalar_tensor_47: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_47: "f32[8, 56, 56, 56]" = torch.ops.aten.where.self(le_47, scalar_tensor_47, getitem_271);  le_47 = scalar_tensor_47 = getitem_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_632: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_633: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 2);  unsqueeze_632 = None
    unsqueeze_634: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_633, 3);  unsqueeze_633 = None
    sum_90: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_47, [0, 2, 3])
    sub_208: "f32[8, 56, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_634)
    mul_711: "f32[8, 56, 56, 56]" = torch.ops.aten.mul.Tensor(where_47, sub_208);  sub_208 = None
    sum_91: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_711, [0, 2, 3]);  mul_711 = None
    mul_712: "f32[56]" = torch.ops.aten.mul.Tensor(sum_90, 3.985969387755102e-05)
    unsqueeze_635: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_712, 0);  mul_712 = None
    unsqueeze_636: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 2);  unsqueeze_635 = None
    unsqueeze_637: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, 3);  unsqueeze_636 = None
    mul_713: "f32[56]" = torch.ops.aten.mul.Tensor(sum_91, 3.985969387755102e-05)
    mul_714: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_715: "f32[56]" = torch.ops.aten.mul.Tensor(mul_713, mul_714);  mul_713 = mul_714 = None
    unsqueeze_638: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_715, 0);  mul_715 = None
    unsqueeze_639: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 2);  unsqueeze_638 = None
    unsqueeze_640: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_639, 3);  unsqueeze_639 = None
    mul_716: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_11);  primals_11 = None
    unsqueeze_641: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_716, 0);  mul_716 = None
    unsqueeze_642: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 2);  unsqueeze_641 = None
    unsqueeze_643: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, 3);  unsqueeze_642 = None
    sub_209: "f32[8, 56, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_634);  convolution_7 = unsqueeze_634 = None
    mul_717: "f32[8, 56, 56, 56]" = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_640);  sub_209 = unsqueeze_640 = None
    sub_210: "f32[8, 56, 56, 56]" = torch.ops.aten.sub.Tensor(where_47, mul_717);  where_47 = mul_717 = None
    sub_211: "f32[8, 56, 56, 56]" = torch.ops.aten.sub.Tensor(sub_210, unsqueeze_637);  sub_210 = unsqueeze_637 = None
    mul_718: "f32[8, 56, 56, 56]" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_643);  sub_211 = unsqueeze_643 = None
    mul_719: "f32[56]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_16);  sum_91 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_718, relu_4, primals_98, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_718 = relu_4 = primals_98 = None
    getitem_274: "f32[8, 24, 56, 56]" = convolution_backward_62[0]
    getitem_275: "f32[56, 24, 1, 1]" = convolution_backward_62[1];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_256: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(getitem_259, getitem_274);  getitem_259 = getitem_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    alias_198: "f32[8, 24, 56, 56]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    le_48: "b8[8, 24, 56, 56]" = torch.ops.aten.le.Scalar(alias_198, 0);  alias_198 = None
    scalar_tensor_48: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_48: "f32[8, 24, 56, 56]" = torch.ops.aten.where.self(le_48, scalar_tensor_48, add_256);  le_48 = scalar_tensor_48 = add_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_644: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_645: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 2);  unsqueeze_644 = None
    unsqueeze_646: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_645, 3);  unsqueeze_645 = None
    sum_92: "f32[24]" = torch.ops.aten.sum.dim_IntList(where_48, [0, 2, 3])
    sub_212: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_646)
    mul_720: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(where_48, sub_212);  sub_212 = None
    sum_93: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_720, [0, 2, 3]);  mul_720 = None
    mul_721: "f32[24]" = torch.ops.aten.mul.Tensor(sum_92, 3.985969387755102e-05)
    unsqueeze_647: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_721, 0);  mul_721 = None
    unsqueeze_648: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 2);  unsqueeze_647 = None
    unsqueeze_649: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, 3);  unsqueeze_648 = None
    mul_722: "f32[24]" = torch.ops.aten.mul.Tensor(sum_93, 3.985969387755102e-05)
    mul_723: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_724: "f32[24]" = torch.ops.aten.mul.Tensor(mul_722, mul_723);  mul_722 = mul_723 = None
    unsqueeze_650: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_724, 0);  mul_724 = None
    unsqueeze_651: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 2);  unsqueeze_650 = None
    unsqueeze_652: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_651, 3);  unsqueeze_651 = None
    mul_725: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_9);  primals_9 = None
    unsqueeze_653: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_725, 0);  mul_725 = None
    unsqueeze_654: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 2);  unsqueeze_653 = None
    unsqueeze_655: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, 3);  unsqueeze_654 = None
    sub_213: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_646);  convolution_6 = unsqueeze_646 = None
    mul_726: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_652);  sub_213 = unsqueeze_652 = None
    sub_214: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(where_48, mul_726);  mul_726 = None
    sub_215: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_214, unsqueeze_649);  sub_214 = unsqueeze_649 = None
    mul_727: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_215, unsqueeze_655);  sub_215 = unsqueeze_655 = None
    mul_728: "f32[24]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_13);  sum_93 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_727, relu, primals_97, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_727 = primals_97 = None
    getitem_277: "f32[8, 32, 112, 112]" = convolution_backward_63[0]
    getitem_278: "f32[24, 32, 1, 1]" = convolution_backward_63[1];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_656: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_657: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 2);  unsqueeze_656 = None
    unsqueeze_658: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_657, 3);  unsqueeze_657 = None
    sum_94: "f32[24]" = torch.ops.aten.sum.dim_IntList(where_48, [0, 2, 3])
    sub_216: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_658)
    mul_729: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(where_48, sub_216);  sub_216 = None
    sum_95: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_729, [0, 2, 3]);  mul_729 = None
    mul_730: "f32[24]" = torch.ops.aten.mul.Tensor(sum_94, 3.985969387755102e-05)
    unsqueeze_659: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_730, 0);  mul_730 = None
    unsqueeze_660: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 2);  unsqueeze_659 = None
    unsqueeze_661: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, 3);  unsqueeze_660 = None
    mul_731: "f32[24]" = torch.ops.aten.mul.Tensor(sum_95, 3.985969387755102e-05)
    mul_732: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_733: "f32[24]" = torch.ops.aten.mul.Tensor(mul_731, mul_732);  mul_731 = mul_732 = None
    unsqueeze_662: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_733, 0);  mul_733 = None
    unsqueeze_663: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 2);  unsqueeze_662 = None
    unsqueeze_664: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_663, 3);  unsqueeze_663 = None
    mul_734: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_7);  primals_7 = None
    unsqueeze_665: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_734, 0);  mul_734 = None
    unsqueeze_666: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 2);  unsqueeze_665 = None
    unsqueeze_667: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, 3);  unsqueeze_666 = None
    sub_217: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_658);  convolution_5 = unsqueeze_658 = None
    mul_735: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_664);  sub_217 = unsqueeze_664 = None
    sub_218: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(where_48, mul_735);  where_48 = mul_735 = None
    sub_219: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_218, unsqueeze_661);  sub_218 = unsqueeze_661 = None
    mul_736: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_667);  sub_219 = unsqueeze_667 = None
    mul_737: "f32[24]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_10);  sum_95 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(mul_736, mul_21, primals_96, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_736 = mul_21 = primals_96 = None
    getitem_280: "f32[8, 24, 56, 56]" = convolution_backward_64[0]
    getitem_281: "f32[24, 24, 1, 1]" = convolution_backward_64[1];  convolution_backward_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_738: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_280, relu_2)
    mul_739: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_280, sigmoid);  getitem_280 = sigmoid = None
    sum_96: "f32[8, 24, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_738, [2, 3], True);  mul_738 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    alias_199: "f32[8, 24, 1, 1]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    sub_220: "f32[8, 24, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_199)
    mul_740: "f32[8, 24, 1, 1]" = torch.ops.aten.mul.Tensor(alias_199, sub_220);  alias_199 = sub_220 = None
    mul_741: "f32[8, 24, 1, 1]" = torch.ops.aten.mul.Tensor(sum_96, mul_740);  sum_96 = mul_740 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(mul_741, relu_3, primals_94, [24], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_741 = primals_94 = None
    getitem_283: "f32[8, 8, 1, 1]" = convolution_backward_65[0]
    getitem_284: "f32[24, 8, 1, 1]" = convolution_backward_65[1]
    getitem_285: "f32[24]" = convolution_backward_65[2];  convolution_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    alias_201: "f32[8, 8, 1, 1]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_202: "f32[8, 8, 1, 1]" = torch.ops.aten.alias.default(alias_201);  alias_201 = None
    le_49: "b8[8, 8, 1, 1]" = torch.ops.aten.le.Scalar(alias_202, 0);  alias_202 = None
    scalar_tensor_49: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_49: "f32[8, 8, 1, 1]" = torch.ops.aten.where.self(le_49, scalar_tensor_49, getitem_283);  le_49 = scalar_tensor_49 = getitem_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_66 = torch.ops.aten.convolution_backward.default(where_49, mean, primals_92, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_49 = mean = primals_92 = None
    getitem_286: "f32[8, 24, 1, 1]" = convolution_backward_66[0]
    getitem_287: "f32[8, 24, 1, 1]" = convolution_backward_66[1]
    getitem_288: "f32[8]" = convolution_backward_66[2];  convolution_backward_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_13: "f32[8, 24, 56, 56]" = torch.ops.aten.expand.default(getitem_286, [8, 24, 56, 56]);  getitem_286 = None
    div_13: "f32[8, 24, 56, 56]" = torch.ops.aten.div.Scalar(expand_13, 3136);  expand_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_257: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_739, div_13);  mul_739 = div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_204: "f32[8, 24, 56, 56]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_205: "f32[8, 24, 56, 56]" = torch.ops.aten.alias.default(alias_204);  alias_204 = None
    le_50: "b8[8, 24, 56, 56]" = torch.ops.aten.le.Scalar(alias_205, 0);  alias_205 = None
    scalar_tensor_50: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_50: "f32[8, 24, 56, 56]" = torch.ops.aten.where.self(le_50, scalar_tensor_50, add_257);  le_50 = scalar_tensor_50 = add_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_668: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_669: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 2);  unsqueeze_668 = None
    unsqueeze_670: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_669, 3);  unsqueeze_669 = None
    sum_97: "f32[24]" = torch.ops.aten.sum.dim_IntList(where_50, [0, 2, 3])
    sub_221: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_670)
    mul_742: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(where_50, sub_221);  sub_221 = None
    sum_98: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_742, [0, 2, 3]);  mul_742 = None
    mul_743: "f32[24]" = torch.ops.aten.mul.Tensor(sum_97, 3.985969387755102e-05)
    unsqueeze_671: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_743, 0);  mul_743 = None
    unsqueeze_672: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 2);  unsqueeze_671 = None
    unsqueeze_673: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, 3);  unsqueeze_672 = None
    mul_744: "f32[24]" = torch.ops.aten.mul.Tensor(sum_98, 3.985969387755102e-05)
    mul_745: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_746: "f32[24]" = torch.ops.aten.mul.Tensor(mul_744, mul_745);  mul_744 = mul_745 = None
    unsqueeze_674: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_746, 0);  mul_746 = None
    unsqueeze_675: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 2);  unsqueeze_674 = None
    unsqueeze_676: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_675, 3);  unsqueeze_675 = None
    mul_747: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_5);  primals_5 = None
    unsqueeze_677: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_747, 0);  mul_747 = None
    unsqueeze_678: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 2);  unsqueeze_677 = None
    unsqueeze_679: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, 3);  unsqueeze_678 = None
    sub_222: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_670);  convolution_2 = unsqueeze_670 = None
    mul_748: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_676);  sub_222 = unsqueeze_676 = None
    sub_223: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(where_50, mul_748);  where_50 = mul_748 = None
    sub_224: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_223, unsqueeze_673);  sub_223 = unsqueeze_673 = None
    mul_749: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_679);  sub_224 = unsqueeze_679 = None
    mul_750: "f32[24]" = torch.ops.aten.mul.Tensor(sum_98, squeeze_7);  sum_98 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_67 = torch.ops.aten.convolution_backward.default(mul_749, relu_1, primals_91, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 3, [True, True, False]);  mul_749 = primals_91 = None
    getitem_289: "f32[8, 24, 112, 112]" = convolution_backward_67[0]
    getitem_290: "f32[24, 8, 3, 3]" = convolution_backward_67[1];  convolution_backward_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_207: "f32[8, 24, 112, 112]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_208: "f32[8, 24, 112, 112]" = torch.ops.aten.alias.default(alias_207);  alias_207 = None
    le_51: "b8[8, 24, 112, 112]" = torch.ops.aten.le.Scalar(alias_208, 0);  alias_208 = None
    scalar_tensor_51: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_51: "f32[8, 24, 112, 112]" = torch.ops.aten.where.self(le_51, scalar_tensor_51, getitem_289);  le_51 = scalar_tensor_51 = getitem_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_680: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_681: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 2);  unsqueeze_680 = None
    unsqueeze_682: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_681, 3);  unsqueeze_681 = None
    sum_99: "f32[24]" = torch.ops.aten.sum.dim_IntList(where_51, [0, 2, 3])
    sub_225: "f32[8, 24, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_682)
    mul_751: "f32[8, 24, 112, 112]" = torch.ops.aten.mul.Tensor(where_51, sub_225);  sub_225 = None
    sum_100: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_751, [0, 2, 3]);  mul_751 = None
    mul_752: "f32[24]" = torch.ops.aten.mul.Tensor(sum_99, 9.964923469387754e-06)
    unsqueeze_683: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_752, 0);  mul_752 = None
    unsqueeze_684: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 2);  unsqueeze_683 = None
    unsqueeze_685: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, 3);  unsqueeze_684 = None
    mul_753: "f32[24]" = torch.ops.aten.mul.Tensor(sum_100, 9.964923469387754e-06)
    mul_754: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_755: "f32[24]" = torch.ops.aten.mul.Tensor(mul_753, mul_754);  mul_753 = mul_754 = None
    unsqueeze_686: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_755, 0);  mul_755 = None
    unsqueeze_687: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 2);  unsqueeze_686 = None
    unsqueeze_688: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_687, 3);  unsqueeze_687 = None
    mul_756: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_3);  primals_3 = None
    unsqueeze_689: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_756, 0);  mul_756 = None
    unsqueeze_690: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 2);  unsqueeze_689 = None
    unsqueeze_691: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, 3);  unsqueeze_690 = None
    sub_226: "f32[8, 24, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_682);  convolution_1 = unsqueeze_682 = None
    mul_757: "f32[8, 24, 112, 112]" = torch.ops.aten.mul.Tensor(sub_226, unsqueeze_688);  sub_226 = unsqueeze_688 = None
    sub_227: "f32[8, 24, 112, 112]" = torch.ops.aten.sub.Tensor(where_51, mul_757);  where_51 = mul_757 = None
    sub_228: "f32[8, 24, 112, 112]" = torch.ops.aten.sub.Tensor(sub_227, unsqueeze_685);  sub_227 = unsqueeze_685 = None
    mul_758: "f32[8, 24, 112, 112]" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_691);  sub_228 = unsqueeze_691 = None
    mul_759: "f32[24]" = torch.ops.aten.mul.Tensor(sum_100, squeeze_4);  sum_100 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_758, relu, primals_90, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_758 = primals_90 = None
    getitem_292: "f32[8, 32, 112, 112]" = convolution_backward_68[0]
    getitem_293: "f32[24, 32, 1, 1]" = convolution_backward_68[1];  convolution_backward_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_258: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(getitem_277, getitem_292);  getitem_277 = getitem_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_210: "f32[8, 32, 112, 112]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_211: "f32[8, 32, 112, 112]" = torch.ops.aten.alias.default(alias_210);  alias_210 = None
    le_52: "b8[8, 32, 112, 112]" = torch.ops.aten.le.Scalar(alias_211, 0);  alias_211 = None
    scalar_tensor_52: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_52: "f32[8, 32, 112, 112]" = torch.ops.aten.where.self(le_52, scalar_tensor_52, add_258);  le_52 = scalar_tensor_52 = add_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_692: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_693: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 2);  unsqueeze_692 = None
    unsqueeze_694: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_693, 3);  unsqueeze_693 = None
    sum_101: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_52, [0, 2, 3])
    sub_229: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_694)
    mul_760: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_52, sub_229);  sub_229 = None
    sum_102: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_760, [0, 2, 3]);  mul_760 = None
    mul_761: "f32[32]" = torch.ops.aten.mul.Tensor(sum_101, 9.964923469387754e-06)
    unsqueeze_695: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_761, 0);  mul_761 = None
    unsqueeze_696: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 2);  unsqueeze_695 = None
    unsqueeze_697: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, 3);  unsqueeze_696 = None
    mul_762: "f32[32]" = torch.ops.aten.mul.Tensor(sum_102, 9.964923469387754e-06)
    mul_763: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_764: "f32[32]" = torch.ops.aten.mul.Tensor(mul_762, mul_763);  mul_762 = mul_763 = None
    unsqueeze_698: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_764, 0);  mul_764 = None
    unsqueeze_699: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 2);  unsqueeze_698 = None
    unsqueeze_700: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_699, 3);  unsqueeze_699 = None
    mul_765: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_1);  primals_1 = None
    unsqueeze_701: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_765, 0);  mul_765 = None
    unsqueeze_702: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 2);  unsqueeze_701 = None
    unsqueeze_703: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, 3);  unsqueeze_702 = None
    sub_230: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_694);  convolution = unsqueeze_694 = None
    mul_766: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_700);  sub_230 = unsqueeze_700 = None
    sub_231: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(where_52, mul_766);  where_52 = mul_766 = None
    sub_232: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_231, unsqueeze_697);  sub_231 = unsqueeze_697 = None
    mul_767: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_703);  sub_232 = unsqueeze_703 = None
    mul_768: "f32[32]" = torch.ops.aten.mul.Tensor(sum_102, squeeze_1);  sum_102 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_69 = torch.ops.aten.convolution_backward.default(mul_767, primals_319, primals_89, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_767 = primals_319 = primals_89 = None
    getitem_296: "f32[32, 3, 3, 3]" = convolution_backward_69[1];  convolution_backward_69 = None
    
    # No stacktrace found for following nodes
    copy_: "i64[]" = torch.ops.aten.copy_.default(primals_187, add);  primals_187 = add = None
    copy__1: "f32[32]" = torch.ops.aten.copy_.default(primals_188, add_2);  primals_188 = add_2 = None
    copy__2: "f32[32]" = torch.ops.aten.copy_.default(primals_189, add_3);  primals_189 = add_3 = None
    copy__3: "i64[]" = torch.ops.aten.copy_.default(primals_190, add_5);  primals_190 = add_5 = None
    copy__4: "f32[24]" = torch.ops.aten.copy_.default(primals_191, add_7);  primals_191 = add_7 = None
    copy__5: "f32[24]" = torch.ops.aten.copy_.default(primals_192, add_8);  primals_192 = add_8 = None
    copy__6: "i64[]" = torch.ops.aten.copy_.default(primals_193, add_10);  primals_193 = add_10 = None
    copy__7: "f32[24]" = torch.ops.aten.copy_.default(primals_194, add_12);  primals_194 = add_12 = None
    copy__8: "f32[24]" = torch.ops.aten.copy_.default(primals_195, add_13);  primals_195 = add_13 = None
    copy__9: "i64[]" = torch.ops.aten.copy_.default(primals_196, add_15);  primals_196 = add_15 = None
    copy__10: "f32[24]" = torch.ops.aten.copy_.default(primals_197, add_17);  primals_197 = add_17 = None
    copy__11: "f32[24]" = torch.ops.aten.copy_.default(primals_198, add_18);  primals_198 = add_18 = None
    copy__12: "i64[]" = torch.ops.aten.copy_.default(primals_199, add_20);  primals_199 = add_20 = None
    copy__13: "f32[24]" = torch.ops.aten.copy_.default(primals_200, add_22);  primals_200 = add_22 = None
    copy__14: "f32[24]" = torch.ops.aten.copy_.default(primals_201, add_23);  primals_201 = add_23 = None
    copy__15: "i64[]" = torch.ops.aten.copy_.default(primals_202, add_26);  primals_202 = add_26 = None
    copy__16: "f32[56]" = torch.ops.aten.copy_.default(primals_203, add_28);  primals_203 = add_28 = None
    copy__17: "f32[56]" = torch.ops.aten.copy_.default(primals_204, add_29);  primals_204 = add_29 = None
    copy__18: "i64[]" = torch.ops.aten.copy_.default(primals_205, add_31);  primals_205 = add_31 = None
    copy__19: "f32[56]" = torch.ops.aten.copy_.default(primals_206, add_33);  primals_206 = add_33 = None
    copy__20: "f32[56]" = torch.ops.aten.copy_.default(primals_207, add_34);  primals_207 = add_34 = None
    copy__21: "i64[]" = torch.ops.aten.copy_.default(primals_208, add_36);  primals_208 = add_36 = None
    copy__22: "f32[56]" = torch.ops.aten.copy_.default(primals_209, add_38);  primals_209 = add_38 = None
    copy__23: "f32[56]" = torch.ops.aten.copy_.default(primals_210, add_39);  primals_210 = add_39 = None
    copy__24: "i64[]" = torch.ops.aten.copy_.default(primals_211, add_41);  primals_211 = add_41 = None
    copy__25: "f32[56]" = torch.ops.aten.copy_.default(primals_212, add_43);  primals_212 = add_43 = None
    copy__26: "f32[56]" = torch.ops.aten.copy_.default(primals_213, add_44);  primals_213 = add_44 = None
    copy__27: "i64[]" = torch.ops.aten.copy_.default(primals_214, add_47);  primals_214 = add_47 = None
    copy__28: "f32[152]" = torch.ops.aten.copy_.default(primals_215, add_49);  primals_215 = add_49 = None
    copy__29: "f32[152]" = torch.ops.aten.copy_.default(primals_216, add_50);  primals_216 = add_50 = None
    copy__30: "i64[]" = torch.ops.aten.copy_.default(primals_217, add_52);  primals_217 = add_52 = None
    copy__31: "f32[152]" = torch.ops.aten.copy_.default(primals_218, add_54);  primals_218 = add_54 = None
    copy__32: "f32[152]" = torch.ops.aten.copy_.default(primals_219, add_55);  primals_219 = add_55 = None
    copy__33: "i64[]" = torch.ops.aten.copy_.default(primals_220, add_57);  primals_220 = add_57 = None
    copy__34: "f32[152]" = torch.ops.aten.copy_.default(primals_221, add_59);  primals_221 = add_59 = None
    copy__35: "f32[152]" = torch.ops.aten.copy_.default(primals_222, add_60);  primals_222 = add_60 = None
    copy__36: "i64[]" = torch.ops.aten.copy_.default(primals_223, add_62);  primals_223 = add_62 = None
    copy__37: "f32[152]" = torch.ops.aten.copy_.default(primals_224, add_64);  primals_224 = add_64 = None
    copy__38: "f32[152]" = torch.ops.aten.copy_.default(primals_225, add_65);  primals_225 = add_65 = None
    copy__39: "i64[]" = torch.ops.aten.copy_.default(primals_226, add_68);  primals_226 = add_68 = None
    copy__40: "f32[152]" = torch.ops.aten.copy_.default(primals_227, add_70);  primals_227 = add_70 = None
    copy__41: "f32[152]" = torch.ops.aten.copy_.default(primals_228, add_71);  primals_228 = add_71 = None
    copy__42: "i64[]" = torch.ops.aten.copy_.default(primals_229, add_73);  primals_229 = add_73 = None
    copy__43: "f32[152]" = torch.ops.aten.copy_.default(primals_230, add_75);  primals_230 = add_75 = None
    copy__44: "f32[152]" = torch.ops.aten.copy_.default(primals_231, add_76);  primals_231 = add_76 = None
    copy__45: "i64[]" = torch.ops.aten.copy_.default(primals_232, add_78);  primals_232 = add_78 = None
    copy__46: "f32[152]" = torch.ops.aten.copy_.default(primals_233, add_80);  primals_233 = add_80 = None
    copy__47: "f32[152]" = torch.ops.aten.copy_.default(primals_234, add_81);  primals_234 = add_81 = None
    copy__48: "i64[]" = torch.ops.aten.copy_.default(primals_235, add_84);  primals_235 = add_84 = None
    copy__49: "f32[152]" = torch.ops.aten.copy_.default(primals_236, add_86);  primals_236 = add_86 = None
    copy__50: "f32[152]" = torch.ops.aten.copy_.default(primals_237, add_87);  primals_237 = add_87 = None
    copy__51: "i64[]" = torch.ops.aten.copy_.default(primals_238, add_89);  primals_238 = add_89 = None
    copy__52: "f32[152]" = torch.ops.aten.copy_.default(primals_239, add_91);  primals_239 = add_91 = None
    copy__53: "f32[152]" = torch.ops.aten.copy_.default(primals_240, add_92);  primals_240 = add_92 = None
    copy__54: "i64[]" = torch.ops.aten.copy_.default(primals_241, add_94);  primals_241 = add_94 = None
    copy__55: "f32[152]" = torch.ops.aten.copy_.default(primals_242, add_96);  primals_242 = add_96 = None
    copy__56: "f32[152]" = torch.ops.aten.copy_.default(primals_243, add_97);  primals_243 = add_97 = None
    copy__57: "i64[]" = torch.ops.aten.copy_.default(primals_244, add_100);  primals_244 = add_100 = None
    copy__58: "f32[152]" = torch.ops.aten.copy_.default(primals_245, add_102);  primals_245 = add_102 = None
    copy__59: "f32[152]" = torch.ops.aten.copy_.default(primals_246, add_103);  primals_246 = add_103 = None
    copy__60: "i64[]" = torch.ops.aten.copy_.default(primals_247, add_105);  primals_247 = add_105 = None
    copy__61: "f32[152]" = torch.ops.aten.copy_.default(primals_248, add_107);  primals_248 = add_107 = None
    copy__62: "f32[152]" = torch.ops.aten.copy_.default(primals_249, add_108);  primals_249 = add_108 = None
    copy__63: "i64[]" = torch.ops.aten.copy_.default(primals_250, add_110);  primals_250 = add_110 = None
    copy__64: "f32[152]" = torch.ops.aten.copy_.default(primals_251, add_112);  primals_251 = add_112 = None
    copy__65: "f32[152]" = torch.ops.aten.copy_.default(primals_252, add_113);  primals_252 = add_113 = None
    copy__66: "i64[]" = torch.ops.aten.copy_.default(primals_253, add_116);  primals_253 = add_116 = None
    copy__67: "f32[368]" = torch.ops.aten.copy_.default(primals_254, add_118);  primals_254 = add_118 = None
    copy__68: "f32[368]" = torch.ops.aten.copy_.default(primals_255, add_119);  primals_255 = add_119 = None
    copy__69: "i64[]" = torch.ops.aten.copy_.default(primals_256, add_121);  primals_256 = add_121 = None
    copy__70: "f32[368]" = torch.ops.aten.copy_.default(primals_257, add_123);  primals_257 = add_123 = None
    copy__71: "f32[368]" = torch.ops.aten.copy_.default(primals_258, add_124);  primals_258 = add_124 = None
    copy__72: "i64[]" = torch.ops.aten.copy_.default(primals_259, add_126);  primals_259 = add_126 = None
    copy__73: "f32[368]" = torch.ops.aten.copy_.default(primals_260, add_128);  primals_260 = add_128 = None
    copy__74: "f32[368]" = torch.ops.aten.copy_.default(primals_261, add_129);  primals_261 = add_129 = None
    copy__75: "i64[]" = torch.ops.aten.copy_.default(primals_262, add_131);  primals_262 = add_131 = None
    copy__76: "f32[368]" = torch.ops.aten.copy_.default(primals_263, add_133);  primals_263 = add_133 = None
    copy__77: "f32[368]" = torch.ops.aten.copy_.default(primals_264, add_134);  primals_264 = add_134 = None
    copy__78: "i64[]" = torch.ops.aten.copy_.default(primals_265, add_137);  primals_265 = add_137 = None
    copy__79: "f32[368]" = torch.ops.aten.copy_.default(primals_266, add_139);  primals_266 = add_139 = None
    copy__80: "f32[368]" = torch.ops.aten.copy_.default(primals_267, add_140);  primals_267 = add_140 = None
    copy__81: "i64[]" = torch.ops.aten.copy_.default(primals_268, add_142);  primals_268 = add_142 = None
    copy__82: "f32[368]" = torch.ops.aten.copy_.default(primals_269, add_144);  primals_269 = add_144 = None
    copy__83: "f32[368]" = torch.ops.aten.copy_.default(primals_270, add_145);  primals_270 = add_145 = None
    copy__84: "i64[]" = torch.ops.aten.copy_.default(primals_271, add_147);  primals_271 = add_147 = None
    copy__85: "f32[368]" = torch.ops.aten.copy_.default(primals_272, add_149);  primals_272 = add_149 = None
    copy__86: "f32[368]" = torch.ops.aten.copy_.default(primals_273, add_150);  primals_273 = add_150 = None
    copy__87: "i64[]" = torch.ops.aten.copy_.default(primals_274, add_153);  primals_274 = add_153 = None
    copy__88: "f32[368]" = torch.ops.aten.copy_.default(primals_275, add_155);  primals_275 = add_155 = None
    copy__89: "f32[368]" = torch.ops.aten.copy_.default(primals_276, add_156);  primals_276 = add_156 = None
    copy__90: "i64[]" = torch.ops.aten.copy_.default(primals_277, add_158);  primals_277 = add_158 = None
    copy__91: "f32[368]" = torch.ops.aten.copy_.default(primals_278, add_160);  primals_278 = add_160 = None
    copy__92: "f32[368]" = torch.ops.aten.copy_.default(primals_279, add_161);  primals_279 = add_161 = None
    copy__93: "i64[]" = torch.ops.aten.copy_.default(primals_280, add_163);  primals_280 = add_163 = None
    copy__94: "f32[368]" = torch.ops.aten.copy_.default(primals_281, add_165);  primals_281 = add_165 = None
    copy__95: "f32[368]" = torch.ops.aten.copy_.default(primals_282, add_166);  primals_282 = add_166 = None
    copy__96: "i64[]" = torch.ops.aten.copy_.default(primals_283, add_169);  primals_283 = add_169 = None
    copy__97: "f32[368]" = torch.ops.aten.copy_.default(primals_284, add_171);  primals_284 = add_171 = None
    copy__98: "f32[368]" = torch.ops.aten.copy_.default(primals_285, add_172);  primals_285 = add_172 = None
    copy__99: "i64[]" = torch.ops.aten.copy_.default(primals_286, add_174);  primals_286 = add_174 = None
    copy__100: "f32[368]" = torch.ops.aten.copy_.default(primals_287, add_176);  primals_287 = add_176 = None
    copy__101: "f32[368]" = torch.ops.aten.copy_.default(primals_288, add_177);  primals_288 = add_177 = None
    copy__102: "i64[]" = torch.ops.aten.copy_.default(primals_289, add_179);  primals_289 = add_179 = None
    copy__103: "f32[368]" = torch.ops.aten.copy_.default(primals_290, add_181);  primals_290 = add_181 = None
    copy__104: "f32[368]" = torch.ops.aten.copy_.default(primals_291, add_182);  primals_291 = add_182 = None
    copy__105: "i64[]" = torch.ops.aten.copy_.default(primals_292, add_185);  primals_292 = add_185 = None
    copy__106: "f32[368]" = torch.ops.aten.copy_.default(primals_293, add_187);  primals_293 = add_187 = None
    copy__107: "f32[368]" = torch.ops.aten.copy_.default(primals_294, add_188);  primals_294 = add_188 = None
    copy__108: "i64[]" = torch.ops.aten.copy_.default(primals_295, add_190);  primals_295 = add_190 = None
    copy__109: "f32[368]" = torch.ops.aten.copy_.default(primals_296, add_192);  primals_296 = add_192 = None
    copy__110: "f32[368]" = torch.ops.aten.copy_.default(primals_297, add_193);  primals_297 = add_193 = None
    copy__111: "i64[]" = torch.ops.aten.copy_.default(primals_298, add_195);  primals_298 = add_195 = None
    copy__112: "f32[368]" = torch.ops.aten.copy_.default(primals_299, add_197);  primals_299 = add_197 = None
    copy__113: "f32[368]" = torch.ops.aten.copy_.default(primals_300, add_198);  primals_300 = add_198 = None
    copy__114: "i64[]" = torch.ops.aten.copy_.default(primals_301, add_201);  primals_301 = add_201 = None
    copy__115: "f32[368]" = torch.ops.aten.copy_.default(primals_302, add_203);  primals_302 = add_203 = None
    copy__116: "f32[368]" = torch.ops.aten.copy_.default(primals_303, add_204);  primals_303 = add_204 = None
    copy__117: "i64[]" = torch.ops.aten.copy_.default(primals_304, add_206);  primals_304 = add_206 = None
    copy__118: "f32[368]" = torch.ops.aten.copy_.default(primals_305, add_208);  primals_305 = add_208 = None
    copy__119: "f32[368]" = torch.ops.aten.copy_.default(primals_306, add_209);  primals_306 = add_209 = None
    copy__120: "i64[]" = torch.ops.aten.copy_.default(primals_307, add_211);  primals_307 = add_211 = None
    copy__121: "f32[368]" = torch.ops.aten.copy_.default(primals_308, add_213);  primals_308 = add_213 = None
    copy__122: "f32[368]" = torch.ops.aten.copy_.default(primals_309, add_214);  primals_309 = add_214 = None
    copy__123: "i64[]" = torch.ops.aten.copy_.default(primals_310, add_217);  primals_310 = add_217 = None
    copy__124: "f32[368]" = torch.ops.aten.copy_.default(primals_311, add_219);  primals_311 = add_219 = None
    copy__125: "f32[368]" = torch.ops.aten.copy_.default(primals_312, add_220);  primals_312 = add_220 = None
    copy__126: "i64[]" = torch.ops.aten.copy_.default(primals_313, add_222);  primals_313 = add_222 = None
    copy__127: "f32[368]" = torch.ops.aten.copy_.default(primals_314, add_224);  primals_314 = add_224 = None
    copy__128: "f32[368]" = torch.ops.aten.copy_.default(primals_315, add_225);  primals_315 = add_225 = None
    copy__129: "i64[]" = torch.ops.aten.copy_.default(primals_316, add_227);  primals_316 = add_227 = None
    copy__130: "f32[368]" = torch.ops.aten.copy_.default(primals_317, add_229);  primals_317 = add_229 = None
    copy__131: "f32[368]" = torch.ops.aten.copy_.default(primals_318, add_230);  primals_318 = add_230 = None
    return pytree.tree_unflatten([addmm, mul_768, sum_101, mul_759, sum_99, mul_750, sum_97, mul_737, sum_94, mul_728, sum_92, mul_719, sum_90, mul_710, sum_88, mul_697, sum_85, mul_688, sum_83, mul_679, sum_81, mul_670, sum_79, mul_657, sum_76, mul_648, sum_74, mul_639, sum_72, mul_630, sum_70, mul_617, sum_67, mul_608, sum_65, mul_599, sum_63, mul_586, sum_60, mul_577, sum_58, mul_568, sum_56, mul_555, sum_53, mul_546, sum_51, mul_537, sum_49, mul_524, sum_46, mul_515, sum_44, mul_506, sum_42, mul_497, sum_40, mul_484, sum_37, mul_475, sum_35, mul_466, sum_33, mul_453, sum_30, mul_444, sum_28, mul_435, sum_26, mul_422, sum_23, mul_413, sum_21, mul_404, sum_19, mul_391, sum_16, mul_382, sum_14, mul_373, sum_12, mul_360, sum_9, mul_351, sum_7, mul_342, sum_5, mul_329, sum_2, getitem_296, getitem_293, getitem_290, getitem_287, getitem_288, getitem_284, getitem_285, getitem_281, getitem_278, getitem_275, getitem_272, getitem_269, getitem_270, getitem_266, getitem_267, getitem_263, getitem_260, getitem_257, getitem_254, getitem_251, getitem_252, getitem_248, getitem_249, getitem_245, getitem_242, getitem_239, getitem_236, getitem_233, getitem_234, getitem_230, getitem_231, getitem_227, getitem_224, getitem_221, getitem_218, getitem_219, getitem_215, getitem_216, getitem_212, getitem_209, getitem_206, getitem_203, getitem_204, getitem_200, getitem_201, getitem_197, getitem_194, getitem_191, getitem_188, getitem_189, getitem_185, getitem_186, getitem_182, getitem_179, getitem_176, getitem_173, getitem_170, getitem_171, getitem_167, getitem_168, getitem_164, getitem_161, getitem_158, getitem_155, getitem_156, getitem_152, getitem_153, getitem_149, getitem_146, getitem_143, getitem_140, getitem_141, getitem_137, getitem_138, getitem_134, getitem_131, getitem_128, getitem_125, getitem_126, getitem_122, getitem_123, getitem_119, getitem_116, getitem_113, getitem_110, getitem_111, getitem_107, getitem_108, getitem_104, getitem_101, getitem_98, getitem_95, getitem_96, getitem_92, getitem_93, getitem_89, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    