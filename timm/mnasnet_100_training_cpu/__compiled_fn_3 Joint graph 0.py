from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[32]"; primals_2: "f32[32]"; primals_3: "f32[32]"; primals_4: "f32[32]"; primals_5: "f32[16]"; primals_6: "f32[16]"; primals_7: "f32[48]"; primals_8: "f32[48]"; primals_9: "f32[48]"; primals_10: "f32[48]"; primals_11: "f32[24]"; primals_12: "f32[24]"; primals_13: "f32[72]"; primals_14: "f32[72]"; primals_15: "f32[72]"; primals_16: "f32[72]"; primals_17: "f32[24]"; primals_18: "f32[24]"; primals_19: "f32[72]"; primals_20: "f32[72]"; primals_21: "f32[72]"; primals_22: "f32[72]"; primals_23: "f32[24]"; primals_24: "f32[24]"; primals_25: "f32[72]"; primals_26: "f32[72]"; primals_27: "f32[72]"; primals_28: "f32[72]"; primals_29: "f32[40]"; primals_30: "f32[40]"; primals_31: "f32[120]"; primals_32: "f32[120]"; primals_33: "f32[120]"; primals_34: "f32[120]"; primals_35: "f32[40]"; primals_36: "f32[40]"; primals_37: "f32[120]"; primals_38: "f32[120]"; primals_39: "f32[120]"; primals_40: "f32[120]"; primals_41: "f32[40]"; primals_42: "f32[40]"; primals_43: "f32[240]"; primals_44: "f32[240]"; primals_45: "f32[240]"; primals_46: "f32[240]"; primals_47: "f32[80]"; primals_48: "f32[80]"; primals_49: "f32[480]"; primals_50: "f32[480]"; primals_51: "f32[480]"; primals_52: "f32[480]"; primals_53: "f32[80]"; primals_54: "f32[80]"; primals_55: "f32[480]"; primals_56: "f32[480]"; primals_57: "f32[480]"; primals_58: "f32[480]"; primals_59: "f32[80]"; primals_60: "f32[80]"; primals_61: "f32[480]"; primals_62: "f32[480]"; primals_63: "f32[480]"; primals_64: "f32[480]"; primals_65: "f32[96]"; primals_66: "f32[96]"; primals_67: "f32[576]"; primals_68: "f32[576]"; primals_69: "f32[576]"; primals_70: "f32[576]"; primals_71: "f32[96]"; primals_72: "f32[96]"; primals_73: "f32[576]"; primals_74: "f32[576]"; primals_75: "f32[576]"; primals_76: "f32[576]"; primals_77: "f32[192]"; primals_78: "f32[192]"; primals_79: "f32[1152]"; primals_80: "f32[1152]"; primals_81: "f32[1152]"; primals_82: "f32[1152]"; primals_83: "f32[192]"; primals_84: "f32[192]"; primals_85: "f32[1152]"; primals_86: "f32[1152]"; primals_87: "f32[1152]"; primals_88: "f32[1152]"; primals_89: "f32[192]"; primals_90: "f32[192]"; primals_91: "f32[1152]"; primals_92: "f32[1152]"; primals_93: "f32[1152]"; primals_94: "f32[1152]"; primals_95: "f32[192]"; primals_96: "f32[192]"; primals_97: "f32[1152]"; primals_98: "f32[1152]"; primals_99: "f32[1152]"; primals_100: "f32[1152]"; primals_101: "f32[320]"; primals_102: "f32[320]"; primals_103: "f32[1280]"; primals_104: "f32[1280]"; primals_105: "f32[32, 3, 3, 3]"; primals_106: "f32[32, 1, 3, 3]"; primals_107: "f32[16, 32, 1, 1]"; primals_108: "f32[48, 16, 1, 1]"; primals_109: "f32[48, 1, 3, 3]"; primals_110: "f32[24, 48, 1, 1]"; primals_111: "f32[72, 24, 1, 1]"; primals_112: "f32[72, 1, 3, 3]"; primals_113: "f32[24, 72, 1, 1]"; primals_114: "f32[72, 24, 1, 1]"; primals_115: "f32[72, 1, 3, 3]"; primals_116: "f32[24, 72, 1, 1]"; primals_117: "f32[72, 24, 1, 1]"; primals_118: "f32[72, 1, 5, 5]"; primals_119: "f32[40, 72, 1, 1]"; primals_120: "f32[120, 40, 1, 1]"; primals_121: "f32[120, 1, 5, 5]"; primals_122: "f32[40, 120, 1, 1]"; primals_123: "f32[120, 40, 1, 1]"; primals_124: "f32[120, 1, 5, 5]"; primals_125: "f32[40, 120, 1, 1]"; primals_126: "f32[240, 40, 1, 1]"; primals_127: "f32[240, 1, 5, 5]"; primals_128: "f32[80, 240, 1, 1]"; primals_129: "f32[480, 80, 1, 1]"; primals_130: "f32[480, 1, 5, 5]"; primals_131: "f32[80, 480, 1, 1]"; primals_132: "f32[480, 80, 1, 1]"; primals_133: "f32[480, 1, 5, 5]"; primals_134: "f32[80, 480, 1, 1]"; primals_135: "f32[480, 80, 1, 1]"; primals_136: "f32[480, 1, 3, 3]"; primals_137: "f32[96, 480, 1, 1]"; primals_138: "f32[576, 96, 1, 1]"; primals_139: "f32[576, 1, 3, 3]"; primals_140: "f32[96, 576, 1, 1]"; primals_141: "f32[576, 96, 1, 1]"; primals_142: "f32[576, 1, 5, 5]"; primals_143: "f32[192, 576, 1, 1]"; primals_144: "f32[1152, 192, 1, 1]"; primals_145: "f32[1152, 1, 5, 5]"; primals_146: "f32[192, 1152, 1, 1]"; primals_147: "f32[1152, 192, 1, 1]"; primals_148: "f32[1152, 1, 5, 5]"; primals_149: "f32[192, 1152, 1, 1]"; primals_150: "f32[1152, 192, 1, 1]"; primals_151: "f32[1152, 1, 5, 5]"; primals_152: "f32[192, 1152, 1, 1]"; primals_153: "f32[1152, 192, 1, 1]"; primals_154: "f32[1152, 1, 3, 3]"; primals_155: "f32[320, 1152, 1, 1]"; primals_156: "f32[1280, 320, 1, 1]"; primals_157: "f32[1000, 1280]"; primals_158: "f32[1000]"; primals_159: "i64[]"; primals_160: "f32[32]"; primals_161: "f32[32]"; primals_162: "i64[]"; primals_163: "f32[32]"; primals_164: "f32[32]"; primals_165: "i64[]"; primals_166: "f32[16]"; primals_167: "f32[16]"; primals_168: "i64[]"; primals_169: "f32[48]"; primals_170: "f32[48]"; primals_171: "i64[]"; primals_172: "f32[48]"; primals_173: "f32[48]"; primals_174: "i64[]"; primals_175: "f32[24]"; primals_176: "f32[24]"; primals_177: "i64[]"; primals_178: "f32[72]"; primals_179: "f32[72]"; primals_180: "i64[]"; primals_181: "f32[72]"; primals_182: "f32[72]"; primals_183: "i64[]"; primals_184: "f32[24]"; primals_185: "f32[24]"; primals_186: "i64[]"; primals_187: "f32[72]"; primals_188: "f32[72]"; primals_189: "i64[]"; primals_190: "f32[72]"; primals_191: "f32[72]"; primals_192: "i64[]"; primals_193: "f32[24]"; primals_194: "f32[24]"; primals_195: "i64[]"; primals_196: "f32[72]"; primals_197: "f32[72]"; primals_198: "i64[]"; primals_199: "f32[72]"; primals_200: "f32[72]"; primals_201: "i64[]"; primals_202: "f32[40]"; primals_203: "f32[40]"; primals_204: "i64[]"; primals_205: "f32[120]"; primals_206: "f32[120]"; primals_207: "i64[]"; primals_208: "f32[120]"; primals_209: "f32[120]"; primals_210: "i64[]"; primals_211: "f32[40]"; primals_212: "f32[40]"; primals_213: "i64[]"; primals_214: "f32[120]"; primals_215: "f32[120]"; primals_216: "i64[]"; primals_217: "f32[120]"; primals_218: "f32[120]"; primals_219: "i64[]"; primals_220: "f32[40]"; primals_221: "f32[40]"; primals_222: "i64[]"; primals_223: "f32[240]"; primals_224: "f32[240]"; primals_225: "i64[]"; primals_226: "f32[240]"; primals_227: "f32[240]"; primals_228: "i64[]"; primals_229: "f32[80]"; primals_230: "f32[80]"; primals_231: "i64[]"; primals_232: "f32[480]"; primals_233: "f32[480]"; primals_234: "i64[]"; primals_235: "f32[480]"; primals_236: "f32[480]"; primals_237: "i64[]"; primals_238: "f32[80]"; primals_239: "f32[80]"; primals_240: "i64[]"; primals_241: "f32[480]"; primals_242: "f32[480]"; primals_243: "i64[]"; primals_244: "f32[480]"; primals_245: "f32[480]"; primals_246: "i64[]"; primals_247: "f32[80]"; primals_248: "f32[80]"; primals_249: "i64[]"; primals_250: "f32[480]"; primals_251: "f32[480]"; primals_252: "i64[]"; primals_253: "f32[480]"; primals_254: "f32[480]"; primals_255: "i64[]"; primals_256: "f32[96]"; primals_257: "f32[96]"; primals_258: "i64[]"; primals_259: "f32[576]"; primals_260: "f32[576]"; primals_261: "i64[]"; primals_262: "f32[576]"; primals_263: "f32[576]"; primals_264: "i64[]"; primals_265: "f32[96]"; primals_266: "f32[96]"; primals_267: "i64[]"; primals_268: "f32[576]"; primals_269: "f32[576]"; primals_270: "i64[]"; primals_271: "f32[576]"; primals_272: "f32[576]"; primals_273: "i64[]"; primals_274: "f32[192]"; primals_275: "f32[192]"; primals_276: "i64[]"; primals_277: "f32[1152]"; primals_278: "f32[1152]"; primals_279: "i64[]"; primals_280: "f32[1152]"; primals_281: "f32[1152]"; primals_282: "i64[]"; primals_283: "f32[192]"; primals_284: "f32[192]"; primals_285: "i64[]"; primals_286: "f32[1152]"; primals_287: "f32[1152]"; primals_288: "i64[]"; primals_289: "f32[1152]"; primals_290: "f32[1152]"; primals_291: "i64[]"; primals_292: "f32[192]"; primals_293: "f32[192]"; primals_294: "i64[]"; primals_295: "f32[1152]"; primals_296: "f32[1152]"; primals_297: "i64[]"; primals_298: "f32[1152]"; primals_299: "f32[1152]"; primals_300: "i64[]"; primals_301: "f32[192]"; primals_302: "f32[192]"; primals_303: "i64[]"; primals_304: "f32[1152]"; primals_305: "f32[1152]"; primals_306: "i64[]"; primals_307: "f32[1152]"; primals_308: "f32[1152]"; primals_309: "i64[]"; primals_310: "f32[320]"; primals_311: "f32[320]"; primals_312: "i64[]"; primals_313: "f32[1280]"; primals_314: "f32[1280]"; primals_315: "f32[8, 3, 224, 224]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:162, code: x = self.conv_stem(x)
    convolution: "f32[8, 32, 112, 112]" = torch.ops.aten.convolution.default(primals_315, primals_105, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_159, 1)
    
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
    mul_2: "f32[32]" = torch.ops.aten.mul.Tensor(primals_160, 0.9)
    add_2: "f32[32]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[32]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.00000996502277);  squeeze_2 = None
    mul_4: "f32[32]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[32]" = torch.ops.aten.mul.Tensor(primals_161, 0.9)
    add_3: "f32[32]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_1, -1)
    unsqueeze_1: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1);  primals_2 = None
    unsqueeze_3: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu: "f32[8, 32, 112, 112]" = torch.ops.aten.relu.default(add_4);  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_1: "f32[8, 32, 112, 112]" = torch.ops.aten.convolution.default(relu, primals_106, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_162, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[1, 32, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 32, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
    rsqrt_1: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_1: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
    mul_7: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
    squeeze_4: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_8: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_9: "f32[32]" = torch.ops.aten.mul.Tensor(primals_163, 0.9)
    add_7: "f32[32]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
    squeeze_5: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_10: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.00000996502277);  squeeze_5 = None
    mul_11: "f32[32]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
    mul_12: "f32[32]" = torch.ops.aten.mul.Tensor(primals_164, 0.9)
    add_8: "f32[32]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
    unsqueeze_4: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1)
    unsqueeze_5: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_13: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
    unsqueeze_6: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_4, -1);  primals_4 = None
    unsqueeze_7: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_9: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_1: "f32[8, 32, 112, 112]" = torch.ops.aten.relu.default(add_9);  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_2: "f32[8, 16, 112, 112]" = torch.ops.aten.convolution.default(relu_1, primals_107, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_165, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 16, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 16, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
    rsqrt_2: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_2: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
    mul_14: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_15: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_16: "f32[16]" = torch.ops.aten.mul.Tensor(primals_166, 0.9)
    add_12: "f32[16]" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
    squeeze_8: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_17: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.00000996502277);  squeeze_8 = None
    mul_18: "f32[16]" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
    mul_19: "f32[16]" = torch.ops.aten.mul.Tensor(primals_167, 0.9)
    add_13: "f32[16]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
    unsqueeze_8: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_9: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_20: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
    unsqueeze_10: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_11: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_14: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_3: "f32[8, 48, 112, 112]" = torch.ops.aten.convolution.default(add_14, primals_108, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_15: "i64[]" = torch.ops.aten.add.Tensor(primals_168, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
    getitem_6: "f32[1, 48, 1, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 48, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_16: "f32[1, 48, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
    rsqrt_3: "f32[1, 48, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_3: "f32[8, 48, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_7)
    mul_21: "f32[8, 48, 112, 112]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
    squeeze_10: "f32[48]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_22: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_23: "f32[48]" = torch.ops.aten.mul.Tensor(primals_169, 0.9)
    add_17: "f32[48]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
    squeeze_11: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
    mul_24: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.00000996502277);  squeeze_11 = None
    mul_25: "f32[48]" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
    mul_26: "f32[48]" = torch.ops.aten.mul.Tensor(primals_170, 0.9)
    add_18: "f32[48]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
    unsqueeze_12: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1)
    unsqueeze_13: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_27: "f32[8, 48, 112, 112]" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
    unsqueeze_14: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1);  primals_8 = None
    unsqueeze_15: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_19: "f32[8, 48, 112, 112]" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_2: "f32[8, 48, 112, 112]" = torch.ops.aten.relu.default(add_19);  add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_4: "f32[8, 48, 56, 56]" = torch.ops.aten.convolution.default(relu_2, primals_109, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 48)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_20: "i64[]" = torch.ops.aten.add.Tensor(primals_171, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
    getitem_8: "f32[1, 48, 1, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 48, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_21: "f32[1, 48, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
    rsqrt_4: "f32[1, 48, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_4: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_9)
    mul_28: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
    squeeze_13: "f32[48]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
    mul_29: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_30: "f32[48]" = torch.ops.aten.mul.Tensor(primals_172, 0.9)
    add_22: "f32[48]" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
    squeeze_14: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
    mul_31: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000398612827361);  squeeze_14 = None
    mul_32: "f32[48]" = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
    mul_33: "f32[48]" = torch.ops.aten.mul.Tensor(primals_173, 0.9)
    add_23: "f32[48]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
    unsqueeze_16: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1)
    unsqueeze_17: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    mul_34: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
    unsqueeze_18: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_10, -1);  primals_10 = None
    unsqueeze_19: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    add_24: "f32[8, 48, 56, 56]" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_3: "f32[8, 48, 56, 56]" = torch.ops.aten.relu.default(add_24);  add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_5: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(relu_3, primals_110, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_25: "i64[]" = torch.ops.aten.add.Tensor(primals_174, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
    getitem_10: "f32[1, 24, 1, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 24, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_26: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
    rsqrt_5: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_5: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_11)
    mul_35: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_15: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
    squeeze_16: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    mul_36: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_37: "f32[24]" = torch.ops.aten.mul.Tensor(primals_175, 0.9)
    add_27: "f32[24]" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
    squeeze_17: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
    mul_38: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000398612827361);  squeeze_17 = None
    mul_39: "f32[24]" = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
    mul_40: "f32[24]" = torch.ops.aten.mul.Tensor(primals_176, 0.9)
    add_28: "f32[24]" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
    unsqueeze_20: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_21: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_41: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
    unsqueeze_22: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_23: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_29: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_6: "f32[8, 72, 56, 56]" = torch.ops.aten.convolution.default(add_29, primals_111, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_30: "i64[]" = torch.ops.aten.add.Tensor(primals_177, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_6 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
    getitem_12: "f32[1, 72, 1, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 72, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_31: "f32[1, 72, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
    rsqrt_6: "f32[1, 72, 1, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_6: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_13)
    mul_42: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_18: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
    squeeze_19: "f32[72]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
    mul_43: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_44: "f32[72]" = torch.ops.aten.mul.Tensor(primals_178, 0.9)
    add_32: "f32[72]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
    squeeze_20: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
    mul_45: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0000398612827361);  squeeze_20 = None
    mul_46: "f32[72]" = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
    mul_47: "f32[72]" = torch.ops.aten.mul.Tensor(primals_179, 0.9)
    add_33: "f32[72]" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
    unsqueeze_24: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1)
    unsqueeze_25: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    mul_48: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
    unsqueeze_26: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1);  primals_14 = None
    unsqueeze_27: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    add_34: "f32[8, 72, 56, 56]" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_4: "f32[8, 72, 56, 56]" = torch.ops.aten.relu.default(add_34);  add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_7: "f32[8, 72, 56, 56]" = torch.ops.aten.convolution.default(relu_4, primals_112, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_35: "i64[]" = torch.ops.aten.add.Tensor(primals_180, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_7 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
    getitem_14: "f32[1, 72, 1, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 72, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_36: "f32[1, 72, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
    rsqrt_7: "f32[1, 72, 1, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_7: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_15)
    mul_49: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_21: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
    squeeze_22: "f32[72]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
    mul_50: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_51: "f32[72]" = torch.ops.aten.mul.Tensor(primals_181, 0.9)
    add_37: "f32[72]" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
    squeeze_23: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
    mul_52: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0000398612827361);  squeeze_23 = None
    mul_53: "f32[72]" = torch.ops.aten.mul.Tensor(mul_52, 0.1);  mul_52 = None
    mul_54: "f32[72]" = torch.ops.aten.mul.Tensor(primals_182, 0.9)
    add_38: "f32[72]" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
    unsqueeze_28: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1)
    unsqueeze_29: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_55: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
    unsqueeze_30: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1);  primals_16 = None
    unsqueeze_31: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_39: "f32[8, 72, 56, 56]" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_5: "f32[8, 72, 56, 56]" = torch.ops.aten.relu.default(add_39);  add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_8: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(relu_5, primals_113, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_40: "i64[]" = torch.ops.aten.add.Tensor(primals_183, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_8 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
    getitem_16: "f32[1, 24, 1, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 24, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_41: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
    rsqrt_8: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
    sub_8: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_17)
    mul_56: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_24: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
    squeeze_25: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
    mul_57: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_58: "f32[24]" = torch.ops.aten.mul.Tensor(primals_184, 0.9)
    add_42: "f32[24]" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
    squeeze_26: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
    mul_59: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0000398612827361);  squeeze_26 = None
    mul_60: "f32[24]" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
    mul_61: "f32[24]" = torch.ops.aten.mul.Tensor(primals_185, 0.9)
    add_43: "f32[24]" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
    unsqueeze_32: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_33: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    mul_62: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
    unsqueeze_34: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_35: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    add_44: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_45: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(add_44, add_29);  add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_9: "f32[8, 72, 56, 56]" = torch.ops.aten.convolution.default(add_45, primals_114, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_46: "i64[]" = torch.ops.aten.add.Tensor(primals_186, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_9 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
    getitem_18: "f32[1, 72, 1, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 72, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_47: "f32[1, 72, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
    rsqrt_9: "f32[1, 72, 1, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_9: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_19)
    mul_63: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_27: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
    squeeze_28: "f32[72]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_64: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_65: "f32[72]" = torch.ops.aten.mul.Tensor(primals_187, 0.9)
    add_48: "f32[72]" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
    squeeze_29: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
    mul_66: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0000398612827361);  squeeze_29 = None
    mul_67: "f32[72]" = torch.ops.aten.mul.Tensor(mul_66, 0.1);  mul_66 = None
    mul_68: "f32[72]" = torch.ops.aten.mul.Tensor(primals_188, 0.9)
    add_49: "f32[72]" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
    unsqueeze_36: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
    unsqueeze_37: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_69: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
    unsqueeze_38: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
    unsqueeze_39: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_50: "f32[8, 72, 56, 56]" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_6: "f32[8, 72, 56, 56]" = torch.ops.aten.relu.default(add_50);  add_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_10: "f32[8, 72, 56, 56]" = torch.ops.aten.convolution.default(relu_6, primals_115, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_51: "i64[]" = torch.ops.aten.add.Tensor(primals_189, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_10 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[1, 72, 1, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 72, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_52: "f32[1, 72, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
    rsqrt_10: "f32[1, 72, 1, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_10: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_21)
    mul_70: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_30: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
    squeeze_31: "f32[72]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
    mul_71: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_72: "f32[72]" = torch.ops.aten.mul.Tensor(primals_190, 0.9)
    add_53: "f32[72]" = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
    squeeze_32: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
    mul_73: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0000398612827361);  squeeze_32 = None
    mul_74: "f32[72]" = torch.ops.aten.mul.Tensor(mul_73, 0.1);  mul_73 = None
    mul_75: "f32[72]" = torch.ops.aten.mul.Tensor(primals_191, 0.9)
    add_54: "f32[72]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
    unsqueeze_40: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1)
    unsqueeze_41: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    mul_76: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_41);  mul_70 = unsqueeze_41 = None
    unsqueeze_42: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1);  primals_22 = None
    unsqueeze_43: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    add_55: "f32[8, 72, 56, 56]" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_43);  mul_76 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_7: "f32[8, 72, 56, 56]" = torch.ops.aten.relu.default(add_55);  add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_11: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(relu_7, primals_116, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_56: "i64[]" = torch.ops.aten.add.Tensor(primals_192, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_11 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[1, 24, 1, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 24, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_57: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
    rsqrt_11: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    sub_11: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_23)
    mul_77: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    squeeze_33: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
    squeeze_34: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
    mul_78: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_79: "f32[24]" = torch.ops.aten.mul.Tensor(primals_193, 0.9)
    add_58: "f32[24]" = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
    squeeze_35: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
    mul_80: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0000398612827361);  squeeze_35 = None
    mul_81: "f32[24]" = torch.ops.aten.mul.Tensor(mul_80, 0.1);  mul_80 = None
    mul_82: "f32[24]" = torch.ops.aten.mul.Tensor(primals_194, 0.9)
    add_59: "f32[24]" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
    unsqueeze_44: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_45: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_83: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
    unsqueeze_46: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_47: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_60: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_61: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(add_60, add_45);  add_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_12: "f32[8, 72, 56, 56]" = torch.ops.aten.convolution.default(add_61, primals_117, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_62: "i64[]" = torch.ops.aten.add.Tensor(primals_195, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_12 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
    getitem_24: "f32[1, 72, 1, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 72, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_63: "f32[1, 72, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
    rsqrt_12: "f32[1, 72, 1, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_12: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_25)
    mul_84: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_36: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
    squeeze_37: "f32[72]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
    mul_85: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_86: "f32[72]" = torch.ops.aten.mul.Tensor(primals_196, 0.9)
    add_64: "f32[72]" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
    squeeze_38: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
    mul_87: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0000398612827361);  squeeze_38 = None
    mul_88: "f32[72]" = torch.ops.aten.mul.Tensor(mul_87, 0.1);  mul_87 = None
    mul_89: "f32[72]" = torch.ops.aten.mul.Tensor(primals_197, 0.9)
    add_65: "f32[72]" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
    unsqueeze_48: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1)
    unsqueeze_49: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    mul_90: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_49);  mul_84 = unsqueeze_49 = None
    unsqueeze_50: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1);  primals_26 = None
    unsqueeze_51: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    add_66: "f32[8, 72, 56, 56]" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_51);  mul_90 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_8: "f32[8, 72, 56, 56]" = torch.ops.aten.relu.default(add_66);  add_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_13: "f32[8, 72, 28, 28]" = torch.ops.aten.convolution.default(relu_8, primals_118, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_67: "i64[]" = torch.ops.aten.add.Tensor(primals_198, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_13 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
    getitem_26: "f32[1, 72, 1, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 72, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_68: "f32[1, 72, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
    rsqrt_13: "f32[1, 72, 1, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    sub_13: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_27)
    mul_91: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_39: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
    squeeze_40: "f32[72]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_92: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_93: "f32[72]" = torch.ops.aten.mul.Tensor(primals_199, 0.9)
    add_69: "f32[72]" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
    squeeze_41: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
    mul_94: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0001594642002871);  squeeze_41 = None
    mul_95: "f32[72]" = torch.ops.aten.mul.Tensor(mul_94, 0.1);  mul_94 = None
    mul_96: "f32[72]" = torch.ops.aten.mul.Tensor(primals_200, 0.9)
    add_70: "f32[72]" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
    unsqueeze_52: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1)
    unsqueeze_53: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_97: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
    unsqueeze_54: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1);  primals_28 = None
    unsqueeze_55: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_71: "f32[8, 72, 28, 28]" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_9: "f32[8, 72, 28, 28]" = torch.ops.aten.relu.default(add_71);  add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_14: "f32[8, 40, 28, 28]" = torch.ops.aten.convolution.default(relu_9, primals_119, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_72: "i64[]" = torch.ops.aten.add.Tensor(primals_201, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_14 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
    getitem_28: "f32[1, 40, 1, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 40, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_73: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
    rsqrt_14: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    sub_14: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_29)
    mul_98: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_42: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
    squeeze_43: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
    mul_99: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_100: "f32[40]" = torch.ops.aten.mul.Tensor(primals_202, 0.9)
    add_74: "f32[40]" = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
    squeeze_44: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
    mul_101: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0001594642002871);  squeeze_44 = None
    mul_102: "f32[40]" = torch.ops.aten.mul.Tensor(mul_101, 0.1);  mul_101 = None
    mul_103: "f32[40]" = torch.ops.aten.mul.Tensor(primals_203, 0.9)
    add_75: "f32[40]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
    unsqueeze_56: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_57: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    mul_104: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
    unsqueeze_58: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_59: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    add_76: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_15: "f32[8, 120, 28, 28]" = torch.ops.aten.convolution.default(add_76, primals_120, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_77: "i64[]" = torch.ops.aten.add.Tensor(primals_204, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_15 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
    getitem_30: "f32[1, 120, 1, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 120, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_78: "f32[1, 120, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
    rsqrt_15: "f32[1, 120, 1, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_15: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_31)
    mul_105: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    squeeze_45: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
    squeeze_46: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
    mul_106: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_107: "f32[120]" = torch.ops.aten.mul.Tensor(primals_205, 0.9)
    add_79: "f32[120]" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
    squeeze_47: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
    mul_108: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0001594642002871);  squeeze_47 = None
    mul_109: "f32[120]" = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
    mul_110: "f32[120]" = torch.ops.aten.mul.Tensor(primals_206, 0.9)
    add_80: "f32[120]" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
    unsqueeze_60: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1)
    unsqueeze_61: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_111: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_61);  mul_105 = unsqueeze_61 = None
    unsqueeze_62: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1);  primals_32 = None
    unsqueeze_63: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_81: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_63);  mul_111 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_10: "f32[8, 120, 28, 28]" = torch.ops.aten.relu.default(add_81);  add_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_16: "f32[8, 120, 28, 28]" = torch.ops.aten.convolution.default(relu_10, primals_121, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_82: "i64[]" = torch.ops.aten.add.Tensor(primals_207, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_16 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
    getitem_32: "f32[1, 120, 1, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 120, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_83: "f32[1, 120, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
    rsqrt_16: "f32[1, 120, 1, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
    sub_16: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_33)
    mul_112: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_48: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
    squeeze_49: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
    mul_113: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_114: "f32[120]" = torch.ops.aten.mul.Tensor(primals_208, 0.9)
    add_84: "f32[120]" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
    squeeze_50: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
    mul_115: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0001594642002871);  squeeze_50 = None
    mul_116: "f32[120]" = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
    mul_117: "f32[120]" = torch.ops.aten.mul.Tensor(primals_209, 0.9)
    add_85: "f32[120]" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
    unsqueeze_64: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1)
    unsqueeze_65: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    mul_118: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
    unsqueeze_66: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1);  primals_34 = None
    unsqueeze_67: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    add_86: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_11: "f32[8, 120, 28, 28]" = torch.ops.aten.relu.default(add_86);  add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_17: "f32[8, 40, 28, 28]" = torch.ops.aten.convolution.default(relu_11, primals_122, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_87: "i64[]" = torch.ops.aten.add.Tensor(primals_210, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_17 = torch.ops.aten.var_mean.correction(convolution_17, [0, 2, 3], correction = 0, keepdim = True)
    getitem_34: "f32[1, 40, 1, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 40, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_88: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
    rsqrt_17: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
    sub_17: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, getitem_35)
    mul_119: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    squeeze_51: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
    squeeze_52: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
    mul_120: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_121: "f32[40]" = torch.ops.aten.mul.Tensor(primals_211, 0.9)
    add_89: "f32[40]" = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
    squeeze_53: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
    mul_122: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0001594642002871);  squeeze_53 = None
    mul_123: "f32[40]" = torch.ops.aten.mul.Tensor(mul_122, 0.1);  mul_122 = None
    mul_124: "f32[40]" = torch.ops.aten.mul.Tensor(primals_212, 0.9)
    add_90: "f32[40]" = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
    unsqueeze_68: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_69: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_125: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_69);  mul_119 = unsqueeze_69 = None
    unsqueeze_70: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_71: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_91: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_71);  mul_125 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_92: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_91, add_76);  add_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_18: "f32[8, 120, 28, 28]" = torch.ops.aten.convolution.default(add_92, primals_123, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_93: "i64[]" = torch.ops.aten.add.Tensor(primals_213, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_18 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
    getitem_36: "f32[1, 120, 1, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 120, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_94: "f32[1, 120, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
    rsqrt_18: "f32[1, 120, 1, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    sub_18: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_37)
    mul_126: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    squeeze_54: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
    squeeze_55: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
    mul_127: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_128: "f32[120]" = torch.ops.aten.mul.Tensor(primals_214, 0.9)
    add_95: "f32[120]" = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
    squeeze_56: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
    mul_129: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0001594642002871);  squeeze_56 = None
    mul_130: "f32[120]" = torch.ops.aten.mul.Tensor(mul_129, 0.1);  mul_129 = None
    mul_131: "f32[120]" = torch.ops.aten.mul.Tensor(primals_215, 0.9)
    add_96: "f32[120]" = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
    unsqueeze_72: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1)
    unsqueeze_73: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    mul_132: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_73);  mul_126 = unsqueeze_73 = None
    unsqueeze_74: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1);  primals_38 = None
    unsqueeze_75: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    add_97: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_75);  mul_132 = unsqueeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_12: "f32[8, 120, 28, 28]" = torch.ops.aten.relu.default(add_97);  add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_19: "f32[8, 120, 28, 28]" = torch.ops.aten.convolution.default(relu_12, primals_124, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_98: "i64[]" = torch.ops.aten.add.Tensor(primals_216, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_19 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
    getitem_38: "f32[1, 120, 1, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 120, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_99: "f32[1, 120, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
    rsqrt_19: "f32[1, 120, 1, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
    sub_19: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_39)
    mul_133: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    squeeze_57: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
    squeeze_58: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
    mul_134: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_135: "f32[120]" = torch.ops.aten.mul.Tensor(primals_217, 0.9)
    add_100: "f32[120]" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
    squeeze_59: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
    mul_136: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0001594642002871);  squeeze_59 = None
    mul_137: "f32[120]" = torch.ops.aten.mul.Tensor(mul_136, 0.1);  mul_136 = None
    mul_138: "f32[120]" = torch.ops.aten.mul.Tensor(primals_218, 0.9)
    add_101: "f32[120]" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
    unsqueeze_76: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1)
    unsqueeze_77: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_139: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_77);  mul_133 = unsqueeze_77 = None
    unsqueeze_78: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1);  primals_40 = None
    unsqueeze_79: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_102: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_139, unsqueeze_79);  mul_139 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_13: "f32[8, 120, 28, 28]" = torch.ops.aten.relu.default(add_102);  add_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_20: "f32[8, 40, 28, 28]" = torch.ops.aten.convolution.default(relu_13, primals_125, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_103: "i64[]" = torch.ops.aten.add.Tensor(primals_219, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_20 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
    getitem_40: "f32[1, 40, 1, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 40, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_104: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
    rsqrt_20: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
    sub_20: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_41)
    mul_140: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    squeeze_60: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
    squeeze_61: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
    mul_141: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_142: "f32[40]" = torch.ops.aten.mul.Tensor(primals_220, 0.9)
    add_105: "f32[40]" = torch.ops.aten.add.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
    squeeze_62: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
    mul_143: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0001594642002871);  squeeze_62 = None
    mul_144: "f32[40]" = torch.ops.aten.mul.Tensor(mul_143, 0.1);  mul_143 = None
    mul_145: "f32[40]" = torch.ops.aten.mul.Tensor(primals_221, 0.9)
    add_106: "f32[40]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
    unsqueeze_80: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_81: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    mul_146: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_140, unsqueeze_81);  mul_140 = unsqueeze_81 = None
    unsqueeze_82: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_83: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    add_107: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_83);  mul_146 = unsqueeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_108: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_107, add_92);  add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_21: "f32[8, 240, 28, 28]" = torch.ops.aten.convolution.default(add_108, primals_126, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_109: "i64[]" = torch.ops.aten.add.Tensor(primals_222, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_21 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
    getitem_42: "f32[1, 240, 1, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 240, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_110: "f32[1, 240, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
    rsqrt_21: "f32[1, 240, 1, 1]" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
    sub_21: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_43)
    mul_147: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    squeeze_63: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
    squeeze_64: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
    mul_148: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_149: "f32[240]" = torch.ops.aten.mul.Tensor(primals_223, 0.9)
    add_111: "f32[240]" = torch.ops.aten.add.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
    squeeze_65: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
    mul_150: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0001594642002871);  squeeze_65 = None
    mul_151: "f32[240]" = torch.ops.aten.mul.Tensor(mul_150, 0.1);  mul_150 = None
    mul_152: "f32[240]" = torch.ops.aten.mul.Tensor(primals_224, 0.9)
    add_112: "f32[240]" = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
    unsqueeze_84: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1)
    unsqueeze_85: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_153: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_147, unsqueeze_85);  mul_147 = unsqueeze_85 = None
    unsqueeze_86: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1);  primals_44 = None
    unsqueeze_87: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_113: "f32[8, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_153, unsqueeze_87);  mul_153 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_14: "f32[8, 240, 28, 28]" = torch.ops.aten.relu.default(add_113);  add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_22: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(relu_14, primals_127, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 240)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_114: "i64[]" = torch.ops.aten.add.Tensor(primals_225, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_22 = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
    getitem_44: "f32[1, 240, 1, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 240, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_115: "f32[1, 240, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05)
    rsqrt_22: "f32[1, 240, 1, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    sub_22: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, getitem_45)
    mul_154: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    squeeze_66: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
    squeeze_67: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
    mul_155: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_156: "f32[240]" = torch.ops.aten.mul.Tensor(primals_226, 0.9)
    add_116: "f32[240]" = torch.ops.aten.add.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
    squeeze_68: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
    mul_157: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0006381620931717);  squeeze_68 = None
    mul_158: "f32[240]" = torch.ops.aten.mul.Tensor(mul_157, 0.1);  mul_157 = None
    mul_159: "f32[240]" = torch.ops.aten.mul.Tensor(primals_227, 0.9)
    add_117: "f32[240]" = torch.ops.aten.add.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
    unsqueeze_88: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1)
    unsqueeze_89: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    mul_160: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_89);  mul_154 = unsqueeze_89 = None
    unsqueeze_90: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1);  primals_46 = None
    unsqueeze_91: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    add_118: "f32[8, 240, 14, 14]" = torch.ops.aten.add.Tensor(mul_160, unsqueeze_91);  mul_160 = unsqueeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_15: "f32[8, 240, 14, 14]" = torch.ops.aten.relu.default(add_118);  add_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_23: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(relu_15, primals_128, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_119: "i64[]" = torch.ops.aten.add.Tensor(primals_228, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_23 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
    getitem_46: "f32[1, 80, 1, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 80, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_120: "f32[1, 80, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
    rsqrt_23: "f32[1, 80, 1, 1]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
    sub_23: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_47)
    mul_161: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    squeeze_69: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
    squeeze_70: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
    mul_162: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
    mul_163: "f32[80]" = torch.ops.aten.mul.Tensor(primals_229, 0.9)
    add_121: "f32[80]" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
    squeeze_71: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
    mul_164: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0006381620931717);  squeeze_71 = None
    mul_165: "f32[80]" = torch.ops.aten.mul.Tensor(mul_164, 0.1);  mul_164 = None
    mul_166: "f32[80]" = torch.ops.aten.mul.Tensor(primals_230, 0.9)
    add_122: "f32[80]" = torch.ops.aten.add.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
    unsqueeze_92: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_93: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_167: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_93);  mul_161 = unsqueeze_93 = None
    unsqueeze_94: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_95: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_123: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_95);  mul_167 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_24: "f32[8, 480, 14, 14]" = torch.ops.aten.convolution.default(add_123, primals_129, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_124: "i64[]" = torch.ops.aten.add.Tensor(primals_231, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_24 = torch.ops.aten.var_mean.correction(convolution_24, [0, 2, 3], correction = 0, keepdim = True)
    getitem_48: "f32[1, 480, 1, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 480, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_125: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05)
    rsqrt_24: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_125);  add_125 = None
    sub_24: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_24, getitem_49)
    mul_168: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    squeeze_72: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
    squeeze_73: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
    mul_169: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_170: "f32[480]" = torch.ops.aten.mul.Tensor(primals_232, 0.9)
    add_126: "f32[480]" = torch.ops.aten.add.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
    squeeze_74: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
    mul_171: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0006381620931717);  squeeze_74 = None
    mul_172: "f32[480]" = torch.ops.aten.mul.Tensor(mul_171, 0.1);  mul_171 = None
    mul_173: "f32[480]" = torch.ops.aten.mul.Tensor(primals_233, 0.9)
    add_127: "f32[480]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
    unsqueeze_96: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1)
    unsqueeze_97: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    mul_174: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_168, unsqueeze_97);  mul_168 = unsqueeze_97 = None
    unsqueeze_98: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1);  primals_50 = None
    unsqueeze_99: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    add_128: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_174, unsqueeze_99);  mul_174 = unsqueeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_16: "f32[8, 480, 14, 14]" = torch.ops.aten.relu.default(add_128);  add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_25: "f32[8, 480, 14, 14]" = torch.ops.aten.convolution.default(relu_16, primals_130, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 480)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_129: "i64[]" = torch.ops.aten.add.Tensor(primals_234, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_25 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
    getitem_50: "f32[1, 480, 1, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 480, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_130: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
    rsqrt_25: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
    sub_25: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_51)
    mul_175: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    squeeze_75: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
    squeeze_76: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
    mul_176: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_177: "f32[480]" = torch.ops.aten.mul.Tensor(primals_235, 0.9)
    add_131: "f32[480]" = torch.ops.aten.add.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
    squeeze_77: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
    mul_178: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0006381620931717);  squeeze_77 = None
    mul_179: "f32[480]" = torch.ops.aten.mul.Tensor(mul_178, 0.1);  mul_178 = None
    mul_180: "f32[480]" = torch.ops.aten.mul.Tensor(primals_236, 0.9)
    add_132: "f32[480]" = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
    unsqueeze_100: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1)
    unsqueeze_101: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_181: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_101);  mul_175 = unsqueeze_101 = None
    unsqueeze_102: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1);  primals_52 = None
    unsqueeze_103: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_133: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_103);  mul_181 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_17: "f32[8, 480, 14, 14]" = torch.ops.aten.relu.default(add_133);  add_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_26: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(relu_17, primals_131, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_134: "i64[]" = torch.ops.aten.add.Tensor(primals_237, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_26 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
    getitem_52: "f32[1, 80, 1, 1]" = var_mean_26[0]
    getitem_53: "f32[1, 80, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_135: "f32[1, 80, 1, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05)
    rsqrt_26: "f32[1, 80, 1, 1]" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
    sub_26: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_53)
    mul_182: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    squeeze_78: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
    squeeze_79: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
    mul_183: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_184: "f32[80]" = torch.ops.aten.mul.Tensor(primals_238, 0.9)
    add_136: "f32[80]" = torch.ops.aten.add.Tensor(mul_183, mul_184);  mul_183 = mul_184 = None
    squeeze_80: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
    mul_185: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0006381620931717);  squeeze_80 = None
    mul_186: "f32[80]" = torch.ops.aten.mul.Tensor(mul_185, 0.1);  mul_185 = None
    mul_187: "f32[80]" = torch.ops.aten.mul.Tensor(primals_239, 0.9)
    add_137: "f32[80]" = torch.ops.aten.add.Tensor(mul_186, mul_187);  mul_186 = mul_187 = None
    unsqueeze_104: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_105: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    mul_188: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_105);  mul_182 = unsqueeze_105 = None
    unsqueeze_106: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_107: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    add_138: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_107);  mul_188 = unsqueeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_139: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_138, add_123);  add_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_27: "f32[8, 480, 14, 14]" = torch.ops.aten.convolution.default(add_139, primals_132, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_140: "i64[]" = torch.ops.aten.add.Tensor(primals_240, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_27 = torch.ops.aten.var_mean.correction(convolution_27, [0, 2, 3], correction = 0, keepdim = True)
    getitem_54: "f32[1, 480, 1, 1]" = var_mean_27[0]
    getitem_55: "f32[1, 480, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_141: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05)
    rsqrt_27: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
    sub_27: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, getitem_55)
    mul_189: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    squeeze_81: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
    squeeze_82: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
    mul_190: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
    mul_191: "f32[480]" = torch.ops.aten.mul.Tensor(primals_241, 0.9)
    add_142: "f32[480]" = torch.ops.aten.add.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
    squeeze_83: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
    mul_192: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0006381620931717);  squeeze_83 = None
    mul_193: "f32[480]" = torch.ops.aten.mul.Tensor(mul_192, 0.1);  mul_192 = None
    mul_194: "f32[480]" = torch.ops.aten.mul.Tensor(primals_242, 0.9)
    add_143: "f32[480]" = torch.ops.aten.add.Tensor(mul_193, mul_194);  mul_193 = mul_194 = None
    unsqueeze_108: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1)
    unsqueeze_109: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_195: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_189, unsqueeze_109);  mul_189 = unsqueeze_109 = None
    unsqueeze_110: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1);  primals_56 = None
    unsqueeze_111: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_144: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_195, unsqueeze_111);  mul_195 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_18: "f32[8, 480, 14, 14]" = torch.ops.aten.relu.default(add_144);  add_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_28: "f32[8, 480, 14, 14]" = torch.ops.aten.convolution.default(relu_18, primals_133, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 480)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_145: "i64[]" = torch.ops.aten.add.Tensor(primals_243, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_28 = torch.ops.aten.var_mean.correction(convolution_28, [0, 2, 3], correction = 0, keepdim = True)
    getitem_56: "f32[1, 480, 1, 1]" = var_mean_28[0]
    getitem_57: "f32[1, 480, 1, 1]" = var_mean_28[1];  var_mean_28 = None
    add_146: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05)
    rsqrt_28: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
    sub_28: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, getitem_57)
    mul_196: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    squeeze_84: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
    squeeze_85: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
    mul_197: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
    mul_198: "f32[480]" = torch.ops.aten.mul.Tensor(primals_244, 0.9)
    add_147: "f32[480]" = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
    squeeze_86: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
    mul_199: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0006381620931717);  squeeze_86 = None
    mul_200: "f32[480]" = torch.ops.aten.mul.Tensor(mul_199, 0.1);  mul_199 = None
    mul_201: "f32[480]" = torch.ops.aten.mul.Tensor(primals_245, 0.9)
    add_148: "f32[480]" = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
    unsqueeze_112: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1)
    unsqueeze_113: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    mul_202: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_113);  mul_196 = unsqueeze_113 = None
    unsqueeze_114: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_58, -1);  primals_58 = None
    unsqueeze_115: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    add_149: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_115);  mul_202 = unsqueeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_19: "f32[8, 480, 14, 14]" = torch.ops.aten.relu.default(add_149);  add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_29: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(relu_19, primals_134, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_150: "i64[]" = torch.ops.aten.add.Tensor(primals_246, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_29 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
    getitem_58: "f32[1, 80, 1, 1]" = var_mean_29[0]
    getitem_59: "f32[1, 80, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_151: "f32[1, 80, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
    rsqrt_29: "f32[1, 80, 1, 1]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
    sub_29: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, getitem_59)
    mul_203: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    squeeze_87: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
    squeeze_88: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
    mul_204: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
    mul_205: "f32[80]" = torch.ops.aten.mul.Tensor(primals_247, 0.9)
    add_152: "f32[80]" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
    squeeze_89: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
    mul_206: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0006381620931717);  squeeze_89 = None
    mul_207: "f32[80]" = torch.ops.aten.mul.Tensor(mul_206, 0.1);  mul_206 = None
    mul_208: "f32[80]" = torch.ops.aten.mul.Tensor(primals_248, 0.9)
    add_153: "f32[80]" = torch.ops.aten.add.Tensor(mul_207, mul_208);  mul_207 = mul_208 = None
    unsqueeze_116: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_117: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_209: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_117);  mul_203 = unsqueeze_117 = None
    unsqueeze_118: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_119: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_154: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_119);  mul_209 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_155: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_154, add_139);  add_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_30: "f32[8, 480, 14, 14]" = torch.ops.aten.convolution.default(add_155, primals_135, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_156: "i64[]" = torch.ops.aten.add.Tensor(primals_249, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_30 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
    getitem_60: "f32[1, 480, 1, 1]" = var_mean_30[0]
    getitem_61: "f32[1, 480, 1, 1]" = var_mean_30[1];  var_mean_30 = None
    add_157: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
    rsqrt_30: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
    sub_30: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, getitem_61)
    mul_210: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    squeeze_90: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
    squeeze_91: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
    mul_211: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
    mul_212: "f32[480]" = torch.ops.aten.mul.Tensor(primals_250, 0.9)
    add_158: "f32[480]" = torch.ops.aten.add.Tensor(mul_211, mul_212);  mul_211 = mul_212 = None
    squeeze_92: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
    mul_213: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0006381620931717);  squeeze_92 = None
    mul_214: "f32[480]" = torch.ops.aten.mul.Tensor(mul_213, 0.1);  mul_213 = None
    mul_215: "f32[480]" = torch.ops.aten.mul.Tensor(primals_251, 0.9)
    add_159: "f32[480]" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
    unsqueeze_120: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_61, -1)
    unsqueeze_121: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    mul_216: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_121);  mul_210 = unsqueeze_121 = None
    unsqueeze_122: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1);  primals_62 = None
    unsqueeze_123: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    add_160: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_216, unsqueeze_123);  mul_216 = unsqueeze_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_20: "f32[8, 480, 14, 14]" = torch.ops.aten.relu.default(add_160);  add_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_31: "f32[8, 480, 14, 14]" = torch.ops.aten.convolution.default(relu_20, primals_136, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_161: "i64[]" = torch.ops.aten.add.Tensor(primals_252, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_31 = torch.ops.aten.var_mean.correction(convolution_31, [0, 2, 3], correction = 0, keepdim = True)
    getitem_62: "f32[1, 480, 1, 1]" = var_mean_31[0]
    getitem_63: "f32[1, 480, 1, 1]" = var_mean_31[1];  var_mean_31 = None
    add_162: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
    rsqrt_31: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
    sub_31: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, getitem_63)
    mul_217: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    squeeze_93: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
    squeeze_94: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
    mul_218: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
    mul_219: "f32[480]" = torch.ops.aten.mul.Tensor(primals_253, 0.9)
    add_163: "f32[480]" = torch.ops.aten.add.Tensor(mul_218, mul_219);  mul_218 = mul_219 = None
    squeeze_95: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
    mul_220: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0006381620931717);  squeeze_95 = None
    mul_221: "f32[480]" = torch.ops.aten.mul.Tensor(mul_220, 0.1);  mul_220 = None
    mul_222: "f32[480]" = torch.ops.aten.mul.Tensor(primals_254, 0.9)
    add_164: "f32[480]" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
    unsqueeze_124: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1)
    unsqueeze_125: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_223: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_125);  mul_217 = unsqueeze_125 = None
    unsqueeze_126: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_64, -1);  primals_64 = None
    unsqueeze_127: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_165: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_223, unsqueeze_127);  mul_223 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_21: "f32[8, 480, 14, 14]" = torch.ops.aten.relu.default(add_165);  add_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_32: "f32[8, 96, 14, 14]" = torch.ops.aten.convolution.default(relu_21, primals_137, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_166: "i64[]" = torch.ops.aten.add.Tensor(primals_255, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_32 = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
    getitem_64: "f32[1, 96, 1, 1]" = var_mean_32[0]
    getitem_65: "f32[1, 96, 1, 1]" = var_mean_32[1];  var_mean_32 = None
    add_167: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
    rsqrt_32: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_167);  add_167 = None
    sub_32: "f32[8, 96, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, getitem_65)
    mul_224: "f32[8, 96, 14, 14]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    squeeze_96: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
    squeeze_97: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
    mul_225: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
    mul_226: "f32[96]" = torch.ops.aten.mul.Tensor(primals_256, 0.9)
    add_168: "f32[96]" = torch.ops.aten.add.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
    squeeze_98: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
    mul_227: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0006381620931717);  squeeze_98 = None
    mul_228: "f32[96]" = torch.ops.aten.mul.Tensor(mul_227, 0.1);  mul_227 = None
    mul_229: "f32[96]" = torch.ops.aten.mul.Tensor(primals_257, 0.9)
    add_169: "f32[96]" = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
    unsqueeze_128: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
    unsqueeze_129: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    mul_230: "f32[8, 96, 14, 14]" = torch.ops.aten.mul.Tensor(mul_224, unsqueeze_129);  mul_224 = unsqueeze_129 = None
    unsqueeze_130: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
    unsqueeze_131: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    add_170: "f32[8, 96, 14, 14]" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_131);  mul_230 = unsqueeze_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_33: "f32[8, 576, 14, 14]" = torch.ops.aten.convolution.default(add_170, primals_138, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_171: "i64[]" = torch.ops.aten.add.Tensor(primals_258, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_33 = torch.ops.aten.var_mean.correction(convolution_33, [0, 2, 3], correction = 0, keepdim = True)
    getitem_66: "f32[1, 576, 1, 1]" = var_mean_33[0]
    getitem_67: "f32[1, 576, 1, 1]" = var_mean_33[1];  var_mean_33 = None
    add_172: "f32[1, 576, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
    rsqrt_33: "f32[1, 576, 1, 1]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
    sub_33: "f32[8, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, getitem_67)
    mul_231: "f32[8, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    squeeze_99: "f32[576]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
    squeeze_100: "f32[576]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
    mul_232: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
    mul_233: "f32[576]" = torch.ops.aten.mul.Tensor(primals_259, 0.9)
    add_173: "f32[576]" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
    squeeze_101: "f32[576]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
    mul_234: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0006381620931717);  squeeze_101 = None
    mul_235: "f32[576]" = torch.ops.aten.mul.Tensor(mul_234, 0.1);  mul_234 = None
    mul_236: "f32[576]" = torch.ops.aten.mul.Tensor(primals_260, 0.9)
    add_174: "f32[576]" = torch.ops.aten.add.Tensor(mul_235, mul_236);  mul_235 = mul_236 = None
    unsqueeze_132: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_67, -1)
    unsqueeze_133: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_237: "f32[8, 576, 14, 14]" = torch.ops.aten.mul.Tensor(mul_231, unsqueeze_133);  mul_231 = unsqueeze_133 = None
    unsqueeze_134: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1);  primals_68 = None
    unsqueeze_135: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_175: "f32[8, 576, 14, 14]" = torch.ops.aten.add.Tensor(mul_237, unsqueeze_135);  mul_237 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_22: "f32[8, 576, 14, 14]" = torch.ops.aten.relu.default(add_175);  add_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_34: "f32[8, 576, 14, 14]" = torch.ops.aten.convolution.default(relu_22, primals_139, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 576)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_176: "i64[]" = torch.ops.aten.add.Tensor(primals_261, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_34 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
    getitem_68: "f32[1, 576, 1, 1]" = var_mean_34[0]
    getitem_69: "f32[1, 576, 1, 1]" = var_mean_34[1];  var_mean_34 = None
    add_177: "f32[1, 576, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05)
    rsqrt_34: "f32[1, 576, 1, 1]" = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
    sub_34: "f32[8, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_69)
    mul_238: "f32[8, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    squeeze_102: "f32[576]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
    squeeze_103: "f32[576]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
    mul_239: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
    mul_240: "f32[576]" = torch.ops.aten.mul.Tensor(primals_262, 0.9)
    add_178: "f32[576]" = torch.ops.aten.add.Tensor(mul_239, mul_240);  mul_239 = mul_240 = None
    squeeze_104: "f32[576]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
    mul_241: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0006381620931717);  squeeze_104 = None
    mul_242: "f32[576]" = torch.ops.aten.mul.Tensor(mul_241, 0.1);  mul_241 = None
    mul_243: "f32[576]" = torch.ops.aten.mul.Tensor(primals_263, 0.9)
    add_179: "f32[576]" = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
    unsqueeze_136: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1)
    unsqueeze_137: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    mul_244: "f32[8, 576, 14, 14]" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_137);  mul_238 = unsqueeze_137 = None
    unsqueeze_138: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_70, -1);  primals_70 = None
    unsqueeze_139: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    add_180: "f32[8, 576, 14, 14]" = torch.ops.aten.add.Tensor(mul_244, unsqueeze_139);  mul_244 = unsqueeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_23: "f32[8, 576, 14, 14]" = torch.ops.aten.relu.default(add_180);  add_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_35: "f32[8, 96, 14, 14]" = torch.ops.aten.convolution.default(relu_23, primals_140, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_181: "i64[]" = torch.ops.aten.add.Tensor(primals_264, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_35 = torch.ops.aten.var_mean.correction(convolution_35, [0, 2, 3], correction = 0, keepdim = True)
    getitem_70: "f32[1, 96, 1, 1]" = var_mean_35[0]
    getitem_71: "f32[1, 96, 1, 1]" = var_mean_35[1];  var_mean_35 = None
    add_182: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05)
    rsqrt_35: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
    sub_35: "f32[8, 96, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, getitem_71)
    mul_245: "f32[8, 96, 14, 14]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    squeeze_105: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
    squeeze_106: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
    mul_246: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
    mul_247: "f32[96]" = torch.ops.aten.mul.Tensor(primals_265, 0.9)
    add_183: "f32[96]" = torch.ops.aten.add.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
    squeeze_107: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
    mul_248: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0006381620931717);  squeeze_107 = None
    mul_249: "f32[96]" = torch.ops.aten.mul.Tensor(mul_248, 0.1);  mul_248 = None
    mul_250: "f32[96]" = torch.ops.aten.mul.Tensor(primals_266, 0.9)
    add_184: "f32[96]" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
    unsqueeze_140: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_141: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_251: "f32[8, 96, 14, 14]" = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_141);  mul_245 = unsqueeze_141 = None
    unsqueeze_142: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_143: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_185: "f32[8, 96, 14, 14]" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_143);  mul_251 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_186: "f32[8, 96, 14, 14]" = torch.ops.aten.add.Tensor(add_185, add_170);  add_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_36: "f32[8, 576, 14, 14]" = torch.ops.aten.convolution.default(add_186, primals_141, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_187: "i64[]" = torch.ops.aten.add.Tensor(primals_267, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_36 = torch.ops.aten.var_mean.correction(convolution_36, [0, 2, 3], correction = 0, keepdim = True)
    getitem_72: "f32[1, 576, 1, 1]" = var_mean_36[0]
    getitem_73: "f32[1, 576, 1, 1]" = var_mean_36[1];  var_mean_36 = None
    add_188: "f32[1, 576, 1, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05)
    rsqrt_36: "f32[1, 576, 1, 1]" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
    sub_36: "f32[8, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, getitem_73)
    mul_252: "f32[8, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    squeeze_108: "f32[576]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
    squeeze_109: "f32[576]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
    mul_253: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
    mul_254: "f32[576]" = torch.ops.aten.mul.Tensor(primals_268, 0.9)
    add_189: "f32[576]" = torch.ops.aten.add.Tensor(mul_253, mul_254);  mul_253 = mul_254 = None
    squeeze_110: "f32[576]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
    mul_255: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0006381620931717);  squeeze_110 = None
    mul_256: "f32[576]" = torch.ops.aten.mul.Tensor(mul_255, 0.1);  mul_255 = None
    mul_257: "f32[576]" = torch.ops.aten.mul.Tensor(primals_269, 0.9)
    add_190: "f32[576]" = torch.ops.aten.add.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
    unsqueeze_144: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_73, -1)
    unsqueeze_145: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    mul_258: "f32[8, 576, 14, 14]" = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_145);  mul_252 = unsqueeze_145 = None
    unsqueeze_146: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1);  primals_74 = None
    unsqueeze_147: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    add_191: "f32[8, 576, 14, 14]" = torch.ops.aten.add.Tensor(mul_258, unsqueeze_147);  mul_258 = unsqueeze_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_24: "f32[8, 576, 14, 14]" = torch.ops.aten.relu.default(add_191);  add_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_37: "f32[8, 576, 7, 7]" = torch.ops.aten.convolution.default(relu_24, primals_142, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 576)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_192: "i64[]" = torch.ops.aten.add.Tensor(primals_270, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_37 = torch.ops.aten.var_mean.correction(convolution_37, [0, 2, 3], correction = 0, keepdim = True)
    getitem_74: "f32[1, 576, 1, 1]" = var_mean_37[0]
    getitem_75: "f32[1, 576, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_193: "f32[1, 576, 1, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05)
    rsqrt_37: "f32[1, 576, 1, 1]" = torch.ops.aten.rsqrt.default(add_193);  add_193 = None
    sub_37: "f32[8, 576, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_37, getitem_75)
    mul_259: "f32[8, 576, 7, 7]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    squeeze_111: "f32[576]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
    squeeze_112: "f32[576]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
    mul_260: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
    mul_261: "f32[576]" = torch.ops.aten.mul.Tensor(primals_271, 0.9)
    add_194: "f32[576]" = torch.ops.aten.add.Tensor(mul_260, mul_261);  mul_260 = mul_261 = None
    squeeze_113: "f32[576]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
    mul_262: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0025575447570332);  squeeze_113 = None
    mul_263: "f32[576]" = torch.ops.aten.mul.Tensor(mul_262, 0.1);  mul_262 = None
    mul_264: "f32[576]" = torch.ops.aten.mul.Tensor(primals_272, 0.9)
    add_195: "f32[576]" = torch.ops.aten.add.Tensor(mul_263, mul_264);  mul_263 = mul_264 = None
    unsqueeze_148: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1)
    unsqueeze_149: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_265: "f32[8, 576, 7, 7]" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_149);  mul_259 = unsqueeze_149 = None
    unsqueeze_150: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_76, -1);  primals_76 = None
    unsqueeze_151: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_196: "f32[8, 576, 7, 7]" = torch.ops.aten.add.Tensor(mul_265, unsqueeze_151);  mul_265 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_25: "f32[8, 576, 7, 7]" = torch.ops.aten.relu.default(add_196);  add_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_38: "f32[8, 192, 7, 7]" = torch.ops.aten.convolution.default(relu_25, primals_143, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_197: "i64[]" = torch.ops.aten.add.Tensor(primals_273, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_38 = torch.ops.aten.var_mean.correction(convolution_38, [0, 2, 3], correction = 0, keepdim = True)
    getitem_76: "f32[1, 192, 1, 1]" = var_mean_38[0]
    getitem_77: "f32[1, 192, 1, 1]" = var_mean_38[1];  var_mean_38 = None
    add_198: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
    rsqrt_38: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_198);  add_198 = None
    sub_38: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_38, getitem_77)
    mul_266: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    squeeze_114: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
    squeeze_115: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
    mul_267: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
    mul_268: "f32[192]" = torch.ops.aten.mul.Tensor(primals_274, 0.9)
    add_199: "f32[192]" = torch.ops.aten.add.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
    squeeze_116: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
    mul_269: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0025575447570332);  squeeze_116 = None
    mul_270: "f32[192]" = torch.ops.aten.mul.Tensor(mul_269, 0.1);  mul_269 = None
    mul_271: "f32[192]" = torch.ops.aten.mul.Tensor(primals_275, 0.9)
    add_200: "f32[192]" = torch.ops.aten.add.Tensor(mul_270, mul_271);  mul_270 = mul_271 = None
    unsqueeze_152: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
    unsqueeze_153: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    mul_272: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(mul_266, unsqueeze_153);  mul_266 = unsqueeze_153 = None
    unsqueeze_154: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
    unsqueeze_155: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    add_201: "f32[8, 192, 7, 7]" = torch.ops.aten.add.Tensor(mul_272, unsqueeze_155);  mul_272 = unsqueeze_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_39: "f32[8, 1152, 7, 7]" = torch.ops.aten.convolution.default(add_201, primals_144, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_202: "i64[]" = torch.ops.aten.add.Tensor(primals_276, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_39 = torch.ops.aten.var_mean.correction(convolution_39, [0, 2, 3], correction = 0, keepdim = True)
    getitem_78: "f32[1, 1152, 1, 1]" = var_mean_39[0]
    getitem_79: "f32[1, 1152, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_203: "f32[1, 1152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
    rsqrt_39: "f32[1, 1152, 1, 1]" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
    sub_39: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_39, getitem_79)
    mul_273: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    squeeze_117: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
    squeeze_118: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
    mul_274: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
    mul_275: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_277, 0.9)
    add_204: "f32[1152]" = torch.ops.aten.add.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None
    squeeze_119: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
    mul_276: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0025575447570332);  squeeze_119 = None
    mul_277: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_276, 0.1);  mul_276 = None
    mul_278: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_278, 0.9)
    add_205: "f32[1152]" = torch.ops.aten.add.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
    unsqueeze_156: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_79, -1)
    unsqueeze_157: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_279: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_273, unsqueeze_157);  mul_273 = unsqueeze_157 = None
    unsqueeze_158: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1);  primals_80 = None
    unsqueeze_159: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_206: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_279, unsqueeze_159);  mul_279 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_26: "f32[8, 1152, 7, 7]" = torch.ops.aten.relu.default(add_206);  add_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_40: "f32[8, 1152, 7, 7]" = torch.ops.aten.convolution.default(relu_26, primals_145, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_207: "i64[]" = torch.ops.aten.add.Tensor(primals_279, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_40 = torch.ops.aten.var_mean.correction(convolution_40, [0, 2, 3], correction = 0, keepdim = True)
    getitem_80: "f32[1, 1152, 1, 1]" = var_mean_40[0]
    getitem_81: "f32[1, 1152, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_208: "f32[1, 1152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05)
    rsqrt_40: "f32[1, 1152, 1, 1]" = torch.ops.aten.rsqrt.default(add_208);  add_208 = None
    sub_40: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_81)
    mul_280: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    squeeze_120: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
    squeeze_121: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
    mul_281: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
    mul_282: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_280, 0.9)
    add_209: "f32[1152]" = torch.ops.aten.add.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
    squeeze_122: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
    mul_283: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0025575447570332);  squeeze_122 = None
    mul_284: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_283, 0.1);  mul_283 = None
    mul_285: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_281, 0.9)
    add_210: "f32[1152]" = torch.ops.aten.add.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
    unsqueeze_160: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1)
    unsqueeze_161: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    mul_286: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_161);  mul_280 = unsqueeze_161 = None
    unsqueeze_162: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_82, -1);  primals_82 = None
    unsqueeze_163: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    add_211: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_286, unsqueeze_163);  mul_286 = unsqueeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_27: "f32[8, 1152, 7, 7]" = torch.ops.aten.relu.default(add_211);  add_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_41: "f32[8, 192, 7, 7]" = torch.ops.aten.convolution.default(relu_27, primals_146, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_212: "i64[]" = torch.ops.aten.add.Tensor(primals_282, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_41 = torch.ops.aten.var_mean.correction(convolution_41, [0, 2, 3], correction = 0, keepdim = True)
    getitem_82: "f32[1, 192, 1, 1]" = var_mean_41[0]
    getitem_83: "f32[1, 192, 1, 1]" = var_mean_41[1];  var_mean_41 = None
    add_213: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05)
    rsqrt_41: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
    sub_41: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_41, getitem_83)
    mul_287: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
    squeeze_123: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
    squeeze_124: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
    mul_288: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
    mul_289: "f32[192]" = torch.ops.aten.mul.Tensor(primals_283, 0.9)
    add_214: "f32[192]" = torch.ops.aten.add.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
    squeeze_125: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
    mul_290: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0025575447570332);  squeeze_125 = None
    mul_291: "f32[192]" = torch.ops.aten.mul.Tensor(mul_290, 0.1);  mul_290 = None
    mul_292: "f32[192]" = torch.ops.aten.mul.Tensor(primals_284, 0.9)
    add_215: "f32[192]" = torch.ops.aten.add.Tensor(mul_291, mul_292);  mul_291 = mul_292 = None
    unsqueeze_164: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1)
    unsqueeze_165: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_293: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(mul_287, unsqueeze_165);  mul_287 = unsqueeze_165 = None
    unsqueeze_166: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
    unsqueeze_167: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_216: "f32[8, 192, 7, 7]" = torch.ops.aten.add.Tensor(mul_293, unsqueeze_167);  mul_293 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_217: "f32[8, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_216, add_201);  add_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_42: "f32[8, 1152, 7, 7]" = torch.ops.aten.convolution.default(add_217, primals_147, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_218: "i64[]" = torch.ops.aten.add.Tensor(primals_285, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_42 = torch.ops.aten.var_mean.correction(convolution_42, [0, 2, 3], correction = 0, keepdim = True)
    getitem_84: "f32[1, 1152, 1, 1]" = var_mean_42[0]
    getitem_85: "f32[1, 1152, 1, 1]" = var_mean_42[1];  var_mean_42 = None
    add_219: "f32[1, 1152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05)
    rsqrt_42: "f32[1, 1152, 1, 1]" = torch.ops.aten.rsqrt.default(add_219);  add_219 = None
    sub_42: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_42, getitem_85)
    mul_294: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
    squeeze_126: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
    squeeze_127: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
    mul_295: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
    mul_296: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_286, 0.9)
    add_220: "f32[1152]" = torch.ops.aten.add.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None
    squeeze_128: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
    mul_297: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0025575447570332);  squeeze_128 = None
    mul_298: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_297, 0.1);  mul_297 = None
    mul_299: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_287, 0.9)
    add_221: "f32[1152]" = torch.ops.aten.add.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
    unsqueeze_168: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_85, -1)
    unsqueeze_169: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    mul_300: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_294, unsqueeze_169);  mul_294 = unsqueeze_169 = None
    unsqueeze_170: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1);  primals_86 = None
    unsqueeze_171: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    add_222: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_300, unsqueeze_171);  mul_300 = unsqueeze_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_28: "f32[8, 1152, 7, 7]" = torch.ops.aten.relu.default(add_222);  add_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_43: "f32[8, 1152, 7, 7]" = torch.ops.aten.convolution.default(relu_28, primals_148, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_223: "i64[]" = torch.ops.aten.add.Tensor(primals_288, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_43 = torch.ops.aten.var_mean.correction(convolution_43, [0, 2, 3], correction = 0, keepdim = True)
    getitem_86: "f32[1, 1152, 1, 1]" = var_mean_43[0]
    getitem_87: "f32[1, 1152, 1, 1]" = var_mean_43[1];  var_mean_43 = None
    add_224: "f32[1, 1152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05)
    rsqrt_43: "f32[1, 1152, 1, 1]" = torch.ops.aten.rsqrt.default(add_224);  add_224 = None
    sub_43: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_43, getitem_87)
    mul_301: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
    squeeze_129: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
    squeeze_130: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
    mul_302: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
    mul_303: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_289, 0.9)
    add_225: "f32[1152]" = torch.ops.aten.add.Tensor(mul_302, mul_303);  mul_302 = mul_303 = None
    squeeze_131: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
    mul_304: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0025575447570332);  squeeze_131 = None
    mul_305: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_304, 0.1);  mul_304 = None
    mul_306: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_290, 0.9)
    add_226: "f32[1152]" = torch.ops.aten.add.Tensor(mul_305, mul_306);  mul_305 = mul_306 = None
    unsqueeze_172: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1)
    unsqueeze_173: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_307: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_173);  mul_301 = unsqueeze_173 = None
    unsqueeze_174: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_88, -1);  primals_88 = None
    unsqueeze_175: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_227: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_307, unsqueeze_175);  mul_307 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_29: "f32[8, 1152, 7, 7]" = torch.ops.aten.relu.default(add_227);  add_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_44: "f32[8, 192, 7, 7]" = torch.ops.aten.convolution.default(relu_29, primals_149, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_228: "i64[]" = torch.ops.aten.add.Tensor(primals_291, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_44 = torch.ops.aten.var_mean.correction(convolution_44, [0, 2, 3], correction = 0, keepdim = True)
    getitem_88: "f32[1, 192, 1, 1]" = var_mean_44[0]
    getitem_89: "f32[1, 192, 1, 1]" = var_mean_44[1];  var_mean_44 = None
    add_229: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05)
    rsqrt_44: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
    sub_44: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_44, getitem_89)
    mul_308: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
    squeeze_132: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
    squeeze_133: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
    mul_309: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1)
    mul_310: "f32[192]" = torch.ops.aten.mul.Tensor(primals_292, 0.9)
    add_230: "f32[192]" = torch.ops.aten.add.Tensor(mul_309, mul_310);  mul_309 = mul_310 = None
    squeeze_134: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
    mul_311: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_134, 1.0025575447570332);  squeeze_134 = None
    mul_312: "f32[192]" = torch.ops.aten.mul.Tensor(mul_311, 0.1);  mul_311 = None
    mul_313: "f32[192]" = torch.ops.aten.mul.Tensor(primals_293, 0.9)
    add_231: "f32[192]" = torch.ops.aten.add.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None
    unsqueeze_176: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1)
    unsqueeze_177: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    mul_314: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(mul_308, unsqueeze_177);  mul_308 = unsqueeze_177 = None
    unsqueeze_178: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1);  primals_90 = None
    unsqueeze_179: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    add_232: "f32[8, 192, 7, 7]" = torch.ops.aten.add.Tensor(mul_314, unsqueeze_179);  mul_314 = unsqueeze_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_233: "f32[8, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_232, add_217);  add_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_45: "f32[8, 1152, 7, 7]" = torch.ops.aten.convolution.default(add_233, primals_150, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_234: "i64[]" = torch.ops.aten.add.Tensor(primals_294, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_45 = torch.ops.aten.var_mean.correction(convolution_45, [0, 2, 3], correction = 0, keepdim = True)
    getitem_90: "f32[1, 1152, 1, 1]" = var_mean_45[0]
    getitem_91: "f32[1, 1152, 1, 1]" = var_mean_45[1];  var_mean_45 = None
    add_235: "f32[1, 1152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05)
    rsqrt_45: "f32[1, 1152, 1, 1]" = torch.ops.aten.rsqrt.default(add_235);  add_235 = None
    sub_45: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_45, getitem_91)
    mul_315: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
    squeeze_135: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
    squeeze_136: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
    mul_316: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1)
    mul_317: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_295, 0.9)
    add_236: "f32[1152]" = torch.ops.aten.add.Tensor(mul_316, mul_317);  mul_316 = mul_317 = None
    squeeze_137: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
    mul_318: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_137, 1.0025575447570332);  squeeze_137 = None
    mul_319: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_318, 0.1);  mul_318 = None
    mul_320: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_296, 0.9)
    add_237: "f32[1152]" = torch.ops.aten.add.Tensor(mul_319, mul_320);  mul_319 = mul_320 = None
    unsqueeze_180: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_91, -1)
    unsqueeze_181: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_321: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_315, unsqueeze_181);  mul_315 = unsqueeze_181 = None
    unsqueeze_182: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1);  primals_92 = None
    unsqueeze_183: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_238: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_321, unsqueeze_183);  mul_321 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_30: "f32[8, 1152, 7, 7]" = torch.ops.aten.relu.default(add_238);  add_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_46: "f32[8, 1152, 7, 7]" = torch.ops.aten.convolution.default(relu_30, primals_151, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_239: "i64[]" = torch.ops.aten.add.Tensor(primals_297, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_46 = torch.ops.aten.var_mean.correction(convolution_46, [0, 2, 3], correction = 0, keepdim = True)
    getitem_92: "f32[1, 1152, 1, 1]" = var_mean_46[0]
    getitem_93: "f32[1, 1152, 1, 1]" = var_mean_46[1];  var_mean_46 = None
    add_240: "f32[1, 1152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05)
    rsqrt_46: "f32[1, 1152, 1, 1]" = torch.ops.aten.rsqrt.default(add_240);  add_240 = None
    sub_46: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_46, getitem_93)
    mul_322: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
    squeeze_138: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
    squeeze_139: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
    mul_323: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_138, 0.1)
    mul_324: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_298, 0.9)
    add_241: "f32[1152]" = torch.ops.aten.add.Tensor(mul_323, mul_324);  mul_323 = mul_324 = None
    squeeze_140: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_92, [0, 2, 3]);  getitem_92 = None
    mul_325: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_140, 1.0025575447570332);  squeeze_140 = None
    mul_326: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_325, 0.1);  mul_325 = None
    mul_327: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_299, 0.9)
    add_242: "f32[1152]" = torch.ops.aten.add.Tensor(mul_326, mul_327);  mul_326 = mul_327 = None
    unsqueeze_184: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_93, -1)
    unsqueeze_185: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    mul_328: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_322, unsqueeze_185);  mul_322 = unsqueeze_185 = None
    unsqueeze_186: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_94, -1);  primals_94 = None
    unsqueeze_187: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    add_243: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_328, unsqueeze_187);  mul_328 = unsqueeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_31: "f32[8, 1152, 7, 7]" = torch.ops.aten.relu.default(add_243);  add_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_47: "f32[8, 192, 7, 7]" = torch.ops.aten.convolution.default(relu_31, primals_152, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_244: "i64[]" = torch.ops.aten.add.Tensor(primals_300, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_47 = torch.ops.aten.var_mean.correction(convolution_47, [0, 2, 3], correction = 0, keepdim = True)
    getitem_94: "f32[1, 192, 1, 1]" = var_mean_47[0]
    getitem_95: "f32[1, 192, 1, 1]" = var_mean_47[1];  var_mean_47 = None
    add_245: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05)
    rsqrt_47: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_245);  add_245 = None
    sub_47: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_47, getitem_95)
    mul_329: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
    squeeze_141: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
    squeeze_142: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
    mul_330: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_141, 0.1)
    mul_331: "f32[192]" = torch.ops.aten.mul.Tensor(primals_301, 0.9)
    add_246: "f32[192]" = torch.ops.aten.add.Tensor(mul_330, mul_331);  mul_330 = mul_331 = None
    squeeze_143: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
    mul_332: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_143, 1.0025575447570332);  squeeze_143 = None
    mul_333: "f32[192]" = torch.ops.aten.mul.Tensor(mul_332, 0.1);  mul_332 = None
    mul_334: "f32[192]" = torch.ops.aten.mul.Tensor(primals_302, 0.9)
    add_247: "f32[192]" = torch.ops.aten.add.Tensor(mul_333, mul_334);  mul_333 = mul_334 = None
    unsqueeze_188: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1)
    unsqueeze_189: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_335: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(mul_329, unsqueeze_189);  mul_329 = unsqueeze_189 = None
    unsqueeze_190: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1);  primals_96 = None
    unsqueeze_191: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_248: "f32[8, 192, 7, 7]" = torch.ops.aten.add.Tensor(mul_335, unsqueeze_191);  mul_335 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_249: "f32[8, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_248, add_233);  add_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_48: "f32[8, 1152, 7, 7]" = torch.ops.aten.convolution.default(add_249, primals_153, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_250: "i64[]" = torch.ops.aten.add.Tensor(primals_303, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_48 = torch.ops.aten.var_mean.correction(convolution_48, [0, 2, 3], correction = 0, keepdim = True)
    getitem_96: "f32[1, 1152, 1, 1]" = var_mean_48[0]
    getitem_97: "f32[1, 1152, 1, 1]" = var_mean_48[1];  var_mean_48 = None
    add_251: "f32[1, 1152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05)
    rsqrt_48: "f32[1, 1152, 1, 1]" = torch.ops.aten.rsqrt.default(add_251);  add_251 = None
    sub_48: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_48, getitem_97)
    mul_336: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
    squeeze_144: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
    squeeze_145: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
    mul_337: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_144, 0.1)
    mul_338: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_304, 0.9)
    add_252: "f32[1152]" = torch.ops.aten.add.Tensor(mul_337, mul_338);  mul_337 = mul_338 = None
    squeeze_146: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
    mul_339: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_146, 1.0025575447570332);  squeeze_146 = None
    mul_340: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_339, 0.1);  mul_339 = None
    mul_341: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_305, 0.9)
    add_253: "f32[1152]" = torch.ops.aten.add.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
    unsqueeze_192: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_97, -1)
    unsqueeze_193: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    mul_342: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_336, unsqueeze_193);  mul_336 = unsqueeze_193 = None
    unsqueeze_194: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1);  primals_98 = None
    unsqueeze_195: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    add_254: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_342, unsqueeze_195);  mul_342 = unsqueeze_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_32: "f32[8, 1152, 7, 7]" = torch.ops.aten.relu.default(add_254);  add_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_49: "f32[8, 1152, 7, 7]" = torch.ops.aten.convolution.default(relu_32, primals_154, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1152)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_255: "i64[]" = torch.ops.aten.add.Tensor(primals_306, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_49 = torch.ops.aten.var_mean.correction(convolution_49, [0, 2, 3], correction = 0, keepdim = True)
    getitem_98: "f32[1, 1152, 1, 1]" = var_mean_49[0]
    getitem_99: "f32[1, 1152, 1, 1]" = var_mean_49[1];  var_mean_49 = None
    add_256: "f32[1, 1152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05)
    rsqrt_49: "f32[1, 1152, 1, 1]" = torch.ops.aten.rsqrt.default(add_256);  add_256 = None
    sub_49: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_49, getitem_99)
    mul_343: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
    squeeze_147: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
    squeeze_148: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
    mul_344: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_147, 0.1)
    mul_345: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_307, 0.9)
    add_257: "f32[1152]" = torch.ops.aten.add.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
    squeeze_149: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
    mul_346: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_149, 1.0025575447570332);  squeeze_149 = None
    mul_347: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_346, 0.1);  mul_346 = None
    mul_348: "f32[1152]" = torch.ops.aten.mul.Tensor(primals_308, 0.9)
    add_258: "f32[1152]" = torch.ops.aten.add.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
    unsqueeze_196: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1)
    unsqueeze_197: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_349: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_197);  mul_343 = unsqueeze_197 = None
    unsqueeze_198: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_100, -1);  primals_100 = None
    unsqueeze_199: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_259: "f32[8, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_349, unsqueeze_199);  mul_349 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_33: "f32[8, 1152, 7, 7]" = torch.ops.aten.relu.default(add_259);  add_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_50: "f32[8, 320, 7, 7]" = torch.ops.aten.convolution.default(relu_33, primals_155, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_260: "i64[]" = torch.ops.aten.add.Tensor(primals_309, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_50 = torch.ops.aten.var_mean.correction(convolution_50, [0, 2, 3], correction = 0, keepdim = True)
    getitem_100: "f32[1, 320, 1, 1]" = var_mean_50[0]
    getitem_101: "f32[1, 320, 1, 1]" = var_mean_50[1];  var_mean_50 = None
    add_261: "f32[1, 320, 1, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05)
    rsqrt_50: "f32[1, 320, 1, 1]" = torch.ops.aten.rsqrt.default(add_261);  add_261 = None
    sub_50: "f32[8, 320, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, getitem_101)
    mul_350: "f32[8, 320, 7, 7]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
    squeeze_150: "f32[320]" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
    squeeze_151: "f32[320]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
    mul_351: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_150, 0.1)
    mul_352: "f32[320]" = torch.ops.aten.mul.Tensor(primals_310, 0.9)
    add_262: "f32[320]" = torch.ops.aten.add.Tensor(mul_351, mul_352);  mul_351 = mul_352 = None
    squeeze_152: "f32[320]" = torch.ops.aten.squeeze.dims(getitem_100, [0, 2, 3]);  getitem_100 = None
    mul_353: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_152, 1.0025575447570332);  squeeze_152 = None
    mul_354: "f32[320]" = torch.ops.aten.mul.Tensor(mul_353, 0.1);  mul_353 = None
    mul_355: "f32[320]" = torch.ops.aten.mul.Tensor(primals_311, 0.9)
    add_263: "f32[320]" = torch.ops.aten.add.Tensor(mul_354, mul_355);  mul_354 = mul_355 = None
    unsqueeze_200: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_101, -1)
    unsqueeze_201: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    mul_356: "f32[8, 320, 7, 7]" = torch.ops.aten.mul.Tensor(mul_350, unsqueeze_201);  mul_350 = unsqueeze_201 = None
    unsqueeze_202: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1);  primals_102 = None
    unsqueeze_203: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    add_264: "f32[8, 320, 7, 7]" = torch.ops.aten.add.Tensor(mul_356, unsqueeze_203);  mul_356 = unsqueeze_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:168, code: x = self.conv_head(x)
    convolution_51: "f32[8, 1280, 7, 7]" = torch.ops.aten.convolution.default(add_264, primals_156, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_265: "i64[]" = torch.ops.aten.add.Tensor(primals_312, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_51 = torch.ops.aten.var_mean.correction(convolution_51, [0, 2, 3], correction = 0, keepdim = True)
    getitem_102: "f32[1, 1280, 1, 1]" = var_mean_51[0]
    getitem_103: "f32[1, 1280, 1, 1]" = var_mean_51[1];  var_mean_51 = None
    add_266: "f32[1, 1280, 1, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05)
    rsqrt_51: "f32[1, 1280, 1, 1]" = torch.ops.aten.rsqrt.default(add_266);  add_266 = None
    sub_51: "f32[8, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, getitem_103)
    mul_357: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
    squeeze_153: "f32[1280]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
    squeeze_154: "f32[1280]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
    mul_358: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_153, 0.1)
    mul_359: "f32[1280]" = torch.ops.aten.mul.Tensor(primals_313, 0.9)
    add_267: "f32[1280]" = torch.ops.aten.add.Tensor(mul_358, mul_359);  mul_358 = mul_359 = None
    squeeze_155: "f32[1280]" = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
    mul_360: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_155, 1.0025575447570332);  squeeze_155 = None
    mul_361: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_360, 0.1);  mul_360 = None
    mul_362: "f32[1280]" = torch.ops.aten.mul.Tensor(primals_314, 0.9)
    add_268: "f32[1280]" = torch.ops.aten.add.Tensor(mul_361, mul_362);  mul_361 = mul_362 = None
    unsqueeze_204: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_103, -1)
    unsqueeze_205: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_363: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_357, unsqueeze_205);  mul_357 = unsqueeze_205 = None
    unsqueeze_206: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_104, -1);  primals_104 = None
    unsqueeze_207: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_269: "f32[8, 1280, 7, 7]" = torch.ops.aten.add.Tensor(mul_363, unsqueeze_207);  mul_363 = unsqueeze_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_34: "f32[8, 1280, 7, 7]" = torch.ops.aten.relu.default(add_269);  add_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean: "f32[8, 1280, 1, 1]" = torch.ops.aten.mean.dim(relu_34, [-1, -2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view: "f32[8, 1280]" = torch.ops.aten.view.default(mean, [8, 1280]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:176, code: return x if pre_logits else self.classifier(x)
    permute: "f32[1280, 1000]" = torch.ops.aten.permute.default(primals_157, [1, 0]);  primals_157 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_158, view, permute);  primals_158 = None
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
    alias_36: "f32[8, 1280, 7, 7]" = torch.ops.aten.alias.default(relu_34);  relu_34 = None
    alias_37: "f32[8, 1280, 7, 7]" = torch.ops.aten.alias.default(alias_36);  alias_36 = None
    le: "b8[8, 1280, 7, 7]" = torch.ops.aten.le.Scalar(alias_37, 0);  alias_37 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where: "f32[8, 1280, 7, 7]" = torch.ops.aten.where.self(le, scalar_tensor, div);  le = scalar_tensor = div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_208: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
    unsqueeze_209: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, 2);  unsqueeze_208 = None
    unsqueeze_210: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 3);  unsqueeze_209 = None
    sum_2: "f32[1280]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_52: "f32[8, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_210)
    mul_364: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_52);  sub_52 = None
    sum_3: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_364, [0, 2, 3]);  mul_364 = None
    mul_365: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_2, 0.002551020408163265)
    unsqueeze_211: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_365, 0);  mul_365 = None
    unsqueeze_212: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 2);  unsqueeze_211 = None
    unsqueeze_213: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 3);  unsqueeze_212 = None
    mul_366: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_3, 0.002551020408163265)
    mul_367: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
    mul_368: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_366, mul_367);  mul_366 = mul_367 = None
    unsqueeze_214: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_368, 0);  mul_368 = None
    unsqueeze_215: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, 2);  unsqueeze_214 = None
    unsqueeze_216: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 3);  unsqueeze_215 = None
    mul_369: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_103);  primals_103 = None
    unsqueeze_217: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_369, 0);  mul_369 = None
    unsqueeze_218: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 2);  unsqueeze_217 = None
    unsqueeze_219: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 3);  unsqueeze_218 = None
    sub_53: "f32[8, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_210);  convolution_51 = unsqueeze_210 = None
    mul_370: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_216);  sub_53 = unsqueeze_216 = None
    sub_54: "f32[8, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(where, mul_370);  where = mul_370 = None
    sub_55: "f32[8, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(sub_54, unsqueeze_213);  sub_54 = unsqueeze_213 = None
    mul_371: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_219);  sub_55 = unsqueeze_219 = None
    mul_372: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_154);  sum_3 = squeeze_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:168, code: x = self.conv_head(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_371, add_264, primals_156, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_371 = add_264 = primals_156 = None
    getitem_104: "f32[8, 320, 7, 7]" = convolution_backward[0]
    getitem_105: "f32[1280, 320, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_220: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(squeeze_150, 0);  squeeze_150 = None
    unsqueeze_221: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, 2);  unsqueeze_220 = None
    unsqueeze_222: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 3);  unsqueeze_221 = None
    sum_4: "f32[320]" = torch.ops.aten.sum.dim_IntList(getitem_104, [0, 2, 3])
    sub_56: "f32[8, 320, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_222)
    mul_373: "f32[8, 320, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_104, sub_56);  sub_56 = None
    sum_5: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_373, [0, 2, 3]);  mul_373 = None
    mul_374: "f32[320]" = torch.ops.aten.mul.Tensor(sum_4, 0.002551020408163265)
    unsqueeze_223: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_374, 0);  mul_374 = None
    unsqueeze_224: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 2);  unsqueeze_223 = None
    unsqueeze_225: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 3);  unsqueeze_224 = None
    mul_375: "f32[320]" = torch.ops.aten.mul.Tensor(sum_5, 0.002551020408163265)
    mul_376: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
    mul_377: "f32[320]" = torch.ops.aten.mul.Tensor(mul_375, mul_376);  mul_375 = mul_376 = None
    unsqueeze_226: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_377, 0);  mul_377 = None
    unsqueeze_227: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 2);  unsqueeze_226 = None
    unsqueeze_228: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 3);  unsqueeze_227 = None
    mul_378: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_101);  primals_101 = None
    unsqueeze_229: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_378, 0);  mul_378 = None
    unsqueeze_230: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 2);  unsqueeze_229 = None
    unsqueeze_231: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 3);  unsqueeze_230 = None
    sub_57: "f32[8, 320, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_222);  convolution_50 = unsqueeze_222 = None
    mul_379: "f32[8, 320, 7, 7]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_228);  sub_57 = unsqueeze_228 = None
    sub_58: "f32[8, 320, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_104, mul_379);  getitem_104 = mul_379 = None
    sub_59: "f32[8, 320, 7, 7]" = torch.ops.aten.sub.Tensor(sub_58, unsqueeze_225);  sub_58 = unsqueeze_225 = None
    mul_380: "f32[8, 320, 7, 7]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_231);  sub_59 = unsqueeze_231 = None
    mul_381: "f32[320]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_151);  sum_5 = squeeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_380, relu_33, primals_155, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_380 = primals_155 = None
    getitem_107: "f32[8, 1152, 7, 7]" = convolution_backward_1[0]
    getitem_108: "f32[320, 1152, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_39: "f32[8, 1152, 7, 7]" = torch.ops.aten.alias.default(relu_33);  relu_33 = None
    alias_40: "f32[8, 1152, 7, 7]" = torch.ops.aten.alias.default(alias_39);  alias_39 = None
    le_1: "b8[8, 1152, 7, 7]" = torch.ops.aten.le.Scalar(alias_40, 0);  alias_40 = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_1: "f32[8, 1152, 7, 7]" = torch.ops.aten.where.self(le_1, scalar_tensor_1, getitem_107);  le_1 = scalar_tensor_1 = getitem_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_232: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_147, 0);  squeeze_147 = None
    unsqueeze_233: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 2);  unsqueeze_232 = None
    unsqueeze_234: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 3);  unsqueeze_233 = None
    sum_6: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_60: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_234)
    mul_382: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, sub_60);  sub_60 = None
    sum_7: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_382, [0, 2, 3]);  mul_382 = None
    mul_383: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_6, 0.002551020408163265)
    unsqueeze_235: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_383, 0);  mul_383 = None
    unsqueeze_236: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 2);  unsqueeze_235 = None
    unsqueeze_237: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 3);  unsqueeze_236 = None
    mul_384: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_7, 0.002551020408163265)
    mul_385: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
    mul_386: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_384, mul_385);  mul_384 = mul_385 = None
    unsqueeze_238: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_386, 0);  mul_386 = None
    unsqueeze_239: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 2);  unsqueeze_238 = None
    unsqueeze_240: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 3);  unsqueeze_239 = None
    mul_387: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_99);  primals_99 = None
    unsqueeze_241: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_387, 0);  mul_387 = None
    unsqueeze_242: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 2);  unsqueeze_241 = None
    unsqueeze_243: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 3);  unsqueeze_242 = None
    sub_61: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_234);  convolution_49 = unsqueeze_234 = None
    mul_388: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_240);  sub_61 = unsqueeze_240 = None
    sub_62: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(where_1, mul_388);  where_1 = mul_388 = None
    sub_63: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_62, unsqueeze_237);  sub_62 = unsqueeze_237 = None
    mul_389: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_243);  sub_63 = unsqueeze_243 = None
    mul_390: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_148);  sum_7 = squeeze_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_389, relu_32, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_389 = primals_154 = None
    getitem_110: "f32[8, 1152, 7, 7]" = convolution_backward_2[0]
    getitem_111: "f32[1152, 1, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_42: "f32[8, 1152, 7, 7]" = torch.ops.aten.alias.default(relu_32);  relu_32 = None
    alias_43: "f32[8, 1152, 7, 7]" = torch.ops.aten.alias.default(alias_42);  alias_42 = None
    le_2: "b8[8, 1152, 7, 7]" = torch.ops.aten.le.Scalar(alias_43, 0);  alias_43 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_2: "f32[8, 1152, 7, 7]" = torch.ops.aten.where.self(le_2, scalar_tensor_2, getitem_110);  le_2 = scalar_tensor_2 = getitem_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_244: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
    unsqueeze_245: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 2);  unsqueeze_244 = None
    unsqueeze_246: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 3);  unsqueeze_245 = None
    sum_8: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_64: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_246)
    mul_391: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_64);  sub_64 = None
    sum_9: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_391, [0, 2, 3]);  mul_391 = None
    mul_392: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_8, 0.002551020408163265)
    unsqueeze_247: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_392, 0);  mul_392 = None
    unsqueeze_248: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 2);  unsqueeze_247 = None
    unsqueeze_249: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 3);  unsqueeze_248 = None
    mul_393: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_9, 0.002551020408163265)
    mul_394: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
    mul_395: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_393, mul_394);  mul_393 = mul_394 = None
    unsqueeze_250: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_395, 0);  mul_395 = None
    unsqueeze_251: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 2);  unsqueeze_250 = None
    unsqueeze_252: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 3);  unsqueeze_251 = None
    mul_396: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_97);  primals_97 = None
    unsqueeze_253: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_396, 0);  mul_396 = None
    unsqueeze_254: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None
    unsqueeze_255: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 3);  unsqueeze_254 = None
    sub_65: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_246);  convolution_48 = unsqueeze_246 = None
    mul_397: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_252);  sub_65 = unsqueeze_252 = None
    sub_66: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(where_2, mul_397);  where_2 = mul_397 = None
    sub_67: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_66, unsqueeze_249);  sub_66 = unsqueeze_249 = None
    mul_398: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_255);  sub_67 = unsqueeze_255 = None
    mul_399: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_145);  sum_9 = squeeze_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_398, add_249, primals_153, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_398 = add_249 = primals_153 = None
    getitem_113: "f32[8, 192, 7, 7]" = convolution_backward_3[0]
    getitem_114: "f32[1152, 192, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_256: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
    unsqueeze_257: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 2);  unsqueeze_256 = None
    unsqueeze_258: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 3);  unsqueeze_257 = None
    sum_10: "f32[192]" = torch.ops.aten.sum.dim_IntList(getitem_113, [0, 2, 3])
    sub_68: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_258)
    mul_400: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_113, sub_68);  sub_68 = None
    sum_11: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_400, [0, 2, 3]);  mul_400 = None
    mul_401: "f32[192]" = torch.ops.aten.mul.Tensor(sum_10, 0.002551020408163265)
    unsqueeze_259: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_401, 0);  mul_401 = None
    unsqueeze_260: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 2);  unsqueeze_259 = None
    unsqueeze_261: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 3);  unsqueeze_260 = None
    mul_402: "f32[192]" = torch.ops.aten.mul.Tensor(sum_11, 0.002551020408163265)
    mul_403: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
    mul_404: "f32[192]" = torch.ops.aten.mul.Tensor(mul_402, mul_403);  mul_402 = mul_403 = None
    unsqueeze_262: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_404, 0);  mul_404 = None
    unsqueeze_263: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 2);  unsqueeze_262 = None
    unsqueeze_264: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 3);  unsqueeze_263 = None
    mul_405: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_95);  primals_95 = None
    unsqueeze_265: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_405, 0);  mul_405 = None
    unsqueeze_266: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
    unsqueeze_267: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 3);  unsqueeze_266 = None
    sub_69: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_258);  convolution_47 = unsqueeze_258 = None
    mul_406: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_264);  sub_69 = unsqueeze_264 = None
    sub_70: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_113, mul_406);  mul_406 = None
    sub_71: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(sub_70, unsqueeze_261);  sub_70 = unsqueeze_261 = None
    mul_407: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_267);  sub_71 = unsqueeze_267 = None
    mul_408: "f32[192]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_142);  sum_11 = squeeze_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_407, relu_31, primals_152, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_407 = primals_152 = None
    getitem_116: "f32[8, 1152, 7, 7]" = convolution_backward_4[0]
    getitem_117: "f32[192, 1152, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_45: "f32[8, 1152, 7, 7]" = torch.ops.aten.alias.default(relu_31);  relu_31 = None
    alias_46: "f32[8, 1152, 7, 7]" = torch.ops.aten.alias.default(alias_45);  alias_45 = None
    le_3: "b8[8, 1152, 7, 7]" = torch.ops.aten.le.Scalar(alias_46, 0);  alias_46 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_3: "f32[8, 1152, 7, 7]" = torch.ops.aten.where.self(le_3, scalar_tensor_3, getitem_116);  le_3 = scalar_tensor_3 = getitem_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_268: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
    unsqueeze_269: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 2);  unsqueeze_268 = None
    unsqueeze_270: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 3);  unsqueeze_269 = None
    sum_12: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_72: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_270)
    mul_409: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_72);  sub_72 = None
    sum_13: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_409, [0, 2, 3]);  mul_409 = None
    mul_410: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_12, 0.002551020408163265)
    unsqueeze_271: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_410, 0);  mul_410 = None
    unsqueeze_272: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 2);  unsqueeze_271 = None
    unsqueeze_273: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 3);  unsqueeze_272 = None
    mul_411: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_13, 0.002551020408163265)
    mul_412: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
    mul_413: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_411, mul_412);  mul_411 = mul_412 = None
    unsqueeze_274: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_413, 0);  mul_413 = None
    unsqueeze_275: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 2);  unsqueeze_274 = None
    unsqueeze_276: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 3);  unsqueeze_275 = None
    mul_414: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_93);  primals_93 = None
    unsqueeze_277: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_414, 0);  mul_414 = None
    unsqueeze_278: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None
    unsqueeze_279: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 3);  unsqueeze_278 = None
    sub_73: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_270);  convolution_46 = unsqueeze_270 = None
    mul_415: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_276);  sub_73 = unsqueeze_276 = None
    sub_74: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(where_3, mul_415);  where_3 = mul_415 = None
    sub_75: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_74, unsqueeze_273);  sub_74 = unsqueeze_273 = None
    mul_416: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_279);  sub_75 = unsqueeze_279 = None
    mul_417: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_139);  sum_13 = squeeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_416, relu_30, primals_151, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_416 = primals_151 = None
    getitem_119: "f32[8, 1152, 7, 7]" = convolution_backward_5[0]
    getitem_120: "f32[1152, 1, 5, 5]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_48: "f32[8, 1152, 7, 7]" = torch.ops.aten.alias.default(relu_30);  relu_30 = None
    alias_49: "f32[8, 1152, 7, 7]" = torch.ops.aten.alias.default(alias_48);  alias_48 = None
    le_4: "b8[8, 1152, 7, 7]" = torch.ops.aten.le.Scalar(alias_49, 0);  alias_49 = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_4: "f32[8, 1152, 7, 7]" = torch.ops.aten.where.self(le_4, scalar_tensor_4, getitem_119);  le_4 = scalar_tensor_4 = getitem_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_280: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
    unsqueeze_281: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 2);  unsqueeze_280 = None
    unsqueeze_282: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 3);  unsqueeze_281 = None
    sum_14: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_76: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_282)
    mul_418: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, sub_76);  sub_76 = None
    sum_15: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_418, [0, 2, 3]);  mul_418 = None
    mul_419: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_14, 0.002551020408163265)
    unsqueeze_283: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_419, 0);  mul_419 = None
    unsqueeze_284: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 2);  unsqueeze_283 = None
    unsqueeze_285: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 3);  unsqueeze_284 = None
    mul_420: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_15, 0.002551020408163265)
    mul_421: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
    mul_422: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_420, mul_421);  mul_420 = mul_421 = None
    unsqueeze_286: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_422, 0);  mul_422 = None
    unsqueeze_287: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 2);  unsqueeze_286 = None
    unsqueeze_288: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 3);  unsqueeze_287 = None
    mul_423: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_91);  primals_91 = None
    unsqueeze_289: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_423, 0);  mul_423 = None
    unsqueeze_290: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 2);  unsqueeze_289 = None
    unsqueeze_291: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 3);  unsqueeze_290 = None
    sub_77: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_282);  convolution_45 = unsqueeze_282 = None
    mul_424: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_288);  sub_77 = unsqueeze_288 = None
    sub_78: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(where_4, mul_424);  where_4 = mul_424 = None
    sub_79: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_78, unsqueeze_285);  sub_78 = unsqueeze_285 = None
    mul_425: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_291);  sub_79 = unsqueeze_291 = None
    mul_426: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_136);  sum_15 = squeeze_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_425, add_233, primals_150, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_425 = add_233 = primals_150 = None
    getitem_122: "f32[8, 192, 7, 7]" = convolution_backward_6[0]
    getitem_123: "f32[1152, 192, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_270: "f32[8, 192, 7, 7]" = torch.ops.aten.add.Tensor(getitem_113, getitem_122);  getitem_113 = getitem_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_292: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
    unsqueeze_293: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 2);  unsqueeze_292 = None
    unsqueeze_294: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 3);  unsqueeze_293 = None
    sum_16: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_270, [0, 2, 3])
    sub_80: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_294)
    mul_427: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_270, sub_80);  sub_80 = None
    sum_17: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_427, [0, 2, 3]);  mul_427 = None
    mul_428: "f32[192]" = torch.ops.aten.mul.Tensor(sum_16, 0.002551020408163265)
    unsqueeze_295: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_428, 0);  mul_428 = None
    unsqueeze_296: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 2);  unsqueeze_295 = None
    unsqueeze_297: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 3);  unsqueeze_296 = None
    mul_429: "f32[192]" = torch.ops.aten.mul.Tensor(sum_17, 0.002551020408163265)
    mul_430: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
    mul_431: "f32[192]" = torch.ops.aten.mul.Tensor(mul_429, mul_430);  mul_429 = mul_430 = None
    unsqueeze_298: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_431, 0);  mul_431 = None
    unsqueeze_299: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 2);  unsqueeze_298 = None
    unsqueeze_300: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 3);  unsqueeze_299 = None
    mul_432: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_89);  primals_89 = None
    unsqueeze_301: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_432, 0);  mul_432 = None
    unsqueeze_302: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None
    unsqueeze_303: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 3);  unsqueeze_302 = None
    sub_81: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_294);  convolution_44 = unsqueeze_294 = None
    mul_433: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_300);  sub_81 = unsqueeze_300 = None
    sub_82: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(add_270, mul_433);  mul_433 = None
    sub_83: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(sub_82, unsqueeze_297);  sub_82 = unsqueeze_297 = None
    mul_434: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_303);  sub_83 = unsqueeze_303 = None
    mul_435: "f32[192]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_133);  sum_17 = squeeze_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_434, relu_29, primals_149, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_434 = primals_149 = None
    getitem_125: "f32[8, 1152, 7, 7]" = convolution_backward_7[0]
    getitem_126: "f32[192, 1152, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_51: "f32[8, 1152, 7, 7]" = torch.ops.aten.alias.default(relu_29);  relu_29 = None
    alias_52: "f32[8, 1152, 7, 7]" = torch.ops.aten.alias.default(alias_51);  alias_51 = None
    le_5: "b8[8, 1152, 7, 7]" = torch.ops.aten.le.Scalar(alias_52, 0);  alias_52 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_5: "f32[8, 1152, 7, 7]" = torch.ops.aten.where.self(le_5, scalar_tensor_5, getitem_125);  le_5 = scalar_tensor_5 = getitem_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_304: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
    unsqueeze_305: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 2);  unsqueeze_304 = None
    unsqueeze_306: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 3);  unsqueeze_305 = None
    sum_18: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_84: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_306)
    mul_436: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, sub_84);  sub_84 = None
    sum_19: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_436, [0, 2, 3]);  mul_436 = None
    mul_437: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_18, 0.002551020408163265)
    unsqueeze_307: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_437, 0);  mul_437 = None
    unsqueeze_308: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 2);  unsqueeze_307 = None
    unsqueeze_309: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 3);  unsqueeze_308 = None
    mul_438: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_19, 0.002551020408163265)
    mul_439: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
    mul_440: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_438, mul_439);  mul_438 = mul_439 = None
    unsqueeze_310: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_440, 0);  mul_440 = None
    unsqueeze_311: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 2);  unsqueeze_310 = None
    unsqueeze_312: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 3);  unsqueeze_311 = None
    mul_441: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_87);  primals_87 = None
    unsqueeze_313: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_441, 0);  mul_441 = None
    unsqueeze_314: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 2);  unsqueeze_313 = None
    unsqueeze_315: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 3);  unsqueeze_314 = None
    sub_85: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_306);  convolution_43 = unsqueeze_306 = None
    mul_442: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_312);  sub_85 = unsqueeze_312 = None
    sub_86: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(where_5, mul_442);  where_5 = mul_442 = None
    sub_87: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_86, unsqueeze_309);  sub_86 = unsqueeze_309 = None
    mul_443: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_315);  sub_87 = unsqueeze_315 = None
    mul_444: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_130);  sum_19 = squeeze_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_443, relu_28, primals_148, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_443 = primals_148 = None
    getitem_128: "f32[8, 1152, 7, 7]" = convolution_backward_8[0]
    getitem_129: "f32[1152, 1, 5, 5]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_54: "f32[8, 1152, 7, 7]" = torch.ops.aten.alias.default(relu_28);  relu_28 = None
    alias_55: "f32[8, 1152, 7, 7]" = torch.ops.aten.alias.default(alias_54);  alias_54 = None
    le_6: "b8[8, 1152, 7, 7]" = torch.ops.aten.le.Scalar(alias_55, 0);  alias_55 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_6: "f32[8, 1152, 7, 7]" = torch.ops.aten.where.self(le_6, scalar_tensor_6, getitem_128);  le_6 = scalar_tensor_6 = getitem_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_316: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
    unsqueeze_317: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, 2);  unsqueeze_316 = None
    unsqueeze_318: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 3);  unsqueeze_317 = None
    sum_20: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_88: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_318)
    mul_445: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, sub_88);  sub_88 = None
    sum_21: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_445, [0, 2, 3]);  mul_445 = None
    mul_446: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_20, 0.002551020408163265)
    unsqueeze_319: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_446, 0);  mul_446 = None
    unsqueeze_320: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 2);  unsqueeze_319 = None
    unsqueeze_321: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 3);  unsqueeze_320 = None
    mul_447: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_21, 0.002551020408163265)
    mul_448: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
    mul_449: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_447, mul_448);  mul_447 = mul_448 = None
    unsqueeze_322: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_449, 0);  mul_449 = None
    unsqueeze_323: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 2);  unsqueeze_322 = None
    unsqueeze_324: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 3);  unsqueeze_323 = None
    mul_450: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_85);  primals_85 = None
    unsqueeze_325: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_450, 0);  mul_450 = None
    unsqueeze_326: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None
    unsqueeze_327: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 3);  unsqueeze_326 = None
    sub_89: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_318);  convolution_42 = unsqueeze_318 = None
    mul_451: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_324);  sub_89 = unsqueeze_324 = None
    sub_90: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(where_6, mul_451);  where_6 = mul_451 = None
    sub_91: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_90, unsqueeze_321);  sub_90 = unsqueeze_321 = None
    mul_452: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_327);  sub_91 = unsqueeze_327 = None
    mul_453: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_127);  sum_21 = squeeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_452, add_217, primals_147, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_452 = add_217 = primals_147 = None
    getitem_131: "f32[8, 192, 7, 7]" = convolution_backward_9[0]
    getitem_132: "f32[1152, 192, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_271: "f32[8, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_270, getitem_131);  add_270 = getitem_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_328: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
    unsqueeze_329: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 2);  unsqueeze_328 = None
    unsqueeze_330: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 3);  unsqueeze_329 = None
    sum_22: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_271, [0, 2, 3])
    sub_92: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_330)
    mul_454: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_271, sub_92);  sub_92 = None
    sum_23: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_454, [0, 2, 3]);  mul_454 = None
    mul_455: "f32[192]" = torch.ops.aten.mul.Tensor(sum_22, 0.002551020408163265)
    unsqueeze_331: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_455, 0);  mul_455 = None
    unsqueeze_332: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 2);  unsqueeze_331 = None
    unsqueeze_333: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 3);  unsqueeze_332 = None
    mul_456: "f32[192]" = torch.ops.aten.mul.Tensor(sum_23, 0.002551020408163265)
    mul_457: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
    mul_458: "f32[192]" = torch.ops.aten.mul.Tensor(mul_456, mul_457);  mul_456 = mul_457 = None
    unsqueeze_334: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_458, 0);  mul_458 = None
    unsqueeze_335: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 2);  unsqueeze_334 = None
    unsqueeze_336: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 3);  unsqueeze_335 = None
    mul_459: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_83);  primals_83 = None
    unsqueeze_337: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_459, 0);  mul_459 = None
    unsqueeze_338: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 2);  unsqueeze_337 = None
    unsqueeze_339: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 3);  unsqueeze_338 = None
    sub_93: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_330);  convolution_41 = unsqueeze_330 = None
    mul_460: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_336);  sub_93 = unsqueeze_336 = None
    sub_94: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(add_271, mul_460);  mul_460 = None
    sub_95: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(sub_94, unsqueeze_333);  sub_94 = unsqueeze_333 = None
    mul_461: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_339);  sub_95 = unsqueeze_339 = None
    mul_462: "f32[192]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_124);  sum_23 = squeeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_461, relu_27, primals_146, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_461 = primals_146 = None
    getitem_134: "f32[8, 1152, 7, 7]" = convolution_backward_10[0]
    getitem_135: "f32[192, 1152, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_57: "f32[8, 1152, 7, 7]" = torch.ops.aten.alias.default(relu_27);  relu_27 = None
    alias_58: "f32[8, 1152, 7, 7]" = torch.ops.aten.alias.default(alias_57);  alias_57 = None
    le_7: "b8[8, 1152, 7, 7]" = torch.ops.aten.le.Scalar(alias_58, 0);  alias_58 = None
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_7: "f32[8, 1152, 7, 7]" = torch.ops.aten.where.self(le_7, scalar_tensor_7, getitem_134);  le_7 = scalar_tensor_7 = getitem_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_340: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
    unsqueeze_341: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 2);  unsqueeze_340 = None
    unsqueeze_342: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 3);  unsqueeze_341 = None
    sum_24: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_96: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_342)
    mul_463: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, sub_96);  sub_96 = None
    sum_25: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_463, [0, 2, 3]);  mul_463 = None
    mul_464: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_24, 0.002551020408163265)
    unsqueeze_343: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_464, 0);  mul_464 = None
    unsqueeze_344: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 2);  unsqueeze_343 = None
    unsqueeze_345: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 3);  unsqueeze_344 = None
    mul_465: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_25, 0.002551020408163265)
    mul_466: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_467: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_465, mul_466);  mul_465 = mul_466 = None
    unsqueeze_346: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_467, 0);  mul_467 = None
    unsqueeze_347: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 2);  unsqueeze_346 = None
    unsqueeze_348: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 3);  unsqueeze_347 = None
    mul_468: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_81);  primals_81 = None
    unsqueeze_349: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_468, 0);  mul_468 = None
    unsqueeze_350: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None
    unsqueeze_351: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 3);  unsqueeze_350 = None
    sub_97: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_342);  convolution_40 = unsqueeze_342 = None
    mul_469: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_348);  sub_97 = unsqueeze_348 = None
    sub_98: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(where_7, mul_469);  where_7 = mul_469 = None
    sub_99: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_98, unsqueeze_345);  sub_98 = unsqueeze_345 = None
    mul_470: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_351);  sub_99 = unsqueeze_351 = None
    mul_471: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_121);  sum_25 = squeeze_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_470, relu_26, primals_145, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_470 = primals_145 = None
    getitem_137: "f32[8, 1152, 7, 7]" = convolution_backward_11[0]
    getitem_138: "f32[1152, 1, 5, 5]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_60: "f32[8, 1152, 7, 7]" = torch.ops.aten.alias.default(relu_26);  relu_26 = None
    alias_61: "f32[8, 1152, 7, 7]" = torch.ops.aten.alias.default(alias_60);  alias_60 = None
    le_8: "b8[8, 1152, 7, 7]" = torch.ops.aten.le.Scalar(alias_61, 0);  alias_61 = None
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_8: "f32[8, 1152, 7, 7]" = torch.ops.aten.where.self(le_8, scalar_tensor_8, getitem_137);  le_8 = scalar_tensor_8 = getitem_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_352: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
    unsqueeze_353: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 2);  unsqueeze_352 = None
    unsqueeze_354: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 3);  unsqueeze_353 = None
    sum_26: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_100: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_354)
    mul_472: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_8, sub_100);  sub_100 = None
    sum_27: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_472, [0, 2, 3]);  mul_472 = None
    mul_473: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_26, 0.002551020408163265)
    unsqueeze_355: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_473, 0);  mul_473 = None
    unsqueeze_356: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 2);  unsqueeze_355 = None
    unsqueeze_357: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 3);  unsqueeze_356 = None
    mul_474: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_27, 0.002551020408163265)
    mul_475: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_476: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_474, mul_475);  mul_474 = mul_475 = None
    unsqueeze_358: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_476, 0);  mul_476 = None
    unsqueeze_359: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 2);  unsqueeze_358 = None
    unsqueeze_360: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 3);  unsqueeze_359 = None
    mul_477: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_79);  primals_79 = None
    unsqueeze_361: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_477, 0);  mul_477 = None
    unsqueeze_362: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
    unsqueeze_363: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 3);  unsqueeze_362 = None
    sub_101: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_354);  convolution_39 = unsqueeze_354 = None
    mul_478: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_360);  sub_101 = unsqueeze_360 = None
    sub_102: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(where_8, mul_478);  where_8 = mul_478 = None
    sub_103: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_102, unsqueeze_357);  sub_102 = unsqueeze_357 = None
    mul_479: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_363);  sub_103 = unsqueeze_363 = None
    mul_480: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_118);  sum_27 = squeeze_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_479, add_201, primals_144, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_479 = add_201 = primals_144 = None
    getitem_140: "f32[8, 192, 7, 7]" = convolution_backward_12[0]
    getitem_141: "f32[1152, 192, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_272: "f32[8, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_271, getitem_140);  add_271 = getitem_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_364: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
    unsqueeze_365: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 2);  unsqueeze_364 = None
    unsqueeze_366: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 3);  unsqueeze_365 = None
    sum_28: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_272, [0, 2, 3])
    sub_104: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_366)
    mul_481: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_272, sub_104);  sub_104 = None
    sum_29: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_481, [0, 2, 3]);  mul_481 = None
    mul_482: "f32[192]" = torch.ops.aten.mul.Tensor(sum_28, 0.002551020408163265)
    unsqueeze_367: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_482, 0);  mul_482 = None
    unsqueeze_368: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
    unsqueeze_369: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 3);  unsqueeze_368 = None
    mul_483: "f32[192]" = torch.ops.aten.mul.Tensor(sum_29, 0.002551020408163265)
    mul_484: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_485: "f32[192]" = torch.ops.aten.mul.Tensor(mul_483, mul_484);  mul_483 = mul_484 = None
    unsqueeze_370: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_485, 0);  mul_485 = None
    unsqueeze_371: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 2);  unsqueeze_370 = None
    unsqueeze_372: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 3);  unsqueeze_371 = None
    mul_486: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_77);  primals_77 = None
    unsqueeze_373: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_486, 0);  mul_486 = None
    unsqueeze_374: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
    unsqueeze_375: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 3);  unsqueeze_374 = None
    sub_105: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_366);  convolution_38 = unsqueeze_366 = None
    mul_487: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_372);  sub_105 = unsqueeze_372 = None
    sub_106: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(add_272, mul_487);  add_272 = mul_487 = None
    sub_107: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(sub_106, unsqueeze_369);  sub_106 = unsqueeze_369 = None
    mul_488: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_375);  sub_107 = unsqueeze_375 = None
    mul_489: "f32[192]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_115);  sum_29 = squeeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_488, relu_25, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_488 = primals_143 = None
    getitem_143: "f32[8, 576, 7, 7]" = convolution_backward_13[0]
    getitem_144: "f32[192, 576, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_63: "f32[8, 576, 7, 7]" = torch.ops.aten.alias.default(relu_25);  relu_25 = None
    alias_64: "f32[8, 576, 7, 7]" = torch.ops.aten.alias.default(alias_63);  alias_63 = None
    le_9: "b8[8, 576, 7, 7]" = torch.ops.aten.le.Scalar(alias_64, 0);  alias_64 = None
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_9: "f32[8, 576, 7, 7]" = torch.ops.aten.where.self(le_9, scalar_tensor_9, getitem_143);  le_9 = scalar_tensor_9 = getitem_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_376: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
    unsqueeze_377: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 2);  unsqueeze_376 = None
    unsqueeze_378: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 3);  unsqueeze_377 = None
    sum_30: "f32[576]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_108: "f32[8, 576, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_378)
    mul_490: "f32[8, 576, 7, 7]" = torch.ops.aten.mul.Tensor(where_9, sub_108);  sub_108 = None
    sum_31: "f32[576]" = torch.ops.aten.sum.dim_IntList(mul_490, [0, 2, 3]);  mul_490 = None
    mul_491: "f32[576]" = torch.ops.aten.mul.Tensor(sum_30, 0.002551020408163265)
    unsqueeze_379: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_491, 0);  mul_491 = None
    unsqueeze_380: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
    unsqueeze_381: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 3);  unsqueeze_380 = None
    mul_492: "f32[576]" = torch.ops.aten.mul.Tensor(sum_31, 0.002551020408163265)
    mul_493: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_494: "f32[576]" = torch.ops.aten.mul.Tensor(mul_492, mul_493);  mul_492 = mul_493 = None
    unsqueeze_382: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_494, 0);  mul_494 = None
    unsqueeze_383: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 2);  unsqueeze_382 = None
    unsqueeze_384: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 3);  unsqueeze_383 = None
    mul_495: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_75);  primals_75 = None
    unsqueeze_385: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_495, 0);  mul_495 = None
    unsqueeze_386: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
    unsqueeze_387: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
    sub_109: "f32[8, 576, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_378);  convolution_37 = unsqueeze_378 = None
    mul_496: "f32[8, 576, 7, 7]" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_384);  sub_109 = unsqueeze_384 = None
    sub_110: "f32[8, 576, 7, 7]" = torch.ops.aten.sub.Tensor(where_9, mul_496);  where_9 = mul_496 = None
    sub_111: "f32[8, 576, 7, 7]" = torch.ops.aten.sub.Tensor(sub_110, unsqueeze_381);  sub_110 = unsqueeze_381 = None
    mul_497: "f32[8, 576, 7, 7]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_387);  sub_111 = unsqueeze_387 = None
    mul_498: "f32[576]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_112);  sum_31 = squeeze_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_497, relu_24, primals_142, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 576, [True, True, False]);  mul_497 = primals_142 = None
    getitem_146: "f32[8, 576, 14, 14]" = convolution_backward_14[0]
    getitem_147: "f32[576, 1, 5, 5]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_66: "f32[8, 576, 14, 14]" = torch.ops.aten.alias.default(relu_24);  relu_24 = None
    alias_67: "f32[8, 576, 14, 14]" = torch.ops.aten.alias.default(alias_66);  alias_66 = None
    le_10: "b8[8, 576, 14, 14]" = torch.ops.aten.le.Scalar(alias_67, 0);  alias_67 = None
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_10: "f32[8, 576, 14, 14]" = torch.ops.aten.where.self(le_10, scalar_tensor_10, getitem_146);  le_10 = scalar_tensor_10 = getitem_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_388: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
    unsqueeze_389: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 2);  unsqueeze_388 = None
    unsqueeze_390: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 3);  unsqueeze_389 = None
    sum_32: "f32[576]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_112: "f32[8, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_390)
    mul_499: "f32[8, 576, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, sub_112);  sub_112 = None
    sum_33: "f32[576]" = torch.ops.aten.sum.dim_IntList(mul_499, [0, 2, 3]);  mul_499 = None
    mul_500: "f32[576]" = torch.ops.aten.mul.Tensor(sum_32, 0.0006377551020408163)
    unsqueeze_391: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_500, 0);  mul_500 = None
    unsqueeze_392: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
    unsqueeze_393: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 3);  unsqueeze_392 = None
    mul_501: "f32[576]" = torch.ops.aten.mul.Tensor(sum_33, 0.0006377551020408163)
    mul_502: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_503: "f32[576]" = torch.ops.aten.mul.Tensor(mul_501, mul_502);  mul_501 = mul_502 = None
    unsqueeze_394: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_503, 0);  mul_503 = None
    unsqueeze_395: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 2);  unsqueeze_394 = None
    unsqueeze_396: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 3);  unsqueeze_395 = None
    mul_504: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_73);  primals_73 = None
    unsqueeze_397: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_504, 0);  mul_504 = None
    unsqueeze_398: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
    unsqueeze_399: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 3);  unsqueeze_398 = None
    sub_113: "f32[8, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_390);  convolution_36 = unsqueeze_390 = None
    mul_505: "f32[8, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_396);  sub_113 = unsqueeze_396 = None
    sub_114: "f32[8, 576, 14, 14]" = torch.ops.aten.sub.Tensor(where_10, mul_505);  where_10 = mul_505 = None
    sub_115: "f32[8, 576, 14, 14]" = torch.ops.aten.sub.Tensor(sub_114, unsqueeze_393);  sub_114 = unsqueeze_393 = None
    mul_506: "f32[8, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_399);  sub_115 = unsqueeze_399 = None
    mul_507: "f32[576]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_109);  sum_33 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_506, add_186, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_506 = add_186 = primals_141 = None
    getitem_149: "f32[8, 96, 14, 14]" = convolution_backward_15[0]
    getitem_150: "f32[576, 96, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_400: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
    unsqueeze_401: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 2);  unsqueeze_400 = None
    unsqueeze_402: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 3);  unsqueeze_401 = None
    sum_34: "f32[96]" = torch.ops.aten.sum.dim_IntList(getitem_149, [0, 2, 3])
    sub_116: "f32[8, 96, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_402)
    mul_508: "f32[8, 96, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_149, sub_116);  sub_116 = None
    sum_35: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_508, [0, 2, 3]);  mul_508 = None
    mul_509: "f32[96]" = torch.ops.aten.mul.Tensor(sum_34, 0.0006377551020408163)
    unsqueeze_403: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_509, 0);  mul_509 = None
    unsqueeze_404: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 2);  unsqueeze_403 = None
    unsqueeze_405: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 3);  unsqueeze_404 = None
    mul_510: "f32[96]" = torch.ops.aten.mul.Tensor(sum_35, 0.0006377551020408163)
    mul_511: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_512: "f32[96]" = torch.ops.aten.mul.Tensor(mul_510, mul_511);  mul_510 = mul_511 = None
    unsqueeze_406: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_512, 0);  mul_512 = None
    unsqueeze_407: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 2);  unsqueeze_406 = None
    unsqueeze_408: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 3);  unsqueeze_407 = None
    mul_513: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_71);  primals_71 = None
    unsqueeze_409: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_513, 0);  mul_513 = None
    unsqueeze_410: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 2);  unsqueeze_409 = None
    unsqueeze_411: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 3);  unsqueeze_410 = None
    sub_117: "f32[8, 96, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_402);  convolution_35 = unsqueeze_402 = None
    mul_514: "f32[8, 96, 14, 14]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_408);  sub_117 = unsqueeze_408 = None
    sub_118: "f32[8, 96, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_149, mul_514);  mul_514 = None
    sub_119: "f32[8, 96, 14, 14]" = torch.ops.aten.sub.Tensor(sub_118, unsqueeze_405);  sub_118 = unsqueeze_405 = None
    mul_515: "f32[8, 96, 14, 14]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_411);  sub_119 = unsqueeze_411 = None
    mul_516: "f32[96]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_106);  sum_35 = squeeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_515, relu_23, primals_140, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_515 = primals_140 = None
    getitem_152: "f32[8, 576, 14, 14]" = convolution_backward_16[0]
    getitem_153: "f32[96, 576, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_69: "f32[8, 576, 14, 14]" = torch.ops.aten.alias.default(relu_23);  relu_23 = None
    alias_70: "f32[8, 576, 14, 14]" = torch.ops.aten.alias.default(alias_69);  alias_69 = None
    le_11: "b8[8, 576, 14, 14]" = torch.ops.aten.le.Scalar(alias_70, 0);  alias_70 = None
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_11: "f32[8, 576, 14, 14]" = torch.ops.aten.where.self(le_11, scalar_tensor_11, getitem_152);  le_11 = scalar_tensor_11 = getitem_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_412: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
    unsqueeze_413: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 2);  unsqueeze_412 = None
    unsqueeze_414: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 3);  unsqueeze_413 = None
    sum_36: "f32[576]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_120: "f32[8, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_414)
    mul_517: "f32[8, 576, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, sub_120);  sub_120 = None
    sum_37: "f32[576]" = torch.ops.aten.sum.dim_IntList(mul_517, [0, 2, 3]);  mul_517 = None
    mul_518: "f32[576]" = torch.ops.aten.mul.Tensor(sum_36, 0.0006377551020408163)
    unsqueeze_415: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_518, 0);  mul_518 = None
    unsqueeze_416: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 2);  unsqueeze_415 = None
    unsqueeze_417: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 3);  unsqueeze_416 = None
    mul_519: "f32[576]" = torch.ops.aten.mul.Tensor(sum_37, 0.0006377551020408163)
    mul_520: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_521: "f32[576]" = torch.ops.aten.mul.Tensor(mul_519, mul_520);  mul_519 = mul_520 = None
    unsqueeze_418: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_521, 0);  mul_521 = None
    unsqueeze_419: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 2);  unsqueeze_418 = None
    unsqueeze_420: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 3);  unsqueeze_419 = None
    mul_522: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_69);  primals_69 = None
    unsqueeze_421: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_522, 0);  mul_522 = None
    unsqueeze_422: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None
    unsqueeze_423: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 3);  unsqueeze_422 = None
    sub_121: "f32[8, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_414);  convolution_34 = unsqueeze_414 = None
    mul_523: "f32[8, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_420);  sub_121 = unsqueeze_420 = None
    sub_122: "f32[8, 576, 14, 14]" = torch.ops.aten.sub.Tensor(where_11, mul_523);  where_11 = mul_523 = None
    sub_123: "f32[8, 576, 14, 14]" = torch.ops.aten.sub.Tensor(sub_122, unsqueeze_417);  sub_122 = unsqueeze_417 = None
    mul_524: "f32[8, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_423);  sub_123 = unsqueeze_423 = None
    mul_525: "f32[576]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_103);  sum_37 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_524, relu_22, primals_139, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 576, [True, True, False]);  mul_524 = primals_139 = None
    getitem_155: "f32[8, 576, 14, 14]" = convolution_backward_17[0]
    getitem_156: "f32[576, 1, 3, 3]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_72: "f32[8, 576, 14, 14]" = torch.ops.aten.alias.default(relu_22);  relu_22 = None
    alias_73: "f32[8, 576, 14, 14]" = torch.ops.aten.alias.default(alias_72);  alias_72 = None
    le_12: "b8[8, 576, 14, 14]" = torch.ops.aten.le.Scalar(alias_73, 0);  alias_73 = None
    scalar_tensor_12: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_12: "f32[8, 576, 14, 14]" = torch.ops.aten.where.self(le_12, scalar_tensor_12, getitem_155);  le_12 = scalar_tensor_12 = getitem_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_424: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
    unsqueeze_425: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 2);  unsqueeze_424 = None
    unsqueeze_426: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 3);  unsqueeze_425 = None
    sum_38: "f32[576]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_124: "f32[8, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_426)
    mul_526: "f32[8, 576, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, sub_124);  sub_124 = None
    sum_39: "f32[576]" = torch.ops.aten.sum.dim_IntList(mul_526, [0, 2, 3]);  mul_526 = None
    mul_527: "f32[576]" = torch.ops.aten.mul.Tensor(sum_38, 0.0006377551020408163)
    unsqueeze_427: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_527, 0);  mul_527 = None
    unsqueeze_428: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 2);  unsqueeze_427 = None
    unsqueeze_429: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 3);  unsqueeze_428 = None
    mul_528: "f32[576]" = torch.ops.aten.mul.Tensor(sum_39, 0.0006377551020408163)
    mul_529: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_530: "f32[576]" = torch.ops.aten.mul.Tensor(mul_528, mul_529);  mul_528 = mul_529 = None
    unsqueeze_430: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_530, 0);  mul_530 = None
    unsqueeze_431: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 2);  unsqueeze_430 = None
    unsqueeze_432: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 3);  unsqueeze_431 = None
    mul_531: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_67);  primals_67 = None
    unsqueeze_433: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_531, 0);  mul_531 = None
    unsqueeze_434: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
    unsqueeze_435: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 3);  unsqueeze_434 = None
    sub_125: "f32[8, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_426);  convolution_33 = unsqueeze_426 = None
    mul_532: "f32[8, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_432);  sub_125 = unsqueeze_432 = None
    sub_126: "f32[8, 576, 14, 14]" = torch.ops.aten.sub.Tensor(where_12, mul_532);  where_12 = mul_532 = None
    sub_127: "f32[8, 576, 14, 14]" = torch.ops.aten.sub.Tensor(sub_126, unsqueeze_429);  sub_126 = unsqueeze_429 = None
    mul_533: "f32[8, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_435);  sub_127 = unsqueeze_435 = None
    mul_534: "f32[576]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_100);  sum_39 = squeeze_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_533, add_170, primals_138, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_533 = add_170 = primals_138 = None
    getitem_158: "f32[8, 96, 14, 14]" = convolution_backward_18[0]
    getitem_159: "f32[576, 96, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_273: "f32[8, 96, 14, 14]" = torch.ops.aten.add.Tensor(getitem_149, getitem_158);  getitem_149 = getitem_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_436: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
    unsqueeze_437: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 2);  unsqueeze_436 = None
    unsqueeze_438: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 3);  unsqueeze_437 = None
    sum_40: "f32[96]" = torch.ops.aten.sum.dim_IntList(add_273, [0, 2, 3])
    sub_128: "f32[8, 96, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_438)
    mul_535: "f32[8, 96, 14, 14]" = torch.ops.aten.mul.Tensor(add_273, sub_128);  sub_128 = None
    sum_41: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_535, [0, 2, 3]);  mul_535 = None
    mul_536: "f32[96]" = torch.ops.aten.mul.Tensor(sum_40, 0.0006377551020408163)
    unsqueeze_439: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_536, 0);  mul_536 = None
    unsqueeze_440: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 2);  unsqueeze_439 = None
    unsqueeze_441: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 3);  unsqueeze_440 = None
    mul_537: "f32[96]" = torch.ops.aten.mul.Tensor(sum_41, 0.0006377551020408163)
    mul_538: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_539: "f32[96]" = torch.ops.aten.mul.Tensor(mul_537, mul_538);  mul_537 = mul_538 = None
    unsqueeze_442: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_539, 0);  mul_539 = None
    unsqueeze_443: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 2);  unsqueeze_442 = None
    unsqueeze_444: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 3);  unsqueeze_443 = None
    mul_540: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_65);  primals_65 = None
    unsqueeze_445: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_540, 0);  mul_540 = None
    unsqueeze_446: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
    unsqueeze_447: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 3);  unsqueeze_446 = None
    sub_129: "f32[8, 96, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_438);  convolution_32 = unsqueeze_438 = None
    mul_541: "f32[8, 96, 14, 14]" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_444);  sub_129 = unsqueeze_444 = None
    sub_130: "f32[8, 96, 14, 14]" = torch.ops.aten.sub.Tensor(add_273, mul_541);  add_273 = mul_541 = None
    sub_131: "f32[8, 96, 14, 14]" = torch.ops.aten.sub.Tensor(sub_130, unsqueeze_441);  sub_130 = unsqueeze_441 = None
    mul_542: "f32[8, 96, 14, 14]" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_447);  sub_131 = unsqueeze_447 = None
    mul_543: "f32[96]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_97);  sum_41 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_542, relu_21, primals_137, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_542 = primals_137 = None
    getitem_161: "f32[8, 480, 14, 14]" = convolution_backward_19[0]
    getitem_162: "f32[96, 480, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_75: "f32[8, 480, 14, 14]" = torch.ops.aten.alias.default(relu_21);  relu_21 = None
    alias_76: "f32[8, 480, 14, 14]" = torch.ops.aten.alias.default(alias_75);  alias_75 = None
    le_13: "b8[8, 480, 14, 14]" = torch.ops.aten.le.Scalar(alias_76, 0);  alias_76 = None
    scalar_tensor_13: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_13: "f32[8, 480, 14, 14]" = torch.ops.aten.where.self(le_13, scalar_tensor_13, getitem_161);  le_13 = scalar_tensor_13 = getitem_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_448: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
    unsqueeze_449: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, 2);  unsqueeze_448 = None
    unsqueeze_450: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 3);  unsqueeze_449 = None
    sum_42: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_132: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_450)
    mul_544: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_13, sub_132);  sub_132 = None
    sum_43: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_544, [0, 2, 3]);  mul_544 = None
    mul_545: "f32[480]" = torch.ops.aten.mul.Tensor(sum_42, 0.0006377551020408163)
    unsqueeze_451: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_545, 0);  mul_545 = None
    unsqueeze_452: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 2);  unsqueeze_451 = None
    unsqueeze_453: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 3);  unsqueeze_452 = None
    mul_546: "f32[480]" = torch.ops.aten.mul.Tensor(sum_43, 0.0006377551020408163)
    mul_547: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_548: "f32[480]" = torch.ops.aten.mul.Tensor(mul_546, mul_547);  mul_546 = mul_547 = None
    unsqueeze_454: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_548, 0);  mul_548 = None
    unsqueeze_455: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 2);  unsqueeze_454 = None
    unsqueeze_456: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 3);  unsqueeze_455 = None
    mul_549: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_63);  primals_63 = None
    unsqueeze_457: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_549, 0);  mul_549 = None
    unsqueeze_458: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 2);  unsqueeze_457 = None
    unsqueeze_459: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 3);  unsqueeze_458 = None
    sub_133: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_450);  convolution_31 = unsqueeze_450 = None
    mul_550: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_456);  sub_133 = unsqueeze_456 = None
    sub_134: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(where_13, mul_550);  where_13 = mul_550 = None
    sub_135: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_134, unsqueeze_453);  sub_134 = unsqueeze_453 = None
    mul_551: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_459);  sub_135 = unsqueeze_459 = None
    mul_552: "f32[480]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_94);  sum_43 = squeeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_551, relu_20, primals_136, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_551 = primals_136 = None
    getitem_164: "f32[8, 480, 14, 14]" = convolution_backward_20[0]
    getitem_165: "f32[480, 1, 3, 3]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_78: "f32[8, 480, 14, 14]" = torch.ops.aten.alias.default(relu_20);  relu_20 = None
    alias_79: "f32[8, 480, 14, 14]" = torch.ops.aten.alias.default(alias_78);  alias_78 = None
    le_14: "b8[8, 480, 14, 14]" = torch.ops.aten.le.Scalar(alias_79, 0);  alias_79 = None
    scalar_tensor_14: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_14: "f32[8, 480, 14, 14]" = torch.ops.aten.where.self(le_14, scalar_tensor_14, getitem_164);  le_14 = scalar_tensor_14 = getitem_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_460: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_461: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, 2);  unsqueeze_460 = None
    unsqueeze_462: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 3);  unsqueeze_461 = None
    sum_44: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_136: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_462)
    mul_553: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, sub_136);  sub_136 = None
    sum_45: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_553, [0, 2, 3]);  mul_553 = None
    mul_554: "f32[480]" = torch.ops.aten.mul.Tensor(sum_44, 0.0006377551020408163)
    unsqueeze_463: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_554, 0);  mul_554 = None
    unsqueeze_464: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 2);  unsqueeze_463 = None
    unsqueeze_465: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 3);  unsqueeze_464 = None
    mul_555: "f32[480]" = torch.ops.aten.mul.Tensor(sum_45, 0.0006377551020408163)
    mul_556: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_557: "f32[480]" = torch.ops.aten.mul.Tensor(mul_555, mul_556);  mul_555 = mul_556 = None
    unsqueeze_466: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_557, 0);  mul_557 = None
    unsqueeze_467: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 2);  unsqueeze_466 = None
    unsqueeze_468: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 3);  unsqueeze_467 = None
    mul_558: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_61);  primals_61 = None
    unsqueeze_469: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_558, 0);  mul_558 = None
    unsqueeze_470: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None
    unsqueeze_471: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 3);  unsqueeze_470 = None
    sub_137: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_462);  convolution_30 = unsqueeze_462 = None
    mul_559: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_468);  sub_137 = unsqueeze_468 = None
    sub_138: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(where_14, mul_559);  where_14 = mul_559 = None
    sub_139: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_138, unsqueeze_465);  sub_138 = unsqueeze_465 = None
    mul_560: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_471);  sub_139 = unsqueeze_471 = None
    mul_561: "f32[480]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_91);  sum_45 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_560, add_155, primals_135, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_560 = add_155 = primals_135 = None
    getitem_167: "f32[8, 80, 14, 14]" = convolution_backward_21[0]
    getitem_168: "f32[480, 80, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_472: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    unsqueeze_473: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, 2);  unsqueeze_472 = None
    unsqueeze_474: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 3);  unsqueeze_473 = None
    sum_46: "f32[80]" = torch.ops.aten.sum.dim_IntList(getitem_167, [0, 2, 3])
    sub_140: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_474)
    mul_562: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_167, sub_140);  sub_140 = None
    sum_47: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_562, [0, 2, 3]);  mul_562 = None
    mul_563: "f32[80]" = torch.ops.aten.mul.Tensor(sum_46, 0.0006377551020408163)
    unsqueeze_475: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_563, 0);  mul_563 = None
    unsqueeze_476: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 2);  unsqueeze_475 = None
    unsqueeze_477: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 3);  unsqueeze_476 = None
    mul_564: "f32[80]" = torch.ops.aten.mul.Tensor(sum_47, 0.0006377551020408163)
    mul_565: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_566: "f32[80]" = torch.ops.aten.mul.Tensor(mul_564, mul_565);  mul_564 = mul_565 = None
    unsqueeze_478: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_566, 0);  mul_566 = None
    unsqueeze_479: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 2);  unsqueeze_478 = None
    unsqueeze_480: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 3);  unsqueeze_479 = None
    mul_567: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_59);  primals_59 = None
    unsqueeze_481: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_567, 0);  mul_567 = None
    unsqueeze_482: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 2);  unsqueeze_481 = None
    unsqueeze_483: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 3);  unsqueeze_482 = None
    sub_141: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_474);  convolution_29 = unsqueeze_474 = None
    mul_568: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_480);  sub_141 = unsqueeze_480 = None
    sub_142: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_167, mul_568);  mul_568 = None
    sub_143: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(sub_142, unsqueeze_477);  sub_142 = unsqueeze_477 = None
    mul_569: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_483);  sub_143 = unsqueeze_483 = None
    mul_570: "f32[80]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_88);  sum_47 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_569, relu_19, primals_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_569 = primals_134 = None
    getitem_170: "f32[8, 480, 14, 14]" = convolution_backward_22[0]
    getitem_171: "f32[80, 480, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_81: "f32[8, 480, 14, 14]" = torch.ops.aten.alias.default(relu_19);  relu_19 = None
    alias_82: "f32[8, 480, 14, 14]" = torch.ops.aten.alias.default(alias_81);  alias_81 = None
    le_15: "b8[8, 480, 14, 14]" = torch.ops.aten.le.Scalar(alias_82, 0);  alias_82 = None
    scalar_tensor_15: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_15: "f32[8, 480, 14, 14]" = torch.ops.aten.where.self(le_15, scalar_tensor_15, getitem_170);  le_15 = scalar_tensor_15 = getitem_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_484: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_485: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 2);  unsqueeze_484 = None
    unsqueeze_486: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 3);  unsqueeze_485 = None
    sum_48: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_144: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_486)
    mul_571: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, sub_144);  sub_144 = None
    sum_49: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_571, [0, 2, 3]);  mul_571 = None
    mul_572: "f32[480]" = torch.ops.aten.mul.Tensor(sum_48, 0.0006377551020408163)
    unsqueeze_487: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_572, 0);  mul_572 = None
    unsqueeze_488: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
    unsqueeze_489: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 3);  unsqueeze_488 = None
    mul_573: "f32[480]" = torch.ops.aten.mul.Tensor(sum_49, 0.0006377551020408163)
    mul_574: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_575: "f32[480]" = torch.ops.aten.mul.Tensor(mul_573, mul_574);  mul_573 = mul_574 = None
    unsqueeze_490: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_575, 0);  mul_575 = None
    unsqueeze_491: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 2);  unsqueeze_490 = None
    unsqueeze_492: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 3);  unsqueeze_491 = None
    mul_576: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_57);  primals_57 = None
    unsqueeze_493: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_576, 0);  mul_576 = None
    unsqueeze_494: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
    unsqueeze_495: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 3);  unsqueeze_494 = None
    sub_145: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_486);  convolution_28 = unsqueeze_486 = None
    mul_577: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_492);  sub_145 = unsqueeze_492 = None
    sub_146: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(where_15, mul_577);  where_15 = mul_577 = None
    sub_147: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_146, unsqueeze_489);  sub_146 = unsqueeze_489 = None
    mul_578: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_495);  sub_147 = unsqueeze_495 = None
    mul_579: "f32[480]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_85);  sum_49 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_578, relu_18, primals_133, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_578 = primals_133 = None
    getitem_173: "f32[8, 480, 14, 14]" = convolution_backward_23[0]
    getitem_174: "f32[480, 1, 5, 5]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_84: "f32[8, 480, 14, 14]" = torch.ops.aten.alias.default(relu_18);  relu_18 = None
    alias_85: "f32[8, 480, 14, 14]" = torch.ops.aten.alias.default(alias_84);  alias_84 = None
    le_16: "b8[8, 480, 14, 14]" = torch.ops.aten.le.Scalar(alias_85, 0);  alias_85 = None
    scalar_tensor_16: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_16: "f32[8, 480, 14, 14]" = torch.ops.aten.where.self(le_16, scalar_tensor_16, getitem_173);  le_16 = scalar_tensor_16 = getitem_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_496: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_497: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 2);  unsqueeze_496 = None
    unsqueeze_498: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 3);  unsqueeze_497 = None
    sum_50: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_148: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_498)
    mul_580: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, sub_148);  sub_148 = None
    sum_51: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_580, [0, 2, 3]);  mul_580 = None
    mul_581: "f32[480]" = torch.ops.aten.mul.Tensor(sum_50, 0.0006377551020408163)
    unsqueeze_499: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_581, 0);  mul_581 = None
    unsqueeze_500: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 2);  unsqueeze_499 = None
    unsqueeze_501: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 3);  unsqueeze_500 = None
    mul_582: "f32[480]" = torch.ops.aten.mul.Tensor(sum_51, 0.0006377551020408163)
    mul_583: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_584: "f32[480]" = torch.ops.aten.mul.Tensor(mul_582, mul_583);  mul_582 = mul_583 = None
    unsqueeze_502: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_584, 0);  mul_584 = None
    unsqueeze_503: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 2);  unsqueeze_502 = None
    unsqueeze_504: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 3);  unsqueeze_503 = None
    mul_585: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_55);  primals_55 = None
    unsqueeze_505: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_585, 0);  mul_585 = None
    unsqueeze_506: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 2);  unsqueeze_505 = None
    unsqueeze_507: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 3);  unsqueeze_506 = None
    sub_149: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_498);  convolution_27 = unsqueeze_498 = None
    mul_586: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_504);  sub_149 = unsqueeze_504 = None
    sub_150: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(where_16, mul_586);  where_16 = mul_586 = None
    sub_151: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_150, unsqueeze_501);  sub_150 = unsqueeze_501 = None
    mul_587: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_507);  sub_151 = unsqueeze_507 = None
    mul_588: "f32[480]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_82);  sum_51 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_587, add_139, primals_132, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_587 = add_139 = primals_132 = None
    getitem_176: "f32[8, 80, 14, 14]" = convolution_backward_24[0]
    getitem_177: "f32[480, 80, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_274: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(getitem_167, getitem_176);  getitem_167 = getitem_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_508: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_509: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, 2);  unsqueeze_508 = None
    unsqueeze_510: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 3);  unsqueeze_509 = None
    sum_52: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_274, [0, 2, 3])
    sub_152: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_510)
    mul_589: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_274, sub_152);  sub_152 = None
    sum_53: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_589, [0, 2, 3]);  mul_589 = None
    mul_590: "f32[80]" = torch.ops.aten.mul.Tensor(sum_52, 0.0006377551020408163)
    unsqueeze_511: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_590, 0);  mul_590 = None
    unsqueeze_512: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 2);  unsqueeze_511 = None
    unsqueeze_513: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 3);  unsqueeze_512 = None
    mul_591: "f32[80]" = torch.ops.aten.mul.Tensor(sum_53, 0.0006377551020408163)
    mul_592: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_593: "f32[80]" = torch.ops.aten.mul.Tensor(mul_591, mul_592);  mul_591 = mul_592 = None
    unsqueeze_514: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_593, 0);  mul_593 = None
    unsqueeze_515: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 2);  unsqueeze_514 = None
    unsqueeze_516: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 3);  unsqueeze_515 = None
    mul_594: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_53);  primals_53 = None
    unsqueeze_517: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_594, 0);  mul_594 = None
    unsqueeze_518: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 2);  unsqueeze_517 = None
    unsqueeze_519: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 3);  unsqueeze_518 = None
    sub_153: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_510);  convolution_26 = unsqueeze_510 = None
    mul_595: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_516);  sub_153 = unsqueeze_516 = None
    sub_154: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(add_274, mul_595);  mul_595 = None
    sub_155: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(sub_154, unsqueeze_513);  sub_154 = unsqueeze_513 = None
    mul_596: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_519);  sub_155 = unsqueeze_519 = None
    mul_597: "f32[80]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_79);  sum_53 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_596, relu_17, primals_131, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_596 = primals_131 = None
    getitem_179: "f32[8, 480, 14, 14]" = convolution_backward_25[0]
    getitem_180: "f32[80, 480, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_87: "f32[8, 480, 14, 14]" = torch.ops.aten.alias.default(relu_17);  relu_17 = None
    alias_88: "f32[8, 480, 14, 14]" = torch.ops.aten.alias.default(alias_87);  alias_87 = None
    le_17: "b8[8, 480, 14, 14]" = torch.ops.aten.le.Scalar(alias_88, 0);  alias_88 = None
    scalar_tensor_17: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_17: "f32[8, 480, 14, 14]" = torch.ops.aten.where.self(le_17, scalar_tensor_17, getitem_179);  le_17 = scalar_tensor_17 = getitem_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_520: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_521: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 2);  unsqueeze_520 = None
    unsqueeze_522: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 3);  unsqueeze_521 = None
    sum_54: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_156: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_522)
    mul_598: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_17, sub_156);  sub_156 = None
    sum_55: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_598, [0, 2, 3]);  mul_598 = None
    mul_599: "f32[480]" = torch.ops.aten.mul.Tensor(sum_54, 0.0006377551020408163)
    unsqueeze_523: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_599, 0);  mul_599 = None
    unsqueeze_524: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 2);  unsqueeze_523 = None
    unsqueeze_525: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 3);  unsqueeze_524 = None
    mul_600: "f32[480]" = torch.ops.aten.mul.Tensor(sum_55, 0.0006377551020408163)
    mul_601: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_602: "f32[480]" = torch.ops.aten.mul.Tensor(mul_600, mul_601);  mul_600 = mul_601 = None
    unsqueeze_526: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_602, 0);  mul_602 = None
    unsqueeze_527: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 2);  unsqueeze_526 = None
    unsqueeze_528: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 3);  unsqueeze_527 = None
    mul_603: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_51);  primals_51 = None
    unsqueeze_529: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_603, 0);  mul_603 = None
    unsqueeze_530: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 2);  unsqueeze_529 = None
    unsqueeze_531: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 3);  unsqueeze_530 = None
    sub_157: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_522);  convolution_25 = unsqueeze_522 = None
    mul_604: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_528);  sub_157 = unsqueeze_528 = None
    sub_158: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(where_17, mul_604);  where_17 = mul_604 = None
    sub_159: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_158, unsqueeze_525);  sub_158 = unsqueeze_525 = None
    mul_605: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_531);  sub_159 = unsqueeze_531 = None
    mul_606: "f32[480]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_76);  sum_55 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_605, relu_16, primals_130, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_605 = primals_130 = None
    getitem_182: "f32[8, 480, 14, 14]" = convolution_backward_26[0]
    getitem_183: "f32[480, 1, 5, 5]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_90: "f32[8, 480, 14, 14]" = torch.ops.aten.alias.default(relu_16);  relu_16 = None
    alias_91: "f32[8, 480, 14, 14]" = torch.ops.aten.alias.default(alias_90);  alias_90 = None
    le_18: "b8[8, 480, 14, 14]" = torch.ops.aten.le.Scalar(alias_91, 0);  alias_91 = None
    scalar_tensor_18: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_18: "f32[8, 480, 14, 14]" = torch.ops.aten.where.self(le_18, scalar_tensor_18, getitem_182);  le_18 = scalar_tensor_18 = getitem_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_532: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_533: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 2);  unsqueeze_532 = None
    unsqueeze_534: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 3);  unsqueeze_533 = None
    sum_56: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_160: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_534)
    mul_607: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, sub_160);  sub_160 = None
    sum_57: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_607, [0, 2, 3]);  mul_607 = None
    mul_608: "f32[480]" = torch.ops.aten.mul.Tensor(sum_56, 0.0006377551020408163)
    unsqueeze_535: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_608, 0);  mul_608 = None
    unsqueeze_536: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 2);  unsqueeze_535 = None
    unsqueeze_537: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 3);  unsqueeze_536 = None
    mul_609: "f32[480]" = torch.ops.aten.mul.Tensor(sum_57, 0.0006377551020408163)
    mul_610: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_611: "f32[480]" = torch.ops.aten.mul.Tensor(mul_609, mul_610);  mul_609 = mul_610 = None
    unsqueeze_538: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_611, 0);  mul_611 = None
    unsqueeze_539: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 2);  unsqueeze_538 = None
    unsqueeze_540: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 3);  unsqueeze_539 = None
    mul_612: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_49);  primals_49 = None
    unsqueeze_541: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_612, 0);  mul_612 = None
    unsqueeze_542: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 2);  unsqueeze_541 = None
    unsqueeze_543: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 3);  unsqueeze_542 = None
    sub_161: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_534);  convolution_24 = unsqueeze_534 = None
    mul_613: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_540);  sub_161 = unsqueeze_540 = None
    sub_162: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(where_18, mul_613);  where_18 = mul_613 = None
    sub_163: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_162, unsqueeze_537);  sub_162 = unsqueeze_537 = None
    mul_614: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_543);  sub_163 = unsqueeze_543 = None
    mul_615: "f32[480]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_73);  sum_57 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_614, add_123, primals_129, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_614 = add_123 = primals_129 = None
    getitem_185: "f32[8, 80, 14, 14]" = convolution_backward_27[0]
    getitem_186: "f32[480, 80, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_275: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_274, getitem_185);  add_274 = getitem_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_544: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_545: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, 2);  unsqueeze_544 = None
    unsqueeze_546: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 3);  unsqueeze_545 = None
    sum_58: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_275, [0, 2, 3])
    sub_164: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_546)
    mul_616: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_275, sub_164);  sub_164 = None
    sum_59: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_616, [0, 2, 3]);  mul_616 = None
    mul_617: "f32[80]" = torch.ops.aten.mul.Tensor(sum_58, 0.0006377551020408163)
    unsqueeze_547: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_617, 0);  mul_617 = None
    unsqueeze_548: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_547, 2);  unsqueeze_547 = None
    unsqueeze_549: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 3);  unsqueeze_548 = None
    mul_618: "f32[80]" = torch.ops.aten.mul.Tensor(sum_59, 0.0006377551020408163)
    mul_619: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_620: "f32[80]" = torch.ops.aten.mul.Tensor(mul_618, mul_619);  mul_618 = mul_619 = None
    unsqueeze_550: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_620, 0);  mul_620 = None
    unsqueeze_551: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 2);  unsqueeze_550 = None
    unsqueeze_552: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 3);  unsqueeze_551 = None
    mul_621: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_47);  primals_47 = None
    unsqueeze_553: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_621, 0);  mul_621 = None
    unsqueeze_554: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 2);  unsqueeze_553 = None
    unsqueeze_555: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 3);  unsqueeze_554 = None
    sub_165: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_546);  convolution_23 = unsqueeze_546 = None
    mul_622: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_552);  sub_165 = unsqueeze_552 = None
    sub_166: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(add_275, mul_622);  add_275 = mul_622 = None
    sub_167: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(sub_166, unsqueeze_549);  sub_166 = unsqueeze_549 = None
    mul_623: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_555);  sub_167 = unsqueeze_555 = None
    mul_624: "f32[80]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_70);  sum_59 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_623, relu_15, primals_128, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_623 = primals_128 = None
    getitem_188: "f32[8, 240, 14, 14]" = convolution_backward_28[0]
    getitem_189: "f32[80, 240, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_93: "f32[8, 240, 14, 14]" = torch.ops.aten.alias.default(relu_15);  relu_15 = None
    alias_94: "f32[8, 240, 14, 14]" = torch.ops.aten.alias.default(alias_93);  alias_93 = None
    le_19: "b8[8, 240, 14, 14]" = torch.ops.aten.le.Scalar(alias_94, 0);  alias_94 = None
    scalar_tensor_19: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_19: "f32[8, 240, 14, 14]" = torch.ops.aten.where.self(le_19, scalar_tensor_19, getitem_188);  le_19 = scalar_tensor_19 = getitem_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_556: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_557: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 2);  unsqueeze_556 = None
    unsqueeze_558: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 3);  unsqueeze_557 = None
    sum_60: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_168: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_558)
    mul_625: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, sub_168);  sub_168 = None
    sum_61: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_625, [0, 2, 3]);  mul_625 = None
    mul_626: "f32[240]" = torch.ops.aten.mul.Tensor(sum_60, 0.0006377551020408163)
    unsqueeze_559: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_626, 0);  mul_626 = None
    unsqueeze_560: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 2);  unsqueeze_559 = None
    unsqueeze_561: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 3);  unsqueeze_560 = None
    mul_627: "f32[240]" = torch.ops.aten.mul.Tensor(sum_61, 0.0006377551020408163)
    mul_628: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_629: "f32[240]" = torch.ops.aten.mul.Tensor(mul_627, mul_628);  mul_627 = mul_628 = None
    unsqueeze_562: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_629, 0);  mul_629 = None
    unsqueeze_563: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 2);  unsqueeze_562 = None
    unsqueeze_564: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 3);  unsqueeze_563 = None
    mul_630: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_45);  primals_45 = None
    unsqueeze_565: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_630, 0);  mul_630 = None
    unsqueeze_566: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 2);  unsqueeze_565 = None
    unsqueeze_567: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 3);  unsqueeze_566 = None
    sub_169: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_558);  convolution_22 = unsqueeze_558 = None
    mul_631: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_564);  sub_169 = unsqueeze_564 = None
    sub_170: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(where_19, mul_631);  where_19 = mul_631 = None
    sub_171: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(sub_170, unsqueeze_561);  sub_170 = unsqueeze_561 = None
    mul_632: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_567);  sub_171 = unsqueeze_567 = None
    mul_633: "f32[240]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_67);  sum_61 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_632, relu_14, primals_127, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 240, [True, True, False]);  mul_632 = primals_127 = None
    getitem_191: "f32[8, 240, 28, 28]" = convolution_backward_29[0]
    getitem_192: "f32[240, 1, 5, 5]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_96: "f32[8, 240, 28, 28]" = torch.ops.aten.alias.default(relu_14);  relu_14 = None
    alias_97: "f32[8, 240, 28, 28]" = torch.ops.aten.alias.default(alias_96);  alias_96 = None
    le_20: "b8[8, 240, 28, 28]" = torch.ops.aten.le.Scalar(alias_97, 0);  alias_97 = None
    scalar_tensor_20: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_20: "f32[8, 240, 28, 28]" = torch.ops.aten.where.self(le_20, scalar_tensor_20, getitem_191);  le_20 = scalar_tensor_20 = getitem_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_568: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_569: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, 2);  unsqueeze_568 = None
    unsqueeze_570: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 3);  unsqueeze_569 = None
    sum_62: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_172: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_570)
    mul_634: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(where_20, sub_172);  sub_172 = None
    sum_63: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_634, [0, 2, 3]);  mul_634 = None
    mul_635: "f32[240]" = torch.ops.aten.mul.Tensor(sum_62, 0.00015943877551020407)
    unsqueeze_571: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_635, 0);  mul_635 = None
    unsqueeze_572: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_571, 2);  unsqueeze_571 = None
    unsqueeze_573: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 3);  unsqueeze_572 = None
    mul_636: "f32[240]" = torch.ops.aten.mul.Tensor(sum_63, 0.00015943877551020407)
    mul_637: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_638: "f32[240]" = torch.ops.aten.mul.Tensor(mul_636, mul_637);  mul_636 = mul_637 = None
    unsqueeze_574: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_638, 0);  mul_638 = None
    unsqueeze_575: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, 2);  unsqueeze_574 = None
    unsqueeze_576: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 3);  unsqueeze_575 = None
    mul_639: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_43);  primals_43 = None
    unsqueeze_577: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_639, 0);  mul_639 = None
    unsqueeze_578: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 2);  unsqueeze_577 = None
    unsqueeze_579: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 3);  unsqueeze_578 = None
    sub_173: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_570);  convolution_21 = unsqueeze_570 = None
    mul_640: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_576);  sub_173 = unsqueeze_576 = None
    sub_174: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(where_20, mul_640);  where_20 = mul_640 = None
    sub_175: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(sub_174, unsqueeze_573);  sub_174 = unsqueeze_573 = None
    mul_641: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_579);  sub_175 = unsqueeze_579 = None
    mul_642: "f32[240]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_64);  sum_63 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_641, add_108, primals_126, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_641 = add_108 = primals_126 = None
    getitem_194: "f32[8, 40, 28, 28]" = convolution_backward_30[0]
    getitem_195: "f32[240, 40, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_580: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_581: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 2);  unsqueeze_580 = None
    unsqueeze_582: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 3);  unsqueeze_581 = None
    sum_64: "f32[40]" = torch.ops.aten.sum.dim_IntList(getitem_194, [0, 2, 3])
    sub_176: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_582)
    mul_643: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_194, sub_176);  sub_176 = None
    sum_65: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_643, [0, 2, 3]);  mul_643 = None
    mul_644: "f32[40]" = torch.ops.aten.mul.Tensor(sum_64, 0.00015943877551020407)
    unsqueeze_583: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_644, 0);  mul_644 = None
    unsqueeze_584: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_583, 2);  unsqueeze_583 = None
    unsqueeze_585: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 3);  unsqueeze_584 = None
    mul_645: "f32[40]" = torch.ops.aten.mul.Tensor(sum_65, 0.00015943877551020407)
    mul_646: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_647: "f32[40]" = torch.ops.aten.mul.Tensor(mul_645, mul_646);  mul_645 = mul_646 = None
    unsqueeze_586: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_647, 0);  mul_647 = None
    unsqueeze_587: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 2);  unsqueeze_586 = None
    unsqueeze_588: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 3);  unsqueeze_587 = None
    mul_648: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_41);  primals_41 = None
    unsqueeze_589: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_648, 0);  mul_648 = None
    unsqueeze_590: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 2);  unsqueeze_589 = None
    unsqueeze_591: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 3);  unsqueeze_590 = None
    sub_177: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_582);  convolution_20 = unsqueeze_582 = None
    mul_649: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_588);  sub_177 = unsqueeze_588 = None
    sub_178: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_194, mul_649);  mul_649 = None
    sub_179: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_178, unsqueeze_585);  sub_178 = unsqueeze_585 = None
    mul_650: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_591);  sub_179 = unsqueeze_591 = None
    mul_651: "f32[40]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_61);  sum_65 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_650, relu_13, primals_125, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_650 = primals_125 = None
    getitem_197: "f32[8, 120, 28, 28]" = convolution_backward_31[0]
    getitem_198: "f32[40, 120, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_99: "f32[8, 120, 28, 28]" = torch.ops.aten.alias.default(relu_13);  relu_13 = None
    alias_100: "f32[8, 120, 28, 28]" = torch.ops.aten.alias.default(alias_99);  alias_99 = None
    le_21: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(alias_100, 0);  alias_100 = None
    scalar_tensor_21: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_21: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_21, scalar_tensor_21, getitem_197);  le_21 = scalar_tensor_21 = getitem_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_592: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_593: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 2);  unsqueeze_592 = None
    unsqueeze_594: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 3);  unsqueeze_593 = None
    sum_66: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_180: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_594)
    mul_652: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_21, sub_180);  sub_180 = None
    sum_67: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_652, [0, 2, 3]);  mul_652 = None
    mul_653: "f32[120]" = torch.ops.aten.mul.Tensor(sum_66, 0.00015943877551020407)
    unsqueeze_595: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_653, 0);  mul_653 = None
    unsqueeze_596: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 2);  unsqueeze_595 = None
    unsqueeze_597: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 3);  unsqueeze_596 = None
    mul_654: "f32[120]" = torch.ops.aten.mul.Tensor(sum_67, 0.00015943877551020407)
    mul_655: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_656: "f32[120]" = torch.ops.aten.mul.Tensor(mul_654, mul_655);  mul_654 = mul_655 = None
    unsqueeze_598: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_656, 0);  mul_656 = None
    unsqueeze_599: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 2);  unsqueeze_598 = None
    unsqueeze_600: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 3);  unsqueeze_599 = None
    mul_657: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_39);  primals_39 = None
    unsqueeze_601: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_657, 0);  mul_657 = None
    unsqueeze_602: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 2);  unsqueeze_601 = None
    unsqueeze_603: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 3);  unsqueeze_602 = None
    sub_181: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_594);  convolution_19 = unsqueeze_594 = None
    mul_658: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_600);  sub_181 = unsqueeze_600 = None
    sub_182: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_21, mul_658);  where_21 = mul_658 = None
    sub_183: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_182, unsqueeze_597);  sub_182 = unsqueeze_597 = None
    mul_659: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_603);  sub_183 = unsqueeze_603 = None
    mul_660: "f32[120]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_58);  sum_67 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_659, relu_12, primals_124, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False]);  mul_659 = primals_124 = None
    getitem_200: "f32[8, 120, 28, 28]" = convolution_backward_32[0]
    getitem_201: "f32[120, 1, 5, 5]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_102: "f32[8, 120, 28, 28]" = torch.ops.aten.alias.default(relu_12);  relu_12 = None
    alias_103: "f32[8, 120, 28, 28]" = torch.ops.aten.alias.default(alias_102);  alias_102 = None
    le_22: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(alias_103, 0);  alias_103 = None
    scalar_tensor_22: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_22: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_22, scalar_tensor_22, getitem_200);  le_22 = scalar_tensor_22 = getitem_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_604: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_605: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, 2);  unsqueeze_604 = None
    unsqueeze_606: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 3);  unsqueeze_605 = None
    sum_68: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_184: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_606)
    mul_661: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_22, sub_184);  sub_184 = None
    sum_69: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_661, [0, 2, 3]);  mul_661 = None
    mul_662: "f32[120]" = torch.ops.aten.mul.Tensor(sum_68, 0.00015943877551020407)
    unsqueeze_607: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_662, 0);  mul_662 = None
    unsqueeze_608: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_607, 2);  unsqueeze_607 = None
    unsqueeze_609: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 3);  unsqueeze_608 = None
    mul_663: "f32[120]" = torch.ops.aten.mul.Tensor(sum_69, 0.00015943877551020407)
    mul_664: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_665: "f32[120]" = torch.ops.aten.mul.Tensor(mul_663, mul_664);  mul_663 = mul_664 = None
    unsqueeze_610: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_665, 0);  mul_665 = None
    unsqueeze_611: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 2);  unsqueeze_610 = None
    unsqueeze_612: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 3);  unsqueeze_611 = None
    mul_666: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_37);  primals_37 = None
    unsqueeze_613: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_666, 0);  mul_666 = None
    unsqueeze_614: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 2);  unsqueeze_613 = None
    unsqueeze_615: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 3);  unsqueeze_614 = None
    sub_185: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_606);  convolution_18 = unsqueeze_606 = None
    mul_667: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_612);  sub_185 = unsqueeze_612 = None
    sub_186: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_22, mul_667);  where_22 = mul_667 = None
    sub_187: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_186, unsqueeze_609);  sub_186 = unsqueeze_609 = None
    mul_668: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_615);  sub_187 = unsqueeze_615 = None
    mul_669: "f32[120]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_55);  sum_69 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_668, add_92, primals_123, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_668 = add_92 = primals_123 = None
    getitem_203: "f32[8, 40, 28, 28]" = convolution_backward_33[0]
    getitem_204: "f32[120, 40, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_276: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(getitem_194, getitem_203);  getitem_194 = getitem_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_616: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_617: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 2);  unsqueeze_616 = None
    unsqueeze_618: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 3);  unsqueeze_617 = None
    sum_70: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_276, [0, 2, 3])
    sub_188: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_618)
    mul_670: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_276, sub_188);  sub_188 = None
    sum_71: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_670, [0, 2, 3]);  mul_670 = None
    mul_671: "f32[40]" = torch.ops.aten.mul.Tensor(sum_70, 0.00015943877551020407)
    unsqueeze_619: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_671, 0);  mul_671 = None
    unsqueeze_620: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_619, 2);  unsqueeze_619 = None
    unsqueeze_621: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 3);  unsqueeze_620 = None
    mul_672: "f32[40]" = torch.ops.aten.mul.Tensor(sum_71, 0.00015943877551020407)
    mul_673: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_674: "f32[40]" = torch.ops.aten.mul.Tensor(mul_672, mul_673);  mul_672 = mul_673 = None
    unsqueeze_622: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_674, 0);  mul_674 = None
    unsqueeze_623: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 2);  unsqueeze_622 = None
    unsqueeze_624: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 3);  unsqueeze_623 = None
    mul_675: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_35);  primals_35 = None
    unsqueeze_625: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_675, 0);  mul_675 = None
    unsqueeze_626: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 2);  unsqueeze_625 = None
    unsqueeze_627: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 3);  unsqueeze_626 = None
    sub_189: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_618);  convolution_17 = unsqueeze_618 = None
    mul_676: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_624);  sub_189 = unsqueeze_624 = None
    sub_190: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(add_276, mul_676);  mul_676 = None
    sub_191: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_190, unsqueeze_621);  sub_190 = unsqueeze_621 = None
    mul_677: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_627);  sub_191 = unsqueeze_627 = None
    mul_678: "f32[40]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_52);  sum_71 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_677, relu_11, primals_122, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_677 = primals_122 = None
    getitem_206: "f32[8, 120, 28, 28]" = convolution_backward_34[0]
    getitem_207: "f32[40, 120, 1, 1]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_105: "f32[8, 120, 28, 28]" = torch.ops.aten.alias.default(relu_11);  relu_11 = None
    alias_106: "f32[8, 120, 28, 28]" = torch.ops.aten.alias.default(alias_105);  alias_105 = None
    le_23: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(alias_106, 0);  alias_106 = None
    scalar_tensor_23: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_23: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_23, scalar_tensor_23, getitem_206);  le_23 = scalar_tensor_23 = getitem_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_628: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_629: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, 2);  unsqueeze_628 = None
    unsqueeze_630: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 3);  unsqueeze_629 = None
    sum_72: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_192: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_630)
    mul_679: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_23, sub_192);  sub_192 = None
    sum_73: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_679, [0, 2, 3]);  mul_679 = None
    mul_680: "f32[120]" = torch.ops.aten.mul.Tensor(sum_72, 0.00015943877551020407)
    unsqueeze_631: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_680, 0);  mul_680 = None
    unsqueeze_632: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_631, 2);  unsqueeze_631 = None
    unsqueeze_633: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 3);  unsqueeze_632 = None
    mul_681: "f32[120]" = torch.ops.aten.mul.Tensor(sum_73, 0.00015943877551020407)
    mul_682: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_683: "f32[120]" = torch.ops.aten.mul.Tensor(mul_681, mul_682);  mul_681 = mul_682 = None
    unsqueeze_634: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_683, 0);  mul_683 = None
    unsqueeze_635: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, 2);  unsqueeze_634 = None
    unsqueeze_636: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 3);  unsqueeze_635 = None
    mul_684: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_33);  primals_33 = None
    unsqueeze_637: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_684, 0);  mul_684 = None
    unsqueeze_638: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 2);  unsqueeze_637 = None
    unsqueeze_639: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 3);  unsqueeze_638 = None
    sub_193: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_630);  convolution_16 = unsqueeze_630 = None
    mul_685: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_636);  sub_193 = unsqueeze_636 = None
    sub_194: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_23, mul_685);  where_23 = mul_685 = None
    sub_195: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_194, unsqueeze_633);  sub_194 = unsqueeze_633 = None
    mul_686: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_195, unsqueeze_639);  sub_195 = unsqueeze_639 = None
    mul_687: "f32[120]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_49);  sum_73 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_686, relu_10, primals_121, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False]);  mul_686 = primals_121 = None
    getitem_209: "f32[8, 120, 28, 28]" = convolution_backward_35[0]
    getitem_210: "f32[120, 1, 5, 5]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_108: "f32[8, 120, 28, 28]" = torch.ops.aten.alias.default(relu_10);  relu_10 = None
    alias_109: "f32[8, 120, 28, 28]" = torch.ops.aten.alias.default(alias_108);  alias_108 = None
    le_24: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(alias_109, 0);  alias_109 = None
    scalar_tensor_24: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_24: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_24, scalar_tensor_24, getitem_209);  le_24 = scalar_tensor_24 = getitem_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_640: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_641: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, 2);  unsqueeze_640 = None
    unsqueeze_642: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 3);  unsqueeze_641 = None
    sum_74: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_196: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_642)
    mul_688: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_24, sub_196);  sub_196 = None
    sum_75: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_688, [0, 2, 3]);  mul_688 = None
    mul_689: "f32[120]" = torch.ops.aten.mul.Tensor(sum_74, 0.00015943877551020407)
    unsqueeze_643: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_689, 0);  mul_689 = None
    unsqueeze_644: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_643, 2);  unsqueeze_643 = None
    unsqueeze_645: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 3);  unsqueeze_644 = None
    mul_690: "f32[120]" = torch.ops.aten.mul.Tensor(sum_75, 0.00015943877551020407)
    mul_691: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_692: "f32[120]" = torch.ops.aten.mul.Tensor(mul_690, mul_691);  mul_690 = mul_691 = None
    unsqueeze_646: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_692, 0);  mul_692 = None
    unsqueeze_647: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, 2);  unsqueeze_646 = None
    unsqueeze_648: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 3);  unsqueeze_647 = None
    mul_693: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_31);  primals_31 = None
    unsqueeze_649: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_693, 0);  mul_693 = None
    unsqueeze_650: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 2);  unsqueeze_649 = None
    unsqueeze_651: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 3);  unsqueeze_650 = None
    sub_197: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_642);  convolution_15 = unsqueeze_642 = None
    mul_694: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_648);  sub_197 = unsqueeze_648 = None
    sub_198: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_24, mul_694);  where_24 = mul_694 = None
    sub_199: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_198, unsqueeze_645);  sub_198 = unsqueeze_645 = None
    mul_695: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_651);  sub_199 = unsqueeze_651 = None
    mul_696: "f32[120]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_46);  sum_75 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_695, add_76, primals_120, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_695 = add_76 = primals_120 = None
    getitem_212: "f32[8, 40, 28, 28]" = convolution_backward_36[0]
    getitem_213: "f32[120, 40, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_277: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_276, getitem_212);  add_276 = getitem_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_652: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_653: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, 2);  unsqueeze_652 = None
    unsqueeze_654: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 3);  unsqueeze_653 = None
    sum_76: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_277, [0, 2, 3])
    sub_200: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_654)
    mul_697: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_277, sub_200);  sub_200 = None
    sum_77: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_697, [0, 2, 3]);  mul_697 = None
    mul_698: "f32[40]" = torch.ops.aten.mul.Tensor(sum_76, 0.00015943877551020407)
    unsqueeze_655: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_698, 0);  mul_698 = None
    unsqueeze_656: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_655, 2);  unsqueeze_655 = None
    unsqueeze_657: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 3);  unsqueeze_656 = None
    mul_699: "f32[40]" = torch.ops.aten.mul.Tensor(sum_77, 0.00015943877551020407)
    mul_700: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_701: "f32[40]" = torch.ops.aten.mul.Tensor(mul_699, mul_700);  mul_699 = mul_700 = None
    unsqueeze_658: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_701, 0);  mul_701 = None
    unsqueeze_659: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, 2);  unsqueeze_658 = None
    unsqueeze_660: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 3);  unsqueeze_659 = None
    mul_702: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_29);  primals_29 = None
    unsqueeze_661: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_702, 0);  mul_702 = None
    unsqueeze_662: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 2);  unsqueeze_661 = None
    unsqueeze_663: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 3);  unsqueeze_662 = None
    sub_201: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_654);  convolution_14 = unsqueeze_654 = None
    mul_703: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_660);  sub_201 = unsqueeze_660 = None
    sub_202: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(add_277, mul_703);  add_277 = mul_703 = None
    sub_203: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_202, unsqueeze_657);  sub_202 = unsqueeze_657 = None
    mul_704: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_663);  sub_203 = unsqueeze_663 = None
    mul_705: "f32[40]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_43);  sum_77 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_704, relu_9, primals_119, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_704 = primals_119 = None
    getitem_215: "f32[8, 72, 28, 28]" = convolution_backward_37[0]
    getitem_216: "f32[40, 72, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_111: "f32[8, 72, 28, 28]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
    alias_112: "f32[8, 72, 28, 28]" = torch.ops.aten.alias.default(alias_111);  alias_111 = None
    le_25: "b8[8, 72, 28, 28]" = torch.ops.aten.le.Scalar(alias_112, 0);  alias_112 = None
    scalar_tensor_25: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_25: "f32[8, 72, 28, 28]" = torch.ops.aten.where.self(le_25, scalar_tensor_25, getitem_215);  le_25 = scalar_tensor_25 = getitem_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_664: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_665: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, 2);  unsqueeze_664 = None
    unsqueeze_666: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 3);  unsqueeze_665 = None
    sum_78: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_204: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_666)
    mul_706: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(where_25, sub_204);  sub_204 = None
    sum_79: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_706, [0, 2, 3]);  mul_706 = None
    mul_707: "f32[72]" = torch.ops.aten.mul.Tensor(sum_78, 0.00015943877551020407)
    unsqueeze_667: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_707, 0);  mul_707 = None
    unsqueeze_668: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_667, 2);  unsqueeze_667 = None
    unsqueeze_669: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 3);  unsqueeze_668 = None
    mul_708: "f32[72]" = torch.ops.aten.mul.Tensor(sum_79, 0.00015943877551020407)
    mul_709: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_710: "f32[72]" = torch.ops.aten.mul.Tensor(mul_708, mul_709);  mul_708 = mul_709 = None
    unsqueeze_670: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_710, 0);  mul_710 = None
    unsqueeze_671: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, 2);  unsqueeze_670 = None
    unsqueeze_672: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 3);  unsqueeze_671 = None
    mul_711: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_27);  primals_27 = None
    unsqueeze_673: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_711, 0);  mul_711 = None
    unsqueeze_674: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 2);  unsqueeze_673 = None
    unsqueeze_675: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 3);  unsqueeze_674 = None
    sub_205: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_666);  convolution_13 = unsqueeze_666 = None
    mul_712: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_672);  sub_205 = unsqueeze_672 = None
    sub_206: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(where_25, mul_712);  where_25 = mul_712 = None
    sub_207: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(sub_206, unsqueeze_669);  sub_206 = unsqueeze_669 = None
    mul_713: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_675);  sub_207 = unsqueeze_675 = None
    mul_714: "f32[72]" = torch.ops.aten.mul.Tensor(sum_79, squeeze_40);  sum_79 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_713, relu_8, primals_118, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 72, [True, True, False]);  mul_713 = primals_118 = None
    getitem_218: "f32[8, 72, 56, 56]" = convolution_backward_38[0]
    getitem_219: "f32[72, 1, 5, 5]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_114: "f32[8, 72, 56, 56]" = torch.ops.aten.alias.default(relu_8);  relu_8 = None
    alias_115: "f32[8, 72, 56, 56]" = torch.ops.aten.alias.default(alias_114);  alias_114 = None
    le_26: "b8[8, 72, 56, 56]" = torch.ops.aten.le.Scalar(alias_115, 0);  alias_115 = None
    scalar_tensor_26: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_26: "f32[8, 72, 56, 56]" = torch.ops.aten.where.self(le_26, scalar_tensor_26, getitem_218);  le_26 = scalar_tensor_26 = getitem_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_676: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_677: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, 2);  unsqueeze_676 = None
    unsqueeze_678: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 3);  unsqueeze_677 = None
    sum_80: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_208: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_678)
    mul_715: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_26, sub_208);  sub_208 = None
    sum_81: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_715, [0, 2, 3]);  mul_715 = None
    mul_716: "f32[72]" = torch.ops.aten.mul.Tensor(sum_80, 3.985969387755102e-05)
    unsqueeze_679: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_716, 0);  mul_716 = None
    unsqueeze_680: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_679, 2);  unsqueeze_679 = None
    unsqueeze_681: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 3);  unsqueeze_680 = None
    mul_717: "f32[72]" = torch.ops.aten.mul.Tensor(sum_81, 3.985969387755102e-05)
    mul_718: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_719: "f32[72]" = torch.ops.aten.mul.Tensor(mul_717, mul_718);  mul_717 = mul_718 = None
    unsqueeze_682: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_719, 0);  mul_719 = None
    unsqueeze_683: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, 2);  unsqueeze_682 = None
    unsqueeze_684: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 3);  unsqueeze_683 = None
    mul_720: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_25);  primals_25 = None
    unsqueeze_685: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_720, 0);  mul_720 = None
    unsqueeze_686: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 2);  unsqueeze_685 = None
    unsqueeze_687: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 3);  unsqueeze_686 = None
    sub_209: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_678);  convolution_12 = unsqueeze_678 = None
    mul_721: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_684);  sub_209 = unsqueeze_684 = None
    sub_210: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(where_26, mul_721);  where_26 = mul_721 = None
    sub_211: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(sub_210, unsqueeze_681);  sub_210 = unsqueeze_681 = None
    mul_722: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_687);  sub_211 = unsqueeze_687 = None
    mul_723: "f32[72]" = torch.ops.aten.mul.Tensor(sum_81, squeeze_37);  sum_81 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_722, add_61, primals_117, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_722 = add_61 = primals_117 = None
    getitem_221: "f32[8, 24, 56, 56]" = convolution_backward_39[0]
    getitem_222: "f32[72, 24, 1, 1]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_688: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_689: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, 2);  unsqueeze_688 = None
    unsqueeze_690: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 3);  unsqueeze_689 = None
    sum_82: "f32[24]" = torch.ops.aten.sum.dim_IntList(getitem_221, [0, 2, 3])
    sub_212: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_690)
    mul_724: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_221, sub_212);  sub_212 = None
    sum_83: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_724, [0, 2, 3]);  mul_724 = None
    mul_725: "f32[24]" = torch.ops.aten.mul.Tensor(sum_82, 3.985969387755102e-05)
    unsqueeze_691: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_725, 0);  mul_725 = None
    unsqueeze_692: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_691, 2);  unsqueeze_691 = None
    unsqueeze_693: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 3);  unsqueeze_692 = None
    mul_726: "f32[24]" = torch.ops.aten.mul.Tensor(sum_83, 3.985969387755102e-05)
    mul_727: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_728: "f32[24]" = torch.ops.aten.mul.Tensor(mul_726, mul_727);  mul_726 = mul_727 = None
    unsqueeze_694: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_728, 0);  mul_728 = None
    unsqueeze_695: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 2);  unsqueeze_694 = None
    unsqueeze_696: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 3);  unsqueeze_695 = None
    mul_729: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_23);  primals_23 = None
    unsqueeze_697: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_729, 0);  mul_729 = None
    unsqueeze_698: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 2);  unsqueeze_697 = None
    unsqueeze_699: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 3);  unsqueeze_698 = None
    sub_213: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_690);  convolution_11 = unsqueeze_690 = None
    mul_730: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_696);  sub_213 = unsqueeze_696 = None
    sub_214: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(getitem_221, mul_730);  mul_730 = None
    sub_215: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_214, unsqueeze_693);  sub_214 = unsqueeze_693 = None
    mul_731: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_215, unsqueeze_699);  sub_215 = unsqueeze_699 = None
    mul_732: "f32[24]" = torch.ops.aten.mul.Tensor(sum_83, squeeze_34);  sum_83 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_731, relu_7, primals_116, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_731 = primals_116 = None
    getitem_224: "f32[8, 72, 56, 56]" = convolution_backward_40[0]
    getitem_225: "f32[24, 72, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_117: "f32[8, 72, 56, 56]" = torch.ops.aten.alias.default(relu_7);  relu_7 = None
    alias_118: "f32[8, 72, 56, 56]" = torch.ops.aten.alias.default(alias_117);  alias_117 = None
    le_27: "b8[8, 72, 56, 56]" = torch.ops.aten.le.Scalar(alias_118, 0);  alias_118 = None
    scalar_tensor_27: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_27: "f32[8, 72, 56, 56]" = torch.ops.aten.where.self(le_27, scalar_tensor_27, getitem_224);  le_27 = scalar_tensor_27 = getitem_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_700: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_701: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, 2);  unsqueeze_700 = None
    unsqueeze_702: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 3);  unsqueeze_701 = None
    sum_84: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_216: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_702)
    mul_733: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_27, sub_216);  sub_216 = None
    sum_85: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_733, [0, 2, 3]);  mul_733 = None
    mul_734: "f32[72]" = torch.ops.aten.mul.Tensor(sum_84, 3.985969387755102e-05)
    unsqueeze_703: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_734, 0);  mul_734 = None
    unsqueeze_704: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_703, 2);  unsqueeze_703 = None
    unsqueeze_705: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 3);  unsqueeze_704 = None
    mul_735: "f32[72]" = torch.ops.aten.mul.Tensor(sum_85, 3.985969387755102e-05)
    mul_736: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_737: "f32[72]" = torch.ops.aten.mul.Tensor(mul_735, mul_736);  mul_735 = mul_736 = None
    unsqueeze_706: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_737, 0);  mul_737 = None
    unsqueeze_707: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 2);  unsqueeze_706 = None
    unsqueeze_708: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 3);  unsqueeze_707 = None
    mul_738: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_21);  primals_21 = None
    unsqueeze_709: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_738, 0);  mul_738 = None
    unsqueeze_710: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 2);  unsqueeze_709 = None
    unsqueeze_711: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 3);  unsqueeze_710 = None
    sub_217: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_702);  convolution_10 = unsqueeze_702 = None
    mul_739: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_708);  sub_217 = unsqueeze_708 = None
    sub_218: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(where_27, mul_739);  where_27 = mul_739 = None
    sub_219: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(sub_218, unsqueeze_705);  sub_218 = unsqueeze_705 = None
    mul_740: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_711);  sub_219 = unsqueeze_711 = None
    mul_741: "f32[72]" = torch.ops.aten.mul.Tensor(sum_85, squeeze_31);  sum_85 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_740, relu_6, primals_115, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 72, [True, True, False]);  mul_740 = primals_115 = None
    getitem_227: "f32[8, 72, 56, 56]" = convolution_backward_41[0]
    getitem_228: "f32[72, 1, 3, 3]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_120: "f32[8, 72, 56, 56]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_121: "f32[8, 72, 56, 56]" = torch.ops.aten.alias.default(alias_120);  alias_120 = None
    le_28: "b8[8, 72, 56, 56]" = torch.ops.aten.le.Scalar(alias_121, 0);  alias_121 = None
    scalar_tensor_28: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_28: "f32[8, 72, 56, 56]" = torch.ops.aten.where.self(le_28, scalar_tensor_28, getitem_227);  le_28 = scalar_tensor_28 = getitem_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_712: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_713: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, 2);  unsqueeze_712 = None
    unsqueeze_714: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 3);  unsqueeze_713 = None
    sum_86: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    sub_220: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_714)
    mul_742: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_28, sub_220);  sub_220 = None
    sum_87: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_742, [0, 2, 3]);  mul_742 = None
    mul_743: "f32[72]" = torch.ops.aten.mul.Tensor(sum_86, 3.985969387755102e-05)
    unsqueeze_715: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_743, 0);  mul_743 = None
    unsqueeze_716: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_715, 2);  unsqueeze_715 = None
    unsqueeze_717: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 3);  unsqueeze_716 = None
    mul_744: "f32[72]" = torch.ops.aten.mul.Tensor(sum_87, 3.985969387755102e-05)
    mul_745: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_746: "f32[72]" = torch.ops.aten.mul.Tensor(mul_744, mul_745);  mul_744 = mul_745 = None
    unsqueeze_718: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_746, 0);  mul_746 = None
    unsqueeze_719: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 2);  unsqueeze_718 = None
    unsqueeze_720: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 3);  unsqueeze_719 = None
    mul_747: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_19);  primals_19 = None
    unsqueeze_721: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_747, 0);  mul_747 = None
    unsqueeze_722: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 2);  unsqueeze_721 = None
    unsqueeze_723: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 3);  unsqueeze_722 = None
    sub_221: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_714);  convolution_9 = unsqueeze_714 = None
    mul_748: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_221, unsqueeze_720);  sub_221 = unsqueeze_720 = None
    sub_222: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(where_28, mul_748);  where_28 = mul_748 = None
    sub_223: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(sub_222, unsqueeze_717);  sub_222 = unsqueeze_717 = None
    mul_749: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_223, unsqueeze_723);  sub_223 = unsqueeze_723 = None
    mul_750: "f32[72]" = torch.ops.aten.mul.Tensor(sum_87, squeeze_28);  sum_87 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_749, add_45, primals_114, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_749 = add_45 = primals_114 = None
    getitem_230: "f32[8, 24, 56, 56]" = convolution_backward_42[0]
    getitem_231: "f32[72, 24, 1, 1]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_278: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(getitem_221, getitem_230);  getitem_221 = getitem_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_724: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_725: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, 2);  unsqueeze_724 = None
    unsqueeze_726: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 3);  unsqueeze_725 = None
    sum_88: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_278, [0, 2, 3])
    sub_224: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_726)
    mul_751: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_278, sub_224);  sub_224 = None
    sum_89: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_751, [0, 2, 3]);  mul_751 = None
    mul_752: "f32[24]" = torch.ops.aten.mul.Tensor(sum_88, 3.985969387755102e-05)
    unsqueeze_727: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_752, 0);  mul_752 = None
    unsqueeze_728: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_727, 2);  unsqueeze_727 = None
    unsqueeze_729: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 3);  unsqueeze_728 = None
    mul_753: "f32[24]" = torch.ops.aten.mul.Tensor(sum_89, 3.985969387755102e-05)
    mul_754: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_755: "f32[24]" = torch.ops.aten.mul.Tensor(mul_753, mul_754);  mul_753 = mul_754 = None
    unsqueeze_730: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_755, 0);  mul_755 = None
    unsqueeze_731: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, 2);  unsqueeze_730 = None
    unsqueeze_732: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 3);  unsqueeze_731 = None
    mul_756: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_17);  primals_17 = None
    unsqueeze_733: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_756, 0);  mul_756 = None
    unsqueeze_734: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 2);  unsqueeze_733 = None
    unsqueeze_735: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 3);  unsqueeze_734 = None
    sub_225: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_726);  convolution_8 = unsqueeze_726 = None
    mul_757: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_732);  sub_225 = unsqueeze_732 = None
    sub_226: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(add_278, mul_757);  mul_757 = None
    sub_227: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_226, unsqueeze_729);  sub_226 = unsqueeze_729 = None
    mul_758: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_735);  sub_227 = unsqueeze_735 = None
    mul_759: "f32[24]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_25);  sum_89 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_758, relu_5, primals_113, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_758 = primals_113 = None
    getitem_233: "f32[8, 72, 56, 56]" = convolution_backward_43[0]
    getitem_234: "f32[24, 72, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_123: "f32[8, 72, 56, 56]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_124: "f32[8, 72, 56, 56]" = torch.ops.aten.alias.default(alias_123);  alias_123 = None
    le_29: "b8[8, 72, 56, 56]" = torch.ops.aten.le.Scalar(alias_124, 0);  alias_124 = None
    scalar_tensor_29: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_29: "f32[8, 72, 56, 56]" = torch.ops.aten.where.self(le_29, scalar_tensor_29, getitem_233);  le_29 = scalar_tensor_29 = getitem_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_736: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_737: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, 2);  unsqueeze_736 = None
    unsqueeze_738: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 3);  unsqueeze_737 = None
    sum_90: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    sub_228: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_738)
    mul_760: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_29, sub_228);  sub_228 = None
    sum_91: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_760, [0, 2, 3]);  mul_760 = None
    mul_761: "f32[72]" = torch.ops.aten.mul.Tensor(sum_90, 3.985969387755102e-05)
    unsqueeze_739: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_761, 0);  mul_761 = None
    unsqueeze_740: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_739, 2);  unsqueeze_739 = None
    unsqueeze_741: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 3);  unsqueeze_740 = None
    mul_762: "f32[72]" = torch.ops.aten.mul.Tensor(sum_91, 3.985969387755102e-05)
    mul_763: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_764: "f32[72]" = torch.ops.aten.mul.Tensor(mul_762, mul_763);  mul_762 = mul_763 = None
    unsqueeze_742: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_764, 0);  mul_764 = None
    unsqueeze_743: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, 2);  unsqueeze_742 = None
    unsqueeze_744: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 3);  unsqueeze_743 = None
    mul_765: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_15);  primals_15 = None
    unsqueeze_745: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_765, 0);  mul_765 = None
    unsqueeze_746: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_745, 2);  unsqueeze_745 = None
    unsqueeze_747: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 3);  unsqueeze_746 = None
    sub_229: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_738);  convolution_7 = unsqueeze_738 = None
    mul_766: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_744);  sub_229 = unsqueeze_744 = None
    sub_230: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(where_29, mul_766);  where_29 = mul_766 = None
    sub_231: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(sub_230, unsqueeze_741);  sub_230 = unsqueeze_741 = None
    mul_767: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_231, unsqueeze_747);  sub_231 = unsqueeze_747 = None
    mul_768: "f32[72]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_22);  sum_91 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_767, relu_4, primals_112, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 72, [True, True, False]);  mul_767 = primals_112 = None
    getitem_236: "f32[8, 72, 56, 56]" = convolution_backward_44[0]
    getitem_237: "f32[72, 1, 3, 3]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_126: "f32[8, 72, 56, 56]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_127: "f32[8, 72, 56, 56]" = torch.ops.aten.alias.default(alias_126);  alias_126 = None
    le_30: "b8[8, 72, 56, 56]" = torch.ops.aten.le.Scalar(alias_127, 0);  alias_127 = None
    scalar_tensor_30: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_30: "f32[8, 72, 56, 56]" = torch.ops.aten.where.self(le_30, scalar_tensor_30, getitem_236);  le_30 = scalar_tensor_30 = getitem_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_748: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_749: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, 2);  unsqueeze_748 = None
    unsqueeze_750: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 3);  unsqueeze_749 = None
    sum_92: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
    sub_232: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_750)
    mul_769: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_30, sub_232);  sub_232 = None
    sum_93: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_769, [0, 2, 3]);  mul_769 = None
    mul_770: "f32[72]" = torch.ops.aten.mul.Tensor(sum_92, 3.985969387755102e-05)
    unsqueeze_751: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_770, 0);  mul_770 = None
    unsqueeze_752: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_751, 2);  unsqueeze_751 = None
    unsqueeze_753: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 3);  unsqueeze_752 = None
    mul_771: "f32[72]" = torch.ops.aten.mul.Tensor(sum_93, 3.985969387755102e-05)
    mul_772: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_773: "f32[72]" = torch.ops.aten.mul.Tensor(mul_771, mul_772);  mul_771 = mul_772 = None
    unsqueeze_754: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_773, 0);  mul_773 = None
    unsqueeze_755: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, 2);  unsqueeze_754 = None
    unsqueeze_756: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 3);  unsqueeze_755 = None
    mul_774: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_13);  primals_13 = None
    unsqueeze_757: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_774, 0);  mul_774 = None
    unsqueeze_758: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_757, 2);  unsqueeze_757 = None
    unsqueeze_759: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 3);  unsqueeze_758 = None
    sub_233: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_750);  convolution_6 = unsqueeze_750 = None
    mul_775: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_233, unsqueeze_756);  sub_233 = unsqueeze_756 = None
    sub_234: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(where_30, mul_775);  where_30 = mul_775 = None
    sub_235: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(sub_234, unsqueeze_753);  sub_234 = unsqueeze_753 = None
    mul_776: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_235, unsqueeze_759);  sub_235 = unsqueeze_759 = None
    mul_777: "f32[72]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_19);  sum_93 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_776, add_29, primals_111, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_776 = add_29 = primals_111 = None
    getitem_239: "f32[8, 24, 56, 56]" = convolution_backward_45[0]
    getitem_240: "f32[72, 24, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_279: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(add_278, getitem_239);  add_278 = getitem_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_760: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_761: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, 2);  unsqueeze_760 = None
    unsqueeze_762: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 3);  unsqueeze_761 = None
    sum_94: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_279, [0, 2, 3])
    sub_236: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_762)
    mul_778: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_279, sub_236);  sub_236 = None
    sum_95: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_778, [0, 2, 3]);  mul_778 = None
    mul_779: "f32[24]" = torch.ops.aten.mul.Tensor(sum_94, 3.985969387755102e-05)
    unsqueeze_763: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_779, 0);  mul_779 = None
    unsqueeze_764: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_763, 2);  unsqueeze_763 = None
    unsqueeze_765: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 3);  unsqueeze_764 = None
    mul_780: "f32[24]" = torch.ops.aten.mul.Tensor(sum_95, 3.985969387755102e-05)
    mul_781: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_782: "f32[24]" = torch.ops.aten.mul.Tensor(mul_780, mul_781);  mul_780 = mul_781 = None
    unsqueeze_766: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_782, 0);  mul_782 = None
    unsqueeze_767: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, 2);  unsqueeze_766 = None
    unsqueeze_768: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 3);  unsqueeze_767 = None
    mul_783: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_11);  primals_11 = None
    unsqueeze_769: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_783, 0);  mul_783 = None
    unsqueeze_770: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_769, 2);  unsqueeze_769 = None
    unsqueeze_771: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 3);  unsqueeze_770 = None
    sub_237: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_762);  convolution_5 = unsqueeze_762 = None
    mul_784: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_768);  sub_237 = unsqueeze_768 = None
    sub_238: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(add_279, mul_784);  add_279 = mul_784 = None
    sub_239: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_238, unsqueeze_765);  sub_238 = unsqueeze_765 = None
    mul_785: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_239, unsqueeze_771);  sub_239 = unsqueeze_771 = None
    mul_786: "f32[24]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_16);  sum_95 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_785, relu_3, primals_110, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_785 = primals_110 = None
    getitem_242: "f32[8, 48, 56, 56]" = convolution_backward_46[0]
    getitem_243: "f32[24, 48, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_129: "f32[8, 48, 56, 56]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_130: "f32[8, 48, 56, 56]" = torch.ops.aten.alias.default(alias_129);  alias_129 = None
    le_31: "b8[8, 48, 56, 56]" = torch.ops.aten.le.Scalar(alias_130, 0);  alias_130 = None
    scalar_tensor_31: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_31: "f32[8, 48, 56, 56]" = torch.ops.aten.where.self(le_31, scalar_tensor_31, getitem_242);  le_31 = scalar_tensor_31 = getitem_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_772: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_773: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, 2);  unsqueeze_772 = None
    unsqueeze_774: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 3);  unsqueeze_773 = None
    sum_96: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_240: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_774)
    mul_787: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(where_31, sub_240);  sub_240 = None
    sum_97: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_787, [0, 2, 3]);  mul_787 = None
    mul_788: "f32[48]" = torch.ops.aten.mul.Tensor(sum_96, 3.985969387755102e-05)
    unsqueeze_775: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_788, 0);  mul_788 = None
    unsqueeze_776: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_775, 2);  unsqueeze_775 = None
    unsqueeze_777: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 3);  unsqueeze_776 = None
    mul_789: "f32[48]" = torch.ops.aten.mul.Tensor(sum_97, 3.985969387755102e-05)
    mul_790: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_791: "f32[48]" = torch.ops.aten.mul.Tensor(mul_789, mul_790);  mul_789 = mul_790 = None
    unsqueeze_778: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_791, 0);  mul_791 = None
    unsqueeze_779: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, 2);  unsqueeze_778 = None
    unsqueeze_780: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 3);  unsqueeze_779 = None
    mul_792: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_9);  primals_9 = None
    unsqueeze_781: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_792, 0);  mul_792 = None
    unsqueeze_782: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_781, 2);  unsqueeze_781 = None
    unsqueeze_783: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 3);  unsqueeze_782 = None
    sub_241: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_774);  convolution_4 = unsqueeze_774 = None
    mul_793: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_241, unsqueeze_780);  sub_241 = unsqueeze_780 = None
    sub_242: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(where_31, mul_793);  where_31 = mul_793 = None
    sub_243: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(sub_242, unsqueeze_777);  sub_242 = unsqueeze_777 = None
    mul_794: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_783);  sub_243 = unsqueeze_783 = None
    mul_795: "f32[48]" = torch.ops.aten.mul.Tensor(sum_97, squeeze_13);  sum_97 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_794, relu_2, primals_109, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 48, [True, True, False]);  mul_794 = primals_109 = None
    getitem_245: "f32[8, 48, 112, 112]" = convolution_backward_47[0]
    getitem_246: "f32[48, 1, 3, 3]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_132: "f32[8, 48, 112, 112]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_133: "f32[8, 48, 112, 112]" = torch.ops.aten.alias.default(alias_132);  alias_132 = None
    le_32: "b8[8, 48, 112, 112]" = torch.ops.aten.le.Scalar(alias_133, 0);  alias_133 = None
    scalar_tensor_32: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_32: "f32[8, 48, 112, 112]" = torch.ops.aten.where.self(le_32, scalar_tensor_32, getitem_245);  le_32 = scalar_tensor_32 = getitem_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_784: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_785: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, 2);  unsqueeze_784 = None
    unsqueeze_786: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 3);  unsqueeze_785 = None
    sum_98: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
    sub_244: "f32[8, 48, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_786)
    mul_796: "f32[8, 48, 112, 112]" = torch.ops.aten.mul.Tensor(where_32, sub_244);  sub_244 = None
    sum_99: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_796, [0, 2, 3]);  mul_796 = None
    mul_797: "f32[48]" = torch.ops.aten.mul.Tensor(sum_98, 9.964923469387754e-06)
    unsqueeze_787: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_797, 0);  mul_797 = None
    unsqueeze_788: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_787, 2);  unsqueeze_787 = None
    unsqueeze_789: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 3);  unsqueeze_788 = None
    mul_798: "f32[48]" = torch.ops.aten.mul.Tensor(sum_99, 9.964923469387754e-06)
    mul_799: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_800: "f32[48]" = torch.ops.aten.mul.Tensor(mul_798, mul_799);  mul_798 = mul_799 = None
    unsqueeze_790: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_800, 0);  mul_800 = None
    unsqueeze_791: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, 2);  unsqueeze_790 = None
    unsqueeze_792: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_791, 3);  unsqueeze_791 = None
    mul_801: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_7);  primals_7 = None
    unsqueeze_793: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_801, 0);  mul_801 = None
    unsqueeze_794: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_793, 2);  unsqueeze_793 = None
    unsqueeze_795: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 3);  unsqueeze_794 = None
    sub_245: "f32[8, 48, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_786);  convolution_3 = unsqueeze_786 = None
    mul_802: "f32[8, 48, 112, 112]" = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_792);  sub_245 = unsqueeze_792 = None
    sub_246: "f32[8, 48, 112, 112]" = torch.ops.aten.sub.Tensor(where_32, mul_802);  where_32 = mul_802 = None
    sub_247: "f32[8, 48, 112, 112]" = torch.ops.aten.sub.Tensor(sub_246, unsqueeze_789);  sub_246 = unsqueeze_789 = None
    mul_803: "f32[8, 48, 112, 112]" = torch.ops.aten.mul.Tensor(sub_247, unsqueeze_795);  sub_247 = unsqueeze_795 = None
    mul_804: "f32[48]" = torch.ops.aten.mul.Tensor(sum_99, squeeze_10);  sum_99 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_803, add_14, primals_108, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_803 = add_14 = primals_108 = None
    getitem_248: "f32[8, 16, 112, 112]" = convolution_backward_48[0]
    getitem_249: "f32[48, 16, 1, 1]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_796: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_797: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, 2);  unsqueeze_796 = None
    unsqueeze_798: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 3);  unsqueeze_797 = None
    sum_100: "f32[16]" = torch.ops.aten.sum.dim_IntList(getitem_248, [0, 2, 3])
    sub_248: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_798)
    mul_805: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_248, sub_248);  sub_248 = None
    sum_101: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_805, [0, 2, 3]);  mul_805 = None
    mul_806: "f32[16]" = torch.ops.aten.mul.Tensor(sum_100, 9.964923469387754e-06)
    unsqueeze_799: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_806, 0);  mul_806 = None
    unsqueeze_800: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_799, 2);  unsqueeze_799 = None
    unsqueeze_801: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 3);  unsqueeze_800 = None
    mul_807: "f32[16]" = torch.ops.aten.mul.Tensor(sum_101, 9.964923469387754e-06)
    mul_808: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_809: "f32[16]" = torch.ops.aten.mul.Tensor(mul_807, mul_808);  mul_807 = mul_808 = None
    unsqueeze_802: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_809, 0);  mul_809 = None
    unsqueeze_803: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_802, 2);  unsqueeze_802 = None
    unsqueeze_804: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_803, 3);  unsqueeze_803 = None
    mul_810: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_5);  primals_5 = None
    unsqueeze_805: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_810, 0);  mul_810 = None
    unsqueeze_806: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_805, 2);  unsqueeze_805 = None
    unsqueeze_807: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 3);  unsqueeze_806 = None
    sub_249: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_798);  convolution_2 = unsqueeze_798 = None
    mul_811: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_249, unsqueeze_804);  sub_249 = unsqueeze_804 = None
    sub_250: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(getitem_248, mul_811);  getitem_248 = mul_811 = None
    sub_251: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_250, unsqueeze_801);  sub_250 = unsqueeze_801 = None
    mul_812: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_251, unsqueeze_807);  sub_251 = unsqueeze_807 = None
    mul_813: "f32[16]" = torch.ops.aten.mul.Tensor(sum_101, squeeze_7);  sum_101 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_812, relu_1, primals_107, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_812 = primals_107 = None
    getitem_251: "f32[8, 32, 112, 112]" = convolution_backward_49[0]
    getitem_252: "f32[16, 32, 1, 1]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_135: "f32[8, 32, 112, 112]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_136: "f32[8, 32, 112, 112]" = torch.ops.aten.alias.default(alias_135);  alias_135 = None
    le_33: "b8[8, 32, 112, 112]" = torch.ops.aten.le.Scalar(alias_136, 0);  alias_136 = None
    scalar_tensor_33: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_33: "f32[8, 32, 112, 112]" = torch.ops.aten.where.self(le_33, scalar_tensor_33, getitem_251);  le_33 = scalar_tensor_33 = getitem_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_808: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_809: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, 2);  unsqueeze_808 = None
    unsqueeze_810: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 3);  unsqueeze_809 = None
    sum_102: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    sub_252: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_810)
    mul_814: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_33, sub_252);  sub_252 = None
    sum_103: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_814, [0, 2, 3]);  mul_814 = None
    mul_815: "f32[32]" = torch.ops.aten.mul.Tensor(sum_102, 9.964923469387754e-06)
    unsqueeze_811: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_815, 0);  mul_815 = None
    unsqueeze_812: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_811, 2);  unsqueeze_811 = None
    unsqueeze_813: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 3);  unsqueeze_812 = None
    mul_816: "f32[32]" = torch.ops.aten.mul.Tensor(sum_103, 9.964923469387754e-06)
    mul_817: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_818: "f32[32]" = torch.ops.aten.mul.Tensor(mul_816, mul_817);  mul_816 = mul_817 = None
    unsqueeze_814: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_818, 0);  mul_818 = None
    unsqueeze_815: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_814, 2);  unsqueeze_814 = None
    unsqueeze_816: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_815, 3);  unsqueeze_815 = None
    mul_819: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_3);  primals_3 = None
    unsqueeze_817: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_819, 0);  mul_819 = None
    unsqueeze_818: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_817, 2);  unsqueeze_817 = None
    unsqueeze_819: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 3);  unsqueeze_818 = None
    sub_253: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_810);  convolution_1 = unsqueeze_810 = None
    mul_820: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_816);  sub_253 = unsqueeze_816 = None
    sub_254: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(where_33, mul_820);  where_33 = mul_820 = None
    sub_255: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_254, unsqueeze_813);  sub_254 = unsqueeze_813 = None
    mul_821: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_819);  sub_255 = unsqueeze_819 = None
    mul_822: "f32[32]" = torch.ops.aten.mul.Tensor(sum_103, squeeze_4);  sum_103 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_821, relu, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_821 = primals_106 = None
    getitem_254: "f32[8, 32, 112, 112]" = convolution_backward_50[0]
    getitem_255: "f32[32, 1, 3, 3]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_138: "f32[8, 32, 112, 112]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_139: "f32[8, 32, 112, 112]" = torch.ops.aten.alias.default(alias_138);  alias_138 = None
    le_34: "b8[8, 32, 112, 112]" = torch.ops.aten.le.Scalar(alias_139, 0);  alias_139 = None
    scalar_tensor_34: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_34: "f32[8, 32, 112, 112]" = torch.ops.aten.where.self(le_34, scalar_tensor_34, getitem_254);  le_34 = scalar_tensor_34 = getitem_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_820: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_821: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, 2);  unsqueeze_820 = None
    unsqueeze_822: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 3);  unsqueeze_821 = None
    sum_104: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
    sub_256: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_822)
    mul_823: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_34, sub_256);  sub_256 = None
    sum_105: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_823, [0, 2, 3]);  mul_823 = None
    mul_824: "f32[32]" = torch.ops.aten.mul.Tensor(sum_104, 9.964923469387754e-06)
    unsqueeze_823: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_824, 0);  mul_824 = None
    unsqueeze_824: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_823, 2);  unsqueeze_823 = None
    unsqueeze_825: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 3);  unsqueeze_824 = None
    mul_825: "f32[32]" = torch.ops.aten.mul.Tensor(sum_105, 9.964923469387754e-06)
    mul_826: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_827: "f32[32]" = torch.ops.aten.mul.Tensor(mul_825, mul_826);  mul_825 = mul_826 = None
    unsqueeze_826: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_827, 0);  mul_827 = None
    unsqueeze_827: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, 2);  unsqueeze_826 = None
    unsqueeze_828: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_827, 3);  unsqueeze_827 = None
    mul_828: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_1);  primals_1 = None
    unsqueeze_829: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_828, 0);  mul_828 = None
    unsqueeze_830: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_829, 2);  unsqueeze_829 = None
    unsqueeze_831: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 3);  unsqueeze_830 = None
    sub_257: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_822);  convolution = unsqueeze_822 = None
    mul_829: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_828);  sub_257 = unsqueeze_828 = None
    sub_258: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(where_34, mul_829);  where_34 = mul_829 = None
    sub_259: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_258, unsqueeze_825);  sub_258 = unsqueeze_825 = None
    mul_830: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_259, unsqueeze_831);  sub_259 = unsqueeze_831 = None
    mul_831: "f32[32]" = torch.ops.aten.mul.Tensor(sum_105, squeeze_1);  sum_105 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:162, code: x = self.conv_stem(x)
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_830, primals_315, primals_105, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_830 = primals_315 = primals_105 = None
    getitem_258: "f32[32, 3, 3, 3]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # No stacktrace found for following nodes
    copy_: "i64[]" = torch.ops.aten.copy_.default(primals_159, add);  primals_159 = add = None
    copy__1: "f32[32]" = torch.ops.aten.copy_.default(primals_160, add_2);  primals_160 = add_2 = None
    copy__2: "f32[32]" = torch.ops.aten.copy_.default(primals_161, add_3);  primals_161 = add_3 = None
    copy__3: "i64[]" = torch.ops.aten.copy_.default(primals_162, add_5);  primals_162 = add_5 = None
    copy__4: "f32[32]" = torch.ops.aten.copy_.default(primals_163, add_7);  primals_163 = add_7 = None
    copy__5: "f32[32]" = torch.ops.aten.copy_.default(primals_164, add_8);  primals_164 = add_8 = None
    copy__6: "i64[]" = torch.ops.aten.copy_.default(primals_165, add_10);  primals_165 = add_10 = None
    copy__7: "f32[16]" = torch.ops.aten.copy_.default(primals_166, add_12);  primals_166 = add_12 = None
    copy__8: "f32[16]" = torch.ops.aten.copy_.default(primals_167, add_13);  primals_167 = add_13 = None
    copy__9: "i64[]" = torch.ops.aten.copy_.default(primals_168, add_15);  primals_168 = add_15 = None
    copy__10: "f32[48]" = torch.ops.aten.copy_.default(primals_169, add_17);  primals_169 = add_17 = None
    copy__11: "f32[48]" = torch.ops.aten.copy_.default(primals_170, add_18);  primals_170 = add_18 = None
    copy__12: "i64[]" = torch.ops.aten.copy_.default(primals_171, add_20);  primals_171 = add_20 = None
    copy__13: "f32[48]" = torch.ops.aten.copy_.default(primals_172, add_22);  primals_172 = add_22 = None
    copy__14: "f32[48]" = torch.ops.aten.copy_.default(primals_173, add_23);  primals_173 = add_23 = None
    copy__15: "i64[]" = torch.ops.aten.copy_.default(primals_174, add_25);  primals_174 = add_25 = None
    copy__16: "f32[24]" = torch.ops.aten.copy_.default(primals_175, add_27);  primals_175 = add_27 = None
    copy__17: "f32[24]" = torch.ops.aten.copy_.default(primals_176, add_28);  primals_176 = add_28 = None
    copy__18: "i64[]" = torch.ops.aten.copy_.default(primals_177, add_30);  primals_177 = add_30 = None
    copy__19: "f32[72]" = torch.ops.aten.copy_.default(primals_178, add_32);  primals_178 = add_32 = None
    copy__20: "f32[72]" = torch.ops.aten.copy_.default(primals_179, add_33);  primals_179 = add_33 = None
    copy__21: "i64[]" = torch.ops.aten.copy_.default(primals_180, add_35);  primals_180 = add_35 = None
    copy__22: "f32[72]" = torch.ops.aten.copy_.default(primals_181, add_37);  primals_181 = add_37 = None
    copy__23: "f32[72]" = torch.ops.aten.copy_.default(primals_182, add_38);  primals_182 = add_38 = None
    copy__24: "i64[]" = torch.ops.aten.copy_.default(primals_183, add_40);  primals_183 = add_40 = None
    copy__25: "f32[24]" = torch.ops.aten.copy_.default(primals_184, add_42);  primals_184 = add_42 = None
    copy__26: "f32[24]" = torch.ops.aten.copy_.default(primals_185, add_43);  primals_185 = add_43 = None
    copy__27: "i64[]" = torch.ops.aten.copy_.default(primals_186, add_46);  primals_186 = add_46 = None
    copy__28: "f32[72]" = torch.ops.aten.copy_.default(primals_187, add_48);  primals_187 = add_48 = None
    copy__29: "f32[72]" = torch.ops.aten.copy_.default(primals_188, add_49);  primals_188 = add_49 = None
    copy__30: "i64[]" = torch.ops.aten.copy_.default(primals_189, add_51);  primals_189 = add_51 = None
    copy__31: "f32[72]" = torch.ops.aten.copy_.default(primals_190, add_53);  primals_190 = add_53 = None
    copy__32: "f32[72]" = torch.ops.aten.copy_.default(primals_191, add_54);  primals_191 = add_54 = None
    copy__33: "i64[]" = torch.ops.aten.copy_.default(primals_192, add_56);  primals_192 = add_56 = None
    copy__34: "f32[24]" = torch.ops.aten.copy_.default(primals_193, add_58);  primals_193 = add_58 = None
    copy__35: "f32[24]" = torch.ops.aten.copy_.default(primals_194, add_59);  primals_194 = add_59 = None
    copy__36: "i64[]" = torch.ops.aten.copy_.default(primals_195, add_62);  primals_195 = add_62 = None
    copy__37: "f32[72]" = torch.ops.aten.copy_.default(primals_196, add_64);  primals_196 = add_64 = None
    copy__38: "f32[72]" = torch.ops.aten.copy_.default(primals_197, add_65);  primals_197 = add_65 = None
    copy__39: "i64[]" = torch.ops.aten.copy_.default(primals_198, add_67);  primals_198 = add_67 = None
    copy__40: "f32[72]" = torch.ops.aten.copy_.default(primals_199, add_69);  primals_199 = add_69 = None
    copy__41: "f32[72]" = torch.ops.aten.copy_.default(primals_200, add_70);  primals_200 = add_70 = None
    copy__42: "i64[]" = torch.ops.aten.copy_.default(primals_201, add_72);  primals_201 = add_72 = None
    copy__43: "f32[40]" = torch.ops.aten.copy_.default(primals_202, add_74);  primals_202 = add_74 = None
    copy__44: "f32[40]" = torch.ops.aten.copy_.default(primals_203, add_75);  primals_203 = add_75 = None
    copy__45: "i64[]" = torch.ops.aten.copy_.default(primals_204, add_77);  primals_204 = add_77 = None
    copy__46: "f32[120]" = torch.ops.aten.copy_.default(primals_205, add_79);  primals_205 = add_79 = None
    copy__47: "f32[120]" = torch.ops.aten.copy_.default(primals_206, add_80);  primals_206 = add_80 = None
    copy__48: "i64[]" = torch.ops.aten.copy_.default(primals_207, add_82);  primals_207 = add_82 = None
    copy__49: "f32[120]" = torch.ops.aten.copy_.default(primals_208, add_84);  primals_208 = add_84 = None
    copy__50: "f32[120]" = torch.ops.aten.copy_.default(primals_209, add_85);  primals_209 = add_85 = None
    copy__51: "i64[]" = torch.ops.aten.copy_.default(primals_210, add_87);  primals_210 = add_87 = None
    copy__52: "f32[40]" = torch.ops.aten.copy_.default(primals_211, add_89);  primals_211 = add_89 = None
    copy__53: "f32[40]" = torch.ops.aten.copy_.default(primals_212, add_90);  primals_212 = add_90 = None
    copy__54: "i64[]" = torch.ops.aten.copy_.default(primals_213, add_93);  primals_213 = add_93 = None
    copy__55: "f32[120]" = torch.ops.aten.copy_.default(primals_214, add_95);  primals_214 = add_95 = None
    copy__56: "f32[120]" = torch.ops.aten.copy_.default(primals_215, add_96);  primals_215 = add_96 = None
    copy__57: "i64[]" = torch.ops.aten.copy_.default(primals_216, add_98);  primals_216 = add_98 = None
    copy__58: "f32[120]" = torch.ops.aten.copy_.default(primals_217, add_100);  primals_217 = add_100 = None
    copy__59: "f32[120]" = torch.ops.aten.copy_.default(primals_218, add_101);  primals_218 = add_101 = None
    copy__60: "i64[]" = torch.ops.aten.copy_.default(primals_219, add_103);  primals_219 = add_103 = None
    copy__61: "f32[40]" = torch.ops.aten.copy_.default(primals_220, add_105);  primals_220 = add_105 = None
    copy__62: "f32[40]" = torch.ops.aten.copy_.default(primals_221, add_106);  primals_221 = add_106 = None
    copy__63: "i64[]" = torch.ops.aten.copy_.default(primals_222, add_109);  primals_222 = add_109 = None
    copy__64: "f32[240]" = torch.ops.aten.copy_.default(primals_223, add_111);  primals_223 = add_111 = None
    copy__65: "f32[240]" = torch.ops.aten.copy_.default(primals_224, add_112);  primals_224 = add_112 = None
    copy__66: "i64[]" = torch.ops.aten.copy_.default(primals_225, add_114);  primals_225 = add_114 = None
    copy__67: "f32[240]" = torch.ops.aten.copy_.default(primals_226, add_116);  primals_226 = add_116 = None
    copy__68: "f32[240]" = torch.ops.aten.copy_.default(primals_227, add_117);  primals_227 = add_117 = None
    copy__69: "i64[]" = torch.ops.aten.copy_.default(primals_228, add_119);  primals_228 = add_119 = None
    copy__70: "f32[80]" = torch.ops.aten.copy_.default(primals_229, add_121);  primals_229 = add_121 = None
    copy__71: "f32[80]" = torch.ops.aten.copy_.default(primals_230, add_122);  primals_230 = add_122 = None
    copy__72: "i64[]" = torch.ops.aten.copy_.default(primals_231, add_124);  primals_231 = add_124 = None
    copy__73: "f32[480]" = torch.ops.aten.copy_.default(primals_232, add_126);  primals_232 = add_126 = None
    copy__74: "f32[480]" = torch.ops.aten.copy_.default(primals_233, add_127);  primals_233 = add_127 = None
    copy__75: "i64[]" = torch.ops.aten.copy_.default(primals_234, add_129);  primals_234 = add_129 = None
    copy__76: "f32[480]" = torch.ops.aten.copy_.default(primals_235, add_131);  primals_235 = add_131 = None
    copy__77: "f32[480]" = torch.ops.aten.copy_.default(primals_236, add_132);  primals_236 = add_132 = None
    copy__78: "i64[]" = torch.ops.aten.copy_.default(primals_237, add_134);  primals_237 = add_134 = None
    copy__79: "f32[80]" = torch.ops.aten.copy_.default(primals_238, add_136);  primals_238 = add_136 = None
    copy__80: "f32[80]" = torch.ops.aten.copy_.default(primals_239, add_137);  primals_239 = add_137 = None
    copy__81: "i64[]" = torch.ops.aten.copy_.default(primals_240, add_140);  primals_240 = add_140 = None
    copy__82: "f32[480]" = torch.ops.aten.copy_.default(primals_241, add_142);  primals_241 = add_142 = None
    copy__83: "f32[480]" = torch.ops.aten.copy_.default(primals_242, add_143);  primals_242 = add_143 = None
    copy__84: "i64[]" = torch.ops.aten.copy_.default(primals_243, add_145);  primals_243 = add_145 = None
    copy__85: "f32[480]" = torch.ops.aten.copy_.default(primals_244, add_147);  primals_244 = add_147 = None
    copy__86: "f32[480]" = torch.ops.aten.copy_.default(primals_245, add_148);  primals_245 = add_148 = None
    copy__87: "i64[]" = torch.ops.aten.copy_.default(primals_246, add_150);  primals_246 = add_150 = None
    copy__88: "f32[80]" = torch.ops.aten.copy_.default(primals_247, add_152);  primals_247 = add_152 = None
    copy__89: "f32[80]" = torch.ops.aten.copy_.default(primals_248, add_153);  primals_248 = add_153 = None
    copy__90: "i64[]" = torch.ops.aten.copy_.default(primals_249, add_156);  primals_249 = add_156 = None
    copy__91: "f32[480]" = torch.ops.aten.copy_.default(primals_250, add_158);  primals_250 = add_158 = None
    copy__92: "f32[480]" = torch.ops.aten.copy_.default(primals_251, add_159);  primals_251 = add_159 = None
    copy__93: "i64[]" = torch.ops.aten.copy_.default(primals_252, add_161);  primals_252 = add_161 = None
    copy__94: "f32[480]" = torch.ops.aten.copy_.default(primals_253, add_163);  primals_253 = add_163 = None
    copy__95: "f32[480]" = torch.ops.aten.copy_.default(primals_254, add_164);  primals_254 = add_164 = None
    copy__96: "i64[]" = torch.ops.aten.copy_.default(primals_255, add_166);  primals_255 = add_166 = None
    copy__97: "f32[96]" = torch.ops.aten.copy_.default(primals_256, add_168);  primals_256 = add_168 = None
    copy__98: "f32[96]" = torch.ops.aten.copy_.default(primals_257, add_169);  primals_257 = add_169 = None
    copy__99: "i64[]" = torch.ops.aten.copy_.default(primals_258, add_171);  primals_258 = add_171 = None
    copy__100: "f32[576]" = torch.ops.aten.copy_.default(primals_259, add_173);  primals_259 = add_173 = None
    copy__101: "f32[576]" = torch.ops.aten.copy_.default(primals_260, add_174);  primals_260 = add_174 = None
    copy__102: "i64[]" = torch.ops.aten.copy_.default(primals_261, add_176);  primals_261 = add_176 = None
    copy__103: "f32[576]" = torch.ops.aten.copy_.default(primals_262, add_178);  primals_262 = add_178 = None
    copy__104: "f32[576]" = torch.ops.aten.copy_.default(primals_263, add_179);  primals_263 = add_179 = None
    copy__105: "i64[]" = torch.ops.aten.copy_.default(primals_264, add_181);  primals_264 = add_181 = None
    copy__106: "f32[96]" = torch.ops.aten.copy_.default(primals_265, add_183);  primals_265 = add_183 = None
    copy__107: "f32[96]" = torch.ops.aten.copy_.default(primals_266, add_184);  primals_266 = add_184 = None
    copy__108: "i64[]" = torch.ops.aten.copy_.default(primals_267, add_187);  primals_267 = add_187 = None
    copy__109: "f32[576]" = torch.ops.aten.copy_.default(primals_268, add_189);  primals_268 = add_189 = None
    copy__110: "f32[576]" = torch.ops.aten.copy_.default(primals_269, add_190);  primals_269 = add_190 = None
    copy__111: "i64[]" = torch.ops.aten.copy_.default(primals_270, add_192);  primals_270 = add_192 = None
    copy__112: "f32[576]" = torch.ops.aten.copy_.default(primals_271, add_194);  primals_271 = add_194 = None
    copy__113: "f32[576]" = torch.ops.aten.copy_.default(primals_272, add_195);  primals_272 = add_195 = None
    copy__114: "i64[]" = torch.ops.aten.copy_.default(primals_273, add_197);  primals_273 = add_197 = None
    copy__115: "f32[192]" = torch.ops.aten.copy_.default(primals_274, add_199);  primals_274 = add_199 = None
    copy__116: "f32[192]" = torch.ops.aten.copy_.default(primals_275, add_200);  primals_275 = add_200 = None
    copy__117: "i64[]" = torch.ops.aten.copy_.default(primals_276, add_202);  primals_276 = add_202 = None
    copy__118: "f32[1152]" = torch.ops.aten.copy_.default(primals_277, add_204);  primals_277 = add_204 = None
    copy__119: "f32[1152]" = torch.ops.aten.copy_.default(primals_278, add_205);  primals_278 = add_205 = None
    copy__120: "i64[]" = torch.ops.aten.copy_.default(primals_279, add_207);  primals_279 = add_207 = None
    copy__121: "f32[1152]" = torch.ops.aten.copy_.default(primals_280, add_209);  primals_280 = add_209 = None
    copy__122: "f32[1152]" = torch.ops.aten.copy_.default(primals_281, add_210);  primals_281 = add_210 = None
    copy__123: "i64[]" = torch.ops.aten.copy_.default(primals_282, add_212);  primals_282 = add_212 = None
    copy__124: "f32[192]" = torch.ops.aten.copy_.default(primals_283, add_214);  primals_283 = add_214 = None
    copy__125: "f32[192]" = torch.ops.aten.copy_.default(primals_284, add_215);  primals_284 = add_215 = None
    copy__126: "i64[]" = torch.ops.aten.copy_.default(primals_285, add_218);  primals_285 = add_218 = None
    copy__127: "f32[1152]" = torch.ops.aten.copy_.default(primals_286, add_220);  primals_286 = add_220 = None
    copy__128: "f32[1152]" = torch.ops.aten.copy_.default(primals_287, add_221);  primals_287 = add_221 = None
    copy__129: "i64[]" = torch.ops.aten.copy_.default(primals_288, add_223);  primals_288 = add_223 = None
    copy__130: "f32[1152]" = torch.ops.aten.copy_.default(primals_289, add_225);  primals_289 = add_225 = None
    copy__131: "f32[1152]" = torch.ops.aten.copy_.default(primals_290, add_226);  primals_290 = add_226 = None
    copy__132: "i64[]" = torch.ops.aten.copy_.default(primals_291, add_228);  primals_291 = add_228 = None
    copy__133: "f32[192]" = torch.ops.aten.copy_.default(primals_292, add_230);  primals_292 = add_230 = None
    copy__134: "f32[192]" = torch.ops.aten.copy_.default(primals_293, add_231);  primals_293 = add_231 = None
    copy__135: "i64[]" = torch.ops.aten.copy_.default(primals_294, add_234);  primals_294 = add_234 = None
    copy__136: "f32[1152]" = torch.ops.aten.copy_.default(primals_295, add_236);  primals_295 = add_236 = None
    copy__137: "f32[1152]" = torch.ops.aten.copy_.default(primals_296, add_237);  primals_296 = add_237 = None
    copy__138: "i64[]" = torch.ops.aten.copy_.default(primals_297, add_239);  primals_297 = add_239 = None
    copy__139: "f32[1152]" = torch.ops.aten.copy_.default(primals_298, add_241);  primals_298 = add_241 = None
    copy__140: "f32[1152]" = torch.ops.aten.copy_.default(primals_299, add_242);  primals_299 = add_242 = None
    copy__141: "i64[]" = torch.ops.aten.copy_.default(primals_300, add_244);  primals_300 = add_244 = None
    copy__142: "f32[192]" = torch.ops.aten.copy_.default(primals_301, add_246);  primals_301 = add_246 = None
    copy__143: "f32[192]" = torch.ops.aten.copy_.default(primals_302, add_247);  primals_302 = add_247 = None
    copy__144: "i64[]" = torch.ops.aten.copy_.default(primals_303, add_250);  primals_303 = add_250 = None
    copy__145: "f32[1152]" = torch.ops.aten.copy_.default(primals_304, add_252);  primals_304 = add_252 = None
    copy__146: "f32[1152]" = torch.ops.aten.copy_.default(primals_305, add_253);  primals_305 = add_253 = None
    copy__147: "i64[]" = torch.ops.aten.copy_.default(primals_306, add_255);  primals_306 = add_255 = None
    copy__148: "f32[1152]" = torch.ops.aten.copy_.default(primals_307, add_257);  primals_307 = add_257 = None
    copy__149: "f32[1152]" = torch.ops.aten.copy_.default(primals_308, add_258);  primals_308 = add_258 = None
    copy__150: "i64[]" = torch.ops.aten.copy_.default(primals_309, add_260);  primals_309 = add_260 = None
    copy__151: "f32[320]" = torch.ops.aten.copy_.default(primals_310, add_262);  primals_310 = add_262 = None
    copy__152: "f32[320]" = torch.ops.aten.copy_.default(primals_311, add_263);  primals_311 = add_263 = None
    copy__153: "i64[]" = torch.ops.aten.copy_.default(primals_312, add_265);  primals_312 = add_265 = None
    copy__154: "f32[1280]" = torch.ops.aten.copy_.default(primals_313, add_267);  primals_313 = add_267 = None
    copy__155: "f32[1280]" = torch.ops.aten.copy_.default(primals_314, add_268);  primals_314 = add_268 = None
    return pytree.tree_unflatten([addmm, mul_831, sum_104, mul_822, sum_102, mul_813, sum_100, mul_804, sum_98, mul_795, sum_96, mul_786, sum_94, mul_777, sum_92, mul_768, sum_90, mul_759, sum_88, mul_750, sum_86, mul_741, sum_84, mul_732, sum_82, mul_723, sum_80, mul_714, sum_78, mul_705, sum_76, mul_696, sum_74, mul_687, sum_72, mul_678, sum_70, mul_669, sum_68, mul_660, sum_66, mul_651, sum_64, mul_642, sum_62, mul_633, sum_60, mul_624, sum_58, mul_615, sum_56, mul_606, sum_54, mul_597, sum_52, mul_588, sum_50, mul_579, sum_48, mul_570, sum_46, mul_561, sum_44, mul_552, sum_42, mul_543, sum_40, mul_534, sum_38, mul_525, sum_36, mul_516, sum_34, mul_507, sum_32, mul_498, sum_30, mul_489, sum_28, mul_480, sum_26, mul_471, sum_24, mul_462, sum_22, mul_453, sum_20, mul_444, sum_18, mul_435, sum_16, mul_426, sum_14, mul_417, sum_12, mul_408, sum_10, mul_399, sum_8, mul_390, sum_6, mul_381, sum_4, mul_372, sum_2, getitem_258, getitem_255, getitem_252, getitem_249, getitem_246, getitem_243, getitem_240, getitem_237, getitem_234, getitem_231, getitem_228, getitem_225, getitem_222, getitem_219, getitem_216, getitem_213, getitem_210, getitem_207, getitem_204, getitem_201, getitem_198, getitem_195, getitem_192, getitem_189, getitem_186, getitem_183, getitem_180, getitem_177, getitem_174, getitem_171, getitem_168, getitem_165, getitem_162, getitem_159, getitem_156, getitem_153, getitem_150, getitem_147, getitem_144, getitem_141, getitem_138, getitem_135, getitem_132, getitem_129, getitem_126, getitem_123, getitem_120, getitem_117, getitem_114, getitem_111, getitem_108, getitem_105, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    