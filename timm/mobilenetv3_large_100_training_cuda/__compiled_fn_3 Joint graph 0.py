from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[16]"; primals_2: "f32[16]"; primals_3: "f32[16]"; primals_4: "f32[16]"; primals_5: "f32[16]"; primals_6: "f32[16]"; primals_7: "f32[64]"; primals_8: "f32[64]"; primals_9: "f32[64]"; primals_10: "f32[64]"; primals_11: "f32[24]"; primals_12: "f32[24]"; primals_13: "f32[72]"; primals_14: "f32[72]"; primals_15: "f32[72]"; primals_16: "f32[72]"; primals_17: "f32[24]"; primals_18: "f32[24]"; primals_19: "f32[72]"; primals_20: "f32[72]"; primals_21: "f32[72]"; primals_22: "f32[72]"; primals_23: "f32[40]"; primals_24: "f32[40]"; primals_25: "f32[120]"; primals_26: "f32[120]"; primals_27: "f32[120]"; primals_28: "f32[120]"; primals_29: "f32[40]"; primals_30: "f32[40]"; primals_31: "f32[120]"; primals_32: "f32[120]"; primals_33: "f32[120]"; primals_34: "f32[120]"; primals_35: "f32[40]"; primals_36: "f32[40]"; primals_37: "f32[240]"; primals_38: "f32[240]"; primals_39: "f32[240]"; primals_40: "f32[240]"; primals_41: "f32[80]"; primals_42: "f32[80]"; primals_43: "f32[200]"; primals_44: "f32[200]"; primals_45: "f32[200]"; primals_46: "f32[200]"; primals_47: "f32[80]"; primals_48: "f32[80]"; primals_49: "f32[184]"; primals_50: "f32[184]"; primals_51: "f32[184]"; primals_52: "f32[184]"; primals_53: "f32[80]"; primals_54: "f32[80]"; primals_55: "f32[184]"; primals_56: "f32[184]"; primals_57: "f32[184]"; primals_58: "f32[184]"; primals_59: "f32[80]"; primals_60: "f32[80]"; primals_61: "f32[480]"; primals_62: "f32[480]"; primals_63: "f32[480]"; primals_64: "f32[480]"; primals_65: "f32[112]"; primals_66: "f32[112]"; primals_67: "f32[672]"; primals_68: "f32[672]"; primals_69: "f32[672]"; primals_70: "f32[672]"; primals_71: "f32[112]"; primals_72: "f32[112]"; primals_73: "f32[672]"; primals_74: "f32[672]"; primals_75: "f32[672]"; primals_76: "f32[672]"; primals_77: "f32[160]"; primals_78: "f32[160]"; primals_79: "f32[960]"; primals_80: "f32[960]"; primals_81: "f32[960]"; primals_82: "f32[960]"; primals_83: "f32[160]"; primals_84: "f32[160]"; primals_85: "f32[960]"; primals_86: "f32[960]"; primals_87: "f32[960]"; primals_88: "f32[960]"; primals_89: "f32[160]"; primals_90: "f32[160]"; primals_91: "f32[960]"; primals_92: "f32[960]"; primals_93: "f32[1000, 1280]"; primals_94: "f32[1000]"; primals_95: "f32[16, 3, 3, 3]"; primals_96: "f32[16, 1, 3, 3]"; primals_97: "f32[16, 16, 1, 1]"; primals_98: "f32[64, 16, 1, 1]"; primals_99: "f32[64, 1, 3, 3]"; primals_100: "f32[24, 64, 1, 1]"; primals_101: "f32[72, 24, 1, 1]"; primals_102: "f32[72, 1, 3, 3]"; primals_103: "f32[24, 72, 1, 1]"; primals_104: "f32[72, 24, 1, 1]"; primals_105: "f32[72, 1, 5, 5]"; primals_106: "f32[24, 72, 1, 1]"; primals_107: "f32[24]"; primals_108: "f32[72, 24, 1, 1]"; primals_109: "f32[72]"; primals_110: "f32[40, 72, 1, 1]"; primals_111: "f32[120, 40, 1, 1]"; primals_112: "f32[120, 1, 5, 5]"; primals_113: "f32[32, 120, 1, 1]"; primals_114: "f32[32]"; primals_115: "f32[120, 32, 1, 1]"; primals_116: "f32[120]"; primals_117: "f32[40, 120, 1, 1]"; primals_118: "f32[120, 40, 1, 1]"; primals_119: "f32[120, 1, 5, 5]"; primals_120: "f32[32, 120, 1, 1]"; primals_121: "f32[32]"; primals_122: "f32[120, 32, 1, 1]"; primals_123: "f32[120]"; primals_124: "f32[40, 120, 1, 1]"; primals_125: "f32[240, 40, 1, 1]"; primals_126: "f32[240, 1, 3, 3]"; primals_127: "f32[80, 240, 1, 1]"; primals_128: "f32[200, 80, 1, 1]"; primals_129: "f32[200, 1, 3, 3]"; primals_130: "f32[80, 200, 1, 1]"; primals_131: "f32[184, 80, 1, 1]"; primals_132: "f32[184, 1, 3, 3]"; primals_133: "f32[80, 184, 1, 1]"; primals_134: "f32[184, 80, 1, 1]"; primals_135: "f32[184, 1, 3, 3]"; primals_136: "f32[80, 184, 1, 1]"; primals_137: "f32[480, 80, 1, 1]"; primals_138: "f32[480, 1, 3, 3]"; primals_139: "f32[120, 480, 1, 1]"; primals_140: "f32[120]"; primals_141: "f32[480, 120, 1, 1]"; primals_142: "f32[480]"; primals_143: "f32[112, 480, 1, 1]"; primals_144: "f32[672, 112, 1, 1]"; primals_145: "f32[672, 1, 3, 3]"; primals_146: "f32[168, 672, 1, 1]"; primals_147: "f32[168]"; primals_148: "f32[672, 168, 1, 1]"; primals_149: "f32[672]"; primals_150: "f32[112, 672, 1, 1]"; primals_151: "f32[672, 112, 1, 1]"; primals_152: "f32[672, 1, 5, 5]"; primals_153: "f32[168, 672, 1, 1]"; primals_154: "f32[168]"; primals_155: "f32[672, 168, 1, 1]"; primals_156: "f32[672]"; primals_157: "f32[160, 672, 1, 1]"; primals_158: "f32[960, 160, 1, 1]"; primals_159: "f32[960, 1, 5, 5]"; primals_160: "f32[240, 960, 1, 1]"; primals_161: "f32[240]"; primals_162: "f32[960, 240, 1, 1]"; primals_163: "f32[960]"; primals_164: "f32[160, 960, 1, 1]"; primals_165: "f32[960, 160, 1, 1]"; primals_166: "f32[960, 1, 5, 5]"; primals_167: "f32[240, 960, 1, 1]"; primals_168: "f32[240]"; primals_169: "f32[960, 240, 1, 1]"; primals_170: "f32[960]"; primals_171: "f32[160, 960, 1, 1]"; primals_172: "f32[960, 160, 1, 1]"; primals_173: "f32[1280, 960, 1, 1]"; primals_174: "f32[1280]"; primals_175: "i64[]"; primals_176: "f32[16]"; primals_177: "f32[16]"; primals_178: "i64[]"; primals_179: "f32[16]"; primals_180: "f32[16]"; primals_181: "i64[]"; primals_182: "f32[16]"; primals_183: "f32[16]"; primals_184: "i64[]"; primals_185: "f32[64]"; primals_186: "f32[64]"; primals_187: "i64[]"; primals_188: "f32[64]"; primals_189: "f32[64]"; primals_190: "i64[]"; primals_191: "f32[24]"; primals_192: "f32[24]"; primals_193: "i64[]"; primals_194: "f32[72]"; primals_195: "f32[72]"; primals_196: "i64[]"; primals_197: "f32[72]"; primals_198: "f32[72]"; primals_199: "i64[]"; primals_200: "f32[24]"; primals_201: "f32[24]"; primals_202: "i64[]"; primals_203: "f32[72]"; primals_204: "f32[72]"; primals_205: "i64[]"; primals_206: "f32[72]"; primals_207: "f32[72]"; primals_208: "i64[]"; primals_209: "f32[40]"; primals_210: "f32[40]"; primals_211: "i64[]"; primals_212: "f32[120]"; primals_213: "f32[120]"; primals_214: "i64[]"; primals_215: "f32[120]"; primals_216: "f32[120]"; primals_217: "i64[]"; primals_218: "f32[40]"; primals_219: "f32[40]"; primals_220: "i64[]"; primals_221: "f32[120]"; primals_222: "f32[120]"; primals_223: "i64[]"; primals_224: "f32[120]"; primals_225: "f32[120]"; primals_226: "i64[]"; primals_227: "f32[40]"; primals_228: "f32[40]"; primals_229: "i64[]"; primals_230: "f32[240]"; primals_231: "f32[240]"; primals_232: "i64[]"; primals_233: "f32[240]"; primals_234: "f32[240]"; primals_235: "i64[]"; primals_236: "f32[80]"; primals_237: "f32[80]"; primals_238: "i64[]"; primals_239: "f32[200]"; primals_240: "f32[200]"; primals_241: "i64[]"; primals_242: "f32[200]"; primals_243: "f32[200]"; primals_244: "i64[]"; primals_245: "f32[80]"; primals_246: "f32[80]"; primals_247: "i64[]"; primals_248: "f32[184]"; primals_249: "f32[184]"; primals_250: "i64[]"; primals_251: "f32[184]"; primals_252: "f32[184]"; primals_253: "i64[]"; primals_254: "f32[80]"; primals_255: "f32[80]"; primals_256: "i64[]"; primals_257: "f32[184]"; primals_258: "f32[184]"; primals_259: "i64[]"; primals_260: "f32[184]"; primals_261: "f32[184]"; primals_262: "i64[]"; primals_263: "f32[80]"; primals_264: "f32[80]"; primals_265: "i64[]"; primals_266: "f32[480]"; primals_267: "f32[480]"; primals_268: "i64[]"; primals_269: "f32[480]"; primals_270: "f32[480]"; primals_271: "i64[]"; primals_272: "f32[112]"; primals_273: "f32[112]"; primals_274: "i64[]"; primals_275: "f32[672]"; primals_276: "f32[672]"; primals_277: "i64[]"; primals_278: "f32[672]"; primals_279: "f32[672]"; primals_280: "i64[]"; primals_281: "f32[112]"; primals_282: "f32[112]"; primals_283: "i64[]"; primals_284: "f32[672]"; primals_285: "f32[672]"; primals_286: "i64[]"; primals_287: "f32[672]"; primals_288: "f32[672]"; primals_289: "i64[]"; primals_290: "f32[160]"; primals_291: "f32[160]"; primals_292: "i64[]"; primals_293: "f32[960]"; primals_294: "f32[960]"; primals_295: "i64[]"; primals_296: "f32[960]"; primals_297: "f32[960]"; primals_298: "i64[]"; primals_299: "f32[160]"; primals_300: "f32[160]"; primals_301: "i64[]"; primals_302: "f32[960]"; primals_303: "f32[960]"; primals_304: "i64[]"; primals_305: "f32[960]"; primals_306: "f32[960]"; primals_307: "i64[]"; primals_308: "f32[160]"; primals_309: "f32[160]"; primals_310: "i64[]"; primals_311: "f32[960]"; primals_312: "f32[960]"; primals_313: "f32[8, 3, 224, 224]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:135, code: x = self.conv_stem(x)
    convolution: "f32[8, 16, 112, 112]" = torch.ops.aten.convolution.default(primals_313, primals_95, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_175, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
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
    mul_2: "f32[16]" = torch.ops.aten.mul.Tensor(primals_176, 0.9)
    add_2: "f32[16]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[16]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.00000996502277);  squeeze_2 = None
    mul_4: "f32[16]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[16]" = torch.ops.aten.mul.Tensor(primals_177, 0.9)
    add_3: "f32[16]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_1, -1)
    unsqueeze_1: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1);  primals_2 = None
    unsqueeze_3: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone: "f32[8, 16, 112, 112]" = torch.ops.aten.clone.default(add_4)
    add_5: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(add_4, 3)
    clamp_min: "f32[8, 16, 112, 112]" = torch.ops.aten.clamp_min.default(add_5, 0);  add_5 = None
    clamp_max: "f32[8, 16, 112, 112]" = torch.ops.aten.clamp_max.default(clamp_min, 6);  clamp_min = None
    mul_7: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(add_4, clamp_max);  add_4 = clamp_max = None
    div: "f32[8, 16, 112, 112]" = torch.ops.aten.div.Tensor(mul_7, 6);  mul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_1: "f32[8, 16, 112, 112]" = torch.ops.aten.convolution.default(div, primals_96, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_6: "i64[]" = torch.ops.aten.add.Tensor(primals_178, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[1, 16, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 16, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_7: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
    rsqrt_1: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    sub_1: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
    mul_8: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
    squeeze_4: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_9: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_10: "f32[16]" = torch.ops.aten.mul.Tensor(primals_179, 0.9)
    add_8: "f32[16]" = torch.ops.aten.add.Tensor(mul_9, mul_10);  mul_9 = mul_10 = None
    squeeze_5: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_11: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.00000996502277);  squeeze_5 = None
    mul_12: "f32[16]" = torch.ops.aten.mul.Tensor(mul_11, 0.1);  mul_11 = None
    mul_13: "f32[16]" = torch.ops.aten.mul.Tensor(primals_180, 0.9)
    add_9: "f32[16]" = torch.ops.aten.add.Tensor(mul_12, mul_13);  mul_12 = mul_13 = None
    unsqueeze_4: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1)
    unsqueeze_5: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_14: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_8, unsqueeze_5);  mul_8 = unsqueeze_5 = None
    unsqueeze_6: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_4, -1);  primals_4 = None
    unsqueeze_7: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_10: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_7);  mul_14 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu: "f32[8, 16, 112, 112]" = torch.ops.aten.relu.default(add_10);  add_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_2: "f32[8, 16, 112, 112]" = torch.ops.aten.convolution.default(relu, primals_97, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_11: "i64[]" = torch.ops.aten.add.Tensor(primals_181, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 16, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 16, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_12: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
    rsqrt_2: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    sub_2: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
    mul_15: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_16: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_17: "f32[16]" = torch.ops.aten.mul.Tensor(primals_182, 0.9)
    add_13: "f32[16]" = torch.ops.aten.add.Tensor(mul_16, mul_17);  mul_16 = mul_17 = None
    squeeze_8: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_18: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.00000996502277);  squeeze_8 = None
    mul_19: "f32[16]" = torch.ops.aten.mul.Tensor(mul_18, 0.1);  mul_18 = None
    mul_20: "f32[16]" = torch.ops.aten.mul.Tensor(primals_183, 0.9)
    add_14: "f32[16]" = torch.ops.aten.add.Tensor(mul_19, mul_20);  mul_19 = mul_20 = None
    unsqueeze_8: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_9: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_21: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_15, unsqueeze_9);  mul_15 = unsqueeze_9 = None
    unsqueeze_10: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_11: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_15: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_21, unsqueeze_11);  mul_21 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:129, code: x = self.drop_path(x) + shortcut
    add_16: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(add_15, div);  add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_3: "f32[8, 64, 112, 112]" = torch.ops.aten.convolution.default(add_16, primals_98, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_17: "i64[]" = torch.ops.aten.add.Tensor(primals_184, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
    getitem_6: "f32[1, 64, 1, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 64, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_18: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
    rsqrt_3: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_3: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_7)
    mul_22: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
    squeeze_10: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_23: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_24: "f32[64]" = torch.ops.aten.mul.Tensor(primals_185, 0.9)
    add_19: "f32[64]" = torch.ops.aten.add.Tensor(mul_23, mul_24);  mul_23 = mul_24 = None
    squeeze_11: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
    mul_25: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.00000996502277);  squeeze_11 = None
    mul_26: "f32[64]" = torch.ops.aten.mul.Tensor(mul_25, 0.1);  mul_25 = None
    mul_27: "f32[64]" = torch.ops.aten.mul.Tensor(primals_186, 0.9)
    add_20: "f32[64]" = torch.ops.aten.add.Tensor(mul_26, mul_27);  mul_26 = mul_27 = None
    unsqueeze_12: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1)
    unsqueeze_13: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_28: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_13);  mul_22 = unsqueeze_13 = None
    unsqueeze_14: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1);  primals_8 = None
    unsqueeze_15: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_21: "f32[8, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_28, unsqueeze_15);  mul_28 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_1: "f32[8, 64, 112, 112]" = torch.ops.aten.relu.default(add_21);  add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_4: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_1, primals_99, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_22: "i64[]" = torch.ops.aten.add.Tensor(primals_187, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
    getitem_8: "f32[1, 64, 1, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 64, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_23: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
    rsqrt_4: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
    sub_4: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_9)
    mul_29: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
    squeeze_13: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
    mul_30: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_31: "f32[64]" = torch.ops.aten.mul.Tensor(primals_188, 0.9)
    add_24: "f32[64]" = torch.ops.aten.add.Tensor(mul_30, mul_31);  mul_30 = mul_31 = None
    squeeze_14: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
    mul_32: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000398612827361);  squeeze_14 = None
    mul_33: "f32[64]" = torch.ops.aten.mul.Tensor(mul_32, 0.1);  mul_32 = None
    mul_34: "f32[64]" = torch.ops.aten.mul.Tensor(primals_189, 0.9)
    add_25: "f32[64]" = torch.ops.aten.add.Tensor(mul_33, mul_34);  mul_33 = mul_34 = None
    unsqueeze_16: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1)
    unsqueeze_17: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    mul_35: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_29, unsqueeze_17);  mul_29 = unsqueeze_17 = None
    unsqueeze_18: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_10, -1);  primals_10 = None
    unsqueeze_19: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    add_26: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_19);  mul_35 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_2: "f32[8, 64, 56, 56]" = torch.ops.aten.relu.default(add_26);  add_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_5: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(relu_2, primals_100, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_27: "i64[]" = torch.ops.aten.add.Tensor(primals_190, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
    getitem_10: "f32[1, 24, 1, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 24, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_28: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
    rsqrt_5: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_5: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_11)
    mul_36: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_15: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
    squeeze_16: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    mul_37: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_38: "f32[24]" = torch.ops.aten.mul.Tensor(primals_191, 0.9)
    add_29: "f32[24]" = torch.ops.aten.add.Tensor(mul_37, mul_38);  mul_37 = mul_38 = None
    squeeze_17: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
    mul_39: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000398612827361);  squeeze_17 = None
    mul_40: "f32[24]" = torch.ops.aten.mul.Tensor(mul_39, 0.1);  mul_39 = None
    mul_41: "f32[24]" = torch.ops.aten.mul.Tensor(primals_192, 0.9)
    add_30: "f32[24]" = torch.ops.aten.add.Tensor(mul_40, mul_41);  mul_40 = mul_41 = None
    unsqueeze_20: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_21: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_42: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_36, unsqueeze_21);  mul_36 = unsqueeze_21 = None
    unsqueeze_22: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_23: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_31: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_42, unsqueeze_23);  mul_42 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_6: "f32[8, 72, 56, 56]" = torch.ops.aten.convolution.default(add_31, primals_101, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_32: "i64[]" = torch.ops.aten.add.Tensor(primals_193, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_6 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
    getitem_12: "f32[1, 72, 1, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 72, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_33: "f32[1, 72, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
    rsqrt_6: "f32[1, 72, 1, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
    sub_6: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_13)
    mul_43: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_18: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
    squeeze_19: "f32[72]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
    mul_44: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_45: "f32[72]" = torch.ops.aten.mul.Tensor(primals_194, 0.9)
    add_34: "f32[72]" = torch.ops.aten.add.Tensor(mul_44, mul_45);  mul_44 = mul_45 = None
    squeeze_20: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
    mul_46: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0000398612827361);  squeeze_20 = None
    mul_47: "f32[72]" = torch.ops.aten.mul.Tensor(mul_46, 0.1);  mul_46 = None
    mul_48: "f32[72]" = torch.ops.aten.mul.Tensor(primals_195, 0.9)
    add_35: "f32[72]" = torch.ops.aten.add.Tensor(mul_47, mul_48);  mul_47 = mul_48 = None
    unsqueeze_24: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1)
    unsqueeze_25: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    mul_49: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_25);  mul_43 = unsqueeze_25 = None
    unsqueeze_26: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1);  primals_14 = None
    unsqueeze_27: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    add_36: "f32[8, 72, 56, 56]" = torch.ops.aten.add.Tensor(mul_49, unsqueeze_27);  mul_49 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_3: "f32[8, 72, 56, 56]" = torch.ops.aten.relu.default(add_36);  add_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_7: "f32[8, 72, 56, 56]" = torch.ops.aten.convolution.default(relu_3, primals_102, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_37: "i64[]" = torch.ops.aten.add.Tensor(primals_196, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_7 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
    getitem_14: "f32[1, 72, 1, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 72, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_38: "f32[1, 72, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
    rsqrt_7: "f32[1, 72, 1, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_7: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_15)
    mul_50: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_21: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
    squeeze_22: "f32[72]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
    mul_51: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_52: "f32[72]" = torch.ops.aten.mul.Tensor(primals_197, 0.9)
    add_39: "f32[72]" = torch.ops.aten.add.Tensor(mul_51, mul_52);  mul_51 = mul_52 = None
    squeeze_23: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
    mul_53: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0000398612827361);  squeeze_23 = None
    mul_54: "f32[72]" = torch.ops.aten.mul.Tensor(mul_53, 0.1);  mul_53 = None
    mul_55: "f32[72]" = torch.ops.aten.mul.Tensor(primals_198, 0.9)
    add_40: "f32[72]" = torch.ops.aten.add.Tensor(mul_54, mul_55);  mul_54 = mul_55 = None
    unsqueeze_28: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1)
    unsqueeze_29: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_56: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(mul_50, unsqueeze_29);  mul_50 = unsqueeze_29 = None
    unsqueeze_30: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1);  primals_16 = None
    unsqueeze_31: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_41: "f32[8, 72, 56, 56]" = torch.ops.aten.add.Tensor(mul_56, unsqueeze_31);  mul_56 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_4: "f32[8, 72, 56, 56]" = torch.ops.aten.relu.default(add_41);  add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_8: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(relu_4, primals_103, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_42: "i64[]" = torch.ops.aten.add.Tensor(primals_199, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_8 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
    getitem_16: "f32[1, 24, 1, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 24, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_43: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
    rsqrt_8: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
    sub_8: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_17)
    mul_57: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_24: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
    squeeze_25: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
    mul_58: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_59: "f32[24]" = torch.ops.aten.mul.Tensor(primals_200, 0.9)
    add_44: "f32[24]" = torch.ops.aten.add.Tensor(mul_58, mul_59);  mul_58 = mul_59 = None
    squeeze_26: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
    mul_60: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0000398612827361);  squeeze_26 = None
    mul_61: "f32[24]" = torch.ops.aten.mul.Tensor(mul_60, 0.1);  mul_60 = None
    mul_62: "f32[24]" = torch.ops.aten.mul.Tensor(primals_201, 0.9)
    add_45: "f32[24]" = torch.ops.aten.add.Tensor(mul_61, mul_62);  mul_61 = mul_62 = None
    unsqueeze_32: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_33: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    mul_63: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_57, unsqueeze_33);  mul_57 = unsqueeze_33 = None
    unsqueeze_34: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_35: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    add_46: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_63, unsqueeze_35);  mul_63 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_47: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(add_46, add_31);  add_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_9: "f32[8, 72, 56, 56]" = torch.ops.aten.convolution.default(add_47, primals_104, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_48: "i64[]" = torch.ops.aten.add.Tensor(primals_202, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_9 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
    getitem_18: "f32[1, 72, 1, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 72, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_49: "f32[1, 72, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
    rsqrt_9: "f32[1, 72, 1, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_9: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_19)
    mul_64: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_27: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
    squeeze_28: "f32[72]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_65: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_66: "f32[72]" = torch.ops.aten.mul.Tensor(primals_203, 0.9)
    add_50: "f32[72]" = torch.ops.aten.add.Tensor(mul_65, mul_66);  mul_65 = mul_66 = None
    squeeze_29: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
    mul_67: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0000398612827361);  squeeze_29 = None
    mul_68: "f32[72]" = torch.ops.aten.mul.Tensor(mul_67, 0.1);  mul_67 = None
    mul_69: "f32[72]" = torch.ops.aten.mul.Tensor(primals_204, 0.9)
    add_51: "f32[72]" = torch.ops.aten.add.Tensor(mul_68, mul_69);  mul_68 = mul_69 = None
    unsqueeze_36: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
    unsqueeze_37: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_70: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(mul_64, unsqueeze_37);  mul_64 = unsqueeze_37 = None
    unsqueeze_38: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
    unsqueeze_39: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_52: "f32[8, 72, 56, 56]" = torch.ops.aten.add.Tensor(mul_70, unsqueeze_39);  mul_70 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_5: "f32[8, 72, 56, 56]" = torch.ops.aten.relu.default(add_52);  add_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_10: "f32[8, 72, 28, 28]" = torch.ops.aten.convolution.default(relu_5, primals_105, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_53: "i64[]" = torch.ops.aten.add.Tensor(primals_205, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_10 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[1, 72, 1, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 72, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_54: "f32[1, 72, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
    rsqrt_10: "f32[1, 72, 1, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
    sub_10: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_21)
    mul_71: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_30: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
    squeeze_31: "f32[72]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
    mul_72: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_73: "f32[72]" = torch.ops.aten.mul.Tensor(primals_206, 0.9)
    add_55: "f32[72]" = torch.ops.aten.add.Tensor(mul_72, mul_73);  mul_72 = mul_73 = None
    squeeze_32: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
    mul_74: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0001594642002871);  squeeze_32 = None
    mul_75: "f32[72]" = torch.ops.aten.mul.Tensor(mul_74, 0.1);  mul_74 = None
    mul_76: "f32[72]" = torch.ops.aten.mul.Tensor(primals_207, 0.9)
    add_56: "f32[72]" = torch.ops.aten.add.Tensor(mul_75, mul_76);  mul_75 = mul_76 = None
    unsqueeze_40: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1)
    unsqueeze_41: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    mul_77: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(mul_71, unsqueeze_41);  mul_71 = unsqueeze_41 = None
    unsqueeze_42: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1);  primals_22 = None
    unsqueeze_43: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    add_57: "f32[8, 72, 28, 28]" = torch.ops.aten.add.Tensor(mul_77, unsqueeze_43);  mul_77 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_6: "f32[8, 72, 28, 28]" = torch.ops.aten.relu.default(add_57);  add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean: "f32[8, 72, 1, 1]" = torch.ops.aten.mean.dim(relu_6, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_11: "f32[8, 24, 1, 1]" = torch.ops.aten.convolution.default(mean, primals_106, primals_107, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    relu_7: "f32[8, 24, 1, 1]" = torch.ops.aten.relu.default(convolution_11);  convolution_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_12: "f32[8, 72, 1, 1]" = torch.ops.aten.convolution.default(relu_7, primals_108, primals_109, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_58: "f32[8, 72, 1, 1]" = torch.ops.aten.add.Tensor(convolution_12, 3)
    clamp_min_1: "f32[8, 72, 1, 1]" = torch.ops.aten.clamp_min.default(add_58, 0);  add_58 = None
    clamp_max_1: "f32[8, 72, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_1, 6);  clamp_min_1 = None
    div_1: "f32[8, 72, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_1, 6);  clamp_max_1 = None
    mul_78: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(relu_6, div_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_13: "f32[8, 40, 28, 28]" = torch.ops.aten.convolution.default(mul_78, primals_110, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_59: "i64[]" = torch.ops.aten.add.Tensor(primals_208, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_11 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[1, 40, 1, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 40, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_60: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
    rsqrt_11: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_11: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_23)
    mul_79: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    squeeze_33: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
    squeeze_34: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
    mul_80: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_81: "f32[40]" = torch.ops.aten.mul.Tensor(primals_209, 0.9)
    add_61: "f32[40]" = torch.ops.aten.add.Tensor(mul_80, mul_81);  mul_80 = mul_81 = None
    squeeze_35: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
    mul_82: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0001594642002871);  squeeze_35 = None
    mul_83: "f32[40]" = torch.ops.aten.mul.Tensor(mul_82, 0.1);  mul_82 = None
    mul_84: "f32[40]" = torch.ops.aten.mul.Tensor(primals_210, 0.9)
    add_62: "f32[40]" = torch.ops.aten.add.Tensor(mul_83, mul_84);  mul_83 = mul_84 = None
    unsqueeze_44: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_45: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_85: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_79, unsqueeze_45);  mul_79 = unsqueeze_45 = None
    unsqueeze_46: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_47: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_63: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_85, unsqueeze_47);  mul_85 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_14: "f32[8, 120, 28, 28]" = torch.ops.aten.convolution.default(add_63, primals_111, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_64: "i64[]" = torch.ops.aten.add.Tensor(primals_211, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_12 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
    getitem_24: "f32[1, 120, 1, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 120, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_65: "f32[1, 120, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
    rsqrt_12: "f32[1, 120, 1, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
    sub_12: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_25)
    mul_86: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_36: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
    squeeze_37: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
    mul_87: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_88: "f32[120]" = torch.ops.aten.mul.Tensor(primals_212, 0.9)
    add_66: "f32[120]" = torch.ops.aten.add.Tensor(mul_87, mul_88);  mul_87 = mul_88 = None
    squeeze_38: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
    mul_89: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0001594642002871);  squeeze_38 = None
    mul_90: "f32[120]" = torch.ops.aten.mul.Tensor(mul_89, 0.1);  mul_89 = None
    mul_91: "f32[120]" = torch.ops.aten.mul.Tensor(primals_213, 0.9)
    add_67: "f32[120]" = torch.ops.aten.add.Tensor(mul_90, mul_91);  mul_90 = mul_91 = None
    unsqueeze_48: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1)
    unsqueeze_49: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    mul_92: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_86, unsqueeze_49);  mul_86 = unsqueeze_49 = None
    unsqueeze_50: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1);  primals_26 = None
    unsqueeze_51: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    add_68: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_92, unsqueeze_51);  mul_92 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_8: "f32[8, 120, 28, 28]" = torch.ops.aten.relu.default(add_68);  add_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_15: "f32[8, 120, 28, 28]" = torch.ops.aten.convolution.default(relu_8, primals_112, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_69: "i64[]" = torch.ops.aten.add.Tensor(primals_214, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_13 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
    getitem_26: "f32[1, 120, 1, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 120, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_70: "f32[1, 120, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
    rsqrt_13: "f32[1, 120, 1, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    sub_13: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_27)
    mul_93: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_39: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
    squeeze_40: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_94: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_95: "f32[120]" = torch.ops.aten.mul.Tensor(primals_215, 0.9)
    add_71: "f32[120]" = torch.ops.aten.add.Tensor(mul_94, mul_95);  mul_94 = mul_95 = None
    squeeze_41: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
    mul_96: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0001594642002871);  squeeze_41 = None
    mul_97: "f32[120]" = torch.ops.aten.mul.Tensor(mul_96, 0.1);  mul_96 = None
    mul_98: "f32[120]" = torch.ops.aten.mul.Tensor(primals_216, 0.9)
    add_72: "f32[120]" = torch.ops.aten.add.Tensor(mul_97, mul_98);  mul_97 = mul_98 = None
    unsqueeze_52: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1)
    unsqueeze_53: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_99: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_93, unsqueeze_53);  mul_93 = unsqueeze_53 = None
    unsqueeze_54: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1);  primals_28 = None
    unsqueeze_55: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_73: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_99, unsqueeze_55);  mul_99 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_9: "f32[8, 120, 28, 28]" = torch.ops.aten.relu.default(add_73);  add_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_1: "f32[8, 120, 1, 1]" = torch.ops.aten.mean.dim(relu_9, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_16: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_1, primals_113, primals_114, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    relu_10: "f32[8, 32, 1, 1]" = torch.ops.aten.relu.default(convolution_16);  convolution_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_17: "f32[8, 120, 1, 1]" = torch.ops.aten.convolution.default(relu_10, primals_115, primals_116, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_74: "f32[8, 120, 1, 1]" = torch.ops.aten.add.Tensor(convolution_17, 3)
    clamp_min_2: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_min.default(add_74, 0);  add_74 = None
    clamp_max_2: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_2, 6);  clamp_min_2 = None
    div_2: "f32[8, 120, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_2, 6);  clamp_max_2 = None
    mul_100: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(relu_9, div_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_18: "f32[8, 40, 28, 28]" = torch.ops.aten.convolution.default(mul_100, primals_117, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_75: "i64[]" = torch.ops.aten.add.Tensor(primals_217, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_14 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
    getitem_28: "f32[1, 40, 1, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 40, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_76: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
    rsqrt_14: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    sub_14: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_29)
    mul_101: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_42: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
    squeeze_43: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
    mul_102: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_103: "f32[40]" = torch.ops.aten.mul.Tensor(primals_218, 0.9)
    add_77: "f32[40]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
    squeeze_44: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
    mul_104: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0001594642002871);  squeeze_44 = None
    mul_105: "f32[40]" = torch.ops.aten.mul.Tensor(mul_104, 0.1);  mul_104 = None
    mul_106: "f32[40]" = torch.ops.aten.mul.Tensor(primals_219, 0.9)
    add_78: "f32[40]" = torch.ops.aten.add.Tensor(mul_105, mul_106);  mul_105 = mul_106 = None
    unsqueeze_56: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_57: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    mul_107: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_101, unsqueeze_57);  mul_101 = unsqueeze_57 = None
    unsqueeze_58: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_59: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    add_79: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_107, unsqueeze_59);  mul_107 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_80: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_79, add_63);  add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_19: "f32[8, 120, 28, 28]" = torch.ops.aten.convolution.default(add_80, primals_118, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_81: "i64[]" = torch.ops.aten.add.Tensor(primals_220, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_15 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
    getitem_30: "f32[1, 120, 1, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 120, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_82: "f32[1, 120, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
    rsqrt_15: "f32[1, 120, 1, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    sub_15: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_31)
    mul_108: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    squeeze_45: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
    squeeze_46: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
    mul_109: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_110: "f32[120]" = torch.ops.aten.mul.Tensor(primals_221, 0.9)
    add_83: "f32[120]" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
    squeeze_47: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
    mul_111: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0001594642002871);  squeeze_47 = None
    mul_112: "f32[120]" = torch.ops.aten.mul.Tensor(mul_111, 0.1);  mul_111 = None
    mul_113: "f32[120]" = torch.ops.aten.mul.Tensor(primals_222, 0.9)
    add_84: "f32[120]" = torch.ops.aten.add.Tensor(mul_112, mul_113);  mul_112 = mul_113 = None
    unsqueeze_60: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1)
    unsqueeze_61: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_114: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_108, unsqueeze_61);  mul_108 = unsqueeze_61 = None
    unsqueeze_62: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1);  primals_32 = None
    unsqueeze_63: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_85: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_114, unsqueeze_63);  mul_114 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_11: "f32[8, 120, 28, 28]" = torch.ops.aten.relu.default(add_85);  add_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_20: "f32[8, 120, 28, 28]" = torch.ops.aten.convolution.default(relu_11, primals_119, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_86: "i64[]" = torch.ops.aten.add.Tensor(primals_223, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_16 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
    getitem_32: "f32[1, 120, 1, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 120, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_87: "f32[1, 120, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
    rsqrt_16: "f32[1, 120, 1, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    sub_16: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_33)
    mul_115: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_48: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
    squeeze_49: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
    mul_116: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_117: "f32[120]" = torch.ops.aten.mul.Tensor(primals_224, 0.9)
    add_88: "f32[120]" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
    squeeze_50: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
    mul_118: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0001594642002871);  squeeze_50 = None
    mul_119: "f32[120]" = torch.ops.aten.mul.Tensor(mul_118, 0.1);  mul_118 = None
    mul_120: "f32[120]" = torch.ops.aten.mul.Tensor(primals_225, 0.9)
    add_89: "f32[120]" = torch.ops.aten.add.Tensor(mul_119, mul_120);  mul_119 = mul_120 = None
    unsqueeze_64: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1)
    unsqueeze_65: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    mul_121: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_115, unsqueeze_65);  mul_115 = unsqueeze_65 = None
    unsqueeze_66: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1);  primals_34 = None
    unsqueeze_67: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    add_90: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_121, unsqueeze_67);  mul_121 = unsqueeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_12: "f32[8, 120, 28, 28]" = torch.ops.aten.relu.default(add_90);  add_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_2: "f32[8, 120, 1, 1]" = torch.ops.aten.mean.dim(relu_12, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_21: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_2, primals_120, primals_121, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    relu_13: "f32[8, 32, 1, 1]" = torch.ops.aten.relu.default(convolution_21);  convolution_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_22: "f32[8, 120, 1, 1]" = torch.ops.aten.convolution.default(relu_13, primals_122, primals_123, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_91: "f32[8, 120, 1, 1]" = torch.ops.aten.add.Tensor(convolution_22, 3)
    clamp_min_3: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_min.default(add_91, 0);  add_91 = None
    clamp_max_3: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_3, 6);  clamp_min_3 = None
    div_3: "f32[8, 120, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_3, 6);  clamp_max_3 = None
    mul_122: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(relu_12, div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_23: "f32[8, 40, 28, 28]" = torch.ops.aten.convolution.default(mul_122, primals_124, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_92: "i64[]" = torch.ops.aten.add.Tensor(primals_226, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_17 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
    getitem_34: "f32[1, 40, 1, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 40, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_93: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
    rsqrt_17: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
    sub_17: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_35)
    mul_123: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    squeeze_51: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
    squeeze_52: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
    mul_124: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_125: "f32[40]" = torch.ops.aten.mul.Tensor(primals_227, 0.9)
    add_94: "f32[40]" = torch.ops.aten.add.Tensor(mul_124, mul_125);  mul_124 = mul_125 = None
    squeeze_53: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
    mul_126: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0001594642002871);  squeeze_53 = None
    mul_127: "f32[40]" = torch.ops.aten.mul.Tensor(mul_126, 0.1);  mul_126 = None
    mul_128: "f32[40]" = torch.ops.aten.mul.Tensor(primals_228, 0.9)
    add_95: "f32[40]" = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
    unsqueeze_68: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_69: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_129: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_123, unsqueeze_69);  mul_123 = unsqueeze_69 = None
    unsqueeze_70: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_71: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_96: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_129, unsqueeze_71);  mul_129 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_97: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_96, add_80);  add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_24: "f32[8, 240, 28, 28]" = torch.ops.aten.convolution.default(add_97, primals_125, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_98: "i64[]" = torch.ops.aten.add.Tensor(primals_229, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_18 = torch.ops.aten.var_mean.correction(convolution_24, [0, 2, 3], correction = 0, keepdim = True)
    getitem_36: "f32[1, 240, 1, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 240, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_99: "f32[1, 240, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
    rsqrt_18: "f32[1, 240, 1, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
    sub_18: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, getitem_37)
    mul_130: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    squeeze_54: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
    squeeze_55: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
    mul_131: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_132: "f32[240]" = torch.ops.aten.mul.Tensor(primals_230, 0.9)
    add_100: "f32[240]" = torch.ops.aten.add.Tensor(mul_131, mul_132);  mul_131 = mul_132 = None
    squeeze_56: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
    mul_133: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0001594642002871);  squeeze_56 = None
    mul_134: "f32[240]" = torch.ops.aten.mul.Tensor(mul_133, 0.1);  mul_133 = None
    mul_135: "f32[240]" = torch.ops.aten.mul.Tensor(primals_231, 0.9)
    add_101: "f32[240]" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
    unsqueeze_72: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1)
    unsqueeze_73: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    mul_136: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_130, unsqueeze_73);  mul_130 = unsqueeze_73 = None
    unsqueeze_74: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1);  primals_38 = None
    unsqueeze_75: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    add_102: "f32[8, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_136, unsqueeze_75);  mul_136 = unsqueeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_1: "f32[8, 240, 28, 28]" = torch.ops.aten.clone.default(add_102)
    add_103: "f32[8, 240, 28, 28]" = torch.ops.aten.add.Tensor(add_102, 3)
    clamp_min_4: "f32[8, 240, 28, 28]" = torch.ops.aten.clamp_min.default(add_103, 0);  add_103 = None
    clamp_max_4: "f32[8, 240, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_4, 6);  clamp_min_4 = None
    mul_137: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_102, clamp_max_4);  add_102 = clamp_max_4 = None
    div_4: "f32[8, 240, 28, 28]" = torch.ops.aten.div.Tensor(mul_137, 6);  mul_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_25: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(div_4, primals_126, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 240)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_104: "i64[]" = torch.ops.aten.add.Tensor(primals_232, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_19 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
    getitem_38: "f32[1, 240, 1, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 240, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_105: "f32[1, 240, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
    rsqrt_19: "f32[1, 240, 1, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    sub_19: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_39)
    mul_138: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    squeeze_57: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
    squeeze_58: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
    mul_139: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_140: "f32[240]" = torch.ops.aten.mul.Tensor(primals_233, 0.9)
    add_106: "f32[240]" = torch.ops.aten.add.Tensor(mul_139, mul_140);  mul_139 = mul_140 = None
    squeeze_59: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
    mul_141: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0006381620931717);  squeeze_59 = None
    mul_142: "f32[240]" = torch.ops.aten.mul.Tensor(mul_141, 0.1);  mul_141 = None
    mul_143: "f32[240]" = torch.ops.aten.mul.Tensor(primals_234, 0.9)
    add_107: "f32[240]" = torch.ops.aten.add.Tensor(mul_142, mul_143);  mul_142 = mul_143 = None
    unsqueeze_76: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1)
    unsqueeze_77: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_144: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_138, unsqueeze_77);  mul_138 = unsqueeze_77 = None
    unsqueeze_78: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1);  primals_40 = None
    unsqueeze_79: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_108: "f32[8, 240, 14, 14]" = torch.ops.aten.add.Tensor(mul_144, unsqueeze_79);  mul_144 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_2: "f32[8, 240, 14, 14]" = torch.ops.aten.clone.default(add_108)
    add_109: "f32[8, 240, 14, 14]" = torch.ops.aten.add.Tensor(add_108, 3)
    clamp_min_5: "f32[8, 240, 14, 14]" = torch.ops.aten.clamp_min.default(add_109, 0);  add_109 = None
    clamp_max_5: "f32[8, 240, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_5, 6);  clamp_min_5 = None
    mul_145: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(add_108, clamp_max_5);  add_108 = clamp_max_5 = None
    div_5: "f32[8, 240, 14, 14]" = torch.ops.aten.div.Tensor(mul_145, 6);  mul_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_26: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(div_5, primals_127, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_110: "i64[]" = torch.ops.aten.add.Tensor(primals_235, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_20 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
    getitem_40: "f32[1, 80, 1, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 80, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_111: "f32[1, 80, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
    rsqrt_20: "f32[1, 80, 1, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
    sub_20: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_41)
    mul_146: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    squeeze_60: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
    squeeze_61: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
    mul_147: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_148: "f32[80]" = torch.ops.aten.mul.Tensor(primals_236, 0.9)
    add_112: "f32[80]" = torch.ops.aten.add.Tensor(mul_147, mul_148);  mul_147 = mul_148 = None
    squeeze_62: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
    mul_149: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0006381620931717);  squeeze_62 = None
    mul_150: "f32[80]" = torch.ops.aten.mul.Tensor(mul_149, 0.1);  mul_149 = None
    mul_151: "f32[80]" = torch.ops.aten.mul.Tensor(primals_237, 0.9)
    add_113: "f32[80]" = torch.ops.aten.add.Tensor(mul_150, mul_151);  mul_150 = mul_151 = None
    unsqueeze_80: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_81: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    mul_152: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_146, unsqueeze_81);  mul_146 = unsqueeze_81 = None
    unsqueeze_82: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_83: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    add_114: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_152, unsqueeze_83);  mul_152 = unsqueeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_27: "f32[8, 200, 14, 14]" = torch.ops.aten.convolution.default(add_114, primals_128, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_115: "i64[]" = torch.ops.aten.add.Tensor(primals_238, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_21 = torch.ops.aten.var_mean.correction(convolution_27, [0, 2, 3], correction = 0, keepdim = True)
    getitem_42: "f32[1, 200, 1, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 200, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_116: "f32[1, 200, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
    rsqrt_21: "f32[1, 200, 1, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
    sub_21: "f32[8, 200, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, getitem_43)
    mul_153: "f32[8, 200, 14, 14]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    squeeze_63: "f32[200]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
    squeeze_64: "f32[200]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
    mul_154: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_155: "f32[200]" = torch.ops.aten.mul.Tensor(primals_239, 0.9)
    add_117: "f32[200]" = torch.ops.aten.add.Tensor(mul_154, mul_155);  mul_154 = mul_155 = None
    squeeze_65: "f32[200]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
    mul_156: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0006381620931717);  squeeze_65 = None
    mul_157: "f32[200]" = torch.ops.aten.mul.Tensor(mul_156, 0.1);  mul_156 = None
    mul_158: "f32[200]" = torch.ops.aten.mul.Tensor(primals_240, 0.9)
    add_118: "f32[200]" = torch.ops.aten.add.Tensor(mul_157, mul_158);  mul_157 = mul_158 = None
    unsqueeze_84: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1)
    unsqueeze_85: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_159: "f32[8, 200, 14, 14]" = torch.ops.aten.mul.Tensor(mul_153, unsqueeze_85);  mul_153 = unsqueeze_85 = None
    unsqueeze_86: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1);  primals_44 = None
    unsqueeze_87: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_119: "f32[8, 200, 14, 14]" = torch.ops.aten.add.Tensor(mul_159, unsqueeze_87);  mul_159 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_3: "f32[8, 200, 14, 14]" = torch.ops.aten.clone.default(add_119)
    add_120: "f32[8, 200, 14, 14]" = torch.ops.aten.add.Tensor(add_119, 3)
    clamp_min_6: "f32[8, 200, 14, 14]" = torch.ops.aten.clamp_min.default(add_120, 0);  add_120 = None
    clamp_max_6: "f32[8, 200, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_6, 6);  clamp_min_6 = None
    mul_160: "f32[8, 200, 14, 14]" = torch.ops.aten.mul.Tensor(add_119, clamp_max_6);  add_119 = clamp_max_6 = None
    div_6: "f32[8, 200, 14, 14]" = torch.ops.aten.div.Tensor(mul_160, 6);  mul_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_28: "f32[8, 200, 14, 14]" = torch.ops.aten.convolution.default(div_6, primals_129, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 200)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_121: "i64[]" = torch.ops.aten.add.Tensor(primals_241, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_22 = torch.ops.aten.var_mean.correction(convolution_28, [0, 2, 3], correction = 0, keepdim = True)
    getitem_44: "f32[1, 200, 1, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 200, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_122: "f32[1, 200, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05)
    rsqrt_22: "f32[1, 200, 1, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    sub_22: "f32[8, 200, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, getitem_45)
    mul_161: "f32[8, 200, 14, 14]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    squeeze_66: "f32[200]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
    squeeze_67: "f32[200]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
    mul_162: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_163: "f32[200]" = torch.ops.aten.mul.Tensor(primals_242, 0.9)
    add_123: "f32[200]" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
    squeeze_68: "f32[200]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
    mul_164: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0006381620931717);  squeeze_68 = None
    mul_165: "f32[200]" = torch.ops.aten.mul.Tensor(mul_164, 0.1);  mul_164 = None
    mul_166: "f32[200]" = torch.ops.aten.mul.Tensor(primals_243, 0.9)
    add_124: "f32[200]" = torch.ops.aten.add.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
    unsqueeze_88: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1)
    unsqueeze_89: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    mul_167: "f32[8, 200, 14, 14]" = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_89);  mul_161 = unsqueeze_89 = None
    unsqueeze_90: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1);  primals_46 = None
    unsqueeze_91: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    add_125: "f32[8, 200, 14, 14]" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_91);  mul_167 = unsqueeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_4: "f32[8, 200, 14, 14]" = torch.ops.aten.clone.default(add_125)
    add_126: "f32[8, 200, 14, 14]" = torch.ops.aten.add.Tensor(add_125, 3)
    clamp_min_7: "f32[8, 200, 14, 14]" = torch.ops.aten.clamp_min.default(add_126, 0);  add_126 = None
    clamp_max_7: "f32[8, 200, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_7, 6);  clamp_min_7 = None
    mul_168: "f32[8, 200, 14, 14]" = torch.ops.aten.mul.Tensor(add_125, clamp_max_7);  add_125 = clamp_max_7 = None
    div_7: "f32[8, 200, 14, 14]" = torch.ops.aten.div.Tensor(mul_168, 6);  mul_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_29: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(div_7, primals_130, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_127: "i64[]" = torch.ops.aten.add.Tensor(primals_244, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_23 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
    getitem_46: "f32[1, 80, 1, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 80, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_128: "f32[1, 80, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
    rsqrt_23: "f32[1, 80, 1, 1]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
    sub_23: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, getitem_47)
    mul_169: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    squeeze_69: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
    squeeze_70: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
    mul_170: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
    mul_171: "f32[80]" = torch.ops.aten.mul.Tensor(primals_245, 0.9)
    add_129: "f32[80]" = torch.ops.aten.add.Tensor(mul_170, mul_171);  mul_170 = mul_171 = None
    squeeze_71: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
    mul_172: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0006381620931717);  squeeze_71 = None
    mul_173: "f32[80]" = torch.ops.aten.mul.Tensor(mul_172, 0.1);  mul_172 = None
    mul_174: "f32[80]" = torch.ops.aten.mul.Tensor(primals_246, 0.9)
    add_130: "f32[80]" = torch.ops.aten.add.Tensor(mul_173, mul_174);  mul_173 = mul_174 = None
    unsqueeze_92: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_93: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_175: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_169, unsqueeze_93);  mul_169 = unsqueeze_93 = None
    unsqueeze_94: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_95: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_131: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_175, unsqueeze_95);  mul_175 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_132: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_131, add_114);  add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_30: "f32[8, 184, 14, 14]" = torch.ops.aten.convolution.default(add_132, primals_131, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_133: "i64[]" = torch.ops.aten.add.Tensor(primals_247, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_24 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
    getitem_48: "f32[1, 184, 1, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 184, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_134: "f32[1, 184, 1, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05)
    rsqrt_24: "f32[1, 184, 1, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
    sub_24: "f32[8, 184, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, getitem_49)
    mul_176: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    squeeze_72: "f32[184]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
    squeeze_73: "f32[184]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
    mul_177: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_178: "f32[184]" = torch.ops.aten.mul.Tensor(primals_248, 0.9)
    add_135: "f32[184]" = torch.ops.aten.add.Tensor(mul_177, mul_178);  mul_177 = mul_178 = None
    squeeze_74: "f32[184]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
    mul_179: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0006381620931717);  squeeze_74 = None
    mul_180: "f32[184]" = torch.ops.aten.mul.Tensor(mul_179, 0.1);  mul_179 = None
    mul_181: "f32[184]" = torch.ops.aten.mul.Tensor(primals_249, 0.9)
    add_136: "f32[184]" = torch.ops.aten.add.Tensor(mul_180, mul_181);  mul_180 = mul_181 = None
    unsqueeze_96: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1)
    unsqueeze_97: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    mul_182: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(mul_176, unsqueeze_97);  mul_176 = unsqueeze_97 = None
    unsqueeze_98: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1);  primals_50 = None
    unsqueeze_99: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    add_137: "f32[8, 184, 14, 14]" = torch.ops.aten.add.Tensor(mul_182, unsqueeze_99);  mul_182 = unsqueeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_5: "f32[8, 184, 14, 14]" = torch.ops.aten.clone.default(add_137)
    add_138: "f32[8, 184, 14, 14]" = torch.ops.aten.add.Tensor(add_137, 3)
    clamp_min_8: "f32[8, 184, 14, 14]" = torch.ops.aten.clamp_min.default(add_138, 0);  add_138 = None
    clamp_max_8: "f32[8, 184, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_8, 6);  clamp_min_8 = None
    mul_183: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(add_137, clamp_max_8);  add_137 = clamp_max_8 = None
    div_8: "f32[8, 184, 14, 14]" = torch.ops.aten.div.Tensor(mul_183, 6);  mul_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_31: "f32[8, 184, 14, 14]" = torch.ops.aten.convolution.default(div_8, primals_132, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 184)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_139: "i64[]" = torch.ops.aten.add.Tensor(primals_250, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_25 = torch.ops.aten.var_mean.correction(convolution_31, [0, 2, 3], correction = 0, keepdim = True)
    getitem_50: "f32[1, 184, 1, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 184, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_140: "f32[1, 184, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
    rsqrt_25: "f32[1, 184, 1, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    sub_25: "f32[8, 184, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, getitem_51)
    mul_184: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    squeeze_75: "f32[184]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
    squeeze_76: "f32[184]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
    mul_185: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_186: "f32[184]" = torch.ops.aten.mul.Tensor(primals_251, 0.9)
    add_141: "f32[184]" = torch.ops.aten.add.Tensor(mul_185, mul_186);  mul_185 = mul_186 = None
    squeeze_77: "f32[184]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
    mul_187: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0006381620931717);  squeeze_77 = None
    mul_188: "f32[184]" = torch.ops.aten.mul.Tensor(mul_187, 0.1);  mul_187 = None
    mul_189: "f32[184]" = torch.ops.aten.mul.Tensor(primals_252, 0.9)
    add_142: "f32[184]" = torch.ops.aten.add.Tensor(mul_188, mul_189);  mul_188 = mul_189 = None
    unsqueeze_100: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1)
    unsqueeze_101: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_190: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(mul_184, unsqueeze_101);  mul_184 = unsqueeze_101 = None
    unsqueeze_102: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1);  primals_52 = None
    unsqueeze_103: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_143: "f32[8, 184, 14, 14]" = torch.ops.aten.add.Tensor(mul_190, unsqueeze_103);  mul_190 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_6: "f32[8, 184, 14, 14]" = torch.ops.aten.clone.default(add_143)
    add_144: "f32[8, 184, 14, 14]" = torch.ops.aten.add.Tensor(add_143, 3)
    clamp_min_9: "f32[8, 184, 14, 14]" = torch.ops.aten.clamp_min.default(add_144, 0);  add_144 = None
    clamp_max_9: "f32[8, 184, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_9, 6);  clamp_min_9 = None
    mul_191: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(add_143, clamp_max_9);  add_143 = clamp_max_9 = None
    div_9: "f32[8, 184, 14, 14]" = torch.ops.aten.div.Tensor(mul_191, 6);  mul_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_32: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(div_9, primals_133, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_145: "i64[]" = torch.ops.aten.add.Tensor(primals_253, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_26 = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
    getitem_52: "f32[1, 80, 1, 1]" = var_mean_26[0]
    getitem_53: "f32[1, 80, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_146: "f32[1, 80, 1, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05)
    rsqrt_26: "f32[1, 80, 1, 1]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
    sub_26: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, getitem_53)
    mul_192: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    squeeze_78: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
    squeeze_79: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
    mul_193: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_194: "f32[80]" = torch.ops.aten.mul.Tensor(primals_254, 0.9)
    add_147: "f32[80]" = torch.ops.aten.add.Tensor(mul_193, mul_194);  mul_193 = mul_194 = None
    squeeze_80: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
    mul_195: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0006381620931717);  squeeze_80 = None
    mul_196: "f32[80]" = torch.ops.aten.mul.Tensor(mul_195, 0.1);  mul_195 = None
    mul_197: "f32[80]" = torch.ops.aten.mul.Tensor(primals_255, 0.9)
    add_148: "f32[80]" = torch.ops.aten.add.Tensor(mul_196, mul_197);  mul_196 = mul_197 = None
    unsqueeze_104: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_105: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    mul_198: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_192, unsqueeze_105);  mul_192 = unsqueeze_105 = None
    unsqueeze_106: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_107: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    add_149: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_198, unsqueeze_107);  mul_198 = unsqueeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_150: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_149, add_132);  add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_33: "f32[8, 184, 14, 14]" = torch.ops.aten.convolution.default(add_150, primals_134, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_151: "i64[]" = torch.ops.aten.add.Tensor(primals_256, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_27 = torch.ops.aten.var_mean.correction(convolution_33, [0, 2, 3], correction = 0, keepdim = True)
    getitem_54: "f32[1, 184, 1, 1]" = var_mean_27[0]
    getitem_55: "f32[1, 184, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_152: "f32[1, 184, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05)
    rsqrt_27: "f32[1, 184, 1, 1]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
    sub_27: "f32[8, 184, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, getitem_55)
    mul_199: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    squeeze_81: "f32[184]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
    squeeze_82: "f32[184]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
    mul_200: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
    mul_201: "f32[184]" = torch.ops.aten.mul.Tensor(primals_257, 0.9)
    add_153: "f32[184]" = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
    squeeze_83: "f32[184]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
    mul_202: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0006381620931717);  squeeze_83 = None
    mul_203: "f32[184]" = torch.ops.aten.mul.Tensor(mul_202, 0.1);  mul_202 = None
    mul_204: "f32[184]" = torch.ops.aten.mul.Tensor(primals_258, 0.9)
    add_154: "f32[184]" = torch.ops.aten.add.Tensor(mul_203, mul_204);  mul_203 = mul_204 = None
    unsqueeze_108: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1)
    unsqueeze_109: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_205: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(mul_199, unsqueeze_109);  mul_199 = unsqueeze_109 = None
    unsqueeze_110: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1);  primals_56 = None
    unsqueeze_111: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_155: "f32[8, 184, 14, 14]" = torch.ops.aten.add.Tensor(mul_205, unsqueeze_111);  mul_205 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_7: "f32[8, 184, 14, 14]" = torch.ops.aten.clone.default(add_155)
    add_156: "f32[8, 184, 14, 14]" = torch.ops.aten.add.Tensor(add_155, 3)
    clamp_min_10: "f32[8, 184, 14, 14]" = torch.ops.aten.clamp_min.default(add_156, 0);  add_156 = None
    clamp_max_10: "f32[8, 184, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_10, 6);  clamp_min_10 = None
    mul_206: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(add_155, clamp_max_10);  add_155 = clamp_max_10 = None
    div_10: "f32[8, 184, 14, 14]" = torch.ops.aten.div.Tensor(mul_206, 6);  mul_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_34: "f32[8, 184, 14, 14]" = torch.ops.aten.convolution.default(div_10, primals_135, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 184)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_157: "i64[]" = torch.ops.aten.add.Tensor(primals_259, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_28 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
    getitem_56: "f32[1, 184, 1, 1]" = var_mean_28[0]
    getitem_57: "f32[1, 184, 1, 1]" = var_mean_28[1];  var_mean_28 = None
    add_158: "f32[1, 184, 1, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05)
    rsqrt_28: "f32[1, 184, 1, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
    sub_28: "f32[8, 184, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_57)
    mul_207: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    squeeze_84: "f32[184]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
    squeeze_85: "f32[184]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
    mul_208: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
    mul_209: "f32[184]" = torch.ops.aten.mul.Tensor(primals_260, 0.9)
    add_159: "f32[184]" = torch.ops.aten.add.Tensor(mul_208, mul_209);  mul_208 = mul_209 = None
    squeeze_86: "f32[184]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
    mul_210: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0006381620931717);  squeeze_86 = None
    mul_211: "f32[184]" = torch.ops.aten.mul.Tensor(mul_210, 0.1);  mul_210 = None
    mul_212: "f32[184]" = torch.ops.aten.mul.Tensor(primals_261, 0.9)
    add_160: "f32[184]" = torch.ops.aten.add.Tensor(mul_211, mul_212);  mul_211 = mul_212 = None
    unsqueeze_112: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1)
    unsqueeze_113: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    mul_213: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(mul_207, unsqueeze_113);  mul_207 = unsqueeze_113 = None
    unsqueeze_114: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_58, -1);  primals_58 = None
    unsqueeze_115: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    add_161: "f32[8, 184, 14, 14]" = torch.ops.aten.add.Tensor(mul_213, unsqueeze_115);  mul_213 = unsqueeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_8: "f32[8, 184, 14, 14]" = torch.ops.aten.clone.default(add_161)
    add_162: "f32[8, 184, 14, 14]" = torch.ops.aten.add.Tensor(add_161, 3)
    clamp_min_11: "f32[8, 184, 14, 14]" = torch.ops.aten.clamp_min.default(add_162, 0);  add_162 = None
    clamp_max_11: "f32[8, 184, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_11, 6);  clamp_min_11 = None
    mul_214: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(add_161, clamp_max_11);  add_161 = clamp_max_11 = None
    div_11: "f32[8, 184, 14, 14]" = torch.ops.aten.div.Tensor(mul_214, 6);  mul_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_35: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(div_11, primals_136, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_163: "i64[]" = torch.ops.aten.add.Tensor(primals_262, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_29 = torch.ops.aten.var_mean.correction(convolution_35, [0, 2, 3], correction = 0, keepdim = True)
    getitem_58: "f32[1, 80, 1, 1]" = var_mean_29[0]
    getitem_59: "f32[1, 80, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_164: "f32[1, 80, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
    rsqrt_29: "f32[1, 80, 1, 1]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
    sub_29: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, getitem_59)
    mul_215: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    squeeze_87: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
    squeeze_88: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
    mul_216: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
    mul_217: "f32[80]" = torch.ops.aten.mul.Tensor(primals_263, 0.9)
    add_165: "f32[80]" = torch.ops.aten.add.Tensor(mul_216, mul_217);  mul_216 = mul_217 = None
    squeeze_89: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
    mul_218: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0006381620931717);  squeeze_89 = None
    mul_219: "f32[80]" = torch.ops.aten.mul.Tensor(mul_218, 0.1);  mul_218 = None
    mul_220: "f32[80]" = torch.ops.aten.mul.Tensor(primals_264, 0.9)
    add_166: "f32[80]" = torch.ops.aten.add.Tensor(mul_219, mul_220);  mul_219 = mul_220 = None
    unsqueeze_116: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_117: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_221: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_215, unsqueeze_117);  mul_215 = unsqueeze_117 = None
    unsqueeze_118: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_119: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_167: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_221, unsqueeze_119);  mul_221 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_168: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_167, add_150);  add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_36: "f32[8, 480, 14, 14]" = torch.ops.aten.convolution.default(add_168, primals_137, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_169: "i64[]" = torch.ops.aten.add.Tensor(primals_265, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_30 = torch.ops.aten.var_mean.correction(convolution_36, [0, 2, 3], correction = 0, keepdim = True)
    getitem_60: "f32[1, 480, 1, 1]" = var_mean_30[0]
    getitem_61: "f32[1, 480, 1, 1]" = var_mean_30[1];  var_mean_30 = None
    add_170: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
    rsqrt_30: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
    sub_30: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, getitem_61)
    mul_222: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    squeeze_90: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
    squeeze_91: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
    mul_223: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
    mul_224: "f32[480]" = torch.ops.aten.mul.Tensor(primals_266, 0.9)
    add_171: "f32[480]" = torch.ops.aten.add.Tensor(mul_223, mul_224);  mul_223 = mul_224 = None
    squeeze_92: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
    mul_225: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0006381620931717);  squeeze_92 = None
    mul_226: "f32[480]" = torch.ops.aten.mul.Tensor(mul_225, 0.1);  mul_225 = None
    mul_227: "f32[480]" = torch.ops.aten.mul.Tensor(primals_267, 0.9)
    add_172: "f32[480]" = torch.ops.aten.add.Tensor(mul_226, mul_227);  mul_226 = mul_227 = None
    unsqueeze_120: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_61, -1)
    unsqueeze_121: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    mul_228: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_222, unsqueeze_121);  mul_222 = unsqueeze_121 = None
    unsqueeze_122: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1);  primals_62 = None
    unsqueeze_123: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    add_173: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_228, unsqueeze_123);  mul_228 = unsqueeze_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_9: "f32[8, 480, 14, 14]" = torch.ops.aten.clone.default(add_173)
    add_174: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(add_173, 3)
    clamp_min_12: "f32[8, 480, 14, 14]" = torch.ops.aten.clamp_min.default(add_174, 0);  add_174 = None
    clamp_max_12: "f32[8, 480, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_12, 6);  clamp_min_12 = None
    mul_229: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_173, clamp_max_12);  add_173 = clamp_max_12 = None
    div_12: "f32[8, 480, 14, 14]" = torch.ops.aten.div.Tensor(mul_229, 6);  mul_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_37: "f32[8, 480, 14, 14]" = torch.ops.aten.convolution.default(div_12, primals_138, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_175: "i64[]" = torch.ops.aten.add.Tensor(primals_268, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_31 = torch.ops.aten.var_mean.correction(convolution_37, [0, 2, 3], correction = 0, keepdim = True)
    getitem_62: "f32[1, 480, 1, 1]" = var_mean_31[0]
    getitem_63: "f32[1, 480, 1, 1]" = var_mean_31[1];  var_mean_31 = None
    add_176: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
    rsqrt_31: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
    sub_31: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, getitem_63)
    mul_230: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    squeeze_93: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
    squeeze_94: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
    mul_231: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
    mul_232: "f32[480]" = torch.ops.aten.mul.Tensor(primals_269, 0.9)
    add_177: "f32[480]" = torch.ops.aten.add.Tensor(mul_231, mul_232);  mul_231 = mul_232 = None
    squeeze_95: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
    mul_233: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0006381620931717);  squeeze_95 = None
    mul_234: "f32[480]" = torch.ops.aten.mul.Tensor(mul_233, 0.1);  mul_233 = None
    mul_235: "f32[480]" = torch.ops.aten.mul.Tensor(primals_270, 0.9)
    add_178: "f32[480]" = torch.ops.aten.add.Tensor(mul_234, mul_235);  mul_234 = mul_235 = None
    unsqueeze_124: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1)
    unsqueeze_125: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_236: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_230, unsqueeze_125);  mul_230 = unsqueeze_125 = None
    unsqueeze_126: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_64, -1);  primals_64 = None
    unsqueeze_127: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_179: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_236, unsqueeze_127);  mul_236 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_10: "f32[8, 480, 14, 14]" = torch.ops.aten.clone.default(add_179)
    add_180: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(add_179, 3)
    clamp_min_13: "f32[8, 480, 14, 14]" = torch.ops.aten.clamp_min.default(add_180, 0);  add_180 = None
    clamp_max_13: "f32[8, 480, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_13, 6);  clamp_min_13 = None
    mul_237: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_179, clamp_max_13);  add_179 = clamp_max_13 = None
    div_13: "f32[8, 480, 14, 14]" = torch.ops.aten.div.Tensor(mul_237, 6);  mul_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_3: "f32[8, 480, 1, 1]" = torch.ops.aten.mean.dim(div_13, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_38: "f32[8, 120, 1, 1]" = torch.ops.aten.convolution.default(mean_3, primals_139, primals_140, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    relu_14: "f32[8, 120, 1, 1]" = torch.ops.aten.relu.default(convolution_38);  convolution_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_39: "f32[8, 480, 1, 1]" = torch.ops.aten.convolution.default(relu_14, primals_141, primals_142, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_181: "f32[8, 480, 1, 1]" = torch.ops.aten.add.Tensor(convolution_39, 3)
    clamp_min_14: "f32[8, 480, 1, 1]" = torch.ops.aten.clamp_min.default(add_181, 0);  add_181 = None
    clamp_max_14: "f32[8, 480, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_14, 6);  clamp_min_14 = None
    div_14: "f32[8, 480, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_14, 6);  clamp_max_14 = None
    mul_238: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(div_13, div_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_40: "f32[8, 112, 14, 14]" = torch.ops.aten.convolution.default(mul_238, primals_143, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_182: "i64[]" = torch.ops.aten.add.Tensor(primals_271, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_32 = torch.ops.aten.var_mean.correction(convolution_40, [0, 2, 3], correction = 0, keepdim = True)
    getitem_64: "f32[1, 112, 1, 1]" = var_mean_32[0]
    getitem_65: "f32[1, 112, 1, 1]" = var_mean_32[1];  var_mean_32 = None
    add_183: "f32[1, 112, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
    rsqrt_32: "f32[1, 112, 1, 1]" = torch.ops.aten.rsqrt.default(add_183);  add_183 = None
    sub_32: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_65)
    mul_239: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    squeeze_96: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
    squeeze_97: "f32[112]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
    mul_240: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
    mul_241: "f32[112]" = torch.ops.aten.mul.Tensor(primals_272, 0.9)
    add_184: "f32[112]" = torch.ops.aten.add.Tensor(mul_240, mul_241);  mul_240 = mul_241 = None
    squeeze_98: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
    mul_242: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0006381620931717);  squeeze_98 = None
    mul_243: "f32[112]" = torch.ops.aten.mul.Tensor(mul_242, 0.1);  mul_242 = None
    mul_244: "f32[112]" = torch.ops.aten.mul.Tensor(primals_273, 0.9)
    add_185: "f32[112]" = torch.ops.aten.add.Tensor(mul_243, mul_244);  mul_243 = mul_244 = None
    unsqueeze_128: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
    unsqueeze_129: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    mul_245: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(mul_239, unsqueeze_129);  mul_239 = unsqueeze_129 = None
    unsqueeze_130: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
    unsqueeze_131: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    add_186: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(mul_245, unsqueeze_131);  mul_245 = unsqueeze_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_41: "f32[8, 672, 14, 14]" = torch.ops.aten.convolution.default(add_186, primals_144, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_187: "i64[]" = torch.ops.aten.add.Tensor(primals_274, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_33 = torch.ops.aten.var_mean.correction(convolution_41, [0, 2, 3], correction = 0, keepdim = True)
    getitem_66: "f32[1, 672, 1, 1]" = var_mean_33[0]
    getitem_67: "f32[1, 672, 1, 1]" = var_mean_33[1];  var_mean_33 = None
    add_188: "f32[1, 672, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
    rsqrt_33: "f32[1, 672, 1, 1]" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
    sub_33: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, getitem_67)
    mul_246: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    squeeze_99: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
    squeeze_100: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
    mul_247: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
    mul_248: "f32[672]" = torch.ops.aten.mul.Tensor(primals_275, 0.9)
    add_189: "f32[672]" = torch.ops.aten.add.Tensor(mul_247, mul_248);  mul_247 = mul_248 = None
    squeeze_101: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
    mul_249: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0006381620931717);  squeeze_101 = None
    mul_250: "f32[672]" = torch.ops.aten.mul.Tensor(mul_249, 0.1);  mul_249 = None
    mul_251: "f32[672]" = torch.ops.aten.mul.Tensor(primals_276, 0.9)
    add_190: "f32[672]" = torch.ops.aten.add.Tensor(mul_250, mul_251);  mul_250 = mul_251 = None
    unsqueeze_132: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_67, -1)
    unsqueeze_133: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_252: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_246, unsqueeze_133);  mul_246 = unsqueeze_133 = None
    unsqueeze_134: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1);  primals_68 = None
    unsqueeze_135: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_191: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_252, unsqueeze_135);  mul_252 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_11: "f32[8, 672, 14, 14]" = torch.ops.aten.clone.default(add_191)
    add_192: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(add_191, 3)
    clamp_min_15: "f32[8, 672, 14, 14]" = torch.ops.aten.clamp_min.default(add_192, 0);  add_192 = None
    clamp_max_15: "f32[8, 672, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_15, 6);  clamp_min_15 = None
    mul_253: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_191, clamp_max_15);  add_191 = clamp_max_15 = None
    div_15: "f32[8, 672, 14, 14]" = torch.ops.aten.div.Tensor(mul_253, 6);  mul_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_42: "f32[8, 672, 14, 14]" = torch.ops.aten.convolution.default(div_15, primals_145, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 672)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_193: "i64[]" = torch.ops.aten.add.Tensor(primals_277, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_34 = torch.ops.aten.var_mean.correction(convolution_42, [0, 2, 3], correction = 0, keepdim = True)
    getitem_68: "f32[1, 672, 1, 1]" = var_mean_34[0]
    getitem_69: "f32[1, 672, 1, 1]" = var_mean_34[1];  var_mean_34 = None
    add_194: "f32[1, 672, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05)
    rsqrt_34: "f32[1, 672, 1, 1]" = torch.ops.aten.rsqrt.default(add_194);  add_194 = None
    sub_34: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, getitem_69)
    mul_254: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    squeeze_102: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
    squeeze_103: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
    mul_255: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
    mul_256: "f32[672]" = torch.ops.aten.mul.Tensor(primals_278, 0.9)
    add_195: "f32[672]" = torch.ops.aten.add.Tensor(mul_255, mul_256);  mul_255 = mul_256 = None
    squeeze_104: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
    mul_257: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0006381620931717);  squeeze_104 = None
    mul_258: "f32[672]" = torch.ops.aten.mul.Tensor(mul_257, 0.1);  mul_257 = None
    mul_259: "f32[672]" = torch.ops.aten.mul.Tensor(primals_279, 0.9)
    add_196: "f32[672]" = torch.ops.aten.add.Tensor(mul_258, mul_259);  mul_258 = mul_259 = None
    unsqueeze_136: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1)
    unsqueeze_137: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    mul_260: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_254, unsqueeze_137);  mul_254 = unsqueeze_137 = None
    unsqueeze_138: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_70, -1);  primals_70 = None
    unsqueeze_139: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    add_197: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_260, unsqueeze_139);  mul_260 = unsqueeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_12: "f32[8, 672, 14, 14]" = torch.ops.aten.clone.default(add_197)
    add_198: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(add_197, 3)
    clamp_min_16: "f32[8, 672, 14, 14]" = torch.ops.aten.clamp_min.default(add_198, 0);  add_198 = None
    clamp_max_16: "f32[8, 672, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_16, 6);  clamp_min_16 = None
    mul_261: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_197, clamp_max_16);  add_197 = clamp_max_16 = None
    div_16: "f32[8, 672, 14, 14]" = torch.ops.aten.div.Tensor(mul_261, 6);  mul_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_4: "f32[8, 672, 1, 1]" = torch.ops.aten.mean.dim(div_16, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_43: "f32[8, 168, 1, 1]" = torch.ops.aten.convolution.default(mean_4, primals_146, primals_147, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    relu_15: "f32[8, 168, 1, 1]" = torch.ops.aten.relu.default(convolution_43);  convolution_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_44: "f32[8, 672, 1, 1]" = torch.ops.aten.convolution.default(relu_15, primals_148, primals_149, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_199: "f32[8, 672, 1, 1]" = torch.ops.aten.add.Tensor(convolution_44, 3)
    clamp_min_17: "f32[8, 672, 1, 1]" = torch.ops.aten.clamp_min.default(add_199, 0);  add_199 = None
    clamp_max_17: "f32[8, 672, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_17, 6);  clamp_min_17 = None
    div_17: "f32[8, 672, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_17, 6);  clamp_max_17 = None
    mul_262: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(div_16, div_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_45: "f32[8, 112, 14, 14]" = torch.ops.aten.convolution.default(mul_262, primals_150, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_200: "i64[]" = torch.ops.aten.add.Tensor(primals_280, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_35 = torch.ops.aten.var_mean.correction(convolution_45, [0, 2, 3], correction = 0, keepdim = True)
    getitem_70: "f32[1, 112, 1, 1]" = var_mean_35[0]
    getitem_71: "f32[1, 112, 1, 1]" = var_mean_35[1];  var_mean_35 = None
    add_201: "f32[1, 112, 1, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05)
    rsqrt_35: "f32[1, 112, 1, 1]" = torch.ops.aten.rsqrt.default(add_201);  add_201 = None
    sub_35: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, getitem_71)
    mul_263: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    squeeze_105: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
    squeeze_106: "f32[112]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
    mul_264: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
    mul_265: "f32[112]" = torch.ops.aten.mul.Tensor(primals_281, 0.9)
    add_202: "f32[112]" = torch.ops.aten.add.Tensor(mul_264, mul_265);  mul_264 = mul_265 = None
    squeeze_107: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
    mul_266: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0006381620931717);  squeeze_107 = None
    mul_267: "f32[112]" = torch.ops.aten.mul.Tensor(mul_266, 0.1);  mul_266 = None
    mul_268: "f32[112]" = torch.ops.aten.mul.Tensor(primals_282, 0.9)
    add_203: "f32[112]" = torch.ops.aten.add.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
    unsqueeze_140: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_141: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_269: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(mul_263, unsqueeze_141);  mul_263 = unsqueeze_141 = None
    unsqueeze_142: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_143: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_204: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(mul_269, unsqueeze_143);  mul_269 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_205: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(add_204, add_186);  add_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_46: "f32[8, 672, 14, 14]" = torch.ops.aten.convolution.default(add_205, primals_151, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_206: "i64[]" = torch.ops.aten.add.Tensor(primals_283, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_36 = torch.ops.aten.var_mean.correction(convolution_46, [0, 2, 3], correction = 0, keepdim = True)
    getitem_72: "f32[1, 672, 1, 1]" = var_mean_36[0]
    getitem_73: "f32[1, 672, 1, 1]" = var_mean_36[1];  var_mean_36 = None
    add_207: "f32[1, 672, 1, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05)
    rsqrt_36: "f32[1, 672, 1, 1]" = torch.ops.aten.rsqrt.default(add_207);  add_207 = None
    sub_36: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, getitem_73)
    mul_270: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    squeeze_108: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
    squeeze_109: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
    mul_271: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
    mul_272: "f32[672]" = torch.ops.aten.mul.Tensor(primals_284, 0.9)
    add_208: "f32[672]" = torch.ops.aten.add.Tensor(mul_271, mul_272);  mul_271 = mul_272 = None
    squeeze_110: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
    mul_273: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0006381620931717);  squeeze_110 = None
    mul_274: "f32[672]" = torch.ops.aten.mul.Tensor(mul_273, 0.1);  mul_273 = None
    mul_275: "f32[672]" = torch.ops.aten.mul.Tensor(primals_285, 0.9)
    add_209: "f32[672]" = torch.ops.aten.add.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None
    unsqueeze_144: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_73, -1)
    unsqueeze_145: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    mul_276: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_270, unsqueeze_145);  mul_270 = unsqueeze_145 = None
    unsqueeze_146: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1);  primals_74 = None
    unsqueeze_147: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    add_210: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_276, unsqueeze_147);  mul_276 = unsqueeze_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_13: "f32[8, 672, 14, 14]" = torch.ops.aten.clone.default(add_210)
    add_211: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(add_210, 3)
    clamp_min_18: "f32[8, 672, 14, 14]" = torch.ops.aten.clamp_min.default(add_211, 0);  add_211 = None
    clamp_max_18: "f32[8, 672, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_18, 6);  clamp_min_18 = None
    mul_277: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_210, clamp_max_18);  add_210 = clamp_max_18 = None
    div_18: "f32[8, 672, 14, 14]" = torch.ops.aten.div.Tensor(mul_277, 6);  mul_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_47: "f32[8, 672, 7, 7]" = torch.ops.aten.convolution.default(div_18, primals_152, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 672)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_212: "i64[]" = torch.ops.aten.add.Tensor(primals_286, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_37 = torch.ops.aten.var_mean.correction(convolution_47, [0, 2, 3], correction = 0, keepdim = True)
    getitem_74: "f32[1, 672, 1, 1]" = var_mean_37[0]
    getitem_75: "f32[1, 672, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_213: "f32[1, 672, 1, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05)
    rsqrt_37: "f32[1, 672, 1, 1]" = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
    sub_37: "f32[8, 672, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_47, getitem_75)
    mul_278: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    squeeze_111: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
    squeeze_112: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
    mul_279: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
    mul_280: "f32[672]" = torch.ops.aten.mul.Tensor(primals_287, 0.9)
    add_214: "f32[672]" = torch.ops.aten.add.Tensor(mul_279, mul_280);  mul_279 = mul_280 = None
    squeeze_113: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
    mul_281: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0025575447570332);  squeeze_113 = None
    mul_282: "f32[672]" = torch.ops.aten.mul.Tensor(mul_281, 0.1);  mul_281 = None
    mul_283: "f32[672]" = torch.ops.aten.mul.Tensor(primals_288, 0.9)
    add_215: "f32[672]" = torch.ops.aten.add.Tensor(mul_282, mul_283);  mul_282 = mul_283 = None
    unsqueeze_148: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1)
    unsqueeze_149: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_284: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(mul_278, unsqueeze_149);  mul_278 = unsqueeze_149 = None
    unsqueeze_150: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_76, -1);  primals_76 = None
    unsqueeze_151: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_216: "f32[8, 672, 7, 7]" = torch.ops.aten.add.Tensor(mul_284, unsqueeze_151);  mul_284 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_14: "f32[8, 672, 7, 7]" = torch.ops.aten.clone.default(add_216)
    add_217: "f32[8, 672, 7, 7]" = torch.ops.aten.add.Tensor(add_216, 3)
    clamp_min_19: "f32[8, 672, 7, 7]" = torch.ops.aten.clamp_min.default(add_217, 0);  add_217 = None
    clamp_max_19: "f32[8, 672, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_19, 6);  clamp_min_19 = None
    mul_285: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(add_216, clamp_max_19);  add_216 = clamp_max_19 = None
    div_19: "f32[8, 672, 7, 7]" = torch.ops.aten.div.Tensor(mul_285, 6);  mul_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_5: "f32[8, 672, 1, 1]" = torch.ops.aten.mean.dim(div_19, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_48: "f32[8, 168, 1, 1]" = torch.ops.aten.convolution.default(mean_5, primals_153, primals_154, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    relu_16: "f32[8, 168, 1, 1]" = torch.ops.aten.relu.default(convolution_48);  convolution_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_49: "f32[8, 672, 1, 1]" = torch.ops.aten.convolution.default(relu_16, primals_155, primals_156, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_218: "f32[8, 672, 1, 1]" = torch.ops.aten.add.Tensor(convolution_49, 3)
    clamp_min_20: "f32[8, 672, 1, 1]" = torch.ops.aten.clamp_min.default(add_218, 0);  add_218 = None
    clamp_max_20: "f32[8, 672, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_20, 6);  clamp_min_20 = None
    div_20: "f32[8, 672, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_20, 6);  clamp_max_20 = None
    mul_286: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(div_19, div_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_50: "f32[8, 160, 7, 7]" = torch.ops.aten.convolution.default(mul_286, primals_157, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_219: "i64[]" = torch.ops.aten.add.Tensor(primals_289, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_38 = torch.ops.aten.var_mean.correction(convolution_50, [0, 2, 3], correction = 0, keepdim = True)
    getitem_76: "f32[1, 160, 1, 1]" = var_mean_38[0]
    getitem_77: "f32[1, 160, 1, 1]" = var_mean_38[1];  var_mean_38 = None
    add_220: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
    rsqrt_38: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_220);  add_220 = None
    sub_38: "f32[8, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, getitem_77)
    mul_287: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    squeeze_114: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
    squeeze_115: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
    mul_288: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
    mul_289: "f32[160]" = torch.ops.aten.mul.Tensor(primals_290, 0.9)
    add_221: "f32[160]" = torch.ops.aten.add.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
    squeeze_116: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
    mul_290: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0025575447570332);  squeeze_116 = None
    mul_291: "f32[160]" = torch.ops.aten.mul.Tensor(mul_290, 0.1);  mul_290 = None
    mul_292: "f32[160]" = torch.ops.aten.mul.Tensor(primals_291, 0.9)
    add_222: "f32[160]" = torch.ops.aten.add.Tensor(mul_291, mul_292);  mul_291 = mul_292 = None
    unsqueeze_152: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
    unsqueeze_153: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    mul_293: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(mul_287, unsqueeze_153);  mul_287 = unsqueeze_153 = None
    unsqueeze_154: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
    unsqueeze_155: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    add_223: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(mul_293, unsqueeze_155);  mul_293 = unsqueeze_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_51: "f32[8, 960, 7, 7]" = torch.ops.aten.convolution.default(add_223, primals_158, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_224: "i64[]" = torch.ops.aten.add.Tensor(primals_292, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_39 = torch.ops.aten.var_mean.correction(convolution_51, [0, 2, 3], correction = 0, keepdim = True)
    getitem_78: "f32[1, 960, 1, 1]" = var_mean_39[0]
    getitem_79: "f32[1, 960, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_225: "f32[1, 960, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
    rsqrt_39: "f32[1, 960, 1, 1]" = torch.ops.aten.rsqrt.default(add_225);  add_225 = None
    sub_39: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, getitem_79)
    mul_294: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    squeeze_117: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
    squeeze_118: "f32[960]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
    mul_295: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
    mul_296: "f32[960]" = torch.ops.aten.mul.Tensor(primals_293, 0.9)
    add_226: "f32[960]" = torch.ops.aten.add.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None
    squeeze_119: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
    mul_297: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0025575447570332);  squeeze_119 = None
    mul_298: "f32[960]" = torch.ops.aten.mul.Tensor(mul_297, 0.1);  mul_297 = None
    mul_299: "f32[960]" = torch.ops.aten.mul.Tensor(primals_294, 0.9)
    add_227: "f32[960]" = torch.ops.aten.add.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
    unsqueeze_156: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_79, -1)
    unsqueeze_157: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_300: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_294, unsqueeze_157);  mul_294 = unsqueeze_157 = None
    unsqueeze_158: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1);  primals_80 = None
    unsqueeze_159: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_228: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_300, unsqueeze_159);  mul_300 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_15: "f32[8, 960, 7, 7]" = torch.ops.aten.clone.default(add_228)
    add_229: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(add_228, 3)
    clamp_min_21: "f32[8, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_229, 0);  add_229 = None
    clamp_max_21: "f32[8, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_21, 6);  clamp_min_21 = None
    mul_301: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_228, clamp_max_21);  add_228 = clamp_max_21 = None
    div_21: "f32[8, 960, 7, 7]" = torch.ops.aten.div.Tensor(mul_301, 6);  mul_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_52: "f32[8, 960, 7, 7]" = torch.ops.aten.convolution.default(div_21, primals_159, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 960)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_230: "i64[]" = torch.ops.aten.add.Tensor(primals_295, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_40 = torch.ops.aten.var_mean.correction(convolution_52, [0, 2, 3], correction = 0, keepdim = True)
    getitem_80: "f32[1, 960, 1, 1]" = var_mean_40[0]
    getitem_81: "f32[1, 960, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_231: "f32[1, 960, 1, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05)
    rsqrt_40: "f32[1, 960, 1, 1]" = torch.ops.aten.rsqrt.default(add_231);  add_231 = None
    sub_40: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_52, getitem_81)
    mul_302: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    squeeze_120: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
    squeeze_121: "f32[960]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
    mul_303: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
    mul_304: "f32[960]" = torch.ops.aten.mul.Tensor(primals_296, 0.9)
    add_232: "f32[960]" = torch.ops.aten.add.Tensor(mul_303, mul_304);  mul_303 = mul_304 = None
    squeeze_122: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
    mul_305: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0025575447570332);  squeeze_122 = None
    mul_306: "f32[960]" = torch.ops.aten.mul.Tensor(mul_305, 0.1);  mul_305 = None
    mul_307: "f32[960]" = torch.ops.aten.mul.Tensor(primals_297, 0.9)
    add_233: "f32[960]" = torch.ops.aten.add.Tensor(mul_306, mul_307);  mul_306 = mul_307 = None
    unsqueeze_160: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1)
    unsqueeze_161: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    mul_308: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_302, unsqueeze_161);  mul_302 = unsqueeze_161 = None
    unsqueeze_162: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_82, -1);  primals_82 = None
    unsqueeze_163: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    add_234: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_308, unsqueeze_163);  mul_308 = unsqueeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_16: "f32[8, 960, 7, 7]" = torch.ops.aten.clone.default(add_234)
    add_235: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(add_234, 3)
    clamp_min_22: "f32[8, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_235, 0);  add_235 = None
    clamp_max_22: "f32[8, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_22, 6);  clamp_min_22 = None
    mul_309: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_234, clamp_max_22);  add_234 = clamp_max_22 = None
    div_22: "f32[8, 960, 7, 7]" = torch.ops.aten.div.Tensor(mul_309, 6);  mul_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_6: "f32[8, 960, 1, 1]" = torch.ops.aten.mean.dim(div_22, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_53: "f32[8, 240, 1, 1]" = torch.ops.aten.convolution.default(mean_6, primals_160, primals_161, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    relu_17: "f32[8, 240, 1, 1]" = torch.ops.aten.relu.default(convolution_53);  convolution_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_54: "f32[8, 960, 1, 1]" = torch.ops.aten.convolution.default(relu_17, primals_162, primals_163, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_236: "f32[8, 960, 1, 1]" = torch.ops.aten.add.Tensor(convolution_54, 3)
    clamp_min_23: "f32[8, 960, 1, 1]" = torch.ops.aten.clamp_min.default(add_236, 0);  add_236 = None
    clamp_max_23: "f32[8, 960, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_23, 6);  clamp_min_23 = None
    div_23: "f32[8, 960, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_23, 6);  clamp_max_23 = None
    mul_310: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(div_22, div_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_55: "f32[8, 160, 7, 7]" = torch.ops.aten.convolution.default(mul_310, primals_164, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_237: "i64[]" = torch.ops.aten.add.Tensor(primals_298, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_41 = torch.ops.aten.var_mean.correction(convolution_55, [0, 2, 3], correction = 0, keepdim = True)
    getitem_82: "f32[1, 160, 1, 1]" = var_mean_41[0]
    getitem_83: "f32[1, 160, 1, 1]" = var_mean_41[1];  var_mean_41 = None
    add_238: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05)
    rsqrt_41: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_238);  add_238 = None
    sub_41: "f32[8, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_55, getitem_83)
    mul_311: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
    squeeze_123: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
    squeeze_124: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
    mul_312: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
    mul_313: "f32[160]" = torch.ops.aten.mul.Tensor(primals_299, 0.9)
    add_239: "f32[160]" = torch.ops.aten.add.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None
    squeeze_125: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
    mul_314: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0025575447570332);  squeeze_125 = None
    mul_315: "f32[160]" = torch.ops.aten.mul.Tensor(mul_314, 0.1);  mul_314 = None
    mul_316: "f32[160]" = torch.ops.aten.mul.Tensor(primals_300, 0.9)
    add_240: "f32[160]" = torch.ops.aten.add.Tensor(mul_315, mul_316);  mul_315 = mul_316 = None
    unsqueeze_164: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1)
    unsqueeze_165: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_317: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(mul_311, unsqueeze_165);  mul_311 = unsqueeze_165 = None
    unsqueeze_166: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
    unsqueeze_167: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_241: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(mul_317, unsqueeze_167);  mul_317 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_242: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(add_241, add_223);  add_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_56: "f32[8, 960, 7, 7]" = torch.ops.aten.convolution.default(add_242, primals_165, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_243: "i64[]" = torch.ops.aten.add.Tensor(primals_301, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_42 = torch.ops.aten.var_mean.correction(convolution_56, [0, 2, 3], correction = 0, keepdim = True)
    getitem_84: "f32[1, 960, 1, 1]" = var_mean_42[0]
    getitem_85: "f32[1, 960, 1, 1]" = var_mean_42[1];  var_mean_42 = None
    add_244: "f32[1, 960, 1, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05)
    rsqrt_42: "f32[1, 960, 1, 1]" = torch.ops.aten.rsqrt.default(add_244);  add_244 = None
    sub_42: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_56, getitem_85)
    mul_318: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
    squeeze_126: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
    squeeze_127: "f32[960]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
    mul_319: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
    mul_320: "f32[960]" = torch.ops.aten.mul.Tensor(primals_302, 0.9)
    add_245: "f32[960]" = torch.ops.aten.add.Tensor(mul_319, mul_320);  mul_319 = mul_320 = None
    squeeze_128: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
    mul_321: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0025575447570332);  squeeze_128 = None
    mul_322: "f32[960]" = torch.ops.aten.mul.Tensor(mul_321, 0.1);  mul_321 = None
    mul_323: "f32[960]" = torch.ops.aten.mul.Tensor(primals_303, 0.9)
    add_246: "f32[960]" = torch.ops.aten.add.Tensor(mul_322, mul_323);  mul_322 = mul_323 = None
    unsqueeze_168: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_85, -1)
    unsqueeze_169: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    mul_324: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_318, unsqueeze_169);  mul_318 = unsqueeze_169 = None
    unsqueeze_170: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1);  primals_86 = None
    unsqueeze_171: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    add_247: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_324, unsqueeze_171);  mul_324 = unsqueeze_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_17: "f32[8, 960, 7, 7]" = torch.ops.aten.clone.default(add_247)
    add_248: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(add_247, 3)
    clamp_min_24: "f32[8, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_248, 0);  add_248 = None
    clamp_max_24: "f32[8, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_24, 6);  clamp_min_24 = None
    mul_325: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_247, clamp_max_24);  add_247 = clamp_max_24 = None
    div_24: "f32[8, 960, 7, 7]" = torch.ops.aten.div.Tensor(mul_325, 6);  mul_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_57: "f32[8, 960, 7, 7]" = torch.ops.aten.convolution.default(div_24, primals_166, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 960)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_249: "i64[]" = torch.ops.aten.add.Tensor(primals_304, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_43 = torch.ops.aten.var_mean.correction(convolution_57, [0, 2, 3], correction = 0, keepdim = True)
    getitem_86: "f32[1, 960, 1, 1]" = var_mean_43[0]
    getitem_87: "f32[1, 960, 1, 1]" = var_mean_43[1];  var_mean_43 = None
    add_250: "f32[1, 960, 1, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05)
    rsqrt_43: "f32[1, 960, 1, 1]" = torch.ops.aten.rsqrt.default(add_250);  add_250 = None
    sub_43: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_57, getitem_87)
    mul_326: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
    squeeze_129: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
    squeeze_130: "f32[960]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
    mul_327: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
    mul_328: "f32[960]" = torch.ops.aten.mul.Tensor(primals_305, 0.9)
    add_251: "f32[960]" = torch.ops.aten.add.Tensor(mul_327, mul_328);  mul_327 = mul_328 = None
    squeeze_131: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
    mul_329: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0025575447570332);  squeeze_131 = None
    mul_330: "f32[960]" = torch.ops.aten.mul.Tensor(mul_329, 0.1);  mul_329 = None
    mul_331: "f32[960]" = torch.ops.aten.mul.Tensor(primals_306, 0.9)
    add_252: "f32[960]" = torch.ops.aten.add.Tensor(mul_330, mul_331);  mul_330 = mul_331 = None
    unsqueeze_172: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1)
    unsqueeze_173: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_332: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_326, unsqueeze_173);  mul_326 = unsqueeze_173 = None
    unsqueeze_174: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_88, -1);  primals_88 = None
    unsqueeze_175: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_253: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_332, unsqueeze_175);  mul_332 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_18: "f32[8, 960, 7, 7]" = torch.ops.aten.clone.default(add_253)
    add_254: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(add_253, 3)
    clamp_min_25: "f32[8, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_254, 0);  add_254 = None
    clamp_max_25: "f32[8, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_25, 6);  clamp_min_25 = None
    mul_333: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_253, clamp_max_25);  add_253 = clamp_max_25 = None
    div_25: "f32[8, 960, 7, 7]" = torch.ops.aten.div.Tensor(mul_333, 6);  mul_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_7: "f32[8, 960, 1, 1]" = torch.ops.aten.mean.dim(div_25, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_58: "f32[8, 240, 1, 1]" = torch.ops.aten.convolution.default(mean_7, primals_167, primals_168, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    relu_18: "f32[8, 240, 1, 1]" = torch.ops.aten.relu.default(convolution_58);  convolution_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_59: "f32[8, 960, 1, 1]" = torch.ops.aten.convolution.default(relu_18, primals_169, primals_170, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_255: "f32[8, 960, 1, 1]" = torch.ops.aten.add.Tensor(convolution_59, 3)
    clamp_min_26: "f32[8, 960, 1, 1]" = torch.ops.aten.clamp_min.default(add_255, 0);  add_255 = None
    clamp_max_26: "f32[8, 960, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_26, 6);  clamp_min_26 = None
    div_26: "f32[8, 960, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_26, 6);  clamp_max_26 = None
    mul_334: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(div_25, div_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_60: "f32[8, 160, 7, 7]" = torch.ops.aten.convolution.default(mul_334, primals_171, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_256: "i64[]" = torch.ops.aten.add.Tensor(primals_307, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_44 = torch.ops.aten.var_mean.correction(convolution_60, [0, 2, 3], correction = 0, keepdim = True)
    getitem_88: "f32[1, 160, 1, 1]" = var_mean_44[0]
    getitem_89: "f32[1, 160, 1, 1]" = var_mean_44[1];  var_mean_44 = None
    add_257: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05)
    rsqrt_44: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_257);  add_257 = None
    sub_44: "f32[8, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_60, getitem_89)
    mul_335: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
    squeeze_132: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
    squeeze_133: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
    mul_336: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1)
    mul_337: "f32[160]" = torch.ops.aten.mul.Tensor(primals_308, 0.9)
    add_258: "f32[160]" = torch.ops.aten.add.Tensor(mul_336, mul_337);  mul_336 = mul_337 = None
    squeeze_134: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
    mul_338: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_134, 1.0025575447570332);  squeeze_134 = None
    mul_339: "f32[160]" = torch.ops.aten.mul.Tensor(mul_338, 0.1);  mul_338 = None
    mul_340: "f32[160]" = torch.ops.aten.mul.Tensor(primals_309, 0.9)
    add_259: "f32[160]" = torch.ops.aten.add.Tensor(mul_339, mul_340);  mul_339 = mul_340 = None
    unsqueeze_176: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1)
    unsqueeze_177: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    mul_341: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(mul_335, unsqueeze_177);  mul_335 = unsqueeze_177 = None
    unsqueeze_178: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1);  primals_90 = None
    unsqueeze_179: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    add_260: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(mul_341, unsqueeze_179);  mul_341 = unsqueeze_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_261: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(add_260, add_242);  add_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:82, code: x = self.conv(x)
    convolution_61: "f32[8, 960, 7, 7]" = torch.ops.aten.convolution.default(add_261, primals_172, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_262: "i64[]" = torch.ops.aten.add.Tensor(primals_310, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_45 = torch.ops.aten.var_mean.correction(convolution_61, [0, 2, 3], correction = 0, keepdim = True)
    getitem_90: "f32[1, 960, 1, 1]" = var_mean_45[0]
    getitem_91: "f32[1, 960, 1, 1]" = var_mean_45[1];  var_mean_45 = None
    add_263: "f32[1, 960, 1, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05)
    rsqrt_45: "f32[1, 960, 1, 1]" = torch.ops.aten.rsqrt.default(add_263);  add_263 = None
    sub_45: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_61, getitem_91)
    mul_342: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
    squeeze_135: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
    squeeze_136: "f32[960]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
    mul_343: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1)
    mul_344: "f32[960]" = torch.ops.aten.mul.Tensor(primals_311, 0.9)
    add_264: "f32[960]" = torch.ops.aten.add.Tensor(mul_343, mul_344);  mul_343 = mul_344 = None
    squeeze_137: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
    mul_345: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_137, 1.0025575447570332);  squeeze_137 = None
    mul_346: "f32[960]" = torch.ops.aten.mul.Tensor(mul_345, 0.1);  mul_345 = None
    mul_347: "f32[960]" = torch.ops.aten.mul.Tensor(primals_312, 0.9)
    add_265: "f32[960]" = torch.ops.aten.add.Tensor(mul_346, mul_347);  mul_346 = mul_347 = None
    unsqueeze_180: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_91, -1)
    unsqueeze_181: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_348: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_342, unsqueeze_181);  mul_342 = unsqueeze_181 = None
    unsqueeze_182: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1);  primals_92 = None
    unsqueeze_183: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_266: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_348, unsqueeze_183);  mul_348 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_19: "f32[8, 960, 7, 7]" = torch.ops.aten.clone.default(add_266)
    add_267: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(add_266, 3)
    clamp_min_27: "f32[8, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_267, 0);  add_267 = None
    clamp_max_27: "f32[8, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_27, 6);  clamp_min_27 = None
    mul_349: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_266, clamp_max_27);  add_266 = clamp_max_27 = None
    div_27: "f32[8, 960, 7, 7]" = torch.ops.aten.div.Tensor(mul_349, 6);  mul_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean_8: "f32[8, 960, 1, 1]" = torch.ops.aten.mean.dim(div_27, [-1, -2], True);  div_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:145, code: x = self.conv_head(x)
    convolution_62: "f32[8, 1280, 1, 1]" = torch.ops.aten.convolution.default(mean_8, primals_173, primals_174, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:146, code: x = self.act2(x)
    clone_20: "f32[8, 1280, 1, 1]" = torch.ops.aten.clone.default(convolution_62)
    add_268: "f32[8, 1280, 1, 1]" = torch.ops.aten.add.Tensor(convolution_62, 3)
    clamp_min_28: "f32[8, 1280, 1, 1]" = torch.ops.aten.clamp_min.default(add_268, 0);  add_268 = None
    clamp_max_28: "f32[8, 1280, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_28, 6);  clamp_min_28 = None
    mul_350: "f32[8, 1280, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_62, clamp_max_28);  convolution_62 = clamp_max_28 = None
    div_28: "f32[8, 1280, 1, 1]" = torch.ops.aten.div.Tensor(mul_350, 6);  mul_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/linear.py:19, code: return F.linear(input, self.weight, self.bias)
    permute: "f32[1280, 1000]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
    view_1: "f32[8, 1280]" = torch.ops.aten.view.default(div_28, [8, 1280]);  div_28 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_94, view_1, permute);  primals_94 = None
    permute_1: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm: "f32[8, 1280]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 1280]" = torch.ops.aten.mm.default(permute_2, view_1);  permute_2 = view_1 = None
    permute_3: "f32[1280, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_2: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:147, code: x = self.flatten(x)
    view_3: "f32[8, 1280, 1, 1]" = torch.ops.aten.view.default(mm, [8, 1280, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:146, code: x = self.act2(x)
    lt: "b8[8, 1280, 1, 1]" = torch.ops.aten.lt.Scalar(clone_20, -3)
    le: "b8[8, 1280, 1, 1]" = torch.ops.aten.le.Scalar(clone_20, 3)
    div_29: "f32[8, 1280, 1, 1]" = torch.ops.aten.div.Tensor(clone_20, 3);  clone_20 = None
    add_269: "f32[8, 1280, 1, 1]" = torch.ops.aten.add.Tensor(div_29, 0.5);  div_29 = None
    mul_351: "f32[8, 1280, 1, 1]" = torch.ops.aten.mul.Tensor(view_3, add_269);  add_269 = None
    where: "f32[8, 1280, 1, 1]" = torch.ops.aten.where.self(le, mul_351, view_3);  le = mul_351 = view_3 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_1: "f32[8, 1280, 1, 1]" = torch.ops.aten.where.self(lt, scalar_tensor, where);  lt = scalar_tensor = where = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:145, code: x = self.conv_head(x)
    sum_2: "f32[1280]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    convolution_backward = torch.ops.aten.convolution_backward.default(where_1, mean_8, primals_173, [1280], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_1 = mean_8 = primals_173 = None
    getitem_92: "f32[8, 960, 1, 1]" = convolution_backward[0]
    getitem_93: "f32[1280, 960, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 960, 7, 7]" = torch.ops.aten.expand.default(getitem_92, [8, 960, 7, 7]);  getitem_92 = None
    div_30: "f32[8, 960, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_1: "b8[8, 960, 7, 7]" = torch.ops.aten.lt.Scalar(clone_19, -3)
    le_1: "b8[8, 960, 7, 7]" = torch.ops.aten.le.Scalar(clone_19, 3)
    div_31: "f32[8, 960, 7, 7]" = torch.ops.aten.div.Tensor(clone_19, 3);  clone_19 = None
    add_270: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(div_31, 0.5);  div_31 = None
    mul_352: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(div_30, add_270);  add_270 = None
    where_2: "f32[8, 960, 7, 7]" = torch.ops.aten.where.self(le_1, mul_352, div_30);  le_1 = mul_352 = div_30 = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_3: "f32[8, 960, 7, 7]" = torch.ops.aten.where.self(lt_1, scalar_tensor_1, where_2);  lt_1 = scalar_tensor_1 = where_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_184: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
    unsqueeze_185: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, 2);  unsqueeze_184 = None
    unsqueeze_186: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 3);  unsqueeze_185 = None
    sum_3: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_46: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_186)
    mul_353: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_46);  sub_46 = None
    sum_4: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_353, [0, 2, 3]);  mul_353 = None
    mul_354: "f32[960]" = torch.ops.aten.mul.Tensor(sum_3, 0.002551020408163265)
    unsqueeze_187: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_354, 0);  mul_354 = None
    unsqueeze_188: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_187, 2);  unsqueeze_187 = None
    unsqueeze_189: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, 3);  unsqueeze_188 = None
    mul_355: "f32[960]" = torch.ops.aten.mul.Tensor(sum_4, 0.002551020408163265)
    mul_356: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
    mul_357: "f32[960]" = torch.ops.aten.mul.Tensor(mul_355, mul_356);  mul_355 = mul_356 = None
    unsqueeze_190: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_357, 0);  mul_357 = None
    unsqueeze_191: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, 2);  unsqueeze_190 = None
    unsqueeze_192: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_191, 3);  unsqueeze_191 = None
    mul_358: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_91);  primals_91 = None
    unsqueeze_193: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_358, 0);  mul_358 = None
    unsqueeze_194: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_193, 2);  unsqueeze_193 = None
    unsqueeze_195: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, 3);  unsqueeze_194 = None
    sub_47: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_186);  convolution_61 = unsqueeze_186 = None
    mul_359: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_192);  sub_47 = unsqueeze_192 = None
    sub_48: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(where_3, mul_359);  where_3 = mul_359 = None
    sub_49: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(sub_48, unsqueeze_189);  sub_48 = unsqueeze_189 = None
    mul_360: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_195);  sub_49 = unsqueeze_195 = None
    mul_361: "f32[960]" = torch.ops.aten.mul.Tensor(sum_4, squeeze_136);  sum_4 = squeeze_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:82, code: x = self.conv(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_360, add_261, primals_172, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_360 = add_261 = primals_172 = None
    getitem_95: "f32[8, 160, 7, 7]" = convolution_backward_1[0]
    getitem_96: "f32[960, 160, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_196: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
    unsqueeze_197: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, 2);  unsqueeze_196 = None
    unsqueeze_198: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 3);  unsqueeze_197 = None
    sum_5: "f32[160]" = torch.ops.aten.sum.dim_IntList(getitem_95, [0, 2, 3])
    sub_50: "f32[8, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_198)
    mul_362: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_95, sub_50);  sub_50 = None
    sum_6: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_362, [0, 2, 3]);  mul_362 = None
    mul_363: "f32[160]" = torch.ops.aten.mul.Tensor(sum_5, 0.002551020408163265)
    unsqueeze_199: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_363, 0);  mul_363 = None
    unsqueeze_200: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_199, 2);  unsqueeze_199 = None
    unsqueeze_201: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, 3);  unsqueeze_200 = None
    mul_364: "f32[160]" = torch.ops.aten.mul.Tensor(sum_6, 0.002551020408163265)
    mul_365: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
    mul_366: "f32[160]" = torch.ops.aten.mul.Tensor(mul_364, mul_365);  mul_364 = mul_365 = None
    unsqueeze_202: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_366, 0);  mul_366 = None
    unsqueeze_203: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, 2);  unsqueeze_202 = None
    unsqueeze_204: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 3);  unsqueeze_203 = None
    mul_367: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_89);  primals_89 = None
    unsqueeze_205: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_367, 0);  mul_367 = None
    unsqueeze_206: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 2);  unsqueeze_205 = None
    unsqueeze_207: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 3);  unsqueeze_206 = None
    sub_51: "f32[8, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_198);  convolution_60 = unsqueeze_198 = None
    mul_368: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_204);  sub_51 = unsqueeze_204 = None
    sub_52: "f32[8, 160, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_95, mul_368);  mul_368 = None
    sub_53: "f32[8, 160, 7, 7]" = torch.ops.aten.sub.Tensor(sub_52, unsqueeze_201);  sub_52 = unsqueeze_201 = None
    mul_369: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_207);  sub_53 = unsqueeze_207 = None
    mul_370: "f32[160]" = torch.ops.aten.mul.Tensor(sum_6, squeeze_133);  sum_6 = squeeze_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_369, mul_334, primals_171, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_369 = mul_334 = primals_171 = None
    getitem_98: "f32[8, 960, 7, 7]" = convolution_backward_2[0]
    getitem_99: "f32[160, 960, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_371: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_98, div_25);  div_25 = None
    mul_372: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_98, div_26);  getitem_98 = div_26 = None
    sum_7: "f32[8, 960, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_371, [2, 3], True);  mul_371 = None
    gt: "b8[8, 960, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_59, -3.0)
    lt_2: "b8[8, 960, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_59, 3.0);  convolution_59 = None
    bitwise_and: "b8[8, 960, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt, lt_2);  gt = lt_2 = None
    mul_373: "f32[8, 960, 1, 1]" = torch.ops.aten.mul.Tensor(sum_7, 0.16666666666666666);  sum_7 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_4: "f32[8, 960, 1, 1]" = torch.ops.aten.where.self(bitwise_and, mul_373, scalar_tensor_2);  bitwise_and = mul_373 = scalar_tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_8: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(where_4, relu_18, primals_169, [960], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_4 = primals_169 = None
    getitem_101: "f32[8, 240, 1, 1]" = convolution_backward_3[0]
    getitem_102: "f32[960, 240, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    alias_20: "f32[8, 240, 1, 1]" = torch.ops.aten.alias.default(relu_18);  relu_18 = None
    alias_21: "f32[8, 240, 1, 1]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
    le_2: "b8[8, 240, 1, 1]" = torch.ops.aten.le.Scalar(alias_21, 0);  alias_21 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_5: "f32[8, 240, 1, 1]" = torch.ops.aten.where.self(le_2, scalar_tensor_3, getitem_101);  le_2 = scalar_tensor_3 = getitem_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_9: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(where_5, mean_7, primals_167, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_5 = mean_7 = primals_167 = None
    getitem_104: "f32[8, 960, 1, 1]" = convolution_backward_4[0]
    getitem_105: "f32[240, 960, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_1: "f32[8, 960, 7, 7]" = torch.ops.aten.expand.default(getitem_104, [8, 960, 7, 7]);  getitem_104 = None
    div_32: "f32[8, 960, 7, 7]" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_271: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_372, div_32);  mul_372 = div_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_3: "b8[8, 960, 7, 7]" = torch.ops.aten.lt.Scalar(clone_18, -3)
    le_3: "b8[8, 960, 7, 7]" = torch.ops.aten.le.Scalar(clone_18, 3)
    div_33: "f32[8, 960, 7, 7]" = torch.ops.aten.div.Tensor(clone_18, 3);  clone_18 = None
    add_272: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(div_33, 0.5);  div_33 = None
    mul_374: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_271, add_272);  add_272 = None
    where_6: "f32[8, 960, 7, 7]" = torch.ops.aten.where.self(le_3, mul_374, add_271);  le_3 = mul_374 = add_271 = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_7: "f32[8, 960, 7, 7]" = torch.ops.aten.where.self(lt_3, scalar_tensor_4, where_6);  lt_3 = scalar_tensor_4 = where_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_208: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
    unsqueeze_209: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, 2);  unsqueeze_208 = None
    unsqueeze_210: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 3);  unsqueeze_209 = None
    sum_10: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_54: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_210)
    mul_375: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, sub_54);  sub_54 = None
    sum_11: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_375, [0, 2, 3]);  mul_375 = None
    mul_376: "f32[960]" = torch.ops.aten.mul.Tensor(sum_10, 0.002551020408163265)
    unsqueeze_211: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_376, 0);  mul_376 = None
    unsqueeze_212: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 2);  unsqueeze_211 = None
    unsqueeze_213: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 3);  unsqueeze_212 = None
    mul_377: "f32[960]" = torch.ops.aten.mul.Tensor(sum_11, 0.002551020408163265)
    mul_378: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
    mul_379: "f32[960]" = torch.ops.aten.mul.Tensor(mul_377, mul_378);  mul_377 = mul_378 = None
    unsqueeze_214: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_379, 0);  mul_379 = None
    unsqueeze_215: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, 2);  unsqueeze_214 = None
    unsqueeze_216: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 3);  unsqueeze_215 = None
    mul_380: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_87);  primals_87 = None
    unsqueeze_217: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_380, 0);  mul_380 = None
    unsqueeze_218: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 2);  unsqueeze_217 = None
    unsqueeze_219: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 3);  unsqueeze_218 = None
    sub_55: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_210);  convolution_57 = unsqueeze_210 = None
    mul_381: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_216);  sub_55 = unsqueeze_216 = None
    sub_56: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(where_7, mul_381);  where_7 = mul_381 = None
    sub_57: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(sub_56, unsqueeze_213);  sub_56 = unsqueeze_213 = None
    mul_382: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_219);  sub_57 = unsqueeze_219 = None
    mul_383: "f32[960]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_130);  sum_11 = squeeze_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_382, div_24, primals_166, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 960, [True, True, False]);  mul_382 = div_24 = primals_166 = None
    getitem_107: "f32[8, 960, 7, 7]" = convolution_backward_5[0]
    getitem_108: "f32[960, 1, 5, 5]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_4: "b8[8, 960, 7, 7]" = torch.ops.aten.lt.Scalar(clone_17, -3)
    le_4: "b8[8, 960, 7, 7]" = torch.ops.aten.le.Scalar(clone_17, 3)
    div_34: "f32[8, 960, 7, 7]" = torch.ops.aten.div.Tensor(clone_17, 3);  clone_17 = None
    add_273: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(div_34, 0.5);  div_34 = None
    mul_384: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_107, add_273);  add_273 = None
    where_8: "f32[8, 960, 7, 7]" = torch.ops.aten.where.self(le_4, mul_384, getitem_107);  le_4 = mul_384 = getitem_107 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_9: "f32[8, 960, 7, 7]" = torch.ops.aten.where.self(lt_4, scalar_tensor_5, where_8);  lt_4 = scalar_tensor_5 = where_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_220: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
    unsqueeze_221: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, 2);  unsqueeze_220 = None
    unsqueeze_222: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 3);  unsqueeze_221 = None
    sum_12: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_58: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_222)
    mul_385: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_9, sub_58);  sub_58 = None
    sum_13: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_385, [0, 2, 3]);  mul_385 = None
    mul_386: "f32[960]" = torch.ops.aten.mul.Tensor(sum_12, 0.002551020408163265)
    unsqueeze_223: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_386, 0);  mul_386 = None
    unsqueeze_224: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 2);  unsqueeze_223 = None
    unsqueeze_225: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 3);  unsqueeze_224 = None
    mul_387: "f32[960]" = torch.ops.aten.mul.Tensor(sum_13, 0.002551020408163265)
    mul_388: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
    mul_389: "f32[960]" = torch.ops.aten.mul.Tensor(mul_387, mul_388);  mul_387 = mul_388 = None
    unsqueeze_226: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_389, 0);  mul_389 = None
    unsqueeze_227: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 2);  unsqueeze_226 = None
    unsqueeze_228: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 3);  unsqueeze_227 = None
    mul_390: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_85);  primals_85 = None
    unsqueeze_229: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_390, 0);  mul_390 = None
    unsqueeze_230: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 2);  unsqueeze_229 = None
    unsqueeze_231: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 3);  unsqueeze_230 = None
    sub_59: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_222);  convolution_56 = unsqueeze_222 = None
    mul_391: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_228);  sub_59 = unsqueeze_228 = None
    sub_60: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(where_9, mul_391);  where_9 = mul_391 = None
    sub_61: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(sub_60, unsqueeze_225);  sub_60 = unsqueeze_225 = None
    mul_392: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_231);  sub_61 = unsqueeze_231 = None
    mul_393: "f32[960]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_127);  sum_13 = squeeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_392, add_242, primals_165, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_392 = add_242 = primals_165 = None
    getitem_110: "f32[8, 160, 7, 7]" = convolution_backward_6[0]
    getitem_111: "f32[960, 160, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_274: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(getitem_95, getitem_110);  getitem_95 = getitem_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_232: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
    unsqueeze_233: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 2);  unsqueeze_232 = None
    unsqueeze_234: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 3);  unsqueeze_233 = None
    sum_14: "f32[160]" = torch.ops.aten.sum.dim_IntList(add_274, [0, 2, 3])
    sub_62: "f32[8, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_234)
    mul_394: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(add_274, sub_62);  sub_62 = None
    sum_15: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_394, [0, 2, 3]);  mul_394 = None
    mul_395: "f32[160]" = torch.ops.aten.mul.Tensor(sum_14, 0.002551020408163265)
    unsqueeze_235: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_395, 0);  mul_395 = None
    unsqueeze_236: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 2);  unsqueeze_235 = None
    unsqueeze_237: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 3);  unsqueeze_236 = None
    mul_396: "f32[160]" = torch.ops.aten.mul.Tensor(sum_15, 0.002551020408163265)
    mul_397: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
    mul_398: "f32[160]" = torch.ops.aten.mul.Tensor(mul_396, mul_397);  mul_396 = mul_397 = None
    unsqueeze_238: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_398, 0);  mul_398 = None
    unsqueeze_239: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 2);  unsqueeze_238 = None
    unsqueeze_240: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 3);  unsqueeze_239 = None
    mul_399: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_83);  primals_83 = None
    unsqueeze_241: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_399, 0);  mul_399 = None
    unsqueeze_242: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 2);  unsqueeze_241 = None
    unsqueeze_243: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 3);  unsqueeze_242 = None
    sub_63: "f32[8, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_234);  convolution_55 = unsqueeze_234 = None
    mul_400: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_240);  sub_63 = unsqueeze_240 = None
    sub_64: "f32[8, 160, 7, 7]" = torch.ops.aten.sub.Tensor(add_274, mul_400);  mul_400 = None
    sub_65: "f32[8, 160, 7, 7]" = torch.ops.aten.sub.Tensor(sub_64, unsqueeze_237);  sub_64 = unsqueeze_237 = None
    mul_401: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_243);  sub_65 = unsqueeze_243 = None
    mul_402: "f32[160]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_124);  sum_15 = squeeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_401, mul_310, primals_164, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_401 = mul_310 = primals_164 = None
    getitem_113: "f32[8, 960, 7, 7]" = convolution_backward_7[0]
    getitem_114: "f32[160, 960, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_403: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_113, div_22);  div_22 = None
    mul_404: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_113, div_23);  getitem_113 = div_23 = None
    sum_16: "f32[8, 960, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_403, [2, 3], True);  mul_403 = None
    gt_1: "b8[8, 960, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_54, -3.0)
    lt_5: "b8[8, 960, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_54, 3.0);  convolution_54 = None
    bitwise_and_1: "b8[8, 960, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_1, lt_5);  gt_1 = lt_5 = None
    mul_405: "f32[8, 960, 1, 1]" = torch.ops.aten.mul.Tensor(sum_16, 0.16666666666666666);  sum_16 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_10: "f32[8, 960, 1, 1]" = torch.ops.aten.where.self(bitwise_and_1, mul_405, scalar_tensor_6);  bitwise_and_1 = mul_405 = scalar_tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_17: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(where_10, relu_17, primals_162, [960], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_10 = primals_162 = None
    getitem_116: "f32[8, 240, 1, 1]" = convolution_backward_8[0]
    getitem_117: "f32[960, 240, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    alias_23: "f32[8, 240, 1, 1]" = torch.ops.aten.alias.default(relu_17);  relu_17 = None
    alias_24: "f32[8, 240, 1, 1]" = torch.ops.aten.alias.default(alias_23);  alias_23 = None
    le_5: "b8[8, 240, 1, 1]" = torch.ops.aten.le.Scalar(alias_24, 0);  alias_24 = None
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_11: "f32[8, 240, 1, 1]" = torch.ops.aten.where.self(le_5, scalar_tensor_7, getitem_116);  le_5 = scalar_tensor_7 = getitem_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_18: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(where_11, mean_6, primals_160, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_11 = mean_6 = primals_160 = None
    getitem_119: "f32[8, 960, 1, 1]" = convolution_backward_9[0]
    getitem_120: "f32[240, 960, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_2: "f32[8, 960, 7, 7]" = torch.ops.aten.expand.default(getitem_119, [8, 960, 7, 7]);  getitem_119 = None
    div_35: "f32[8, 960, 7, 7]" = torch.ops.aten.div.Scalar(expand_2, 49);  expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_275: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_404, div_35);  mul_404 = div_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_6: "b8[8, 960, 7, 7]" = torch.ops.aten.lt.Scalar(clone_16, -3)
    le_6: "b8[8, 960, 7, 7]" = torch.ops.aten.le.Scalar(clone_16, 3)
    div_36: "f32[8, 960, 7, 7]" = torch.ops.aten.div.Tensor(clone_16, 3);  clone_16 = None
    add_276: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(div_36, 0.5);  div_36 = None
    mul_406: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_275, add_276);  add_276 = None
    where_12: "f32[8, 960, 7, 7]" = torch.ops.aten.where.self(le_6, mul_406, add_275);  le_6 = mul_406 = add_275 = None
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_13: "f32[8, 960, 7, 7]" = torch.ops.aten.where.self(lt_6, scalar_tensor_8, where_12);  lt_6 = scalar_tensor_8 = where_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_244: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
    unsqueeze_245: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 2);  unsqueeze_244 = None
    unsqueeze_246: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 3);  unsqueeze_245 = None
    sum_19: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_66: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_246)
    mul_407: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_13, sub_66);  sub_66 = None
    sum_20: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_407, [0, 2, 3]);  mul_407 = None
    mul_408: "f32[960]" = torch.ops.aten.mul.Tensor(sum_19, 0.002551020408163265)
    unsqueeze_247: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_408, 0);  mul_408 = None
    unsqueeze_248: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 2);  unsqueeze_247 = None
    unsqueeze_249: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 3);  unsqueeze_248 = None
    mul_409: "f32[960]" = torch.ops.aten.mul.Tensor(sum_20, 0.002551020408163265)
    mul_410: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_411: "f32[960]" = torch.ops.aten.mul.Tensor(mul_409, mul_410);  mul_409 = mul_410 = None
    unsqueeze_250: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_411, 0);  mul_411 = None
    unsqueeze_251: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 2);  unsqueeze_250 = None
    unsqueeze_252: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 3);  unsqueeze_251 = None
    mul_412: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_81);  primals_81 = None
    unsqueeze_253: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_412, 0);  mul_412 = None
    unsqueeze_254: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None
    unsqueeze_255: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 3);  unsqueeze_254 = None
    sub_67: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_246);  convolution_52 = unsqueeze_246 = None
    mul_413: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_252);  sub_67 = unsqueeze_252 = None
    sub_68: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(where_13, mul_413);  where_13 = mul_413 = None
    sub_69: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(sub_68, unsqueeze_249);  sub_68 = unsqueeze_249 = None
    mul_414: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_255);  sub_69 = unsqueeze_255 = None
    mul_415: "f32[960]" = torch.ops.aten.mul.Tensor(sum_20, squeeze_121);  sum_20 = squeeze_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_414, div_21, primals_159, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 960, [True, True, False]);  mul_414 = div_21 = primals_159 = None
    getitem_122: "f32[8, 960, 7, 7]" = convolution_backward_10[0]
    getitem_123: "f32[960, 1, 5, 5]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_7: "b8[8, 960, 7, 7]" = torch.ops.aten.lt.Scalar(clone_15, -3)
    le_7: "b8[8, 960, 7, 7]" = torch.ops.aten.le.Scalar(clone_15, 3)
    div_37: "f32[8, 960, 7, 7]" = torch.ops.aten.div.Tensor(clone_15, 3);  clone_15 = None
    add_277: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(div_37, 0.5);  div_37 = None
    mul_416: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_122, add_277);  add_277 = None
    where_14: "f32[8, 960, 7, 7]" = torch.ops.aten.where.self(le_7, mul_416, getitem_122);  le_7 = mul_416 = getitem_122 = None
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_15: "f32[8, 960, 7, 7]" = torch.ops.aten.where.self(lt_7, scalar_tensor_9, where_14);  lt_7 = scalar_tensor_9 = where_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_256: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
    unsqueeze_257: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 2);  unsqueeze_256 = None
    unsqueeze_258: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 3);  unsqueeze_257 = None
    sum_21: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_70: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_258)
    mul_417: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_15, sub_70);  sub_70 = None
    sum_22: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_417, [0, 2, 3]);  mul_417 = None
    mul_418: "f32[960]" = torch.ops.aten.mul.Tensor(sum_21, 0.002551020408163265)
    unsqueeze_259: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_418, 0);  mul_418 = None
    unsqueeze_260: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 2);  unsqueeze_259 = None
    unsqueeze_261: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 3);  unsqueeze_260 = None
    mul_419: "f32[960]" = torch.ops.aten.mul.Tensor(sum_22, 0.002551020408163265)
    mul_420: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_421: "f32[960]" = torch.ops.aten.mul.Tensor(mul_419, mul_420);  mul_419 = mul_420 = None
    unsqueeze_262: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_421, 0);  mul_421 = None
    unsqueeze_263: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 2);  unsqueeze_262 = None
    unsqueeze_264: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 3);  unsqueeze_263 = None
    mul_422: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_79);  primals_79 = None
    unsqueeze_265: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_422, 0);  mul_422 = None
    unsqueeze_266: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
    unsqueeze_267: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 3);  unsqueeze_266 = None
    sub_71: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_258);  convolution_51 = unsqueeze_258 = None
    mul_423: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_264);  sub_71 = unsqueeze_264 = None
    sub_72: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(where_15, mul_423);  where_15 = mul_423 = None
    sub_73: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(sub_72, unsqueeze_261);  sub_72 = unsqueeze_261 = None
    mul_424: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_267);  sub_73 = unsqueeze_267 = None
    mul_425: "f32[960]" = torch.ops.aten.mul.Tensor(sum_22, squeeze_118);  sum_22 = squeeze_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_424, add_223, primals_158, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_424 = add_223 = primals_158 = None
    getitem_125: "f32[8, 160, 7, 7]" = convolution_backward_11[0]
    getitem_126: "f32[960, 160, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_278: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(add_274, getitem_125);  add_274 = getitem_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_268: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
    unsqueeze_269: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 2);  unsqueeze_268 = None
    unsqueeze_270: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 3);  unsqueeze_269 = None
    sum_23: "f32[160]" = torch.ops.aten.sum.dim_IntList(add_278, [0, 2, 3])
    sub_74: "f32[8, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_270)
    mul_426: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(add_278, sub_74);  sub_74 = None
    sum_24: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_426, [0, 2, 3]);  mul_426 = None
    mul_427: "f32[160]" = torch.ops.aten.mul.Tensor(sum_23, 0.002551020408163265)
    unsqueeze_271: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_427, 0);  mul_427 = None
    unsqueeze_272: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 2);  unsqueeze_271 = None
    unsqueeze_273: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 3);  unsqueeze_272 = None
    mul_428: "f32[160]" = torch.ops.aten.mul.Tensor(sum_24, 0.002551020408163265)
    mul_429: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_430: "f32[160]" = torch.ops.aten.mul.Tensor(mul_428, mul_429);  mul_428 = mul_429 = None
    unsqueeze_274: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_430, 0);  mul_430 = None
    unsqueeze_275: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 2);  unsqueeze_274 = None
    unsqueeze_276: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 3);  unsqueeze_275 = None
    mul_431: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_77);  primals_77 = None
    unsqueeze_277: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_431, 0);  mul_431 = None
    unsqueeze_278: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None
    unsqueeze_279: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 3);  unsqueeze_278 = None
    sub_75: "f32[8, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_270);  convolution_50 = unsqueeze_270 = None
    mul_432: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_276);  sub_75 = unsqueeze_276 = None
    sub_76: "f32[8, 160, 7, 7]" = torch.ops.aten.sub.Tensor(add_278, mul_432);  add_278 = mul_432 = None
    sub_77: "f32[8, 160, 7, 7]" = torch.ops.aten.sub.Tensor(sub_76, unsqueeze_273);  sub_76 = unsqueeze_273 = None
    mul_433: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_279);  sub_77 = unsqueeze_279 = None
    mul_434: "f32[160]" = torch.ops.aten.mul.Tensor(sum_24, squeeze_115);  sum_24 = squeeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_433, mul_286, primals_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_433 = mul_286 = primals_157 = None
    getitem_128: "f32[8, 672, 7, 7]" = convolution_backward_12[0]
    getitem_129: "f32[160, 672, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_435: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_128, div_19);  div_19 = None
    mul_436: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_128, div_20);  getitem_128 = div_20 = None
    sum_25: "f32[8, 672, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_435, [2, 3], True);  mul_435 = None
    gt_2: "b8[8, 672, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_49, -3.0)
    lt_8: "b8[8, 672, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_49, 3.0);  convolution_49 = None
    bitwise_and_2: "b8[8, 672, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_2, lt_8);  gt_2 = lt_8 = None
    mul_437: "f32[8, 672, 1, 1]" = torch.ops.aten.mul.Tensor(sum_25, 0.16666666666666666);  sum_25 = None
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_16: "f32[8, 672, 1, 1]" = torch.ops.aten.where.self(bitwise_and_2, mul_437, scalar_tensor_10);  bitwise_and_2 = mul_437 = scalar_tensor_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_26: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(where_16, relu_16, primals_155, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_16 = primals_155 = None
    getitem_131: "f32[8, 168, 1, 1]" = convolution_backward_13[0]
    getitem_132: "f32[672, 168, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    alias_26: "f32[8, 168, 1, 1]" = torch.ops.aten.alias.default(relu_16);  relu_16 = None
    alias_27: "f32[8, 168, 1, 1]" = torch.ops.aten.alias.default(alias_26);  alias_26 = None
    le_8: "b8[8, 168, 1, 1]" = torch.ops.aten.le.Scalar(alias_27, 0);  alias_27 = None
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_17: "f32[8, 168, 1, 1]" = torch.ops.aten.where.self(le_8, scalar_tensor_11, getitem_131);  le_8 = scalar_tensor_11 = getitem_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_27: "f32[168]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(where_17, mean_5, primals_153, [168], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_17 = mean_5 = primals_153 = None
    getitem_134: "f32[8, 672, 1, 1]" = convolution_backward_14[0]
    getitem_135: "f32[168, 672, 1, 1]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_3: "f32[8, 672, 7, 7]" = torch.ops.aten.expand.default(getitem_134, [8, 672, 7, 7]);  getitem_134 = None
    div_38: "f32[8, 672, 7, 7]" = torch.ops.aten.div.Scalar(expand_3, 49);  expand_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_279: "f32[8, 672, 7, 7]" = torch.ops.aten.add.Tensor(mul_436, div_38);  mul_436 = div_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_9: "b8[8, 672, 7, 7]" = torch.ops.aten.lt.Scalar(clone_14, -3)
    le_9: "b8[8, 672, 7, 7]" = torch.ops.aten.le.Scalar(clone_14, 3)
    div_39: "f32[8, 672, 7, 7]" = torch.ops.aten.div.Tensor(clone_14, 3);  clone_14 = None
    add_280: "f32[8, 672, 7, 7]" = torch.ops.aten.add.Tensor(div_39, 0.5);  div_39 = None
    mul_438: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(add_279, add_280);  add_280 = None
    where_18: "f32[8, 672, 7, 7]" = torch.ops.aten.where.self(le_9, mul_438, add_279);  le_9 = mul_438 = add_279 = None
    scalar_tensor_12: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_19: "f32[8, 672, 7, 7]" = torch.ops.aten.where.self(lt_9, scalar_tensor_12, where_18);  lt_9 = scalar_tensor_12 = where_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_280: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
    unsqueeze_281: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 2);  unsqueeze_280 = None
    unsqueeze_282: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 3);  unsqueeze_281 = None
    sum_28: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_78: "f32[8, 672, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_282)
    mul_439: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(where_19, sub_78);  sub_78 = None
    sum_29: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_439, [0, 2, 3]);  mul_439 = None
    mul_440: "f32[672]" = torch.ops.aten.mul.Tensor(sum_28, 0.002551020408163265)
    unsqueeze_283: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_440, 0);  mul_440 = None
    unsqueeze_284: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 2);  unsqueeze_283 = None
    unsqueeze_285: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 3);  unsqueeze_284 = None
    mul_441: "f32[672]" = torch.ops.aten.mul.Tensor(sum_29, 0.002551020408163265)
    mul_442: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_443: "f32[672]" = torch.ops.aten.mul.Tensor(mul_441, mul_442);  mul_441 = mul_442 = None
    unsqueeze_286: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_443, 0);  mul_443 = None
    unsqueeze_287: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 2);  unsqueeze_286 = None
    unsqueeze_288: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 3);  unsqueeze_287 = None
    mul_444: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_75);  primals_75 = None
    unsqueeze_289: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_444, 0);  mul_444 = None
    unsqueeze_290: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 2);  unsqueeze_289 = None
    unsqueeze_291: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 3);  unsqueeze_290 = None
    sub_79: "f32[8, 672, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_282);  convolution_47 = unsqueeze_282 = None
    mul_445: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_288);  sub_79 = unsqueeze_288 = None
    sub_80: "f32[8, 672, 7, 7]" = torch.ops.aten.sub.Tensor(where_19, mul_445);  where_19 = mul_445 = None
    sub_81: "f32[8, 672, 7, 7]" = torch.ops.aten.sub.Tensor(sub_80, unsqueeze_285);  sub_80 = unsqueeze_285 = None
    mul_446: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_291);  sub_81 = unsqueeze_291 = None
    mul_447: "f32[672]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_112);  sum_29 = squeeze_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_446, div_18, primals_152, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False]);  mul_446 = div_18 = primals_152 = None
    getitem_137: "f32[8, 672, 14, 14]" = convolution_backward_15[0]
    getitem_138: "f32[672, 1, 5, 5]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_10: "b8[8, 672, 14, 14]" = torch.ops.aten.lt.Scalar(clone_13, -3)
    le_10: "b8[8, 672, 14, 14]" = torch.ops.aten.le.Scalar(clone_13, 3)
    div_40: "f32[8, 672, 14, 14]" = torch.ops.aten.div.Tensor(clone_13, 3);  clone_13 = None
    add_281: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(div_40, 0.5);  div_40 = None
    mul_448: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_137, add_281);  add_281 = None
    where_20: "f32[8, 672, 14, 14]" = torch.ops.aten.where.self(le_10, mul_448, getitem_137);  le_10 = mul_448 = getitem_137 = None
    scalar_tensor_13: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_21: "f32[8, 672, 14, 14]" = torch.ops.aten.where.self(lt_10, scalar_tensor_13, where_20);  lt_10 = scalar_tensor_13 = where_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_292: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
    unsqueeze_293: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 2);  unsqueeze_292 = None
    unsqueeze_294: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 3);  unsqueeze_293 = None
    sum_30: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_82: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_294)
    mul_449: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(where_21, sub_82);  sub_82 = None
    sum_31: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_449, [0, 2, 3]);  mul_449 = None
    mul_450: "f32[672]" = torch.ops.aten.mul.Tensor(sum_30, 0.0006377551020408163)
    unsqueeze_295: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_450, 0);  mul_450 = None
    unsqueeze_296: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 2);  unsqueeze_295 = None
    unsqueeze_297: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 3);  unsqueeze_296 = None
    mul_451: "f32[672]" = torch.ops.aten.mul.Tensor(sum_31, 0.0006377551020408163)
    mul_452: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_453: "f32[672]" = torch.ops.aten.mul.Tensor(mul_451, mul_452);  mul_451 = mul_452 = None
    unsqueeze_298: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_453, 0);  mul_453 = None
    unsqueeze_299: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 2);  unsqueeze_298 = None
    unsqueeze_300: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 3);  unsqueeze_299 = None
    mul_454: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_73);  primals_73 = None
    unsqueeze_301: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_454, 0);  mul_454 = None
    unsqueeze_302: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None
    unsqueeze_303: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 3);  unsqueeze_302 = None
    sub_83: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_294);  convolution_46 = unsqueeze_294 = None
    mul_455: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_300);  sub_83 = unsqueeze_300 = None
    sub_84: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(where_21, mul_455);  where_21 = mul_455 = None
    sub_85: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(sub_84, unsqueeze_297);  sub_84 = unsqueeze_297 = None
    mul_456: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_303);  sub_85 = unsqueeze_303 = None
    mul_457: "f32[672]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_109);  sum_31 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_456, add_205, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_456 = add_205 = primals_151 = None
    getitem_140: "f32[8, 112, 14, 14]" = convolution_backward_16[0]
    getitem_141: "f32[672, 112, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_304: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
    unsqueeze_305: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 2);  unsqueeze_304 = None
    unsqueeze_306: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 3);  unsqueeze_305 = None
    sum_32: "f32[112]" = torch.ops.aten.sum.dim_IntList(getitem_140, [0, 2, 3])
    sub_86: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_306)
    mul_458: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_140, sub_86);  sub_86 = None
    sum_33: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_458, [0, 2, 3]);  mul_458 = None
    mul_459: "f32[112]" = torch.ops.aten.mul.Tensor(sum_32, 0.0006377551020408163)
    unsqueeze_307: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_459, 0);  mul_459 = None
    unsqueeze_308: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 2);  unsqueeze_307 = None
    unsqueeze_309: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 3);  unsqueeze_308 = None
    mul_460: "f32[112]" = torch.ops.aten.mul.Tensor(sum_33, 0.0006377551020408163)
    mul_461: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_462: "f32[112]" = torch.ops.aten.mul.Tensor(mul_460, mul_461);  mul_460 = mul_461 = None
    unsqueeze_310: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_462, 0);  mul_462 = None
    unsqueeze_311: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 2);  unsqueeze_310 = None
    unsqueeze_312: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 3);  unsqueeze_311 = None
    mul_463: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_71);  primals_71 = None
    unsqueeze_313: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_463, 0);  mul_463 = None
    unsqueeze_314: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 2);  unsqueeze_313 = None
    unsqueeze_315: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 3);  unsqueeze_314 = None
    sub_87: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_306);  convolution_45 = unsqueeze_306 = None
    mul_464: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_312);  sub_87 = unsqueeze_312 = None
    sub_88: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_140, mul_464);  mul_464 = None
    sub_89: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(sub_88, unsqueeze_309);  sub_88 = unsqueeze_309 = None
    mul_465: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_315);  sub_89 = unsqueeze_315 = None
    mul_466: "f32[112]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_106);  sum_33 = squeeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_465, mul_262, primals_150, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_465 = mul_262 = primals_150 = None
    getitem_143: "f32[8, 672, 14, 14]" = convolution_backward_17[0]
    getitem_144: "f32[112, 672, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_467: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_143, div_16);  div_16 = None
    mul_468: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_143, div_17);  getitem_143 = div_17 = None
    sum_34: "f32[8, 672, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_467, [2, 3], True);  mul_467 = None
    gt_3: "b8[8, 672, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_44, -3.0)
    lt_11: "b8[8, 672, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_44, 3.0);  convolution_44 = None
    bitwise_and_3: "b8[8, 672, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_3, lt_11);  gt_3 = lt_11 = None
    mul_469: "f32[8, 672, 1, 1]" = torch.ops.aten.mul.Tensor(sum_34, 0.16666666666666666);  sum_34 = None
    scalar_tensor_14: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_22: "f32[8, 672, 1, 1]" = torch.ops.aten.where.self(bitwise_and_3, mul_469, scalar_tensor_14);  bitwise_and_3 = mul_469 = scalar_tensor_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_35: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(where_22, relu_15, primals_148, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_22 = primals_148 = None
    getitem_146: "f32[8, 168, 1, 1]" = convolution_backward_18[0]
    getitem_147: "f32[672, 168, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    alias_29: "f32[8, 168, 1, 1]" = torch.ops.aten.alias.default(relu_15);  relu_15 = None
    alias_30: "f32[8, 168, 1, 1]" = torch.ops.aten.alias.default(alias_29);  alias_29 = None
    le_11: "b8[8, 168, 1, 1]" = torch.ops.aten.le.Scalar(alias_30, 0);  alias_30 = None
    scalar_tensor_15: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_23: "f32[8, 168, 1, 1]" = torch.ops.aten.where.self(le_11, scalar_tensor_15, getitem_146);  le_11 = scalar_tensor_15 = getitem_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_36: "f32[168]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(where_23, mean_4, primals_146, [168], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_23 = mean_4 = primals_146 = None
    getitem_149: "f32[8, 672, 1, 1]" = convolution_backward_19[0]
    getitem_150: "f32[168, 672, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_4: "f32[8, 672, 14, 14]" = torch.ops.aten.expand.default(getitem_149, [8, 672, 14, 14]);  getitem_149 = None
    div_41: "f32[8, 672, 14, 14]" = torch.ops.aten.div.Scalar(expand_4, 196);  expand_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_282: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_468, div_41);  mul_468 = div_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_12: "b8[8, 672, 14, 14]" = torch.ops.aten.lt.Scalar(clone_12, -3)
    le_12: "b8[8, 672, 14, 14]" = torch.ops.aten.le.Scalar(clone_12, 3)
    div_42: "f32[8, 672, 14, 14]" = torch.ops.aten.div.Tensor(clone_12, 3);  clone_12 = None
    add_283: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(div_42, 0.5);  div_42 = None
    mul_470: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_282, add_283);  add_283 = None
    where_24: "f32[8, 672, 14, 14]" = torch.ops.aten.where.self(le_12, mul_470, add_282);  le_12 = mul_470 = add_282 = None
    scalar_tensor_16: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_25: "f32[8, 672, 14, 14]" = torch.ops.aten.where.self(lt_12, scalar_tensor_16, where_24);  lt_12 = scalar_tensor_16 = where_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_316: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
    unsqueeze_317: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, 2);  unsqueeze_316 = None
    unsqueeze_318: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 3);  unsqueeze_317 = None
    sum_37: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_90: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_318)
    mul_471: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(where_25, sub_90);  sub_90 = None
    sum_38: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_471, [0, 2, 3]);  mul_471 = None
    mul_472: "f32[672]" = torch.ops.aten.mul.Tensor(sum_37, 0.0006377551020408163)
    unsqueeze_319: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_472, 0);  mul_472 = None
    unsqueeze_320: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 2);  unsqueeze_319 = None
    unsqueeze_321: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 3);  unsqueeze_320 = None
    mul_473: "f32[672]" = torch.ops.aten.mul.Tensor(sum_38, 0.0006377551020408163)
    mul_474: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_475: "f32[672]" = torch.ops.aten.mul.Tensor(mul_473, mul_474);  mul_473 = mul_474 = None
    unsqueeze_322: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_475, 0);  mul_475 = None
    unsqueeze_323: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 2);  unsqueeze_322 = None
    unsqueeze_324: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 3);  unsqueeze_323 = None
    mul_476: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_69);  primals_69 = None
    unsqueeze_325: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_476, 0);  mul_476 = None
    unsqueeze_326: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None
    unsqueeze_327: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 3);  unsqueeze_326 = None
    sub_91: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_318);  convolution_42 = unsqueeze_318 = None
    mul_477: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_324);  sub_91 = unsqueeze_324 = None
    sub_92: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(where_25, mul_477);  where_25 = mul_477 = None
    sub_93: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(sub_92, unsqueeze_321);  sub_92 = unsqueeze_321 = None
    mul_478: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_327);  sub_93 = unsqueeze_327 = None
    mul_479: "f32[672]" = torch.ops.aten.mul.Tensor(sum_38, squeeze_103);  sum_38 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_478, div_15, primals_145, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 672, [True, True, False]);  mul_478 = div_15 = primals_145 = None
    getitem_152: "f32[8, 672, 14, 14]" = convolution_backward_20[0]
    getitem_153: "f32[672, 1, 3, 3]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_13: "b8[8, 672, 14, 14]" = torch.ops.aten.lt.Scalar(clone_11, -3)
    le_13: "b8[8, 672, 14, 14]" = torch.ops.aten.le.Scalar(clone_11, 3)
    div_43: "f32[8, 672, 14, 14]" = torch.ops.aten.div.Tensor(clone_11, 3);  clone_11 = None
    add_284: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(div_43, 0.5);  div_43 = None
    mul_480: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_152, add_284);  add_284 = None
    where_26: "f32[8, 672, 14, 14]" = torch.ops.aten.where.self(le_13, mul_480, getitem_152);  le_13 = mul_480 = getitem_152 = None
    scalar_tensor_17: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_27: "f32[8, 672, 14, 14]" = torch.ops.aten.where.self(lt_13, scalar_tensor_17, where_26);  lt_13 = scalar_tensor_17 = where_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_328: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
    unsqueeze_329: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 2);  unsqueeze_328 = None
    unsqueeze_330: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 3);  unsqueeze_329 = None
    sum_39: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_94: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_330)
    mul_481: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(where_27, sub_94);  sub_94 = None
    sum_40: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_481, [0, 2, 3]);  mul_481 = None
    mul_482: "f32[672]" = torch.ops.aten.mul.Tensor(sum_39, 0.0006377551020408163)
    unsqueeze_331: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_482, 0);  mul_482 = None
    unsqueeze_332: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 2);  unsqueeze_331 = None
    unsqueeze_333: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 3);  unsqueeze_332 = None
    mul_483: "f32[672]" = torch.ops.aten.mul.Tensor(sum_40, 0.0006377551020408163)
    mul_484: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_485: "f32[672]" = torch.ops.aten.mul.Tensor(mul_483, mul_484);  mul_483 = mul_484 = None
    unsqueeze_334: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_485, 0);  mul_485 = None
    unsqueeze_335: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 2);  unsqueeze_334 = None
    unsqueeze_336: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 3);  unsqueeze_335 = None
    mul_486: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_67);  primals_67 = None
    unsqueeze_337: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_486, 0);  mul_486 = None
    unsqueeze_338: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 2);  unsqueeze_337 = None
    unsqueeze_339: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 3);  unsqueeze_338 = None
    sub_95: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_330);  convolution_41 = unsqueeze_330 = None
    mul_487: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_336);  sub_95 = unsqueeze_336 = None
    sub_96: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(where_27, mul_487);  where_27 = mul_487 = None
    sub_97: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(sub_96, unsqueeze_333);  sub_96 = unsqueeze_333 = None
    mul_488: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_339);  sub_97 = unsqueeze_339 = None
    mul_489: "f32[672]" = torch.ops.aten.mul.Tensor(sum_40, squeeze_100);  sum_40 = squeeze_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_488, add_186, primals_144, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_488 = add_186 = primals_144 = None
    getitem_155: "f32[8, 112, 14, 14]" = convolution_backward_21[0]
    getitem_156: "f32[672, 112, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_285: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(getitem_140, getitem_155);  getitem_140 = getitem_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_340: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
    unsqueeze_341: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 2);  unsqueeze_340 = None
    unsqueeze_342: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 3);  unsqueeze_341 = None
    sum_41: "f32[112]" = torch.ops.aten.sum.dim_IntList(add_285, [0, 2, 3])
    sub_98: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_342)
    mul_490: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(add_285, sub_98);  sub_98 = None
    sum_42: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_490, [0, 2, 3]);  mul_490 = None
    mul_491: "f32[112]" = torch.ops.aten.mul.Tensor(sum_41, 0.0006377551020408163)
    unsqueeze_343: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_491, 0);  mul_491 = None
    unsqueeze_344: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 2);  unsqueeze_343 = None
    unsqueeze_345: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 3);  unsqueeze_344 = None
    mul_492: "f32[112]" = torch.ops.aten.mul.Tensor(sum_42, 0.0006377551020408163)
    mul_493: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_494: "f32[112]" = torch.ops.aten.mul.Tensor(mul_492, mul_493);  mul_492 = mul_493 = None
    unsqueeze_346: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_494, 0);  mul_494 = None
    unsqueeze_347: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 2);  unsqueeze_346 = None
    unsqueeze_348: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 3);  unsqueeze_347 = None
    mul_495: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_65);  primals_65 = None
    unsqueeze_349: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_495, 0);  mul_495 = None
    unsqueeze_350: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None
    unsqueeze_351: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 3);  unsqueeze_350 = None
    sub_99: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_342);  convolution_40 = unsqueeze_342 = None
    mul_496: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_348);  sub_99 = unsqueeze_348 = None
    sub_100: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(add_285, mul_496);  add_285 = mul_496 = None
    sub_101: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(sub_100, unsqueeze_345);  sub_100 = unsqueeze_345 = None
    mul_497: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_351);  sub_101 = unsqueeze_351 = None
    mul_498: "f32[112]" = torch.ops.aten.mul.Tensor(sum_42, squeeze_97);  sum_42 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_497, mul_238, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_497 = mul_238 = primals_143 = None
    getitem_158: "f32[8, 480, 14, 14]" = convolution_backward_22[0]
    getitem_159: "f32[112, 480, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_499: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_158, div_13);  div_13 = None
    mul_500: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_158, div_14);  getitem_158 = div_14 = None
    sum_43: "f32[8, 480, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_499, [2, 3], True);  mul_499 = None
    gt_4: "b8[8, 480, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_39, -3.0)
    lt_14: "b8[8, 480, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_39, 3.0);  convolution_39 = None
    bitwise_and_4: "b8[8, 480, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_4, lt_14);  gt_4 = lt_14 = None
    mul_501: "f32[8, 480, 1, 1]" = torch.ops.aten.mul.Tensor(sum_43, 0.16666666666666666);  sum_43 = None
    scalar_tensor_18: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_28: "f32[8, 480, 1, 1]" = torch.ops.aten.where.self(bitwise_and_4, mul_501, scalar_tensor_18);  bitwise_and_4 = mul_501 = scalar_tensor_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_44: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(where_28, relu_14, primals_141, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_28 = primals_141 = None
    getitem_161: "f32[8, 120, 1, 1]" = convolution_backward_23[0]
    getitem_162: "f32[480, 120, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    alias_32: "f32[8, 120, 1, 1]" = torch.ops.aten.alias.default(relu_14);  relu_14 = None
    alias_33: "f32[8, 120, 1, 1]" = torch.ops.aten.alias.default(alias_32);  alias_32 = None
    le_14: "b8[8, 120, 1, 1]" = torch.ops.aten.le.Scalar(alias_33, 0);  alias_33 = None
    scalar_tensor_19: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_29: "f32[8, 120, 1, 1]" = torch.ops.aten.where.self(le_14, scalar_tensor_19, getitem_161);  le_14 = scalar_tensor_19 = getitem_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_45: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(where_29, mean_3, primals_139, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_29 = mean_3 = primals_139 = None
    getitem_164: "f32[8, 480, 1, 1]" = convolution_backward_24[0]
    getitem_165: "f32[120, 480, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_5: "f32[8, 480, 14, 14]" = torch.ops.aten.expand.default(getitem_164, [8, 480, 14, 14]);  getitem_164 = None
    div_44: "f32[8, 480, 14, 14]" = torch.ops.aten.div.Scalar(expand_5, 196);  expand_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_286: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_500, div_44);  mul_500 = div_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_15: "b8[8, 480, 14, 14]" = torch.ops.aten.lt.Scalar(clone_10, -3)
    le_15: "b8[8, 480, 14, 14]" = torch.ops.aten.le.Scalar(clone_10, 3)
    div_45: "f32[8, 480, 14, 14]" = torch.ops.aten.div.Tensor(clone_10, 3);  clone_10 = None
    add_287: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(div_45, 0.5);  div_45 = None
    mul_502: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_286, add_287);  add_287 = None
    where_30: "f32[8, 480, 14, 14]" = torch.ops.aten.where.self(le_15, mul_502, add_286);  le_15 = mul_502 = add_286 = None
    scalar_tensor_20: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_31: "f32[8, 480, 14, 14]" = torch.ops.aten.where.self(lt_15, scalar_tensor_20, where_30);  lt_15 = scalar_tensor_20 = where_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_352: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
    unsqueeze_353: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 2);  unsqueeze_352 = None
    unsqueeze_354: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 3);  unsqueeze_353 = None
    sum_46: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_102: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_354)
    mul_503: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_31, sub_102);  sub_102 = None
    sum_47: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_503, [0, 2, 3]);  mul_503 = None
    mul_504: "f32[480]" = torch.ops.aten.mul.Tensor(sum_46, 0.0006377551020408163)
    unsqueeze_355: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_504, 0);  mul_504 = None
    unsqueeze_356: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 2);  unsqueeze_355 = None
    unsqueeze_357: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 3);  unsqueeze_356 = None
    mul_505: "f32[480]" = torch.ops.aten.mul.Tensor(sum_47, 0.0006377551020408163)
    mul_506: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_507: "f32[480]" = torch.ops.aten.mul.Tensor(mul_505, mul_506);  mul_505 = mul_506 = None
    unsqueeze_358: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_507, 0);  mul_507 = None
    unsqueeze_359: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 2);  unsqueeze_358 = None
    unsqueeze_360: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 3);  unsqueeze_359 = None
    mul_508: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_63);  primals_63 = None
    unsqueeze_361: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_508, 0);  mul_508 = None
    unsqueeze_362: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
    unsqueeze_363: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 3);  unsqueeze_362 = None
    sub_103: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_354);  convolution_37 = unsqueeze_354 = None
    mul_509: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_360);  sub_103 = unsqueeze_360 = None
    sub_104: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(where_31, mul_509);  where_31 = mul_509 = None
    sub_105: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_104, unsqueeze_357);  sub_104 = unsqueeze_357 = None
    mul_510: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_363);  sub_105 = unsqueeze_363 = None
    mul_511: "f32[480]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_94);  sum_47 = squeeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_510, div_12, primals_138, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_510 = div_12 = primals_138 = None
    getitem_167: "f32[8, 480, 14, 14]" = convolution_backward_25[0]
    getitem_168: "f32[480, 1, 3, 3]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_16: "b8[8, 480, 14, 14]" = torch.ops.aten.lt.Scalar(clone_9, -3)
    le_16: "b8[8, 480, 14, 14]" = torch.ops.aten.le.Scalar(clone_9, 3)
    div_46: "f32[8, 480, 14, 14]" = torch.ops.aten.div.Tensor(clone_9, 3);  clone_9 = None
    add_288: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(div_46, 0.5);  div_46 = None
    mul_512: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_167, add_288);  add_288 = None
    where_32: "f32[8, 480, 14, 14]" = torch.ops.aten.where.self(le_16, mul_512, getitem_167);  le_16 = mul_512 = getitem_167 = None
    scalar_tensor_21: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_33: "f32[8, 480, 14, 14]" = torch.ops.aten.where.self(lt_16, scalar_tensor_21, where_32);  lt_16 = scalar_tensor_21 = where_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_364: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_365: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 2);  unsqueeze_364 = None
    unsqueeze_366: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 3);  unsqueeze_365 = None
    sum_48: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    sub_106: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_366)
    mul_513: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_33, sub_106);  sub_106 = None
    sum_49: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_513, [0, 2, 3]);  mul_513 = None
    mul_514: "f32[480]" = torch.ops.aten.mul.Tensor(sum_48, 0.0006377551020408163)
    unsqueeze_367: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_514, 0);  mul_514 = None
    unsqueeze_368: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
    unsqueeze_369: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 3);  unsqueeze_368 = None
    mul_515: "f32[480]" = torch.ops.aten.mul.Tensor(sum_49, 0.0006377551020408163)
    mul_516: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_517: "f32[480]" = torch.ops.aten.mul.Tensor(mul_515, mul_516);  mul_515 = mul_516 = None
    unsqueeze_370: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_517, 0);  mul_517 = None
    unsqueeze_371: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 2);  unsqueeze_370 = None
    unsqueeze_372: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 3);  unsqueeze_371 = None
    mul_518: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_61);  primals_61 = None
    unsqueeze_373: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_518, 0);  mul_518 = None
    unsqueeze_374: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
    unsqueeze_375: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 3);  unsqueeze_374 = None
    sub_107: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_366);  convolution_36 = unsqueeze_366 = None
    mul_519: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_372);  sub_107 = unsqueeze_372 = None
    sub_108: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(where_33, mul_519);  where_33 = mul_519 = None
    sub_109: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_108, unsqueeze_369);  sub_108 = unsqueeze_369 = None
    mul_520: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_375);  sub_109 = unsqueeze_375 = None
    mul_521: "f32[480]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_91);  sum_49 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_520, add_168, primals_137, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_520 = add_168 = primals_137 = None
    getitem_170: "f32[8, 80, 14, 14]" = convolution_backward_26[0]
    getitem_171: "f32[480, 80, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_376: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    unsqueeze_377: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 2);  unsqueeze_376 = None
    unsqueeze_378: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 3);  unsqueeze_377 = None
    sum_50: "f32[80]" = torch.ops.aten.sum.dim_IntList(getitem_170, [0, 2, 3])
    sub_110: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_378)
    mul_522: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_170, sub_110);  sub_110 = None
    sum_51: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_522, [0, 2, 3]);  mul_522 = None
    mul_523: "f32[80]" = torch.ops.aten.mul.Tensor(sum_50, 0.0006377551020408163)
    unsqueeze_379: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_523, 0);  mul_523 = None
    unsqueeze_380: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
    unsqueeze_381: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 3);  unsqueeze_380 = None
    mul_524: "f32[80]" = torch.ops.aten.mul.Tensor(sum_51, 0.0006377551020408163)
    mul_525: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_526: "f32[80]" = torch.ops.aten.mul.Tensor(mul_524, mul_525);  mul_524 = mul_525 = None
    unsqueeze_382: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_526, 0);  mul_526 = None
    unsqueeze_383: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 2);  unsqueeze_382 = None
    unsqueeze_384: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 3);  unsqueeze_383 = None
    mul_527: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_59);  primals_59 = None
    unsqueeze_385: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_527, 0);  mul_527 = None
    unsqueeze_386: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
    unsqueeze_387: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
    sub_111: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_378);  convolution_35 = unsqueeze_378 = None
    mul_528: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_384);  sub_111 = unsqueeze_384 = None
    sub_112: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_170, mul_528);  mul_528 = None
    sub_113: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(sub_112, unsqueeze_381);  sub_112 = unsqueeze_381 = None
    mul_529: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_387);  sub_113 = unsqueeze_387 = None
    mul_530: "f32[80]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_88);  sum_51 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_529, div_11, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_529 = div_11 = primals_136 = None
    getitem_173: "f32[8, 184, 14, 14]" = convolution_backward_27[0]
    getitem_174: "f32[80, 184, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_17: "b8[8, 184, 14, 14]" = torch.ops.aten.lt.Scalar(clone_8, -3)
    le_17: "b8[8, 184, 14, 14]" = torch.ops.aten.le.Scalar(clone_8, 3)
    div_47: "f32[8, 184, 14, 14]" = torch.ops.aten.div.Tensor(clone_8, 3);  clone_8 = None
    add_289: "f32[8, 184, 14, 14]" = torch.ops.aten.add.Tensor(div_47, 0.5);  div_47 = None
    mul_531: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_173, add_289);  add_289 = None
    where_34: "f32[8, 184, 14, 14]" = torch.ops.aten.where.self(le_17, mul_531, getitem_173);  le_17 = mul_531 = getitem_173 = None
    scalar_tensor_22: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_35: "f32[8, 184, 14, 14]" = torch.ops.aten.where.self(lt_17, scalar_tensor_22, where_34);  lt_17 = scalar_tensor_22 = where_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_388: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_389: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 2);  unsqueeze_388 = None
    unsqueeze_390: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 3);  unsqueeze_389 = None
    sum_52: "f32[184]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    sub_114: "f32[8, 184, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_390)
    mul_532: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(where_35, sub_114);  sub_114 = None
    sum_53: "f32[184]" = torch.ops.aten.sum.dim_IntList(mul_532, [0, 2, 3]);  mul_532 = None
    mul_533: "f32[184]" = torch.ops.aten.mul.Tensor(sum_52, 0.0006377551020408163)
    unsqueeze_391: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_533, 0);  mul_533 = None
    unsqueeze_392: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
    unsqueeze_393: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 3);  unsqueeze_392 = None
    mul_534: "f32[184]" = torch.ops.aten.mul.Tensor(sum_53, 0.0006377551020408163)
    mul_535: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_536: "f32[184]" = torch.ops.aten.mul.Tensor(mul_534, mul_535);  mul_534 = mul_535 = None
    unsqueeze_394: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_536, 0);  mul_536 = None
    unsqueeze_395: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 2);  unsqueeze_394 = None
    unsqueeze_396: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 3);  unsqueeze_395 = None
    mul_537: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_57);  primals_57 = None
    unsqueeze_397: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_537, 0);  mul_537 = None
    unsqueeze_398: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
    unsqueeze_399: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 3);  unsqueeze_398 = None
    sub_115: "f32[8, 184, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_390);  convolution_34 = unsqueeze_390 = None
    mul_538: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_396);  sub_115 = unsqueeze_396 = None
    sub_116: "f32[8, 184, 14, 14]" = torch.ops.aten.sub.Tensor(where_35, mul_538);  where_35 = mul_538 = None
    sub_117: "f32[8, 184, 14, 14]" = torch.ops.aten.sub.Tensor(sub_116, unsqueeze_393);  sub_116 = unsqueeze_393 = None
    mul_539: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_399);  sub_117 = unsqueeze_399 = None
    mul_540: "f32[184]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_85);  sum_53 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_539, div_10, primals_135, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 184, [True, True, False]);  mul_539 = div_10 = primals_135 = None
    getitem_176: "f32[8, 184, 14, 14]" = convolution_backward_28[0]
    getitem_177: "f32[184, 1, 3, 3]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_18: "b8[8, 184, 14, 14]" = torch.ops.aten.lt.Scalar(clone_7, -3)
    le_18: "b8[8, 184, 14, 14]" = torch.ops.aten.le.Scalar(clone_7, 3)
    div_48: "f32[8, 184, 14, 14]" = torch.ops.aten.div.Tensor(clone_7, 3);  clone_7 = None
    add_290: "f32[8, 184, 14, 14]" = torch.ops.aten.add.Tensor(div_48, 0.5);  div_48 = None
    mul_541: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_176, add_290);  add_290 = None
    where_36: "f32[8, 184, 14, 14]" = torch.ops.aten.where.self(le_18, mul_541, getitem_176);  le_18 = mul_541 = getitem_176 = None
    scalar_tensor_23: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_37: "f32[8, 184, 14, 14]" = torch.ops.aten.where.self(lt_18, scalar_tensor_23, where_36);  lt_18 = scalar_tensor_23 = where_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_400: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_401: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 2);  unsqueeze_400 = None
    unsqueeze_402: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 3);  unsqueeze_401 = None
    sum_54: "f32[184]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
    sub_118: "f32[8, 184, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_402)
    mul_542: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(where_37, sub_118);  sub_118 = None
    sum_55: "f32[184]" = torch.ops.aten.sum.dim_IntList(mul_542, [0, 2, 3]);  mul_542 = None
    mul_543: "f32[184]" = torch.ops.aten.mul.Tensor(sum_54, 0.0006377551020408163)
    unsqueeze_403: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_543, 0);  mul_543 = None
    unsqueeze_404: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 2);  unsqueeze_403 = None
    unsqueeze_405: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 3);  unsqueeze_404 = None
    mul_544: "f32[184]" = torch.ops.aten.mul.Tensor(sum_55, 0.0006377551020408163)
    mul_545: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_546: "f32[184]" = torch.ops.aten.mul.Tensor(mul_544, mul_545);  mul_544 = mul_545 = None
    unsqueeze_406: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_546, 0);  mul_546 = None
    unsqueeze_407: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 2);  unsqueeze_406 = None
    unsqueeze_408: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 3);  unsqueeze_407 = None
    mul_547: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_55);  primals_55 = None
    unsqueeze_409: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_547, 0);  mul_547 = None
    unsqueeze_410: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 2);  unsqueeze_409 = None
    unsqueeze_411: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 3);  unsqueeze_410 = None
    sub_119: "f32[8, 184, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_402);  convolution_33 = unsqueeze_402 = None
    mul_548: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_408);  sub_119 = unsqueeze_408 = None
    sub_120: "f32[8, 184, 14, 14]" = torch.ops.aten.sub.Tensor(where_37, mul_548);  where_37 = mul_548 = None
    sub_121: "f32[8, 184, 14, 14]" = torch.ops.aten.sub.Tensor(sub_120, unsqueeze_405);  sub_120 = unsqueeze_405 = None
    mul_549: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_411);  sub_121 = unsqueeze_411 = None
    mul_550: "f32[184]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_82);  sum_55 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_549, add_150, primals_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_549 = add_150 = primals_134 = None
    getitem_179: "f32[8, 80, 14, 14]" = convolution_backward_29[0]
    getitem_180: "f32[184, 80, 1, 1]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_291: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(getitem_170, getitem_179);  getitem_170 = getitem_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_412: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_413: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 2);  unsqueeze_412 = None
    unsqueeze_414: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 3);  unsqueeze_413 = None
    sum_56: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_291, [0, 2, 3])
    sub_122: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_414)
    mul_551: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_291, sub_122);  sub_122 = None
    sum_57: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_551, [0, 2, 3]);  mul_551 = None
    mul_552: "f32[80]" = torch.ops.aten.mul.Tensor(sum_56, 0.0006377551020408163)
    unsqueeze_415: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_552, 0);  mul_552 = None
    unsqueeze_416: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 2);  unsqueeze_415 = None
    unsqueeze_417: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 3);  unsqueeze_416 = None
    mul_553: "f32[80]" = torch.ops.aten.mul.Tensor(sum_57, 0.0006377551020408163)
    mul_554: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_555: "f32[80]" = torch.ops.aten.mul.Tensor(mul_553, mul_554);  mul_553 = mul_554 = None
    unsqueeze_418: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_555, 0);  mul_555 = None
    unsqueeze_419: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 2);  unsqueeze_418 = None
    unsqueeze_420: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 3);  unsqueeze_419 = None
    mul_556: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_53);  primals_53 = None
    unsqueeze_421: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_556, 0);  mul_556 = None
    unsqueeze_422: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None
    unsqueeze_423: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 3);  unsqueeze_422 = None
    sub_123: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_414);  convolution_32 = unsqueeze_414 = None
    mul_557: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_420);  sub_123 = unsqueeze_420 = None
    sub_124: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(add_291, mul_557);  mul_557 = None
    sub_125: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(sub_124, unsqueeze_417);  sub_124 = unsqueeze_417 = None
    mul_558: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_423);  sub_125 = unsqueeze_423 = None
    mul_559: "f32[80]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_79);  sum_57 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_558, div_9, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_558 = div_9 = primals_133 = None
    getitem_182: "f32[8, 184, 14, 14]" = convolution_backward_30[0]
    getitem_183: "f32[80, 184, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_19: "b8[8, 184, 14, 14]" = torch.ops.aten.lt.Scalar(clone_6, -3)
    le_19: "b8[8, 184, 14, 14]" = torch.ops.aten.le.Scalar(clone_6, 3)
    div_49: "f32[8, 184, 14, 14]" = torch.ops.aten.div.Tensor(clone_6, 3);  clone_6 = None
    add_292: "f32[8, 184, 14, 14]" = torch.ops.aten.add.Tensor(div_49, 0.5);  div_49 = None
    mul_560: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_182, add_292);  add_292 = None
    where_38: "f32[8, 184, 14, 14]" = torch.ops.aten.where.self(le_19, mul_560, getitem_182);  le_19 = mul_560 = getitem_182 = None
    scalar_tensor_24: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_39: "f32[8, 184, 14, 14]" = torch.ops.aten.where.self(lt_19, scalar_tensor_24, where_38);  lt_19 = scalar_tensor_24 = where_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_424: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_425: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 2);  unsqueeze_424 = None
    unsqueeze_426: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 3);  unsqueeze_425 = None
    sum_58: "f32[184]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
    sub_126: "f32[8, 184, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_426)
    mul_561: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(where_39, sub_126);  sub_126 = None
    sum_59: "f32[184]" = torch.ops.aten.sum.dim_IntList(mul_561, [0, 2, 3]);  mul_561 = None
    mul_562: "f32[184]" = torch.ops.aten.mul.Tensor(sum_58, 0.0006377551020408163)
    unsqueeze_427: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_562, 0);  mul_562 = None
    unsqueeze_428: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 2);  unsqueeze_427 = None
    unsqueeze_429: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 3);  unsqueeze_428 = None
    mul_563: "f32[184]" = torch.ops.aten.mul.Tensor(sum_59, 0.0006377551020408163)
    mul_564: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_565: "f32[184]" = torch.ops.aten.mul.Tensor(mul_563, mul_564);  mul_563 = mul_564 = None
    unsqueeze_430: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_565, 0);  mul_565 = None
    unsqueeze_431: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 2);  unsqueeze_430 = None
    unsqueeze_432: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 3);  unsqueeze_431 = None
    mul_566: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_51);  primals_51 = None
    unsqueeze_433: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_566, 0);  mul_566 = None
    unsqueeze_434: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
    unsqueeze_435: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 3);  unsqueeze_434 = None
    sub_127: "f32[8, 184, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_426);  convolution_31 = unsqueeze_426 = None
    mul_567: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_432);  sub_127 = unsqueeze_432 = None
    sub_128: "f32[8, 184, 14, 14]" = torch.ops.aten.sub.Tensor(where_39, mul_567);  where_39 = mul_567 = None
    sub_129: "f32[8, 184, 14, 14]" = torch.ops.aten.sub.Tensor(sub_128, unsqueeze_429);  sub_128 = unsqueeze_429 = None
    mul_568: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_435);  sub_129 = unsqueeze_435 = None
    mul_569: "f32[184]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_76);  sum_59 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_568, div_8, primals_132, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 184, [True, True, False]);  mul_568 = div_8 = primals_132 = None
    getitem_185: "f32[8, 184, 14, 14]" = convolution_backward_31[0]
    getitem_186: "f32[184, 1, 3, 3]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_20: "b8[8, 184, 14, 14]" = torch.ops.aten.lt.Scalar(clone_5, -3)
    le_20: "b8[8, 184, 14, 14]" = torch.ops.aten.le.Scalar(clone_5, 3)
    div_50: "f32[8, 184, 14, 14]" = torch.ops.aten.div.Tensor(clone_5, 3);  clone_5 = None
    add_293: "f32[8, 184, 14, 14]" = torch.ops.aten.add.Tensor(div_50, 0.5);  div_50 = None
    mul_570: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_185, add_293);  add_293 = None
    where_40: "f32[8, 184, 14, 14]" = torch.ops.aten.where.self(le_20, mul_570, getitem_185);  le_20 = mul_570 = getitem_185 = None
    scalar_tensor_25: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_41: "f32[8, 184, 14, 14]" = torch.ops.aten.where.self(lt_20, scalar_tensor_25, where_40);  lt_20 = scalar_tensor_25 = where_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_436: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_437: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 2);  unsqueeze_436 = None
    unsqueeze_438: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 3);  unsqueeze_437 = None
    sum_60: "f32[184]" = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
    sub_130: "f32[8, 184, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_438)
    mul_571: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(where_41, sub_130);  sub_130 = None
    sum_61: "f32[184]" = torch.ops.aten.sum.dim_IntList(mul_571, [0, 2, 3]);  mul_571 = None
    mul_572: "f32[184]" = torch.ops.aten.mul.Tensor(sum_60, 0.0006377551020408163)
    unsqueeze_439: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_572, 0);  mul_572 = None
    unsqueeze_440: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 2);  unsqueeze_439 = None
    unsqueeze_441: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 3);  unsqueeze_440 = None
    mul_573: "f32[184]" = torch.ops.aten.mul.Tensor(sum_61, 0.0006377551020408163)
    mul_574: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_575: "f32[184]" = torch.ops.aten.mul.Tensor(mul_573, mul_574);  mul_573 = mul_574 = None
    unsqueeze_442: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_575, 0);  mul_575 = None
    unsqueeze_443: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 2);  unsqueeze_442 = None
    unsqueeze_444: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 3);  unsqueeze_443 = None
    mul_576: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_49);  primals_49 = None
    unsqueeze_445: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_576, 0);  mul_576 = None
    unsqueeze_446: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
    unsqueeze_447: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 3);  unsqueeze_446 = None
    sub_131: "f32[8, 184, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_438);  convolution_30 = unsqueeze_438 = None
    mul_577: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_444);  sub_131 = unsqueeze_444 = None
    sub_132: "f32[8, 184, 14, 14]" = torch.ops.aten.sub.Tensor(where_41, mul_577);  where_41 = mul_577 = None
    sub_133: "f32[8, 184, 14, 14]" = torch.ops.aten.sub.Tensor(sub_132, unsqueeze_441);  sub_132 = unsqueeze_441 = None
    mul_578: "f32[8, 184, 14, 14]" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_447);  sub_133 = unsqueeze_447 = None
    mul_579: "f32[184]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_73);  sum_61 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_578, add_132, primals_131, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_578 = add_132 = primals_131 = None
    getitem_188: "f32[8, 80, 14, 14]" = convolution_backward_32[0]
    getitem_189: "f32[184, 80, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_294: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_291, getitem_188);  add_291 = getitem_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_448: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_449: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, 2);  unsqueeze_448 = None
    unsqueeze_450: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 3);  unsqueeze_449 = None
    sum_62: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_294, [0, 2, 3])
    sub_134: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_450)
    mul_580: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_294, sub_134);  sub_134 = None
    sum_63: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_580, [0, 2, 3]);  mul_580 = None
    mul_581: "f32[80]" = torch.ops.aten.mul.Tensor(sum_62, 0.0006377551020408163)
    unsqueeze_451: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_581, 0);  mul_581 = None
    unsqueeze_452: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 2);  unsqueeze_451 = None
    unsqueeze_453: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 3);  unsqueeze_452 = None
    mul_582: "f32[80]" = torch.ops.aten.mul.Tensor(sum_63, 0.0006377551020408163)
    mul_583: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_584: "f32[80]" = torch.ops.aten.mul.Tensor(mul_582, mul_583);  mul_582 = mul_583 = None
    unsqueeze_454: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_584, 0);  mul_584 = None
    unsqueeze_455: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 2);  unsqueeze_454 = None
    unsqueeze_456: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 3);  unsqueeze_455 = None
    mul_585: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_47);  primals_47 = None
    unsqueeze_457: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_585, 0);  mul_585 = None
    unsqueeze_458: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 2);  unsqueeze_457 = None
    unsqueeze_459: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 3);  unsqueeze_458 = None
    sub_135: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_450);  convolution_29 = unsqueeze_450 = None
    mul_586: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_456);  sub_135 = unsqueeze_456 = None
    sub_136: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(add_294, mul_586);  mul_586 = None
    sub_137: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(sub_136, unsqueeze_453);  sub_136 = unsqueeze_453 = None
    mul_587: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_459);  sub_137 = unsqueeze_459 = None
    mul_588: "f32[80]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_70);  sum_63 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_587, div_7, primals_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_587 = div_7 = primals_130 = None
    getitem_191: "f32[8, 200, 14, 14]" = convolution_backward_33[0]
    getitem_192: "f32[80, 200, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_21: "b8[8, 200, 14, 14]" = torch.ops.aten.lt.Scalar(clone_4, -3)
    le_21: "b8[8, 200, 14, 14]" = torch.ops.aten.le.Scalar(clone_4, 3)
    div_51: "f32[8, 200, 14, 14]" = torch.ops.aten.div.Tensor(clone_4, 3);  clone_4 = None
    add_295: "f32[8, 200, 14, 14]" = torch.ops.aten.add.Tensor(div_51, 0.5);  div_51 = None
    mul_589: "f32[8, 200, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_191, add_295);  add_295 = None
    where_42: "f32[8, 200, 14, 14]" = torch.ops.aten.where.self(le_21, mul_589, getitem_191);  le_21 = mul_589 = getitem_191 = None
    scalar_tensor_26: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_43: "f32[8, 200, 14, 14]" = torch.ops.aten.where.self(lt_21, scalar_tensor_26, where_42);  lt_21 = scalar_tensor_26 = where_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_460: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_461: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, 2);  unsqueeze_460 = None
    unsqueeze_462: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 3);  unsqueeze_461 = None
    sum_64: "f32[200]" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
    sub_138: "f32[8, 200, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_462)
    mul_590: "f32[8, 200, 14, 14]" = torch.ops.aten.mul.Tensor(where_43, sub_138);  sub_138 = None
    sum_65: "f32[200]" = torch.ops.aten.sum.dim_IntList(mul_590, [0, 2, 3]);  mul_590 = None
    mul_591: "f32[200]" = torch.ops.aten.mul.Tensor(sum_64, 0.0006377551020408163)
    unsqueeze_463: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_591, 0);  mul_591 = None
    unsqueeze_464: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 2);  unsqueeze_463 = None
    unsqueeze_465: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 3);  unsqueeze_464 = None
    mul_592: "f32[200]" = torch.ops.aten.mul.Tensor(sum_65, 0.0006377551020408163)
    mul_593: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_594: "f32[200]" = torch.ops.aten.mul.Tensor(mul_592, mul_593);  mul_592 = mul_593 = None
    unsqueeze_466: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_594, 0);  mul_594 = None
    unsqueeze_467: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 2);  unsqueeze_466 = None
    unsqueeze_468: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 3);  unsqueeze_467 = None
    mul_595: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_45);  primals_45 = None
    unsqueeze_469: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_595, 0);  mul_595 = None
    unsqueeze_470: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None
    unsqueeze_471: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 3);  unsqueeze_470 = None
    sub_139: "f32[8, 200, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_462);  convolution_28 = unsqueeze_462 = None
    mul_596: "f32[8, 200, 14, 14]" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_468);  sub_139 = unsqueeze_468 = None
    sub_140: "f32[8, 200, 14, 14]" = torch.ops.aten.sub.Tensor(where_43, mul_596);  where_43 = mul_596 = None
    sub_141: "f32[8, 200, 14, 14]" = torch.ops.aten.sub.Tensor(sub_140, unsqueeze_465);  sub_140 = unsqueeze_465 = None
    mul_597: "f32[8, 200, 14, 14]" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_471);  sub_141 = unsqueeze_471 = None
    mul_598: "f32[200]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_67);  sum_65 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_597, div_6, primals_129, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 200, [True, True, False]);  mul_597 = div_6 = primals_129 = None
    getitem_194: "f32[8, 200, 14, 14]" = convolution_backward_34[0]
    getitem_195: "f32[200, 1, 3, 3]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_22: "b8[8, 200, 14, 14]" = torch.ops.aten.lt.Scalar(clone_3, -3)
    le_22: "b8[8, 200, 14, 14]" = torch.ops.aten.le.Scalar(clone_3, 3)
    div_52: "f32[8, 200, 14, 14]" = torch.ops.aten.div.Tensor(clone_3, 3);  clone_3 = None
    add_296: "f32[8, 200, 14, 14]" = torch.ops.aten.add.Tensor(div_52, 0.5);  div_52 = None
    mul_599: "f32[8, 200, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_194, add_296);  add_296 = None
    where_44: "f32[8, 200, 14, 14]" = torch.ops.aten.where.self(le_22, mul_599, getitem_194);  le_22 = mul_599 = getitem_194 = None
    scalar_tensor_27: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_45: "f32[8, 200, 14, 14]" = torch.ops.aten.where.self(lt_22, scalar_tensor_27, where_44);  lt_22 = scalar_tensor_27 = where_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_472: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_473: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, 2);  unsqueeze_472 = None
    unsqueeze_474: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 3);  unsqueeze_473 = None
    sum_66: "f32[200]" = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
    sub_142: "f32[8, 200, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_474)
    mul_600: "f32[8, 200, 14, 14]" = torch.ops.aten.mul.Tensor(where_45, sub_142);  sub_142 = None
    sum_67: "f32[200]" = torch.ops.aten.sum.dim_IntList(mul_600, [0, 2, 3]);  mul_600 = None
    mul_601: "f32[200]" = torch.ops.aten.mul.Tensor(sum_66, 0.0006377551020408163)
    unsqueeze_475: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_601, 0);  mul_601 = None
    unsqueeze_476: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 2);  unsqueeze_475 = None
    unsqueeze_477: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 3);  unsqueeze_476 = None
    mul_602: "f32[200]" = torch.ops.aten.mul.Tensor(sum_67, 0.0006377551020408163)
    mul_603: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_604: "f32[200]" = torch.ops.aten.mul.Tensor(mul_602, mul_603);  mul_602 = mul_603 = None
    unsqueeze_478: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_604, 0);  mul_604 = None
    unsqueeze_479: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 2);  unsqueeze_478 = None
    unsqueeze_480: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 3);  unsqueeze_479 = None
    mul_605: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_43);  primals_43 = None
    unsqueeze_481: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_605, 0);  mul_605 = None
    unsqueeze_482: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 2);  unsqueeze_481 = None
    unsqueeze_483: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 3);  unsqueeze_482 = None
    sub_143: "f32[8, 200, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_474);  convolution_27 = unsqueeze_474 = None
    mul_606: "f32[8, 200, 14, 14]" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_480);  sub_143 = unsqueeze_480 = None
    sub_144: "f32[8, 200, 14, 14]" = torch.ops.aten.sub.Tensor(where_45, mul_606);  where_45 = mul_606 = None
    sub_145: "f32[8, 200, 14, 14]" = torch.ops.aten.sub.Tensor(sub_144, unsqueeze_477);  sub_144 = unsqueeze_477 = None
    mul_607: "f32[8, 200, 14, 14]" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_483);  sub_145 = unsqueeze_483 = None
    mul_608: "f32[200]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_64);  sum_67 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_607, add_114, primals_128, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_607 = add_114 = primals_128 = None
    getitem_197: "f32[8, 80, 14, 14]" = convolution_backward_35[0]
    getitem_198: "f32[200, 80, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_297: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_294, getitem_197);  add_294 = getitem_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_484: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_485: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 2);  unsqueeze_484 = None
    unsqueeze_486: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 3);  unsqueeze_485 = None
    sum_68: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_297, [0, 2, 3])
    sub_146: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_486)
    mul_609: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_297, sub_146);  sub_146 = None
    sum_69: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_609, [0, 2, 3]);  mul_609 = None
    mul_610: "f32[80]" = torch.ops.aten.mul.Tensor(sum_68, 0.0006377551020408163)
    unsqueeze_487: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_610, 0);  mul_610 = None
    unsqueeze_488: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
    unsqueeze_489: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 3);  unsqueeze_488 = None
    mul_611: "f32[80]" = torch.ops.aten.mul.Tensor(sum_69, 0.0006377551020408163)
    mul_612: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_613: "f32[80]" = torch.ops.aten.mul.Tensor(mul_611, mul_612);  mul_611 = mul_612 = None
    unsqueeze_490: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_613, 0);  mul_613 = None
    unsqueeze_491: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 2);  unsqueeze_490 = None
    unsqueeze_492: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 3);  unsqueeze_491 = None
    mul_614: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_41);  primals_41 = None
    unsqueeze_493: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_614, 0);  mul_614 = None
    unsqueeze_494: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
    unsqueeze_495: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 3);  unsqueeze_494 = None
    sub_147: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_486);  convolution_26 = unsqueeze_486 = None
    mul_615: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_492);  sub_147 = unsqueeze_492 = None
    sub_148: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(add_297, mul_615);  add_297 = mul_615 = None
    sub_149: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(sub_148, unsqueeze_489);  sub_148 = unsqueeze_489 = None
    mul_616: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_495);  sub_149 = unsqueeze_495 = None
    mul_617: "f32[80]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_61);  sum_69 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_616, div_5, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_616 = div_5 = primals_127 = None
    getitem_200: "f32[8, 240, 14, 14]" = convolution_backward_36[0]
    getitem_201: "f32[80, 240, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_23: "b8[8, 240, 14, 14]" = torch.ops.aten.lt.Scalar(clone_2, -3)
    le_23: "b8[8, 240, 14, 14]" = torch.ops.aten.le.Scalar(clone_2, 3)
    div_53: "f32[8, 240, 14, 14]" = torch.ops.aten.div.Tensor(clone_2, 3);  clone_2 = None
    add_298: "f32[8, 240, 14, 14]" = torch.ops.aten.add.Tensor(div_53, 0.5);  div_53 = None
    mul_618: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_200, add_298);  add_298 = None
    where_46: "f32[8, 240, 14, 14]" = torch.ops.aten.where.self(le_23, mul_618, getitem_200);  le_23 = mul_618 = getitem_200 = None
    scalar_tensor_28: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_47: "f32[8, 240, 14, 14]" = torch.ops.aten.where.self(lt_23, scalar_tensor_28, where_46);  lt_23 = scalar_tensor_28 = where_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_496: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_497: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 2);  unsqueeze_496 = None
    unsqueeze_498: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 3);  unsqueeze_497 = None
    sum_70: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_47, [0, 2, 3])
    sub_150: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_498)
    mul_619: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(where_47, sub_150);  sub_150 = None
    sum_71: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_619, [0, 2, 3]);  mul_619 = None
    mul_620: "f32[240]" = torch.ops.aten.mul.Tensor(sum_70, 0.0006377551020408163)
    unsqueeze_499: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_620, 0);  mul_620 = None
    unsqueeze_500: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 2);  unsqueeze_499 = None
    unsqueeze_501: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 3);  unsqueeze_500 = None
    mul_621: "f32[240]" = torch.ops.aten.mul.Tensor(sum_71, 0.0006377551020408163)
    mul_622: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_623: "f32[240]" = torch.ops.aten.mul.Tensor(mul_621, mul_622);  mul_621 = mul_622 = None
    unsqueeze_502: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_623, 0);  mul_623 = None
    unsqueeze_503: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 2);  unsqueeze_502 = None
    unsqueeze_504: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 3);  unsqueeze_503 = None
    mul_624: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_39);  primals_39 = None
    unsqueeze_505: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_624, 0);  mul_624 = None
    unsqueeze_506: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 2);  unsqueeze_505 = None
    unsqueeze_507: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 3);  unsqueeze_506 = None
    sub_151: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_498);  convolution_25 = unsqueeze_498 = None
    mul_625: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_504);  sub_151 = unsqueeze_504 = None
    sub_152: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(where_47, mul_625);  where_47 = mul_625 = None
    sub_153: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(sub_152, unsqueeze_501);  sub_152 = unsqueeze_501 = None
    mul_626: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_507);  sub_153 = unsqueeze_507 = None
    mul_627: "f32[240]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_58);  sum_71 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_626, div_4, primals_126, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 240, [True, True, False]);  mul_626 = div_4 = primals_126 = None
    getitem_203: "f32[8, 240, 28, 28]" = convolution_backward_37[0]
    getitem_204: "f32[240, 1, 3, 3]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_24: "b8[8, 240, 28, 28]" = torch.ops.aten.lt.Scalar(clone_1, -3)
    le_24: "b8[8, 240, 28, 28]" = torch.ops.aten.le.Scalar(clone_1, 3)
    div_54: "f32[8, 240, 28, 28]" = torch.ops.aten.div.Tensor(clone_1, 3);  clone_1 = None
    add_299: "f32[8, 240, 28, 28]" = torch.ops.aten.add.Tensor(div_54, 0.5);  div_54 = None
    mul_628: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_203, add_299);  add_299 = None
    where_48: "f32[8, 240, 28, 28]" = torch.ops.aten.where.self(le_24, mul_628, getitem_203);  le_24 = mul_628 = getitem_203 = None
    scalar_tensor_29: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_49: "f32[8, 240, 28, 28]" = torch.ops.aten.where.self(lt_24, scalar_tensor_29, where_48);  lt_24 = scalar_tensor_29 = where_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_508: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_509: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, 2);  unsqueeze_508 = None
    unsqueeze_510: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 3);  unsqueeze_509 = None
    sum_72: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_49, [0, 2, 3])
    sub_154: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_510)
    mul_629: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(where_49, sub_154);  sub_154 = None
    sum_73: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_629, [0, 2, 3]);  mul_629 = None
    mul_630: "f32[240]" = torch.ops.aten.mul.Tensor(sum_72, 0.00015943877551020407)
    unsqueeze_511: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_630, 0);  mul_630 = None
    unsqueeze_512: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 2);  unsqueeze_511 = None
    unsqueeze_513: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 3);  unsqueeze_512 = None
    mul_631: "f32[240]" = torch.ops.aten.mul.Tensor(sum_73, 0.00015943877551020407)
    mul_632: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_633: "f32[240]" = torch.ops.aten.mul.Tensor(mul_631, mul_632);  mul_631 = mul_632 = None
    unsqueeze_514: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_633, 0);  mul_633 = None
    unsqueeze_515: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 2);  unsqueeze_514 = None
    unsqueeze_516: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 3);  unsqueeze_515 = None
    mul_634: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_37);  primals_37 = None
    unsqueeze_517: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_634, 0);  mul_634 = None
    unsqueeze_518: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 2);  unsqueeze_517 = None
    unsqueeze_519: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 3);  unsqueeze_518 = None
    sub_155: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_510);  convolution_24 = unsqueeze_510 = None
    mul_635: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_516);  sub_155 = unsqueeze_516 = None
    sub_156: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(where_49, mul_635);  where_49 = mul_635 = None
    sub_157: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(sub_156, unsqueeze_513);  sub_156 = unsqueeze_513 = None
    mul_636: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_519);  sub_157 = unsqueeze_519 = None
    mul_637: "f32[240]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_55);  sum_73 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_636, add_97, primals_125, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_636 = add_97 = primals_125 = None
    getitem_206: "f32[8, 40, 28, 28]" = convolution_backward_38[0]
    getitem_207: "f32[240, 40, 1, 1]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_520: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_521: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 2);  unsqueeze_520 = None
    unsqueeze_522: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 3);  unsqueeze_521 = None
    sum_74: "f32[40]" = torch.ops.aten.sum.dim_IntList(getitem_206, [0, 2, 3])
    sub_158: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_522)
    mul_638: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_206, sub_158);  sub_158 = None
    sum_75: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_638, [0, 2, 3]);  mul_638 = None
    mul_639: "f32[40]" = torch.ops.aten.mul.Tensor(sum_74, 0.00015943877551020407)
    unsqueeze_523: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_639, 0);  mul_639 = None
    unsqueeze_524: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 2);  unsqueeze_523 = None
    unsqueeze_525: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 3);  unsqueeze_524 = None
    mul_640: "f32[40]" = torch.ops.aten.mul.Tensor(sum_75, 0.00015943877551020407)
    mul_641: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_642: "f32[40]" = torch.ops.aten.mul.Tensor(mul_640, mul_641);  mul_640 = mul_641 = None
    unsqueeze_526: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_642, 0);  mul_642 = None
    unsqueeze_527: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 2);  unsqueeze_526 = None
    unsqueeze_528: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 3);  unsqueeze_527 = None
    mul_643: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_35);  primals_35 = None
    unsqueeze_529: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_643, 0);  mul_643 = None
    unsqueeze_530: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 2);  unsqueeze_529 = None
    unsqueeze_531: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 3);  unsqueeze_530 = None
    sub_159: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_522);  convolution_23 = unsqueeze_522 = None
    mul_644: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_528);  sub_159 = unsqueeze_528 = None
    sub_160: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_206, mul_644);  mul_644 = None
    sub_161: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_160, unsqueeze_525);  sub_160 = unsqueeze_525 = None
    mul_645: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_531);  sub_161 = unsqueeze_531 = None
    mul_646: "f32[40]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_52);  sum_75 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_645, mul_122, primals_124, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_645 = mul_122 = primals_124 = None
    getitem_209: "f32[8, 120, 28, 28]" = convolution_backward_39[0]
    getitem_210: "f32[40, 120, 1, 1]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_647: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_209, relu_12)
    mul_648: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_209, div_3);  getitem_209 = div_3 = None
    sum_76: "f32[8, 120, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_647, [2, 3], True);  mul_647 = None
    gt_5: "b8[8, 120, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_22, -3.0)
    lt_25: "b8[8, 120, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_22, 3.0);  convolution_22 = None
    bitwise_and_5: "b8[8, 120, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_5, lt_25);  gt_5 = lt_25 = None
    mul_649: "f32[8, 120, 1, 1]" = torch.ops.aten.mul.Tensor(sum_76, 0.16666666666666666);  sum_76 = None
    scalar_tensor_30: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_50: "f32[8, 120, 1, 1]" = torch.ops.aten.where.self(bitwise_and_5, mul_649, scalar_tensor_30);  bitwise_and_5 = mul_649 = scalar_tensor_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_77: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_50, [0, 2, 3])
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(where_50, relu_13, primals_122, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_50 = primals_122 = None
    getitem_212: "f32[8, 32, 1, 1]" = convolution_backward_40[0]
    getitem_213: "f32[120, 32, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    alias_35: "f32[8, 32, 1, 1]" = torch.ops.aten.alias.default(relu_13);  relu_13 = None
    alias_36: "f32[8, 32, 1, 1]" = torch.ops.aten.alias.default(alias_35);  alias_35 = None
    le_25: "b8[8, 32, 1, 1]" = torch.ops.aten.le.Scalar(alias_36, 0);  alias_36 = None
    scalar_tensor_31: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_51: "f32[8, 32, 1, 1]" = torch.ops.aten.where.self(le_25, scalar_tensor_31, getitem_212);  le_25 = scalar_tensor_31 = getitem_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_78: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_51, [0, 2, 3])
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(where_51, mean_2, primals_120, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_51 = mean_2 = primals_120 = None
    getitem_215: "f32[8, 120, 1, 1]" = convolution_backward_41[0]
    getitem_216: "f32[32, 120, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_6: "f32[8, 120, 28, 28]" = torch.ops.aten.expand.default(getitem_215, [8, 120, 28, 28]);  getitem_215 = None
    div_55: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Scalar(expand_6, 784);  expand_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_300: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_648, div_55);  mul_648 = div_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_38: "f32[8, 120, 28, 28]" = torch.ops.aten.alias.default(relu_12);  relu_12 = None
    alias_39: "f32[8, 120, 28, 28]" = torch.ops.aten.alias.default(alias_38);  alias_38 = None
    le_26: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(alias_39, 0);  alias_39 = None
    scalar_tensor_32: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_52: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_26, scalar_tensor_32, add_300);  le_26 = scalar_tensor_32 = add_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_532: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_533: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 2);  unsqueeze_532 = None
    unsqueeze_534: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 3);  unsqueeze_533 = None
    sum_79: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_52, [0, 2, 3])
    sub_162: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_534)
    mul_650: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_52, sub_162);  sub_162 = None
    sum_80: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_650, [0, 2, 3]);  mul_650 = None
    mul_651: "f32[120]" = torch.ops.aten.mul.Tensor(sum_79, 0.00015943877551020407)
    unsqueeze_535: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_651, 0);  mul_651 = None
    unsqueeze_536: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 2);  unsqueeze_535 = None
    unsqueeze_537: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 3);  unsqueeze_536 = None
    mul_652: "f32[120]" = torch.ops.aten.mul.Tensor(sum_80, 0.00015943877551020407)
    mul_653: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_654: "f32[120]" = torch.ops.aten.mul.Tensor(mul_652, mul_653);  mul_652 = mul_653 = None
    unsqueeze_538: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_654, 0);  mul_654 = None
    unsqueeze_539: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 2);  unsqueeze_538 = None
    unsqueeze_540: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 3);  unsqueeze_539 = None
    mul_655: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_33);  primals_33 = None
    unsqueeze_541: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_655, 0);  mul_655 = None
    unsqueeze_542: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 2);  unsqueeze_541 = None
    unsqueeze_543: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 3);  unsqueeze_542 = None
    sub_163: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_534);  convolution_20 = unsqueeze_534 = None
    mul_656: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_540);  sub_163 = unsqueeze_540 = None
    sub_164: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_52, mul_656);  where_52 = mul_656 = None
    sub_165: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_164, unsqueeze_537);  sub_164 = unsqueeze_537 = None
    mul_657: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_543);  sub_165 = unsqueeze_543 = None
    mul_658: "f32[120]" = torch.ops.aten.mul.Tensor(sum_80, squeeze_49);  sum_80 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_657, relu_11, primals_119, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False]);  mul_657 = primals_119 = None
    getitem_218: "f32[8, 120, 28, 28]" = convolution_backward_42[0]
    getitem_219: "f32[120, 1, 5, 5]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_41: "f32[8, 120, 28, 28]" = torch.ops.aten.alias.default(relu_11);  relu_11 = None
    alias_42: "f32[8, 120, 28, 28]" = torch.ops.aten.alias.default(alias_41);  alias_41 = None
    le_27: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(alias_42, 0);  alias_42 = None
    scalar_tensor_33: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_53: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_27, scalar_tensor_33, getitem_218);  le_27 = scalar_tensor_33 = getitem_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_544: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_545: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, 2);  unsqueeze_544 = None
    unsqueeze_546: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 3);  unsqueeze_545 = None
    sum_81: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_53, [0, 2, 3])
    sub_166: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_546)
    mul_659: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_53, sub_166);  sub_166 = None
    sum_82: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_659, [0, 2, 3]);  mul_659 = None
    mul_660: "f32[120]" = torch.ops.aten.mul.Tensor(sum_81, 0.00015943877551020407)
    unsqueeze_547: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_660, 0);  mul_660 = None
    unsqueeze_548: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_547, 2);  unsqueeze_547 = None
    unsqueeze_549: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 3);  unsqueeze_548 = None
    mul_661: "f32[120]" = torch.ops.aten.mul.Tensor(sum_82, 0.00015943877551020407)
    mul_662: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_663: "f32[120]" = torch.ops.aten.mul.Tensor(mul_661, mul_662);  mul_661 = mul_662 = None
    unsqueeze_550: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_663, 0);  mul_663 = None
    unsqueeze_551: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 2);  unsqueeze_550 = None
    unsqueeze_552: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 3);  unsqueeze_551 = None
    mul_664: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_31);  primals_31 = None
    unsqueeze_553: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_664, 0);  mul_664 = None
    unsqueeze_554: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 2);  unsqueeze_553 = None
    unsqueeze_555: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 3);  unsqueeze_554 = None
    sub_167: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_546);  convolution_19 = unsqueeze_546 = None
    mul_665: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_552);  sub_167 = unsqueeze_552 = None
    sub_168: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_53, mul_665);  where_53 = mul_665 = None
    sub_169: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_168, unsqueeze_549);  sub_168 = unsqueeze_549 = None
    mul_666: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_555);  sub_169 = unsqueeze_555 = None
    mul_667: "f32[120]" = torch.ops.aten.mul.Tensor(sum_82, squeeze_46);  sum_82 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_666, add_80, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_666 = add_80 = primals_118 = None
    getitem_221: "f32[8, 40, 28, 28]" = convolution_backward_43[0]
    getitem_222: "f32[120, 40, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_301: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(getitem_206, getitem_221);  getitem_206 = getitem_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_556: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_557: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 2);  unsqueeze_556 = None
    unsqueeze_558: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 3);  unsqueeze_557 = None
    sum_83: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_301, [0, 2, 3])
    sub_170: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_558)
    mul_668: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_301, sub_170);  sub_170 = None
    sum_84: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_668, [0, 2, 3]);  mul_668 = None
    mul_669: "f32[40]" = torch.ops.aten.mul.Tensor(sum_83, 0.00015943877551020407)
    unsqueeze_559: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_669, 0);  mul_669 = None
    unsqueeze_560: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 2);  unsqueeze_559 = None
    unsqueeze_561: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 3);  unsqueeze_560 = None
    mul_670: "f32[40]" = torch.ops.aten.mul.Tensor(sum_84, 0.00015943877551020407)
    mul_671: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_672: "f32[40]" = torch.ops.aten.mul.Tensor(mul_670, mul_671);  mul_670 = mul_671 = None
    unsqueeze_562: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_672, 0);  mul_672 = None
    unsqueeze_563: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 2);  unsqueeze_562 = None
    unsqueeze_564: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 3);  unsqueeze_563 = None
    mul_673: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_29);  primals_29 = None
    unsqueeze_565: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_673, 0);  mul_673 = None
    unsqueeze_566: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 2);  unsqueeze_565 = None
    unsqueeze_567: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 3);  unsqueeze_566 = None
    sub_171: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_558);  convolution_18 = unsqueeze_558 = None
    mul_674: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_564);  sub_171 = unsqueeze_564 = None
    sub_172: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(add_301, mul_674);  mul_674 = None
    sub_173: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_172, unsqueeze_561);  sub_172 = unsqueeze_561 = None
    mul_675: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_567);  sub_173 = unsqueeze_567 = None
    mul_676: "f32[40]" = torch.ops.aten.mul.Tensor(sum_84, squeeze_43);  sum_84 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_675, mul_100, primals_117, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_675 = mul_100 = primals_117 = None
    getitem_224: "f32[8, 120, 28, 28]" = convolution_backward_44[0]
    getitem_225: "f32[40, 120, 1, 1]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_677: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_224, relu_9)
    mul_678: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_224, div_2);  getitem_224 = div_2 = None
    sum_85: "f32[8, 120, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_677, [2, 3], True);  mul_677 = None
    gt_6: "b8[8, 120, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_17, -3.0)
    lt_26: "b8[8, 120, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_17, 3.0);  convolution_17 = None
    bitwise_and_6: "b8[8, 120, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_6, lt_26);  gt_6 = lt_26 = None
    mul_679: "f32[8, 120, 1, 1]" = torch.ops.aten.mul.Tensor(sum_85, 0.16666666666666666);  sum_85 = None
    scalar_tensor_34: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_54: "f32[8, 120, 1, 1]" = torch.ops.aten.where.self(bitwise_and_6, mul_679, scalar_tensor_34);  bitwise_and_6 = mul_679 = scalar_tensor_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_86: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_54, [0, 2, 3])
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(where_54, relu_10, primals_115, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_54 = primals_115 = None
    getitem_227: "f32[8, 32, 1, 1]" = convolution_backward_45[0]
    getitem_228: "f32[120, 32, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    alias_44: "f32[8, 32, 1, 1]" = torch.ops.aten.alias.default(relu_10);  relu_10 = None
    alias_45: "f32[8, 32, 1, 1]" = torch.ops.aten.alias.default(alias_44);  alias_44 = None
    le_28: "b8[8, 32, 1, 1]" = torch.ops.aten.le.Scalar(alias_45, 0);  alias_45 = None
    scalar_tensor_35: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_55: "f32[8, 32, 1, 1]" = torch.ops.aten.where.self(le_28, scalar_tensor_35, getitem_227);  le_28 = scalar_tensor_35 = getitem_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_87: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_55, [0, 2, 3])
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(where_55, mean_1, primals_113, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_55 = mean_1 = primals_113 = None
    getitem_230: "f32[8, 120, 1, 1]" = convolution_backward_46[0]
    getitem_231: "f32[32, 120, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_7: "f32[8, 120, 28, 28]" = torch.ops.aten.expand.default(getitem_230, [8, 120, 28, 28]);  getitem_230 = None
    div_56: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Scalar(expand_7, 784);  expand_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_302: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_678, div_56);  mul_678 = div_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_47: "f32[8, 120, 28, 28]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
    alias_48: "f32[8, 120, 28, 28]" = torch.ops.aten.alias.default(alias_47);  alias_47 = None
    le_29: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(alias_48, 0);  alias_48 = None
    scalar_tensor_36: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_56: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_29, scalar_tensor_36, add_302);  le_29 = scalar_tensor_36 = add_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_568: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_569: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, 2);  unsqueeze_568 = None
    unsqueeze_570: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 3);  unsqueeze_569 = None
    sum_88: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_56, [0, 2, 3])
    sub_174: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_570)
    mul_680: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_56, sub_174);  sub_174 = None
    sum_89: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_680, [0, 2, 3]);  mul_680 = None
    mul_681: "f32[120]" = torch.ops.aten.mul.Tensor(sum_88, 0.00015943877551020407)
    unsqueeze_571: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_681, 0);  mul_681 = None
    unsqueeze_572: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_571, 2);  unsqueeze_571 = None
    unsqueeze_573: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 3);  unsqueeze_572 = None
    mul_682: "f32[120]" = torch.ops.aten.mul.Tensor(sum_89, 0.00015943877551020407)
    mul_683: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_684: "f32[120]" = torch.ops.aten.mul.Tensor(mul_682, mul_683);  mul_682 = mul_683 = None
    unsqueeze_574: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_684, 0);  mul_684 = None
    unsqueeze_575: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, 2);  unsqueeze_574 = None
    unsqueeze_576: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 3);  unsqueeze_575 = None
    mul_685: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_27);  primals_27 = None
    unsqueeze_577: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_685, 0);  mul_685 = None
    unsqueeze_578: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 2);  unsqueeze_577 = None
    unsqueeze_579: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 3);  unsqueeze_578 = None
    sub_175: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_570);  convolution_15 = unsqueeze_570 = None
    mul_686: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_576);  sub_175 = unsqueeze_576 = None
    sub_176: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_56, mul_686);  where_56 = mul_686 = None
    sub_177: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_176, unsqueeze_573);  sub_176 = unsqueeze_573 = None
    mul_687: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_579);  sub_177 = unsqueeze_579 = None
    mul_688: "f32[120]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_40);  sum_89 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_687, relu_8, primals_112, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False]);  mul_687 = primals_112 = None
    getitem_233: "f32[8, 120, 28, 28]" = convolution_backward_47[0]
    getitem_234: "f32[120, 1, 5, 5]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_50: "f32[8, 120, 28, 28]" = torch.ops.aten.alias.default(relu_8);  relu_8 = None
    alias_51: "f32[8, 120, 28, 28]" = torch.ops.aten.alias.default(alias_50);  alias_50 = None
    le_30: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(alias_51, 0);  alias_51 = None
    scalar_tensor_37: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_57: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_30, scalar_tensor_37, getitem_233);  le_30 = scalar_tensor_37 = getitem_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_580: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_581: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 2);  unsqueeze_580 = None
    unsqueeze_582: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 3);  unsqueeze_581 = None
    sum_90: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_57, [0, 2, 3])
    sub_178: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_582)
    mul_689: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_57, sub_178);  sub_178 = None
    sum_91: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_689, [0, 2, 3]);  mul_689 = None
    mul_690: "f32[120]" = torch.ops.aten.mul.Tensor(sum_90, 0.00015943877551020407)
    unsqueeze_583: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_690, 0);  mul_690 = None
    unsqueeze_584: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_583, 2);  unsqueeze_583 = None
    unsqueeze_585: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 3);  unsqueeze_584 = None
    mul_691: "f32[120]" = torch.ops.aten.mul.Tensor(sum_91, 0.00015943877551020407)
    mul_692: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_693: "f32[120]" = torch.ops.aten.mul.Tensor(mul_691, mul_692);  mul_691 = mul_692 = None
    unsqueeze_586: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_693, 0);  mul_693 = None
    unsqueeze_587: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 2);  unsqueeze_586 = None
    unsqueeze_588: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 3);  unsqueeze_587 = None
    mul_694: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_25);  primals_25 = None
    unsqueeze_589: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_694, 0);  mul_694 = None
    unsqueeze_590: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 2);  unsqueeze_589 = None
    unsqueeze_591: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 3);  unsqueeze_590 = None
    sub_179: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_582);  convolution_14 = unsqueeze_582 = None
    mul_695: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_588);  sub_179 = unsqueeze_588 = None
    sub_180: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_57, mul_695);  where_57 = mul_695 = None
    sub_181: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_180, unsqueeze_585);  sub_180 = unsqueeze_585 = None
    mul_696: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_591);  sub_181 = unsqueeze_591 = None
    mul_697: "f32[120]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_37);  sum_91 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_696, add_63, primals_111, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_696 = add_63 = primals_111 = None
    getitem_236: "f32[8, 40, 28, 28]" = convolution_backward_48[0]
    getitem_237: "f32[120, 40, 1, 1]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_303: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_301, getitem_236);  add_301 = getitem_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_592: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_593: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 2);  unsqueeze_592 = None
    unsqueeze_594: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 3);  unsqueeze_593 = None
    sum_92: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_303, [0, 2, 3])
    sub_182: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_594)
    mul_698: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_303, sub_182);  sub_182 = None
    sum_93: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_698, [0, 2, 3]);  mul_698 = None
    mul_699: "f32[40]" = torch.ops.aten.mul.Tensor(sum_92, 0.00015943877551020407)
    unsqueeze_595: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_699, 0);  mul_699 = None
    unsqueeze_596: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 2);  unsqueeze_595 = None
    unsqueeze_597: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 3);  unsqueeze_596 = None
    mul_700: "f32[40]" = torch.ops.aten.mul.Tensor(sum_93, 0.00015943877551020407)
    mul_701: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_702: "f32[40]" = torch.ops.aten.mul.Tensor(mul_700, mul_701);  mul_700 = mul_701 = None
    unsqueeze_598: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_702, 0);  mul_702 = None
    unsqueeze_599: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 2);  unsqueeze_598 = None
    unsqueeze_600: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 3);  unsqueeze_599 = None
    mul_703: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_23);  primals_23 = None
    unsqueeze_601: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_703, 0);  mul_703 = None
    unsqueeze_602: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 2);  unsqueeze_601 = None
    unsqueeze_603: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 3);  unsqueeze_602 = None
    sub_183: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_594);  convolution_13 = unsqueeze_594 = None
    mul_704: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_600);  sub_183 = unsqueeze_600 = None
    sub_184: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(add_303, mul_704);  add_303 = mul_704 = None
    sub_185: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_184, unsqueeze_597);  sub_184 = unsqueeze_597 = None
    mul_705: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_603);  sub_185 = unsqueeze_603 = None
    mul_706: "f32[40]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_34);  sum_93 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_705, mul_78, primals_110, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_705 = mul_78 = primals_110 = None
    getitem_239: "f32[8, 72, 28, 28]" = convolution_backward_49[0]
    getitem_240: "f32[40, 72, 1, 1]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_707: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_239, relu_6)
    mul_708: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_239, div_1);  getitem_239 = div_1 = None
    sum_94: "f32[8, 72, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_707, [2, 3], True);  mul_707 = None
    gt_7: "b8[8, 72, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_12, -3.0)
    lt_27: "b8[8, 72, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_12, 3.0);  convolution_12 = None
    bitwise_and_7: "b8[8, 72, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_7, lt_27);  gt_7 = lt_27 = None
    mul_709: "f32[8, 72, 1, 1]" = torch.ops.aten.mul.Tensor(sum_94, 0.16666666666666666);  sum_94 = None
    scalar_tensor_38: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_58: "f32[8, 72, 1, 1]" = torch.ops.aten.where.self(bitwise_and_7, mul_709, scalar_tensor_38);  bitwise_and_7 = mul_709 = scalar_tensor_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_95: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_58, [0, 2, 3])
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(where_58, relu_7, primals_108, [72], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_58 = primals_108 = None
    getitem_242: "f32[8, 24, 1, 1]" = convolution_backward_50[0]
    getitem_243: "f32[72, 24, 1, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    alias_53: "f32[8, 24, 1, 1]" = torch.ops.aten.alias.default(relu_7);  relu_7 = None
    alias_54: "f32[8, 24, 1, 1]" = torch.ops.aten.alias.default(alias_53);  alias_53 = None
    le_31: "b8[8, 24, 1, 1]" = torch.ops.aten.le.Scalar(alias_54, 0);  alias_54 = None
    scalar_tensor_39: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_59: "f32[8, 24, 1, 1]" = torch.ops.aten.where.self(le_31, scalar_tensor_39, getitem_242);  le_31 = scalar_tensor_39 = getitem_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_96: "f32[24]" = torch.ops.aten.sum.dim_IntList(where_59, [0, 2, 3])
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(where_59, mean, primals_106, [24], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_59 = mean = primals_106 = None
    getitem_245: "f32[8, 72, 1, 1]" = convolution_backward_51[0]
    getitem_246: "f32[24, 72, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_8: "f32[8, 72, 28, 28]" = torch.ops.aten.expand.default(getitem_245, [8, 72, 28, 28]);  getitem_245 = None
    div_57: "f32[8, 72, 28, 28]" = torch.ops.aten.div.Scalar(expand_8, 784);  expand_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_304: "f32[8, 72, 28, 28]" = torch.ops.aten.add.Tensor(mul_708, div_57);  mul_708 = div_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_56: "f32[8, 72, 28, 28]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_57: "f32[8, 72, 28, 28]" = torch.ops.aten.alias.default(alias_56);  alias_56 = None
    le_32: "b8[8, 72, 28, 28]" = torch.ops.aten.le.Scalar(alias_57, 0);  alias_57 = None
    scalar_tensor_40: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_60: "f32[8, 72, 28, 28]" = torch.ops.aten.where.self(le_32, scalar_tensor_40, add_304);  le_32 = scalar_tensor_40 = add_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_604: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_605: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, 2);  unsqueeze_604 = None
    unsqueeze_606: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 3);  unsqueeze_605 = None
    sum_97: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_60, [0, 2, 3])
    sub_186: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_606)
    mul_710: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(where_60, sub_186);  sub_186 = None
    sum_98: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_710, [0, 2, 3]);  mul_710 = None
    mul_711: "f32[72]" = torch.ops.aten.mul.Tensor(sum_97, 0.00015943877551020407)
    unsqueeze_607: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_711, 0);  mul_711 = None
    unsqueeze_608: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_607, 2);  unsqueeze_607 = None
    unsqueeze_609: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 3);  unsqueeze_608 = None
    mul_712: "f32[72]" = torch.ops.aten.mul.Tensor(sum_98, 0.00015943877551020407)
    mul_713: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_714: "f32[72]" = torch.ops.aten.mul.Tensor(mul_712, mul_713);  mul_712 = mul_713 = None
    unsqueeze_610: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_714, 0);  mul_714 = None
    unsqueeze_611: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 2);  unsqueeze_610 = None
    unsqueeze_612: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 3);  unsqueeze_611 = None
    mul_715: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_21);  primals_21 = None
    unsqueeze_613: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_715, 0);  mul_715 = None
    unsqueeze_614: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 2);  unsqueeze_613 = None
    unsqueeze_615: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 3);  unsqueeze_614 = None
    sub_187: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_606);  convolution_10 = unsqueeze_606 = None
    mul_716: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_612);  sub_187 = unsqueeze_612 = None
    sub_188: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(where_60, mul_716);  where_60 = mul_716 = None
    sub_189: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(sub_188, unsqueeze_609);  sub_188 = unsqueeze_609 = None
    mul_717: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_615);  sub_189 = unsqueeze_615 = None
    mul_718: "f32[72]" = torch.ops.aten.mul.Tensor(sum_98, squeeze_31);  sum_98 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_717, relu_5, primals_105, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 72, [True, True, False]);  mul_717 = primals_105 = None
    getitem_248: "f32[8, 72, 56, 56]" = convolution_backward_52[0]
    getitem_249: "f32[72, 1, 5, 5]" = convolution_backward_52[1];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_59: "f32[8, 72, 56, 56]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_60: "f32[8, 72, 56, 56]" = torch.ops.aten.alias.default(alias_59);  alias_59 = None
    le_33: "b8[8, 72, 56, 56]" = torch.ops.aten.le.Scalar(alias_60, 0);  alias_60 = None
    scalar_tensor_41: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_61: "f32[8, 72, 56, 56]" = torch.ops.aten.where.self(le_33, scalar_tensor_41, getitem_248);  le_33 = scalar_tensor_41 = getitem_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_616: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_617: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 2);  unsqueeze_616 = None
    unsqueeze_618: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 3);  unsqueeze_617 = None
    sum_99: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_61, [0, 2, 3])
    sub_190: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_618)
    mul_719: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_61, sub_190);  sub_190 = None
    sum_100: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_719, [0, 2, 3]);  mul_719 = None
    mul_720: "f32[72]" = torch.ops.aten.mul.Tensor(sum_99, 3.985969387755102e-05)
    unsqueeze_619: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_720, 0);  mul_720 = None
    unsqueeze_620: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_619, 2);  unsqueeze_619 = None
    unsqueeze_621: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 3);  unsqueeze_620 = None
    mul_721: "f32[72]" = torch.ops.aten.mul.Tensor(sum_100, 3.985969387755102e-05)
    mul_722: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_723: "f32[72]" = torch.ops.aten.mul.Tensor(mul_721, mul_722);  mul_721 = mul_722 = None
    unsqueeze_622: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_723, 0);  mul_723 = None
    unsqueeze_623: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 2);  unsqueeze_622 = None
    unsqueeze_624: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 3);  unsqueeze_623 = None
    mul_724: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_19);  primals_19 = None
    unsqueeze_625: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_724, 0);  mul_724 = None
    unsqueeze_626: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 2);  unsqueeze_625 = None
    unsqueeze_627: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 3);  unsqueeze_626 = None
    sub_191: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_618);  convolution_9 = unsqueeze_618 = None
    mul_725: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_624);  sub_191 = unsqueeze_624 = None
    sub_192: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(where_61, mul_725);  where_61 = mul_725 = None
    sub_193: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(sub_192, unsqueeze_621);  sub_192 = unsqueeze_621 = None
    mul_726: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_627);  sub_193 = unsqueeze_627 = None
    mul_727: "f32[72]" = torch.ops.aten.mul.Tensor(sum_100, squeeze_28);  sum_100 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_726, add_47, primals_104, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_726 = add_47 = primals_104 = None
    getitem_251: "f32[8, 24, 56, 56]" = convolution_backward_53[0]
    getitem_252: "f32[72, 24, 1, 1]" = convolution_backward_53[1];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_628: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_629: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, 2);  unsqueeze_628 = None
    unsqueeze_630: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 3);  unsqueeze_629 = None
    sum_101: "f32[24]" = torch.ops.aten.sum.dim_IntList(getitem_251, [0, 2, 3])
    sub_194: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_630)
    mul_728: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_251, sub_194);  sub_194 = None
    sum_102: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_728, [0, 2, 3]);  mul_728 = None
    mul_729: "f32[24]" = torch.ops.aten.mul.Tensor(sum_101, 3.985969387755102e-05)
    unsqueeze_631: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_729, 0);  mul_729 = None
    unsqueeze_632: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_631, 2);  unsqueeze_631 = None
    unsqueeze_633: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 3);  unsqueeze_632 = None
    mul_730: "f32[24]" = torch.ops.aten.mul.Tensor(sum_102, 3.985969387755102e-05)
    mul_731: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_732: "f32[24]" = torch.ops.aten.mul.Tensor(mul_730, mul_731);  mul_730 = mul_731 = None
    unsqueeze_634: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_732, 0);  mul_732 = None
    unsqueeze_635: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, 2);  unsqueeze_634 = None
    unsqueeze_636: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 3);  unsqueeze_635 = None
    mul_733: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_17);  primals_17 = None
    unsqueeze_637: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_733, 0);  mul_733 = None
    unsqueeze_638: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 2);  unsqueeze_637 = None
    unsqueeze_639: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 3);  unsqueeze_638 = None
    sub_195: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_630);  convolution_8 = unsqueeze_630 = None
    mul_734: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_195, unsqueeze_636);  sub_195 = unsqueeze_636 = None
    sub_196: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(getitem_251, mul_734);  mul_734 = None
    sub_197: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_196, unsqueeze_633);  sub_196 = unsqueeze_633 = None
    mul_735: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_639);  sub_197 = unsqueeze_639 = None
    mul_736: "f32[24]" = torch.ops.aten.mul.Tensor(sum_102, squeeze_25);  sum_102 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_735, relu_4, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_735 = primals_103 = None
    getitem_254: "f32[8, 72, 56, 56]" = convolution_backward_54[0]
    getitem_255: "f32[24, 72, 1, 1]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_62: "f32[8, 72, 56, 56]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_63: "f32[8, 72, 56, 56]" = torch.ops.aten.alias.default(alias_62);  alias_62 = None
    le_34: "b8[8, 72, 56, 56]" = torch.ops.aten.le.Scalar(alias_63, 0);  alias_63 = None
    scalar_tensor_42: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_62: "f32[8, 72, 56, 56]" = torch.ops.aten.where.self(le_34, scalar_tensor_42, getitem_254);  le_34 = scalar_tensor_42 = getitem_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_640: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_641: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, 2);  unsqueeze_640 = None
    unsqueeze_642: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 3);  unsqueeze_641 = None
    sum_103: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_62, [0, 2, 3])
    sub_198: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_642)
    mul_737: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_62, sub_198);  sub_198 = None
    sum_104: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_737, [0, 2, 3]);  mul_737 = None
    mul_738: "f32[72]" = torch.ops.aten.mul.Tensor(sum_103, 3.985969387755102e-05)
    unsqueeze_643: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_738, 0);  mul_738 = None
    unsqueeze_644: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_643, 2);  unsqueeze_643 = None
    unsqueeze_645: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 3);  unsqueeze_644 = None
    mul_739: "f32[72]" = torch.ops.aten.mul.Tensor(sum_104, 3.985969387755102e-05)
    mul_740: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_741: "f32[72]" = torch.ops.aten.mul.Tensor(mul_739, mul_740);  mul_739 = mul_740 = None
    unsqueeze_646: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_741, 0);  mul_741 = None
    unsqueeze_647: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, 2);  unsqueeze_646 = None
    unsqueeze_648: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 3);  unsqueeze_647 = None
    mul_742: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_15);  primals_15 = None
    unsqueeze_649: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_742, 0);  mul_742 = None
    unsqueeze_650: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 2);  unsqueeze_649 = None
    unsqueeze_651: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 3);  unsqueeze_650 = None
    sub_199: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_642);  convolution_7 = unsqueeze_642 = None
    mul_743: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_648);  sub_199 = unsqueeze_648 = None
    sub_200: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(where_62, mul_743);  where_62 = mul_743 = None
    sub_201: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(sub_200, unsqueeze_645);  sub_200 = unsqueeze_645 = None
    mul_744: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_651);  sub_201 = unsqueeze_651 = None
    mul_745: "f32[72]" = torch.ops.aten.mul.Tensor(sum_104, squeeze_22);  sum_104 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_744, relu_3, primals_102, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 72, [True, True, False]);  mul_744 = primals_102 = None
    getitem_257: "f32[8, 72, 56, 56]" = convolution_backward_55[0]
    getitem_258: "f32[72, 1, 3, 3]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_65: "f32[8, 72, 56, 56]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_66: "f32[8, 72, 56, 56]" = torch.ops.aten.alias.default(alias_65);  alias_65 = None
    le_35: "b8[8, 72, 56, 56]" = torch.ops.aten.le.Scalar(alias_66, 0);  alias_66 = None
    scalar_tensor_43: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_63: "f32[8, 72, 56, 56]" = torch.ops.aten.where.self(le_35, scalar_tensor_43, getitem_257);  le_35 = scalar_tensor_43 = getitem_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_652: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_653: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, 2);  unsqueeze_652 = None
    unsqueeze_654: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 3);  unsqueeze_653 = None
    sum_105: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_63, [0, 2, 3])
    sub_202: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_654)
    mul_746: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_63, sub_202);  sub_202 = None
    sum_106: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_746, [0, 2, 3]);  mul_746 = None
    mul_747: "f32[72]" = torch.ops.aten.mul.Tensor(sum_105, 3.985969387755102e-05)
    unsqueeze_655: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_747, 0);  mul_747 = None
    unsqueeze_656: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_655, 2);  unsqueeze_655 = None
    unsqueeze_657: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 3);  unsqueeze_656 = None
    mul_748: "f32[72]" = torch.ops.aten.mul.Tensor(sum_106, 3.985969387755102e-05)
    mul_749: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_750: "f32[72]" = torch.ops.aten.mul.Tensor(mul_748, mul_749);  mul_748 = mul_749 = None
    unsqueeze_658: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_750, 0);  mul_750 = None
    unsqueeze_659: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, 2);  unsqueeze_658 = None
    unsqueeze_660: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 3);  unsqueeze_659 = None
    mul_751: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_13);  primals_13 = None
    unsqueeze_661: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_751, 0);  mul_751 = None
    unsqueeze_662: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 2);  unsqueeze_661 = None
    unsqueeze_663: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 3);  unsqueeze_662 = None
    sub_203: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_654);  convolution_6 = unsqueeze_654 = None
    mul_752: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_660);  sub_203 = unsqueeze_660 = None
    sub_204: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(where_63, mul_752);  where_63 = mul_752 = None
    sub_205: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(sub_204, unsqueeze_657);  sub_204 = unsqueeze_657 = None
    mul_753: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_663);  sub_205 = unsqueeze_663 = None
    mul_754: "f32[72]" = torch.ops.aten.mul.Tensor(sum_106, squeeze_19);  sum_106 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_753, add_31, primals_101, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_753 = add_31 = primals_101 = None
    getitem_260: "f32[8, 24, 56, 56]" = convolution_backward_56[0]
    getitem_261: "f32[72, 24, 1, 1]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_305: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(getitem_251, getitem_260);  getitem_251 = getitem_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_664: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_665: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, 2);  unsqueeze_664 = None
    unsqueeze_666: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 3);  unsqueeze_665 = None
    sum_107: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_305, [0, 2, 3])
    sub_206: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_666)
    mul_755: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_305, sub_206);  sub_206 = None
    sum_108: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_755, [0, 2, 3]);  mul_755 = None
    mul_756: "f32[24]" = torch.ops.aten.mul.Tensor(sum_107, 3.985969387755102e-05)
    unsqueeze_667: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_756, 0);  mul_756 = None
    unsqueeze_668: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_667, 2);  unsqueeze_667 = None
    unsqueeze_669: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 3);  unsqueeze_668 = None
    mul_757: "f32[24]" = torch.ops.aten.mul.Tensor(sum_108, 3.985969387755102e-05)
    mul_758: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_759: "f32[24]" = torch.ops.aten.mul.Tensor(mul_757, mul_758);  mul_757 = mul_758 = None
    unsqueeze_670: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_759, 0);  mul_759 = None
    unsqueeze_671: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, 2);  unsqueeze_670 = None
    unsqueeze_672: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 3);  unsqueeze_671 = None
    mul_760: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_11);  primals_11 = None
    unsqueeze_673: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_760, 0);  mul_760 = None
    unsqueeze_674: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 2);  unsqueeze_673 = None
    unsqueeze_675: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 3);  unsqueeze_674 = None
    sub_207: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_666);  convolution_5 = unsqueeze_666 = None
    mul_761: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_672);  sub_207 = unsqueeze_672 = None
    sub_208: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(add_305, mul_761);  add_305 = mul_761 = None
    sub_209: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_208, unsqueeze_669);  sub_208 = unsqueeze_669 = None
    mul_762: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_675);  sub_209 = unsqueeze_675 = None
    mul_763: "f32[24]" = torch.ops.aten.mul.Tensor(sum_108, squeeze_16);  sum_108 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_762, relu_2, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_762 = primals_100 = None
    getitem_263: "f32[8, 64, 56, 56]" = convolution_backward_57[0]
    getitem_264: "f32[24, 64, 1, 1]" = convolution_backward_57[1];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_68: "f32[8, 64, 56, 56]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_69: "f32[8, 64, 56, 56]" = torch.ops.aten.alias.default(alias_68);  alias_68 = None
    le_36: "b8[8, 64, 56, 56]" = torch.ops.aten.le.Scalar(alias_69, 0);  alias_69 = None
    scalar_tensor_44: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_64: "f32[8, 64, 56, 56]" = torch.ops.aten.where.self(le_36, scalar_tensor_44, getitem_263);  le_36 = scalar_tensor_44 = getitem_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_676: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_677: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, 2);  unsqueeze_676 = None
    unsqueeze_678: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 3);  unsqueeze_677 = None
    sum_109: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_64, [0, 2, 3])
    sub_210: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_678)
    mul_764: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_64, sub_210);  sub_210 = None
    sum_110: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_764, [0, 2, 3]);  mul_764 = None
    mul_765: "f32[64]" = torch.ops.aten.mul.Tensor(sum_109, 3.985969387755102e-05)
    unsqueeze_679: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_765, 0);  mul_765 = None
    unsqueeze_680: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_679, 2);  unsqueeze_679 = None
    unsqueeze_681: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 3);  unsqueeze_680 = None
    mul_766: "f32[64]" = torch.ops.aten.mul.Tensor(sum_110, 3.985969387755102e-05)
    mul_767: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_768: "f32[64]" = torch.ops.aten.mul.Tensor(mul_766, mul_767);  mul_766 = mul_767 = None
    unsqueeze_682: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_768, 0);  mul_768 = None
    unsqueeze_683: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, 2);  unsqueeze_682 = None
    unsqueeze_684: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 3);  unsqueeze_683 = None
    mul_769: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_9);  primals_9 = None
    unsqueeze_685: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_769, 0);  mul_769 = None
    unsqueeze_686: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 2);  unsqueeze_685 = None
    unsqueeze_687: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 3);  unsqueeze_686 = None
    sub_211: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_678);  convolution_4 = unsqueeze_678 = None
    mul_770: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_684);  sub_211 = unsqueeze_684 = None
    sub_212: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(where_64, mul_770);  where_64 = mul_770 = None
    sub_213: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(sub_212, unsqueeze_681);  sub_212 = unsqueeze_681 = None
    mul_771: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_687);  sub_213 = unsqueeze_687 = None
    mul_772: "f32[64]" = torch.ops.aten.mul.Tensor(sum_110, squeeze_13);  sum_110 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_771, relu_1, primals_99, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False]);  mul_771 = primals_99 = None
    getitem_266: "f32[8, 64, 112, 112]" = convolution_backward_58[0]
    getitem_267: "f32[64, 1, 3, 3]" = convolution_backward_58[1];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_71: "f32[8, 64, 112, 112]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_72: "f32[8, 64, 112, 112]" = torch.ops.aten.alias.default(alias_71);  alias_71 = None
    le_37: "b8[8, 64, 112, 112]" = torch.ops.aten.le.Scalar(alias_72, 0);  alias_72 = None
    scalar_tensor_45: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_65: "f32[8, 64, 112, 112]" = torch.ops.aten.where.self(le_37, scalar_tensor_45, getitem_266);  le_37 = scalar_tensor_45 = getitem_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_688: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_689: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, 2);  unsqueeze_688 = None
    unsqueeze_690: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 3);  unsqueeze_689 = None
    sum_111: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_65, [0, 2, 3])
    sub_214: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_690)
    mul_773: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_65, sub_214);  sub_214 = None
    sum_112: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_773, [0, 2, 3]);  mul_773 = None
    mul_774: "f32[64]" = torch.ops.aten.mul.Tensor(sum_111, 9.964923469387754e-06)
    unsqueeze_691: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_774, 0);  mul_774 = None
    unsqueeze_692: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_691, 2);  unsqueeze_691 = None
    unsqueeze_693: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 3);  unsqueeze_692 = None
    mul_775: "f32[64]" = torch.ops.aten.mul.Tensor(sum_112, 9.964923469387754e-06)
    mul_776: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_777: "f32[64]" = torch.ops.aten.mul.Tensor(mul_775, mul_776);  mul_775 = mul_776 = None
    unsqueeze_694: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_777, 0);  mul_777 = None
    unsqueeze_695: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 2);  unsqueeze_694 = None
    unsqueeze_696: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 3);  unsqueeze_695 = None
    mul_778: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_7);  primals_7 = None
    unsqueeze_697: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_778, 0);  mul_778 = None
    unsqueeze_698: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 2);  unsqueeze_697 = None
    unsqueeze_699: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 3);  unsqueeze_698 = None
    sub_215: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_690);  convolution_3 = unsqueeze_690 = None
    mul_779: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_215, unsqueeze_696);  sub_215 = unsqueeze_696 = None
    sub_216: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(where_65, mul_779);  where_65 = mul_779 = None
    sub_217: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(sub_216, unsqueeze_693);  sub_216 = unsqueeze_693 = None
    mul_780: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_699);  sub_217 = unsqueeze_699 = None
    mul_781: "f32[64]" = torch.ops.aten.mul.Tensor(sum_112, squeeze_10);  sum_112 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(mul_780, add_16, primals_98, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_780 = add_16 = primals_98 = None
    getitem_269: "f32[8, 16, 112, 112]" = convolution_backward_59[0]
    getitem_270: "f32[64, 16, 1, 1]" = convolution_backward_59[1];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_700: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_701: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, 2);  unsqueeze_700 = None
    unsqueeze_702: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 3);  unsqueeze_701 = None
    sum_113: "f32[16]" = torch.ops.aten.sum.dim_IntList(getitem_269, [0, 2, 3])
    sub_218: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_702)
    mul_782: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_269, sub_218);  sub_218 = None
    sum_114: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_782, [0, 2, 3]);  mul_782 = None
    mul_783: "f32[16]" = torch.ops.aten.mul.Tensor(sum_113, 9.964923469387754e-06)
    unsqueeze_703: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_783, 0);  mul_783 = None
    unsqueeze_704: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_703, 2);  unsqueeze_703 = None
    unsqueeze_705: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 3);  unsqueeze_704 = None
    mul_784: "f32[16]" = torch.ops.aten.mul.Tensor(sum_114, 9.964923469387754e-06)
    mul_785: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_786: "f32[16]" = torch.ops.aten.mul.Tensor(mul_784, mul_785);  mul_784 = mul_785 = None
    unsqueeze_706: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_786, 0);  mul_786 = None
    unsqueeze_707: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 2);  unsqueeze_706 = None
    unsqueeze_708: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 3);  unsqueeze_707 = None
    mul_787: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_5);  primals_5 = None
    unsqueeze_709: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_787, 0);  mul_787 = None
    unsqueeze_710: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 2);  unsqueeze_709 = None
    unsqueeze_711: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 3);  unsqueeze_710 = None
    sub_219: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_702);  convolution_2 = unsqueeze_702 = None
    mul_788: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_708);  sub_219 = unsqueeze_708 = None
    sub_220: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(getitem_269, mul_788);  mul_788 = None
    sub_221: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_220, unsqueeze_705);  sub_220 = unsqueeze_705 = None
    mul_789: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_221, unsqueeze_711);  sub_221 = unsqueeze_711 = None
    mul_790: "f32[16]" = torch.ops.aten.mul.Tensor(sum_114, squeeze_7);  sum_114 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_789, relu, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_789 = primals_97 = None
    getitem_272: "f32[8, 16, 112, 112]" = convolution_backward_60[0]
    getitem_273: "f32[16, 16, 1, 1]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_74: "f32[8, 16, 112, 112]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_75: "f32[8, 16, 112, 112]" = torch.ops.aten.alias.default(alias_74);  alias_74 = None
    le_38: "b8[8, 16, 112, 112]" = torch.ops.aten.le.Scalar(alias_75, 0);  alias_75 = None
    scalar_tensor_46: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_66: "f32[8, 16, 112, 112]" = torch.ops.aten.where.self(le_38, scalar_tensor_46, getitem_272);  le_38 = scalar_tensor_46 = getitem_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_712: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_713: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, 2);  unsqueeze_712 = None
    unsqueeze_714: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 3);  unsqueeze_713 = None
    sum_115: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_66, [0, 2, 3])
    sub_222: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_714)
    mul_791: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(where_66, sub_222);  sub_222 = None
    sum_116: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_791, [0, 2, 3]);  mul_791 = None
    mul_792: "f32[16]" = torch.ops.aten.mul.Tensor(sum_115, 9.964923469387754e-06)
    unsqueeze_715: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_792, 0);  mul_792 = None
    unsqueeze_716: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_715, 2);  unsqueeze_715 = None
    unsqueeze_717: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 3);  unsqueeze_716 = None
    mul_793: "f32[16]" = torch.ops.aten.mul.Tensor(sum_116, 9.964923469387754e-06)
    mul_794: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_795: "f32[16]" = torch.ops.aten.mul.Tensor(mul_793, mul_794);  mul_793 = mul_794 = None
    unsqueeze_718: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_795, 0);  mul_795 = None
    unsqueeze_719: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 2);  unsqueeze_718 = None
    unsqueeze_720: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 3);  unsqueeze_719 = None
    mul_796: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_3);  primals_3 = None
    unsqueeze_721: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_796, 0);  mul_796 = None
    unsqueeze_722: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 2);  unsqueeze_721 = None
    unsqueeze_723: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 3);  unsqueeze_722 = None
    sub_223: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_714);  convolution_1 = unsqueeze_714 = None
    mul_797: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_223, unsqueeze_720);  sub_223 = unsqueeze_720 = None
    sub_224: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(where_66, mul_797);  where_66 = mul_797 = None
    sub_225: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_224, unsqueeze_717);  sub_224 = unsqueeze_717 = None
    mul_798: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_723);  sub_225 = unsqueeze_723 = None
    mul_799: "f32[16]" = torch.ops.aten.mul.Tensor(sum_116, squeeze_4);  sum_116 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_798, div, primals_96, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False]);  mul_798 = div = primals_96 = None
    getitem_275: "f32[8, 16, 112, 112]" = convolution_backward_61[0]
    getitem_276: "f32[16, 1, 3, 3]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    add_306: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(getitem_269, getitem_275);  getitem_269 = getitem_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_28: "b8[8, 16, 112, 112]" = torch.ops.aten.lt.Scalar(clone, -3)
    le_39: "b8[8, 16, 112, 112]" = torch.ops.aten.le.Scalar(clone, 3)
    div_58: "f32[8, 16, 112, 112]" = torch.ops.aten.div.Tensor(clone, 3);  clone = None
    add_307: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(div_58, 0.5);  div_58 = None
    mul_800: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(add_306, add_307);  add_307 = None
    where_67: "f32[8, 16, 112, 112]" = torch.ops.aten.where.self(le_39, mul_800, add_306);  le_39 = mul_800 = add_306 = None
    scalar_tensor_47: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_68: "f32[8, 16, 112, 112]" = torch.ops.aten.where.self(lt_28, scalar_tensor_47, where_67);  lt_28 = scalar_tensor_47 = where_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_724: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_725: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, 2);  unsqueeze_724 = None
    unsqueeze_726: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 3);  unsqueeze_725 = None
    sum_117: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_68, [0, 2, 3])
    sub_226: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_726)
    mul_801: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(where_68, sub_226);  sub_226 = None
    sum_118: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_801, [0, 2, 3]);  mul_801 = None
    mul_802: "f32[16]" = torch.ops.aten.mul.Tensor(sum_117, 9.964923469387754e-06)
    unsqueeze_727: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_802, 0);  mul_802 = None
    unsqueeze_728: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_727, 2);  unsqueeze_727 = None
    unsqueeze_729: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 3);  unsqueeze_728 = None
    mul_803: "f32[16]" = torch.ops.aten.mul.Tensor(sum_118, 9.964923469387754e-06)
    mul_804: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_805: "f32[16]" = torch.ops.aten.mul.Tensor(mul_803, mul_804);  mul_803 = mul_804 = None
    unsqueeze_730: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_805, 0);  mul_805 = None
    unsqueeze_731: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, 2);  unsqueeze_730 = None
    unsqueeze_732: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 3);  unsqueeze_731 = None
    mul_806: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_1);  primals_1 = None
    unsqueeze_733: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_806, 0);  mul_806 = None
    unsqueeze_734: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 2);  unsqueeze_733 = None
    unsqueeze_735: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 3);  unsqueeze_734 = None
    sub_227: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_726);  convolution = unsqueeze_726 = None
    mul_807: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_732);  sub_227 = unsqueeze_732 = None
    sub_228: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(where_68, mul_807);  where_68 = mul_807 = None
    sub_229: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_228, unsqueeze_729);  sub_228 = unsqueeze_729 = None
    mul_808: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_735);  sub_229 = unsqueeze_735 = None
    mul_809: "f32[16]" = torch.ops.aten.mul.Tensor(sum_118, squeeze_1);  sum_118 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:135, code: x = self.conv_stem(x)
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_808, primals_313, primals_95, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_808 = primals_313 = primals_95 = None
    getitem_279: "f32[16, 3, 3, 3]" = convolution_backward_62[1];  convolution_backward_62 = None
    
    # No stacktrace found for following nodes
    copy_: "i64[]" = torch.ops.aten.copy_.default(primals_175, add);  primals_175 = add = None
    copy__1: "f32[16]" = torch.ops.aten.copy_.default(primals_176, add_2);  primals_176 = add_2 = None
    copy__2: "f32[16]" = torch.ops.aten.copy_.default(primals_177, add_3);  primals_177 = add_3 = None
    copy__3: "i64[]" = torch.ops.aten.copy_.default(primals_178, add_6);  primals_178 = add_6 = None
    copy__4: "f32[16]" = torch.ops.aten.copy_.default(primals_179, add_8);  primals_179 = add_8 = None
    copy__5: "f32[16]" = torch.ops.aten.copy_.default(primals_180, add_9);  primals_180 = add_9 = None
    copy__6: "i64[]" = torch.ops.aten.copy_.default(primals_181, add_11);  primals_181 = add_11 = None
    copy__7: "f32[16]" = torch.ops.aten.copy_.default(primals_182, add_13);  primals_182 = add_13 = None
    copy__8: "f32[16]" = torch.ops.aten.copy_.default(primals_183, add_14);  primals_183 = add_14 = None
    copy__9: "i64[]" = torch.ops.aten.copy_.default(primals_184, add_17);  primals_184 = add_17 = None
    copy__10: "f32[64]" = torch.ops.aten.copy_.default(primals_185, add_19);  primals_185 = add_19 = None
    copy__11: "f32[64]" = torch.ops.aten.copy_.default(primals_186, add_20);  primals_186 = add_20 = None
    copy__12: "i64[]" = torch.ops.aten.copy_.default(primals_187, add_22);  primals_187 = add_22 = None
    copy__13: "f32[64]" = torch.ops.aten.copy_.default(primals_188, add_24);  primals_188 = add_24 = None
    copy__14: "f32[64]" = torch.ops.aten.copy_.default(primals_189, add_25);  primals_189 = add_25 = None
    copy__15: "i64[]" = torch.ops.aten.copy_.default(primals_190, add_27);  primals_190 = add_27 = None
    copy__16: "f32[24]" = torch.ops.aten.copy_.default(primals_191, add_29);  primals_191 = add_29 = None
    copy__17: "f32[24]" = torch.ops.aten.copy_.default(primals_192, add_30);  primals_192 = add_30 = None
    copy__18: "i64[]" = torch.ops.aten.copy_.default(primals_193, add_32);  primals_193 = add_32 = None
    copy__19: "f32[72]" = torch.ops.aten.copy_.default(primals_194, add_34);  primals_194 = add_34 = None
    copy__20: "f32[72]" = torch.ops.aten.copy_.default(primals_195, add_35);  primals_195 = add_35 = None
    copy__21: "i64[]" = torch.ops.aten.copy_.default(primals_196, add_37);  primals_196 = add_37 = None
    copy__22: "f32[72]" = torch.ops.aten.copy_.default(primals_197, add_39);  primals_197 = add_39 = None
    copy__23: "f32[72]" = torch.ops.aten.copy_.default(primals_198, add_40);  primals_198 = add_40 = None
    copy__24: "i64[]" = torch.ops.aten.copy_.default(primals_199, add_42);  primals_199 = add_42 = None
    copy__25: "f32[24]" = torch.ops.aten.copy_.default(primals_200, add_44);  primals_200 = add_44 = None
    copy__26: "f32[24]" = torch.ops.aten.copy_.default(primals_201, add_45);  primals_201 = add_45 = None
    copy__27: "i64[]" = torch.ops.aten.copy_.default(primals_202, add_48);  primals_202 = add_48 = None
    copy__28: "f32[72]" = torch.ops.aten.copy_.default(primals_203, add_50);  primals_203 = add_50 = None
    copy__29: "f32[72]" = torch.ops.aten.copy_.default(primals_204, add_51);  primals_204 = add_51 = None
    copy__30: "i64[]" = torch.ops.aten.copy_.default(primals_205, add_53);  primals_205 = add_53 = None
    copy__31: "f32[72]" = torch.ops.aten.copy_.default(primals_206, add_55);  primals_206 = add_55 = None
    copy__32: "f32[72]" = torch.ops.aten.copy_.default(primals_207, add_56);  primals_207 = add_56 = None
    copy__33: "i64[]" = torch.ops.aten.copy_.default(primals_208, add_59);  primals_208 = add_59 = None
    copy__34: "f32[40]" = torch.ops.aten.copy_.default(primals_209, add_61);  primals_209 = add_61 = None
    copy__35: "f32[40]" = torch.ops.aten.copy_.default(primals_210, add_62);  primals_210 = add_62 = None
    copy__36: "i64[]" = torch.ops.aten.copy_.default(primals_211, add_64);  primals_211 = add_64 = None
    copy__37: "f32[120]" = torch.ops.aten.copy_.default(primals_212, add_66);  primals_212 = add_66 = None
    copy__38: "f32[120]" = torch.ops.aten.copy_.default(primals_213, add_67);  primals_213 = add_67 = None
    copy__39: "i64[]" = torch.ops.aten.copy_.default(primals_214, add_69);  primals_214 = add_69 = None
    copy__40: "f32[120]" = torch.ops.aten.copy_.default(primals_215, add_71);  primals_215 = add_71 = None
    copy__41: "f32[120]" = torch.ops.aten.copy_.default(primals_216, add_72);  primals_216 = add_72 = None
    copy__42: "i64[]" = torch.ops.aten.copy_.default(primals_217, add_75);  primals_217 = add_75 = None
    copy__43: "f32[40]" = torch.ops.aten.copy_.default(primals_218, add_77);  primals_218 = add_77 = None
    copy__44: "f32[40]" = torch.ops.aten.copy_.default(primals_219, add_78);  primals_219 = add_78 = None
    copy__45: "i64[]" = torch.ops.aten.copy_.default(primals_220, add_81);  primals_220 = add_81 = None
    copy__46: "f32[120]" = torch.ops.aten.copy_.default(primals_221, add_83);  primals_221 = add_83 = None
    copy__47: "f32[120]" = torch.ops.aten.copy_.default(primals_222, add_84);  primals_222 = add_84 = None
    copy__48: "i64[]" = torch.ops.aten.copy_.default(primals_223, add_86);  primals_223 = add_86 = None
    copy__49: "f32[120]" = torch.ops.aten.copy_.default(primals_224, add_88);  primals_224 = add_88 = None
    copy__50: "f32[120]" = torch.ops.aten.copy_.default(primals_225, add_89);  primals_225 = add_89 = None
    copy__51: "i64[]" = torch.ops.aten.copy_.default(primals_226, add_92);  primals_226 = add_92 = None
    copy__52: "f32[40]" = torch.ops.aten.copy_.default(primals_227, add_94);  primals_227 = add_94 = None
    copy__53: "f32[40]" = torch.ops.aten.copy_.default(primals_228, add_95);  primals_228 = add_95 = None
    copy__54: "i64[]" = torch.ops.aten.copy_.default(primals_229, add_98);  primals_229 = add_98 = None
    copy__55: "f32[240]" = torch.ops.aten.copy_.default(primals_230, add_100);  primals_230 = add_100 = None
    copy__56: "f32[240]" = torch.ops.aten.copy_.default(primals_231, add_101);  primals_231 = add_101 = None
    copy__57: "i64[]" = torch.ops.aten.copy_.default(primals_232, add_104);  primals_232 = add_104 = None
    copy__58: "f32[240]" = torch.ops.aten.copy_.default(primals_233, add_106);  primals_233 = add_106 = None
    copy__59: "f32[240]" = torch.ops.aten.copy_.default(primals_234, add_107);  primals_234 = add_107 = None
    copy__60: "i64[]" = torch.ops.aten.copy_.default(primals_235, add_110);  primals_235 = add_110 = None
    copy__61: "f32[80]" = torch.ops.aten.copy_.default(primals_236, add_112);  primals_236 = add_112 = None
    copy__62: "f32[80]" = torch.ops.aten.copy_.default(primals_237, add_113);  primals_237 = add_113 = None
    copy__63: "i64[]" = torch.ops.aten.copy_.default(primals_238, add_115);  primals_238 = add_115 = None
    copy__64: "f32[200]" = torch.ops.aten.copy_.default(primals_239, add_117);  primals_239 = add_117 = None
    copy__65: "f32[200]" = torch.ops.aten.copy_.default(primals_240, add_118);  primals_240 = add_118 = None
    copy__66: "i64[]" = torch.ops.aten.copy_.default(primals_241, add_121);  primals_241 = add_121 = None
    copy__67: "f32[200]" = torch.ops.aten.copy_.default(primals_242, add_123);  primals_242 = add_123 = None
    copy__68: "f32[200]" = torch.ops.aten.copy_.default(primals_243, add_124);  primals_243 = add_124 = None
    copy__69: "i64[]" = torch.ops.aten.copy_.default(primals_244, add_127);  primals_244 = add_127 = None
    copy__70: "f32[80]" = torch.ops.aten.copy_.default(primals_245, add_129);  primals_245 = add_129 = None
    copy__71: "f32[80]" = torch.ops.aten.copy_.default(primals_246, add_130);  primals_246 = add_130 = None
    copy__72: "i64[]" = torch.ops.aten.copy_.default(primals_247, add_133);  primals_247 = add_133 = None
    copy__73: "f32[184]" = torch.ops.aten.copy_.default(primals_248, add_135);  primals_248 = add_135 = None
    copy__74: "f32[184]" = torch.ops.aten.copy_.default(primals_249, add_136);  primals_249 = add_136 = None
    copy__75: "i64[]" = torch.ops.aten.copy_.default(primals_250, add_139);  primals_250 = add_139 = None
    copy__76: "f32[184]" = torch.ops.aten.copy_.default(primals_251, add_141);  primals_251 = add_141 = None
    copy__77: "f32[184]" = torch.ops.aten.copy_.default(primals_252, add_142);  primals_252 = add_142 = None
    copy__78: "i64[]" = torch.ops.aten.copy_.default(primals_253, add_145);  primals_253 = add_145 = None
    copy__79: "f32[80]" = torch.ops.aten.copy_.default(primals_254, add_147);  primals_254 = add_147 = None
    copy__80: "f32[80]" = torch.ops.aten.copy_.default(primals_255, add_148);  primals_255 = add_148 = None
    copy__81: "i64[]" = torch.ops.aten.copy_.default(primals_256, add_151);  primals_256 = add_151 = None
    copy__82: "f32[184]" = torch.ops.aten.copy_.default(primals_257, add_153);  primals_257 = add_153 = None
    copy__83: "f32[184]" = torch.ops.aten.copy_.default(primals_258, add_154);  primals_258 = add_154 = None
    copy__84: "i64[]" = torch.ops.aten.copy_.default(primals_259, add_157);  primals_259 = add_157 = None
    copy__85: "f32[184]" = torch.ops.aten.copy_.default(primals_260, add_159);  primals_260 = add_159 = None
    copy__86: "f32[184]" = torch.ops.aten.copy_.default(primals_261, add_160);  primals_261 = add_160 = None
    copy__87: "i64[]" = torch.ops.aten.copy_.default(primals_262, add_163);  primals_262 = add_163 = None
    copy__88: "f32[80]" = torch.ops.aten.copy_.default(primals_263, add_165);  primals_263 = add_165 = None
    copy__89: "f32[80]" = torch.ops.aten.copy_.default(primals_264, add_166);  primals_264 = add_166 = None
    copy__90: "i64[]" = torch.ops.aten.copy_.default(primals_265, add_169);  primals_265 = add_169 = None
    copy__91: "f32[480]" = torch.ops.aten.copy_.default(primals_266, add_171);  primals_266 = add_171 = None
    copy__92: "f32[480]" = torch.ops.aten.copy_.default(primals_267, add_172);  primals_267 = add_172 = None
    copy__93: "i64[]" = torch.ops.aten.copy_.default(primals_268, add_175);  primals_268 = add_175 = None
    copy__94: "f32[480]" = torch.ops.aten.copy_.default(primals_269, add_177);  primals_269 = add_177 = None
    copy__95: "f32[480]" = torch.ops.aten.copy_.default(primals_270, add_178);  primals_270 = add_178 = None
    copy__96: "i64[]" = torch.ops.aten.copy_.default(primals_271, add_182);  primals_271 = add_182 = None
    copy__97: "f32[112]" = torch.ops.aten.copy_.default(primals_272, add_184);  primals_272 = add_184 = None
    copy__98: "f32[112]" = torch.ops.aten.copy_.default(primals_273, add_185);  primals_273 = add_185 = None
    copy__99: "i64[]" = torch.ops.aten.copy_.default(primals_274, add_187);  primals_274 = add_187 = None
    copy__100: "f32[672]" = torch.ops.aten.copy_.default(primals_275, add_189);  primals_275 = add_189 = None
    copy__101: "f32[672]" = torch.ops.aten.copy_.default(primals_276, add_190);  primals_276 = add_190 = None
    copy__102: "i64[]" = torch.ops.aten.copy_.default(primals_277, add_193);  primals_277 = add_193 = None
    copy__103: "f32[672]" = torch.ops.aten.copy_.default(primals_278, add_195);  primals_278 = add_195 = None
    copy__104: "f32[672]" = torch.ops.aten.copy_.default(primals_279, add_196);  primals_279 = add_196 = None
    copy__105: "i64[]" = torch.ops.aten.copy_.default(primals_280, add_200);  primals_280 = add_200 = None
    copy__106: "f32[112]" = torch.ops.aten.copy_.default(primals_281, add_202);  primals_281 = add_202 = None
    copy__107: "f32[112]" = torch.ops.aten.copy_.default(primals_282, add_203);  primals_282 = add_203 = None
    copy__108: "i64[]" = torch.ops.aten.copy_.default(primals_283, add_206);  primals_283 = add_206 = None
    copy__109: "f32[672]" = torch.ops.aten.copy_.default(primals_284, add_208);  primals_284 = add_208 = None
    copy__110: "f32[672]" = torch.ops.aten.copy_.default(primals_285, add_209);  primals_285 = add_209 = None
    copy__111: "i64[]" = torch.ops.aten.copy_.default(primals_286, add_212);  primals_286 = add_212 = None
    copy__112: "f32[672]" = torch.ops.aten.copy_.default(primals_287, add_214);  primals_287 = add_214 = None
    copy__113: "f32[672]" = torch.ops.aten.copy_.default(primals_288, add_215);  primals_288 = add_215 = None
    copy__114: "i64[]" = torch.ops.aten.copy_.default(primals_289, add_219);  primals_289 = add_219 = None
    copy__115: "f32[160]" = torch.ops.aten.copy_.default(primals_290, add_221);  primals_290 = add_221 = None
    copy__116: "f32[160]" = torch.ops.aten.copy_.default(primals_291, add_222);  primals_291 = add_222 = None
    copy__117: "i64[]" = torch.ops.aten.copy_.default(primals_292, add_224);  primals_292 = add_224 = None
    copy__118: "f32[960]" = torch.ops.aten.copy_.default(primals_293, add_226);  primals_293 = add_226 = None
    copy__119: "f32[960]" = torch.ops.aten.copy_.default(primals_294, add_227);  primals_294 = add_227 = None
    copy__120: "i64[]" = torch.ops.aten.copy_.default(primals_295, add_230);  primals_295 = add_230 = None
    copy__121: "f32[960]" = torch.ops.aten.copy_.default(primals_296, add_232);  primals_296 = add_232 = None
    copy__122: "f32[960]" = torch.ops.aten.copy_.default(primals_297, add_233);  primals_297 = add_233 = None
    copy__123: "i64[]" = torch.ops.aten.copy_.default(primals_298, add_237);  primals_298 = add_237 = None
    copy__124: "f32[160]" = torch.ops.aten.copy_.default(primals_299, add_239);  primals_299 = add_239 = None
    copy__125: "f32[160]" = torch.ops.aten.copy_.default(primals_300, add_240);  primals_300 = add_240 = None
    copy__126: "i64[]" = torch.ops.aten.copy_.default(primals_301, add_243);  primals_301 = add_243 = None
    copy__127: "f32[960]" = torch.ops.aten.copy_.default(primals_302, add_245);  primals_302 = add_245 = None
    copy__128: "f32[960]" = torch.ops.aten.copy_.default(primals_303, add_246);  primals_303 = add_246 = None
    copy__129: "i64[]" = torch.ops.aten.copy_.default(primals_304, add_249);  primals_304 = add_249 = None
    copy__130: "f32[960]" = torch.ops.aten.copy_.default(primals_305, add_251);  primals_305 = add_251 = None
    copy__131: "f32[960]" = torch.ops.aten.copy_.default(primals_306, add_252);  primals_306 = add_252 = None
    copy__132: "i64[]" = torch.ops.aten.copy_.default(primals_307, add_256);  primals_307 = add_256 = None
    copy__133: "f32[160]" = torch.ops.aten.copy_.default(primals_308, add_258);  primals_308 = add_258 = None
    copy__134: "f32[160]" = torch.ops.aten.copy_.default(primals_309, add_259);  primals_309 = add_259 = None
    copy__135: "i64[]" = torch.ops.aten.copy_.default(primals_310, add_262);  primals_310 = add_262 = None
    copy__136: "f32[960]" = torch.ops.aten.copy_.default(primals_311, add_264);  primals_311 = add_264 = None
    copy__137: "f32[960]" = torch.ops.aten.copy_.default(primals_312, add_265);  primals_312 = add_265 = None
    return pytree.tree_unflatten([addmm, mul_809, sum_117, mul_799, sum_115, mul_790, sum_113, mul_781, sum_111, mul_772, sum_109, mul_763, sum_107, mul_754, sum_105, mul_745, sum_103, mul_736, sum_101, mul_727, sum_99, mul_718, sum_97, mul_706, sum_92, mul_697, sum_90, mul_688, sum_88, mul_676, sum_83, mul_667, sum_81, mul_658, sum_79, mul_646, sum_74, mul_637, sum_72, mul_627, sum_70, mul_617, sum_68, mul_608, sum_66, mul_598, sum_64, mul_588, sum_62, mul_579, sum_60, mul_569, sum_58, mul_559, sum_56, mul_550, sum_54, mul_540, sum_52, mul_530, sum_50, mul_521, sum_48, mul_511, sum_46, mul_498, sum_41, mul_489, sum_39, mul_479, sum_37, mul_466, sum_32, mul_457, sum_30, mul_447, sum_28, mul_434, sum_23, mul_425, sum_21, mul_415, sum_19, mul_402, sum_14, mul_393, sum_12, mul_383, sum_10, mul_370, sum_5, mul_361, sum_3, permute_4, view_2, getitem_279, getitem_276, getitem_273, getitem_270, getitem_267, getitem_264, getitem_261, getitem_258, getitem_255, getitem_252, getitem_249, getitem_246, sum_96, getitem_243, sum_95, getitem_240, getitem_237, getitem_234, getitem_231, sum_87, getitem_228, sum_86, getitem_225, getitem_222, getitem_219, getitem_216, sum_78, getitem_213, sum_77, getitem_210, getitem_207, getitem_204, getitem_201, getitem_198, getitem_195, getitem_192, getitem_189, getitem_186, getitem_183, getitem_180, getitem_177, getitem_174, getitem_171, getitem_168, getitem_165, sum_45, getitem_162, sum_44, getitem_159, getitem_156, getitem_153, getitem_150, sum_36, getitem_147, sum_35, getitem_144, getitem_141, getitem_138, getitem_135, sum_27, getitem_132, sum_26, getitem_129, getitem_126, getitem_123, getitem_120, sum_18, getitem_117, sum_17, getitem_114, getitem_111, getitem_108, getitem_105, sum_9, getitem_102, sum_8, getitem_99, getitem_96, getitem_93, sum_2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    