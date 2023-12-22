from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[16]"; primals_2: "f32[16]"; primals_3: "f32[64]"; primals_4: "f32[64]"; primals_5: "f32[64]"; primals_6: "f32[64]"; primals_7: "f32[32]"; primals_8: "f32[32]"; primals_9: "f32[128]"; primals_10: "f32[128]"; primals_11: "f32[128]"; primals_12: "f32[128]"; primals_13: "f32[64]"; primals_14: "f32[64]"; primals_15: "f32[256]"; primals_16: "f32[256]"; primals_17: "f32[256]"; primals_18: "f32[256]"; primals_19: "f32[64]"; primals_20: "f32[64]"; primals_21: "f32[256]"; primals_22: "f32[256]"; primals_23: "f32[256]"; primals_24: "f32[256]"; primals_25: "f32[64]"; primals_26: "f32[64]"; primals_27: "f32[256]"; primals_28: "f32[256]"; primals_29: "f32[256]"; primals_30: "f32[256]"; primals_31: "f32[96]"; primals_32: "f32[96]"; primals_33: "f32[96]"; primals_34: "f32[96]"; primals_35: "f32[96]"; primals_36: "f32[96]"; primals_37: "f32[96]"; primals_38: "f32[96]"; primals_39: "f32[384]"; primals_40: "f32[384]"; primals_41: "f32[384]"; primals_42: "f32[384]"; primals_43: "f32[128]"; primals_44: "f32[128]"; primals_45: "f32[128]"; primals_46: "f32[128]"; primals_47: "f32[128]"; primals_48: "f32[128]"; primals_49: "f32[128]"; primals_50: "f32[128]"; primals_51: "f32[512]"; primals_52: "f32[512]"; primals_53: "f32[512]"; primals_54: "f32[512]"; primals_55: "f32[160]"; primals_56: "f32[160]"; primals_57: "f32[160]"; primals_58: "f32[160]"; primals_59: "f32[160]"; primals_60: "f32[160]"; primals_61: "f32[160]"; primals_62: "f32[160]"; primals_63: "f32[640]"; primals_64: "f32[640]"; primals_65: "f32[16, 3, 3, 3]"; primals_66: "f32[64, 16, 1, 1]"; primals_67: "f32[64, 1, 3, 3]"; primals_68: "f32[32, 64, 1, 1]"; primals_69: "f32[128, 32, 1, 1]"; primals_70: "f32[128, 1, 3, 3]"; primals_71: "f32[64, 128, 1, 1]"; primals_72: "f32[256, 64, 1, 1]"; primals_73: "f32[256, 1, 3, 3]"; primals_74: "f32[64, 256, 1, 1]"; primals_75: "f32[256, 64, 1, 1]"; primals_76: "f32[256, 1, 3, 3]"; primals_77: "f32[64, 256, 1, 1]"; primals_78: "f32[256, 64, 1, 1]"; primals_79: "f32[256, 1, 3, 3]"; primals_80: "f32[96, 256, 1, 1]"; primals_81: "f32[96, 96, 3, 3]"; primals_82: "f32[144, 96, 1, 1]"; primals_83: "f32[144]"; primals_84: "f32[144]"; primals_85: "f32[432, 144]"; primals_86: "f32[432]"; primals_87: "f32[144, 144]"; primals_88: "f32[144]"; primals_89: "f32[144]"; primals_90: "f32[144]"; primals_91: "f32[288, 144]"; primals_92: "f32[288]"; primals_93: "f32[144, 288]"; primals_94: "f32[144]"; primals_95: "f32[144]"; primals_96: "f32[144]"; primals_97: "f32[432, 144]"; primals_98: "f32[432]"; primals_99: "f32[144, 144]"; primals_100: "f32[144]"; primals_101: "f32[144]"; primals_102: "f32[144]"; primals_103: "f32[288, 144]"; primals_104: "f32[288]"; primals_105: "f32[144, 288]"; primals_106: "f32[144]"; primals_107: "f32[144]"; primals_108: "f32[144]"; primals_109: "f32[96, 144, 1, 1]"; primals_110: "f32[96, 192, 3, 3]"; primals_111: "f32[384, 96, 1, 1]"; primals_112: "f32[384, 1, 3, 3]"; primals_113: "f32[128, 384, 1, 1]"; primals_114: "f32[128, 128, 3, 3]"; primals_115: "f32[192, 128, 1, 1]"; primals_116: "f32[192]"; primals_117: "f32[192]"; primals_118: "f32[576, 192]"; primals_119: "f32[576]"; primals_120: "f32[192, 192]"; primals_121: "f32[192]"; primals_122: "f32[192]"; primals_123: "f32[192]"; primals_124: "f32[384, 192]"; primals_125: "f32[384]"; primals_126: "f32[192, 384]"; primals_127: "f32[192]"; primals_128: "f32[192]"; primals_129: "f32[192]"; primals_130: "f32[576, 192]"; primals_131: "f32[576]"; primals_132: "f32[192, 192]"; primals_133: "f32[192]"; primals_134: "f32[192]"; primals_135: "f32[192]"; primals_136: "f32[384, 192]"; primals_137: "f32[384]"; primals_138: "f32[192, 384]"; primals_139: "f32[192]"; primals_140: "f32[192]"; primals_141: "f32[192]"; primals_142: "f32[576, 192]"; primals_143: "f32[576]"; primals_144: "f32[192, 192]"; primals_145: "f32[192]"; primals_146: "f32[192]"; primals_147: "f32[192]"; primals_148: "f32[384, 192]"; primals_149: "f32[384]"; primals_150: "f32[192, 384]"; primals_151: "f32[192]"; primals_152: "f32[192]"; primals_153: "f32[192]"; primals_154: "f32[576, 192]"; primals_155: "f32[576]"; primals_156: "f32[192, 192]"; primals_157: "f32[192]"; primals_158: "f32[192]"; primals_159: "f32[192]"; primals_160: "f32[384, 192]"; primals_161: "f32[384]"; primals_162: "f32[192, 384]"; primals_163: "f32[192]"; primals_164: "f32[192]"; primals_165: "f32[192]"; primals_166: "f32[128, 192, 1, 1]"; primals_167: "f32[128, 256, 3, 3]"; primals_168: "f32[512, 128, 1, 1]"; primals_169: "f32[512, 1, 3, 3]"; primals_170: "f32[160, 512, 1, 1]"; primals_171: "f32[160, 160, 3, 3]"; primals_172: "f32[240, 160, 1, 1]"; primals_173: "f32[240]"; primals_174: "f32[240]"; primals_175: "f32[720, 240]"; primals_176: "f32[720]"; primals_177: "f32[240, 240]"; primals_178: "f32[240]"; primals_179: "f32[240]"; primals_180: "f32[240]"; primals_181: "f32[480, 240]"; primals_182: "f32[480]"; primals_183: "f32[240, 480]"; primals_184: "f32[240]"; primals_185: "f32[240]"; primals_186: "f32[240]"; primals_187: "f32[720, 240]"; primals_188: "f32[720]"; primals_189: "f32[240, 240]"; primals_190: "f32[240]"; primals_191: "f32[240]"; primals_192: "f32[240]"; primals_193: "f32[480, 240]"; primals_194: "f32[480]"; primals_195: "f32[240, 480]"; primals_196: "f32[240]"; primals_197: "f32[240]"; primals_198: "f32[240]"; primals_199: "f32[720, 240]"; primals_200: "f32[720]"; primals_201: "f32[240, 240]"; primals_202: "f32[240]"; primals_203: "f32[240]"; primals_204: "f32[240]"; primals_205: "f32[480, 240]"; primals_206: "f32[480]"; primals_207: "f32[240, 480]"; primals_208: "f32[240]"; primals_209: "f32[240]"; primals_210: "f32[240]"; primals_211: "f32[160, 240, 1, 1]"; primals_212: "f32[160, 320, 3, 3]"; primals_213: "f32[640, 160, 1, 1]"; primals_214: "f32[1000, 640]"; primals_215: "f32[1000]"; primals_216: "i64[]"; primals_217: "f32[16]"; primals_218: "f32[16]"; primals_219: "i64[]"; primals_220: "f32[64]"; primals_221: "f32[64]"; primals_222: "i64[]"; primals_223: "f32[64]"; primals_224: "f32[64]"; primals_225: "i64[]"; primals_226: "f32[32]"; primals_227: "f32[32]"; primals_228: "i64[]"; primals_229: "f32[128]"; primals_230: "f32[128]"; primals_231: "i64[]"; primals_232: "f32[128]"; primals_233: "f32[128]"; primals_234: "i64[]"; primals_235: "f32[64]"; primals_236: "f32[64]"; primals_237: "i64[]"; primals_238: "f32[256]"; primals_239: "f32[256]"; primals_240: "i64[]"; primals_241: "f32[256]"; primals_242: "f32[256]"; primals_243: "i64[]"; primals_244: "f32[64]"; primals_245: "f32[64]"; primals_246: "i64[]"; primals_247: "f32[256]"; primals_248: "f32[256]"; primals_249: "i64[]"; primals_250: "f32[256]"; primals_251: "f32[256]"; primals_252: "i64[]"; primals_253: "f32[64]"; primals_254: "f32[64]"; primals_255: "i64[]"; primals_256: "f32[256]"; primals_257: "f32[256]"; primals_258: "i64[]"; primals_259: "f32[256]"; primals_260: "f32[256]"; primals_261: "i64[]"; primals_262: "f32[96]"; primals_263: "f32[96]"; primals_264: "i64[]"; primals_265: "f32[96]"; primals_266: "f32[96]"; primals_267: "i64[]"; primals_268: "f32[96]"; primals_269: "f32[96]"; primals_270: "i64[]"; primals_271: "f32[96]"; primals_272: "f32[96]"; primals_273: "i64[]"; primals_274: "f32[384]"; primals_275: "f32[384]"; primals_276: "i64[]"; primals_277: "f32[384]"; primals_278: "f32[384]"; primals_279: "i64[]"; primals_280: "f32[128]"; primals_281: "f32[128]"; primals_282: "i64[]"; primals_283: "f32[128]"; primals_284: "f32[128]"; primals_285: "i64[]"; primals_286: "f32[128]"; primals_287: "f32[128]"; primals_288: "i64[]"; primals_289: "f32[128]"; primals_290: "f32[128]"; primals_291: "i64[]"; primals_292: "f32[512]"; primals_293: "f32[512]"; primals_294: "i64[]"; primals_295: "f32[512]"; primals_296: "f32[512]"; primals_297: "i64[]"; primals_298: "f32[160]"; primals_299: "f32[160]"; primals_300: "i64[]"; primals_301: "f32[160]"; primals_302: "f32[160]"; primals_303: "i64[]"; primals_304: "f32[160]"; primals_305: "f32[160]"; primals_306: "i64[]"; primals_307: "f32[160]"; primals_308: "f32[160]"; primals_309: "i64[]"; primals_310: "f32[640]"; primals_311: "f32[640]"; primals_312: "f32[8, 3, 256, 256]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution: "f32[8, 16, 128, 128]" = torch.ops.aten.convolution.default(primals_312, primals_65, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_216, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
    getitem: "f32[1, 16, 1, 1]" = var_mean[0]
    getitem_1: "f32[1, 16, 1, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
    rsqrt: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 16, 128, 128]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
    mul: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    squeeze: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
    squeeze_1: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
    mul_1: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
    mul_2: "f32[16]" = torch.ops.aten.mul.Tensor(primals_217, 0.9)
    add_2: "f32[16]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[16]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000076294527394);  squeeze_2 = None
    mul_4: "f32[16]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[16]" = torch.ops.aten.mul.Tensor(primals_218, 0.9)
    add_3: "f32[16]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_1, -1)
    unsqueeze_1: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1);  primals_2 = None
    unsqueeze_3: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 16, 128, 128]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone: "f32[8, 16, 128, 128]" = torch.ops.aten.clone.default(add_4)
    sigmoid: "f32[8, 16, 128, 128]" = torch.ops.aten.sigmoid.default(add_4)
    mul_7: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(add_4, sigmoid);  add_4 = sigmoid = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_1: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(mul_7, primals_66, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_219, 1)
    
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
    mul_10: "f32[64]" = torch.ops.aten.mul.Tensor(primals_220, 0.9)
    add_7: "f32[64]" = torch.ops.aten.add.Tensor(mul_9, mul_10);  mul_9 = mul_10 = None
    squeeze_5: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_11: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000076294527394);  squeeze_5 = None
    mul_12: "f32[64]" = torch.ops.aten.mul.Tensor(mul_11, 0.1);  mul_11 = None
    mul_13: "f32[64]" = torch.ops.aten.mul.Tensor(primals_221, 0.9)
    add_8: "f32[64]" = torch.ops.aten.add.Tensor(mul_12, mul_13);  mul_12 = mul_13 = None
    unsqueeze_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1)
    unsqueeze_5: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_14: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_8, unsqueeze_5);  mul_8 = unsqueeze_5 = None
    unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_4, -1);  primals_4 = None
    unsqueeze_7: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_9: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_7);  mul_14 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_1: "f32[8, 64, 128, 128]" = torch.ops.aten.clone.default(add_9)
    sigmoid_1: "f32[8, 64, 128, 128]" = torch.ops.aten.sigmoid.default(add_9)
    mul_15: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(add_9, sigmoid_1);  add_9 = sigmoid_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_2: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(mul_15, primals_67, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_222, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 64, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 64, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
    rsqrt_2: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_2: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
    mul_16: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_17: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_18: "f32[64]" = torch.ops.aten.mul.Tensor(primals_223, 0.9)
    add_12: "f32[64]" = torch.ops.aten.add.Tensor(mul_17, mul_18);  mul_17 = mul_18 = None
    squeeze_8: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_19: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0000076294527394);  squeeze_8 = None
    mul_20: "f32[64]" = torch.ops.aten.mul.Tensor(mul_19, 0.1);  mul_19 = None
    mul_21: "f32[64]" = torch.ops.aten.mul.Tensor(primals_224, 0.9)
    add_13: "f32[64]" = torch.ops.aten.add.Tensor(mul_20, mul_21);  mul_20 = mul_21 = None
    unsqueeze_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_22: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_9);  mul_16 = unsqueeze_9 = None
    unsqueeze_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_14: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_22, unsqueeze_11);  mul_22 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_2: "f32[8, 64, 128, 128]" = torch.ops.aten.clone.default(add_14)
    sigmoid_2: "f32[8, 64, 128, 128]" = torch.ops.aten.sigmoid.default(add_14)
    mul_23: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(add_14, sigmoid_2);  add_14 = sigmoid_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_3: "f32[8, 32, 128, 128]" = torch.ops.aten.convolution.default(mul_23, primals_68, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_15: "i64[]" = torch.ops.aten.add.Tensor(primals_225, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
    getitem_6: "f32[1, 32, 1, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 32, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_16: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
    rsqrt_3: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_3: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_7)
    mul_24: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
    squeeze_10: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_25: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_26: "f32[32]" = torch.ops.aten.mul.Tensor(primals_226, 0.9)
    add_17: "f32[32]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
    squeeze_11: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
    mul_27: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0000076294527394);  squeeze_11 = None
    mul_28: "f32[32]" = torch.ops.aten.mul.Tensor(mul_27, 0.1);  mul_27 = None
    mul_29: "f32[32]" = torch.ops.aten.mul.Tensor(primals_227, 0.9)
    add_18: "f32[32]" = torch.ops.aten.add.Tensor(mul_28, mul_29);  mul_28 = mul_29 = None
    unsqueeze_12: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1)
    unsqueeze_13: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_30: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(mul_24, unsqueeze_13);  mul_24 = unsqueeze_13 = None
    unsqueeze_14: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1);  primals_8 = None
    unsqueeze_15: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_19: "f32[8, 32, 128, 128]" = torch.ops.aten.add.Tensor(mul_30, unsqueeze_15);  mul_30 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_4: "f32[8, 128, 128, 128]" = torch.ops.aten.convolution.default(add_19, primals_69, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_20: "i64[]" = torch.ops.aten.add.Tensor(primals_228, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
    getitem_8: "f32[1, 128, 1, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 128, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_21: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
    rsqrt_4: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_4: "f32[8, 128, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_9)
    mul_31: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
    squeeze_13: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
    mul_32: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_33: "f32[128]" = torch.ops.aten.mul.Tensor(primals_229, 0.9)
    add_22: "f32[128]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
    squeeze_14: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
    mul_34: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000076294527394);  squeeze_14 = None
    mul_35: "f32[128]" = torch.ops.aten.mul.Tensor(mul_34, 0.1);  mul_34 = None
    mul_36: "f32[128]" = torch.ops.aten.mul.Tensor(primals_230, 0.9)
    add_23: "f32[128]" = torch.ops.aten.add.Tensor(mul_35, mul_36);  mul_35 = mul_36 = None
    unsqueeze_16: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1)
    unsqueeze_17: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    mul_37: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(mul_31, unsqueeze_17);  mul_31 = unsqueeze_17 = None
    unsqueeze_18: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_10, -1);  primals_10 = None
    unsqueeze_19: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    add_24: "f32[8, 128, 128, 128]" = torch.ops.aten.add.Tensor(mul_37, unsqueeze_19);  mul_37 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_3: "f32[8, 128, 128, 128]" = torch.ops.aten.clone.default(add_24)
    sigmoid_3: "f32[8, 128, 128, 128]" = torch.ops.aten.sigmoid.default(add_24)
    mul_38: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(add_24, sigmoid_3);  add_24 = sigmoid_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_5: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(mul_38, primals_70, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 128)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_25: "i64[]" = torch.ops.aten.add.Tensor(primals_231, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
    getitem_10: "f32[1, 128, 1, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 128, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_26: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
    rsqrt_5: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_5: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_11)
    mul_39: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_15: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
    squeeze_16: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    mul_40: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_41: "f32[128]" = torch.ops.aten.mul.Tensor(primals_232, 0.9)
    add_27: "f32[128]" = torch.ops.aten.add.Tensor(mul_40, mul_41);  mul_40 = mul_41 = None
    squeeze_17: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
    mul_42: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.000030518509476);  squeeze_17 = None
    mul_43: "f32[128]" = torch.ops.aten.mul.Tensor(mul_42, 0.1);  mul_42 = None
    mul_44: "f32[128]" = torch.ops.aten.mul.Tensor(primals_233, 0.9)
    add_28: "f32[128]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
    unsqueeze_20: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_21: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_45: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_39, unsqueeze_21);  mul_39 = unsqueeze_21 = None
    unsqueeze_22: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_23: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_29: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_45, unsqueeze_23);  mul_45 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_4: "f32[8, 128, 64, 64]" = torch.ops.aten.clone.default(add_29)
    sigmoid_4: "f32[8, 128, 64, 64]" = torch.ops.aten.sigmoid.default(add_29)
    mul_46: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(add_29, sigmoid_4);  add_29 = sigmoid_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_6: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_46, primals_71, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_30: "i64[]" = torch.ops.aten.add.Tensor(primals_234, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_6 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
    getitem_12: "f32[1, 64, 1, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 64, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_31: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
    rsqrt_6: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_6: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_13)
    mul_47: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_18: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
    squeeze_19: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
    mul_48: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_49: "f32[64]" = torch.ops.aten.mul.Tensor(primals_235, 0.9)
    add_32: "f32[64]" = torch.ops.aten.add.Tensor(mul_48, mul_49);  mul_48 = mul_49 = None
    squeeze_20: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
    mul_50: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.000030518509476);  squeeze_20 = None
    mul_51: "f32[64]" = torch.ops.aten.mul.Tensor(mul_50, 0.1);  mul_50 = None
    mul_52: "f32[64]" = torch.ops.aten.mul.Tensor(primals_236, 0.9)
    add_33: "f32[64]" = torch.ops.aten.add.Tensor(mul_51, mul_52);  mul_51 = mul_52 = None
    unsqueeze_24: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1)
    unsqueeze_25: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    mul_53: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_47, unsqueeze_25);  mul_47 = unsqueeze_25 = None
    unsqueeze_26: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1);  primals_14 = None
    unsqueeze_27: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    add_34: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_53, unsqueeze_27);  mul_53 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_7: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(add_34, primals_72, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_35: "i64[]" = torch.ops.aten.add.Tensor(primals_237, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_7 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
    getitem_14: "f32[1, 256, 1, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 256, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_36: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
    rsqrt_7: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_7: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_15)
    mul_54: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_21: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
    squeeze_22: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
    mul_55: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_56: "f32[256]" = torch.ops.aten.mul.Tensor(primals_238, 0.9)
    add_37: "f32[256]" = torch.ops.aten.add.Tensor(mul_55, mul_56);  mul_55 = mul_56 = None
    squeeze_23: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
    mul_57: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.000030518509476);  squeeze_23 = None
    mul_58: "f32[256]" = torch.ops.aten.mul.Tensor(mul_57, 0.1);  mul_57 = None
    mul_59: "f32[256]" = torch.ops.aten.mul.Tensor(primals_239, 0.9)
    add_38: "f32[256]" = torch.ops.aten.add.Tensor(mul_58, mul_59);  mul_58 = mul_59 = None
    unsqueeze_28: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1)
    unsqueeze_29: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_60: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_54, unsqueeze_29);  mul_54 = unsqueeze_29 = None
    unsqueeze_30: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1);  primals_16 = None
    unsqueeze_31: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_39: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_60, unsqueeze_31);  mul_60 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_5: "f32[8, 256, 64, 64]" = torch.ops.aten.clone.default(add_39)
    sigmoid_5: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_39)
    mul_61: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_39, sigmoid_5);  add_39 = sigmoid_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_8: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(mul_61, primals_73, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 256)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_40: "i64[]" = torch.ops.aten.add.Tensor(primals_240, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_8 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
    getitem_16: "f32[1, 256, 1, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 256, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_41: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
    rsqrt_8: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
    sub_8: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_17)
    mul_62: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_24: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
    squeeze_25: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
    mul_63: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_64: "f32[256]" = torch.ops.aten.mul.Tensor(primals_241, 0.9)
    add_42: "f32[256]" = torch.ops.aten.add.Tensor(mul_63, mul_64);  mul_63 = mul_64 = None
    squeeze_26: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
    mul_65: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.000030518509476);  squeeze_26 = None
    mul_66: "f32[256]" = torch.ops.aten.mul.Tensor(mul_65, 0.1);  mul_65 = None
    mul_67: "f32[256]" = torch.ops.aten.mul.Tensor(primals_242, 0.9)
    add_43: "f32[256]" = torch.ops.aten.add.Tensor(mul_66, mul_67);  mul_66 = mul_67 = None
    unsqueeze_32: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_33: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    mul_68: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_62, unsqueeze_33);  mul_62 = unsqueeze_33 = None
    unsqueeze_34: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_35: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    add_44: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_68, unsqueeze_35);  mul_68 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_6: "f32[8, 256, 64, 64]" = torch.ops.aten.clone.default(add_44)
    sigmoid_6: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_44)
    mul_69: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_44, sigmoid_6);  add_44 = sigmoid_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_9: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_69, primals_74, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_45: "i64[]" = torch.ops.aten.add.Tensor(primals_243, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_9 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
    getitem_18: "f32[1, 64, 1, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 64, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_46: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
    rsqrt_9: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    sub_9: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_19)
    mul_70: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_27: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
    squeeze_28: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_71: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_72: "f32[64]" = torch.ops.aten.mul.Tensor(primals_244, 0.9)
    add_47: "f32[64]" = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
    squeeze_29: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
    mul_73: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.000030518509476);  squeeze_29 = None
    mul_74: "f32[64]" = torch.ops.aten.mul.Tensor(mul_73, 0.1);  mul_73 = None
    mul_75: "f32[64]" = torch.ops.aten.mul.Tensor(primals_245, 0.9)
    add_48: "f32[64]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
    unsqueeze_36: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
    unsqueeze_37: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_76: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_37);  mul_70 = unsqueeze_37 = None
    unsqueeze_38: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
    unsqueeze_39: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_49: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_39);  mul_76 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_50: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(add_49, add_34);  add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_10: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(add_50, primals_75, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_51: "i64[]" = torch.ops.aten.add.Tensor(primals_246, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_10 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[1, 256, 1, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 256, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_52: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
    rsqrt_10: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_10: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_21)
    mul_77: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_30: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
    squeeze_31: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
    mul_78: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_79: "f32[256]" = torch.ops.aten.mul.Tensor(primals_247, 0.9)
    add_53: "f32[256]" = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
    squeeze_32: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
    mul_80: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.000030518509476);  squeeze_32 = None
    mul_81: "f32[256]" = torch.ops.aten.mul.Tensor(mul_80, 0.1);  mul_80 = None
    mul_82: "f32[256]" = torch.ops.aten.mul.Tensor(primals_248, 0.9)
    add_54: "f32[256]" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
    unsqueeze_40: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1)
    unsqueeze_41: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    mul_83: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_41);  mul_77 = unsqueeze_41 = None
    unsqueeze_42: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1);  primals_22 = None
    unsqueeze_43: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    add_55: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_43);  mul_83 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_7: "f32[8, 256, 64, 64]" = torch.ops.aten.clone.default(add_55)
    sigmoid_7: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_55)
    mul_84: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_55, sigmoid_7);  add_55 = sigmoid_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_11: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(mul_84, primals_76, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 256)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_56: "i64[]" = torch.ops.aten.add.Tensor(primals_249, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_11 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[1, 256, 1, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 256, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_57: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
    rsqrt_11: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    sub_11: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_23)
    mul_85: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    squeeze_33: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
    squeeze_34: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
    mul_86: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_87: "f32[256]" = torch.ops.aten.mul.Tensor(primals_250, 0.9)
    add_58: "f32[256]" = torch.ops.aten.add.Tensor(mul_86, mul_87);  mul_86 = mul_87 = None
    squeeze_35: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
    mul_88: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.000030518509476);  squeeze_35 = None
    mul_89: "f32[256]" = torch.ops.aten.mul.Tensor(mul_88, 0.1);  mul_88 = None
    mul_90: "f32[256]" = torch.ops.aten.mul.Tensor(primals_251, 0.9)
    add_59: "f32[256]" = torch.ops.aten.add.Tensor(mul_89, mul_90);  mul_89 = mul_90 = None
    unsqueeze_44: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_45: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_91: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_85, unsqueeze_45);  mul_85 = unsqueeze_45 = None
    unsqueeze_46: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_47: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_60: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_91, unsqueeze_47);  mul_91 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_8: "f32[8, 256, 64, 64]" = torch.ops.aten.clone.default(add_60)
    sigmoid_8: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_60)
    mul_92: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_60, sigmoid_8);  add_60 = sigmoid_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_12: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_92, primals_77, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_61: "i64[]" = torch.ops.aten.add.Tensor(primals_252, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_12 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
    getitem_24: "f32[1, 64, 1, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 64, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_62: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
    rsqrt_12: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    sub_12: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_25)
    mul_93: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_36: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
    squeeze_37: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
    mul_94: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_95: "f32[64]" = torch.ops.aten.mul.Tensor(primals_253, 0.9)
    add_63: "f32[64]" = torch.ops.aten.add.Tensor(mul_94, mul_95);  mul_94 = mul_95 = None
    squeeze_38: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
    mul_96: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.000030518509476);  squeeze_38 = None
    mul_97: "f32[64]" = torch.ops.aten.mul.Tensor(mul_96, 0.1);  mul_96 = None
    mul_98: "f32[64]" = torch.ops.aten.mul.Tensor(primals_254, 0.9)
    add_64: "f32[64]" = torch.ops.aten.add.Tensor(mul_97, mul_98);  mul_97 = mul_98 = None
    unsqueeze_48: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1)
    unsqueeze_49: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    mul_99: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_93, unsqueeze_49);  mul_93 = unsqueeze_49 = None
    unsqueeze_50: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1);  primals_26 = None
    unsqueeze_51: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    add_65: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_99, unsqueeze_51);  mul_99 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_66: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(add_65, add_50);  add_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_13: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(add_66, primals_78, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_67: "i64[]" = torch.ops.aten.add.Tensor(primals_255, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_13 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
    getitem_26: "f32[1, 256, 1, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 256, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_68: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
    rsqrt_13: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    sub_13: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_27)
    mul_100: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_39: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
    squeeze_40: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_101: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_102: "f32[256]" = torch.ops.aten.mul.Tensor(primals_256, 0.9)
    add_69: "f32[256]" = torch.ops.aten.add.Tensor(mul_101, mul_102);  mul_101 = mul_102 = None
    squeeze_41: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
    mul_103: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.000030518509476);  squeeze_41 = None
    mul_104: "f32[256]" = torch.ops.aten.mul.Tensor(mul_103, 0.1);  mul_103 = None
    mul_105: "f32[256]" = torch.ops.aten.mul.Tensor(primals_257, 0.9)
    add_70: "f32[256]" = torch.ops.aten.add.Tensor(mul_104, mul_105);  mul_104 = mul_105 = None
    unsqueeze_52: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1)
    unsqueeze_53: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_106: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_100, unsqueeze_53);  mul_100 = unsqueeze_53 = None
    unsqueeze_54: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1);  primals_28 = None
    unsqueeze_55: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_71: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_106, unsqueeze_55);  mul_106 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_9: "f32[8, 256, 64, 64]" = torch.ops.aten.clone.default(add_71)
    sigmoid_9: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_71)
    mul_107: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_71, sigmoid_9);  add_71 = sigmoid_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_14: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(mul_107, primals_79, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 256)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_72: "i64[]" = torch.ops.aten.add.Tensor(primals_258, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_14 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
    getitem_28: "f32[1, 256, 1, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 256, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_73: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
    rsqrt_14: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    sub_14: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_29)
    mul_108: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_42: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
    squeeze_43: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
    mul_109: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_110: "f32[256]" = torch.ops.aten.mul.Tensor(primals_259, 0.9)
    add_74: "f32[256]" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
    squeeze_44: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
    mul_111: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0001220852154804);  squeeze_44 = None
    mul_112: "f32[256]" = torch.ops.aten.mul.Tensor(mul_111, 0.1);  mul_111 = None
    mul_113: "f32[256]" = torch.ops.aten.mul.Tensor(primals_260, 0.9)
    add_75: "f32[256]" = torch.ops.aten.add.Tensor(mul_112, mul_113);  mul_112 = mul_113 = None
    unsqueeze_56: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_57: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    mul_114: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_108, unsqueeze_57);  mul_108 = unsqueeze_57 = None
    unsqueeze_58: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_59: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    add_76: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_114, unsqueeze_59);  mul_114 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_10: "f32[8, 256, 32, 32]" = torch.ops.aten.clone.default(add_76)
    sigmoid_10: "f32[8, 256, 32, 32]" = torch.ops.aten.sigmoid.default(add_76)
    mul_115: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(add_76, sigmoid_10);  add_76 = sigmoid_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_15: "f32[8, 96, 32, 32]" = torch.ops.aten.convolution.default(mul_115, primals_80, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_77: "i64[]" = torch.ops.aten.add.Tensor(primals_261, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_15 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
    getitem_30: "f32[1, 96, 1, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 96, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_78: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
    rsqrt_15: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_15: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_31)
    mul_116: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    squeeze_45: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
    squeeze_46: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
    mul_117: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_118: "f32[96]" = torch.ops.aten.mul.Tensor(primals_262, 0.9)
    add_79: "f32[96]" = torch.ops.aten.add.Tensor(mul_117, mul_118);  mul_117 = mul_118 = None
    squeeze_47: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
    mul_119: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0001220852154804);  squeeze_47 = None
    mul_120: "f32[96]" = torch.ops.aten.mul.Tensor(mul_119, 0.1);  mul_119 = None
    mul_121: "f32[96]" = torch.ops.aten.mul.Tensor(primals_263, 0.9)
    add_80: "f32[96]" = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
    unsqueeze_60: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1)
    unsqueeze_61: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_122: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(mul_116, unsqueeze_61);  mul_116 = unsqueeze_61 = None
    unsqueeze_62: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1);  primals_32 = None
    unsqueeze_63: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_81: "f32[8, 96, 32, 32]" = torch.ops.aten.add.Tensor(mul_122, unsqueeze_63);  mul_122 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_16: "f32[8, 96, 32, 32]" = torch.ops.aten.convolution.default(add_81, primals_81, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_82: "i64[]" = torch.ops.aten.add.Tensor(primals_264, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_16 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
    getitem_32: "f32[1, 96, 1, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 96, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_83: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
    rsqrt_16: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
    sub_16: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_33)
    mul_123: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_48: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
    squeeze_49: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
    mul_124: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_125: "f32[96]" = torch.ops.aten.mul.Tensor(primals_265, 0.9)
    add_84: "f32[96]" = torch.ops.aten.add.Tensor(mul_124, mul_125);  mul_124 = mul_125 = None
    squeeze_50: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
    mul_126: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0001220852154804);  squeeze_50 = None
    mul_127: "f32[96]" = torch.ops.aten.mul.Tensor(mul_126, 0.1);  mul_126 = None
    mul_128: "f32[96]" = torch.ops.aten.mul.Tensor(primals_266, 0.9)
    add_85: "f32[96]" = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
    unsqueeze_64: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1)
    unsqueeze_65: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    mul_129: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(mul_123, unsqueeze_65);  mul_123 = unsqueeze_65 = None
    unsqueeze_66: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1);  primals_34 = None
    unsqueeze_67: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    add_86: "f32[8, 96, 32, 32]" = torch.ops.aten.add.Tensor(mul_129, unsqueeze_67);  mul_129 = unsqueeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_11: "f32[8, 96, 32, 32]" = torch.ops.aten.clone.default(add_86)
    sigmoid_11: "f32[8, 96, 32, 32]" = torch.ops.aten.sigmoid.default(add_86)
    mul_130: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(add_86, sigmoid_11);  add_86 = sigmoid_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:234, code: x = self.conv_1x1(x)
    convolution_17: "f32[8, 144, 32, 32]" = torch.ops.aten.convolution.default(mul_130, primals_82, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:249, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
    view: "f32[18432, 2, 16, 2]" = torch.ops.aten.view.default(convolution_17, [18432, 2, 16, 2]);  convolution_17 = None
    permute: "f32[18432, 16, 2, 2]" = torch.ops.aten.permute.default(view, [0, 2, 1, 3]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:251, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
    clone_12: "f32[18432, 16, 2, 2]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
    view_1: "f32[8, 144, 256, 4]" = torch.ops.aten.view.default(clone_12, [8, 144, 256, 4]);  clone_12 = None
    permute_1: "f32[8, 4, 256, 144]" = torch.ops.aten.permute.default(view_1, [0, 3, 2, 1]);  view_1 = None
    clone_13: "f32[8, 4, 256, 144]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
    view_2: "f32[32, 256, 144]" = torch.ops.aten.view.default(clone_13, [32, 256, 144]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_17 = torch.ops.aten.var_mean.correction(view_2, [2], correction = 0, keepdim = True)
    getitem_34: "f32[32, 256, 1]" = var_mean_17[0]
    getitem_35: "f32[32, 256, 1]" = var_mean_17[1];  var_mean_17 = None
    add_87: "f32[32, 256, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
    rsqrt_17: "f32[32, 256, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    sub_17: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(view_2, getitem_35)
    mul_131: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    mul_132: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_131, primals_83);  mul_131 = None
    add_88: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(mul_132, primals_84);  mul_132 = primals_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_3: "f32[8192, 144]" = torch.ops.aten.view.default(add_88, [8192, 144]);  add_88 = None
    permute_2: "f32[144, 432]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
    addmm: "f32[8192, 432]" = torch.ops.aten.addmm.default(primals_86, view_3, permute_2);  primals_86 = None
    view_4: "f32[32, 256, 432]" = torch.ops.aten.view.default(addmm, [32, 256, 432]);  addmm = None
    view_5: "f32[32, 256, 3, 4, 36]" = torch.ops.aten.view.default(view_4, [32, 256, 3, 4, 36]);  view_4 = None
    permute_3: "f32[3, 32, 4, 256, 36]" = torch.ops.aten.permute.default(view_5, [2, 0, 3, 1, 4]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind = torch.ops.aten.unbind.int(permute_3);  permute_3 = None
    getitem_36: "f32[32, 4, 256, 36]" = unbind[0]
    getitem_37: "f32[32, 4, 256, 36]" = unbind[1]
    getitem_38: "f32[32, 4, 256, 36]" = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_36, getitem_37, getitem_38)
    getitem_39: "f32[32, 4, 256, 36]" = _scaled_dot_product_flash_attention[0]
    getitem_40: "f32[32, 4, 256]" = _scaled_dot_product_flash_attention[1]
    getitem_41: "i32[]" = _scaled_dot_product_flash_attention[2]
    getitem_42: "i32[]" = _scaled_dot_product_flash_attention[3]
    getitem_45: "i64[]" = _scaled_dot_product_flash_attention[6]
    getitem_46: "i64[]" = _scaled_dot_product_flash_attention[7];  _scaled_dot_product_flash_attention = None
    alias: "f32[32, 4, 256, 36]" = torch.ops.aten.alias.default(getitem_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_4: "f32[32, 256, 4, 36]" = torch.ops.aten.permute.default(getitem_39, [0, 2, 1, 3]);  getitem_39 = None
    view_6: "f32[32, 256, 144]" = torch.ops.aten.view.default(permute_4, [32, 256, 144]);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_7: "f32[8192, 144]" = torch.ops.aten.view.default(view_6, [8192, 144]);  view_6 = None
    permute_5: "f32[144, 144]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
    addmm_1: "f32[8192, 144]" = torch.ops.aten.addmm.default(primals_88, view_7, permute_5);  primals_88 = None
    view_8: "f32[32, 256, 144]" = torch.ops.aten.view.default(addmm_1, [32, 256, 144]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_14: "f32[32, 256, 144]" = torch.ops.aten.clone.default(view_8);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_89: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(view_2, clone_14);  clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_18 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
    getitem_48: "f32[32, 256, 1]" = var_mean_18[0]
    getitem_49: "f32[32, 256, 1]" = var_mean_18[1];  var_mean_18 = None
    add_90: "f32[32, 256, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
    rsqrt_18: "f32[32, 256, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
    sub_18: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(add_89, getitem_49)
    mul_133: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    mul_134: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_133, primals_89);  mul_133 = None
    add_91: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(mul_134, primals_90);  mul_134 = primals_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_9: "f32[8192, 144]" = torch.ops.aten.view.default(add_91, [8192, 144]);  add_91 = None
    permute_6: "f32[144, 288]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
    addmm_2: "f32[8192, 288]" = torch.ops.aten.addmm.default(primals_92, view_9, permute_6);  primals_92 = None
    view_10: "f32[32, 256, 288]" = torch.ops.aten.view.default(addmm_2, [32, 256, 288]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_12: "f32[32, 256, 288]" = torch.ops.aten.sigmoid.default(view_10)
    mul_135: "f32[32, 256, 288]" = torch.ops.aten.mul.Tensor(view_10, sigmoid_12);  sigmoid_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_15: "f32[32, 256, 288]" = torch.ops.aten.clone.default(mul_135);  mul_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_11: "f32[8192, 288]" = torch.ops.aten.view.default(clone_15, [8192, 288]);  clone_15 = None
    permute_7: "f32[288, 144]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
    addmm_3: "f32[8192, 144]" = torch.ops.aten.addmm.default(primals_94, view_11, permute_7);  primals_94 = None
    view_12: "f32[32, 256, 144]" = torch.ops.aten.view.default(addmm_3, [32, 256, 144]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_16: "f32[32, 256, 144]" = torch.ops.aten.clone.default(view_12);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_92: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(add_89, clone_16);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_19 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
    getitem_50: "f32[32, 256, 1]" = var_mean_19[0]
    getitem_51: "f32[32, 256, 1]" = var_mean_19[1];  var_mean_19 = None
    add_93: "f32[32, 256, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
    rsqrt_19: "f32[32, 256, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
    sub_19: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(add_92, getitem_51)
    mul_136: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    mul_137: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_136, primals_95);  mul_136 = None
    add_94: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(mul_137, primals_96);  mul_137 = primals_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_13: "f32[8192, 144]" = torch.ops.aten.view.default(add_94, [8192, 144]);  add_94 = None
    permute_8: "f32[144, 432]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
    addmm_4: "f32[8192, 432]" = torch.ops.aten.addmm.default(primals_98, view_13, permute_8);  primals_98 = None
    view_14: "f32[32, 256, 432]" = torch.ops.aten.view.default(addmm_4, [32, 256, 432]);  addmm_4 = None
    view_15: "f32[32, 256, 3, 4, 36]" = torch.ops.aten.view.default(view_14, [32, 256, 3, 4, 36]);  view_14 = None
    permute_9: "f32[3, 32, 4, 256, 36]" = torch.ops.aten.permute.default(view_15, [2, 0, 3, 1, 4]);  view_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_1 = torch.ops.aten.unbind.int(permute_9);  permute_9 = None
    getitem_52: "f32[32, 4, 256, 36]" = unbind_1[0]
    getitem_53: "f32[32, 4, 256, 36]" = unbind_1[1]
    getitem_54: "f32[32, 4, 256, 36]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_1 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_52, getitem_53, getitem_54)
    getitem_55: "f32[32, 4, 256, 36]" = _scaled_dot_product_flash_attention_1[0]
    getitem_56: "f32[32, 4, 256]" = _scaled_dot_product_flash_attention_1[1]
    getitem_57: "i32[]" = _scaled_dot_product_flash_attention_1[2]
    getitem_58: "i32[]" = _scaled_dot_product_flash_attention_1[3]
    getitem_61: "i64[]" = _scaled_dot_product_flash_attention_1[6]
    getitem_62: "i64[]" = _scaled_dot_product_flash_attention_1[7];  _scaled_dot_product_flash_attention_1 = None
    alias_1: "f32[32, 4, 256, 36]" = torch.ops.aten.alias.default(getitem_55)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_10: "f32[32, 256, 4, 36]" = torch.ops.aten.permute.default(getitem_55, [0, 2, 1, 3]);  getitem_55 = None
    view_16: "f32[32, 256, 144]" = torch.ops.aten.view.default(permute_10, [32, 256, 144]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_17: "f32[8192, 144]" = torch.ops.aten.view.default(view_16, [8192, 144]);  view_16 = None
    permute_11: "f32[144, 144]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
    addmm_5: "f32[8192, 144]" = torch.ops.aten.addmm.default(primals_100, view_17, permute_11);  primals_100 = None
    view_18: "f32[32, 256, 144]" = torch.ops.aten.view.default(addmm_5, [32, 256, 144]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_17: "f32[32, 256, 144]" = torch.ops.aten.clone.default(view_18);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_95: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(add_92, clone_17);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_20 = torch.ops.aten.var_mean.correction(add_95, [2], correction = 0, keepdim = True)
    getitem_64: "f32[32, 256, 1]" = var_mean_20[0]
    getitem_65: "f32[32, 256, 1]" = var_mean_20[1];  var_mean_20 = None
    add_96: "f32[32, 256, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
    rsqrt_20: "f32[32, 256, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
    sub_20: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(add_95, getitem_65)
    mul_138: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    mul_139: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_138, primals_101);  mul_138 = None
    add_97: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(mul_139, primals_102);  mul_139 = primals_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_19: "f32[8192, 144]" = torch.ops.aten.view.default(add_97, [8192, 144]);  add_97 = None
    permute_12: "f32[144, 288]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
    addmm_6: "f32[8192, 288]" = torch.ops.aten.addmm.default(primals_104, view_19, permute_12);  primals_104 = None
    view_20: "f32[32, 256, 288]" = torch.ops.aten.view.default(addmm_6, [32, 256, 288]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_13: "f32[32, 256, 288]" = torch.ops.aten.sigmoid.default(view_20)
    mul_140: "f32[32, 256, 288]" = torch.ops.aten.mul.Tensor(view_20, sigmoid_13);  sigmoid_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_18: "f32[32, 256, 288]" = torch.ops.aten.clone.default(mul_140);  mul_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_21: "f32[8192, 288]" = torch.ops.aten.view.default(clone_18, [8192, 288]);  clone_18 = None
    permute_13: "f32[288, 144]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
    addmm_7: "f32[8192, 144]" = torch.ops.aten.addmm.default(primals_106, view_21, permute_13);  primals_106 = None
    view_22: "f32[32, 256, 144]" = torch.ops.aten.view.default(addmm_7, [32, 256, 144]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_19: "f32[32, 256, 144]" = torch.ops.aten.clone.default(view_22);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_98: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(add_95, clone_19);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:255, code: x = self.norm(x)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_98, [2], correction = 0, keepdim = True)
    getitem_66: "f32[32, 256, 1]" = var_mean_21[0]
    getitem_67: "f32[32, 256, 1]" = var_mean_21[1];  var_mean_21 = None
    add_99: "f32[32, 256, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
    rsqrt_21: "f32[32, 256, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
    sub_21: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(add_98, getitem_67)
    mul_141: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    mul_142: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_141, primals_107);  mul_141 = None
    add_100: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(mul_142, primals_108);  mul_142 = primals_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:259, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
    view_23: "f32[8, 4, 256, 144]" = torch.ops.aten.view.default(add_100, [8, 4, 256, -1]);  add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:260, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
    permute_14: "f32[8, 144, 256, 4]" = torch.ops.aten.permute.default(view_23, [0, 3, 2, 1]);  view_23 = None
    clone_20: "f32[8, 144, 256, 4]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
    view_24: "f32[18432, 16, 2, 2]" = torch.ops.aten.view.default(clone_20, [18432, 16, 2, 2]);  clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:262, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
    permute_15: "f32[18432, 2, 16, 2]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
    clone_21: "f32[18432, 2, 16, 2]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    view_25: "f32[8, 144, 32, 32]" = torch.ops.aten.view.default(clone_21, [8, 144, 32, 32]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_18: "f32[8, 96, 32, 32]" = torch.ops.aten.convolution.default(view_25, primals_109, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_101: "i64[]" = torch.ops.aten.add.Tensor(primals_267, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_22 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
    getitem_68: "f32[1, 96, 1, 1]" = var_mean_22[0]
    getitem_69: "f32[1, 96, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_102: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05)
    rsqrt_22: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
    sub_22: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_69)
    mul_143: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    squeeze_51: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
    squeeze_52: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
    mul_144: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_145: "f32[96]" = torch.ops.aten.mul.Tensor(primals_268, 0.9)
    add_103: "f32[96]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
    squeeze_53: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
    mul_146: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0001220852154804);  squeeze_53 = None
    mul_147: "f32[96]" = torch.ops.aten.mul.Tensor(mul_146, 0.1);  mul_146 = None
    mul_148: "f32[96]" = torch.ops.aten.mul.Tensor(primals_269, 0.9)
    add_104: "f32[96]" = torch.ops.aten.add.Tensor(mul_147, mul_148);  mul_147 = mul_148 = None
    unsqueeze_68: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_69: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_149: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(mul_143, unsqueeze_69);  mul_143 = unsqueeze_69 = None
    unsqueeze_70: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_71: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_105: "f32[8, 96, 32, 32]" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_71);  mul_149 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_22: "f32[8, 96, 32, 32]" = torch.ops.aten.clone.default(add_105)
    sigmoid_14: "f32[8, 96, 32, 32]" = torch.ops.aten.sigmoid.default(add_105)
    mul_150: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(add_105, sigmoid_14);  add_105 = sigmoid_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:268, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
    cat: "f32[8, 192, 32, 32]" = torch.ops.aten.cat.default([add_81, mul_150], 1);  mul_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_19: "f32[8, 96, 32, 32]" = torch.ops.aten.convolution.default(cat, primals_110, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_106: "i64[]" = torch.ops.aten.add.Tensor(primals_270, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_23 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
    getitem_70: "f32[1, 96, 1, 1]" = var_mean_23[0]
    getitem_71: "f32[1, 96, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_107: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05)
    rsqrt_23: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
    sub_23: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_71)
    mul_151: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    squeeze_54: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
    squeeze_55: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
    mul_152: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_153: "f32[96]" = torch.ops.aten.mul.Tensor(primals_271, 0.9)
    add_108: "f32[96]" = torch.ops.aten.add.Tensor(mul_152, mul_153);  mul_152 = mul_153 = None
    squeeze_56: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
    mul_154: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0001220852154804);  squeeze_56 = None
    mul_155: "f32[96]" = torch.ops.aten.mul.Tensor(mul_154, 0.1);  mul_154 = None
    mul_156: "f32[96]" = torch.ops.aten.mul.Tensor(primals_272, 0.9)
    add_109: "f32[96]" = torch.ops.aten.add.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
    unsqueeze_72: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1)
    unsqueeze_73: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    mul_157: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(mul_151, unsqueeze_73);  mul_151 = unsqueeze_73 = None
    unsqueeze_74: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1);  primals_38 = None
    unsqueeze_75: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    add_110: "f32[8, 96, 32, 32]" = torch.ops.aten.add.Tensor(mul_157, unsqueeze_75);  mul_157 = unsqueeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_23: "f32[8, 96, 32, 32]" = torch.ops.aten.clone.default(add_110)
    sigmoid_15: "f32[8, 96, 32, 32]" = torch.ops.aten.sigmoid.default(add_110)
    mul_158: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(add_110, sigmoid_15);  add_110 = sigmoid_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_20: "f32[8, 384, 32, 32]" = torch.ops.aten.convolution.default(mul_158, primals_111, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_111: "i64[]" = torch.ops.aten.add.Tensor(primals_273, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_24 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
    getitem_72: "f32[1, 384, 1, 1]" = var_mean_24[0]
    getitem_73: "f32[1, 384, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_112: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05)
    rsqrt_24: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
    sub_24: "f32[8, 384, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_73)
    mul_159: "f32[8, 384, 32, 32]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    squeeze_57: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
    squeeze_58: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
    mul_160: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_161: "f32[384]" = torch.ops.aten.mul.Tensor(primals_274, 0.9)
    add_113: "f32[384]" = torch.ops.aten.add.Tensor(mul_160, mul_161);  mul_160 = mul_161 = None
    squeeze_59: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
    mul_162: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0001220852154804);  squeeze_59 = None
    mul_163: "f32[384]" = torch.ops.aten.mul.Tensor(mul_162, 0.1);  mul_162 = None
    mul_164: "f32[384]" = torch.ops.aten.mul.Tensor(primals_275, 0.9)
    add_114: "f32[384]" = torch.ops.aten.add.Tensor(mul_163, mul_164);  mul_163 = mul_164 = None
    unsqueeze_76: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1)
    unsqueeze_77: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_165: "f32[8, 384, 32, 32]" = torch.ops.aten.mul.Tensor(mul_159, unsqueeze_77);  mul_159 = unsqueeze_77 = None
    unsqueeze_78: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1);  primals_40 = None
    unsqueeze_79: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_115: "f32[8, 384, 32, 32]" = torch.ops.aten.add.Tensor(mul_165, unsqueeze_79);  mul_165 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_24: "f32[8, 384, 32, 32]" = torch.ops.aten.clone.default(add_115)
    sigmoid_16: "f32[8, 384, 32, 32]" = torch.ops.aten.sigmoid.default(add_115)
    mul_166: "f32[8, 384, 32, 32]" = torch.ops.aten.mul.Tensor(add_115, sigmoid_16);  add_115 = sigmoid_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_21: "f32[8, 384, 16, 16]" = torch.ops.aten.convolution.default(mul_166, primals_112, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 384)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_116: "i64[]" = torch.ops.aten.add.Tensor(primals_276, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_25 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
    getitem_74: "f32[1, 384, 1, 1]" = var_mean_25[0]
    getitem_75: "f32[1, 384, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_117: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05)
    rsqrt_25: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
    sub_25: "f32[8, 384, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_75)
    mul_167: "f32[8, 384, 16, 16]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    squeeze_60: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
    squeeze_61: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
    mul_168: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_169: "f32[384]" = torch.ops.aten.mul.Tensor(primals_277, 0.9)
    add_118: "f32[384]" = torch.ops.aten.add.Tensor(mul_168, mul_169);  mul_168 = mul_169 = None
    squeeze_62: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
    mul_170: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0004885197850513);  squeeze_62 = None
    mul_171: "f32[384]" = torch.ops.aten.mul.Tensor(mul_170, 0.1);  mul_170 = None
    mul_172: "f32[384]" = torch.ops.aten.mul.Tensor(primals_278, 0.9)
    add_119: "f32[384]" = torch.ops.aten.add.Tensor(mul_171, mul_172);  mul_171 = mul_172 = None
    unsqueeze_80: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_81: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    mul_173: "f32[8, 384, 16, 16]" = torch.ops.aten.mul.Tensor(mul_167, unsqueeze_81);  mul_167 = unsqueeze_81 = None
    unsqueeze_82: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_83: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    add_120: "f32[8, 384, 16, 16]" = torch.ops.aten.add.Tensor(mul_173, unsqueeze_83);  mul_173 = unsqueeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_25: "f32[8, 384, 16, 16]" = torch.ops.aten.clone.default(add_120)
    sigmoid_17: "f32[8, 384, 16, 16]" = torch.ops.aten.sigmoid.default(add_120)
    mul_174: "f32[8, 384, 16, 16]" = torch.ops.aten.mul.Tensor(add_120, sigmoid_17);  add_120 = sigmoid_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_22: "f32[8, 128, 16, 16]" = torch.ops.aten.convolution.default(mul_174, primals_113, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_121: "i64[]" = torch.ops.aten.add.Tensor(primals_279, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_26 = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
    getitem_76: "f32[1, 128, 1, 1]" = var_mean_26[0]
    getitem_77: "f32[1, 128, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_122: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
    rsqrt_26: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    sub_26: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_22, getitem_77)
    mul_175: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    squeeze_63: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
    squeeze_64: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
    mul_176: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_177: "f32[128]" = torch.ops.aten.mul.Tensor(primals_280, 0.9)
    add_123: "f32[128]" = torch.ops.aten.add.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
    squeeze_65: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
    mul_178: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0004885197850513);  squeeze_65 = None
    mul_179: "f32[128]" = torch.ops.aten.mul.Tensor(mul_178, 0.1);  mul_178 = None
    mul_180: "f32[128]" = torch.ops.aten.mul.Tensor(primals_281, 0.9)
    add_124: "f32[128]" = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
    unsqueeze_84: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1)
    unsqueeze_85: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_181: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_85);  mul_175 = unsqueeze_85 = None
    unsqueeze_86: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1);  primals_44 = None
    unsqueeze_87: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_125: "f32[8, 128, 16, 16]" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_87);  mul_181 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_23: "f32[8, 128, 16, 16]" = torch.ops.aten.convolution.default(add_125, primals_114, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_126: "i64[]" = torch.ops.aten.add.Tensor(primals_282, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_27 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
    getitem_78: "f32[1, 128, 1, 1]" = var_mean_27[0]
    getitem_79: "f32[1, 128, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_127: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
    rsqrt_27: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
    sub_27: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_79)
    mul_182: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    squeeze_66: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
    squeeze_67: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
    mul_183: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_184: "f32[128]" = torch.ops.aten.mul.Tensor(primals_283, 0.9)
    add_128: "f32[128]" = torch.ops.aten.add.Tensor(mul_183, mul_184);  mul_183 = mul_184 = None
    squeeze_68: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
    mul_185: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0004885197850513);  squeeze_68 = None
    mul_186: "f32[128]" = torch.ops.aten.mul.Tensor(mul_185, 0.1);  mul_185 = None
    mul_187: "f32[128]" = torch.ops.aten.mul.Tensor(primals_284, 0.9)
    add_129: "f32[128]" = torch.ops.aten.add.Tensor(mul_186, mul_187);  mul_186 = mul_187 = None
    unsqueeze_88: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1)
    unsqueeze_89: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    mul_188: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_89);  mul_182 = unsqueeze_89 = None
    unsqueeze_90: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1);  primals_46 = None
    unsqueeze_91: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    add_130: "f32[8, 128, 16, 16]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_91);  mul_188 = unsqueeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_26: "f32[8, 128, 16, 16]" = torch.ops.aten.clone.default(add_130)
    sigmoid_18: "f32[8, 128, 16, 16]" = torch.ops.aten.sigmoid.default(add_130)
    mul_189: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(add_130, sigmoid_18);  add_130 = sigmoid_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:234, code: x = self.conv_1x1(x)
    convolution_24: "f32[8, 192, 16, 16]" = torch.ops.aten.convolution.default(mul_189, primals_115, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:249, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
    view_26: "f32[12288, 2, 8, 2]" = torch.ops.aten.view.default(convolution_24, [12288, 2, 8, 2]);  convolution_24 = None
    permute_16: "f32[12288, 8, 2, 2]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:251, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
    clone_27: "f32[12288, 8, 2, 2]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    view_27: "f32[8, 192, 64, 4]" = torch.ops.aten.view.default(clone_27, [8, 192, 64, 4]);  clone_27 = None
    permute_17: "f32[8, 4, 64, 192]" = torch.ops.aten.permute.default(view_27, [0, 3, 2, 1]);  view_27 = None
    clone_28: "f32[8, 4, 64, 192]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
    view_28: "f32[32, 64, 192]" = torch.ops.aten.view.default(clone_28, [32, 64, 192]);  clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_28 = torch.ops.aten.var_mean.correction(view_28, [2], correction = 0, keepdim = True)
    getitem_80: "f32[32, 64, 1]" = var_mean_28[0]
    getitem_81: "f32[32, 64, 1]" = var_mean_28[1];  var_mean_28 = None
    add_131: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
    rsqrt_28: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
    sub_28: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(view_28, getitem_81)
    mul_190: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    mul_191: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_190, primals_116);  mul_190 = None
    add_132: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_191, primals_117);  mul_191 = primals_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_29: "f32[2048, 192]" = torch.ops.aten.view.default(add_132, [2048, 192]);  add_132 = None
    permute_18: "f32[192, 576]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
    addmm_8: "f32[2048, 576]" = torch.ops.aten.addmm.default(primals_119, view_29, permute_18);  primals_119 = None
    view_30: "f32[32, 64, 576]" = torch.ops.aten.view.default(addmm_8, [32, 64, 576]);  addmm_8 = None
    view_31: "f32[32, 64, 3, 4, 48]" = torch.ops.aten.view.default(view_30, [32, 64, 3, 4, 48]);  view_30 = None
    permute_19: "f32[3, 32, 4, 64, 48]" = torch.ops.aten.permute.default(view_31, [2, 0, 3, 1, 4]);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_2 = torch.ops.aten.unbind.int(permute_19);  permute_19 = None
    getitem_82: "f32[32, 4, 64, 48]" = unbind_2[0]
    getitem_83: "f32[32, 4, 64, 48]" = unbind_2[1]
    getitem_84: "f32[32, 4, 64, 48]" = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_2 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_82, getitem_83, getitem_84)
    getitem_85: "f32[32, 4, 64, 48]" = _scaled_dot_product_flash_attention_2[0]
    getitem_86: "f32[32, 4, 64]" = _scaled_dot_product_flash_attention_2[1]
    getitem_87: "i32[]" = _scaled_dot_product_flash_attention_2[2]
    getitem_88: "i32[]" = _scaled_dot_product_flash_attention_2[3]
    getitem_91: "i64[]" = _scaled_dot_product_flash_attention_2[6]
    getitem_92: "i64[]" = _scaled_dot_product_flash_attention_2[7];  _scaled_dot_product_flash_attention_2 = None
    alias_2: "f32[32, 4, 64, 48]" = torch.ops.aten.alias.default(getitem_85)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_20: "f32[32, 64, 4, 48]" = torch.ops.aten.permute.default(getitem_85, [0, 2, 1, 3]);  getitem_85 = None
    view_32: "f32[32, 64, 192]" = torch.ops.aten.view.default(permute_20, [32, 64, 192]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_33: "f32[2048, 192]" = torch.ops.aten.view.default(view_32, [2048, 192]);  view_32 = None
    permute_21: "f32[192, 192]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
    addmm_9: "f32[2048, 192]" = torch.ops.aten.addmm.default(primals_121, view_33, permute_21);  primals_121 = None
    view_34: "f32[32, 64, 192]" = torch.ops.aten.view.default(addmm_9, [32, 64, 192]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_29: "f32[32, 64, 192]" = torch.ops.aten.clone.default(view_34);  view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_133: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(view_28, clone_29);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_29 = torch.ops.aten.var_mean.correction(add_133, [2], correction = 0, keepdim = True)
    getitem_94: "f32[32, 64, 1]" = var_mean_29[0]
    getitem_95: "f32[32, 64, 1]" = var_mean_29[1];  var_mean_29 = None
    add_134: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
    rsqrt_29: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
    sub_29: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_133, getitem_95)
    mul_192: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    mul_193: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_192, primals_122);  mul_192 = None
    add_135: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_193, primals_123);  mul_193 = primals_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_35: "f32[2048, 192]" = torch.ops.aten.view.default(add_135, [2048, 192]);  add_135 = None
    permute_22: "f32[192, 384]" = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
    addmm_10: "f32[2048, 384]" = torch.ops.aten.addmm.default(primals_125, view_35, permute_22);  primals_125 = None
    view_36: "f32[32, 64, 384]" = torch.ops.aten.view.default(addmm_10, [32, 64, 384]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_19: "f32[32, 64, 384]" = torch.ops.aten.sigmoid.default(view_36)
    mul_194: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(view_36, sigmoid_19);  sigmoid_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_30: "f32[32, 64, 384]" = torch.ops.aten.clone.default(mul_194);  mul_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_37: "f32[2048, 384]" = torch.ops.aten.view.default(clone_30, [2048, 384]);  clone_30 = None
    permute_23: "f32[384, 192]" = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
    addmm_11: "f32[2048, 192]" = torch.ops.aten.addmm.default(primals_127, view_37, permute_23);  primals_127 = None
    view_38: "f32[32, 64, 192]" = torch.ops.aten.view.default(addmm_11, [32, 64, 192]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_31: "f32[32, 64, 192]" = torch.ops.aten.clone.default(view_38);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_136: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_133, clone_31);  clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_30 = torch.ops.aten.var_mean.correction(add_136, [2], correction = 0, keepdim = True)
    getitem_96: "f32[32, 64, 1]" = var_mean_30[0]
    getitem_97: "f32[32, 64, 1]" = var_mean_30[1];  var_mean_30 = None
    add_137: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
    rsqrt_30: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
    sub_30: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_136, getitem_97)
    mul_195: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    mul_196: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_195, primals_128);  mul_195 = None
    add_138: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_196, primals_129);  mul_196 = primals_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_39: "f32[2048, 192]" = torch.ops.aten.view.default(add_138, [2048, 192]);  add_138 = None
    permute_24: "f32[192, 576]" = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
    addmm_12: "f32[2048, 576]" = torch.ops.aten.addmm.default(primals_131, view_39, permute_24);  primals_131 = None
    view_40: "f32[32, 64, 576]" = torch.ops.aten.view.default(addmm_12, [32, 64, 576]);  addmm_12 = None
    view_41: "f32[32, 64, 3, 4, 48]" = torch.ops.aten.view.default(view_40, [32, 64, 3, 4, 48]);  view_40 = None
    permute_25: "f32[3, 32, 4, 64, 48]" = torch.ops.aten.permute.default(view_41, [2, 0, 3, 1, 4]);  view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_3 = torch.ops.aten.unbind.int(permute_25);  permute_25 = None
    getitem_98: "f32[32, 4, 64, 48]" = unbind_3[0]
    getitem_99: "f32[32, 4, 64, 48]" = unbind_3[1]
    getitem_100: "f32[32, 4, 64, 48]" = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_3 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_98, getitem_99, getitem_100)
    getitem_101: "f32[32, 4, 64, 48]" = _scaled_dot_product_flash_attention_3[0]
    getitem_102: "f32[32, 4, 64]" = _scaled_dot_product_flash_attention_3[1]
    getitem_103: "i32[]" = _scaled_dot_product_flash_attention_3[2]
    getitem_104: "i32[]" = _scaled_dot_product_flash_attention_3[3]
    getitem_107: "i64[]" = _scaled_dot_product_flash_attention_3[6]
    getitem_108: "i64[]" = _scaled_dot_product_flash_attention_3[7];  _scaled_dot_product_flash_attention_3 = None
    alias_3: "f32[32, 4, 64, 48]" = torch.ops.aten.alias.default(getitem_101)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_26: "f32[32, 64, 4, 48]" = torch.ops.aten.permute.default(getitem_101, [0, 2, 1, 3]);  getitem_101 = None
    view_42: "f32[32, 64, 192]" = torch.ops.aten.view.default(permute_26, [32, 64, 192]);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_43: "f32[2048, 192]" = torch.ops.aten.view.default(view_42, [2048, 192]);  view_42 = None
    permute_27: "f32[192, 192]" = torch.ops.aten.permute.default(primals_132, [1, 0]);  primals_132 = None
    addmm_13: "f32[2048, 192]" = torch.ops.aten.addmm.default(primals_133, view_43, permute_27);  primals_133 = None
    view_44: "f32[32, 64, 192]" = torch.ops.aten.view.default(addmm_13, [32, 64, 192]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_32: "f32[32, 64, 192]" = torch.ops.aten.clone.default(view_44);  view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_139: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_136, clone_32);  clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_31 = torch.ops.aten.var_mean.correction(add_139, [2], correction = 0, keepdim = True)
    getitem_110: "f32[32, 64, 1]" = var_mean_31[0]
    getitem_111: "f32[32, 64, 1]" = var_mean_31[1];  var_mean_31 = None
    add_140: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
    rsqrt_31: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    sub_31: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_139, getitem_111)
    mul_197: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    mul_198: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_197, primals_134);  mul_197 = None
    add_141: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_198, primals_135);  mul_198 = primals_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_45: "f32[2048, 192]" = torch.ops.aten.view.default(add_141, [2048, 192]);  add_141 = None
    permute_28: "f32[192, 384]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
    addmm_14: "f32[2048, 384]" = torch.ops.aten.addmm.default(primals_137, view_45, permute_28);  primals_137 = None
    view_46: "f32[32, 64, 384]" = torch.ops.aten.view.default(addmm_14, [32, 64, 384]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_20: "f32[32, 64, 384]" = torch.ops.aten.sigmoid.default(view_46)
    mul_199: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(view_46, sigmoid_20);  sigmoid_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_33: "f32[32, 64, 384]" = torch.ops.aten.clone.default(mul_199);  mul_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_47: "f32[2048, 384]" = torch.ops.aten.view.default(clone_33, [2048, 384]);  clone_33 = None
    permute_29: "f32[384, 192]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
    addmm_15: "f32[2048, 192]" = torch.ops.aten.addmm.default(primals_139, view_47, permute_29);  primals_139 = None
    view_48: "f32[32, 64, 192]" = torch.ops.aten.view.default(addmm_15, [32, 64, 192]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_34: "f32[32, 64, 192]" = torch.ops.aten.clone.default(view_48);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_142: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_139, clone_34);  clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_32 = torch.ops.aten.var_mean.correction(add_142, [2], correction = 0, keepdim = True)
    getitem_112: "f32[32, 64, 1]" = var_mean_32[0]
    getitem_113: "f32[32, 64, 1]" = var_mean_32[1];  var_mean_32 = None
    add_143: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
    rsqrt_32: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
    sub_32: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_142, getitem_113)
    mul_200: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    mul_201: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_200, primals_140);  mul_200 = None
    add_144: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_201, primals_141);  mul_201 = primals_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_49: "f32[2048, 192]" = torch.ops.aten.view.default(add_144, [2048, 192]);  add_144 = None
    permute_30: "f32[192, 576]" = torch.ops.aten.permute.default(primals_142, [1, 0]);  primals_142 = None
    addmm_16: "f32[2048, 576]" = torch.ops.aten.addmm.default(primals_143, view_49, permute_30);  primals_143 = None
    view_50: "f32[32, 64, 576]" = torch.ops.aten.view.default(addmm_16, [32, 64, 576]);  addmm_16 = None
    view_51: "f32[32, 64, 3, 4, 48]" = torch.ops.aten.view.default(view_50, [32, 64, 3, 4, 48]);  view_50 = None
    permute_31: "f32[3, 32, 4, 64, 48]" = torch.ops.aten.permute.default(view_51, [2, 0, 3, 1, 4]);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_4 = torch.ops.aten.unbind.int(permute_31);  permute_31 = None
    getitem_114: "f32[32, 4, 64, 48]" = unbind_4[0]
    getitem_115: "f32[32, 4, 64, 48]" = unbind_4[1]
    getitem_116: "f32[32, 4, 64, 48]" = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_4 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_114, getitem_115, getitem_116)
    getitem_117: "f32[32, 4, 64, 48]" = _scaled_dot_product_flash_attention_4[0]
    getitem_118: "f32[32, 4, 64]" = _scaled_dot_product_flash_attention_4[1]
    getitem_119: "i32[]" = _scaled_dot_product_flash_attention_4[2]
    getitem_120: "i32[]" = _scaled_dot_product_flash_attention_4[3]
    getitem_123: "i64[]" = _scaled_dot_product_flash_attention_4[6]
    getitem_124: "i64[]" = _scaled_dot_product_flash_attention_4[7];  _scaled_dot_product_flash_attention_4 = None
    alias_4: "f32[32, 4, 64, 48]" = torch.ops.aten.alias.default(getitem_117)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_32: "f32[32, 64, 4, 48]" = torch.ops.aten.permute.default(getitem_117, [0, 2, 1, 3]);  getitem_117 = None
    view_52: "f32[32, 64, 192]" = torch.ops.aten.view.default(permute_32, [32, 64, 192]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_53: "f32[2048, 192]" = torch.ops.aten.view.default(view_52, [2048, 192]);  view_52 = None
    permute_33: "f32[192, 192]" = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
    addmm_17: "f32[2048, 192]" = torch.ops.aten.addmm.default(primals_145, view_53, permute_33);  primals_145 = None
    view_54: "f32[32, 64, 192]" = torch.ops.aten.view.default(addmm_17, [32, 64, 192]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_35: "f32[32, 64, 192]" = torch.ops.aten.clone.default(view_54);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_145: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_142, clone_35);  clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_33 = torch.ops.aten.var_mean.correction(add_145, [2], correction = 0, keepdim = True)
    getitem_126: "f32[32, 64, 1]" = var_mean_33[0]
    getitem_127: "f32[32, 64, 1]" = var_mean_33[1];  var_mean_33 = None
    add_146: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-05);  getitem_126 = None
    rsqrt_33: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
    sub_33: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_145, getitem_127)
    mul_202: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    mul_203: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_202, primals_146);  mul_202 = None
    add_147: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_203, primals_147);  mul_203 = primals_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_55: "f32[2048, 192]" = torch.ops.aten.view.default(add_147, [2048, 192]);  add_147 = None
    permute_34: "f32[192, 384]" = torch.ops.aten.permute.default(primals_148, [1, 0]);  primals_148 = None
    addmm_18: "f32[2048, 384]" = torch.ops.aten.addmm.default(primals_149, view_55, permute_34);  primals_149 = None
    view_56: "f32[32, 64, 384]" = torch.ops.aten.view.default(addmm_18, [32, 64, 384]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_21: "f32[32, 64, 384]" = torch.ops.aten.sigmoid.default(view_56)
    mul_204: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(view_56, sigmoid_21);  sigmoid_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_36: "f32[32, 64, 384]" = torch.ops.aten.clone.default(mul_204);  mul_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_57: "f32[2048, 384]" = torch.ops.aten.view.default(clone_36, [2048, 384]);  clone_36 = None
    permute_35: "f32[384, 192]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
    addmm_19: "f32[2048, 192]" = torch.ops.aten.addmm.default(primals_151, view_57, permute_35);  primals_151 = None
    view_58: "f32[32, 64, 192]" = torch.ops.aten.view.default(addmm_19, [32, 64, 192]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_37: "f32[32, 64, 192]" = torch.ops.aten.clone.default(view_58);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_148: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_145, clone_37);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_34 = torch.ops.aten.var_mean.correction(add_148, [2], correction = 0, keepdim = True)
    getitem_128: "f32[32, 64, 1]" = var_mean_34[0]
    getitem_129: "f32[32, 64, 1]" = var_mean_34[1];  var_mean_34 = None
    add_149: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-05);  getitem_128 = None
    rsqrt_34: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
    sub_34: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_148, getitem_129)
    mul_205: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    mul_206: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_205, primals_152);  mul_205 = None
    add_150: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_206, primals_153);  mul_206 = primals_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_59: "f32[2048, 192]" = torch.ops.aten.view.default(add_150, [2048, 192]);  add_150 = None
    permute_36: "f32[192, 576]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
    addmm_20: "f32[2048, 576]" = torch.ops.aten.addmm.default(primals_155, view_59, permute_36);  primals_155 = None
    view_60: "f32[32, 64, 576]" = torch.ops.aten.view.default(addmm_20, [32, 64, 576]);  addmm_20 = None
    view_61: "f32[32, 64, 3, 4, 48]" = torch.ops.aten.view.default(view_60, [32, 64, 3, 4, 48]);  view_60 = None
    permute_37: "f32[3, 32, 4, 64, 48]" = torch.ops.aten.permute.default(view_61, [2, 0, 3, 1, 4]);  view_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_5 = torch.ops.aten.unbind.int(permute_37);  permute_37 = None
    getitem_130: "f32[32, 4, 64, 48]" = unbind_5[0]
    getitem_131: "f32[32, 4, 64, 48]" = unbind_5[1]
    getitem_132: "f32[32, 4, 64, 48]" = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_5 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_130, getitem_131, getitem_132)
    getitem_133: "f32[32, 4, 64, 48]" = _scaled_dot_product_flash_attention_5[0]
    getitem_134: "f32[32, 4, 64]" = _scaled_dot_product_flash_attention_5[1]
    getitem_135: "i32[]" = _scaled_dot_product_flash_attention_5[2]
    getitem_136: "i32[]" = _scaled_dot_product_flash_attention_5[3]
    getitem_139: "i64[]" = _scaled_dot_product_flash_attention_5[6]
    getitem_140: "i64[]" = _scaled_dot_product_flash_attention_5[7];  _scaled_dot_product_flash_attention_5 = None
    alias_5: "f32[32, 4, 64, 48]" = torch.ops.aten.alias.default(getitem_133)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_38: "f32[32, 64, 4, 48]" = torch.ops.aten.permute.default(getitem_133, [0, 2, 1, 3]);  getitem_133 = None
    view_62: "f32[32, 64, 192]" = torch.ops.aten.view.default(permute_38, [32, 64, 192]);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_63: "f32[2048, 192]" = torch.ops.aten.view.default(view_62, [2048, 192]);  view_62 = None
    permute_39: "f32[192, 192]" = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
    addmm_21: "f32[2048, 192]" = torch.ops.aten.addmm.default(primals_157, view_63, permute_39);  primals_157 = None
    view_64: "f32[32, 64, 192]" = torch.ops.aten.view.default(addmm_21, [32, 64, 192]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_38: "f32[32, 64, 192]" = torch.ops.aten.clone.default(view_64);  view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_151: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_148, clone_38);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_35 = torch.ops.aten.var_mean.correction(add_151, [2], correction = 0, keepdim = True)
    getitem_142: "f32[32, 64, 1]" = var_mean_35[0]
    getitem_143: "f32[32, 64, 1]" = var_mean_35[1];  var_mean_35 = None
    add_152: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-05);  getitem_142 = None
    rsqrt_35: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
    sub_35: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_151, getitem_143)
    mul_207: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    mul_208: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_207, primals_158);  mul_207 = None
    add_153: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_208, primals_159);  mul_208 = primals_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_65: "f32[2048, 192]" = torch.ops.aten.view.default(add_153, [2048, 192]);  add_153 = None
    permute_40: "f32[192, 384]" = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
    addmm_22: "f32[2048, 384]" = torch.ops.aten.addmm.default(primals_161, view_65, permute_40);  primals_161 = None
    view_66: "f32[32, 64, 384]" = torch.ops.aten.view.default(addmm_22, [32, 64, 384]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_22: "f32[32, 64, 384]" = torch.ops.aten.sigmoid.default(view_66)
    mul_209: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(view_66, sigmoid_22);  sigmoid_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_39: "f32[32, 64, 384]" = torch.ops.aten.clone.default(mul_209);  mul_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_67: "f32[2048, 384]" = torch.ops.aten.view.default(clone_39, [2048, 384]);  clone_39 = None
    permute_41: "f32[384, 192]" = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
    addmm_23: "f32[2048, 192]" = torch.ops.aten.addmm.default(primals_163, view_67, permute_41);  primals_163 = None
    view_68: "f32[32, 64, 192]" = torch.ops.aten.view.default(addmm_23, [32, 64, 192]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_40: "f32[32, 64, 192]" = torch.ops.aten.clone.default(view_68);  view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_154: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_151, clone_40);  clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:255, code: x = self.norm(x)
    var_mean_36 = torch.ops.aten.var_mean.correction(add_154, [2], correction = 0, keepdim = True)
    getitem_144: "f32[32, 64, 1]" = var_mean_36[0]
    getitem_145: "f32[32, 64, 1]" = var_mean_36[1];  var_mean_36 = None
    add_155: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-05);  getitem_144 = None
    rsqrt_36: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
    sub_36: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_154, getitem_145)
    mul_210: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    mul_211: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_210, primals_164);  mul_210 = None
    add_156: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_211, primals_165);  mul_211 = primals_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:259, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
    view_69: "f32[8, 4, 64, 192]" = torch.ops.aten.view.default(add_156, [8, 4, 64, -1]);  add_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:260, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
    permute_42: "f32[8, 192, 64, 4]" = torch.ops.aten.permute.default(view_69, [0, 3, 2, 1]);  view_69 = None
    clone_41: "f32[8, 192, 64, 4]" = torch.ops.aten.clone.default(permute_42, memory_format = torch.contiguous_format);  permute_42 = None
    view_70: "f32[12288, 8, 2, 2]" = torch.ops.aten.view.default(clone_41, [12288, 8, 2, 2]);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:262, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
    permute_43: "f32[12288, 2, 8, 2]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
    clone_42: "f32[12288, 2, 8, 2]" = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None
    view_71: "f32[8, 192, 16, 16]" = torch.ops.aten.view.default(clone_42, [8, 192, 16, 16]);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_25: "f32[8, 128, 16, 16]" = torch.ops.aten.convolution.default(view_71, primals_166, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_157: "i64[]" = torch.ops.aten.add.Tensor(primals_285, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_37 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
    getitem_146: "f32[1, 128, 1, 1]" = var_mean_37[0]
    getitem_147: "f32[1, 128, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_158: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_146, 1e-05)
    rsqrt_37: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
    sub_37: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_147)
    mul_212: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    squeeze_69: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_147, [0, 2, 3]);  getitem_147 = None
    squeeze_70: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
    mul_213: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
    mul_214: "f32[128]" = torch.ops.aten.mul.Tensor(primals_286, 0.9)
    add_159: "f32[128]" = torch.ops.aten.add.Tensor(mul_213, mul_214);  mul_213 = mul_214 = None
    squeeze_71: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_146, [0, 2, 3]);  getitem_146 = None
    mul_215: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0004885197850513);  squeeze_71 = None
    mul_216: "f32[128]" = torch.ops.aten.mul.Tensor(mul_215, 0.1);  mul_215 = None
    mul_217: "f32[128]" = torch.ops.aten.mul.Tensor(primals_287, 0.9)
    add_160: "f32[128]" = torch.ops.aten.add.Tensor(mul_216, mul_217);  mul_216 = mul_217 = None
    unsqueeze_92: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_93: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_218: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(mul_212, unsqueeze_93);  mul_212 = unsqueeze_93 = None
    unsqueeze_94: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_95: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_161: "f32[8, 128, 16, 16]" = torch.ops.aten.add.Tensor(mul_218, unsqueeze_95);  mul_218 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_43: "f32[8, 128, 16, 16]" = torch.ops.aten.clone.default(add_161)
    sigmoid_23: "f32[8, 128, 16, 16]" = torch.ops.aten.sigmoid.default(add_161)
    mul_219: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(add_161, sigmoid_23);  add_161 = sigmoid_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:268, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
    cat_1: "f32[8, 256, 16, 16]" = torch.ops.aten.cat.default([add_125, mul_219], 1);  mul_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_26: "f32[8, 128, 16, 16]" = torch.ops.aten.convolution.default(cat_1, primals_167, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_162: "i64[]" = torch.ops.aten.add.Tensor(primals_288, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_38 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
    getitem_148: "f32[1, 128, 1, 1]" = var_mean_38[0]
    getitem_149: "f32[1, 128, 1, 1]" = var_mean_38[1];  var_mean_38 = None
    add_163: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_148, 1e-05)
    rsqrt_38: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
    sub_38: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_149)
    mul_220: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    squeeze_72: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_149, [0, 2, 3]);  getitem_149 = None
    squeeze_73: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
    mul_221: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_222: "f32[128]" = torch.ops.aten.mul.Tensor(primals_289, 0.9)
    add_164: "f32[128]" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
    squeeze_74: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_148, [0, 2, 3]);  getitem_148 = None
    mul_223: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0004885197850513);  squeeze_74 = None
    mul_224: "f32[128]" = torch.ops.aten.mul.Tensor(mul_223, 0.1);  mul_223 = None
    mul_225: "f32[128]" = torch.ops.aten.mul.Tensor(primals_290, 0.9)
    add_165: "f32[128]" = torch.ops.aten.add.Tensor(mul_224, mul_225);  mul_224 = mul_225 = None
    unsqueeze_96: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1)
    unsqueeze_97: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    mul_226: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(mul_220, unsqueeze_97);  mul_220 = unsqueeze_97 = None
    unsqueeze_98: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1);  primals_50 = None
    unsqueeze_99: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    add_166: "f32[8, 128, 16, 16]" = torch.ops.aten.add.Tensor(mul_226, unsqueeze_99);  mul_226 = unsqueeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_44: "f32[8, 128, 16, 16]" = torch.ops.aten.clone.default(add_166)
    sigmoid_24: "f32[8, 128, 16, 16]" = torch.ops.aten.sigmoid.default(add_166)
    mul_227: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(add_166, sigmoid_24);  add_166 = sigmoid_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_27: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(mul_227, primals_168, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_167: "i64[]" = torch.ops.aten.add.Tensor(primals_291, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_39 = torch.ops.aten.var_mean.correction(convolution_27, [0, 2, 3], correction = 0, keepdim = True)
    getitem_150: "f32[1, 512, 1, 1]" = var_mean_39[0]
    getitem_151: "f32[1, 512, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_168: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_150, 1e-05)
    rsqrt_39: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    sub_39: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_27, getitem_151)
    mul_228: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    squeeze_75: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_151, [0, 2, 3]);  getitem_151 = None
    squeeze_76: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
    mul_229: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_230: "f32[512]" = torch.ops.aten.mul.Tensor(primals_292, 0.9)
    add_169: "f32[512]" = torch.ops.aten.add.Tensor(mul_229, mul_230);  mul_229 = mul_230 = None
    squeeze_77: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_150, [0, 2, 3]);  getitem_150 = None
    mul_231: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0004885197850513);  squeeze_77 = None
    mul_232: "f32[512]" = torch.ops.aten.mul.Tensor(mul_231, 0.1);  mul_231 = None
    mul_233: "f32[512]" = torch.ops.aten.mul.Tensor(primals_293, 0.9)
    add_170: "f32[512]" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
    unsqueeze_100: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1)
    unsqueeze_101: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_234: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_228, unsqueeze_101);  mul_228 = unsqueeze_101 = None
    unsqueeze_102: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1);  primals_52 = None
    unsqueeze_103: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_171: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_234, unsqueeze_103);  mul_234 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_45: "f32[8, 512, 16, 16]" = torch.ops.aten.clone.default(add_171)
    sigmoid_25: "f32[8, 512, 16, 16]" = torch.ops.aten.sigmoid.default(add_171)
    mul_235: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(add_171, sigmoid_25);  add_171 = sigmoid_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_28: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(mul_235, primals_169, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 512)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_172: "i64[]" = torch.ops.aten.add.Tensor(primals_294, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_40 = torch.ops.aten.var_mean.correction(convolution_28, [0, 2, 3], correction = 0, keepdim = True)
    getitem_152: "f32[1, 512, 1, 1]" = var_mean_40[0]
    getitem_153: "f32[1, 512, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_173: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_152, 1e-05)
    rsqrt_40: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_173);  add_173 = None
    sub_40: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_28, getitem_153)
    mul_236: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    squeeze_78: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_153, [0, 2, 3]);  getitem_153 = None
    squeeze_79: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
    mul_237: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_238: "f32[512]" = torch.ops.aten.mul.Tensor(primals_295, 0.9)
    add_174: "f32[512]" = torch.ops.aten.add.Tensor(mul_237, mul_238);  mul_237 = mul_238 = None
    squeeze_80: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_152, [0, 2, 3]);  getitem_152 = None
    mul_239: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0019569471624266);  squeeze_80 = None
    mul_240: "f32[512]" = torch.ops.aten.mul.Tensor(mul_239, 0.1);  mul_239 = None
    mul_241: "f32[512]" = torch.ops.aten.mul.Tensor(primals_296, 0.9)
    add_175: "f32[512]" = torch.ops.aten.add.Tensor(mul_240, mul_241);  mul_240 = mul_241 = None
    unsqueeze_104: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_105: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    mul_242: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_236, unsqueeze_105);  mul_236 = unsqueeze_105 = None
    unsqueeze_106: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_107: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    add_176: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_242, unsqueeze_107);  mul_242 = unsqueeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_46: "f32[8, 512, 8, 8]" = torch.ops.aten.clone.default(add_176)
    sigmoid_26: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(add_176)
    mul_243: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_176, sigmoid_26);  add_176 = sigmoid_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_29: "f32[8, 160, 8, 8]" = torch.ops.aten.convolution.default(mul_243, primals_170, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_177: "i64[]" = torch.ops.aten.add.Tensor(primals_297, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_41 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
    getitem_154: "f32[1, 160, 1, 1]" = var_mean_41[0]
    getitem_155: "f32[1, 160, 1, 1]" = var_mean_41[1];  var_mean_41 = None
    add_178: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_154, 1e-05)
    rsqrt_41: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
    sub_41: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_29, getitem_155)
    mul_244: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
    squeeze_81: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_155, [0, 2, 3]);  getitem_155 = None
    squeeze_82: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
    mul_245: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
    mul_246: "f32[160]" = torch.ops.aten.mul.Tensor(primals_298, 0.9)
    add_179: "f32[160]" = torch.ops.aten.add.Tensor(mul_245, mul_246);  mul_245 = mul_246 = None
    squeeze_83: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_154, [0, 2, 3]);  getitem_154 = None
    mul_247: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0019569471624266);  squeeze_83 = None
    mul_248: "f32[160]" = torch.ops.aten.mul.Tensor(mul_247, 0.1);  mul_247 = None
    mul_249: "f32[160]" = torch.ops.aten.mul.Tensor(primals_299, 0.9)
    add_180: "f32[160]" = torch.ops.aten.add.Tensor(mul_248, mul_249);  mul_248 = mul_249 = None
    unsqueeze_108: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1)
    unsqueeze_109: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_250: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(mul_244, unsqueeze_109);  mul_244 = unsqueeze_109 = None
    unsqueeze_110: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1);  primals_56 = None
    unsqueeze_111: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_181: "f32[8, 160, 8, 8]" = torch.ops.aten.add.Tensor(mul_250, unsqueeze_111);  mul_250 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_30: "f32[8, 160, 8, 8]" = torch.ops.aten.convolution.default(add_181, primals_171, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_182: "i64[]" = torch.ops.aten.add.Tensor(primals_300, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_42 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
    getitem_156: "f32[1, 160, 1, 1]" = var_mean_42[0]
    getitem_157: "f32[1, 160, 1, 1]" = var_mean_42[1];  var_mean_42 = None
    add_183: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-05)
    rsqrt_42: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_183);  add_183 = None
    sub_42: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_30, getitem_157)
    mul_251: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
    squeeze_84: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_157, [0, 2, 3]);  getitem_157 = None
    squeeze_85: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
    mul_252: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
    mul_253: "f32[160]" = torch.ops.aten.mul.Tensor(primals_301, 0.9)
    add_184: "f32[160]" = torch.ops.aten.add.Tensor(mul_252, mul_253);  mul_252 = mul_253 = None
    squeeze_86: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_156, [0, 2, 3]);  getitem_156 = None
    mul_254: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0019569471624266);  squeeze_86 = None
    mul_255: "f32[160]" = torch.ops.aten.mul.Tensor(mul_254, 0.1);  mul_254 = None
    mul_256: "f32[160]" = torch.ops.aten.mul.Tensor(primals_302, 0.9)
    add_185: "f32[160]" = torch.ops.aten.add.Tensor(mul_255, mul_256);  mul_255 = mul_256 = None
    unsqueeze_112: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1)
    unsqueeze_113: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    mul_257: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(mul_251, unsqueeze_113);  mul_251 = unsqueeze_113 = None
    unsqueeze_114: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_58, -1);  primals_58 = None
    unsqueeze_115: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    add_186: "f32[8, 160, 8, 8]" = torch.ops.aten.add.Tensor(mul_257, unsqueeze_115);  mul_257 = unsqueeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_47: "f32[8, 160, 8, 8]" = torch.ops.aten.clone.default(add_186)
    sigmoid_27: "f32[8, 160, 8, 8]" = torch.ops.aten.sigmoid.default(add_186)
    mul_258: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(add_186, sigmoid_27);  add_186 = sigmoid_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:234, code: x = self.conv_1x1(x)
    convolution_31: "f32[8, 240, 8, 8]" = torch.ops.aten.convolution.default(mul_258, primals_172, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:249, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
    view_72: "f32[7680, 2, 4, 2]" = torch.ops.aten.view.default(convolution_31, [7680, 2, 4, 2]);  convolution_31 = None
    permute_44: "f32[7680, 4, 2, 2]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:251, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
    clone_48: "f32[7680, 4, 2, 2]" = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
    view_73: "f32[8, 240, 16, 4]" = torch.ops.aten.view.default(clone_48, [8, 240, 16, 4]);  clone_48 = None
    permute_45: "f32[8, 4, 16, 240]" = torch.ops.aten.permute.default(view_73, [0, 3, 2, 1]);  view_73 = None
    clone_49: "f32[8, 4, 16, 240]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
    view_74: "f32[32, 16, 240]" = torch.ops.aten.view.default(clone_49, [32, 16, 240]);  clone_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_43 = torch.ops.aten.var_mean.correction(view_74, [2], correction = 0, keepdim = True)
    getitem_158: "f32[32, 16, 1]" = var_mean_43[0]
    getitem_159: "f32[32, 16, 1]" = var_mean_43[1];  var_mean_43 = None
    add_187: "f32[32, 16, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-05);  getitem_158 = None
    rsqrt_43: "f32[32, 16, 1]" = torch.ops.aten.rsqrt.default(add_187);  add_187 = None
    sub_43: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(view_74, getitem_159)
    mul_259: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
    mul_260: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_259, primals_173);  mul_259 = None
    add_188: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(mul_260, primals_174);  mul_260 = primals_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_75: "f32[512, 240]" = torch.ops.aten.view.default(add_188, [512, 240]);  add_188 = None
    permute_46: "f32[240, 720]" = torch.ops.aten.permute.default(primals_175, [1, 0]);  primals_175 = None
    addmm_24: "f32[512, 720]" = torch.ops.aten.addmm.default(primals_176, view_75, permute_46);  primals_176 = None
    view_76: "f32[32, 16, 720]" = torch.ops.aten.view.default(addmm_24, [32, 16, 720]);  addmm_24 = None
    view_77: "f32[32, 16, 3, 4, 60]" = torch.ops.aten.view.default(view_76, [32, 16, 3, 4, 60]);  view_76 = None
    permute_47: "f32[3, 32, 4, 16, 60]" = torch.ops.aten.permute.default(view_77, [2, 0, 3, 1, 4]);  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_6 = torch.ops.aten.unbind.int(permute_47);  permute_47 = None
    getitem_160: "f32[32, 4, 16, 60]" = unbind_6[0]
    getitem_161: "f32[32, 4, 16, 60]" = unbind_6[1]
    getitem_162: "f32[32, 4, 16, 60]" = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_6 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_160, getitem_161, getitem_162)
    getitem_163: "f32[32, 4, 16, 60]" = _scaled_dot_product_flash_attention_6[0]
    getitem_164: "f32[32, 4, 16]" = _scaled_dot_product_flash_attention_6[1]
    getitem_165: "i32[]" = _scaled_dot_product_flash_attention_6[2]
    getitem_166: "i32[]" = _scaled_dot_product_flash_attention_6[3]
    getitem_169: "i64[]" = _scaled_dot_product_flash_attention_6[6]
    getitem_170: "i64[]" = _scaled_dot_product_flash_attention_6[7];  _scaled_dot_product_flash_attention_6 = None
    alias_6: "f32[32, 4, 16, 60]" = torch.ops.aten.alias.default(getitem_163)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_48: "f32[32, 16, 4, 60]" = torch.ops.aten.permute.default(getitem_163, [0, 2, 1, 3]);  getitem_163 = None
    view_78: "f32[32, 16, 240]" = torch.ops.aten.view.default(permute_48, [32, 16, 240]);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_79: "f32[512, 240]" = torch.ops.aten.view.default(view_78, [512, 240]);  view_78 = None
    permute_49: "f32[240, 240]" = torch.ops.aten.permute.default(primals_177, [1, 0]);  primals_177 = None
    addmm_25: "f32[512, 240]" = torch.ops.aten.addmm.default(primals_178, view_79, permute_49);  primals_178 = None
    view_80: "f32[32, 16, 240]" = torch.ops.aten.view.default(addmm_25, [32, 16, 240]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_50: "f32[32, 16, 240]" = torch.ops.aten.clone.default(view_80);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_189: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(view_74, clone_50);  clone_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_44 = torch.ops.aten.var_mean.correction(add_189, [2], correction = 0, keepdim = True)
    getitem_172: "f32[32, 16, 1]" = var_mean_44[0]
    getitem_173: "f32[32, 16, 1]" = var_mean_44[1];  var_mean_44 = None
    add_190: "f32[32, 16, 1]" = torch.ops.aten.add.Tensor(getitem_172, 1e-05);  getitem_172 = None
    rsqrt_44: "f32[32, 16, 1]" = torch.ops.aten.rsqrt.default(add_190);  add_190 = None
    sub_44: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(add_189, getitem_173)
    mul_261: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
    mul_262: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_261, primals_179);  mul_261 = None
    add_191: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(mul_262, primals_180);  mul_262 = primals_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_81: "f32[512, 240]" = torch.ops.aten.view.default(add_191, [512, 240]);  add_191 = None
    permute_50: "f32[240, 480]" = torch.ops.aten.permute.default(primals_181, [1, 0]);  primals_181 = None
    addmm_26: "f32[512, 480]" = torch.ops.aten.addmm.default(primals_182, view_81, permute_50);  primals_182 = None
    view_82: "f32[32, 16, 480]" = torch.ops.aten.view.default(addmm_26, [32, 16, 480]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_28: "f32[32, 16, 480]" = torch.ops.aten.sigmoid.default(view_82)
    mul_263: "f32[32, 16, 480]" = torch.ops.aten.mul.Tensor(view_82, sigmoid_28);  sigmoid_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_51: "f32[32, 16, 480]" = torch.ops.aten.clone.default(mul_263);  mul_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_83: "f32[512, 480]" = torch.ops.aten.view.default(clone_51, [512, 480]);  clone_51 = None
    permute_51: "f32[480, 240]" = torch.ops.aten.permute.default(primals_183, [1, 0]);  primals_183 = None
    addmm_27: "f32[512, 240]" = torch.ops.aten.addmm.default(primals_184, view_83, permute_51);  primals_184 = None
    view_84: "f32[32, 16, 240]" = torch.ops.aten.view.default(addmm_27, [32, 16, 240]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_52: "f32[32, 16, 240]" = torch.ops.aten.clone.default(view_84);  view_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_192: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(add_189, clone_52);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_45 = torch.ops.aten.var_mean.correction(add_192, [2], correction = 0, keepdim = True)
    getitem_174: "f32[32, 16, 1]" = var_mean_45[0]
    getitem_175: "f32[32, 16, 1]" = var_mean_45[1];  var_mean_45 = None
    add_193: "f32[32, 16, 1]" = torch.ops.aten.add.Tensor(getitem_174, 1e-05);  getitem_174 = None
    rsqrt_45: "f32[32, 16, 1]" = torch.ops.aten.rsqrt.default(add_193);  add_193 = None
    sub_45: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(add_192, getitem_175)
    mul_264: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
    mul_265: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_264, primals_185);  mul_264 = None
    add_194: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(mul_265, primals_186);  mul_265 = primals_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_85: "f32[512, 240]" = torch.ops.aten.view.default(add_194, [512, 240]);  add_194 = None
    permute_52: "f32[240, 720]" = torch.ops.aten.permute.default(primals_187, [1, 0]);  primals_187 = None
    addmm_28: "f32[512, 720]" = torch.ops.aten.addmm.default(primals_188, view_85, permute_52);  primals_188 = None
    view_86: "f32[32, 16, 720]" = torch.ops.aten.view.default(addmm_28, [32, 16, 720]);  addmm_28 = None
    view_87: "f32[32, 16, 3, 4, 60]" = torch.ops.aten.view.default(view_86, [32, 16, 3, 4, 60]);  view_86 = None
    permute_53: "f32[3, 32, 4, 16, 60]" = torch.ops.aten.permute.default(view_87, [2, 0, 3, 1, 4]);  view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_7 = torch.ops.aten.unbind.int(permute_53);  permute_53 = None
    getitem_176: "f32[32, 4, 16, 60]" = unbind_7[0]
    getitem_177: "f32[32, 4, 16, 60]" = unbind_7[1]
    getitem_178: "f32[32, 4, 16, 60]" = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_7 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_176, getitem_177, getitem_178)
    getitem_179: "f32[32, 4, 16, 60]" = _scaled_dot_product_flash_attention_7[0]
    getitem_180: "f32[32, 4, 16]" = _scaled_dot_product_flash_attention_7[1]
    getitem_181: "i32[]" = _scaled_dot_product_flash_attention_7[2]
    getitem_182: "i32[]" = _scaled_dot_product_flash_attention_7[3]
    getitem_185: "i64[]" = _scaled_dot_product_flash_attention_7[6]
    getitem_186: "i64[]" = _scaled_dot_product_flash_attention_7[7];  _scaled_dot_product_flash_attention_7 = None
    alias_7: "f32[32, 4, 16, 60]" = torch.ops.aten.alias.default(getitem_179)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_54: "f32[32, 16, 4, 60]" = torch.ops.aten.permute.default(getitem_179, [0, 2, 1, 3]);  getitem_179 = None
    view_88: "f32[32, 16, 240]" = torch.ops.aten.view.default(permute_54, [32, 16, 240]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_89: "f32[512, 240]" = torch.ops.aten.view.default(view_88, [512, 240]);  view_88 = None
    permute_55: "f32[240, 240]" = torch.ops.aten.permute.default(primals_189, [1, 0]);  primals_189 = None
    addmm_29: "f32[512, 240]" = torch.ops.aten.addmm.default(primals_190, view_89, permute_55);  primals_190 = None
    view_90: "f32[32, 16, 240]" = torch.ops.aten.view.default(addmm_29, [32, 16, 240]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_53: "f32[32, 16, 240]" = torch.ops.aten.clone.default(view_90);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_195: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(add_192, clone_53);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_46 = torch.ops.aten.var_mean.correction(add_195, [2], correction = 0, keepdim = True)
    getitem_188: "f32[32, 16, 1]" = var_mean_46[0]
    getitem_189: "f32[32, 16, 1]" = var_mean_46[1];  var_mean_46 = None
    add_196: "f32[32, 16, 1]" = torch.ops.aten.add.Tensor(getitem_188, 1e-05);  getitem_188 = None
    rsqrt_46: "f32[32, 16, 1]" = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
    sub_46: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(add_195, getitem_189)
    mul_266: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
    mul_267: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_266, primals_191);  mul_266 = None
    add_197: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(mul_267, primals_192);  mul_267 = primals_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_91: "f32[512, 240]" = torch.ops.aten.view.default(add_197, [512, 240]);  add_197 = None
    permute_56: "f32[240, 480]" = torch.ops.aten.permute.default(primals_193, [1, 0]);  primals_193 = None
    addmm_30: "f32[512, 480]" = torch.ops.aten.addmm.default(primals_194, view_91, permute_56);  primals_194 = None
    view_92: "f32[32, 16, 480]" = torch.ops.aten.view.default(addmm_30, [32, 16, 480]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_29: "f32[32, 16, 480]" = torch.ops.aten.sigmoid.default(view_92)
    mul_268: "f32[32, 16, 480]" = torch.ops.aten.mul.Tensor(view_92, sigmoid_29);  sigmoid_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_54: "f32[32, 16, 480]" = torch.ops.aten.clone.default(mul_268);  mul_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_93: "f32[512, 480]" = torch.ops.aten.view.default(clone_54, [512, 480]);  clone_54 = None
    permute_57: "f32[480, 240]" = torch.ops.aten.permute.default(primals_195, [1, 0]);  primals_195 = None
    addmm_31: "f32[512, 240]" = torch.ops.aten.addmm.default(primals_196, view_93, permute_57);  primals_196 = None
    view_94: "f32[32, 16, 240]" = torch.ops.aten.view.default(addmm_31, [32, 16, 240]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_55: "f32[32, 16, 240]" = torch.ops.aten.clone.default(view_94);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_198: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(add_195, clone_55);  clone_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_47 = torch.ops.aten.var_mean.correction(add_198, [2], correction = 0, keepdim = True)
    getitem_190: "f32[32, 16, 1]" = var_mean_47[0]
    getitem_191: "f32[32, 16, 1]" = var_mean_47[1];  var_mean_47 = None
    add_199: "f32[32, 16, 1]" = torch.ops.aten.add.Tensor(getitem_190, 1e-05);  getitem_190 = None
    rsqrt_47: "f32[32, 16, 1]" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
    sub_47: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(add_198, getitem_191)
    mul_269: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
    mul_270: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_269, primals_197);  mul_269 = None
    add_200: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(mul_270, primals_198);  mul_270 = primals_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_95: "f32[512, 240]" = torch.ops.aten.view.default(add_200, [512, 240]);  add_200 = None
    permute_58: "f32[240, 720]" = torch.ops.aten.permute.default(primals_199, [1, 0]);  primals_199 = None
    addmm_32: "f32[512, 720]" = torch.ops.aten.addmm.default(primals_200, view_95, permute_58);  primals_200 = None
    view_96: "f32[32, 16, 720]" = torch.ops.aten.view.default(addmm_32, [32, 16, 720]);  addmm_32 = None
    view_97: "f32[32, 16, 3, 4, 60]" = torch.ops.aten.view.default(view_96, [32, 16, 3, 4, 60]);  view_96 = None
    permute_59: "f32[3, 32, 4, 16, 60]" = torch.ops.aten.permute.default(view_97, [2, 0, 3, 1, 4]);  view_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_8 = torch.ops.aten.unbind.int(permute_59);  permute_59 = None
    getitem_192: "f32[32, 4, 16, 60]" = unbind_8[0]
    getitem_193: "f32[32, 4, 16, 60]" = unbind_8[1]
    getitem_194: "f32[32, 4, 16, 60]" = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_8 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_192, getitem_193, getitem_194)
    getitem_195: "f32[32, 4, 16, 60]" = _scaled_dot_product_flash_attention_8[0]
    getitem_196: "f32[32, 4, 16]" = _scaled_dot_product_flash_attention_8[1]
    getitem_197: "i32[]" = _scaled_dot_product_flash_attention_8[2]
    getitem_198: "i32[]" = _scaled_dot_product_flash_attention_8[3]
    getitem_201: "i64[]" = _scaled_dot_product_flash_attention_8[6]
    getitem_202: "i64[]" = _scaled_dot_product_flash_attention_8[7];  _scaled_dot_product_flash_attention_8 = None
    alias_8: "f32[32, 4, 16, 60]" = torch.ops.aten.alias.default(getitem_195)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_60: "f32[32, 16, 4, 60]" = torch.ops.aten.permute.default(getitem_195, [0, 2, 1, 3]);  getitem_195 = None
    view_98: "f32[32, 16, 240]" = torch.ops.aten.view.default(permute_60, [32, 16, 240]);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_99: "f32[512, 240]" = torch.ops.aten.view.default(view_98, [512, 240]);  view_98 = None
    permute_61: "f32[240, 240]" = torch.ops.aten.permute.default(primals_201, [1, 0]);  primals_201 = None
    addmm_33: "f32[512, 240]" = torch.ops.aten.addmm.default(primals_202, view_99, permute_61);  primals_202 = None
    view_100: "f32[32, 16, 240]" = torch.ops.aten.view.default(addmm_33, [32, 16, 240]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_56: "f32[32, 16, 240]" = torch.ops.aten.clone.default(view_100);  view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_201: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(add_198, clone_56);  clone_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_48 = torch.ops.aten.var_mean.correction(add_201, [2], correction = 0, keepdim = True)
    getitem_204: "f32[32, 16, 1]" = var_mean_48[0]
    getitem_205: "f32[32, 16, 1]" = var_mean_48[1];  var_mean_48 = None
    add_202: "f32[32, 16, 1]" = torch.ops.aten.add.Tensor(getitem_204, 1e-05);  getitem_204 = None
    rsqrt_48: "f32[32, 16, 1]" = torch.ops.aten.rsqrt.default(add_202);  add_202 = None
    sub_48: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(add_201, getitem_205)
    mul_271: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
    mul_272: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_271, primals_203);  mul_271 = None
    add_203: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(mul_272, primals_204);  mul_272 = primals_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_101: "f32[512, 240]" = torch.ops.aten.view.default(add_203, [512, 240]);  add_203 = None
    permute_62: "f32[240, 480]" = torch.ops.aten.permute.default(primals_205, [1, 0]);  primals_205 = None
    addmm_34: "f32[512, 480]" = torch.ops.aten.addmm.default(primals_206, view_101, permute_62);  primals_206 = None
    view_102: "f32[32, 16, 480]" = torch.ops.aten.view.default(addmm_34, [32, 16, 480]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_30: "f32[32, 16, 480]" = torch.ops.aten.sigmoid.default(view_102)
    mul_273: "f32[32, 16, 480]" = torch.ops.aten.mul.Tensor(view_102, sigmoid_30);  sigmoid_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_57: "f32[32, 16, 480]" = torch.ops.aten.clone.default(mul_273);  mul_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_103: "f32[512, 480]" = torch.ops.aten.view.default(clone_57, [512, 480]);  clone_57 = None
    permute_63: "f32[480, 240]" = torch.ops.aten.permute.default(primals_207, [1, 0]);  primals_207 = None
    addmm_35: "f32[512, 240]" = torch.ops.aten.addmm.default(primals_208, view_103, permute_63);  primals_208 = None
    view_104: "f32[32, 16, 240]" = torch.ops.aten.view.default(addmm_35, [32, 16, 240]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_58: "f32[32, 16, 240]" = torch.ops.aten.clone.default(view_104);  view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_204: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(add_201, clone_58);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:255, code: x = self.norm(x)
    var_mean_49 = torch.ops.aten.var_mean.correction(add_204, [2], correction = 0, keepdim = True)
    getitem_206: "f32[32, 16, 1]" = var_mean_49[0]
    getitem_207: "f32[32, 16, 1]" = var_mean_49[1];  var_mean_49 = None
    add_205: "f32[32, 16, 1]" = torch.ops.aten.add.Tensor(getitem_206, 1e-05);  getitem_206 = None
    rsqrt_49: "f32[32, 16, 1]" = torch.ops.aten.rsqrt.default(add_205);  add_205 = None
    sub_49: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(add_204, getitem_207)
    mul_274: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
    mul_275: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_274, primals_209);  mul_274 = None
    add_206: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(mul_275, primals_210);  mul_275 = primals_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:259, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
    view_105: "f32[8, 4, 16, 240]" = torch.ops.aten.view.default(add_206, [8, 4, 16, -1]);  add_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:260, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
    permute_64: "f32[8, 240, 16, 4]" = torch.ops.aten.permute.default(view_105, [0, 3, 2, 1]);  view_105 = None
    clone_59: "f32[8, 240, 16, 4]" = torch.ops.aten.clone.default(permute_64, memory_format = torch.contiguous_format);  permute_64 = None
    view_106: "f32[7680, 4, 2, 2]" = torch.ops.aten.view.default(clone_59, [7680, 4, 2, 2]);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:262, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
    permute_65: "f32[7680, 2, 4, 2]" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
    clone_60: "f32[7680, 2, 4, 2]" = torch.ops.aten.clone.default(permute_65, memory_format = torch.contiguous_format);  permute_65 = None
    view_107: "f32[8, 240, 8, 8]" = torch.ops.aten.view.default(clone_60, [8, 240, 8, 8]);  clone_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_32: "f32[8, 160, 8, 8]" = torch.ops.aten.convolution.default(view_107, primals_211, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_207: "i64[]" = torch.ops.aten.add.Tensor(primals_303, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_50 = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
    getitem_208: "f32[1, 160, 1, 1]" = var_mean_50[0]
    getitem_209: "f32[1, 160, 1, 1]" = var_mean_50[1];  var_mean_50 = None
    add_208: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_208, 1e-05)
    rsqrt_50: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_208);  add_208 = None
    sub_50: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_32, getitem_209)
    mul_276: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
    squeeze_87: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_209, [0, 2, 3]);  getitem_209 = None
    squeeze_88: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
    mul_277: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
    mul_278: "f32[160]" = torch.ops.aten.mul.Tensor(primals_304, 0.9)
    add_209: "f32[160]" = torch.ops.aten.add.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
    squeeze_89: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_208, [0, 2, 3]);  getitem_208 = None
    mul_279: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0019569471624266);  squeeze_89 = None
    mul_280: "f32[160]" = torch.ops.aten.mul.Tensor(mul_279, 0.1);  mul_279 = None
    mul_281: "f32[160]" = torch.ops.aten.mul.Tensor(primals_305, 0.9)
    add_210: "f32[160]" = torch.ops.aten.add.Tensor(mul_280, mul_281);  mul_280 = mul_281 = None
    unsqueeze_116: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_117: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_282: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(mul_276, unsqueeze_117);  mul_276 = unsqueeze_117 = None
    unsqueeze_118: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_119: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_211: "f32[8, 160, 8, 8]" = torch.ops.aten.add.Tensor(mul_282, unsqueeze_119);  mul_282 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_61: "f32[8, 160, 8, 8]" = torch.ops.aten.clone.default(add_211)
    sigmoid_31: "f32[8, 160, 8, 8]" = torch.ops.aten.sigmoid.default(add_211)
    mul_283: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(add_211, sigmoid_31);  add_211 = sigmoid_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:268, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
    cat_2: "f32[8, 320, 8, 8]" = torch.ops.aten.cat.default([add_181, mul_283], 1);  mul_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_33: "f32[8, 160, 8, 8]" = torch.ops.aten.convolution.default(cat_2, primals_212, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_212: "i64[]" = torch.ops.aten.add.Tensor(primals_306, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_51 = torch.ops.aten.var_mean.correction(convolution_33, [0, 2, 3], correction = 0, keepdim = True)
    getitem_210: "f32[1, 160, 1, 1]" = var_mean_51[0]
    getitem_211: "f32[1, 160, 1, 1]" = var_mean_51[1];  var_mean_51 = None
    add_213: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_210, 1e-05)
    rsqrt_51: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
    sub_51: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_33, getitem_211)
    mul_284: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
    squeeze_90: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_211, [0, 2, 3]);  getitem_211 = None
    squeeze_91: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
    mul_285: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
    mul_286: "f32[160]" = torch.ops.aten.mul.Tensor(primals_307, 0.9)
    add_214: "f32[160]" = torch.ops.aten.add.Tensor(mul_285, mul_286);  mul_285 = mul_286 = None
    squeeze_92: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_210, [0, 2, 3]);  getitem_210 = None
    mul_287: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0019569471624266);  squeeze_92 = None
    mul_288: "f32[160]" = torch.ops.aten.mul.Tensor(mul_287, 0.1);  mul_287 = None
    mul_289: "f32[160]" = torch.ops.aten.mul.Tensor(primals_308, 0.9)
    add_215: "f32[160]" = torch.ops.aten.add.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
    unsqueeze_120: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_61, -1)
    unsqueeze_121: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    mul_290: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(mul_284, unsqueeze_121);  mul_284 = unsqueeze_121 = None
    unsqueeze_122: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1);  primals_62 = None
    unsqueeze_123: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    add_216: "f32[8, 160, 8, 8]" = torch.ops.aten.add.Tensor(mul_290, unsqueeze_123);  mul_290 = unsqueeze_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_62: "f32[8, 160, 8, 8]" = torch.ops.aten.clone.default(add_216)
    sigmoid_32: "f32[8, 160, 8, 8]" = torch.ops.aten.sigmoid.default(add_216)
    mul_291: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(add_216, sigmoid_32);  add_216 = sigmoid_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_34: "f32[8, 640, 8, 8]" = torch.ops.aten.convolution.default(mul_291, primals_213, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_217: "i64[]" = torch.ops.aten.add.Tensor(primals_309, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_52 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
    getitem_212: "f32[1, 640, 1, 1]" = var_mean_52[0]
    getitem_213: "f32[1, 640, 1, 1]" = var_mean_52[1];  var_mean_52 = None
    add_218: "f32[1, 640, 1, 1]" = torch.ops.aten.add.Tensor(getitem_212, 1e-05)
    rsqrt_52: "f32[1, 640, 1, 1]" = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
    sub_52: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_213)
    mul_292: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
    squeeze_93: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_213, [0, 2, 3]);  getitem_213 = None
    squeeze_94: "f32[640]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
    mul_293: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
    mul_294: "f32[640]" = torch.ops.aten.mul.Tensor(primals_310, 0.9)
    add_219: "f32[640]" = torch.ops.aten.add.Tensor(mul_293, mul_294);  mul_293 = mul_294 = None
    squeeze_95: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_212, [0, 2, 3]);  getitem_212 = None
    mul_295: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0019569471624266);  squeeze_95 = None
    mul_296: "f32[640]" = torch.ops.aten.mul.Tensor(mul_295, 0.1);  mul_295 = None
    mul_297: "f32[640]" = torch.ops.aten.mul.Tensor(primals_311, 0.9)
    add_220: "f32[640]" = torch.ops.aten.add.Tensor(mul_296, mul_297);  mul_296 = mul_297 = None
    unsqueeze_124: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1)
    unsqueeze_125: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_298: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(mul_292, unsqueeze_125);  mul_292 = unsqueeze_125 = None
    unsqueeze_126: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_64, -1);  primals_64 = None
    unsqueeze_127: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_221: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Tensor(mul_298, unsqueeze_127);  mul_298 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_63: "f32[8, 640, 8, 8]" = torch.ops.aten.clone.default(add_221)
    sigmoid_33: "f32[8, 640, 8, 8]" = torch.ops.aten.sigmoid.default(add_221)
    mul_299: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(add_221, sigmoid_33);  add_221 = sigmoid_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean: "f32[8, 640, 1, 1]" = torch.ops.aten.mean.dim(mul_299, [-1, -2], True);  mul_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_108: "f32[8, 640]" = torch.ops.aten.view.default(mean, [8, 640]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:131, code: x = self.drop(x)
    clone_64: "f32[8, 640]" = torch.ops.aten.clone.default(view_108);  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    permute_66: "f32[640, 1000]" = torch.ops.aten.permute.default(primals_214, [1, 0]);  primals_214 = None
    addmm_36: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_215, clone_64, permute_66);  primals_215 = None
    permute_67: "f32[1000, 640]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    mm: "f32[8, 640]" = torch.ops.aten.mm.default(tangents_1, permute_67);  permute_67 = None
    permute_68: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 640]" = torch.ops.aten.mm.default(permute_68, clone_64);  permute_68 = clone_64 = None
    permute_69: "f32[640, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_109: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_70: "f32[1000, 640]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_110: "f32[8, 640, 1, 1]" = torch.ops.aten.view.default(mm, [8, 640, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 640, 8, 8]" = torch.ops.aten.expand.default(view_110, [8, 640, 8, 8]);  view_110 = None
    div: "f32[8, 640, 8, 8]" = torch.ops.aten.div.Scalar(expand, 64);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_34: "f32[8, 640, 8, 8]" = torch.ops.aten.sigmoid.default(clone_63)
    full: "f32[8, 640, 8, 8]" = torch.ops.aten.full.default([8, 640, 8, 8], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_53: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(full, sigmoid_34);  full = None
    mul_300: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(clone_63, sub_53);  clone_63 = sub_53 = None
    add_222: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Scalar(mul_300, 1);  mul_300 = None
    mul_301: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sigmoid_34, add_222);  sigmoid_34 = add_222 = None
    mul_302: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(div, mul_301);  div = mul_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_128: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
    unsqueeze_129: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, 2);  unsqueeze_128 = None
    unsqueeze_130: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_129, 3);  unsqueeze_129 = None
    sum_2: "f32[640]" = torch.ops.aten.sum.dim_IntList(mul_302, [0, 2, 3])
    sub_54: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_130)
    mul_303: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(mul_302, sub_54);  sub_54 = None
    sum_3: "f32[640]" = torch.ops.aten.sum.dim_IntList(mul_303, [0, 2, 3]);  mul_303 = None
    mul_304: "f32[640]" = torch.ops.aten.mul.Tensor(sum_2, 0.001953125)
    unsqueeze_131: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_304, 0);  mul_304 = None
    unsqueeze_132: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_131, 2);  unsqueeze_131 = None
    unsqueeze_133: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, 3);  unsqueeze_132 = None
    mul_305: "f32[640]" = torch.ops.aten.mul.Tensor(sum_3, 0.001953125)
    mul_306: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_307: "f32[640]" = torch.ops.aten.mul.Tensor(mul_305, mul_306);  mul_305 = mul_306 = None
    unsqueeze_134: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_307, 0);  mul_307 = None
    unsqueeze_135: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, 2);  unsqueeze_134 = None
    unsqueeze_136: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_135, 3);  unsqueeze_135 = None
    mul_308: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_63);  primals_63 = None
    unsqueeze_137: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_308, 0);  mul_308 = None
    unsqueeze_138: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_137, 2);  unsqueeze_137 = None
    unsqueeze_139: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, 3);  unsqueeze_138 = None
    sub_55: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_130);  convolution_34 = unsqueeze_130 = None
    mul_309: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_136);  sub_55 = unsqueeze_136 = None
    sub_56: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(mul_302, mul_309);  mul_302 = mul_309 = None
    sub_57: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(sub_56, unsqueeze_133);  sub_56 = unsqueeze_133 = None
    mul_310: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_139);  sub_57 = unsqueeze_139 = None
    mul_311: "f32[640]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_94);  sum_3 = squeeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_310, mul_291, primals_213, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_310 = mul_291 = primals_213 = None
    getitem_214: "f32[8, 160, 8, 8]" = convolution_backward[0]
    getitem_215: "f32[640, 160, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_35: "f32[8, 160, 8, 8]" = torch.ops.aten.sigmoid.default(clone_62)
    full_1: "f32[8, 160, 8, 8]" = torch.ops.aten.full.default([8, 160, 8, 8], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_58: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(full_1, sigmoid_35);  full_1 = None
    mul_312: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(clone_62, sub_58);  clone_62 = sub_58 = None
    add_223: "f32[8, 160, 8, 8]" = torch.ops.aten.add.Scalar(mul_312, 1);  mul_312 = None
    mul_313: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sigmoid_35, add_223);  sigmoid_35 = add_223 = None
    mul_314: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_214, mul_313);  getitem_214 = mul_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_140: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_141: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, 2);  unsqueeze_140 = None
    unsqueeze_142: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_141, 3);  unsqueeze_141 = None
    sum_4: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_314, [0, 2, 3])
    sub_59: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_142)
    mul_315: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(mul_314, sub_59);  sub_59 = None
    sum_5: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_315, [0, 2, 3]);  mul_315 = None
    mul_316: "f32[160]" = torch.ops.aten.mul.Tensor(sum_4, 0.001953125)
    unsqueeze_143: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_316, 0);  mul_316 = None
    unsqueeze_144: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_143, 2);  unsqueeze_143 = None
    unsqueeze_145: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, 3);  unsqueeze_144 = None
    mul_317: "f32[160]" = torch.ops.aten.mul.Tensor(sum_5, 0.001953125)
    mul_318: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_319: "f32[160]" = torch.ops.aten.mul.Tensor(mul_317, mul_318);  mul_317 = mul_318 = None
    unsqueeze_146: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_319, 0);  mul_319 = None
    unsqueeze_147: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, 2);  unsqueeze_146 = None
    unsqueeze_148: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_147, 3);  unsqueeze_147 = None
    mul_320: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_61);  primals_61 = None
    unsqueeze_149: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_320, 0);  mul_320 = None
    unsqueeze_150: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_149, 2);  unsqueeze_149 = None
    unsqueeze_151: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, 3);  unsqueeze_150 = None
    sub_60: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_142);  convolution_33 = unsqueeze_142 = None
    mul_321: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_148);  sub_60 = unsqueeze_148 = None
    sub_61: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(mul_314, mul_321);  mul_314 = mul_321 = None
    sub_62: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(sub_61, unsqueeze_145);  sub_61 = unsqueeze_145 = None
    mul_322: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_151);  sub_62 = unsqueeze_151 = None
    mul_323: "f32[160]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_91);  sum_5 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_322, cat_2, primals_212, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_322 = cat_2 = primals_212 = None
    getitem_217: "f32[8, 320, 8, 8]" = convolution_backward_1[0]
    getitem_218: "f32[160, 320, 3, 3]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:268, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
    slice_1: "f32[8, 160, 8, 8]" = torch.ops.aten.slice.Tensor(getitem_217, 1, 0, 160)
    slice_2: "f32[8, 160, 8, 8]" = torch.ops.aten.slice.Tensor(getitem_217, 1, 160, 320);  getitem_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_36: "f32[8, 160, 8, 8]" = torch.ops.aten.sigmoid.default(clone_61)
    full_2: "f32[8, 160, 8, 8]" = torch.ops.aten.full.default([8, 160, 8, 8], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_63: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(full_2, sigmoid_36);  full_2 = None
    mul_324: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(clone_61, sub_63);  clone_61 = sub_63 = None
    add_224: "f32[8, 160, 8, 8]" = torch.ops.aten.add.Scalar(mul_324, 1);  mul_324 = None
    mul_325: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sigmoid_36, add_224);  sigmoid_36 = add_224 = None
    mul_326: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(slice_2, mul_325);  slice_2 = mul_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_152: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    unsqueeze_153: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, 2);  unsqueeze_152 = None
    unsqueeze_154: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_153, 3);  unsqueeze_153 = None
    sum_6: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_326, [0, 2, 3])
    sub_64: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_154)
    mul_327: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(mul_326, sub_64);  sub_64 = None
    sum_7: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_327, [0, 2, 3]);  mul_327 = None
    mul_328: "f32[160]" = torch.ops.aten.mul.Tensor(sum_6, 0.001953125)
    unsqueeze_155: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_328, 0);  mul_328 = None
    unsqueeze_156: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_155, 2);  unsqueeze_155 = None
    unsqueeze_157: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, 3);  unsqueeze_156 = None
    mul_329: "f32[160]" = torch.ops.aten.mul.Tensor(sum_7, 0.001953125)
    mul_330: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_331: "f32[160]" = torch.ops.aten.mul.Tensor(mul_329, mul_330);  mul_329 = mul_330 = None
    unsqueeze_158: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_331, 0);  mul_331 = None
    unsqueeze_159: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, 2);  unsqueeze_158 = None
    unsqueeze_160: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_159, 3);  unsqueeze_159 = None
    mul_332: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_59);  primals_59 = None
    unsqueeze_161: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_332, 0);  mul_332 = None
    unsqueeze_162: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_161, 2);  unsqueeze_161 = None
    unsqueeze_163: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, 3);  unsqueeze_162 = None
    sub_65: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_154);  convolution_32 = unsqueeze_154 = None
    mul_333: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_160);  sub_65 = unsqueeze_160 = None
    sub_66: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(mul_326, mul_333);  mul_326 = mul_333 = None
    sub_67: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(sub_66, unsqueeze_157);  sub_66 = unsqueeze_157 = None
    mul_334: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_163);  sub_67 = unsqueeze_163 = None
    mul_335: "f32[160]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_88);  sum_7 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_334, view_107, primals_211, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_334 = view_107 = primals_211 = None
    getitem_220: "f32[8, 240, 8, 8]" = convolution_backward_2[0]
    getitem_221: "f32[160, 240, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:262, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
    view_111: "f32[7680, 2, 4, 2]" = torch.ops.aten.view.default(getitem_220, [7680, 2, 4, 2]);  getitem_220 = None
    permute_74: "f32[7680, 4, 2, 2]" = torch.ops.aten.permute.default(view_111, [0, 2, 1, 3]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:260, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
    clone_65: "f32[7680, 4, 2, 2]" = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None
    view_112: "f32[8, 240, 16, 4]" = torch.ops.aten.view.default(clone_65, [8, 240, 16, 4]);  clone_65 = None
    permute_75: "f32[8, 4, 16, 240]" = torch.ops.aten.permute.default(view_112, [0, 3, 2, 1]);  view_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:259, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
    clone_66: "f32[8, 4, 16, 240]" = torch.ops.aten.clone.default(permute_75, memory_format = torch.contiguous_format);  permute_75 = None
    view_113: "f32[32, 16, 240]" = torch.ops.aten.view.default(clone_66, [32, 16, 240]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:255, code: x = self.norm(x)
    sub_68: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(add_204, getitem_207);  add_204 = getitem_207 = None
    mul_336: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_49);  sub_68 = None
    mul_337: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(view_113, primals_209);  primals_209 = None
    mul_338: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_337, 240)
    sum_8: "f32[32, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_337, [2], True)
    mul_339: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_337, mul_336);  mul_337 = None
    sum_9: "f32[32, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_339, [2], True);  mul_339 = None
    mul_340: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_336, sum_9);  sum_9 = None
    sub_69: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(mul_338, sum_8);  mul_338 = sum_8 = None
    sub_70: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(sub_69, mul_340);  sub_69 = mul_340 = None
    div_1: "f32[32, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_49, 240);  rsqrt_49 = None
    mul_341: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(div_1, sub_70);  div_1 = sub_70 = None
    mul_342: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(view_113, mul_336);  mul_336 = None
    sum_10: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_342, [0, 1]);  mul_342 = None
    sum_11: "f32[240]" = torch.ops.aten.sum.dim_IntList(view_113, [0, 1]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_114: "f32[512, 240]" = torch.ops.aten.view.default(mul_341, [512, 240])
    permute_76: "f32[240, 480]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    mm_2: "f32[512, 480]" = torch.ops.aten.mm.default(view_114, permute_76);  permute_76 = None
    permute_77: "f32[240, 512]" = torch.ops.aten.permute.default(view_114, [1, 0])
    mm_3: "f32[240, 480]" = torch.ops.aten.mm.default(permute_77, view_103);  permute_77 = view_103 = None
    permute_78: "f32[480, 240]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_12: "f32[1, 240]" = torch.ops.aten.sum.dim_IntList(view_114, [0], True);  view_114 = None
    view_115: "f32[240]" = torch.ops.aten.view.default(sum_12, [240]);  sum_12 = None
    permute_79: "f32[240, 480]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    view_116: "f32[32, 16, 480]" = torch.ops.aten.view.default(mm_2, [32, 16, 480]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_37: "f32[32, 16, 480]" = torch.ops.aten.sigmoid.default(view_102)
    full_3: "f32[32, 16, 480]" = torch.ops.aten.full.default([32, 16, 480], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_71: "f32[32, 16, 480]" = torch.ops.aten.sub.Tensor(full_3, sigmoid_37);  full_3 = None
    mul_343: "f32[32, 16, 480]" = torch.ops.aten.mul.Tensor(view_102, sub_71);  view_102 = sub_71 = None
    add_225: "f32[32, 16, 480]" = torch.ops.aten.add.Scalar(mul_343, 1);  mul_343 = None
    mul_344: "f32[32, 16, 480]" = torch.ops.aten.mul.Tensor(sigmoid_37, add_225);  sigmoid_37 = add_225 = None
    mul_345: "f32[32, 16, 480]" = torch.ops.aten.mul.Tensor(view_116, mul_344);  view_116 = mul_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_117: "f32[512, 480]" = torch.ops.aten.view.default(mul_345, [512, 480]);  mul_345 = None
    permute_81: "f32[480, 240]" = torch.ops.aten.permute.default(permute_62, [1, 0]);  permute_62 = None
    mm_4: "f32[512, 240]" = torch.ops.aten.mm.default(view_117, permute_81);  permute_81 = None
    permute_82: "f32[480, 512]" = torch.ops.aten.permute.default(view_117, [1, 0])
    mm_5: "f32[480, 240]" = torch.ops.aten.mm.default(permute_82, view_101);  permute_82 = view_101 = None
    permute_83: "f32[240, 480]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_13: "f32[1, 480]" = torch.ops.aten.sum.dim_IntList(view_117, [0], True);  view_117 = None
    view_118: "f32[480]" = torch.ops.aten.view.default(sum_13, [480]);  sum_13 = None
    permute_84: "f32[480, 240]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
    view_119: "f32[32, 16, 240]" = torch.ops.aten.view.default(mm_4, [32, 16, 240]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_72: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(add_201, getitem_205);  add_201 = getitem_205 = None
    mul_346: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_48);  sub_72 = None
    mul_347: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(view_119, primals_203);  primals_203 = None
    mul_348: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_347, 240)
    sum_14: "f32[32, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_347, [2], True)
    mul_349: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_347, mul_346);  mul_347 = None
    sum_15: "f32[32, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_349, [2], True);  mul_349 = None
    mul_350: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_346, sum_15);  sum_15 = None
    sub_73: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(mul_348, sum_14);  mul_348 = sum_14 = None
    sub_74: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(sub_73, mul_350);  sub_73 = mul_350 = None
    div_2: "f32[32, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 240);  rsqrt_48 = None
    mul_351: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(div_2, sub_74);  div_2 = sub_74 = None
    mul_352: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(view_119, mul_346);  mul_346 = None
    sum_16: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_352, [0, 1]);  mul_352 = None
    sum_17: "f32[240]" = torch.ops.aten.sum.dim_IntList(view_119, [0, 1]);  view_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_226: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(mul_341, mul_351);  mul_341 = mul_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_120: "f32[512, 240]" = torch.ops.aten.view.default(add_226, [512, 240])
    permute_85: "f32[240, 240]" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
    mm_6: "f32[512, 240]" = torch.ops.aten.mm.default(view_120, permute_85);  permute_85 = None
    permute_86: "f32[240, 512]" = torch.ops.aten.permute.default(view_120, [1, 0])
    mm_7: "f32[240, 240]" = torch.ops.aten.mm.default(permute_86, view_99);  permute_86 = view_99 = None
    permute_87: "f32[240, 240]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_18: "f32[1, 240]" = torch.ops.aten.sum.dim_IntList(view_120, [0], True);  view_120 = None
    view_121: "f32[240]" = torch.ops.aten.view.default(sum_18, [240]);  sum_18 = None
    permute_88: "f32[240, 240]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    view_122: "f32[32, 16, 240]" = torch.ops.aten.view.default(mm_6, [32, 16, 240]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_123: "f32[32, 16, 4, 60]" = torch.ops.aten.view.default(view_122, [32, 16, 4, 60]);  view_122 = None
    permute_89: "f32[32, 4, 16, 60]" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_9: "f32[32, 4, 16, 60]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    _scaled_dot_product_flash_attention_backward = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_89, getitem_192, getitem_193, getitem_194, alias_9, getitem_196, getitem_197, getitem_198, 0, 0, 0.0, False, getitem_201, getitem_202);  permute_89 = getitem_192 = getitem_193 = getitem_194 = alias_9 = getitem_196 = getitem_197 = getitem_198 = getitem_201 = getitem_202 = None
    getitem_223: "f32[32, 4, 16, 60]" = _scaled_dot_product_flash_attention_backward[0]
    getitem_224: "f32[32, 4, 16, 60]" = _scaled_dot_product_flash_attention_backward[1]
    getitem_225: "f32[32, 4, 16, 60]" = _scaled_dot_product_flash_attention_backward[2];  _scaled_dot_product_flash_attention_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_3: "f32[96, 4, 16, 60]" = torch.ops.aten.cat.default([getitem_223, getitem_224, getitem_225]);  getitem_223 = getitem_224 = getitem_225 = None
    view_124: "f32[3, 32, 4, 16, 60]" = torch.ops.aten.view.default(cat_3, [3, 32, 4, 16, 60]);  cat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_90: "f32[32, 16, 3, 4, 60]" = torch.ops.aten.permute.default(view_124, [1, 3, 0, 2, 4]);  view_124 = None
    clone_67: "f32[32, 16, 3, 4, 60]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
    view_125: "f32[32, 16, 720]" = torch.ops.aten.view.default(clone_67, [32, 16, 720]);  clone_67 = None
    view_126: "f32[512, 720]" = torch.ops.aten.view.default(view_125, [512, 720]);  view_125 = None
    permute_91: "f32[720, 240]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    mm_8: "f32[512, 240]" = torch.ops.aten.mm.default(view_126, permute_91);  permute_91 = None
    permute_92: "f32[720, 512]" = torch.ops.aten.permute.default(view_126, [1, 0])
    mm_9: "f32[720, 240]" = torch.ops.aten.mm.default(permute_92, view_95);  permute_92 = view_95 = None
    permute_93: "f32[240, 720]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_19: "f32[1, 720]" = torch.ops.aten.sum.dim_IntList(view_126, [0], True);  view_126 = None
    view_127: "f32[720]" = torch.ops.aten.view.default(sum_19, [720]);  sum_19 = None
    permute_94: "f32[720, 240]" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
    view_128: "f32[32, 16, 240]" = torch.ops.aten.view.default(mm_8, [32, 16, 240]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_75: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(add_198, getitem_191);  add_198 = getitem_191 = None
    mul_353: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_47);  sub_75 = None
    mul_354: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(view_128, primals_197);  primals_197 = None
    mul_355: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_354, 240)
    sum_20: "f32[32, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_354, [2], True)
    mul_356: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_354, mul_353);  mul_354 = None
    sum_21: "f32[32, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_356, [2], True);  mul_356 = None
    mul_357: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_353, sum_21);  sum_21 = None
    sub_76: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(mul_355, sum_20);  mul_355 = sum_20 = None
    sub_77: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(sub_76, mul_357);  sub_76 = mul_357 = None
    div_3: "f32[32, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 240);  rsqrt_47 = None
    mul_358: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(div_3, sub_77);  div_3 = sub_77 = None
    mul_359: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(view_128, mul_353);  mul_353 = None
    sum_22: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_359, [0, 1]);  mul_359 = None
    sum_23: "f32[240]" = torch.ops.aten.sum.dim_IntList(view_128, [0, 1]);  view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_227: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(add_226, mul_358);  add_226 = mul_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_129: "f32[512, 240]" = torch.ops.aten.view.default(add_227, [512, 240])
    permute_95: "f32[240, 480]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    mm_10: "f32[512, 480]" = torch.ops.aten.mm.default(view_129, permute_95);  permute_95 = None
    permute_96: "f32[240, 512]" = torch.ops.aten.permute.default(view_129, [1, 0])
    mm_11: "f32[240, 480]" = torch.ops.aten.mm.default(permute_96, view_93);  permute_96 = view_93 = None
    permute_97: "f32[480, 240]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_24: "f32[1, 240]" = torch.ops.aten.sum.dim_IntList(view_129, [0], True);  view_129 = None
    view_130: "f32[240]" = torch.ops.aten.view.default(sum_24, [240]);  sum_24 = None
    permute_98: "f32[240, 480]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    view_131: "f32[32, 16, 480]" = torch.ops.aten.view.default(mm_10, [32, 16, 480]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_38: "f32[32, 16, 480]" = torch.ops.aten.sigmoid.default(view_92)
    full_4: "f32[32, 16, 480]" = torch.ops.aten.full.default([32, 16, 480], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_78: "f32[32, 16, 480]" = torch.ops.aten.sub.Tensor(full_4, sigmoid_38);  full_4 = None
    mul_360: "f32[32, 16, 480]" = torch.ops.aten.mul.Tensor(view_92, sub_78);  view_92 = sub_78 = None
    add_228: "f32[32, 16, 480]" = torch.ops.aten.add.Scalar(mul_360, 1);  mul_360 = None
    mul_361: "f32[32, 16, 480]" = torch.ops.aten.mul.Tensor(sigmoid_38, add_228);  sigmoid_38 = add_228 = None
    mul_362: "f32[32, 16, 480]" = torch.ops.aten.mul.Tensor(view_131, mul_361);  view_131 = mul_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_132: "f32[512, 480]" = torch.ops.aten.view.default(mul_362, [512, 480]);  mul_362 = None
    permute_100: "f32[480, 240]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    mm_12: "f32[512, 240]" = torch.ops.aten.mm.default(view_132, permute_100);  permute_100 = None
    permute_101: "f32[480, 512]" = torch.ops.aten.permute.default(view_132, [1, 0])
    mm_13: "f32[480, 240]" = torch.ops.aten.mm.default(permute_101, view_91);  permute_101 = view_91 = None
    permute_102: "f32[240, 480]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_25: "f32[1, 480]" = torch.ops.aten.sum.dim_IntList(view_132, [0], True);  view_132 = None
    view_133: "f32[480]" = torch.ops.aten.view.default(sum_25, [480]);  sum_25 = None
    permute_103: "f32[480, 240]" = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
    view_134: "f32[32, 16, 240]" = torch.ops.aten.view.default(mm_12, [32, 16, 240]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_79: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(add_195, getitem_189);  add_195 = getitem_189 = None
    mul_363: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_46);  sub_79 = None
    mul_364: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(view_134, primals_191);  primals_191 = None
    mul_365: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_364, 240)
    sum_26: "f32[32, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_364, [2], True)
    mul_366: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_364, mul_363);  mul_364 = None
    sum_27: "f32[32, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_366, [2], True);  mul_366 = None
    mul_367: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_363, sum_27);  sum_27 = None
    sub_80: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(mul_365, sum_26);  mul_365 = sum_26 = None
    sub_81: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(sub_80, mul_367);  sub_80 = mul_367 = None
    div_4: "f32[32, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 240);  rsqrt_46 = None
    mul_368: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(div_4, sub_81);  div_4 = sub_81 = None
    mul_369: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(view_134, mul_363);  mul_363 = None
    sum_28: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_369, [0, 1]);  mul_369 = None
    sum_29: "f32[240]" = torch.ops.aten.sum.dim_IntList(view_134, [0, 1]);  view_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_229: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(add_227, mul_368);  add_227 = mul_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_135: "f32[512, 240]" = torch.ops.aten.view.default(add_229, [512, 240])
    permute_104: "f32[240, 240]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    mm_14: "f32[512, 240]" = torch.ops.aten.mm.default(view_135, permute_104);  permute_104 = None
    permute_105: "f32[240, 512]" = torch.ops.aten.permute.default(view_135, [1, 0])
    mm_15: "f32[240, 240]" = torch.ops.aten.mm.default(permute_105, view_89);  permute_105 = view_89 = None
    permute_106: "f32[240, 240]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_30: "f32[1, 240]" = torch.ops.aten.sum.dim_IntList(view_135, [0], True);  view_135 = None
    view_136: "f32[240]" = torch.ops.aten.view.default(sum_30, [240]);  sum_30 = None
    permute_107: "f32[240, 240]" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
    view_137: "f32[32, 16, 240]" = torch.ops.aten.view.default(mm_14, [32, 16, 240]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_138: "f32[32, 16, 4, 60]" = torch.ops.aten.view.default(view_137, [32, 16, 4, 60]);  view_137 = None
    permute_108: "f32[32, 4, 16, 60]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_10: "f32[32, 4, 16, 60]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    _scaled_dot_product_flash_attention_backward_1 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_108, getitem_176, getitem_177, getitem_178, alias_10, getitem_180, getitem_181, getitem_182, 0, 0, 0.0, False, getitem_185, getitem_186);  permute_108 = getitem_176 = getitem_177 = getitem_178 = alias_10 = getitem_180 = getitem_181 = getitem_182 = getitem_185 = getitem_186 = None
    getitem_226: "f32[32, 4, 16, 60]" = _scaled_dot_product_flash_attention_backward_1[0]
    getitem_227: "f32[32, 4, 16, 60]" = _scaled_dot_product_flash_attention_backward_1[1]
    getitem_228: "f32[32, 4, 16, 60]" = _scaled_dot_product_flash_attention_backward_1[2];  _scaled_dot_product_flash_attention_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_4: "f32[96, 4, 16, 60]" = torch.ops.aten.cat.default([getitem_226, getitem_227, getitem_228]);  getitem_226 = getitem_227 = getitem_228 = None
    view_139: "f32[3, 32, 4, 16, 60]" = torch.ops.aten.view.default(cat_4, [3, 32, 4, 16, 60]);  cat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_109: "f32[32, 16, 3, 4, 60]" = torch.ops.aten.permute.default(view_139, [1, 3, 0, 2, 4]);  view_139 = None
    clone_68: "f32[32, 16, 3, 4, 60]" = torch.ops.aten.clone.default(permute_109, memory_format = torch.contiguous_format);  permute_109 = None
    view_140: "f32[32, 16, 720]" = torch.ops.aten.view.default(clone_68, [32, 16, 720]);  clone_68 = None
    view_141: "f32[512, 720]" = torch.ops.aten.view.default(view_140, [512, 720]);  view_140 = None
    permute_110: "f32[720, 240]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    mm_16: "f32[512, 240]" = torch.ops.aten.mm.default(view_141, permute_110);  permute_110 = None
    permute_111: "f32[720, 512]" = torch.ops.aten.permute.default(view_141, [1, 0])
    mm_17: "f32[720, 240]" = torch.ops.aten.mm.default(permute_111, view_85);  permute_111 = view_85 = None
    permute_112: "f32[240, 720]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_31: "f32[1, 720]" = torch.ops.aten.sum.dim_IntList(view_141, [0], True);  view_141 = None
    view_142: "f32[720]" = torch.ops.aten.view.default(sum_31, [720]);  sum_31 = None
    permute_113: "f32[720, 240]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
    view_143: "f32[32, 16, 240]" = torch.ops.aten.view.default(mm_16, [32, 16, 240]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_82: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(add_192, getitem_175);  add_192 = getitem_175 = None
    mul_370: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_45);  sub_82 = None
    mul_371: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(view_143, primals_185);  primals_185 = None
    mul_372: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_371, 240)
    sum_32: "f32[32, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_371, [2], True)
    mul_373: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_371, mul_370);  mul_371 = None
    sum_33: "f32[32, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_373, [2], True);  mul_373 = None
    mul_374: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_370, sum_33);  sum_33 = None
    sub_83: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(mul_372, sum_32);  mul_372 = sum_32 = None
    sub_84: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(sub_83, mul_374);  sub_83 = mul_374 = None
    div_5: "f32[32, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 240);  rsqrt_45 = None
    mul_375: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(div_5, sub_84);  div_5 = sub_84 = None
    mul_376: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(view_143, mul_370);  mul_370 = None
    sum_34: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_376, [0, 1]);  mul_376 = None
    sum_35: "f32[240]" = torch.ops.aten.sum.dim_IntList(view_143, [0, 1]);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_230: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(add_229, mul_375);  add_229 = mul_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_144: "f32[512, 240]" = torch.ops.aten.view.default(add_230, [512, 240])
    permute_114: "f32[240, 480]" = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
    mm_18: "f32[512, 480]" = torch.ops.aten.mm.default(view_144, permute_114);  permute_114 = None
    permute_115: "f32[240, 512]" = torch.ops.aten.permute.default(view_144, [1, 0])
    mm_19: "f32[240, 480]" = torch.ops.aten.mm.default(permute_115, view_83);  permute_115 = view_83 = None
    permute_116: "f32[480, 240]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_36: "f32[1, 240]" = torch.ops.aten.sum.dim_IntList(view_144, [0], True);  view_144 = None
    view_145: "f32[240]" = torch.ops.aten.view.default(sum_36, [240]);  sum_36 = None
    permute_117: "f32[240, 480]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
    view_146: "f32[32, 16, 480]" = torch.ops.aten.view.default(mm_18, [32, 16, 480]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_39: "f32[32, 16, 480]" = torch.ops.aten.sigmoid.default(view_82)
    full_5: "f32[32, 16, 480]" = torch.ops.aten.full.default([32, 16, 480], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_85: "f32[32, 16, 480]" = torch.ops.aten.sub.Tensor(full_5, sigmoid_39);  full_5 = None
    mul_377: "f32[32, 16, 480]" = torch.ops.aten.mul.Tensor(view_82, sub_85);  view_82 = sub_85 = None
    add_231: "f32[32, 16, 480]" = torch.ops.aten.add.Scalar(mul_377, 1);  mul_377 = None
    mul_378: "f32[32, 16, 480]" = torch.ops.aten.mul.Tensor(sigmoid_39, add_231);  sigmoid_39 = add_231 = None
    mul_379: "f32[32, 16, 480]" = torch.ops.aten.mul.Tensor(view_146, mul_378);  view_146 = mul_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_147: "f32[512, 480]" = torch.ops.aten.view.default(mul_379, [512, 480]);  mul_379 = None
    permute_119: "f32[480, 240]" = torch.ops.aten.permute.default(permute_50, [1, 0]);  permute_50 = None
    mm_20: "f32[512, 240]" = torch.ops.aten.mm.default(view_147, permute_119);  permute_119 = None
    permute_120: "f32[480, 512]" = torch.ops.aten.permute.default(view_147, [1, 0])
    mm_21: "f32[480, 240]" = torch.ops.aten.mm.default(permute_120, view_81);  permute_120 = view_81 = None
    permute_121: "f32[240, 480]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_37: "f32[1, 480]" = torch.ops.aten.sum.dim_IntList(view_147, [0], True);  view_147 = None
    view_148: "f32[480]" = torch.ops.aten.view.default(sum_37, [480]);  sum_37 = None
    permute_122: "f32[480, 240]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
    view_149: "f32[32, 16, 240]" = torch.ops.aten.view.default(mm_20, [32, 16, 240]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_86: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(add_189, getitem_173);  add_189 = getitem_173 = None
    mul_380: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_44);  sub_86 = None
    mul_381: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(view_149, primals_179);  primals_179 = None
    mul_382: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_381, 240)
    sum_38: "f32[32, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_381, [2], True)
    mul_383: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_381, mul_380);  mul_381 = None
    sum_39: "f32[32, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_383, [2], True);  mul_383 = None
    mul_384: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_380, sum_39);  sum_39 = None
    sub_87: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(mul_382, sum_38);  mul_382 = sum_38 = None
    sub_88: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(sub_87, mul_384);  sub_87 = mul_384 = None
    div_6: "f32[32, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 240);  rsqrt_44 = None
    mul_385: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(div_6, sub_88);  div_6 = sub_88 = None
    mul_386: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(view_149, mul_380);  mul_380 = None
    sum_40: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_386, [0, 1]);  mul_386 = None
    sum_41: "f32[240]" = torch.ops.aten.sum.dim_IntList(view_149, [0, 1]);  view_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_232: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(add_230, mul_385);  add_230 = mul_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_150: "f32[512, 240]" = torch.ops.aten.view.default(add_232, [512, 240])
    permute_123: "f32[240, 240]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
    mm_22: "f32[512, 240]" = torch.ops.aten.mm.default(view_150, permute_123);  permute_123 = None
    permute_124: "f32[240, 512]" = torch.ops.aten.permute.default(view_150, [1, 0])
    mm_23: "f32[240, 240]" = torch.ops.aten.mm.default(permute_124, view_79);  permute_124 = view_79 = None
    permute_125: "f32[240, 240]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_42: "f32[1, 240]" = torch.ops.aten.sum.dim_IntList(view_150, [0], True);  view_150 = None
    view_151: "f32[240]" = torch.ops.aten.view.default(sum_42, [240]);  sum_42 = None
    permute_126: "f32[240, 240]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
    view_152: "f32[32, 16, 240]" = torch.ops.aten.view.default(mm_22, [32, 16, 240]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_153: "f32[32, 16, 4, 60]" = torch.ops.aten.view.default(view_152, [32, 16, 4, 60]);  view_152 = None
    permute_127: "f32[32, 4, 16, 60]" = torch.ops.aten.permute.default(view_153, [0, 2, 1, 3]);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_11: "f32[32, 4, 16, 60]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    _scaled_dot_product_flash_attention_backward_2 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_127, getitem_160, getitem_161, getitem_162, alias_11, getitem_164, getitem_165, getitem_166, 0, 0, 0.0, False, getitem_169, getitem_170);  permute_127 = getitem_160 = getitem_161 = getitem_162 = alias_11 = getitem_164 = getitem_165 = getitem_166 = getitem_169 = getitem_170 = None
    getitem_229: "f32[32, 4, 16, 60]" = _scaled_dot_product_flash_attention_backward_2[0]
    getitem_230: "f32[32, 4, 16, 60]" = _scaled_dot_product_flash_attention_backward_2[1]
    getitem_231: "f32[32, 4, 16, 60]" = _scaled_dot_product_flash_attention_backward_2[2];  _scaled_dot_product_flash_attention_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_5: "f32[96, 4, 16, 60]" = torch.ops.aten.cat.default([getitem_229, getitem_230, getitem_231]);  getitem_229 = getitem_230 = getitem_231 = None
    view_154: "f32[3, 32, 4, 16, 60]" = torch.ops.aten.view.default(cat_5, [3, 32, 4, 16, 60]);  cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_128: "f32[32, 16, 3, 4, 60]" = torch.ops.aten.permute.default(view_154, [1, 3, 0, 2, 4]);  view_154 = None
    clone_69: "f32[32, 16, 3, 4, 60]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    view_155: "f32[32, 16, 720]" = torch.ops.aten.view.default(clone_69, [32, 16, 720]);  clone_69 = None
    view_156: "f32[512, 720]" = torch.ops.aten.view.default(view_155, [512, 720]);  view_155 = None
    permute_129: "f32[720, 240]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    mm_24: "f32[512, 240]" = torch.ops.aten.mm.default(view_156, permute_129);  permute_129 = None
    permute_130: "f32[720, 512]" = torch.ops.aten.permute.default(view_156, [1, 0])
    mm_25: "f32[720, 240]" = torch.ops.aten.mm.default(permute_130, view_75);  permute_130 = view_75 = None
    permute_131: "f32[240, 720]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_43: "f32[1, 720]" = torch.ops.aten.sum.dim_IntList(view_156, [0], True);  view_156 = None
    view_157: "f32[720]" = torch.ops.aten.view.default(sum_43, [720]);  sum_43 = None
    permute_132: "f32[720, 240]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    view_158: "f32[32, 16, 240]" = torch.ops.aten.view.default(mm_24, [32, 16, 240]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_89: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(view_74, getitem_159);  view_74 = getitem_159 = None
    mul_387: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_43);  sub_89 = None
    mul_388: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(view_158, primals_173);  primals_173 = None
    mul_389: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_388, 240)
    sum_44: "f32[32, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_388, [2], True)
    mul_390: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_388, mul_387);  mul_388 = None
    sum_45: "f32[32, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_390, [2], True);  mul_390 = None
    mul_391: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_387, sum_45);  sum_45 = None
    sub_90: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(mul_389, sum_44);  mul_389 = sum_44 = None
    sub_91: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(sub_90, mul_391);  sub_90 = mul_391 = None
    div_7: "f32[32, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 240);  rsqrt_43 = None
    mul_392: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(div_7, sub_91);  div_7 = sub_91 = None
    mul_393: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(view_158, mul_387);  mul_387 = None
    sum_46: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_393, [0, 1]);  mul_393 = None
    sum_47: "f32[240]" = torch.ops.aten.sum.dim_IntList(view_158, [0, 1]);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_233: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(add_232, mul_392);  add_232 = mul_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:251, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
    view_159: "f32[8, 4, 16, 240]" = torch.ops.aten.view.default(add_233, [8, 4, 16, 240]);  add_233 = None
    permute_133: "f32[8, 240, 16, 4]" = torch.ops.aten.permute.default(view_159, [0, 3, 2, 1]);  view_159 = None
    clone_70: "f32[8, 240, 16, 4]" = torch.ops.aten.clone.default(permute_133, memory_format = torch.contiguous_format);  permute_133 = None
    view_160: "f32[7680, 4, 2, 2]" = torch.ops.aten.view.default(clone_70, [7680, 4, 2, 2]);  clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:249, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
    permute_134: "f32[7680, 2, 4, 2]" = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None
    clone_71: "f32[7680, 2, 4, 2]" = torch.ops.aten.clone.default(permute_134, memory_format = torch.contiguous_format);  permute_134 = None
    view_161: "f32[8, 240, 8, 8]" = torch.ops.aten.view.default(clone_71, [8, 240, 8, 8]);  clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:234, code: x = self.conv_1x1(x)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(view_161, mul_258, primals_172, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_161 = mul_258 = primals_172 = None
    getitem_232: "f32[8, 160, 8, 8]" = convolution_backward_3[0]
    getitem_233: "f32[240, 160, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_40: "f32[8, 160, 8, 8]" = torch.ops.aten.sigmoid.default(clone_47)
    full_6: "f32[8, 160, 8, 8]" = torch.ops.aten.full.default([8, 160, 8, 8], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_92: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(full_6, sigmoid_40);  full_6 = None
    mul_394: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(clone_47, sub_92);  clone_47 = sub_92 = None
    add_234: "f32[8, 160, 8, 8]" = torch.ops.aten.add.Scalar(mul_394, 1);  mul_394 = None
    mul_395: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sigmoid_40, add_234);  sigmoid_40 = add_234 = None
    mul_396: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_232, mul_395);  getitem_232 = mul_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_164: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_165: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, 2);  unsqueeze_164 = None
    unsqueeze_166: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_165, 3);  unsqueeze_165 = None
    sum_48: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_396, [0, 2, 3])
    sub_93: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_166)
    mul_397: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(mul_396, sub_93);  sub_93 = None
    sum_49: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_397, [0, 2, 3]);  mul_397 = None
    mul_398: "f32[160]" = torch.ops.aten.mul.Tensor(sum_48, 0.001953125)
    unsqueeze_167: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_398, 0);  mul_398 = None
    unsqueeze_168: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 2);  unsqueeze_167 = None
    unsqueeze_169: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, 3);  unsqueeze_168 = None
    mul_399: "f32[160]" = torch.ops.aten.mul.Tensor(sum_49, 0.001953125)
    mul_400: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_401: "f32[160]" = torch.ops.aten.mul.Tensor(mul_399, mul_400);  mul_399 = mul_400 = None
    unsqueeze_170: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_401, 0);  mul_401 = None
    unsqueeze_171: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, 2);  unsqueeze_170 = None
    unsqueeze_172: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_171, 3);  unsqueeze_171 = None
    mul_402: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_57);  primals_57 = None
    unsqueeze_173: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_402, 0);  mul_402 = None
    unsqueeze_174: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 2);  unsqueeze_173 = None
    unsqueeze_175: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, 3);  unsqueeze_174 = None
    sub_94: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_166);  convolution_30 = unsqueeze_166 = None
    mul_403: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_172);  sub_94 = unsqueeze_172 = None
    sub_95: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(mul_396, mul_403);  mul_396 = mul_403 = None
    sub_96: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(sub_95, unsqueeze_169);  sub_95 = unsqueeze_169 = None
    mul_404: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_175);  sub_96 = unsqueeze_175 = None
    mul_405: "f32[160]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_85);  sum_49 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_404, add_181, primals_171, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_404 = add_181 = primals_171 = None
    getitem_235: "f32[8, 160, 8, 8]" = convolution_backward_4[0]
    getitem_236: "f32[160, 160, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_235: "f32[8, 160, 8, 8]" = torch.ops.aten.add.Tensor(slice_1, getitem_235);  slice_1 = getitem_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_176: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_177: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, 2);  unsqueeze_176 = None
    unsqueeze_178: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_177, 3);  unsqueeze_177 = None
    sum_50: "f32[160]" = torch.ops.aten.sum.dim_IntList(add_235, [0, 2, 3])
    sub_97: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_178)
    mul_406: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(add_235, sub_97);  sub_97 = None
    sum_51: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_406, [0, 2, 3]);  mul_406 = None
    mul_407: "f32[160]" = torch.ops.aten.mul.Tensor(sum_50, 0.001953125)
    unsqueeze_179: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_407, 0);  mul_407 = None
    unsqueeze_180: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_179, 2);  unsqueeze_179 = None
    unsqueeze_181: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, 3);  unsqueeze_180 = None
    mul_408: "f32[160]" = torch.ops.aten.mul.Tensor(sum_51, 0.001953125)
    mul_409: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_410: "f32[160]" = torch.ops.aten.mul.Tensor(mul_408, mul_409);  mul_408 = mul_409 = None
    unsqueeze_182: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_410, 0);  mul_410 = None
    unsqueeze_183: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, 2);  unsqueeze_182 = None
    unsqueeze_184: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 3);  unsqueeze_183 = None
    mul_411: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_55);  primals_55 = None
    unsqueeze_185: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_411, 0);  mul_411 = None
    unsqueeze_186: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 2);  unsqueeze_185 = None
    unsqueeze_187: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, 3);  unsqueeze_186 = None
    sub_98: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_178);  convolution_29 = unsqueeze_178 = None
    mul_412: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_184);  sub_98 = unsqueeze_184 = None
    sub_99: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(add_235, mul_412);  add_235 = mul_412 = None
    sub_100: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(sub_99, unsqueeze_181);  sub_99 = unsqueeze_181 = None
    mul_413: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_187);  sub_100 = unsqueeze_187 = None
    mul_414: "f32[160]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_82);  sum_51 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_413, mul_243, primals_170, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_413 = mul_243 = primals_170 = None
    getitem_238: "f32[8, 512, 8, 8]" = convolution_backward_5[0]
    getitem_239: "f32[160, 512, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_41: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(clone_46)
    full_7: "f32[8, 512, 8, 8]" = torch.ops.aten.full.default([8, 512, 8, 8], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_101: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(full_7, sigmoid_41);  full_7 = None
    mul_415: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(clone_46, sub_101);  clone_46 = sub_101 = None
    add_236: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Scalar(mul_415, 1);  mul_415 = None
    mul_416: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sigmoid_41, add_236);  sigmoid_41 = add_236 = None
    mul_417: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_238, mul_416);  getitem_238 = mul_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_188: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_189: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, 2);  unsqueeze_188 = None
    unsqueeze_190: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_189, 3);  unsqueeze_189 = None
    sum_52: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_417, [0, 2, 3])
    sub_102: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_190)
    mul_418: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_417, sub_102);  sub_102 = None
    sum_53: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_418, [0, 2, 3]);  mul_418 = None
    mul_419: "f32[512]" = torch.ops.aten.mul.Tensor(sum_52, 0.001953125)
    unsqueeze_191: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_419, 0);  mul_419 = None
    unsqueeze_192: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_191, 2);  unsqueeze_191 = None
    unsqueeze_193: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, 3);  unsqueeze_192 = None
    mul_420: "f32[512]" = torch.ops.aten.mul.Tensor(sum_53, 0.001953125)
    mul_421: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_422: "f32[512]" = torch.ops.aten.mul.Tensor(mul_420, mul_421);  mul_420 = mul_421 = None
    unsqueeze_194: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_422, 0);  mul_422 = None
    unsqueeze_195: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, 2);  unsqueeze_194 = None
    unsqueeze_196: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_195, 3);  unsqueeze_195 = None
    mul_423: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_53);  primals_53 = None
    unsqueeze_197: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_423, 0);  mul_423 = None
    unsqueeze_198: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 2);  unsqueeze_197 = None
    unsqueeze_199: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, 3);  unsqueeze_198 = None
    sub_103: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_190);  convolution_28 = unsqueeze_190 = None
    mul_424: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_196);  sub_103 = unsqueeze_196 = None
    sub_104: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(mul_417, mul_424);  mul_417 = mul_424 = None
    sub_105: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_104, unsqueeze_193);  sub_104 = unsqueeze_193 = None
    mul_425: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_199);  sub_105 = unsqueeze_199 = None
    mul_426: "f32[512]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_79);  sum_53 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_425, mul_235, primals_169, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 512, [True, True, False]);  mul_425 = mul_235 = primals_169 = None
    getitem_241: "f32[8, 512, 16, 16]" = convolution_backward_6[0]
    getitem_242: "f32[512, 1, 3, 3]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_42: "f32[8, 512, 16, 16]" = torch.ops.aten.sigmoid.default(clone_45)
    full_8: "f32[8, 512, 16, 16]" = torch.ops.aten.full.default([8, 512, 16, 16], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_106: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(full_8, sigmoid_42);  full_8 = None
    mul_427: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(clone_45, sub_106);  clone_45 = sub_106 = None
    add_237: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Scalar(mul_427, 1);  mul_427 = None
    mul_428: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_42, add_237);  sigmoid_42 = add_237 = None
    mul_429: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_241, mul_428);  getitem_241 = mul_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_200: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_201: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, 2);  unsqueeze_200 = None
    unsqueeze_202: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_201, 3);  unsqueeze_201 = None
    sum_54: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_429, [0, 2, 3])
    sub_107: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_202)
    mul_430: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_429, sub_107);  sub_107 = None
    sum_55: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_430, [0, 2, 3]);  mul_430 = None
    mul_431: "f32[512]" = torch.ops.aten.mul.Tensor(sum_54, 0.00048828125)
    unsqueeze_203: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_431, 0);  mul_431 = None
    unsqueeze_204: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 2);  unsqueeze_203 = None
    unsqueeze_205: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, 3);  unsqueeze_204 = None
    mul_432: "f32[512]" = torch.ops.aten.mul.Tensor(sum_55, 0.00048828125)
    mul_433: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_434: "f32[512]" = torch.ops.aten.mul.Tensor(mul_432, mul_433);  mul_432 = mul_433 = None
    unsqueeze_206: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_434, 0);  mul_434 = None
    unsqueeze_207: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 2);  unsqueeze_206 = None
    unsqueeze_208: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_207, 3);  unsqueeze_207 = None
    mul_435: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_51);  primals_51 = None
    unsqueeze_209: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_435, 0);  mul_435 = None
    unsqueeze_210: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 2);  unsqueeze_209 = None
    unsqueeze_211: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, 3);  unsqueeze_210 = None
    sub_108: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_202);  convolution_27 = unsqueeze_202 = None
    mul_436: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_208);  sub_108 = unsqueeze_208 = None
    sub_109: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(mul_429, mul_436);  mul_429 = mul_436 = None
    sub_110: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_109, unsqueeze_205);  sub_109 = unsqueeze_205 = None
    mul_437: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_211);  sub_110 = unsqueeze_211 = None
    mul_438: "f32[512]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_76);  sum_55 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_437, mul_227, primals_168, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_437 = mul_227 = primals_168 = None
    getitem_244: "f32[8, 128, 16, 16]" = convolution_backward_7[0]
    getitem_245: "f32[512, 128, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_43: "f32[8, 128, 16, 16]" = torch.ops.aten.sigmoid.default(clone_44)
    full_9: "f32[8, 128, 16, 16]" = torch.ops.aten.full.default([8, 128, 16, 16], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_111: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(full_9, sigmoid_43);  full_9 = None
    mul_439: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(clone_44, sub_111);  clone_44 = sub_111 = None
    add_238: "f32[8, 128, 16, 16]" = torch.ops.aten.add.Scalar(mul_439, 1);  mul_439 = None
    mul_440: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_43, add_238);  sigmoid_43 = add_238 = None
    mul_441: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_244, mul_440);  getitem_244 = mul_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_212: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_213: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 2);  unsqueeze_212 = None
    unsqueeze_214: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 3);  unsqueeze_213 = None
    sum_56: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_441, [0, 2, 3])
    sub_112: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_214)
    mul_442: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(mul_441, sub_112);  sub_112 = None
    sum_57: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_442, [0, 2, 3]);  mul_442 = None
    mul_443: "f32[128]" = torch.ops.aten.mul.Tensor(sum_56, 0.00048828125)
    unsqueeze_215: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_443, 0);  mul_443 = None
    unsqueeze_216: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 2);  unsqueeze_215 = None
    unsqueeze_217: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, 3);  unsqueeze_216 = None
    mul_444: "f32[128]" = torch.ops.aten.mul.Tensor(sum_57, 0.00048828125)
    mul_445: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_446: "f32[128]" = torch.ops.aten.mul.Tensor(mul_444, mul_445);  mul_444 = mul_445 = None
    unsqueeze_218: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_446, 0);  mul_446 = None
    unsqueeze_219: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 2);  unsqueeze_218 = None
    unsqueeze_220: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 3);  unsqueeze_219 = None
    mul_447: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_49);  primals_49 = None
    unsqueeze_221: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_447, 0);  mul_447 = None
    unsqueeze_222: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 2);  unsqueeze_221 = None
    unsqueeze_223: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, 3);  unsqueeze_222 = None
    sub_113: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_214);  convolution_26 = unsqueeze_214 = None
    mul_448: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_220);  sub_113 = unsqueeze_220 = None
    sub_114: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(mul_441, mul_448);  mul_441 = mul_448 = None
    sub_115: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(sub_114, unsqueeze_217);  sub_114 = unsqueeze_217 = None
    mul_449: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_223);  sub_115 = unsqueeze_223 = None
    mul_450: "f32[128]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_73);  sum_57 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_449, cat_1, primals_167, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_449 = cat_1 = primals_167 = None
    getitem_247: "f32[8, 256, 16, 16]" = convolution_backward_8[0]
    getitem_248: "f32[128, 256, 3, 3]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:268, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
    slice_3: "f32[8, 128, 16, 16]" = torch.ops.aten.slice.Tensor(getitem_247, 1, 0, 128)
    slice_4: "f32[8, 128, 16, 16]" = torch.ops.aten.slice.Tensor(getitem_247, 1, 128, 256);  getitem_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_44: "f32[8, 128, 16, 16]" = torch.ops.aten.sigmoid.default(clone_43)
    full_10: "f32[8, 128, 16, 16]" = torch.ops.aten.full.default([8, 128, 16, 16], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_116: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(full_10, sigmoid_44);  full_10 = None
    mul_451: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(clone_43, sub_116);  clone_43 = sub_116 = None
    add_239: "f32[8, 128, 16, 16]" = torch.ops.aten.add.Scalar(mul_451, 1);  mul_451 = None
    mul_452: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_44, add_239);  sigmoid_44 = add_239 = None
    mul_453: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(slice_4, mul_452);  slice_4 = mul_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_224: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_225: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 2);  unsqueeze_224 = None
    unsqueeze_226: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 3);  unsqueeze_225 = None
    sum_58: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_453, [0, 2, 3])
    sub_117: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_226)
    mul_454: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(mul_453, sub_117);  sub_117 = None
    sum_59: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_454, [0, 2, 3]);  mul_454 = None
    mul_455: "f32[128]" = torch.ops.aten.mul.Tensor(sum_58, 0.00048828125)
    unsqueeze_227: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_455, 0);  mul_455 = None
    unsqueeze_228: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 2);  unsqueeze_227 = None
    unsqueeze_229: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, 3);  unsqueeze_228 = None
    mul_456: "f32[128]" = torch.ops.aten.mul.Tensor(sum_59, 0.00048828125)
    mul_457: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_458: "f32[128]" = torch.ops.aten.mul.Tensor(mul_456, mul_457);  mul_456 = mul_457 = None
    unsqueeze_230: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_458, 0);  mul_458 = None
    unsqueeze_231: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 2);  unsqueeze_230 = None
    unsqueeze_232: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 3);  unsqueeze_231 = None
    mul_459: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_47);  primals_47 = None
    unsqueeze_233: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_459, 0);  mul_459 = None
    unsqueeze_234: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 2);  unsqueeze_233 = None
    unsqueeze_235: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, 3);  unsqueeze_234 = None
    sub_118: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_226);  convolution_25 = unsqueeze_226 = None
    mul_460: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_232);  sub_118 = unsqueeze_232 = None
    sub_119: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(mul_453, mul_460);  mul_453 = mul_460 = None
    sub_120: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(sub_119, unsqueeze_229);  sub_119 = unsqueeze_229 = None
    mul_461: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_235);  sub_120 = unsqueeze_235 = None
    mul_462: "f32[128]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_70);  sum_59 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_461, view_71, primals_166, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_461 = view_71 = primals_166 = None
    getitem_250: "f32[8, 192, 16, 16]" = convolution_backward_9[0]
    getitem_251: "f32[128, 192, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:262, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
    view_162: "f32[12288, 2, 8, 2]" = torch.ops.aten.view.default(getitem_250, [12288, 2, 8, 2]);  getitem_250 = None
    permute_140: "f32[12288, 8, 2, 2]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:260, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
    clone_72: "f32[12288, 8, 2, 2]" = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
    view_163: "f32[8, 192, 64, 4]" = torch.ops.aten.view.default(clone_72, [8, 192, 64, 4]);  clone_72 = None
    permute_141: "f32[8, 4, 64, 192]" = torch.ops.aten.permute.default(view_163, [0, 3, 2, 1]);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:259, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
    clone_73: "f32[8, 4, 64, 192]" = torch.ops.aten.clone.default(permute_141, memory_format = torch.contiguous_format);  permute_141 = None
    view_164: "f32[32, 64, 192]" = torch.ops.aten.view.default(clone_73, [32, 64, 192]);  clone_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:255, code: x = self.norm(x)
    sub_121: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_154, getitem_145);  add_154 = getitem_145 = None
    mul_463: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_121, rsqrt_36);  sub_121 = None
    mul_464: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_164, primals_164);  primals_164 = None
    mul_465: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_464, 192)
    sum_60: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_464, [2], True)
    mul_466: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_464, mul_463);  mul_464 = None
    sum_61: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_466, [2], True);  mul_466 = None
    mul_467: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_463, sum_61);  sum_61 = None
    sub_122: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(mul_465, sum_60);  mul_465 = sum_60 = None
    sub_123: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(sub_122, mul_467);  sub_122 = mul_467 = None
    div_8: "f32[32, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 192);  rsqrt_36 = None
    mul_468: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(div_8, sub_123);  div_8 = sub_123 = None
    mul_469: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_164, mul_463);  mul_463 = None
    sum_62: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_469, [0, 1]);  mul_469 = None
    sum_63: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_164, [0, 1]);  view_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_165: "f32[2048, 192]" = torch.ops.aten.view.default(mul_468, [2048, 192])
    permute_142: "f32[192, 384]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    mm_26: "f32[2048, 384]" = torch.ops.aten.mm.default(view_165, permute_142);  permute_142 = None
    permute_143: "f32[192, 2048]" = torch.ops.aten.permute.default(view_165, [1, 0])
    mm_27: "f32[192, 384]" = torch.ops.aten.mm.default(permute_143, view_67);  permute_143 = view_67 = None
    permute_144: "f32[384, 192]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_64: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_165, [0], True);  view_165 = None
    view_166: "f32[192]" = torch.ops.aten.view.default(sum_64, [192]);  sum_64 = None
    permute_145: "f32[192, 384]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
    view_167: "f32[32, 64, 384]" = torch.ops.aten.view.default(mm_26, [32, 64, 384]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_45: "f32[32, 64, 384]" = torch.ops.aten.sigmoid.default(view_66)
    full_11: "f32[32, 64, 384]" = torch.ops.aten.full.default([32, 64, 384], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_124: "f32[32, 64, 384]" = torch.ops.aten.sub.Tensor(full_11, sigmoid_45);  full_11 = None
    mul_470: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(view_66, sub_124);  view_66 = sub_124 = None
    add_240: "f32[32, 64, 384]" = torch.ops.aten.add.Scalar(mul_470, 1);  mul_470 = None
    mul_471: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(sigmoid_45, add_240);  sigmoid_45 = add_240 = None
    mul_472: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(view_167, mul_471);  view_167 = mul_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_168: "f32[2048, 384]" = torch.ops.aten.view.default(mul_472, [2048, 384]);  mul_472 = None
    permute_147: "f32[384, 192]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
    mm_28: "f32[2048, 192]" = torch.ops.aten.mm.default(view_168, permute_147);  permute_147 = None
    permute_148: "f32[384, 2048]" = torch.ops.aten.permute.default(view_168, [1, 0])
    mm_29: "f32[384, 192]" = torch.ops.aten.mm.default(permute_148, view_65);  permute_148 = view_65 = None
    permute_149: "f32[192, 384]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_65: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_168, [0], True);  view_168 = None
    view_169: "f32[384]" = torch.ops.aten.view.default(sum_65, [384]);  sum_65 = None
    permute_150: "f32[384, 192]" = torch.ops.aten.permute.default(permute_149, [1, 0]);  permute_149 = None
    view_170: "f32[32, 64, 192]" = torch.ops.aten.view.default(mm_28, [32, 64, 192]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_125: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_151, getitem_143);  add_151 = getitem_143 = None
    mul_473: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_125, rsqrt_35);  sub_125 = None
    mul_474: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_170, primals_158);  primals_158 = None
    mul_475: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_474, 192)
    sum_66: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_474, [2], True)
    mul_476: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_474, mul_473);  mul_474 = None
    sum_67: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_476, [2], True);  mul_476 = None
    mul_477: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_473, sum_67);  sum_67 = None
    sub_126: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(mul_475, sum_66);  mul_475 = sum_66 = None
    sub_127: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(sub_126, mul_477);  sub_126 = mul_477 = None
    div_9: "f32[32, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 192);  rsqrt_35 = None
    mul_478: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(div_9, sub_127);  div_9 = sub_127 = None
    mul_479: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_170, mul_473);  mul_473 = None
    sum_68: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_479, [0, 1]);  mul_479 = None
    sum_69: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_170, [0, 1]);  view_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_241: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_468, mul_478);  mul_468 = mul_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_171: "f32[2048, 192]" = torch.ops.aten.view.default(add_241, [2048, 192])
    permute_151: "f32[192, 192]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
    mm_30: "f32[2048, 192]" = torch.ops.aten.mm.default(view_171, permute_151);  permute_151 = None
    permute_152: "f32[192, 2048]" = torch.ops.aten.permute.default(view_171, [1, 0])
    mm_31: "f32[192, 192]" = torch.ops.aten.mm.default(permute_152, view_63);  permute_152 = view_63 = None
    permute_153: "f32[192, 192]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_70: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_171, [0], True);  view_171 = None
    view_172: "f32[192]" = torch.ops.aten.view.default(sum_70, [192]);  sum_70 = None
    permute_154: "f32[192, 192]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
    view_173: "f32[32, 64, 192]" = torch.ops.aten.view.default(mm_30, [32, 64, 192]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_174: "f32[32, 64, 4, 48]" = torch.ops.aten.view.default(view_173, [32, 64, 4, 48]);  view_173 = None
    permute_155: "f32[32, 4, 64, 48]" = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_12: "f32[32, 4, 64, 48]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    _scaled_dot_product_flash_attention_backward_3 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_155, getitem_130, getitem_131, getitem_132, alias_12, getitem_134, getitem_135, getitem_136, 0, 0, 0.0, False, getitem_139, getitem_140);  permute_155 = getitem_130 = getitem_131 = getitem_132 = alias_12 = getitem_134 = getitem_135 = getitem_136 = getitem_139 = getitem_140 = None
    getitem_253: "f32[32, 4, 64, 48]" = _scaled_dot_product_flash_attention_backward_3[0]
    getitem_254: "f32[32, 4, 64, 48]" = _scaled_dot_product_flash_attention_backward_3[1]
    getitem_255: "f32[32, 4, 64, 48]" = _scaled_dot_product_flash_attention_backward_3[2];  _scaled_dot_product_flash_attention_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_6: "f32[96, 4, 64, 48]" = torch.ops.aten.cat.default([getitem_253, getitem_254, getitem_255]);  getitem_253 = getitem_254 = getitem_255 = None
    view_175: "f32[3, 32, 4, 64, 48]" = torch.ops.aten.view.default(cat_6, [3, 32, 4, 64, 48]);  cat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_156: "f32[32, 64, 3, 4, 48]" = torch.ops.aten.permute.default(view_175, [1, 3, 0, 2, 4]);  view_175 = None
    clone_74: "f32[32, 64, 3, 4, 48]" = torch.ops.aten.clone.default(permute_156, memory_format = torch.contiguous_format);  permute_156 = None
    view_176: "f32[32, 64, 576]" = torch.ops.aten.view.default(clone_74, [32, 64, 576]);  clone_74 = None
    view_177: "f32[2048, 576]" = torch.ops.aten.view.default(view_176, [2048, 576]);  view_176 = None
    permute_157: "f32[576, 192]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    mm_32: "f32[2048, 192]" = torch.ops.aten.mm.default(view_177, permute_157);  permute_157 = None
    permute_158: "f32[576, 2048]" = torch.ops.aten.permute.default(view_177, [1, 0])
    mm_33: "f32[576, 192]" = torch.ops.aten.mm.default(permute_158, view_59);  permute_158 = view_59 = None
    permute_159: "f32[192, 576]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_71: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_177, [0], True);  view_177 = None
    view_178: "f32[576]" = torch.ops.aten.view.default(sum_71, [576]);  sum_71 = None
    permute_160: "f32[576, 192]" = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None
    view_179: "f32[32, 64, 192]" = torch.ops.aten.view.default(mm_32, [32, 64, 192]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_128: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_148, getitem_129);  add_148 = getitem_129 = None
    mul_480: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_128, rsqrt_34);  sub_128 = None
    mul_481: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_179, primals_152);  primals_152 = None
    mul_482: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_481, 192)
    sum_72: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_481, [2], True)
    mul_483: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_481, mul_480);  mul_481 = None
    sum_73: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_483, [2], True);  mul_483 = None
    mul_484: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_480, sum_73);  sum_73 = None
    sub_129: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(mul_482, sum_72);  mul_482 = sum_72 = None
    sub_130: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(sub_129, mul_484);  sub_129 = mul_484 = None
    div_10: "f32[32, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 192);  rsqrt_34 = None
    mul_485: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(div_10, sub_130);  div_10 = sub_130 = None
    mul_486: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_179, mul_480);  mul_480 = None
    sum_74: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_486, [0, 1]);  mul_486 = None
    sum_75: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_179, [0, 1]);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_242: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_241, mul_485);  add_241 = mul_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_180: "f32[2048, 192]" = torch.ops.aten.view.default(add_242, [2048, 192])
    permute_161: "f32[192, 384]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    mm_34: "f32[2048, 384]" = torch.ops.aten.mm.default(view_180, permute_161);  permute_161 = None
    permute_162: "f32[192, 2048]" = torch.ops.aten.permute.default(view_180, [1, 0])
    mm_35: "f32[192, 384]" = torch.ops.aten.mm.default(permute_162, view_57);  permute_162 = view_57 = None
    permute_163: "f32[384, 192]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_76: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_180, [0], True);  view_180 = None
    view_181: "f32[192]" = torch.ops.aten.view.default(sum_76, [192]);  sum_76 = None
    permute_164: "f32[192, 384]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
    view_182: "f32[32, 64, 384]" = torch.ops.aten.view.default(mm_34, [32, 64, 384]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_46: "f32[32, 64, 384]" = torch.ops.aten.sigmoid.default(view_56)
    full_12: "f32[32, 64, 384]" = torch.ops.aten.full.default([32, 64, 384], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_131: "f32[32, 64, 384]" = torch.ops.aten.sub.Tensor(full_12, sigmoid_46);  full_12 = None
    mul_487: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(view_56, sub_131);  view_56 = sub_131 = None
    add_243: "f32[32, 64, 384]" = torch.ops.aten.add.Scalar(mul_487, 1);  mul_487 = None
    mul_488: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(sigmoid_46, add_243);  sigmoid_46 = add_243 = None
    mul_489: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(view_182, mul_488);  view_182 = mul_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_183: "f32[2048, 384]" = torch.ops.aten.view.default(mul_489, [2048, 384]);  mul_489 = None
    permute_166: "f32[384, 192]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    mm_36: "f32[2048, 192]" = torch.ops.aten.mm.default(view_183, permute_166);  permute_166 = None
    permute_167: "f32[384, 2048]" = torch.ops.aten.permute.default(view_183, [1, 0])
    mm_37: "f32[384, 192]" = torch.ops.aten.mm.default(permute_167, view_55);  permute_167 = view_55 = None
    permute_168: "f32[192, 384]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_77: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_183, [0], True);  view_183 = None
    view_184: "f32[384]" = torch.ops.aten.view.default(sum_77, [384]);  sum_77 = None
    permute_169: "f32[384, 192]" = torch.ops.aten.permute.default(permute_168, [1, 0]);  permute_168 = None
    view_185: "f32[32, 64, 192]" = torch.ops.aten.view.default(mm_36, [32, 64, 192]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_132: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_145, getitem_127);  add_145 = getitem_127 = None
    mul_490: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_132, rsqrt_33);  sub_132 = None
    mul_491: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_185, primals_146);  primals_146 = None
    mul_492: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_491, 192)
    sum_78: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_491, [2], True)
    mul_493: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_491, mul_490);  mul_491 = None
    sum_79: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_493, [2], True);  mul_493 = None
    mul_494: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_490, sum_79);  sum_79 = None
    sub_133: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(mul_492, sum_78);  mul_492 = sum_78 = None
    sub_134: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(sub_133, mul_494);  sub_133 = mul_494 = None
    div_11: "f32[32, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 192);  rsqrt_33 = None
    mul_495: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(div_11, sub_134);  div_11 = sub_134 = None
    mul_496: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_185, mul_490);  mul_490 = None
    sum_80: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_496, [0, 1]);  mul_496 = None
    sum_81: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_185, [0, 1]);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_244: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_242, mul_495);  add_242 = mul_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_186: "f32[2048, 192]" = torch.ops.aten.view.default(add_244, [2048, 192])
    permute_170: "f32[192, 192]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    mm_38: "f32[2048, 192]" = torch.ops.aten.mm.default(view_186, permute_170);  permute_170 = None
    permute_171: "f32[192, 2048]" = torch.ops.aten.permute.default(view_186, [1, 0])
    mm_39: "f32[192, 192]" = torch.ops.aten.mm.default(permute_171, view_53);  permute_171 = view_53 = None
    permute_172: "f32[192, 192]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_82: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_186, [0], True);  view_186 = None
    view_187: "f32[192]" = torch.ops.aten.view.default(sum_82, [192]);  sum_82 = None
    permute_173: "f32[192, 192]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
    view_188: "f32[32, 64, 192]" = torch.ops.aten.view.default(mm_38, [32, 64, 192]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_189: "f32[32, 64, 4, 48]" = torch.ops.aten.view.default(view_188, [32, 64, 4, 48]);  view_188 = None
    permute_174: "f32[32, 4, 64, 48]" = torch.ops.aten.permute.default(view_189, [0, 2, 1, 3]);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_13: "f32[32, 4, 64, 48]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    _scaled_dot_product_flash_attention_backward_4 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_174, getitem_114, getitem_115, getitem_116, alias_13, getitem_118, getitem_119, getitem_120, 0, 0, 0.0, False, getitem_123, getitem_124);  permute_174 = getitem_114 = getitem_115 = getitem_116 = alias_13 = getitem_118 = getitem_119 = getitem_120 = getitem_123 = getitem_124 = None
    getitem_256: "f32[32, 4, 64, 48]" = _scaled_dot_product_flash_attention_backward_4[0]
    getitem_257: "f32[32, 4, 64, 48]" = _scaled_dot_product_flash_attention_backward_4[1]
    getitem_258: "f32[32, 4, 64, 48]" = _scaled_dot_product_flash_attention_backward_4[2];  _scaled_dot_product_flash_attention_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_7: "f32[96, 4, 64, 48]" = torch.ops.aten.cat.default([getitem_256, getitem_257, getitem_258]);  getitem_256 = getitem_257 = getitem_258 = None
    view_190: "f32[3, 32, 4, 64, 48]" = torch.ops.aten.view.default(cat_7, [3, 32, 4, 64, 48]);  cat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_175: "f32[32, 64, 3, 4, 48]" = torch.ops.aten.permute.default(view_190, [1, 3, 0, 2, 4]);  view_190 = None
    clone_75: "f32[32, 64, 3, 4, 48]" = torch.ops.aten.clone.default(permute_175, memory_format = torch.contiguous_format);  permute_175 = None
    view_191: "f32[32, 64, 576]" = torch.ops.aten.view.default(clone_75, [32, 64, 576]);  clone_75 = None
    view_192: "f32[2048, 576]" = torch.ops.aten.view.default(view_191, [2048, 576]);  view_191 = None
    permute_176: "f32[576, 192]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    mm_40: "f32[2048, 192]" = torch.ops.aten.mm.default(view_192, permute_176);  permute_176 = None
    permute_177: "f32[576, 2048]" = torch.ops.aten.permute.default(view_192, [1, 0])
    mm_41: "f32[576, 192]" = torch.ops.aten.mm.default(permute_177, view_49);  permute_177 = view_49 = None
    permute_178: "f32[192, 576]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_83: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_192, [0], True);  view_192 = None
    view_193: "f32[576]" = torch.ops.aten.view.default(sum_83, [576]);  sum_83 = None
    permute_179: "f32[576, 192]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
    view_194: "f32[32, 64, 192]" = torch.ops.aten.view.default(mm_40, [32, 64, 192]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_135: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_142, getitem_113);  add_142 = getitem_113 = None
    mul_497: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_135, rsqrt_32);  sub_135 = None
    mul_498: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_194, primals_140);  primals_140 = None
    mul_499: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_498, 192)
    sum_84: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_498, [2], True)
    mul_500: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_498, mul_497);  mul_498 = None
    sum_85: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_500, [2], True);  mul_500 = None
    mul_501: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_497, sum_85);  sum_85 = None
    sub_136: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(mul_499, sum_84);  mul_499 = sum_84 = None
    sub_137: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(sub_136, mul_501);  sub_136 = mul_501 = None
    div_12: "f32[32, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 192);  rsqrt_32 = None
    mul_502: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(div_12, sub_137);  div_12 = sub_137 = None
    mul_503: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_194, mul_497);  mul_497 = None
    sum_86: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_503, [0, 1]);  mul_503 = None
    sum_87: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_194, [0, 1]);  view_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_245: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_244, mul_502);  add_244 = mul_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_195: "f32[2048, 192]" = torch.ops.aten.view.default(add_245, [2048, 192])
    permute_180: "f32[192, 384]" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
    mm_42: "f32[2048, 384]" = torch.ops.aten.mm.default(view_195, permute_180);  permute_180 = None
    permute_181: "f32[192, 2048]" = torch.ops.aten.permute.default(view_195, [1, 0])
    mm_43: "f32[192, 384]" = torch.ops.aten.mm.default(permute_181, view_47);  permute_181 = view_47 = None
    permute_182: "f32[384, 192]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_88: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_195, [0], True);  view_195 = None
    view_196: "f32[192]" = torch.ops.aten.view.default(sum_88, [192]);  sum_88 = None
    permute_183: "f32[192, 384]" = torch.ops.aten.permute.default(permute_182, [1, 0]);  permute_182 = None
    view_197: "f32[32, 64, 384]" = torch.ops.aten.view.default(mm_42, [32, 64, 384]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_47: "f32[32, 64, 384]" = torch.ops.aten.sigmoid.default(view_46)
    full_13: "f32[32, 64, 384]" = torch.ops.aten.full.default([32, 64, 384], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_138: "f32[32, 64, 384]" = torch.ops.aten.sub.Tensor(full_13, sigmoid_47);  full_13 = None
    mul_504: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(view_46, sub_138);  view_46 = sub_138 = None
    add_246: "f32[32, 64, 384]" = torch.ops.aten.add.Scalar(mul_504, 1);  mul_504 = None
    mul_505: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(sigmoid_47, add_246);  sigmoid_47 = add_246 = None
    mul_506: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(view_197, mul_505);  view_197 = mul_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_198: "f32[2048, 384]" = torch.ops.aten.view.default(mul_506, [2048, 384]);  mul_506 = None
    permute_185: "f32[384, 192]" = torch.ops.aten.permute.default(permute_28, [1, 0]);  permute_28 = None
    mm_44: "f32[2048, 192]" = torch.ops.aten.mm.default(view_198, permute_185);  permute_185 = None
    permute_186: "f32[384, 2048]" = torch.ops.aten.permute.default(view_198, [1, 0])
    mm_45: "f32[384, 192]" = torch.ops.aten.mm.default(permute_186, view_45);  permute_186 = view_45 = None
    permute_187: "f32[192, 384]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_89: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_198, [0], True);  view_198 = None
    view_199: "f32[384]" = torch.ops.aten.view.default(sum_89, [384]);  sum_89 = None
    permute_188: "f32[384, 192]" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
    view_200: "f32[32, 64, 192]" = torch.ops.aten.view.default(mm_44, [32, 64, 192]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_139: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_139, getitem_111);  add_139 = getitem_111 = None
    mul_507: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_139, rsqrt_31);  sub_139 = None
    mul_508: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_200, primals_134);  primals_134 = None
    mul_509: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_508, 192)
    sum_90: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_508, [2], True)
    mul_510: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_508, mul_507);  mul_508 = None
    sum_91: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_510, [2], True);  mul_510 = None
    mul_511: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_507, sum_91);  sum_91 = None
    sub_140: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(mul_509, sum_90);  mul_509 = sum_90 = None
    sub_141: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(sub_140, mul_511);  sub_140 = mul_511 = None
    div_13: "f32[32, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 192);  rsqrt_31 = None
    mul_512: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(div_13, sub_141);  div_13 = sub_141 = None
    mul_513: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_200, mul_507);  mul_507 = None
    sum_92: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_513, [0, 1]);  mul_513 = None
    sum_93: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_200, [0, 1]);  view_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_247: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_245, mul_512);  add_245 = mul_512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_201: "f32[2048, 192]" = torch.ops.aten.view.default(add_247, [2048, 192])
    permute_189: "f32[192, 192]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
    mm_46: "f32[2048, 192]" = torch.ops.aten.mm.default(view_201, permute_189);  permute_189 = None
    permute_190: "f32[192, 2048]" = torch.ops.aten.permute.default(view_201, [1, 0])
    mm_47: "f32[192, 192]" = torch.ops.aten.mm.default(permute_190, view_43);  permute_190 = view_43 = None
    permute_191: "f32[192, 192]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_94: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_201, [0], True);  view_201 = None
    view_202: "f32[192]" = torch.ops.aten.view.default(sum_94, [192]);  sum_94 = None
    permute_192: "f32[192, 192]" = torch.ops.aten.permute.default(permute_191, [1, 0]);  permute_191 = None
    view_203: "f32[32, 64, 192]" = torch.ops.aten.view.default(mm_46, [32, 64, 192]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_204: "f32[32, 64, 4, 48]" = torch.ops.aten.view.default(view_203, [32, 64, 4, 48]);  view_203 = None
    permute_193: "f32[32, 4, 64, 48]" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_14: "f32[32, 4, 64, 48]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    _scaled_dot_product_flash_attention_backward_5 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_193, getitem_98, getitem_99, getitem_100, alias_14, getitem_102, getitem_103, getitem_104, 0, 0, 0.0, False, getitem_107, getitem_108);  permute_193 = getitem_98 = getitem_99 = getitem_100 = alias_14 = getitem_102 = getitem_103 = getitem_104 = getitem_107 = getitem_108 = None
    getitem_259: "f32[32, 4, 64, 48]" = _scaled_dot_product_flash_attention_backward_5[0]
    getitem_260: "f32[32, 4, 64, 48]" = _scaled_dot_product_flash_attention_backward_5[1]
    getitem_261: "f32[32, 4, 64, 48]" = _scaled_dot_product_flash_attention_backward_5[2];  _scaled_dot_product_flash_attention_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_8: "f32[96, 4, 64, 48]" = torch.ops.aten.cat.default([getitem_259, getitem_260, getitem_261]);  getitem_259 = getitem_260 = getitem_261 = None
    view_205: "f32[3, 32, 4, 64, 48]" = torch.ops.aten.view.default(cat_8, [3, 32, 4, 64, 48]);  cat_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_194: "f32[32, 64, 3, 4, 48]" = torch.ops.aten.permute.default(view_205, [1, 3, 0, 2, 4]);  view_205 = None
    clone_76: "f32[32, 64, 3, 4, 48]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
    view_206: "f32[32, 64, 576]" = torch.ops.aten.view.default(clone_76, [32, 64, 576]);  clone_76 = None
    view_207: "f32[2048, 576]" = torch.ops.aten.view.default(view_206, [2048, 576]);  view_206 = None
    permute_195: "f32[576, 192]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    mm_48: "f32[2048, 192]" = torch.ops.aten.mm.default(view_207, permute_195);  permute_195 = None
    permute_196: "f32[576, 2048]" = torch.ops.aten.permute.default(view_207, [1, 0])
    mm_49: "f32[576, 192]" = torch.ops.aten.mm.default(permute_196, view_39);  permute_196 = view_39 = None
    permute_197: "f32[192, 576]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_95: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_207, [0], True);  view_207 = None
    view_208: "f32[576]" = torch.ops.aten.view.default(sum_95, [576]);  sum_95 = None
    permute_198: "f32[576, 192]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
    view_209: "f32[32, 64, 192]" = torch.ops.aten.view.default(mm_48, [32, 64, 192]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_142: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_136, getitem_97);  add_136 = getitem_97 = None
    mul_514: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_142, rsqrt_30);  sub_142 = None
    mul_515: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_209, primals_128);  primals_128 = None
    mul_516: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_515, 192)
    sum_96: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_515, [2], True)
    mul_517: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_515, mul_514);  mul_515 = None
    sum_97: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_517, [2], True);  mul_517 = None
    mul_518: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_514, sum_97);  sum_97 = None
    sub_143: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(mul_516, sum_96);  mul_516 = sum_96 = None
    sub_144: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(sub_143, mul_518);  sub_143 = mul_518 = None
    div_14: "f32[32, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 192);  rsqrt_30 = None
    mul_519: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(div_14, sub_144);  div_14 = sub_144 = None
    mul_520: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_209, mul_514);  mul_514 = None
    sum_98: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_520, [0, 1]);  mul_520 = None
    sum_99: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_209, [0, 1]);  view_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_248: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_247, mul_519);  add_247 = mul_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_210: "f32[2048, 192]" = torch.ops.aten.view.default(add_248, [2048, 192])
    permute_199: "f32[192, 384]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    mm_50: "f32[2048, 384]" = torch.ops.aten.mm.default(view_210, permute_199);  permute_199 = None
    permute_200: "f32[192, 2048]" = torch.ops.aten.permute.default(view_210, [1, 0])
    mm_51: "f32[192, 384]" = torch.ops.aten.mm.default(permute_200, view_37);  permute_200 = view_37 = None
    permute_201: "f32[384, 192]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_100: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_210, [0], True);  view_210 = None
    view_211: "f32[192]" = torch.ops.aten.view.default(sum_100, [192]);  sum_100 = None
    permute_202: "f32[192, 384]" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
    view_212: "f32[32, 64, 384]" = torch.ops.aten.view.default(mm_50, [32, 64, 384]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_48: "f32[32, 64, 384]" = torch.ops.aten.sigmoid.default(view_36)
    full_14: "f32[32, 64, 384]" = torch.ops.aten.full.default([32, 64, 384], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_145: "f32[32, 64, 384]" = torch.ops.aten.sub.Tensor(full_14, sigmoid_48);  full_14 = None
    mul_521: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(view_36, sub_145);  view_36 = sub_145 = None
    add_249: "f32[32, 64, 384]" = torch.ops.aten.add.Scalar(mul_521, 1);  mul_521 = None
    mul_522: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(sigmoid_48, add_249);  sigmoid_48 = add_249 = None
    mul_523: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(view_212, mul_522);  view_212 = mul_522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_213: "f32[2048, 384]" = torch.ops.aten.view.default(mul_523, [2048, 384]);  mul_523 = None
    permute_204: "f32[384, 192]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    mm_52: "f32[2048, 192]" = torch.ops.aten.mm.default(view_213, permute_204);  permute_204 = None
    permute_205: "f32[384, 2048]" = torch.ops.aten.permute.default(view_213, [1, 0])
    mm_53: "f32[384, 192]" = torch.ops.aten.mm.default(permute_205, view_35);  permute_205 = view_35 = None
    permute_206: "f32[192, 384]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_101: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_213, [0], True);  view_213 = None
    view_214: "f32[384]" = torch.ops.aten.view.default(sum_101, [384]);  sum_101 = None
    permute_207: "f32[384, 192]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
    view_215: "f32[32, 64, 192]" = torch.ops.aten.view.default(mm_52, [32, 64, 192]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_146: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_133, getitem_95);  add_133 = getitem_95 = None
    mul_524: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_146, rsqrt_29);  sub_146 = None
    mul_525: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_215, primals_122);  primals_122 = None
    mul_526: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_525, 192)
    sum_102: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_525, [2], True)
    mul_527: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_525, mul_524);  mul_525 = None
    sum_103: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_527, [2], True);  mul_527 = None
    mul_528: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_524, sum_103);  sum_103 = None
    sub_147: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(mul_526, sum_102);  mul_526 = sum_102 = None
    sub_148: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(sub_147, mul_528);  sub_147 = mul_528 = None
    div_15: "f32[32, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 192);  rsqrt_29 = None
    mul_529: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(div_15, sub_148);  div_15 = sub_148 = None
    mul_530: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_215, mul_524);  mul_524 = None
    sum_104: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_530, [0, 1]);  mul_530 = None
    sum_105: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_215, [0, 1]);  view_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_250: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_248, mul_529);  add_248 = mul_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_216: "f32[2048, 192]" = torch.ops.aten.view.default(add_250, [2048, 192])
    permute_208: "f32[192, 192]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    mm_54: "f32[2048, 192]" = torch.ops.aten.mm.default(view_216, permute_208);  permute_208 = None
    permute_209: "f32[192, 2048]" = torch.ops.aten.permute.default(view_216, [1, 0])
    mm_55: "f32[192, 192]" = torch.ops.aten.mm.default(permute_209, view_33);  permute_209 = view_33 = None
    permute_210: "f32[192, 192]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_106: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_216, [0], True);  view_216 = None
    view_217: "f32[192]" = torch.ops.aten.view.default(sum_106, [192]);  sum_106 = None
    permute_211: "f32[192, 192]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
    view_218: "f32[32, 64, 192]" = torch.ops.aten.view.default(mm_54, [32, 64, 192]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_219: "f32[32, 64, 4, 48]" = torch.ops.aten.view.default(view_218, [32, 64, 4, 48]);  view_218 = None
    permute_212: "f32[32, 4, 64, 48]" = torch.ops.aten.permute.default(view_219, [0, 2, 1, 3]);  view_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_15: "f32[32, 4, 64, 48]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    _scaled_dot_product_flash_attention_backward_6 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_212, getitem_82, getitem_83, getitem_84, alias_15, getitem_86, getitem_87, getitem_88, 0, 0, 0.0, False, getitem_91, getitem_92);  permute_212 = getitem_82 = getitem_83 = getitem_84 = alias_15 = getitem_86 = getitem_87 = getitem_88 = getitem_91 = getitem_92 = None
    getitem_262: "f32[32, 4, 64, 48]" = _scaled_dot_product_flash_attention_backward_6[0]
    getitem_263: "f32[32, 4, 64, 48]" = _scaled_dot_product_flash_attention_backward_6[1]
    getitem_264: "f32[32, 4, 64, 48]" = _scaled_dot_product_flash_attention_backward_6[2];  _scaled_dot_product_flash_attention_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_9: "f32[96, 4, 64, 48]" = torch.ops.aten.cat.default([getitem_262, getitem_263, getitem_264]);  getitem_262 = getitem_263 = getitem_264 = None
    view_220: "f32[3, 32, 4, 64, 48]" = torch.ops.aten.view.default(cat_9, [3, 32, 4, 64, 48]);  cat_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_213: "f32[32, 64, 3, 4, 48]" = torch.ops.aten.permute.default(view_220, [1, 3, 0, 2, 4]);  view_220 = None
    clone_77: "f32[32, 64, 3, 4, 48]" = torch.ops.aten.clone.default(permute_213, memory_format = torch.contiguous_format);  permute_213 = None
    view_221: "f32[32, 64, 576]" = torch.ops.aten.view.default(clone_77, [32, 64, 576]);  clone_77 = None
    view_222: "f32[2048, 576]" = torch.ops.aten.view.default(view_221, [2048, 576]);  view_221 = None
    permute_214: "f32[576, 192]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
    mm_56: "f32[2048, 192]" = torch.ops.aten.mm.default(view_222, permute_214);  permute_214 = None
    permute_215: "f32[576, 2048]" = torch.ops.aten.permute.default(view_222, [1, 0])
    mm_57: "f32[576, 192]" = torch.ops.aten.mm.default(permute_215, view_29);  permute_215 = view_29 = None
    permute_216: "f32[192, 576]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_107: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_222, [0], True);  view_222 = None
    view_223: "f32[576]" = torch.ops.aten.view.default(sum_107, [576]);  sum_107 = None
    permute_217: "f32[576, 192]" = torch.ops.aten.permute.default(permute_216, [1, 0]);  permute_216 = None
    view_224: "f32[32, 64, 192]" = torch.ops.aten.view.default(mm_56, [32, 64, 192]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_149: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(view_28, getitem_81);  view_28 = getitem_81 = None
    mul_531: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_149, rsqrt_28);  sub_149 = None
    mul_532: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_224, primals_116);  primals_116 = None
    mul_533: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_532, 192)
    sum_108: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_532, [2], True)
    mul_534: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_532, mul_531);  mul_532 = None
    sum_109: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_534, [2], True);  mul_534 = None
    mul_535: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_531, sum_109);  sum_109 = None
    sub_150: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(mul_533, sum_108);  mul_533 = sum_108 = None
    sub_151: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(sub_150, mul_535);  sub_150 = mul_535 = None
    div_16: "f32[32, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 192);  rsqrt_28 = None
    mul_536: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(div_16, sub_151);  div_16 = sub_151 = None
    mul_537: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_224, mul_531);  mul_531 = None
    sum_110: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_537, [0, 1]);  mul_537 = None
    sum_111: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_224, [0, 1]);  view_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_251: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_250, mul_536);  add_250 = mul_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:251, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
    view_225: "f32[8, 4, 64, 192]" = torch.ops.aten.view.default(add_251, [8, 4, 64, 192]);  add_251 = None
    permute_218: "f32[8, 192, 64, 4]" = torch.ops.aten.permute.default(view_225, [0, 3, 2, 1]);  view_225 = None
    clone_78: "f32[8, 192, 64, 4]" = torch.ops.aten.clone.default(permute_218, memory_format = torch.contiguous_format);  permute_218 = None
    view_226: "f32[12288, 8, 2, 2]" = torch.ops.aten.view.default(clone_78, [12288, 8, 2, 2]);  clone_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:249, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
    permute_219: "f32[12288, 2, 8, 2]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
    clone_79: "f32[12288, 2, 8, 2]" = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None
    view_227: "f32[8, 192, 16, 16]" = torch.ops.aten.view.default(clone_79, [8, 192, 16, 16]);  clone_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:234, code: x = self.conv_1x1(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(view_227, mul_189, primals_115, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_227 = mul_189 = primals_115 = None
    getitem_265: "f32[8, 128, 16, 16]" = convolution_backward_10[0]
    getitem_266: "f32[192, 128, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_49: "f32[8, 128, 16, 16]" = torch.ops.aten.sigmoid.default(clone_26)
    full_15: "f32[8, 128, 16, 16]" = torch.ops.aten.full.default([8, 128, 16, 16], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_152: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(full_15, sigmoid_49);  full_15 = None
    mul_538: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(clone_26, sub_152);  clone_26 = sub_152 = None
    add_252: "f32[8, 128, 16, 16]" = torch.ops.aten.add.Scalar(mul_538, 1);  mul_538 = None
    mul_539: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_49, add_252);  sigmoid_49 = add_252 = None
    mul_540: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_265, mul_539);  getitem_265 = mul_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_236: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_237: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 2);  unsqueeze_236 = None
    unsqueeze_238: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_237, 3);  unsqueeze_237 = None
    sum_112: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_540, [0, 2, 3])
    sub_153: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_238)
    mul_541: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(mul_540, sub_153);  sub_153 = None
    sum_113: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_541, [0, 2, 3]);  mul_541 = None
    mul_542: "f32[128]" = torch.ops.aten.mul.Tensor(sum_112, 0.00048828125)
    unsqueeze_239: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_542, 0);  mul_542 = None
    unsqueeze_240: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 2);  unsqueeze_239 = None
    unsqueeze_241: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, 3);  unsqueeze_240 = None
    mul_543: "f32[128]" = torch.ops.aten.mul.Tensor(sum_113, 0.00048828125)
    mul_544: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_545: "f32[128]" = torch.ops.aten.mul.Tensor(mul_543, mul_544);  mul_543 = mul_544 = None
    unsqueeze_242: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_545, 0);  mul_545 = None
    unsqueeze_243: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 2);  unsqueeze_242 = None
    unsqueeze_244: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_243, 3);  unsqueeze_243 = None
    mul_546: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_45);  primals_45 = None
    unsqueeze_245: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_546, 0);  mul_546 = None
    unsqueeze_246: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 2);  unsqueeze_245 = None
    unsqueeze_247: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, 3);  unsqueeze_246 = None
    sub_154: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_238);  convolution_23 = unsqueeze_238 = None
    mul_547: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_244);  sub_154 = unsqueeze_244 = None
    sub_155: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(mul_540, mul_547);  mul_540 = mul_547 = None
    sub_156: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(sub_155, unsqueeze_241);  sub_155 = unsqueeze_241 = None
    mul_548: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_247);  sub_156 = unsqueeze_247 = None
    mul_549: "f32[128]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_67);  sum_113 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_548, add_125, primals_114, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_548 = add_125 = primals_114 = None
    getitem_268: "f32[8, 128, 16, 16]" = convolution_backward_11[0]
    getitem_269: "f32[128, 128, 3, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_253: "f32[8, 128, 16, 16]" = torch.ops.aten.add.Tensor(slice_3, getitem_268);  slice_3 = getitem_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_248: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_249: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 2);  unsqueeze_248 = None
    unsqueeze_250: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 3);  unsqueeze_249 = None
    sum_114: "f32[128]" = torch.ops.aten.sum.dim_IntList(add_253, [0, 2, 3])
    sub_157: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_250)
    mul_550: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(add_253, sub_157);  sub_157 = None
    sum_115: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_550, [0, 2, 3]);  mul_550 = None
    mul_551: "f32[128]" = torch.ops.aten.mul.Tensor(sum_114, 0.00048828125)
    unsqueeze_251: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_551, 0);  mul_551 = None
    unsqueeze_252: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 2);  unsqueeze_251 = None
    unsqueeze_253: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, 3);  unsqueeze_252 = None
    mul_552: "f32[128]" = torch.ops.aten.mul.Tensor(sum_115, 0.00048828125)
    mul_553: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_554: "f32[128]" = torch.ops.aten.mul.Tensor(mul_552, mul_553);  mul_552 = mul_553 = None
    unsqueeze_254: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_554, 0);  mul_554 = None
    unsqueeze_255: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 2);  unsqueeze_254 = None
    unsqueeze_256: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 3);  unsqueeze_255 = None
    mul_555: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_43);  primals_43 = None
    unsqueeze_257: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_555, 0);  mul_555 = None
    unsqueeze_258: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 2);  unsqueeze_257 = None
    unsqueeze_259: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, 3);  unsqueeze_258 = None
    sub_158: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_250);  convolution_22 = unsqueeze_250 = None
    mul_556: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_256);  sub_158 = unsqueeze_256 = None
    sub_159: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(add_253, mul_556);  add_253 = mul_556 = None
    sub_160: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(sub_159, unsqueeze_253);  sub_159 = unsqueeze_253 = None
    mul_557: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_259);  sub_160 = unsqueeze_259 = None
    mul_558: "f32[128]" = torch.ops.aten.mul.Tensor(sum_115, squeeze_64);  sum_115 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_557, mul_174, primals_113, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_557 = mul_174 = primals_113 = None
    getitem_271: "f32[8, 384, 16, 16]" = convolution_backward_12[0]
    getitem_272: "f32[128, 384, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_50: "f32[8, 384, 16, 16]" = torch.ops.aten.sigmoid.default(clone_25)
    full_16: "f32[8, 384, 16, 16]" = torch.ops.aten.full.default([8, 384, 16, 16], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_161: "f32[8, 384, 16, 16]" = torch.ops.aten.sub.Tensor(full_16, sigmoid_50);  full_16 = None
    mul_559: "f32[8, 384, 16, 16]" = torch.ops.aten.mul.Tensor(clone_25, sub_161);  clone_25 = sub_161 = None
    add_254: "f32[8, 384, 16, 16]" = torch.ops.aten.add.Scalar(mul_559, 1);  mul_559 = None
    mul_560: "f32[8, 384, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_50, add_254);  sigmoid_50 = add_254 = None
    mul_561: "f32[8, 384, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_271, mul_560);  getitem_271 = mul_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_260: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_261: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 2);  unsqueeze_260 = None
    unsqueeze_262: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 3);  unsqueeze_261 = None
    sum_116: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_561, [0, 2, 3])
    sub_162: "f32[8, 384, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_262)
    mul_562: "f32[8, 384, 16, 16]" = torch.ops.aten.mul.Tensor(mul_561, sub_162);  sub_162 = None
    sum_117: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_562, [0, 2, 3]);  mul_562 = None
    mul_563: "f32[384]" = torch.ops.aten.mul.Tensor(sum_116, 0.00048828125)
    unsqueeze_263: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_563, 0);  mul_563 = None
    unsqueeze_264: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 2);  unsqueeze_263 = None
    unsqueeze_265: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 3);  unsqueeze_264 = None
    mul_564: "f32[384]" = torch.ops.aten.mul.Tensor(sum_117, 0.00048828125)
    mul_565: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_566: "f32[384]" = torch.ops.aten.mul.Tensor(mul_564, mul_565);  mul_564 = mul_565 = None
    unsqueeze_266: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_566, 0);  mul_566 = None
    unsqueeze_267: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 2);  unsqueeze_266 = None
    unsqueeze_268: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 3);  unsqueeze_267 = None
    mul_567: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_41);  primals_41 = None
    unsqueeze_269: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_567, 0);  mul_567 = None
    unsqueeze_270: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 2);  unsqueeze_269 = None
    unsqueeze_271: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 3);  unsqueeze_270 = None
    sub_163: "f32[8, 384, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_262);  convolution_21 = unsqueeze_262 = None
    mul_568: "f32[8, 384, 16, 16]" = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_268);  sub_163 = unsqueeze_268 = None
    sub_164: "f32[8, 384, 16, 16]" = torch.ops.aten.sub.Tensor(mul_561, mul_568);  mul_561 = mul_568 = None
    sub_165: "f32[8, 384, 16, 16]" = torch.ops.aten.sub.Tensor(sub_164, unsqueeze_265);  sub_164 = unsqueeze_265 = None
    mul_569: "f32[8, 384, 16, 16]" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_271);  sub_165 = unsqueeze_271 = None
    mul_570: "f32[384]" = torch.ops.aten.mul.Tensor(sum_117, squeeze_61);  sum_117 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_569, mul_166, primals_112, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 384, [True, True, False]);  mul_569 = mul_166 = primals_112 = None
    getitem_274: "f32[8, 384, 32, 32]" = convolution_backward_13[0]
    getitem_275: "f32[384, 1, 3, 3]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_51: "f32[8, 384, 32, 32]" = torch.ops.aten.sigmoid.default(clone_24)
    full_17: "f32[8, 384, 32, 32]" = torch.ops.aten.full.default([8, 384, 32, 32], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_166: "f32[8, 384, 32, 32]" = torch.ops.aten.sub.Tensor(full_17, sigmoid_51);  full_17 = None
    mul_571: "f32[8, 384, 32, 32]" = torch.ops.aten.mul.Tensor(clone_24, sub_166);  clone_24 = sub_166 = None
    add_255: "f32[8, 384, 32, 32]" = torch.ops.aten.add.Scalar(mul_571, 1);  mul_571 = None
    mul_572: "f32[8, 384, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_51, add_255);  sigmoid_51 = add_255 = None
    mul_573: "f32[8, 384, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_274, mul_572);  getitem_274 = mul_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_272: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_273: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 2);  unsqueeze_272 = None
    unsqueeze_274: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 3);  unsqueeze_273 = None
    sum_118: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_573, [0, 2, 3])
    sub_167: "f32[8, 384, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_274)
    mul_574: "f32[8, 384, 32, 32]" = torch.ops.aten.mul.Tensor(mul_573, sub_167);  sub_167 = None
    sum_119: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_574, [0, 2, 3]);  mul_574 = None
    mul_575: "f32[384]" = torch.ops.aten.mul.Tensor(sum_118, 0.0001220703125)
    unsqueeze_275: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_575, 0);  mul_575 = None
    unsqueeze_276: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 2);  unsqueeze_275 = None
    unsqueeze_277: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 3);  unsqueeze_276 = None
    mul_576: "f32[384]" = torch.ops.aten.mul.Tensor(sum_119, 0.0001220703125)
    mul_577: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_578: "f32[384]" = torch.ops.aten.mul.Tensor(mul_576, mul_577);  mul_576 = mul_577 = None
    unsqueeze_278: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_578, 0);  mul_578 = None
    unsqueeze_279: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 2);  unsqueeze_278 = None
    unsqueeze_280: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 3);  unsqueeze_279 = None
    mul_579: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_39);  primals_39 = None
    unsqueeze_281: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_579, 0);  mul_579 = None
    unsqueeze_282: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 2);  unsqueeze_281 = None
    unsqueeze_283: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 3);  unsqueeze_282 = None
    sub_168: "f32[8, 384, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_274);  convolution_20 = unsqueeze_274 = None
    mul_580: "f32[8, 384, 32, 32]" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_280);  sub_168 = unsqueeze_280 = None
    sub_169: "f32[8, 384, 32, 32]" = torch.ops.aten.sub.Tensor(mul_573, mul_580);  mul_573 = mul_580 = None
    sub_170: "f32[8, 384, 32, 32]" = torch.ops.aten.sub.Tensor(sub_169, unsqueeze_277);  sub_169 = unsqueeze_277 = None
    mul_581: "f32[8, 384, 32, 32]" = torch.ops.aten.mul.Tensor(sub_170, unsqueeze_283);  sub_170 = unsqueeze_283 = None
    mul_582: "f32[384]" = torch.ops.aten.mul.Tensor(sum_119, squeeze_58);  sum_119 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_581, mul_158, primals_111, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_581 = mul_158 = primals_111 = None
    getitem_277: "f32[8, 96, 32, 32]" = convolution_backward_14[0]
    getitem_278: "f32[384, 96, 1, 1]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_52: "f32[8, 96, 32, 32]" = torch.ops.aten.sigmoid.default(clone_23)
    full_18: "f32[8, 96, 32, 32]" = torch.ops.aten.full.default([8, 96, 32, 32], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_171: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(full_18, sigmoid_52);  full_18 = None
    mul_583: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(clone_23, sub_171);  clone_23 = sub_171 = None
    add_256: "f32[8, 96, 32, 32]" = torch.ops.aten.add.Scalar(mul_583, 1);  mul_583 = None
    mul_584: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_52, add_256);  sigmoid_52 = add_256 = None
    mul_585: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_277, mul_584);  getitem_277 = mul_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_284: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_285: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 2);  unsqueeze_284 = None
    unsqueeze_286: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 3);  unsqueeze_285 = None
    sum_120: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_585, [0, 2, 3])
    sub_172: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_286)
    mul_586: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(mul_585, sub_172);  sub_172 = None
    sum_121: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_586, [0, 2, 3]);  mul_586 = None
    mul_587: "f32[96]" = torch.ops.aten.mul.Tensor(sum_120, 0.0001220703125)
    unsqueeze_287: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_587, 0);  mul_587 = None
    unsqueeze_288: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 2);  unsqueeze_287 = None
    unsqueeze_289: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 3);  unsqueeze_288 = None
    mul_588: "f32[96]" = torch.ops.aten.mul.Tensor(sum_121, 0.0001220703125)
    mul_589: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_590: "f32[96]" = torch.ops.aten.mul.Tensor(mul_588, mul_589);  mul_588 = mul_589 = None
    unsqueeze_290: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_590, 0);  mul_590 = None
    unsqueeze_291: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 2);  unsqueeze_290 = None
    unsqueeze_292: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 3);  unsqueeze_291 = None
    mul_591: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_37);  primals_37 = None
    unsqueeze_293: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_591, 0);  mul_591 = None
    unsqueeze_294: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 2);  unsqueeze_293 = None
    unsqueeze_295: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 3);  unsqueeze_294 = None
    sub_173: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_286);  convolution_19 = unsqueeze_286 = None
    mul_592: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_292);  sub_173 = unsqueeze_292 = None
    sub_174: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(mul_585, mul_592);  mul_585 = mul_592 = None
    sub_175: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(sub_174, unsqueeze_289);  sub_174 = unsqueeze_289 = None
    mul_593: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_295);  sub_175 = unsqueeze_295 = None
    mul_594: "f32[96]" = torch.ops.aten.mul.Tensor(sum_121, squeeze_55);  sum_121 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_593, cat, primals_110, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_593 = cat = primals_110 = None
    getitem_280: "f32[8, 192, 32, 32]" = convolution_backward_15[0]
    getitem_281: "f32[96, 192, 3, 3]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:268, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
    slice_5: "f32[8, 96, 32, 32]" = torch.ops.aten.slice.Tensor(getitem_280, 1, 0, 96)
    slice_6: "f32[8, 96, 32, 32]" = torch.ops.aten.slice.Tensor(getitem_280, 1, 96, 192);  getitem_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_53: "f32[8, 96, 32, 32]" = torch.ops.aten.sigmoid.default(clone_22)
    full_19: "f32[8, 96, 32, 32]" = torch.ops.aten.full.default([8, 96, 32, 32], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_176: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(full_19, sigmoid_53);  full_19 = None
    mul_595: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(clone_22, sub_176);  clone_22 = sub_176 = None
    add_257: "f32[8, 96, 32, 32]" = torch.ops.aten.add.Scalar(mul_595, 1);  mul_595 = None
    mul_596: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_53, add_257);  sigmoid_53 = add_257 = None
    mul_597: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(slice_6, mul_596);  slice_6 = mul_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_296: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_297: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 2);  unsqueeze_296 = None
    unsqueeze_298: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 3);  unsqueeze_297 = None
    sum_122: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_597, [0, 2, 3])
    sub_177: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_298)
    mul_598: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(mul_597, sub_177);  sub_177 = None
    sum_123: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_598, [0, 2, 3]);  mul_598 = None
    mul_599: "f32[96]" = torch.ops.aten.mul.Tensor(sum_122, 0.0001220703125)
    unsqueeze_299: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_599, 0);  mul_599 = None
    unsqueeze_300: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 2);  unsqueeze_299 = None
    unsqueeze_301: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 3);  unsqueeze_300 = None
    mul_600: "f32[96]" = torch.ops.aten.mul.Tensor(sum_123, 0.0001220703125)
    mul_601: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_602: "f32[96]" = torch.ops.aten.mul.Tensor(mul_600, mul_601);  mul_600 = mul_601 = None
    unsqueeze_302: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_602, 0);  mul_602 = None
    unsqueeze_303: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 2);  unsqueeze_302 = None
    unsqueeze_304: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 3);  unsqueeze_303 = None
    mul_603: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_35);  primals_35 = None
    unsqueeze_305: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_603, 0);  mul_603 = None
    unsqueeze_306: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 2);  unsqueeze_305 = None
    unsqueeze_307: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, 3);  unsqueeze_306 = None
    sub_178: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_298);  convolution_18 = unsqueeze_298 = None
    mul_604: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_178, unsqueeze_304);  sub_178 = unsqueeze_304 = None
    sub_179: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(mul_597, mul_604);  mul_597 = mul_604 = None
    sub_180: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(sub_179, unsqueeze_301);  sub_179 = unsqueeze_301 = None
    mul_605: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_307);  sub_180 = unsqueeze_307 = None
    mul_606: "f32[96]" = torch.ops.aten.mul.Tensor(sum_123, squeeze_52);  sum_123 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_605, view_25, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_605 = view_25 = primals_109 = None
    getitem_283: "f32[8, 144, 32, 32]" = convolution_backward_16[0]
    getitem_284: "f32[96, 144, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:262, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
    view_228: "f32[18432, 2, 16, 2]" = torch.ops.aten.view.default(getitem_283, [18432, 2, 16, 2]);  getitem_283 = None
    permute_225: "f32[18432, 16, 2, 2]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:260, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
    clone_80: "f32[18432, 16, 2, 2]" = torch.ops.aten.clone.default(permute_225, memory_format = torch.contiguous_format);  permute_225 = None
    view_229: "f32[8, 144, 256, 4]" = torch.ops.aten.view.default(clone_80, [8, 144, 256, 4]);  clone_80 = None
    permute_226: "f32[8, 4, 256, 144]" = torch.ops.aten.permute.default(view_229, [0, 3, 2, 1]);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:259, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
    clone_81: "f32[8, 4, 256, 144]" = torch.ops.aten.clone.default(permute_226, memory_format = torch.contiguous_format);  permute_226 = None
    view_230: "f32[32, 256, 144]" = torch.ops.aten.view.default(clone_81, [32, 256, 144]);  clone_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:255, code: x = self.norm(x)
    sub_181: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(add_98, getitem_67);  add_98 = getitem_67 = None
    mul_607: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(sub_181, rsqrt_21);  sub_181 = None
    mul_608: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(view_230, primals_107);  primals_107 = None
    mul_609: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_608, 144)
    sum_124: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_608, [2], True)
    mul_610: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_608, mul_607);  mul_608 = None
    sum_125: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_610, [2], True);  mul_610 = None
    mul_611: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_607, sum_125);  sum_125 = None
    sub_182: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(mul_609, sum_124);  mul_609 = sum_124 = None
    sub_183: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(sub_182, mul_611);  sub_182 = mul_611 = None
    div_17: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 144);  rsqrt_21 = None
    mul_612: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(div_17, sub_183);  div_17 = sub_183 = None
    mul_613: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(view_230, mul_607);  mul_607 = None
    sum_126: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_613, [0, 1]);  mul_613 = None
    sum_127: "f32[144]" = torch.ops.aten.sum.dim_IntList(view_230, [0, 1]);  view_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_231: "f32[8192, 144]" = torch.ops.aten.view.default(mul_612, [8192, 144])
    permute_227: "f32[144, 288]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    mm_58: "f32[8192, 288]" = torch.ops.aten.mm.default(view_231, permute_227);  permute_227 = None
    permute_228: "f32[144, 8192]" = torch.ops.aten.permute.default(view_231, [1, 0])
    mm_59: "f32[144, 288]" = torch.ops.aten.mm.default(permute_228, view_21);  permute_228 = view_21 = None
    permute_229: "f32[288, 144]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_128: "f32[1, 144]" = torch.ops.aten.sum.dim_IntList(view_231, [0], True);  view_231 = None
    view_232: "f32[144]" = torch.ops.aten.view.default(sum_128, [144]);  sum_128 = None
    permute_230: "f32[144, 288]" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
    view_233: "f32[32, 256, 288]" = torch.ops.aten.view.default(mm_58, [32, 256, 288]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_54: "f32[32, 256, 288]" = torch.ops.aten.sigmoid.default(view_20)
    full_20: "f32[32, 256, 288]" = torch.ops.aten.full.default([32, 256, 288], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_184: "f32[32, 256, 288]" = torch.ops.aten.sub.Tensor(full_20, sigmoid_54);  full_20 = None
    mul_614: "f32[32, 256, 288]" = torch.ops.aten.mul.Tensor(view_20, sub_184);  view_20 = sub_184 = None
    add_258: "f32[32, 256, 288]" = torch.ops.aten.add.Scalar(mul_614, 1);  mul_614 = None
    mul_615: "f32[32, 256, 288]" = torch.ops.aten.mul.Tensor(sigmoid_54, add_258);  sigmoid_54 = add_258 = None
    mul_616: "f32[32, 256, 288]" = torch.ops.aten.mul.Tensor(view_233, mul_615);  view_233 = mul_615 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_234: "f32[8192, 288]" = torch.ops.aten.view.default(mul_616, [8192, 288]);  mul_616 = None
    permute_232: "f32[288, 144]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    mm_60: "f32[8192, 144]" = torch.ops.aten.mm.default(view_234, permute_232);  permute_232 = None
    permute_233: "f32[288, 8192]" = torch.ops.aten.permute.default(view_234, [1, 0])
    mm_61: "f32[288, 144]" = torch.ops.aten.mm.default(permute_233, view_19);  permute_233 = view_19 = None
    permute_234: "f32[144, 288]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_129: "f32[1, 288]" = torch.ops.aten.sum.dim_IntList(view_234, [0], True);  view_234 = None
    view_235: "f32[288]" = torch.ops.aten.view.default(sum_129, [288]);  sum_129 = None
    permute_235: "f32[288, 144]" = torch.ops.aten.permute.default(permute_234, [1, 0]);  permute_234 = None
    view_236: "f32[32, 256, 144]" = torch.ops.aten.view.default(mm_60, [32, 256, 144]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_185: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(add_95, getitem_65);  add_95 = getitem_65 = None
    mul_617: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(sub_185, rsqrt_20);  sub_185 = None
    mul_618: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(view_236, primals_101);  primals_101 = None
    mul_619: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_618, 144)
    sum_130: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_618, [2], True)
    mul_620: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_618, mul_617);  mul_618 = None
    sum_131: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_620, [2], True);  mul_620 = None
    mul_621: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_617, sum_131);  sum_131 = None
    sub_186: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(mul_619, sum_130);  mul_619 = sum_130 = None
    sub_187: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(sub_186, mul_621);  sub_186 = mul_621 = None
    div_18: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 144);  rsqrt_20 = None
    mul_622: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(div_18, sub_187);  div_18 = sub_187 = None
    mul_623: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(view_236, mul_617);  mul_617 = None
    sum_132: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_623, [0, 1]);  mul_623 = None
    sum_133: "f32[144]" = torch.ops.aten.sum.dim_IntList(view_236, [0, 1]);  view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_259: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(mul_612, mul_622);  mul_612 = mul_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_237: "f32[8192, 144]" = torch.ops.aten.view.default(add_259, [8192, 144])
    permute_236: "f32[144, 144]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    mm_62: "f32[8192, 144]" = torch.ops.aten.mm.default(view_237, permute_236);  permute_236 = None
    permute_237: "f32[144, 8192]" = torch.ops.aten.permute.default(view_237, [1, 0])
    mm_63: "f32[144, 144]" = torch.ops.aten.mm.default(permute_237, view_17);  permute_237 = view_17 = None
    permute_238: "f32[144, 144]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_134: "f32[1, 144]" = torch.ops.aten.sum.dim_IntList(view_237, [0], True);  view_237 = None
    view_238: "f32[144]" = torch.ops.aten.view.default(sum_134, [144]);  sum_134 = None
    permute_239: "f32[144, 144]" = torch.ops.aten.permute.default(permute_238, [1, 0]);  permute_238 = None
    view_239: "f32[32, 256, 144]" = torch.ops.aten.view.default(mm_62, [32, 256, 144]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_240: "f32[32, 256, 4, 36]" = torch.ops.aten.view.default(view_239, [32, 256, 4, 36]);  view_239 = None
    permute_240: "f32[32, 4, 256, 36]" = torch.ops.aten.permute.default(view_240, [0, 2, 1, 3]);  view_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_16: "f32[32, 4, 256, 36]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    _scaled_dot_product_flash_attention_backward_7 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_240, getitem_52, getitem_53, getitem_54, alias_16, getitem_56, getitem_57, getitem_58, 0, 0, 0.0, False, getitem_61, getitem_62);  permute_240 = getitem_52 = getitem_53 = getitem_54 = alias_16 = getitem_56 = getitem_57 = getitem_58 = getitem_61 = getitem_62 = None
    getitem_286: "f32[32, 4, 256, 36]" = _scaled_dot_product_flash_attention_backward_7[0]
    getitem_287: "f32[32, 4, 256, 36]" = _scaled_dot_product_flash_attention_backward_7[1]
    getitem_288: "f32[32, 4, 256, 36]" = _scaled_dot_product_flash_attention_backward_7[2];  _scaled_dot_product_flash_attention_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_10: "f32[96, 4, 256, 36]" = torch.ops.aten.cat.default([getitem_286, getitem_287, getitem_288]);  getitem_286 = getitem_287 = getitem_288 = None
    view_241: "f32[3, 32, 4, 256, 36]" = torch.ops.aten.view.default(cat_10, [3, 32, 4, 256, 36]);  cat_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_241: "f32[32, 256, 3, 4, 36]" = torch.ops.aten.permute.default(view_241, [1, 3, 0, 2, 4]);  view_241 = None
    clone_82: "f32[32, 256, 3, 4, 36]" = torch.ops.aten.clone.default(permute_241, memory_format = torch.contiguous_format);  permute_241 = None
    view_242: "f32[32, 256, 432]" = torch.ops.aten.view.default(clone_82, [32, 256, 432]);  clone_82 = None
    view_243: "f32[8192, 432]" = torch.ops.aten.view.default(view_242, [8192, 432]);  view_242 = None
    permute_242: "f32[432, 144]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    mm_64: "f32[8192, 144]" = torch.ops.aten.mm.default(view_243, permute_242);  permute_242 = None
    permute_243: "f32[432, 8192]" = torch.ops.aten.permute.default(view_243, [1, 0])
    mm_65: "f32[432, 144]" = torch.ops.aten.mm.default(permute_243, view_13);  permute_243 = view_13 = None
    permute_244: "f32[144, 432]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_135: "f32[1, 432]" = torch.ops.aten.sum.dim_IntList(view_243, [0], True);  view_243 = None
    view_244: "f32[432]" = torch.ops.aten.view.default(sum_135, [432]);  sum_135 = None
    permute_245: "f32[432, 144]" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
    view_245: "f32[32, 256, 144]" = torch.ops.aten.view.default(mm_64, [32, 256, 144]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_188: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(add_92, getitem_51);  add_92 = getitem_51 = None
    mul_624: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(sub_188, rsqrt_19);  sub_188 = None
    mul_625: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(view_245, primals_95);  primals_95 = None
    mul_626: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_625, 144)
    sum_136: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_625, [2], True)
    mul_627: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_625, mul_624);  mul_625 = None
    sum_137: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_627, [2], True);  mul_627 = None
    mul_628: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_624, sum_137);  sum_137 = None
    sub_189: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(mul_626, sum_136);  mul_626 = sum_136 = None
    sub_190: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(sub_189, mul_628);  sub_189 = mul_628 = None
    div_19: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 144);  rsqrt_19 = None
    mul_629: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(div_19, sub_190);  div_19 = sub_190 = None
    mul_630: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(view_245, mul_624);  mul_624 = None
    sum_138: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_630, [0, 1]);  mul_630 = None
    sum_139: "f32[144]" = torch.ops.aten.sum.dim_IntList(view_245, [0, 1]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_260: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(add_259, mul_629);  add_259 = mul_629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_246: "f32[8192, 144]" = torch.ops.aten.view.default(add_260, [8192, 144])
    permute_246: "f32[144, 288]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
    mm_66: "f32[8192, 288]" = torch.ops.aten.mm.default(view_246, permute_246);  permute_246 = None
    permute_247: "f32[144, 8192]" = torch.ops.aten.permute.default(view_246, [1, 0])
    mm_67: "f32[144, 288]" = torch.ops.aten.mm.default(permute_247, view_11);  permute_247 = view_11 = None
    permute_248: "f32[288, 144]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_140: "f32[1, 144]" = torch.ops.aten.sum.dim_IntList(view_246, [0], True);  view_246 = None
    view_247: "f32[144]" = torch.ops.aten.view.default(sum_140, [144]);  sum_140 = None
    permute_249: "f32[144, 288]" = torch.ops.aten.permute.default(permute_248, [1, 0]);  permute_248 = None
    view_248: "f32[32, 256, 288]" = torch.ops.aten.view.default(mm_66, [32, 256, 288]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_55: "f32[32, 256, 288]" = torch.ops.aten.sigmoid.default(view_10)
    full_21: "f32[32, 256, 288]" = torch.ops.aten.full.default([32, 256, 288], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_191: "f32[32, 256, 288]" = torch.ops.aten.sub.Tensor(full_21, sigmoid_55);  full_21 = None
    mul_631: "f32[32, 256, 288]" = torch.ops.aten.mul.Tensor(view_10, sub_191);  view_10 = sub_191 = None
    add_261: "f32[32, 256, 288]" = torch.ops.aten.add.Scalar(mul_631, 1);  mul_631 = None
    mul_632: "f32[32, 256, 288]" = torch.ops.aten.mul.Tensor(sigmoid_55, add_261);  sigmoid_55 = add_261 = None
    mul_633: "f32[32, 256, 288]" = torch.ops.aten.mul.Tensor(view_248, mul_632);  view_248 = mul_632 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_249: "f32[8192, 288]" = torch.ops.aten.view.default(mul_633, [8192, 288]);  mul_633 = None
    permute_251: "f32[288, 144]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
    mm_68: "f32[8192, 144]" = torch.ops.aten.mm.default(view_249, permute_251);  permute_251 = None
    permute_252: "f32[288, 8192]" = torch.ops.aten.permute.default(view_249, [1, 0])
    mm_69: "f32[288, 144]" = torch.ops.aten.mm.default(permute_252, view_9);  permute_252 = view_9 = None
    permute_253: "f32[144, 288]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_141: "f32[1, 288]" = torch.ops.aten.sum.dim_IntList(view_249, [0], True);  view_249 = None
    view_250: "f32[288]" = torch.ops.aten.view.default(sum_141, [288]);  sum_141 = None
    permute_254: "f32[288, 144]" = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None
    view_251: "f32[32, 256, 144]" = torch.ops.aten.view.default(mm_68, [32, 256, 144]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_192: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(add_89, getitem_49);  add_89 = getitem_49 = None
    mul_634: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(sub_192, rsqrt_18);  sub_192 = None
    mul_635: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(view_251, primals_89);  primals_89 = None
    mul_636: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_635, 144)
    sum_142: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_635, [2], True)
    mul_637: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_635, mul_634);  mul_635 = None
    sum_143: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_637, [2], True);  mul_637 = None
    mul_638: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_634, sum_143);  sum_143 = None
    sub_193: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(mul_636, sum_142);  mul_636 = sum_142 = None
    sub_194: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(sub_193, mul_638);  sub_193 = mul_638 = None
    div_20: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 144);  rsqrt_18 = None
    mul_639: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(div_20, sub_194);  div_20 = sub_194 = None
    mul_640: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(view_251, mul_634);  mul_634 = None
    sum_144: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_640, [0, 1]);  mul_640 = None
    sum_145: "f32[144]" = torch.ops.aten.sum.dim_IntList(view_251, [0, 1]);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_262: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(add_260, mul_639);  add_260 = mul_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_252: "f32[8192, 144]" = torch.ops.aten.view.default(add_262, [8192, 144])
    permute_255: "f32[144, 144]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
    mm_70: "f32[8192, 144]" = torch.ops.aten.mm.default(view_252, permute_255);  permute_255 = None
    permute_256: "f32[144, 8192]" = torch.ops.aten.permute.default(view_252, [1, 0])
    mm_71: "f32[144, 144]" = torch.ops.aten.mm.default(permute_256, view_7);  permute_256 = view_7 = None
    permute_257: "f32[144, 144]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_146: "f32[1, 144]" = torch.ops.aten.sum.dim_IntList(view_252, [0], True);  view_252 = None
    view_253: "f32[144]" = torch.ops.aten.view.default(sum_146, [144]);  sum_146 = None
    permute_258: "f32[144, 144]" = torch.ops.aten.permute.default(permute_257, [1, 0]);  permute_257 = None
    view_254: "f32[32, 256, 144]" = torch.ops.aten.view.default(mm_70, [32, 256, 144]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_255: "f32[32, 256, 4, 36]" = torch.ops.aten.view.default(view_254, [32, 256, 4, 36]);  view_254 = None
    permute_259: "f32[32, 4, 256, 36]" = torch.ops.aten.permute.default(view_255, [0, 2, 1, 3]);  view_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_17: "f32[32, 4, 256, 36]" = torch.ops.aten.alias.default(alias);  alias = None
    _scaled_dot_product_flash_attention_backward_8 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_259, getitem_36, getitem_37, getitem_38, alias_17, getitem_40, getitem_41, getitem_42, 0, 0, 0.0, False, getitem_45, getitem_46);  permute_259 = getitem_36 = getitem_37 = getitem_38 = alias_17 = getitem_40 = getitem_41 = getitem_42 = getitem_45 = getitem_46 = None
    getitem_289: "f32[32, 4, 256, 36]" = _scaled_dot_product_flash_attention_backward_8[0]
    getitem_290: "f32[32, 4, 256, 36]" = _scaled_dot_product_flash_attention_backward_8[1]
    getitem_291: "f32[32, 4, 256, 36]" = _scaled_dot_product_flash_attention_backward_8[2];  _scaled_dot_product_flash_attention_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_11: "f32[96, 4, 256, 36]" = torch.ops.aten.cat.default([getitem_289, getitem_290, getitem_291]);  getitem_289 = getitem_290 = getitem_291 = None
    view_256: "f32[3, 32, 4, 256, 36]" = torch.ops.aten.view.default(cat_11, [3, 32, 4, 256, 36]);  cat_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_260: "f32[32, 256, 3, 4, 36]" = torch.ops.aten.permute.default(view_256, [1, 3, 0, 2, 4]);  view_256 = None
    clone_83: "f32[32, 256, 3, 4, 36]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
    view_257: "f32[32, 256, 432]" = torch.ops.aten.view.default(clone_83, [32, 256, 432]);  clone_83 = None
    view_258: "f32[8192, 432]" = torch.ops.aten.view.default(view_257, [8192, 432]);  view_257 = None
    permute_261: "f32[432, 144]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    mm_72: "f32[8192, 144]" = torch.ops.aten.mm.default(view_258, permute_261);  permute_261 = None
    permute_262: "f32[432, 8192]" = torch.ops.aten.permute.default(view_258, [1, 0])
    mm_73: "f32[432, 144]" = torch.ops.aten.mm.default(permute_262, view_3);  permute_262 = view_3 = None
    permute_263: "f32[144, 432]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_147: "f32[1, 432]" = torch.ops.aten.sum.dim_IntList(view_258, [0], True);  view_258 = None
    view_259: "f32[432]" = torch.ops.aten.view.default(sum_147, [432]);  sum_147 = None
    permute_264: "f32[432, 144]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
    view_260: "f32[32, 256, 144]" = torch.ops.aten.view.default(mm_72, [32, 256, 144]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_195: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(view_2, getitem_35);  view_2 = getitem_35 = None
    mul_641: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(sub_195, rsqrt_17);  sub_195 = None
    mul_642: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(view_260, primals_83);  primals_83 = None
    mul_643: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_642, 144)
    sum_148: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_642, [2], True)
    mul_644: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_642, mul_641);  mul_642 = None
    sum_149: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_644, [2], True);  mul_644 = None
    mul_645: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_641, sum_149);  sum_149 = None
    sub_196: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(mul_643, sum_148);  mul_643 = sum_148 = None
    sub_197: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(sub_196, mul_645);  sub_196 = mul_645 = None
    div_21: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 144);  rsqrt_17 = None
    mul_646: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(div_21, sub_197);  div_21 = sub_197 = None
    mul_647: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(view_260, mul_641);  mul_641 = None
    sum_150: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_647, [0, 1]);  mul_647 = None
    sum_151: "f32[144]" = torch.ops.aten.sum.dim_IntList(view_260, [0, 1]);  view_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_263: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(add_262, mul_646);  add_262 = mul_646 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:251, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
    view_261: "f32[8, 4, 256, 144]" = torch.ops.aten.view.default(add_263, [8, 4, 256, 144]);  add_263 = None
    permute_265: "f32[8, 144, 256, 4]" = torch.ops.aten.permute.default(view_261, [0, 3, 2, 1]);  view_261 = None
    clone_84: "f32[8, 144, 256, 4]" = torch.ops.aten.clone.default(permute_265, memory_format = torch.contiguous_format);  permute_265 = None
    view_262: "f32[18432, 16, 2, 2]" = torch.ops.aten.view.default(clone_84, [18432, 16, 2, 2]);  clone_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:249, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
    permute_266: "f32[18432, 2, 16, 2]" = torch.ops.aten.permute.default(view_262, [0, 2, 1, 3]);  view_262 = None
    clone_85: "f32[18432, 2, 16, 2]" = torch.ops.aten.clone.default(permute_266, memory_format = torch.contiguous_format);  permute_266 = None
    view_263: "f32[8, 144, 32, 32]" = torch.ops.aten.view.default(clone_85, [8, 144, 32, 32]);  clone_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:234, code: x = self.conv_1x1(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(view_263, mul_130, primals_82, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_263 = mul_130 = primals_82 = None
    getitem_292: "f32[8, 96, 32, 32]" = convolution_backward_17[0]
    getitem_293: "f32[144, 96, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_56: "f32[8, 96, 32, 32]" = torch.ops.aten.sigmoid.default(clone_11)
    full_22: "f32[8, 96, 32, 32]" = torch.ops.aten.full.default([8, 96, 32, 32], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_198: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(full_22, sigmoid_56);  full_22 = None
    mul_648: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(clone_11, sub_198);  clone_11 = sub_198 = None
    add_264: "f32[8, 96, 32, 32]" = torch.ops.aten.add.Scalar(mul_648, 1);  mul_648 = None
    mul_649: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_56, add_264);  sigmoid_56 = add_264 = None
    mul_650: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_292, mul_649);  getitem_292 = mul_649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_308: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_309: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 2);  unsqueeze_308 = None
    unsqueeze_310: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 3);  unsqueeze_309 = None
    sum_152: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_650, [0, 2, 3])
    sub_199: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_310)
    mul_651: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(mul_650, sub_199);  sub_199 = None
    sum_153: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_651, [0, 2, 3]);  mul_651 = None
    mul_652: "f32[96]" = torch.ops.aten.mul.Tensor(sum_152, 0.0001220703125)
    unsqueeze_311: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_652, 0);  mul_652 = None
    unsqueeze_312: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 2);  unsqueeze_311 = None
    unsqueeze_313: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 3);  unsqueeze_312 = None
    mul_653: "f32[96]" = torch.ops.aten.mul.Tensor(sum_153, 0.0001220703125)
    mul_654: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_655: "f32[96]" = torch.ops.aten.mul.Tensor(mul_653, mul_654);  mul_653 = mul_654 = None
    unsqueeze_314: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_655, 0);  mul_655 = None
    unsqueeze_315: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 2);  unsqueeze_314 = None
    unsqueeze_316: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 3);  unsqueeze_315 = None
    mul_656: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_33);  primals_33 = None
    unsqueeze_317: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_656, 0);  mul_656 = None
    unsqueeze_318: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None
    unsqueeze_319: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, 3);  unsqueeze_318 = None
    sub_200: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_310);  convolution_16 = unsqueeze_310 = None
    mul_657: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_316);  sub_200 = unsqueeze_316 = None
    sub_201: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(mul_650, mul_657);  mul_650 = mul_657 = None
    sub_202: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(sub_201, unsqueeze_313);  sub_201 = unsqueeze_313 = None
    mul_658: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_319);  sub_202 = unsqueeze_319 = None
    mul_659: "f32[96]" = torch.ops.aten.mul.Tensor(sum_153, squeeze_49);  sum_153 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_658, add_81, primals_81, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_658 = add_81 = primals_81 = None
    getitem_295: "f32[8, 96, 32, 32]" = convolution_backward_18[0]
    getitem_296: "f32[96, 96, 3, 3]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_265: "f32[8, 96, 32, 32]" = torch.ops.aten.add.Tensor(slice_5, getitem_295);  slice_5 = getitem_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_320: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_321: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 2);  unsqueeze_320 = None
    unsqueeze_322: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 3);  unsqueeze_321 = None
    sum_154: "f32[96]" = torch.ops.aten.sum.dim_IntList(add_265, [0, 2, 3])
    sub_203: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_322)
    mul_660: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(add_265, sub_203);  sub_203 = None
    sum_155: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_660, [0, 2, 3]);  mul_660 = None
    mul_661: "f32[96]" = torch.ops.aten.mul.Tensor(sum_154, 0.0001220703125)
    unsqueeze_323: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_661, 0);  mul_661 = None
    unsqueeze_324: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 2);  unsqueeze_323 = None
    unsqueeze_325: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, 3);  unsqueeze_324 = None
    mul_662: "f32[96]" = torch.ops.aten.mul.Tensor(sum_155, 0.0001220703125)
    mul_663: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_664: "f32[96]" = torch.ops.aten.mul.Tensor(mul_662, mul_663);  mul_662 = mul_663 = None
    unsqueeze_326: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_664, 0);  mul_664 = None
    unsqueeze_327: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 2);  unsqueeze_326 = None
    unsqueeze_328: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 3);  unsqueeze_327 = None
    mul_665: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_31);  primals_31 = None
    unsqueeze_329: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_665, 0);  mul_665 = None
    unsqueeze_330: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 2);  unsqueeze_329 = None
    unsqueeze_331: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 3);  unsqueeze_330 = None
    sub_204: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_322);  convolution_15 = unsqueeze_322 = None
    mul_666: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_328);  sub_204 = unsqueeze_328 = None
    sub_205: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(add_265, mul_666);  add_265 = mul_666 = None
    sub_206: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(sub_205, unsqueeze_325);  sub_205 = unsqueeze_325 = None
    mul_667: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_331);  sub_206 = unsqueeze_331 = None
    mul_668: "f32[96]" = torch.ops.aten.mul.Tensor(sum_155, squeeze_46);  sum_155 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_667, mul_115, primals_80, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_667 = mul_115 = primals_80 = None
    getitem_298: "f32[8, 256, 32, 32]" = convolution_backward_19[0]
    getitem_299: "f32[96, 256, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_57: "f32[8, 256, 32, 32]" = torch.ops.aten.sigmoid.default(clone_10)
    full_23: "f32[8, 256, 32, 32]" = torch.ops.aten.full.default([8, 256, 32, 32], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_207: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(full_23, sigmoid_57);  full_23 = None
    mul_669: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(clone_10, sub_207);  clone_10 = sub_207 = None
    add_266: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Scalar(mul_669, 1);  mul_669 = None
    mul_670: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_57, add_266);  sigmoid_57 = add_266 = None
    mul_671: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_298, mul_670);  getitem_298 = mul_670 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_332: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_333: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 2);  unsqueeze_332 = None
    unsqueeze_334: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 3);  unsqueeze_333 = None
    sum_156: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_671, [0, 2, 3])
    sub_208: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_334)
    mul_672: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_671, sub_208);  sub_208 = None
    sum_157: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_672, [0, 2, 3]);  mul_672 = None
    mul_673: "f32[256]" = torch.ops.aten.mul.Tensor(sum_156, 0.0001220703125)
    unsqueeze_335: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_673, 0);  mul_673 = None
    unsqueeze_336: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 2);  unsqueeze_335 = None
    unsqueeze_337: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 3);  unsqueeze_336 = None
    mul_674: "f32[256]" = torch.ops.aten.mul.Tensor(sum_157, 0.0001220703125)
    mul_675: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_676: "f32[256]" = torch.ops.aten.mul.Tensor(mul_674, mul_675);  mul_674 = mul_675 = None
    unsqueeze_338: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_676, 0);  mul_676 = None
    unsqueeze_339: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 2);  unsqueeze_338 = None
    unsqueeze_340: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 3);  unsqueeze_339 = None
    mul_677: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_29);  primals_29 = None
    unsqueeze_341: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_677, 0);  mul_677 = None
    unsqueeze_342: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 2);  unsqueeze_341 = None
    unsqueeze_343: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 3);  unsqueeze_342 = None
    sub_209: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_334);  convolution_14 = unsqueeze_334 = None
    mul_678: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_340);  sub_209 = unsqueeze_340 = None
    sub_210: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(mul_671, mul_678);  mul_671 = mul_678 = None
    sub_211: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(sub_210, unsqueeze_337);  sub_210 = unsqueeze_337 = None
    mul_679: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_343);  sub_211 = unsqueeze_343 = None
    mul_680: "f32[256]" = torch.ops.aten.mul.Tensor(sum_157, squeeze_43);  sum_157 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_679, mul_107, primals_79, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 256, [True, True, False]);  mul_679 = mul_107 = primals_79 = None
    getitem_301: "f32[8, 256, 64, 64]" = convolution_backward_20[0]
    getitem_302: "f32[256, 1, 3, 3]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_58: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(clone_9)
    full_24: "f32[8, 256, 64, 64]" = torch.ops.aten.full.default([8, 256, 64, 64], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_212: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(full_24, sigmoid_58);  full_24 = None
    mul_681: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(clone_9, sub_212);  clone_9 = sub_212 = None
    add_267: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Scalar(mul_681, 1);  mul_681 = None
    mul_682: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_58, add_267);  sigmoid_58 = add_267 = None
    mul_683: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_301, mul_682);  getitem_301 = mul_682 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_344: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_345: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 2);  unsqueeze_344 = None
    unsqueeze_346: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 3);  unsqueeze_345 = None
    sum_158: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_683, [0, 2, 3])
    sub_213: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_346)
    mul_684: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_683, sub_213);  sub_213 = None
    sum_159: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_684, [0, 2, 3]);  mul_684 = None
    mul_685: "f32[256]" = torch.ops.aten.mul.Tensor(sum_158, 3.0517578125e-05)
    unsqueeze_347: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_685, 0);  mul_685 = None
    unsqueeze_348: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 2);  unsqueeze_347 = None
    unsqueeze_349: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 3);  unsqueeze_348 = None
    mul_686: "f32[256]" = torch.ops.aten.mul.Tensor(sum_159, 3.0517578125e-05)
    mul_687: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_688: "f32[256]" = torch.ops.aten.mul.Tensor(mul_686, mul_687);  mul_686 = mul_687 = None
    unsqueeze_350: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_688, 0);  mul_688 = None
    unsqueeze_351: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 2);  unsqueeze_350 = None
    unsqueeze_352: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 3);  unsqueeze_351 = None
    mul_689: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_27);  primals_27 = None
    unsqueeze_353: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_689, 0);  mul_689 = None
    unsqueeze_354: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 2);  unsqueeze_353 = None
    unsqueeze_355: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, 3);  unsqueeze_354 = None
    sub_214: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_346);  convolution_13 = unsqueeze_346 = None
    mul_690: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_352);  sub_214 = unsqueeze_352 = None
    sub_215: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(mul_683, mul_690);  mul_683 = mul_690 = None
    sub_216: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(sub_215, unsqueeze_349);  sub_215 = unsqueeze_349 = None
    mul_691: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_355);  sub_216 = unsqueeze_355 = None
    mul_692: "f32[256]" = torch.ops.aten.mul.Tensor(sum_159, squeeze_40);  sum_159 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_691, add_66, primals_78, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_691 = add_66 = primals_78 = None
    getitem_304: "f32[8, 64, 64, 64]" = convolution_backward_21[0]
    getitem_305: "f32[256, 64, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_356: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_357: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 2);  unsqueeze_356 = None
    unsqueeze_358: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 3);  unsqueeze_357 = None
    sum_160: "f32[64]" = torch.ops.aten.sum.dim_IntList(getitem_304, [0, 2, 3])
    sub_217: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_358)
    mul_693: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_304, sub_217);  sub_217 = None
    sum_161: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_693, [0, 2, 3]);  mul_693 = None
    mul_694: "f32[64]" = torch.ops.aten.mul.Tensor(sum_160, 3.0517578125e-05)
    unsqueeze_359: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_694, 0);  mul_694 = None
    unsqueeze_360: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 2);  unsqueeze_359 = None
    unsqueeze_361: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, 3);  unsqueeze_360 = None
    mul_695: "f32[64]" = torch.ops.aten.mul.Tensor(sum_161, 3.0517578125e-05)
    mul_696: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_697: "f32[64]" = torch.ops.aten.mul.Tensor(mul_695, mul_696);  mul_695 = mul_696 = None
    unsqueeze_362: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_697, 0);  mul_697 = None
    unsqueeze_363: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 2);  unsqueeze_362 = None
    unsqueeze_364: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 3);  unsqueeze_363 = None
    mul_698: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_25);  primals_25 = None
    unsqueeze_365: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_698, 0);  mul_698 = None
    unsqueeze_366: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 2);  unsqueeze_365 = None
    unsqueeze_367: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, 3);  unsqueeze_366 = None
    sub_218: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_358);  convolution_12 = unsqueeze_358 = None
    mul_699: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_218, unsqueeze_364);  sub_218 = unsqueeze_364 = None
    sub_219: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(getitem_304, mul_699);  mul_699 = None
    sub_220: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_219, unsqueeze_361);  sub_219 = unsqueeze_361 = None
    mul_700: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_367);  sub_220 = unsqueeze_367 = None
    mul_701: "f32[64]" = torch.ops.aten.mul.Tensor(sum_161, squeeze_37);  sum_161 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_700, mul_92, primals_77, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_700 = mul_92 = primals_77 = None
    getitem_307: "f32[8, 256, 64, 64]" = convolution_backward_22[0]
    getitem_308: "f32[64, 256, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_59: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(clone_8)
    full_25: "f32[8, 256, 64, 64]" = torch.ops.aten.full.default([8, 256, 64, 64], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_221: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(full_25, sigmoid_59);  full_25 = None
    mul_702: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(clone_8, sub_221);  clone_8 = sub_221 = None
    add_268: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Scalar(mul_702, 1);  mul_702 = None
    mul_703: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_59, add_268);  sigmoid_59 = add_268 = None
    mul_704: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_307, mul_703);  getitem_307 = mul_703 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_368: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_369: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 2);  unsqueeze_368 = None
    unsqueeze_370: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 3);  unsqueeze_369 = None
    sum_162: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_704, [0, 2, 3])
    sub_222: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_370)
    mul_705: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_704, sub_222);  sub_222 = None
    sum_163: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_705, [0, 2, 3]);  mul_705 = None
    mul_706: "f32[256]" = torch.ops.aten.mul.Tensor(sum_162, 3.0517578125e-05)
    unsqueeze_371: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_706, 0);  mul_706 = None
    unsqueeze_372: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 2);  unsqueeze_371 = None
    unsqueeze_373: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, 3);  unsqueeze_372 = None
    mul_707: "f32[256]" = torch.ops.aten.mul.Tensor(sum_163, 3.0517578125e-05)
    mul_708: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_709: "f32[256]" = torch.ops.aten.mul.Tensor(mul_707, mul_708);  mul_707 = mul_708 = None
    unsqueeze_374: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_709, 0);  mul_709 = None
    unsqueeze_375: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 2);  unsqueeze_374 = None
    unsqueeze_376: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 3);  unsqueeze_375 = None
    mul_710: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_23);  primals_23 = None
    unsqueeze_377: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_710, 0);  mul_710 = None
    unsqueeze_378: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 2);  unsqueeze_377 = None
    unsqueeze_379: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 3);  unsqueeze_378 = None
    sub_223: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_370);  convolution_11 = unsqueeze_370 = None
    mul_711: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_223, unsqueeze_376);  sub_223 = unsqueeze_376 = None
    sub_224: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(mul_704, mul_711);  mul_704 = mul_711 = None
    sub_225: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(sub_224, unsqueeze_373);  sub_224 = unsqueeze_373 = None
    mul_712: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_379);  sub_225 = unsqueeze_379 = None
    mul_713: "f32[256]" = torch.ops.aten.mul.Tensor(sum_163, squeeze_34);  sum_163 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_712, mul_84, primals_76, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 256, [True, True, False]);  mul_712 = mul_84 = primals_76 = None
    getitem_310: "f32[8, 256, 64, 64]" = convolution_backward_23[0]
    getitem_311: "f32[256, 1, 3, 3]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_60: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(clone_7)
    full_26: "f32[8, 256, 64, 64]" = torch.ops.aten.full.default([8, 256, 64, 64], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_226: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(full_26, sigmoid_60);  full_26 = None
    mul_714: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(clone_7, sub_226);  clone_7 = sub_226 = None
    add_269: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Scalar(mul_714, 1);  mul_714 = None
    mul_715: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_60, add_269);  sigmoid_60 = add_269 = None
    mul_716: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_310, mul_715);  getitem_310 = mul_715 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_380: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_381: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 2);  unsqueeze_380 = None
    unsqueeze_382: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 3);  unsqueeze_381 = None
    sum_164: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_716, [0, 2, 3])
    sub_227: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_382)
    mul_717: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_716, sub_227);  sub_227 = None
    sum_165: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_717, [0, 2, 3]);  mul_717 = None
    mul_718: "f32[256]" = torch.ops.aten.mul.Tensor(sum_164, 3.0517578125e-05)
    unsqueeze_383: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_718, 0);  mul_718 = None
    unsqueeze_384: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 2);  unsqueeze_383 = None
    unsqueeze_385: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, 3);  unsqueeze_384 = None
    mul_719: "f32[256]" = torch.ops.aten.mul.Tensor(sum_165, 3.0517578125e-05)
    mul_720: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_721: "f32[256]" = torch.ops.aten.mul.Tensor(mul_719, mul_720);  mul_719 = mul_720 = None
    unsqueeze_386: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_721, 0);  mul_721 = None
    unsqueeze_387: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 2);  unsqueeze_386 = None
    unsqueeze_388: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 3);  unsqueeze_387 = None
    mul_722: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_21);  primals_21 = None
    unsqueeze_389: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_722, 0);  mul_722 = None
    unsqueeze_390: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 2);  unsqueeze_389 = None
    unsqueeze_391: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, 3);  unsqueeze_390 = None
    sub_228: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_382);  convolution_10 = unsqueeze_382 = None
    mul_723: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_388);  sub_228 = unsqueeze_388 = None
    sub_229: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(mul_716, mul_723);  mul_716 = mul_723 = None
    sub_230: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(sub_229, unsqueeze_385);  sub_229 = unsqueeze_385 = None
    mul_724: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_391);  sub_230 = unsqueeze_391 = None
    mul_725: "f32[256]" = torch.ops.aten.mul.Tensor(sum_165, squeeze_31);  sum_165 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_724, add_50, primals_75, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_724 = add_50 = primals_75 = None
    getitem_313: "f32[8, 64, 64, 64]" = convolution_backward_24[0]
    getitem_314: "f32[256, 64, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_270: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(getitem_304, getitem_313);  getitem_304 = getitem_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_392: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_393: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 2);  unsqueeze_392 = None
    unsqueeze_394: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 3);  unsqueeze_393 = None
    sum_166: "f32[64]" = torch.ops.aten.sum.dim_IntList(add_270, [0, 2, 3])
    sub_231: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_394)
    mul_726: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_270, sub_231);  sub_231 = None
    sum_167: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_726, [0, 2, 3]);  mul_726 = None
    mul_727: "f32[64]" = torch.ops.aten.mul.Tensor(sum_166, 3.0517578125e-05)
    unsqueeze_395: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_727, 0);  mul_727 = None
    unsqueeze_396: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 2);  unsqueeze_395 = None
    unsqueeze_397: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, 3);  unsqueeze_396 = None
    mul_728: "f32[64]" = torch.ops.aten.mul.Tensor(sum_167, 3.0517578125e-05)
    mul_729: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_730: "f32[64]" = torch.ops.aten.mul.Tensor(mul_728, mul_729);  mul_728 = mul_729 = None
    unsqueeze_398: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_730, 0);  mul_730 = None
    unsqueeze_399: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 2);  unsqueeze_398 = None
    unsqueeze_400: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 3);  unsqueeze_399 = None
    mul_731: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_19);  primals_19 = None
    unsqueeze_401: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_731, 0);  mul_731 = None
    unsqueeze_402: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 2);  unsqueeze_401 = None
    unsqueeze_403: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 3);  unsqueeze_402 = None
    sub_232: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_394);  convolution_9 = unsqueeze_394 = None
    mul_732: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_400);  sub_232 = unsqueeze_400 = None
    sub_233: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(add_270, mul_732);  mul_732 = None
    sub_234: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_233, unsqueeze_397);  sub_233 = unsqueeze_397 = None
    mul_733: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_234, unsqueeze_403);  sub_234 = unsqueeze_403 = None
    mul_734: "f32[64]" = torch.ops.aten.mul.Tensor(sum_167, squeeze_28);  sum_167 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_733, mul_69, primals_74, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_733 = mul_69 = primals_74 = None
    getitem_316: "f32[8, 256, 64, 64]" = convolution_backward_25[0]
    getitem_317: "f32[64, 256, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_61: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(clone_6)
    full_27: "f32[8, 256, 64, 64]" = torch.ops.aten.full.default([8, 256, 64, 64], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_235: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(full_27, sigmoid_61);  full_27 = None
    mul_735: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(clone_6, sub_235);  clone_6 = sub_235 = None
    add_271: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Scalar(mul_735, 1);  mul_735 = None
    mul_736: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_61, add_271);  sigmoid_61 = add_271 = None
    mul_737: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_316, mul_736);  getitem_316 = mul_736 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_404: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_405: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 2);  unsqueeze_404 = None
    unsqueeze_406: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 3);  unsqueeze_405 = None
    sum_168: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_737, [0, 2, 3])
    sub_236: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_406)
    mul_738: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_737, sub_236);  sub_236 = None
    sum_169: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_738, [0, 2, 3]);  mul_738 = None
    mul_739: "f32[256]" = torch.ops.aten.mul.Tensor(sum_168, 3.0517578125e-05)
    unsqueeze_407: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_739, 0);  mul_739 = None
    unsqueeze_408: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 2);  unsqueeze_407 = None
    unsqueeze_409: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, 3);  unsqueeze_408 = None
    mul_740: "f32[256]" = torch.ops.aten.mul.Tensor(sum_169, 3.0517578125e-05)
    mul_741: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_742: "f32[256]" = torch.ops.aten.mul.Tensor(mul_740, mul_741);  mul_740 = mul_741 = None
    unsqueeze_410: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_742, 0);  mul_742 = None
    unsqueeze_411: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 2);  unsqueeze_410 = None
    unsqueeze_412: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 3);  unsqueeze_411 = None
    mul_743: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_17);  primals_17 = None
    unsqueeze_413: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_743, 0);  mul_743 = None
    unsqueeze_414: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 2);  unsqueeze_413 = None
    unsqueeze_415: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 3);  unsqueeze_414 = None
    sub_237: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_406);  convolution_8 = unsqueeze_406 = None
    mul_744: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_412);  sub_237 = unsqueeze_412 = None
    sub_238: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(mul_737, mul_744);  mul_737 = mul_744 = None
    sub_239: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(sub_238, unsqueeze_409);  sub_238 = unsqueeze_409 = None
    mul_745: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_239, unsqueeze_415);  sub_239 = unsqueeze_415 = None
    mul_746: "f32[256]" = torch.ops.aten.mul.Tensor(sum_169, squeeze_25);  sum_169 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_745, mul_61, primals_73, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 256, [True, True, False]);  mul_745 = mul_61 = primals_73 = None
    getitem_319: "f32[8, 256, 64, 64]" = convolution_backward_26[0]
    getitem_320: "f32[256, 1, 3, 3]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_62: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(clone_5)
    full_28: "f32[8, 256, 64, 64]" = torch.ops.aten.full.default([8, 256, 64, 64], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_240: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(full_28, sigmoid_62);  full_28 = None
    mul_747: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(clone_5, sub_240);  clone_5 = sub_240 = None
    add_272: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Scalar(mul_747, 1);  mul_747 = None
    mul_748: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_62, add_272);  sigmoid_62 = add_272 = None
    mul_749: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_319, mul_748);  getitem_319 = mul_748 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_416: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_417: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 2);  unsqueeze_416 = None
    unsqueeze_418: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 3);  unsqueeze_417 = None
    sum_170: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_749, [0, 2, 3])
    sub_241: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_418)
    mul_750: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_749, sub_241);  sub_241 = None
    sum_171: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_750, [0, 2, 3]);  mul_750 = None
    mul_751: "f32[256]" = torch.ops.aten.mul.Tensor(sum_170, 3.0517578125e-05)
    unsqueeze_419: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_751, 0);  mul_751 = None
    unsqueeze_420: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 2);  unsqueeze_419 = None
    unsqueeze_421: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 3);  unsqueeze_420 = None
    mul_752: "f32[256]" = torch.ops.aten.mul.Tensor(sum_171, 3.0517578125e-05)
    mul_753: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_754: "f32[256]" = torch.ops.aten.mul.Tensor(mul_752, mul_753);  mul_752 = mul_753 = None
    unsqueeze_422: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_754, 0);  mul_754 = None
    unsqueeze_423: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 2);  unsqueeze_422 = None
    unsqueeze_424: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 3);  unsqueeze_423 = None
    mul_755: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_15);  primals_15 = None
    unsqueeze_425: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_755, 0);  mul_755 = None
    unsqueeze_426: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 2);  unsqueeze_425 = None
    unsqueeze_427: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 3);  unsqueeze_426 = None
    sub_242: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_418);  convolution_7 = unsqueeze_418 = None
    mul_756: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_242, unsqueeze_424);  sub_242 = unsqueeze_424 = None
    sub_243: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(mul_749, mul_756);  mul_749 = mul_756 = None
    sub_244: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(sub_243, unsqueeze_421);  sub_243 = unsqueeze_421 = None
    mul_757: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_427);  sub_244 = unsqueeze_427 = None
    mul_758: "f32[256]" = torch.ops.aten.mul.Tensor(sum_171, squeeze_22);  sum_171 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_757, add_34, primals_72, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_757 = add_34 = primals_72 = None
    getitem_322: "f32[8, 64, 64, 64]" = convolution_backward_27[0]
    getitem_323: "f32[256, 64, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_273: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(add_270, getitem_322);  add_270 = getitem_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_428: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_429: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 2);  unsqueeze_428 = None
    unsqueeze_430: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 3);  unsqueeze_429 = None
    sum_172: "f32[64]" = torch.ops.aten.sum.dim_IntList(add_273, [0, 2, 3])
    sub_245: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_430)
    mul_759: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_273, sub_245);  sub_245 = None
    sum_173: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_759, [0, 2, 3]);  mul_759 = None
    mul_760: "f32[64]" = torch.ops.aten.mul.Tensor(sum_172, 3.0517578125e-05)
    unsqueeze_431: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_760, 0);  mul_760 = None
    unsqueeze_432: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 2);  unsqueeze_431 = None
    unsqueeze_433: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 3);  unsqueeze_432 = None
    mul_761: "f32[64]" = torch.ops.aten.mul.Tensor(sum_173, 3.0517578125e-05)
    mul_762: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_763: "f32[64]" = torch.ops.aten.mul.Tensor(mul_761, mul_762);  mul_761 = mul_762 = None
    unsqueeze_434: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_763, 0);  mul_763 = None
    unsqueeze_435: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 2);  unsqueeze_434 = None
    unsqueeze_436: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 3);  unsqueeze_435 = None
    mul_764: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_13);  primals_13 = None
    unsqueeze_437: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_764, 0);  mul_764 = None
    unsqueeze_438: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 2);  unsqueeze_437 = None
    unsqueeze_439: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 3);  unsqueeze_438 = None
    sub_246: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_430);  convolution_6 = unsqueeze_430 = None
    mul_765: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_246, unsqueeze_436);  sub_246 = unsqueeze_436 = None
    sub_247: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(add_273, mul_765);  add_273 = mul_765 = None
    sub_248: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_247, unsqueeze_433);  sub_247 = unsqueeze_433 = None
    mul_766: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_439);  sub_248 = unsqueeze_439 = None
    mul_767: "f32[64]" = torch.ops.aten.mul.Tensor(sum_173, squeeze_19);  sum_173 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_766, mul_46, primals_71, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_766 = mul_46 = primals_71 = None
    getitem_325: "f32[8, 128, 64, 64]" = convolution_backward_28[0]
    getitem_326: "f32[64, 128, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_63: "f32[8, 128, 64, 64]" = torch.ops.aten.sigmoid.default(clone_4)
    full_29: "f32[8, 128, 64, 64]" = torch.ops.aten.full.default([8, 128, 64, 64], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_249: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(full_29, sigmoid_63);  full_29 = None
    mul_768: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(clone_4, sub_249);  clone_4 = sub_249 = None
    add_274: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Scalar(mul_768, 1);  mul_768 = None
    mul_769: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_63, add_274);  sigmoid_63 = add_274 = None
    mul_770: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_325, mul_769);  getitem_325 = mul_769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_440: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_441: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 2);  unsqueeze_440 = None
    unsqueeze_442: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 3);  unsqueeze_441 = None
    sum_174: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_770, [0, 2, 3])
    sub_250: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_442)
    mul_771: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_770, sub_250);  sub_250 = None
    sum_175: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_771, [0, 2, 3]);  mul_771 = None
    mul_772: "f32[128]" = torch.ops.aten.mul.Tensor(sum_174, 3.0517578125e-05)
    unsqueeze_443: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_772, 0);  mul_772 = None
    unsqueeze_444: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 2);  unsqueeze_443 = None
    unsqueeze_445: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, 3);  unsqueeze_444 = None
    mul_773: "f32[128]" = torch.ops.aten.mul.Tensor(sum_175, 3.0517578125e-05)
    mul_774: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_775: "f32[128]" = torch.ops.aten.mul.Tensor(mul_773, mul_774);  mul_773 = mul_774 = None
    unsqueeze_446: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_775, 0);  mul_775 = None
    unsqueeze_447: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 2);  unsqueeze_446 = None
    unsqueeze_448: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 3);  unsqueeze_447 = None
    mul_776: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_11);  primals_11 = None
    unsqueeze_449: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_776, 0);  mul_776 = None
    unsqueeze_450: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
    unsqueeze_451: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 3);  unsqueeze_450 = None
    sub_251: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_442);  convolution_5 = unsqueeze_442 = None
    mul_777: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_251, unsqueeze_448);  sub_251 = unsqueeze_448 = None
    sub_252: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(mul_770, mul_777);  mul_770 = mul_777 = None
    sub_253: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(sub_252, unsqueeze_445);  sub_252 = unsqueeze_445 = None
    mul_778: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_451);  sub_253 = unsqueeze_451 = None
    mul_779: "f32[128]" = torch.ops.aten.mul.Tensor(sum_175, squeeze_16);  sum_175 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_778, mul_38, primals_70, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 128, [True, True, False]);  mul_778 = mul_38 = primals_70 = None
    getitem_328: "f32[8, 128, 128, 128]" = convolution_backward_29[0]
    getitem_329: "f32[128, 1, 3, 3]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_64: "f32[8, 128, 128, 128]" = torch.ops.aten.sigmoid.default(clone_3)
    full_30: "f32[8, 128, 128, 128]" = torch.ops.aten.full.default([8, 128, 128, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_254: "f32[8, 128, 128, 128]" = torch.ops.aten.sub.Tensor(full_30, sigmoid_64);  full_30 = None
    mul_780: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(clone_3, sub_254);  clone_3 = sub_254 = None
    add_275: "f32[8, 128, 128, 128]" = torch.ops.aten.add.Scalar(mul_780, 1);  mul_780 = None
    mul_781: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(sigmoid_64, add_275);  sigmoid_64 = add_275 = None
    mul_782: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(getitem_328, mul_781);  getitem_328 = mul_781 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_452: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_453: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 2);  unsqueeze_452 = None
    unsqueeze_454: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 3);  unsqueeze_453 = None
    sum_176: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_782, [0, 2, 3])
    sub_255: "f32[8, 128, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_454)
    mul_783: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(mul_782, sub_255);  sub_255 = None
    sum_177: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_783, [0, 2, 3]);  mul_783 = None
    mul_784: "f32[128]" = torch.ops.aten.mul.Tensor(sum_176, 7.62939453125e-06)
    unsqueeze_455: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_784, 0);  mul_784 = None
    unsqueeze_456: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 2);  unsqueeze_455 = None
    unsqueeze_457: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, 3);  unsqueeze_456 = None
    mul_785: "f32[128]" = torch.ops.aten.mul.Tensor(sum_177, 7.62939453125e-06)
    mul_786: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_787: "f32[128]" = torch.ops.aten.mul.Tensor(mul_785, mul_786);  mul_785 = mul_786 = None
    unsqueeze_458: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_787, 0);  mul_787 = None
    unsqueeze_459: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 2);  unsqueeze_458 = None
    unsqueeze_460: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 3);  unsqueeze_459 = None
    mul_788: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_9);  primals_9 = None
    unsqueeze_461: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_788, 0);  mul_788 = None
    unsqueeze_462: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
    unsqueeze_463: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 3);  unsqueeze_462 = None
    sub_256: "f32[8, 128, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_454);  convolution_4 = unsqueeze_454 = None
    mul_789: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_460);  sub_256 = unsqueeze_460 = None
    sub_257: "f32[8, 128, 128, 128]" = torch.ops.aten.sub.Tensor(mul_782, mul_789);  mul_782 = mul_789 = None
    sub_258: "f32[8, 128, 128, 128]" = torch.ops.aten.sub.Tensor(sub_257, unsqueeze_457);  sub_257 = unsqueeze_457 = None
    mul_790: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(sub_258, unsqueeze_463);  sub_258 = unsqueeze_463 = None
    mul_791: "f32[128]" = torch.ops.aten.mul.Tensor(sum_177, squeeze_13);  sum_177 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_790, add_19, primals_69, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_790 = add_19 = primals_69 = None
    getitem_331: "f32[8, 32, 128, 128]" = convolution_backward_30[0]
    getitem_332: "f32[128, 32, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_464: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_465: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 2);  unsqueeze_464 = None
    unsqueeze_466: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 3);  unsqueeze_465 = None
    sum_178: "f32[32]" = torch.ops.aten.sum.dim_IntList(getitem_331, [0, 2, 3])
    sub_259: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_466)
    mul_792: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(getitem_331, sub_259);  sub_259 = None
    sum_179: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_792, [0, 2, 3]);  mul_792 = None
    mul_793: "f32[32]" = torch.ops.aten.mul.Tensor(sum_178, 7.62939453125e-06)
    unsqueeze_467: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_793, 0);  mul_793 = None
    unsqueeze_468: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 2);  unsqueeze_467 = None
    unsqueeze_469: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, 3);  unsqueeze_468 = None
    mul_794: "f32[32]" = torch.ops.aten.mul.Tensor(sum_179, 7.62939453125e-06)
    mul_795: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_796: "f32[32]" = torch.ops.aten.mul.Tensor(mul_794, mul_795);  mul_794 = mul_795 = None
    unsqueeze_470: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_796, 0);  mul_796 = None
    unsqueeze_471: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 2);  unsqueeze_470 = None
    unsqueeze_472: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 3);  unsqueeze_471 = None
    mul_797: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_7);  primals_7 = None
    unsqueeze_473: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_797, 0);  mul_797 = None
    unsqueeze_474: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
    unsqueeze_475: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 3);  unsqueeze_474 = None
    sub_260: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_466);  convolution_3 = unsqueeze_466 = None
    mul_798: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_472);  sub_260 = unsqueeze_472 = None
    sub_261: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(getitem_331, mul_798);  getitem_331 = mul_798 = None
    sub_262: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(sub_261, unsqueeze_469);  sub_261 = unsqueeze_469 = None
    mul_799: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_262, unsqueeze_475);  sub_262 = unsqueeze_475 = None
    mul_800: "f32[32]" = torch.ops.aten.mul.Tensor(sum_179, squeeze_10);  sum_179 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_799, mul_23, primals_68, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_799 = mul_23 = primals_68 = None
    getitem_334: "f32[8, 64, 128, 128]" = convolution_backward_31[0]
    getitem_335: "f32[32, 64, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_65: "f32[8, 64, 128, 128]" = torch.ops.aten.sigmoid.default(clone_2)
    full_31: "f32[8, 64, 128, 128]" = torch.ops.aten.full.default([8, 64, 128, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_263: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(full_31, sigmoid_65);  full_31 = None
    mul_801: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(clone_2, sub_263);  clone_2 = sub_263 = None
    add_276: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Scalar(mul_801, 1);  mul_801 = None
    mul_802: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sigmoid_65, add_276);  sigmoid_65 = add_276 = None
    mul_803: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(getitem_334, mul_802);  getitem_334 = mul_802 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_476: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_477: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 2);  unsqueeze_476 = None
    unsqueeze_478: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 3);  unsqueeze_477 = None
    sum_180: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_803, [0, 2, 3])
    sub_264: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_478)
    mul_804: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_803, sub_264);  sub_264 = None
    sum_181: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_804, [0, 2, 3]);  mul_804 = None
    mul_805: "f32[64]" = torch.ops.aten.mul.Tensor(sum_180, 7.62939453125e-06)
    unsqueeze_479: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_805, 0);  mul_805 = None
    unsqueeze_480: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 2);  unsqueeze_479 = None
    unsqueeze_481: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 3);  unsqueeze_480 = None
    mul_806: "f32[64]" = torch.ops.aten.mul.Tensor(sum_181, 7.62939453125e-06)
    mul_807: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_808: "f32[64]" = torch.ops.aten.mul.Tensor(mul_806, mul_807);  mul_806 = mul_807 = None
    unsqueeze_482: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_808, 0);  mul_808 = None
    unsqueeze_483: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 2);  unsqueeze_482 = None
    unsqueeze_484: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 3);  unsqueeze_483 = None
    mul_809: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_5);  primals_5 = None
    unsqueeze_485: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_809, 0);  mul_809 = None
    unsqueeze_486: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
    unsqueeze_487: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 3);  unsqueeze_486 = None
    sub_265: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_478);  convolution_2 = unsqueeze_478 = None
    mul_810: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_265, unsqueeze_484);  sub_265 = unsqueeze_484 = None
    sub_266: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(mul_803, mul_810);  mul_803 = mul_810 = None
    sub_267: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(sub_266, unsqueeze_481);  sub_266 = unsqueeze_481 = None
    mul_811: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_267, unsqueeze_487);  sub_267 = unsqueeze_487 = None
    mul_812: "f32[64]" = torch.ops.aten.mul.Tensor(sum_181, squeeze_7);  sum_181 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_811, mul_15, primals_67, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False]);  mul_811 = mul_15 = primals_67 = None
    getitem_337: "f32[8, 64, 128, 128]" = convolution_backward_32[0]
    getitem_338: "f32[64, 1, 3, 3]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_66: "f32[8, 64, 128, 128]" = torch.ops.aten.sigmoid.default(clone_1)
    full_32: "f32[8, 64, 128, 128]" = torch.ops.aten.full.default([8, 64, 128, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_268: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(full_32, sigmoid_66);  full_32 = None
    mul_813: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(clone_1, sub_268);  clone_1 = sub_268 = None
    add_277: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Scalar(mul_813, 1);  mul_813 = None
    mul_814: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sigmoid_66, add_277);  sigmoid_66 = add_277 = None
    mul_815: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(getitem_337, mul_814);  getitem_337 = mul_814 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_488: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_489: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 2);  unsqueeze_488 = None
    unsqueeze_490: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 3);  unsqueeze_489 = None
    sum_182: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_815, [0, 2, 3])
    sub_269: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_490)
    mul_816: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_815, sub_269);  sub_269 = None
    sum_183: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_816, [0, 2, 3]);  mul_816 = None
    mul_817: "f32[64]" = torch.ops.aten.mul.Tensor(sum_182, 7.62939453125e-06)
    unsqueeze_491: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_817, 0);  mul_817 = None
    unsqueeze_492: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 2);  unsqueeze_491 = None
    unsqueeze_493: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, 3);  unsqueeze_492 = None
    mul_818: "f32[64]" = torch.ops.aten.mul.Tensor(sum_183, 7.62939453125e-06)
    mul_819: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_820: "f32[64]" = torch.ops.aten.mul.Tensor(mul_818, mul_819);  mul_818 = mul_819 = None
    unsqueeze_494: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_820, 0);  mul_820 = None
    unsqueeze_495: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 2);  unsqueeze_494 = None
    unsqueeze_496: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 3);  unsqueeze_495 = None
    mul_821: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_3);  primals_3 = None
    unsqueeze_497: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_821, 0);  mul_821 = None
    unsqueeze_498: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
    unsqueeze_499: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 3);  unsqueeze_498 = None
    sub_270: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_490);  convolution_1 = unsqueeze_490 = None
    mul_822: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_270, unsqueeze_496);  sub_270 = unsqueeze_496 = None
    sub_271: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(mul_815, mul_822);  mul_815 = mul_822 = None
    sub_272: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(sub_271, unsqueeze_493);  sub_271 = unsqueeze_493 = None
    mul_823: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_499);  sub_272 = unsqueeze_499 = None
    mul_824: "f32[64]" = torch.ops.aten.mul.Tensor(sum_183, squeeze_4);  sum_183 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_823, mul_7, primals_66, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_823 = mul_7 = primals_66 = None
    getitem_340: "f32[8, 16, 128, 128]" = convolution_backward_33[0]
    getitem_341: "f32[64, 16, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_67: "f32[8, 16, 128, 128]" = torch.ops.aten.sigmoid.default(clone)
    full_33: "f32[8, 16, 128, 128]" = torch.ops.aten.full.default([8, 16, 128, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_273: "f32[8, 16, 128, 128]" = torch.ops.aten.sub.Tensor(full_33, sigmoid_67);  full_33 = None
    mul_825: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(clone, sub_273);  clone = sub_273 = None
    add_278: "f32[8, 16, 128, 128]" = torch.ops.aten.add.Scalar(mul_825, 1);  mul_825 = None
    mul_826: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(sigmoid_67, add_278);  sigmoid_67 = add_278 = None
    mul_827: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(getitem_340, mul_826);  getitem_340 = mul_826 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_500: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_501: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 2);  unsqueeze_500 = None
    unsqueeze_502: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 3);  unsqueeze_501 = None
    sum_184: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_827, [0, 2, 3])
    sub_274: "f32[8, 16, 128, 128]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_502)
    mul_828: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(mul_827, sub_274);  sub_274 = None
    sum_185: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_828, [0, 2, 3]);  mul_828 = None
    mul_829: "f32[16]" = torch.ops.aten.mul.Tensor(sum_184, 7.62939453125e-06)
    unsqueeze_503: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_829, 0);  mul_829 = None
    unsqueeze_504: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 2);  unsqueeze_503 = None
    unsqueeze_505: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, 3);  unsqueeze_504 = None
    mul_830: "f32[16]" = torch.ops.aten.mul.Tensor(sum_185, 7.62939453125e-06)
    mul_831: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_832: "f32[16]" = torch.ops.aten.mul.Tensor(mul_830, mul_831);  mul_830 = mul_831 = None
    unsqueeze_506: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_832, 0);  mul_832 = None
    unsqueeze_507: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 2);  unsqueeze_506 = None
    unsqueeze_508: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 3);  unsqueeze_507 = None
    mul_833: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_1);  primals_1 = None
    unsqueeze_509: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_833, 0);  mul_833 = None
    unsqueeze_510: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 2);  unsqueeze_509 = None
    unsqueeze_511: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 3);  unsqueeze_510 = None
    sub_275: "f32[8, 16, 128, 128]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_502);  convolution = unsqueeze_502 = None
    mul_834: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(sub_275, unsqueeze_508);  sub_275 = unsqueeze_508 = None
    sub_276: "f32[8, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_827, mul_834);  mul_827 = mul_834 = None
    sub_277: "f32[8, 16, 128, 128]" = torch.ops.aten.sub.Tensor(sub_276, unsqueeze_505);  sub_276 = unsqueeze_505 = None
    mul_835: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(sub_277, unsqueeze_511);  sub_277 = unsqueeze_511 = None
    mul_836: "f32[16]" = torch.ops.aten.mul.Tensor(sum_185, squeeze_1);  sum_185 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_835, primals_312, primals_65, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_835 = primals_312 = primals_65 = None
    getitem_344: "f32[16, 3, 3, 3]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # No stacktrace found for following nodes
    copy_: "i64[]" = torch.ops.aten.copy_.default(primals_216, add);  primals_216 = add = None
    copy__1: "f32[16]" = torch.ops.aten.copy_.default(primals_217, add_2);  primals_217 = add_2 = None
    copy__2: "f32[16]" = torch.ops.aten.copy_.default(primals_218, add_3);  primals_218 = add_3 = None
    copy__3: "i64[]" = torch.ops.aten.copy_.default(primals_219, add_5);  primals_219 = add_5 = None
    copy__4: "f32[64]" = torch.ops.aten.copy_.default(primals_220, add_7);  primals_220 = add_7 = None
    copy__5: "f32[64]" = torch.ops.aten.copy_.default(primals_221, add_8);  primals_221 = add_8 = None
    copy__6: "i64[]" = torch.ops.aten.copy_.default(primals_222, add_10);  primals_222 = add_10 = None
    copy__7: "f32[64]" = torch.ops.aten.copy_.default(primals_223, add_12);  primals_223 = add_12 = None
    copy__8: "f32[64]" = torch.ops.aten.copy_.default(primals_224, add_13);  primals_224 = add_13 = None
    copy__9: "i64[]" = torch.ops.aten.copy_.default(primals_225, add_15);  primals_225 = add_15 = None
    copy__10: "f32[32]" = torch.ops.aten.copy_.default(primals_226, add_17);  primals_226 = add_17 = None
    copy__11: "f32[32]" = torch.ops.aten.copy_.default(primals_227, add_18);  primals_227 = add_18 = None
    copy__12: "i64[]" = torch.ops.aten.copy_.default(primals_228, add_20);  primals_228 = add_20 = None
    copy__13: "f32[128]" = torch.ops.aten.copy_.default(primals_229, add_22);  primals_229 = add_22 = None
    copy__14: "f32[128]" = torch.ops.aten.copy_.default(primals_230, add_23);  primals_230 = add_23 = None
    copy__15: "i64[]" = torch.ops.aten.copy_.default(primals_231, add_25);  primals_231 = add_25 = None
    copy__16: "f32[128]" = torch.ops.aten.copy_.default(primals_232, add_27);  primals_232 = add_27 = None
    copy__17: "f32[128]" = torch.ops.aten.copy_.default(primals_233, add_28);  primals_233 = add_28 = None
    copy__18: "i64[]" = torch.ops.aten.copy_.default(primals_234, add_30);  primals_234 = add_30 = None
    copy__19: "f32[64]" = torch.ops.aten.copy_.default(primals_235, add_32);  primals_235 = add_32 = None
    copy__20: "f32[64]" = torch.ops.aten.copy_.default(primals_236, add_33);  primals_236 = add_33 = None
    copy__21: "i64[]" = torch.ops.aten.copy_.default(primals_237, add_35);  primals_237 = add_35 = None
    copy__22: "f32[256]" = torch.ops.aten.copy_.default(primals_238, add_37);  primals_238 = add_37 = None
    copy__23: "f32[256]" = torch.ops.aten.copy_.default(primals_239, add_38);  primals_239 = add_38 = None
    copy__24: "i64[]" = torch.ops.aten.copy_.default(primals_240, add_40);  primals_240 = add_40 = None
    copy__25: "f32[256]" = torch.ops.aten.copy_.default(primals_241, add_42);  primals_241 = add_42 = None
    copy__26: "f32[256]" = torch.ops.aten.copy_.default(primals_242, add_43);  primals_242 = add_43 = None
    copy__27: "i64[]" = torch.ops.aten.copy_.default(primals_243, add_45);  primals_243 = add_45 = None
    copy__28: "f32[64]" = torch.ops.aten.copy_.default(primals_244, add_47);  primals_244 = add_47 = None
    copy__29: "f32[64]" = torch.ops.aten.copy_.default(primals_245, add_48);  primals_245 = add_48 = None
    copy__30: "i64[]" = torch.ops.aten.copy_.default(primals_246, add_51);  primals_246 = add_51 = None
    copy__31: "f32[256]" = torch.ops.aten.copy_.default(primals_247, add_53);  primals_247 = add_53 = None
    copy__32: "f32[256]" = torch.ops.aten.copy_.default(primals_248, add_54);  primals_248 = add_54 = None
    copy__33: "i64[]" = torch.ops.aten.copy_.default(primals_249, add_56);  primals_249 = add_56 = None
    copy__34: "f32[256]" = torch.ops.aten.copy_.default(primals_250, add_58);  primals_250 = add_58 = None
    copy__35: "f32[256]" = torch.ops.aten.copy_.default(primals_251, add_59);  primals_251 = add_59 = None
    copy__36: "i64[]" = torch.ops.aten.copy_.default(primals_252, add_61);  primals_252 = add_61 = None
    copy__37: "f32[64]" = torch.ops.aten.copy_.default(primals_253, add_63);  primals_253 = add_63 = None
    copy__38: "f32[64]" = torch.ops.aten.copy_.default(primals_254, add_64);  primals_254 = add_64 = None
    copy__39: "i64[]" = torch.ops.aten.copy_.default(primals_255, add_67);  primals_255 = add_67 = None
    copy__40: "f32[256]" = torch.ops.aten.copy_.default(primals_256, add_69);  primals_256 = add_69 = None
    copy__41: "f32[256]" = torch.ops.aten.copy_.default(primals_257, add_70);  primals_257 = add_70 = None
    copy__42: "i64[]" = torch.ops.aten.copy_.default(primals_258, add_72);  primals_258 = add_72 = None
    copy__43: "f32[256]" = torch.ops.aten.copy_.default(primals_259, add_74);  primals_259 = add_74 = None
    copy__44: "f32[256]" = torch.ops.aten.copy_.default(primals_260, add_75);  primals_260 = add_75 = None
    copy__45: "i64[]" = torch.ops.aten.copy_.default(primals_261, add_77);  primals_261 = add_77 = None
    copy__46: "f32[96]" = torch.ops.aten.copy_.default(primals_262, add_79);  primals_262 = add_79 = None
    copy__47: "f32[96]" = torch.ops.aten.copy_.default(primals_263, add_80);  primals_263 = add_80 = None
    copy__48: "i64[]" = torch.ops.aten.copy_.default(primals_264, add_82);  primals_264 = add_82 = None
    copy__49: "f32[96]" = torch.ops.aten.copy_.default(primals_265, add_84);  primals_265 = add_84 = None
    copy__50: "f32[96]" = torch.ops.aten.copy_.default(primals_266, add_85);  primals_266 = add_85 = None
    copy__51: "i64[]" = torch.ops.aten.copy_.default(primals_267, add_101);  primals_267 = add_101 = None
    copy__52: "f32[96]" = torch.ops.aten.copy_.default(primals_268, add_103);  primals_268 = add_103 = None
    copy__53: "f32[96]" = torch.ops.aten.copy_.default(primals_269, add_104);  primals_269 = add_104 = None
    copy__54: "i64[]" = torch.ops.aten.copy_.default(primals_270, add_106);  primals_270 = add_106 = None
    copy__55: "f32[96]" = torch.ops.aten.copy_.default(primals_271, add_108);  primals_271 = add_108 = None
    copy__56: "f32[96]" = torch.ops.aten.copy_.default(primals_272, add_109);  primals_272 = add_109 = None
    copy__57: "i64[]" = torch.ops.aten.copy_.default(primals_273, add_111);  primals_273 = add_111 = None
    copy__58: "f32[384]" = torch.ops.aten.copy_.default(primals_274, add_113);  primals_274 = add_113 = None
    copy__59: "f32[384]" = torch.ops.aten.copy_.default(primals_275, add_114);  primals_275 = add_114 = None
    copy__60: "i64[]" = torch.ops.aten.copy_.default(primals_276, add_116);  primals_276 = add_116 = None
    copy__61: "f32[384]" = torch.ops.aten.copy_.default(primals_277, add_118);  primals_277 = add_118 = None
    copy__62: "f32[384]" = torch.ops.aten.copy_.default(primals_278, add_119);  primals_278 = add_119 = None
    copy__63: "i64[]" = torch.ops.aten.copy_.default(primals_279, add_121);  primals_279 = add_121 = None
    copy__64: "f32[128]" = torch.ops.aten.copy_.default(primals_280, add_123);  primals_280 = add_123 = None
    copy__65: "f32[128]" = torch.ops.aten.copy_.default(primals_281, add_124);  primals_281 = add_124 = None
    copy__66: "i64[]" = torch.ops.aten.copy_.default(primals_282, add_126);  primals_282 = add_126 = None
    copy__67: "f32[128]" = torch.ops.aten.copy_.default(primals_283, add_128);  primals_283 = add_128 = None
    copy__68: "f32[128]" = torch.ops.aten.copy_.default(primals_284, add_129);  primals_284 = add_129 = None
    copy__69: "i64[]" = torch.ops.aten.copy_.default(primals_285, add_157);  primals_285 = add_157 = None
    copy__70: "f32[128]" = torch.ops.aten.copy_.default(primals_286, add_159);  primals_286 = add_159 = None
    copy__71: "f32[128]" = torch.ops.aten.copy_.default(primals_287, add_160);  primals_287 = add_160 = None
    copy__72: "i64[]" = torch.ops.aten.copy_.default(primals_288, add_162);  primals_288 = add_162 = None
    copy__73: "f32[128]" = torch.ops.aten.copy_.default(primals_289, add_164);  primals_289 = add_164 = None
    copy__74: "f32[128]" = torch.ops.aten.copy_.default(primals_290, add_165);  primals_290 = add_165 = None
    copy__75: "i64[]" = torch.ops.aten.copy_.default(primals_291, add_167);  primals_291 = add_167 = None
    copy__76: "f32[512]" = torch.ops.aten.copy_.default(primals_292, add_169);  primals_292 = add_169 = None
    copy__77: "f32[512]" = torch.ops.aten.copy_.default(primals_293, add_170);  primals_293 = add_170 = None
    copy__78: "i64[]" = torch.ops.aten.copy_.default(primals_294, add_172);  primals_294 = add_172 = None
    copy__79: "f32[512]" = torch.ops.aten.copy_.default(primals_295, add_174);  primals_295 = add_174 = None
    copy__80: "f32[512]" = torch.ops.aten.copy_.default(primals_296, add_175);  primals_296 = add_175 = None
    copy__81: "i64[]" = torch.ops.aten.copy_.default(primals_297, add_177);  primals_297 = add_177 = None
    copy__82: "f32[160]" = torch.ops.aten.copy_.default(primals_298, add_179);  primals_298 = add_179 = None
    copy__83: "f32[160]" = torch.ops.aten.copy_.default(primals_299, add_180);  primals_299 = add_180 = None
    copy__84: "i64[]" = torch.ops.aten.copy_.default(primals_300, add_182);  primals_300 = add_182 = None
    copy__85: "f32[160]" = torch.ops.aten.copy_.default(primals_301, add_184);  primals_301 = add_184 = None
    copy__86: "f32[160]" = torch.ops.aten.copy_.default(primals_302, add_185);  primals_302 = add_185 = None
    copy__87: "i64[]" = torch.ops.aten.copy_.default(primals_303, add_207);  primals_303 = add_207 = None
    copy__88: "f32[160]" = torch.ops.aten.copy_.default(primals_304, add_209);  primals_304 = add_209 = None
    copy__89: "f32[160]" = torch.ops.aten.copy_.default(primals_305, add_210);  primals_305 = add_210 = None
    copy__90: "i64[]" = torch.ops.aten.copy_.default(primals_306, add_212);  primals_306 = add_212 = None
    copy__91: "f32[160]" = torch.ops.aten.copy_.default(primals_307, add_214);  primals_307 = add_214 = None
    copy__92: "f32[160]" = torch.ops.aten.copy_.default(primals_308, add_215);  primals_308 = add_215 = None
    copy__93: "i64[]" = torch.ops.aten.copy_.default(primals_309, add_217);  primals_309 = add_217 = None
    copy__94: "f32[640]" = torch.ops.aten.copy_.default(primals_310, add_219);  primals_310 = add_219 = None
    copy__95: "f32[640]" = torch.ops.aten.copy_.default(primals_311, add_220);  primals_311 = add_220 = None
    return pytree.tree_unflatten([addmm_36, mul_836, sum_184, mul_824, sum_182, mul_812, sum_180, mul_800, sum_178, mul_791, sum_176, mul_779, sum_174, mul_767, sum_172, mul_758, sum_170, mul_746, sum_168, mul_734, sum_166, mul_725, sum_164, mul_713, sum_162, mul_701, sum_160, mul_692, sum_158, mul_680, sum_156, mul_668, sum_154, mul_659, sum_152, mul_606, sum_122, mul_594, sum_120, mul_582, sum_118, mul_570, sum_116, mul_558, sum_114, mul_549, sum_112, mul_462, sum_58, mul_450, sum_56, mul_438, sum_54, mul_426, sum_52, mul_414, sum_50, mul_405, sum_48, mul_335, sum_6, mul_323, sum_4, mul_311, sum_2, getitem_344, getitem_341, getitem_338, getitem_335, getitem_332, getitem_329, getitem_326, getitem_323, getitem_320, getitem_317, getitem_314, getitem_311, getitem_308, getitem_305, getitem_302, getitem_299, getitem_296, getitem_293, sum_150, sum_151, permute_264, view_259, permute_258, view_253, sum_144, sum_145, permute_254, view_250, permute_249, view_247, sum_138, sum_139, permute_245, view_244, permute_239, view_238, sum_132, sum_133, permute_235, view_235, permute_230, view_232, sum_126, sum_127, getitem_284, getitem_281, getitem_278, getitem_275, getitem_272, getitem_269, getitem_266, sum_110, sum_111, permute_217, view_223, permute_211, view_217, sum_104, sum_105, permute_207, view_214, permute_202, view_211, sum_98, sum_99, permute_198, view_208, permute_192, view_202, sum_92, sum_93, permute_188, view_199, permute_183, view_196, sum_86, sum_87, permute_179, view_193, permute_173, view_187, sum_80, sum_81, permute_169, view_184, permute_164, view_181, sum_74, sum_75, permute_160, view_178, permute_154, view_172, sum_68, sum_69, permute_150, view_169, permute_145, view_166, sum_62, sum_63, getitem_251, getitem_248, getitem_245, getitem_242, getitem_239, getitem_236, getitem_233, sum_46, sum_47, permute_132, view_157, permute_126, view_151, sum_40, sum_41, permute_122, view_148, permute_117, view_145, sum_34, sum_35, permute_113, view_142, permute_107, view_136, sum_28, sum_29, permute_103, view_133, permute_98, view_130, sum_22, sum_23, permute_94, view_127, permute_88, view_121, sum_16, sum_17, permute_84, view_118, permute_79, view_115, sum_10, sum_11, getitem_221, getitem_218, getitem_215, permute_70, view_109, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    