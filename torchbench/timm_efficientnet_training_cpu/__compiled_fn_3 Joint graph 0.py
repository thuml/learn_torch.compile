from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[32]"; primals_2: "f32[32]"; primals_3: "f32[32]"; primals_4: "f32[32]"; primals_5: "f32[16]"; primals_6: "f32[16]"; primals_7: "f32[96]"; primals_8: "f32[96]"; primals_9: "f32[96]"; primals_10: "f32[96]"; primals_11: "f32[24]"; primals_12: "f32[24]"; primals_13: "f32[144]"; primals_14: "f32[144]"; primals_15: "f32[144]"; primals_16: "f32[144]"; primals_17: "f32[24]"; primals_18: "f32[24]"; primals_19: "f32[144]"; primals_20: "f32[144]"; primals_21: "f32[144]"; primals_22: "f32[144]"; primals_23: "f32[40]"; primals_24: "f32[40]"; primals_25: "f32[240]"; primals_26: "f32[240]"; primals_27: "f32[240]"; primals_28: "f32[240]"; primals_29: "f32[40]"; primals_30: "f32[40]"; primals_31: "f32[240]"; primals_32: "f32[240]"; primals_33: "f32[240]"; primals_34: "f32[240]"; primals_35: "f32[80]"; primals_36: "f32[80]"; primals_37: "f32[480]"; primals_38: "f32[480]"; primals_39: "f32[480]"; primals_40: "f32[480]"; primals_41: "f32[80]"; primals_42: "f32[80]"; primals_43: "f32[480]"; primals_44: "f32[480]"; primals_45: "f32[480]"; primals_46: "f32[480]"; primals_47: "f32[80]"; primals_48: "f32[80]"; primals_49: "f32[480]"; primals_50: "f32[480]"; primals_51: "f32[480]"; primals_52: "f32[480]"; primals_53: "f32[112]"; primals_54: "f32[112]"; primals_55: "f32[672]"; primals_56: "f32[672]"; primals_57: "f32[672]"; primals_58: "f32[672]"; primals_59: "f32[112]"; primals_60: "f32[112]"; primals_61: "f32[672]"; primals_62: "f32[672]"; primals_63: "f32[672]"; primals_64: "f32[672]"; primals_65: "f32[112]"; primals_66: "f32[112]"; primals_67: "f32[672]"; primals_68: "f32[672]"; primals_69: "f32[672]"; primals_70: "f32[672]"; primals_71: "f32[192]"; primals_72: "f32[192]"; primals_73: "f32[1152]"; primals_74: "f32[1152]"; primals_75: "f32[1152]"; primals_76: "f32[1152]"; primals_77: "f32[192]"; primals_78: "f32[192]"; primals_79: "f32[1152]"; primals_80: "f32[1152]"; primals_81: "f32[1152]"; primals_82: "f32[1152]"; primals_83: "f32[192]"; primals_84: "f32[192]"; primals_85: "f32[1152]"; primals_86: "f32[1152]"; primals_87: "f32[1152]"; primals_88: "f32[1152]"; primals_89: "f32[192]"; primals_90: "f32[192]"; primals_91: "f32[1152]"; primals_92: "f32[1152]"; primals_93: "f32[1152]"; primals_94: "f32[1152]"; primals_95: "f32[320]"; primals_96: "f32[320]"; primals_97: "f32[1280]"; primals_98: "f32[1280]"; primals_99: "f32[32, 3, 3, 3]"; primals_100: "f32[32, 1, 3, 3]"; primals_101: "f32[8, 32, 1, 1]"; primals_102: "f32[8]"; primals_103: "f32[32, 8, 1, 1]"; primals_104: "f32[32]"; primals_105: "f32[16, 32, 1, 1]"; primals_106: "f32[96, 16, 1, 1]"; primals_107: "f32[96, 1, 3, 3]"; primals_108: "f32[4, 96, 1, 1]"; primals_109: "f32[4]"; primals_110: "f32[96, 4, 1, 1]"; primals_111: "f32[96]"; primals_112: "f32[24, 96, 1, 1]"; primals_113: "f32[144, 24, 1, 1]"; primals_114: "f32[144, 1, 3, 3]"; primals_115: "f32[6, 144, 1, 1]"; primals_116: "f32[6]"; primals_117: "f32[144, 6, 1, 1]"; primals_118: "f32[144]"; primals_119: "f32[24, 144, 1, 1]"; primals_120: "f32[144, 24, 1, 1]"; primals_121: "f32[144, 1, 5, 5]"; primals_122: "f32[6, 144, 1, 1]"; primals_123: "f32[6]"; primals_124: "f32[144, 6, 1, 1]"; primals_125: "f32[144]"; primals_126: "f32[40, 144, 1, 1]"; primals_127: "f32[240, 40, 1, 1]"; primals_128: "f32[240, 1, 5, 5]"; primals_129: "f32[10, 240, 1, 1]"; primals_130: "f32[10]"; primals_131: "f32[240, 10, 1, 1]"; primals_132: "f32[240]"; primals_133: "f32[40, 240, 1, 1]"; primals_134: "f32[240, 40, 1, 1]"; primals_135: "f32[240, 1, 3, 3]"; primals_136: "f32[10, 240, 1, 1]"; primals_137: "f32[10]"; primals_138: "f32[240, 10, 1, 1]"; primals_139: "f32[240]"; primals_140: "f32[80, 240, 1, 1]"; primals_141: "f32[480, 80, 1, 1]"; primals_142: "f32[480, 1, 3, 3]"; primals_143: "f32[20, 480, 1, 1]"; primals_144: "f32[20]"; primals_145: "f32[480, 20, 1, 1]"; primals_146: "f32[480]"; primals_147: "f32[80, 480, 1, 1]"; primals_148: "f32[480, 80, 1, 1]"; primals_149: "f32[480, 1, 3, 3]"; primals_150: "f32[20, 480, 1, 1]"; primals_151: "f32[20]"; primals_152: "f32[480, 20, 1, 1]"; primals_153: "f32[480]"; primals_154: "f32[80, 480, 1, 1]"; primals_155: "f32[480, 80, 1, 1]"; primals_156: "f32[480, 1, 5, 5]"; primals_157: "f32[20, 480, 1, 1]"; primals_158: "f32[20]"; primals_159: "f32[480, 20, 1, 1]"; primals_160: "f32[480]"; primals_161: "f32[112, 480, 1, 1]"; primals_162: "f32[672, 112, 1, 1]"; primals_163: "f32[672, 1, 5, 5]"; primals_164: "f32[28, 672, 1, 1]"; primals_165: "f32[28]"; primals_166: "f32[672, 28, 1, 1]"; primals_167: "f32[672]"; primals_168: "f32[112, 672, 1, 1]"; primals_169: "f32[672, 112, 1, 1]"; primals_170: "f32[672, 1, 5, 5]"; primals_171: "f32[28, 672, 1, 1]"; primals_172: "f32[28]"; primals_173: "f32[672, 28, 1, 1]"; primals_174: "f32[672]"; primals_175: "f32[112, 672, 1, 1]"; primals_176: "f32[672, 112, 1, 1]"; primals_177: "f32[672, 1, 5, 5]"; primals_178: "f32[28, 672, 1, 1]"; primals_179: "f32[28]"; primals_180: "f32[672, 28, 1, 1]"; primals_181: "f32[672]"; primals_182: "f32[192, 672, 1, 1]"; primals_183: "f32[1152, 192, 1, 1]"; primals_184: "f32[1152, 1, 5, 5]"; primals_185: "f32[48, 1152, 1, 1]"; primals_186: "f32[48]"; primals_187: "f32[1152, 48, 1, 1]"; primals_188: "f32[1152]"; primals_189: "f32[192, 1152, 1, 1]"; primals_190: "f32[1152, 192, 1, 1]"; primals_191: "f32[1152, 1, 5, 5]"; primals_192: "f32[48, 1152, 1, 1]"; primals_193: "f32[48]"; primals_194: "f32[1152, 48, 1, 1]"; primals_195: "f32[1152]"; primals_196: "f32[192, 1152, 1, 1]"; primals_197: "f32[1152, 192, 1, 1]"; primals_198: "f32[1152, 1, 5, 5]"; primals_199: "f32[48, 1152, 1, 1]"; primals_200: "f32[48]"; primals_201: "f32[1152, 48, 1, 1]"; primals_202: "f32[1152]"; primals_203: "f32[192, 1152, 1, 1]"; primals_204: "f32[1152, 192, 1, 1]"; primals_205: "f32[1152, 1, 3, 3]"; primals_206: "f32[48, 1152, 1, 1]"; primals_207: "f32[48]"; primals_208: "f32[1152, 48, 1, 1]"; primals_209: "f32[1152]"; primals_210: "f32[320, 1152, 1, 1]"; primals_211: "f32[1280, 320, 1, 1]"; primals_212: "f32[1000, 1280]"; primals_213: "f32[1000]"; primals_214: "f32[32]"; primals_215: "f32[32]"; primals_216: "f32[32]"; primals_217: "f32[32]"; primals_218: "f32[16]"; primals_219: "f32[16]"; primals_220: "f32[96]"; primals_221: "f32[96]"; primals_222: "f32[96]"; primals_223: "f32[96]"; primals_224: "f32[24]"; primals_225: "f32[24]"; primals_226: "f32[144]"; primals_227: "f32[144]"; primals_228: "f32[144]"; primals_229: "f32[144]"; primals_230: "f32[24]"; primals_231: "f32[24]"; primals_232: "f32[144]"; primals_233: "f32[144]"; primals_234: "f32[144]"; primals_235: "f32[144]"; primals_236: "f32[40]"; primals_237: "f32[40]"; primals_238: "f32[240]"; primals_239: "f32[240]"; primals_240: "f32[240]"; primals_241: "f32[240]"; primals_242: "f32[40]"; primals_243: "f32[40]"; primals_244: "f32[240]"; primals_245: "f32[240]"; primals_246: "f32[240]"; primals_247: "f32[240]"; primals_248: "f32[80]"; primals_249: "f32[80]"; primals_250: "f32[480]"; primals_251: "f32[480]"; primals_252: "f32[480]"; primals_253: "f32[480]"; primals_254: "f32[80]"; primals_255: "f32[80]"; primals_256: "f32[480]"; primals_257: "f32[480]"; primals_258: "f32[480]"; primals_259: "f32[480]"; primals_260: "f32[80]"; primals_261: "f32[80]"; primals_262: "f32[480]"; primals_263: "f32[480]"; primals_264: "f32[480]"; primals_265: "f32[480]"; primals_266: "f32[112]"; primals_267: "f32[112]"; primals_268: "f32[672]"; primals_269: "f32[672]"; primals_270: "f32[672]"; primals_271: "f32[672]"; primals_272: "f32[112]"; primals_273: "f32[112]"; primals_274: "f32[672]"; primals_275: "f32[672]"; primals_276: "f32[672]"; primals_277: "f32[672]"; primals_278: "f32[112]"; primals_279: "f32[112]"; primals_280: "f32[672]"; primals_281: "f32[672]"; primals_282: "f32[672]"; primals_283: "f32[672]"; primals_284: "f32[192]"; primals_285: "f32[192]"; primals_286: "f32[1152]"; primals_287: "f32[1152]"; primals_288: "f32[1152]"; primals_289: "f32[1152]"; primals_290: "f32[192]"; primals_291: "f32[192]"; primals_292: "f32[1152]"; primals_293: "f32[1152]"; primals_294: "f32[1152]"; primals_295: "f32[1152]"; primals_296: "f32[192]"; primals_297: "f32[192]"; primals_298: "f32[1152]"; primals_299: "f32[1152]"; primals_300: "f32[1152]"; primals_301: "f32[1152]"; primals_302: "f32[192]"; primals_303: "f32[192]"; primals_304: "f32[1152]"; primals_305: "f32[1152]"; primals_306: "f32[1152]"; primals_307: "f32[1152]"; primals_308: "f32[320]"; primals_309: "f32[320]"; primals_310: "f32[1280]"; primals_311: "f32[1280]"; primals_312: "f32[4, 3, 224, 224]"; tangents_1: "f32[4, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:162, code: x = self.conv_stem(x)
    convolution: "f32[4, 32, 112, 112]" = torch.ops.aten.convolution.default(primals_312, primals_99, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type: "f32[32]" = torch.ops.prims.convert_element_type.default(primals_214, torch.float32)
    convert_element_type_1: "f32[32]" = torch.ops.prims.convert_element_type.default(primals_215, torch.float32)
    add: "f32[32]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1e-05);  convert_element_type_1 = None
    sqrt: "f32[32]" = torch.ops.aten.sqrt.default(add);  add = None
    reciprocal: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
    mul: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
    unsqueeze: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
    unsqueeze_1: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    unsqueeze_2: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
    unsqueeze_3: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    sub: "f32[4, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  unsqueeze_1 = None
    mul_1: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
    unsqueeze_4: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_1, -1)
    unsqueeze_5: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_2: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
    unsqueeze_6: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1);  primals_2 = None
    unsqueeze_7: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_1: "f32[4, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone: "f32[4, 32, 112, 112]" = torch.ops.aten.clone.default(add_1)
    sigmoid: "f32[4, 32, 112, 112]" = torch.ops.aten.sigmoid.default(add_1)
    mul_3: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(add_1, sigmoid);  add_1 = sigmoid = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_1: "f32[4, 32, 112, 112]" = torch.ops.aten.convolution.default(mul_3, primals_100, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_2: "f32[32]" = torch.ops.prims.convert_element_type.default(primals_216, torch.float32)
    convert_element_type_3: "f32[32]" = torch.ops.prims.convert_element_type.default(primals_217, torch.float32)
    add_2: "f32[32]" = torch.ops.aten.add.Tensor(convert_element_type_3, 1e-05);  convert_element_type_3 = None
    sqrt_1: "f32[32]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
    reciprocal_1: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_4: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_8: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, -1);  convert_element_type_2 = None
    unsqueeze_9: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    unsqueeze_10: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_4, -1);  mul_4 = None
    unsqueeze_11: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    sub_1: "f32[4, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  unsqueeze_9 = None
    mul_5: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1)
    unsqueeze_13: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_6: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_5, unsqueeze_13);  mul_5 = unsqueeze_13 = None
    unsqueeze_14: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_4, -1);  primals_4 = None
    unsqueeze_15: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_3: "f32[4, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_15);  mul_6 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_1: "f32[4, 32, 112, 112]" = torch.ops.aten.clone.default(add_3)
    sigmoid_1: "f32[4, 32, 112, 112]" = torch.ops.aten.sigmoid.default(add_3)
    mul_7: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(add_3, sigmoid_1);  add_3 = sigmoid_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean: "f32[4, 32, 1, 1]" = torch.ops.aten.mean.dim(mul_7, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_2: "f32[4, 8, 1, 1]" = torch.ops.aten.convolution.default(mean, primals_101, primals_102, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_2: "f32[4, 8, 1, 1]" = torch.ops.aten.clone.default(convolution_2)
    sigmoid_2: "f32[4, 8, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_2)
    mul_8: "f32[4, 8, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_2, sigmoid_2);  convolution_2 = sigmoid_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_3: "f32[4, 32, 1, 1]" = torch.ops.aten.convolution.default(mul_8, primals_103, primals_104, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_3: "f32[4, 32, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_3);  convolution_3 = None
    alias: "f32[4, 32, 1, 1]" = torch.ops.aten.alias.default(sigmoid_3)
    mul_9: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_7, sigmoid_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_4: "f32[4, 16, 112, 112]" = torch.ops.aten.convolution.default(mul_9, primals_105, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_4: "f32[16]" = torch.ops.prims.convert_element_type.default(primals_218, torch.float32)
    convert_element_type_5: "f32[16]" = torch.ops.prims.convert_element_type.default(primals_219, torch.float32)
    add_4: "f32[16]" = torch.ops.aten.add.Tensor(convert_element_type_5, 1e-05);  convert_element_type_5 = None
    sqrt_2: "f32[16]" = torch.ops.aten.sqrt.default(add_4);  add_4 = None
    reciprocal_2: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_10: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_16: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, -1);  convert_element_type_4 = None
    unsqueeze_17: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    unsqueeze_18: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_10, -1);  mul_10 = None
    unsqueeze_19: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    sub_2: "f32[4, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_17);  unsqueeze_17 = None
    mul_11: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_21: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_12: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_11, unsqueeze_21);  mul_11 = unsqueeze_21 = None
    unsqueeze_22: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_23: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_5: "f32[4, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_12, unsqueeze_23);  mul_12 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_5: "f32[4, 96, 112, 112]" = torch.ops.aten.convolution.default(add_5, primals_106, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_6: "f32[96]" = torch.ops.prims.convert_element_type.default(primals_220, torch.float32)
    convert_element_type_7: "f32[96]" = torch.ops.prims.convert_element_type.default(primals_221, torch.float32)
    add_6: "f32[96]" = torch.ops.aten.add.Tensor(convert_element_type_7, 1e-05);  convert_element_type_7 = None
    sqrt_3: "f32[96]" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
    reciprocal_3: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
    mul_13: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
    unsqueeze_24: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_6, -1);  convert_element_type_6 = None
    unsqueeze_25: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    unsqueeze_26: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_13, -1);  mul_13 = None
    unsqueeze_27: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    sub_3: "f32[4, 96, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_25);  unsqueeze_25 = None
    mul_14: "f32[4, 96, 112, 112]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
    unsqueeze_28: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1)
    unsqueeze_29: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_15: "f32[4, 96, 112, 112]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_29);  mul_14 = unsqueeze_29 = None
    unsqueeze_30: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1);  primals_8 = None
    unsqueeze_31: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_7: "f32[4, 96, 112, 112]" = torch.ops.aten.add.Tensor(mul_15, unsqueeze_31);  mul_15 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_3: "f32[4, 96, 112, 112]" = torch.ops.aten.clone.default(add_7)
    sigmoid_4: "f32[4, 96, 112, 112]" = torch.ops.aten.sigmoid.default(add_7)
    mul_16: "f32[4, 96, 112, 112]" = torch.ops.aten.mul.Tensor(add_7, sigmoid_4);  add_7 = sigmoid_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_6: "f32[4, 96, 56, 56]" = torch.ops.aten.convolution.default(mul_16, primals_107, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 96)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_8: "f32[96]" = torch.ops.prims.convert_element_type.default(primals_222, torch.float32)
    convert_element_type_9: "f32[96]" = torch.ops.prims.convert_element_type.default(primals_223, torch.float32)
    add_8: "f32[96]" = torch.ops.aten.add.Tensor(convert_element_type_9, 1e-05);  convert_element_type_9 = None
    sqrt_4: "f32[96]" = torch.ops.aten.sqrt.default(add_8);  add_8 = None
    reciprocal_4: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
    mul_17: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
    unsqueeze_32: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_8, -1);  convert_element_type_8 = None
    unsqueeze_33: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    unsqueeze_34: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_17, -1);  mul_17 = None
    unsqueeze_35: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    sub_4: "f32[4, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_33);  unsqueeze_33 = None
    mul_18: "f32[4, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
    unsqueeze_36: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1)
    unsqueeze_37: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_19: "f32[4, 96, 56, 56]" = torch.ops.aten.mul.Tensor(mul_18, unsqueeze_37);  mul_18 = unsqueeze_37 = None
    unsqueeze_38: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_10, -1);  primals_10 = None
    unsqueeze_39: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_9: "f32[4, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_19, unsqueeze_39);  mul_19 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_4: "f32[4, 96, 56, 56]" = torch.ops.aten.clone.default(add_9)
    sigmoid_5: "f32[4, 96, 56, 56]" = torch.ops.aten.sigmoid.default(add_9)
    mul_20: "f32[4, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_9, sigmoid_5);  add_9 = sigmoid_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_1: "f32[4, 96, 1, 1]" = torch.ops.aten.mean.dim(mul_20, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_7: "f32[4, 4, 1, 1]" = torch.ops.aten.convolution.default(mean_1, primals_108, primals_109, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_5: "f32[4, 4, 1, 1]" = torch.ops.aten.clone.default(convolution_7)
    sigmoid_6: "f32[4, 4, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_7)
    mul_21: "f32[4, 4, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_7, sigmoid_6);  convolution_7 = sigmoid_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_8: "f32[4, 96, 1, 1]" = torch.ops.aten.convolution.default(mul_21, primals_110, primals_111, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_7: "f32[4, 96, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_8);  convolution_8 = None
    alias_1: "f32[4, 96, 1, 1]" = torch.ops.aten.alias.default(sigmoid_7)
    mul_22: "f32[4, 96, 56, 56]" = torch.ops.aten.mul.Tensor(mul_20, sigmoid_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_9: "f32[4, 24, 56, 56]" = torch.ops.aten.convolution.default(mul_22, primals_112, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_10: "f32[24]" = torch.ops.prims.convert_element_type.default(primals_224, torch.float32)
    convert_element_type_11: "f32[24]" = torch.ops.prims.convert_element_type.default(primals_225, torch.float32)
    add_10: "f32[24]" = torch.ops.aten.add.Tensor(convert_element_type_11, 1e-05);  convert_element_type_11 = None
    sqrt_5: "f32[24]" = torch.ops.aten.sqrt.default(add_10);  add_10 = None
    reciprocal_5: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
    mul_23: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
    unsqueeze_40: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_10, -1);  convert_element_type_10 = None
    unsqueeze_41: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    unsqueeze_42: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_23, -1);  mul_23 = None
    unsqueeze_43: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    sub_5: "f32[4, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_41);  unsqueeze_41 = None
    mul_24: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
    unsqueeze_44: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_45: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_25: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_24, unsqueeze_45);  mul_24 = unsqueeze_45 = None
    unsqueeze_46: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_47: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_11: "f32[4, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_25, unsqueeze_47);  mul_25 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_10: "f32[4, 144, 56, 56]" = torch.ops.aten.convolution.default(add_11, primals_113, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_12: "f32[144]" = torch.ops.prims.convert_element_type.default(primals_226, torch.float32)
    convert_element_type_13: "f32[144]" = torch.ops.prims.convert_element_type.default(primals_227, torch.float32)
    add_12: "f32[144]" = torch.ops.aten.add.Tensor(convert_element_type_13, 1e-05);  convert_element_type_13 = None
    sqrt_6: "f32[144]" = torch.ops.aten.sqrt.default(add_12);  add_12 = None
    reciprocal_6: "f32[144]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
    mul_26: "f32[144]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
    unsqueeze_48: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_12, -1);  convert_element_type_12 = None
    unsqueeze_49: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    unsqueeze_50: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(mul_26, -1);  mul_26 = None
    unsqueeze_51: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    sub_6: "f32[4, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_49);  unsqueeze_49 = None
    mul_27: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
    unsqueeze_52: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1)
    unsqueeze_53: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_28: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_27, unsqueeze_53);  mul_27 = unsqueeze_53 = None
    unsqueeze_54: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1);  primals_14 = None
    unsqueeze_55: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_13: "f32[4, 144, 56, 56]" = torch.ops.aten.add.Tensor(mul_28, unsqueeze_55);  mul_28 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_6: "f32[4, 144, 56, 56]" = torch.ops.aten.clone.default(add_13)
    sigmoid_8: "f32[4, 144, 56, 56]" = torch.ops.aten.sigmoid.default(add_13)
    mul_29: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(add_13, sigmoid_8);  add_13 = sigmoid_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_11: "f32[4, 144, 56, 56]" = torch.ops.aten.convolution.default(mul_29, primals_114, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 144)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_14: "f32[144]" = torch.ops.prims.convert_element_type.default(primals_228, torch.float32)
    convert_element_type_15: "f32[144]" = torch.ops.prims.convert_element_type.default(primals_229, torch.float32)
    add_14: "f32[144]" = torch.ops.aten.add.Tensor(convert_element_type_15, 1e-05);  convert_element_type_15 = None
    sqrt_7: "f32[144]" = torch.ops.aten.sqrt.default(add_14);  add_14 = None
    reciprocal_7: "f32[144]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
    mul_30: "f32[144]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
    unsqueeze_56: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_14, -1);  convert_element_type_14 = None
    unsqueeze_57: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    unsqueeze_58: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(mul_30, -1);  mul_30 = None
    unsqueeze_59: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    sub_7: "f32[4, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_57);  unsqueeze_57 = None
    mul_31: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
    unsqueeze_60: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1)
    unsqueeze_61: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_32: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_31, unsqueeze_61);  mul_31 = unsqueeze_61 = None
    unsqueeze_62: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1);  primals_16 = None
    unsqueeze_63: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_15: "f32[4, 144, 56, 56]" = torch.ops.aten.add.Tensor(mul_32, unsqueeze_63);  mul_32 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_7: "f32[4, 144, 56, 56]" = torch.ops.aten.clone.default(add_15)
    sigmoid_9: "f32[4, 144, 56, 56]" = torch.ops.aten.sigmoid.default(add_15)
    mul_33: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(add_15, sigmoid_9);  add_15 = sigmoid_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_2: "f32[4, 144, 1, 1]" = torch.ops.aten.mean.dim(mul_33, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_12: "f32[4, 6, 1, 1]" = torch.ops.aten.convolution.default(mean_2, primals_115, primals_116, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_8: "f32[4, 6, 1, 1]" = torch.ops.aten.clone.default(convolution_12)
    sigmoid_10: "f32[4, 6, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_12)
    mul_34: "f32[4, 6, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_12, sigmoid_10);  convolution_12 = sigmoid_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_13: "f32[4, 144, 1, 1]" = torch.ops.aten.convolution.default(mul_34, primals_117, primals_118, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_11: "f32[4, 144, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_13);  convolution_13 = None
    alias_2: "f32[4, 144, 1, 1]" = torch.ops.aten.alias.default(sigmoid_11)
    mul_35: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_33, sigmoid_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_14: "f32[4, 24, 56, 56]" = torch.ops.aten.convolution.default(mul_35, primals_119, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_16: "f32[24]" = torch.ops.prims.convert_element_type.default(primals_230, torch.float32)
    convert_element_type_17: "f32[24]" = torch.ops.prims.convert_element_type.default(primals_231, torch.float32)
    add_16: "f32[24]" = torch.ops.aten.add.Tensor(convert_element_type_17, 1e-05);  convert_element_type_17 = None
    sqrt_8: "f32[24]" = torch.ops.aten.sqrt.default(add_16);  add_16 = None
    reciprocal_8: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
    mul_36: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
    unsqueeze_64: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_16, -1);  convert_element_type_16 = None
    unsqueeze_65: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    unsqueeze_66: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_36, -1);  mul_36 = None
    unsqueeze_67: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    sub_8: "f32[4, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_65);  unsqueeze_65 = None
    mul_37: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
    unsqueeze_68: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_69: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_38: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_69);  mul_37 = unsqueeze_69 = None
    unsqueeze_70: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_71: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_17: "f32[4, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_38, unsqueeze_71);  mul_38 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_18: "f32[4, 24, 56, 56]" = torch.ops.aten.add.Tensor(add_17, add_11);  add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_15: "f32[4, 144, 56, 56]" = torch.ops.aten.convolution.default(add_18, primals_120, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_18: "f32[144]" = torch.ops.prims.convert_element_type.default(primals_232, torch.float32)
    convert_element_type_19: "f32[144]" = torch.ops.prims.convert_element_type.default(primals_233, torch.float32)
    add_19: "f32[144]" = torch.ops.aten.add.Tensor(convert_element_type_19, 1e-05);  convert_element_type_19 = None
    sqrt_9: "f32[144]" = torch.ops.aten.sqrt.default(add_19);  add_19 = None
    reciprocal_9: "f32[144]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
    mul_39: "f32[144]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
    unsqueeze_72: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_18, -1);  convert_element_type_18 = None
    unsqueeze_73: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    unsqueeze_74: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(mul_39, -1);  mul_39 = None
    unsqueeze_75: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    sub_9: "f32[4, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_73);  unsqueeze_73 = None
    mul_40: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
    unsqueeze_76: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
    unsqueeze_77: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_41: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_40, unsqueeze_77);  mul_40 = unsqueeze_77 = None
    unsqueeze_78: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
    unsqueeze_79: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_20: "f32[4, 144, 56, 56]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_79);  mul_41 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_9: "f32[4, 144, 56, 56]" = torch.ops.aten.clone.default(add_20)
    sigmoid_12: "f32[4, 144, 56, 56]" = torch.ops.aten.sigmoid.default(add_20)
    mul_42: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(add_20, sigmoid_12);  add_20 = sigmoid_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_16: "f32[4, 144, 28, 28]" = torch.ops.aten.convolution.default(mul_42, primals_121, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 144)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_20: "f32[144]" = torch.ops.prims.convert_element_type.default(primals_234, torch.float32)
    convert_element_type_21: "f32[144]" = torch.ops.prims.convert_element_type.default(primals_235, torch.float32)
    add_21: "f32[144]" = torch.ops.aten.add.Tensor(convert_element_type_21, 1e-05);  convert_element_type_21 = None
    sqrt_10: "f32[144]" = torch.ops.aten.sqrt.default(add_21);  add_21 = None
    reciprocal_10: "f32[144]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
    mul_43: "f32[144]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
    unsqueeze_80: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_20, -1);  convert_element_type_20 = None
    unsqueeze_81: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    unsqueeze_82: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(mul_43, -1);  mul_43 = None
    unsqueeze_83: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    sub_10: "f32[4, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_81);  unsqueeze_81 = None
    mul_44: "f32[4, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
    unsqueeze_84: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1)
    unsqueeze_85: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_45: "f32[4, 144, 28, 28]" = torch.ops.aten.mul.Tensor(mul_44, unsqueeze_85);  mul_44 = unsqueeze_85 = None
    unsqueeze_86: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1);  primals_22 = None
    unsqueeze_87: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_22: "f32[4, 144, 28, 28]" = torch.ops.aten.add.Tensor(mul_45, unsqueeze_87);  mul_45 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_10: "f32[4, 144, 28, 28]" = torch.ops.aten.clone.default(add_22)
    sigmoid_13: "f32[4, 144, 28, 28]" = torch.ops.aten.sigmoid.default(add_22)
    mul_46: "f32[4, 144, 28, 28]" = torch.ops.aten.mul.Tensor(add_22, sigmoid_13);  add_22 = sigmoid_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_3: "f32[4, 144, 1, 1]" = torch.ops.aten.mean.dim(mul_46, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_17: "f32[4, 6, 1, 1]" = torch.ops.aten.convolution.default(mean_3, primals_122, primals_123, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_11: "f32[4, 6, 1, 1]" = torch.ops.aten.clone.default(convolution_17)
    sigmoid_14: "f32[4, 6, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_17)
    mul_47: "f32[4, 6, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_17, sigmoid_14);  convolution_17 = sigmoid_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_18: "f32[4, 144, 1, 1]" = torch.ops.aten.convolution.default(mul_47, primals_124, primals_125, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_15: "f32[4, 144, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_18);  convolution_18 = None
    alias_3: "f32[4, 144, 1, 1]" = torch.ops.aten.alias.default(sigmoid_15)
    mul_48: "f32[4, 144, 28, 28]" = torch.ops.aten.mul.Tensor(mul_46, sigmoid_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_19: "f32[4, 40, 28, 28]" = torch.ops.aten.convolution.default(mul_48, primals_126, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_22: "f32[40]" = torch.ops.prims.convert_element_type.default(primals_236, torch.float32)
    convert_element_type_23: "f32[40]" = torch.ops.prims.convert_element_type.default(primals_237, torch.float32)
    add_23: "f32[40]" = torch.ops.aten.add.Tensor(convert_element_type_23, 1e-05);  convert_element_type_23 = None
    sqrt_11: "f32[40]" = torch.ops.aten.sqrt.default(add_23);  add_23 = None
    reciprocal_11: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
    mul_49: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
    unsqueeze_88: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_22, -1);  convert_element_type_22 = None
    unsqueeze_89: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    unsqueeze_90: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_49, -1);  mul_49 = None
    unsqueeze_91: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    sub_11: "f32[4, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_89);  unsqueeze_89 = None
    mul_50: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
    unsqueeze_92: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_93: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_51: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_50, unsqueeze_93);  mul_50 = unsqueeze_93 = None
    unsqueeze_94: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_95: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_24: "f32[4, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_51, unsqueeze_95);  mul_51 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_20: "f32[4, 240, 28, 28]" = torch.ops.aten.convolution.default(add_24, primals_127, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_24: "f32[240]" = torch.ops.prims.convert_element_type.default(primals_238, torch.float32)
    convert_element_type_25: "f32[240]" = torch.ops.prims.convert_element_type.default(primals_239, torch.float32)
    add_25: "f32[240]" = torch.ops.aten.add.Tensor(convert_element_type_25, 1e-05);  convert_element_type_25 = None
    sqrt_12: "f32[240]" = torch.ops.aten.sqrt.default(add_25);  add_25 = None
    reciprocal_12: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
    mul_52: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
    unsqueeze_96: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_24, -1);  convert_element_type_24 = None
    unsqueeze_97: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    unsqueeze_98: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_52, -1);  mul_52 = None
    unsqueeze_99: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    sub_12: "f32[4, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_97);  unsqueeze_97 = None
    mul_53: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
    unsqueeze_100: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1)
    unsqueeze_101: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_54: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_53, unsqueeze_101);  mul_53 = unsqueeze_101 = None
    unsqueeze_102: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1);  primals_26 = None
    unsqueeze_103: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_26: "f32[4, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_54, unsqueeze_103);  mul_54 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_12: "f32[4, 240, 28, 28]" = torch.ops.aten.clone.default(add_26)
    sigmoid_16: "f32[4, 240, 28, 28]" = torch.ops.aten.sigmoid.default(add_26)
    mul_55: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_26, sigmoid_16);  add_26 = sigmoid_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_21: "f32[4, 240, 28, 28]" = torch.ops.aten.convolution.default(mul_55, primals_128, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 240)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_26: "f32[240]" = torch.ops.prims.convert_element_type.default(primals_240, torch.float32)
    convert_element_type_27: "f32[240]" = torch.ops.prims.convert_element_type.default(primals_241, torch.float32)
    add_27: "f32[240]" = torch.ops.aten.add.Tensor(convert_element_type_27, 1e-05);  convert_element_type_27 = None
    sqrt_13: "f32[240]" = torch.ops.aten.sqrt.default(add_27);  add_27 = None
    reciprocal_13: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
    mul_56: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
    unsqueeze_104: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_26, -1);  convert_element_type_26 = None
    unsqueeze_105: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    unsqueeze_106: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_56, -1);  mul_56 = None
    unsqueeze_107: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    sub_13: "f32[4, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_105);  unsqueeze_105 = None
    mul_57: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
    unsqueeze_108: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1)
    unsqueeze_109: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_58: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_57, unsqueeze_109);  mul_57 = unsqueeze_109 = None
    unsqueeze_110: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1);  primals_28 = None
    unsqueeze_111: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_28: "f32[4, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_58, unsqueeze_111);  mul_58 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_13: "f32[4, 240, 28, 28]" = torch.ops.aten.clone.default(add_28)
    sigmoid_17: "f32[4, 240, 28, 28]" = torch.ops.aten.sigmoid.default(add_28)
    mul_59: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_28, sigmoid_17);  add_28 = sigmoid_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_4: "f32[4, 240, 1, 1]" = torch.ops.aten.mean.dim(mul_59, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_22: "f32[4, 10, 1, 1]" = torch.ops.aten.convolution.default(mean_4, primals_129, primals_130, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_14: "f32[4, 10, 1, 1]" = torch.ops.aten.clone.default(convolution_22)
    sigmoid_18: "f32[4, 10, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_22)
    mul_60: "f32[4, 10, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_22, sigmoid_18);  convolution_22 = sigmoid_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_23: "f32[4, 240, 1, 1]" = torch.ops.aten.convolution.default(mul_60, primals_131, primals_132, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_19: "f32[4, 240, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_23);  convolution_23 = None
    alias_4: "f32[4, 240, 1, 1]" = torch.ops.aten.alias.default(sigmoid_19)
    mul_61: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_59, sigmoid_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_24: "f32[4, 40, 28, 28]" = torch.ops.aten.convolution.default(mul_61, primals_133, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_28: "f32[40]" = torch.ops.prims.convert_element_type.default(primals_242, torch.float32)
    convert_element_type_29: "f32[40]" = torch.ops.prims.convert_element_type.default(primals_243, torch.float32)
    add_29: "f32[40]" = torch.ops.aten.add.Tensor(convert_element_type_29, 1e-05);  convert_element_type_29 = None
    sqrt_14: "f32[40]" = torch.ops.aten.sqrt.default(add_29);  add_29 = None
    reciprocal_14: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
    mul_62: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
    unsqueeze_112: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_28, -1);  convert_element_type_28 = None
    unsqueeze_113: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    unsqueeze_114: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_62, -1);  mul_62 = None
    unsqueeze_115: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    sub_14: "f32[4, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_113);  unsqueeze_113 = None
    mul_63: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
    unsqueeze_116: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_117: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_64: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_117);  mul_63 = unsqueeze_117 = None
    unsqueeze_118: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_119: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_30: "f32[4, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_64, unsqueeze_119);  mul_64 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_31: "f32[4, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_30, add_24);  add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_25: "f32[4, 240, 28, 28]" = torch.ops.aten.convolution.default(add_31, primals_134, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_30: "f32[240]" = torch.ops.prims.convert_element_type.default(primals_244, torch.float32)
    convert_element_type_31: "f32[240]" = torch.ops.prims.convert_element_type.default(primals_245, torch.float32)
    add_32: "f32[240]" = torch.ops.aten.add.Tensor(convert_element_type_31, 1e-05);  convert_element_type_31 = None
    sqrt_15: "f32[240]" = torch.ops.aten.sqrt.default(add_32);  add_32 = None
    reciprocal_15: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
    mul_65: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
    unsqueeze_120: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_30, -1);  convert_element_type_30 = None
    unsqueeze_121: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    unsqueeze_122: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_65, -1);  mul_65 = None
    unsqueeze_123: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    sub_15: "f32[4, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_121);  unsqueeze_121 = None
    mul_66: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
    unsqueeze_124: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1)
    unsqueeze_125: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_67: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_66, unsqueeze_125);  mul_66 = unsqueeze_125 = None
    unsqueeze_126: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1);  primals_32 = None
    unsqueeze_127: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_33: "f32[4, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_67, unsqueeze_127);  mul_67 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_15: "f32[4, 240, 28, 28]" = torch.ops.aten.clone.default(add_33)
    sigmoid_20: "f32[4, 240, 28, 28]" = torch.ops.aten.sigmoid.default(add_33)
    mul_68: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_33, sigmoid_20);  add_33 = sigmoid_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_26: "f32[4, 240, 14, 14]" = torch.ops.aten.convolution.default(mul_68, primals_135, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 240)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_32: "f32[240]" = torch.ops.prims.convert_element_type.default(primals_246, torch.float32)
    convert_element_type_33: "f32[240]" = torch.ops.prims.convert_element_type.default(primals_247, torch.float32)
    add_34: "f32[240]" = torch.ops.aten.add.Tensor(convert_element_type_33, 1e-05);  convert_element_type_33 = None
    sqrt_16: "f32[240]" = torch.ops.aten.sqrt.default(add_34);  add_34 = None
    reciprocal_16: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
    mul_69: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
    unsqueeze_128: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_32, -1);  convert_element_type_32 = None
    unsqueeze_129: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    unsqueeze_130: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_69, -1);  mul_69 = None
    unsqueeze_131: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    sub_16: "f32[4, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_129);  unsqueeze_129 = None
    mul_70: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
    unsqueeze_132: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1)
    unsqueeze_133: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_71: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_133);  mul_70 = unsqueeze_133 = None
    unsqueeze_134: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1);  primals_34 = None
    unsqueeze_135: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_35: "f32[4, 240, 14, 14]" = torch.ops.aten.add.Tensor(mul_71, unsqueeze_135);  mul_71 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_16: "f32[4, 240, 14, 14]" = torch.ops.aten.clone.default(add_35)
    sigmoid_21: "f32[4, 240, 14, 14]" = torch.ops.aten.sigmoid.default(add_35)
    mul_72: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(add_35, sigmoid_21);  add_35 = sigmoid_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_5: "f32[4, 240, 1, 1]" = torch.ops.aten.mean.dim(mul_72, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_27: "f32[4, 10, 1, 1]" = torch.ops.aten.convolution.default(mean_5, primals_136, primals_137, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_17: "f32[4, 10, 1, 1]" = torch.ops.aten.clone.default(convolution_27)
    sigmoid_22: "f32[4, 10, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_27)
    mul_73: "f32[4, 10, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_27, sigmoid_22);  convolution_27 = sigmoid_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_28: "f32[4, 240, 1, 1]" = torch.ops.aten.convolution.default(mul_73, primals_138, primals_139, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_23: "f32[4, 240, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_28);  convolution_28 = None
    alias_5: "f32[4, 240, 1, 1]" = torch.ops.aten.alias.default(sigmoid_23)
    mul_74: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_72, sigmoid_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_29: "f32[4, 80, 14, 14]" = torch.ops.aten.convolution.default(mul_74, primals_140, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_34: "f32[80]" = torch.ops.prims.convert_element_type.default(primals_248, torch.float32)
    convert_element_type_35: "f32[80]" = torch.ops.prims.convert_element_type.default(primals_249, torch.float32)
    add_36: "f32[80]" = torch.ops.aten.add.Tensor(convert_element_type_35, 1e-05);  convert_element_type_35 = None
    sqrt_17: "f32[80]" = torch.ops.aten.sqrt.default(add_36);  add_36 = None
    reciprocal_17: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
    mul_75: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
    unsqueeze_136: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_34, -1);  convert_element_type_34 = None
    unsqueeze_137: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    unsqueeze_138: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_75, -1);  mul_75 = None
    unsqueeze_139: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    sub_17: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_137);  unsqueeze_137 = None
    mul_76: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_139);  sub_17 = unsqueeze_139 = None
    unsqueeze_140: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_141: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_77: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_76, unsqueeze_141);  mul_76 = unsqueeze_141 = None
    unsqueeze_142: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_143: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_37: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_77, unsqueeze_143);  mul_77 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_30: "f32[4, 480, 14, 14]" = torch.ops.aten.convolution.default(add_37, primals_141, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_36: "f32[480]" = torch.ops.prims.convert_element_type.default(primals_250, torch.float32)
    convert_element_type_37: "f32[480]" = torch.ops.prims.convert_element_type.default(primals_251, torch.float32)
    add_38: "f32[480]" = torch.ops.aten.add.Tensor(convert_element_type_37, 1e-05);  convert_element_type_37 = None
    sqrt_18: "f32[480]" = torch.ops.aten.sqrt.default(add_38);  add_38 = None
    reciprocal_18: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
    mul_78: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
    unsqueeze_144: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_36, -1);  convert_element_type_36 = None
    unsqueeze_145: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    unsqueeze_146: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_78, -1);  mul_78 = None
    unsqueeze_147: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    sub_18: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_145);  unsqueeze_145 = None
    mul_79: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_147);  sub_18 = unsqueeze_147 = None
    unsqueeze_148: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1)
    unsqueeze_149: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_80: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_79, unsqueeze_149);  mul_79 = unsqueeze_149 = None
    unsqueeze_150: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1);  primals_38 = None
    unsqueeze_151: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_39: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_80, unsqueeze_151);  mul_80 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_18: "f32[4, 480, 14, 14]" = torch.ops.aten.clone.default(add_39)
    sigmoid_24: "f32[4, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_39)
    mul_81: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_39, sigmoid_24);  add_39 = sigmoid_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_31: "f32[4, 480, 14, 14]" = torch.ops.aten.convolution.default(mul_81, primals_142, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_38: "f32[480]" = torch.ops.prims.convert_element_type.default(primals_252, torch.float32)
    convert_element_type_39: "f32[480]" = torch.ops.prims.convert_element_type.default(primals_253, torch.float32)
    add_40: "f32[480]" = torch.ops.aten.add.Tensor(convert_element_type_39, 1e-05);  convert_element_type_39 = None
    sqrt_19: "f32[480]" = torch.ops.aten.sqrt.default(add_40);  add_40 = None
    reciprocal_19: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
    mul_82: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
    unsqueeze_152: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_38, -1);  convert_element_type_38 = None
    unsqueeze_153: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    unsqueeze_154: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_82, -1);  mul_82 = None
    unsqueeze_155: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    sub_19: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_153);  unsqueeze_153 = None
    mul_83: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_155);  sub_19 = unsqueeze_155 = None
    unsqueeze_156: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1)
    unsqueeze_157: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_84: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_83, unsqueeze_157);  mul_83 = unsqueeze_157 = None
    unsqueeze_158: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1);  primals_40 = None
    unsqueeze_159: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_41: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_84, unsqueeze_159);  mul_84 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_19: "f32[4, 480, 14, 14]" = torch.ops.aten.clone.default(add_41)
    sigmoid_25: "f32[4, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_41)
    mul_85: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_41, sigmoid_25);  add_41 = sigmoid_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_6: "f32[4, 480, 1, 1]" = torch.ops.aten.mean.dim(mul_85, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_32: "f32[4, 20, 1, 1]" = torch.ops.aten.convolution.default(mean_6, primals_143, primals_144, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_20: "f32[4, 20, 1, 1]" = torch.ops.aten.clone.default(convolution_32)
    sigmoid_26: "f32[4, 20, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_32)
    mul_86: "f32[4, 20, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_32, sigmoid_26);  convolution_32 = sigmoid_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_33: "f32[4, 480, 1, 1]" = torch.ops.aten.convolution.default(mul_86, primals_145, primals_146, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_27: "f32[4, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_33);  convolution_33 = None
    alias_6: "f32[4, 480, 1, 1]" = torch.ops.aten.alias.default(sigmoid_27)
    mul_87: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_85, sigmoid_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_34: "f32[4, 80, 14, 14]" = torch.ops.aten.convolution.default(mul_87, primals_147, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_40: "f32[80]" = torch.ops.prims.convert_element_type.default(primals_254, torch.float32)
    convert_element_type_41: "f32[80]" = torch.ops.prims.convert_element_type.default(primals_255, torch.float32)
    add_42: "f32[80]" = torch.ops.aten.add.Tensor(convert_element_type_41, 1e-05);  convert_element_type_41 = None
    sqrt_20: "f32[80]" = torch.ops.aten.sqrt.default(add_42);  add_42 = None
    reciprocal_20: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
    mul_88: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
    unsqueeze_160: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_40, -1);  convert_element_type_40 = None
    unsqueeze_161: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    unsqueeze_162: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_88, -1);  mul_88 = None
    unsqueeze_163: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    sub_20: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_161);  unsqueeze_161 = None
    mul_89: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_163);  sub_20 = unsqueeze_163 = None
    unsqueeze_164: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_165: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_90: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_89, unsqueeze_165);  mul_89 = unsqueeze_165 = None
    unsqueeze_166: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_167: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_43: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_167);  mul_90 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_44: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_43, add_37);  add_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_35: "f32[4, 480, 14, 14]" = torch.ops.aten.convolution.default(add_44, primals_148, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_42: "f32[480]" = torch.ops.prims.convert_element_type.default(primals_256, torch.float32)
    convert_element_type_43: "f32[480]" = torch.ops.prims.convert_element_type.default(primals_257, torch.float32)
    add_45: "f32[480]" = torch.ops.aten.add.Tensor(convert_element_type_43, 1e-05);  convert_element_type_43 = None
    sqrt_21: "f32[480]" = torch.ops.aten.sqrt.default(add_45);  add_45 = None
    reciprocal_21: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
    mul_91: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
    unsqueeze_168: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_42, -1);  convert_element_type_42 = None
    unsqueeze_169: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    unsqueeze_170: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_91, -1);  mul_91 = None
    unsqueeze_171: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    sub_21: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_169);  unsqueeze_169 = None
    mul_92: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_21, unsqueeze_171);  sub_21 = unsqueeze_171 = None
    unsqueeze_172: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1)
    unsqueeze_173: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_93: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_92, unsqueeze_173);  mul_92 = unsqueeze_173 = None
    unsqueeze_174: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1);  primals_44 = None
    unsqueeze_175: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_46: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_93, unsqueeze_175);  mul_93 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_21: "f32[4, 480, 14, 14]" = torch.ops.aten.clone.default(add_46)
    sigmoid_28: "f32[4, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_46)
    mul_94: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_46, sigmoid_28);  add_46 = sigmoid_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_36: "f32[4, 480, 14, 14]" = torch.ops.aten.convolution.default(mul_94, primals_149, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_44: "f32[480]" = torch.ops.prims.convert_element_type.default(primals_258, torch.float32)
    convert_element_type_45: "f32[480]" = torch.ops.prims.convert_element_type.default(primals_259, torch.float32)
    add_47: "f32[480]" = torch.ops.aten.add.Tensor(convert_element_type_45, 1e-05);  convert_element_type_45 = None
    sqrt_22: "f32[480]" = torch.ops.aten.sqrt.default(add_47);  add_47 = None
    reciprocal_22: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
    mul_95: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
    unsqueeze_176: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_44, -1);  convert_element_type_44 = None
    unsqueeze_177: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    unsqueeze_178: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_95, -1);  mul_95 = None
    unsqueeze_179: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    sub_22: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_177);  unsqueeze_177 = None
    mul_96: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_179);  sub_22 = unsqueeze_179 = None
    unsqueeze_180: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1)
    unsqueeze_181: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_97: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_96, unsqueeze_181);  mul_96 = unsqueeze_181 = None
    unsqueeze_182: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1);  primals_46 = None
    unsqueeze_183: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_48: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_183);  mul_97 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_22: "f32[4, 480, 14, 14]" = torch.ops.aten.clone.default(add_48)
    sigmoid_29: "f32[4, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_48)
    mul_98: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_48, sigmoid_29);  add_48 = sigmoid_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_7: "f32[4, 480, 1, 1]" = torch.ops.aten.mean.dim(mul_98, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_37: "f32[4, 20, 1, 1]" = torch.ops.aten.convolution.default(mean_7, primals_150, primals_151, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_23: "f32[4, 20, 1, 1]" = torch.ops.aten.clone.default(convolution_37)
    sigmoid_30: "f32[4, 20, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_37)
    mul_99: "f32[4, 20, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_37, sigmoid_30);  convolution_37 = sigmoid_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_38: "f32[4, 480, 1, 1]" = torch.ops.aten.convolution.default(mul_99, primals_152, primals_153, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_31: "f32[4, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_38);  convolution_38 = None
    alias_7: "f32[4, 480, 1, 1]" = torch.ops.aten.alias.default(sigmoid_31)
    mul_100: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_98, sigmoid_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_39: "f32[4, 80, 14, 14]" = torch.ops.aten.convolution.default(mul_100, primals_154, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_46: "f32[80]" = torch.ops.prims.convert_element_type.default(primals_260, torch.float32)
    convert_element_type_47: "f32[80]" = torch.ops.prims.convert_element_type.default(primals_261, torch.float32)
    add_49: "f32[80]" = torch.ops.aten.add.Tensor(convert_element_type_47, 1e-05);  convert_element_type_47 = None
    sqrt_23: "f32[80]" = torch.ops.aten.sqrt.default(add_49);  add_49 = None
    reciprocal_23: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
    mul_101: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
    unsqueeze_184: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_46, -1);  convert_element_type_46 = None
    unsqueeze_185: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    unsqueeze_186: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_101, -1);  mul_101 = None
    unsqueeze_187: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    sub_23: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_185);  unsqueeze_185 = None
    mul_102: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_187);  sub_23 = unsqueeze_187 = None
    unsqueeze_188: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_189: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_103: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_102, unsqueeze_189);  mul_102 = unsqueeze_189 = None
    unsqueeze_190: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_191: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_50: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_103, unsqueeze_191);  mul_103 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_51: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_50, add_44);  add_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_40: "f32[4, 480, 14, 14]" = torch.ops.aten.convolution.default(add_51, primals_155, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_48: "f32[480]" = torch.ops.prims.convert_element_type.default(primals_262, torch.float32)
    convert_element_type_49: "f32[480]" = torch.ops.prims.convert_element_type.default(primals_263, torch.float32)
    add_52: "f32[480]" = torch.ops.aten.add.Tensor(convert_element_type_49, 1e-05);  convert_element_type_49 = None
    sqrt_24: "f32[480]" = torch.ops.aten.sqrt.default(add_52);  add_52 = None
    reciprocal_24: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
    mul_104: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
    unsqueeze_192: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_48, -1);  convert_element_type_48 = None
    unsqueeze_193: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    unsqueeze_194: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_104, -1);  mul_104 = None
    unsqueeze_195: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    sub_24: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_193);  unsqueeze_193 = None
    mul_105: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_195);  sub_24 = unsqueeze_195 = None
    unsqueeze_196: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1)
    unsqueeze_197: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_106: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_197);  mul_105 = unsqueeze_197 = None
    unsqueeze_198: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1);  primals_50 = None
    unsqueeze_199: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_53: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_106, unsqueeze_199);  mul_106 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_24: "f32[4, 480, 14, 14]" = torch.ops.aten.clone.default(add_53)
    sigmoid_32: "f32[4, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_53)
    mul_107: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_53, sigmoid_32);  add_53 = sigmoid_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_41: "f32[4, 480, 14, 14]" = torch.ops.aten.convolution.default(mul_107, primals_156, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 480)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_50: "f32[480]" = torch.ops.prims.convert_element_type.default(primals_264, torch.float32)
    convert_element_type_51: "f32[480]" = torch.ops.prims.convert_element_type.default(primals_265, torch.float32)
    add_54: "f32[480]" = torch.ops.aten.add.Tensor(convert_element_type_51, 1e-05);  convert_element_type_51 = None
    sqrt_25: "f32[480]" = torch.ops.aten.sqrt.default(add_54);  add_54 = None
    reciprocal_25: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
    mul_108: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
    unsqueeze_200: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_50, -1);  convert_element_type_50 = None
    unsqueeze_201: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    unsqueeze_202: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_108, -1);  mul_108 = None
    unsqueeze_203: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    sub_25: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_201);  unsqueeze_201 = None
    mul_109: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_203);  sub_25 = unsqueeze_203 = None
    unsqueeze_204: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1)
    unsqueeze_205: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_110: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_109, unsqueeze_205);  mul_109 = unsqueeze_205 = None
    unsqueeze_206: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1);  primals_52 = None
    unsqueeze_207: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_55: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_110, unsqueeze_207);  mul_110 = unsqueeze_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_25: "f32[4, 480, 14, 14]" = torch.ops.aten.clone.default(add_55)
    sigmoid_33: "f32[4, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_55)
    mul_111: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_55, sigmoid_33);  add_55 = sigmoid_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_8: "f32[4, 480, 1, 1]" = torch.ops.aten.mean.dim(mul_111, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_42: "f32[4, 20, 1, 1]" = torch.ops.aten.convolution.default(mean_8, primals_157, primals_158, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_26: "f32[4, 20, 1, 1]" = torch.ops.aten.clone.default(convolution_42)
    sigmoid_34: "f32[4, 20, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_42)
    mul_112: "f32[4, 20, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_42, sigmoid_34);  convolution_42 = sigmoid_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_43: "f32[4, 480, 1, 1]" = torch.ops.aten.convolution.default(mul_112, primals_159, primals_160, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_35: "f32[4, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_43);  convolution_43 = None
    alias_8: "f32[4, 480, 1, 1]" = torch.ops.aten.alias.default(sigmoid_35)
    mul_113: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_111, sigmoid_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_44: "f32[4, 112, 14, 14]" = torch.ops.aten.convolution.default(mul_113, primals_161, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_52: "f32[112]" = torch.ops.prims.convert_element_type.default(primals_266, torch.float32)
    convert_element_type_53: "f32[112]" = torch.ops.prims.convert_element_type.default(primals_267, torch.float32)
    add_56: "f32[112]" = torch.ops.aten.add.Tensor(convert_element_type_53, 1e-05);  convert_element_type_53 = None
    sqrt_26: "f32[112]" = torch.ops.aten.sqrt.default(add_56);  add_56 = None
    reciprocal_26: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
    mul_114: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
    unsqueeze_208: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_52, -1);  convert_element_type_52 = None
    unsqueeze_209: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    unsqueeze_210: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_114, -1);  mul_114 = None
    unsqueeze_211: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    sub_26: "f32[4, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_209);  unsqueeze_209 = None
    mul_115: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_211);  sub_26 = unsqueeze_211 = None
    unsqueeze_212: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_213: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_116: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(mul_115, unsqueeze_213);  mul_115 = unsqueeze_213 = None
    unsqueeze_214: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_215: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_57: "f32[4, 112, 14, 14]" = torch.ops.aten.add.Tensor(mul_116, unsqueeze_215);  mul_116 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_45: "f32[4, 672, 14, 14]" = torch.ops.aten.convolution.default(add_57, primals_162, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_54: "f32[672]" = torch.ops.prims.convert_element_type.default(primals_268, torch.float32)
    convert_element_type_55: "f32[672]" = torch.ops.prims.convert_element_type.default(primals_269, torch.float32)
    add_58: "f32[672]" = torch.ops.aten.add.Tensor(convert_element_type_55, 1e-05);  convert_element_type_55 = None
    sqrt_27: "f32[672]" = torch.ops.aten.sqrt.default(add_58);  add_58 = None
    reciprocal_27: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
    mul_117: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
    unsqueeze_216: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_54, -1);  convert_element_type_54 = None
    unsqueeze_217: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    unsqueeze_218: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_117, -1);  mul_117 = None
    unsqueeze_219: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    sub_27: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_217);  unsqueeze_217 = None
    mul_118: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_219);  sub_27 = unsqueeze_219 = None
    unsqueeze_220: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1)
    unsqueeze_221: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_119: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_118, unsqueeze_221);  mul_118 = unsqueeze_221 = None
    unsqueeze_222: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1);  primals_56 = None
    unsqueeze_223: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_59: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_119, unsqueeze_223);  mul_119 = unsqueeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_27: "f32[4, 672, 14, 14]" = torch.ops.aten.clone.default(add_59)
    sigmoid_36: "f32[4, 672, 14, 14]" = torch.ops.aten.sigmoid.default(add_59)
    mul_120: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_59, sigmoid_36);  add_59 = sigmoid_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_46: "f32[4, 672, 14, 14]" = torch.ops.aten.convolution.default(mul_120, primals_163, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 672)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_56: "f32[672]" = torch.ops.prims.convert_element_type.default(primals_270, torch.float32)
    convert_element_type_57: "f32[672]" = torch.ops.prims.convert_element_type.default(primals_271, torch.float32)
    add_60: "f32[672]" = torch.ops.aten.add.Tensor(convert_element_type_57, 1e-05);  convert_element_type_57 = None
    sqrt_28: "f32[672]" = torch.ops.aten.sqrt.default(add_60);  add_60 = None
    reciprocal_28: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
    mul_121: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
    unsqueeze_224: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_56, -1);  convert_element_type_56 = None
    unsqueeze_225: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    unsqueeze_226: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_121, -1);  mul_121 = None
    unsqueeze_227: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    sub_28: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_225);  unsqueeze_225 = None
    mul_122: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_227);  sub_28 = unsqueeze_227 = None
    unsqueeze_228: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1)
    unsqueeze_229: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_123: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_122, unsqueeze_229);  mul_122 = unsqueeze_229 = None
    unsqueeze_230: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_58, -1);  primals_58 = None
    unsqueeze_231: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_61: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_123, unsqueeze_231);  mul_123 = unsqueeze_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_28: "f32[4, 672, 14, 14]" = torch.ops.aten.clone.default(add_61)
    sigmoid_37: "f32[4, 672, 14, 14]" = torch.ops.aten.sigmoid.default(add_61)
    mul_124: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_61, sigmoid_37);  add_61 = sigmoid_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_9: "f32[4, 672, 1, 1]" = torch.ops.aten.mean.dim(mul_124, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_47: "f32[4, 28, 1, 1]" = torch.ops.aten.convolution.default(mean_9, primals_164, primals_165, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_29: "f32[4, 28, 1, 1]" = torch.ops.aten.clone.default(convolution_47)
    sigmoid_38: "f32[4, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_47)
    mul_125: "f32[4, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_47, sigmoid_38);  convolution_47 = sigmoid_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_48: "f32[4, 672, 1, 1]" = torch.ops.aten.convolution.default(mul_125, primals_166, primals_167, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_39: "f32[4, 672, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_48);  convolution_48 = None
    alias_9: "f32[4, 672, 1, 1]" = torch.ops.aten.alias.default(sigmoid_39)
    mul_126: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_124, sigmoid_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_49: "f32[4, 112, 14, 14]" = torch.ops.aten.convolution.default(mul_126, primals_168, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_58: "f32[112]" = torch.ops.prims.convert_element_type.default(primals_272, torch.float32)
    convert_element_type_59: "f32[112]" = torch.ops.prims.convert_element_type.default(primals_273, torch.float32)
    add_62: "f32[112]" = torch.ops.aten.add.Tensor(convert_element_type_59, 1e-05);  convert_element_type_59 = None
    sqrt_29: "f32[112]" = torch.ops.aten.sqrt.default(add_62);  add_62 = None
    reciprocal_29: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
    mul_127: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
    unsqueeze_232: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_58, -1);  convert_element_type_58 = None
    unsqueeze_233: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    unsqueeze_234: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_127, -1);  mul_127 = None
    unsqueeze_235: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    sub_29: "f32[4, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_233);  unsqueeze_233 = None
    mul_128: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_235);  sub_29 = unsqueeze_235 = None
    unsqueeze_236: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_237: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_129: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(mul_128, unsqueeze_237);  mul_128 = unsqueeze_237 = None
    unsqueeze_238: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_239: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_63: "f32[4, 112, 14, 14]" = torch.ops.aten.add.Tensor(mul_129, unsqueeze_239);  mul_129 = unsqueeze_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_64: "f32[4, 112, 14, 14]" = torch.ops.aten.add.Tensor(add_63, add_57);  add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_50: "f32[4, 672, 14, 14]" = torch.ops.aten.convolution.default(add_64, primals_169, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_60: "f32[672]" = torch.ops.prims.convert_element_type.default(primals_274, torch.float32)
    convert_element_type_61: "f32[672]" = torch.ops.prims.convert_element_type.default(primals_275, torch.float32)
    add_65: "f32[672]" = torch.ops.aten.add.Tensor(convert_element_type_61, 1e-05);  convert_element_type_61 = None
    sqrt_30: "f32[672]" = torch.ops.aten.sqrt.default(add_65);  add_65 = None
    reciprocal_30: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
    mul_130: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
    unsqueeze_240: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_60, -1);  convert_element_type_60 = None
    unsqueeze_241: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    unsqueeze_242: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_130, -1);  mul_130 = None
    unsqueeze_243: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    sub_30: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_241);  unsqueeze_241 = None
    mul_131: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_243);  sub_30 = unsqueeze_243 = None
    unsqueeze_244: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_61, -1)
    unsqueeze_245: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_132: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_131, unsqueeze_245);  mul_131 = unsqueeze_245 = None
    unsqueeze_246: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1);  primals_62 = None
    unsqueeze_247: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_66: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_247);  mul_132 = unsqueeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_30: "f32[4, 672, 14, 14]" = torch.ops.aten.clone.default(add_66)
    sigmoid_40: "f32[4, 672, 14, 14]" = torch.ops.aten.sigmoid.default(add_66)
    mul_133: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_66, sigmoid_40);  add_66 = sigmoid_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_51: "f32[4, 672, 14, 14]" = torch.ops.aten.convolution.default(mul_133, primals_170, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 672)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_62: "f32[672]" = torch.ops.prims.convert_element_type.default(primals_276, torch.float32)
    convert_element_type_63: "f32[672]" = torch.ops.prims.convert_element_type.default(primals_277, torch.float32)
    add_67: "f32[672]" = torch.ops.aten.add.Tensor(convert_element_type_63, 1e-05);  convert_element_type_63 = None
    sqrt_31: "f32[672]" = torch.ops.aten.sqrt.default(add_67);  add_67 = None
    reciprocal_31: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
    mul_134: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
    unsqueeze_248: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_62, -1);  convert_element_type_62 = None
    unsqueeze_249: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    unsqueeze_250: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_134, -1);  mul_134 = None
    unsqueeze_251: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    sub_31: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_249);  unsqueeze_249 = None
    mul_135: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_251);  sub_31 = unsqueeze_251 = None
    unsqueeze_252: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1)
    unsqueeze_253: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_136: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_135, unsqueeze_253);  mul_135 = unsqueeze_253 = None
    unsqueeze_254: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_64, -1);  primals_64 = None
    unsqueeze_255: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_68: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_136, unsqueeze_255);  mul_136 = unsqueeze_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_31: "f32[4, 672, 14, 14]" = torch.ops.aten.clone.default(add_68)
    sigmoid_41: "f32[4, 672, 14, 14]" = torch.ops.aten.sigmoid.default(add_68)
    mul_137: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_68, sigmoid_41);  add_68 = sigmoid_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_10: "f32[4, 672, 1, 1]" = torch.ops.aten.mean.dim(mul_137, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_52: "f32[4, 28, 1, 1]" = torch.ops.aten.convolution.default(mean_10, primals_171, primals_172, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_32: "f32[4, 28, 1, 1]" = torch.ops.aten.clone.default(convolution_52)
    sigmoid_42: "f32[4, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_52)
    mul_138: "f32[4, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_52, sigmoid_42);  convolution_52 = sigmoid_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_53: "f32[4, 672, 1, 1]" = torch.ops.aten.convolution.default(mul_138, primals_173, primals_174, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_43: "f32[4, 672, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_53);  convolution_53 = None
    alias_10: "f32[4, 672, 1, 1]" = torch.ops.aten.alias.default(sigmoid_43)
    mul_139: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_137, sigmoid_43)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_54: "f32[4, 112, 14, 14]" = torch.ops.aten.convolution.default(mul_139, primals_175, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_64: "f32[112]" = torch.ops.prims.convert_element_type.default(primals_278, torch.float32)
    convert_element_type_65: "f32[112]" = torch.ops.prims.convert_element_type.default(primals_279, torch.float32)
    add_69: "f32[112]" = torch.ops.aten.add.Tensor(convert_element_type_65, 1e-05);  convert_element_type_65 = None
    sqrt_32: "f32[112]" = torch.ops.aten.sqrt.default(add_69);  add_69 = None
    reciprocal_32: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
    mul_140: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
    unsqueeze_256: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_64, -1);  convert_element_type_64 = None
    unsqueeze_257: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    unsqueeze_258: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_140, -1);  mul_140 = None
    unsqueeze_259: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    sub_32: "f32[4, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_257);  unsqueeze_257 = None
    mul_141: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_259);  sub_32 = unsqueeze_259 = None
    unsqueeze_260: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
    unsqueeze_261: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
    mul_142: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(mul_141, unsqueeze_261);  mul_141 = unsqueeze_261 = None
    unsqueeze_262: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
    unsqueeze_263: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
    add_70: "f32[4, 112, 14, 14]" = torch.ops.aten.add.Tensor(mul_142, unsqueeze_263);  mul_142 = unsqueeze_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_71: "f32[4, 112, 14, 14]" = torch.ops.aten.add.Tensor(add_70, add_64);  add_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_55: "f32[4, 672, 14, 14]" = torch.ops.aten.convolution.default(add_71, primals_176, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_66: "f32[672]" = torch.ops.prims.convert_element_type.default(primals_280, torch.float32)
    convert_element_type_67: "f32[672]" = torch.ops.prims.convert_element_type.default(primals_281, torch.float32)
    add_72: "f32[672]" = torch.ops.aten.add.Tensor(convert_element_type_67, 1e-05);  convert_element_type_67 = None
    sqrt_33: "f32[672]" = torch.ops.aten.sqrt.default(add_72);  add_72 = None
    reciprocal_33: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
    mul_143: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
    unsqueeze_264: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_66, -1);  convert_element_type_66 = None
    unsqueeze_265: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
    unsqueeze_266: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_143, -1);  mul_143 = None
    unsqueeze_267: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
    sub_33: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_265);  unsqueeze_265 = None
    mul_144: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_33, unsqueeze_267);  sub_33 = unsqueeze_267 = None
    unsqueeze_268: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_67, -1)
    unsqueeze_269: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
    mul_145: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_144, unsqueeze_269);  mul_144 = unsqueeze_269 = None
    unsqueeze_270: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1);  primals_68 = None
    unsqueeze_271: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
    add_73: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_145, unsqueeze_271);  mul_145 = unsqueeze_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_33: "f32[4, 672, 14, 14]" = torch.ops.aten.clone.default(add_73)
    sigmoid_44: "f32[4, 672, 14, 14]" = torch.ops.aten.sigmoid.default(add_73)
    mul_146: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_73, sigmoid_44);  add_73 = sigmoid_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_56: "f32[4, 672, 7, 7]" = torch.ops.aten.convolution.default(mul_146, primals_177, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 672)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_68: "f32[672]" = torch.ops.prims.convert_element_type.default(primals_282, torch.float32)
    convert_element_type_69: "f32[672]" = torch.ops.prims.convert_element_type.default(primals_283, torch.float32)
    add_74: "f32[672]" = torch.ops.aten.add.Tensor(convert_element_type_69, 1e-05);  convert_element_type_69 = None
    sqrt_34: "f32[672]" = torch.ops.aten.sqrt.default(add_74);  add_74 = None
    reciprocal_34: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
    mul_147: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
    unsqueeze_272: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_68, -1);  convert_element_type_68 = None
    unsqueeze_273: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
    unsqueeze_274: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_147, -1);  mul_147 = None
    unsqueeze_275: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
    sub_34: "f32[4, 672, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_273);  unsqueeze_273 = None
    mul_148: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_275);  sub_34 = unsqueeze_275 = None
    unsqueeze_276: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1)
    unsqueeze_277: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
    mul_149: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(mul_148, unsqueeze_277);  mul_148 = unsqueeze_277 = None
    unsqueeze_278: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_70, -1);  primals_70 = None
    unsqueeze_279: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
    add_75: "f32[4, 672, 7, 7]" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_279);  mul_149 = unsqueeze_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_34: "f32[4, 672, 7, 7]" = torch.ops.aten.clone.default(add_75)
    sigmoid_45: "f32[4, 672, 7, 7]" = torch.ops.aten.sigmoid.default(add_75)
    mul_150: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(add_75, sigmoid_45);  add_75 = sigmoid_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_11: "f32[4, 672, 1, 1]" = torch.ops.aten.mean.dim(mul_150, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_57: "f32[4, 28, 1, 1]" = torch.ops.aten.convolution.default(mean_11, primals_178, primals_179, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_35: "f32[4, 28, 1, 1]" = torch.ops.aten.clone.default(convolution_57)
    sigmoid_46: "f32[4, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_57)
    mul_151: "f32[4, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_57, sigmoid_46);  convolution_57 = sigmoid_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_58: "f32[4, 672, 1, 1]" = torch.ops.aten.convolution.default(mul_151, primals_180, primals_181, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_47: "f32[4, 672, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_58);  convolution_58 = None
    alias_11: "f32[4, 672, 1, 1]" = torch.ops.aten.alias.default(sigmoid_47)
    mul_152: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(mul_150, sigmoid_47)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_59: "f32[4, 192, 7, 7]" = torch.ops.aten.convolution.default(mul_152, primals_182, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_70: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_284, torch.float32)
    convert_element_type_71: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_285, torch.float32)
    add_76: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_71, 1e-05);  convert_element_type_71 = None
    sqrt_35: "f32[192]" = torch.ops.aten.sqrt.default(add_76);  add_76 = None
    reciprocal_35: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
    mul_153: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
    unsqueeze_280: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_70, -1);  convert_element_type_70 = None
    unsqueeze_281: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
    unsqueeze_282: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_153, -1);  mul_153 = None
    unsqueeze_283: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
    sub_35: "f32[4, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_281);  unsqueeze_281 = None
    mul_154: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_283);  sub_35 = unsqueeze_283 = None
    unsqueeze_284: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_285: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
    mul_155: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_285);  mul_154 = unsqueeze_285 = None
    unsqueeze_286: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_287: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
    add_77: "f32[4, 192, 7, 7]" = torch.ops.aten.add.Tensor(mul_155, unsqueeze_287);  mul_155 = unsqueeze_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_60: "f32[4, 1152, 7, 7]" = torch.ops.aten.convolution.default(add_77, primals_183, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_72: "f32[1152]" = torch.ops.prims.convert_element_type.default(primals_286, torch.float32)
    convert_element_type_73: "f32[1152]" = torch.ops.prims.convert_element_type.default(primals_287, torch.float32)
    add_78: "f32[1152]" = torch.ops.aten.add.Tensor(convert_element_type_73, 1e-05);  convert_element_type_73 = None
    sqrt_36: "f32[1152]" = torch.ops.aten.sqrt.default(add_78);  add_78 = None
    reciprocal_36: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
    mul_156: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
    unsqueeze_288: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_72, -1);  convert_element_type_72 = None
    unsqueeze_289: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
    unsqueeze_290: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_156, -1);  mul_156 = None
    unsqueeze_291: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
    sub_36: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_289);  unsqueeze_289 = None
    mul_157: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_291);  sub_36 = unsqueeze_291 = None
    unsqueeze_292: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_73, -1)
    unsqueeze_293: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
    mul_158: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_157, unsqueeze_293);  mul_157 = unsqueeze_293 = None
    unsqueeze_294: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1);  primals_74 = None
    unsqueeze_295: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
    add_79: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_158, unsqueeze_295);  mul_158 = unsqueeze_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_36: "f32[4, 1152, 7, 7]" = torch.ops.aten.clone.default(add_79)
    sigmoid_48: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_79)
    mul_159: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_79, sigmoid_48);  add_79 = sigmoid_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_61: "f32[4, 1152, 7, 7]" = torch.ops.aten.convolution.default(mul_159, primals_184, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_74: "f32[1152]" = torch.ops.prims.convert_element_type.default(primals_288, torch.float32)
    convert_element_type_75: "f32[1152]" = torch.ops.prims.convert_element_type.default(primals_289, torch.float32)
    add_80: "f32[1152]" = torch.ops.aten.add.Tensor(convert_element_type_75, 1e-05);  convert_element_type_75 = None
    sqrt_37: "f32[1152]" = torch.ops.aten.sqrt.default(add_80);  add_80 = None
    reciprocal_37: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
    mul_160: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
    unsqueeze_296: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_74, -1);  convert_element_type_74 = None
    unsqueeze_297: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
    unsqueeze_298: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_160, -1);  mul_160 = None
    unsqueeze_299: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
    sub_37: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_297);  unsqueeze_297 = None
    mul_161: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_299);  sub_37 = unsqueeze_299 = None
    unsqueeze_300: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1)
    unsqueeze_301: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
    mul_162: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_301);  mul_161 = unsqueeze_301 = None
    unsqueeze_302: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_76, -1);  primals_76 = None
    unsqueeze_303: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
    add_81: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_162, unsqueeze_303);  mul_162 = unsqueeze_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_37: "f32[4, 1152, 7, 7]" = torch.ops.aten.clone.default(add_81)
    sigmoid_49: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_81)
    mul_163: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_81, sigmoid_49);  add_81 = sigmoid_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_12: "f32[4, 1152, 1, 1]" = torch.ops.aten.mean.dim(mul_163, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_62: "f32[4, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_12, primals_185, primals_186, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_38: "f32[4, 48, 1, 1]" = torch.ops.aten.clone.default(convolution_62)
    sigmoid_50: "f32[4, 48, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_62)
    mul_164: "f32[4, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_62, sigmoid_50);  convolution_62 = sigmoid_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_63: "f32[4, 1152, 1, 1]" = torch.ops.aten.convolution.default(mul_164, primals_187, primals_188, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_51: "f32[4, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_63);  convolution_63 = None
    alias_12: "f32[4, 1152, 1, 1]" = torch.ops.aten.alias.default(sigmoid_51)
    mul_165: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_163, sigmoid_51)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_64: "f32[4, 192, 7, 7]" = torch.ops.aten.convolution.default(mul_165, primals_189, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_76: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_290, torch.float32)
    convert_element_type_77: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_291, torch.float32)
    add_82: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_77, 1e-05);  convert_element_type_77 = None
    sqrt_38: "f32[192]" = torch.ops.aten.sqrt.default(add_82);  add_82 = None
    reciprocal_38: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
    mul_166: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
    unsqueeze_304: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_76, -1);  convert_element_type_76 = None
    unsqueeze_305: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
    unsqueeze_306: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_166, -1);  mul_166 = None
    unsqueeze_307: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
    sub_38: "f32[4, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_305);  unsqueeze_305 = None
    mul_167: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_307);  sub_38 = unsqueeze_307 = None
    unsqueeze_308: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
    unsqueeze_309: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
    mul_168: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(mul_167, unsqueeze_309);  mul_167 = unsqueeze_309 = None
    unsqueeze_310: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
    unsqueeze_311: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
    add_83: "f32[4, 192, 7, 7]" = torch.ops.aten.add.Tensor(mul_168, unsqueeze_311);  mul_168 = unsqueeze_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_84: "f32[4, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_83, add_77);  add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_65: "f32[4, 1152, 7, 7]" = torch.ops.aten.convolution.default(add_84, primals_190, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_78: "f32[1152]" = torch.ops.prims.convert_element_type.default(primals_292, torch.float32)
    convert_element_type_79: "f32[1152]" = torch.ops.prims.convert_element_type.default(primals_293, torch.float32)
    add_85: "f32[1152]" = torch.ops.aten.add.Tensor(convert_element_type_79, 1e-05);  convert_element_type_79 = None
    sqrt_39: "f32[1152]" = torch.ops.aten.sqrt.default(add_85);  add_85 = None
    reciprocal_39: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
    mul_169: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
    unsqueeze_312: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_78, -1);  convert_element_type_78 = None
    unsqueeze_313: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
    unsqueeze_314: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_169, -1);  mul_169 = None
    unsqueeze_315: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
    sub_39: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_313);  unsqueeze_313 = None
    mul_170: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_315);  sub_39 = unsqueeze_315 = None
    unsqueeze_316: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_79, -1)
    unsqueeze_317: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
    mul_171: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_170, unsqueeze_317);  mul_170 = unsqueeze_317 = None
    unsqueeze_318: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1);  primals_80 = None
    unsqueeze_319: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
    add_86: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_171, unsqueeze_319);  mul_171 = unsqueeze_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_39: "f32[4, 1152, 7, 7]" = torch.ops.aten.clone.default(add_86)
    sigmoid_52: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_86)
    mul_172: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_86, sigmoid_52);  add_86 = sigmoid_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_66: "f32[4, 1152, 7, 7]" = torch.ops.aten.convolution.default(mul_172, primals_191, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_80: "f32[1152]" = torch.ops.prims.convert_element_type.default(primals_294, torch.float32)
    convert_element_type_81: "f32[1152]" = torch.ops.prims.convert_element_type.default(primals_295, torch.float32)
    add_87: "f32[1152]" = torch.ops.aten.add.Tensor(convert_element_type_81, 1e-05);  convert_element_type_81 = None
    sqrt_40: "f32[1152]" = torch.ops.aten.sqrt.default(add_87);  add_87 = None
    reciprocal_40: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
    mul_173: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
    unsqueeze_320: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_80, -1);  convert_element_type_80 = None
    unsqueeze_321: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
    unsqueeze_322: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_173, -1);  mul_173 = None
    unsqueeze_323: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
    sub_40: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_321);  unsqueeze_321 = None
    mul_174: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_323);  sub_40 = unsqueeze_323 = None
    unsqueeze_324: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1)
    unsqueeze_325: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
    mul_175: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_174, unsqueeze_325);  mul_174 = unsqueeze_325 = None
    unsqueeze_326: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_82, -1);  primals_82 = None
    unsqueeze_327: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
    add_88: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_175, unsqueeze_327);  mul_175 = unsqueeze_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_40: "f32[4, 1152, 7, 7]" = torch.ops.aten.clone.default(add_88)
    sigmoid_53: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_88)
    mul_176: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_88, sigmoid_53);  add_88 = sigmoid_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_13: "f32[4, 1152, 1, 1]" = torch.ops.aten.mean.dim(mul_176, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_67: "f32[4, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_13, primals_192, primals_193, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_41: "f32[4, 48, 1, 1]" = torch.ops.aten.clone.default(convolution_67)
    sigmoid_54: "f32[4, 48, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_67)
    mul_177: "f32[4, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_67, sigmoid_54);  convolution_67 = sigmoid_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_68: "f32[4, 1152, 1, 1]" = torch.ops.aten.convolution.default(mul_177, primals_194, primals_195, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_55: "f32[4, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_68);  convolution_68 = None
    alias_13: "f32[4, 1152, 1, 1]" = torch.ops.aten.alias.default(sigmoid_55)
    mul_178: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_176, sigmoid_55)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_69: "f32[4, 192, 7, 7]" = torch.ops.aten.convolution.default(mul_178, primals_196, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_82: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_296, torch.float32)
    convert_element_type_83: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_297, torch.float32)
    add_89: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_83, 1e-05);  convert_element_type_83 = None
    sqrt_41: "f32[192]" = torch.ops.aten.sqrt.default(add_89);  add_89 = None
    reciprocal_41: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
    mul_179: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
    unsqueeze_328: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_82, -1);  convert_element_type_82 = None
    unsqueeze_329: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
    unsqueeze_330: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_179, -1);  mul_179 = None
    unsqueeze_331: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
    sub_41: "f32[4, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_329);  unsqueeze_329 = None
    mul_180: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_331);  sub_41 = unsqueeze_331 = None
    unsqueeze_332: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1)
    unsqueeze_333: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
    mul_181: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(mul_180, unsqueeze_333);  mul_180 = unsqueeze_333 = None
    unsqueeze_334: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
    unsqueeze_335: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
    add_90: "f32[4, 192, 7, 7]" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_335);  mul_181 = unsqueeze_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_91: "f32[4, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_90, add_84);  add_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_70: "f32[4, 1152, 7, 7]" = torch.ops.aten.convolution.default(add_91, primals_197, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_84: "f32[1152]" = torch.ops.prims.convert_element_type.default(primals_298, torch.float32)
    convert_element_type_85: "f32[1152]" = torch.ops.prims.convert_element_type.default(primals_299, torch.float32)
    add_92: "f32[1152]" = torch.ops.aten.add.Tensor(convert_element_type_85, 1e-05);  convert_element_type_85 = None
    sqrt_42: "f32[1152]" = torch.ops.aten.sqrt.default(add_92);  add_92 = None
    reciprocal_42: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
    mul_182: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
    unsqueeze_336: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_84, -1);  convert_element_type_84 = None
    unsqueeze_337: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
    unsqueeze_338: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_182, -1);  mul_182 = None
    unsqueeze_339: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
    sub_42: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_337);  unsqueeze_337 = None
    mul_183: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_339);  sub_42 = unsqueeze_339 = None
    unsqueeze_340: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_85, -1)
    unsqueeze_341: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
    mul_184: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_183, unsqueeze_341);  mul_183 = unsqueeze_341 = None
    unsqueeze_342: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1);  primals_86 = None
    unsqueeze_343: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
    add_93: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_184, unsqueeze_343);  mul_184 = unsqueeze_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_42: "f32[4, 1152, 7, 7]" = torch.ops.aten.clone.default(add_93)
    sigmoid_56: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_93)
    mul_185: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_93, sigmoid_56);  add_93 = sigmoid_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_71: "f32[4, 1152, 7, 7]" = torch.ops.aten.convolution.default(mul_185, primals_198, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_86: "f32[1152]" = torch.ops.prims.convert_element_type.default(primals_300, torch.float32)
    convert_element_type_87: "f32[1152]" = torch.ops.prims.convert_element_type.default(primals_301, torch.float32)
    add_94: "f32[1152]" = torch.ops.aten.add.Tensor(convert_element_type_87, 1e-05);  convert_element_type_87 = None
    sqrt_43: "f32[1152]" = torch.ops.aten.sqrt.default(add_94);  add_94 = None
    reciprocal_43: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
    mul_186: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
    unsqueeze_344: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_86, -1);  convert_element_type_86 = None
    unsqueeze_345: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
    unsqueeze_346: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_186, -1);  mul_186 = None
    unsqueeze_347: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
    sub_43: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_345);  unsqueeze_345 = None
    mul_187: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_347);  sub_43 = unsqueeze_347 = None
    unsqueeze_348: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1)
    unsqueeze_349: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
    mul_188: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_187, unsqueeze_349);  mul_187 = unsqueeze_349 = None
    unsqueeze_350: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_88, -1);  primals_88 = None
    unsqueeze_351: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
    add_95: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_351);  mul_188 = unsqueeze_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_43: "f32[4, 1152, 7, 7]" = torch.ops.aten.clone.default(add_95)
    sigmoid_57: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_95)
    mul_189: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_95, sigmoid_57);  add_95 = sigmoid_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_14: "f32[4, 1152, 1, 1]" = torch.ops.aten.mean.dim(mul_189, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_72: "f32[4, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_14, primals_199, primals_200, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_44: "f32[4, 48, 1, 1]" = torch.ops.aten.clone.default(convolution_72)
    sigmoid_58: "f32[4, 48, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_72)
    mul_190: "f32[4, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_72, sigmoid_58);  convolution_72 = sigmoid_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_73: "f32[4, 1152, 1, 1]" = torch.ops.aten.convolution.default(mul_190, primals_201, primals_202, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_59: "f32[4, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_73);  convolution_73 = None
    alias_14: "f32[4, 1152, 1, 1]" = torch.ops.aten.alias.default(sigmoid_59)
    mul_191: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_189, sigmoid_59)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_74: "f32[4, 192, 7, 7]" = torch.ops.aten.convolution.default(mul_191, primals_203, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_88: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_302, torch.float32)
    convert_element_type_89: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_303, torch.float32)
    add_96: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_89, 1e-05);  convert_element_type_89 = None
    sqrt_44: "f32[192]" = torch.ops.aten.sqrt.default(add_96);  add_96 = None
    reciprocal_44: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
    mul_192: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
    unsqueeze_352: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_88, -1);  convert_element_type_88 = None
    unsqueeze_353: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
    unsqueeze_354: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_192, -1);  mul_192 = None
    unsqueeze_355: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
    sub_44: "f32[4, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_353);  unsqueeze_353 = None
    mul_193: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_355);  sub_44 = unsqueeze_355 = None
    unsqueeze_356: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1)
    unsqueeze_357: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
    mul_194: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(mul_193, unsqueeze_357);  mul_193 = unsqueeze_357 = None
    unsqueeze_358: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1);  primals_90 = None
    unsqueeze_359: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
    add_97: "f32[4, 192, 7, 7]" = torch.ops.aten.add.Tensor(mul_194, unsqueeze_359);  mul_194 = unsqueeze_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_98: "f32[4, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_97, add_91);  add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_75: "f32[4, 1152, 7, 7]" = torch.ops.aten.convolution.default(add_98, primals_204, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_90: "f32[1152]" = torch.ops.prims.convert_element_type.default(primals_304, torch.float32)
    convert_element_type_91: "f32[1152]" = torch.ops.prims.convert_element_type.default(primals_305, torch.float32)
    add_99: "f32[1152]" = torch.ops.aten.add.Tensor(convert_element_type_91, 1e-05);  convert_element_type_91 = None
    sqrt_45: "f32[1152]" = torch.ops.aten.sqrt.default(add_99);  add_99 = None
    reciprocal_45: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
    mul_195: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
    unsqueeze_360: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_90, -1);  convert_element_type_90 = None
    unsqueeze_361: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
    unsqueeze_362: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_195, -1);  mul_195 = None
    unsqueeze_363: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
    sub_45: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_361);  unsqueeze_361 = None
    mul_196: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_363);  sub_45 = unsqueeze_363 = None
    unsqueeze_364: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_91, -1)
    unsqueeze_365: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
    mul_197: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_365);  mul_196 = unsqueeze_365 = None
    unsqueeze_366: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1);  primals_92 = None
    unsqueeze_367: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
    add_100: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_197, unsqueeze_367);  mul_197 = unsqueeze_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_45: "f32[4, 1152, 7, 7]" = torch.ops.aten.clone.default(add_100)
    sigmoid_60: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_100)
    mul_198: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_100, sigmoid_60);  add_100 = sigmoid_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_76: "f32[4, 1152, 7, 7]" = torch.ops.aten.convolution.default(mul_198, primals_205, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1152)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_92: "f32[1152]" = torch.ops.prims.convert_element_type.default(primals_306, torch.float32)
    convert_element_type_93: "f32[1152]" = torch.ops.prims.convert_element_type.default(primals_307, torch.float32)
    add_101: "f32[1152]" = torch.ops.aten.add.Tensor(convert_element_type_93, 1e-05);  convert_element_type_93 = None
    sqrt_46: "f32[1152]" = torch.ops.aten.sqrt.default(add_101);  add_101 = None
    reciprocal_46: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
    mul_199: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
    unsqueeze_368: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_92, -1);  convert_element_type_92 = None
    unsqueeze_369: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
    unsqueeze_370: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_199, -1);  mul_199 = None
    unsqueeze_371: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
    sub_46: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_369);  unsqueeze_369 = None
    mul_200: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_371);  sub_46 = unsqueeze_371 = None
    unsqueeze_372: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_93, -1)
    unsqueeze_373: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
    mul_201: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_200, unsqueeze_373);  mul_200 = unsqueeze_373 = None
    unsqueeze_374: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_94, -1);  primals_94 = None
    unsqueeze_375: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
    add_102: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_201, unsqueeze_375);  mul_201 = unsqueeze_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_46: "f32[4, 1152, 7, 7]" = torch.ops.aten.clone.default(add_102)
    sigmoid_61: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(add_102)
    mul_202: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_102, sigmoid_61);  add_102 = sigmoid_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_15: "f32[4, 1152, 1, 1]" = torch.ops.aten.mean.dim(mul_202, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_77: "f32[4, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_15, primals_206, primals_207, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_47: "f32[4, 48, 1, 1]" = torch.ops.aten.clone.default(convolution_77)
    sigmoid_62: "f32[4, 48, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_77)
    mul_203: "f32[4, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_77, sigmoid_62);  convolution_77 = sigmoid_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_78: "f32[4, 1152, 1, 1]" = torch.ops.aten.convolution.default(mul_203, primals_208, primals_209, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_63: "f32[4, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_78);  convolution_78 = None
    alias_15: "f32[4, 1152, 1, 1]" = torch.ops.aten.alias.default(sigmoid_63)
    mul_204: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_202, sigmoid_63)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_79: "f32[4, 320, 7, 7]" = torch.ops.aten.convolution.default(mul_204, primals_210, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_94: "f32[320]" = torch.ops.prims.convert_element_type.default(primals_308, torch.float32)
    convert_element_type_95: "f32[320]" = torch.ops.prims.convert_element_type.default(primals_309, torch.float32)
    add_103: "f32[320]" = torch.ops.aten.add.Tensor(convert_element_type_95, 1e-05);  convert_element_type_95 = None
    sqrt_47: "f32[320]" = torch.ops.aten.sqrt.default(add_103);  add_103 = None
    reciprocal_47: "f32[320]" = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
    mul_205: "f32[320]" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
    unsqueeze_376: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_94, -1);  convert_element_type_94 = None
    unsqueeze_377: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
    unsqueeze_378: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(mul_205, -1);  mul_205 = None
    unsqueeze_379: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
    sub_47: "f32[4, 320, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_377);  unsqueeze_377 = None
    mul_206: "f32[4, 320, 7, 7]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_379);  sub_47 = unsqueeze_379 = None
    unsqueeze_380: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1)
    unsqueeze_381: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
    mul_207: "f32[4, 320, 7, 7]" = torch.ops.aten.mul.Tensor(mul_206, unsqueeze_381);  mul_206 = unsqueeze_381 = None
    unsqueeze_382: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1);  primals_96 = None
    unsqueeze_383: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
    add_104: "f32[4, 320, 7, 7]" = torch.ops.aten.add.Tensor(mul_207, unsqueeze_383);  mul_207 = unsqueeze_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:168, code: x = self.conv_head(x)
    convolution_80: "f32[4, 1280, 7, 7]" = torch.ops.aten.convolution.default(add_104, primals_211, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_96: "f32[1280]" = torch.ops.prims.convert_element_type.default(primals_310, torch.float32)
    convert_element_type_97: "f32[1280]" = torch.ops.prims.convert_element_type.default(primals_311, torch.float32)
    add_105: "f32[1280]" = torch.ops.aten.add.Tensor(convert_element_type_97, 1e-05);  convert_element_type_97 = None
    sqrt_48: "f32[1280]" = torch.ops.aten.sqrt.default(add_105);  add_105 = None
    reciprocal_48: "f32[1280]" = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
    mul_208: "f32[1280]" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
    unsqueeze_384: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_96, -1);  convert_element_type_96 = None
    unsqueeze_385: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
    unsqueeze_386: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(mul_208, -1);  mul_208 = None
    unsqueeze_387: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
    sub_48: "f32[4, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_385);  unsqueeze_385 = None
    mul_209: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_387);  sub_48 = unsqueeze_387 = None
    unsqueeze_388: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_97, -1)
    unsqueeze_389: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
    mul_210: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_209, unsqueeze_389);  mul_209 = unsqueeze_389 = None
    unsqueeze_390: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1);  primals_98 = None
    unsqueeze_391: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
    add_106: "f32[4, 1280, 7, 7]" = torch.ops.aten.add.Tensor(mul_210, unsqueeze_391);  mul_210 = unsqueeze_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_48: "f32[4, 1280, 7, 7]" = torch.ops.aten.clone.default(add_106)
    sigmoid_64: "f32[4, 1280, 7, 7]" = torch.ops.aten.sigmoid.default(add_106)
    mul_211: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(add_106, sigmoid_64);  add_106 = sigmoid_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean_16: "f32[4, 1280, 1, 1]" = torch.ops.aten.mean.dim(mul_211, [-1, -2], True);  mul_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view: "f32[4, 1280]" = torch.ops.aten.view.default(mean_16, [4, 1280]);  mean_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:176, code: return x if pre_logits else self.classifier(x)
    permute: "f32[1280, 1000]" = torch.ops.aten.permute.default(primals_212, [1, 0]);  primals_212 = None
    addmm: "f32[4, 1000]" = torch.ops.aten.addmm.default(primals_213, view, permute);  primals_213 = None
    permute_1: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm: "f32[4, 1280]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 4]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 1280]" = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
    permute_3: "f32[1280, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_2: "f32[4, 1280, 1, 1]" = torch.ops.aten.view.default(mm, [4, 1280, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[4, 1280, 7, 7]" = torch.ops.aten.expand.default(view_2, [4, 1280, 7, 7]);  view_2 = None
    div: "f32[4, 1280, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_65: "f32[4, 1280, 7, 7]" = torch.ops.aten.sigmoid.default(clone_48)
    full_98: "f32[4, 1280, 7, 7]" = torch.ops.aten.full.default([4, 1280, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_49: "f32[4, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(full_98, sigmoid_65);  full_98 = None
    mul_212: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(clone_48, sub_49);  clone_48 = sub_49 = None
    add_107: "f32[4, 1280, 7, 7]" = torch.ops.aten.add.Scalar(mul_212, 1);  mul_212 = None
    mul_213: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_65, add_107);  sigmoid_65 = add_107 = None
    mul_214: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(div, mul_213);  div = mul_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_108: "f32[1280]" = torch.ops.aten.add.Tensor(primals_311, 1e-05);  primals_311 = None
    rsqrt: "f32[1280]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    unsqueeze_392: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(primals_310, 0);  primals_310 = None
    unsqueeze_393: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 2);  unsqueeze_392 = None
    unsqueeze_394: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 3);  unsqueeze_393 = None
    sum_2: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_214, [0, 2, 3])
    sub_50: "f32[4, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_394);  convolution_80 = unsqueeze_394 = None
    mul_215: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_214, sub_50);  sub_50 = None
    sum_3: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_215, [0, 2, 3]);  mul_215 = None
    mul_220: "f32[1280]" = torch.ops.aten.mul.Tensor(rsqrt, primals_97);  primals_97 = None
    unsqueeze_401: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_220, 0);  mul_220 = None
    unsqueeze_402: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 2);  unsqueeze_401 = None
    unsqueeze_403: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 3);  unsqueeze_402 = None
    mul_221: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_214, unsqueeze_403);  mul_214 = unsqueeze_403 = None
    mul_222: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_3, rsqrt);  sum_3 = rsqrt = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:168, code: x = self.conv_head(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_221, add_104, primals_211, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_221 = add_104 = primals_211 = None
    getitem: "f32[4, 320, 7, 7]" = convolution_backward[0]
    getitem_1: "f32[1280, 320, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_109: "f32[320]" = torch.ops.aten.add.Tensor(primals_309, 1e-05);  primals_309 = None
    rsqrt_1: "f32[320]" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
    unsqueeze_404: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_308, 0);  primals_308 = None
    unsqueeze_405: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 2);  unsqueeze_404 = None
    unsqueeze_406: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 3);  unsqueeze_405 = None
    sum_4: "f32[320]" = torch.ops.aten.sum.dim_IntList(getitem, [0, 2, 3])
    sub_51: "f32[4, 320, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_406);  convolution_79 = unsqueeze_406 = None
    mul_223: "f32[4, 320, 7, 7]" = torch.ops.aten.mul.Tensor(getitem, sub_51);  sub_51 = None
    sum_5: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_223, [0, 2, 3]);  mul_223 = None
    mul_228: "f32[320]" = torch.ops.aten.mul.Tensor(rsqrt_1, primals_95);  primals_95 = None
    unsqueeze_413: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_228, 0);  mul_228 = None
    unsqueeze_414: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 2);  unsqueeze_413 = None
    unsqueeze_415: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 3);  unsqueeze_414 = None
    mul_229: "f32[4, 320, 7, 7]" = torch.ops.aten.mul.Tensor(getitem, unsqueeze_415);  getitem = unsqueeze_415 = None
    mul_230: "f32[320]" = torch.ops.aten.mul.Tensor(sum_5, rsqrt_1);  sum_5 = rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_229, mul_204, primals_210, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_229 = mul_204 = primals_210 = None
    getitem_3: "f32[4, 1152, 7, 7]" = convolution_backward_1[0]
    getitem_4: "f32[320, 1152, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_231: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_3, mul_202);  mul_202 = None
    mul_232: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_3, sigmoid_63);  getitem_3 = sigmoid_63 = None
    sum_6: "f32[4, 1152, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_231, [2, 3], True);  mul_231 = None
    alias_16: "f32[4, 1152, 1, 1]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    sub_52: "f32[4, 1152, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_16)
    mul_233: "f32[4, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(alias_16, sub_52);  alias_16 = sub_52 = None
    mul_234: "f32[4, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(sum_6, mul_233);  sum_6 = mul_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_234, mul_203, primals_208, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_234 = mul_203 = primals_208 = None
    getitem_6: "f32[4, 48, 1, 1]" = convolution_backward_2[0]
    getitem_7: "f32[1152, 48, 1, 1]" = convolution_backward_2[1]
    getitem_8: "f32[1152]" = convolution_backward_2[2];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_66: "f32[4, 48, 1, 1]" = torch.ops.aten.sigmoid.default(clone_47)
    full_99: "f32[4, 48, 1, 1]" = torch.ops.aten.full.default([4, 48, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_53: "f32[4, 48, 1, 1]" = torch.ops.aten.sub.Tensor(full_99, sigmoid_66);  full_99 = None
    mul_235: "f32[4, 48, 1, 1]" = torch.ops.aten.mul.Tensor(clone_47, sub_53);  clone_47 = sub_53 = None
    add_110: "f32[4, 48, 1, 1]" = torch.ops.aten.add.Scalar(mul_235, 1);  mul_235 = None
    mul_236: "f32[4, 48, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_66, add_110);  sigmoid_66 = add_110 = None
    mul_237: "f32[4, 48, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_6, mul_236);  getitem_6 = mul_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_237, mean_15, primals_206, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_237 = mean_15 = primals_206 = None
    getitem_9: "f32[4, 1152, 1, 1]" = convolution_backward_3[0]
    getitem_10: "f32[48, 1152, 1, 1]" = convolution_backward_3[1]
    getitem_11: "f32[48]" = convolution_backward_3[2];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_1: "f32[4, 1152, 7, 7]" = torch.ops.aten.expand.default(getitem_9, [4, 1152, 7, 7]);  getitem_9 = None
    div_1: "f32[4, 1152, 7, 7]" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_111: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_232, div_1);  mul_232 = div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_67: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(clone_46)
    full_100: "f32[4, 1152, 7, 7]" = torch.ops.aten.full.default([4, 1152, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_54: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(full_100, sigmoid_67);  full_100 = None
    mul_238: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(clone_46, sub_54);  clone_46 = sub_54 = None
    add_112: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Scalar(mul_238, 1);  mul_238 = None
    mul_239: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_67, add_112);  sigmoid_67 = add_112 = None
    mul_240: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_111, mul_239);  add_111 = mul_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_113: "f32[1152]" = torch.ops.aten.add.Tensor(primals_307, 1e-05);  primals_307 = None
    rsqrt_2: "f32[1152]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
    unsqueeze_416: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(primals_306, 0);  primals_306 = None
    unsqueeze_417: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 2);  unsqueeze_416 = None
    unsqueeze_418: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 3);  unsqueeze_417 = None
    sum_7: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_240, [0, 2, 3])
    sub_55: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_418);  convolution_76 = unsqueeze_418 = None
    mul_241: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_240, sub_55);  sub_55 = None
    sum_8: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_241, [0, 2, 3]);  mul_241 = None
    mul_246: "f32[1152]" = torch.ops.aten.mul.Tensor(rsqrt_2, primals_93);  primals_93 = None
    unsqueeze_425: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_246, 0);  mul_246 = None
    unsqueeze_426: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 2);  unsqueeze_425 = None
    unsqueeze_427: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 3);  unsqueeze_426 = None
    mul_247: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_240, unsqueeze_427);  mul_240 = unsqueeze_427 = None
    mul_248: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_8, rsqrt_2);  sum_8 = rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_247, mul_198, primals_205, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_247 = mul_198 = primals_205 = None
    getitem_12: "f32[4, 1152, 7, 7]" = convolution_backward_4[0]
    getitem_13: "f32[1152, 1, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_68: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(clone_45)
    full_101: "f32[4, 1152, 7, 7]" = torch.ops.aten.full.default([4, 1152, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_56: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(full_101, sigmoid_68);  full_101 = None
    mul_249: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(clone_45, sub_56);  clone_45 = sub_56 = None
    add_114: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Scalar(mul_249, 1);  mul_249 = None
    mul_250: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_68, add_114);  sigmoid_68 = add_114 = None
    mul_251: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_12, mul_250);  getitem_12 = mul_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_115: "f32[1152]" = torch.ops.aten.add.Tensor(primals_305, 1e-05);  primals_305 = None
    rsqrt_3: "f32[1152]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    unsqueeze_428: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(primals_304, 0);  primals_304 = None
    unsqueeze_429: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 2);  unsqueeze_428 = None
    unsqueeze_430: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 3);  unsqueeze_429 = None
    sum_9: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_251, [0, 2, 3])
    sub_57: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_430);  convolution_75 = unsqueeze_430 = None
    mul_252: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_251, sub_57);  sub_57 = None
    sum_10: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_252, [0, 2, 3]);  mul_252 = None
    mul_257: "f32[1152]" = torch.ops.aten.mul.Tensor(rsqrt_3, primals_91);  primals_91 = None
    unsqueeze_437: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_257, 0);  mul_257 = None
    unsqueeze_438: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 2);  unsqueeze_437 = None
    unsqueeze_439: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 3);  unsqueeze_438 = None
    mul_258: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_251, unsqueeze_439);  mul_251 = unsqueeze_439 = None
    mul_259: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_10, rsqrt_3);  sum_10 = rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_258, add_98, primals_204, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_258 = add_98 = primals_204 = None
    getitem_15: "f32[4, 192, 7, 7]" = convolution_backward_5[0]
    getitem_16: "f32[1152, 192, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_116: "f32[192]" = torch.ops.aten.add.Tensor(primals_303, 1e-05);  primals_303 = None
    rsqrt_4: "f32[192]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
    unsqueeze_440: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_302, 0);  primals_302 = None
    unsqueeze_441: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 2);  unsqueeze_440 = None
    unsqueeze_442: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 3);  unsqueeze_441 = None
    sum_11: "f32[192]" = torch.ops.aten.sum.dim_IntList(getitem_15, [0, 2, 3])
    sub_58: "f32[4, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_442);  convolution_74 = unsqueeze_442 = None
    mul_260: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_15, sub_58);  sub_58 = None
    sum_12: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_260, [0, 2, 3]);  mul_260 = None
    mul_265: "f32[192]" = torch.ops.aten.mul.Tensor(rsqrt_4, primals_89);  primals_89 = None
    unsqueeze_449: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_265, 0);  mul_265 = None
    unsqueeze_450: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
    unsqueeze_451: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 3);  unsqueeze_450 = None
    mul_266: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_15, unsqueeze_451);  unsqueeze_451 = None
    mul_267: "f32[192]" = torch.ops.aten.mul.Tensor(sum_12, rsqrt_4);  sum_12 = rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_266, mul_191, primals_203, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_266 = mul_191 = primals_203 = None
    getitem_18: "f32[4, 1152, 7, 7]" = convolution_backward_6[0]
    getitem_19: "f32[192, 1152, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_268: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_18, mul_189);  mul_189 = None
    mul_269: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_18, sigmoid_59);  getitem_18 = sigmoid_59 = None
    sum_13: "f32[4, 1152, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_268, [2, 3], True);  mul_268 = None
    alias_17: "f32[4, 1152, 1, 1]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    sub_59: "f32[4, 1152, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_17)
    mul_270: "f32[4, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(alias_17, sub_59);  alias_17 = sub_59 = None
    mul_271: "f32[4, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(sum_13, mul_270);  sum_13 = mul_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_271, mul_190, primals_201, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_271 = mul_190 = primals_201 = None
    getitem_21: "f32[4, 48, 1, 1]" = convolution_backward_7[0]
    getitem_22: "f32[1152, 48, 1, 1]" = convolution_backward_7[1]
    getitem_23: "f32[1152]" = convolution_backward_7[2];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_69: "f32[4, 48, 1, 1]" = torch.ops.aten.sigmoid.default(clone_44)
    full_102: "f32[4, 48, 1, 1]" = torch.ops.aten.full.default([4, 48, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_60: "f32[4, 48, 1, 1]" = torch.ops.aten.sub.Tensor(full_102, sigmoid_69);  full_102 = None
    mul_272: "f32[4, 48, 1, 1]" = torch.ops.aten.mul.Tensor(clone_44, sub_60);  clone_44 = sub_60 = None
    add_117: "f32[4, 48, 1, 1]" = torch.ops.aten.add.Scalar(mul_272, 1);  mul_272 = None
    mul_273: "f32[4, 48, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_69, add_117);  sigmoid_69 = add_117 = None
    mul_274: "f32[4, 48, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_21, mul_273);  getitem_21 = mul_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_274, mean_14, primals_199, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_274 = mean_14 = primals_199 = None
    getitem_24: "f32[4, 1152, 1, 1]" = convolution_backward_8[0]
    getitem_25: "f32[48, 1152, 1, 1]" = convolution_backward_8[1]
    getitem_26: "f32[48]" = convolution_backward_8[2];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_2: "f32[4, 1152, 7, 7]" = torch.ops.aten.expand.default(getitem_24, [4, 1152, 7, 7]);  getitem_24 = None
    div_2: "f32[4, 1152, 7, 7]" = torch.ops.aten.div.Scalar(expand_2, 49);  expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_118: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_269, div_2);  mul_269 = div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_70: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(clone_43)
    full_103: "f32[4, 1152, 7, 7]" = torch.ops.aten.full.default([4, 1152, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_61: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(full_103, sigmoid_70);  full_103 = None
    mul_275: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(clone_43, sub_61);  clone_43 = sub_61 = None
    add_119: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Scalar(mul_275, 1);  mul_275 = None
    mul_276: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_70, add_119);  sigmoid_70 = add_119 = None
    mul_277: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_118, mul_276);  add_118 = mul_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_120: "f32[1152]" = torch.ops.aten.add.Tensor(primals_301, 1e-05);  primals_301 = None
    rsqrt_5: "f32[1152]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
    unsqueeze_452: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(primals_300, 0);  primals_300 = None
    unsqueeze_453: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 2);  unsqueeze_452 = None
    unsqueeze_454: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 3);  unsqueeze_453 = None
    sum_14: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_277, [0, 2, 3])
    sub_62: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_454);  convolution_71 = unsqueeze_454 = None
    mul_278: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_277, sub_62);  sub_62 = None
    sum_15: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_278, [0, 2, 3]);  mul_278 = None
    mul_283: "f32[1152]" = torch.ops.aten.mul.Tensor(rsqrt_5, primals_87);  primals_87 = None
    unsqueeze_461: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_283, 0);  mul_283 = None
    unsqueeze_462: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
    unsqueeze_463: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 3);  unsqueeze_462 = None
    mul_284: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_277, unsqueeze_463);  mul_277 = unsqueeze_463 = None
    mul_285: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_15, rsqrt_5);  sum_15 = rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_284, mul_185, primals_198, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_284 = mul_185 = primals_198 = None
    getitem_27: "f32[4, 1152, 7, 7]" = convolution_backward_9[0]
    getitem_28: "f32[1152, 1, 5, 5]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_71: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(clone_42)
    full_104: "f32[4, 1152, 7, 7]" = torch.ops.aten.full.default([4, 1152, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_63: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(full_104, sigmoid_71);  full_104 = None
    mul_286: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(clone_42, sub_63);  clone_42 = sub_63 = None
    add_121: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Scalar(mul_286, 1);  mul_286 = None
    mul_287: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_71, add_121);  sigmoid_71 = add_121 = None
    mul_288: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_27, mul_287);  getitem_27 = mul_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_122: "f32[1152]" = torch.ops.aten.add.Tensor(primals_299, 1e-05);  primals_299 = None
    rsqrt_6: "f32[1152]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    unsqueeze_464: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(primals_298, 0);  primals_298 = None
    unsqueeze_465: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 2);  unsqueeze_464 = None
    unsqueeze_466: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 3);  unsqueeze_465 = None
    sum_16: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_288, [0, 2, 3])
    sub_64: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_466);  convolution_70 = unsqueeze_466 = None
    mul_289: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_288, sub_64);  sub_64 = None
    sum_17: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_289, [0, 2, 3]);  mul_289 = None
    mul_294: "f32[1152]" = torch.ops.aten.mul.Tensor(rsqrt_6, primals_85);  primals_85 = None
    unsqueeze_473: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_294, 0);  mul_294 = None
    unsqueeze_474: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
    unsqueeze_475: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 3);  unsqueeze_474 = None
    mul_295: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_288, unsqueeze_475);  mul_288 = unsqueeze_475 = None
    mul_296: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_17, rsqrt_6);  sum_17 = rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_295, add_91, primals_197, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_295 = add_91 = primals_197 = None
    getitem_30: "f32[4, 192, 7, 7]" = convolution_backward_10[0]
    getitem_31: "f32[1152, 192, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_123: "f32[4, 192, 7, 7]" = torch.ops.aten.add.Tensor(getitem_15, getitem_30);  getitem_15 = getitem_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_124: "f32[192]" = torch.ops.aten.add.Tensor(primals_297, 1e-05);  primals_297 = None
    rsqrt_7: "f32[192]" = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
    unsqueeze_476: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_296, 0);  primals_296 = None
    unsqueeze_477: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 2);  unsqueeze_476 = None
    unsqueeze_478: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 3);  unsqueeze_477 = None
    sum_18: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_123, [0, 2, 3])
    sub_65: "f32[4, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_478);  convolution_69 = unsqueeze_478 = None
    mul_297: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_123, sub_65);  sub_65 = None
    sum_19: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_297, [0, 2, 3]);  mul_297 = None
    mul_302: "f32[192]" = torch.ops.aten.mul.Tensor(rsqrt_7, primals_83);  primals_83 = None
    unsqueeze_485: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_302, 0);  mul_302 = None
    unsqueeze_486: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
    unsqueeze_487: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 3);  unsqueeze_486 = None
    mul_303: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_123, unsqueeze_487);  unsqueeze_487 = None
    mul_304: "f32[192]" = torch.ops.aten.mul.Tensor(sum_19, rsqrt_7);  sum_19 = rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_303, mul_178, primals_196, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_303 = mul_178 = primals_196 = None
    getitem_33: "f32[4, 1152, 7, 7]" = convolution_backward_11[0]
    getitem_34: "f32[192, 1152, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_305: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_33, mul_176);  mul_176 = None
    mul_306: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_33, sigmoid_55);  getitem_33 = sigmoid_55 = None
    sum_20: "f32[4, 1152, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_305, [2, 3], True);  mul_305 = None
    alias_18: "f32[4, 1152, 1, 1]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    sub_66: "f32[4, 1152, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_18)
    mul_307: "f32[4, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(alias_18, sub_66);  alias_18 = sub_66 = None
    mul_308: "f32[4, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(sum_20, mul_307);  sum_20 = mul_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_308, mul_177, primals_194, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_308 = mul_177 = primals_194 = None
    getitem_36: "f32[4, 48, 1, 1]" = convolution_backward_12[0]
    getitem_37: "f32[1152, 48, 1, 1]" = convolution_backward_12[1]
    getitem_38: "f32[1152]" = convolution_backward_12[2];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_72: "f32[4, 48, 1, 1]" = torch.ops.aten.sigmoid.default(clone_41)
    full_105: "f32[4, 48, 1, 1]" = torch.ops.aten.full.default([4, 48, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_67: "f32[4, 48, 1, 1]" = torch.ops.aten.sub.Tensor(full_105, sigmoid_72);  full_105 = None
    mul_309: "f32[4, 48, 1, 1]" = torch.ops.aten.mul.Tensor(clone_41, sub_67);  clone_41 = sub_67 = None
    add_125: "f32[4, 48, 1, 1]" = torch.ops.aten.add.Scalar(mul_309, 1);  mul_309 = None
    mul_310: "f32[4, 48, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_72, add_125);  sigmoid_72 = add_125 = None
    mul_311: "f32[4, 48, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_36, mul_310);  getitem_36 = mul_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_311, mean_13, primals_192, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_311 = mean_13 = primals_192 = None
    getitem_39: "f32[4, 1152, 1, 1]" = convolution_backward_13[0]
    getitem_40: "f32[48, 1152, 1, 1]" = convolution_backward_13[1]
    getitem_41: "f32[48]" = convolution_backward_13[2];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_3: "f32[4, 1152, 7, 7]" = torch.ops.aten.expand.default(getitem_39, [4, 1152, 7, 7]);  getitem_39 = None
    div_3: "f32[4, 1152, 7, 7]" = torch.ops.aten.div.Scalar(expand_3, 49);  expand_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_126: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_306, div_3);  mul_306 = div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_73: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(clone_40)
    full_106: "f32[4, 1152, 7, 7]" = torch.ops.aten.full.default([4, 1152, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_68: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(full_106, sigmoid_73);  full_106 = None
    mul_312: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(clone_40, sub_68);  clone_40 = sub_68 = None
    add_127: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Scalar(mul_312, 1);  mul_312 = None
    mul_313: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_73, add_127);  sigmoid_73 = add_127 = None
    mul_314: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_126, mul_313);  add_126 = mul_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_128: "f32[1152]" = torch.ops.aten.add.Tensor(primals_295, 1e-05);  primals_295 = None
    rsqrt_8: "f32[1152]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
    unsqueeze_488: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(primals_294, 0);  primals_294 = None
    unsqueeze_489: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 2);  unsqueeze_488 = None
    unsqueeze_490: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 3);  unsqueeze_489 = None
    sum_21: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_314, [0, 2, 3])
    sub_69: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_490);  convolution_66 = unsqueeze_490 = None
    mul_315: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_314, sub_69);  sub_69 = None
    sum_22: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_315, [0, 2, 3]);  mul_315 = None
    mul_320: "f32[1152]" = torch.ops.aten.mul.Tensor(rsqrt_8, primals_81);  primals_81 = None
    unsqueeze_497: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_320, 0);  mul_320 = None
    unsqueeze_498: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
    unsqueeze_499: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 3);  unsqueeze_498 = None
    mul_321: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_314, unsqueeze_499);  mul_314 = unsqueeze_499 = None
    mul_322: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_22, rsqrt_8);  sum_22 = rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_321, mul_172, primals_191, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_321 = mul_172 = primals_191 = None
    getitem_42: "f32[4, 1152, 7, 7]" = convolution_backward_14[0]
    getitem_43: "f32[1152, 1, 5, 5]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_74: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(clone_39)
    full_107: "f32[4, 1152, 7, 7]" = torch.ops.aten.full.default([4, 1152, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_70: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(full_107, sigmoid_74);  full_107 = None
    mul_323: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(clone_39, sub_70);  clone_39 = sub_70 = None
    add_129: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Scalar(mul_323, 1);  mul_323 = None
    mul_324: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_74, add_129);  sigmoid_74 = add_129 = None
    mul_325: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_42, mul_324);  getitem_42 = mul_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_130: "f32[1152]" = torch.ops.aten.add.Tensor(primals_293, 1e-05);  primals_293 = None
    rsqrt_9: "f32[1152]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
    unsqueeze_500: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(primals_292, 0);  primals_292 = None
    unsqueeze_501: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 2);  unsqueeze_500 = None
    unsqueeze_502: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 3);  unsqueeze_501 = None
    sum_23: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_325, [0, 2, 3])
    sub_71: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_502);  convolution_65 = unsqueeze_502 = None
    mul_326: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_325, sub_71);  sub_71 = None
    sum_24: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_326, [0, 2, 3]);  mul_326 = None
    mul_331: "f32[1152]" = torch.ops.aten.mul.Tensor(rsqrt_9, primals_79);  primals_79 = None
    unsqueeze_509: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_331, 0);  mul_331 = None
    unsqueeze_510: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 2);  unsqueeze_509 = None
    unsqueeze_511: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 3);  unsqueeze_510 = None
    mul_332: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_325, unsqueeze_511);  mul_325 = unsqueeze_511 = None
    mul_333: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_24, rsqrt_9);  sum_24 = rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_332, add_84, primals_190, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_332 = add_84 = primals_190 = None
    getitem_45: "f32[4, 192, 7, 7]" = convolution_backward_15[0]
    getitem_46: "f32[1152, 192, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_131: "f32[4, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_123, getitem_45);  add_123 = getitem_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_132: "f32[192]" = torch.ops.aten.add.Tensor(primals_291, 1e-05);  primals_291 = None
    rsqrt_10: "f32[192]" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
    unsqueeze_512: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_290, 0);  primals_290 = None
    unsqueeze_513: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 2);  unsqueeze_512 = None
    unsqueeze_514: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 3);  unsqueeze_513 = None
    sum_25: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_131, [0, 2, 3])
    sub_72: "f32[4, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_514);  convolution_64 = unsqueeze_514 = None
    mul_334: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_131, sub_72);  sub_72 = None
    sum_26: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_334, [0, 2, 3]);  mul_334 = None
    mul_339: "f32[192]" = torch.ops.aten.mul.Tensor(rsqrt_10, primals_77);  primals_77 = None
    unsqueeze_521: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_339, 0);  mul_339 = None
    unsqueeze_522: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 2);  unsqueeze_521 = None
    unsqueeze_523: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, 3);  unsqueeze_522 = None
    mul_340: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_131, unsqueeze_523);  unsqueeze_523 = None
    mul_341: "f32[192]" = torch.ops.aten.mul.Tensor(sum_26, rsqrt_10);  sum_26 = rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_340, mul_165, primals_189, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_340 = mul_165 = primals_189 = None
    getitem_48: "f32[4, 1152, 7, 7]" = convolution_backward_16[0]
    getitem_49: "f32[192, 1152, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_342: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_48, mul_163);  mul_163 = None
    mul_343: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_48, sigmoid_51);  getitem_48 = sigmoid_51 = None
    sum_27: "f32[4, 1152, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_342, [2, 3], True);  mul_342 = None
    alias_19: "f32[4, 1152, 1, 1]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    sub_73: "f32[4, 1152, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_19)
    mul_344: "f32[4, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(alias_19, sub_73);  alias_19 = sub_73 = None
    mul_345: "f32[4, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(sum_27, mul_344);  sum_27 = mul_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_345, mul_164, primals_187, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_345 = mul_164 = primals_187 = None
    getitem_51: "f32[4, 48, 1, 1]" = convolution_backward_17[0]
    getitem_52: "f32[1152, 48, 1, 1]" = convolution_backward_17[1]
    getitem_53: "f32[1152]" = convolution_backward_17[2];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_75: "f32[4, 48, 1, 1]" = torch.ops.aten.sigmoid.default(clone_38)
    full_108: "f32[4, 48, 1, 1]" = torch.ops.aten.full.default([4, 48, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_74: "f32[4, 48, 1, 1]" = torch.ops.aten.sub.Tensor(full_108, sigmoid_75);  full_108 = None
    mul_346: "f32[4, 48, 1, 1]" = torch.ops.aten.mul.Tensor(clone_38, sub_74);  clone_38 = sub_74 = None
    add_133: "f32[4, 48, 1, 1]" = torch.ops.aten.add.Scalar(mul_346, 1);  mul_346 = None
    mul_347: "f32[4, 48, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_75, add_133);  sigmoid_75 = add_133 = None
    mul_348: "f32[4, 48, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_51, mul_347);  getitem_51 = mul_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_348, mean_12, primals_185, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_348 = mean_12 = primals_185 = None
    getitem_54: "f32[4, 1152, 1, 1]" = convolution_backward_18[0]
    getitem_55: "f32[48, 1152, 1, 1]" = convolution_backward_18[1]
    getitem_56: "f32[48]" = convolution_backward_18[2];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_4: "f32[4, 1152, 7, 7]" = torch.ops.aten.expand.default(getitem_54, [4, 1152, 7, 7]);  getitem_54 = None
    div_4: "f32[4, 1152, 7, 7]" = torch.ops.aten.div.Scalar(expand_4, 49);  expand_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_134: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_343, div_4);  mul_343 = div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_76: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(clone_37)
    full_109: "f32[4, 1152, 7, 7]" = torch.ops.aten.full.default([4, 1152, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_75: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(full_109, sigmoid_76);  full_109 = None
    mul_349: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(clone_37, sub_75);  clone_37 = sub_75 = None
    add_135: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Scalar(mul_349, 1);  mul_349 = None
    mul_350: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_76, add_135);  sigmoid_76 = add_135 = None
    mul_351: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_134, mul_350);  add_134 = mul_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_136: "f32[1152]" = torch.ops.aten.add.Tensor(primals_289, 1e-05);  primals_289 = None
    rsqrt_11: "f32[1152]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
    unsqueeze_524: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(primals_288, 0);  primals_288 = None
    unsqueeze_525: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 2);  unsqueeze_524 = None
    unsqueeze_526: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_525, 3);  unsqueeze_525 = None
    sum_28: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_351, [0, 2, 3])
    sub_76: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_526);  convolution_61 = unsqueeze_526 = None
    mul_352: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_351, sub_76);  sub_76 = None
    sum_29: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_352, [0, 2, 3]);  mul_352 = None
    mul_357: "f32[1152]" = torch.ops.aten.mul.Tensor(rsqrt_11, primals_75);  primals_75 = None
    unsqueeze_533: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_357, 0);  mul_357 = None
    unsqueeze_534: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 2);  unsqueeze_533 = None
    unsqueeze_535: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 3);  unsqueeze_534 = None
    mul_358: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_351, unsqueeze_535);  mul_351 = unsqueeze_535 = None
    mul_359: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_29, rsqrt_11);  sum_29 = rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_358, mul_159, primals_184, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_358 = mul_159 = primals_184 = None
    getitem_57: "f32[4, 1152, 7, 7]" = convolution_backward_19[0]
    getitem_58: "f32[1152, 1, 5, 5]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_77: "f32[4, 1152, 7, 7]" = torch.ops.aten.sigmoid.default(clone_36)
    full_110: "f32[4, 1152, 7, 7]" = torch.ops.aten.full.default([4, 1152, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_77: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(full_110, sigmoid_77);  full_110 = None
    mul_360: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(clone_36, sub_77);  clone_36 = sub_77 = None
    add_137: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Scalar(mul_360, 1);  mul_360 = None
    mul_361: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_77, add_137);  sigmoid_77 = add_137 = None
    mul_362: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_57, mul_361);  getitem_57 = mul_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_138: "f32[1152]" = torch.ops.aten.add.Tensor(primals_287, 1e-05);  primals_287 = None
    rsqrt_12: "f32[1152]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
    unsqueeze_536: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(primals_286, 0);  primals_286 = None
    unsqueeze_537: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 2);  unsqueeze_536 = None
    unsqueeze_538: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_537, 3);  unsqueeze_537 = None
    sum_30: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_362, [0, 2, 3])
    sub_78: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_538);  convolution_60 = unsqueeze_538 = None
    mul_363: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_362, sub_78);  sub_78 = None
    sum_31: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_363, [0, 2, 3]);  mul_363 = None
    mul_368: "f32[1152]" = torch.ops.aten.mul.Tensor(rsqrt_12, primals_73);  primals_73 = None
    unsqueeze_545: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_368, 0);  mul_368 = None
    unsqueeze_546: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 2);  unsqueeze_545 = None
    unsqueeze_547: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, 3);  unsqueeze_546 = None
    mul_369: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_362, unsqueeze_547);  mul_362 = unsqueeze_547 = None
    mul_370: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_31, rsqrt_12);  sum_31 = rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_369, add_77, primals_183, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_369 = add_77 = primals_183 = None
    getitem_60: "f32[4, 192, 7, 7]" = convolution_backward_20[0]
    getitem_61: "f32[1152, 192, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_139: "f32[4, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_131, getitem_60);  add_131 = getitem_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_140: "f32[192]" = torch.ops.aten.add.Tensor(primals_285, 1e-05);  primals_285 = None
    rsqrt_13: "f32[192]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    unsqueeze_548: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_284, 0);  primals_284 = None
    unsqueeze_549: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 2);  unsqueeze_548 = None
    unsqueeze_550: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_549, 3);  unsqueeze_549 = None
    sum_32: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_139, [0, 2, 3])
    sub_79: "f32[4, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_550);  convolution_59 = unsqueeze_550 = None
    mul_371: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_139, sub_79);  sub_79 = None
    sum_33: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_371, [0, 2, 3]);  mul_371 = None
    mul_376: "f32[192]" = torch.ops.aten.mul.Tensor(rsqrt_13, primals_71);  primals_71 = None
    unsqueeze_557: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_376, 0);  mul_376 = None
    unsqueeze_558: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 2);  unsqueeze_557 = None
    unsqueeze_559: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, 3);  unsqueeze_558 = None
    mul_377: "f32[4, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_139, unsqueeze_559);  add_139 = unsqueeze_559 = None
    mul_378: "f32[192]" = torch.ops.aten.mul.Tensor(sum_33, rsqrt_13);  sum_33 = rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_377, mul_152, primals_182, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_377 = mul_152 = primals_182 = None
    getitem_63: "f32[4, 672, 7, 7]" = convolution_backward_21[0]
    getitem_64: "f32[192, 672, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_379: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_63, mul_150);  mul_150 = None
    mul_380: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_63, sigmoid_47);  getitem_63 = sigmoid_47 = None
    sum_34: "f32[4, 672, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_379, [2, 3], True);  mul_379 = None
    alias_20: "f32[4, 672, 1, 1]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    sub_80: "f32[4, 672, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_20)
    mul_381: "f32[4, 672, 1, 1]" = torch.ops.aten.mul.Tensor(alias_20, sub_80);  alias_20 = sub_80 = None
    mul_382: "f32[4, 672, 1, 1]" = torch.ops.aten.mul.Tensor(sum_34, mul_381);  sum_34 = mul_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_382, mul_151, primals_180, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_382 = mul_151 = primals_180 = None
    getitem_66: "f32[4, 28, 1, 1]" = convolution_backward_22[0]
    getitem_67: "f32[672, 28, 1, 1]" = convolution_backward_22[1]
    getitem_68: "f32[672]" = convolution_backward_22[2];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_78: "f32[4, 28, 1, 1]" = torch.ops.aten.sigmoid.default(clone_35)
    full_111: "f32[4, 28, 1, 1]" = torch.ops.aten.full.default([4, 28, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_81: "f32[4, 28, 1, 1]" = torch.ops.aten.sub.Tensor(full_111, sigmoid_78);  full_111 = None
    mul_383: "f32[4, 28, 1, 1]" = torch.ops.aten.mul.Tensor(clone_35, sub_81);  clone_35 = sub_81 = None
    add_141: "f32[4, 28, 1, 1]" = torch.ops.aten.add.Scalar(mul_383, 1);  mul_383 = None
    mul_384: "f32[4, 28, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_78, add_141);  sigmoid_78 = add_141 = None
    mul_385: "f32[4, 28, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_66, mul_384);  getitem_66 = mul_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_385, mean_11, primals_178, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_385 = mean_11 = primals_178 = None
    getitem_69: "f32[4, 672, 1, 1]" = convolution_backward_23[0]
    getitem_70: "f32[28, 672, 1, 1]" = convolution_backward_23[1]
    getitem_71: "f32[28]" = convolution_backward_23[2];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_5: "f32[4, 672, 7, 7]" = torch.ops.aten.expand.default(getitem_69, [4, 672, 7, 7]);  getitem_69 = None
    div_5: "f32[4, 672, 7, 7]" = torch.ops.aten.div.Scalar(expand_5, 49);  expand_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_142: "f32[4, 672, 7, 7]" = torch.ops.aten.add.Tensor(mul_380, div_5);  mul_380 = div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_79: "f32[4, 672, 7, 7]" = torch.ops.aten.sigmoid.default(clone_34)
    full_112: "f32[4, 672, 7, 7]" = torch.ops.aten.full.default([4, 672, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_82: "f32[4, 672, 7, 7]" = torch.ops.aten.sub.Tensor(full_112, sigmoid_79);  full_112 = None
    mul_386: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(clone_34, sub_82);  clone_34 = sub_82 = None
    add_143: "f32[4, 672, 7, 7]" = torch.ops.aten.add.Scalar(mul_386, 1);  mul_386 = None
    mul_387: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_79, add_143);  sigmoid_79 = add_143 = None
    mul_388: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(add_142, mul_387);  add_142 = mul_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_144: "f32[672]" = torch.ops.aten.add.Tensor(primals_283, 1e-05);  primals_283 = None
    rsqrt_14: "f32[672]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    unsqueeze_560: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(primals_282, 0);  primals_282 = None
    unsqueeze_561: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 2);  unsqueeze_560 = None
    unsqueeze_562: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_561, 3);  unsqueeze_561 = None
    sum_35: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_388, [0, 2, 3])
    sub_83: "f32[4, 672, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_562);  convolution_56 = unsqueeze_562 = None
    mul_389: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(mul_388, sub_83);  sub_83 = None
    sum_36: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_389, [0, 2, 3]);  mul_389 = None
    mul_394: "f32[672]" = torch.ops.aten.mul.Tensor(rsqrt_14, primals_69);  primals_69 = None
    unsqueeze_569: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_394, 0);  mul_394 = None
    unsqueeze_570: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 2);  unsqueeze_569 = None
    unsqueeze_571: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, 3);  unsqueeze_570 = None
    mul_395: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(mul_388, unsqueeze_571);  mul_388 = unsqueeze_571 = None
    mul_396: "f32[672]" = torch.ops.aten.mul.Tensor(sum_36, rsqrt_14);  sum_36 = rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_395, mul_146, primals_177, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False]);  mul_395 = mul_146 = primals_177 = None
    getitem_72: "f32[4, 672, 14, 14]" = convolution_backward_24[0]
    getitem_73: "f32[672, 1, 5, 5]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_80: "f32[4, 672, 14, 14]" = torch.ops.aten.sigmoid.default(clone_33)
    full_113: "f32[4, 672, 14, 14]" = torch.ops.aten.full.default([4, 672, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_84: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(full_113, sigmoid_80);  full_113 = None
    mul_397: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(clone_33, sub_84);  clone_33 = sub_84 = None
    add_145: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Scalar(mul_397, 1);  mul_397 = None
    mul_398: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_80, add_145);  sigmoid_80 = add_145 = None
    mul_399: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_72, mul_398);  getitem_72 = mul_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_146: "f32[672]" = torch.ops.aten.add.Tensor(primals_281, 1e-05);  primals_281 = None
    rsqrt_15: "f32[672]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
    unsqueeze_572: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(primals_280, 0);  primals_280 = None
    unsqueeze_573: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 2);  unsqueeze_572 = None
    unsqueeze_574: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_573, 3);  unsqueeze_573 = None
    sum_37: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_399, [0, 2, 3])
    sub_85: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_574);  convolution_55 = unsqueeze_574 = None
    mul_400: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_399, sub_85);  sub_85 = None
    sum_38: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_400, [0, 2, 3]);  mul_400 = None
    mul_405: "f32[672]" = torch.ops.aten.mul.Tensor(rsqrt_15, primals_67);  primals_67 = None
    unsqueeze_581: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_405, 0);  mul_405 = None
    unsqueeze_582: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 2);  unsqueeze_581 = None
    unsqueeze_583: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, 3);  unsqueeze_582 = None
    mul_406: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_399, unsqueeze_583);  mul_399 = unsqueeze_583 = None
    mul_407: "f32[672]" = torch.ops.aten.mul.Tensor(sum_38, rsqrt_15);  sum_38 = rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_406, add_71, primals_176, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_406 = add_71 = primals_176 = None
    getitem_75: "f32[4, 112, 14, 14]" = convolution_backward_25[0]
    getitem_76: "f32[672, 112, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_147: "f32[112]" = torch.ops.aten.add.Tensor(primals_279, 1e-05);  primals_279 = None
    rsqrt_16: "f32[112]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
    unsqueeze_584: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(primals_278, 0);  primals_278 = None
    unsqueeze_585: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 2);  unsqueeze_584 = None
    unsqueeze_586: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_585, 3);  unsqueeze_585 = None
    sum_39: "f32[112]" = torch.ops.aten.sum.dim_IntList(getitem_75, [0, 2, 3])
    sub_86: "f32[4, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_586);  convolution_54 = unsqueeze_586 = None
    mul_408: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_75, sub_86);  sub_86 = None
    sum_40: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_408, [0, 2, 3]);  mul_408 = None
    mul_413: "f32[112]" = torch.ops.aten.mul.Tensor(rsqrt_16, primals_65);  primals_65 = None
    unsqueeze_593: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_413, 0);  mul_413 = None
    unsqueeze_594: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 2);  unsqueeze_593 = None
    unsqueeze_595: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, 3);  unsqueeze_594 = None
    mul_414: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_75, unsqueeze_595);  unsqueeze_595 = None
    mul_415: "f32[112]" = torch.ops.aten.mul.Tensor(sum_40, rsqrt_16);  sum_40 = rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_414, mul_139, primals_175, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_414 = mul_139 = primals_175 = None
    getitem_78: "f32[4, 672, 14, 14]" = convolution_backward_26[0]
    getitem_79: "f32[112, 672, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_416: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_78, mul_137);  mul_137 = None
    mul_417: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_78, sigmoid_43);  getitem_78 = sigmoid_43 = None
    sum_41: "f32[4, 672, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_416, [2, 3], True);  mul_416 = None
    alias_21: "f32[4, 672, 1, 1]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    sub_87: "f32[4, 672, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_21)
    mul_418: "f32[4, 672, 1, 1]" = torch.ops.aten.mul.Tensor(alias_21, sub_87);  alias_21 = sub_87 = None
    mul_419: "f32[4, 672, 1, 1]" = torch.ops.aten.mul.Tensor(sum_41, mul_418);  sum_41 = mul_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_419, mul_138, primals_173, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_419 = mul_138 = primals_173 = None
    getitem_81: "f32[4, 28, 1, 1]" = convolution_backward_27[0]
    getitem_82: "f32[672, 28, 1, 1]" = convolution_backward_27[1]
    getitem_83: "f32[672]" = convolution_backward_27[2];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_81: "f32[4, 28, 1, 1]" = torch.ops.aten.sigmoid.default(clone_32)
    full_114: "f32[4, 28, 1, 1]" = torch.ops.aten.full.default([4, 28, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_88: "f32[4, 28, 1, 1]" = torch.ops.aten.sub.Tensor(full_114, sigmoid_81);  full_114 = None
    mul_420: "f32[4, 28, 1, 1]" = torch.ops.aten.mul.Tensor(clone_32, sub_88);  clone_32 = sub_88 = None
    add_148: "f32[4, 28, 1, 1]" = torch.ops.aten.add.Scalar(mul_420, 1);  mul_420 = None
    mul_421: "f32[4, 28, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_81, add_148);  sigmoid_81 = add_148 = None
    mul_422: "f32[4, 28, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_81, mul_421);  getitem_81 = mul_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_422, mean_10, primals_171, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_422 = mean_10 = primals_171 = None
    getitem_84: "f32[4, 672, 1, 1]" = convolution_backward_28[0]
    getitem_85: "f32[28, 672, 1, 1]" = convolution_backward_28[1]
    getitem_86: "f32[28]" = convolution_backward_28[2];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_6: "f32[4, 672, 14, 14]" = torch.ops.aten.expand.default(getitem_84, [4, 672, 14, 14]);  getitem_84 = None
    div_6: "f32[4, 672, 14, 14]" = torch.ops.aten.div.Scalar(expand_6, 196);  expand_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_149: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_417, div_6);  mul_417 = div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_82: "f32[4, 672, 14, 14]" = torch.ops.aten.sigmoid.default(clone_31)
    full_115: "f32[4, 672, 14, 14]" = torch.ops.aten.full.default([4, 672, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_89: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(full_115, sigmoid_82);  full_115 = None
    mul_423: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(clone_31, sub_89);  clone_31 = sub_89 = None
    add_150: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Scalar(mul_423, 1);  mul_423 = None
    mul_424: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_82, add_150);  sigmoid_82 = add_150 = None
    mul_425: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_149, mul_424);  add_149 = mul_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_151: "f32[672]" = torch.ops.aten.add.Tensor(primals_277, 1e-05);  primals_277 = None
    rsqrt_17: "f32[672]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
    unsqueeze_596: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(primals_276, 0);  primals_276 = None
    unsqueeze_597: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 2);  unsqueeze_596 = None
    unsqueeze_598: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_597, 3);  unsqueeze_597 = None
    sum_42: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_425, [0, 2, 3])
    sub_90: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_598);  convolution_51 = unsqueeze_598 = None
    mul_426: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_425, sub_90);  sub_90 = None
    sum_43: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_426, [0, 2, 3]);  mul_426 = None
    mul_431: "f32[672]" = torch.ops.aten.mul.Tensor(rsqrt_17, primals_63);  primals_63 = None
    unsqueeze_605: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_431, 0);  mul_431 = None
    unsqueeze_606: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 2);  unsqueeze_605 = None
    unsqueeze_607: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, 3);  unsqueeze_606 = None
    mul_432: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_425, unsqueeze_607);  mul_425 = unsqueeze_607 = None
    mul_433: "f32[672]" = torch.ops.aten.mul.Tensor(sum_43, rsqrt_17);  sum_43 = rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_432, mul_133, primals_170, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False]);  mul_432 = mul_133 = primals_170 = None
    getitem_87: "f32[4, 672, 14, 14]" = convolution_backward_29[0]
    getitem_88: "f32[672, 1, 5, 5]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_83: "f32[4, 672, 14, 14]" = torch.ops.aten.sigmoid.default(clone_30)
    full_116: "f32[4, 672, 14, 14]" = torch.ops.aten.full.default([4, 672, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_91: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(full_116, sigmoid_83);  full_116 = None
    mul_434: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(clone_30, sub_91);  clone_30 = sub_91 = None
    add_152: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Scalar(mul_434, 1);  mul_434 = None
    mul_435: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_83, add_152);  sigmoid_83 = add_152 = None
    mul_436: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_87, mul_435);  getitem_87 = mul_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_153: "f32[672]" = torch.ops.aten.add.Tensor(primals_275, 1e-05);  primals_275 = None
    rsqrt_18: "f32[672]" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
    unsqueeze_608: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(primals_274, 0);  primals_274 = None
    unsqueeze_609: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 2);  unsqueeze_608 = None
    unsqueeze_610: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_609, 3);  unsqueeze_609 = None
    sum_44: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_436, [0, 2, 3])
    sub_92: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_610);  convolution_50 = unsqueeze_610 = None
    mul_437: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_436, sub_92);  sub_92 = None
    sum_45: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_437, [0, 2, 3]);  mul_437 = None
    mul_442: "f32[672]" = torch.ops.aten.mul.Tensor(rsqrt_18, primals_61);  primals_61 = None
    unsqueeze_617: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_442, 0);  mul_442 = None
    unsqueeze_618: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 2);  unsqueeze_617 = None
    unsqueeze_619: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, 3);  unsqueeze_618 = None
    mul_443: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_436, unsqueeze_619);  mul_436 = unsqueeze_619 = None
    mul_444: "f32[672]" = torch.ops.aten.mul.Tensor(sum_45, rsqrt_18);  sum_45 = rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_443, add_64, primals_169, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_443 = add_64 = primals_169 = None
    getitem_90: "f32[4, 112, 14, 14]" = convolution_backward_30[0]
    getitem_91: "f32[672, 112, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_154: "f32[4, 112, 14, 14]" = torch.ops.aten.add.Tensor(getitem_75, getitem_90);  getitem_75 = getitem_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_155: "f32[112]" = torch.ops.aten.add.Tensor(primals_273, 1e-05);  primals_273 = None
    rsqrt_19: "f32[112]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
    unsqueeze_620: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(primals_272, 0);  primals_272 = None
    unsqueeze_621: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 2);  unsqueeze_620 = None
    unsqueeze_622: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_621, 3);  unsqueeze_621 = None
    sum_46: "f32[112]" = torch.ops.aten.sum.dim_IntList(add_154, [0, 2, 3])
    sub_93: "f32[4, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_622);  convolution_49 = unsqueeze_622 = None
    mul_445: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(add_154, sub_93);  sub_93 = None
    sum_47: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_445, [0, 2, 3]);  mul_445 = None
    mul_450: "f32[112]" = torch.ops.aten.mul.Tensor(rsqrt_19, primals_59);  primals_59 = None
    unsqueeze_629: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_450, 0);  mul_450 = None
    unsqueeze_630: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 2);  unsqueeze_629 = None
    unsqueeze_631: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, 3);  unsqueeze_630 = None
    mul_451: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(add_154, unsqueeze_631);  unsqueeze_631 = None
    mul_452: "f32[112]" = torch.ops.aten.mul.Tensor(sum_47, rsqrt_19);  sum_47 = rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_451, mul_126, primals_168, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_451 = mul_126 = primals_168 = None
    getitem_93: "f32[4, 672, 14, 14]" = convolution_backward_31[0]
    getitem_94: "f32[112, 672, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_453: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_93, mul_124);  mul_124 = None
    mul_454: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_93, sigmoid_39);  getitem_93 = sigmoid_39 = None
    sum_48: "f32[4, 672, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_453, [2, 3], True);  mul_453 = None
    alias_22: "f32[4, 672, 1, 1]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    sub_94: "f32[4, 672, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_22)
    mul_455: "f32[4, 672, 1, 1]" = torch.ops.aten.mul.Tensor(alias_22, sub_94);  alias_22 = sub_94 = None
    mul_456: "f32[4, 672, 1, 1]" = torch.ops.aten.mul.Tensor(sum_48, mul_455);  sum_48 = mul_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_456, mul_125, primals_166, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_456 = mul_125 = primals_166 = None
    getitem_96: "f32[4, 28, 1, 1]" = convolution_backward_32[0]
    getitem_97: "f32[672, 28, 1, 1]" = convolution_backward_32[1]
    getitem_98: "f32[672]" = convolution_backward_32[2];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_84: "f32[4, 28, 1, 1]" = torch.ops.aten.sigmoid.default(clone_29)
    full_117: "f32[4, 28, 1, 1]" = torch.ops.aten.full.default([4, 28, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_95: "f32[4, 28, 1, 1]" = torch.ops.aten.sub.Tensor(full_117, sigmoid_84);  full_117 = None
    mul_457: "f32[4, 28, 1, 1]" = torch.ops.aten.mul.Tensor(clone_29, sub_95);  clone_29 = sub_95 = None
    add_156: "f32[4, 28, 1, 1]" = torch.ops.aten.add.Scalar(mul_457, 1);  mul_457 = None
    mul_458: "f32[4, 28, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_84, add_156);  sigmoid_84 = add_156 = None
    mul_459: "f32[4, 28, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_96, mul_458);  getitem_96 = mul_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_459, mean_9, primals_164, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_459 = mean_9 = primals_164 = None
    getitem_99: "f32[4, 672, 1, 1]" = convolution_backward_33[0]
    getitem_100: "f32[28, 672, 1, 1]" = convolution_backward_33[1]
    getitem_101: "f32[28]" = convolution_backward_33[2];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_7: "f32[4, 672, 14, 14]" = torch.ops.aten.expand.default(getitem_99, [4, 672, 14, 14]);  getitem_99 = None
    div_7: "f32[4, 672, 14, 14]" = torch.ops.aten.div.Scalar(expand_7, 196);  expand_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_157: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_454, div_7);  mul_454 = div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_85: "f32[4, 672, 14, 14]" = torch.ops.aten.sigmoid.default(clone_28)
    full_118: "f32[4, 672, 14, 14]" = torch.ops.aten.full.default([4, 672, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_96: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(full_118, sigmoid_85);  full_118 = None
    mul_460: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(clone_28, sub_96);  clone_28 = sub_96 = None
    add_158: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Scalar(mul_460, 1);  mul_460 = None
    mul_461: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_85, add_158);  sigmoid_85 = add_158 = None
    mul_462: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_157, mul_461);  add_157 = mul_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_159: "f32[672]" = torch.ops.aten.add.Tensor(primals_271, 1e-05);  primals_271 = None
    rsqrt_20: "f32[672]" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
    unsqueeze_632: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(primals_270, 0);  primals_270 = None
    unsqueeze_633: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 2);  unsqueeze_632 = None
    unsqueeze_634: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_633, 3);  unsqueeze_633 = None
    sum_49: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_462, [0, 2, 3])
    sub_97: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_634);  convolution_46 = unsqueeze_634 = None
    mul_463: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_462, sub_97);  sub_97 = None
    sum_50: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_463, [0, 2, 3]);  mul_463 = None
    mul_468: "f32[672]" = torch.ops.aten.mul.Tensor(rsqrt_20, primals_57);  primals_57 = None
    unsqueeze_641: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_468, 0);  mul_468 = None
    unsqueeze_642: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 2);  unsqueeze_641 = None
    unsqueeze_643: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, 3);  unsqueeze_642 = None
    mul_469: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_462, unsqueeze_643);  mul_462 = unsqueeze_643 = None
    mul_470: "f32[672]" = torch.ops.aten.mul.Tensor(sum_50, rsqrt_20);  sum_50 = rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_469, mul_120, primals_163, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False]);  mul_469 = mul_120 = primals_163 = None
    getitem_102: "f32[4, 672, 14, 14]" = convolution_backward_34[0]
    getitem_103: "f32[672, 1, 5, 5]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_86: "f32[4, 672, 14, 14]" = torch.ops.aten.sigmoid.default(clone_27)
    full_119: "f32[4, 672, 14, 14]" = torch.ops.aten.full.default([4, 672, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_98: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(full_119, sigmoid_86);  full_119 = None
    mul_471: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(clone_27, sub_98);  clone_27 = sub_98 = None
    add_160: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Scalar(mul_471, 1);  mul_471 = None
    mul_472: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_86, add_160);  sigmoid_86 = add_160 = None
    mul_473: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_102, mul_472);  getitem_102 = mul_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_161: "f32[672]" = torch.ops.aten.add.Tensor(primals_269, 1e-05);  primals_269 = None
    rsqrt_21: "f32[672]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    unsqueeze_644: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(primals_268, 0);  primals_268 = None
    unsqueeze_645: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 2);  unsqueeze_644 = None
    unsqueeze_646: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_645, 3);  unsqueeze_645 = None
    sum_51: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_473, [0, 2, 3])
    sub_99: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_646);  convolution_45 = unsqueeze_646 = None
    mul_474: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_473, sub_99);  sub_99 = None
    sum_52: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_474, [0, 2, 3]);  mul_474 = None
    mul_479: "f32[672]" = torch.ops.aten.mul.Tensor(rsqrt_21, primals_55);  primals_55 = None
    unsqueeze_653: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_479, 0);  mul_479 = None
    unsqueeze_654: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 2);  unsqueeze_653 = None
    unsqueeze_655: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, 3);  unsqueeze_654 = None
    mul_480: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_473, unsqueeze_655);  mul_473 = unsqueeze_655 = None
    mul_481: "f32[672]" = torch.ops.aten.mul.Tensor(sum_52, rsqrt_21);  sum_52 = rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_480, add_57, primals_162, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_480 = add_57 = primals_162 = None
    getitem_105: "f32[4, 112, 14, 14]" = convolution_backward_35[0]
    getitem_106: "f32[672, 112, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_162: "f32[4, 112, 14, 14]" = torch.ops.aten.add.Tensor(add_154, getitem_105);  add_154 = getitem_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_163: "f32[112]" = torch.ops.aten.add.Tensor(primals_267, 1e-05);  primals_267 = None
    rsqrt_22: "f32[112]" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
    unsqueeze_656: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(primals_266, 0);  primals_266 = None
    unsqueeze_657: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 2);  unsqueeze_656 = None
    unsqueeze_658: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_657, 3);  unsqueeze_657 = None
    sum_53: "f32[112]" = torch.ops.aten.sum.dim_IntList(add_162, [0, 2, 3])
    sub_100: "f32[4, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_658);  convolution_44 = unsqueeze_658 = None
    mul_482: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(add_162, sub_100);  sub_100 = None
    sum_54: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_482, [0, 2, 3]);  mul_482 = None
    mul_487: "f32[112]" = torch.ops.aten.mul.Tensor(rsqrt_22, primals_53);  primals_53 = None
    unsqueeze_665: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_487, 0);  mul_487 = None
    unsqueeze_666: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 2);  unsqueeze_665 = None
    unsqueeze_667: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, 3);  unsqueeze_666 = None
    mul_488: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(add_162, unsqueeze_667);  add_162 = unsqueeze_667 = None
    mul_489: "f32[112]" = torch.ops.aten.mul.Tensor(sum_54, rsqrt_22);  sum_54 = rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_488, mul_113, primals_161, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_488 = mul_113 = primals_161 = None
    getitem_108: "f32[4, 480, 14, 14]" = convolution_backward_36[0]
    getitem_109: "f32[112, 480, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_490: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_108, mul_111);  mul_111 = None
    mul_491: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_108, sigmoid_35);  getitem_108 = sigmoid_35 = None
    sum_55: "f32[4, 480, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_490, [2, 3], True);  mul_490 = None
    alias_23: "f32[4, 480, 1, 1]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    sub_101: "f32[4, 480, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_23)
    mul_492: "f32[4, 480, 1, 1]" = torch.ops.aten.mul.Tensor(alias_23, sub_101);  alias_23 = sub_101 = None
    mul_493: "f32[4, 480, 1, 1]" = torch.ops.aten.mul.Tensor(sum_55, mul_492);  sum_55 = mul_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_493, mul_112, primals_159, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_493 = mul_112 = primals_159 = None
    getitem_111: "f32[4, 20, 1, 1]" = convolution_backward_37[0]
    getitem_112: "f32[480, 20, 1, 1]" = convolution_backward_37[1]
    getitem_113: "f32[480]" = convolution_backward_37[2];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_87: "f32[4, 20, 1, 1]" = torch.ops.aten.sigmoid.default(clone_26)
    full_120: "f32[4, 20, 1, 1]" = torch.ops.aten.full.default([4, 20, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_102: "f32[4, 20, 1, 1]" = torch.ops.aten.sub.Tensor(full_120, sigmoid_87);  full_120 = None
    mul_494: "f32[4, 20, 1, 1]" = torch.ops.aten.mul.Tensor(clone_26, sub_102);  clone_26 = sub_102 = None
    add_164: "f32[4, 20, 1, 1]" = torch.ops.aten.add.Scalar(mul_494, 1);  mul_494 = None
    mul_495: "f32[4, 20, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_87, add_164);  sigmoid_87 = add_164 = None
    mul_496: "f32[4, 20, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_111, mul_495);  getitem_111 = mul_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_496, mean_8, primals_157, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_496 = mean_8 = primals_157 = None
    getitem_114: "f32[4, 480, 1, 1]" = convolution_backward_38[0]
    getitem_115: "f32[20, 480, 1, 1]" = convolution_backward_38[1]
    getitem_116: "f32[20]" = convolution_backward_38[2];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_8: "f32[4, 480, 14, 14]" = torch.ops.aten.expand.default(getitem_114, [4, 480, 14, 14]);  getitem_114 = None
    div_8: "f32[4, 480, 14, 14]" = torch.ops.aten.div.Scalar(expand_8, 196);  expand_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_165: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_491, div_8);  mul_491 = div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_88: "f32[4, 480, 14, 14]" = torch.ops.aten.sigmoid.default(clone_25)
    full_121: "f32[4, 480, 14, 14]" = torch.ops.aten.full.default([4, 480, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_103: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(full_121, sigmoid_88);  full_121 = None
    mul_497: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(clone_25, sub_103);  clone_25 = sub_103 = None
    add_166: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Scalar(mul_497, 1);  mul_497 = None
    mul_498: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_88, add_166);  sigmoid_88 = add_166 = None
    mul_499: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_165, mul_498);  add_165 = mul_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_167: "f32[480]" = torch.ops.aten.add.Tensor(primals_265, 1e-05);  primals_265 = None
    rsqrt_23: "f32[480]" = torch.ops.aten.rsqrt.default(add_167);  add_167 = None
    unsqueeze_668: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(primals_264, 0);  primals_264 = None
    unsqueeze_669: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 2);  unsqueeze_668 = None
    unsqueeze_670: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_669, 3);  unsqueeze_669 = None
    sum_56: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_499, [0, 2, 3])
    sub_104: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_670);  convolution_41 = unsqueeze_670 = None
    mul_500: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_499, sub_104);  sub_104 = None
    sum_57: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_500, [0, 2, 3]);  mul_500 = None
    mul_505: "f32[480]" = torch.ops.aten.mul.Tensor(rsqrt_23, primals_51);  primals_51 = None
    unsqueeze_677: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_505, 0);  mul_505 = None
    unsqueeze_678: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 2);  unsqueeze_677 = None
    unsqueeze_679: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, 3);  unsqueeze_678 = None
    mul_506: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_499, unsqueeze_679);  mul_499 = unsqueeze_679 = None
    mul_507: "f32[480]" = torch.ops.aten.mul.Tensor(sum_57, rsqrt_23);  sum_57 = rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_506, mul_107, primals_156, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_506 = mul_107 = primals_156 = None
    getitem_117: "f32[4, 480, 14, 14]" = convolution_backward_39[0]
    getitem_118: "f32[480, 1, 5, 5]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_89: "f32[4, 480, 14, 14]" = torch.ops.aten.sigmoid.default(clone_24)
    full_122: "f32[4, 480, 14, 14]" = torch.ops.aten.full.default([4, 480, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_105: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(full_122, sigmoid_89);  full_122 = None
    mul_508: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(clone_24, sub_105);  clone_24 = sub_105 = None
    add_168: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Scalar(mul_508, 1);  mul_508 = None
    mul_509: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_89, add_168);  sigmoid_89 = add_168 = None
    mul_510: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_117, mul_509);  getitem_117 = mul_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_169: "f32[480]" = torch.ops.aten.add.Tensor(primals_263, 1e-05);  primals_263 = None
    rsqrt_24: "f32[480]" = torch.ops.aten.rsqrt.default(add_169);  add_169 = None
    unsqueeze_680: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(primals_262, 0);  primals_262 = None
    unsqueeze_681: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 2);  unsqueeze_680 = None
    unsqueeze_682: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_681, 3);  unsqueeze_681 = None
    sum_58: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_510, [0, 2, 3])
    sub_106: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_682);  convolution_40 = unsqueeze_682 = None
    mul_511: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_510, sub_106);  sub_106 = None
    sum_59: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_511, [0, 2, 3]);  mul_511 = None
    mul_516: "f32[480]" = torch.ops.aten.mul.Tensor(rsqrt_24, primals_49);  primals_49 = None
    unsqueeze_689: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_516, 0);  mul_516 = None
    unsqueeze_690: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 2);  unsqueeze_689 = None
    unsqueeze_691: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, 3);  unsqueeze_690 = None
    mul_517: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_510, unsqueeze_691);  mul_510 = unsqueeze_691 = None
    mul_518: "f32[480]" = torch.ops.aten.mul.Tensor(sum_59, rsqrt_24);  sum_59 = rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_517, add_51, primals_155, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_517 = add_51 = primals_155 = None
    getitem_120: "f32[4, 80, 14, 14]" = convolution_backward_40[0]
    getitem_121: "f32[480, 80, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_170: "f32[80]" = torch.ops.aten.add.Tensor(primals_261, 1e-05);  primals_261 = None
    rsqrt_25: "f32[80]" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
    unsqueeze_692: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(primals_260, 0);  primals_260 = None
    unsqueeze_693: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 2);  unsqueeze_692 = None
    unsqueeze_694: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_693, 3);  unsqueeze_693 = None
    sum_60: "f32[80]" = torch.ops.aten.sum.dim_IntList(getitem_120, [0, 2, 3])
    sub_107: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_694);  convolution_39 = unsqueeze_694 = None
    mul_519: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_120, sub_107);  sub_107 = None
    sum_61: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_519, [0, 2, 3]);  mul_519 = None
    mul_524: "f32[80]" = torch.ops.aten.mul.Tensor(rsqrt_25, primals_47);  primals_47 = None
    unsqueeze_701: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_524, 0);  mul_524 = None
    unsqueeze_702: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 2);  unsqueeze_701 = None
    unsqueeze_703: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, 3);  unsqueeze_702 = None
    mul_525: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_120, unsqueeze_703);  unsqueeze_703 = None
    mul_526: "f32[80]" = torch.ops.aten.mul.Tensor(sum_61, rsqrt_25);  sum_61 = rsqrt_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_525, mul_100, primals_154, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_525 = mul_100 = primals_154 = None
    getitem_123: "f32[4, 480, 14, 14]" = convolution_backward_41[0]
    getitem_124: "f32[80, 480, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_527: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_123, mul_98);  mul_98 = None
    mul_528: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_123, sigmoid_31);  getitem_123 = sigmoid_31 = None
    sum_62: "f32[4, 480, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_527, [2, 3], True);  mul_527 = None
    alias_24: "f32[4, 480, 1, 1]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    sub_108: "f32[4, 480, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_24)
    mul_529: "f32[4, 480, 1, 1]" = torch.ops.aten.mul.Tensor(alias_24, sub_108);  alias_24 = sub_108 = None
    mul_530: "f32[4, 480, 1, 1]" = torch.ops.aten.mul.Tensor(sum_62, mul_529);  sum_62 = mul_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_530, mul_99, primals_152, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_530 = mul_99 = primals_152 = None
    getitem_126: "f32[4, 20, 1, 1]" = convolution_backward_42[0]
    getitem_127: "f32[480, 20, 1, 1]" = convolution_backward_42[1]
    getitem_128: "f32[480]" = convolution_backward_42[2];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_90: "f32[4, 20, 1, 1]" = torch.ops.aten.sigmoid.default(clone_23)
    full_123: "f32[4, 20, 1, 1]" = torch.ops.aten.full.default([4, 20, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_109: "f32[4, 20, 1, 1]" = torch.ops.aten.sub.Tensor(full_123, sigmoid_90);  full_123 = None
    mul_531: "f32[4, 20, 1, 1]" = torch.ops.aten.mul.Tensor(clone_23, sub_109);  clone_23 = sub_109 = None
    add_171: "f32[4, 20, 1, 1]" = torch.ops.aten.add.Scalar(mul_531, 1);  mul_531 = None
    mul_532: "f32[4, 20, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_90, add_171);  sigmoid_90 = add_171 = None
    mul_533: "f32[4, 20, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_126, mul_532);  getitem_126 = mul_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_533, mean_7, primals_150, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_533 = mean_7 = primals_150 = None
    getitem_129: "f32[4, 480, 1, 1]" = convolution_backward_43[0]
    getitem_130: "f32[20, 480, 1, 1]" = convolution_backward_43[1]
    getitem_131: "f32[20]" = convolution_backward_43[2];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_9: "f32[4, 480, 14, 14]" = torch.ops.aten.expand.default(getitem_129, [4, 480, 14, 14]);  getitem_129 = None
    div_9: "f32[4, 480, 14, 14]" = torch.ops.aten.div.Scalar(expand_9, 196);  expand_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_172: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_528, div_9);  mul_528 = div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_91: "f32[4, 480, 14, 14]" = torch.ops.aten.sigmoid.default(clone_22)
    full_124: "f32[4, 480, 14, 14]" = torch.ops.aten.full.default([4, 480, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_110: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(full_124, sigmoid_91);  full_124 = None
    mul_534: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(clone_22, sub_110);  clone_22 = sub_110 = None
    add_173: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Scalar(mul_534, 1);  mul_534 = None
    mul_535: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_91, add_173);  sigmoid_91 = add_173 = None
    mul_536: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_172, mul_535);  add_172 = mul_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_174: "f32[480]" = torch.ops.aten.add.Tensor(primals_259, 1e-05);  primals_259 = None
    rsqrt_26: "f32[480]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
    unsqueeze_704: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(primals_258, 0);  primals_258 = None
    unsqueeze_705: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 2);  unsqueeze_704 = None
    unsqueeze_706: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_705, 3);  unsqueeze_705 = None
    sum_63: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_536, [0, 2, 3])
    sub_111: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_706);  convolution_36 = unsqueeze_706 = None
    mul_537: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_536, sub_111);  sub_111 = None
    sum_64: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_537, [0, 2, 3]);  mul_537 = None
    mul_542: "f32[480]" = torch.ops.aten.mul.Tensor(rsqrt_26, primals_45);  primals_45 = None
    unsqueeze_713: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_542, 0);  mul_542 = None
    unsqueeze_714: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 2);  unsqueeze_713 = None
    unsqueeze_715: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, 3);  unsqueeze_714 = None
    mul_543: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_536, unsqueeze_715);  mul_536 = unsqueeze_715 = None
    mul_544: "f32[480]" = torch.ops.aten.mul.Tensor(sum_64, rsqrt_26);  sum_64 = rsqrt_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_543, mul_94, primals_149, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_543 = mul_94 = primals_149 = None
    getitem_132: "f32[4, 480, 14, 14]" = convolution_backward_44[0]
    getitem_133: "f32[480, 1, 3, 3]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_92: "f32[4, 480, 14, 14]" = torch.ops.aten.sigmoid.default(clone_21)
    full_125: "f32[4, 480, 14, 14]" = torch.ops.aten.full.default([4, 480, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_112: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(full_125, sigmoid_92);  full_125 = None
    mul_545: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(clone_21, sub_112);  clone_21 = sub_112 = None
    add_175: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Scalar(mul_545, 1);  mul_545 = None
    mul_546: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_92, add_175);  sigmoid_92 = add_175 = None
    mul_547: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_132, mul_546);  getitem_132 = mul_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_176: "f32[480]" = torch.ops.aten.add.Tensor(primals_257, 1e-05);  primals_257 = None
    rsqrt_27: "f32[480]" = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
    unsqueeze_716: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(primals_256, 0);  primals_256 = None
    unsqueeze_717: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 2);  unsqueeze_716 = None
    unsqueeze_718: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_717, 3);  unsqueeze_717 = None
    sum_65: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_547, [0, 2, 3])
    sub_113: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_718);  convolution_35 = unsqueeze_718 = None
    mul_548: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_547, sub_113);  sub_113 = None
    sum_66: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_548, [0, 2, 3]);  mul_548 = None
    mul_553: "f32[480]" = torch.ops.aten.mul.Tensor(rsqrt_27, primals_43);  primals_43 = None
    unsqueeze_725: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_553, 0);  mul_553 = None
    unsqueeze_726: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 2);  unsqueeze_725 = None
    unsqueeze_727: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, 3);  unsqueeze_726 = None
    mul_554: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_547, unsqueeze_727);  mul_547 = unsqueeze_727 = None
    mul_555: "f32[480]" = torch.ops.aten.mul.Tensor(sum_66, rsqrt_27);  sum_66 = rsqrt_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_554, add_44, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_554 = add_44 = primals_148 = None
    getitem_135: "f32[4, 80, 14, 14]" = convolution_backward_45[0]
    getitem_136: "f32[480, 80, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_177: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(getitem_120, getitem_135);  getitem_120 = getitem_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_178: "f32[80]" = torch.ops.aten.add.Tensor(primals_255, 1e-05);  primals_255 = None
    rsqrt_28: "f32[80]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
    unsqueeze_728: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(primals_254, 0);  primals_254 = None
    unsqueeze_729: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 2);  unsqueeze_728 = None
    unsqueeze_730: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_729, 3);  unsqueeze_729 = None
    sum_67: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_177, [0, 2, 3])
    sub_114: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_730);  convolution_34 = unsqueeze_730 = None
    mul_556: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_177, sub_114);  sub_114 = None
    sum_68: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_556, [0, 2, 3]);  mul_556 = None
    mul_561: "f32[80]" = torch.ops.aten.mul.Tensor(rsqrt_28, primals_41);  primals_41 = None
    unsqueeze_737: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_561, 0);  mul_561 = None
    unsqueeze_738: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 2);  unsqueeze_737 = None
    unsqueeze_739: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, 3);  unsqueeze_738 = None
    mul_562: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_177, unsqueeze_739);  unsqueeze_739 = None
    mul_563: "f32[80]" = torch.ops.aten.mul.Tensor(sum_68, rsqrt_28);  sum_68 = rsqrt_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_562, mul_87, primals_147, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_562 = mul_87 = primals_147 = None
    getitem_138: "f32[4, 480, 14, 14]" = convolution_backward_46[0]
    getitem_139: "f32[80, 480, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_564: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_138, mul_85);  mul_85 = None
    mul_565: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_138, sigmoid_27);  getitem_138 = sigmoid_27 = None
    sum_69: "f32[4, 480, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_564, [2, 3], True);  mul_564 = None
    alias_25: "f32[4, 480, 1, 1]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    sub_115: "f32[4, 480, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_25)
    mul_566: "f32[4, 480, 1, 1]" = torch.ops.aten.mul.Tensor(alias_25, sub_115);  alias_25 = sub_115 = None
    mul_567: "f32[4, 480, 1, 1]" = torch.ops.aten.mul.Tensor(sum_69, mul_566);  sum_69 = mul_566 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_567, mul_86, primals_145, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_567 = mul_86 = primals_145 = None
    getitem_141: "f32[4, 20, 1, 1]" = convolution_backward_47[0]
    getitem_142: "f32[480, 20, 1, 1]" = convolution_backward_47[1]
    getitem_143: "f32[480]" = convolution_backward_47[2];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_93: "f32[4, 20, 1, 1]" = torch.ops.aten.sigmoid.default(clone_20)
    full_126: "f32[4, 20, 1, 1]" = torch.ops.aten.full.default([4, 20, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_116: "f32[4, 20, 1, 1]" = torch.ops.aten.sub.Tensor(full_126, sigmoid_93);  full_126 = None
    mul_568: "f32[4, 20, 1, 1]" = torch.ops.aten.mul.Tensor(clone_20, sub_116);  clone_20 = sub_116 = None
    add_179: "f32[4, 20, 1, 1]" = torch.ops.aten.add.Scalar(mul_568, 1);  mul_568 = None
    mul_569: "f32[4, 20, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_93, add_179);  sigmoid_93 = add_179 = None
    mul_570: "f32[4, 20, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_141, mul_569);  getitem_141 = mul_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_570, mean_6, primals_143, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_570 = mean_6 = primals_143 = None
    getitem_144: "f32[4, 480, 1, 1]" = convolution_backward_48[0]
    getitem_145: "f32[20, 480, 1, 1]" = convolution_backward_48[1]
    getitem_146: "f32[20]" = convolution_backward_48[2];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_10: "f32[4, 480, 14, 14]" = torch.ops.aten.expand.default(getitem_144, [4, 480, 14, 14]);  getitem_144 = None
    div_10: "f32[4, 480, 14, 14]" = torch.ops.aten.div.Scalar(expand_10, 196);  expand_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_180: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_565, div_10);  mul_565 = div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_94: "f32[4, 480, 14, 14]" = torch.ops.aten.sigmoid.default(clone_19)
    full_127: "f32[4, 480, 14, 14]" = torch.ops.aten.full.default([4, 480, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_117: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(full_127, sigmoid_94);  full_127 = None
    mul_571: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(clone_19, sub_117);  clone_19 = sub_117 = None
    add_181: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Scalar(mul_571, 1);  mul_571 = None
    mul_572: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_94, add_181);  sigmoid_94 = add_181 = None
    mul_573: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_180, mul_572);  add_180 = mul_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_182: "f32[480]" = torch.ops.aten.add.Tensor(primals_253, 1e-05);  primals_253 = None
    rsqrt_29: "f32[480]" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
    unsqueeze_740: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(primals_252, 0);  primals_252 = None
    unsqueeze_741: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 2);  unsqueeze_740 = None
    unsqueeze_742: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_741, 3);  unsqueeze_741 = None
    sum_70: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_573, [0, 2, 3])
    sub_118: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_742);  convolution_31 = unsqueeze_742 = None
    mul_574: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_573, sub_118);  sub_118 = None
    sum_71: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_574, [0, 2, 3]);  mul_574 = None
    mul_579: "f32[480]" = torch.ops.aten.mul.Tensor(rsqrt_29, primals_39);  primals_39 = None
    unsqueeze_749: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_579, 0);  mul_579 = None
    unsqueeze_750: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 2);  unsqueeze_749 = None
    unsqueeze_751: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, 3);  unsqueeze_750 = None
    mul_580: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_573, unsqueeze_751);  mul_573 = unsqueeze_751 = None
    mul_581: "f32[480]" = torch.ops.aten.mul.Tensor(sum_71, rsqrt_29);  sum_71 = rsqrt_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_580, mul_81, primals_142, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_580 = mul_81 = primals_142 = None
    getitem_147: "f32[4, 480, 14, 14]" = convolution_backward_49[0]
    getitem_148: "f32[480, 1, 3, 3]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_95: "f32[4, 480, 14, 14]" = torch.ops.aten.sigmoid.default(clone_18)
    full_128: "f32[4, 480, 14, 14]" = torch.ops.aten.full.default([4, 480, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_119: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(full_128, sigmoid_95);  full_128 = None
    mul_582: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(clone_18, sub_119);  clone_18 = sub_119 = None
    add_183: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Scalar(mul_582, 1);  mul_582 = None
    mul_583: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_95, add_183);  sigmoid_95 = add_183 = None
    mul_584: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_147, mul_583);  getitem_147 = mul_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_184: "f32[480]" = torch.ops.aten.add.Tensor(primals_251, 1e-05);  primals_251 = None
    rsqrt_30: "f32[480]" = torch.ops.aten.rsqrt.default(add_184);  add_184 = None
    unsqueeze_752: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(primals_250, 0);  primals_250 = None
    unsqueeze_753: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 2);  unsqueeze_752 = None
    unsqueeze_754: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_753, 3);  unsqueeze_753 = None
    sum_72: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_584, [0, 2, 3])
    sub_120: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_754);  convolution_30 = unsqueeze_754 = None
    mul_585: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_584, sub_120);  sub_120 = None
    sum_73: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_585, [0, 2, 3]);  mul_585 = None
    mul_590: "f32[480]" = torch.ops.aten.mul.Tensor(rsqrt_30, primals_37);  primals_37 = None
    unsqueeze_761: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_590, 0);  mul_590 = None
    unsqueeze_762: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 2);  unsqueeze_761 = None
    unsqueeze_763: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, 3);  unsqueeze_762 = None
    mul_591: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_584, unsqueeze_763);  mul_584 = unsqueeze_763 = None
    mul_592: "f32[480]" = torch.ops.aten.mul.Tensor(sum_73, rsqrt_30);  sum_73 = rsqrt_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_591, add_37, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_591 = add_37 = primals_141 = None
    getitem_150: "f32[4, 80, 14, 14]" = convolution_backward_50[0]
    getitem_151: "f32[480, 80, 1, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_185: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_177, getitem_150);  add_177 = getitem_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_186: "f32[80]" = torch.ops.aten.add.Tensor(primals_249, 1e-05);  primals_249 = None
    rsqrt_31: "f32[80]" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
    unsqueeze_764: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(primals_248, 0);  primals_248 = None
    unsqueeze_765: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 2);  unsqueeze_764 = None
    unsqueeze_766: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_765, 3);  unsqueeze_765 = None
    sum_74: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_185, [0, 2, 3])
    sub_121: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_766);  convolution_29 = unsqueeze_766 = None
    mul_593: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_185, sub_121);  sub_121 = None
    sum_75: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_593, [0, 2, 3]);  mul_593 = None
    mul_598: "f32[80]" = torch.ops.aten.mul.Tensor(rsqrt_31, primals_35);  primals_35 = None
    unsqueeze_773: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_598, 0);  mul_598 = None
    unsqueeze_774: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 2);  unsqueeze_773 = None
    unsqueeze_775: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, 3);  unsqueeze_774 = None
    mul_599: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_185, unsqueeze_775);  add_185 = unsqueeze_775 = None
    mul_600: "f32[80]" = torch.ops.aten.mul.Tensor(sum_75, rsqrt_31);  sum_75 = rsqrt_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_599, mul_74, primals_140, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_599 = mul_74 = primals_140 = None
    getitem_153: "f32[4, 240, 14, 14]" = convolution_backward_51[0]
    getitem_154: "f32[80, 240, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_601: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_153, mul_72);  mul_72 = None
    mul_602: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_153, sigmoid_23);  getitem_153 = sigmoid_23 = None
    sum_76: "f32[4, 240, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_601, [2, 3], True);  mul_601 = None
    alias_26: "f32[4, 240, 1, 1]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    sub_122: "f32[4, 240, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_26)
    mul_603: "f32[4, 240, 1, 1]" = torch.ops.aten.mul.Tensor(alias_26, sub_122);  alias_26 = sub_122 = None
    mul_604: "f32[4, 240, 1, 1]" = torch.ops.aten.mul.Tensor(sum_76, mul_603);  sum_76 = mul_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_604, mul_73, primals_138, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_604 = mul_73 = primals_138 = None
    getitem_156: "f32[4, 10, 1, 1]" = convolution_backward_52[0]
    getitem_157: "f32[240, 10, 1, 1]" = convolution_backward_52[1]
    getitem_158: "f32[240]" = convolution_backward_52[2];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_96: "f32[4, 10, 1, 1]" = torch.ops.aten.sigmoid.default(clone_17)
    full_129: "f32[4, 10, 1, 1]" = torch.ops.aten.full.default([4, 10, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_123: "f32[4, 10, 1, 1]" = torch.ops.aten.sub.Tensor(full_129, sigmoid_96);  full_129 = None
    mul_605: "f32[4, 10, 1, 1]" = torch.ops.aten.mul.Tensor(clone_17, sub_123);  clone_17 = sub_123 = None
    add_187: "f32[4, 10, 1, 1]" = torch.ops.aten.add.Scalar(mul_605, 1);  mul_605 = None
    mul_606: "f32[4, 10, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_96, add_187);  sigmoid_96 = add_187 = None
    mul_607: "f32[4, 10, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_156, mul_606);  getitem_156 = mul_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_607, mean_5, primals_136, [10], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_607 = mean_5 = primals_136 = None
    getitem_159: "f32[4, 240, 1, 1]" = convolution_backward_53[0]
    getitem_160: "f32[10, 240, 1, 1]" = convolution_backward_53[1]
    getitem_161: "f32[10]" = convolution_backward_53[2];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_11: "f32[4, 240, 14, 14]" = torch.ops.aten.expand.default(getitem_159, [4, 240, 14, 14]);  getitem_159 = None
    div_11: "f32[4, 240, 14, 14]" = torch.ops.aten.div.Scalar(expand_11, 196);  expand_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_188: "f32[4, 240, 14, 14]" = torch.ops.aten.add.Tensor(mul_602, div_11);  mul_602 = div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_97: "f32[4, 240, 14, 14]" = torch.ops.aten.sigmoid.default(clone_16)
    full_130: "f32[4, 240, 14, 14]" = torch.ops.aten.full.default([4, 240, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_124: "f32[4, 240, 14, 14]" = torch.ops.aten.sub.Tensor(full_130, sigmoid_97);  full_130 = None
    mul_608: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(clone_16, sub_124);  clone_16 = sub_124 = None
    add_189: "f32[4, 240, 14, 14]" = torch.ops.aten.add.Scalar(mul_608, 1);  mul_608 = None
    mul_609: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_97, add_189);  sigmoid_97 = add_189 = None
    mul_610: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(add_188, mul_609);  add_188 = mul_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_190: "f32[240]" = torch.ops.aten.add.Tensor(primals_247, 1e-05);  primals_247 = None
    rsqrt_32: "f32[240]" = torch.ops.aten.rsqrt.default(add_190);  add_190 = None
    unsqueeze_776: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(primals_246, 0);  primals_246 = None
    unsqueeze_777: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 2);  unsqueeze_776 = None
    unsqueeze_778: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_777, 3);  unsqueeze_777 = None
    sum_77: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_610, [0, 2, 3])
    sub_125: "f32[4, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_778);  convolution_26 = unsqueeze_778 = None
    mul_611: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_610, sub_125);  sub_125 = None
    sum_78: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_611, [0, 2, 3]);  mul_611 = None
    mul_616: "f32[240]" = torch.ops.aten.mul.Tensor(rsqrt_32, primals_33);  primals_33 = None
    unsqueeze_785: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_616, 0);  mul_616 = None
    unsqueeze_786: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 2);  unsqueeze_785 = None
    unsqueeze_787: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, 3);  unsqueeze_786 = None
    mul_617: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_610, unsqueeze_787);  mul_610 = unsqueeze_787 = None
    mul_618: "f32[240]" = torch.ops.aten.mul.Tensor(sum_78, rsqrt_32);  sum_78 = rsqrt_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_617, mul_68, primals_135, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 240, [True, True, False]);  mul_617 = mul_68 = primals_135 = None
    getitem_162: "f32[4, 240, 28, 28]" = convolution_backward_54[0]
    getitem_163: "f32[240, 1, 3, 3]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_98: "f32[4, 240, 28, 28]" = torch.ops.aten.sigmoid.default(clone_15)
    full_131: "f32[4, 240, 28, 28]" = torch.ops.aten.full.default([4, 240, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_126: "f32[4, 240, 28, 28]" = torch.ops.aten.sub.Tensor(full_131, sigmoid_98);  full_131 = None
    mul_619: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(clone_15, sub_126);  clone_15 = sub_126 = None
    add_191: "f32[4, 240, 28, 28]" = torch.ops.aten.add.Scalar(mul_619, 1);  mul_619 = None
    mul_620: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_98, add_191);  sigmoid_98 = add_191 = None
    mul_621: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_162, mul_620);  getitem_162 = mul_620 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_192: "f32[240]" = torch.ops.aten.add.Tensor(primals_245, 1e-05);  primals_245 = None
    rsqrt_33: "f32[240]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
    unsqueeze_788: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(primals_244, 0);  primals_244 = None
    unsqueeze_789: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 2);  unsqueeze_788 = None
    unsqueeze_790: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_789, 3);  unsqueeze_789 = None
    sum_79: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_621, [0, 2, 3])
    sub_127: "f32[4, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_790);  convolution_25 = unsqueeze_790 = None
    mul_622: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_621, sub_127);  sub_127 = None
    sum_80: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_622, [0, 2, 3]);  mul_622 = None
    mul_627: "f32[240]" = torch.ops.aten.mul.Tensor(rsqrt_33, primals_31);  primals_31 = None
    unsqueeze_797: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_627, 0);  mul_627 = None
    unsqueeze_798: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 2);  unsqueeze_797 = None
    unsqueeze_799: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, 3);  unsqueeze_798 = None
    mul_628: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_621, unsqueeze_799);  mul_621 = unsqueeze_799 = None
    mul_629: "f32[240]" = torch.ops.aten.mul.Tensor(sum_80, rsqrt_33);  sum_80 = rsqrt_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_628, add_31, primals_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_628 = add_31 = primals_134 = None
    getitem_165: "f32[4, 40, 28, 28]" = convolution_backward_55[0]
    getitem_166: "f32[240, 40, 1, 1]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_193: "f32[40]" = torch.ops.aten.add.Tensor(primals_243, 1e-05);  primals_243 = None
    rsqrt_34: "f32[40]" = torch.ops.aten.rsqrt.default(add_193);  add_193 = None
    unsqueeze_800: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(primals_242, 0);  primals_242 = None
    unsqueeze_801: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 2);  unsqueeze_800 = None
    unsqueeze_802: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_801, 3);  unsqueeze_801 = None
    sum_81: "f32[40]" = torch.ops.aten.sum.dim_IntList(getitem_165, [0, 2, 3])
    sub_128: "f32[4, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_802);  convolution_24 = unsqueeze_802 = None
    mul_630: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_165, sub_128);  sub_128 = None
    sum_82: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_630, [0, 2, 3]);  mul_630 = None
    mul_635: "f32[40]" = torch.ops.aten.mul.Tensor(rsqrt_34, primals_29);  primals_29 = None
    unsqueeze_809: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_635, 0);  mul_635 = None
    unsqueeze_810: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 2);  unsqueeze_809 = None
    unsqueeze_811: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, 3);  unsqueeze_810 = None
    mul_636: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_165, unsqueeze_811);  unsqueeze_811 = None
    mul_637: "f32[40]" = torch.ops.aten.mul.Tensor(sum_82, rsqrt_34);  sum_82 = rsqrt_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_636, mul_61, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_636 = mul_61 = primals_133 = None
    getitem_168: "f32[4, 240, 28, 28]" = convolution_backward_56[0]
    getitem_169: "f32[40, 240, 1, 1]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_638: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_168, mul_59);  mul_59 = None
    mul_639: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_168, sigmoid_19);  getitem_168 = sigmoid_19 = None
    sum_83: "f32[4, 240, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_638, [2, 3], True);  mul_638 = None
    alias_27: "f32[4, 240, 1, 1]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    sub_129: "f32[4, 240, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_27)
    mul_640: "f32[4, 240, 1, 1]" = torch.ops.aten.mul.Tensor(alias_27, sub_129);  alias_27 = sub_129 = None
    mul_641: "f32[4, 240, 1, 1]" = torch.ops.aten.mul.Tensor(sum_83, mul_640);  sum_83 = mul_640 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_641, mul_60, primals_131, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_641 = mul_60 = primals_131 = None
    getitem_171: "f32[4, 10, 1, 1]" = convolution_backward_57[0]
    getitem_172: "f32[240, 10, 1, 1]" = convolution_backward_57[1]
    getitem_173: "f32[240]" = convolution_backward_57[2];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_99: "f32[4, 10, 1, 1]" = torch.ops.aten.sigmoid.default(clone_14)
    full_132: "f32[4, 10, 1, 1]" = torch.ops.aten.full.default([4, 10, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_130: "f32[4, 10, 1, 1]" = torch.ops.aten.sub.Tensor(full_132, sigmoid_99);  full_132 = None
    mul_642: "f32[4, 10, 1, 1]" = torch.ops.aten.mul.Tensor(clone_14, sub_130);  clone_14 = sub_130 = None
    add_194: "f32[4, 10, 1, 1]" = torch.ops.aten.add.Scalar(mul_642, 1);  mul_642 = None
    mul_643: "f32[4, 10, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_99, add_194);  sigmoid_99 = add_194 = None
    mul_644: "f32[4, 10, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_171, mul_643);  getitem_171 = mul_643 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_644, mean_4, primals_129, [10], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_644 = mean_4 = primals_129 = None
    getitem_174: "f32[4, 240, 1, 1]" = convolution_backward_58[0]
    getitem_175: "f32[10, 240, 1, 1]" = convolution_backward_58[1]
    getitem_176: "f32[10]" = convolution_backward_58[2];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_12: "f32[4, 240, 28, 28]" = torch.ops.aten.expand.default(getitem_174, [4, 240, 28, 28]);  getitem_174 = None
    div_12: "f32[4, 240, 28, 28]" = torch.ops.aten.div.Scalar(expand_12, 784);  expand_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_195: "f32[4, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_639, div_12);  mul_639 = div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_100: "f32[4, 240, 28, 28]" = torch.ops.aten.sigmoid.default(clone_13)
    full_133: "f32[4, 240, 28, 28]" = torch.ops.aten.full.default([4, 240, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_131: "f32[4, 240, 28, 28]" = torch.ops.aten.sub.Tensor(full_133, sigmoid_100);  full_133 = None
    mul_645: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(clone_13, sub_131);  clone_13 = sub_131 = None
    add_196: "f32[4, 240, 28, 28]" = torch.ops.aten.add.Scalar(mul_645, 1);  mul_645 = None
    mul_646: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_100, add_196);  sigmoid_100 = add_196 = None
    mul_647: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_195, mul_646);  add_195 = mul_646 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_197: "f32[240]" = torch.ops.aten.add.Tensor(primals_241, 1e-05);  primals_241 = None
    rsqrt_35: "f32[240]" = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
    unsqueeze_812: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(primals_240, 0);  primals_240 = None
    unsqueeze_813: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 2);  unsqueeze_812 = None
    unsqueeze_814: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_813, 3);  unsqueeze_813 = None
    sum_84: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_647, [0, 2, 3])
    sub_132: "f32[4, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_814);  convolution_21 = unsqueeze_814 = None
    mul_648: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_647, sub_132);  sub_132 = None
    sum_85: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_648, [0, 2, 3]);  mul_648 = None
    mul_653: "f32[240]" = torch.ops.aten.mul.Tensor(rsqrt_35, primals_27);  primals_27 = None
    unsqueeze_821: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_653, 0);  mul_653 = None
    unsqueeze_822: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 2);  unsqueeze_821 = None
    unsqueeze_823: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, 3);  unsqueeze_822 = None
    mul_654: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_647, unsqueeze_823);  mul_647 = unsqueeze_823 = None
    mul_655: "f32[240]" = torch.ops.aten.mul.Tensor(sum_85, rsqrt_35);  sum_85 = rsqrt_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(mul_654, mul_55, primals_128, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 240, [True, True, False]);  mul_654 = mul_55 = primals_128 = None
    getitem_177: "f32[4, 240, 28, 28]" = convolution_backward_59[0]
    getitem_178: "f32[240, 1, 5, 5]" = convolution_backward_59[1];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_101: "f32[4, 240, 28, 28]" = torch.ops.aten.sigmoid.default(clone_12)
    full_134: "f32[4, 240, 28, 28]" = torch.ops.aten.full.default([4, 240, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_133: "f32[4, 240, 28, 28]" = torch.ops.aten.sub.Tensor(full_134, sigmoid_101);  full_134 = None
    mul_656: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(clone_12, sub_133);  clone_12 = sub_133 = None
    add_198: "f32[4, 240, 28, 28]" = torch.ops.aten.add.Scalar(mul_656, 1);  mul_656 = None
    mul_657: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_101, add_198);  sigmoid_101 = add_198 = None
    mul_658: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_177, mul_657);  getitem_177 = mul_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_199: "f32[240]" = torch.ops.aten.add.Tensor(primals_239, 1e-05);  primals_239 = None
    rsqrt_36: "f32[240]" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
    unsqueeze_824: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(primals_238, 0);  primals_238 = None
    unsqueeze_825: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 2);  unsqueeze_824 = None
    unsqueeze_826: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_825, 3);  unsqueeze_825 = None
    sum_86: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_658, [0, 2, 3])
    sub_134: "f32[4, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_826);  convolution_20 = unsqueeze_826 = None
    mul_659: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_658, sub_134);  sub_134 = None
    sum_87: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_659, [0, 2, 3]);  mul_659 = None
    mul_664: "f32[240]" = torch.ops.aten.mul.Tensor(rsqrt_36, primals_25);  primals_25 = None
    unsqueeze_833: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_664, 0);  mul_664 = None
    unsqueeze_834: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 2);  unsqueeze_833 = None
    unsqueeze_835: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, 3);  unsqueeze_834 = None
    mul_665: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_658, unsqueeze_835);  mul_658 = unsqueeze_835 = None
    mul_666: "f32[240]" = torch.ops.aten.mul.Tensor(sum_87, rsqrt_36);  sum_87 = rsqrt_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_665, add_24, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_665 = add_24 = primals_127 = None
    getitem_180: "f32[4, 40, 28, 28]" = convolution_backward_60[0]
    getitem_181: "f32[240, 40, 1, 1]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_200: "f32[4, 40, 28, 28]" = torch.ops.aten.add.Tensor(getitem_165, getitem_180);  getitem_165 = getitem_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_201: "f32[40]" = torch.ops.aten.add.Tensor(primals_237, 1e-05);  primals_237 = None
    rsqrt_37: "f32[40]" = torch.ops.aten.rsqrt.default(add_201);  add_201 = None
    unsqueeze_836: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(primals_236, 0);  primals_236 = None
    unsqueeze_837: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, 2);  unsqueeze_836 = None
    unsqueeze_838: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_837, 3);  unsqueeze_837 = None
    sum_88: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_200, [0, 2, 3])
    sub_135: "f32[4, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_838);  convolution_19 = unsqueeze_838 = None
    mul_667: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_200, sub_135);  sub_135 = None
    sum_89: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_667, [0, 2, 3]);  mul_667 = None
    mul_672: "f32[40]" = torch.ops.aten.mul.Tensor(rsqrt_37, primals_23);  primals_23 = None
    unsqueeze_845: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_672, 0);  mul_672 = None
    unsqueeze_846: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 2);  unsqueeze_845 = None
    unsqueeze_847: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, 3);  unsqueeze_846 = None
    mul_673: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_200, unsqueeze_847);  add_200 = unsqueeze_847 = None
    mul_674: "f32[40]" = torch.ops.aten.mul.Tensor(sum_89, rsqrt_37);  sum_89 = rsqrt_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_673, mul_48, primals_126, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_673 = mul_48 = primals_126 = None
    getitem_183: "f32[4, 144, 28, 28]" = convolution_backward_61[0]
    getitem_184: "f32[40, 144, 1, 1]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_675: "f32[4, 144, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_183, mul_46);  mul_46 = None
    mul_676: "f32[4, 144, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_183, sigmoid_15);  getitem_183 = sigmoid_15 = None
    sum_90: "f32[4, 144, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_675, [2, 3], True);  mul_675 = None
    alias_28: "f32[4, 144, 1, 1]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    sub_136: "f32[4, 144, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_28)
    mul_677: "f32[4, 144, 1, 1]" = torch.ops.aten.mul.Tensor(alias_28, sub_136);  alias_28 = sub_136 = None
    mul_678: "f32[4, 144, 1, 1]" = torch.ops.aten.mul.Tensor(sum_90, mul_677);  sum_90 = mul_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_678, mul_47, primals_124, [144], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_678 = mul_47 = primals_124 = None
    getitem_186: "f32[4, 6, 1, 1]" = convolution_backward_62[0]
    getitem_187: "f32[144, 6, 1, 1]" = convolution_backward_62[1]
    getitem_188: "f32[144]" = convolution_backward_62[2];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_102: "f32[4, 6, 1, 1]" = torch.ops.aten.sigmoid.default(clone_11)
    full_135: "f32[4, 6, 1, 1]" = torch.ops.aten.full.default([4, 6, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_137: "f32[4, 6, 1, 1]" = torch.ops.aten.sub.Tensor(full_135, sigmoid_102);  full_135 = None
    mul_679: "f32[4, 6, 1, 1]" = torch.ops.aten.mul.Tensor(clone_11, sub_137);  clone_11 = sub_137 = None
    add_202: "f32[4, 6, 1, 1]" = torch.ops.aten.add.Scalar(mul_679, 1);  mul_679 = None
    mul_680: "f32[4, 6, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_102, add_202);  sigmoid_102 = add_202 = None
    mul_681: "f32[4, 6, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_186, mul_680);  getitem_186 = mul_680 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_681, mean_3, primals_122, [6], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_681 = mean_3 = primals_122 = None
    getitem_189: "f32[4, 144, 1, 1]" = convolution_backward_63[0]
    getitem_190: "f32[6, 144, 1, 1]" = convolution_backward_63[1]
    getitem_191: "f32[6]" = convolution_backward_63[2];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_13: "f32[4, 144, 28, 28]" = torch.ops.aten.expand.default(getitem_189, [4, 144, 28, 28]);  getitem_189 = None
    div_13: "f32[4, 144, 28, 28]" = torch.ops.aten.div.Scalar(expand_13, 784);  expand_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_203: "f32[4, 144, 28, 28]" = torch.ops.aten.add.Tensor(mul_676, div_13);  mul_676 = div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_103: "f32[4, 144, 28, 28]" = torch.ops.aten.sigmoid.default(clone_10)
    full_136: "f32[4, 144, 28, 28]" = torch.ops.aten.full.default([4, 144, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_138: "f32[4, 144, 28, 28]" = torch.ops.aten.sub.Tensor(full_136, sigmoid_103);  full_136 = None
    mul_682: "f32[4, 144, 28, 28]" = torch.ops.aten.mul.Tensor(clone_10, sub_138);  clone_10 = sub_138 = None
    add_204: "f32[4, 144, 28, 28]" = torch.ops.aten.add.Scalar(mul_682, 1);  mul_682 = None
    mul_683: "f32[4, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_103, add_204);  sigmoid_103 = add_204 = None
    mul_684: "f32[4, 144, 28, 28]" = torch.ops.aten.mul.Tensor(add_203, mul_683);  add_203 = mul_683 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_205: "f32[144]" = torch.ops.aten.add.Tensor(primals_235, 1e-05);  primals_235 = None
    rsqrt_38: "f32[144]" = torch.ops.aten.rsqrt.default(add_205);  add_205 = None
    unsqueeze_848: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(primals_234, 0);  primals_234 = None
    unsqueeze_849: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, 2);  unsqueeze_848 = None
    unsqueeze_850: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_849, 3);  unsqueeze_849 = None
    sum_91: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_684, [0, 2, 3])
    sub_139: "f32[4, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_850);  convolution_16 = unsqueeze_850 = None
    mul_685: "f32[4, 144, 28, 28]" = torch.ops.aten.mul.Tensor(mul_684, sub_139);  sub_139 = None
    sum_92: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_685, [0, 2, 3]);  mul_685 = None
    mul_690: "f32[144]" = torch.ops.aten.mul.Tensor(rsqrt_38, primals_21);  primals_21 = None
    unsqueeze_857: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_690, 0);  mul_690 = None
    unsqueeze_858: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 2);  unsqueeze_857 = None
    unsqueeze_859: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, 3);  unsqueeze_858 = None
    mul_691: "f32[4, 144, 28, 28]" = torch.ops.aten.mul.Tensor(mul_684, unsqueeze_859);  mul_684 = unsqueeze_859 = None
    mul_692: "f32[144]" = torch.ops.aten.mul.Tensor(sum_92, rsqrt_38);  sum_92 = rsqrt_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(mul_691, mul_42, primals_121, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 144, [True, True, False]);  mul_691 = mul_42 = primals_121 = None
    getitem_192: "f32[4, 144, 56, 56]" = convolution_backward_64[0]
    getitem_193: "f32[144, 1, 5, 5]" = convolution_backward_64[1];  convolution_backward_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_104: "f32[4, 144, 56, 56]" = torch.ops.aten.sigmoid.default(clone_9)
    full_137: "f32[4, 144, 56, 56]" = torch.ops.aten.full.default([4, 144, 56, 56], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_140: "f32[4, 144, 56, 56]" = torch.ops.aten.sub.Tensor(full_137, sigmoid_104);  full_137 = None
    mul_693: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(clone_9, sub_140);  clone_9 = sub_140 = None
    add_206: "f32[4, 144, 56, 56]" = torch.ops.aten.add.Scalar(mul_693, 1);  mul_693 = None
    mul_694: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sigmoid_104, add_206);  sigmoid_104 = add_206 = None
    mul_695: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_192, mul_694);  getitem_192 = mul_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_207: "f32[144]" = torch.ops.aten.add.Tensor(primals_233, 1e-05);  primals_233 = None
    rsqrt_39: "f32[144]" = torch.ops.aten.rsqrt.default(add_207);  add_207 = None
    unsqueeze_860: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(primals_232, 0);  primals_232 = None
    unsqueeze_861: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, 2);  unsqueeze_860 = None
    unsqueeze_862: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_861, 3);  unsqueeze_861 = None
    sum_93: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_695, [0, 2, 3])
    sub_141: "f32[4, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_862);  convolution_15 = unsqueeze_862 = None
    mul_696: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_695, sub_141);  sub_141 = None
    sum_94: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_696, [0, 2, 3]);  mul_696 = None
    mul_701: "f32[144]" = torch.ops.aten.mul.Tensor(rsqrt_39, primals_19);  primals_19 = None
    unsqueeze_869: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_701, 0);  mul_701 = None
    unsqueeze_870: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 2);  unsqueeze_869 = None
    unsqueeze_871: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, 3);  unsqueeze_870 = None
    mul_702: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_695, unsqueeze_871);  mul_695 = unsqueeze_871 = None
    mul_703: "f32[144]" = torch.ops.aten.mul.Tensor(sum_94, rsqrt_39);  sum_94 = rsqrt_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(mul_702, add_18, primals_120, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_702 = add_18 = primals_120 = None
    getitem_195: "f32[4, 24, 56, 56]" = convolution_backward_65[0]
    getitem_196: "f32[144, 24, 1, 1]" = convolution_backward_65[1];  convolution_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_208: "f32[24]" = torch.ops.aten.add.Tensor(primals_231, 1e-05);  primals_231 = None
    rsqrt_40: "f32[24]" = torch.ops.aten.rsqrt.default(add_208);  add_208 = None
    unsqueeze_872: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(primals_230, 0);  primals_230 = None
    unsqueeze_873: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, 2);  unsqueeze_872 = None
    unsqueeze_874: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_873, 3);  unsqueeze_873 = None
    sum_95: "f32[24]" = torch.ops.aten.sum.dim_IntList(getitem_195, [0, 2, 3])
    sub_142: "f32[4, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_874);  convolution_14 = unsqueeze_874 = None
    mul_704: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_195, sub_142);  sub_142 = None
    sum_96: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_704, [0, 2, 3]);  mul_704 = None
    mul_709: "f32[24]" = torch.ops.aten.mul.Tensor(rsqrt_40, primals_17);  primals_17 = None
    unsqueeze_881: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_709, 0);  mul_709 = None
    unsqueeze_882: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 2);  unsqueeze_881 = None
    unsqueeze_883: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, 3);  unsqueeze_882 = None
    mul_710: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_195, unsqueeze_883);  unsqueeze_883 = None
    mul_711: "f32[24]" = torch.ops.aten.mul.Tensor(sum_96, rsqrt_40);  sum_96 = rsqrt_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_710, mul_35, primals_119, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_710 = mul_35 = primals_119 = None
    getitem_198: "f32[4, 144, 56, 56]" = convolution_backward_66[0]
    getitem_199: "f32[24, 144, 1, 1]" = convolution_backward_66[1];  convolution_backward_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_712: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_198, mul_33);  mul_33 = None
    mul_713: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_198, sigmoid_11);  getitem_198 = sigmoid_11 = None
    sum_97: "f32[4, 144, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_712, [2, 3], True);  mul_712 = None
    alias_29: "f32[4, 144, 1, 1]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    sub_143: "f32[4, 144, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_29)
    mul_714: "f32[4, 144, 1, 1]" = torch.ops.aten.mul.Tensor(alias_29, sub_143);  alias_29 = sub_143 = None
    mul_715: "f32[4, 144, 1, 1]" = torch.ops.aten.mul.Tensor(sum_97, mul_714);  sum_97 = mul_714 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_67 = torch.ops.aten.convolution_backward.default(mul_715, mul_34, primals_117, [144], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_715 = mul_34 = primals_117 = None
    getitem_201: "f32[4, 6, 1, 1]" = convolution_backward_67[0]
    getitem_202: "f32[144, 6, 1, 1]" = convolution_backward_67[1]
    getitem_203: "f32[144]" = convolution_backward_67[2];  convolution_backward_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_105: "f32[4, 6, 1, 1]" = torch.ops.aten.sigmoid.default(clone_8)
    full_138: "f32[4, 6, 1, 1]" = torch.ops.aten.full.default([4, 6, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_144: "f32[4, 6, 1, 1]" = torch.ops.aten.sub.Tensor(full_138, sigmoid_105);  full_138 = None
    mul_716: "f32[4, 6, 1, 1]" = torch.ops.aten.mul.Tensor(clone_8, sub_144);  clone_8 = sub_144 = None
    add_209: "f32[4, 6, 1, 1]" = torch.ops.aten.add.Scalar(mul_716, 1);  mul_716 = None
    mul_717: "f32[4, 6, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_105, add_209);  sigmoid_105 = add_209 = None
    mul_718: "f32[4, 6, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_201, mul_717);  getitem_201 = mul_717 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_718, mean_2, primals_115, [6], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_718 = mean_2 = primals_115 = None
    getitem_204: "f32[4, 144, 1, 1]" = convolution_backward_68[0]
    getitem_205: "f32[6, 144, 1, 1]" = convolution_backward_68[1]
    getitem_206: "f32[6]" = convolution_backward_68[2];  convolution_backward_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_14: "f32[4, 144, 56, 56]" = torch.ops.aten.expand.default(getitem_204, [4, 144, 56, 56]);  getitem_204 = None
    div_14: "f32[4, 144, 56, 56]" = torch.ops.aten.div.Scalar(expand_14, 3136);  expand_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_210: "f32[4, 144, 56, 56]" = torch.ops.aten.add.Tensor(mul_713, div_14);  mul_713 = div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_106: "f32[4, 144, 56, 56]" = torch.ops.aten.sigmoid.default(clone_7)
    full_139: "f32[4, 144, 56, 56]" = torch.ops.aten.full.default([4, 144, 56, 56], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_145: "f32[4, 144, 56, 56]" = torch.ops.aten.sub.Tensor(full_139, sigmoid_106);  full_139 = None
    mul_719: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(clone_7, sub_145);  clone_7 = sub_145 = None
    add_211: "f32[4, 144, 56, 56]" = torch.ops.aten.add.Scalar(mul_719, 1);  mul_719 = None
    mul_720: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sigmoid_106, add_211);  sigmoid_106 = add_211 = None
    mul_721: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(add_210, mul_720);  add_210 = mul_720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_212: "f32[144]" = torch.ops.aten.add.Tensor(primals_229, 1e-05);  primals_229 = None
    rsqrt_41: "f32[144]" = torch.ops.aten.rsqrt.default(add_212);  add_212 = None
    unsqueeze_884: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(primals_228, 0);  primals_228 = None
    unsqueeze_885: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, 2);  unsqueeze_884 = None
    unsqueeze_886: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_885, 3);  unsqueeze_885 = None
    sum_98: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_721, [0, 2, 3])
    sub_146: "f32[4, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_886);  convolution_11 = unsqueeze_886 = None
    mul_722: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_721, sub_146);  sub_146 = None
    sum_99: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_722, [0, 2, 3]);  mul_722 = None
    mul_727: "f32[144]" = torch.ops.aten.mul.Tensor(rsqrt_41, primals_15);  primals_15 = None
    unsqueeze_893: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_727, 0);  mul_727 = None
    unsqueeze_894: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_893, 2);  unsqueeze_893 = None
    unsqueeze_895: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, 3);  unsqueeze_894 = None
    mul_728: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_721, unsqueeze_895);  mul_721 = unsqueeze_895 = None
    mul_729: "f32[144]" = torch.ops.aten.mul.Tensor(sum_99, rsqrt_41);  sum_99 = rsqrt_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_69 = torch.ops.aten.convolution_backward.default(mul_728, mul_29, primals_114, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 144, [True, True, False]);  mul_728 = mul_29 = primals_114 = None
    getitem_207: "f32[4, 144, 56, 56]" = convolution_backward_69[0]
    getitem_208: "f32[144, 1, 3, 3]" = convolution_backward_69[1];  convolution_backward_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_107: "f32[4, 144, 56, 56]" = torch.ops.aten.sigmoid.default(clone_6)
    full_140: "f32[4, 144, 56, 56]" = torch.ops.aten.full.default([4, 144, 56, 56], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_147: "f32[4, 144, 56, 56]" = torch.ops.aten.sub.Tensor(full_140, sigmoid_107);  full_140 = None
    mul_730: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(clone_6, sub_147);  clone_6 = sub_147 = None
    add_213: "f32[4, 144, 56, 56]" = torch.ops.aten.add.Scalar(mul_730, 1);  mul_730 = None
    mul_731: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sigmoid_107, add_213);  sigmoid_107 = add_213 = None
    mul_732: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_207, mul_731);  getitem_207 = mul_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_214: "f32[144]" = torch.ops.aten.add.Tensor(primals_227, 1e-05);  primals_227 = None
    rsqrt_42: "f32[144]" = torch.ops.aten.rsqrt.default(add_214);  add_214 = None
    unsqueeze_896: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(primals_226, 0);  primals_226 = None
    unsqueeze_897: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, 2);  unsqueeze_896 = None
    unsqueeze_898: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_897, 3);  unsqueeze_897 = None
    sum_100: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_732, [0, 2, 3])
    sub_148: "f32[4, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_898);  convolution_10 = unsqueeze_898 = None
    mul_733: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_732, sub_148);  sub_148 = None
    sum_101: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_733, [0, 2, 3]);  mul_733 = None
    mul_738: "f32[144]" = torch.ops.aten.mul.Tensor(rsqrt_42, primals_13);  primals_13 = None
    unsqueeze_905: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_738, 0);  mul_738 = None
    unsqueeze_906: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_905, 2);  unsqueeze_905 = None
    unsqueeze_907: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, 3);  unsqueeze_906 = None
    mul_739: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_732, unsqueeze_907);  mul_732 = unsqueeze_907 = None
    mul_740: "f32[144]" = torch.ops.aten.mul.Tensor(sum_101, rsqrt_42);  sum_101 = rsqrt_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_739, add_11, primals_113, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_739 = add_11 = primals_113 = None
    getitem_210: "f32[4, 24, 56, 56]" = convolution_backward_70[0]
    getitem_211: "f32[144, 24, 1, 1]" = convolution_backward_70[1];  convolution_backward_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_215: "f32[4, 24, 56, 56]" = torch.ops.aten.add.Tensor(getitem_195, getitem_210);  getitem_195 = getitem_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_216: "f32[24]" = torch.ops.aten.add.Tensor(primals_225, 1e-05);  primals_225 = None
    rsqrt_43: "f32[24]" = torch.ops.aten.rsqrt.default(add_216);  add_216 = None
    unsqueeze_908: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(primals_224, 0);  primals_224 = None
    unsqueeze_909: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, 2);  unsqueeze_908 = None
    unsqueeze_910: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_909, 3);  unsqueeze_909 = None
    sum_102: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_215, [0, 2, 3])
    sub_149: "f32[4, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_910);  convolution_9 = unsqueeze_910 = None
    mul_741: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_215, sub_149);  sub_149 = None
    sum_103: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_741, [0, 2, 3]);  mul_741 = None
    mul_746: "f32[24]" = torch.ops.aten.mul.Tensor(rsqrt_43, primals_11);  primals_11 = None
    unsqueeze_917: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_746, 0);  mul_746 = None
    unsqueeze_918: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_917, 2);  unsqueeze_917 = None
    unsqueeze_919: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, 3);  unsqueeze_918 = None
    mul_747: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_215, unsqueeze_919);  add_215 = unsqueeze_919 = None
    mul_748: "f32[24]" = torch.ops.aten.mul.Tensor(sum_103, rsqrt_43);  sum_103 = rsqrt_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_71 = torch.ops.aten.convolution_backward.default(mul_747, mul_22, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_747 = mul_22 = primals_112 = None
    getitem_213: "f32[4, 96, 56, 56]" = convolution_backward_71[0]
    getitem_214: "f32[24, 96, 1, 1]" = convolution_backward_71[1];  convolution_backward_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_749: "f32[4, 96, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_213, mul_20);  mul_20 = None
    mul_750: "f32[4, 96, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_213, sigmoid_7);  getitem_213 = sigmoid_7 = None
    sum_104: "f32[4, 96, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_749, [2, 3], True);  mul_749 = None
    alias_30: "f32[4, 96, 1, 1]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    sub_150: "f32[4, 96, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_30)
    mul_751: "f32[4, 96, 1, 1]" = torch.ops.aten.mul.Tensor(alias_30, sub_150);  alias_30 = sub_150 = None
    mul_752: "f32[4, 96, 1, 1]" = torch.ops.aten.mul.Tensor(sum_104, mul_751);  sum_104 = mul_751 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_72 = torch.ops.aten.convolution_backward.default(mul_752, mul_21, primals_110, [96], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_752 = mul_21 = primals_110 = None
    getitem_216: "f32[4, 4, 1, 1]" = convolution_backward_72[0]
    getitem_217: "f32[96, 4, 1, 1]" = convolution_backward_72[1]
    getitem_218: "f32[96]" = convolution_backward_72[2];  convolution_backward_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_108: "f32[4, 4, 1, 1]" = torch.ops.aten.sigmoid.default(clone_5)
    full_141: "f32[4, 4, 1, 1]" = torch.ops.aten.full.default([4, 4, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_151: "f32[4, 4, 1, 1]" = torch.ops.aten.sub.Tensor(full_141, sigmoid_108);  full_141 = None
    mul_753: "f32[4, 4, 1, 1]" = torch.ops.aten.mul.Tensor(clone_5, sub_151);  clone_5 = sub_151 = None
    add_217: "f32[4, 4, 1, 1]" = torch.ops.aten.add.Scalar(mul_753, 1);  mul_753 = None
    mul_754: "f32[4, 4, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_108, add_217);  sigmoid_108 = add_217 = None
    mul_755: "f32[4, 4, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_216, mul_754);  getitem_216 = mul_754 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_73 = torch.ops.aten.convolution_backward.default(mul_755, mean_1, primals_108, [4], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_755 = mean_1 = primals_108 = None
    getitem_219: "f32[4, 96, 1, 1]" = convolution_backward_73[0]
    getitem_220: "f32[4, 96, 1, 1]" = convolution_backward_73[1]
    getitem_221: "f32[4]" = convolution_backward_73[2];  convolution_backward_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_15: "f32[4, 96, 56, 56]" = torch.ops.aten.expand.default(getitem_219, [4, 96, 56, 56]);  getitem_219 = None
    div_15: "f32[4, 96, 56, 56]" = torch.ops.aten.div.Scalar(expand_15, 3136);  expand_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_218: "f32[4, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_750, div_15);  mul_750 = div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_109: "f32[4, 96, 56, 56]" = torch.ops.aten.sigmoid.default(clone_4)
    full_142: "f32[4, 96, 56, 56]" = torch.ops.aten.full.default([4, 96, 56, 56], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_152: "f32[4, 96, 56, 56]" = torch.ops.aten.sub.Tensor(full_142, sigmoid_109);  full_142 = None
    mul_756: "f32[4, 96, 56, 56]" = torch.ops.aten.mul.Tensor(clone_4, sub_152);  clone_4 = sub_152 = None
    add_219: "f32[4, 96, 56, 56]" = torch.ops.aten.add.Scalar(mul_756, 1);  mul_756 = None
    mul_757: "f32[4, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sigmoid_109, add_219);  sigmoid_109 = add_219 = None
    mul_758: "f32[4, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_218, mul_757);  add_218 = mul_757 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_220: "f32[96]" = torch.ops.aten.add.Tensor(primals_223, 1e-05);  primals_223 = None
    rsqrt_44: "f32[96]" = torch.ops.aten.rsqrt.default(add_220);  add_220 = None
    unsqueeze_920: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_222, 0);  primals_222 = None
    unsqueeze_921: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_920, 2);  unsqueeze_920 = None
    unsqueeze_922: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_921, 3);  unsqueeze_921 = None
    sum_105: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_758, [0, 2, 3])
    sub_153: "f32[4, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_922);  convolution_6 = unsqueeze_922 = None
    mul_759: "f32[4, 96, 56, 56]" = torch.ops.aten.mul.Tensor(mul_758, sub_153);  sub_153 = None
    sum_106: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_759, [0, 2, 3]);  mul_759 = None
    mul_764: "f32[96]" = torch.ops.aten.mul.Tensor(rsqrt_44, primals_9);  primals_9 = None
    unsqueeze_929: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_764, 0);  mul_764 = None
    unsqueeze_930: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_929, 2);  unsqueeze_929 = None
    unsqueeze_931: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_930, 3);  unsqueeze_930 = None
    mul_765: "f32[4, 96, 56, 56]" = torch.ops.aten.mul.Tensor(mul_758, unsqueeze_931);  mul_758 = unsqueeze_931 = None
    mul_766: "f32[96]" = torch.ops.aten.mul.Tensor(sum_106, rsqrt_44);  sum_106 = rsqrt_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_74 = torch.ops.aten.convolution_backward.default(mul_765, mul_16, primals_107, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 96, [True, True, False]);  mul_765 = mul_16 = primals_107 = None
    getitem_222: "f32[4, 96, 112, 112]" = convolution_backward_74[0]
    getitem_223: "f32[96, 1, 3, 3]" = convolution_backward_74[1];  convolution_backward_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_110: "f32[4, 96, 112, 112]" = torch.ops.aten.sigmoid.default(clone_3)
    full_143: "f32[4, 96, 112, 112]" = torch.ops.aten.full.default([4, 96, 112, 112], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_154: "f32[4, 96, 112, 112]" = torch.ops.aten.sub.Tensor(full_143, sigmoid_110);  full_143 = None
    mul_767: "f32[4, 96, 112, 112]" = torch.ops.aten.mul.Tensor(clone_3, sub_154);  clone_3 = sub_154 = None
    add_221: "f32[4, 96, 112, 112]" = torch.ops.aten.add.Scalar(mul_767, 1);  mul_767 = None
    mul_768: "f32[4, 96, 112, 112]" = torch.ops.aten.mul.Tensor(sigmoid_110, add_221);  sigmoid_110 = add_221 = None
    mul_769: "f32[4, 96, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_222, mul_768);  getitem_222 = mul_768 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_222: "f32[96]" = torch.ops.aten.add.Tensor(primals_221, 1e-05);  primals_221 = None
    rsqrt_45: "f32[96]" = torch.ops.aten.rsqrt.default(add_222);  add_222 = None
    unsqueeze_932: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_220, 0);  primals_220 = None
    unsqueeze_933: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_932, 2);  unsqueeze_932 = None
    unsqueeze_934: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_933, 3);  unsqueeze_933 = None
    sum_107: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_769, [0, 2, 3])
    sub_155: "f32[4, 96, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_934);  convolution_5 = unsqueeze_934 = None
    mul_770: "f32[4, 96, 112, 112]" = torch.ops.aten.mul.Tensor(mul_769, sub_155);  sub_155 = None
    sum_108: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_770, [0, 2, 3]);  mul_770 = None
    mul_775: "f32[96]" = torch.ops.aten.mul.Tensor(rsqrt_45, primals_7);  primals_7 = None
    unsqueeze_941: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_775, 0);  mul_775 = None
    unsqueeze_942: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_941, 2);  unsqueeze_941 = None
    unsqueeze_943: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_942, 3);  unsqueeze_942 = None
    mul_776: "f32[4, 96, 112, 112]" = torch.ops.aten.mul.Tensor(mul_769, unsqueeze_943);  mul_769 = unsqueeze_943 = None
    mul_777: "f32[96]" = torch.ops.aten.mul.Tensor(sum_108, rsqrt_45);  sum_108 = rsqrt_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_75 = torch.ops.aten.convolution_backward.default(mul_776, add_5, primals_106, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_776 = add_5 = primals_106 = None
    getitem_225: "f32[4, 16, 112, 112]" = convolution_backward_75[0]
    getitem_226: "f32[96, 16, 1, 1]" = convolution_backward_75[1];  convolution_backward_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_223: "f32[16]" = torch.ops.aten.add.Tensor(primals_219, 1e-05);  primals_219 = None
    rsqrt_46: "f32[16]" = torch.ops.aten.rsqrt.default(add_223);  add_223 = None
    unsqueeze_944: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(primals_218, 0);  primals_218 = None
    unsqueeze_945: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_944, 2);  unsqueeze_944 = None
    unsqueeze_946: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_945, 3);  unsqueeze_945 = None
    sum_109: "f32[16]" = torch.ops.aten.sum.dim_IntList(getitem_225, [0, 2, 3])
    sub_156: "f32[4, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_946);  convolution_4 = unsqueeze_946 = None
    mul_778: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_225, sub_156);  sub_156 = None
    sum_110: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_778, [0, 2, 3]);  mul_778 = None
    mul_783: "f32[16]" = torch.ops.aten.mul.Tensor(rsqrt_46, primals_5);  primals_5 = None
    unsqueeze_953: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_783, 0);  mul_783 = None
    unsqueeze_954: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_953, 2);  unsqueeze_953 = None
    unsqueeze_955: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_954, 3);  unsqueeze_954 = None
    mul_784: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_225, unsqueeze_955);  getitem_225 = unsqueeze_955 = None
    mul_785: "f32[16]" = torch.ops.aten.mul.Tensor(sum_110, rsqrt_46);  sum_110 = rsqrt_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_76 = torch.ops.aten.convolution_backward.default(mul_784, mul_9, primals_105, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_784 = mul_9 = primals_105 = None
    getitem_228: "f32[4, 32, 112, 112]" = convolution_backward_76[0]
    getitem_229: "f32[16, 32, 1, 1]" = convolution_backward_76[1];  convolution_backward_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_786: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_228, mul_7);  mul_7 = None
    mul_787: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_228, sigmoid_3);  getitem_228 = sigmoid_3 = None
    sum_111: "f32[4, 32, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_786, [2, 3], True);  mul_786 = None
    alias_31: "f32[4, 32, 1, 1]" = torch.ops.aten.alias.default(alias);  alias = None
    sub_157: "f32[4, 32, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_31)
    mul_788: "f32[4, 32, 1, 1]" = torch.ops.aten.mul.Tensor(alias_31, sub_157);  alias_31 = sub_157 = None
    mul_789: "f32[4, 32, 1, 1]" = torch.ops.aten.mul.Tensor(sum_111, mul_788);  sum_111 = mul_788 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_77 = torch.ops.aten.convolution_backward.default(mul_789, mul_8, primals_103, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_789 = mul_8 = primals_103 = None
    getitem_231: "f32[4, 8, 1, 1]" = convolution_backward_77[0]
    getitem_232: "f32[32, 8, 1, 1]" = convolution_backward_77[1]
    getitem_233: "f32[32]" = convolution_backward_77[2];  convolution_backward_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_111: "f32[4, 8, 1, 1]" = torch.ops.aten.sigmoid.default(clone_2)
    full_144: "f32[4, 8, 1, 1]" = torch.ops.aten.full.default([4, 8, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_158: "f32[4, 8, 1, 1]" = torch.ops.aten.sub.Tensor(full_144, sigmoid_111);  full_144 = None
    mul_790: "f32[4, 8, 1, 1]" = torch.ops.aten.mul.Tensor(clone_2, sub_158);  clone_2 = sub_158 = None
    add_224: "f32[4, 8, 1, 1]" = torch.ops.aten.add.Scalar(mul_790, 1);  mul_790 = None
    mul_791: "f32[4, 8, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_111, add_224);  sigmoid_111 = add_224 = None
    mul_792: "f32[4, 8, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_231, mul_791);  getitem_231 = mul_791 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_78 = torch.ops.aten.convolution_backward.default(mul_792, mean, primals_101, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_792 = mean = primals_101 = None
    getitem_234: "f32[4, 32, 1, 1]" = convolution_backward_78[0]
    getitem_235: "f32[8, 32, 1, 1]" = convolution_backward_78[1]
    getitem_236: "f32[8]" = convolution_backward_78[2];  convolution_backward_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_16: "f32[4, 32, 112, 112]" = torch.ops.aten.expand.default(getitem_234, [4, 32, 112, 112]);  getitem_234 = None
    div_16: "f32[4, 32, 112, 112]" = torch.ops.aten.div.Scalar(expand_16, 12544);  expand_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_225: "f32[4, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_787, div_16);  mul_787 = div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_112: "f32[4, 32, 112, 112]" = torch.ops.aten.sigmoid.default(clone_1)
    full_145: "f32[4, 32, 112, 112]" = torch.ops.aten.full.default([4, 32, 112, 112], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_159: "f32[4, 32, 112, 112]" = torch.ops.aten.sub.Tensor(full_145, sigmoid_112);  full_145 = None
    mul_793: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(clone_1, sub_159);  clone_1 = sub_159 = None
    add_226: "f32[4, 32, 112, 112]" = torch.ops.aten.add.Scalar(mul_793, 1);  mul_793 = None
    mul_794: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sigmoid_112, add_226);  sigmoid_112 = add_226 = None
    mul_795: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(add_225, mul_794);  add_225 = mul_794 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_227: "f32[32]" = torch.ops.aten.add.Tensor(primals_217, 1e-05);  primals_217 = None
    rsqrt_47: "f32[32]" = torch.ops.aten.rsqrt.default(add_227);  add_227 = None
    unsqueeze_956: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(primals_216, 0);  primals_216 = None
    unsqueeze_957: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_956, 2);  unsqueeze_956 = None
    unsqueeze_958: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_957, 3);  unsqueeze_957 = None
    sum_112: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_795, [0, 2, 3])
    sub_160: "f32[4, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_958);  convolution_1 = unsqueeze_958 = None
    mul_796: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_795, sub_160);  sub_160 = None
    sum_113: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_796, [0, 2, 3]);  mul_796 = None
    mul_801: "f32[32]" = torch.ops.aten.mul.Tensor(rsqrt_47, primals_3);  primals_3 = None
    unsqueeze_965: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_801, 0);  mul_801 = None
    unsqueeze_966: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_965, 2);  unsqueeze_965 = None
    unsqueeze_967: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_966, 3);  unsqueeze_966 = None
    mul_802: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_795, unsqueeze_967);  mul_795 = unsqueeze_967 = None
    mul_803: "f32[32]" = torch.ops.aten.mul.Tensor(sum_113, rsqrt_47);  sum_113 = rsqrt_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_79 = torch.ops.aten.convolution_backward.default(mul_802, mul_3, primals_100, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_802 = mul_3 = primals_100 = None
    getitem_237: "f32[4, 32, 112, 112]" = convolution_backward_79[0]
    getitem_238: "f32[32, 1, 3, 3]" = convolution_backward_79[1];  convolution_backward_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_113: "f32[4, 32, 112, 112]" = torch.ops.aten.sigmoid.default(clone)
    full_146: "f32[4, 32, 112, 112]" = torch.ops.aten.full.default([4, 32, 112, 112], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_161: "f32[4, 32, 112, 112]" = torch.ops.aten.sub.Tensor(full_146, sigmoid_113);  full_146 = None
    mul_804: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(clone, sub_161);  clone = sub_161 = None
    add_228: "f32[4, 32, 112, 112]" = torch.ops.aten.add.Scalar(mul_804, 1);  mul_804 = None
    mul_805: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sigmoid_113, add_228);  sigmoid_113 = add_228 = None
    mul_806: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_237, mul_805);  getitem_237 = mul_805 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_229: "f32[32]" = torch.ops.aten.add.Tensor(primals_215, 1e-05);  primals_215 = None
    rsqrt_48: "f32[32]" = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
    unsqueeze_968: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(primals_214, 0);  primals_214 = None
    unsqueeze_969: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_968, 2);  unsqueeze_968 = None
    unsqueeze_970: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_969, 3);  unsqueeze_969 = None
    sum_114: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_806, [0, 2, 3])
    sub_162: "f32[4, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_970);  convolution = unsqueeze_970 = None
    mul_807: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_806, sub_162);  sub_162 = None
    sum_115: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_807, [0, 2, 3]);  mul_807 = None
    mul_812: "f32[32]" = torch.ops.aten.mul.Tensor(rsqrt_48, primals_1);  primals_1 = None
    unsqueeze_977: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_812, 0);  mul_812 = None
    unsqueeze_978: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_977, 2);  unsqueeze_977 = None
    unsqueeze_979: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_978, 3);  unsqueeze_978 = None
    mul_813: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_806, unsqueeze_979);  mul_806 = unsqueeze_979 = None
    mul_814: "f32[32]" = torch.ops.aten.mul.Tensor(sum_115, rsqrt_48);  sum_115 = rsqrt_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:162, code: x = self.conv_stem(x)
    convolution_backward_80 = torch.ops.aten.convolution_backward.default(mul_813, primals_312, primals_99, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_813 = primals_312 = primals_99 = None
    getitem_241: "f32[32, 3, 3, 3]" = convolution_backward_80[1];  convolution_backward_80 = None
    return pytree.tree_unflatten([addmm, mul_814, sum_114, mul_803, sum_112, mul_785, sum_109, mul_777, sum_107, mul_766, sum_105, mul_748, sum_102, mul_740, sum_100, mul_729, sum_98, mul_711, sum_95, mul_703, sum_93, mul_692, sum_91, mul_674, sum_88, mul_666, sum_86, mul_655, sum_84, mul_637, sum_81, mul_629, sum_79, mul_618, sum_77, mul_600, sum_74, mul_592, sum_72, mul_581, sum_70, mul_563, sum_67, mul_555, sum_65, mul_544, sum_63, mul_526, sum_60, mul_518, sum_58, mul_507, sum_56, mul_489, sum_53, mul_481, sum_51, mul_470, sum_49, mul_452, sum_46, mul_444, sum_44, mul_433, sum_42, mul_415, sum_39, mul_407, sum_37, mul_396, sum_35, mul_378, sum_32, mul_370, sum_30, mul_359, sum_28, mul_341, sum_25, mul_333, sum_23, mul_322, sum_21, mul_304, sum_18, mul_296, sum_16, mul_285, sum_14, mul_267, sum_11, mul_259, sum_9, mul_248, sum_7, mul_230, sum_4, mul_222, sum_2, getitem_241, getitem_238, getitem_235, getitem_236, getitem_232, getitem_233, getitem_229, getitem_226, getitem_223, getitem_220, getitem_221, getitem_217, getitem_218, getitem_214, getitem_211, getitem_208, getitem_205, getitem_206, getitem_202, getitem_203, getitem_199, getitem_196, getitem_193, getitem_190, getitem_191, getitem_187, getitem_188, getitem_184, getitem_181, getitem_178, getitem_175, getitem_176, getitem_172, getitem_173, getitem_169, getitem_166, getitem_163, getitem_160, getitem_161, getitem_157, getitem_158, getitem_154, getitem_151, getitem_148, getitem_145, getitem_146, getitem_142, getitem_143, getitem_139, getitem_136, getitem_133, getitem_130, getitem_131, getitem_127, getitem_128, getitem_124, getitem_121, getitem_118, getitem_115, getitem_116, getitem_112, getitem_113, getitem_109, getitem_106, getitem_103, getitem_100, getitem_101, getitem_97, getitem_98, getitem_94, getitem_91, getitem_88, getitem_85, getitem_86, getitem_82, getitem_83, getitem_79, getitem_76, getitem_73, getitem_70, getitem_71, getitem_67, getitem_68, getitem_64, getitem_61, getitem_58, getitem_55, getitem_56, getitem_52, getitem_53, getitem_49, getitem_46, getitem_43, getitem_40, getitem_41, getitem_37, getitem_38, getitem_34, getitem_31, getitem_28, getitem_25, getitem_26, getitem_22, getitem_23, getitem_19, getitem_16, getitem_13, getitem_10, getitem_11, getitem_7, getitem_8, getitem_4, getitem_1, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    