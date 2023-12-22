from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[64, 3, 7, 7]"; primals_2: "f32[64]"; primals_3: "f32[64]"; primals_4: "f32[64, 64, 1, 1]"; primals_5: "f32[64]"; primals_6: "f32[64]"; primals_7: "f32[64, 64, 3, 3]"; primals_8: "f32[64]"; primals_9: "f32[64]"; primals_10: "f32[256, 64, 1, 1]"; primals_11: "f32[256]"; primals_12: "f32[256]"; primals_13: "f32[256, 64, 1, 1]"; primals_14: "f32[256]"; primals_15: "f32[256]"; primals_16: "f32[64, 256, 1, 1]"; primals_17: "f32[64]"; primals_18: "f32[64]"; primals_19: "f32[64, 64, 3, 3]"; primals_20: "f32[64]"; primals_21: "f32[64]"; primals_22: "f32[256, 64, 1, 1]"; primals_23: "f32[256]"; primals_24: "f32[256]"; primals_25: "f32[64, 256, 1, 1]"; primals_26: "f32[64]"; primals_27: "f32[64]"; primals_28: "f32[64, 64, 3, 3]"; primals_29: "f32[64]"; primals_30: "f32[64]"; primals_31: "f32[256, 64, 1, 1]"; primals_32: "f32[256]"; primals_33: "f32[256]"; primals_34: "f32[128, 256, 1, 1]"; primals_35: "f32[128]"; primals_36: "f32[128]"; primals_37: "f32[128, 128, 3, 3]"; primals_38: "f32[128]"; primals_39: "f32[128]"; primals_40: "f32[512, 128, 1, 1]"; primals_41: "f32[512]"; primals_42: "f32[512]"; primals_43: "f32[512, 256, 1, 1]"; primals_44: "f32[512]"; primals_45: "f32[512]"; primals_46: "f32[128, 512, 1, 1]"; primals_47: "f32[128]"; primals_48: "f32[128]"; primals_49: "f32[128, 128, 3, 3]"; primals_50: "f32[128]"; primals_51: "f32[128]"; primals_52: "f32[512, 128, 1, 1]"; primals_53: "f32[512]"; primals_54: "f32[512]"; primals_55: "f32[128, 512, 1, 1]"; primals_56: "f32[128]"; primals_57: "f32[128]"; primals_58: "f32[128, 128, 3, 3]"; primals_59: "f32[128]"; primals_60: "f32[128]"; primals_61: "f32[512, 128, 1, 1]"; primals_62: "f32[512]"; primals_63: "f32[512]"; primals_64: "f32[128, 512, 1, 1]"; primals_65: "f32[128]"; primals_66: "f32[128]"; primals_67: "f32[128, 128, 3, 3]"; primals_68: "f32[128]"; primals_69: "f32[128]"; primals_70: "f32[512, 128, 1, 1]"; primals_71: "f32[512]"; primals_72: "f32[512]"; primals_73: "f32[256, 512, 1, 1]"; primals_74: "f32[256]"; primals_75: "f32[256]"; primals_76: "f32[256, 256, 3, 3]"; primals_77: "f32[256]"; primals_78: "f32[256]"; primals_79: "f32[1024, 256, 1, 1]"; primals_80: "f32[1024]"; primals_81: "f32[1024]"; primals_82: "f32[1024, 512, 1, 1]"; primals_83: "f32[1024]"; primals_84: "f32[1024]"; primals_85: "f32[256, 1024, 1, 1]"; primals_86: "f32[256]"; primals_87: "f32[256]"; primals_88: "f32[256, 256, 3, 3]"; primals_89: "f32[256]"; primals_90: "f32[256]"; primals_91: "f32[1024, 256, 1, 1]"; primals_92: "f32[1024]"; primals_93: "f32[1024]"; primals_94: "f32[256, 1024, 1, 1]"; primals_95: "f32[256]"; primals_96: "f32[256]"; primals_97: "f32[256, 256, 3, 3]"; primals_98: "f32[256]"; primals_99: "f32[256]"; primals_100: "f32[1024, 256, 1, 1]"; primals_101: "f32[1024]"; primals_102: "f32[1024]"; primals_103: "f32[256, 1024, 1, 1]"; primals_104: "f32[256]"; primals_105: "f32[256]"; primals_106: "f32[256, 256, 3, 3]"; primals_107: "f32[256]"; primals_108: "f32[256]"; primals_109: "f32[1024, 256, 1, 1]"; primals_110: "f32[1024]"; primals_111: "f32[1024]"; primals_112: "f32[256, 1024, 1, 1]"; primals_113: "f32[256]"; primals_114: "f32[256]"; primals_115: "f32[256, 256, 3, 3]"; primals_116: "f32[256]"; primals_117: "f32[256]"; primals_118: "f32[1024, 256, 1, 1]"; primals_119: "f32[1024]"; primals_120: "f32[1024]"; primals_121: "f32[256, 1024, 1, 1]"; primals_122: "f32[256]"; primals_123: "f32[256]"; primals_124: "f32[256, 256, 3, 3]"; primals_125: "f32[256]"; primals_126: "f32[256]"; primals_127: "f32[1024, 256, 1, 1]"; primals_128: "f32[1024]"; primals_129: "f32[1024]"; primals_130: "f32[512, 1024, 1, 1]"; primals_131: "f32[512]"; primals_132: "f32[512]"; primals_133: "f32[512, 512, 3, 3]"; primals_134: "f32[512]"; primals_135: "f32[512]"; primals_136: "f32[2048, 512, 1, 1]"; primals_137: "f32[2048]"; primals_138: "f32[2048]"; primals_139: "f32[2048, 1024, 1, 1]"; primals_140: "f32[2048]"; primals_141: "f32[2048]"; primals_142: "f32[512, 2048, 1, 1]"; primals_143: "f32[512]"; primals_144: "f32[512]"; primals_145: "f32[512, 512, 3, 3]"; primals_146: "f32[512]"; primals_147: "f32[512]"; primals_148: "f32[2048, 512, 1, 1]"; primals_149: "f32[2048]"; primals_150: "f32[2048]"; primals_151: "f32[512, 2048, 1, 1]"; primals_152: "f32[512]"; primals_153: "f32[512]"; primals_154: "f32[512, 512, 3, 3]"; primals_155: "f32[512]"; primals_156: "f32[512]"; primals_157: "f32[2048, 512, 1, 1]"; primals_158: "f32[2048]"; primals_159: "f32[2048]"; primals_160: "f32[1000, 2048]"; primals_161: "f32[1000]"; primals_162: "f32[64]"; primals_163: "f32[64]"; primals_164: "i64[]"; primals_165: "f32[64]"; primals_166: "f32[64]"; primals_167: "i64[]"; primals_168: "f32[64]"; primals_169: "f32[64]"; primals_170: "i64[]"; primals_171: "f32[256]"; primals_172: "f32[256]"; primals_173: "i64[]"; primals_174: "f32[256]"; primals_175: "f32[256]"; primals_176: "i64[]"; primals_177: "f32[64]"; primals_178: "f32[64]"; primals_179: "i64[]"; primals_180: "f32[64]"; primals_181: "f32[64]"; primals_182: "i64[]"; primals_183: "f32[256]"; primals_184: "f32[256]"; primals_185: "i64[]"; primals_186: "f32[64]"; primals_187: "f32[64]"; primals_188: "i64[]"; primals_189: "f32[64]"; primals_190: "f32[64]"; primals_191: "i64[]"; primals_192: "f32[256]"; primals_193: "f32[256]"; primals_194: "i64[]"; primals_195: "f32[128]"; primals_196: "f32[128]"; primals_197: "i64[]"; primals_198: "f32[128]"; primals_199: "f32[128]"; primals_200: "i64[]"; primals_201: "f32[512]"; primals_202: "f32[512]"; primals_203: "i64[]"; primals_204: "f32[512]"; primals_205: "f32[512]"; primals_206: "i64[]"; primals_207: "f32[128]"; primals_208: "f32[128]"; primals_209: "i64[]"; primals_210: "f32[128]"; primals_211: "f32[128]"; primals_212: "i64[]"; primals_213: "f32[512]"; primals_214: "f32[512]"; primals_215: "i64[]"; primals_216: "f32[128]"; primals_217: "f32[128]"; primals_218: "i64[]"; primals_219: "f32[128]"; primals_220: "f32[128]"; primals_221: "i64[]"; primals_222: "f32[512]"; primals_223: "f32[512]"; primals_224: "i64[]"; primals_225: "f32[128]"; primals_226: "f32[128]"; primals_227: "i64[]"; primals_228: "f32[128]"; primals_229: "f32[128]"; primals_230: "i64[]"; primals_231: "f32[512]"; primals_232: "f32[512]"; primals_233: "i64[]"; primals_234: "f32[256]"; primals_235: "f32[256]"; primals_236: "i64[]"; primals_237: "f32[256]"; primals_238: "f32[256]"; primals_239: "i64[]"; primals_240: "f32[1024]"; primals_241: "f32[1024]"; primals_242: "i64[]"; primals_243: "f32[1024]"; primals_244: "f32[1024]"; primals_245: "i64[]"; primals_246: "f32[256]"; primals_247: "f32[256]"; primals_248: "i64[]"; primals_249: "f32[256]"; primals_250: "f32[256]"; primals_251: "i64[]"; primals_252: "f32[1024]"; primals_253: "f32[1024]"; primals_254: "i64[]"; primals_255: "f32[256]"; primals_256: "f32[256]"; primals_257: "i64[]"; primals_258: "f32[256]"; primals_259: "f32[256]"; primals_260: "i64[]"; primals_261: "f32[1024]"; primals_262: "f32[1024]"; primals_263: "i64[]"; primals_264: "f32[256]"; primals_265: "f32[256]"; primals_266: "i64[]"; primals_267: "f32[256]"; primals_268: "f32[256]"; primals_269: "i64[]"; primals_270: "f32[1024]"; primals_271: "f32[1024]"; primals_272: "i64[]"; primals_273: "f32[256]"; primals_274: "f32[256]"; primals_275: "i64[]"; primals_276: "f32[256]"; primals_277: "f32[256]"; primals_278: "i64[]"; primals_279: "f32[1024]"; primals_280: "f32[1024]"; primals_281: "i64[]"; primals_282: "f32[256]"; primals_283: "f32[256]"; primals_284: "i64[]"; primals_285: "f32[256]"; primals_286: "f32[256]"; primals_287: "i64[]"; primals_288: "f32[1024]"; primals_289: "f32[1024]"; primals_290: "i64[]"; primals_291: "f32[512]"; primals_292: "f32[512]"; primals_293: "i64[]"; primals_294: "f32[512]"; primals_295: "f32[512]"; primals_296: "i64[]"; primals_297: "f32[2048]"; primals_298: "f32[2048]"; primals_299: "i64[]"; primals_300: "f32[2048]"; primals_301: "f32[2048]"; primals_302: "i64[]"; primals_303: "f32[512]"; primals_304: "f32[512]"; primals_305: "i64[]"; primals_306: "f32[512]"; primals_307: "f32[512]"; primals_308: "i64[]"; primals_309: "f32[2048]"; primals_310: "f32[2048]"; primals_311: "i64[]"; primals_312: "f32[512]"; primals_313: "f32[512]"; primals_314: "i64[]"; primals_315: "f32[512]"; primals_316: "f32[512]"; primals_317: "i64[]"; primals_318: "f32[2048]"; primals_319: "f32[2048]"; primals_320: "i64[]"; primals_321: "f32[4, 3, 224, 224]"; tangents_1: "f32[4, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:268, code: x = self.conv1(x)
    convolution: "f32[4, 64, 112, 112]" = torch.ops.aten.convolution.default(primals_321, primals_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:269, code: x = self.bn1(x)
    convert_element_type: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_162, torch.float32)
    convert_element_type_1: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_163, torch.float32)
    add: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1e-05);  convert_element_type_1 = None
    sqrt: "f32[64]" = torch.ops.aten.sqrt.default(add);  add = None
    reciprocal: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
    mul: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
    unsqueeze: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
    unsqueeze_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
    unsqueeze_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    sub: "f32[4, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  unsqueeze_1 = None
    mul_1: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
    unsqueeze_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1)
    unsqueeze_5: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_2: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
    unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1);  primals_3 = None
    unsqueeze_7: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_1: "f32[4, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:270, code: x = self.relu(x)
    relu: "f32[4, 64, 112, 112]" = torch.ops.aten.relu.default(add_1);  add_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:271, code: x = self.maxpool(x)
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(relu, [3, 3], [2, 2], [1, 1])
    getitem: "f32[4, 64, 56, 56]" = max_pool2d_with_indices[0]
    getitem_1: "i64[4, 64, 56, 56]" = max_pool2d_with_indices[1];  max_pool2d_with_indices = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_1: "f32[4, 64, 56, 56]" = torch.ops.aten.convolution.default(getitem, primals_4, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    convert_element_type_2: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_165, torch.float32)
    convert_element_type_3: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_166, torch.float32)
    add_2: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_3, 1e-05);  convert_element_type_3 = None
    sqrt_1: "f32[64]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
    reciprocal_1: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_3: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, -1);  convert_element_type_2 = None
    unsqueeze_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    unsqueeze_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
    unsqueeze_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    sub_1: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  unsqueeze_9 = None
    mul_4: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_13: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_5: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
    unsqueeze_14: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_15: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_3: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    relu_1: "f32[4, 64, 56, 56]" = torch.ops.aten.relu.default(add_3);  add_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_2: "f32[4, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_1, primals_7, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    convert_element_type_4: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_168, torch.float32)
    convert_element_type_5: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_169, torch.float32)
    add_4: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_5, 1e-05);  convert_element_type_5 = None
    sqrt_2: "f32[64]" = torch.ops.aten.sqrt.default(add_4);  add_4 = None
    reciprocal_2: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_6: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_16: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, -1);  convert_element_type_4 = None
    unsqueeze_17: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    unsqueeze_18: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
    unsqueeze_19: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    sub_2: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  unsqueeze_17 = None
    mul_7: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1)
    unsqueeze_21: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_8: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_21);  mul_7 = unsqueeze_21 = None
    unsqueeze_22: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1);  primals_9 = None
    unsqueeze_23: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_5: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_8, unsqueeze_23);  mul_8 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    relu_2: "f32[4, 64, 56, 56]" = torch.ops.aten.relu.default(add_5);  add_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_3: "f32[4, 256, 56, 56]" = torch.ops.aten.convolution.default(relu_2, primals_10, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    convert_element_type_6: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_171, torch.float32)
    convert_element_type_7: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_172, torch.float32)
    add_6: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_7, 1e-05);  convert_element_type_7 = None
    sqrt_3: "f32[256]" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
    reciprocal_3: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
    mul_9: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
    unsqueeze_24: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_6, -1);  convert_element_type_6 = None
    unsqueeze_25: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    unsqueeze_26: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_9, -1);  mul_9 = None
    unsqueeze_27: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    sub_3: "f32[4, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  unsqueeze_25 = None
    mul_10: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
    unsqueeze_28: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_29: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_11: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_29);  mul_10 = unsqueeze_29 = None
    unsqueeze_30: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_31: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_7: "f32[4, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_11, unsqueeze_31);  mul_11 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:158, code: identity = self.downsample(x)
    convolution_4: "f32[4, 256, 56, 56]" = torch.ops.aten.convolution.default(getitem, primals_13, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_8: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_174, torch.float32)
    convert_element_type_9: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_175, torch.float32)
    add_8: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_9, 1e-05);  convert_element_type_9 = None
    sqrt_4: "f32[256]" = torch.ops.aten.sqrt.default(add_8);  add_8 = None
    reciprocal_4: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
    mul_12: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
    unsqueeze_32: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_8, -1);  convert_element_type_8 = None
    unsqueeze_33: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    unsqueeze_34: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
    unsqueeze_35: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    sub_4: "f32[4, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  unsqueeze_33 = None
    mul_13: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
    unsqueeze_36: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1)
    unsqueeze_37: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_14: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_37);  mul_13 = unsqueeze_37 = None
    unsqueeze_38: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1);  primals_15 = None
    unsqueeze_39: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_9: "f32[4, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_39);  mul_14 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:160, code: out += identity
    add_10: "f32[4, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_7, add_9);  add_7 = add_9 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    relu_3: "f32[4, 256, 56, 56]" = torch.ops.aten.relu.default(add_10);  add_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_5: "f32[4, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_3, primals_16, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    convert_element_type_10: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_177, torch.float32)
    convert_element_type_11: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_178, torch.float32)
    add_11: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_11, 1e-05);  convert_element_type_11 = None
    sqrt_5: "f32[64]" = torch.ops.aten.sqrt.default(add_11);  add_11 = None
    reciprocal_5: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
    mul_15: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
    unsqueeze_40: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_10, -1);  convert_element_type_10 = None
    unsqueeze_41: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    unsqueeze_42: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_15, -1);  mul_15 = None
    unsqueeze_43: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    sub_5: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_41);  unsqueeze_41 = None
    mul_16: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
    unsqueeze_44: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_45: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_17: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_45);  mul_16 = unsqueeze_45 = None
    unsqueeze_46: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_47: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_12: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_17, unsqueeze_47);  mul_17 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    relu_4: "f32[4, 64, 56, 56]" = torch.ops.aten.relu.default(add_12);  add_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_6: "f32[4, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_4, primals_19, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    convert_element_type_12: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_180, torch.float32)
    convert_element_type_13: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_181, torch.float32)
    add_13: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_13, 1e-05);  convert_element_type_13 = None
    sqrt_6: "f32[64]" = torch.ops.aten.sqrt.default(add_13);  add_13 = None
    reciprocal_6: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
    mul_18: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
    unsqueeze_48: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_12, -1);  convert_element_type_12 = None
    unsqueeze_49: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    unsqueeze_50: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_18, -1);  mul_18 = None
    unsqueeze_51: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    sub_6: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_49);  unsqueeze_49 = None
    mul_19: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
    unsqueeze_52: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1)
    unsqueeze_53: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_20: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_19, unsqueeze_53);  mul_19 = unsqueeze_53 = None
    unsqueeze_54: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1);  primals_21 = None
    unsqueeze_55: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_14: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_55);  mul_20 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    relu_5: "f32[4, 64, 56, 56]" = torch.ops.aten.relu.default(add_14);  add_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_7: "f32[4, 256, 56, 56]" = torch.ops.aten.convolution.default(relu_5, primals_22, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    convert_element_type_14: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_183, torch.float32)
    convert_element_type_15: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_184, torch.float32)
    add_15: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_15, 1e-05);  convert_element_type_15 = None
    sqrt_7: "f32[256]" = torch.ops.aten.sqrt.default(add_15);  add_15 = None
    reciprocal_7: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
    mul_21: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
    unsqueeze_56: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_14, -1);  convert_element_type_14 = None
    unsqueeze_57: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    unsqueeze_58: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_21, -1);  mul_21 = None
    unsqueeze_59: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    sub_7: "f32[4, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_57);  unsqueeze_57 = None
    mul_22: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
    unsqueeze_60: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_61: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_23: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_61);  mul_22 = unsqueeze_61 = None
    unsqueeze_62: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_63: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_16: "f32[4, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_63);  mul_23 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:160, code: out += identity
    add_17: "f32[4, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_16, relu_3);  add_16 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    relu_6: "f32[4, 256, 56, 56]" = torch.ops.aten.relu.default(add_17);  add_17 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_8: "f32[4, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_6, primals_25, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    convert_element_type_16: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_186, torch.float32)
    convert_element_type_17: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_187, torch.float32)
    add_18: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_17, 1e-05);  convert_element_type_17 = None
    sqrt_8: "f32[64]" = torch.ops.aten.sqrt.default(add_18);  add_18 = None
    reciprocal_8: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
    mul_24: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
    unsqueeze_64: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_16, -1);  convert_element_type_16 = None
    unsqueeze_65: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    unsqueeze_66: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_24, -1);  mul_24 = None
    unsqueeze_67: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    sub_8: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_65);  unsqueeze_65 = None
    mul_25: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
    unsqueeze_68: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1)
    unsqueeze_69: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_26: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_69);  mul_25 = unsqueeze_69 = None
    unsqueeze_70: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1);  primals_27 = None
    unsqueeze_71: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_19: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_26, unsqueeze_71);  mul_26 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    relu_7: "f32[4, 64, 56, 56]" = torch.ops.aten.relu.default(add_19);  add_19 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_9: "f32[4, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_7, primals_28, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    convert_element_type_18: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_189, torch.float32)
    convert_element_type_19: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_190, torch.float32)
    add_20: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_19, 1e-05);  convert_element_type_19 = None
    sqrt_9: "f32[64]" = torch.ops.aten.sqrt.default(add_20);  add_20 = None
    reciprocal_9: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
    mul_27: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
    unsqueeze_72: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_18, -1);  convert_element_type_18 = None
    unsqueeze_73: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    unsqueeze_74: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_27, -1);  mul_27 = None
    unsqueeze_75: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    sub_9: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_73);  unsqueeze_73 = None
    mul_28: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
    unsqueeze_76: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_77: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_29: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_77);  mul_28 = unsqueeze_77 = None
    unsqueeze_78: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_79: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_21: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_29, unsqueeze_79);  mul_29 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    relu_8: "f32[4, 64, 56, 56]" = torch.ops.aten.relu.default(add_21);  add_21 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_10: "f32[4, 256, 56, 56]" = torch.ops.aten.convolution.default(relu_8, primals_31, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    convert_element_type_20: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_192, torch.float32)
    convert_element_type_21: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_193, torch.float32)
    add_22: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_21, 1e-05);  convert_element_type_21 = None
    sqrt_10: "f32[256]" = torch.ops.aten.sqrt.default(add_22);  add_22 = None
    reciprocal_10: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
    mul_30: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
    unsqueeze_80: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_20, -1);  convert_element_type_20 = None
    unsqueeze_81: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    unsqueeze_82: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_30, -1);  mul_30 = None
    unsqueeze_83: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    sub_10: "f32[4, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_81);  unsqueeze_81 = None
    mul_31: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
    unsqueeze_84: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1)
    unsqueeze_85: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_32: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_31, unsqueeze_85);  mul_31 = unsqueeze_85 = None
    unsqueeze_86: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1);  primals_33 = None
    unsqueeze_87: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_23: "f32[4, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_32, unsqueeze_87);  mul_32 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:160, code: out += identity
    add_24: "f32[4, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_23, relu_6);  add_23 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    relu_9: "f32[4, 256, 56, 56]" = torch.ops.aten.relu.default(add_24);  add_24 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_11: "f32[4, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_9, primals_34, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    convert_element_type_22: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_195, torch.float32)
    convert_element_type_23: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_196, torch.float32)
    add_25: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_23, 1e-05);  convert_element_type_23 = None
    sqrt_11: "f32[128]" = torch.ops.aten.sqrt.default(add_25);  add_25 = None
    reciprocal_11: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
    mul_33: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
    unsqueeze_88: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_22, -1);  convert_element_type_22 = None
    unsqueeze_89: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    unsqueeze_90: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_33, -1);  mul_33 = None
    unsqueeze_91: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    sub_11: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_89);  unsqueeze_89 = None
    mul_34: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
    unsqueeze_92: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_93: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_35: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_34, unsqueeze_93);  mul_34 = unsqueeze_93 = None
    unsqueeze_94: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_95: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_26: "f32[4, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_95);  mul_35 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    relu_10: "f32[4, 128, 56, 56]" = torch.ops.aten.relu.default(add_26);  add_26 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_12: "f32[4, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_10, primals_37, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    convert_element_type_24: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_198, torch.float32)
    convert_element_type_25: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_199, torch.float32)
    add_27: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_25, 1e-05);  convert_element_type_25 = None
    sqrt_12: "f32[128]" = torch.ops.aten.sqrt.default(add_27);  add_27 = None
    reciprocal_12: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
    mul_36: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
    unsqueeze_96: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_24, -1);  convert_element_type_24 = None
    unsqueeze_97: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    unsqueeze_98: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_36, -1);  mul_36 = None
    unsqueeze_99: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    sub_12: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_97);  unsqueeze_97 = None
    mul_37: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
    unsqueeze_100: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1)
    unsqueeze_101: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_38: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_101);  mul_37 = unsqueeze_101 = None
    unsqueeze_102: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1);  primals_39 = None
    unsqueeze_103: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_28: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_38, unsqueeze_103);  mul_38 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    relu_11: "f32[4, 128, 28, 28]" = torch.ops.aten.relu.default(add_28);  add_28 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_13: "f32[4, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_11, primals_40, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    convert_element_type_26: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_201, torch.float32)
    convert_element_type_27: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_202, torch.float32)
    add_29: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_27, 1e-05);  convert_element_type_27 = None
    sqrt_13: "f32[512]" = torch.ops.aten.sqrt.default(add_29);  add_29 = None
    reciprocal_13: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
    mul_39: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
    unsqueeze_104: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_26, -1);  convert_element_type_26 = None
    unsqueeze_105: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    unsqueeze_106: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_39, -1);  mul_39 = None
    unsqueeze_107: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    sub_13: "f32[4, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_105);  unsqueeze_105 = None
    mul_40: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
    unsqueeze_108: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_109: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_41: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_40, unsqueeze_109);  mul_40 = unsqueeze_109 = None
    unsqueeze_110: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_111: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_30: "f32[4, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_111);  mul_41 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:158, code: identity = self.downsample(x)
    convolution_14: "f32[4, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_9, primals_43, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_28: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_204, torch.float32)
    convert_element_type_29: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_205, torch.float32)
    add_31: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_29, 1e-05);  convert_element_type_29 = None
    sqrt_14: "f32[512]" = torch.ops.aten.sqrt.default(add_31);  add_31 = None
    reciprocal_14: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
    mul_42: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
    unsqueeze_112: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_28, -1);  convert_element_type_28 = None
    unsqueeze_113: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    unsqueeze_114: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_42, -1);  mul_42 = None
    unsqueeze_115: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    sub_14: "f32[4, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_113);  unsqueeze_113 = None
    mul_43: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
    unsqueeze_116: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1)
    unsqueeze_117: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_44: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_117);  mul_43 = unsqueeze_117 = None
    unsqueeze_118: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1);  primals_45 = None
    unsqueeze_119: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_32: "f32[4, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_44, unsqueeze_119);  mul_44 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:160, code: out += identity
    add_33: "f32[4, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_30, add_32);  add_30 = add_32 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    relu_12: "f32[4, 512, 28, 28]" = torch.ops.aten.relu.default(add_33);  add_33 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_15: "f32[4, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_12, primals_46, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    convert_element_type_30: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_207, torch.float32)
    convert_element_type_31: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_208, torch.float32)
    add_34: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_31, 1e-05);  convert_element_type_31 = None
    sqrt_15: "f32[128]" = torch.ops.aten.sqrt.default(add_34);  add_34 = None
    reciprocal_15: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
    mul_45: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
    unsqueeze_120: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_30, -1);  convert_element_type_30 = None
    unsqueeze_121: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    unsqueeze_122: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_45, -1);  mul_45 = None
    unsqueeze_123: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    sub_15: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_121);  unsqueeze_121 = None
    mul_46: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
    unsqueeze_124: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_125: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_47: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_46, unsqueeze_125);  mul_46 = unsqueeze_125 = None
    unsqueeze_126: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_127: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_35: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_47, unsqueeze_127);  mul_47 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    relu_13: "f32[4, 128, 28, 28]" = torch.ops.aten.relu.default(add_35);  add_35 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_16: "f32[4, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_13, primals_49, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    convert_element_type_32: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_210, torch.float32)
    convert_element_type_33: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_211, torch.float32)
    add_36: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_33, 1e-05);  convert_element_type_33 = None
    sqrt_16: "f32[128]" = torch.ops.aten.sqrt.default(add_36);  add_36 = None
    reciprocal_16: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
    mul_48: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
    unsqueeze_128: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_32, -1);  convert_element_type_32 = None
    unsqueeze_129: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    unsqueeze_130: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_48, -1);  mul_48 = None
    unsqueeze_131: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    sub_16: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_129);  unsqueeze_129 = None
    mul_49: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
    unsqueeze_132: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1)
    unsqueeze_133: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_50: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_133);  mul_49 = unsqueeze_133 = None
    unsqueeze_134: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1);  primals_51 = None
    unsqueeze_135: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_37: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_50, unsqueeze_135);  mul_50 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    relu_14: "f32[4, 128, 28, 28]" = torch.ops.aten.relu.default(add_37);  add_37 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_17: "f32[4, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_14, primals_52, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    convert_element_type_34: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_213, torch.float32)
    convert_element_type_35: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_214, torch.float32)
    add_38: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_35, 1e-05);  convert_element_type_35 = None
    sqrt_17: "f32[512]" = torch.ops.aten.sqrt.default(add_38);  add_38 = None
    reciprocal_17: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
    mul_51: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
    unsqueeze_136: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_34, -1);  convert_element_type_34 = None
    unsqueeze_137: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    unsqueeze_138: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_51, -1);  mul_51 = None
    unsqueeze_139: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    sub_17: "f32[4, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_137);  unsqueeze_137 = None
    mul_52: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_139);  sub_17 = unsqueeze_139 = None
    unsqueeze_140: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_141: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_53: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_52, unsqueeze_141);  mul_52 = unsqueeze_141 = None
    unsqueeze_142: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_143: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_39: "f32[4, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_53, unsqueeze_143);  mul_53 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:160, code: out += identity
    add_40: "f32[4, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_39, relu_12);  add_39 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    relu_15: "f32[4, 512, 28, 28]" = torch.ops.aten.relu.default(add_40);  add_40 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_18: "f32[4, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_15, primals_55, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    convert_element_type_36: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_216, torch.float32)
    convert_element_type_37: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_217, torch.float32)
    add_41: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_37, 1e-05);  convert_element_type_37 = None
    sqrt_18: "f32[128]" = torch.ops.aten.sqrt.default(add_41);  add_41 = None
    reciprocal_18: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
    mul_54: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
    unsqueeze_144: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_36, -1);  convert_element_type_36 = None
    unsqueeze_145: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    unsqueeze_146: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_54, -1);  mul_54 = None
    unsqueeze_147: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    sub_18: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_145);  unsqueeze_145 = None
    mul_55: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_147);  sub_18 = unsqueeze_147 = None
    unsqueeze_148: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1)
    unsqueeze_149: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_56: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_55, unsqueeze_149);  mul_55 = unsqueeze_149 = None
    unsqueeze_150: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1);  primals_57 = None
    unsqueeze_151: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_42: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_56, unsqueeze_151);  mul_56 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    relu_16: "f32[4, 128, 28, 28]" = torch.ops.aten.relu.default(add_42);  add_42 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_19: "f32[4, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_16, primals_58, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    convert_element_type_38: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_219, torch.float32)
    convert_element_type_39: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_220, torch.float32)
    add_43: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_39, 1e-05);  convert_element_type_39 = None
    sqrt_19: "f32[128]" = torch.ops.aten.sqrt.default(add_43);  add_43 = None
    reciprocal_19: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
    mul_57: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
    unsqueeze_152: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_38, -1);  convert_element_type_38 = None
    unsqueeze_153: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    unsqueeze_154: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_57, -1);  mul_57 = None
    unsqueeze_155: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    sub_19: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_153);  unsqueeze_153 = None
    mul_58: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_155);  sub_19 = unsqueeze_155 = None
    unsqueeze_156: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_157: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_59: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_157);  mul_58 = unsqueeze_157 = None
    unsqueeze_158: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_159: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_44: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_59, unsqueeze_159);  mul_59 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    relu_17: "f32[4, 128, 28, 28]" = torch.ops.aten.relu.default(add_44);  add_44 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_20: "f32[4, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_17, primals_61, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    convert_element_type_40: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_222, torch.float32)
    convert_element_type_41: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_223, torch.float32)
    add_45: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_41, 1e-05);  convert_element_type_41 = None
    sqrt_20: "f32[512]" = torch.ops.aten.sqrt.default(add_45);  add_45 = None
    reciprocal_20: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
    mul_60: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
    unsqueeze_160: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_40, -1);  convert_element_type_40 = None
    unsqueeze_161: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    unsqueeze_162: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_60, -1);  mul_60 = None
    unsqueeze_163: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    sub_20: "f32[4, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_161);  unsqueeze_161 = None
    mul_61: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_163);  sub_20 = unsqueeze_163 = None
    unsqueeze_164: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1)
    unsqueeze_165: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_62: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_61, unsqueeze_165);  mul_61 = unsqueeze_165 = None
    unsqueeze_166: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1);  primals_63 = None
    unsqueeze_167: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_46: "f32[4, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_167);  mul_62 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:160, code: out += identity
    add_47: "f32[4, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_46, relu_15);  add_46 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    relu_18: "f32[4, 512, 28, 28]" = torch.ops.aten.relu.default(add_47);  add_47 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_21: "f32[4, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_18, primals_64, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    convert_element_type_42: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_225, torch.float32)
    convert_element_type_43: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_226, torch.float32)
    add_48: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_43, 1e-05);  convert_element_type_43 = None
    sqrt_21: "f32[128]" = torch.ops.aten.sqrt.default(add_48);  add_48 = None
    reciprocal_21: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
    mul_63: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
    unsqueeze_168: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_42, -1);  convert_element_type_42 = None
    unsqueeze_169: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    unsqueeze_170: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_63, -1);  mul_63 = None
    unsqueeze_171: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    sub_21: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_169);  unsqueeze_169 = None
    mul_64: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_21, unsqueeze_171);  sub_21 = unsqueeze_171 = None
    unsqueeze_172: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
    unsqueeze_173: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_65: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_64, unsqueeze_173);  mul_64 = unsqueeze_173 = None
    unsqueeze_174: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
    unsqueeze_175: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_49: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_65, unsqueeze_175);  mul_65 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    relu_19: "f32[4, 128, 28, 28]" = torch.ops.aten.relu.default(add_49);  add_49 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_22: "f32[4, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_19, primals_67, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    convert_element_type_44: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_228, torch.float32)
    convert_element_type_45: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_229, torch.float32)
    add_50: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_45, 1e-05);  convert_element_type_45 = None
    sqrt_22: "f32[128]" = torch.ops.aten.sqrt.default(add_50);  add_50 = None
    reciprocal_22: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
    mul_66: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
    unsqueeze_176: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_44, -1);  convert_element_type_44 = None
    unsqueeze_177: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    unsqueeze_178: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_66, -1);  mul_66 = None
    unsqueeze_179: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    sub_22: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_177);  unsqueeze_177 = None
    mul_67: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_179);  sub_22 = unsqueeze_179 = None
    unsqueeze_180: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1)
    unsqueeze_181: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_68: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_67, unsqueeze_181);  mul_67 = unsqueeze_181 = None
    unsqueeze_182: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1);  primals_69 = None
    unsqueeze_183: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_51: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_68, unsqueeze_183);  mul_68 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    relu_20: "f32[4, 128, 28, 28]" = torch.ops.aten.relu.default(add_51);  add_51 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_23: "f32[4, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_20, primals_70, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    convert_element_type_46: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_231, torch.float32)
    convert_element_type_47: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_232, torch.float32)
    add_52: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_47, 1e-05);  convert_element_type_47 = None
    sqrt_23: "f32[512]" = torch.ops.aten.sqrt.default(add_52);  add_52 = None
    reciprocal_23: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
    mul_69: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
    unsqueeze_184: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_46, -1);  convert_element_type_46 = None
    unsqueeze_185: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    unsqueeze_186: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_69, -1);  mul_69 = None
    unsqueeze_187: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    sub_23: "f32[4, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_185);  unsqueeze_185 = None
    mul_70: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_187);  sub_23 = unsqueeze_187 = None
    unsqueeze_188: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_189: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_71: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_189);  mul_70 = unsqueeze_189 = None
    unsqueeze_190: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_191: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_53: "f32[4, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_71, unsqueeze_191);  mul_71 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:160, code: out += identity
    add_54: "f32[4, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_53, relu_18);  add_53 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    relu_21: "f32[4, 512, 28, 28]" = torch.ops.aten.relu.default(add_54);  add_54 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_24: "f32[4, 256, 28, 28]" = torch.ops.aten.convolution.default(relu_21, primals_73, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    convert_element_type_48: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_234, torch.float32)
    convert_element_type_49: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_235, torch.float32)
    add_55: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_49, 1e-05);  convert_element_type_49 = None
    sqrt_24: "f32[256]" = torch.ops.aten.sqrt.default(add_55);  add_55 = None
    reciprocal_24: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
    mul_72: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
    unsqueeze_192: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_48, -1);  convert_element_type_48 = None
    unsqueeze_193: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    unsqueeze_194: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_72, -1);  mul_72 = None
    unsqueeze_195: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    sub_24: "f32[4, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_193);  unsqueeze_193 = None
    mul_73: "f32[4, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_195);  sub_24 = unsqueeze_195 = None
    unsqueeze_196: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1)
    unsqueeze_197: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_74: "f32[4, 256, 28, 28]" = torch.ops.aten.mul.Tensor(mul_73, unsqueeze_197);  mul_73 = unsqueeze_197 = None
    unsqueeze_198: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1);  primals_75 = None
    unsqueeze_199: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_56: "f32[4, 256, 28, 28]" = torch.ops.aten.add.Tensor(mul_74, unsqueeze_199);  mul_74 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    relu_22: "f32[4, 256, 28, 28]" = torch.ops.aten.relu.default(add_56);  add_56 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_25: "f32[4, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_22, primals_76, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    convert_element_type_50: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_237, torch.float32)
    convert_element_type_51: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_238, torch.float32)
    add_57: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_51, 1e-05);  convert_element_type_51 = None
    sqrt_25: "f32[256]" = torch.ops.aten.sqrt.default(add_57);  add_57 = None
    reciprocal_25: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
    mul_75: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
    unsqueeze_200: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_50, -1);  convert_element_type_50 = None
    unsqueeze_201: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    unsqueeze_202: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_75, -1);  mul_75 = None
    unsqueeze_203: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    sub_25: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_201);  unsqueeze_201 = None
    mul_76: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_203);  sub_25 = unsqueeze_203 = None
    unsqueeze_204: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
    unsqueeze_205: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_77: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_76, unsqueeze_205);  mul_76 = unsqueeze_205 = None
    unsqueeze_206: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
    unsqueeze_207: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_58: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_77, unsqueeze_207);  mul_77 = unsqueeze_207 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    relu_23: "f32[4, 256, 14, 14]" = torch.ops.aten.relu.default(add_58);  add_58 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_26: "f32[4, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_23, primals_79, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    convert_element_type_52: "f32[1024]" = torch.ops.prims.convert_element_type.default(primals_240, torch.float32)
    convert_element_type_53: "f32[1024]" = torch.ops.prims.convert_element_type.default(primals_241, torch.float32)
    add_59: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_53, 1e-05);  convert_element_type_53 = None
    sqrt_26: "f32[1024]" = torch.ops.aten.sqrt.default(add_59);  add_59 = None
    reciprocal_26: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
    mul_78: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
    unsqueeze_208: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_52, -1);  convert_element_type_52 = None
    unsqueeze_209: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    unsqueeze_210: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_78, -1);  mul_78 = None
    unsqueeze_211: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    sub_26: "f32[4, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_209);  unsqueeze_209 = None
    mul_79: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_211);  sub_26 = unsqueeze_211 = None
    unsqueeze_212: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1)
    unsqueeze_213: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_80: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_79, unsqueeze_213);  mul_79 = unsqueeze_213 = None
    unsqueeze_214: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1);  primals_81 = None
    unsqueeze_215: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_60: "f32[4, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_80, unsqueeze_215);  mul_80 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:158, code: identity = self.downsample(x)
    convolution_27: "f32[4, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_21, primals_82, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_54: "f32[1024]" = torch.ops.prims.convert_element_type.default(primals_243, torch.float32)
    convert_element_type_55: "f32[1024]" = torch.ops.prims.convert_element_type.default(primals_244, torch.float32)
    add_61: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_55, 1e-05);  convert_element_type_55 = None
    sqrt_27: "f32[1024]" = torch.ops.aten.sqrt.default(add_61);  add_61 = None
    reciprocal_27: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
    mul_81: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
    unsqueeze_216: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_54, -1);  convert_element_type_54 = None
    unsqueeze_217: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    unsqueeze_218: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_81, -1);  mul_81 = None
    unsqueeze_219: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    sub_27: "f32[4, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_217);  unsqueeze_217 = None
    mul_82: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_219);  sub_27 = unsqueeze_219 = None
    unsqueeze_220: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1)
    unsqueeze_221: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_83: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_82, unsqueeze_221);  mul_82 = unsqueeze_221 = None
    unsqueeze_222: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
    unsqueeze_223: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_62: "f32[4, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_223);  mul_83 = unsqueeze_223 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:160, code: out += identity
    add_63: "f32[4, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_60, add_62);  add_60 = add_62 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    relu_24: "f32[4, 1024, 14, 14]" = torch.ops.aten.relu.default(add_63);  add_63 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_28: "f32[4, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_24, primals_85, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    convert_element_type_56: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_246, torch.float32)
    convert_element_type_57: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_247, torch.float32)
    add_64: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_57, 1e-05);  convert_element_type_57 = None
    sqrt_28: "f32[256]" = torch.ops.aten.sqrt.default(add_64);  add_64 = None
    reciprocal_28: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
    mul_84: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
    unsqueeze_224: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_56, -1);  convert_element_type_56 = None
    unsqueeze_225: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    unsqueeze_226: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_84, -1);  mul_84 = None
    unsqueeze_227: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    sub_28: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_225);  unsqueeze_225 = None
    mul_85: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_227);  sub_28 = unsqueeze_227 = None
    unsqueeze_228: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1)
    unsqueeze_229: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_86: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_85, unsqueeze_229);  mul_85 = unsqueeze_229 = None
    unsqueeze_230: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1);  primals_87 = None
    unsqueeze_231: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_65: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_86, unsqueeze_231);  mul_86 = unsqueeze_231 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    relu_25: "f32[4, 256, 14, 14]" = torch.ops.aten.relu.default(add_65);  add_65 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_29: "f32[4, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_25, primals_88, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    convert_element_type_58: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_249, torch.float32)
    convert_element_type_59: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_250, torch.float32)
    add_66: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_59, 1e-05);  convert_element_type_59 = None
    sqrt_29: "f32[256]" = torch.ops.aten.sqrt.default(add_66);  add_66 = None
    reciprocal_29: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
    mul_87: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
    unsqueeze_232: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_58, -1);  convert_element_type_58 = None
    unsqueeze_233: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    unsqueeze_234: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_87, -1);  mul_87 = None
    unsqueeze_235: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    sub_29: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_233);  unsqueeze_233 = None
    mul_88: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_235);  sub_29 = unsqueeze_235 = None
    unsqueeze_236: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1)
    unsqueeze_237: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_89: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_88, unsqueeze_237);  mul_88 = unsqueeze_237 = None
    unsqueeze_238: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1);  primals_90 = None
    unsqueeze_239: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_67: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_89, unsqueeze_239);  mul_89 = unsqueeze_239 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    relu_26: "f32[4, 256, 14, 14]" = torch.ops.aten.relu.default(add_67);  add_67 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_30: "f32[4, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_26, primals_91, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    convert_element_type_60: "f32[1024]" = torch.ops.prims.convert_element_type.default(primals_252, torch.float32)
    convert_element_type_61: "f32[1024]" = torch.ops.prims.convert_element_type.default(primals_253, torch.float32)
    add_68: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_61, 1e-05);  convert_element_type_61 = None
    sqrt_30: "f32[1024]" = torch.ops.aten.sqrt.default(add_68);  add_68 = None
    reciprocal_30: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
    mul_90: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
    unsqueeze_240: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_60, -1);  convert_element_type_60 = None
    unsqueeze_241: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    unsqueeze_242: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_90, -1);  mul_90 = None
    unsqueeze_243: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    sub_30: "f32[4, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_241);  unsqueeze_241 = None
    mul_91: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_243);  sub_30 = unsqueeze_243 = None
    unsqueeze_244: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1)
    unsqueeze_245: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_92: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_245);  mul_91 = unsqueeze_245 = None
    unsqueeze_246: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_93, -1);  primals_93 = None
    unsqueeze_247: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_69: "f32[4, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_92, unsqueeze_247);  mul_92 = unsqueeze_247 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:160, code: out += identity
    add_70: "f32[4, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_69, relu_24);  add_69 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    relu_27: "f32[4, 1024, 14, 14]" = torch.ops.aten.relu.default(add_70);  add_70 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_31: "f32[4, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_27, primals_94, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    convert_element_type_62: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_255, torch.float32)
    convert_element_type_63: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_256, torch.float32)
    add_71: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_63, 1e-05);  convert_element_type_63 = None
    sqrt_31: "f32[256]" = torch.ops.aten.sqrt.default(add_71);  add_71 = None
    reciprocal_31: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
    mul_93: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
    unsqueeze_248: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_62, -1);  convert_element_type_62 = None
    unsqueeze_249: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    unsqueeze_250: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_93, -1);  mul_93 = None
    unsqueeze_251: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    sub_31: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_249);  unsqueeze_249 = None
    mul_94: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_251);  sub_31 = unsqueeze_251 = None
    unsqueeze_252: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1)
    unsqueeze_253: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_95: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_94, unsqueeze_253);  mul_94 = unsqueeze_253 = None
    unsqueeze_254: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1);  primals_96 = None
    unsqueeze_255: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_72: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_95, unsqueeze_255);  mul_95 = unsqueeze_255 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    relu_28: "f32[4, 256, 14, 14]" = torch.ops.aten.relu.default(add_72);  add_72 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_32: "f32[4, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_28, primals_97, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    convert_element_type_64: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_258, torch.float32)
    convert_element_type_65: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_259, torch.float32)
    add_73: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_65, 1e-05);  convert_element_type_65 = None
    sqrt_32: "f32[256]" = torch.ops.aten.sqrt.default(add_73);  add_73 = None
    reciprocal_32: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
    mul_96: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
    unsqueeze_256: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_64, -1);  convert_element_type_64 = None
    unsqueeze_257: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    unsqueeze_258: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_96, -1);  mul_96 = None
    unsqueeze_259: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    sub_32: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_257);  unsqueeze_257 = None
    mul_97: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_259);  sub_32 = unsqueeze_259 = None
    unsqueeze_260: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1)
    unsqueeze_261: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
    mul_98: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_97, unsqueeze_261);  mul_97 = unsqueeze_261 = None
    unsqueeze_262: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1);  primals_99 = None
    unsqueeze_263: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
    add_74: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_98, unsqueeze_263);  mul_98 = unsqueeze_263 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    relu_29: "f32[4, 256, 14, 14]" = torch.ops.aten.relu.default(add_74);  add_74 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_33: "f32[4, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_29, primals_100, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    convert_element_type_66: "f32[1024]" = torch.ops.prims.convert_element_type.default(primals_261, torch.float32)
    convert_element_type_67: "f32[1024]" = torch.ops.prims.convert_element_type.default(primals_262, torch.float32)
    add_75: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_67, 1e-05);  convert_element_type_67 = None
    sqrt_33: "f32[1024]" = torch.ops.aten.sqrt.default(add_75);  add_75 = None
    reciprocal_33: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
    mul_99: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
    unsqueeze_264: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_66, -1);  convert_element_type_66 = None
    unsqueeze_265: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
    unsqueeze_266: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_99, -1);  mul_99 = None
    unsqueeze_267: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
    sub_33: "f32[4, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_265);  unsqueeze_265 = None
    mul_100: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_33, unsqueeze_267);  sub_33 = unsqueeze_267 = None
    unsqueeze_268: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_101, -1)
    unsqueeze_269: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
    mul_101: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_100, unsqueeze_269);  mul_100 = unsqueeze_269 = None
    unsqueeze_270: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1);  primals_102 = None
    unsqueeze_271: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
    add_76: "f32[4, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_101, unsqueeze_271);  mul_101 = unsqueeze_271 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:160, code: out += identity
    add_77: "f32[4, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_76, relu_27);  add_76 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    relu_30: "f32[4, 1024, 14, 14]" = torch.ops.aten.relu.default(add_77);  add_77 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_34: "f32[4, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_30, primals_103, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    convert_element_type_68: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_264, torch.float32)
    convert_element_type_69: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_265, torch.float32)
    add_78: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_69, 1e-05);  convert_element_type_69 = None
    sqrt_34: "f32[256]" = torch.ops.aten.sqrt.default(add_78);  add_78 = None
    reciprocal_34: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
    mul_102: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
    unsqueeze_272: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_68, -1);  convert_element_type_68 = None
    unsqueeze_273: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
    unsqueeze_274: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_102, -1);  mul_102 = None
    unsqueeze_275: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
    sub_34: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_273);  unsqueeze_273 = None
    mul_103: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_275);  sub_34 = unsqueeze_275 = None
    unsqueeze_276: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_104, -1)
    unsqueeze_277: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
    mul_104: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_103, unsqueeze_277);  mul_103 = unsqueeze_277 = None
    unsqueeze_278: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_105, -1);  primals_105 = None
    unsqueeze_279: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
    add_79: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_279);  mul_104 = unsqueeze_279 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    relu_31: "f32[4, 256, 14, 14]" = torch.ops.aten.relu.default(add_79);  add_79 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_35: "f32[4, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_31, primals_106, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    convert_element_type_70: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_267, torch.float32)
    convert_element_type_71: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_268, torch.float32)
    add_80: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_71, 1e-05);  convert_element_type_71 = None
    sqrt_35: "f32[256]" = torch.ops.aten.sqrt.default(add_80);  add_80 = None
    reciprocal_35: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
    mul_105: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
    unsqueeze_280: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_70, -1);  convert_element_type_70 = None
    unsqueeze_281: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
    unsqueeze_282: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_105, -1);  mul_105 = None
    unsqueeze_283: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
    sub_35: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_281);  unsqueeze_281 = None
    mul_106: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_283);  sub_35 = unsqueeze_283 = None
    unsqueeze_284: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_107, -1)
    unsqueeze_285: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
    mul_107: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_106, unsqueeze_285);  mul_106 = unsqueeze_285 = None
    unsqueeze_286: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1);  primals_108 = None
    unsqueeze_287: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
    add_81: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_107, unsqueeze_287);  mul_107 = unsqueeze_287 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    relu_32: "f32[4, 256, 14, 14]" = torch.ops.aten.relu.default(add_81);  add_81 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_36: "f32[4, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_32, primals_109, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    convert_element_type_72: "f32[1024]" = torch.ops.prims.convert_element_type.default(primals_270, torch.float32)
    convert_element_type_73: "f32[1024]" = torch.ops.prims.convert_element_type.default(primals_271, torch.float32)
    add_82: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_73, 1e-05);  convert_element_type_73 = None
    sqrt_36: "f32[1024]" = torch.ops.aten.sqrt.default(add_82);  add_82 = None
    reciprocal_36: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
    mul_108: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
    unsqueeze_288: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_72, -1);  convert_element_type_72 = None
    unsqueeze_289: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
    unsqueeze_290: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_108, -1);  mul_108 = None
    unsqueeze_291: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
    sub_36: "f32[4, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_289);  unsqueeze_289 = None
    mul_109: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_291);  sub_36 = unsqueeze_291 = None
    unsqueeze_292: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_110, -1)
    unsqueeze_293: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
    mul_110: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_109, unsqueeze_293);  mul_109 = unsqueeze_293 = None
    unsqueeze_294: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_111, -1);  primals_111 = None
    unsqueeze_295: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
    add_83: "f32[4, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_110, unsqueeze_295);  mul_110 = unsqueeze_295 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:160, code: out += identity
    add_84: "f32[4, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_83, relu_30);  add_83 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    relu_33: "f32[4, 1024, 14, 14]" = torch.ops.aten.relu.default(add_84);  add_84 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_37: "f32[4, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_33, primals_112, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    convert_element_type_74: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_273, torch.float32)
    convert_element_type_75: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_274, torch.float32)
    add_85: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_75, 1e-05);  convert_element_type_75 = None
    sqrt_37: "f32[256]" = torch.ops.aten.sqrt.default(add_85);  add_85 = None
    reciprocal_37: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
    mul_111: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
    unsqueeze_296: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_74, -1);  convert_element_type_74 = None
    unsqueeze_297: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
    unsqueeze_298: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_111, -1);  mul_111 = None
    unsqueeze_299: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
    sub_37: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_297);  unsqueeze_297 = None
    mul_112: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_299);  sub_37 = unsqueeze_299 = None
    unsqueeze_300: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_113, -1)
    unsqueeze_301: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
    mul_113: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_301);  mul_112 = unsqueeze_301 = None
    unsqueeze_302: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_114, -1);  primals_114 = None
    unsqueeze_303: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
    add_86: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_113, unsqueeze_303);  mul_113 = unsqueeze_303 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    relu_34: "f32[4, 256, 14, 14]" = torch.ops.aten.relu.default(add_86);  add_86 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_38: "f32[4, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_34, primals_115, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    convert_element_type_76: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_276, torch.float32)
    convert_element_type_77: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_277, torch.float32)
    add_87: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_77, 1e-05);  convert_element_type_77 = None
    sqrt_38: "f32[256]" = torch.ops.aten.sqrt.default(add_87);  add_87 = None
    reciprocal_38: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
    mul_114: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
    unsqueeze_304: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_76, -1);  convert_element_type_76 = None
    unsqueeze_305: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
    unsqueeze_306: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_114, -1);  mul_114 = None
    unsqueeze_307: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
    sub_38: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_305);  unsqueeze_305 = None
    mul_115: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_307);  sub_38 = unsqueeze_307 = None
    unsqueeze_308: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_116, -1)
    unsqueeze_309: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
    mul_116: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_115, unsqueeze_309);  mul_115 = unsqueeze_309 = None
    unsqueeze_310: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_117, -1);  primals_117 = None
    unsqueeze_311: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
    add_88: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_116, unsqueeze_311);  mul_116 = unsqueeze_311 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    relu_35: "f32[4, 256, 14, 14]" = torch.ops.aten.relu.default(add_88);  add_88 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_39: "f32[4, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_35, primals_118, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    convert_element_type_78: "f32[1024]" = torch.ops.prims.convert_element_type.default(primals_279, torch.float32)
    convert_element_type_79: "f32[1024]" = torch.ops.prims.convert_element_type.default(primals_280, torch.float32)
    add_89: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_79, 1e-05);  convert_element_type_79 = None
    sqrt_39: "f32[1024]" = torch.ops.aten.sqrt.default(add_89);  add_89 = None
    reciprocal_39: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
    mul_117: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
    unsqueeze_312: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_78, -1);  convert_element_type_78 = None
    unsqueeze_313: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
    unsqueeze_314: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_117, -1);  mul_117 = None
    unsqueeze_315: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
    sub_39: "f32[4, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_313);  unsqueeze_313 = None
    mul_118: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_315);  sub_39 = unsqueeze_315 = None
    unsqueeze_316: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_119, -1)
    unsqueeze_317: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
    mul_119: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_118, unsqueeze_317);  mul_118 = unsqueeze_317 = None
    unsqueeze_318: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_120, -1);  primals_120 = None
    unsqueeze_319: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
    add_90: "f32[4, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_119, unsqueeze_319);  mul_119 = unsqueeze_319 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:160, code: out += identity
    add_91: "f32[4, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_90, relu_33);  add_90 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    relu_36: "f32[4, 1024, 14, 14]" = torch.ops.aten.relu.default(add_91);  add_91 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_40: "f32[4, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_36, primals_121, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    convert_element_type_80: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_282, torch.float32)
    convert_element_type_81: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_283, torch.float32)
    add_92: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_81, 1e-05);  convert_element_type_81 = None
    sqrt_40: "f32[256]" = torch.ops.aten.sqrt.default(add_92);  add_92 = None
    reciprocal_40: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
    mul_120: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
    unsqueeze_320: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_80, -1);  convert_element_type_80 = None
    unsqueeze_321: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
    unsqueeze_322: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_120, -1);  mul_120 = None
    unsqueeze_323: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
    sub_40: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_321);  unsqueeze_321 = None
    mul_121: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_323);  sub_40 = unsqueeze_323 = None
    unsqueeze_324: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_122, -1)
    unsqueeze_325: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
    mul_122: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_121, unsqueeze_325);  mul_121 = unsqueeze_325 = None
    unsqueeze_326: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_123, -1);  primals_123 = None
    unsqueeze_327: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
    add_93: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_122, unsqueeze_327);  mul_122 = unsqueeze_327 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    relu_37: "f32[4, 256, 14, 14]" = torch.ops.aten.relu.default(add_93);  add_93 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_41: "f32[4, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_37, primals_124, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    convert_element_type_82: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_285, torch.float32)
    convert_element_type_83: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_286, torch.float32)
    add_94: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_83, 1e-05);  convert_element_type_83 = None
    sqrt_41: "f32[256]" = torch.ops.aten.sqrt.default(add_94);  add_94 = None
    reciprocal_41: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
    mul_123: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
    unsqueeze_328: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_82, -1);  convert_element_type_82 = None
    unsqueeze_329: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
    unsqueeze_330: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_123, -1);  mul_123 = None
    unsqueeze_331: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
    sub_41: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_329);  unsqueeze_329 = None
    mul_124: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_331);  sub_41 = unsqueeze_331 = None
    unsqueeze_332: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_125, -1)
    unsqueeze_333: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
    mul_125: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_124, unsqueeze_333);  mul_124 = unsqueeze_333 = None
    unsqueeze_334: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_126, -1);  primals_126 = None
    unsqueeze_335: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
    add_95: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_335);  mul_125 = unsqueeze_335 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    relu_38: "f32[4, 256, 14, 14]" = torch.ops.aten.relu.default(add_95);  add_95 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_42: "f32[4, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_38, primals_127, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    convert_element_type_84: "f32[1024]" = torch.ops.prims.convert_element_type.default(primals_288, torch.float32)
    convert_element_type_85: "f32[1024]" = torch.ops.prims.convert_element_type.default(primals_289, torch.float32)
    add_96: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_85, 1e-05);  convert_element_type_85 = None
    sqrt_42: "f32[1024]" = torch.ops.aten.sqrt.default(add_96);  add_96 = None
    reciprocal_42: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
    mul_126: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
    unsqueeze_336: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_84, -1);  convert_element_type_84 = None
    unsqueeze_337: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
    unsqueeze_338: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_126, -1);  mul_126 = None
    unsqueeze_339: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
    sub_42: "f32[4, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_337);  unsqueeze_337 = None
    mul_127: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_339);  sub_42 = unsqueeze_339 = None
    unsqueeze_340: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_128, -1)
    unsqueeze_341: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
    mul_128: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_127, unsqueeze_341);  mul_127 = unsqueeze_341 = None
    unsqueeze_342: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_129, -1);  primals_129 = None
    unsqueeze_343: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
    add_97: "f32[4, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_128, unsqueeze_343);  mul_128 = unsqueeze_343 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:160, code: out += identity
    add_98: "f32[4, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_97, relu_36);  add_97 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    relu_39: "f32[4, 1024, 14, 14]" = torch.ops.aten.relu.default(add_98);  add_98 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_43: "f32[4, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_39, primals_130, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    convert_element_type_86: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_291, torch.float32)
    convert_element_type_87: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_292, torch.float32)
    add_99: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_87, 1e-05);  convert_element_type_87 = None
    sqrt_43: "f32[512]" = torch.ops.aten.sqrt.default(add_99);  add_99 = None
    reciprocal_43: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
    mul_129: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
    unsqueeze_344: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_86, -1);  convert_element_type_86 = None
    unsqueeze_345: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
    unsqueeze_346: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_129, -1);  mul_129 = None
    unsqueeze_347: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
    sub_43: "f32[4, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_345);  unsqueeze_345 = None
    mul_130: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_347);  sub_43 = unsqueeze_347 = None
    unsqueeze_348: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_131, -1)
    unsqueeze_349: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
    mul_131: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_130, unsqueeze_349);  mul_130 = unsqueeze_349 = None
    unsqueeze_350: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_132, -1);  primals_132 = None
    unsqueeze_351: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
    add_100: "f32[4, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_131, unsqueeze_351);  mul_131 = unsqueeze_351 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    relu_40: "f32[4, 512, 14, 14]" = torch.ops.aten.relu.default(add_100);  add_100 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_44: "f32[4, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_40, primals_133, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    convert_element_type_88: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_294, torch.float32)
    convert_element_type_89: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_295, torch.float32)
    add_101: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_89, 1e-05);  convert_element_type_89 = None
    sqrt_44: "f32[512]" = torch.ops.aten.sqrt.default(add_101);  add_101 = None
    reciprocal_44: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
    mul_132: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
    unsqueeze_352: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_88, -1);  convert_element_type_88 = None
    unsqueeze_353: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
    unsqueeze_354: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_132, -1);  mul_132 = None
    unsqueeze_355: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
    sub_44: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_353);  unsqueeze_353 = None
    mul_133: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_355);  sub_44 = unsqueeze_355 = None
    unsqueeze_356: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_134, -1)
    unsqueeze_357: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
    mul_134: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_357);  mul_133 = unsqueeze_357 = None
    unsqueeze_358: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_135, -1);  primals_135 = None
    unsqueeze_359: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
    add_102: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_134, unsqueeze_359);  mul_134 = unsqueeze_359 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    relu_41: "f32[4, 512, 7, 7]" = torch.ops.aten.relu.default(add_102);  add_102 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_45: "f32[4, 2048, 7, 7]" = torch.ops.aten.convolution.default(relu_41, primals_136, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    convert_element_type_90: "f32[2048]" = torch.ops.prims.convert_element_type.default(primals_297, torch.float32)
    convert_element_type_91: "f32[2048]" = torch.ops.prims.convert_element_type.default(primals_298, torch.float32)
    add_103: "f32[2048]" = torch.ops.aten.add.Tensor(convert_element_type_91, 1e-05);  convert_element_type_91 = None
    sqrt_45: "f32[2048]" = torch.ops.aten.sqrt.default(add_103);  add_103 = None
    reciprocal_45: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
    mul_135: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
    unsqueeze_360: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_90, -1);  convert_element_type_90 = None
    unsqueeze_361: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
    unsqueeze_362: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_135, -1);  mul_135 = None
    unsqueeze_363: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
    sub_45: "f32[4, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_361);  unsqueeze_361 = None
    mul_136: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_363);  sub_45 = unsqueeze_363 = None
    unsqueeze_364: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_137, -1)
    unsqueeze_365: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
    mul_137: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_365);  mul_136 = unsqueeze_365 = None
    unsqueeze_366: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_138, -1);  primals_138 = None
    unsqueeze_367: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
    add_104: "f32[4, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_137, unsqueeze_367);  mul_137 = unsqueeze_367 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:158, code: identity = self.downsample(x)
    convolution_46: "f32[4, 2048, 7, 7]" = torch.ops.aten.convolution.default(relu_39, primals_139, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_92: "f32[2048]" = torch.ops.prims.convert_element_type.default(primals_300, torch.float32)
    convert_element_type_93: "f32[2048]" = torch.ops.prims.convert_element_type.default(primals_301, torch.float32)
    add_105: "f32[2048]" = torch.ops.aten.add.Tensor(convert_element_type_93, 1e-05);  convert_element_type_93 = None
    sqrt_46: "f32[2048]" = torch.ops.aten.sqrt.default(add_105);  add_105 = None
    reciprocal_46: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
    mul_138: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
    unsqueeze_368: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_92, -1);  convert_element_type_92 = None
    unsqueeze_369: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
    unsqueeze_370: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_138, -1);  mul_138 = None
    unsqueeze_371: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
    sub_46: "f32[4, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_369);  unsqueeze_369 = None
    mul_139: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_371);  sub_46 = unsqueeze_371 = None
    unsqueeze_372: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_140, -1)
    unsqueeze_373: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
    mul_140: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_139, unsqueeze_373);  mul_139 = unsqueeze_373 = None
    unsqueeze_374: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_141, -1);  primals_141 = None
    unsqueeze_375: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
    add_106: "f32[4, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_140, unsqueeze_375);  mul_140 = unsqueeze_375 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:160, code: out += identity
    add_107: "f32[4, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_104, add_106);  add_104 = add_106 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    relu_42: "f32[4, 2048, 7, 7]" = torch.ops.aten.relu.default(add_107);  add_107 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_47: "f32[4, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_42, primals_142, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    convert_element_type_94: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_303, torch.float32)
    convert_element_type_95: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_304, torch.float32)
    add_108: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_95, 1e-05);  convert_element_type_95 = None
    sqrt_47: "f32[512]" = torch.ops.aten.sqrt.default(add_108);  add_108 = None
    reciprocal_47: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
    mul_141: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
    unsqueeze_376: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_94, -1);  convert_element_type_94 = None
    unsqueeze_377: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
    unsqueeze_378: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_141, -1);  mul_141 = None
    unsqueeze_379: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
    sub_47: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_377);  unsqueeze_377 = None
    mul_142: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_379);  sub_47 = unsqueeze_379 = None
    unsqueeze_380: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_143, -1)
    unsqueeze_381: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
    mul_143: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_142, unsqueeze_381);  mul_142 = unsqueeze_381 = None
    unsqueeze_382: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_144, -1);  primals_144 = None
    unsqueeze_383: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
    add_109: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_143, unsqueeze_383);  mul_143 = unsqueeze_383 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    relu_43: "f32[4, 512, 7, 7]" = torch.ops.aten.relu.default(add_109);  add_109 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_48: "f32[4, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_43, primals_145, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    convert_element_type_96: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_306, torch.float32)
    convert_element_type_97: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_307, torch.float32)
    add_110: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_97, 1e-05);  convert_element_type_97 = None
    sqrt_48: "f32[512]" = torch.ops.aten.sqrt.default(add_110);  add_110 = None
    reciprocal_48: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
    mul_144: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
    unsqueeze_384: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_96, -1);  convert_element_type_96 = None
    unsqueeze_385: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
    unsqueeze_386: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_144, -1);  mul_144 = None
    unsqueeze_387: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
    sub_48: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_385);  unsqueeze_385 = None
    mul_145: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_387);  sub_48 = unsqueeze_387 = None
    unsqueeze_388: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_146, -1)
    unsqueeze_389: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
    mul_146: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_145, unsqueeze_389);  mul_145 = unsqueeze_389 = None
    unsqueeze_390: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_147, -1);  primals_147 = None
    unsqueeze_391: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
    add_111: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_391);  mul_146 = unsqueeze_391 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    relu_44: "f32[4, 512, 7, 7]" = torch.ops.aten.relu.default(add_111);  add_111 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_49: "f32[4, 2048, 7, 7]" = torch.ops.aten.convolution.default(relu_44, primals_148, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    convert_element_type_98: "f32[2048]" = torch.ops.prims.convert_element_type.default(primals_309, torch.float32)
    convert_element_type_99: "f32[2048]" = torch.ops.prims.convert_element_type.default(primals_310, torch.float32)
    add_112: "f32[2048]" = torch.ops.aten.add.Tensor(convert_element_type_99, 1e-05);  convert_element_type_99 = None
    sqrt_49: "f32[2048]" = torch.ops.aten.sqrt.default(add_112);  add_112 = None
    reciprocal_49: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
    mul_147: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
    unsqueeze_392: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_98, -1);  convert_element_type_98 = None
    unsqueeze_393: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
    unsqueeze_394: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_147, -1);  mul_147 = None
    unsqueeze_395: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
    sub_49: "f32[4, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_393);  unsqueeze_393 = None
    mul_148: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_395);  sub_49 = unsqueeze_395 = None
    unsqueeze_396: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_149, -1)
    unsqueeze_397: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
    mul_149: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_148, unsqueeze_397);  mul_148 = unsqueeze_397 = None
    unsqueeze_398: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_150, -1);  primals_150 = None
    unsqueeze_399: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
    add_113: "f32[4, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_399);  mul_149 = unsqueeze_399 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:160, code: out += identity
    add_114: "f32[4, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_113, relu_42);  add_113 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    relu_45: "f32[4, 2048, 7, 7]" = torch.ops.aten.relu.default(add_114);  add_114 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_50: "f32[4, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_45, primals_151, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    convert_element_type_100: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_312, torch.float32)
    convert_element_type_101: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_313, torch.float32)
    add_115: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_101, 1e-05);  convert_element_type_101 = None
    sqrt_50: "f32[512]" = torch.ops.aten.sqrt.default(add_115);  add_115 = None
    reciprocal_50: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
    mul_150: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
    unsqueeze_400: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_100, -1);  convert_element_type_100 = None
    unsqueeze_401: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
    unsqueeze_402: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_150, -1);  mul_150 = None
    unsqueeze_403: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
    sub_50: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_401);  unsqueeze_401 = None
    mul_151: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_403);  sub_50 = unsqueeze_403 = None
    unsqueeze_404: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_152, -1)
    unsqueeze_405: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
    mul_152: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_151, unsqueeze_405);  mul_151 = unsqueeze_405 = None
    unsqueeze_406: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_153, -1);  primals_153 = None
    unsqueeze_407: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
    add_116: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_152, unsqueeze_407);  mul_152 = unsqueeze_407 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    relu_46: "f32[4, 512, 7, 7]" = torch.ops.aten.relu.default(add_116);  add_116 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_51: "f32[4, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_46, primals_154, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    convert_element_type_102: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_315, torch.float32)
    convert_element_type_103: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_316, torch.float32)
    add_117: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_103, 1e-05);  convert_element_type_103 = None
    sqrt_51: "f32[512]" = torch.ops.aten.sqrt.default(add_117);  add_117 = None
    reciprocal_51: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
    mul_153: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
    unsqueeze_408: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_102, -1);  convert_element_type_102 = None
    unsqueeze_409: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
    unsqueeze_410: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_153, -1);  mul_153 = None
    unsqueeze_411: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
    sub_51: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_409);  unsqueeze_409 = None
    mul_154: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_411);  sub_51 = unsqueeze_411 = None
    unsqueeze_412: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_155, -1)
    unsqueeze_413: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
    mul_155: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_413);  mul_154 = unsqueeze_413 = None
    unsqueeze_414: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_156, -1);  primals_156 = None
    unsqueeze_415: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
    add_118: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_155, unsqueeze_415);  mul_155 = unsqueeze_415 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    relu_47: "f32[4, 512, 7, 7]" = torch.ops.aten.relu.default(add_118);  add_118 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_52: "f32[4, 2048, 7, 7]" = torch.ops.aten.convolution.default(relu_47, primals_157, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    convert_element_type_104: "f32[2048]" = torch.ops.prims.convert_element_type.default(primals_318, torch.float32)
    convert_element_type_105: "f32[2048]" = torch.ops.prims.convert_element_type.default(primals_319, torch.float32)
    add_119: "f32[2048]" = torch.ops.aten.add.Tensor(convert_element_type_105, 1e-05);  convert_element_type_105 = None
    sqrt_52: "f32[2048]" = torch.ops.aten.sqrt.default(add_119);  add_119 = None
    reciprocal_52: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
    mul_156: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
    unsqueeze_416: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_104, -1);  convert_element_type_104 = None
    unsqueeze_417: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
    unsqueeze_418: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_156, -1);  mul_156 = None
    unsqueeze_419: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
    sub_52: "f32[4, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_417);  unsqueeze_417 = None
    mul_157: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_419);  sub_52 = unsqueeze_419 = None
    unsqueeze_420: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_158, -1)
    unsqueeze_421: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
    mul_158: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_157, unsqueeze_421);  mul_157 = unsqueeze_421 = None
    unsqueeze_422: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_159, -1);  primals_159 = None
    unsqueeze_423: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
    add_120: "f32[4, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_158, unsqueeze_423);  mul_158 = unsqueeze_423 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:160, code: out += identity
    add_121: "f32[4, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_120, relu_45);  add_120 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    relu_48: "f32[4, 2048, 7, 7]" = torch.ops.aten.relu.default(add_121);  add_121 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:278, code: x = self.avgpool(x)
    mean: "f32[4, 2048, 1, 1]" = torch.ops.aten.mean.dim(relu_48, [-1, -2], True)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:279, code: x = torch.flatten(x, 1)
    view: "f32[4, 2048]" = torch.ops.aten.view.default(mean, [4, 2048]);  mean = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:280, code: x = self.fc(x)
    permute: "f32[2048, 1000]" = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
    addmm: "f32[4, 1000]" = torch.ops.aten.addmm.default(primals_161, view, permute);  primals_161 = None
    permute_1: "f32[1000, 2048]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm: "f32[4, 2048]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 4]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 2048]" = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
    permute_3: "f32[2048, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 2048]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:279, code: x = torch.flatten(x, 1)
    view_2: "f32[4, 2048, 1, 1]" = torch.ops.aten.view.default(mm, [4, 2048, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:278, code: x = self.avgpool(x)
    expand: "f32[4, 2048, 7, 7]" = torch.ops.aten.expand.default(view_2, [4, 2048, 7, 7]);  view_2 = None
    div: "f32[4, 2048, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    alias_50: "f32[4, 2048, 7, 7]" = torch.ops.aten.alias.default(relu_48);  relu_48 = None
    alias_51: "f32[4, 2048, 7, 7]" = torch.ops.aten.alias.default(alias_50);  alias_50 = None
    le: "b8[4, 2048, 7, 7]" = torch.ops.aten.le.Scalar(alias_51, 0);  alias_51 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where: "f32[4, 2048, 7, 7]" = torch.ops.aten.where.self(le, scalar_tensor, div);  le = scalar_tensor = div = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    add_122: "f32[2048]" = torch.ops.aten.add.Tensor(primals_319, 1e-05);  primals_319 = None
    rsqrt: "f32[2048]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    unsqueeze_424: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(primals_318, 0);  primals_318 = None
    unsqueeze_425: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 2);  unsqueeze_424 = None
    unsqueeze_426: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 3);  unsqueeze_425 = None
    sum_2: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_53: "f32[4, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_426);  convolution_52 = unsqueeze_426 = None
    mul_159: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_53);  sub_53 = None
    sum_3: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_159, [0, 2, 3]);  mul_159 = None
    mul_164: "f32[2048]" = torch.ops.aten.mul.Tensor(rsqrt, primals_158);  primals_158 = None
    unsqueeze_433: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_164, 0);  mul_164 = None
    unsqueeze_434: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
    unsqueeze_435: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 3);  unsqueeze_434 = None
    mul_165: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where, unsqueeze_435);  unsqueeze_435 = None
    mul_166: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_3, rsqrt);  sum_3 = rsqrt = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_165, relu_47, primals_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_165 = primals_157 = None
    getitem_2: "f32[4, 512, 7, 7]" = convolution_backward[0]
    getitem_3: "f32[2048, 512, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    alias_53: "f32[4, 512, 7, 7]" = torch.ops.aten.alias.default(relu_47);  relu_47 = None
    alias_54: "f32[4, 512, 7, 7]" = torch.ops.aten.alias.default(alias_53);  alias_53 = None
    le_1: "b8[4, 512, 7, 7]" = torch.ops.aten.le.Scalar(alias_54, 0);  alias_54 = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_1: "f32[4, 512, 7, 7]" = torch.ops.aten.where.self(le_1, scalar_tensor_1, getitem_2);  le_1 = scalar_tensor_1 = getitem_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    add_123: "f32[512]" = torch.ops.aten.add.Tensor(primals_316, 1e-05);  primals_316 = None
    rsqrt_1: "f32[512]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
    unsqueeze_436: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_315, 0);  primals_315 = None
    unsqueeze_437: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 2);  unsqueeze_436 = None
    unsqueeze_438: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 3);  unsqueeze_437 = None
    sum_4: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_54: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_438);  convolution_51 = unsqueeze_438 = None
    mul_167: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, sub_54);  sub_54 = None
    sum_5: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_167, [0, 2, 3]);  mul_167 = None
    mul_172: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_1, primals_155);  primals_155 = None
    unsqueeze_445: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_172, 0);  mul_172 = None
    unsqueeze_446: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
    unsqueeze_447: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 3);  unsqueeze_446 = None
    mul_173: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, unsqueeze_447);  where_1 = unsqueeze_447 = None
    mul_174: "f32[512]" = torch.ops.aten.mul.Tensor(sum_5, rsqrt_1);  sum_5 = rsqrt_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_173, relu_46, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_173 = primals_154 = None
    getitem_5: "f32[4, 512, 7, 7]" = convolution_backward_1[0]
    getitem_6: "f32[512, 512, 3, 3]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    alias_56: "f32[4, 512, 7, 7]" = torch.ops.aten.alias.default(relu_46);  relu_46 = None
    alias_57: "f32[4, 512, 7, 7]" = torch.ops.aten.alias.default(alias_56);  alias_56 = None
    le_2: "b8[4, 512, 7, 7]" = torch.ops.aten.le.Scalar(alias_57, 0);  alias_57 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_2: "f32[4, 512, 7, 7]" = torch.ops.aten.where.self(le_2, scalar_tensor_2, getitem_5);  le_2 = scalar_tensor_2 = getitem_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    add_124: "f32[512]" = torch.ops.aten.add.Tensor(primals_313, 1e-05);  primals_313 = None
    rsqrt_2: "f32[512]" = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
    unsqueeze_448: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_312, 0);  primals_312 = None
    unsqueeze_449: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, 2);  unsqueeze_448 = None
    unsqueeze_450: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 3);  unsqueeze_449 = None
    sum_6: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_55: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_450);  convolution_50 = unsqueeze_450 = None
    mul_175: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_55);  sub_55 = None
    sum_7: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_175, [0, 2, 3]);  mul_175 = None
    mul_180: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_2, primals_152);  primals_152 = None
    unsqueeze_457: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_180, 0);  mul_180 = None
    unsqueeze_458: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 2);  unsqueeze_457 = None
    unsqueeze_459: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 3);  unsqueeze_458 = None
    mul_181: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, unsqueeze_459);  where_2 = unsqueeze_459 = None
    mul_182: "f32[512]" = torch.ops.aten.mul.Tensor(sum_7, rsqrt_2);  sum_7 = rsqrt_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_181, relu_45, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_181 = primals_151 = None
    getitem_8: "f32[4, 2048, 7, 7]" = convolution_backward_2[0]
    getitem_9: "f32[512, 2048, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    add_125: "f32[4, 2048, 7, 7]" = torch.ops.aten.add.Tensor(where, getitem_8);  where = getitem_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    alias_59: "f32[4, 2048, 7, 7]" = torch.ops.aten.alias.default(relu_45);  relu_45 = None
    alias_60: "f32[4, 2048, 7, 7]" = torch.ops.aten.alias.default(alias_59);  alias_59 = None
    le_3: "b8[4, 2048, 7, 7]" = torch.ops.aten.le.Scalar(alias_60, 0);  alias_60 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_3: "f32[4, 2048, 7, 7]" = torch.ops.aten.where.self(le_3, scalar_tensor_3, add_125);  le_3 = scalar_tensor_3 = add_125 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    add_126: "f32[2048]" = torch.ops.aten.add.Tensor(primals_310, 1e-05);  primals_310 = None
    rsqrt_3: "f32[2048]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    unsqueeze_460: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(primals_309, 0);  primals_309 = None
    unsqueeze_461: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, 2);  unsqueeze_460 = None
    unsqueeze_462: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 3);  unsqueeze_461 = None
    sum_8: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_56: "f32[4, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_462);  convolution_49 = unsqueeze_462 = None
    mul_183: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_56);  sub_56 = None
    sum_9: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_183, [0, 2, 3]);  mul_183 = None
    mul_188: "f32[2048]" = torch.ops.aten.mul.Tensor(rsqrt_3, primals_149);  primals_149 = None
    unsqueeze_469: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_188, 0);  mul_188 = None
    unsqueeze_470: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None
    unsqueeze_471: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 3);  unsqueeze_470 = None
    mul_189: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, unsqueeze_471);  unsqueeze_471 = None
    mul_190: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_9, rsqrt_3);  sum_9 = rsqrt_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_189, relu_44, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_189 = primals_148 = None
    getitem_11: "f32[4, 512, 7, 7]" = convolution_backward_3[0]
    getitem_12: "f32[2048, 512, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    alias_62: "f32[4, 512, 7, 7]" = torch.ops.aten.alias.default(relu_44);  relu_44 = None
    alias_63: "f32[4, 512, 7, 7]" = torch.ops.aten.alias.default(alias_62);  alias_62 = None
    le_4: "b8[4, 512, 7, 7]" = torch.ops.aten.le.Scalar(alias_63, 0);  alias_63 = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_4: "f32[4, 512, 7, 7]" = torch.ops.aten.where.self(le_4, scalar_tensor_4, getitem_11);  le_4 = scalar_tensor_4 = getitem_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    add_127: "f32[512]" = torch.ops.aten.add.Tensor(primals_307, 1e-05);  primals_307 = None
    rsqrt_4: "f32[512]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
    unsqueeze_472: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_306, 0);  primals_306 = None
    unsqueeze_473: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, 2);  unsqueeze_472 = None
    unsqueeze_474: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 3);  unsqueeze_473 = None
    sum_10: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_57: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_474);  convolution_48 = unsqueeze_474 = None
    mul_191: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, sub_57);  sub_57 = None
    sum_11: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_191, [0, 2, 3]);  mul_191 = None
    mul_196: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_4, primals_146);  primals_146 = None
    unsqueeze_481: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_196, 0);  mul_196 = None
    unsqueeze_482: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 2);  unsqueeze_481 = None
    unsqueeze_483: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 3);  unsqueeze_482 = None
    mul_197: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, unsqueeze_483);  where_4 = unsqueeze_483 = None
    mul_198: "f32[512]" = torch.ops.aten.mul.Tensor(sum_11, rsqrt_4);  sum_11 = rsqrt_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_197, relu_43, primals_145, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_197 = primals_145 = None
    getitem_14: "f32[4, 512, 7, 7]" = convolution_backward_4[0]
    getitem_15: "f32[512, 512, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    alias_65: "f32[4, 512, 7, 7]" = torch.ops.aten.alias.default(relu_43);  relu_43 = None
    alias_66: "f32[4, 512, 7, 7]" = torch.ops.aten.alias.default(alias_65);  alias_65 = None
    le_5: "b8[4, 512, 7, 7]" = torch.ops.aten.le.Scalar(alias_66, 0);  alias_66 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_5: "f32[4, 512, 7, 7]" = torch.ops.aten.where.self(le_5, scalar_tensor_5, getitem_14);  le_5 = scalar_tensor_5 = getitem_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    add_128: "f32[512]" = torch.ops.aten.add.Tensor(primals_304, 1e-05);  primals_304 = None
    rsqrt_5: "f32[512]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
    unsqueeze_484: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_303, 0);  primals_303 = None
    unsqueeze_485: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 2);  unsqueeze_484 = None
    unsqueeze_486: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 3);  unsqueeze_485 = None
    sum_12: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_58: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_486);  convolution_47 = unsqueeze_486 = None
    mul_199: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, sub_58);  sub_58 = None
    sum_13: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_199, [0, 2, 3]);  mul_199 = None
    mul_204: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_5, primals_143);  primals_143 = None
    unsqueeze_493: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_204, 0);  mul_204 = None
    unsqueeze_494: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
    unsqueeze_495: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 3);  unsqueeze_494 = None
    mul_205: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, unsqueeze_495);  where_5 = unsqueeze_495 = None
    mul_206: "f32[512]" = torch.ops.aten.mul.Tensor(sum_13, rsqrt_5);  sum_13 = rsqrt_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_205, relu_42, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_205 = primals_142 = None
    getitem_17: "f32[4, 2048, 7, 7]" = convolution_backward_5[0]
    getitem_18: "f32[512, 2048, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    add_129: "f32[4, 2048, 7, 7]" = torch.ops.aten.add.Tensor(where_3, getitem_17);  where_3 = getitem_17 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    alias_68: "f32[4, 2048, 7, 7]" = torch.ops.aten.alias.default(relu_42);  relu_42 = None
    alias_69: "f32[4, 2048, 7, 7]" = torch.ops.aten.alias.default(alias_68);  alias_68 = None
    le_6: "b8[4, 2048, 7, 7]" = torch.ops.aten.le.Scalar(alias_69, 0);  alias_69 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_6: "f32[4, 2048, 7, 7]" = torch.ops.aten.where.self(le_6, scalar_tensor_6, add_129);  le_6 = scalar_tensor_6 = add_129 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:158, code: identity = self.downsample(x)
    add_130: "f32[2048]" = torch.ops.aten.add.Tensor(primals_301, 1e-05);  primals_301 = None
    rsqrt_6: "f32[2048]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
    unsqueeze_496: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(primals_300, 0);  primals_300 = None
    unsqueeze_497: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 2);  unsqueeze_496 = None
    unsqueeze_498: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 3);  unsqueeze_497 = None
    sum_14: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_59: "f32[4, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_498);  convolution_46 = unsqueeze_498 = None
    mul_207: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, sub_59);  sub_59 = None
    sum_15: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_207, [0, 2, 3]);  mul_207 = None
    mul_212: "f32[2048]" = torch.ops.aten.mul.Tensor(rsqrt_6, primals_140);  primals_140 = None
    unsqueeze_505: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_212, 0);  mul_212 = None
    unsqueeze_506: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 2);  unsqueeze_505 = None
    unsqueeze_507: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 3);  unsqueeze_506 = None
    mul_213: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, unsqueeze_507);  unsqueeze_507 = None
    mul_214: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_15, rsqrt_6);  sum_15 = rsqrt_6 = None
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_213, relu_39, primals_139, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_213 = primals_139 = None
    getitem_20: "f32[4, 1024, 14, 14]" = convolution_backward_6[0]
    getitem_21: "f32[2048, 1024, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    add_131: "f32[2048]" = torch.ops.aten.add.Tensor(primals_298, 1e-05);  primals_298 = None
    rsqrt_7: "f32[2048]" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
    unsqueeze_508: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(primals_297, 0);  primals_297 = None
    unsqueeze_509: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, 2);  unsqueeze_508 = None
    unsqueeze_510: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 3);  unsqueeze_509 = None
    sum_16: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_60: "f32[4, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_510);  convolution_45 = unsqueeze_510 = None
    mul_215: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, sub_60);  sub_60 = None
    sum_17: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_215, [0, 2, 3]);  mul_215 = None
    mul_220: "f32[2048]" = torch.ops.aten.mul.Tensor(rsqrt_7, primals_137);  primals_137 = None
    unsqueeze_517: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_220, 0);  mul_220 = None
    unsqueeze_518: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 2);  unsqueeze_517 = None
    unsqueeze_519: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 3);  unsqueeze_518 = None
    mul_221: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, unsqueeze_519);  where_6 = unsqueeze_519 = None
    mul_222: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_17, rsqrt_7);  sum_17 = rsqrt_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_221, relu_41, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_221 = primals_136 = None
    getitem_23: "f32[4, 512, 7, 7]" = convolution_backward_7[0]
    getitem_24: "f32[2048, 512, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    alias_71: "f32[4, 512, 7, 7]" = torch.ops.aten.alias.default(relu_41);  relu_41 = None
    alias_72: "f32[4, 512, 7, 7]" = torch.ops.aten.alias.default(alias_71);  alias_71 = None
    le_7: "b8[4, 512, 7, 7]" = torch.ops.aten.le.Scalar(alias_72, 0);  alias_72 = None
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_7: "f32[4, 512, 7, 7]" = torch.ops.aten.where.self(le_7, scalar_tensor_7, getitem_23);  le_7 = scalar_tensor_7 = getitem_23 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    add_132: "f32[512]" = torch.ops.aten.add.Tensor(primals_295, 1e-05);  primals_295 = None
    rsqrt_8: "f32[512]" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
    unsqueeze_520: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_294, 0);  primals_294 = None
    unsqueeze_521: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 2);  unsqueeze_520 = None
    unsqueeze_522: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 3);  unsqueeze_521 = None
    sum_18: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_61: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_522);  convolution_44 = unsqueeze_522 = None
    mul_223: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, sub_61);  sub_61 = None
    sum_19: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_223, [0, 2, 3]);  mul_223 = None
    mul_228: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_8, primals_134);  primals_134 = None
    unsqueeze_529: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_228, 0);  mul_228 = None
    unsqueeze_530: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 2);  unsqueeze_529 = None
    unsqueeze_531: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 3);  unsqueeze_530 = None
    mul_229: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, unsqueeze_531);  where_7 = unsqueeze_531 = None
    mul_230: "f32[512]" = torch.ops.aten.mul.Tensor(sum_19, rsqrt_8);  sum_19 = rsqrt_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_229, relu_40, primals_133, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_229 = primals_133 = None
    getitem_26: "f32[4, 512, 14, 14]" = convolution_backward_8[0]
    getitem_27: "f32[512, 512, 3, 3]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    alias_74: "f32[4, 512, 14, 14]" = torch.ops.aten.alias.default(relu_40);  relu_40 = None
    alias_75: "f32[4, 512, 14, 14]" = torch.ops.aten.alias.default(alias_74);  alias_74 = None
    le_8: "b8[4, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_75, 0);  alias_75 = None
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_8: "f32[4, 512, 14, 14]" = torch.ops.aten.where.self(le_8, scalar_tensor_8, getitem_26);  le_8 = scalar_tensor_8 = getitem_26 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    add_133: "f32[512]" = torch.ops.aten.add.Tensor(primals_292, 1e-05);  primals_292 = None
    rsqrt_9: "f32[512]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
    unsqueeze_532: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_291, 0);  primals_291 = None
    unsqueeze_533: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 2);  unsqueeze_532 = None
    unsqueeze_534: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 3);  unsqueeze_533 = None
    sum_20: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_62: "f32[4, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_534);  convolution_43 = unsqueeze_534 = None
    mul_231: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_8, sub_62);  sub_62 = None
    sum_21: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_231, [0, 2, 3]);  mul_231 = None
    mul_236: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_9, primals_131);  primals_131 = None
    unsqueeze_541: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_236, 0);  mul_236 = None
    unsqueeze_542: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 2);  unsqueeze_541 = None
    unsqueeze_543: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 3);  unsqueeze_542 = None
    mul_237: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_8, unsqueeze_543);  where_8 = unsqueeze_543 = None
    mul_238: "f32[512]" = torch.ops.aten.mul.Tensor(sum_21, rsqrt_9);  sum_21 = rsqrt_9 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_237, relu_39, primals_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_237 = primals_130 = None
    getitem_29: "f32[4, 1024, 14, 14]" = convolution_backward_9[0]
    getitem_30: "f32[512, 1024, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    add_134: "f32[4, 1024, 14, 14]" = torch.ops.aten.add.Tensor(getitem_20, getitem_29);  getitem_20 = getitem_29 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    alias_77: "f32[4, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_39);  relu_39 = None
    alias_78: "f32[4, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_77);  alias_77 = None
    le_9: "b8[4, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_78, 0);  alias_78 = None
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_9: "f32[4, 1024, 14, 14]" = torch.ops.aten.where.self(le_9, scalar_tensor_9, add_134);  le_9 = scalar_tensor_9 = add_134 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    add_135: "f32[1024]" = torch.ops.aten.add.Tensor(primals_289, 1e-05);  primals_289 = None
    rsqrt_10: "f32[1024]" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
    unsqueeze_544: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(primals_288, 0);  primals_288 = None
    unsqueeze_545: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, 2);  unsqueeze_544 = None
    unsqueeze_546: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 3);  unsqueeze_545 = None
    sum_22: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_63: "f32[4, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_546);  convolution_42 = unsqueeze_546 = None
    mul_239: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_9, sub_63);  sub_63 = None
    sum_23: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_239, [0, 2, 3]);  mul_239 = None
    mul_244: "f32[1024]" = torch.ops.aten.mul.Tensor(rsqrt_10, primals_128);  primals_128 = None
    unsqueeze_553: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_244, 0);  mul_244 = None
    unsqueeze_554: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 2);  unsqueeze_553 = None
    unsqueeze_555: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 3);  unsqueeze_554 = None
    mul_245: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_9, unsqueeze_555);  unsqueeze_555 = None
    mul_246: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_23, rsqrt_10);  sum_23 = rsqrt_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_245, relu_38, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_245 = primals_127 = None
    getitem_32: "f32[4, 256, 14, 14]" = convolution_backward_10[0]
    getitem_33: "f32[1024, 256, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    alias_80: "f32[4, 256, 14, 14]" = torch.ops.aten.alias.default(relu_38);  relu_38 = None
    alias_81: "f32[4, 256, 14, 14]" = torch.ops.aten.alias.default(alias_80);  alias_80 = None
    le_10: "b8[4, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_81, 0);  alias_81 = None
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_10: "f32[4, 256, 14, 14]" = torch.ops.aten.where.self(le_10, scalar_tensor_10, getitem_32);  le_10 = scalar_tensor_10 = getitem_32 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    add_136: "f32[256]" = torch.ops.aten.add.Tensor(primals_286, 1e-05);  primals_286 = None
    rsqrt_11: "f32[256]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
    unsqueeze_556: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_285, 0);  primals_285 = None
    unsqueeze_557: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 2);  unsqueeze_556 = None
    unsqueeze_558: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 3);  unsqueeze_557 = None
    sum_24: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_64: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_558);  convolution_41 = unsqueeze_558 = None
    mul_247: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, sub_64);  sub_64 = None
    sum_25: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_247, [0, 2, 3]);  mul_247 = None
    mul_252: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_11, primals_125);  primals_125 = None
    unsqueeze_565: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_252, 0);  mul_252 = None
    unsqueeze_566: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 2);  unsqueeze_565 = None
    unsqueeze_567: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 3);  unsqueeze_566 = None
    mul_253: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, unsqueeze_567);  where_10 = unsqueeze_567 = None
    mul_254: "f32[256]" = torch.ops.aten.mul.Tensor(sum_25, rsqrt_11);  sum_25 = rsqrt_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_253, relu_37, primals_124, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_253 = primals_124 = None
    getitem_35: "f32[4, 256, 14, 14]" = convolution_backward_11[0]
    getitem_36: "f32[256, 256, 3, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    alias_83: "f32[4, 256, 14, 14]" = torch.ops.aten.alias.default(relu_37);  relu_37 = None
    alias_84: "f32[4, 256, 14, 14]" = torch.ops.aten.alias.default(alias_83);  alias_83 = None
    le_11: "b8[4, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_84, 0);  alias_84 = None
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_11: "f32[4, 256, 14, 14]" = torch.ops.aten.where.self(le_11, scalar_tensor_11, getitem_35);  le_11 = scalar_tensor_11 = getitem_35 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    add_137: "f32[256]" = torch.ops.aten.add.Tensor(primals_283, 1e-05);  primals_283 = None
    rsqrt_12: "f32[256]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
    unsqueeze_568: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_282, 0);  primals_282 = None
    unsqueeze_569: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, 2);  unsqueeze_568 = None
    unsqueeze_570: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 3);  unsqueeze_569 = None
    sum_26: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_65: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_570);  convolution_40 = unsqueeze_570 = None
    mul_255: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, sub_65);  sub_65 = None
    sum_27: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_255, [0, 2, 3]);  mul_255 = None
    mul_260: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_12, primals_122);  primals_122 = None
    unsqueeze_577: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_260, 0);  mul_260 = None
    unsqueeze_578: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 2);  unsqueeze_577 = None
    unsqueeze_579: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 3);  unsqueeze_578 = None
    mul_261: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, unsqueeze_579);  where_11 = unsqueeze_579 = None
    mul_262: "f32[256]" = torch.ops.aten.mul.Tensor(sum_27, rsqrt_12);  sum_27 = rsqrt_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_261, relu_36, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_261 = primals_121 = None
    getitem_38: "f32[4, 1024, 14, 14]" = convolution_backward_12[0]
    getitem_39: "f32[256, 1024, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    add_138: "f32[4, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_9, getitem_38);  where_9 = getitem_38 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    alias_86: "f32[4, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_36);  relu_36 = None
    alias_87: "f32[4, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_86);  alias_86 = None
    le_12: "b8[4, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_87, 0);  alias_87 = None
    scalar_tensor_12: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_12: "f32[4, 1024, 14, 14]" = torch.ops.aten.where.self(le_12, scalar_tensor_12, add_138);  le_12 = scalar_tensor_12 = add_138 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    add_139: "f32[1024]" = torch.ops.aten.add.Tensor(primals_280, 1e-05);  primals_280 = None
    rsqrt_13: "f32[1024]" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
    unsqueeze_580: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(primals_279, 0);  primals_279 = None
    unsqueeze_581: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 2);  unsqueeze_580 = None
    unsqueeze_582: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 3);  unsqueeze_581 = None
    sum_28: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_66: "f32[4, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_582);  convolution_39 = unsqueeze_582 = None
    mul_263: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, sub_66);  sub_66 = None
    sum_29: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_263, [0, 2, 3]);  mul_263 = None
    mul_268: "f32[1024]" = torch.ops.aten.mul.Tensor(rsqrt_13, primals_119);  primals_119 = None
    unsqueeze_589: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_268, 0);  mul_268 = None
    unsqueeze_590: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 2);  unsqueeze_589 = None
    unsqueeze_591: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 3);  unsqueeze_590 = None
    mul_269: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, unsqueeze_591);  unsqueeze_591 = None
    mul_270: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_29, rsqrt_13);  sum_29 = rsqrt_13 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_269, relu_35, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_269 = primals_118 = None
    getitem_41: "f32[4, 256, 14, 14]" = convolution_backward_13[0]
    getitem_42: "f32[1024, 256, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    alias_89: "f32[4, 256, 14, 14]" = torch.ops.aten.alias.default(relu_35);  relu_35 = None
    alias_90: "f32[4, 256, 14, 14]" = torch.ops.aten.alias.default(alias_89);  alias_89 = None
    le_13: "b8[4, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_90, 0);  alias_90 = None
    scalar_tensor_13: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_13: "f32[4, 256, 14, 14]" = torch.ops.aten.where.self(le_13, scalar_tensor_13, getitem_41);  le_13 = scalar_tensor_13 = getitem_41 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    add_140: "f32[256]" = torch.ops.aten.add.Tensor(primals_277, 1e-05);  primals_277 = None
    rsqrt_14: "f32[256]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    unsqueeze_592: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_276, 0);  primals_276 = None
    unsqueeze_593: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 2);  unsqueeze_592 = None
    unsqueeze_594: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 3);  unsqueeze_593 = None
    sum_30: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_67: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_594);  convolution_38 = unsqueeze_594 = None
    mul_271: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_13, sub_67);  sub_67 = None
    sum_31: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_271, [0, 2, 3]);  mul_271 = None
    mul_276: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_14, primals_116);  primals_116 = None
    unsqueeze_601: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_276, 0);  mul_276 = None
    unsqueeze_602: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 2);  unsqueeze_601 = None
    unsqueeze_603: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 3);  unsqueeze_602 = None
    mul_277: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_13, unsqueeze_603);  where_13 = unsqueeze_603 = None
    mul_278: "f32[256]" = torch.ops.aten.mul.Tensor(sum_31, rsqrt_14);  sum_31 = rsqrt_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_277, relu_34, primals_115, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_277 = primals_115 = None
    getitem_44: "f32[4, 256, 14, 14]" = convolution_backward_14[0]
    getitem_45: "f32[256, 256, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    alias_92: "f32[4, 256, 14, 14]" = torch.ops.aten.alias.default(relu_34);  relu_34 = None
    alias_93: "f32[4, 256, 14, 14]" = torch.ops.aten.alias.default(alias_92);  alias_92 = None
    le_14: "b8[4, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_93, 0);  alias_93 = None
    scalar_tensor_14: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_14: "f32[4, 256, 14, 14]" = torch.ops.aten.where.self(le_14, scalar_tensor_14, getitem_44);  le_14 = scalar_tensor_14 = getitem_44 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    add_141: "f32[256]" = torch.ops.aten.add.Tensor(primals_274, 1e-05);  primals_274 = None
    rsqrt_15: "f32[256]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
    unsqueeze_604: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_273, 0);  primals_273 = None
    unsqueeze_605: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, 2);  unsqueeze_604 = None
    unsqueeze_606: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 3);  unsqueeze_605 = None
    sum_32: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_68: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_606);  convolution_37 = unsqueeze_606 = None
    mul_279: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, sub_68);  sub_68 = None
    sum_33: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_279, [0, 2, 3]);  mul_279 = None
    mul_284: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_15, primals_113);  primals_113 = None
    unsqueeze_613: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_284, 0);  mul_284 = None
    unsqueeze_614: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 2);  unsqueeze_613 = None
    unsqueeze_615: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 3);  unsqueeze_614 = None
    mul_285: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, unsqueeze_615);  where_14 = unsqueeze_615 = None
    mul_286: "f32[256]" = torch.ops.aten.mul.Tensor(sum_33, rsqrt_15);  sum_33 = rsqrt_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_285, relu_33, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_285 = primals_112 = None
    getitem_47: "f32[4, 1024, 14, 14]" = convolution_backward_15[0]
    getitem_48: "f32[256, 1024, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    add_142: "f32[4, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_12, getitem_47);  where_12 = getitem_47 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    alias_95: "f32[4, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_33);  relu_33 = None
    alias_96: "f32[4, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_95);  alias_95 = None
    le_15: "b8[4, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_96, 0);  alias_96 = None
    scalar_tensor_15: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_15: "f32[4, 1024, 14, 14]" = torch.ops.aten.where.self(le_15, scalar_tensor_15, add_142);  le_15 = scalar_tensor_15 = add_142 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    add_143: "f32[1024]" = torch.ops.aten.add.Tensor(primals_271, 1e-05);  primals_271 = None
    rsqrt_16: "f32[1024]" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
    unsqueeze_616: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(primals_270, 0);  primals_270 = None
    unsqueeze_617: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 2);  unsqueeze_616 = None
    unsqueeze_618: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 3);  unsqueeze_617 = None
    sum_34: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_69: "f32[4, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_618);  convolution_36 = unsqueeze_618 = None
    mul_287: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, sub_69);  sub_69 = None
    sum_35: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_287, [0, 2, 3]);  mul_287 = None
    mul_292: "f32[1024]" = torch.ops.aten.mul.Tensor(rsqrt_16, primals_110);  primals_110 = None
    unsqueeze_625: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_292, 0);  mul_292 = None
    unsqueeze_626: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 2);  unsqueeze_625 = None
    unsqueeze_627: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 3);  unsqueeze_626 = None
    mul_293: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, unsqueeze_627);  unsqueeze_627 = None
    mul_294: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_35, rsqrt_16);  sum_35 = rsqrt_16 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_293, relu_32, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_293 = primals_109 = None
    getitem_50: "f32[4, 256, 14, 14]" = convolution_backward_16[0]
    getitem_51: "f32[1024, 256, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    alias_98: "f32[4, 256, 14, 14]" = torch.ops.aten.alias.default(relu_32);  relu_32 = None
    alias_99: "f32[4, 256, 14, 14]" = torch.ops.aten.alias.default(alias_98);  alias_98 = None
    le_16: "b8[4, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_99, 0);  alias_99 = None
    scalar_tensor_16: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_16: "f32[4, 256, 14, 14]" = torch.ops.aten.where.self(le_16, scalar_tensor_16, getitem_50);  le_16 = scalar_tensor_16 = getitem_50 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    add_144: "f32[256]" = torch.ops.aten.add.Tensor(primals_268, 1e-05);  primals_268 = None
    rsqrt_17: "f32[256]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    unsqueeze_628: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_267, 0);  primals_267 = None
    unsqueeze_629: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, 2);  unsqueeze_628 = None
    unsqueeze_630: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 3);  unsqueeze_629 = None
    sum_36: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_70: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_630);  convolution_35 = unsqueeze_630 = None
    mul_295: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, sub_70);  sub_70 = None
    sum_37: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_295, [0, 2, 3]);  mul_295 = None
    mul_300: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_17, primals_107);  primals_107 = None
    unsqueeze_637: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_300, 0);  mul_300 = None
    unsqueeze_638: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 2);  unsqueeze_637 = None
    unsqueeze_639: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 3);  unsqueeze_638 = None
    mul_301: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, unsqueeze_639);  where_16 = unsqueeze_639 = None
    mul_302: "f32[256]" = torch.ops.aten.mul.Tensor(sum_37, rsqrt_17);  sum_37 = rsqrt_17 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_301, relu_31, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_301 = primals_106 = None
    getitem_53: "f32[4, 256, 14, 14]" = convolution_backward_17[0]
    getitem_54: "f32[256, 256, 3, 3]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    alias_101: "f32[4, 256, 14, 14]" = torch.ops.aten.alias.default(relu_31);  relu_31 = None
    alias_102: "f32[4, 256, 14, 14]" = torch.ops.aten.alias.default(alias_101);  alias_101 = None
    le_17: "b8[4, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_102, 0);  alias_102 = None
    scalar_tensor_17: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_17: "f32[4, 256, 14, 14]" = torch.ops.aten.where.self(le_17, scalar_tensor_17, getitem_53);  le_17 = scalar_tensor_17 = getitem_53 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    add_145: "f32[256]" = torch.ops.aten.add.Tensor(primals_265, 1e-05);  primals_265 = None
    rsqrt_18: "f32[256]" = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
    unsqueeze_640: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_264, 0);  primals_264 = None
    unsqueeze_641: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, 2);  unsqueeze_640 = None
    unsqueeze_642: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 3);  unsqueeze_641 = None
    sum_38: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_71: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_642);  convolution_34 = unsqueeze_642 = None
    mul_303: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_17, sub_71);  sub_71 = None
    sum_39: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_303, [0, 2, 3]);  mul_303 = None
    mul_308: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_18, primals_104);  primals_104 = None
    unsqueeze_649: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_308, 0);  mul_308 = None
    unsqueeze_650: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 2);  unsqueeze_649 = None
    unsqueeze_651: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 3);  unsqueeze_650 = None
    mul_309: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_17, unsqueeze_651);  where_17 = unsqueeze_651 = None
    mul_310: "f32[256]" = torch.ops.aten.mul.Tensor(sum_39, rsqrt_18);  sum_39 = rsqrt_18 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_309, relu_30, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_309 = primals_103 = None
    getitem_56: "f32[4, 1024, 14, 14]" = convolution_backward_18[0]
    getitem_57: "f32[256, 1024, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    add_146: "f32[4, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_15, getitem_56);  where_15 = getitem_56 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    alias_104: "f32[4, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_30);  relu_30 = None
    alias_105: "f32[4, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_104);  alias_104 = None
    le_18: "b8[4, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_105, 0);  alias_105 = None
    scalar_tensor_18: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_18: "f32[4, 1024, 14, 14]" = torch.ops.aten.where.self(le_18, scalar_tensor_18, add_146);  le_18 = scalar_tensor_18 = add_146 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    add_147: "f32[1024]" = torch.ops.aten.add.Tensor(primals_262, 1e-05);  primals_262 = None
    rsqrt_19: "f32[1024]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
    unsqueeze_652: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(primals_261, 0);  primals_261 = None
    unsqueeze_653: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, 2);  unsqueeze_652 = None
    unsqueeze_654: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 3);  unsqueeze_653 = None
    sum_40: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_72: "f32[4, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_654);  convolution_33 = unsqueeze_654 = None
    mul_311: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, sub_72);  sub_72 = None
    sum_41: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_311, [0, 2, 3]);  mul_311 = None
    mul_316: "f32[1024]" = torch.ops.aten.mul.Tensor(rsqrt_19, primals_101);  primals_101 = None
    unsqueeze_661: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_316, 0);  mul_316 = None
    unsqueeze_662: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 2);  unsqueeze_661 = None
    unsqueeze_663: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 3);  unsqueeze_662 = None
    mul_317: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, unsqueeze_663);  unsqueeze_663 = None
    mul_318: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_41, rsqrt_19);  sum_41 = rsqrt_19 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_317, relu_29, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_317 = primals_100 = None
    getitem_59: "f32[4, 256, 14, 14]" = convolution_backward_19[0]
    getitem_60: "f32[1024, 256, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    alias_107: "f32[4, 256, 14, 14]" = torch.ops.aten.alias.default(relu_29);  relu_29 = None
    alias_108: "f32[4, 256, 14, 14]" = torch.ops.aten.alias.default(alias_107);  alias_107 = None
    le_19: "b8[4, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_108, 0);  alias_108 = None
    scalar_tensor_19: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_19: "f32[4, 256, 14, 14]" = torch.ops.aten.where.self(le_19, scalar_tensor_19, getitem_59);  le_19 = scalar_tensor_19 = getitem_59 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    add_148: "f32[256]" = torch.ops.aten.add.Tensor(primals_259, 1e-05);  primals_259 = None
    rsqrt_20: "f32[256]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
    unsqueeze_664: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_258, 0);  primals_258 = None
    unsqueeze_665: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, 2);  unsqueeze_664 = None
    unsqueeze_666: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 3);  unsqueeze_665 = None
    sum_42: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_73: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_666);  convolution_32 = unsqueeze_666 = None
    mul_319: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, sub_73);  sub_73 = None
    sum_43: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_319, [0, 2, 3]);  mul_319 = None
    mul_324: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_20, primals_98);  primals_98 = None
    unsqueeze_673: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_324, 0);  mul_324 = None
    unsqueeze_674: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 2);  unsqueeze_673 = None
    unsqueeze_675: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 3);  unsqueeze_674 = None
    mul_325: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, unsqueeze_675);  where_19 = unsqueeze_675 = None
    mul_326: "f32[256]" = torch.ops.aten.mul.Tensor(sum_43, rsqrt_20);  sum_43 = rsqrt_20 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_325, relu_28, primals_97, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_325 = primals_97 = None
    getitem_62: "f32[4, 256, 14, 14]" = convolution_backward_20[0]
    getitem_63: "f32[256, 256, 3, 3]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    alias_110: "f32[4, 256, 14, 14]" = torch.ops.aten.alias.default(relu_28);  relu_28 = None
    alias_111: "f32[4, 256, 14, 14]" = torch.ops.aten.alias.default(alias_110);  alias_110 = None
    le_20: "b8[4, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_111, 0);  alias_111 = None
    scalar_tensor_20: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_20: "f32[4, 256, 14, 14]" = torch.ops.aten.where.self(le_20, scalar_tensor_20, getitem_62);  le_20 = scalar_tensor_20 = getitem_62 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    add_149: "f32[256]" = torch.ops.aten.add.Tensor(primals_256, 1e-05);  primals_256 = None
    rsqrt_21: "f32[256]" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
    unsqueeze_676: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_255, 0);  primals_255 = None
    unsqueeze_677: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, 2);  unsqueeze_676 = None
    unsqueeze_678: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 3);  unsqueeze_677 = None
    sum_44: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_74: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_678);  convolution_31 = unsqueeze_678 = None
    mul_327: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_20, sub_74);  sub_74 = None
    sum_45: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_327, [0, 2, 3]);  mul_327 = None
    mul_332: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_21, primals_95);  primals_95 = None
    unsqueeze_685: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_332, 0);  mul_332 = None
    unsqueeze_686: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 2);  unsqueeze_685 = None
    unsqueeze_687: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 3);  unsqueeze_686 = None
    mul_333: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_20, unsqueeze_687);  where_20 = unsqueeze_687 = None
    mul_334: "f32[256]" = torch.ops.aten.mul.Tensor(sum_45, rsqrt_21);  sum_45 = rsqrt_21 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_333, relu_27, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_333 = primals_94 = None
    getitem_65: "f32[4, 1024, 14, 14]" = convolution_backward_21[0]
    getitem_66: "f32[256, 1024, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    add_150: "f32[4, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_18, getitem_65);  where_18 = getitem_65 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    alias_113: "f32[4, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_27);  relu_27 = None
    alias_114: "f32[4, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_113);  alias_113 = None
    le_21: "b8[4, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_114, 0);  alias_114 = None
    scalar_tensor_21: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_21: "f32[4, 1024, 14, 14]" = torch.ops.aten.where.self(le_21, scalar_tensor_21, add_150);  le_21 = scalar_tensor_21 = add_150 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    add_151: "f32[1024]" = torch.ops.aten.add.Tensor(primals_253, 1e-05);  primals_253 = None
    rsqrt_22: "f32[1024]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
    unsqueeze_688: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(primals_252, 0);  primals_252 = None
    unsqueeze_689: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, 2);  unsqueeze_688 = None
    unsqueeze_690: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 3);  unsqueeze_689 = None
    sum_46: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_75: "f32[4, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_690);  convolution_30 = unsqueeze_690 = None
    mul_335: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_21, sub_75);  sub_75 = None
    sum_47: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_335, [0, 2, 3]);  mul_335 = None
    mul_340: "f32[1024]" = torch.ops.aten.mul.Tensor(rsqrt_22, primals_92);  primals_92 = None
    unsqueeze_697: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_340, 0);  mul_340 = None
    unsqueeze_698: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 2);  unsqueeze_697 = None
    unsqueeze_699: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 3);  unsqueeze_698 = None
    mul_341: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_21, unsqueeze_699);  unsqueeze_699 = None
    mul_342: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_47, rsqrt_22);  sum_47 = rsqrt_22 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_341, relu_26, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_341 = primals_91 = None
    getitem_68: "f32[4, 256, 14, 14]" = convolution_backward_22[0]
    getitem_69: "f32[1024, 256, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    alias_116: "f32[4, 256, 14, 14]" = torch.ops.aten.alias.default(relu_26);  relu_26 = None
    alias_117: "f32[4, 256, 14, 14]" = torch.ops.aten.alias.default(alias_116);  alias_116 = None
    le_22: "b8[4, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_117, 0);  alias_117 = None
    scalar_tensor_22: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_22: "f32[4, 256, 14, 14]" = torch.ops.aten.where.self(le_22, scalar_tensor_22, getitem_68);  le_22 = scalar_tensor_22 = getitem_68 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    add_152: "f32[256]" = torch.ops.aten.add.Tensor(primals_250, 1e-05);  primals_250 = None
    rsqrt_23: "f32[256]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
    unsqueeze_700: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_249, 0);  primals_249 = None
    unsqueeze_701: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, 2);  unsqueeze_700 = None
    unsqueeze_702: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 3);  unsqueeze_701 = None
    sum_48: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_76: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_702);  convolution_29 = unsqueeze_702 = None
    mul_343: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_22, sub_76);  sub_76 = None
    sum_49: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_343, [0, 2, 3]);  mul_343 = None
    mul_348: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_23, primals_89);  primals_89 = None
    unsqueeze_709: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_348, 0);  mul_348 = None
    unsqueeze_710: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 2);  unsqueeze_709 = None
    unsqueeze_711: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 3);  unsqueeze_710 = None
    mul_349: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_22, unsqueeze_711);  where_22 = unsqueeze_711 = None
    mul_350: "f32[256]" = torch.ops.aten.mul.Tensor(sum_49, rsqrt_23);  sum_49 = rsqrt_23 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_349, relu_25, primals_88, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_349 = primals_88 = None
    getitem_71: "f32[4, 256, 14, 14]" = convolution_backward_23[0]
    getitem_72: "f32[256, 256, 3, 3]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    alias_119: "f32[4, 256, 14, 14]" = torch.ops.aten.alias.default(relu_25);  relu_25 = None
    alias_120: "f32[4, 256, 14, 14]" = torch.ops.aten.alias.default(alias_119);  alias_119 = None
    le_23: "b8[4, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_120, 0);  alias_120 = None
    scalar_tensor_23: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_23: "f32[4, 256, 14, 14]" = torch.ops.aten.where.self(le_23, scalar_tensor_23, getitem_71);  le_23 = scalar_tensor_23 = getitem_71 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    add_153: "f32[256]" = torch.ops.aten.add.Tensor(primals_247, 1e-05);  primals_247 = None
    rsqrt_24: "f32[256]" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
    unsqueeze_712: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_246, 0);  primals_246 = None
    unsqueeze_713: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, 2);  unsqueeze_712 = None
    unsqueeze_714: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 3);  unsqueeze_713 = None
    sum_50: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_77: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_714);  convolution_28 = unsqueeze_714 = None
    mul_351: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_23, sub_77);  sub_77 = None
    sum_51: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_351, [0, 2, 3]);  mul_351 = None
    mul_356: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_24, primals_86);  primals_86 = None
    unsqueeze_721: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_356, 0);  mul_356 = None
    unsqueeze_722: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 2);  unsqueeze_721 = None
    unsqueeze_723: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 3);  unsqueeze_722 = None
    mul_357: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_23, unsqueeze_723);  where_23 = unsqueeze_723 = None
    mul_358: "f32[256]" = torch.ops.aten.mul.Tensor(sum_51, rsqrt_24);  sum_51 = rsqrt_24 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_357, relu_24, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_357 = primals_85 = None
    getitem_74: "f32[4, 1024, 14, 14]" = convolution_backward_24[0]
    getitem_75: "f32[256, 1024, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    add_154: "f32[4, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_21, getitem_74);  where_21 = getitem_74 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    alias_122: "f32[4, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_24);  relu_24 = None
    alias_123: "f32[4, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_122);  alias_122 = None
    le_24: "b8[4, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_123, 0);  alias_123 = None
    scalar_tensor_24: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_24: "f32[4, 1024, 14, 14]" = torch.ops.aten.where.self(le_24, scalar_tensor_24, add_154);  le_24 = scalar_tensor_24 = add_154 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:158, code: identity = self.downsample(x)
    add_155: "f32[1024]" = torch.ops.aten.add.Tensor(primals_244, 1e-05);  primals_244 = None
    rsqrt_25: "f32[1024]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
    unsqueeze_724: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(primals_243, 0);  primals_243 = None
    unsqueeze_725: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, 2);  unsqueeze_724 = None
    unsqueeze_726: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 3);  unsqueeze_725 = None
    sum_52: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_78: "f32[4, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_726);  convolution_27 = unsqueeze_726 = None
    mul_359: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_24, sub_78);  sub_78 = None
    sum_53: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_359, [0, 2, 3]);  mul_359 = None
    mul_364: "f32[1024]" = torch.ops.aten.mul.Tensor(rsqrt_25, primals_83);  primals_83 = None
    unsqueeze_733: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_364, 0);  mul_364 = None
    unsqueeze_734: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 2);  unsqueeze_733 = None
    unsqueeze_735: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 3);  unsqueeze_734 = None
    mul_365: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_24, unsqueeze_735);  unsqueeze_735 = None
    mul_366: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_53, rsqrt_25);  sum_53 = rsqrt_25 = None
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_365, relu_21, primals_82, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_365 = primals_82 = None
    getitem_77: "f32[4, 512, 28, 28]" = convolution_backward_25[0]
    getitem_78: "f32[1024, 512, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    add_156: "f32[1024]" = torch.ops.aten.add.Tensor(primals_241, 1e-05);  primals_241 = None
    rsqrt_26: "f32[1024]" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
    unsqueeze_736: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(primals_240, 0);  primals_240 = None
    unsqueeze_737: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, 2);  unsqueeze_736 = None
    unsqueeze_738: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 3);  unsqueeze_737 = None
    sum_54: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_79: "f32[4, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_738);  convolution_26 = unsqueeze_738 = None
    mul_367: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_24, sub_79);  sub_79 = None
    sum_55: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_367, [0, 2, 3]);  mul_367 = None
    mul_372: "f32[1024]" = torch.ops.aten.mul.Tensor(rsqrt_26, primals_80);  primals_80 = None
    unsqueeze_745: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_372, 0);  mul_372 = None
    unsqueeze_746: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_745, 2);  unsqueeze_745 = None
    unsqueeze_747: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 3);  unsqueeze_746 = None
    mul_373: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_24, unsqueeze_747);  where_24 = unsqueeze_747 = None
    mul_374: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_55, rsqrt_26);  sum_55 = rsqrt_26 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_373, relu_23, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_373 = primals_79 = None
    getitem_80: "f32[4, 256, 14, 14]" = convolution_backward_26[0]
    getitem_81: "f32[1024, 256, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    alias_125: "f32[4, 256, 14, 14]" = torch.ops.aten.alias.default(relu_23);  relu_23 = None
    alias_126: "f32[4, 256, 14, 14]" = torch.ops.aten.alias.default(alias_125);  alias_125 = None
    le_25: "b8[4, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_126, 0);  alias_126 = None
    scalar_tensor_25: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_25: "f32[4, 256, 14, 14]" = torch.ops.aten.where.self(le_25, scalar_tensor_25, getitem_80);  le_25 = scalar_tensor_25 = getitem_80 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    add_157: "f32[256]" = torch.ops.aten.add.Tensor(primals_238, 1e-05);  primals_238 = None
    rsqrt_27: "f32[256]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
    unsqueeze_748: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_237, 0);  primals_237 = None
    unsqueeze_749: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, 2);  unsqueeze_748 = None
    unsqueeze_750: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 3);  unsqueeze_749 = None
    sum_56: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_80: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_750);  convolution_25 = unsqueeze_750 = None
    mul_375: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_25, sub_80);  sub_80 = None
    sum_57: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_375, [0, 2, 3]);  mul_375 = None
    mul_380: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_27, primals_77);  primals_77 = None
    unsqueeze_757: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_380, 0);  mul_380 = None
    unsqueeze_758: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_757, 2);  unsqueeze_757 = None
    unsqueeze_759: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 3);  unsqueeze_758 = None
    mul_381: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_25, unsqueeze_759);  where_25 = unsqueeze_759 = None
    mul_382: "f32[256]" = torch.ops.aten.mul.Tensor(sum_57, rsqrt_27);  sum_57 = rsqrt_27 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_381, relu_22, primals_76, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_381 = primals_76 = None
    getitem_83: "f32[4, 256, 28, 28]" = convolution_backward_27[0]
    getitem_84: "f32[256, 256, 3, 3]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    alias_128: "f32[4, 256, 28, 28]" = torch.ops.aten.alias.default(relu_22);  relu_22 = None
    alias_129: "f32[4, 256, 28, 28]" = torch.ops.aten.alias.default(alias_128);  alias_128 = None
    le_26: "b8[4, 256, 28, 28]" = torch.ops.aten.le.Scalar(alias_129, 0);  alias_129 = None
    scalar_tensor_26: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_26: "f32[4, 256, 28, 28]" = torch.ops.aten.where.self(le_26, scalar_tensor_26, getitem_83);  le_26 = scalar_tensor_26 = getitem_83 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    add_158: "f32[256]" = torch.ops.aten.add.Tensor(primals_235, 1e-05);  primals_235 = None
    rsqrt_28: "f32[256]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
    unsqueeze_760: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_234, 0);  primals_234 = None
    unsqueeze_761: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, 2);  unsqueeze_760 = None
    unsqueeze_762: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 3);  unsqueeze_761 = None
    sum_58: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_81: "f32[4, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_762);  convolution_24 = unsqueeze_762 = None
    mul_383: "f32[4, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_26, sub_81);  sub_81 = None
    sum_59: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_383, [0, 2, 3]);  mul_383 = None
    mul_388: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_28, primals_74);  primals_74 = None
    unsqueeze_769: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_388, 0);  mul_388 = None
    unsqueeze_770: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_769, 2);  unsqueeze_769 = None
    unsqueeze_771: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 3);  unsqueeze_770 = None
    mul_389: "f32[4, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_26, unsqueeze_771);  where_26 = unsqueeze_771 = None
    mul_390: "f32[256]" = torch.ops.aten.mul.Tensor(sum_59, rsqrt_28);  sum_59 = rsqrt_28 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_389, relu_21, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_389 = primals_73 = None
    getitem_86: "f32[4, 512, 28, 28]" = convolution_backward_28[0]
    getitem_87: "f32[256, 512, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    add_159: "f32[4, 512, 28, 28]" = torch.ops.aten.add.Tensor(getitem_77, getitem_86);  getitem_77 = getitem_86 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    alias_131: "f32[4, 512, 28, 28]" = torch.ops.aten.alias.default(relu_21);  relu_21 = None
    alias_132: "f32[4, 512, 28, 28]" = torch.ops.aten.alias.default(alias_131);  alias_131 = None
    le_27: "b8[4, 512, 28, 28]" = torch.ops.aten.le.Scalar(alias_132, 0);  alias_132 = None
    scalar_tensor_27: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_27: "f32[4, 512, 28, 28]" = torch.ops.aten.where.self(le_27, scalar_tensor_27, add_159);  le_27 = scalar_tensor_27 = add_159 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    add_160: "f32[512]" = torch.ops.aten.add.Tensor(primals_232, 1e-05);  primals_232 = None
    rsqrt_29: "f32[512]" = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
    unsqueeze_772: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_231, 0);  primals_231 = None
    unsqueeze_773: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, 2);  unsqueeze_772 = None
    unsqueeze_774: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 3);  unsqueeze_773 = None
    sum_60: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_82: "f32[4, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_774);  convolution_23 = unsqueeze_774 = None
    mul_391: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_27, sub_82);  sub_82 = None
    sum_61: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_391, [0, 2, 3]);  mul_391 = None
    mul_396: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_29, primals_71);  primals_71 = None
    unsqueeze_781: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_396, 0);  mul_396 = None
    unsqueeze_782: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_781, 2);  unsqueeze_781 = None
    unsqueeze_783: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 3);  unsqueeze_782 = None
    mul_397: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_27, unsqueeze_783);  unsqueeze_783 = None
    mul_398: "f32[512]" = torch.ops.aten.mul.Tensor(sum_61, rsqrt_29);  sum_61 = rsqrt_29 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_397, relu_20, primals_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_397 = primals_70 = None
    getitem_89: "f32[4, 128, 28, 28]" = convolution_backward_29[0]
    getitem_90: "f32[512, 128, 1, 1]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    alias_134: "f32[4, 128, 28, 28]" = torch.ops.aten.alias.default(relu_20);  relu_20 = None
    alias_135: "f32[4, 128, 28, 28]" = torch.ops.aten.alias.default(alias_134);  alias_134 = None
    le_28: "b8[4, 128, 28, 28]" = torch.ops.aten.le.Scalar(alias_135, 0);  alias_135 = None
    scalar_tensor_28: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_28: "f32[4, 128, 28, 28]" = torch.ops.aten.where.self(le_28, scalar_tensor_28, getitem_89);  le_28 = scalar_tensor_28 = getitem_89 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    add_161: "f32[128]" = torch.ops.aten.add.Tensor(primals_229, 1e-05);  primals_229 = None
    rsqrt_30: "f32[128]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    unsqueeze_784: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_228, 0);  primals_228 = None
    unsqueeze_785: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, 2);  unsqueeze_784 = None
    unsqueeze_786: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 3);  unsqueeze_785 = None
    sum_62: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    sub_83: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_786);  convolution_22 = unsqueeze_786 = None
    mul_399: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_28, sub_83);  sub_83 = None
    sum_63: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_399, [0, 2, 3]);  mul_399 = None
    mul_404: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_30, primals_68);  primals_68 = None
    unsqueeze_793: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_404, 0);  mul_404 = None
    unsqueeze_794: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_793, 2);  unsqueeze_793 = None
    unsqueeze_795: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 3);  unsqueeze_794 = None
    mul_405: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_28, unsqueeze_795);  where_28 = unsqueeze_795 = None
    mul_406: "f32[128]" = torch.ops.aten.mul.Tensor(sum_63, rsqrt_30);  sum_63 = rsqrt_30 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_405, relu_19, primals_67, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_405 = primals_67 = None
    getitem_92: "f32[4, 128, 28, 28]" = convolution_backward_30[0]
    getitem_93: "f32[128, 128, 3, 3]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    alias_137: "f32[4, 128, 28, 28]" = torch.ops.aten.alias.default(relu_19);  relu_19 = None
    alias_138: "f32[4, 128, 28, 28]" = torch.ops.aten.alias.default(alias_137);  alias_137 = None
    le_29: "b8[4, 128, 28, 28]" = torch.ops.aten.le.Scalar(alias_138, 0);  alias_138 = None
    scalar_tensor_29: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_29: "f32[4, 128, 28, 28]" = torch.ops.aten.where.self(le_29, scalar_tensor_29, getitem_92);  le_29 = scalar_tensor_29 = getitem_92 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    add_162: "f32[128]" = torch.ops.aten.add.Tensor(primals_226, 1e-05);  primals_226 = None
    rsqrt_31: "f32[128]" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
    unsqueeze_796: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_225, 0);  primals_225 = None
    unsqueeze_797: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, 2);  unsqueeze_796 = None
    unsqueeze_798: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 3);  unsqueeze_797 = None
    sum_64: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    sub_84: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_798);  convolution_21 = unsqueeze_798 = None
    mul_407: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_29, sub_84);  sub_84 = None
    sum_65: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_407, [0, 2, 3]);  mul_407 = None
    mul_412: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_31, primals_65);  primals_65 = None
    unsqueeze_805: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_412, 0);  mul_412 = None
    unsqueeze_806: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_805, 2);  unsqueeze_805 = None
    unsqueeze_807: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 3);  unsqueeze_806 = None
    mul_413: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_29, unsqueeze_807);  where_29 = unsqueeze_807 = None
    mul_414: "f32[128]" = torch.ops.aten.mul.Tensor(sum_65, rsqrt_31);  sum_65 = rsqrt_31 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_413, relu_18, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_413 = primals_64 = None
    getitem_95: "f32[4, 512, 28, 28]" = convolution_backward_31[0]
    getitem_96: "f32[128, 512, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    add_163: "f32[4, 512, 28, 28]" = torch.ops.aten.add.Tensor(where_27, getitem_95);  where_27 = getitem_95 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    alias_140: "f32[4, 512, 28, 28]" = torch.ops.aten.alias.default(relu_18);  relu_18 = None
    alias_141: "f32[4, 512, 28, 28]" = torch.ops.aten.alias.default(alias_140);  alias_140 = None
    le_30: "b8[4, 512, 28, 28]" = torch.ops.aten.le.Scalar(alias_141, 0);  alias_141 = None
    scalar_tensor_30: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_30: "f32[4, 512, 28, 28]" = torch.ops.aten.where.self(le_30, scalar_tensor_30, add_163);  le_30 = scalar_tensor_30 = add_163 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    add_164: "f32[512]" = torch.ops.aten.add.Tensor(primals_223, 1e-05);  primals_223 = None
    rsqrt_32: "f32[512]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
    unsqueeze_808: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_222, 0);  primals_222 = None
    unsqueeze_809: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, 2);  unsqueeze_808 = None
    unsqueeze_810: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 3);  unsqueeze_809 = None
    sum_66: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
    sub_85: "f32[4, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_810);  convolution_20 = unsqueeze_810 = None
    mul_415: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_30, sub_85);  sub_85 = None
    sum_67: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_415, [0, 2, 3]);  mul_415 = None
    mul_420: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_32, primals_62);  primals_62 = None
    unsqueeze_817: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_420, 0);  mul_420 = None
    unsqueeze_818: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_817, 2);  unsqueeze_817 = None
    unsqueeze_819: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 3);  unsqueeze_818 = None
    mul_421: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_30, unsqueeze_819);  unsqueeze_819 = None
    mul_422: "f32[512]" = torch.ops.aten.mul.Tensor(sum_67, rsqrt_32);  sum_67 = rsqrt_32 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_421, relu_17, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_421 = primals_61 = None
    getitem_98: "f32[4, 128, 28, 28]" = convolution_backward_32[0]
    getitem_99: "f32[512, 128, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    alias_143: "f32[4, 128, 28, 28]" = torch.ops.aten.alias.default(relu_17);  relu_17 = None
    alias_144: "f32[4, 128, 28, 28]" = torch.ops.aten.alias.default(alias_143);  alias_143 = None
    le_31: "b8[4, 128, 28, 28]" = torch.ops.aten.le.Scalar(alias_144, 0);  alias_144 = None
    scalar_tensor_31: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_31: "f32[4, 128, 28, 28]" = torch.ops.aten.where.self(le_31, scalar_tensor_31, getitem_98);  le_31 = scalar_tensor_31 = getitem_98 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    add_165: "f32[128]" = torch.ops.aten.add.Tensor(primals_220, 1e-05);  primals_220 = None
    rsqrt_33: "f32[128]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
    unsqueeze_820: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_219, 0);  primals_219 = None
    unsqueeze_821: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, 2);  unsqueeze_820 = None
    unsqueeze_822: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 3);  unsqueeze_821 = None
    sum_68: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_86: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_822);  convolution_19 = unsqueeze_822 = None
    mul_423: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_31, sub_86);  sub_86 = None
    sum_69: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_423, [0, 2, 3]);  mul_423 = None
    mul_428: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_33, primals_59);  primals_59 = None
    unsqueeze_829: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_428, 0);  mul_428 = None
    unsqueeze_830: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_829, 2);  unsqueeze_829 = None
    unsqueeze_831: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 3);  unsqueeze_830 = None
    mul_429: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_31, unsqueeze_831);  where_31 = unsqueeze_831 = None
    mul_430: "f32[128]" = torch.ops.aten.mul.Tensor(sum_69, rsqrt_33);  sum_69 = rsqrt_33 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_429, relu_16, primals_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_429 = primals_58 = None
    getitem_101: "f32[4, 128, 28, 28]" = convolution_backward_33[0]
    getitem_102: "f32[128, 128, 3, 3]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    alias_146: "f32[4, 128, 28, 28]" = torch.ops.aten.alias.default(relu_16);  relu_16 = None
    alias_147: "f32[4, 128, 28, 28]" = torch.ops.aten.alias.default(alias_146);  alias_146 = None
    le_32: "b8[4, 128, 28, 28]" = torch.ops.aten.le.Scalar(alias_147, 0);  alias_147 = None
    scalar_tensor_32: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_32: "f32[4, 128, 28, 28]" = torch.ops.aten.where.self(le_32, scalar_tensor_32, getitem_101);  le_32 = scalar_tensor_32 = getitem_101 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    add_166: "f32[128]" = torch.ops.aten.add.Tensor(primals_217, 1e-05);  primals_217 = None
    rsqrt_34: "f32[128]" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
    unsqueeze_832: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_216, 0);  primals_216 = None
    unsqueeze_833: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, 2);  unsqueeze_832 = None
    unsqueeze_834: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 3);  unsqueeze_833 = None
    sum_70: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
    sub_87: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_834);  convolution_18 = unsqueeze_834 = None
    mul_431: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_32, sub_87);  sub_87 = None
    sum_71: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_431, [0, 2, 3]);  mul_431 = None
    mul_436: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_34, primals_56);  primals_56 = None
    unsqueeze_841: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_436, 0);  mul_436 = None
    unsqueeze_842: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_841, 2);  unsqueeze_841 = None
    unsqueeze_843: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, 3);  unsqueeze_842 = None
    mul_437: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_32, unsqueeze_843);  where_32 = unsqueeze_843 = None
    mul_438: "f32[128]" = torch.ops.aten.mul.Tensor(sum_71, rsqrt_34);  sum_71 = rsqrt_34 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_437, relu_15, primals_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_437 = primals_55 = None
    getitem_104: "f32[4, 512, 28, 28]" = convolution_backward_34[0]
    getitem_105: "f32[128, 512, 1, 1]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    add_167: "f32[4, 512, 28, 28]" = torch.ops.aten.add.Tensor(where_30, getitem_104);  where_30 = getitem_104 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    alias_149: "f32[4, 512, 28, 28]" = torch.ops.aten.alias.default(relu_15);  relu_15 = None
    alias_150: "f32[4, 512, 28, 28]" = torch.ops.aten.alias.default(alias_149);  alias_149 = None
    le_33: "b8[4, 512, 28, 28]" = torch.ops.aten.le.Scalar(alias_150, 0);  alias_150 = None
    scalar_tensor_33: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_33: "f32[4, 512, 28, 28]" = torch.ops.aten.where.self(le_33, scalar_tensor_33, add_167);  le_33 = scalar_tensor_33 = add_167 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    add_168: "f32[512]" = torch.ops.aten.add.Tensor(primals_214, 1e-05);  primals_214 = None
    rsqrt_35: "f32[512]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    unsqueeze_844: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_213, 0);  primals_213 = None
    unsqueeze_845: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_844, 2);  unsqueeze_844 = None
    unsqueeze_846: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 3);  unsqueeze_845 = None
    sum_72: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    sub_88: "f32[4, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_846);  convolution_17 = unsqueeze_846 = None
    mul_439: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_33, sub_88);  sub_88 = None
    sum_73: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_439, [0, 2, 3]);  mul_439 = None
    mul_444: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_35, primals_53);  primals_53 = None
    unsqueeze_853: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_444, 0);  mul_444 = None
    unsqueeze_854: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_853, 2);  unsqueeze_853 = None
    unsqueeze_855: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, 3);  unsqueeze_854 = None
    mul_445: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_33, unsqueeze_855);  unsqueeze_855 = None
    mul_446: "f32[512]" = torch.ops.aten.mul.Tensor(sum_73, rsqrt_35);  sum_73 = rsqrt_35 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_445, relu_14, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_445 = primals_52 = None
    getitem_107: "f32[4, 128, 28, 28]" = convolution_backward_35[0]
    getitem_108: "f32[512, 128, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    alias_152: "f32[4, 128, 28, 28]" = torch.ops.aten.alias.default(relu_14);  relu_14 = None
    alias_153: "f32[4, 128, 28, 28]" = torch.ops.aten.alias.default(alias_152);  alias_152 = None
    le_34: "b8[4, 128, 28, 28]" = torch.ops.aten.le.Scalar(alias_153, 0);  alias_153 = None
    scalar_tensor_34: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_34: "f32[4, 128, 28, 28]" = torch.ops.aten.where.self(le_34, scalar_tensor_34, getitem_107);  le_34 = scalar_tensor_34 = getitem_107 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    add_169: "f32[128]" = torch.ops.aten.add.Tensor(primals_211, 1e-05);  primals_211 = None
    rsqrt_36: "f32[128]" = torch.ops.aten.rsqrt.default(add_169);  add_169 = None
    unsqueeze_856: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_210, 0);  primals_210 = None
    unsqueeze_857: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, 2);  unsqueeze_856 = None
    unsqueeze_858: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 3);  unsqueeze_857 = None
    sum_74: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
    sub_89: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_858);  convolution_16 = unsqueeze_858 = None
    mul_447: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_34, sub_89);  sub_89 = None
    sum_75: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_447, [0, 2, 3]);  mul_447 = None
    mul_452: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_36, primals_50);  primals_50 = None
    unsqueeze_865: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_452, 0);  mul_452 = None
    unsqueeze_866: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_865, 2);  unsqueeze_865 = None
    unsqueeze_867: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, 3);  unsqueeze_866 = None
    mul_453: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_34, unsqueeze_867);  where_34 = unsqueeze_867 = None
    mul_454: "f32[128]" = torch.ops.aten.mul.Tensor(sum_75, rsqrt_36);  sum_75 = rsqrt_36 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_453, relu_13, primals_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_453 = primals_49 = None
    getitem_110: "f32[4, 128, 28, 28]" = convolution_backward_36[0]
    getitem_111: "f32[128, 128, 3, 3]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    alias_155: "f32[4, 128, 28, 28]" = torch.ops.aten.alias.default(relu_13);  relu_13 = None
    alias_156: "f32[4, 128, 28, 28]" = torch.ops.aten.alias.default(alias_155);  alias_155 = None
    le_35: "b8[4, 128, 28, 28]" = torch.ops.aten.le.Scalar(alias_156, 0);  alias_156 = None
    scalar_tensor_35: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_35: "f32[4, 128, 28, 28]" = torch.ops.aten.where.self(le_35, scalar_tensor_35, getitem_110);  le_35 = scalar_tensor_35 = getitem_110 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    add_170: "f32[128]" = torch.ops.aten.add.Tensor(primals_208, 1e-05);  primals_208 = None
    rsqrt_37: "f32[128]" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
    unsqueeze_868: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_207, 0);  primals_207 = None
    unsqueeze_869: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, 2);  unsqueeze_868 = None
    unsqueeze_870: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 3);  unsqueeze_869 = None
    sum_76: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    sub_90: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_870);  convolution_15 = unsqueeze_870 = None
    mul_455: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_35, sub_90);  sub_90 = None
    sum_77: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_455, [0, 2, 3]);  mul_455 = None
    mul_460: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_37, primals_47);  primals_47 = None
    unsqueeze_877: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_460, 0);  mul_460 = None
    unsqueeze_878: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_877, 2);  unsqueeze_877 = None
    unsqueeze_879: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, 3);  unsqueeze_878 = None
    mul_461: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_35, unsqueeze_879);  where_35 = unsqueeze_879 = None
    mul_462: "f32[128]" = torch.ops.aten.mul.Tensor(sum_77, rsqrt_37);  sum_77 = rsqrt_37 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_461, relu_12, primals_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_461 = primals_46 = None
    getitem_113: "f32[4, 512, 28, 28]" = convolution_backward_37[0]
    getitem_114: "f32[128, 512, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    add_171: "f32[4, 512, 28, 28]" = torch.ops.aten.add.Tensor(where_33, getitem_113);  where_33 = getitem_113 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    alias_158: "f32[4, 512, 28, 28]" = torch.ops.aten.alias.default(relu_12);  relu_12 = None
    alias_159: "f32[4, 512, 28, 28]" = torch.ops.aten.alias.default(alias_158);  alias_158 = None
    le_36: "b8[4, 512, 28, 28]" = torch.ops.aten.le.Scalar(alias_159, 0);  alias_159 = None
    scalar_tensor_36: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_36: "f32[4, 512, 28, 28]" = torch.ops.aten.where.self(le_36, scalar_tensor_36, add_171);  le_36 = scalar_tensor_36 = add_171 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:158, code: identity = self.downsample(x)
    add_172: "f32[512]" = torch.ops.aten.add.Tensor(primals_205, 1e-05);  primals_205 = None
    rsqrt_38: "f32[512]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
    unsqueeze_880: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_204, 0);  primals_204 = None
    unsqueeze_881: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_880, 2);  unsqueeze_880 = None
    unsqueeze_882: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 3);  unsqueeze_881 = None
    sum_78: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
    sub_91: "f32[4, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_882);  convolution_14 = unsqueeze_882 = None
    mul_463: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_36, sub_91);  sub_91 = None
    sum_79: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_463, [0, 2, 3]);  mul_463 = None
    mul_468: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_38, primals_44);  primals_44 = None
    unsqueeze_889: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_468, 0);  mul_468 = None
    unsqueeze_890: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_889, 2);  unsqueeze_889 = None
    unsqueeze_891: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, 3);  unsqueeze_890 = None
    mul_469: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_36, unsqueeze_891);  unsqueeze_891 = None
    mul_470: "f32[512]" = torch.ops.aten.mul.Tensor(sum_79, rsqrt_38);  sum_79 = rsqrt_38 = None
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_469, relu_9, primals_43, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_469 = primals_43 = None
    getitem_116: "f32[4, 256, 56, 56]" = convolution_backward_38[0]
    getitem_117: "f32[512, 256, 1, 1]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    add_173: "f32[512]" = torch.ops.aten.add.Tensor(primals_202, 1e-05);  primals_202 = None
    rsqrt_39: "f32[512]" = torch.ops.aten.rsqrt.default(add_173);  add_173 = None
    unsqueeze_892: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_201, 0);  primals_201 = None
    unsqueeze_893: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_892, 2);  unsqueeze_892 = None
    unsqueeze_894: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_893, 3);  unsqueeze_893 = None
    sum_80: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
    sub_92: "f32[4, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_894);  convolution_13 = unsqueeze_894 = None
    mul_471: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_36, sub_92);  sub_92 = None
    sum_81: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_471, [0, 2, 3]);  mul_471 = None
    mul_476: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_39, primals_41);  primals_41 = None
    unsqueeze_901: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_476, 0);  mul_476 = None
    unsqueeze_902: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_901, 2);  unsqueeze_901 = None
    unsqueeze_903: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, 3);  unsqueeze_902 = None
    mul_477: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_36, unsqueeze_903);  where_36 = unsqueeze_903 = None
    mul_478: "f32[512]" = torch.ops.aten.mul.Tensor(sum_81, rsqrt_39);  sum_81 = rsqrt_39 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_477, relu_11, primals_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_477 = primals_40 = None
    getitem_119: "f32[4, 128, 28, 28]" = convolution_backward_39[0]
    getitem_120: "f32[512, 128, 1, 1]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    alias_161: "f32[4, 128, 28, 28]" = torch.ops.aten.alias.default(relu_11);  relu_11 = None
    alias_162: "f32[4, 128, 28, 28]" = torch.ops.aten.alias.default(alias_161);  alias_161 = None
    le_37: "b8[4, 128, 28, 28]" = torch.ops.aten.le.Scalar(alias_162, 0);  alias_162 = None
    scalar_tensor_37: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_37: "f32[4, 128, 28, 28]" = torch.ops.aten.where.self(le_37, scalar_tensor_37, getitem_119);  le_37 = scalar_tensor_37 = getitem_119 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    add_174: "f32[128]" = torch.ops.aten.add.Tensor(primals_199, 1e-05);  primals_199 = None
    rsqrt_40: "f32[128]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
    unsqueeze_904: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_198, 0);  primals_198 = None
    unsqueeze_905: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_904, 2);  unsqueeze_904 = None
    unsqueeze_906: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_905, 3);  unsqueeze_905 = None
    sum_82: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
    sub_93: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_906);  convolution_12 = unsqueeze_906 = None
    mul_479: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_37, sub_93);  sub_93 = None
    sum_83: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_479, [0, 2, 3]);  mul_479 = None
    mul_484: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_40, primals_38);  primals_38 = None
    unsqueeze_913: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_484, 0);  mul_484 = None
    unsqueeze_914: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_913, 2);  unsqueeze_913 = None
    unsqueeze_915: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, 3);  unsqueeze_914 = None
    mul_485: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_37, unsqueeze_915);  where_37 = unsqueeze_915 = None
    mul_486: "f32[128]" = torch.ops.aten.mul.Tensor(sum_83, rsqrt_40);  sum_83 = rsqrt_40 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_485, relu_10, primals_37, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_485 = primals_37 = None
    getitem_122: "f32[4, 128, 56, 56]" = convolution_backward_40[0]
    getitem_123: "f32[128, 128, 3, 3]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    alias_164: "f32[4, 128, 56, 56]" = torch.ops.aten.alias.default(relu_10);  relu_10 = None
    alias_165: "f32[4, 128, 56, 56]" = torch.ops.aten.alias.default(alias_164);  alias_164 = None
    le_38: "b8[4, 128, 56, 56]" = torch.ops.aten.le.Scalar(alias_165, 0);  alias_165 = None
    scalar_tensor_38: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_38: "f32[4, 128, 56, 56]" = torch.ops.aten.where.self(le_38, scalar_tensor_38, getitem_122);  le_38 = scalar_tensor_38 = getitem_122 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    add_175: "f32[128]" = torch.ops.aten.add.Tensor(primals_196, 1e-05);  primals_196 = None
    rsqrt_41: "f32[128]" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
    unsqueeze_916: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_195, 0);  primals_195 = None
    unsqueeze_917: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_916, 2);  unsqueeze_916 = None
    unsqueeze_918: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_917, 3);  unsqueeze_917 = None
    sum_84: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
    sub_94: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_918);  convolution_11 = unsqueeze_918 = None
    mul_487: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_38, sub_94);  sub_94 = None
    sum_85: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_487, [0, 2, 3]);  mul_487 = None
    mul_492: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_41, primals_35);  primals_35 = None
    unsqueeze_925: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_492, 0);  mul_492 = None
    unsqueeze_926: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_925, 2);  unsqueeze_925 = None
    unsqueeze_927: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, 3);  unsqueeze_926 = None
    mul_493: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_38, unsqueeze_927);  where_38 = unsqueeze_927 = None
    mul_494: "f32[128]" = torch.ops.aten.mul.Tensor(sum_85, rsqrt_41);  sum_85 = rsqrt_41 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_493, relu_9, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_493 = primals_34 = None
    getitem_125: "f32[4, 256, 56, 56]" = convolution_backward_41[0]
    getitem_126: "f32[128, 256, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    add_176: "f32[4, 256, 56, 56]" = torch.ops.aten.add.Tensor(getitem_116, getitem_125);  getitem_116 = getitem_125 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    alias_167: "f32[4, 256, 56, 56]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
    alias_168: "f32[4, 256, 56, 56]" = torch.ops.aten.alias.default(alias_167);  alias_167 = None
    le_39: "b8[4, 256, 56, 56]" = torch.ops.aten.le.Scalar(alias_168, 0);  alias_168 = None
    scalar_tensor_39: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_39: "f32[4, 256, 56, 56]" = torch.ops.aten.where.self(le_39, scalar_tensor_39, add_176);  le_39 = scalar_tensor_39 = add_176 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    add_177: "f32[256]" = torch.ops.aten.add.Tensor(primals_193, 1e-05);  primals_193 = None
    rsqrt_42: "f32[256]" = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
    unsqueeze_928: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_192, 0);  primals_192 = None
    unsqueeze_929: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_928, 2);  unsqueeze_928 = None
    unsqueeze_930: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_929, 3);  unsqueeze_929 = None
    sum_86: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
    sub_95: "f32[4, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_930);  convolution_10 = unsqueeze_930 = None
    mul_495: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_39, sub_95);  sub_95 = None
    sum_87: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_495, [0, 2, 3]);  mul_495 = None
    mul_500: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_42, primals_32);  primals_32 = None
    unsqueeze_937: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_500, 0);  mul_500 = None
    unsqueeze_938: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_937, 2);  unsqueeze_937 = None
    unsqueeze_939: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_938, 3);  unsqueeze_938 = None
    mul_501: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_39, unsqueeze_939);  unsqueeze_939 = None
    mul_502: "f32[256]" = torch.ops.aten.mul.Tensor(sum_87, rsqrt_42);  sum_87 = rsqrt_42 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_501, relu_8, primals_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_501 = primals_31 = None
    getitem_128: "f32[4, 64, 56, 56]" = convolution_backward_42[0]
    getitem_129: "f32[256, 64, 1, 1]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    alias_170: "f32[4, 64, 56, 56]" = torch.ops.aten.alias.default(relu_8);  relu_8 = None
    alias_171: "f32[4, 64, 56, 56]" = torch.ops.aten.alias.default(alias_170);  alias_170 = None
    le_40: "b8[4, 64, 56, 56]" = torch.ops.aten.le.Scalar(alias_171, 0);  alias_171 = None
    scalar_tensor_40: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_40: "f32[4, 64, 56, 56]" = torch.ops.aten.where.self(le_40, scalar_tensor_40, getitem_128);  le_40 = scalar_tensor_40 = getitem_128 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    add_178: "f32[64]" = torch.ops.aten.add.Tensor(primals_190, 1e-05);  primals_190 = None
    rsqrt_43: "f32[64]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
    unsqueeze_940: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_189, 0);  primals_189 = None
    unsqueeze_941: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_940, 2);  unsqueeze_940 = None
    unsqueeze_942: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_941, 3);  unsqueeze_941 = None
    sum_88: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
    sub_96: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_942);  convolution_9 = unsqueeze_942 = None
    mul_503: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_40, sub_96);  sub_96 = None
    sum_89: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_503, [0, 2, 3]);  mul_503 = None
    mul_508: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_43, primals_29);  primals_29 = None
    unsqueeze_949: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_508, 0);  mul_508 = None
    unsqueeze_950: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_949, 2);  unsqueeze_949 = None
    unsqueeze_951: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_950, 3);  unsqueeze_950 = None
    mul_509: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_40, unsqueeze_951);  where_40 = unsqueeze_951 = None
    mul_510: "f32[64]" = torch.ops.aten.mul.Tensor(sum_89, rsqrt_43);  sum_89 = rsqrt_43 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_509, relu_7, primals_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_509 = primals_28 = None
    getitem_131: "f32[4, 64, 56, 56]" = convolution_backward_43[0]
    getitem_132: "f32[64, 64, 3, 3]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    alias_173: "f32[4, 64, 56, 56]" = torch.ops.aten.alias.default(relu_7);  relu_7 = None
    alias_174: "f32[4, 64, 56, 56]" = torch.ops.aten.alias.default(alias_173);  alias_173 = None
    le_41: "b8[4, 64, 56, 56]" = torch.ops.aten.le.Scalar(alias_174, 0);  alias_174 = None
    scalar_tensor_41: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_41: "f32[4, 64, 56, 56]" = torch.ops.aten.where.self(le_41, scalar_tensor_41, getitem_131);  le_41 = scalar_tensor_41 = getitem_131 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    add_179: "f32[64]" = torch.ops.aten.add.Tensor(primals_187, 1e-05);  primals_187 = None
    rsqrt_44: "f32[64]" = torch.ops.aten.rsqrt.default(add_179);  add_179 = None
    unsqueeze_952: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_186, 0);  primals_186 = None
    unsqueeze_953: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_952, 2);  unsqueeze_952 = None
    unsqueeze_954: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_953, 3);  unsqueeze_953 = None
    sum_90: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
    sub_97: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_954);  convolution_8 = unsqueeze_954 = None
    mul_511: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_41, sub_97);  sub_97 = None
    sum_91: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_511, [0, 2, 3]);  mul_511 = None
    mul_516: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_44, primals_26);  primals_26 = None
    unsqueeze_961: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_516, 0);  mul_516 = None
    unsqueeze_962: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_961, 2);  unsqueeze_961 = None
    unsqueeze_963: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_962, 3);  unsqueeze_962 = None
    mul_517: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_41, unsqueeze_963);  where_41 = unsqueeze_963 = None
    mul_518: "f32[64]" = torch.ops.aten.mul.Tensor(sum_91, rsqrt_44);  sum_91 = rsqrt_44 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_517, relu_6, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_517 = primals_25 = None
    getitem_134: "f32[4, 256, 56, 56]" = convolution_backward_44[0]
    getitem_135: "f32[64, 256, 1, 1]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    add_180: "f32[4, 256, 56, 56]" = torch.ops.aten.add.Tensor(where_39, getitem_134);  where_39 = getitem_134 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    alias_176: "f32[4, 256, 56, 56]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_177: "f32[4, 256, 56, 56]" = torch.ops.aten.alias.default(alias_176);  alias_176 = None
    le_42: "b8[4, 256, 56, 56]" = torch.ops.aten.le.Scalar(alias_177, 0);  alias_177 = None
    scalar_tensor_42: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_42: "f32[4, 256, 56, 56]" = torch.ops.aten.where.self(le_42, scalar_tensor_42, add_180);  le_42 = scalar_tensor_42 = add_180 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    add_181: "f32[256]" = torch.ops.aten.add.Tensor(primals_184, 1e-05);  primals_184 = None
    rsqrt_45: "f32[256]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
    unsqueeze_964: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_183, 0);  primals_183 = None
    unsqueeze_965: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_964, 2);  unsqueeze_964 = None
    unsqueeze_966: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_965, 3);  unsqueeze_965 = None
    sum_92: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_42, [0, 2, 3])
    sub_98: "f32[4, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_966);  convolution_7 = unsqueeze_966 = None
    mul_519: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_42, sub_98);  sub_98 = None
    sum_93: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_519, [0, 2, 3]);  mul_519 = None
    mul_524: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_45, primals_23);  primals_23 = None
    unsqueeze_973: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_524, 0);  mul_524 = None
    unsqueeze_974: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_973, 2);  unsqueeze_973 = None
    unsqueeze_975: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_974, 3);  unsqueeze_974 = None
    mul_525: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_42, unsqueeze_975);  unsqueeze_975 = None
    mul_526: "f32[256]" = torch.ops.aten.mul.Tensor(sum_93, rsqrt_45);  sum_93 = rsqrt_45 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_525, relu_5, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_525 = primals_22 = None
    getitem_137: "f32[4, 64, 56, 56]" = convolution_backward_45[0]
    getitem_138: "f32[256, 64, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    alias_179: "f32[4, 64, 56, 56]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_180: "f32[4, 64, 56, 56]" = torch.ops.aten.alias.default(alias_179);  alias_179 = None
    le_43: "b8[4, 64, 56, 56]" = torch.ops.aten.le.Scalar(alias_180, 0);  alias_180 = None
    scalar_tensor_43: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_43: "f32[4, 64, 56, 56]" = torch.ops.aten.where.self(le_43, scalar_tensor_43, getitem_137);  le_43 = scalar_tensor_43 = getitem_137 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    add_182: "f32[64]" = torch.ops.aten.add.Tensor(primals_181, 1e-05);  primals_181 = None
    rsqrt_46: "f32[64]" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
    unsqueeze_976: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_180, 0);  primals_180 = None
    unsqueeze_977: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_976, 2);  unsqueeze_976 = None
    unsqueeze_978: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_977, 3);  unsqueeze_977 = None
    sum_94: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
    sub_99: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_978);  convolution_6 = unsqueeze_978 = None
    mul_527: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_43, sub_99);  sub_99 = None
    sum_95: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_527, [0, 2, 3]);  mul_527 = None
    mul_532: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_46, primals_20);  primals_20 = None
    unsqueeze_985: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_532, 0);  mul_532 = None
    unsqueeze_986: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_985, 2);  unsqueeze_985 = None
    unsqueeze_987: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_986, 3);  unsqueeze_986 = None
    mul_533: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_43, unsqueeze_987);  where_43 = unsqueeze_987 = None
    mul_534: "f32[64]" = torch.ops.aten.mul.Tensor(sum_95, rsqrt_46);  sum_95 = rsqrt_46 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_533, relu_4, primals_19, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_533 = primals_19 = None
    getitem_140: "f32[4, 64, 56, 56]" = convolution_backward_46[0]
    getitem_141: "f32[64, 64, 3, 3]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    alias_182: "f32[4, 64, 56, 56]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_183: "f32[4, 64, 56, 56]" = torch.ops.aten.alias.default(alias_182);  alias_182 = None
    le_44: "b8[4, 64, 56, 56]" = torch.ops.aten.le.Scalar(alias_183, 0);  alias_183 = None
    scalar_tensor_44: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_44: "f32[4, 64, 56, 56]" = torch.ops.aten.where.self(le_44, scalar_tensor_44, getitem_140);  le_44 = scalar_tensor_44 = getitem_140 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    add_183: "f32[64]" = torch.ops.aten.add.Tensor(primals_178, 1e-05);  primals_178 = None
    rsqrt_47: "f32[64]" = torch.ops.aten.rsqrt.default(add_183);  add_183 = None
    unsqueeze_988: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_177, 0);  primals_177 = None
    unsqueeze_989: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_988, 2);  unsqueeze_988 = None
    unsqueeze_990: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_989, 3);  unsqueeze_989 = None
    sum_96: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_44, [0, 2, 3])
    sub_100: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_990);  convolution_5 = unsqueeze_990 = None
    mul_535: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_44, sub_100);  sub_100 = None
    sum_97: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_535, [0, 2, 3]);  mul_535 = None
    mul_540: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_47, primals_17);  primals_17 = None
    unsqueeze_997: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_540, 0);  mul_540 = None
    unsqueeze_998: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_997, 2);  unsqueeze_997 = None
    unsqueeze_999: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_998, 3);  unsqueeze_998 = None
    mul_541: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_44, unsqueeze_999);  where_44 = unsqueeze_999 = None
    mul_542: "f32[64]" = torch.ops.aten.mul.Tensor(sum_97, rsqrt_47);  sum_97 = rsqrt_47 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_541, relu_3, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_541 = primals_16 = None
    getitem_143: "f32[4, 256, 56, 56]" = convolution_backward_47[0]
    getitem_144: "f32[64, 256, 1, 1]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    add_184: "f32[4, 256, 56, 56]" = torch.ops.aten.add.Tensor(where_42, getitem_143);  where_42 = getitem_143 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    alias_185: "f32[4, 256, 56, 56]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_186: "f32[4, 256, 56, 56]" = torch.ops.aten.alias.default(alias_185);  alias_185 = None
    le_45: "b8[4, 256, 56, 56]" = torch.ops.aten.le.Scalar(alias_186, 0);  alias_186 = None
    scalar_tensor_45: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_45: "f32[4, 256, 56, 56]" = torch.ops.aten.where.self(le_45, scalar_tensor_45, add_184);  le_45 = scalar_tensor_45 = add_184 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:158, code: identity = self.downsample(x)
    add_185: "f32[256]" = torch.ops.aten.add.Tensor(primals_175, 1e-05);  primals_175 = None
    rsqrt_48: "f32[256]" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
    unsqueeze_1000: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_174, 0);  primals_174 = None
    unsqueeze_1001: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1000, 2);  unsqueeze_1000 = None
    unsqueeze_1002: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1001, 3);  unsqueeze_1001 = None
    sum_98: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
    sub_101: "f32[4, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_1002);  convolution_4 = unsqueeze_1002 = None
    mul_543: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_45, sub_101);  sub_101 = None
    sum_99: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_543, [0, 2, 3]);  mul_543 = None
    mul_548: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_48, primals_14);  primals_14 = None
    unsqueeze_1009: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_548, 0);  mul_548 = None
    unsqueeze_1010: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1009, 2);  unsqueeze_1009 = None
    unsqueeze_1011: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1010, 3);  unsqueeze_1010 = None
    mul_549: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_45, unsqueeze_1011);  unsqueeze_1011 = None
    mul_550: "f32[256]" = torch.ops.aten.mul.Tensor(sum_99, rsqrt_48);  sum_99 = rsqrt_48 = None
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_549, getitem, primals_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_549 = primals_13 = None
    getitem_146: "f32[4, 64, 56, 56]" = convolution_backward_48[0]
    getitem_147: "f32[256, 64, 1, 1]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    add_186: "f32[256]" = torch.ops.aten.add.Tensor(primals_172, 1e-05);  primals_172 = None
    rsqrt_49: "f32[256]" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
    unsqueeze_1012: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_171, 0);  primals_171 = None
    unsqueeze_1013: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1012, 2);  unsqueeze_1012 = None
    unsqueeze_1014: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1013, 3);  unsqueeze_1013 = None
    sum_100: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
    sub_102: "f32[4, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_1014);  convolution_3 = unsqueeze_1014 = None
    mul_551: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_45, sub_102);  sub_102 = None
    sum_101: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_551, [0, 2, 3]);  mul_551 = None
    mul_556: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_49, primals_11);  primals_11 = None
    unsqueeze_1021: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_556, 0);  mul_556 = None
    unsqueeze_1022: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1021, 2);  unsqueeze_1021 = None
    unsqueeze_1023: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1022, 3);  unsqueeze_1022 = None
    mul_557: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_45, unsqueeze_1023);  where_45 = unsqueeze_1023 = None
    mul_558: "f32[256]" = torch.ops.aten.mul.Tensor(sum_101, rsqrt_49);  sum_101 = rsqrt_49 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_557, relu_2, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_557 = primals_10 = None
    getitem_149: "f32[4, 64, 56, 56]" = convolution_backward_49[0]
    getitem_150: "f32[256, 64, 1, 1]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    alias_188: "f32[4, 64, 56, 56]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_189: "f32[4, 64, 56, 56]" = torch.ops.aten.alias.default(alias_188);  alias_188 = None
    le_46: "b8[4, 64, 56, 56]" = torch.ops.aten.le.Scalar(alias_189, 0);  alias_189 = None
    scalar_tensor_46: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_46: "f32[4, 64, 56, 56]" = torch.ops.aten.where.self(le_46, scalar_tensor_46, getitem_149);  le_46 = scalar_tensor_46 = getitem_149 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    add_187: "f32[64]" = torch.ops.aten.add.Tensor(primals_169, 1e-05);  primals_169 = None
    rsqrt_50: "f32[64]" = torch.ops.aten.rsqrt.default(add_187);  add_187 = None
    unsqueeze_1024: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_168, 0);  primals_168 = None
    unsqueeze_1025: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1024, 2);  unsqueeze_1024 = None
    unsqueeze_1026: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1025, 3);  unsqueeze_1025 = None
    sum_102: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_46, [0, 2, 3])
    sub_103: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_1026);  convolution_2 = unsqueeze_1026 = None
    mul_559: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_46, sub_103);  sub_103 = None
    sum_103: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_559, [0, 2, 3]);  mul_559 = None
    mul_564: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_50, primals_8);  primals_8 = None
    unsqueeze_1033: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_564, 0);  mul_564 = None
    unsqueeze_1034: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1033, 2);  unsqueeze_1033 = None
    unsqueeze_1035: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1034, 3);  unsqueeze_1034 = None
    mul_565: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_46, unsqueeze_1035);  where_46 = unsqueeze_1035 = None
    mul_566: "f32[64]" = torch.ops.aten.mul.Tensor(sum_103, rsqrt_50);  sum_103 = rsqrt_50 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_565, relu_1, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_565 = primals_7 = None
    getitem_152: "f32[4, 64, 56, 56]" = convolution_backward_50[0]
    getitem_153: "f32[64, 64, 3, 3]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    alias_191: "f32[4, 64, 56, 56]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_192: "f32[4, 64, 56, 56]" = torch.ops.aten.alias.default(alias_191);  alias_191 = None
    le_47: "b8[4, 64, 56, 56]" = torch.ops.aten.le.Scalar(alias_192, 0);  alias_192 = None
    scalar_tensor_47: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_47: "f32[4, 64, 56, 56]" = torch.ops.aten.where.self(le_47, scalar_tensor_47, getitem_152);  le_47 = scalar_tensor_47 = getitem_152 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    add_188: "f32[64]" = torch.ops.aten.add.Tensor(primals_166, 1e-05);  primals_166 = None
    rsqrt_51: "f32[64]" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
    unsqueeze_1036: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_165, 0);  primals_165 = None
    unsqueeze_1037: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1036, 2);  unsqueeze_1036 = None
    unsqueeze_1038: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1037, 3);  unsqueeze_1037 = None
    sum_104: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_47, [0, 2, 3])
    sub_104: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1038);  convolution_1 = unsqueeze_1038 = None
    mul_567: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_47, sub_104);  sub_104 = None
    sum_105: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_567, [0, 2, 3]);  mul_567 = None
    mul_572: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_51, primals_5);  primals_5 = None
    unsqueeze_1045: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_572, 0);  mul_572 = None
    unsqueeze_1046: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1045, 2);  unsqueeze_1045 = None
    unsqueeze_1047: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1046, 3);  unsqueeze_1046 = None
    mul_573: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_47, unsqueeze_1047);  where_47 = unsqueeze_1047 = None
    mul_574: "f32[64]" = torch.ops.aten.mul.Tensor(sum_105, rsqrt_51);  sum_105 = rsqrt_51 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_573, getitem, primals_4, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_573 = getitem = primals_4 = None
    getitem_155: "f32[4, 64, 56, 56]" = convolution_backward_51[0]
    getitem_156: "f32[64, 64, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    add_189: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(getitem_146, getitem_155);  getitem_146 = getitem_155 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:271, code: x = self.maxpool(x)
    max_pool2d_with_indices_backward: "f32[4, 64, 112, 112]" = torch.ops.aten.max_pool2d_with_indices_backward.default(add_189, relu, [3, 3], [2, 2], [1, 1], [1, 1], False, getitem_1);  add_189 = getitem_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:270, code: x = self.relu(x)
    alias_194: "f32[4, 64, 112, 112]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_195: "f32[4, 64, 112, 112]" = torch.ops.aten.alias.default(alias_194);  alias_194 = None
    le_48: "b8[4, 64, 112, 112]" = torch.ops.aten.le.Scalar(alias_195, 0);  alias_195 = None
    scalar_tensor_48: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_48: "f32[4, 64, 112, 112]" = torch.ops.aten.where.self(le_48, scalar_tensor_48, max_pool2d_with_indices_backward);  le_48 = scalar_tensor_48 = max_pool2d_with_indices_backward = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:269, code: x = self.bn1(x)
    add_190: "f32[64]" = torch.ops.aten.add.Tensor(primals_163, 1e-05);  primals_163 = None
    rsqrt_52: "f32[64]" = torch.ops.aten.rsqrt.default(add_190);  add_190 = None
    unsqueeze_1048: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_162, 0);  primals_162 = None
    unsqueeze_1049: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1048, 2);  unsqueeze_1048 = None
    unsqueeze_1050: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1049, 3);  unsqueeze_1049 = None
    sum_106: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_48, [0, 2, 3])
    sub_105: "f32[4, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1050);  convolution = unsqueeze_1050 = None
    mul_575: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_48, sub_105);  sub_105 = None
    sum_107: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_575, [0, 2, 3]);  mul_575 = None
    mul_580: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_52, primals_2);  primals_2 = None
    unsqueeze_1057: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_580, 0);  mul_580 = None
    unsqueeze_1058: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1057, 2);  unsqueeze_1057 = None
    unsqueeze_1059: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1058, 3);  unsqueeze_1058 = None
    mul_581: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_48, unsqueeze_1059);  where_48 = unsqueeze_1059 = None
    mul_582: "f32[64]" = torch.ops.aten.mul.Tensor(sum_107, rsqrt_52);  sum_107 = rsqrt_52 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:268, code: x = self.conv1(x)
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_581, primals_321, primals_1, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_581 = primals_321 = primals_1 = None
    getitem_159: "f32[64, 3, 7, 7]" = convolution_backward_52[1];  convolution_backward_52 = None
    return pytree.tree_unflatten([addmm, getitem_159, mul_582, sum_106, getitem_156, mul_574, sum_104, getitem_153, mul_566, sum_102, getitem_150, mul_558, sum_100, getitem_147, mul_550, sum_98, getitem_144, mul_542, sum_96, getitem_141, mul_534, sum_94, getitem_138, mul_526, sum_92, getitem_135, mul_518, sum_90, getitem_132, mul_510, sum_88, getitem_129, mul_502, sum_86, getitem_126, mul_494, sum_84, getitem_123, mul_486, sum_82, getitem_120, mul_478, sum_80, getitem_117, mul_470, sum_78, getitem_114, mul_462, sum_76, getitem_111, mul_454, sum_74, getitem_108, mul_446, sum_72, getitem_105, mul_438, sum_70, getitem_102, mul_430, sum_68, getitem_99, mul_422, sum_66, getitem_96, mul_414, sum_64, getitem_93, mul_406, sum_62, getitem_90, mul_398, sum_60, getitem_87, mul_390, sum_58, getitem_84, mul_382, sum_56, getitem_81, mul_374, sum_54, getitem_78, mul_366, sum_52, getitem_75, mul_358, sum_50, getitem_72, mul_350, sum_48, getitem_69, mul_342, sum_46, getitem_66, mul_334, sum_44, getitem_63, mul_326, sum_42, getitem_60, mul_318, sum_40, getitem_57, mul_310, sum_38, getitem_54, mul_302, sum_36, getitem_51, mul_294, sum_34, getitem_48, mul_286, sum_32, getitem_45, mul_278, sum_30, getitem_42, mul_270, sum_28, getitem_39, mul_262, sum_26, getitem_36, mul_254, sum_24, getitem_33, mul_246, sum_22, getitem_30, mul_238, sum_20, getitem_27, mul_230, sum_18, getitem_24, mul_222, sum_16, getitem_21, mul_214, sum_14, getitem_18, mul_206, sum_12, getitem_15, mul_198, sum_10, getitem_12, mul_190, sum_8, getitem_9, mul_182, sum_6, getitem_6, mul_174, sum_4, getitem_3, mul_166, sum_2, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    