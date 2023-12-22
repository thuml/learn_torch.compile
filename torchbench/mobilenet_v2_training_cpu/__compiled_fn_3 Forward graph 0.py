from __future__ import annotations



def forward(self, primals_1: "f32[32, 3, 3, 3]", primals_2: "f32[32]", primals_3: "f32[32]", primals_4: "f32[32, 1, 3, 3]", primals_5: "f32[32]", primals_6: "f32[32]", primals_7: "f32[16, 32, 1, 1]", primals_8: "f32[16]", primals_9: "f32[16]", primals_10: "f32[96, 16, 1, 1]", primals_11: "f32[96]", primals_12: "f32[96]", primals_13: "f32[96, 1, 3, 3]", primals_14: "f32[96]", primals_15: "f32[96]", primals_16: "f32[24, 96, 1, 1]", primals_17: "f32[24]", primals_18: "f32[24]", primals_19: "f32[144, 24, 1, 1]", primals_20: "f32[144]", primals_21: "f32[144]", primals_22: "f32[144, 1, 3, 3]", primals_23: "f32[144]", primals_24: "f32[144]", primals_25: "f32[24, 144, 1, 1]", primals_26: "f32[24]", primals_27: "f32[24]", primals_28: "f32[144, 24, 1, 1]", primals_29: "f32[144]", primals_30: "f32[144]", primals_31: "f32[144, 1, 3, 3]", primals_32: "f32[144]", primals_33: "f32[144]", primals_34: "f32[32, 144, 1, 1]", primals_35: "f32[32]", primals_36: "f32[32]", primals_37: "f32[192, 32, 1, 1]", primals_38: "f32[192]", primals_39: "f32[192]", primals_40: "f32[192, 1, 3, 3]", primals_41: "f32[192]", primals_42: "f32[192]", primals_43: "f32[32, 192, 1, 1]", primals_44: "f32[32]", primals_45: "f32[32]", primals_46: "f32[192, 32, 1, 1]", primals_47: "f32[192]", primals_48: "f32[192]", primals_49: "f32[192, 1, 3, 3]", primals_50: "f32[192]", primals_51: "f32[192]", primals_52: "f32[32, 192, 1, 1]", primals_53: "f32[32]", primals_54: "f32[32]", primals_55: "f32[192, 32, 1, 1]", primals_56: "f32[192]", primals_57: "f32[192]", primals_58: "f32[192, 1, 3, 3]", primals_59: "f32[192]", primals_60: "f32[192]", primals_61: "f32[64, 192, 1, 1]", primals_62: "f32[64]", primals_63: "f32[64]", primals_64: "f32[384, 64, 1, 1]", primals_65: "f32[384]", primals_66: "f32[384]", primals_67: "f32[384, 1, 3, 3]", primals_68: "f32[384]", primals_69: "f32[384]", primals_70: "f32[64, 384, 1, 1]", primals_71: "f32[64]", primals_72: "f32[64]", primals_73: "f32[384, 64, 1, 1]", primals_74: "f32[384]", primals_75: "f32[384]", primals_76: "f32[384, 1, 3, 3]", primals_77: "f32[384]", primals_78: "f32[384]", primals_79: "f32[64, 384, 1, 1]", primals_80: "f32[64]", primals_81: "f32[64]", primals_82: "f32[384, 64, 1, 1]", primals_83: "f32[384]", primals_84: "f32[384]", primals_85: "f32[384, 1, 3, 3]", primals_86: "f32[384]", primals_87: "f32[384]", primals_88: "f32[64, 384, 1, 1]", primals_89: "f32[64]", primals_90: "f32[64]", primals_91: "f32[384, 64, 1, 1]", primals_92: "f32[384]", primals_93: "f32[384]", primals_94: "f32[384, 1, 3, 3]", primals_95: "f32[384]", primals_96: "f32[384]", primals_97: "f32[96, 384, 1, 1]", primals_98: "f32[96]", primals_99: "f32[96]", primals_100: "f32[576, 96, 1, 1]", primals_101: "f32[576]", primals_102: "f32[576]", primals_103: "f32[576, 1, 3, 3]", primals_104: "f32[576]", primals_105: "f32[576]", primals_106: "f32[96, 576, 1, 1]", primals_107: "f32[96]", primals_108: "f32[96]", primals_109: "f32[576, 96, 1, 1]", primals_110: "f32[576]", primals_111: "f32[576]", primals_112: "f32[576, 1, 3, 3]", primals_113: "f32[576]", primals_114: "f32[576]", primals_115: "f32[96, 576, 1, 1]", primals_116: "f32[96]", primals_117: "f32[96]", primals_118: "f32[576, 96, 1, 1]", primals_119: "f32[576]", primals_120: "f32[576]", primals_121: "f32[576, 1, 3, 3]", primals_122: "f32[576]", primals_123: "f32[576]", primals_124: "f32[160, 576, 1, 1]", primals_125: "f32[160]", primals_126: "f32[160]", primals_127: "f32[960, 160, 1, 1]", primals_128: "f32[960]", primals_129: "f32[960]", primals_130: "f32[960, 1, 3, 3]", primals_131: "f32[960]", primals_132: "f32[960]", primals_133: "f32[160, 960, 1, 1]", primals_134: "f32[160]", primals_135: "f32[160]", primals_136: "f32[960, 160, 1, 1]", primals_137: "f32[960]", primals_138: "f32[960]", primals_139: "f32[960, 1, 3, 3]", primals_140: "f32[960]", primals_141: "f32[960]", primals_142: "f32[160, 960, 1, 1]", primals_143: "f32[160]", primals_144: "f32[160]", primals_145: "f32[960, 160, 1, 1]", primals_146: "f32[960]", primals_147: "f32[960]", primals_148: "f32[960, 1, 3, 3]", primals_149: "f32[960]", primals_150: "f32[960]", primals_151: "f32[320, 960, 1, 1]", primals_152: "f32[320]", primals_153: "f32[320]", primals_154: "f32[1280, 320, 1, 1]", primals_155: "f32[1280]", primals_156: "f32[1280]", primals_157: "f32[1000, 1280]", primals_158: "f32[1000]", primals_159: "f32[32]", primals_160: "f32[32]", primals_161: "i64[]", primals_162: "f32[32]", primals_163: "f32[32]", primals_164: "i64[]", primals_165: "f32[16]", primals_166: "f32[16]", primals_167: "i64[]", primals_168: "f32[96]", primals_169: "f32[96]", primals_170: "i64[]", primals_171: "f32[96]", primals_172: "f32[96]", primals_173: "i64[]", primals_174: "f32[24]", primals_175: "f32[24]", primals_176: "i64[]", primals_177: "f32[144]", primals_178: "f32[144]", primals_179: "i64[]", primals_180: "f32[144]", primals_181: "f32[144]", primals_182: "i64[]", primals_183: "f32[24]", primals_184: "f32[24]", primals_185: "i64[]", primals_186: "f32[144]", primals_187: "f32[144]", primals_188: "i64[]", primals_189: "f32[144]", primals_190: "f32[144]", primals_191: "i64[]", primals_192: "f32[32]", primals_193: "f32[32]", primals_194: "i64[]", primals_195: "f32[192]", primals_196: "f32[192]", primals_197: "i64[]", primals_198: "f32[192]", primals_199: "f32[192]", primals_200: "i64[]", primals_201: "f32[32]", primals_202: "f32[32]", primals_203: "i64[]", primals_204: "f32[192]", primals_205: "f32[192]", primals_206: "i64[]", primals_207: "f32[192]", primals_208: "f32[192]", primals_209: "i64[]", primals_210: "f32[32]", primals_211: "f32[32]", primals_212: "i64[]", primals_213: "f32[192]", primals_214: "f32[192]", primals_215: "i64[]", primals_216: "f32[192]", primals_217: "f32[192]", primals_218: "i64[]", primals_219: "f32[64]", primals_220: "f32[64]", primals_221: "i64[]", primals_222: "f32[384]", primals_223: "f32[384]", primals_224: "i64[]", primals_225: "f32[384]", primals_226: "f32[384]", primals_227: "i64[]", primals_228: "f32[64]", primals_229: "f32[64]", primals_230: "i64[]", primals_231: "f32[384]", primals_232: "f32[384]", primals_233: "i64[]", primals_234: "f32[384]", primals_235: "f32[384]", primals_236: "i64[]", primals_237: "f32[64]", primals_238: "f32[64]", primals_239: "i64[]", primals_240: "f32[384]", primals_241: "f32[384]", primals_242: "i64[]", primals_243: "f32[384]", primals_244: "f32[384]", primals_245: "i64[]", primals_246: "f32[64]", primals_247: "f32[64]", primals_248: "i64[]", primals_249: "f32[384]", primals_250: "f32[384]", primals_251: "i64[]", primals_252: "f32[384]", primals_253: "f32[384]", primals_254: "i64[]", primals_255: "f32[96]", primals_256: "f32[96]", primals_257: "i64[]", primals_258: "f32[576]", primals_259: "f32[576]", primals_260: "i64[]", primals_261: "f32[576]", primals_262: "f32[576]", primals_263: "i64[]", primals_264: "f32[96]", primals_265: "f32[96]", primals_266: "i64[]", primals_267: "f32[576]", primals_268: "f32[576]", primals_269: "i64[]", primals_270: "f32[576]", primals_271: "f32[576]", primals_272: "i64[]", primals_273: "f32[96]", primals_274: "f32[96]", primals_275: "i64[]", primals_276: "f32[576]", primals_277: "f32[576]", primals_278: "i64[]", primals_279: "f32[576]", primals_280: "f32[576]", primals_281: "i64[]", primals_282: "f32[160]", primals_283: "f32[160]", primals_284: "i64[]", primals_285: "f32[960]", primals_286: "f32[960]", primals_287: "i64[]", primals_288: "f32[960]", primals_289: "f32[960]", primals_290: "i64[]", primals_291: "f32[160]", primals_292: "f32[160]", primals_293: "i64[]", primals_294: "f32[960]", primals_295: "f32[960]", primals_296: "i64[]", primals_297: "f32[960]", primals_298: "f32[960]", primals_299: "i64[]", primals_300: "f32[160]", primals_301: "f32[160]", primals_302: "i64[]", primals_303: "f32[960]", primals_304: "f32[960]", primals_305: "i64[]", primals_306: "f32[960]", primals_307: "f32[960]", primals_308: "i64[]", primals_309: "f32[320]", primals_310: "f32[320]", primals_311: "i64[]", primals_312: "f32[1280]", primals_313: "f32[1280]", primals_314: "i64[]", primals_315: "f32[4, 3, 224, 224]"):
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:166, code: x = self.features(x)
    convolution: "f32[4, 32, 112, 112]" = torch.ops.aten.convolution.default(primals_315, primals_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    convert_element_type: "f32[32]" = torch.ops.prims.convert_element_type.default(primals_159, torch.float32)
    convert_element_type_1: "f32[32]" = torch.ops.prims.convert_element_type.default(primals_160, torch.float32)
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
    unsqueeze_4: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1)
    unsqueeze_5: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_2: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
    unsqueeze_6: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1);  primals_3 = None
    unsqueeze_7: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_1: "f32[4, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
    clone: "f32[4, 32, 112, 112]" = torch.ops.aten.clone.default(add_1)
    clamp_min: "f32[4, 32, 112, 112]" = torch.ops.aten.clamp_min.default(add_1, 0.0);  add_1 = None
    clamp_max: "f32[4, 32, 112, 112]" = torch.ops.aten.clamp_max.default(clamp_min, 6.0);  clamp_min = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:64, code: return self.conv(x)
    convolution_1: "f32[4, 32, 112, 112]" = torch.ops.aten.convolution.default(clamp_max, primals_4, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    convert_element_type_2: "f32[32]" = torch.ops.prims.convert_element_type.default(primals_162, torch.float32)
    convert_element_type_3: "f32[32]" = torch.ops.prims.convert_element_type.default(primals_163, torch.float32)
    add_2: "f32[32]" = torch.ops.aten.add.Tensor(convert_element_type_3, 1e-05);  convert_element_type_3 = None
    sqrt_1: "f32[32]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
    reciprocal_1: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_3: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_8: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, -1);  convert_element_type_2 = None
    unsqueeze_9: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    unsqueeze_10: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
    unsqueeze_11: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    sub_1: "f32[4, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  unsqueeze_9 = None
    mul_4: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_13: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_5: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
    unsqueeze_14: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_15: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_3: "f32[4, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None
    clone_1: "f32[4, 32, 112, 112]" = torch.ops.aten.clone.default(add_3)
    clamp_min_1: "f32[4, 32, 112, 112]" = torch.ops.aten.clamp_min.default(add_3, 0.0);  add_3 = None
    clamp_max_1: "f32[4, 32, 112, 112]" = torch.ops.aten.clamp_max.default(clamp_min_1, 6.0);  clamp_min_1 = None
    convolution_2: "f32[4, 16, 112, 112]" = torch.ops.aten.convolution.default(clamp_max_1, primals_7, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_4: "f32[16]" = torch.ops.prims.convert_element_type.default(primals_165, torch.float32)
    convert_element_type_5: "f32[16]" = torch.ops.prims.convert_element_type.default(primals_166, torch.float32)
    add_4: "f32[16]" = torch.ops.aten.add.Tensor(convert_element_type_5, 1e-05);  convert_element_type_5 = None
    sqrt_2: "f32[16]" = torch.ops.aten.sqrt.default(add_4);  add_4 = None
    reciprocal_2: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_6: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_16: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, -1);  convert_element_type_4 = None
    unsqueeze_17: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    unsqueeze_18: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
    unsqueeze_19: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    sub_2: "f32[4, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  unsqueeze_17 = None
    mul_7: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1)
    unsqueeze_21: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_8: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_21);  mul_7 = unsqueeze_21 = None
    unsqueeze_22: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1);  primals_9 = None
    unsqueeze_23: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_5: "f32[4, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_8, unsqueeze_23);  mul_8 = unsqueeze_23 = None
    convolution_3: "f32[4, 96, 112, 112]" = torch.ops.aten.convolution.default(add_5, primals_10, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_6: "f32[96]" = torch.ops.prims.convert_element_type.default(primals_168, torch.float32)
    convert_element_type_7: "f32[96]" = torch.ops.prims.convert_element_type.default(primals_169, torch.float32)
    add_6: "f32[96]" = torch.ops.aten.add.Tensor(convert_element_type_7, 1e-05);  convert_element_type_7 = None
    sqrt_3: "f32[96]" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
    reciprocal_3: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
    mul_9: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
    unsqueeze_24: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_6, -1);  convert_element_type_6 = None
    unsqueeze_25: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    unsqueeze_26: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_9, -1);  mul_9 = None
    unsqueeze_27: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    sub_3: "f32[4, 96, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  unsqueeze_25 = None
    mul_10: "f32[4, 96, 112, 112]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
    unsqueeze_28: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_29: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_11: "f32[4, 96, 112, 112]" = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_29);  mul_10 = unsqueeze_29 = None
    unsqueeze_30: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_31: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_7: "f32[4, 96, 112, 112]" = torch.ops.aten.add.Tensor(mul_11, unsqueeze_31);  mul_11 = unsqueeze_31 = None
    clone_2: "f32[4, 96, 112, 112]" = torch.ops.aten.clone.default(add_7)
    clamp_min_2: "f32[4, 96, 112, 112]" = torch.ops.aten.clamp_min.default(add_7, 0.0);  add_7 = None
    clamp_max_2: "f32[4, 96, 112, 112]" = torch.ops.aten.clamp_max.default(clamp_min_2, 6.0);  clamp_min_2 = None
    convolution_4: "f32[4, 96, 56, 56]" = torch.ops.aten.convolution.default(clamp_max_2, primals_13, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 96)
    convert_element_type_8: "f32[96]" = torch.ops.prims.convert_element_type.default(primals_171, torch.float32)
    convert_element_type_9: "f32[96]" = torch.ops.prims.convert_element_type.default(primals_172, torch.float32)
    add_8: "f32[96]" = torch.ops.aten.add.Tensor(convert_element_type_9, 1e-05);  convert_element_type_9 = None
    sqrt_4: "f32[96]" = torch.ops.aten.sqrt.default(add_8);  add_8 = None
    reciprocal_4: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
    mul_12: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
    unsqueeze_32: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_8, -1);  convert_element_type_8 = None
    unsqueeze_33: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    unsqueeze_34: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
    unsqueeze_35: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    sub_4: "f32[4, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  unsqueeze_33 = None
    mul_13: "f32[4, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
    unsqueeze_36: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1)
    unsqueeze_37: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_14: "f32[4, 96, 56, 56]" = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_37);  mul_13 = unsqueeze_37 = None
    unsqueeze_38: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1);  primals_15 = None
    unsqueeze_39: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_9: "f32[4, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_39);  mul_14 = unsqueeze_39 = None
    clone_3: "f32[4, 96, 56, 56]" = torch.ops.aten.clone.default(add_9)
    clamp_min_3: "f32[4, 96, 56, 56]" = torch.ops.aten.clamp_min.default(add_9, 0.0);  add_9 = None
    clamp_max_3: "f32[4, 96, 56, 56]" = torch.ops.aten.clamp_max.default(clamp_min_3, 6.0);  clamp_min_3 = None
    convolution_5: "f32[4, 24, 56, 56]" = torch.ops.aten.convolution.default(clamp_max_3, primals_16, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_10: "f32[24]" = torch.ops.prims.convert_element_type.default(primals_174, torch.float32)
    convert_element_type_11: "f32[24]" = torch.ops.prims.convert_element_type.default(primals_175, torch.float32)
    add_10: "f32[24]" = torch.ops.aten.add.Tensor(convert_element_type_11, 1e-05);  convert_element_type_11 = None
    sqrt_5: "f32[24]" = torch.ops.aten.sqrt.default(add_10);  add_10 = None
    reciprocal_5: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
    mul_15: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
    unsqueeze_40: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_10, -1);  convert_element_type_10 = None
    unsqueeze_41: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    unsqueeze_42: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_15, -1);  mul_15 = None
    unsqueeze_43: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    sub_5: "f32[4, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_41);  unsqueeze_41 = None
    mul_16: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
    unsqueeze_44: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_45: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_17: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_45);  mul_16 = unsqueeze_45 = None
    unsqueeze_46: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_47: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_11: "f32[4, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_17, unsqueeze_47);  mul_17 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    convolution_6: "f32[4, 144, 56, 56]" = torch.ops.aten.convolution.default(add_11, primals_19, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_12: "f32[144]" = torch.ops.prims.convert_element_type.default(primals_177, torch.float32)
    convert_element_type_13: "f32[144]" = torch.ops.prims.convert_element_type.default(primals_178, torch.float32)
    add_12: "f32[144]" = torch.ops.aten.add.Tensor(convert_element_type_13, 1e-05);  convert_element_type_13 = None
    sqrt_6: "f32[144]" = torch.ops.aten.sqrt.default(add_12);  add_12 = None
    reciprocal_6: "f32[144]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
    mul_18: "f32[144]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
    unsqueeze_48: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_12, -1);  convert_element_type_12 = None
    unsqueeze_49: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    unsqueeze_50: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(mul_18, -1);  mul_18 = None
    unsqueeze_51: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    sub_6: "f32[4, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_49);  unsqueeze_49 = None
    mul_19: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
    unsqueeze_52: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1)
    unsqueeze_53: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_20: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_19, unsqueeze_53);  mul_19 = unsqueeze_53 = None
    unsqueeze_54: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1);  primals_21 = None
    unsqueeze_55: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_13: "f32[4, 144, 56, 56]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_55);  mul_20 = unsqueeze_55 = None
    clone_4: "f32[4, 144, 56, 56]" = torch.ops.aten.clone.default(add_13)
    clamp_min_4: "f32[4, 144, 56, 56]" = torch.ops.aten.clamp_min.default(add_13, 0.0);  add_13 = None
    clamp_max_4: "f32[4, 144, 56, 56]" = torch.ops.aten.clamp_max.default(clamp_min_4, 6.0);  clamp_min_4 = None
    convolution_7: "f32[4, 144, 56, 56]" = torch.ops.aten.convolution.default(clamp_max_4, primals_22, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 144)
    convert_element_type_14: "f32[144]" = torch.ops.prims.convert_element_type.default(primals_180, torch.float32)
    convert_element_type_15: "f32[144]" = torch.ops.prims.convert_element_type.default(primals_181, torch.float32)
    add_14: "f32[144]" = torch.ops.aten.add.Tensor(convert_element_type_15, 1e-05);  convert_element_type_15 = None
    sqrt_7: "f32[144]" = torch.ops.aten.sqrt.default(add_14);  add_14 = None
    reciprocal_7: "f32[144]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
    mul_21: "f32[144]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
    unsqueeze_56: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_14, -1);  convert_element_type_14 = None
    unsqueeze_57: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    unsqueeze_58: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(mul_21, -1);  mul_21 = None
    unsqueeze_59: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    sub_7: "f32[4, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_57);  unsqueeze_57 = None
    mul_22: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
    unsqueeze_60: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_61: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_23: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_61);  mul_22 = unsqueeze_61 = None
    unsqueeze_62: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_63: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_15: "f32[4, 144, 56, 56]" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_63);  mul_23 = unsqueeze_63 = None
    clone_5: "f32[4, 144, 56, 56]" = torch.ops.aten.clone.default(add_15)
    clamp_min_5: "f32[4, 144, 56, 56]" = torch.ops.aten.clamp_min.default(add_15, 0.0);  add_15 = None
    clamp_max_5: "f32[4, 144, 56, 56]" = torch.ops.aten.clamp_max.default(clamp_min_5, 6.0);  clamp_min_5 = None
    convolution_8: "f32[4, 24, 56, 56]" = torch.ops.aten.convolution.default(clamp_max_5, primals_25, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_16: "f32[24]" = torch.ops.prims.convert_element_type.default(primals_183, torch.float32)
    convert_element_type_17: "f32[24]" = torch.ops.prims.convert_element_type.default(primals_184, torch.float32)
    add_16: "f32[24]" = torch.ops.aten.add.Tensor(convert_element_type_17, 1e-05);  convert_element_type_17 = None
    sqrt_8: "f32[24]" = torch.ops.aten.sqrt.default(add_16);  add_16 = None
    reciprocal_8: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
    mul_24: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
    unsqueeze_64: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_16, -1);  convert_element_type_16 = None
    unsqueeze_65: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    unsqueeze_66: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_24, -1);  mul_24 = None
    unsqueeze_67: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    sub_8: "f32[4, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_65);  unsqueeze_65 = None
    mul_25: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
    unsqueeze_68: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1)
    unsqueeze_69: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_26: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_69);  mul_25 = unsqueeze_69 = None
    unsqueeze_70: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1);  primals_27 = None
    unsqueeze_71: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_17: "f32[4, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_26, unsqueeze_71);  mul_26 = unsqueeze_71 = None
    add_18: "f32[4, 24, 56, 56]" = torch.ops.aten.add.Tensor(add_11, add_17);  add_17 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:64, code: return self.conv(x)
    convolution_9: "f32[4, 144, 56, 56]" = torch.ops.aten.convolution.default(add_18, primals_28, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_18: "f32[144]" = torch.ops.prims.convert_element_type.default(primals_186, torch.float32)
    convert_element_type_19: "f32[144]" = torch.ops.prims.convert_element_type.default(primals_187, torch.float32)
    add_19: "f32[144]" = torch.ops.aten.add.Tensor(convert_element_type_19, 1e-05);  convert_element_type_19 = None
    sqrt_9: "f32[144]" = torch.ops.aten.sqrt.default(add_19);  add_19 = None
    reciprocal_9: "f32[144]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
    mul_27: "f32[144]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
    unsqueeze_72: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_18, -1);  convert_element_type_18 = None
    unsqueeze_73: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    unsqueeze_74: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(mul_27, -1);  mul_27 = None
    unsqueeze_75: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    sub_9: "f32[4, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_73);  unsqueeze_73 = None
    mul_28: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
    unsqueeze_76: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_77: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_29: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_77);  mul_28 = unsqueeze_77 = None
    unsqueeze_78: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_79: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_20: "f32[4, 144, 56, 56]" = torch.ops.aten.add.Tensor(mul_29, unsqueeze_79);  mul_29 = unsqueeze_79 = None
    clone_6: "f32[4, 144, 56, 56]" = torch.ops.aten.clone.default(add_20)
    clamp_min_6: "f32[4, 144, 56, 56]" = torch.ops.aten.clamp_min.default(add_20, 0.0);  add_20 = None
    clamp_max_6: "f32[4, 144, 56, 56]" = torch.ops.aten.clamp_max.default(clamp_min_6, 6.0);  clamp_min_6 = None
    convolution_10: "f32[4, 144, 28, 28]" = torch.ops.aten.convolution.default(clamp_max_6, primals_31, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 144)
    convert_element_type_20: "f32[144]" = torch.ops.prims.convert_element_type.default(primals_189, torch.float32)
    convert_element_type_21: "f32[144]" = torch.ops.prims.convert_element_type.default(primals_190, torch.float32)
    add_21: "f32[144]" = torch.ops.aten.add.Tensor(convert_element_type_21, 1e-05);  convert_element_type_21 = None
    sqrt_10: "f32[144]" = torch.ops.aten.sqrt.default(add_21);  add_21 = None
    reciprocal_10: "f32[144]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
    mul_30: "f32[144]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
    unsqueeze_80: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_20, -1);  convert_element_type_20 = None
    unsqueeze_81: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    unsqueeze_82: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(mul_30, -1);  mul_30 = None
    unsqueeze_83: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    sub_10: "f32[4, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_81);  unsqueeze_81 = None
    mul_31: "f32[4, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
    unsqueeze_84: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1)
    unsqueeze_85: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_32: "f32[4, 144, 28, 28]" = torch.ops.aten.mul.Tensor(mul_31, unsqueeze_85);  mul_31 = unsqueeze_85 = None
    unsqueeze_86: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1);  primals_33 = None
    unsqueeze_87: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_22: "f32[4, 144, 28, 28]" = torch.ops.aten.add.Tensor(mul_32, unsqueeze_87);  mul_32 = unsqueeze_87 = None
    clone_7: "f32[4, 144, 28, 28]" = torch.ops.aten.clone.default(add_22)
    clamp_min_7: "f32[4, 144, 28, 28]" = torch.ops.aten.clamp_min.default(add_22, 0.0);  add_22 = None
    clamp_max_7: "f32[4, 144, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_7, 6.0);  clamp_min_7 = None
    convolution_11: "f32[4, 32, 28, 28]" = torch.ops.aten.convolution.default(clamp_max_7, primals_34, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_22: "f32[32]" = torch.ops.prims.convert_element_type.default(primals_192, torch.float32)
    convert_element_type_23: "f32[32]" = torch.ops.prims.convert_element_type.default(primals_193, torch.float32)
    add_23: "f32[32]" = torch.ops.aten.add.Tensor(convert_element_type_23, 1e-05);  convert_element_type_23 = None
    sqrt_11: "f32[32]" = torch.ops.aten.sqrt.default(add_23);  add_23 = None
    reciprocal_11: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
    mul_33: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
    unsqueeze_88: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_22, -1);  convert_element_type_22 = None
    unsqueeze_89: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    unsqueeze_90: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_33, -1);  mul_33 = None
    unsqueeze_91: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    sub_11: "f32[4, 32, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_89);  unsqueeze_89 = None
    mul_34: "f32[4, 32, 28, 28]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
    unsqueeze_92: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_93: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_35: "f32[4, 32, 28, 28]" = torch.ops.aten.mul.Tensor(mul_34, unsqueeze_93);  mul_34 = unsqueeze_93 = None
    unsqueeze_94: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_95: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_24: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_95);  mul_35 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    convolution_12: "f32[4, 192, 28, 28]" = torch.ops.aten.convolution.default(add_24, primals_37, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_24: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_195, torch.float32)
    convert_element_type_25: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_196, torch.float32)
    add_25: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_25, 1e-05);  convert_element_type_25 = None
    sqrt_12: "f32[192]" = torch.ops.aten.sqrt.default(add_25);  add_25 = None
    reciprocal_12: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
    mul_36: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
    unsqueeze_96: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_24, -1);  convert_element_type_24 = None
    unsqueeze_97: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    unsqueeze_98: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_36, -1);  mul_36 = None
    unsqueeze_99: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    sub_12: "f32[4, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_97);  unsqueeze_97 = None
    mul_37: "f32[4, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
    unsqueeze_100: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1)
    unsqueeze_101: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_38: "f32[4, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_101);  mul_37 = unsqueeze_101 = None
    unsqueeze_102: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1);  primals_39 = None
    unsqueeze_103: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_26: "f32[4, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_38, unsqueeze_103);  mul_38 = unsqueeze_103 = None
    clone_8: "f32[4, 192, 28, 28]" = torch.ops.aten.clone.default(add_26)
    clamp_min_8: "f32[4, 192, 28, 28]" = torch.ops.aten.clamp_min.default(add_26, 0.0);  add_26 = None
    clamp_max_8: "f32[4, 192, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_8, 6.0);  clamp_min_8 = None
    convolution_13: "f32[4, 192, 28, 28]" = torch.ops.aten.convolution.default(clamp_max_8, primals_40, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 192)
    convert_element_type_26: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_198, torch.float32)
    convert_element_type_27: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_199, torch.float32)
    add_27: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_27, 1e-05);  convert_element_type_27 = None
    sqrt_13: "f32[192]" = torch.ops.aten.sqrt.default(add_27);  add_27 = None
    reciprocal_13: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
    mul_39: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
    unsqueeze_104: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_26, -1);  convert_element_type_26 = None
    unsqueeze_105: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    unsqueeze_106: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_39, -1);  mul_39 = None
    unsqueeze_107: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    sub_13: "f32[4, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_105);  unsqueeze_105 = None
    mul_40: "f32[4, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
    unsqueeze_108: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_109: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_41: "f32[4, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_40, unsqueeze_109);  mul_40 = unsqueeze_109 = None
    unsqueeze_110: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_111: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_28: "f32[4, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_111);  mul_41 = unsqueeze_111 = None
    clone_9: "f32[4, 192, 28, 28]" = torch.ops.aten.clone.default(add_28)
    clamp_min_9: "f32[4, 192, 28, 28]" = torch.ops.aten.clamp_min.default(add_28, 0.0);  add_28 = None
    clamp_max_9: "f32[4, 192, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_9, 6.0);  clamp_min_9 = None
    convolution_14: "f32[4, 32, 28, 28]" = torch.ops.aten.convolution.default(clamp_max_9, primals_43, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_28: "f32[32]" = torch.ops.prims.convert_element_type.default(primals_201, torch.float32)
    convert_element_type_29: "f32[32]" = torch.ops.prims.convert_element_type.default(primals_202, torch.float32)
    add_29: "f32[32]" = torch.ops.aten.add.Tensor(convert_element_type_29, 1e-05);  convert_element_type_29 = None
    sqrt_14: "f32[32]" = torch.ops.aten.sqrt.default(add_29);  add_29 = None
    reciprocal_14: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
    mul_42: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
    unsqueeze_112: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_28, -1);  convert_element_type_28 = None
    unsqueeze_113: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    unsqueeze_114: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_42, -1);  mul_42 = None
    unsqueeze_115: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    sub_14: "f32[4, 32, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_113);  unsqueeze_113 = None
    mul_43: "f32[4, 32, 28, 28]" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
    unsqueeze_116: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1)
    unsqueeze_117: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_44: "f32[4, 32, 28, 28]" = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_117);  mul_43 = unsqueeze_117 = None
    unsqueeze_118: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1);  primals_45 = None
    unsqueeze_119: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_30: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(mul_44, unsqueeze_119);  mul_44 = unsqueeze_119 = None
    add_31: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_24, add_30);  add_30 = None
    convolution_15: "f32[4, 192, 28, 28]" = torch.ops.aten.convolution.default(add_31, primals_46, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_30: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_204, torch.float32)
    convert_element_type_31: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_205, torch.float32)
    add_32: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_31, 1e-05);  convert_element_type_31 = None
    sqrt_15: "f32[192]" = torch.ops.aten.sqrt.default(add_32);  add_32 = None
    reciprocal_15: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
    mul_45: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
    unsqueeze_120: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_30, -1);  convert_element_type_30 = None
    unsqueeze_121: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    unsqueeze_122: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_45, -1);  mul_45 = None
    unsqueeze_123: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    sub_15: "f32[4, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_121);  unsqueeze_121 = None
    mul_46: "f32[4, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
    unsqueeze_124: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_125: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_47: "f32[4, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_46, unsqueeze_125);  mul_46 = unsqueeze_125 = None
    unsqueeze_126: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_127: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_33: "f32[4, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_47, unsqueeze_127);  mul_47 = unsqueeze_127 = None
    clone_10: "f32[4, 192, 28, 28]" = torch.ops.aten.clone.default(add_33)
    clamp_min_10: "f32[4, 192, 28, 28]" = torch.ops.aten.clamp_min.default(add_33, 0.0);  add_33 = None
    clamp_max_10: "f32[4, 192, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_10, 6.0);  clamp_min_10 = None
    convolution_16: "f32[4, 192, 28, 28]" = torch.ops.aten.convolution.default(clamp_max_10, primals_49, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 192)
    convert_element_type_32: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_207, torch.float32)
    convert_element_type_33: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_208, torch.float32)
    add_34: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_33, 1e-05);  convert_element_type_33 = None
    sqrt_16: "f32[192]" = torch.ops.aten.sqrt.default(add_34);  add_34 = None
    reciprocal_16: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
    mul_48: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
    unsqueeze_128: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_32, -1);  convert_element_type_32 = None
    unsqueeze_129: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    unsqueeze_130: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_48, -1);  mul_48 = None
    unsqueeze_131: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    sub_16: "f32[4, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_129);  unsqueeze_129 = None
    mul_49: "f32[4, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
    unsqueeze_132: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1)
    unsqueeze_133: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_50: "f32[4, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_133);  mul_49 = unsqueeze_133 = None
    unsqueeze_134: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1);  primals_51 = None
    unsqueeze_135: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_35: "f32[4, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_50, unsqueeze_135);  mul_50 = unsqueeze_135 = None
    clone_11: "f32[4, 192, 28, 28]" = torch.ops.aten.clone.default(add_35)
    clamp_min_11: "f32[4, 192, 28, 28]" = torch.ops.aten.clamp_min.default(add_35, 0.0);  add_35 = None
    clamp_max_11: "f32[4, 192, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_11, 6.0);  clamp_min_11 = None
    convolution_17: "f32[4, 32, 28, 28]" = torch.ops.aten.convolution.default(clamp_max_11, primals_52, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_34: "f32[32]" = torch.ops.prims.convert_element_type.default(primals_210, torch.float32)
    convert_element_type_35: "f32[32]" = torch.ops.prims.convert_element_type.default(primals_211, torch.float32)
    add_36: "f32[32]" = torch.ops.aten.add.Tensor(convert_element_type_35, 1e-05);  convert_element_type_35 = None
    sqrt_17: "f32[32]" = torch.ops.aten.sqrt.default(add_36);  add_36 = None
    reciprocal_17: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
    mul_51: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
    unsqueeze_136: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_34, -1);  convert_element_type_34 = None
    unsqueeze_137: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    unsqueeze_138: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_51, -1);  mul_51 = None
    unsqueeze_139: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    sub_17: "f32[4, 32, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_137);  unsqueeze_137 = None
    mul_52: "f32[4, 32, 28, 28]" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_139);  sub_17 = unsqueeze_139 = None
    unsqueeze_140: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_141: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_53: "f32[4, 32, 28, 28]" = torch.ops.aten.mul.Tensor(mul_52, unsqueeze_141);  mul_52 = unsqueeze_141 = None
    unsqueeze_142: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_143: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_37: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(mul_53, unsqueeze_143);  mul_53 = unsqueeze_143 = None
    add_38: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_31, add_37);  add_37 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:64, code: return self.conv(x)
    convolution_18: "f32[4, 192, 28, 28]" = torch.ops.aten.convolution.default(add_38, primals_55, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_36: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_213, torch.float32)
    convert_element_type_37: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_214, torch.float32)
    add_39: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_37, 1e-05);  convert_element_type_37 = None
    sqrt_18: "f32[192]" = torch.ops.aten.sqrt.default(add_39);  add_39 = None
    reciprocal_18: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
    mul_54: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
    unsqueeze_144: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_36, -1);  convert_element_type_36 = None
    unsqueeze_145: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    unsqueeze_146: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_54, -1);  mul_54 = None
    unsqueeze_147: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    sub_18: "f32[4, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_145);  unsqueeze_145 = None
    mul_55: "f32[4, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_147);  sub_18 = unsqueeze_147 = None
    unsqueeze_148: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1)
    unsqueeze_149: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_56: "f32[4, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_55, unsqueeze_149);  mul_55 = unsqueeze_149 = None
    unsqueeze_150: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1);  primals_57 = None
    unsqueeze_151: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_40: "f32[4, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_56, unsqueeze_151);  mul_56 = unsqueeze_151 = None
    clone_12: "f32[4, 192, 28, 28]" = torch.ops.aten.clone.default(add_40)
    clamp_min_12: "f32[4, 192, 28, 28]" = torch.ops.aten.clamp_min.default(add_40, 0.0);  add_40 = None
    clamp_max_12: "f32[4, 192, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_12, 6.0);  clamp_min_12 = None
    convolution_19: "f32[4, 192, 14, 14]" = torch.ops.aten.convolution.default(clamp_max_12, primals_58, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 192)
    convert_element_type_38: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_216, torch.float32)
    convert_element_type_39: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_217, torch.float32)
    add_41: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_39, 1e-05);  convert_element_type_39 = None
    sqrt_19: "f32[192]" = torch.ops.aten.sqrt.default(add_41);  add_41 = None
    reciprocal_19: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
    mul_57: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
    unsqueeze_152: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_38, -1);  convert_element_type_38 = None
    unsqueeze_153: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    unsqueeze_154: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_57, -1);  mul_57 = None
    unsqueeze_155: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    sub_19: "f32[4, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_153);  unsqueeze_153 = None
    mul_58: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_155);  sub_19 = unsqueeze_155 = None
    unsqueeze_156: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_157: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_59: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_157);  mul_58 = unsqueeze_157 = None
    unsqueeze_158: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_159: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_42: "f32[4, 192, 14, 14]" = torch.ops.aten.add.Tensor(mul_59, unsqueeze_159);  mul_59 = unsqueeze_159 = None
    clone_13: "f32[4, 192, 14, 14]" = torch.ops.aten.clone.default(add_42)
    clamp_min_13: "f32[4, 192, 14, 14]" = torch.ops.aten.clamp_min.default(add_42, 0.0);  add_42 = None
    clamp_max_13: "f32[4, 192, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_13, 6.0);  clamp_min_13 = None
    convolution_20: "f32[4, 64, 14, 14]" = torch.ops.aten.convolution.default(clamp_max_13, primals_61, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_40: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_219, torch.float32)
    convert_element_type_41: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_220, torch.float32)
    add_43: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_41, 1e-05);  convert_element_type_41 = None
    sqrt_20: "f32[64]" = torch.ops.aten.sqrt.default(add_43);  add_43 = None
    reciprocal_20: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
    mul_60: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
    unsqueeze_160: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_40, -1);  convert_element_type_40 = None
    unsqueeze_161: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    unsqueeze_162: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_60, -1);  mul_60 = None
    unsqueeze_163: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    sub_20: "f32[4, 64, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_161);  unsqueeze_161 = None
    mul_61: "f32[4, 64, 14, 14]" = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_163);  sub_20 = unsqueeze_163 = None
    unsqueeze_164: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1)
    unsqueeze_165: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_62: "f32[4, 64, 14, 14]" = torch.ops.aten.mul.Tensor(mul_61, unsqueeze_165);  mul_61 = unsqueeze_165 = None
    unsqueeze_166: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1);  primals_63 = None
    unsqueeze_167: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_44: "f32[4, 64, 14, 14]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_167);  mul_62 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    convolution_21: "f32[4, 384, 14, 14]" = torch.ops.aten.convolution.default(add_44, primals_64, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_42: "f32[384]" = torch.ops.prims.convert_element_type.default(primals_222, torch.float32)
    convert_element_type_43: "f32[384]" = torch.ops.prims.convert_element_type.default(primals_223, torch.float32)
    add_45: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_43, 1e-05);  convert_element_type_43 = None
    sqrt_21: "f32[384]" = torch.ops.aten.sqrt.default(add_45);  add_45 = None
    reciprocal_21: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
    mul_63: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
    unsqueeze_168: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_42, -1);  convert_element_type_42 = None
    unsqueeze_169: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    unsqueeze_170: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_63, -1);  mul_63 = None
    unsqueeze_171: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    sub_21: "f32[4, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_169);  unsqueeze_169 = None
    mul_64: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_21, unsqueeze_171);  sub_21 = unsqueeze_171 = None
    unsqueeze_172: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
    unsqueeze_173: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_65: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_64, unsqueeze_173);  mul_64 = unsqueeze_173 = None
    unsqueeze_174: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
    unsqueeze_175: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_46: "f32[4, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_65, unsqueeze_175);  mul_65 = unsqueeze_175 = None
    clone_14: "f32[4, 384, 14, 14]" = torch.ops.aten.clone.default(add_46)
    clamp_min_14: "f32[4, 384, 14, 14]" = torch.ops.aten.clamp_min.default(add_46, 0.0);  add_46 = None
    clamp_max_14: "f32[4, 384, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_14, 6.0);  clamp_min_14 = None
    convolution_22: "f32[4, 384, 14, 14]" = torch.ops.aten.convolution.default(clamp_max_14, primals_67, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 384)
    convert_element_type_44: "f32[384]" = torch.ops.prims.convert_element_type.default(primals_225, torch.float32)
    convert_element_type_45: "f32[384]" = torch.ops.prims.convert_element_type.default(primals_226, torch.float32)
    add_47: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_45, 1e-05);  convert_element_type_45 = None
    sqrt_22: "f32[384]" = torch.ops.aten.sqrt.default(add_47);  add_47 = None
    reciprocal_22: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
    mul_66: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
    unsqueeze_176: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_44, -1);  convert_element_type_44 = None
    unsqueeze_177: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    unsqueeze_178: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_66, -1);  mul_66 = None
    unsqueeze_179: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    sub_22: "f32[4, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_177);  unsqueeze_177 = None
    mul_67: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_179);  sub_22 = unsqueeze_179 = None
    unsqueeze_180: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1)
    unsqueeze_181: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_68: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_67, unsqueeze_181);  mul_67 = unsqueeze_181 = None
    unsqueeze_182: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1);  primals_69 = None
    unsqueeze_183: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_48: "f32[4, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_68, unsqueeze_183);  mul_68 = unsqueeze_183 = None
    clone_15: "f32[4, 384, 14, 14]" = torch.ops.aten.clone.default(add_48)
    clamp_min_15: "f32[4, 384, 14, 14]" = torch.ops.aten.clamp_min.default(add_48, 0.0);  add_48 = None
    clamp_max_15: "f32[4, 384, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_15, 6.0);  clamp_min_15 = None
    convolution_23: "f32[4, 64, 14, 14]" = torch.ops.aten.convolution.default(clamp_max_15, primals_70, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_46: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_228, torch.float32)
    convert_element_type_47: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_229, torch.float32)
    add_49: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_47, 1e-05);  convert_element_type_47 = None
    sqrt_23: "f32[64]" = torch.ops.aten.sqrt.default(add_49);  add_49 = None
    reciprocal_23: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
    mul_69: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
    unsqueeze_184: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_46, -1);  convert_element_type_46 = None
    unsqueeze_185: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    unsqueeze_186: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_69, -1);  mul_69 = None
    unsqueeze_187: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    sub_23: "f32[4, 64, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_185);  unsqueeze_185 = None
    mul_70: "f32[4, 64, 14, 14]" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_187);  sub_23 = unsqueeze_187 = None
    unsqueeze_188: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_189: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_71: "f32[4, 64, 14, 14]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_189);  mul_70 = unsqueeze_189 = None
    unsqueeze_190: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_191: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_50: "f32[4, 64, 14, 14]" = torch.ops.aten.add.Tensor(mul_71, unsqueeze_191);  mul_71 = unsqueeze_191 = None
    add_51: "f32[4, 64, 14, 14]" = torch.ops.aten.add.Tensor(add_44, add_50);  add_50 = None
    convolution_24: "f32[4, 384, 14, 14]" = torch.ops.aten.convolution.default(add_51, primals_73, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_48: "f32[384]" = torch.ops.prims.convert_element_type.default(primals_231, torch.float32)
    convert_element_type_49: "f32[384]" = torch.ops.prims.convert_element_type.default(primals_232, torch.float32)
    add_52: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_49, 1e-05);  convert_element_type_49 = None
    sqrt_24: "f32[384]" = torch.ops.aten.sqrt.default(add_52);  add_52 = None
    reciprocal_24: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
    mul_72: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
    unsqueeze_192: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_48, -1);  convert_element_type_48 = None
    unsqueeze_193: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    unsqueeze_194: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_72, -1);  mul_72 = None
    unsqueeze_195: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    sub_24: "f32[4, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_193);  unsqueeze_193 = None
    mul_73: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_195);  sub_24 = unsqueeze_195 = None
    unsqueeze_196: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1)
    unsqueeze_197: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_74: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_73, unsqueeze_197);  mul_73 = unsqueeze_197 = None
    unsqueeze_198: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1);  primals_75 = None
    unsqueeze_199: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_53: "f32[4, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_74, unsqueeze_199);  mul_74 = unsqueeze_199 = None
    clone_16: "f32[4, 384, 14, 14]" = torch.ops.aten.clone.default(add_53)
    clamp_min_16: "f32[4, 384, 14, 14]" = torch.ops.aten.clamp_min.default(add_53, 0.0);  add_53 = None
    clamp_max_16: "f32[4, 384, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_16, 6.0);  clamp_min_16 = None
    convolution_25: "f32[4, 384, 14, 14]" = torch.ops.aten.convolution.default(clamp_max_16, primals_76, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 384)
    convert_element_type_50: "f32[384]" = torch.ops.prims.convert_element_type.default(primals_234, torch.float32)
    convert_element_type_51: "f32[384]" = torch.ops.prims.convert_element_type.default(primals_235, torch.float32)
    add_54: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_51, 1e-05);  convert_element_type_51 = None
    sqrt_25: "f32[384]" = torch.ops.aten.sqrt.default(add_54);  add_54 = None
    reciprocal_25: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
    mul_75: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
    unsqueeze_200: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_50, -1);  convert_element_type_50 = None
    unsqueeze_201: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    unsqueeze_202: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_75, -1);  mul_75 = None
    unsqueeze_203: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    sub_25: "f32[4, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_201);  unsqueeze_201 = None
    mul_76: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_203);  sub_25 = unsqueeze_203 = None
    unsqueeze_204: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
    unsqueeze_205: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_77: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_76, unsqueeze_205);  mul_76 = unsqueeze_205 = None
    unsqueeze_206: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
    unsqueeze_207: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_55: "f32[4, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_77, unsqueeze_207);  mul_77 = unsqueeze_207 = None
    clone_17: "f32[4, 384, 14, 14]" = torch.ops.aten.clone.default(add_55)
    clamp_min_17: "f32[4, 384, 14, 14]" = torch.ops.aten.clamp_min.default(add_55, 0.0);  add_55 = None
    clamp_max_17: "f32[4, 384, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_17, 6.0);  clamp_min_17 = None
    convolution_26: "f32[4, 64, 14, 14]" = torch.ops.aten.convolution.default(clamp_max_17, primals_79, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_52: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_237, torch.float32)
    convert_element_type_53: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_238, torch.float32)
    add_56: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_53, 1e-05);  convert_element_type_53 = None
    sqrt_26: "f32[64]" = torch.ops.aten.sqrt.default(add_56);  add_56 = None
    reciprocal_26: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
    mul_78: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
    unsqueeze_208: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_52, -1);  convert_element_type_52 = None
    unsqueeze_209: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    unsqueeze_210: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_78, -1);  mul_78 = None
    unsqueeze_211: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    sub_26: "f32[4, 64, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_209);  unsqueeze_209 = None
    mul_79: "f32[4, 64, 14, 14]" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_211);  sub_26 = unsqueeze_211 = None
    unsqueeze_212: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1)
    unsqueeze_213: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_80: "f32[4, 64, 14, 14]" = torch.ops.aten.mul.Tensor(mul_79, unsqueeze_213);  mul_79 = unsqueeze_213 = None
    unsqueeze_214: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1);  primals_81 = None
    unsqueeze_215: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_57: "f32[4, 64, 14, 14]" = torch.ops.aten.add.Tensor(mul_80, unsqueeze_215);  mul_80 = unsqueeze_215 = None
    add_58: "f32[4, 64, 14, 14]" = torch.ops.aten.add.Tensor(add_51, add_57);  add_57 = None
    convolution_27: "f32[4, 384, 14, 14]" = torch.ops.aten.convolution.default(add_58, primals_82, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_54: "f32[384]" = torch.ops.prims.convert_element_type.default(primals_240, torch.float32)
    convert_element_type_55: "f32[384]" = torch.ops.prims.convert_element_type.default(primals_241, torch.float32)
    add_59: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_55, 1e-05);  convert_element_type_55 = None
    sqrt_27: "f32[384]" = torch.ops.aten.sqrt.default(add_59);  add_59 = None
    reciprocal_27: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
    mul_81: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
    unsqueeze_216: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_54, -1);  convert_element_type_54 = None
    unsqueeze_217: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    unsqueeze_218: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_81, -1);  mul_81 = None
    unsqueeze_219: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    sub_27: "f32[4, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_217);  unsqueeze_217 = None
    mul_82: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_219);  sub_27 = unsqueeze_219 = None
    unsqueeze_220: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1)
    unsqueeze_221: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_83: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_82, unsqueeze_221);  mul_82 = unsqueeze_221 = None
    unsqueeze_222: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
    unsqueeze_223: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_60: "f32[4, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_223);  mul_83 = unsqueeze_223 = None
    clone_18: "f32[4, 384, 14, 14]" = torch.ops.aten.clone.default(add_60)
    clamp_min_18: "f32[4, 384, 14, 14]" = torch.ops.aten.clamp_min.default(add_60, 0.0);  add_60 = None
    clamp_max_18: "f32[4, 384, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_18, 6.0);  clamp_min_18 = None
    convolution_28: "f32[4, 384, 14, 14]" = torch.ops.aten.convolution.default(clamp_max_18, primals_85, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 384)
    convert_element_type_56: "f32[384]" = torch.ops.prims.convert_element_type.default(primals_243, torch.float32)
    convert_element_type_57: "f32[384]" = torch.ops.prims.convert_element_type.default(primals_244, torch.float32)
    add_61: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_57, 1e-05);  convert_element_type_57 = None
    sqrt_28: "f32[384]" = torch.ops.aten.sqrt.default(add_61);  add_61 = None
    reciprocal_28: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
    mul_84: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
    unsqueeze_224: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_56, -1);  convert_element_type_56 = None
    unsqueeze_225: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    unsqueeze_226: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_84, -1);  mul_84 = None
    unsqueeze_227: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    sub_28: "f32[4, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_225);  unsqueeze_225 = None
    mul_85: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_227);  sub_28 = unsqueeze_227 = None
    unsqueeze_228: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1)
    unsqueeze_229: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_86: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_85, unsqueeze_229);  mul_85 = unsqueeze_229 = None
    unsqueeze_230: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1);  primals_87 = None
    unsqueeze_231: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_62: "f32[4, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_86, unsqueeze_231);  mul_86 = unsqueeze_231 = None
    clone_19: "f32[4, 384, 14, 14]" = torch.ops.aten.clone.default(add_62)
    clamp_min_19: "f32[4, 384, 14, 14]" = torch.ops.aten.clamp_min.default(add_62, 0.0);  add_62 = None
    clamp_max_19: "f32[4, 384, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_19, 6.0);  clamp_min_19 = None
    convolution_29: "f32[4, 64, 14, 14]" = torch.ops.aten.convolution.default(clamp_max_19, primals_88, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_58: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_246, torch.float32)
    convert_element_type_59: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_247, torch.float32)
    add_63: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_59, 1e-05);  convert_element_type_59 = None
    sqrt_29: "f32[64]" = torch.ops.aten.sqrt.default(add_63);  add_63 = None
    reciprocal_29: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
    mul_87: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
    unsqueeze_232: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_58, -1);  convert_element_type_58 = None
    unsqueeze_233: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    unsqueeze_234: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_87, -1);  mul_87 = None
    unsqueeze_235: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    sub_29: "f32[4, 64, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_233);  unsqueeze_233 = None
    mul_88: "f32[4, 64, 14, 14]" = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_235);  sub_29 = unsqueeze_235 = None
    unsqueeze_236: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1)
    unsqueeze_237: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_89: "f32[4, 64, 14, 14]" = torch.ops.aten.mul.Tensor(mul_88, unsqueeze_237);  mul_88 = unsqueeze_237 = None
    unsqueeze_238: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1);  primals_90 = None
    unsqueeze_239: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_64: "f32[4, 64, 14, 14]" = torch.ops.aten.add.Tensor(mul_89, unsqueeze_239);  mul_89 = unsqueeze_239 = None
    add_65: "f32[4, 64, 14, 14]" = torch.ops.aten.add.Tensor(add_58, add_64);  add_64 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:64, code: return self.conv(x)
    convolution_30: "f32[4, 384, 14, 14]" = torch.ops.aten.convolution.default(add_65, primals_91, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_60: "f32[384]" = torch.ops.prims.convert_element_type.default(primals_249, torch.float32)
    convert_element_type_61: "f32[384]" = torch.ops.prims.convert_element_type.default(primals_250, torch.float32)
    add_66: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_61, 1e-05);  convert_element_type_61 = None
    sqrt_30: "f32[384]" = torch.ops.aten.sqrt.default(add_66);  add_66 = None
    reciprocal_30: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
    mul_90: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
    unsqueeze_240: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_60, -1);  convert_element_type_60 = None
    unsqueeze_241: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    unsqueeze_242: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_90, -1);  mul_90 = None
    unsqueeze_243: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    sub_30: "f32[4, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_241);  unsqueeze_241 = None
    mul_91: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_243);  sub_30 = unsqueeze_243 = None
    unsqueeze_244: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1)
    unsqueeze_245: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_92: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_245);  mul_91 = unsqueeze_245 = None
    unsqueeze_246: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_93, -1);  primals_93 = None
    unsqueeze_247: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_67: "f32[4, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_92, unsqueeze_247);  mul_92 = unsqueeze_247 = None
    clone_20: "f32[4, 384, 14, 14]" = torch.ops.aten.clone.default(add_67)
    clamp_min_20: "f32[4, 384, 14, 14]" = torch.ops.aten.clamp_min.default(add_67, 0.0);  add_67 = None
    clamp_max_20: "f32[4, 384, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_20, 6.0);  clamp_min_20 = None
    convolution_31: "f32[4, 384, 14, 14]" = torch.ops.aten.convolution.default(clamp_max_20, primals_94, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 384)
    convert_element_type_62: "f32[384]" = torch.ops.prims.convert_element_type.default(primals_252, torch.float32)
    convert_element_type_63: "f32[384]" = torch.ops.prims.convert_element_type.default(primals_253, torch.float32)
    add_68: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_63, 1e-05);  convert_element_type_63 = None
    sqrt_31: "f32[384]" = torch.ops.aten.sqrt.default(add_68);  add_68 = None
    reciprocal_31: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
    mul_93: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
    unsqueeze_248: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_62, -1);  convert_element_type_62 = None
    unsqueeze_249: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    unsqueeze_250: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_93, -1);  mul_93 = None
    unsqueeze_251: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    sub_31: "f32[4, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_249);  unsqueeze_249 = None
    mul_94: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_251);  sub_31 = unsqueeze_251 = None
    unsqueeze_252: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1)
    unsqueeze_253: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_95: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_94, unsqueeze_253);  mul_94 = unsqueeze_253 = None
    unsqueeze_254: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1);  primals_96 = None
    unsqueeze_255: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_69: "f32[4, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_95, unsqueeze_255);  mul_95 = unsqueeze_255 = None
    clone_21: "f32[4, 384, 14, 14]" = torch.ops.aten.clone.default(add_69)
    clamp_min_21: "f32[4, 384, 14, 14]" = torch.ops.aten.clamp_min.default(add_69, 0.0);  add_69 = None
    clamp_max_21: "f32[4, 384, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_21, 6.0);  clamp_min_21 = None
    convolution_32: "f32[4, 96, 14, 14]" = torch.ops.aten.convolution.default(clamp_max_21, primals_97, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_64: "f32[96]" = torch.ops.prims.convert_element_type.default(primals_255, torch.float32)
    convert_element_type_65: "f32[96]" = torch.ops.prims.convert_element_type.default(primals_256, torch.float32)
    add_70: "f32[96]" = torch.ops.aten.add.Tensor(convert_element_type_65, 1e-05);  convert_element_type_65 = None
    sqrt_32: "f32[96]" = torch.ops.aten.sqrt.default(add_70);  add_70 = None
    reciprocal_32: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
    mul_96: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
    unsqueeze_256: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_64, -1);  convert_element_type_64 = None
    unsqueeze_257: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    unsqueeze_258: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_96, -1);  mul_96 = None
    unsqueeze_259: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    sub_32: "f32[4, 96, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_257);  unsqueeze_257 = None
    mul_97: "f32[4, 96, 14, 14]" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_259);  sub_32 = unsqueeze_259 = None
    unsqueeze_260: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1)
    unsqueeze_261: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
    mul_98: "f32[4, 96, 14, 14]" = torch.ops.aten.mul.Tensor(mul_97, unsqueeze_261);  mul_97 = unsqueeze_261 = None
    unsqueeze_262: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1);  primals_99 = None
    unsqueeze_263: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
    add_71: "f32[4, 96, 14, 14]" = torch.ops.aten.add.Tensor(mul_98, unsqueeze_263);  mul_98 = unsqueeze_263 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    convolution_33: "f32[4, 576, 14, 14]" = torch.ops.aten.convolution.default(add_71, primals_100, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_66: "f32[576]" = torch.ops.prims.convert_element_type.default(primals_258, torch.float32)
    convert_element_type_67: "f32[576]" = torch.ops.prims.convert_element_type.default(primals_259, torch.float32)
    add_72: "f32[576]" = torch.ops.aten.add.Tensor(convert_element_type_67, 1e-05);  convert_element_type_67 = None
    sqrt_33: "f32[576]" = torch.ops.aten.sqrt.default(add_72);  add_72 = None
    reciprocal_33: "f32[576]" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
    mul_99: "f32[576]" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
    unsqueeze_264: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_66, -1);  convert_element_type_66 = None
    unsqueeze_265: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
    unsqueeze_266: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(mul_99, -1);  mul_99 = None
    unsqueeze_267: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
    sub_33: "f32[4, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_265);  unsqueeze_265 = None
    mul_100: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_33, unsqueeze_267);  sub_33 = unsqueeze_267 = None
    unsqueeze_268: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_101, -1)
    unsqueeze_269: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
    mul_101: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(mul_100, unsqueeze_269);  mul_100 = unsqueeze_269 = None
    unsqueeze_270: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1);  primals_102 = None
    unsqueeze_271: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
    add_73: "f32[4, 576, 14, 14]" = torch.ops.aten.add.Tensor(mul_101, unsqueeze_271);  mul_101 = unsqueeze_271 = None
    clone_22: "f32[4, 576, 14, 14]" = torch.ops.aten.clone.default(add_73)
    clamp_min_22: "f32[4, 576, 14, 14]" = torch.ops.aten.clamp_min.default(add_73, 0.0);  add_73 = None
    clamp_max_22: "f32[4, 576, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_22, 6.0);  clamp_min_22 = None
    convolution_34: "f32[4, 576, 14, 14]" = torch.ops.aten.convolution.default(clamp_max_22, primals_103, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 576)
    convert_element_type_68: "f32[576]" = torch.ops.prims.convert_element_type.default(primals_261, torch.float32)
    convert_element_type_69: "f32[576]" = torch.ops.prims.convert_element_type.default(primals_262, torch.float32)
    add_74: "f32[576]" = torch.ops.aten.add.Tensor(convert_element_type_69, 1e-05);  convert_element_type_69 = None
    sqrt_34: "f32[576]" = torch.ops.aten.sqrt.default(add_74);  add_74 = None
    reciprocal_34: "f32[576]" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
    mul_102: "f32[576]" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
    unsqueeze_272: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_68, -1);  convert_element_type_68 = None
    unsqueeze_273: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
    unsqueeze_274: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(mul_102, -1);  mul_102 = None
    unsqueeze_275: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
    sub_34: "f32[4, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_273);  unsqueeze_273 = None
    mul_103: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_275);  sub_34 = unsqueeze_275 = None
    unsqueeze_276: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_104, -1)
    unsqueeze_277: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
    mul_104: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(mul_103, unsqueeze_277);  mul_103 = unsqueeze_277 = None
    unsqueeze_278: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_105, -1);  primals_105 = None
    unsqueeze_279: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
    add_75: "f32[4, 576, 14, 14]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_279);  mul_104 = unsqueeze_279 = None
    clone_23: "f32[4, 576, 14, 14]" = torch.ops.aten.clone.default(add_75)
    clamp_min_23: "f32[4, 576, 14, 14]" = torch.ops.aten.clamp_min.default(add_75, 0.0);  add_75 = None
    clamp_max_23: "f32[4, 576, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_23, 6.0);  clamp_min_23 = None
    convolution_35: "f32[4, 96, 14, 14]" = torch.ops.aten.convolution.default(clamp_max_23, primals_106, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_70: "f32[96]" = torch.ops.prims.convert_element_type.default(primals_264, torch.float32)
    convert_element_type_71: "f32[96]" = torch.ops.prims.convert_element_type.default(primals_265, torch.float32)
    add_76: "f32[96]" = torch.ops.aten.add.Tensor(convert_element_type_71, 1e-05);  convert_element_type_71 = None
    sqrt_35: "f32[96]" = torch.ops.aten.sqrt.default(add_76);  add_76 = None
    reciprocal_35: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
    mul_105: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
    unsqueeze_280: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_70, -1);  convert_element_type_70 = None
    unsqueeze_281: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
    unsqueeze_282: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_105, -1);  mul_105 = None
    unsqueeze_283: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
    sub_35: "f32[4, 96, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_281);  unsqueeze_281 = None
    mul_106: "f32[4, 96, 14, 14]" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_283);  sub_35 = unsqueeze_283 = None
    unsqueeze_284: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_107, -1)
    unsqueeze_285: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
    mul_107: "f32[4, 96, 14, 14]" = torch.ops.aten.mul.Tensor(mul_106, unsqueeze_285);  mul_106 = unsqueeze_285 = None
    unsqueeze_286: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1);  primals_108 = None
    unsqueeze_287: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
    add_77: "f32[4, 96, 14, 14]" = torch.ops.aten.add.Tensor(mul_107, unsqueeze_287);  mul_107 = unsqueeze_287 = None
    add_78: "f32[4, 96, 14, 14]" = torch.ops.aten.add.Tensor(add_71, add_77);  add_77 = None
    convolution_36: "f32[4, 576, 14, 14]" = torch.ops.aten.convolution.default(add_78, primals_109, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_72: "f32[576]" = torch.ops.prims.convert_element_type.default(primals_267, torch.float32)
    convert_element_type_73: "f32[576]" = torch.ops.prims.convert_element_type.default(primals_268, torch.float32)
    add_79: "f32[576]" = torch.ops.aten.add.Tensor(convert_element_type_73, 1e-05);  convert_element_type_73 = None
    sqrt_36: "f32[576]" = torch.ops.aten.sqrt.default(add_79);  add_79 = None
    reciprocal_36: "f32[576]" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
    mul_108: "f32[576]" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
    unsqueeze_288: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_72, -1);  convert_element_type_72 = None
    unsqueeze_289: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
    unsqueeze_290: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(mul_108, -1);  mul_108 = None
    unsqueeze_291: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
    sub_36: "f32[4, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_289);  unsqueeze_289 = None
    mul_109: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_291);  sub_36 = unsqueeze_291 = None
    unsqueeze_292: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_110, -1)
    unsqueeze_293: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
    mul_110: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(mul_109, unsqueeze_293);  mul_109 = unsqueeze_293 = None
    unsqueeze_294: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_111, -1);  primals_111 = None
    unsqueeze_295: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
    add_80: "f32[4, 576, 14, 14]" = torch.ops.aten.add.Tensor(mul_110, unsqueeze_295);  mul_110 = unsqueeze_295 = None
    clone_24: "f32[4, 576, 14, 14]" = torch.ops.aten.clone.default(add_80)
    clamp_min_24: "f32[4, 576, 14, 14]" = torch.ops.aten.clamp_min.default(add_80, 0.0);  add_80 = None
    clamp_max_24: "f32[4, 576, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_24, 6.0);  clamp_min_24 = None
    convolution_37: "f32[4, 576, 14, 14]" = torch.ops.aten.convolution.default(clamp_max_24, primals_112, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 576)
    convert_element_type_74: "f32[576]" = torch.ops.prims.convert_element_type.default(primals_270, torch.float32)
    convert_element_type_75: "f32[576]" = torch.ops.prims.convert_element_type.default(primals_271, torch.float32)
    add_81: "f32[576]" = torch.ops.aten.add.Tensor(convert_element_type_75, 1e-05);  convert_element_type_75 = None
    sqrt_37: "f32[576]" = torch.ops.aten.sqrt.default(add_81);  add_81 = None
    reciprocal_37: "f32[576]" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
    mul_111: "f32[576]" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
    unsqueeze_296: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_74, -1);  convert_element_type_74 = None
    unsqueeze_297: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
    unsqueeze_298: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(mul_111, -1);  mul_111 = None
    unsqueeze_299: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
    sub_37: "f32[4, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_297);  unsqueeze_297 = None
    mul_112: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_299);  sub_37 = unsqueeze_299 = None
    unsqueeze_300: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_113, -1)
    unsqueeze_301: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
    mul_113: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_301);  mul_112 = unsqueeze_301 = None
    unsqueeze_302: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_114, -1);  primals_114 = None
    unsqueeze_303: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
    add_82: "f32[4, 576, 14, 14]" = torch.ops.aten.add.Tensor(mul_113, unsqueeze_303);  mul_113 = unsqueeze_303 = None
    clone_25: "f32[4, 576, 14, 14]" = torch.ops.aten.clone.default(add_82)
    clamp_min_25: "f32[4, 576, 14, 14]" = torch.ops.aten.clamp_min.default(add_82, 0.0);  add_82 = None
    clamp_max_25: "f32[4, 576, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_25, 6.0);  clamp_min_25 = None
    convolution_38: "f32[4, 96, 14, 14]" = torch.ops.aten.convolution.default(clamp_max_25, primals_115, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_76: "f32[96]" = torch.ops.prims.convert_element_type.default(primals_273, torch.float32)
    convert_element_type_77: "f32[96]" = torch.ops.prims.convert_element_type.default(primals_274, torch.float32)
    add_83: "f32[96]" = torch.ops.aten.add.Tensor(convert_element_type_77, 1e-05);  convert_element_type_77 = None
    sqrt_38: "f32[96]" = torch.ops.aten.sqrt.default(add_83);  add_83 = None
    reciprocal_38: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
    mul_114: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
    unsqueeze_304: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_76, -1);  convert_element_type_76 = None
    unsqueeze_305: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
    unsqueeze_306: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_114, -1);  mul_114 = None
    unsqueeze_307: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
    sub_38: "f32[4, 96, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_305);  unsqueeze_305 = None
    mul_115: "f32[4, 96, 14, 14]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_307);  sub_38 = unsqueeze_307 = None
    unsqueeze_308: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_116, -1)
    unsqueeze_309: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
    mul_116: "f32[4, 96, 14, 14]" = torch.ops.aten.mul.Tensor(mul_115, unsqueeze_309);  mul_115 = unsqueeze_309 = None
    unsqueeze_310: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_117, -1);  primals_117 = None
    unsqueeze_311: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
    add_84: "f32[4, 96, 14, 14]" = torch.ops.aten.add.Tensor(mul_116, unsqueeze_311);  mul_116 = unsqueeze_311 = None
    add_85: "f32[4, 96, 14, 14]" = torch.ops.aten.add.Tensor(add_78, add_84);  add_84 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:64, code: return self.conv(x)
    convolution_39: "f32[4, 576, 14, 14]" = torch.ops.aten.convolution.default(add_85, primals_118, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_78: "f32[576]" = torch.ops.prims.convert_element_type.default(primals_276, torch.float32)
    convert_element_type_79: "f32[576]" = torch.ops.prims.convert_element_type.default(primals_277, torch.float32)
    add_86: "f32[576]" = torch.ops.aten.add.Tensor(convert_element_type_79, 1e-05);  convert_element_type_79 = None
    sqrt_39: "f32[576]" = torch.ops.aten.sqrt.default(add_86);  add_86 = None
    reciprocal_39: "f32[576]" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
    mul_117: "f32[576]" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
    unsqueeze_312: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_78, -1);  convert_element_type_78 = None
    unsqueeze_313: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
    unsqueeze_314: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(mul_117, -1);  mul_117 = None
    unsqueeze_315: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
    sub_39: "f32[4, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_313);  unsqueeze_313 = None
    mul_118: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_315);  sub_39 = unsqueeze_315 = None
    unsqueeze_316: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_119, -1)
    unsqueeze_317: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
    mul_119: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(mul_118, unsqueeze_317);  mul_118 = unsqueeze_317 = None
    unsqueeze_318: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_120, -1);  primals_120 = None
    unsqueeze_319: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
    add_87: "f32[4, 576, 14, 14]" = torch.ops.aten.add.Tensor(mul_119, unsqueeze_319);  mul_119 = unsqueeze_319 = None
    clone_26: "f32[4, 576, 14, 14]" = torch.ops.aten.clone.default(add_87)
    clamp_min_26: "f32[4, 576, 14, 14]" = torch.ops.aten.clamp_min.default(add_87, 0.0);  add_87 = None
    clamp_max_26: "f32[4, 576, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_26, 6.0);  clamp_min_26 = None
    convolution_40: "f32[4, 576, 7, 7]" = torch.ops.aten.convolution.default(clamp_max_26, primals_121, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 576)
    convert_element_type_80: "f32[576]" = torch.ops.prims.convert_element_type.default(primals_279, torch.float32)
    convert_element_type_81: "f32[576]" = torch.ops.prims.convert_element_type.default(primals_280, torch.float32)
    add_88: "f32[576]" = torch.ops.aten.add.Tensor(convert_element_type_81, 1e-05);  convert_element_type_81 = None
    sqrt_40: "f32[576]" = torch.ops.aten.sqrt.default(add_88);  add_88 = None
    reciprocal_40: "f32[576]" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
    mul_120: "f32[576]" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
    unsqueeze_320: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_80, -1);  convert_element_type_80 = None
    unsqueeze_321: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
    unsqueeze_322: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(mul_120, -1);  mul_120 = None
    unsqueeze_323: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
    sub_40: "f32[4, 576, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_321);  unsqueeze_321 = None
    mul_121: "f32[4, 576, 7, 7]" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_323);  sub_40 = unsqueeze_323 = None
    unsqueeze_324: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_122, -1)
    unsqueeze_325: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
    mul_122: "f32[4, 576, 7, 7]" = torch.ops.aten.mul.Tensor(mul_121, unsqueeze_325);  mul_121 = unsqueeze_325 = None
    unsqueeze_326: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_123, -1);  primals_123 = None
    unsqueeze_327: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
    add_89: "f32[4, 576, 7, 7]" = torch.ops.aten.add.Tensor(mul_122, unsqueeze_327);  mul_122 = unsqueeze_327 = None
    clone_27: "f32[4, 576, 7, 7]" = torch.ops.aten.clone.default(add_89)
    clamp_min_27: "f32[4, 576, 7, 7]" = torch.ops.aten.clamp_min.default(add_89, 0.0);  add_89 = None
    clamp_max_27: "f32[4, 576, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_27, 6.0);  clamp_min_27 = None
    convolution_41: "f32[4, 160, 7, 7]" = torch.ops.aten.convolution.default(clamp_max_27, primals_124, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_82: "f32[160]" = torch.ops.prims.convert_element_type.default(primals_282, torch.float32)
    convert_element_type_83: "f32[160]" = torch.ops.prims.convert_element_type.default(primals_283, torch.float32)
    add_90: "f32[160]" = torch.ops.aten.add.Tensor(convert_element_type_83, 1e-05);  convert_element_type_83 = None
    sqrt_41: "f32[160]" = torch.ops.aten.sqrt.default(add_90);  add_90 = None
    reciprocal_41: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
    mul_123: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
    unsqueeze_328: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_82, -1);  convert_element_type_82 = None
    unsqueeze_329: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
    unsqueeze_330: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_123, -1);  mul_123 = None
    unsqueeze_331: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
    sub_41: "f32[4, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_329);  unsqueeze_329 = None
    mul_124: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_331);  sub_41 = unsqueeze_331 = None
    unsqueeze_332: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_125, -1)
    unsqueeze_333: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
    mul_125: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(mul_124, unsqueeze_333);  mul_124 = unsqueeze_333 = None
    unsqueeze_334: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_126, -1);  primals_126 = None
    unsqueeze_335: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
    add_91: "f32[4, 160, 7, 7]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_335);  mul_125 = unsqueeze_335 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    convolution_42: "f32[4, 960, 7, 7]" = torch.ops.aten.convolution.default(add_91, primals_127, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_84: "f32[960]" = torch.ops.prims.convert_element_type.default(primals_285, torch.float32)
    convert_element_type_85: "f32[960]" = torch.ops.prims.convert_element_type.default(primals_286, torch.float32)
    add_92: "f32[960]" = torch.ops.aten.add.Tensor(convert_element_type_85, 1e-05);  convert_element_type_85 = None
    sqrt_42: "f32[960]" = torch.ops.aten.sqrt.default(add_92);  add_92 = None
    reciprocal_42: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
    mul_126: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
    unsqueeze_336: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_84, -1);  convert_element_type_84 = None
    unsqueeze_337: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
    unsqueeze_338: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_126, -1);  mul_126 = None
    unsqueeze_339: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
    sub_42: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_337);  unsqueeze_337 = None
    mul_127: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_339);  sub_42 = unsqueeze_339 = None
    unsqueeze_340: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_128, -1)
    unsqueeze_341: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
    mul_128: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_127, unsqueeze_341);  mul_127 = unsqueeze_341 = None
    unsqueeze_342: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_129, -1);  primals_129 = None
    unsqueeze_343: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
    add_93: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_128, unsqueeze_343);  mul_128 = unsqueeze_343 = None
    clone_28: "f32[4, 960, 7, 7]" = torch.ops.aten.clone.default(add_93)
    clamp_min_28: "f32[4, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_93, 0.0);  add_93 = None
    clamp_max_28: "f32[4, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_28, 6.0);  clamp_min_28 = None
    convolution_43: "f32[4, 960, 7, 7]" = torch.ops.aten.convolution.default(clamp_max_28, primals_130, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 960)
    convert_element_type_86: "f32[960]" = torch.ops.prims.convert_element_type.default(primals_288, torch.float32)
    convert_element_type_87: "f32[960]" = torch.ops.prims.convert_element_type.default(primals_289, torch.float32)
    add_94: "f32[960]" = torch.ops.aten.add.Tensor(convert_element_type_87, 1e-05);  convert_element_type_87 = None
    sqrt_43: "f32[960]" = torch.ops.aten.sqrt.default(add_94);  add_94 = None
    reciprocal_43: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
    mul_129: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
    unsqueeze_344: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_86, -1);  convert_element_type_86 = None
    unsqueeze_345: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
    unsqueeze_346: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_129, -1);  mul_129 = None
    unsqueeze_347: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
    sub_43: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_345);  unsqueeze_345 = None
    mul_130: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_347);  sub_43 = unsqueeze_347 = None
    unsqueeze_348: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_131, -1)
    unsqueeze_349: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
    mul_131: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_130, unsqueeze_349);  mul_130 = unsqueeze_349 = None
    unsqueeze_350: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_132, -1);  primals_132 = None
    unsqueeze_351: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
    add_95: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_131, unsqueeze_351);  mul_131 = unsqueeze_351 = None
    clone_29: "f32[4, 960, 7, 7]" = torch.ops.aten.clone.default(add_95)
    clamp_min_29: "f32[4, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_95, 0.0);  add_95 = None
    clamp_max_29: "f32[4, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_29, 6.0);  clamp_min_29 = None
    convolution_44: "f32[4, 160, 7, 7]" = torch.ops.aten.convolution.default(clamp_max_29, primals_133, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_88: "f32[160]" = torch.ops.prims.convert_element_type.default(primals_291, torch.float32)
    convert_element_type_89: "f32[160]" = torch.ops.prims.convert_element_type.default(primals_292, torch.float32)
    add_96: "f32[160]" = torch.ops.aten.add.Tensor(convert_element_type_89, 1e-05);  convert_element_type_89 = None
    sqrt_44: "f32[160]" = torch.ops.aten.sqrt.default(add_96);  add_96 = None
    reciprocal_44: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
    mul_132: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
    unsqueeze_352: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_88, -1);  convert_element_type_88 = None
    unsqueeze_353: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
    unsqueeze_354: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_132, -1);  mul_132 = None
    unsqueeze_355: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
    sub_44: "f32[4, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_353);  unsqueeze_353 = None
    mul_133: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_355);  sub_44 = unsqueeze_355 = None
    unsqueeze_356: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_134, -1)
    unsqueeze_357: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
    mul_134: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_357);  mul_133 = unsqueeze_357 = None
    unsqueeze_358: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_135, -1);  primals_135 = None
    unsqueeze_359: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
    add_97: "f32[4, 160, 7, 7]" = torch.ops.aten.add.Tensor(mul_134, unsqueeze_359);  mul_134 = unsqueeze_359 = None
    add_98: "f32[4, 160, 7, 7]" = torch.ops.aten.add.Tensor(add_91, add_97);  add_97 = None
    convolution_45: "f32[4, 960, 7, 7]" = torch.ops.aten.convolution.default(add_98, primals_136, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_90: "f32[960]" = torch.ops.prims.convert_element_type.default(primals_294, torch.float32)
    convert_element_type_91: "f32[960]" = torch.ops.prims.convert_element_type.default(primals_295, torch.float32)
    add_99: "f32[960]" = torch.ops.aten.add.Tensor(convert_element_type_91, 1e-05);  convert_element_type_91 = None
    sqrt_45: "f32[960]" = torch.ops.aten.sqrt.default(add_99);  add_99 = None
    reciprocal_45: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
    mul_135: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
    unsqueeze_360: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_90, -1);  convert_element_type_90 = None
    unsqueeze_361: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
    unsqueeze_362: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_135, -1);  mul_135 = None
    unsqueeze_363: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
    sub_45: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_361);  unsqueeze_361 = None
    mul_136: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_363);  sub_45 = unsqueeze_363 = None
    unsqueeze_364: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_137, -1)
    unsqueeze_365: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
    mul_137: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_365);  mul_136 = unsqueeze_365 = None
    unsqueeze_366: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_138, -1);  primals_138 = None
    unsqueeze_367: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
    add_100: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_137, unsqueeze_367);  mul_137 = unsqueeze_367 = None
    clone_30: "f32[4, 960, 7, 7]" = torch.ops.aten.clone.default(add_100)
    clamp_min_30: "f32[4, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_100, 0.0);  add_100 = None
    clamp_max_30: "f32[4, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_30, 6.0);  clamp_min_30 = None
    convolution_46: "f32[4, 960, 7, 7]" = torch.ops.aten.convolution.default(clamp_max_30, primals_139, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 960)
    convert_element_type_92: "f32[960]" = torch.ops.prims.convert_element_type.default(primals_297, torch.float32)
    convert_element_type_93: "f32[960]" = torch.ops.prims.convert_element_type.default(primals_298, torch.float32)
    add_101: "f32[960]" = torch.ops.aten.add.Tensor(convert_element_type_93, 1e-05);  convert_element_type_93 = None
    sqrt_46: "f32[960]" = torch.ops.aten.sqrt.default(add_101);  add_101 = None
    reciprocal_46: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
    mul_138: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
    unsqueeze_368: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_92, -1);  convert_element_type_92 = None
    unsqueeze_369: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
    unsqueeze_370: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_138, -1);  mul_138 = None
    unsqueeze_371: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
    sub_46: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_369);  unsqueeze_369 = None
    mul_139: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_371);  sub_46 = unsqueeze_371 = None
    unsqueeze_372: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_140, -1)
    unsqueeze_373: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
    mul_140: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_139, unsqueeze_373);  mul_139 = unsqueeze_373 = None
    unsqueeze_374: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_141, -1);  primals_141 = None
    unsqueeze_375: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
    add_102: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_140, unsqueeze_375);  mul_140 = unsqueeze_375 = None
    clone_31: "f32[4, 960, 7, 7]" = torch.ops.aten.clone.default(add_102)
    clamp_min_31: "f32[4, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_102, 0.0);  add_102 = None
    clamp_max_31: "f32[4, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_31, 6.0);  clamp_min_31 = None
    convolution_47: "f32[4, 160, 7, 7]" = torch.ops.aten.convolution.default(clamp_max_31, primals_142, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_94: "f32[160]" = torch.ops.prims.convert_element_type.default(primals_300, torch.float32)
    convert_element_type_95: "f32[160]" = torch.ops.prims.convert_element_type.default(primals_301, torch.float32)
    add_103: "f32[160]" = torch.ops.aten.add.Tensor(convert_element_type_95, 1e-05);  convert_element_type_95 = None
    sqrt_47: "f32[160]" = torch.ops.aten.sqrt.default(add_103);  add_103 = None
    reciprocal_47: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
    mul_141: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
    unsqueeze_376: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_94, -1);  convert_element_type_94 = None
    unsqueeze_377: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
    unsqueeze_378: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_141, -1);  mul_141 = None
    unsqueeze_379: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
    sub_47: "f32[4, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_377);  unsqueeze_377 = None
    mul_142: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_379);  sub_47 = unsqueeze_379 = None
    unsqueeze_380: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_143, -1)
    unsqueeze_381: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
    mul_143: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(mul_142, unsqueeze_381);  mul_142 = unsqueeze_381 = None
    unsqueeze_382: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_144, -1);  primals_144 = None
    unsqueeze_383: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
    add_104: "f32[4, 160, 7, 7]" = torch.ops.aten.add.Tensor(mul_143, unsqueeze_383);  mul_143 = unsqueeze_383 = None
    add_105: "f32[4, 160, 7, 7]" = torch.ops.aten.add.Tensor(add_98, add_104);  add_104 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:64, code: return self.conv(x)
    convolution_48: "f32[4, 960, 7, 7]" = torch.ops.aten.convolution.default(add_105, primals_145, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_96: "f32[960]" = torch.ops.prims.convert_element_type.default(primals_303, torch.float32)
    convert_element_type_97: "f32[960]" = torch.ops.prims.convert_element_type.default(primals_304, torch.float32)
    add_106: "f32[960]" = torch.ops.aten.add.Tensor(convert_element_type_97, 1e-05);  convert_element_type_97 = None
    sqrt_48: "f32[960]" = torch.ops.aten.sqrt.default(add_106);  add_106 = None
    reciprocal_48: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
    mul_144: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
    unsqueeze_384: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_96, -1);  convert_element_type_96 = None
    unsqueeze_385: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
    unsqueeze_386: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_144, -1);  mul_144 = None
    unsqueeze_387: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
    sub_48: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_385);  unsqueeze_385 = None
    mul_145: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_387);  sub_48 = unsqueeze_387 = None
    unsqueeze_388: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_146, -1)
    unsqueeze_389: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
    mul_146: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_145, unsqueeze_389);  mul_145 = unsqueeze_389 = None
    unsqueeze_390: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_147, -1);  primals_147 = None
    unsqueeze_391: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
    add_107: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_391);  mul_146 = unsqueeze_391 = None
    clone_32: "f32[4, 960, 7, 7]" = torch.ops.aten.clone.default(add_107)
    clamp_min_32: "f32[4, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_107, 0.0);  add_107 = None
    clamp_max_32: "f32[4, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_32, 6.0);  clamp_min_32 = None
    convolution_49: "f32[4, 960, 7, 7]" = torch.ops.aten.convolution.default(clamp_max_32, primals_148, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 960)
    convert_element_type_98: "f32[960]" = torch.ops.prims.convert_element_type.default(primals_306, torch.float32)
    convert_element_type_99: "f32[960]" = torch.ops.prims.convert_element_type.default(primals_307, torch.float32)
    add_108: "f32[960]" = torch.ops.aten.add.Tensor(convert_element_type_99, 1e-05);  convert_element_type_99 = None
    sqrt_49: "f32[960]" = torch.ops.aten.sqrt.default(add_108);  add_108 = None
    reciprocal_49: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
    mul_147: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
    unsqueeze_392: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_98, -1);  convert_element_type_98 = None
    unsqueeze_393: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
    unsqueeze_394: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_147, -1);  mul_147 = None
    unsqueeze_395: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
    sub_49: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_393);  unsqueeze_393 = None
    mul_148: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_395);  sub_49 = unsqueeze_395 = None
    unsqueeze_396: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_149, -1)
    unsqueeze_397: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
    mul_149: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_148, unsqueeze_397);  mul_148 = unsqueeze_397 = None
    unsqueeze_398: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_150, -1);  primals_150 = None
    unsqueeze_399: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
    add_109: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_399);  mul_149 = unsqueeze_399 = None
    clone_33: "f32[4, 960, 7, 7]" = torch.ops.aten.clone.default(add_109)
    clamp_min_33: "f32[4, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_109, 0.0);  add_109 = None
    clamp_max_33: "f32[4, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_33, 6.0);  clamp_min_33 = None
    convolution_50: "f32[4, 320, 7, 7]" = torch.ops.aten.convolution.default(clamp_max_33, primals_151, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_100: "f32[320]" = torch.ops.prims.convert_element_type.default(primals_309, torch.float32)
    convert_element_type_101: "f32[320]" = torch.ops.prims.convert_element_type.default(primals_310, torch.float32)
    add_110: "f32[320]" = torch.ops.aten.add.Tensor(convert_element_type_101, 1e-05);  convert_element_type_101 = None
    sqrt_50: "f32[320]" = torch.ops.aten.sqrt.default(add_110);  add_110 = None
    reciprocal_50: "f32[320]" = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
    mul_150: "f32[320]" = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
    unsqueeze_400: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_100, -1);  convert_element_type_100 = None
    unsqueeze_401: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
    unsqueeze_402: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(mul_150, -1);  mul_150 = None
    unsqueeze_403: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
    sub_50: "f32[4, 320, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_401);  unsqueeze_401 = None
    mul_151: "f32[4, 320, 7, 7]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_403);  sub_50 = unsqueeze_403 = None
    unsqueeze_404: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_152, -1)
    unsqueeze_405: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
    mul_152: "f32[4, 320, 7, 7]" = torch.ops.aten.mul.Tensor(mul_151, unsqueeze_405);  mul_151 = unsqueeze_405 = None
    unsqueeze_406: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_153, -1);  primals_153 = None
    unsqueeze_407: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
    add_111: "f32[4, 320, 7, 7]" = torch.ops.aten.add.Tensor(mul_152, unsqueeze_407);  mul_152 = unsqueeze_407 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:166, code: x = self.features(x)
    convolution_51: "f32[4, 1280, 7, 7]" = torch.ops.aten.convolution.default(add_111, primals_154, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_102: "f32[1280]" = torch.ops.prims.convert_element_type.default(primals_312, torch.float32)
    convert_element_type_103: "f32[1280]" = torch.ops.prims.convert_element_type.default(primals_313, torch.float32)
    add_112: "f32[1280]" = torch.ops.aten.add.Tensor(convert_element_type_103, 1e-05);  convert_element_type_103 = None
    sqrt_51: "f32[1280]" = torch.ops.aten.sqrt.default(add_112);  add_112 = None
    reciprocal_51: "f32[1280]" = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
    mul_153: "f32[1280]" = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
    unsqueeze_408: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_102, -1);  convert_element_type_102 = None
    unsqueeze_409: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
    unsqueeze_410: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(mul_153, -1);  mul_153 = None
    unsqueeze_411: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
    sub_51: "f32[4, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_409);  unsqueeze_409 = None
    mul_154: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_411);  sub_51 = unsqueeze_411 = None
    unsqueeze_412: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_155, -1)
    unsqueeze_413: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
    mul_155: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_413);  mul_154 = unsqueeze_413 = None
    unsqueeze_414: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_156, -1);  primals_156 = None
    unsqueeze_415: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
    add_113: "f32[4, 1280, 7, 7]" = torch.ops.aten.add.Tensor(mul_155, unsqueeze_415);  mul_155 = unsqueeze_415 = None
    clone_34: "f32[4, 1280, 7, 7]" = torch.ops.aten.clone.default(add_113)
    clamp_min_34: "f32[4, 1280, 7, 7]" = torch.ops.aten.clamp_min.default(add_113, 0.0);  add_113 = None
    clamp_max_34: "f32[4, 1280, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_34, 6.0);  clamp_min_34 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:168, code: x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
    mean: "f32[4, 1280, 1, 1]" = torch.ops.aten.mean.dim(clamp_max_34, [-1, -2], True);  clamp_max_34 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:169, code: x = torch.flatten(x, 1)
    view: "f32[4, 1280]" = torch.ops.aten.view.default(mean, [4, 1280]);  mean = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:170, code: x = self.classifier(x)
    clone_35: "f32[4, 1280]" = torch.ops.aten.clone.default(view);  view = None
    permute: "f32[1280, 1000]" = torch.ops.aten.permute.default(primals_157, [1, 0]);  primals_157 = None
    addmm: "f32[4, 1000]" = torch.ops.aten.addmm.default(primals_158, clone_35, permute);  primals_158 = None
    permute_1: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:166, code: x = self.features(x)
    le: "b8[4, 1280, 7, 7]" = torch.ops.aten.le.Scalar(clone_34, 0.0)
    ge: "b8[4, 1280, 7, 7]" = torch.ops.aten.ge.Scalar(clone_34, 6.0);  clone_34 = None
    bitwise_or: "b8[4, 1280, 7, 7]" = torch.ops.aten.bitwise_or.Tensor(le, ge);  le = ge = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:64, code: return self.conv(x)
    le_1: "b8[4, 960, 7, 7]" = torch.ops.aten.le.Scalar(clone_33, 0.0)
    ge_1: "b8[4, 960, 7, 7]" = torch.ops.aten.ge.Scalar(clone_33, 6.0);  clone_33 = None
    bitwise_or_1: "b8[4, 960, 7, 7]" = torch.ops.aten.bitwise_or.Tensor(le_1, ge_1);  le_1 = ge_1 = None
    le_2: "b8[4, 960, 7, 7]" = torch.ops.aten.le.Scalar(clone_32, 0.0)
    ge_2: "b8[4, 960, 7, 7]" = torch.ops.aten.ge.Scalar(clone_32, 6.0);  clone_32 = None
    bitwise_or_2: "b8[4, 960, 7, 7]" = torch.ops.aten.bitwise_or.Tensor(le_2, ge_2);  le_2 = ge_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    le_3: "b8[4, 960, 7, 7]" = torch.ops.aten.le.Scalar(clone_31, 0.0)
    ge_3: "b8[4, 960, 7, 7]" = torch.ops.aten.ge.Scalar(clone_31, 6.0);  clone_31 = None
    bitwise_or_3: "b8[4, 960, 7, 7]" = torch.ops.aten.bitwise_or.Tensor(le_3, ge_3);  le_3 = ge_3 = None
    le_4: "b8[4, 960, 7, 7]" = torch.ops.aten.le.Scalar(clone_30, 0.0)
    ge_4: "b8[4, 960, 7, 7]" = torch.ops.aten.ge.Scalar(clone_30, 6.0);  clone_30 = None
    bitwise_or_4: "b8[4, 960, 7, 7]" = torch.ops.aten.bitwise_or.Tensor(le_4, ge_4);  le_4 = ge_4 = None
    le_5: "b8[4, 960, 7, 7]" = torch.ops.aten.le.Scalar(clone_29, 0.0)
    ge_5: "b8[4, 960, 7, 7]" = torch.ops.aten.ge.Scalar(clone_29, 6.0);  clone_29 = None
    bitwise_or_5: "b8[4, 960, 7, 7]" = torch.ops.aten.bitwise_or.Tensor(le_5, ge_5);  le_5 = ge_5 = None
    le_6: "b8[4, 960, 7, 7]" = torch.ops.aten.le.Scalar(clone_28, 0.0)
    ge_6: "b8[4, 960, 7, 7]" = torch.ops.aten.ge.Scalar(clone_28, 6.0);  clone_28 = None
    bitwise_or_6: "b8[4, 960, 7, 7]" = torch.ops.aten.bitwise_or.Tensor(le_6, ge_6);  le_6 = ge_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:64, code: return self.conv(x)
    le_7: "b8[4, 576, 7, 7]" = torch.ops.aten.le.Scalar(clone_27, 0.0)
    ge_7: "b8[4, 576, 7, 7]" = torch.ops.aten.ge.Scalar(clone_27, 6.0);  clone_27 = None
    bitwise_or_7: "b8[4, 576, 7, 7]" = torch.ops.aten.bitwise_or.Tensor(le_7, ge_7);  le_7 = ge_7 = None
    le_8: "b8[4, 576, 14, 14]" = torch.ops.aten.le.Scalar(clone_26, 0.0)
    ge_8: "b8[4, 576, 14, 14]" = torch.ops.aten.ge.Scalar(clone_26, 6.0);  clone_26 = None
    bitwise_or_8: "b8[4, 576, 14, 14]" = torch.ops.aten.bitwise_or.Tensor(le_8, ge_8);  le_8 = ge_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    le_9: "b8[4, 576, 14, 14]" = torch.ops.aten.le.Scalar(clone_25, 0.0)
    ge_9: "b8[4, 576, 14, 14]" = torch.ops.aten.ge.Scalar(clone_25, 6.0);  clone_25 = None
    bitwise_or_9: "b8[4, 576, 14, 14]" = torch.ops.aten.bitwise_or.Tensor(le_9, ge_9);  le_9 = ge_9 = None
    le_10: "b8[4, 576, 14, 14]" = torch.ops.aten.le.Scalar(clone_24, 0.0)
    ge_10: "b8[4, 576, 14, 14]" = torch.ops.aten.ge.Scalar(clone_24, 6.0);  clone_24 = None
    bitwise_or_10: "b8[4, 576, 14, 14]" = torch.ops.aten.bitwise_or.Tensor(le_10, ge_10);  le_10 = ge_10 = None
    le_11: "b8[4, 576, 14, 14]" = torch.ops.aten.le.Scalar(clone_23, 0.0)
    ge_11: "b8[4, 576, 14, 14]" = torch.ops.aten.ge.Scalar(clone_23, 6.0);  clone_23 = None
    bitwise_or_11: "b8[4, 576, 14, 14]" = torch.ops.aten.bitwise_or.Tensor(le_11, ge_11);  le_11 = ge_11 = None
    le_12: "b8[4, 576, 14, 14]" = torch.ops.aten.le.Scalar(clone_22, 0.0)
    ge_12: "b8[4, 576, 14, 14]" = torch.ops.aten.ge.Scalar(clone_22, 6.0);  clone_22 = None
    bitwise_or_12: "b8[4, 576, 14, 14]" = torch.ops.aten.bitwise_or.Tensor(le_12, ge_12);  le_12 = ge_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:64, code: return self.conv(x)
    le_13: "b8[4, 384, 14, 14]" = torch.ops.aten.le.Scalar(clone_21, 0.0)
    ge_13: "b8[4, 384, 14, 14]" = torch.ops.aten.ge.Scalar(clone_21, 6.0);  clone_21 = None
    bitwise_or_13: "b8[4, 384, 14, 14]" = torch.ops.aten.bitwise_or.Tensor(le_13, ge_13);  le_13 = ge_13 = None
    le_14: "b8[4, 384, 14, 14]" = torch.ops.aten.le.Scalar(clone_20, 0.0)
    ge_14: "b8[4, 384, 14, 14]" = torch.ops.aten.ge.Scalar(clone_20, 6.0);  clone_20 = None
    bitwise_or_14: "b8[4, 384, 14, 14]" = torch.ops.aten.bitwise_or.Tensor(le_14, ge_14);  le_14 = ge_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    le_15: "b8[4, 384, 14, 14]" = torch.ops.aten.le.Scalar(clone_19, 0.0)
    ge_15: "b8[4, 384, 14, 14]" = torch.ops.aten.ge.Scalar(clone_19, 6.0);  clone_19 = None
    bitwise_or_15: "b8[4, 384, 14, 14]" = torch.ops.aten.bitwise_or.Tensor(le_15, ge_15);  le_15 = ge_15 = None
    le_16: "b8[4, 384, 14, 14]" = torch.ops.aten.le.Scalar(clone_18, 0.0)
    ge_16: "b8[4, 384, 14, 14]" = torch.ops.aten.ge.Scalar(clone_18, 6.0);  clone_18 = None
    bitwise_or_16: "b8[4, 384, 14, 14]" = torch.ops.aten.bitwise_or.Tensor(le_16, ge_16);  le_16 = ge_16 = None
    le_17: "b8[4, 384, 14, 14]" = torch.ops.aten.le.Scalar(clone_17, 0.0)
    ge_17: "b8[4, 384, 14, 14]" = torch.ops.aten.ge.Scalar(clone_17, 6.0);  clone_17 = None
    bitwise_or_17: "b8[4, 384, 14, 14]" = torch.ops.aten.bitwise_or.Tensor(le_17, ge_17);  le_17 = ge_17 = None
    le_18: "b8[4, 384, 14, 14]" = torch.ops.aten.le.Scalar(clone_16, 0.0)
    ge_18: "b8[4, 384, 14, 14]" = torch.ops.aten.ge.Scalar(clone_16, 6.0);  clone_16 = None
    bitwise_or_18: "b8[4, 384, 14, 14]" = torch.ops.aten.bitwise_or.Tensor(le_18, ge_18);  le_18 = ge_18 = None
    le_19: "b8[4, 384, 14, 14]" = torch.ops.aten.le.Scalar(clone_15, 0.0)
    ge_19: "b8[4, 384, 14, 14]" = torch.ops.aten.ge.Scalar(clone_15, 6.0);  clone_15 = None
    bitwise_or_19: "b8[4, 384, 14, 14]" = torch.ops.aten.bitwise_or.Tensor(le_19, ge_19);  le_19 = ge_19 = None
    le_20: "b8[4, 384, 14, 14]" = torch.ops.aten.le.Scalar(clone_14, 0.0)
    ge_20: "b8[4, 384, 14, 14]" = torch.ops.aten.ge.Scalar(clone_14, 6.0);  clone_14 = None
    bitwise_or_20: "b8[4, 384, 14, 14]" = torch.ops.aten.bitwise_or.Tensor(le_20, ge_20);  le_20 = ge_20 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:64, code: return self.conv(x)
    le_21: "b8[4, 192, 14, 14]" = torch.ops.aten.le.Scalar(clone_13, 0.0)
    ge_21: "b8[4, 192, 14, 14]" = torch.ops.aten.ge.Scalar(clone_13, 6.0);  clone_13 = None
    bitwise_or_21: "b8[4, 192, 14, 14]" = torch.ops.aten.bitwise_or.Tensor(le_21, ge_21);  le_21 = ge_21 = None
    le_22: "b8[4, 192, 28, 28]" = torch.ops.aten.le.Scalar(clone_12, 0.0)
    ge_22: "b8[4, 192, 28, 28]" = torch.ops.aten.ge.Scalar(clone_12, 6.0);  clone_12 = None
    bitwise_or_22: "b8[4, 192, 28, 28]" = torch.ops.aten.bitwise_or.Tensor(le_22, ge_22);  le_22 = ge_22 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    le_23: "b8[4, 192, 28, 28]" = torch.ops.aten.le.Scalar(clone_11, 0.0)
    ge_23: "b8[4, 192, 28, 28]" = torch.ops.aten.ge.Scalar(clone_11, 6.0);  clone_11 = None
    bitwise_or_23: "b8[4, 192, 28, 28]" = torch.ops.aten.bitwise_or.Tensor(le_23, ge_23);  le_23 = ge_23 = None
    le_24: "b8[4, 192, 28, 28]" = torch.ops.aten.le.Scalar(clone_10, 0.0)
    ge_24: "b8[4, 192, 28, 28]" = torch.ops.aten.ge.Scalar(clone_10, 6.0);  clone_10 = None
    bitwise_or_24: "b8[4, 192, 28, 28]" = torch.ops.aten.bitwise_or.Tensor(le_24, ge_24);  le_24 = ge_24 = None
    le_25: "b8[4, 192, 28, 28]" = torch.ops.aten.le.Scalar(clone_9, 0.0)
    ge_25: "b8[4, 192, 28, 28]" = torch.ops.aten.ge.Scalar(clone_9, 6.0);  clone_9 = None
    bitwise_or_25: "b8[4, 192, 28, 28]" = torch.ops.aten.bitwise_or.Tensor(le_25, ge_25);  le_25 = ge_25 = None
    le_26: "b8[4, 192, 28, 28]" = torch.ops.aten.le.Scalar(clone_8, 0.0)
    ge_26: "b8[4, 192, 28, 28]" = torch.ops.aten.ge.Scalar(clone_8, 6.0);  clone_8 = None
    bitwise_or_26: "b8[4, 192, 28, 28]" = torch.ops.aten.bitwise_or.Tensor(le_26, ge_26);  le_26 = ge_26 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:64, code: return self.conv(x)
    le_27: "b8[4, 144, 28, 28]" = torch.ops.aten.le.Scalar(clone_7, 0.0)
    ge_27: "b8[4, 144, 28, 28]" = torch.ops.aten.ge.Scalar(clone_7, 6.0);  clone_7 = None
    bitwise_or_27: "b8[4, 144, 28, 28]" = torch.ops.aten.bitwise_or.Tensor(le_27, ge_27);  le_27 = ge_27 = None
    le_28: "b8[4, 144, 56, 56]" = torch.ops.aten.le.Scalar(clone_6, 0.0)
    ge_28: "b8[4, 144, 56, 56]" = torch.ops.aten.ge.Scalar(clone_6, 6.0);  clone_6 = None
    bitwise_or_28: "b8[4, 144, 56, 56]" = torch.ops.aten.bitwise_or.Tensor(le_28, ge_28);  le_28 = ge_28 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    le_29: "b8[4, 144, 56, 56]" = torch.ops.aten.le.Scalar(clone_5, 0.0)
    ge_29: "b8[4, 144, 56, 56]" = torch.ops.aten.ge.Scalar(clone_5, 6.0);  clone_5 = None
    bitwise_or_29: "b8[4, 144, 56, 56]" = torch.ops.aten.bitwise_or.Tensor(le_29, ge_29);  le_29 = ge_29 = None
    le_30: "b8[4, 144, 56, 56]" = torch.ops.aten.le.Scalar(clone_4, 0.0)
    ge_30: "b8[4, 144, 56, 56]" = torch.ops.aten.ge.Scalar(clone_4, 6.0);  clone_4 = None
    bitwise_or_30: "b8[4, 144, 56, 56]" = torch.ops.aten.bitwise_or.Tensor(le_30, ge_30);  le_30 = ge_30 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:64, code: return self.conv(x)
    le_31: "b8[4, 96, 56, 56]" = torch.ops.aten.le.Scalar(clone_3, 0.0)
    ge_31: "b8[4, 96, 56, 56]" = torch.ops.aten.ge.Scalar(clone_3, 6.0);  clone_3 = None
    bitwise_or_31: "b8[4, 96, 56, 56]" = torch.ops.aten.bitwise_or.Tensor(le_31, ge_31);  le_31 = ge_31 = None
    le_32: "b8[4, 96, 112, 112]" = torch.ops.aten.le.Scalar(clone_2, 0.0)
    ge_32: "b8[4, 96, 112, 112]" = torch.ops.aten.ge.Scalar(clone_2, 6.0);  clone_2 = None
    bitwise_or_32: "b8[4, 96, 112, 112]" = torch.ops.aten.bitwise_or.Tensor(le_32, ge_32);  le_32 = ge_32 = None
    le_33: "b8[4, 32, 112, 112]" = torch.ops.aten.le.Scalar(clone_1, 0.0)
    ge_33: "b8[4, 32, 112, 112]" = torch.ops.aten.ge.Scalar(clone_1, 6.0);  clone_1 = None
    bitwise_or_33: "b8[4, 32, 112, 112]" = torch.ops.aten.bitwise_or.Tensor(le_33, ge_33);  le_33 = ge_33 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:166, code: x = self.features(x)
    le_34: "b8[4, 32, 112, 112]" = torch.ops.aten.le.Scalar(clone, 0.0)
    ge_34: "b8[4, 32, 112, 112]" = torch.ops.aten.ge.Scalar(clone, 6.0);  clone = None
    bitwise_or_34: "b8[4, 32, 112, 112]" = torch.ops.aten.bitwise_or.Tensor(le_34, ge_34);  le_34 = ge_34 = None
    return [addmm, primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_159, primals_160, primals_162, primals_163, primals_165, primals_166, primals_168, primals_169, primals_171, primals_172, primals_174, primals_175, primals_177, primals_178, primals_180, primals_181, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_198, primals_199, primals_201, primals_202, primals_204, primals_205, primals_207, primals_208, primals_210, primals_211, primals_213, primals_214, primals_216, primals_217, primals_219, primals_220, primals_222, primals_223, primals_225, primals_226, primals_228, primals_229, primals_231, primals_232, primals_234, primals_235, primals_237, primals_238, primals_240, primals_241, primals_243, primals_244, primals_246, primals_247, primals_249, primals_250, primals_252, primals_253, primals_255, primals_256, primals_258, primals_259, primals_261, primals_262, primals_264, primals_265, primals_267, primals_268, primals_270, primals_271, primals_273, primals_274, primals_276, primals_277, primals_279, primals_280, primals_282, primals_283, primals_285, primals_286, primals_288, primals_289, primals_291, primals_292, primals_294, primals_295, primals_297, primals_298, primals_300, primals_301, primals_303, primals_304, primals_306, primals_307, primals_309, primals_310, primals_312, primals_313, primals_315, convolution, clamp_max, convolution_1, clamp_max_1, convolution_2, add_5, convolution_3, clamp_max_2, convolution_4, clamp_max_3, convolution_5, add_11, convolution_6, clamp_max_4, convolution_7, clamp_max_5, convolution_8, add_18, convolution_9, clamp_max_6, convolution_10, clamp_max_7, convolution_11, add_24, convolution_12, clamp_max_8, convolution_13, clamp_max_9, convolution_14, add_31, convolution_15, clamp_max_10, convolution_16, clamp_max_11, convolution_17, add_38, convolution_18, clamp_max_12, convolution_19, clamp_max_13, convolution_20, add_44, convolution_21, clamp_max_14, convolution_22, clamp_max_15, convolution_23, add_51, convolution_24, clamp_max_16, convolution_25, clamp_max_17, convolution_26, add_58, convolution_27, clamp_max_18, convolution_28, clamp_max_19, convolution_29, add_65, convolution_30, clamp_max_20, convolution_31, clamp_max_21, convolution_32, add_71, convolution_33, clamp_max_22, convolution_34, clamp_max_23, convolution_35, add_78, convolution_36, clamp_max_24, convolution_37, clamp_max_25, convolution_38, add_85, convolution_39, clamp_max_26, convolution_40, clamp_max_27, convolution_41, add_91, convolution_42, clamp_max_28, convolution_43, clamp_max_29, convolution_44, add_98, convolution_45, clamp_max_30, convolution_46, clamp_max_31, convolution_47, add_105, convolution_48, clamp_max_32, convolution_49, clamp_max_33, convolution_50, add_111, convolution_51, clone_35, permute_1, bitwise_or, bitwise_or_1, bitwise_or_2, bitwise_or_3, bitwise_or_4, bitwise_or_5, bitwise_or_6, bitwise_or_7, bitwise_or_8, bitwise_or_9, bitwise_or_10, bitwise_or_11, bitwise_or_12, bitwise_or_13, bitwise_or_14, bitwise_or_15, bitwise_or_16, bitwise_or_17, bitwise_or_18, bitwise_or_19, bitwise_or_20, bitwise_or_21, bitwise_or_22, bitwise_or_23, bitwise_or_24, bitwise_or_25, bitwise_or_26, bitwise_or_27, bitwise_or_28, bitwise_or_29, bitwise_or_30, bitwise_or_31, bitwise_or_32, bitwise_or_33, bitwise_or_34]
    