from __future__ import annotations



def forward(self, primals_1: "f32[16, 3, 3, 3]", primals_2: "f32[16]", primals_3: "f32[16]", primals_4: "f32[16, 1, 3, 3]", primals_5: "f32[16]", primals_6: "f32[16]", primals_7: "f32[16, 16, 1, 1]", primals_8: "f32[16]", primals_9: "f32[16]", primals_10: "f32[64, 16, 1, 1]", primals_11: "f32[64]", primals_12: "f32[64]", primals_13: "f32[64, 1, 3, 3]", primals_14: "f32[64]", primals_15: "f32[64]", primals_16: "f32[24, 64, 1, 1]", primals_17: "f32[24]", primals_18: "f32[24]", primals_19: "f32[72, 24, 1, 1]", primals_20: "f32[72]", primals_21: "f32[72]", primals_22: "f32[72, 1, 3, 3]", primals_23: "f32[72]", primals_24: "f32[72]", primals_25: "f32[24, 72, 1, 1]", primals_26: "f32[24]", primals_27: "f32[24]", primals_28: "f32[72, 24, 1, 1]", primals_29: "f32[72]", primals_30: "f32[72]", primals_31: "f32[72, 1, 5, 5]", primals_32: "f32[72]", primals_33: "f32[72]", primals_34: "f32[24, 72, 1, 1]", primals_35: "f32[24]", primals_36: "f32[72, 24, 1, 1]", primals_37: "f32[72]", primals_38: "f32[40, 72, 1, 1]", primals_39: "f32[40]", primals_40: "f32[40]", primals_41: "f32[120, 40, 1, 1]", primals_42: "f32[120]", primals_43: "f32[120]", primals_44: "f32[120, 1, 5, 5]", primals_45: "f32[120]", primals_46: "f32[120]", primals_47: "f32[32, 120, 1, 1]", primals_48: "f32[32]", primals_49: "f32[120, 32, 1, 1]", primals_50: "f32[120]", primals_51: "f32[40, 120, 1, 1]", primals_52: "f32[40]", primals_53: "f32[40]", primals_54: "f32[120, 40, 1, 1]", primals_55: "f32[120]", primals_56: "f32[120]", primals_57: "f32[120, 1, 5, 5]", primals_58: "f32[120]", primals_59: "f32[120]", primals_60: "f32[32, 120, 1, 1]", primals_61: "f32[32]", primals_62: "f32[120, 32, 1, 1]", primals_63: "f32[120]", primals_64: "f32[40, 120, 1, 1]", primals_65: "f32[40]", primals_66: "f32[40]", primals_67: "f32[240, 40, 1, 1]", primals_68: "f32[240]", primals_69: "f32[240]", primals_70: "f32[240, 1, 3, 3]", primals_71: "f32[240]", primals_72: "f32[240]", primals_73: "f32[80, 240, 1, 1]", primals_74: "f32[80]", primals_75: "f32[80]", primals_76: "f32[200, 80, 1, 1]", primals_77: "f32[200]", primals_78: "f32[200]", primals_79: "f32[200, 1, 3, 3]", primals_80: "f32[200]", primals_81: "f32[200]", primals_82: "f32[80, 200, 1, 1]", primals_83: "f32[80]", primals_84: "f32[80]", primals_85: "f32[184, 80, 1, 1]", primals_86: "f32[184]", primals_87: "f32[184]", primals_88: "f32[184, 1, 3, 3]", primals_89: "f32[184]", primals_90: "f32[184]", primals_91: "f32[80, 184, 1, 1]", primals_92: "f32[80]", primals_93: "f32[80]", primals_94: "f32[184, 80, 1, 1]", primals_95: "f32[184]", primals_96: "f32[184]", primals_97: "f32[184, 1, 3, 3]", primals_98: "f32[184]", primals_99: "f32[184]", primals_100: "f32[80, 184, 1, 1]", primals_101: "f32[80]", primals_102: "f32[80]", primals_103: "f32[480, 80, 1, 1]", primals_104: "f32[480]", primals_105: "f32[480]", primals_106: "f32[480, 1, 3, 3]", primals_107: "f32[480]", primals_108: "f32[480]", primals_109: "f32[120, 480, 1, 1]", primals_110: "f32[120]", primals_111: "f32[480, 120, 1, 1]", primals_112: "f32[480]", primals_113: "f32[112, 480, 1, 1]", primals_114: "f32[112]", primals_115: "f32[112]", primals_116: "f32[672, 112, 1, 1]", primals_117: "f32[672]", primals_118: "f32[672]", primals_119: "f32[672, 1, 3, 3]", primals_120: "f32[672]", primals_121: "f32[672]", primals_122: "f32[168, 672, 1, 1]", primals_123: "f32[168]", primals_124: "f32[672, 168, 1, 1]", primals_125: "f32[672]", primals_126: "f32[112, 672, 1, 1]", primals_127: "f32[112]", primals_128: "f32[112]", primals_129: "f32[672, 112, 1, 1]", primals_130: "f32[672]", primals_131: "f32[672]", primals_132: "f32[672, 1, 5, 5]", primals_133: "f32[672]", primals_134: "f32[672]", primals_135: "f32[168, 672, 1, 1]", primals_136: "f32[168]", primals_137: "f32[672, 168, 1, 1]", primals_138: "f32[672]", primals_139: "f32[160, 672, 1, 1]", primals_140: "f32[160]", primals_141: "f32[160]", primals_142: "f32[960, 160, 1, 1]", primals_143: "f32[960]", primals_144: "f32[960]", primals_145: "f32[960, 1, 5, 5]", primals_146: "f32[960]", primals_147: "f32[960]", primals_148: "f32[240, 960, 1, 1]", primals_149: "f32[240]", primals_150: "f32[960, 240, 1, 1]", primals_151: "f32[960]", primals_152: "f32[160, 960, 1, 1]", primals_153: "f32[160]", primals_154: "f32[160]", primals_155: "f32[960, 160, 1, 1]", primals_156: "f32[960]", primals_157: "f32[960]", primals_158: "f32[960, 1, 5, 5]", primals_159: "f32[960]", primals_160: "f32[960]", primals_161: "f32[240, 960, 1, 1]", primals_162: "f32[240]", primals_163: "f32[960, 240, 1, 1]", primals_164: "f32[960]", primals_165: "f32[160, 960, 1, 1]", primals_166: "f32[160]", primals_167: "f32[160]", primals_168: "f32[960, 160, 1, 1]", primals_169: "f32[960]", primals_170: "f32[960]", primals_171: "f32[1280, 960]", primals_172: "f32[1280]", primals_173: "f32[1000, 1280]", primals_174: "f32[1000]", primals_175: "f32[16]", primals_176: "f32[16]", primals_177: "i64[]", primals_178: "f32[16]", primals_179: "f32[16]", primals_180: "i64[]", primals_181: "f32[16]", primals_182: "f32[16]", primals_183: "i64[]", primals_184: "f32[64]", primals_185: "f32[64]", primals_186: "i64[]", primals_187: "f32[64]", primals_188: "f32[64]", primals_189: "i64[]", primals_190: "f32[24]", primals_191: "f32[24]", primals_192: "i64[]", primals_193: "f32[72]", primals_194: "f32[72]", primals_195: "i64[]", primals_196: "f32[72]", primals_197: "f32[72]", primals_198: "i64[]", primals_199: "f32[24]", primals_200: "f32[24]", primals_201: "i64[]", primals_202: "f32[72]", primals_203: "f32[72]", primals_204: "i64[]", primals_205: "f32[72]", primals_206: "f32[72]", primals_207: "i64[]", primals_208: "f32[40]", primals_209: "f32[40]", primals_210: "i64[]", primals_211: "f32[120]", primals_212: "f32[120]", primals_213: "i64[]", primals_214: "f32[120]", primals_215: "f32[120]", primals_216: "i64[]", primals_217: "f32[40]", primals_218: "f32[40]", primals_219: "i64[]", primals_220: "f32[120]", primals_221: "f32[120]", primals_222: "i64[]", primals_223: "f32[120]", primals_224: "f32[120]", primals_225: "i64[]", primals_226: "f32[40]", primals_227: "f32[40]", primals_228: "i64[]", primals_229: "f32[240]", primals_230: "f32[240]", primals_231: "i64[]", primals_232: "f32[240]", primals_233: "f32[240]", primals_234: "i64[]", primals_235: "f32[80]", primals_236: "f32[80]", primals_237: "i64[]", primals_238: "f32[200]", primals_239: "f32[200]", primals_240: "i64[]", primals_241: "f32[200]", primals_242: "f32[200]", primals_243: "i64[]", primals_244: "f32[80]", primals_245: "f32[80]", primals_246: "i64[]", primals_247: "f32[184]", primals_248: "f32[184]", primals_249: "i64[]", primals_250: "f32[184]", primals_251: "f32[184]", primals_252: "i64[]", primals_253: "f32[80]", primals_254: "f32[80]", primals_255: "i64[]", primals_256: "f32[184]", primals_257: "f32[184]", primals_258: "i64[]", primals_259: "f32[184]", primals_260: "f32[184]", primals_261: "i64[]", primals_262: "f32[80]", primals_263: "f32[80]", primals_264: "i64[]", primals_265: "f32[480]", primals_266: "f32[480]", primals_267: "i64[]", primals_268: "f32[480]", primals_269: "f32[480]", primals_270: "i64[]", primals_271: "f32[112]", primals_272: "f32[112]", primals_273: "i64[]", primals_274: "f32[672]", primals_275: "f32[672]", primals_276: "i64[]", primals_277: "f32[672]", primals_278: "f32[672]", primals_279: "i64[]", primals_280: "f32[112]", primals_281: "f32[112]", primals_282: "i64[]", primals_283: "f32[672]", primals_284: "f32[672]", primals_285: "i64[]", primals_286: "f32[672]", primals_287: "f32[672]", primals_288: "i64[]", primals_289: "f32[160]", primals_290: "f32[160]", primals_291: "i64[]", primals_292: "f32[960]", primals_293: "f32[960]", primals_294: "i64[]", primals_295: "f32[960]", primals_296: "f32[960]", primals_297: "i64[]", primals_298: "f32[160]", primals_299: "f32[160]", primals_300: "i64[]", primals_301: "f32[960]", primals_302: "f32[960]", primals_303: "i64[]", primals_304: "f32[960]", primals_305: "f32[960]", primals_306: "i64[]", primals_307: "f32[160]", primals_308: "f32[160]", primals_309: "i64[]", primals_310: "f32[960]", primals_311: "f32[960]", primals_312: "i64[]", primals_313: "f32[4, 3, 224, 224]"):
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:210, code: x = self.features(x)
    convolution: "f32[4, 16, 112, 112]" = torch.ops.aten.convolution.default(primals_313, primals_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    add: "f32[16]" = torch.ops.aten.add.Tensor(primals_176, 0.001)
    sqrt: "f32[16]" = torch.ops.aten.sqrt.default(add);  add = None
    reciprocal: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
    mul: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
    unsqueeze: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_175, -1)
    unsqueeze_1: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    unsqueeze_2: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
    unsqueeze_3: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    sub: "f32[4, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  unsqueeze_1 = None
    mul_1: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
    unsqueeze_4: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1)
    unsqueeze_5: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_2: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
    unsqueeze_6: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1);  primals_3 = None
    unsqueeze_7: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_1: "f32[4, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
    add_2: "f32[4, 16, 112, 112]" = torch.ops.aten.add.Tensor(add_1, 3)
    clamp_min: "f32[4, 16, 112, 112]" = torch.ops.aten.clamp_min.default(add_2, 0);  add_2 = None
    clamp_max: "f32[4, 16, 112, 112]" = torch.ops.aten.clamp_max.default(clamp_min, 6);  clamp_min = None
    mul_3: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(add_1, clamp_max);  clamp_max = None
    div: "f32[4, 16, 112, 112]" = torch.ops.aten.div.Tensor(mul_3, 6);  mul_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_1: "f32[4, 16, 112, 112]" = torch.ops.aten.convolution.default(div, primals_4, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 16)
    add_3: "f32[16]" = torch.ops.aten.add.Tensor(primals_179, 0.001)
    sqrt_1: "f32[16]" = torch.ops.aten.sqrt.default(add_3);  add_3 = None
    reciprocal_1: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_4: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_8: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_178, -1)
    unsqueeze_9: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    unsqueeze_10: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_4, -1);  mul_4 = None
    unsqueeze_11: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    sub_1: "f32[4, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  unsqueeze_9 = None
    mul_5: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_13: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_6: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_5, unsqueeze_13);  mul_5 = unsqueeze_13 = None
    unsqueeze_14: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_15: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_4: "f32[4, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_15);  mul_6 = unsqueeze_15 = None
    relu: "f32[4, 16, 112, 112]" = torch.ops.aten.relu.default(add_4);  add_4 = None
    convolution_2: "f32[4, 16, 112, 112]" = torch.ops.aten.convolution.default(relu, primals_7, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_5: "f32[16]" = torch.ops.aten.add.Tensor(primals_182, 0.001)
    sqrt_2: "f32[16]" = torch.ops.aten.sqrt.default(add_5);  add_5 = None
    reciprocal_2: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_7: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_16: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_181, -1)
    unsqueeze_17: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    unsqueeze_18: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_7, -1);  mul_7 = None
    unsqueeze_19: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    sub_2: "f32[4, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  unsqueeze_17 = None
    mul_8: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1)
    unsqueeze_21: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_9: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_8, unsqueeze_21);  mul_8 = unsqueeze_21 = None
    unsqueeze_22: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1);  primals_9 = None
    unsqueeze_23: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_6: "f32[4, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_9, unsqueeze_23);  mul_9 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:113, code: result += input
    add_7: "f32[4, 16, 112, 112]" = torch.ops.aten.add.Tensor(add_6, div);  add_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_3: "f32[4, 64, 112, 112]" = torch.ops.aten.convolution.default(add_7, primals_10, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_8: "f32[64]" = torch.ops.aten.add.Tensor(primals_185, 0.001)
    sqrt_3: "f32[64]" = torch.ops.aten.sqrt.default(add_8);  add_8 = None
    reciprocal_3: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
    mul_10: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
    unsqueeze_24: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_184, -1)
    unsqueeze_25: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    unsqueeze_26: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_10, -1);  mul_10 = None
    unsqueeze_27: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    sub_3: "f32[4, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  unsqueeze_25 = None
    mul_11: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
    unsqueeze_28: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_29: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_12: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_11, unsqueeze_29);  mul_11 = unsqueeze_29 = None
    unsqueeze_30: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_31: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_9: "f32[4, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_12, unsqueeze_31);  mul_12 = unsqueeze_31 = None
    relu_1: "f32[4, 64, 112, 112]" = torch.ops.aten.relu.default(add_9);  add_9 = None
    convolution_4: "f32[4, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_1, primals_13, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 64)
    add_10: "f32[64]" = torch.ops.aten.add.Tensor(primals_188, 0.001)
    sqrt_4: "f32[64]" = torch.ops.aten.sqrt.default(add_10);  add_10 = None
    reciprocal_4: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
    mul_13: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
    unsqueeze_32: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_187, -1)
    unsqueeze_33: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    unsqueeze_34: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_13, -1);  mul_13 = None
    unsqueeze_35: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    sub_4: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  unsqueeze_33 = None
    mul_14: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
    unsqueeze_36: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1)
    unsqueeze_37: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_15: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_37);  mul_14 = unsqueeze_37 = None
    unsqueeze_38: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1);  primals_15 = None
    unsqueeze_39: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_11: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_15, unsqueeze_39);  mul_15 = unsqueeze_39 = None
    relu_2: "f32[4, 64, 56, 56]" = torch.ops.aten.relu.default(add_11);  add_11 = None
    convolution_5: "f32[4, 24, 56, 56]" = torch.ops.aten.convolution.default(relu_2, primals_16, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_12: "f32[24]" = torch.ops.aten.add.Tensor(primals_191, 0.001)
    sqrt_5: "f32[24]" = torch.ops.aten.sqrt.default(add_12);  add_12 = None
    reciprocal_5: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
    mul_16: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
    unsqueeze_40: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_190, -1)
    unsqueeze_41: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    unsqueeze_42: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_16, -1);  mul_16 = None
    unsqueeze_43: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    sub_5: "f32[4, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_41);  unsqueeze_41 = None
    mul_17: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
    unsqueeze_44: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_45: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_18: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_17, unsqueeze_45);  mul_17 = unsqueeze_45 = None
    unsqueeze_46: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_47: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_13: "f32[4, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_18, unsqueeze_47);  mul_18 = unsqueeze_47 = None
    convolution_6: "f32[4, 72, 56, 56]" = torch.ops.aten.convolution.default(add_13, primals_19, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_14: "f32[72]" = torch.ops.aten.add.Tensor(primals_194, 0.001)
    sqrt_6: "f32[72]" = torch.ops.aten.sqrt.default(add_14);  add_14 = None
    reciprocal_6: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
    mul_19: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
    unsqueeze_48: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_193, -1)
    unsqueeze_49: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    unsqueeze_50: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_19, -1);  mul_19 = None
    unsqueeze_51: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    sub_6: "f32[4, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_49);  unsqueeze_49 = None
    mul_20: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
    unsqueeze_52: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1)
    unsqueeze_53: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_21: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(mul_20, unsqueeze_53);  mul_20 = unsqueeze_53 = None
    unsqueeze_54: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1);  primals_21 = None
    unsqueeze_55: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_15: "f32[4, 72, 56, 56]" = torch.ops.aten.add.Tensor(mul_21, unsqueeze_55);  mul_21 = unsqueeze_55 = None
    relu_3: "f32[4, 72, 56, 56]" = torch.ops.aten.relu.default(add_15);  add_15 = None
    convolution_7: "f32[4, 72, 56, 56]" = torch.ops.aten.convolution.default(relu_3, primals_22, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 72)
    add_16: "f32[72]" = torch.ops.aten.add.Tensor(primals_197, 0.001)
    sqrt_7: "f32[72]" = torch.ops.aten.sqrt.default(add_16);  add_16 = None
    reciprocal_7: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
    mul_22: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
    unsqueeze_56: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_196, -1)
    unsqueeze_57: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    unsqueeze_58: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_22, -1);  mul_22 = None
    unsqueeze_59: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    sub_7: "f32[4, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_57);  unsqueeze_57 = None
    mul_23: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
    unsqueeze_60: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_61: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_24: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(mul_23, unsqueeze_61);  mul_23 = unsqueeze_61 = None
    unsqueeze_62: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_63: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_17: "f32[4, 72, 56, 56]" = torch.ops.aten.add.Tensor(mul_24, unsqueeze_63);  mul_24 = unsqueeze_63 = None
    relu_4: "f32[4, 72, 56, 56]" = torch.ops.aten.relu.default(add_17);  add_17 = None
    convolution_8: "f32[4, 24, 56, 56]" = torch.ops.aten.convolution.default(relu_4, primals_25, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_18: "f32[24]" = torch.ops.aten.add.Tensor(primals_200, 0.001)
    sqrt_8: "f32[24]" = torch.ops.aten.sqrt.default(add_18);  add_18 = None
    reciprocal_8: "f32[24]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
    mul_25: "f32[24]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
    unsqueeze_64: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_199, -1)
    unsqueeze_65: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    unsqueeze_66: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(mul_25, -1);  mul_25 = None
    unsqueeze_67: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    sub_8: "f32[4, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_65);  unsqueeze_65 = None
    mul_26: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
    unsqueeze_68: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1)
    unsqueeze_69: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_27: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_26, unsqueeze_69);  mul_26 = unsqueeze_69 = None
    unsqueeze_70: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1);  primals_27 = None
    unsqueeze_71: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_19: "f32[4, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_71);  mul_27 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:113, code: result += input
    add_20: "f32[4, 24, 56, 56]" = torch.ops.aten.add.Tensor(add_19, add_13);  add_19 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_9: "f32[4, 72, 56, 56]" = torch.ops.aten.convolution.default(add_20, primals_28, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_21: "f32[72]" = torch.ops.aten.add.Tensor(primals_203, 0.001)
    sqrt_9: "f32[72]" = torch.ops.aten.sqrt.default(add_21);  add_21 = None
    reciprocal_9: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
    mul_28: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
    unsqueeze_72: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_202, -1)
    unsqueeze_73: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    unsqueeze_74: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_28, -1);  mul_28 = None
    unsqueeze_75: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    sub_9: "f32[4, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_73);  unsqueeze_73 = None
    mul_29: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
    unsqueeze_76: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_77: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_30: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(mul_29, unsqueeze_77);  mul_29 = unsqueeze_77 = None
    unsqueeze_78: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_79: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_22: "f32[4, 72, 56, 56]" = torch.ops.aten.add.Tensor(mul_30, unsqueeze_79);  mul_30 = unsqueeze_79 = None
    relu_5: "f32[4, 72, 56, 56]" = torch.ops.aten.relu.default(add_22);  add_22 = None
    convolution_10: "f32[4, 72, 28, 28]" = torch.ops.aten.convolution.default(relu_5, primals_31, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 72)
    add_23: "f32[72]" = torch.ops.aten.add.Tensor(primals_206, 0.001)
    sqrt_10: "f32[72]" = torch.ops.aten.sqrt.default(add_23);  add_23 = None
    reciprocal_10: "f32[72]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
    mul_31: "f32[72]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
    unsqueeze_80: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_205, -1)
    unsqueeze_81: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    unsqueeze_82: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(mul_31, -1);  mul_31 = None
    unsqueeze_83: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    sub_10: "f32[4, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_81);  unsqueeze_81 = None
    mul_32: "f32[4, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
    unsqueeze_84: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1)
    unsqueeze_85: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_33: "f32[4, 72, 28, 28]" = torch.ops.aten.mul.Tensor(mul_32, unsqueeze_85);  mul_32 = unsqueeze_85 = None
    unsqueeze_86: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1);  primals_33 = None
    unsqueeze_87: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_24: "f32[4, 72, 28, 28]" = torch.ops.aten.add.Tensor(mul_33, unsqueeze_87);  mul_33 = unsqueeze_87 = None
    relu_6: "f32[4, 72, 28, 28]" = torch.ops.aten.relu.default(add_24);  add_24 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    mean: "f32[4, 72, 1, 1]" = torch.ops.aten.mean.dim(relu_6, [-1, -2], True)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    convolution_11: "f32[4, 24, 1, 1]" = torch.ops.aten.convolution.default(mean, primals_34, primals_35, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_35 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    relu_7: "f32[4, 24, 1, 1]" = torch.ops.aten.relu.default(convolution_11);  convolution_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    convolution_12: "f32[4, 72, 1, 1]" = torch.ops.aten.convolution.default(relu_7, primals_36, primals_37, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_37 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    add_25: "f32[4, 72, 1, 1]" = torch.ops.aten.add.Tensor(convolution_12, 3)
    clamp_min_1: "f32[4, 72, 1, 1]" = torch.ops.aten.clamp_min.default(add_25, 0);  add_25 = None
    clamp_max_1: "f32[4, 72, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_1, 6);  clamp_min_1 = None
    div_1: "f32[4, 72, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_1, 6);  clamp_max_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_34: "f32[4, 72, 28, 28]" = torch.ops.aten.mul.Tensor(div_1, relu_6)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_13: "f32[4, 40, 28, 28]" = torch.ops.aten.convolution.default(mul_34, primals_38, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_26: "f32[40]" = torch.ops.aten.add.Tensor(primals_209, 0.001)
    sqrt_11: "f32[40]" = torch.ops.aten.sqrt.default(add_26);  add_26 = None
    reciprocal_11: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
    mul_35: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
    unsqueeze_88: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_208, -1)
    unsqueeze_89: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    unsqueeze_90: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_35, -1);  mul_35 = None
    unsqueeze_91: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    sub_11: "f32[4, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_89);  unsqueeze_89 = None
    mul_36: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
    unsqueeze_92: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1)
    unsqueeze_93: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_37: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_36, unsqueeze_93);  mul_36 = unsqueeze_93 = None
    unsqueeze_94: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1);  primals_40 = None
    unsqueeze_95: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_27: "f32[4, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_37, unsqueeze_95);  mul_37 = unsqueeze_95 = None
    convolution_14: "f32[4, 120, 28, 28]" = torch.ops.aten.convolution.default(add_27, primals_41, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_28: "f32[120]" = torch.ops.aten.add.Tensor(primals_212, 0.001)
    sqrt_12: "f32[120]" = torch.ops.aten.sqrt.default(add_28);  add_28 = None
    reciprocal_12: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
    mul_38: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
    unsqueeze_96: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_211, -1)
    unsqueeze_97: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    unsqueeze_98: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_38, -1);  mul_38 = None
    unsqueeze_99: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    sub_12: "f32[4, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_97);  unsqueeze_97 = None
    mul_39: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
    unsqueeze_100: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1)
    unsqueeze_101: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_40: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_39, unsqueeze_101);  mul_39 = unsqueeze_101 = None
    unsqueeze_102: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1);  primals_43 = None
    unsqueeze_103: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_29: "f32[4, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_40, unsqueeze_103);  mul_40 = unsqueeze_103 = None
    relu_8: "f32[4, 120, 28, 28]" = torch.ops.aten.relu.default(add_29);  add_29 = None
    convolution_15: "f32[4, 120, 28, 28]" = torch.ops.aten.convolution.default(relu_8, primals_44, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120)
    add_30: "f32[120]" = torch.ops.aten.add.Tensor(primals_215, 0.001)
    sqrt_13: "f32[120]" = torch.ops.aten.sqrt.default(add_30);  add_30 = None
    reciprocal_13: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
    mul_41: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
    unsqueeze_104: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_214, -1)
    unsqueeze_105: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    unsqueeze_106: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_41, -1);  mul_41 = None
    unsqueeze_107: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    sub_13: "f32[4, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_105);  unsqueeze_105 = None
    mul_42: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
    unsqueeze_108: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1)
    unsqueeze_109: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_43: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_109);  mul_42 = unsqueeze_109 = None
    unsqueeze_110: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1);  primals_46 = None
    unsqueeze_111: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_31: "f32[4, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_43, unsqueeze_111);  mul_43 = unsqueeze_111 = None
    relu_9: "f32[4, 120, 28, 28]" = torch.ops.aten.relu.default(add_31);  add_31 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    mean_1: "f32[4, 120, 1, 1]" = torch.ops.aten.mean.dim(relu_9, [-1, -2], True)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    convolution_16: "f32[4, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_1, primals_47, primals_48, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_48 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    relu_10: "f32[4, 32, 1, 1]" = torch.ops.aten.relu.default(convolution_16);  convolution_16 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    convolution_17: "f32[4, 120, 1, 1]" = torch.ops.aten.convolution.default(relu_10, primals_49, primals_50, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_50 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    add_32: "f32[4, 120, 1, 1]" = torch.ops.aten.add.Tensor(convolution_17, 3)
    clamp_min_2: "f32[4, 120, 1, 1]" = torch.ops.aten.clamp_min.default(add_32, 0);  add_32 = None
    clamp_max_2: "f32[4, 120, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_2, 6);  clamp_min_2 = None
    div_2: "f32[4, 120, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_2, 6);  clamp_max_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_44: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(div_2, relu_9)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_18: "f32[4, 40, 28, 28]" = torch.ops.aten.convolution.default(mul_44, primals_51, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_33: "f32[40]" = torch.ops.aten.add.Tensor(primals_218, 0.001)
    sqrt_14: "f32[40]" = torch.ops.aten.sqrt.default(add_33);  add_33 = None
    reciprocal_14: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
    mul_45: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
    unsqueeze_112: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_217, -1)
    unsqueeze_113: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    unsqueeze_114: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_45, -1);  mul_45 = None
    unsqueeze_115: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    sub_14: "f32[4, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_113);  unsqueeze_113 = None
    mul_46: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
    unsqueeze_116: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1)
    unsqueeze_117: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_47: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_46, unsqueeze_117);  mul_46 = unsqueeze_117 = None
    unsqueeze_118: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1);  primals_53 = None
    unsqueeze_119: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_34: "f32[4, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_47, unsqueeze_119);  mul_47 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:113, code: result += input
    add_35: "f32[4, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_34, add_27);  add_34 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_19: "f32[4, 120, 28, 28]" = torch.ops.aten.convolution.default(add_35, primals_54, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_36: "f32[120]" = torch.ops.aten.add.Tensor(primals_221, 0.001)
    sqrt_15: "f32[120]" = torch.ops.aten.sqrt.default(add_36);  add_36 = None
    reciprocal_15: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
    mul_48: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
    unsqueeze_120: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_220, -1)
    unsqueeze_121: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    unsqueeze_122: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_48, -1);  mul_48 = None
    unsqueeze_123: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    sub_15: "f32[4, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_121);  unsqueeze_121 = None
    mul_49: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
    unsqueeze_124: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1)
    unsqueeze_125: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_50: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_125);  mul_49 = unsqueeze_125 = None
    unsqueeze_126: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1);  primals_56 = None
    unsqueeze_127: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_37: "f32[4, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_50, unsqueeze_127);  mul_50 = unsqueeze_127 = None
    relu_11: "f32[4, 120, 28, 28]" = torch.ops.aten.relu.default(add_37);  add_37 = None
    convolution_20: "f32[4, 120, 28, 28]" = torch.ops.aten.convolution.default(relu_11, primals_57, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120)
    add_38: "f32[120]" = torch.ops.aten.add.Tensor(primals_224, 0.001)
    sqrt_16: "f32[120]" = torch.ops.aten.sqrt.default(add_38);  add_38 = None
    reciprocal_16: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
    mul_51: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
    unsqueeze_128: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_223, -1)
    unsqueeze_129: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    unsqueeze_130: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_51, -1);  mul_51 = None
    unsqueeze_131: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    sub_16: "f32[4, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_129);  unsqueeze_129 = None
    mul_52: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
    unsqueeze_132: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_58, -1)
    unsqueeze_133: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_53: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_52, unsqueeze_133);  mul_52 = unsqueeze_133 = None
    unsqueeze_134: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1);  primals_59 = None
    unsqueeze_135: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_39: "f32[4, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_53, unsqueeze_135);  mul_53 = unsqueeze_135 = None
    relu_12: "f32[4, 120, 28, 28]" = torch.ops.aten.relu.default(add_39);  add_39 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    mean_2: "f32[4, 120, 1, 1]" = torch.ops.aten.mean.dim(relu_12, [-1, -2], True)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    convolution_21: "f32[4, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_2, primals_60, primals_61, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_61 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    relu_13: "f32[4, 32, 1, 1]" = torch.ops.aten.relu.default(convolution_21);  convolution_21 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    convolution_22: "f32[4, 120, 1, 1]" = torch.ops.aten.convolution.default(relu_13, primals_62, primals_63, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_63 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    add_40: "f32[4, 120, 1, 1]" = torch.ops.aten.add.Tensor(convolution_22, 3)
    clamp_min_3: "f32[4, 120, 1, 1]" = torch.ops.aten.clamp_min.default(add_40, 0);  add_40 = None
    clamp_max_3: "f32[4, 120, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_3, 6);  clamp_min_3 = None
    div_3: "f32[4, 120, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_3, 6);  clamp_max_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_54: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(div_3, relu_12)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_23: "f32[4, 40, 28, 28]" = torch.ops.aten.convolution.default(mul_54, primals_64, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_41: "f32[40]" = torch.ops.aten.add.Tensor(primals_227, 0.001)
    sqrt_17: "f32[40]" = torch.ops.aten.sqrt.default(add_41);  add_41 = None
    reciprocal_17: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
    mul_55: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
    unsqueeze_136: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_226, -1)
    unsqueeze_137: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    unsqueeze_138: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_55, -1);  mul_55 = None
    unsqueeze_139: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    sub_17: "f32[4, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_137);  unsqueeze_137 = None
    mul_56: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_139);  sub_17 = unsqueeze_139 = None
    unsqueeze_140: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
    unsqueeze_141: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_57: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_141);  mul_56 = unsqueeze_141 = None
    unsqueeze_142: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
    unsqueeze_143: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_42: "f32[4, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_57, unsqueeze_143);  mul_57 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:113, code: result += input
    add_43: "f32[4, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_42, add_35);  add_42 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_24: "f32[4, 240, 28, 28]" = torch.ops.aten.convolution.default(add_43, primals_67, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_44: "f32[240]" = torch.ops.aten.add.Tensor(primals_230, 0.001)
    sqrt_18: "f32[240]" = torch.ops.aten.sqrt.default(add_44);  add_44 = None
    reciprocal_18: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
    mul_58: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
    unsqueeze_144: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_229, -1)
    unsqueeze_145: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    unsqueeze_146: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_58, -1);  mul_58 = None
    unsqueeze_147: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    sub_18: "f32[4, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_145);  unsqueeze_145 = None
    mul_59: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_147);  sub_18 = unsqueeze_147 = None
    unsqueeze_148: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1)
    unsqueeze_149: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_60: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_59, unsqueeze_149);  mul_59 = unsqueeze_149 = None
    unsqueeze_150: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1);  primals_69 = None
    unsqueeze_151: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_45: "f32[4, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_60, unsqueeze_151);  mul_60 = unsqueeze_151 = None
    add_46: "f32[4, 240, 28, 28]" = torch.ops.aten.add.Tensor(add_45, 3)
    clamp_min_4: "f32[4, 240, 28, 28]" = torch.ops.aten.clamp_min.default(add_46, 0);  add_46 = None
    clamp_max_4: "f32[4, 240, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_4, 6);  clamp_min_4 = None
    mul_61: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_45, clamp_max_4);  clamp_max_4 = None
    div_4: "f32[4, 240, 28, 28]" = torch.ops.aten.div.Tensor(mul_61, 6);  mul_61 = None
    convolution_25: "f32[4, 240, 14, 14]" = torch.ops.aten.convolution.default(div_4, primals_70, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 240)
    add_47: "f32[240]" = torch.ops.aten.add.Tensor(primals_233, 0.001)
    sqrt_19: "f32[240]" = torch.ops.aten.sqrt.default(add_47);  add_47 = None
    reciprocal_19: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
    mul_62: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
    unsqueeze_152: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_232, -1)
    unsqueeze_153: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    unsqueeze_154: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_62, -1);  mul_62 = None
    unsqueeze_155: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    sub_19: "f32[4, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_153);  unsqueeze_153 = None
    mul_63: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_155);  sub_19 = unsqueeze_155 = None
    unsqueeze_156: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_157: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_64: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_157);  mul_63 = unsqueeze_157 = None
    unsqueeze_158: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_159: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_48: "f32[4, 240, 14, 14]" = torch.ops.aten.add.Tensor(mul_64, unsqueeze_159);  mul_64 = unsqueeze_159 = None
    add_49: "f32[4, 240, 14, 14]" = torch.ops.aten.add.Tensor(add_48, 3)
    clamp_min_5: "f32[4, 240, 14, 14]" = torch.ops.aten.clamp_min.default(add_49, 0);  add_49 = None
    clamp_max_5: "f32[4, 240, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_5, 6);  clamp_min_5 = None
    mul_65: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(add_48, clamp_max_5);  clamp_max_5 = None
    div_5: "f32[4, 240, 14, 14]" = torch.ops.aten.div.Tensor(mul_65, 6);  mul_65 = None
    convolution_26: "f32[4, 80, 14, 14]" = torch.ops.aten.convolution.default(div_5, primals_73, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_50: "f32[80]" = torch.ops.aten.add.Tensor(primals_236, 0.001)
    sqrt_20: "f32[80]" = torch.ops.aten.sqrt.default(add_50);  add_50 = None
    reciprocal_20: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
    mul_66: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
    unsqueeze_160: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_235, -1)
    unsqueeze_161: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    unsqueeze_162: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_66, -1);  mul_66 = None
    unsqueeze_163: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    sub_20: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_161);  unsqueeze_161 = None
    mul_67: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_163);  sub_20 = unsqueeze_163 = None
    unsqueeze_164: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1)
    unsqueeze_165: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_68: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_67, unsqueeze_165);  mul_67 = unsqueeze_165 = None
    unsqueeze_166: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1);  primals_75 = None
    unsqueeze_167: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_51: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_68, unsqueeze_167);  mul_68 = unsqueeze_167 = None
    convolution_27: "f32[4, 200, 14, 14]" = torch.ops.aten.convolution.default(add_51, primals_76, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_52: "f32[200]" = torch.ops.aten.add.Tensor(primals_239, 0.001)
    sqrt_21: "f32[200]" = torch.ops.aten.sqrt.default(add_52);  add_52 = None
    reciprocal_21: "f32[200]" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
    mul_69: "f32[200]" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
    unsqueeze_168: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(primals_238, -1)
    unsqueeze_169: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    unsqueeze_170: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(mul_69, -1);  mul_69 = None
    unsqueeze_171: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    sub_21: "f32[4, 200, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_169);  unsqueeze_169 = None
    mul_70: "f32[4, 200, 14, 14]" = torch.ops.aten.mul.Tensor(sub_21, unsqueeze_171);  sub_21 = unsqueeze_171 = None
    unsqueeze_172: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
    unsqueeze_173: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_71: "f32[4, 200, 14, 14]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_173);  mul_70 = unsqueeze_173 = None
    unsqueeze_174: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
    unsqueeze_175: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_53: "f32[4, 200, 14, 14]" = torch.ops.aten.add.Tensor(mul_71, unsqueeze_175);  mul_71 = unsqueeze_175 = None
    add_54: "f32[4, 200, 14, 14]" = torch.ops.aten.add.Tensor(add_53, 3)
    clamp_min_6: "f32[4, 200, 14, 14]" = torch.ops.aten.clamp_min.default(add_54, 0);  add_54 = None
    clamp_max_6: "f32[4, 200, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_6, 6);  clamp_min_6 = None
    mul_72: "f32[4, 200, 14, 14]" = torch.ops.aten.mul.Tensor(add_53, clamp_max_6);  clamp_max_6 = None
    div_6: "f32[4, 200, 14, 14]" = torch.ops.aten.div.Tensor(mul_72, 6);  mul_72 = None
    convolution_28: "f32[4, 200, 14, 14]" = torch.ops.aten.convolution.default(div_6, primals_79, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 200)
    add_55: "f32[200]" = torch.ops.aten.add.Tensor(primals_242, 0.001)
    sqrt_22: "f32[200]" = torch.ops.aten.sqrt.default(add_55);  add_55 = None
    reciprocal_22: "f32[200]" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
    mul_73: "f32[200]" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
    unsqueeze_176: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(primals_241, -1)
    unsqueeze_177: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    unsqueeze_178: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(mul_73, -1);  mul_73 = None
    unsqueeze_179: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    sub_22: "f32[4, 200, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_177);  unsqueeze_177 = None
    mul_74: "f32[4, 200, 14, 14]" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_179);  sub_22 = unsqueeze_179 = None
    unsqueeze_180: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1)
    unsqueeze_181: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_75: "f32[4, 200, 14, 14]" = torch.ops.aten.mul.Tensor(mul_74, unsqueeze_181);  mul_74 = unsqueeze_181 = None
    unsqueeze_182: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1);  primals_81 = None
    unsqueeze_183: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_56: "f32[4, 200, 14, 14]" = torch.ops.aten.add.Tensor(mul_75, unsqueeze_183);  mul_75 = unsqueeze_183 = None
    add_57: "f32[4, 200, 14, 14]" = torch.ops.aten.add.Tensor(add_56, 3)
    clamp_min_7: "f32[4, 200, 14, 14]" = torch.ops.aten.clamp_min.default(add_57, 0);  add_57 = None
    clamp_max_7: "f32[4, 200, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_7, 6);  clamp_min_7 = None
    mul_76: "f32[4, 200, 14, 14]" = torch.ops.aten.mul.Tensor(add_56, clamp_max_7);  clamp_max_7 = None
    div_7: "f32[4, 200, 14, 14]" = torch.ops.aten.div.Tensor(mul_76, 6);  mul_76 = None
    convolution_29: "f32[4, 80, 14, 14]" = torch.ops.aten.convolution.default(div_7, primals_82, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_58: "f32[80]" = torch.ops.aten.add.Tensor(primals_245, 0.001)
    sqrt_23: "f32[80]" = torch.ops.aten.sqrt.default(add_58);  add_58 = None
    reciprocal_23: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
    mul_77: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
    unsqueeze_184: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_244, -1)
    unsqueeze_185: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    unsqueeze_186: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_77, -1);  mul_77 = None
    unsqueeze_187: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    sub_23: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_185);  unsqueeze_185 = None
    mul_78: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_187);  sub_23 = unsqueeze_187 = None
    unsqueeze_188: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1)
    unsqueeze_189: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_79: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_78, unsqueeze_189);  mul_78 = unsqueeze_189 = None
    unsqueeze_190: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
    unsqueeze_191: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_59: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_79, unsqueeze_191);  mul_79 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:113, code: result += input
    add_60: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_59, add_51);  add_59 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_30: "f32[4, 184, 14, 14]" = torch.ops.aten.convolution.default(add_60, primals_85, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_61: "f32[184]" = torch.ops.aten.add.Tensor(primals_248, 0.001)
    sqrt_24: "f32[184]" = torch.ops.aten.sqrt.default(add_61);  add_61 = None
    reciprocal_24: "f32[184]" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
    mul_80: "f32[184]" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
    unsqueeze_192: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_247, -1)
    unsqueeze_193: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    unsqueeze_194: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(mul_80, -1);  mul_80 = None
    unsqueeze_195: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    sub_24: "f32[4, 184, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_193);  unsqueeze_193 = None
    mul_81: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_195);  sub_24 = unsqueeze_195 = None
    unsqueeze_196: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1)
    unsqueeze_197: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_82: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(mul_81, unsqueeze_197);  mul_81 = unsqueeze_197 = None
    unsqueeze_198: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1);  primals_87 = None
    unsqueeze_199: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_62: "f32[4, 184, 14, 14]" = torch.ops.aten.add.Tensor(mul_82, unsqueeze_199);  mul_82 = unsqueeze_199 = None
    add_63: "f32[4, 184, 14, 14]" = torch.ops.aten.add.Tensor(add_62, 3)
    clamp_min_8: "f32[4, 184, 14, 14]" = torch.ops.aten.clamp_min.default(add_63, 0);  add_63 = None
    clamp_max_8: "f32[4, 184, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_8, 6);  clamp_min_8 = None
    mul_83: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(add_62, clamp_max_8);  clamp_max_8 = None
    div_8: "f32[4, 184, 14, 14]" = torch.ops.aten.div.Tensor(mul_83, 6);  mul_83 = None
    convolution_31: "f32[4, 184, 14, 14]" = torch.ops.aten.convolution.default(div_8, primals_88, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 184)
    add_64: "f32[184]" = torch.ops.aten.add.Tensor(primals_251, 0.001)
    sqrt_25: "f32[184]" = torch.ops.aten.sqrt.default(add_64);  add_64 = None
    reciprocal_25: "f32[184]" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
    mul_84: "f32[184]" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
    unsqueeze_200: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_250, -1)
    unsqueeze_201: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    unsqueeze_202: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(mul_84, -1);  mul_84 = None
    unsqueeze_203: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    sub_25: "f32[4, 184, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_201);  unsqueeze_201 = None
    mul_85: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_203);  sub_25 = unsqueeze_203 = None
    unsqueeze_204: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1)
    unsqueeze_205: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_86: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(mul_85, unsqueeze_205);  mul_85 = unsqueeze_205 = None
    unsqueeze_206: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1);  primals_90 = None
    unsqueeze_207: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_65: "f32[4, 184, 14, 14]" = torch.ops.aten.add.Tensor(mul_86, unsqueeze_207);  mul_86 = unsqueeze_207 = None
    add_66: "f32[4, 184, 14, 14]" = torch.ops.aten.add.Tensor(add_65, 3)
    clamp_min_9: "f32[4, 184, 14, 14]" = torch.ops.aten.clamp_min.default(add_66, 0);  add_66 = None
    clamp_max_9: "f32[4, 184, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_9, 6);  clamp_min_9 = None
    mul_87: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(add_65, clamp_max_9);  clamp_max_9 = None
    div_9: "f32[4, 184, 14, 14]" = torch.ops.aten.div.Tensor(mul_87, 6);  mul_87 = None
    convolution_32: "f32[4, 80, 14, 14]" = torch.ops.aten.convolution.default(div_9, primals_91, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_67: "f32[80]" = torch.ops.aten.add.Tensor(primals_254, 0.001)
    sqrt_26: "f32[80]" = torch.ops.aten.sqrt.default(add_67);  add_67 = None
    reciprocal_26: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
    mul_88: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
    unsqueeze_208: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_253, -1)
    unsqueeze_209: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    unsqueeze_210: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_88, -1);  mul_88 = None
    unsqueeze_211: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    sub_26: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_209);  unsqueeze_209 = None
    mul_89: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_211);  sub_26 = unsqueeze_211 = None
    unsqueeze_212: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1)
    unsqueeze_213: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_90: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_89, unsqueeze_213);  mul_89 = unsqueeze_213 = None
    unsqueeze_214: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_93, -1);  primals_93 = None
    unsqueeze_215: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_68: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_215);  mul_90 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:113, code: result += input
    add_69: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_68, add_60);  add_68 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_33: "f32[4, 184, 14, 14]" = torch.ops.aten.convolution.default(add_69, primals_94, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_70: "f32[184]" = torch.ops.aten.add.Tensor(primals_257, 0.001)
    sqrt_27: "f32[184]" = torch.ops.aten.sqrt.default(add_70);  add_70 = None
    reciprocal_27: "f32[184]" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
    mul_91: "f32[184]" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
    unsqueeze_216: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_256, -1)
    unsqueeze_217: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    unsqueeze_218: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(mul_91, -1);  mul_91 = None
    unsqueeze_219: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    sub_27: "f32[4, 184, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_217);  unsqueeze_217 = None
    mul_92: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_219);  sub_27 = unsqueeze_219 = None
    unsqueeze_220: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1)
    unsqueeze_221: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_93: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(mul_92, unsqueeze_221);  mul_92 = unsqueeze_221 = None
    unsqueeze_222: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1);  primals_96 = None
    unsqueeze_223: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_71: "f32[4, 184, 14, 14]" = torch.ops.aten.add.Tensor(mul_93, unsqueeze_223);  mul_93 = unsqueeze_223 = None
    add_72: "f32[4, 184, 14, 14]" = torch.ops.aten.add.Tensor(add_71, 3)
    clamp_min_10: "f32[4, 184, 14, 14]" = torch.ops.aten.clamp_min.default(add_72, 0);  add_72 = None
    clamp_max_10: "f32[4, 184, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_10, 6);  clamp_min_10 = None
    mul_94: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(add_71, clamp_max_10);  clamp_max_10 = None
    div_10: "f32[4, 184, 14, 14]" = torch.ops.aten.div.Tensor(mul_94, 6);  mul_94 = None
    convolution_34: "f32[4, 184, 14, 14]" = torch.ops.aten.convolution.default(div_10, primals_97, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 184)
    add_73: "f32[184]" = torch.ops.aten.add.Tensor(primals_260, 0.001)
    sqrt_28: "f32[184]" = torch.ops.aten.sqrt.default(add_73);  add_73 = None
    reciprocal_28: "f32[184]" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
    mul_95: "f32[184]" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
    unsqueeze_224: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_259, -1)
    unsqueeze_225: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    unsqueeze_226: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(mul_95, -1);  mul_95 = None
    unsqueeze_227: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    sub_28: "f32[4, 184, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_225);  unsqueeze_225 = None
    mul_96: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_227);  sub_28 = unsqueeze_227 = None
    unsqueeze_228: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1)
    unsqueeze_229: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_97: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(mul_96, unsqueeze_229);  mul_96 = unsqueeze_229 = None
    unsqueeze_230: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1);  primals_99 = None
    unsqueeze_231: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_74: "f32[4, 184, 14, 14]" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_231);  mul_97 = unsqueeze_231 = None
    add_75: "f32[4, 184, 14, 14]" = torch.ops.aten.add.Tensor(add_74, 3)
    clamp_min_11: "f32[4, 184, 14, 14]" = torch.ops.aten.clamp_min.default(add_75, 0);  add_75 = None
    clamp_max_11: "f32[4, 184, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_11, 6);  clamp_min_11 = None
    mul_98: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(add_74, clamp_max_11);  clamp_max_11 = None
    div_11: "f32[4, 184, 14, 14]" = torch.ops.aten.div.Tensor(mul_98, 6);  mul_98 = None
    convolution_35: "f32[4, 80, 14, 14]" = torch.ops.aten.convolution.default(div_11, primals_100, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_76: "f32[80]" = torch.ops.aten.add.Tensor(primals_263, 0.001)
    sqrt_29: "f32[80]" = torch.ops.aten.sqrt.default(add_76);  add_76 = None
    reciprocal_29: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
    mul_99: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
    unsqueeze_232: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_262, -1)
    unsqueeze_233: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    unsqueeze_234: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_99, -1);  mul_99 = None
    unsqueeze_235: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    sub_29: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_233);  unsqueeze_233 = None
    mul_100: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_235);  sub_29 = unsqueeze_235 = None
    unsqueeze_236: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_101, -1)
    unsqueeze_237: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_101: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_100, unsqueeze_237);  mul_100 = unsqueeze_237 = None
    unsqueeze_238: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1);  primals_102 = None
    unsqueeze_239: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_77: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_101, unsqueeze_239);  mul_101 = unsqueeze_239 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:113, code: result += input
    add_78: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_77, add_69);  add_77 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_36: "f32[4, 480, 14, 14]" = torch.ops.aten.convolution.default(add_78, primals_103, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_79: "f32[480]" = torch.ops.aten.add.Tensor(primals_266, 0.001)
    sqrt_30: "f32[480]" = torch.ops.aten.sqrt.default(add_79);  add_79 = None
    reciprocal_30: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
    mul_102: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
    unsqueeze_240: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_265, -1)
    unsqueeze_241: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    unsqueeze_242: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_102, -1);  mul_102 = None
    unsqueeze_243: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    sub_30: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_241);  unsqueeze_241 = None
    mul_103: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_243);  sub_30 = unsqueeze_243 = None
    unsqueeze_244: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_104, -1)
    unsqueeze_245: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_104: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_103, unsqueeze_245);  mul_103 = unsqueeze_245 = None
    unsqueeze_246: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_105, -1);  primals_105 = None
    unsqueeze_247: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_80: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_247);  mul_104 = unsqueeze_247 = None
    add_81: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(add_80, 3)
    clamp_min_12: "f32[4, 480, 14, 14]" = torch.ops.aten.clamp_min.default(add_81, 0);  add_81 = None
    clamp_max_12: "f32[4, 480, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_12, 6);  clamp_min_12 = None
    mul_105: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_80, clamp_max_12);  clamp_max_12 = None
    div_12: "f32[4, 480, 14, 14]" = torch.ops.aten.div.Tensor(mul_105, 6);  mul_105 = None
    convolution_37: "f32[4, 480, 14, 14]" = torch.ops.aten.convolution.default(div_12, primals_106, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480)
    add_82: "f32[480]" = torch.ops.aten.add.Tensor(primals_269, 0.001)
    sqrt_31: "f32[480]" = torch.ops.aten.sqrt.default(add_82);  add_82 = None
    reciprocal_31: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
    mul_106: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
    unsqueeze_248: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_268, -1)
    unsqueeze_249: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    unsqueeze_250: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_106, -1);  mul_106 = None
    unsqueeze_251: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    sub_31: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_249);  unsqueeze_249 = None
    mul_107: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_251);  sub_31 = unsqueeze_251 = None
    unsqueeze_252: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_107, -1)
    unsqueeze_253: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_108: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_107, unsqueeze_253);  mul_107 = unsqueeze_253 = None
    unsqueeze_254: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1);  primals_108 = None
    unsqueeze_255: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_83: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_108, unsqueeze_255);  mul_108 = unsqueeze_255 = None
    add_84: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(add_83, 3)
    clamp_min_13: "f32[4, 480, 14, 14]" = torch.ops.aten.clamp_min.default(add_84, 0);  add_84 = None
    clamp_max_13: "f32[4, 480, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_13, 6);  clamp_min_13 = None
    mul_109: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_83, clamp_max_13);  clamp_max_13 = None
    div_13: "f32[4, 480, 14, 14]" = torch.ops.aten.div.Tensor(mul_109, 6);  mul_109 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    mean_3: "f32[4, 480, 1, 1]" = torch.ops.aten.mean.dim(div_13, [-1, -2], True)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    convolution_38: "f32[4, 120, 1, 1]" = torch.ops.aten.convolution.default(mean_3, primals_109, primals_110, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_110 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    relu_14: "f32[4, 120, 1, 1]" = torch.ops.aten.relu.default(convolution_38);  convolution_38 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    convolution_39: "f32[4, 480, 1, 1]" = torch.ops.aten.convolution.default(relu_14, primals_111, primals_112, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_112 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    add_85: "f32[4, 480, 1, 1]" = torch.ops.aten.add.Tensor(convolution_39, 3)
    clamp_min_14: "f32[4, 480, 1, 1]" = torch.ops.aten.clamp_min.default(add_85, 0);  add_85 = None
    clamp_max_14: "f32[4, 480, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_14, 6);  clamp_min_14 = None
    div_14: "f32[4, 480, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_14, 6);  clamp_max_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_110: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(div_14, div_13)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_40: "f32[4, 112, 14, 14]" = torch.ops.aten.convolution.default(mul_110, primals_113, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_86: "f32[112]" = torch.ops.aten.add.Tensor(primals_272, 0.001)
    sqrt_32: "f32[112]" = torch.ops.aten.sqrt.default(add_86);  add_86 = None
    reciprocal_32: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
    mul_111: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
    unsqueeze_256: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_271, -1)
    unsqueeze_257: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    unsqueeze_258: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_111, -1);  mul_111 = None
    unsqueeze_259: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    sub_32: "f32[4, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_257);  unsqueeze_257 = None
    mul_112: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_259);  sub_32 = unsqueeze_259 = None
    unsqueeze_260: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_114, -1)
    unsqueeze_261: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
    mul_113: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_261);  mul_112 = unsqueeze_261 = None
    unsqueeze_262: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_115, -1);  primals_115 = None
    unsqueeze_263: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
    add_87: "f32[4, 112, 14, 14]" = torch.ops.aten.add.Tensor(mul_113, unsqueeze_263);  mul_113 = unsqueeze_263 = None
    convolution_41: "f32[4, 672, 14, 14]" = torch.ops.aten.convolution.default(add_87, primals_116, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_88: "f32[672]" = torch.ops.aten.add.Tensor(primals_275, 0.001)
    sqrt_33: "f32[672]" = torch.ops.aten.sqrt.default(add_88);  add_88 = None
    reciprocal_33: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
    mul_114: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
    unsqueeze_264: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_274, -1)
    unsqueeze_265: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
    unsqueeze_266: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_114, -1);  mul_114 = None
    unsqueeze_267: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
    sub_33: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_265);  unsqueeze_265 = None
    mul_115: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_33, unsqueeze_267);  sub_33 = unsqueeze_267 = None
    unsqueeze_268: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_117, -1)
    unsqueeze_269: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
    mul_116: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_115, unsqueeze_269);  mul_115 = unsqueeze_269 = None
    unsqueeze_270: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_118, -1);  primals_118 = None
    unsqueeze_271: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
    add_89: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_116, unsqueeze_271);  mul_116 = unsqueeze_271 = None
    add_90: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(add_89, 3)
    clamp_min_15: "f32[4, 672, 14, 14]" = torch.ops.aten.clamp_min.default(add_90, 0);  add_90 = None
    clamp_max_15: "f32[4, 672, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_15, 6);  clamp_min_15 = None
    mul_117: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_89, clamp_max_15);  clamp_max_15 = None
    div_15: "f32[4, 672, 14, 14]" = torch.ops.aten.div.Tensor(mul_117, 6);  mul_117 = None
    convolution_42: "f32[4, 672, 14, 14]" = torch.ops.aten.convolution.default(div_15, primals_119, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 672)
    add_91: "f32[672]" = torch.ops.aten.add.Tensor(primals_278, 0.001)
    sqrt_34: "f32[672]" = torch.ops.aten.sqrt.default(add_91);  add_91 = None
    reciprocal_34: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
    mul_118: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
    unsqueeze_272: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_277, -1)
    unsqueeze_273: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
    unsqueeze_274: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_118, -1);  mul_118 = None
    unsqueeze_275: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
    sub_34: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_273);  unsqueeze_273 = None
    mul_119: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_275);  sub_34 = unsqueeze_275 = None
    unsqueeze_276: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_120, -1)
    unsqueeze_277: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
    mul_120: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_277);  mul_119 = unsqueeze_277 = None
    unsqueeze_278: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_121, -1);  primals_121 = None
    unsqueeze_279: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
    add_92: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_120, unsqueeze_279);  mul_120 = unsqueeze_279 = None
    add_93: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(add_92, 3)
    clamp_min_16: "f32[4, 672, 14, 14]" = torch.ops.aten.clamp_min.default(add_93, 0);  add_93 = None
    clamp_max_16: "f32[4, 672, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_16, 6);  clamp_min_16 = None
    mul_121: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_92, clamp_max_16);  clamp_max_16 = None
    div_16: "f32[4, 672, 14, 14]" = torch.ops.aten.div.Tensor(mul_121, 6);  mul_121 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    mean_4: "f32[4, 672, 1, 1]" = torch.ops.aten.mean.dim(div_16, [-1, -2], True)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    convolution_43: "f32[4, 168, 1, 1]" = torch.ops.aten.convolution.default(mean_4, primals_122, primals_123, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_123 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    relu_15: "f32[4, 168, 1, 1]" = torch.ops.aten.relu.default(convolution_43);  convolution_43 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    convolution_44: "f32[4, 672, 1, 1]" = torch.ops.aten.convolution.default(relu_15, primals_124, primals_125, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_125 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    add_94: "f32[4, 672, 1, 1]" = torch.ops.aten.add.Tensor(convolution_44, 3)
    clamp_min_17: "f32[4, 672, 1, 1]" = torch.ops.aten.clamp_min.default(add_94, 0);  add_94 = None
    clamp_max_17: "f32[4, 672, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_17, 6);  clamp_min_17 = None
    div_17: "f32[4, 672, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_17, 6);  clamp_max_17 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_122: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(div_17, div_16)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_45: "f32[4, 112, 14, 14]" = torch.ops.aten.convolution.default(mul_122, primals_126, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_95: "f32[112]" = torch.ops.aten.add.Tensor(primals_281, 0.001)
    sqrt_35: "f32[112]" = torch.ops.aten.sqrt.default(add_95);  add_95 = None
    reciprocal_35: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
    mul_123: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
    unsqueeze_280: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_280, -1)
    unsqueeze_281: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
    unsqueeze_282: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_123, -1);  mul_123 = None
    unsqueeze_283: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
    sub_35: "f32[4, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_281);  unsqueeze_281 = None
    mul_124: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_283);  sub_35 = unsqueeze_283 = None
    unsqueeze_284: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_127, -1)
    unsqueeze_285: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
    mul_125: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(mul_124, unsqueeze_285);  mul_124 = unsqueeze_285 = None
    unsqueeze_286: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_128, -1);  primals_128 = None
    unsqueeze_287: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
    add_96: "f32[4, 112, 14, 14]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_287);  mul_125 = unsqueeze_287 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:113, code: result += input
    add_97: "f32[4, 112, 14, 14]" = torch.ops.aten.add.Tensor(add_96, add_87);  add_96 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_46: "f32[4, 672, 14, 14]" = torch.ops.aten.convolution.default(add_97, primals_129, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_98: "f32[672]" = torch.ops.aten.add.Tensor(primals_284, 0.001)
    sqrt_36: "f32[672]" = torch.ops.aten.sqrt.default(add_98);  add_98 = None
    reciprocal_36: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
    mul_126: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
    unsqueeze_288: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_283, -1)
    unsqueeze_289: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
    unsqueeze_290: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_126, -1);  mul_126 = None
    unsqueeze_291: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
    sub_36: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_289);  unsqueeze_289 = None
    mul_127: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_291);  sub_36 = unsqueeze_291 = None
    unsqueeze_292: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_130, -1)
    unsqueeze_293: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
    mul_128: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_127, unsqueeze_293);  mul_127 = unsqueeze_293 = None
    unsqueeze_294: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_131, -1);  primals_131 = None
    unsqueeze_295: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
    add_99: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_128, unsqueeze_295);  mul_128 = unsqueeze_295 = None
    add_100: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(add_99, 3)
    clamp_min_18: "f32[4, 672, 14, 14]" = torch.ops.aten.clamp_min.default(add_100, 0);  add_100 = None
    clamp_max_18: "f32[4, 672, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_18, 6);  clamp_min_18 = None
    mul_129: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_99, clamp_max_18);  clamp_max_18 = None
    div_18: "f32[4, 672, 14, 14]" = torch.ops.aten.div.Tensor(mul_129, 6);  mul_129 = None
    convolution_47: "f32[4, 672, 7, 7]" = torch.ops.aten.convolution.default(div_18, primals_132, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 672)
    add_101: "f32[672]" = torch.ops.aten.add.Tensor(primals_287, 0.001)
    sqrt_37: "f32[672]" = torch.ops.aten.sqrt.default(add_101);  add_101 = None
    reciprocal_37: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
    mul_130: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
    unsqueeze_296: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_286, -1)
    unsqueeze_297: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
    unsqueeze_298: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_130, -1);  mul_130 = None
    unsqueeze_299: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
    sub_37: "f32[4, 672, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_297);  unsqueeze_297 = None
    mul_131: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_299);  sub_37 = unsqueeze_299 = None
    unsqueeze_300: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_133, -1)
    unsqueeze_301: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
    mul_132: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(mul_131, unsqueeze_301);  mul_131 = unsqueeze_301 = None
    unsqueeze_302: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_134, -1);  primals_134 = None
    unsqueeze_303: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
    add_102: "f32[4, 672, 7, 7]" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_303);  mul_132 = unsqueeze_303 = None
    add_103: "f32[4, 672, 7, 7]" = torch.ops.aten.add.Tensor(add_102, 3)
    clamp_min_19: "f32[4, 672, 7, 7]" = torch.ops.aten.clamp_min.default(add_103, 0);  add_103 = None
    clamp_max_19: "f32[4, 672, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_19, 6);  clamp_min_19 = None
    mul_133: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(add_102, clamp_max_19);  clamp_max_19 = None
    div_19: "f32[4, 672, 7, 7]" = torch.ops.aten.div.Tensor(mul_133, 6);  mul_133 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    mean_5: "f32[4, 672, 1, 1]" = torch.ops.aten.mean.dim(div_19, [-1, -2], True)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    convolution_48: "f32[4, 168, 1, 1]" = torch.ops.aten.convolution.default(mean_5, primals_135, primals_136, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_136 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    relu_16: "f32[4, 168, 1, 1]" = torch.ops.aten.relu.default(convolution_48);  convolution_48 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    convolution_49: "f32[4, 672, 1, 1]" = torch.ops.aten.convolution.default(relu_16, primals_137, primals_138, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_138 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    add_104: "f32[4, 672, 1, 1]" = torch.ops.aten.add.Tensor(convolution_49, 3)
    clamp_min_20: "f32[4, 672, 1, 1]" = torch.ops.aten.clamp_min.default(add_104, 0);  add_104 = None
    clamp_max_20: "f32[4, 672, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_20, 6);  clamp_min_20 = None
    div_20: "f32[4, 672, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_20, 6);  clamp_max_20 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_134: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(div_20, div_19)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_50: "f32[4, 160, 7, 7]" = torch.ops.aten.convolution.default(mul_134, primals_139, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_105: "f32[160]" = torch.ops.aten.add.Tensor(primals_290, 0.001)
    sqrt_38: "f32[160]" = torch.ops.aten.sqrt.default(add_105);  add_105 = None
    reciprocal_38: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
    mul_135: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
    unsqueeze_304: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_289, -1)
    unsqueeze_305: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
    unsqueeze_306: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_135, -1);  mul_135 = None
    unsqueeze_307: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
    sub_38: "f32[4, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_305);  unsqueeze_305 = None
    mul_136: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_307);  sub_38 = unsqueeze_307 = None
    unsqueeze_308: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_140, -1)
    unsqueeze_309: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
    mul_137: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_309);  mul_136 = unsqueeze_309 = None
    unsqueeze_310: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_141, -1);  primals_141 = None
    unsqueeze_311: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
    add_106: "f32[4, 160, 7, 7]" = torch.ops.aten.add.Tensor(mul_137, unsqueeze_311);  mul_137 = unsqueeze_311 = None
    convolution_51: "f32[4, 960, 7, 7]" = torch.ops.aten.convolution.default(add_106, primals_142, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_107: "f32[960]" = torch.ops.aten.add.Tensor(primals_293, 0.001)
    sqrt_39: "f32[960]" = torch.ops.aten.sqrt.default(add_107);  add_107 = None
    reciprocal_39: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
    mul_138: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
    unsqueeze_312: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_292, -1)
    unsqueeze_313: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
    unsqueeze_314: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_138, -1);  mul_138 = None
    unsqueeze_315: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
    sub_39: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_313);  unsqueeze_313 = None
    mul_139: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_315);  sub_39 = unsqueeze_315 = None
    unsqueeze_316: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_143, -1)
    unsqueeze_317: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
    mul_140: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_139, unsqueeze_317);  mul_139 = unsqueeze_317 = None
    unsqueeze_318: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_144, -1);  primals_144 = None
    unsqueeze_319: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
    add_108: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_140, unsqueeze_319);  mul_140 = unsqueeze_319 = None
    add_109: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(add_108, 3)
    clamp_min_21: "f32[4, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_109, 0);  add_109 = None
    clamp_max_21: "f32[4, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_21, 6);  clamp_min_21 = None
    mul_141: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_108, clamp_max_21);  clamp_max_21 = None
    div_21: "f32[4, 960, 7, 7]" = torch.ops.aten.div.Tensor(mul_141, 6);  mul_141 = None
    convolution_52: "f32[4, 960, 7, 7]" = torch.ops.aten.convolution.default(div_21, primals_145, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 960)
    add_110: "f32[960]" = torch.ops.aten.add.Tensor(primals_296, 0.001)
    sqrt_40: "f32[960]" = torch.ops.aten.sqrt.default(add_110);  add_110 = None
    reciprocal_40: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
    mul_142: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
    unsqueeze_320: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_295, -1)
    unsqueeze_321: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
    unsqueeze_322: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_142, -1);  mul_142 = None
    unsqueeze_323: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
    sub_40: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_321);  unsqueeze_321 = None
    mul_143: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_323);  sub_40 = unsqueeze_323 = None
    unsqueeze_324: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_146, -1)
    unsqueeze_325: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
    mul_144: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_143, unsqueeze_325);  mul_143 = unsqueeze_325 = None
    unsqueeze_326: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_147, -1);  primals_147 = None
    unsqueeze_327: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
    add_111: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_144, unsqueeze_327);  mul_144 = unsqueeze_327 = None
    add_112: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(add_111, 3)
    clamp_min_22: "f32[4, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_112, 0);  add_112 = None
    clamp_max_22: "f32[4, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_22, 6);  clamp_min_22 = None
    mul_145: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_111, clamp_max_22);  clamp_max_22 = None
    div_22: "f32[4, 960, 7, 7]" = torch.ops.aten.div.Tensor(mul_145, 6);  mul_145 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    mean_6: "f32[4, 960, 1, 1]" = torch.ops.aten.mean.dim(div_22, [-1, -2], True)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    convolution_53: "f32[4, 240, 1, 1]" = torch.ops.aten.convolution.default(mean_6, primals_148, primals_149, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_149 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    relu_17: "f32[4, 240, 1, 1]" = torch.ops.aten.relu.default(convolution_53);  convolution_53 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    convolution_54: "f32[4, 960, 1, 1]" = torch.ops.aten.convolution.default(relu_17, primals_150, primals_151, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_151 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    add_113: "f32[4, 960, 1, 1]" = torch.ops.aten.add.Tensor(convolution_54, 3)
    clamp_min_23: "f32[4, 960, 1, 1]" = torch.ops.aten.clamp_min.default(add_113, 0);  add_113 = None
    clamp_max_23: "f32[4, 960, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_23, 6);  clamp_min_23 = None
    div_23: "f32[4, 960, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_23, 6);  clamp_max_23 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_146: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(div_23, div_22)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_55: "f32[4, 160, 7, 7]" = torch.ops.aten.convolution.default(mul_146, primals_152, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_114: "f32[160]" = torch.ops.aten.add.Tensor(primals_299, 0.001)
    sqrt_41: "f32[160]" = torch.ops.aten.sqrt.default(add_114);  add_114 = None
    reciprocal_41: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
    mul_147: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
    unsqueeze_328: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_298, -1)
    unsqueeze_329: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
    unsqueeze_330: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_147, -1);  mul_147 = None
    unsqueeze_331: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
    sub_41: "f32[4, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_329);  unsqueeze_329 = None
    mul_148: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_331);  sub_41 = unsqueeze_331 = None
    unsqueeze_332: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_153, -1)
    unsqueeze_333: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
    mul_149: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(mul_148, unsqueeze_333);  mul_148 = unsqueeze_333 = None
    unsqueeze_334: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_154, -1);  primals_154 = None
    unsqueeze_335: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
    add_115: "f32[4, 160, 7, 7]" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_335);  mul_149 = unsqueeze_335 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:113, code: result += input
    add_116: "f32[4, 160, 7, 7]" = torch.ops.aten.add.Tensor(add_115, add_106);  add_115 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_56: "f32[4, 960, 7, 7]" = torch.ops.aten.convolution.default(add_116, primals_155, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_117: "f32[960]" = torch.ops.aten.add.Tensor(primals_302, 0.001)
    sqrt_42: "f32[960]" = torch.ops.aten.sqrt.default(add_117);  add_117 = None
    reciprocal_42: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
    mul_150: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
    unsqueeze_336: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_301, -1)
    unsqueeze_337: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
    unsqueeze_338: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_150, -1);  mul_150 = None
    unsqueeze_339: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
    sub_42: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_337);  unsqueeze_337 = None
    mul_151: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_339);  sub_42 = unsqueeze_339 = None
    unsqueeze_340: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_156, -1)
    unsqueeze_341: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
    mul_152: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_151, unsqueeze_341);  mul_151 = unsqueeze_341 = None
    unsqueeze_342: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_157, -1);  primals_157 = None
    unsqueeze_343: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
    add_118: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_152, unsqueeze_343);  mul_152 = unsqueeze_343 = None
    add_119: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(add_118, 3)
    clamp_min_24: "f32[4, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_119, 0);  add_119 = None
    clamp_max_24: "f32[4, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_24, 6);  clamp_min_24 = None
    mul_153: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_118, clamp_max_24);  clamp_max_24 = None
    div_24: "f32[4, 960, 7, 7]" = torch.ops.aten.div.Tensor(mul_153, 6);  mul_153 = None
    convolution_57: "f32[4, 960, 7, 7]" = torch.ops.aten.convolution.default(div_24, primals_158, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 960)
    add_120: "f32[960]" = torch.ops.aten.add.Tensor(primals_305, 0.001)
    sqrt_43: "f32[960]" = torch.ops.aten.sqrt.default(add_120);  add_120 = None
    reciprocal_43: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
    mul_154: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
    unsqueeze_344: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_304, -1)
    unsqueeze_345: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
    unsqueeze_346: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_154, -1);  mul_154 = None
    unsqueeze_347: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
    sub_43: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_345);  unsqueeze_345 = None
    mul_155: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_347);  sub_43 = unsqueeze_347 = None
    unsqueeze_348: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_159, -1)
    unsqueeze_349: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
    mul_156: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_155, unsqueeze_349);  mul_155 = unsqueeze_349 = None
    unsqueeze_350: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_160, -1);  primals_160 = None
    unsqueeze_351: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
    add_121: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_156, unsqueeze_351);  mul_156 = unsqueeze_351 = None
    add_122: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(add_121, 3)
    clamp_min_25: "f32[4, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_122, 0);  add_122 = None
    clamp_max_25: "f32[4, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_25, 6);  clamp_min_25 = None
    mul_157: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_121, clamp_max_25);  clamp_max_25 = None
    div_25: "f32[4, 960, 7, 7]" = torch.ops.aten.div.Tensor(mul_157, 6);  mul_157 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    mean_7: "f32[4, 960, 1, 1]" = torch.ops.aten.mean.dim(div_25, [-1, -2], True)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    convolution_58: "f32[4, 240, 1, 1]" = torch.ops.aten.convolution.default(mean_7, primals_161, primals_162, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_162 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    relu_18: "f32[4, 240, 1, 1]" = torch.ops.aten.relu.default(convolution_58);  convolution_58 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    convolution_59: "f32[4, 960, 1, 1]" = torch.ops.aten.convolution.default(relu_18, primals_163, primals_164, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_164 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    add_123: "f32[4, 960, 1, 1]" = torch.ops.aten.add.Tensor(convolution_59, 3)
    clamp_min_26: "f32[4, 960, 1, 1]" = torch.ops.aten.clamp_min.default(add_123, 0);  add_123 = None
    clamp_max_26: "f32[4, 960, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_26, 6);  clamp_min_26 = None
    div_26: "f32[4, 960, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_26, 6);  clamp_max_26 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_158: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(div_26, div_25)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    convolution_60: "f32[4, 160, 7, 7]" = torch.ops.aten.convolution.default(mul_158, primals_165, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_124: "f32[160]" = torch.ops.aten.add.Tensor(primals_308, 0.001)
    sqrt_44: "f32[160]" = torch.ops.aten.sqrt.default(add_124);  add_124 = None
    reciprocal_44: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
    mul_159: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
    unsqueeze_352: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_307, -1)
    unsqueeze_353: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
    unsqueeze_354: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_159, -1);  mul_159 = None
    unsqueeze_355: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
    sub_44: "f32[4, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_353);  unsqueeze_353 = None
    mul_160: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_355);  sub_44 = unsqueeze_355 = None
    unsqueeze_356: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_166, -1)
    unsqueeze_357: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
    mul_161: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(mul_160, unsqueeze_357);  mul_160 = unsqueeze_357 = None
    unsqueeze_358: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_167, -1);  primals_167 = None
    unsqueeze_359: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
    add_125: "f32[4, 160, 7, 7]" = torch.ops.aten.add.Tensor(mul_161, unsqueeze_359);  mul_161 = unsqueeze_359 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:113, code: result += input
    add_126: "f32[4, 160, 7, 7]" = torch.ops.aten.add.Tensor(add_125, add_116);  add_125 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:210, code: x = self.features(x)
    convolution_61: "f32[4, 960, 7, 7]" = torch.ops.aten.convolution.default(add_126, primals_168, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_127: "f32[960]" = torch.ops.aten.add.Tensor(primals_311, 0.001)
    sqrt_45: "f32[960]" = torch.ops.aten.sqrt.default(add_127);  add_127 = None
    reciprocal_45: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
    mul_162: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
    unsqueeze_360: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_310, -1)
    unsqueeze_361: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
    unsqueeze_362: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_162, -1);  mul_162 = None
    unsqueeze_363: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
    sub_45: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_361);  unsqueeze_361 = None
    mul_163: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_363);  sub_45 = unsqueeze_363 = None
    unsqueeze_364: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_169, -1)
    unsqueeze_365: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
    mul_164: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_163, unsqueeze_365);  mul_163 = unsqueeze_365 = None
    unsqueeze_366: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_170, -1);  primals_170 = None
    unsqueeze_367: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
    add_128: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_164, unsqueeze_367);  mul_164 = unsqueeze_367 = None
    add_129: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(add_128, 3)
    clamp_min_27: "f32[4, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_129, 0);  add_129 = None
    clamp_max_27: "f32[4, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_27, 6);  clamp_min_27 = None
    mul_165: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_128, clamp_max_27);  clamp_max_27 = None
    div_27: "f32[4, 960, 7, 7]" = torch.ops.aten.div.Tensor(mul_165, 6);  mul_165 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:212, code: x = self.avgpool(x)
    mean_8: "f32[4, 960, 1, 1]" = torch.ops.aten.mean.dim(div_27, [-1, -2], True);  div_27 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:213, code: x = torch.flatten(x, 1)
    view: "f32[4, 960]" = torch.ops.aten.reshape.default(mean_8, [4, 960]);  mean_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:215, code: x = self.classifier(x)
    permute: "f32[960, 1280]" = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
    addmm: "f32[4, 1280]" = torch.ops.aten.addmm.default(primals_172, view, permute);  primals_172 = None
    add_130: "f32[4, 1280]" = torch.ops.aten.add.Tensor(addmm, 3)
    clamp_min_28: "f32[4, 1280]" = torch.ops.aten.clamp_min.default(add_130, 0);  add_130 = None
    clamp_max_28: "f32[4, 1280]" = torch.ops.aten.clamp_max.default(clamp_min_28, 6);  clamp_min_28 = None
    mul_166: "f32[4, 1280]" = torch.ops.aten.mul.Tensor(addmm, clamp_max_28);  clamp_max_28 = None
    div_28: "f32[4, 1280]" = torch.ops.aten.div.Tensor(mul_166, 6);  mul_166 = None
    permute_1: "f32[1280, 1000]" = torch.ops.aten.permute.default(primals_173, [1, 0]);  primals_173 = None
    addmm_1: "f32[4, 1000]" = torch.ops.aten.addmm.default(primals_174, div_28, permute_1);  primals_174 = None
    permute_2: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    permute_6: "f32[1280, 960]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    gt: "b8[4, 960, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_59, -3.0)
    lt_2: "b8[4, 960, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_59, 3.0);  convolution_59 = None
    bitwise_and: "b8[4, 960, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt, lt_2);  gt = lt_2 = None
    gt_1: "b8[4, 960, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_54, -3.0)
    lt_5: "b8[4, 960, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_54, 3.0);  convolution_54 = None
    bitwise_and_1: "b8[4, 960, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_1, lt_5);  gt_1 = lt_5 = None
    gt_2: "b8[4, 672, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_49, -3.0)
    lt_8: "b8[4, 672, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_49, 3.0);  convolution_49 = None
    bitwise_and_2: "b8[4, 672, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_2, lt_8);  gt_2 = lt_8 = None
    gt_3: "b8[4, 672, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_44, -3.0)
    lt_11: "b8[4, 672, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_44, 3.0);  convolution_44 = None
    bitwise_and_3: "b8[4, 672, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_3, lt_11);  gt_3 = lt_11 = None
    gt_4: "b8[4, 480, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_39, -3.0)
    lt_14: "b8[4, 480, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_39, 3.0);  convolution_39 = None
    bitwise_and_4: "b8[4, 480, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_4, lt_14);  gt_4 = lt_14 = None
    gt_5: "b8[4, 120, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_22, -3.0)
    lt_25: "b8[4, 120, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_22, 3.0);  convolution_22 = None
    bitwise_and_5: "b8[4, 120, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_5, lt_25);  gt_5 = lt_25 = None
    gt_6: "b8[4, 120, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_17, -3.0)
    lt_26: "b8[4, 120, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_17, 3.0);  convolution_17 = None
    bitwise_and_6: "b8[4, 120, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_6, lt_26);  gt_6 = lt_26 = None
    gt_7: "b8[4, 72, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_12, -3.0)
    lt_27: "b8[4, 72, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_12, 3.0);  convolution_12 = None
    bitwise_and_7: "b8[4, 72, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_7, lt_27);  gt_7 = lt_27 = None
    return [addmm_1, primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_36, primals_38, primals_39, primals_41, primals_42, primals_44, primals_45, primals_47, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_60, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_111, primals_113, primals_114, primals_116, primals_117, primals_119, primals_120, primals_122, primals_124, primals_126, primals_127, primals_129, primals_130, primals_132, primals_133, primals_135, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_150, primals_152, primals_153, primals_155, primals_156, primals_158, primals_159, primals_161, primals_163, primals_165, primals_166, primals_168, primals_169, primals_175, primals_176, primals_178, primals_179, primals_181, primals_182, primals_184, primals_185, primals_187, primals_188, primals_190, primals_191, primals_193, primals_194, primals_196, primals_197, primals_199, primals_200, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_214, primals_215, primals_217, primals_218, primals_220, primals_221, primals_223, primals_224, primals_226, primals_227, primals_229, primals_230, primals_232, primals_233, primals_235, primals_236, primals_238, primals_239, primals_241, primals_242, primals_244, primals_245, primals_247, primals_248, primals_250, primals_251, primals_253, primals_254, primals_256, primals_257, primals_259, primals_260, primals_262, primals_263, primals_265, primals_266, primals_268, primals_269, primals_271, primals_272, primals_274, primals_275, primals_277, primals_278, primals_280, primals_281, primals_283, primals_284, primals_286, primals_287, primals_289, primals_290, primals_292, primals_293, primals_295, primals_296, primals_298, primals_299, primals_301, primals_302, primals_304, primals_305, primals_307, primals_308, primals_310, primals_311, primals_313, convolution, add_1, div, convolution_1, relu, convolution_2, add_7, convolution_3, relu_1, convolution_4, relu_2, convolution_5, add_13, convolution_6, relu_3, convolution_7, relu_4, convolution_8, add_20, convolution_9, relu_5, convolution_10, relu_6, mean, relu_7, div_1, mul_34, convolution_13, add_27, convolution_14, relu_8, convolution_15, relu_9, mean_1, relu_10, div_2, mul_44, convolution_18, add_35, convolution_19, relu_11, convolution_20, relu_12, mean_2, relu_13, div_3, mul_54, convolution_23, add_43, convolution_24, add_45, div_4, convolution_25, add_48, div_5, convolution_26, add_51, convolution_27, add_53, div_6, convolution_28, add_56, div_7, convolution_29, add_60, convolution_30, add_62, div_8, convolution_31, add_65, div_9, convolution_32, add_69, convolution_33, add_71, div_10, convolution_34, add_74, div_11, convolution_35, add_78, convolution_36, add_80, div_12, convolution_37, add_83, div_13, mean_3, relu_14, div_14, mul_110, convolution_40, add_87, convolution_41, add_89, div_15, convolution_42, add_92, div_16, mean_4, relu_15, div_17, mul_122, convolution_45, add_97, convolution_46, add_99, div_18, convolution_47, add_102, div_19, mean_5, relu_16, div_20, mul_134, convolution_50, add_106, convolution_51, add_108, div_21, convolution_52, add_111, div_22, mean_6, relu_17, div_23, mul_146, convolution_55, add_116, convolution_56, add_118, div_24, convolution_57, add_121, div_25, mean_7, relu_18, div_26, mul_158, convolution_60, add_126, convolution_61, add_128, view, addmm, div_28, permute_2, permute_6, bitwise_and, bitwise_and_1, bitwise_and_2, bitwise_and_3, bitwise_and_4, bitwise_and_5, bitwise_and_6, bitwise_and_7]
    