from __future__ import annotations



def forward(self, primals_1: "f32[32]", primals_2: "f32[32]", primals_3: "f32[224]", primals_4: "f32[224]", primals_5: "f32[224]", primals_6: "f32[224]", primals_7: "f32[224]", primals_8: "f32[224]", primals_9: "f32[224]", primals_10: "f32[224]", primals_11: "f32[224]", primals_12: "f32[224]", primals_13: "f32[224]", primals_14: "f32[224]", primals_15: "f32[224]", primals_16: "f32[224]", primals_17: "f32[448]", primals_18: "f32[448]", primals_19: "f32[448]", primals_20: "f32[448]", primals_21: "f32[448]", primals_22: "f32[448]", primals_23: "f32[448]", primals_24: "f32[448]", primals_25: "f32[448]", primals_26: "f32[448]", primals_27: "f32[448]", primals_28: "f32[448]", primals_29: "f32[448]", primals_30: "f32[448]", primals_31: "f32[448]", primals_32: "f32[448]", primals_33: "f32[448]", primals_34: "f32[448]", primals_35: "f32[448]", primals_36: "f32[448]", primals_37: "f32[448]", primals_38: "f32[448]", primals_39: "f32[448]", primals_40: "f32[448]", primals_41: "f32[448]", primals_42: "f32[448]", primals_43: "f32[448]", primals_44: "f32[448]", primals_45: "f32[448]", primals_46: "f32[448]", primals_47: "f32[448]", primals_48: "f32[448]", primals_49: "f32[896]", primals_50: "f32[896]", primals_51: "f32[896]", primals_52: "f32[896]", primals_53: "f32[896]", primals_54: "f32[896]", primals_55: "f32[896]", primals_56: "f32[896]", primals_57: "f32[896]", primals_58: "f32[896]", primals_59: "f32[896]", primals_60: "f32[896]", primals_61: "f32[896]", primals_62: "f32[896]", primals_63: "f32[896]", primals_64: "f32[896]", primals_65: "f32[896]", primals_66: "f32[896]", primals_67: "f32[896]", primals_68: "f32[896]", primals_69: "f32[896]", primals_70: "f32[896]", primals_71: "f32[896]", primals_72: "f32[896]", primals_73: "f32[896]", primals_74: "f32[896]", primals_75: "f32[896]", primals_76: "f32[896]", primals_77: "f32[896]", primals_78: "f32[896]", primals_79: "f32[896]", primals_80: "f32[896]", primals_81: "f32[896]", primals_82: "f32[896]", primals_83: "f32[896]", primals_84: "f32[896]", primals_85: "f32[896]", primals_86: "f32[896]", primals_87: "f32[896]", primals_88: "f32[896]", primals_89: "f32[896]", primals_90: "f32[896]", primals_91: "f32[896]", primals_92: "f32[896]", primals_93: "f32[896]", primals_94: "f32[896]", primals_95: "f32[896]", primals_96: "f32[896]", primals_97: "f32[896]", primals_98: "f32[896]", primals_99: "f32[896]", primals_100: "f32[896]", primals_101: "f32[896]", primals_102: "f32[896]", primals_103: "f32[896]", primals_104: "f32[896]", primals_105: "f32[896]", primals_106: "f32[896]", primals_107: "f32[896]", primals_108: "f32[896]", primals_109: "f32[896]", primals_110: "f32[896]", primals_111: "f32[896]", primals_112: "f32[896]", primals_113: "f32[896]", primals_114: "f32[896]", primals_115: "f32[896]", primals_116: "f32[896]", primals_117: "f32[2240]", primals_118: "f32[2240]", primals_119: "f32[2240]", primals_120: "f32[2240]", primals_121: "f32[2240]", primals_122: "f32[2240]", primals_123: "f32[2240]", primals_124: "f32[2240]", primals_125: "f32[32, 3, 3, 3]", primals_126: "f32[224, 32, 1, 1]", primals_127: "f32[224, 112, 3, 3]", primals_128: "f32[8, 224, 1, 1]", primals_129: "f32[8]", primals_130: "f32[224, 8, 1, 1]", primals_131: "f32[224]", primals_132: "f32[224, 224, 1, 1]", primals_133: "f32[224, 32, 1, 1]", primals_134: "f32[224, 224, 1, 1]", primals_135: "f32[224, 112, 3, 3]", primals_136: "f32[56, 224, 1, 1]", primals_137: "f32[56]", primals_138: "f32[224, 56, 1, 1]", primals_139: "f32[224]", primals_140: "f32[224, 224, 1, 1]", primals_141: "f32[448, 224, 1, 1]", primals_142: "f32[448, 112, 3, 3]", primals_143: "f32[56, 448, 1, 1]", primals_144: "f32[56]", primals_145: "f32[448, 56, 1, 1]", primals_146: "f32[448]", primals_147: "f32[448, 448, 1, 1]", primals_148: "f32[448, 224, 1, 1]", primals_149: "f32[448, 448, 1, 1]", primals_150: "f32[448, 112, 3, 3]", primals_151: "f32[112, 448, 1, 1]", primals_152: "f32[112]", primals_153: "f32[448, 112, 1, 1]", primals_154: "f32[448]", primals_155: "f32[448, 448, 1, 1]", primals_156: "f32[448, 448, 1, 1]", primals_157: "f32[448, 112, 3, 3]", primals_158: "f32[112, 448, 1, 1]", primals_159: "f32[112]", primals_160: "f32[448, 112, 1, 1]", primals_161: "f32[448]", primals_162: "f32[448, 448, 1, 1]", primals_163: "f32[448, 448, 1, 1]", primals_164: "f32[448, 112, 3, 3]", primals_165: "f32[112, 448, 1, 1]", primals_166: "f32[112]", primals_167: "f32[448, 112, 1, 1]", primals_168: "f32[448]", primals_169: "f32[448, 448, 1, 1]", primals_170: "f32[448, 448, 1, 1]", primals_171: "f32[448, 112, 3, 3]", primals_172: "f32[112, 448, 1, 1]", primals_173: "f32[112]", primals_174: "f32[448, 112, 1, 1]", primals_175: "f32[448]", primals_176: "f32[448, 448, 1, 1]", primals_177: "f32[896, 448, 1, 1]", primals_178: "f32[896, 112, 3, 3]", primals_179: "f32[112, 896, 1, 1]", primals_180: "f32[112]", primals_181: "f32[896, 112, 1, 1]", primals_182: "f32[896]", primals_183: "f32[896, 896, 1, 1]", primals_184: "f32[896, 448, 1, 1]", primals_185: "f32[896, 896, 1, 1]", primals_186: "f32[896, 112, 3, 3]", primals_187: "f32[224, 896, 1, 1]", primals_188: "f32[224]", primals_189: "f32[896, 224, 1, 1]", primals_190: "f32[896]", primals_191: "f32[896, 896, 1, 1]", primals_192: "f32[896, 896, 1, 1]", primals_193: "f32[896, 112, 3, 3]", primals_194: "f32[224, 896, 1, 1]", primals_195: "f32[224]", primals_196: "f32[896, 224, 1, 1]", primals_197: "f32[896]", primals_198: "f32[896, 896, 1, 1]", primals_199: "f32[896, 896, 1, 1]", primals_200: "f32[896, 112, 3, 3]", primals_201: "f32[224, 896, 1, 1]", primals_202: "f32[224]", primals_203: "f32[896, 224, 1, 1]", primals_204: "f32[896]", primals_205: "f32[896, 896, 1, 1]", primals_206: "f32[896, 896, 1, 1]", primals_207: "f32[896, 112, 3, 3]", primals_208: "f32[224, 896, 1, 1]", primals_209: "f32[224]", primals_210: "f32[896, 224, 1, 1]", primals_211: "f32[896]", primals_212: "f32[896, 896, 1, 1]", primals_213: "f32[896, 896, 1, 1]", primals_214: "f32[896, 112, 3, 3]", primals_215: "f32[224, 896, 1, 1]", primals_216: "f32[224]", primals_217: "f32[896, 224, 1, 1]", primals_218: "f32[896]", primals_219: "f32[896, 896, 1, 1]", primals_220: "f32[896, 896, 1, 1]", primals_221: "f32[896, 112, 3, 3]", primals_222: "f32[224, 896, 1, 1]", primals_223: "f32[224]", primals_224: "f32[896, 224, 1, 1]", primals_225: "f32[896]", primals_226: "f32[896, 896, 1, 1]", primals_227: "f32[896, 896, 1, 1]", primals_228: "f32[896, 112, 3, 3]", primals_229: "f32[224, 896, 1, 1]", primals_230: "f32[224]", primals_231: "f32[896, 224, 1, 1]", primals_232: "f32[896]", primals_233: "f32[896, 896, 1, 1]", primals_234: "f32[896, 896, 1, 1]", primals_235: "f32[896, 112, 3, 3]", primals_236: "f32[224, 896, 1, 1]", primals_237: "f32[224]", primals_238: "f32[896, 224, 1, 1]", primals_239: "f32[896]", primals_240: "f32[896, 896, 1, 1]", primals_241: "f32[896, 896, 1, 1]", primals_242: "f32[896, 112, 3, 3]", primals_243: "f32[224, 896, 1, 1]", primals_244: "f32[224]", primals_245: "f32[896, 224, 1, 1]", primals_246: "f32[896]", primals_247: "f32[896, 896, 1, 1]", primals_248: "f32[896, 896, 1, 1]", primals_249: "f32[896, 112, 3, 3]", primals_250: "f32[224, 896, 1, 1]", primals_251: "f32[224]", primals_252: "f32[896, 224, 1, 1]", primals_253: "f32[896]", primals_254: "f32[896, 896, 1, 1]", primals_255: "f32[2240, 896, 1, 1]", primals_256: "f32[2240, 112, 3, 3]", primals_257: "f32[224, 2240, 1, 1]", primals_258: "f32[224]", primals_259: "f32[2240, 224, 1, 1]", primals_260: "f32[2240]", primals_261: "f32[2240, 2240, 1, 1]", primals_262: "f32[2240, 896, 1, 1]", primals_263: "f32[1000, 2240]", primals_264: "f32[1000]", primals_265: "f32[32]", primals_266: "f32[32]", primals_267: "f32[224]", primals_268: "f32[224]", primals_269: "f32[224]", primals_270: "f32[224]", primals_271: "f32[224]", primals_272: "f32[224]", primals_273: "f32[224]", primals_274: "f32[224]", primals_275: "f32[224]", primals_276: "f32[224]", primals_277: "f32[224]", primals_278: "f32[224]", primals_279: "f32[224]", primals_280: "f32[224]", primals_281: "f32[448]", primals_282: "f32[448]", primals_283: "f32[448]", primals_284: "f32[448]", primals_285: "f32[448]", primals_286: "f32[448]", primals_287: "f32[448]", primals_288: "f32[448]", primals_289: "f32[448]", primals_290: "f32[448]", primals_291: "f32[448]", primals_292: "f32[448]", primals_293: "f32[448]", primals_294: "f32[448]", primals_295: "f32[448]", primals_296: "f32[448]", primals_297: "f32[448]", primals_298: "f32[448]", primals_299: "f32[448]", primals_300: "f32[448]", primals_301: "f32[448]", primals_302: "f32[448]", primals_303: "f32[448]", primals_304: "f32[448]", primals_305: "f32[448]", primals_306: "f32[448]", primals_307: "f32[448]", primals_308: "f32[448]", primals_309: "f32[448]", primals_310: "f32[448]", primals_311: "f32[448]", primals_312: "f32[448]", primals_313: "f32[896]", primals_314: "f32[896]", primals_315: "f32[896]", primals_316: "f32[896]", primals_317: "f32[896]", primals_318: "f32[896]", primals_319: "f32[896]", primals_320: "f32[896]", primals_321: "f32[896]", primals_322: "f32[896]", primals_323: "f32[896]", primals_324: "f32[896]", primals_325: "f32[896]", primals_326: "f32[896]", primals_327: "f32[896]", primals_328: "f32[896]", primals_329: "f32[896]", primals_330: "f32[896]", primals_331: "f32[896]", primals_332: "f32[896]", primals_333: "f32[896]", primals_334: "f32[896]", primals_335: "f32[896]", primals_336: "f32[896]", primals_337: "f32[896]", primals_338: "f32[896]", primals_339: "f32[896]", primals_340: "f32[896]", primals_341: "f32[896]", primals_342: "f32[896]", primals_343: "f32[896]", primals_344: "f32[896]", primals_345: "f32[896]", primals_346: "f32[896]", primals_347: "f32[896]", primals_348: "f32[896]", primals_349: "f32[896]", primals_350: "f32[896]", primals_351: "f32[896]", primals_352: "f32[896]", primals_353: "f32[896]", primals_354: "f32[896]", primals_355: "f32[896]", primals_356: "f32[896]", primals_357: "f32[896]", primals_358: "f32[896]", primals_359: "f32[896]", primals_360: "f32[896]", primals_361: "f32[896]", primals_362: "f32[896]", primals_363: "f32[896]", primals_364: "f32[896]", primals_365: "f32[896]", primals_366: "f32[896]", primals_367: "f32[896]", primals_368: "f32[896]", primals_369: "f32[896]", primals_370: "f32[896]", primals_371: "f32[896]", primals_372: "f32[896]", primals_373: "f32[896]", primals_374: "f32[896]", primals_375: "f32[896]", primals_376: "f32[896]", primals_377: "f32[896]", primals_378: "f32[896]", primals_379: "f32[896]", primals_380: "f32[896]", primals_381: "f32[2240]", primals_382: "f32[2240]", primals_383: "f32[2240]", primals_384: "f32[2240]", primals_385: "f32[2240]", primals_386: "f32[2240]", primals_387: "f32[2240]", primals_388: "f32[2240]", primals_389: "f32[4, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution: "f32[4, 32, 112, 112]" = torch.ops.aten.convolution.default(primals_389, primals_125, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add: "f32[32]" = torch.ops.aten.add.Tensor(primals_266, 1e-05)
    sqrt: "f32[32]" = torch.ops.aten.sqrt.default(add);  add = None
    reciprocal: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
    mul: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
    unsqueeze: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_265, -1)
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
    relu: "f32[4, 32, 112, 112]" = torch.ops.aten.relu.default(add_1);  add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_1: "f32[4, 224, 112, 112]" = torch.ops.aten.convolution.default(relu, primals_126, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_2: "f32[224]" = torch.ops.aten.add.Tensor(primals_268, 1e-05)
    sqrt_1: "f32[224]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
    reciprocal_1: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_3: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_8: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_267, -1)
    unsqueeze_9: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    unsqueeze_10: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
    unsqueeze_11: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    sub_1: "f32[4, 224, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  unsqueeze_9 = None
    mul_4: "f32[4, 224, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1)
    unsqueeze_13: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_5: "f32[4, 224, 112, 112]" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
    unsqueeze_14: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_4, -1);  primals_4 = None
    unsqueeze_15: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_3: "f32[4, 224, 112, 112]" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_1: "f32[4, 224, 112, 112]" = torch.ops.aten.relu.default(add_3);  add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_2: "f32[4, 224, 56, 56]" = torch.ops.aten.convolution.default(relu_1, primals_127, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_4: "f32[224]" = torch.ops.aten.add.Tensor(primals_270, 1e-05)
    sqrt_2: "f32[224]" = torch.ops.aten.sqrt.default(add_4);  add_4 = None
    reciprocal_2: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_6: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_16: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_269, -1)
    unsqueeze_17: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    unsqueeze_18: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
    unsqueeze_19: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    sub_2: "f32[4, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  unsqueeze_17 = None
    mul_7: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_21: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_8: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_21);  mul_7 = unsqueeze_21 = None
    unsqueeze_22: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_23: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_5: "f32[4, 224, 56, 56]" = torch.ops.aten.add.Tensor(mul_8, unsqueeze_23);  mul_8 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_2: "f32[4, 224, 56, 56]" = torch.ops.aten.relu.default(add_5);  add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean: "f32[4, 224, 1, 1]" = torch.ops.aten.mean.dim(relu_2, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_3: "f32[4, 8, 1, 1]" = torch.ops.aten.convolution.default(mean, primals_128, primals_129, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_3: "f32[4, 8, 1, 1]" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_4: "f32[4, 224, 1, 1]" = torch.ops.aten.convolution.default(relu_3, primals_130, primals_131, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid: "f32[4, 224, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_9: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(relu_2, sigmoid);  sigmoid = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_5: "f32[4, 224, 56, 56]" = torch.ops.aten.convolution.default(mul_9, primals_132, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_6: "f32[224]" = torch.ops.aten.add.Tensor(primals_272, 1e-05)
    sqrt_3: "f32[224]" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
    reciprocal_3: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
    mul_10: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
    unsqueeze_24: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_271, -1)
    unsqueeze_25: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    unsqueeze_26: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_10, -1);  mul_10 = None
    unsqueeze_27: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    sub_3: "f32[4, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_25);  unsqueeze_25 = None
    mul_11: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
    unsqueeze_28: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1)
    unsqueeze_29: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_12: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(mul_11, unsqueeze_29);  mul_11 = unsqueeze_29 = None
    unsqueeze_30: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1);  primals_8 = None
    unsqueeze_31: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_7: "f32[4, 224, 56, 56]" = torch.ops.aten.add.Tensor(mul_12, unsqueeze_31);  mul_12 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_6: "f32[4, 224, 56, 56]" = torch.ops.aten.convolution.default(relu, primals_133, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_8: "f32[224]" = torch.ops.aten.add.Tensor(primals_274, 1e-05)
    sqrt_4: "f32[224]" = torch.ops.aten.sqrt.default(add_8);  add_8 = None
    reciprocal_4: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
    mul_13: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
    unsqueeze_32: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_273, -1)
    unsqueeze_33: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    unsqueeze_34: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_13, -1);  mul_13 = None
    unsqueeze_35: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    sub_4: "f32[4, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_33);  unsqueeze_33 = None
    mul_14: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
    unsqueeze_36: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1)
    unsqueeze_37: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_15: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_37);  mul_14 = unsqueeze_37 = None
    unsqueeze_38: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_10, -1);  primals_10 = None
    unsqueeze_39: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_9: "f32[4, 224, 56, 56]" = torch.ops.aten.add.Tensor(mul_15, unsqueeze_39);  mul_15 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_10: "f32[4, 224, 56, 56]" = torch.ops.aten.add.Tensor(add_7, add_9);  add_7 = add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_4: "f32[4, 224, 56, 56]" = torch.ops.aten.relu.default(add_10);  add_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_7: "f32[4, 224, 56, 56]" = torch.ops.aten.convolution.default(relu_4, primals_134, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_11: "f32[224]" = torch.ops.aten.add.Tensor(primals_276, 1e-05)
    sqrt_5: "f32[224]" = torch.ops.aten.sqrt.default(add_11);  add_11 = None
    reciprocal_5: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
    mul_16: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
    unsqueeze_40: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_275, -1)
    unsqueeze_41: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    unsqueeze_42: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_16, -1);  mul_16 = None
    unsqueeze_43: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    sub_5: "f32[4, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_41);  unsqueeze_41 = None
    mul_17: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
    unsqueeze_44: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_45: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_18: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(mul_17, unsqueeze_45);  mul_17 = unsqueeze_45 = None
    unsqueeze_46: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_47: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_12: "f32[4, 224, 56, 56]" = torch.ops.aten.add.Tensor(mul_18, unsqueeze_47);  mul_18 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_5: "f32[4, 224, 56, 56]" = torch.ops.aten.relu.default(add_12);  add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_8: "f32[4, 224, 56, 56]" = torch.ops.aten.convolution.default(relu_5, primals_135, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_13: "f32[224]" = torch.ops.aten.add.Tensor(primals_278, 1e-05)
    sqrt_6: "f32[224]" = torch.ops.aten.sqrt.default(add_13);  add_13 = None
    reciprocal_6: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
    mul_19: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
    unsqueeze_48: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_277, -1)
    unsqueeze_49: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    unsqueeze_50: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_19, -1);  mul_19 = None
    unsqueeze_51: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    sub_6: "f32[4, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_49);  unsqueeze_49 = None
    mul_20: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
    unsqueeze_52: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1)
    unsqueeze_53: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_21: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(mul_20, unsqueeze_53);  mul_20 = unsqueeze_53 = None
    unsqueeze_54: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1);  primals_14 = None
    unsqueeze_55: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_14: "f32[4, 224, 56, 56]" = torch.ops.aten.add.Tensor(mul_21, unsqueeze_55);  mul_21 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_6: "f32[4, 224, 56, 56]" = torch.ops.aten.relu.default(add_14);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_1: "f32[4, 224, 1, 1]" = torch.ops.aten.mean.dim(relu_6, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_9: "f32[4, 56, 1, 1]" = torch.ops.aten.convolution.default(mean_1, primals_136, primals_137, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_7: "f32[4, 56, 1, 1]" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_10: "f32[4, 224, 1, 1]" = torch.ops.aten.convolution.default(relu_7, primals_138, primals_139, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_1: "f32[4, 224, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_22: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(relu_6, sigmoid_1);  sigmoid_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_11: "f32[4, 224, 56, 56]" = torch.ops.aten.convolution.default(mul_22, primals_140, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_15: "f32[224]" = torch.ops.aten.add.Tensor(primals_280, 1e-05)
    sqrt_7: "f32[224]" = torch.ops.aten.sqrt.default(add_15);  add_15 = None
    reciprocal_7: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
    mul_23: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
    unsqueeze_56: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_279, -1)
    unsqueeze_57: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    unsqueeze_58: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_23, -1);  mul_23 = None
    unsqueeze_59: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    sub_7: "f32[4, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_57);  unsqueeze_57 = None
    mul_24: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
    unsqueeze_60: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1)
    unsqueeze_61: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_25: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(mul_24, unsqueeze_61);  mul_24 = unsqueeze_61 = None
    unsqueeze_62: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1);  primals_16 = None
    unsqueeze_63: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_16: "f32[4, 224, 56, 56]" = torch.ops.aten.add.Tensor(mul_25, unsqueeze_63);  mul_25 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_17: "f32[4, 224, 56, 56]" = torch.ops.aten.add.Tensor(add_16, relu_4);  add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_8: "f32[4, 224, 56, 56]" = torch.ops.aten.relu.default(add_17);  add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_12: "f32[4, 448, 56, 56]" = torch.ops.aten.convolution.default(relu_8, primals_141, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_18: "f32[448]" = torch.ops.aten.add.Tensor(primals_282, 1e-05)
    sqrt_8: "f32[448]" = torch.ops.aten.sqrt.default(add_18);  add_18 = None
    reciprocal_8: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
    mul_26: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
    unsqueeze_64: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_281, -1)
    unsqueeze_65: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    unsqueeze_66: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_26, -1);  mul_26 = None
    unsqueeze_67: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    sub_8: "f32[4, 448, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_65);  unsqueeze_65 = None
    mul_27: "f32[4, 448, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
    unsqueeze_68: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_69: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_28: "f32[4, 448, 56, 56]" = torch.ops.aten.mul.Tensor(mul_27, unsqueeze_69);  mul_27 = unsqueeze_69 = None
    unsqueeze_70: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_71: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_19: "f32[4, 448, 56, 56]" = torch.ops.aten.add.Tensor(mul_28, unsqueeze_71);  mul_28 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_9: "f32[4, 448, 56, 56]" = torch.ops.aten.relu.default(add_19);  add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_13: "f32[4, 448, 28, 28]" = torch.ops.aten.convolution.default(relu_9, primals_142, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_20: "f32[448]" = torch.ops.aten.add.Tensor(primals_284, 1e-05)
    sqrt_9: "f32[448]" = torch.ops.aten.sqrt.default(add_20);  add_20 = None
    reciprocal_9: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
    mul_29: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
    unsqueeze_72: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_283, -1)
    unsqueeze_73: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    unsqueeze_74: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_29, -1);  mul_29 = None
    unsqueeze_75: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    sub_9: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_73);  unsqueeze_73 = None
    mul_30: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
    unsqueeze_76: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
    unsqueeze_77: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_31: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_30, unsqueeze_77);  mul_30 = unsqueeze_77 = None
    unsqueeze_78: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
    unsqueeze_79: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_21: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_31, unsqueeze_79);  mul_31 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_10: "f32[4, 448, 28, 28]" = torch.ops.aten.relu.default(add_21);  add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_2: "f32[4, 448, 1, 1]" = torch.ops.aten.mean.dim(relu_10, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_14: "f32[4, 56, 1, 1]" = torch.ops.aten.convolution.default(mean_2, primals_143, primals_144, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_11: "f32[4, 56, 1, 1]" = torch.ops.aten.relu.default(convolution_14);  convolution_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_15: "f32[4, 448, 1, 1]" = torch.ops.aten.convolution.default(relu_11, primals_145, primals_146, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_2: "f32[4, 448, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_32: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(relu_10, sigmoid_2);  sigmoid_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_16: "f32[4, 448, 28, 28]" = torch.ops.aten.convolution.default(mul_32, primals_147, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_22: "f32[448]" = torch.ops.aten.add.Tensor(primals_286, 1e-05)
    sqrt_10: "f32[448]" = torch.ops.aten.sqrt.default(add_22);  add_22 = None
    reciprocal_10: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
    mul_33: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
    unsqueeze_80: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_285, -1)
    unsqueeze_81: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    unsqueeze_82: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_33, -1);  mul_33 = None
    unsqueeze_83: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    sub_10: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_81);  unsqueeze_81 = None
    mul_34: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
    unsqueeze_84: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1)
    unsqueeze_85: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_35: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_34, unsqueeze_85);  mul_34 = unsqueeze_85 = None
    unsqueeze_86: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1);  primals_22 = None
    unsqueeze_87: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_23: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_87);  mul_35 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_17: "f32[4, 448, 28, 28]" = torch.ops.aten.convolution.default(relu_8, primals_148, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_24: "f32[448]" = torch.ops.aten.add.Tensor(primals_288, 1e-05)
    sqrt_11: "f32[448]" = torch.ops.aten.sqrt.default(add_24);  add_24 = None
    reciprocal_11: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
    mul_36: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
    unsqueeze_88: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_287, -1)
    unsqueeze_89: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    unsqueeze_90: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_36, -1);  mul_36 = None
    unsqueeze_91: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    sub_11: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_89);  unsqueeze_89 = None
    mul_37: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
    unsqueeze_92: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_93: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_38: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_93);  mul_37 = unsqueeze_93 = None
    unsqueeze_94: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_95: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_25: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_38, unsqueeze_95);  mul_38 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_26: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(add_23, add_25);  add_23 = add_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_12: "f32[4, 448, 28, 28]" = torch.ops.aten.relu.default(add_26);  add_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_18: "f32[4, 448, 28, 28]" = torch.ops.aten.convolution.default(relu_12, primals_149, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_27: "f32[448]" = torch.ops.aten.add.Tensor(primals_290, 1e-05)
    sqrt_12: "f32[448]" = torch.ops.aten.sqrt.default(add_27);  add_27 = None
    reciprocal_12: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
    mul_39: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
    unsqueeze_96: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_289, -1)
    unsqueeze_97: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    unsqueeze_98: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_39, -1);  mul_39 = None
    unsqueeze_99: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    sub_12: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_97);  unsqueeze_97 = None
    mul_40: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
    unsqueeze_100: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1)
    unsqueeze_101: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_41: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_40, unsqueeze_101);  mul_40 = unsqueeze_101 = None
    unsqueeze_102: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1);  primals_26 = None
    unsqueeze_103: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_28: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_103);  mul_41 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_13: "f32[4, 448, 28, 28]" = torch.ops.aten.relu.default(add_28);  add_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_19: "f32[4, 448, 28, 28]" = torch.ops.aten.convolution.default(relu_13, primals_150, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_29: "f32[448]" = torch.ops.aten.add.Tensor(primals_292, 1e-05)
    sqrt_13: "f32[448]" = torch.ops.aten.sqrt.default(add_29);  add_29 = None
    reciprocal_13: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
    mul_42: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
    unsqueeze_104: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_291, -1)
    unsqueeze_105: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    unsqueeze_106: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_42, -1);  mul_42 = None
    unsqueeze_107: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    sub_13: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_105);  unsqueeze_105 = None
    mul_43: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
    unsqueeze_108: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1)
    unsqueeze_109: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_44: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_109);  mul_43 = unsqueeze_109 = None
    unsqueeze_110: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1);  primals_28 = None
    unsqueeze_111: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_30: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_44, unsqueeze_111);  mul_44 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_14: "f32[4, 448, 28, 28]" = torch.ops.aten.relu.default(add_30);  add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_3: "f32[4, 448, 1, 1]" = torch.ops.aten.mean.dim(relu_14, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_20: "f32[4, 112, 1, 1]" = torch.ops.aten.convolution.default(mean_3, primals_151, primals_152, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_15: "f32[4, 112, 1, 1]" = torch.ops.aten.relu.default(convolution_20);  convolution_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_21: "f32[4, 448, 1, 1]" = torch.ops.aten.convolution.default(relu_15, primals_153, primals_154, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_3: "f32[4, 448, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_45: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(relu_14, sigmoid_3);  sigmoid_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_22: "f32[4, 448, 28, 28]" = torch.ops.aten.convolution.default(mul_45, primals_155, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_31: "f32[448]" = torch.ops.aten.add.Tensor(primals_294, 1e-05)
    sqrt_14: "f32[448]" = torch.ops.aten.sqrt.default(add_31);  add_31 = None
    reciprocal_14: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
    mul_46: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
    unsqueeze_112: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_293, -1)
    unsqueeze_113: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    unsqueeze_114: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_46, -1);  mul_46 = None
    unsqueeze_115: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    sub_14: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_113);  unsqueeze_113 = None
    mul_47: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
    unsqueeze_116: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_117: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_48: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_47, unsqueeze_117);  mul_47 = unsqueeze_117 = None
    unsqueeze_118: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_119: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_32: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_119);  mul_48 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_33: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(add_32, relu_12);  add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_16: "f32[4, 448, 28, 28]" = torch.ops.aten.relu.default(add_33);  add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_23: "f32[4, 448, 28, 28]" = torch.ops.aten.convolution.default(relu_16, primals_156, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_34: "f32[448]" = torch.ops.aten.add.Tensor(primals_296, 1e-05)
    sqrt_15: "f32[448]" = torch.ops.aten.sqrt.default(add_34);  add_34 = None
    reciprocal_15: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
    mul_49: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
    unsqueeze_120: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_295, -1)
    unsqueeze_121: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    unsqueeze_122: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_49, -1);  mul_49 = None
    unsqueeze_123: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    sub_15: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_121);  unsqueeze_121 = None
    mul_50: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
    unsqueeze_124: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1)
    unsqueeze_125: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_51: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_50, unsqueeze_125);  mul_50 = unsqueeze_125 = None
    unsqueeze_126: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1);  primals_32 = None
    unsqueeze_127: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_35: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_51, unsqueeze_127);  mul_51 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_17: "f32[4, 448, 28, 28]" = torch.ops.aten.relu.default(add_35);  add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_24: "f32[4, 448, 28, 28]" = torch.ops.aten.convolution.default(relu_17, primals_157, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_36: "f32[448]" = torch.ops.aten.add.Tensor(primals_298, 1e-05)
    sqrt_16: "f32[448]" = torch.ops.aten.sqrt.default(add_36);  add_36 = None
    reciprocal_16: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
    mul_52: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
    unsqueeze_128: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_297, -1)
    unsqueeze_129: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    unsqueeze_130: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_52, -1);  mul_52 = None
    unsqueeze_131: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    sub_16: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_129);  unsqueeze_129 = None
    mul_53: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
    unsqueeze_132: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1)
    unsqueeze_133: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_54: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_53, unsqueeze_133);  mul_53 = unsqueeze_133 = None
    unsqueeze_134: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1);  primals_34 = None
    unsqueeze_135: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_37: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_54, unsqueeze_135);  mul_54 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_18: "f32[4, 448, 28, 28]" = torch.ops.aten.relu.default(add_37);  add_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_4: "f32[4, 448, 1, 1]" = torch.ops.aten.mean.dim(relu_18, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_25: "f32[4, 112, 1, 1]" = torch.ops.aten.convolution.default(mean_4, primals_158, primals_159, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_19: "f32[4, 112, 1, 1]" = torch.ops.aten.relu.default(convolution_25);  convolution_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_26: "f32[4, 448, 1, 1]" = torch.ops.aten.convolution.default(relu_19, primals_160, primals_161, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_4: "f32[4, 448, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_55: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(relu_18, sigmoid_4);  sigmoid_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_27: "f32[4, 448, 28, 28]" = torch.ops.aten.convolution.default(mul_55, primals_162, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_38: "f32[448]" = torch.ops.aten.add.Tensor(primals_300, 1e-05)
    sqrt_17: "f32[448]" = torch.ops.aten.sqrt.default(add_38);  add_38 = None
    reciprocal_17: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
    mul_56: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
    unsqueeze_136: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_299, -1)
    unsqueeze_137: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    unsqueeze_138: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_56, -1);  mul_56 = None
    unsqueeze_139: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    sub_17: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_137);  unsqueeze_137 = None
    mul_57: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_139);  sub_17 = unsqueeze_139 = None
    unsqueeze_140: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_141: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_58: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_57, unsqueeze_141);  mul_57 = unsqueeze_141 = None
    unsqueeze_142: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_143: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_39: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_58, unsqueeze_143);  mul_58 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_40: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(add_39, relu_16);  add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_20: "f32[4, 448, 28, 28]" = torch.ops.aten.relu.default(add_40);  add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_28: "f32[4, 448, 28, 28]" = torch.ops.aten.convolution.default(relu_20, primals_163, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_41: "f32[448]" = torch.ops.aten.add.Tensor(primals_302, 1e-05)
    sqrt_18: "f32[448]" = torch.ops.aten.sqrt.default(add_41);  add_41 = None
    reciprocal_18: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
    mul_59: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
    unsqueeze_144: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_301, -1)
    unsqueeze_145: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    unsqueeze_146: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_59, -1);  mul_59 = None
    unsqueeze_147: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    sub_18: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_145);  unsqueeze_145 = None
    mul_60: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_147);  sub_18 = unsqueeze_147 = None
    unsqueeze_148: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1)
    unsqueeze_149: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_61: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_60, unsqueeze_149);  mul_60 = unsqueeze_149 = None
    unsqueeze_150: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1);  primals_38 = None
    unsqueeze_151: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_42: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_61, unsqueeze_151);  mul_61 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_21: "f32[4, 448, 28, 28]" = torch.ops.aten.relu.default(add_42);  add_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_29: "f32[4, 448, 28, 28]" = torch.ops.aten.convolution.default(relu_21, primals_164, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_43: "f32[448]" = torch.ops.aten.add.Tensor(primals_304, 1e-05)
    sqrt_19: "f32[448]" = torch.ops.aten.sqrt.default(add_43);  add_43 = None
    reciprocal_19: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
    mul_62: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
    unsqueeze_152: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_303, -1)
    unsqueeze_153: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    unsqueeze_154: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_62, -1);  mul_62 = None
    unsqueeze_155: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    sub_19: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_153);  unsqueeze_153 = None
    mul_63: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_155);  sub_19 = unsqueeze_155 = None
    unsqueeze_156: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1)
    unsqueeze_157: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_64: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_157);  mul_63 = unsqueeze_157 = None
    unsqueeze_158: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1);  primals_40 = None
    unsqueeze_159: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_44: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_64, unsqueeze_159);  mul_64 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_22: "f32[4, 448, 28, 28]" = torch.ops.aten.relu.default(add_44);  add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_5: "f32[4, 448, 1, 1]" = torch.ops.aten.mean.dim(relu_22, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_30: "f32[4, 112, 1, 1]" = torch.ops.aten.convolution.default(mean_5, primals_165, primals_166, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_23: "f32[4, 112, 1, 1]" = torch.ops.aten.relu.default(convolution_30);  convolution_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_31: "f32[4, 448, 1, 1]" = torch.ops.aten.convolution.default(relu_23, primals_167, primals_168, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_5: "f32[4, 448, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_65: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(relu_22, sigmoid_5);  sigmoid_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_32: "f32[4, 448, 28, 28]" = torch.ops.aten.convolution.default(mul_65, primals_169, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_45: "f32[448]" = torch.ops.aten.add.Tensor(primals_306, 1e-05)
    sqrt_20: "f32[448]" = torch.ops.aten.sqrt.default(add_45);  add_45 = None
    reciprocal_20: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
    mul_66: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
    unsqueeze_160: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_305, -1)
    unsqueeze_161: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    unsqueeze_162: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_66, -1);  mul_66 = None
    unsqueeze_163: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    sub_20: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_161);  unsqueeze_161 = None
    mul_67: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_163);  sub_20 = unsqueeze_163 = None
    unsqueeze_164: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_165: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_68: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_67, unsqueeze_165);  mul_67 = unsqueeze_165 = None
    unsqueeze_166: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_167: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_46: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_68, unsqueeze_167);  mul_68 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_47: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(add_46, relu_20);  add_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_24: "f32[4, 448, 28, 28]" = torch.ops.aten.relu.default(add_47);  add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_33: "f32[4, 448, 28, 28]" = torch.ops.aten.convolution.default(relu_24, primals_170, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_48: "f32[448]" = torch.ops.aten.add.Tensor(primals_308, 1e-05)
    sqrt_21: "f32[448]" = torch.ops.aten.sqrt.default(add_48);  add_48 = None
    reciprocal_21: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
    mul_69: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
    unsqueeze_168: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_307, -1)
    unsqueeze_169: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    unsqueeze_170: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_69, -1);  mul_69 = None
    unsqueeze_171: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    sub_21: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_169);  unsqueeze_169 = None
    mul_70: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_21, unsqueeze_171);  sub_21 = unsqueeze_171 = None
    unsqueeze_172: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1)
    unsqueeze_173: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_71: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_173);  mul_70 = unsqueeze_173 = None
    unsqueeze_174: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1);  primals_44 = None
    unsqueeze_175: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_49: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_71, unsqueeze_175);  mul_71 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_25: "f32[4, 448, 28, 28]" = torch.ops.aten.relu.default(add_49);  add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_34: "f32[4, 448, 28, 28]" = torch.ops.aten.convolution.default(relu_25, primals_171, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_50: "f32[448]" = torch.ops.aten.add.Tensor(primals_310, 1e-05)
    sqrt_22: "f32[448]" = torch.ops.aten.sqrt.default(add_50);  add_50 = None
    reciprocal_22: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
    mul_72: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
    unsqueeze_176: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_309, -1)
    unsqueeze_177: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    unsqueeze_178: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_72, -1);  mul_72 = None
    unsqueeze_179: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    sub_22: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_177);  unsqueeze_177 = None
    mul_73: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_179);  sub_22 = unsqueeze_179 = None
    unsqueeze_180: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1)
    unsqueeze_181: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_74: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_73, unsqueeze_181);  mul_73 = unsqueeze_181 = None
    unsqueeze_182: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1);  primals_46 = None
    unsqueeze_183: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_51: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_74, unsqueeze_183);  mul_74 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_26: "f32[4, 448, 28, 28]" = torch.ops.aten.relu.default(add_51);  add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_6: "f32[4, 448, 1, 1]" = torch.ops.aten.mean.dim(relu_26, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_35: "f32[4, 112, 1, 1]" = torch.ops.aten.convolution.default(mean_6, primals_172, primals_173, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_27: "f32[4, 112, 1, 1]" = torch.ops.aten.relu.default(convolution_35);  convolution_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_36: "f32[4, 448, 1, 1]" = torch.ops.aten.convolution.default(relu_27, primals_174, primals_175, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_6: "f32[4, 448, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_75: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(relu_26, sigmoid_6);  sigmoid_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_37: "f32[4, 448, 28, 28]" = torch.ops.aten.convolution.default(mul_75, primals_176, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_52: "f32[448]" = torch.ops.aten.add.Tensor(primals_312, 1e-05)
    sqrt_23: "f32[448]" = torch.ops.aten.sqrt.default(add_52);  add_52 = None
    reciprocal_23: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
    mul_76: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
    unsqueeze_184: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_311, -1)
    unsqueeze_185: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    unsqueeze_186: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_76, -1);  mul_76 = None
    unsqueeze_187: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    sub_23: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_185);  unsqueeze_185 = None
    mul_77: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_187);  sub_23 = unsqueeze_187 = None
    unsqueeze_188: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_189: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_78: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_189);  mul_77 = unsqueeze_189 = None
    unsqueeze_190: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_191: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_53: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_78, unsqueeze_191);  mul_78 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_54: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(add_53, relu_24);  add_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_28: "f32[4, 448, 28, 28]" = torch.ops.aten.relu.default(add_54);  add_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_38: "f32[4, 896, 28, 28]" = torch.ops.aten.convolution.default(relu_28, primals_177, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_55: "f32[896]" = torch.ops.aten.add.Tensor(primals_314, 1e-05)
    sqrt_24: "f32[896]" = torch.ops.aten.sqrt.default(add_55);  add_55 = None
    reciprocal_24: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
    mul_79: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
    unsqueeze_192: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_313, -1)
    unsqueeze_193: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    unsqueeze_194: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_79, -1);  mul_79 = None
    unsqueeze_195: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    sub_24: "f32[4, 896, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_193);  unsqueeze_193 = None
    mul_80: "f32[4, 896, 28, 28]" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_195);  sub_24 = unsqueeze_195 = None
    unsqueeze_196: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1)
    unsqueeze_197: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_81: "f32[4, 896, 28, 28]" = torch.ops.aten.mul.Tensor(mul_80, unsqueeze_197);  mul_80 = unsqueeze_197 = None
    unsqueeze_198: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1);  primals_50 = None
    unsqueeze_199: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_56: "f32[4, 896, 28, 28]" = torch.ops.aten.add.Tensor(mul_81, unsqueeze_199);  mul_81 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_29: "f32[4, 896, 28, 28]" = torch.ops.aten.relu.default(add_56);  add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_39: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_29, primals_178, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_57: "f32[896]" = torch.ops.aten.add.Tensor(primals_316, 1e-05)
    sqrt_25: "f32[896]" = torch.ops.aten.sqrt.default(add_57);  add_57 = None
    reciprocal_25: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
    mul_82: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
    unsqueeze_200: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_315, -1)
    unsqueeze_201: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    unsqueeze_202: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_82, -1);  mul_82 = None
    unsqueeze_203: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    sub_25: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_201);  unsqueeze_201 = None
    mul_83: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_203);  sub_25 = unsqueeze_203 = None
    unsqueeze_204: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1)
    unsqueeze_205: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_84: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_83, unsqueeze_205);  mul_83 = unsqueeze_205 = None
    unsqueeze_206: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1);  primals_52 = None
    unsqueeze_207: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_58: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_84, unsqueeze_207);  mul_84 = unsqueeze_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_30: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_58);  add_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_7: "f32[4, 896, 1, 1]" = torch.ops.aten.mean.dim(relu_30, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_40: "f32[4, 112, 1, 1]" = torch.ops.aten.convolution.default(mean_7, primals_179, primals_180, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_31: "f32[4, 112, 1, 1]" = torch.ops.aten.relu.default(convolution_40);  convolution_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_41: "f32[4, 896, 1, 1]" = torch.ops.aten.convolution.default(relu_31, primals_181, primals_182, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_7: "f32[4, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_41)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_85: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(relu_30, sigmoid_7);  sigmoid_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_42: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(mul_85, primals_183, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_59: "f32[896]" = torch.ops.aten.add.Tensor(primals_318, 1e-05)
    sqrt_26: "f32[896]" = torch.ops.aten.sqrt.default(add_59);  add_59 = None
    reciprocal_26: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
    mul_86: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
    unsqueeze_208: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_317, -1)
    unsqueeze_209: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    unsqueeze_210: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_86, -1);  mul_86 = None
    unsqueeze_211: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    sub_26: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_209);  unsqueeze_209 = None
    mul_87: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_211);  sub_26 = unsqueeze_211 = None
    unsqueeze_212: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_213: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_88: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_87, unsqueeze_213);  mul_87 = unsqueeze_213 = None
    unsqueeze_214: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_215: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_60: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_88, unsqueeze_215);  mul_88 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_43: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_28, primals_184, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_61: "f32[896]" = torch.ops.aten.add.Tensor(primals_320, 1e-05)
    sqrt_27: "f32[896]" = torch.ops.aten.sqrt.default(add_61);  add_61 = None
    reciprocal_27: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
    mul_89: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
    unsqueeze_216: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_319, -1)
    unsqueeze_217: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    unsqueeze_218: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_89, -1);  mul_89 = None
    unsqueeze_219: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    sub_27: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_217);  unsqueeze_217 = None
    mul_90: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_219);  sub_27 = unsqueeze_219 = None
    unsqueeze_220: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1)
    unsqueeze_221: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_91: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_90, unsqueeze_221);  mul_90 = unsqueeze_221 = None
    unsqueeze_222: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1);  primals_56 = None
    unsqueeze_223: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_62: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_91, unsqueeze_223);  mul_91 = unsqueeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_63: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(add_60, add_62);  add_60 = add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_32: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_63);  add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_44: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_32, primals_185, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_64: "f32[896]" = torch.ops.aten.add.Tensor(primals_322, 1e-05)
    sqrt_28: "f32[896]" = torch.ops.aten.sqrt.default(add_64);  add_64 = None
    reciprocal_28: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
    mul_92: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
    unsqueeze_224: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_321, -1)
    unsqueeze_225: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    unsqueeze_226: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_92, -1);  mul_92 = None
    unsqueeze_227: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    sub_28: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_225);  unsqueeze_225 = None
    mul_93: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_227);  sub_28 = unsqueeze_227 = None
    unsqueeze_228: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1)
    unsqueeze_229: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_94: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_93, unsqueeze_229);  mul_93 = unsqueeze_229 = None
    unsqueeze_230: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_58, -1);  primals_58 = None
    unsqueeze_231: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_65: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_94, unsqueeze_231);  mul_94 = unsqueeze_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_33: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_65);  add_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_45: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_33, primals_186, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_66: "f32[896]" = torch.ops.aten.add.Tensor(primals_324, 1e-05)
    sqrt_29: "f32[896]" = torch.ops.aten.sqrt.default(add_66);  add_66 = None
    reciprocal_29: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
    mul_95: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
    unsqueeze_232: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_323, -1)
    unsqueeze_233: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    unsqueeze_234: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_95, -1);  mul_95 = None
    unsqueeze_235: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    sub_29: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_233);  unsqueeze_233 = None
    mul_96: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_235);  sub_29 = unsqueeze_235 = None
    unsqueeze_236: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_237: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_97: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_96, unsqueeze_237);  mul_96 = unsqueeze_237 = None
    unsqueeze_238: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_239: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_67: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_239);  mul_97 = unsqueeze_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_34: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_67);  add_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_8: "f32[4, 896, 1, 1]" = torch.ops.aten.mean.dim(relu_34, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_46: "f32[4, 224, 1, 1]" = torch.ops.aten.convolution.default(mean_8, primals_187, primals_188, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_35: "f32[4, 224, 1, 1]" = torch.ops.aten.relu.default(convolution_46);  convolution_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_47: "f32[4, 896, 1, 1]" = torch.ops.aten.convolution.default(relu_35, primals_189, primals_190, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_8: "f32[4, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_47)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_98: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(relu_34, sigmoid_8);  sigmoid_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_48: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(mul_98, primals_191, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_68: "f32[896]" = torch.ops.aten.add.Tensor(primals_326, 1e-05)
    sqrt_30: "f32[896]" = torch.ops.aten.sqrt.default(add_68);  add_68 = None
    reciprocal_30: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
    mul_99: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
    unsqueeze_240: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_325, -1)
    unsqueeze_241: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    unsqueeze_242: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_99, -1);  mul_99 = None
    unsqueeze_243: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    sub_30: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_241);  unsqueeze_241 = None
    mul_100: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_243);  sub_30 = unsqueeze_243 = None
    unsqueeze_244: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_61, -1)
    unsqueeze_245: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_101: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_100, unsqueeze_245);  mul_100 = unsqueeze_245 = None
    unsqueeze_246: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1);  primals_62 = None
    unsqueeze_247: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_69: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_101, unsqueeze_247);  mul_101 = unsqueeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_70: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(add_69, relu_32);  add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_36: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_70);  add_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_49: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_36, primals_192, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_71: "f32[896]" = torch.ops.aten.add.Tensor(primals_328, 1e-05)
    sqrt_31: "f32[896]" = torch.ops.aten.sqrt.default(add_71);  add_71 = None
    reciprocal_31: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
    mul_102: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
    unsqueeze_248: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_327, -1)
    unsqueeze_249: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    unsqueeze_250: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_102, -1);  mul_102 = None
    unsqueeze_251: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    sub_31: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_249);  unsqueeze_249 = None
    mul_103: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_251);  sub_31 = unsqueeze_251 = None
    unsqueeze_252: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1)
    unsqueeze_253: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_104: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_103, unsqueeze_253);  mul_103 = unsqueeze_253 = None
    unsqueeze_254: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_64, -1);  primals_64 = None
    unsqueeze_255: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_72: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_255);  mul_104 = unsqueeze_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_37: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_72);  add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_50: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_37, primals_193, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_73: "f32[896]" = torch.ops.aten.add.Tensor(primals_330, 1e-05)
    sqrt_32: "f32[896]" = torch.ops.aten.sqrt.default(add_73);  add_73 = None
    reciprocal_32: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
    mul_105: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
    unsqueeze_256: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_329, -1)
    unsqueeze_257: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    unsqueeze_258: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_105, -1);  mul_105 = None
    unsqueeze_259: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    sub_32: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_257);  unsqueeze_257 = None
    mul_106: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_259);  sub_32 = unsqueeze_259 = None
    unsqueeze_260: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
    unsqueeze_261: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
    mul_107: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_106, unsqueeze_261);  mul_106 = unsqueeze_261 = None
    unsqueeze_262: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
    unsqueeze_263: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
    add_74: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_107, unsqueeze_263);  mul_107 = unsqueeze_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_38: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_74);  add_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_9: "f32[4, 896, 1, 1]" = torch.ops.aten.mean.dim(relu_38, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_51: "f32[4, 224, 1, 1]" = torch.ops.aten.convolution.default(mean_9, primals_194, primals_195, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_39: "f32[4, 224, 1, 1]" = torch.ops.aten.relu.default(convolution_51);  convolution_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_52: "f32[4, 896, 1, 1]" = torch.ops.aten.convolution.default(relu_39, primals_196, primals_197, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_9: "f32[4, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_52)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_108: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(relu_38, sigmoid_9);  sigmoid_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_53: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(mul_108, primals_198, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_75: "f32[896]" = torch.ops.aten.add.Tensor(primals_332, 1e-05)
    sqrt_33: "f32[896]" = torch.ops.aten.sqrt.default(add_75);  add_75 = None
    reciprocal_33: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
    mul_109: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
    unsqueeze_264: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_331, -1)
    unsqueeze_265: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
    unsqueeze_266: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_109, -1);  mul_109 = None
    unsqueeze_267: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
    sub_33: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_265);  unsqueeze_265 = None
    mul_110: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_33, unsqueeze_267);  sub_33 = unsqueeze_267 = None
    unsqueeze_268: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_67, -1)
    unsqueeze_269: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
    mul_111: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_110, unsqueeze_269);  mul_110 = unsqueeze_269 = None
    unsqueeze_270: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1);  primals_68 = None
    unsqueeze_271: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
    add_76: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_271);  mul_111 = unsqueeze_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_77: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(add_76, relu_36);  add_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_40: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_77);  add_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_54: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_40, primals_199, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_78: "f32[896]" = torch.ops.aten.add.Tensor(primals_334, 1e-05)
    sqrt_34: "f32[896]" = torch.ops.aten.sqrt.default(add_78);  add_78 = None
    reciprocal_34: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
    mul_112: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
    unsqueeze_272: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_333, -1)
    unsqueeze_273: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
    unsqueeze_274: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_112, -1);  mul_112 = None
    unsqueeze_275: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
    sub_34: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_273);  unsqueeze_273 = None
    mul_113: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_275);  sub_34 = unsqueeze_275 = None
    unsqueeze_276: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1)
    unsqueeze_277: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
    mul_114: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_113, unsqueeze_277);  mul_113 = unsqueeze_277 = None
    unsqueeze_278: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_70, -1);  primals_70 = None
    unsqueeze_279: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
    add_79: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_114, unsqueeze_279);  mul_114 = unsqueeze_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_41: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_79);  add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_55: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_41, primals_200, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_80: "f32[896]" = torch.ops.aten.add.Tensor(primals_336, 1e-05)
    sqrt_35: "f32[896]" = torch.ops.aten.sqrt.default(add_80);  add_80 = None
    reciprocal_35: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
    mul_115: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
    unsqueeze_280: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_335, -1)
    unsqueeze_281: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
    unsqueeze_282: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_115, -1);  mul_115 = None
    unsqueeze_283: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
    sub_35: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_281);  unsqueeze_281 = None
    mul_116: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_283);  sub_35 = unsqueeze_283 = None
    unsqueeze_284: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_285: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
    mul_117: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_116, unsqueeze_285);  mul_116 = unsqueeze_285 = None
    unsqueeze_286: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_287: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
    add_81: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_117, unsqueeze_287);  mul_117 = unsqueeze_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_42: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_81);  add_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_10: "f32[4, 896, 1, 1]" = torch.ops.aten.mean.dim(relu_42, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_56: "f32[4, 224, 1, 1]" = torch.ops.aten.convolution.default(mean_10, primals_201, primals_202, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_43: "f32[4, 224, 1, 1]" = torch.ops.aten.relu.default(convolution_56);  convolution_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_57: "f32[4, 896, 1, 1]" = torch.ops.aten.convolution.default(relu_43, primals_203, primals_204, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_10: "f32[4, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_57)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_118: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(relu_42, sigmoid_10);  sigmoid_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_58: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(mul_118, primals_205, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_82: "f32[896]" = torch.ops.aten.add.Tensor(primals_338, 1e-05)
    sqrt_36: "f32[896]" = torch.ops.aten.sqrt.default(add_82);  add_82 = None
    reciprocal_36: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
    mul_119: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
    unsqueeze_288: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_337, -1)
    unsqueeze_289: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
    unsqueeze_290: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_119, -1);  mul_119 = None
    unsqueeze_291: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
    sub_36: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_289);  unsqueeze_289 = None
    mul_120: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_291);  sub_36 = unsqueeze_291 = None
    unsqueeze_292: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_73, -1)
    unsqueeze_293: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
    mul_121: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_120, unsqueeze_293);  mul_120 = unsqueeze_293 = None
    unsqueeze_294: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1);  primals_74 = None
    unsqueeze_295: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
    add_83: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_121, unsqueeze_295);  mul_121 = unsqueeze_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_84: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(add_83, relu_40);  add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_44: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_84);  add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_59: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_44, primals_206, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_85: "f32[896]" = torch.ops.aten.add.Tensor(primals_340, 1e-05)
    sqrt_37: "f32[896]" = torch.ops.aten.sqrt.default(add_85);  add_85 = None
    reciprocal_37: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
    mul_122: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
    unsqueeze_296: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_339, -1)
    unsqueeze_297: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
    unsqueeze_298: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_122, -1);  mul_122 = None
    unsqueeze_299: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
    sub_37: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_297);  unsqueeze_297 = None
    mul_123: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_299);  sub_37 = unsqueeze_299 = None
    unsqueeze_300: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1)
    unsqueeze_301: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
    mul_124: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_123, unsqueeze_301);  mul_123 = unsqueeze_301 = None
    unsqueeze_302: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_76, -1);  primals_76 = None
    unsqueeze_303: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
    add_86: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_124, unsqueeze_303);  mul_124 = unsqueeze_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_45: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_86);  add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_60: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_45, primals_207, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_87: "f32[896]" = torch.ops.aten.add.Tensor(primals_342, 1e-05)
    sqrt_38: "f32[896]" = torch.ops.aten.sqrt.default(add_87);  add_87 = None
    reciprocal_38: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
    mul_125: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
    unsqueeze_304: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_341, -1)
    unsqueeze_305: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
    unsqueeze_306: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_125, -1);  mul_125 = None
    unsqueeze_307: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
    sub_38: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_305);  unsqueeze_305 = None
    mul_126: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_307);  sub_38 = unsqueeze_307 = None
    unsqueeze_308: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
    unsqueeze_309: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
    mul_127: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_309);  mul_126 = unsqueeze_309 = None
    unsqueeze_310: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
    unsqueeze_311: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
    add_88: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_127, unsqueeze_311);  mul_127 = unsqueeze_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_46: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_88);  add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_11: "f32[4, 896, 1, 1]" = torch.ops.aten.mean.dim(relu_46, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_61: "f32[4, 224, 1, 1]" = torch.ops.aten.convolution.default(mean_11, primals_208, primals_209, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_47: "f32[4, 224, 1, 1]" = torch.ops.aten.relu.default(convolution_61);  convolution_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_62: "f32[4, 896, 1, 1]" = torch.ops.aten.convolution.default(relu_47, primals_210, primals_211, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_11: "f32[4, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_62)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_128: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(relu_46, sigmoid_11);  sigmoid_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_63: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(mul_128, primals_212, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_89: "f32[896]" = torch.ops.aten.add.Tensor(primals_344, 1e-05)
    sqrt_39: "f32[896]" = torch.ops.aten.sqrt.default(add_89);  add_89 = None
    reciprocal_39: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
    mul_129: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
    unsqueeze_312: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_343, -1)
    unsqueeze_313: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
    unsqueeze_314: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_129, -1);  mul_129 = None
    unsqueeze_315: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
    sub_39: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_313);  unsqueeze_313 = None
    mul_130: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_315);  sub_39 = unsqueeze_315 = None
    unsqueeze_316: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_79, -1)
    unsqueeze_317: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
    mul_131: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_130, unsqueeze_317);  mul_130 = unsqueeze_317 = None
    unsqueeze_318: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1);  primals_80 = None
    unsqueeze_319: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
    add_90: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_131, unsqueeze_319);  mul_131 = unsqueeze_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_91: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(add_90, relu_44);  add_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_48: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_91);  add_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_64: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_48, primals_213, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_92: "f32[896]" = torch.ops.aten.add.Tensor(primals_346, 1e-05)
    sqrt_40: "f32[896]" = torch.ops.aten.sqrt.default(add_92);  add_92 = None
    reciprocal_40: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
    mul_132: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
    unsqueeze_320: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_345, -1)
    unsqueeze_321: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
    unsqueeze_322: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_132, -1);  mul_132 = None
    unsqueeze_323: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
    sub_40: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_321);  unsqueeze_321 = None
    mul_133: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_323);  sub_40 = unsqueeze_323 = None
    unsqueeze_324: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1)
    unsqueeze_325: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
    mul_134: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_325);  mul_133 = unsqueeze_325 = None
    unsqueeze_326: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_82, -1);  primals_82 = None
    unsqueeze_327: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
    add_93: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_134, unsqueeze_327);  mul_134 = unsqueeze_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_49: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_93);  add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_65: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_49, primals_214, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_94: "f32[896]" = torch.ops.aten.add.Tensor(primals_348, 1e-05)
    sqrt_41: "f32[896]" = torch.ops.aten.sqrt.default(add_94);  add_94 = None
    reciprocal_41: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
    mul_135: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
    unsqueeze_328: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_347, -1)
    unsqueeze_329: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
    unsqueeze_330: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_135, -1);  mul_135 = None
    unsqueeze_331: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
    sub_41: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_329);  unsqueeze_329 = None
    mul_136: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_331);  sub_41 = unsqueeze_331 = None
    unsqueeze_332: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1)
    unsqueeze_333: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
    mul_137: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_333);  mul_136 = unsqueeze_333 = None
    unsqueeze_334: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
    unsqueeze_335: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
    add_95: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_137, unsqueeze_335);  mul_137 = unsqueeze_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_50: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_95);  add_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_12: "f32[4, 896, 1, 1]" = torch.ops.aten.mean.dim(relu_50, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_66: "f32[4, 224, 1, 1]" = torch.ops.aten.convolution.default(mean_12, primals_215, primals_216, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_51: "f32[4, 224, 1, 1]" = torch.ops.aten.relu.default(convolution_66);  convolution_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_67: "f32[4, 896, 1, 1]" = torch.ops.aten.convolution.default(relu_51, primals_217, primals_218, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_12: "f32[4, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_67)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_138: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(relu_50, sigmoid_12);  sigmoid_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_68: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(mul_138, primals_219, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_96: "f32[896]" = torch.ops.aten.add.Tensor(primals_350, 1e-05)
    sqrt_42: "f32[896]" = torch.ops.aten.sqrt.default(add_96);  add_96 = None
    reciprocal_42: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
    mul_139: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
    unsqueeze_336: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_349, -1)
    unsqueeze_337: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
    unsqueeze_338: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_139, -1);  mul_139 = None
    unsqueeze_339: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
    sub_42: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_337);  unsqueeze_337 = None
    mul_140: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_339);  sub_42 = unsqueeze_339 = None
    unsqueeze_340: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_85, -1)
    unsqueeze_341: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
    mul_141: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_140, unsqueeze_341);  mul_140 = unsqueeze_341 = None
    unsqueeze_342: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1);  primals_86 = None
    unsqueeze_343: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
    add_97: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_141, unsqueeze_343);  mul_141 = unsqueeze_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_98: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(add_97, relu_48);  add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_52: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_98);  add_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_69: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_52, primals_220, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_99: "f32[896]" = torch.ops.aten.add.Tensor(primals_352, 1e-05)
    sqrt_43: "f32[896]" = torch.ops.aten.sqrt.default(add_99);  add_99 = None
    reciprocal_43: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
    mul_142: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
    unsqueeze_344: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_351, -1)
    unsqueeze_345: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
    unsqueeze_346: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_142, -1);  mul_142 = None
    unsqueeze_347: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
    sub_43: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_345);  unsqueeze_345 = None
    mul_143: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_347);  sub_43 = unsqueeze_347 = None
    unsqueeze_348: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1)
    unsqueeze_349: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
    mul_144: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_143, unsqueeze_349);  mul_143 = unsqueeze_349 = None
    unsqueeze_350: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_88, -1);  primals_88 = None
    unsqueeze_351: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
    add_100: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_144, unsqueeze_351);  mul_144 = unsqueeze_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_53: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_100);  add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_70: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_53, primals_221, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_101: "f32[896]" = torch.ops.aten.add.Tensor(primals_354, 1e-05)
    sqrt_44: "f32[896]" = torch.ops.aten.sqrt.default(add_101);  add_101 = None
    reciprocal_44: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
    mul_145: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
    unsqueeze_352: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_353, -1)
    unsqueeze_353: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
    unsqueeze_354: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_145, -1);  mul_145 = None
    unsqueeze_355: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
    sub_44: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_353);  unsqueeze_353 = None
    mul_146: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_355);  sub_44 = unsqueeze_355 = None
    unsqueeze_356: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1)
    unsqueeze_357: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
    mul_147: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_146, unsqueeze_357);  mul_146 = unsqueeze_357 = None
    unsqueeze_358: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1);  primals_90 = None
    unsqueeze_359: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
    add_102: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_147, unsqueeze_359);  mul_147 = unsqueeze_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_54: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_102);  add_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_13: "f32[4, 896, 1, 1]" = torch.ops.aten.mean.dim(relu_54, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_71: "f32[4, 224, 1, 1]" = torch.ops.aten.convolution.default(mean_13, primals_222, primals_223, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_55: "f32[4, 224, 1, 1]" = torch.ops.aten.relu.default(convolution_71);  convolution_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_72: "f32[4, 896, 1, 1]" = torch.ops.aten.convolution.default(relu_55, primals_224, primals_225, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_13: "f32[4, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_148: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(relu_54, sigmoid_13);  sigmoid_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_73: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(mul_148, primals_226, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_103: "f32[896]" = torch.ops.aten.add.Tensor(primals_356, 1e-05)
    sqrt_45: "f32[896]" = torch.ops.aten.sqrt.default(add_103);  add_103 = None
    reciprocal_45: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
    mul_149: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
    unsqueeze_360: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_355, -1)
    unsqueeze_361: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
    unsqueeze_362: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_149, -1);  mul_149 = None
    unsqueeze_363: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
    sub_45: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_361);  unsqueeze_361 = None
    mul_150: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_363);  sub_45 = unsqueeze_363 = None
    unsqueeze_364: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_91, -1)
    unsqueeze_365: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
    mul_151: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_150, unsqueeze_365);  mul_150 = unsqueeze_365 = None
    unsqueeze_366: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1);  primals_92 = None
    unsqueeze_367: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
    add_104: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_151, unsqueeze_367);  mul_151 = unsqueeze_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_105: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(add_104, relu_52);  add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_56: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_105);  add_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_74: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_56, primals_227, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_106: "f32[896]" = torch.ops.aten.add.Tensor(primals_358, 1e-05)
    sqrt_46: "f32[896]" = torch.ops.aten.sqrt.default(add_106);  add_106 = None
    reciprocal_46: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
    mul_152: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
    unsqueeze_368: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_357, -1)
    unsqueeze_369: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
    unsqueeze_370: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_152, -1);  mul_152 = None
    unsqueeze_371: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
    sub_46: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_369);  unsqueeze_369 = None
    mul_153: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_371);  sub_46 = unsqueeze_371 = None
    unsqueeze_372: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_93, -1)
    unsqueeze_373: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
    mul_154: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_153, unsqueeze_373);  mul_153 = unsqueeze_373 = None
    unsqueeze_374: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_94, -1);  primals_94 = None
    unsqueeze_375: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
    add_107: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_154, unsqueeze_375);  mul_154 = unsqueeze_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_57: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_107);  add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_75: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_57, primals_228, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_108: "f32[896]" = torch.ops.aten.add.Tensor(primals_360, 1e-05)
    sqrt_47: "f32[896]" = torch.ops.aten.sqrt.default(add_108);  add_108 = None
    reciprocal_47: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
    mul_155: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
    unsqueeze_376: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_359, -1)
    unsqueeze_377: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
    unsqueeze_378: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_155, -1);  mul_155 = None
    unsqueeze_379: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
    sub_47: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_377);  unsqueeze_377 = None
    mul_156: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_379);  sub_47 = unsqueeze_379 = None
    unsqueeze_380: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1)
    unsqueeze_381: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
    mul_157: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_156, unsqueeze_381);  mul_156 = unsqueeze_381 = None
    unsqueeze_382: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1);  primals_96 = None
    unsqueeze_383: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
    add_109: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_157, unsqueeze_383);  mul_157 = unsqueeze_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_58: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_109);  add_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_14: "f32[4, 896, 1, 1]" = torch.ops.aten.mean.dim(relu_58, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_76: "f32[4, 224, 1, 1]" = torch.ops.aten.convolution.default(mean_14, primals_229, primals_230, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_59: "f32[4, 224, 1, 1]" = torch.ops.aten.relu.default(convolution_76);  convolution_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_77: "f32[4, 896, 1, 1]" = torch.ops.aten.convolution.default(relu_59, primals_231, primals_232, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_14: "f32[4, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_77)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_158: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(relu_58, sigmoid_14);  sigmoid_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_78: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(mul_158, primals_233, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_110: "f32[896]" = torch.ops.aten.add.Tensor(primals_362, 1e-05)
    sqrt_48: "f32[896]" = torch.ops.aten.sqrt.default(add_110);  add_110 = None
    reciprocal_48: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
    mul_159: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
    unsqueeze_384: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_361, -1)
    unsqueeze_385: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
    unsqueeze_386: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_159, -1);  mul_159 = None
    unsqueeze_387: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
    sub_48: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_385);  unsqueeze_385 = None
    mul_160: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_387);  sub_48 = unsqueeze_387 = None
    unsqueeze_388: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_97, -1)
    unsqueeze_389: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
    mul_161: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_160, unsqueeze_389);  mul_160 = unsqueeze_389 = None
    unsqueeze_390: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1);  primals_98 = None
    unsqueeze_391: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
    add_111: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_161, unsqueeze_391);  mul_161 = unsqueeze_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_112: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(add_111, relu_56);  add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_60: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_112);  add_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_79: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_60, primals_234, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_113: "f32[896]" = torch.ops.aten.add.Tensor(primals_364, 1e-05)
    sqrt_49: "f32[896]" = torch.ops.aten.sqrt.default(add_113);  add_113 = None
    reciprocal_49: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
    mul_162: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
    unsqueeze_392: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_363, -1)
    unsqueeze_393: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
    unsqueeze_394: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_162, -1);  mul_162 = None
    unsqueeze_395: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
    sub_49: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_393);  unsqueeze_393 = None
    mul_163: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_395);  sub_49 = unsqueeze_395 = None
    unsqueeze_396: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1)
    unsqueeze_397: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
    mul_164: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_163, unsqueeze_397);  mul_163 = unsqueeze_397 = None
    unsqueeze_398: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_100, -1);  primals_100 = None
    unsqueeze_399: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
    add_114: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_164, unsqueeze_399);  mul_164 = unsqueeze_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_61: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_114);  add_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_80: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_61, primals_235, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_115: "f32[896]" = torch.ops.aten.add.Tensor(primals_366, 1e-05)
    sqrt_50: "f32[896]" = torch.ops.aten.sqrt.default(add_115);  add_115 = None
    reciprocal_50: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
    mul_165: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
    unsqueeze_400: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_365, -1)
    unsqueeze_401: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
    unsqueeze_402: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_165, -1);  mul_165 = None
    unsqueeze_403: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
    sub_50: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_401);  unsqueeze_401 = None
    mul_166: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_403);  sub_50 = unsqueeze_403 = None
    unsqueeze_404: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_101, -1)
    unsqueeze_405: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
    mul_167: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_166, unsqueeze_405);  mul_166 = unsqueeze_405 = None
    unsqueeze_406: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1);  primals_102 = None
    unsqueeze_407: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
    add_116: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_407);  mul_167 = unsqueeze_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_62: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_116);  add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_15: "f32[4, 896, 1, 1]" = torch.ops.aten.mean.dim(relu_62, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_81: "f32[4, 224, 1, 1]" = torch.ops.aten.convolution.default(mean_15, primals_236, primals_237, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_63: "f32[4, 224, 1, 1]" = torch.ops.aten.relu.default(convolution_81);  convolution_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_82: "f32[4, 896, 1, 1]" = torch.ops.aten.convolution.default(relu_63, primals_238, primals_239, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_15: "f32[4, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_82)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_168: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(relu_62, sigmoid_15);  sigmoid_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_83: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(mul_168, primals_240, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_117: "f32[896]" = torch.ops.aten.add.Tensor(primals_368, 1e-05)
    sqrt_51: "f32[896]" = torch.ops.aten.sqrt.default(add_117);  add_117 = None
    reciprocal_51: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
    mul_169: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
    unsqueeze_408: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_367, -1)
    unsqueeze_409: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
    unsqueeze_410: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_169, -1);  mul_169 = None
    unsqueeze_411: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
    sub_51: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_409);  unsqueeze_409 = None
    mul_170: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_411);  sub_51 = unsqueeze_411 = None
    unsqueeze_412: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_103, -1)
    unsqueeze_413: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
    mul_171: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_170, unsqueeze_413);  mul_170 = unsqueeze_413 = None
    unsqueeze_414: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_104, -1);  primals_104 = None
    unsqueeze_415: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
    add_118: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_171, unsqueeze_415);  mul_171 = unsqueeze_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_119: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(add_118, relu_60);  add_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_64: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_119);  add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_84: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_64, primals_241, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_120: "f32[896]" = torch.ops.aten.add.Tensor(primals_370, 1e-05)
    sqrt_52: "f32[896]" = torch.ops.aten.sqrt.default(add_120);  add_120 = None
    reciprocal_52: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
    mul_172: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
    unsqueeze_416: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_369, -1)
    unsqueeze_417: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
    unsqueeze_418: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_172, -1);  mul_172 = None
    unsqueeze_419: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
    sub_52: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_417);  unsqueeze_417 = None
    mul_173: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_419);  sub_52 = unsqueeze_419 = None
    unsqueeze_420: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_105, -1)
    unsqueeze_421: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
    mul_174: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_173, unsqueeze_421);  mul_173 = unsqueeze_421 = None
    unsqueeze_422: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_106, -1);  primals_106 = None
    unsqueeze_423: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
    add_121: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_174, unsqueeze_423);  mul_174 = unsqueeze_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_65: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_121);  add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_85: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_65, primals_242, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_122: "f32[896]" = torch.ops.aten.add.Tensor(primals_372, 1e-05)
    sqrt_53: "f32[896]" = torch.ops.aten.sqrt.default(add_122);  add_122 = None
    reciprocal_53: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
    mul_175: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
    unsqueeze_424: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_371, -1)
    unsqueeze_425: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
    unsqueeze_426: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_175, -1);  mul_175 = None
    unsqueeze_427: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
    sub_53: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_425);  unsqueeze_425 = None
    mul_176: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_427);  sub_53 = unsqueeze_427 = None
    unsqueeze_428: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_107, -1)
    unsqueeze_429: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
    mul_177: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_176, unsqueeze_429);  mul_176 = unsqueeze_429 = None
    unsqueeze_430: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1);  primals_108 = None
    unsqueeze_431: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
    add_123: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_177, unsqueeze_431);  mul_177 = unsqueeze_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_66: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_123);  add_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_16: "f32[4, 896, 1, 1]" = torch.ops.aten.mean.dim(relu_66, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_86: "f32[4, 224, 1, 1]" = torch.ops.aten.convolution.default(mean_16, primals_243, primals_244, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_67: "f32[4, 224, 1, 1]" = torch.ops.aten.relu.default(convolution_86);  convolution_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_87: "f32[4, 896, 1, 1]" = torch.ops.aten.convolution.default(relu_67, primals_245, primals_246, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_16: "f32[4, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_87)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_178: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(relu_66, sigmoid_16);  sigmoid_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_88: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(mul_178, primals_247, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_124: "f32[896]" = torch.ops.aten.add.Tensor(primals_374, 1e-05)
    sqrt_54: "f32[896]" = torch.ops.aten.sqrt.default(add_124);  add_124 = None
    reciprocal_54: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_54);  sqrt_54 = None
    mul_179: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_54, 1);  reciprocal_54 = None
    unsqueeze_432: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_373, -1)
    unsqueeze_433: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
    unsqueeze_434: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_179, -1);  mul_179 = None
    unsqueeze_435: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
    sub_54: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_433);  unsqueeze_433 = None
    mul_180: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_435);  sub_54 = unsqueeze_435 = None
    unsqueeze_436: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_109, -1)
    unsqueeze_437: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
    mul_181: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_180, unsqueeze_437);  mul_180 = unsqueeze_437 = None
    unsqueeze_438: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_110, -1);  primals_110 = None
    unsqueeze_439: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
    add_125: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_439);  mul_181 = unsqueeze_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_126: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(add_125, relu_64);  add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_68: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_126);  add_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_89: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_68, primals_248, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_127: "f32[896]" = torch.ops.aten.add.Tensor(primals_376, 1e-05)
    sqrt_55: "f32[896]" = torch.ops.aten.sqrt.default(add_127);  add_127 = None
    reciprocal_55: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_55);  sqrt_55 = None
    mul_182: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_55, 1);  reciprocal_55 = None
    unsqueeze_440: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_375, -1)
    unsqueeze_441: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
    unsqueeze_442: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_182, -1);  mul_182 = None
    unsqueeze_443: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
    sub_55: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_89, unsqueeze_441);  unsqueeze_441 = None
    mul_183: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_443);  sub_55 = unsqueeze_443 = None
    unsqueeze_444: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_111, -1)
    unsqueeze_445: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
    mul_184: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_183, unsqueeze_445);  mul_183 = unsqueeze_445 = None
    unsqueeze_446: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_112, -1);  primals_112 = None
    unsqueeze_447: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
    add_128: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_184, unsqueeze_447);  mul_184 = unsqueeze_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_69: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_128);  add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_90: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_69, primals_249, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_129: "f32[896]" = torch.ops.aten.add.Tensor(primals_378, 1e-05)
    sqrt_56: "f32[896]" = torch.ops.aten.sqrt.default(add_129);  add_129 = None
    reciprocal_56: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_56);  sqrt_56 = None
    mul_185: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_56, 1);  reciprocal_56 = None
    unsqueeze_448: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_377, -1)
    unsqueeze_449: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, -1);  unsqueeze_448 = None
    unsqueeze_450: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_185, -1);  mul_185 = None
    unsqueeze_451: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, -1);  unsqueeze_450 = None
    sub_56: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_449);  unsqueeze_449 = None
    mul_186: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_451);  sub_56 = unsqueeze_451 = None
    unsqueeze_452: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_113, -1)
    unsqueeze_453: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, -1);  unsqueeze_452 = None
    mul_187: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_186, unsqueeze_453);  mul_186 = unsqueeze_453 = None
    unsqueeze_454: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_114, -1);  primals_114 = None
    unsqueeze_455: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, -1);  unsqueeze_454 = None
    add_130: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_187, unsqueeze_455);  mul_187 = unsqueeze_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_70: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_130);  add_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_17: "f32[4, 896, 1, 1]" = torch.ops.aten.mean.dim(relu_70, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_91: "f32[4, 224, 1, 1]" = torch.ops.aten.convolution.default(mean_17, primals_250, primals_251, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_71: "f32[4, 224, 1, 1]" = torch.ops.aten.relu.default(convolution_91);  convolution_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_92: "f32[4, 896, 1, 1]" = torch.ops.aten.convolution.default(relu_71, primals_252, primals_253, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_17: "f32[4, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_92)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_188: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(relu_70, sigmoid_17);  sigmoid_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_93: "f32[4, 896, 14, 14]" = torch.ops.aten.convolution.default(mul_188, primals_254, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_131: "f32[896]" = torch.ops.aten.add.Tensor(primals_380, 1e-05)
    sqrt_57: "f32[896]" = torch.ops.aten.sqrt.default(add_131);  add_131 = None
    reciprocal_57: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_57);  sqrt_57 = None
    mul_189: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_57, 1);  reciprocal_57 = None
    unsqueeze_456: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_379, -1)
    unsqueeze_457: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, -1);  unsqueeze_456 = None
    unsqueeze_458: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_189, -1);  mul_189 = None
    unsqueeze_459: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, -1);  unsqueeze_458 = None
    sub_57: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_457);  unsqueeze_457 = None
    mul_190: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_459);  sub_57 = unsqueeze_459 = None
    unsqueeze_460: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_115, -1)
    unsqueeze_461: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, -1);  unsqueeze_460 = None
    mul_191: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_190, unsqueeze_461);  mul_190 = unsqueeze_461 = None
    unsqueeze_462: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_116, -1);  primals_116 = None
    unsqueeze_463: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, -1);  unsqueeze_462 = None
    add_132: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_191, unsqueeze_463);  mul_191 = unsqueeze_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_133: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(add_132, relu_68);  add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_72: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_133);  add_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_94: "f32[4, 2240, 14, 14]" = torch.ops.aten.convolution.default(relu_72, primals_255, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_134: "f32[2240]" = torch.ops.aten.add.Tensor(primals_382, 1e-05)
    sqrt_58: "f32[2240]" = torch.ops.aten.sqrt.default(add_134);  add_134 = None
    reciprocal_58: "f32[2240]" = torch.ops.aten.reciprocal.default(sqrt_58);  sqrt_58 = None
    mul_192: "f32[2240]" = torch.ops.aten.mul.Tensor(reciprocal_58, 1);  reciprocal_58 = None
    unsqueeze_464: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_381, -1)
    unsqueeze_465: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
    unsqueeze_466: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(mul_192, -1);  mul_192 = None
    unsqueeze_467: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
    sub_58: "f32[4, 2240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_465);  unsqueeze_465 = None
    mul_193: "f32[4, 2240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_467);  sub_58 = unsqueeze_467 = None
    unsqueeze_468: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_117, -1)
    unsqueeze_469: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
    mul_194: "f32[4, 2240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_193, unsqueeze_469);  mul_193 = unsqueeze_469 = None
    unsqueeze_470: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_118, -1);  primals_118 = None
    unsqueeze_471: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
    add_135: "f32[4, 2240, 14, 14]" = torch.ops.aten.add.Tensor(mul_194, unsqueeze_471);  mul_194 = unsqueeze_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_73: "f32[4, 2240, 14, 14]" = torch.ops.aten.relu.default(add_135);  add_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_95: "f32[4, 2240, 7, 7]" = torch.ops.aten.convolution.default(relu_73, primals_256, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_136: "f32[2240]" = torch.ops.aten.add.Tensor(primals_384, 1e-05)
    sqrt_59: "f32[2240]" = torch.ops.aten.sqrt.default(add_136);  add_136 = None
    reciprocal_59: "f32[2240]" = torch.ops.aten.reciprocal.default(sqrt_59);  sqrt_59 = None
    mul_195: "f32[2240]" = torch.ops.aten.mul.Tensor(reciprocal_59, 1);  reciprocal_59 = None
    unsqueeze_472: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_383, -1)
    unsqueeze_473: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
    unsqueeze_474: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(mul_195, -1);  mul_195 = None
    unsqueeze_475: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
    sub_59: "f32[4, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_473);  unsqueeze_473 = None
    mul_196: "f32[4, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_475);  sub_59 = unsqueeze_475 = None
    unsqueeze_476: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_119, -1)
    unsqueeze_477: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
    mul_197: "f32[4, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_477);  mul_196 = unsqueeze_477 = None
    unsqueeze_478: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_120, -1);  primals_120 = None
    unsqueeze_479: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
    add_137: "f32[4, 2240, 7, 7]" = torch.ops.aten.add.Tensor(mul_197, unsqueeze_479);  mul_197 = unsqueeze_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_74: "f32[4, 2240, 7, 7]" = torch.ops.aten.relu.default(add_137);  add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_18: "f32[4, 2240, 1, 1]" = torch.ops.aten.mean.dim(relu_74, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_96: "f32[4, 224, 1, 1]" = torch.ops.aten.convolution.default(mean_18, primals_257, primals_258, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_75: "f32[4, 224, 1, 1]" = torch.ops.aten.relu.default(convolution_96);  convolution_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_97: "f32[4, 2240, 1, 1]" = torch.ops.aten.convolution.default(relu_75, primals_259, primals_260, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_18: "f32[4, 2240, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_97)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_198: "f32[4, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(relu_74, sigmoid_18);  sigmoid_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_98: "f32[4, 2240, 7, 7]" = torch.ops.aten.convolution.default(mul_198, primals_261, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_138: "f32[2240]" = torch.ops.aten.add.Tensor(primals_386, 1e-05)
    sqrt_60: "f32[2240]" = torch.ops.aten.sqrt.default(add_138);  add_138 = None
    reciprocal_60: "f32[2240]" = torch.ops.aten.reciprocal.default(sqrt_60);  sqrt_60 = None
    mul_199: "f32[2240]" = torch.ops.aten.mul.Tensor(reciprocal_60, 1);  reciprocal_60 = None
    unsqueeze_480: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_385, -1)
    unsqueeze_481: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
    unsqueeze_482: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(mul_199, -1);  mul_199 = None
    unsqueeze_483: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
    sub_60: "f32[4, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_481);  unsqueeze_481 = None
    mul_200: "f32[4, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_483);  sub_60 = unsqueeze_483 = None
    unsqueeze_484: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_121, -1)
    unsqueeze_485: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, -1);  unsqueeze_484 = None
    mul_201: "f32[4, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(mul_200, unsqueeze_485);  mul_200 = unsqueeze_485 = None
    unsqueeze_486: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_122, -1);  primals_122 = None
    unsqueeze_487: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
    add_139: "f32[4, 2240, 7, 7]" = torch.ops.aten.add.Tensor(mul_201, unsqueeze_487);  mul_201 = unsqueeze_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_99: "f32[4, 2240, 7, 7]" = torch.ops.aten.convolution.default(relu_72, primals_262, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_140: "f32[2240]" = torch.ops.aten.add.Tensor(primals_388, 1e-05)
    sqrt_61: "f32[2240]" = torch.ops.aten.sqrt.default(add_140);  add_140 = None
    reciprocal_61: "f32[2240]" = torch.ops.aten.reciprocal.default(sqrt_61);  sqrt_61 = None
    mul_202: "f32[2240]" = torch.ops.aten.mul.Tensor(reciprocal_61, 1);  reciprocal_61 = None
    unsqueeze_488: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_387, -1)
    unsqueeze_489: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
    unsqueeze_490: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(mul_202, -1);  mul_202 = None
    unsqueeze_491: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, -1);  unsqueeze_490 = None
    sub_61: "f32[4, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_99, unsqueeze_489);  unsqueeze_489 = None
    mul_203: "f32[4, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_491);  sub_61 = unsqueeze_491 = None
    unsqueeze_492: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_123, -1)
    unsqueeze_493: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, -1);  unsqueeze_492 = None
    mul_204: "f32[4, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_493);  mul_203 = unsqueeze_493 = None
    unsqueeze_494: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_124, -1);  primals_124 = None
    unsqueeze_495: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, -1);  unsqueeze_494 = None
    add_141: "f32[4, 2240, 7, 7]" = torch.ops.aten.add.Tensor(mul_204, unsqueeze_495);  mul_204 = unsqueeze_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_142: "f32[4, 2240, 7, 7]" = torch.ops.aten.add.Tensor(add_139, add_141);  add_139 = add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_76: "f32[4, 2240, 7, 7]" = torch.ops.aten.relu.default(add_142);  add_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean_19: "f32[4, 2240, 1, 1]" = torch.ops.aten.mean.dim(relu_76, [-1, -2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view: "f32[4, 2240]" = torch.ops.aten.reshape.default(mean_19, [4, 2240]);  mean_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    permute: "f32[2240, 1000]" = torch.ops.aten.permute.default(primals_263, [1, 0]);  primals_263 = None
    addmm: "f32[4, 1000]" = torch.ops.aten.addmm.default(primals_264, view, permute);  primals_264 = None
    permute_1: "f32[1000, 2240]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    le: "b8[4, 2240, 7, 7]" = torch.ops.aten.le.Scalar(relu_76, 0);  relu_76 = None
    return [addmm, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_126, primals_127, primals_128, primals_130, primals_132, primals_133, primals_134, primals_135, primals_136, primals_138, primals_140, primals_141, primals_142, primals_143, primals_145, primals_147, primals_148, primals_149, primals_150, primals_151, primals_153, primals_155, primals_156, primals_157, primals_158, primals_160, primals_162, primals_163, primals_164, primals_165, primals_167, primals_169, primals_170, primals_171, primals_172, primals_174, primals_176, primals_177, primals_178, primals_179, primals_181, primals_183, primals_184, primals_185, primals_186, primals_187, primals_189, primals_191, primals_192, primals_193, primals_194, primals_196, primals_198, primals_199, primals_200, primals_201, primals_203, primals_205, primals_206, primals_207, primals_208, primals_210, primals_212, primals_213, primals_214, primals_215, primals_217, primals_219, primals_220, primals_221, primals_222, primals_224, primals_226, primals_227, primals_228, primals_229, primals_231, primals_233, primals_234, primals_235, primals_236, primals_238, primals_240, primals_241, primals_242, primals_243, primals_245, primals_247, primals_248, primals_249, primals_250, primals_252, primals_254, primals_255, primals_256, primals_257, primals_259, primals_261, primals_262, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, convolution, relu, convolution_1, relu_1, convolution_2, relu_2, mean, relu_3, convolution_4, mul_9, convolution_5, convolution_6, relu_4, convolution_7, relu_5, convolution_8, relu_6, mean_1, relu_7, convolution_10, mul_22, convolution_11, relu_8, convolution_12, relu_9, convolution_13, relu_10, mean_2, relu_11, convolution_15, mul_32, convolution_16, convolution_17, relu_12, convolution_18, relu_13, convolution_19, relu_14, mean_3, relu_15, convolution_21, mul_45, convolution_22, relu_16, convolution_23, relu_17, convolution_24, relu_18, mean_4, relu_19, convolution_26, mul_55, convolution_27, relu_20, convolution_28, relu_21, convolution_29, relu_22, mean_5, relu_23, convolution_31, mul_65, convolution_32, relu_24, convolution_33, relu_25, convolution_34, relu_26, mean_6, relu_27, convolution_36, mul_75, convolution_37, relu_28, convolution_38, relu_29, convolution_39, relu_30, mean_7, relu_31, convolution_41, mul_85, convolution_42, convolution_43, relu_32, convolution_44, relu_33, convolution_45, relu_34, mean_8, relu_35, convolution_47, mul_98, convolution_48, relu_36, convolution_49, relu_37, convolution_50, relu_38, mean_9, relu_39, convolution_52, mul_108, convolution_53, relu_40, convolution_54, relu_41, convolution_55, relu_42, mean_10, relu_43, convolution_57, mul_118, convolution_58, relu_44, convolution_59, relu_45, convolution_60, relu_46, mean_11, relu_47, convolution_62, mul_128, convolution_63, relu_48, convolution_64, relu_49, convolution_65, relu_50, mean_12, relu_51, convolution_67, mul_138, convolution_68, relu_52, convolution_69, relu_53, convolution_70, relu_54, mean_13, relu_55, convolution_72, mul_148, convolution_73, relu_56, convolution_74, relu_57, convolution_75, relu_58, mean_14, relu_59, convolution_77, mul_158, convolution_78, relu_60, convolution_79, relu_61, convolution_80, relu_62, mean_15, relu_63, convolution_82, mul_168, convolution_83, relu_64, convolution_84, relu_65, convolution_85, relu_66, mean_16, relu_67, convolution_87, mul_178, convolution_88, relu_68, convolution_89, relu_69, convolution_90, relu_70, mean_17, relu_71, convolution_92, mul_188, convolution_93, relu_72, convolution_94, relu_73, convolution_95, relu_74, mean_18, relu_75, convolution_97, mul_198, convolution_98, convolution_99, view, permute_1, le]
    