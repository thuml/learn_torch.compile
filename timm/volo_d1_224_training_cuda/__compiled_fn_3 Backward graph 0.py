from __future__ import annotations



def forward(self, primals_3: "f32[64, 3, 7, 7]", primals_4: "f32[64]", primals_6: "f32[64, 64, 3, 3]", primals_7: "f32[64]", primals_9: "f32[64, 64, 3, 3]", primals_10: "f32[64]", primals_12: "f32[192, 64, 4, 4]", primals_14: "f32[192]", primals_21: "f32[192]", primals_27: "f32[192]", primals_34: "f32[192]", primals_40: "f32[192]", primals_47: "f32[192]", primals_53: "f32[192]", primals_60: "f32[192]", primals_66: "f32[384, 192, 2, 2]", primals_68: "f32[384]", primals_73: "f32[384]", primals_79: "f32[384]", primals_84: "f32[384]", primals_90: "f32[384]", primals_95: "f32[384]", primals_101: "f32[384]", primals_106: "f32[384]", primals_112: "f32[384]", primals_117: "f32[384]", primals_123: "f32[384]", primals_128: "f32[384]", primals_134: "f32[384]", primals_139: "f32[384]", primals_145: "f32[384]", primals_150: "f32[384]", primals_156: "f32[384]", primals_161: "f32[384]", primals_167: "f32[384]", primals_172: "f32[384]", primals_178: "f32[384]", primals_183: "f32[384]", primals_189: "f32[384]", primals_194: "f32[384]", primals_200: "f32[384]", primals_205: "f32[384]", primals_211: "f32[384]", primals_216: "f32[384]", primals_222: "f32[384]", primals_228: "f32[384]", primals_234: "f32[384]", primals_240: "f32[384]", primals_246: "f32[384]", primals_261: "f32[8, 3, 224, 224]", convolution: "f32[8, 64, 112, 112]", squeeze_1: "f32[64]", relu: "f32[8, 64, 112, 112]", convolution_1: "f32[8, 64, 112, 112]", squeeze_4: "f32[64]", relu_1: "f32[8, 64, 112, 112]", convolution_2: "f32[8, 64, 112, 112]", squeeze_7: "f32[64]", relu_2: "f32[8, 64, 112, 112]", mul_21: "f32[8, 28, 28, 192]", view: "f32[6272, 192]", add_17: "i64[3, 14]", unsqueeze_17: "i64[3, 14, 1, 1]", permute_5: "f32[8, 192, 28, 28]", view_4: "f32[1568, 192]", full_default: "f32[8, 192, 30, 30]", view_12: "f32[6272, 192]", mul_24: "f32[8, 28, 28, 192]", view_14: "f32[6272, 192]", addmm_1: "f32[6272, 576]", view_16: "f32[6272, 576]", mul_29: "f32[8, 28, 28, 192]", view_18: "f32[6272, 192]", permute_19: "f32[8, 192, 28, 28]", view_22: "f32[1568, 192]", view_30: "f32[6272, 192]", mul_32: "f32[8, 28, 28, 192]", view_32: "f32[6272, 192]", addmm_4: "f32[6272, 576]", view_34: "f32[6272, 576]", mul_37: "f32[8, 28, 28, 192]", view_36: "f32[6272, 192]", permute_33: "f32[8, 192, 28, 28]", view_40: "f32[1568, 192]", view_48: "f32[6272, 192]", mul_40: "f32[8, 28, 28, 192]", view_50: "f32[6272, 192]", addmm_7: "f32[6272, 576]", view_52: "f32[6272, 576]", mul_45: "f32[8, 28, 28, 192]", view_54: "f32[6272, 192]", permute_47: "f32[8, 192, 28, 28]", view_58: "f32[1568, 192]", view_66: "f32[6272, 192]", mul_48: "f32[8, 28, 28, 192]", view_68: "f32[6272, 192]", addmm_10: "f32[6272, 576]", view_70: "f32[6272, 576]", permute_57: "f32[8, 192, 28, 28]", mul_53: "f32[8, 14, 14, 384]", view_72: "f32[1568, 384]", view_82: "f32[1568, 384]", mul_56: "f32[8, 14, 14, 384]", view_84: "f32[1568, 384]", addmm_13: "f32[1568, 1152]", view_86: "f32[1568, 1152]", mul_61: "f32[8, 14, 14, 384]", view_88: "f32[1568, 384]", view_98: "f32[1568, 384]", mul_64: "f32[8, 14, 14, 384]", view_100: "f32[1568, 384]", addmm_16: "f32[1568, 1152]", view_102: "f32[1568, 1152]", mul_69: "f32[8, 14, 14, 384]", view_104: "f32[1568, 384]", view_114: "f32[1568, 384]", mul_72: "f32[8, 14, 14, 384]", view_116: "f32[1568, 384]", addmm_19: "f32[1568, 1152]", view_118: "f32[1568, 1152]", mul_77: "f32[8, 14, 14, 384]", view_120: "f32[1568, 384]", view_130: "f32[1568, 384]", mul_80: "f32[8, 14, 14, 384]", view_132: "f32[1568, 384]", addmm_22: "f32[1568, 1152]", view_134: "f32[1568, 1152]", mul_85: "f32[8, 14, 14, 384]", view_136: "f32[1568, 384]", view_146: "f32[1568, 384]", mul_88: "f32[8, 14, 14, 384]", view_148: "f32[1568, 384]", addmm_25: "f32[1568, 1152]", view_150: "f32[1568, 1152]", mul_93: "f32[8, 14, 14, 384]", view_152: "f32[1568, 384]", view_162: "f32[1568, 384]", mul_96: "f32[8, 14, 14, 384]", view_164: "f32[1568, 384]", addmm_28: "f32[1568, 1152]", view_166: "f32[1568, 1152]", mul_101: "f32[8, 14, 14, 384]", view_168: "f32[1568, 384]", view_178: "f32[1568, 384]", mul_104: "f32[8, 14, 14, 384]", view_180: "f32[1568, 384]", addmm_31: "f32[1568, 1152]", view_182: "f32[1568, 1152]", mul_109: "f32[8, 14, 14, 384]", view_184: "f32[1568, 384]", view_194: "f32[1568, 384]", mul_112: "f32[8, 14, 14, 384]", view_196: "f32[1568, 384]", addmm_34: "f32[1568, 1152]", view_198: "f32[1568, 1152]", mul_117: "f32[8, 14, 14, 384]", view_200: "f32[1568, 384]", view_210: "f32[1568, 384]", mul_120: "f32[8, 14, 14, 384]", view_212: "f32[1568, 384]", addmm_37: "f32[1568, 1152]", view_214: "f32[1568, 1152]", mul_125: "f32[8, 14, 14, 384]", view_216: "f32[1568, 384]", view_226: "f32[1568, 384]", mul_128: "f32[8, 14, 14, 384]", view_228: "f32[1568, 384]", addmm_40: "f32[1568, 1152]", view_230: "f32[1568, 1152]", mul_133: "f32[8, 14, 14, 384]", view_232: "f32[1568, 384]", view_242: "f32[1568, 384]", mul_136: "f32[8, 14, 14, 384]", view_244: "f32[1568, 384]", addmm_43: "f32[1568, 1152]", view_246: "f32[1568, 1152]", mul_141: "f32[8, 14, 14, 384]", view_248: "f32[1568, 384]", view_258: "f32[1568, 384]", mul_144: "f32[8, 14, 14, 384]", view_260: "f32[1568, 384]", addmm_46: "f32[1568, 1152]", view_262: "f32[1568, 1152]", mul_149: "f32[8, 14, 14, 384]", view_264: "f32[1568, 384]", view_274: "f32[1568, 384]", mul_152: "f32[8, 14, 14, 384]", view_276: "f32[1568, 384]", addmm_49: "f32[1568, 1152]", view_278: "f32[1568, 1152]", mul_157: "f32[8, 14, 14, 384]", view_280: "f32[1568, 384]", view_290: "f32[1568, 384]", mul_160: "f32[8, 14, 14, 384]", view_292: "f32[1568, 384]", addmm_52: "f32[1568, 1152]", view_294: "f32[1568, 1152]", cat: "f32[8, 197, 384]", getitem_121: "f32[8, 197, 1]", rsqrt_39: "f32[8, 197, 1]", view_297: "f32[1576, 384]", view_300: "f32[8, 384]", view_310: "f32[8, 384]", mul_168: "f32[8, 1, 384]", view_312: "f32[8, 384]", addmm_55: "f32[8, 1152]", view_314: "f32[8, 1152]", cat_1: "f32[8, 197, 384]", getitem_127: "f32[8, 197, 1]", rsqrt_41: "f32[8, 197, 1]", view_316: "f32[1576, 384]", view_319: "f32[8, 384]", view_329: "f32[8, 384]", mul_176: "f32[8, 1, 384]", view_331: "f32[8, 384]", addmm_58: "f32[8, 1152]", view_333: "f32[8, 1152]", cat_2: "f32[8, 197, 384]", getitem_133: "f32[8, 197, 1]", rsqrt_43: "f32[8, 197, 1]", select: "f32[8, 384]", view_335: "f32[1568, 384]", unsqueeze_61: "i64[8, 1, 1000]", permute_177: "f32[1000, 384]", permute_179: "f32[1000, 384]", permute_183: "f32[384, 1152]", permute_187: "f32[1152, 384]", div_21: "f32[8, 1, 1]", permute_191: "f32[384, 384]", permute_196: "f32[96, 197, 1]", permute_197: "f32[96, 32, 197]", alias_23: "f32[8, 12, 1, 197]", permute_198: "f32[96, 32, 1]", permute_199: "f32[96, 197, 32]", permute_203: "f32[384, 384]", permute_208: "f32[768, 384]", permute_210: "f32[384, 1152]", permute_214: "f32[1152, 384]", div_23: "f32[8, 1, 1]", permute_218: "f32[384, 384]", permute_223: "f32[96, 197, 1]", permute_224: "f32[96, 32, 197]", alias_24: "f32[8, 12, 1, 197]", permute_225: "f32[96, 32, 1]", permute_226: "f32[96, 197, 32]", permute_230: "f32[384, 384]", permute_235: "f32[768, 384]", permute_237: "f32[384, 1152]", permute_241: "f32[1152, 384]", div_25: "f32[8, 14, 14, 1]", permute_245: "f32[384, 384]", permute_250: "f32[96, 196, 196]", permute_251: "f32[96, 32, 196]", alias_25: "f32[8, 12, 196, 196]", permute_252: "f32[96, 32, 196]", permute_253: "f32[96, 196, 32]", permute_258: "f32[1152, 384]", div_26: "f32[8, 14, 14, 1]", permute_260: "f32[384, 1152]", permute_264: "f32[1152, 384]", div_27: "f32[8, 14, 14, 1]", permute_268: "f32[384, 384]", permute_273: "f32[96, 196, 196]", permute_274: "f32[96, 32, 196]", alias_26: "f32[8, 12, 196, 196]", permute_275: "f32[96, 32, 196]", permute_276: "f32[96, 196, 32]", permute_281: "f32[1152, 384]", div_28: "f32[8, 14, 14, 1]", permute_283: "f32[384, 1152]", permute_287: "f32[1152, 384]", div_29: "f32[8, 14, 14, 1]", permute_291: "f32[384, 384]", permute_296: "f32[96, 196, 196]", permute_297: "f32[96, 32, 196]", alias_27: "f32[8, 12, 196, 196]", permute_298: "f32[96, 32, 196]", permute_299: "f32[96, 196, 32]", permute_304: "f32[1152, 384]", div_30: "f32[8, 14, 14, 1]", permute_306: "f32[384, 1152]", permute_310: "f32[1152, 384]", div_31: "f32[8, 14, 14, 1]", permute_314: "f32[384, 384]", permute_319: "f32[96, 196, 196]", permute_320: "f32[96, 32, 196]", alias_28: "f32[8, 12, 196, 196]", permute_321: "f32[96, 32, 196]", permute_322: "f32[96, 196, 32]", permute_327: "f32[1152, 384]", div_32: "f32[8, 14, 14, 1]", permute_329: "f32[384, 1152]", permute_333: "f32[1152, 384]", div_33: "f32[8, 14, 14, 1]", permute_337: "f32[384, 384]", permute_342: "f32[96, 196, 196]", permute_343: "f32[96, 32, 196]", alias_29: "f32[8, 12, 196, 196]", permute_344: "f32[96, 32, 196]", permute_345: "f32[96, 196, 32]", permute_350: "f32[1152, 384]", div_34: "f32[8, 14, 14, 1]", permute_352: "f32[384, 1152]", permute_356: "f32[1152, 384]", div_35: "f32[8, 14, 14, 1]", permute_360: "f32[384, 384]", permute_365: "f32[96, 196, 196]", permute_366: "f32[96, 32, 196]", alias_30: "f32[8, 12, 196, 196]", permute_367: "f32[96, 32, 196]", permute_368: "f32[96, 196, 32]", permute_373: "f32[1152, 384]", div_36: "f32[8, 14, 14, 1]", permute_375: "f32[384, 1152]", permute_379: "f32[1152, 384]", div_37: "f32[8, 14, 14, 1]", permute_383: "f32[384, 384]", permute_388: "f32[96, 196, 196]", permute_389: "f32[96, 32, 196]", alias_31: "f32[8, 12, 196, 196]", permute_390: "f32[96, 32, 196]", permute_391: "f32[96, 196, 32]", permute_396: "f32[1152, 384]", div_38: "f32[8, 14, 14, 1]", permute_398: "f32[384, 1152]", permute_402: "f32[1152, 384]", div_39: "f32[8, 14, 14, 1]", permute_406: "f32[384, 384]", permute_411: "f32[96, 196, 196]", permute_412: "f32[96, 32, 196]", alias_32: "f32[8, 12, 196, 196]", permute_413: "f32[96, 32, 196]", permute_414: "f32[96, 196, 32]", permute_419: "f32[1152, 384]", div_40: "f32[8, 14, 14, 1]", permute_421: "f32[384, 1152]", permute_425: "f32[1152, 384]", div_41: "f32[8, 14, 14, 1]", permute_429: "f32[384, 384]", permute_434: "f32[96, 196, 196]", permute_435: "f32[96, 32, 196]", alias_33: "f32[8, 12, 196, 196]", permute_436: "f32[96, 32, 196]", permute_437: "f32[96, 196, 32]", permute_442: "f32[1152, 384]", div_42: "f32[8, 14, 14, 1]", permute_444: "f32[384, 1152]", permute_448: "f32[1152, 384]", div_43: "f32[8, 14, 14, 1]", permute_452: "f32[384, 384]", permute_457: "f32[96, 196, 196]", permute_458: "f32[96, 32, 196]", alias_34: "f32[8, 12, 196, 196]", permute_459: "f32[96, 32, 196]", permute_460: "f32[96, 196, 32]", permute_465: "f32[1152, 384]", div_44: "f32[8, 14, 14, 1]", permute_467: "f32[384, 1152]", permute_471: "f32[1152, 384]", div_45: "f32[8, 14, 14, 1]", permute_475: "f32[384, 384]", permute_480: "f32[96, 196, 196]", permute_481: "f32[96, 32, 196]", alias_35: "f32[8, 12, 196, 196]", permute_482: "f32[96, 32, 196]", permute_483: "f32[96, 196, 32]", permute_488: "f32[1152, 384]", div_46: "f32[8, 14, 14, 1]", permute_490: "f32[384, 1152]", permute_494: "f32[1152, 384]", div_47: "f32[8, 14, 14, 1]", permute_498: "f32[384, 384]", permute_503: "f32[96, 196, 196]", permute_504: "f32[96, 32, 196]", alias_36: "f32[8, 12, 196, 196]", permute_505: "f32[96, 32, 196]", permute_506: "f32[96, 196, 32]", permute_511: "f32[1152, 384]", div_48: "f32[8, 14, 14, 1]", permute_513: "f32[384, 1152]", permute_517: "f32[1152, 384]", div_49: "f32[8, 14, 14, 1]", permute_521: "f32[384, 384]", permute_526: "f32[96, 196, 196]", permute_527: "f32[96, 32, 196]", alias_37: "f32[8, 12, 196, 196]", permute_528: "f32[96, 32, 196]", permute_529: "f32[96, 196, 32]", permute_534: "f32[1152, 384]", div_50: "f32[8, 14, 14, 1]", permute_536: "f32[384, 1152]", permute_540: "f32[1152, 384]", div_51: "f32[8, 14, 14, 1]", permute_544: "f32[384, 384]", permute_549: "f32[96, 196, 196]", permute_550: "f32[96, 32, 196]", alias_38: "f32[8, 12, 196, 196]", permute_551: "f32[96, 32, 196]", permute_552: "f32[96, 196, 32]", permute_557: "f32[1152, 384]", div_52: "f32[8, 14, 14, 1]", permute_561: "f32[192, 576]", permute_565: "f32[576, 192]", div_53: "f32[8, 28, 28, 1]", permute_571: "f32[192, 192]", permute_576: "f32[9408, 9, 9]", permute_577: "f32[9408, 32, 9]", alias_39: "f32[8, 6, 196, 9, 9]", permute_579: "f32[486, 192]", permute_590: "f32[192, 192]", div_54: "f32[8, 28, 28, 1]", permute_592: "f32[192, 576]", permute_596: "f32[576, 192]", div_55: "f32[8, 28, 28, 1]", permute_602: "f32[192, 192]", permute_607: "f32[9408, 9, 9]", permute_608: "f32[9408, 32, 9]", alias_40: "f32[8, 6, 196, 9, 9]", permute_610: "f32[486, 192]", permute_621: "f32[192, 192]", div_56: "f32[8, 28, 28, 1]", permute_623: "f32[192, 576]", permute_627: "f32[576, 192]", div_57: "f32[8, 28, 28, 1]", permute_633: "f32[192, 192]", permute_638: "f32[9408, 9, 9]", permute_639: "f32[9408, 32, 9]", alias_41: "f32[8, 6, 196, 9, 9]", permute_641: "f32[486, 192]", permute_652: "f32[192, 192]", div_58: "f32[8, 28, 28, 1]", permute_654: "f32[192, 576]", permute_658: "f32[576, 192]", div_59: "f32[8, 28, 28, 1]", permute_664: "f32[192, 192]", permute_669: "f32[9408, 9, 9]", permute_670: "f32[9408, 32, 9]", alias_42: "f32[8, 6, 196, 9, 9]", permute_672: "f32[486, 192]", permute_683: "f32[192, 192]", div_60: "f32[8, 28, 28, 1]", unsqueeze_112: "f32[1, 64, 1, 1]", unsqueeze_124: "f32[1, 64, 1, 1]", unsqueeze_136: "f32[1, 64, 1, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_15: "f32[8, 28, 28, 576]" = torch.ops.aten.view.default(addmm_1, [8, 28, 28, 576]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_27: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_15, 0.7071067811865476)
    erf: "f32[8, 28, 28, 576]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
    add_25: "f32[8, 28, 28, 576]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_33: "f32[8, 28, 28, 576]" = torch.ops.aten.view.default(addmm_4, [8, 28, 28, 576]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_35: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_33, 0.7071067811865476)
    erf_1: "f32[8, 28, 28, 576]" = torch.ops.aten.erf.default(mul_35);  mul_35 = None
    add_37: "f32[8, 28, 28, 576]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_51: "f32[8, 28, 28, 576]" = torch.ops.aten.view.default(addmm_7, [8, 28, 28, 576]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_43: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_51, 0.7071067811865476)
    erf_2: "f32[8, 28, 28, 576]" = torch.ops.aten.erf.default(mul_43);  mul_43 = None
    add_49: "f32[8, 28, 28, 576]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_69: "f32[8, 28, 28, 576]" = torch.ops.aten.view.default(addmm_10, [8, 28, 28, 576]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_51: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_69, 0.7071067811865476)
    erf_3: "f32[8, 28, 28, 576]" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
    add_61: "f32[8, 28, 28, 576]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_85: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_13, [8, 14, 14, 1152]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_59: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476)
    erf_4: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_59);  mul_59 = None
    add_69: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_101: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_16, [8, 14, 14, 1152]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_67: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_101, 0.7071067811865476)
    erf_5: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_67);  mul_67 = None
    add_76: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_117: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_19, [8, 14, 14, 1152]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_75: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_117, 0.7071067811865476)
    erf_6: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
    add_83: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_133: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_22, [8, 14, 14, 1152]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_83: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_133, 0.7071067811865476)
    erf_7: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
    add_90: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_149: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_25, [8, 14, 14, 1152]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_91: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_149, 0.7071067811865476)
    erf_8: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_91);  mul_91 = None
    add_97: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_165: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_28, [8, 14, 14, 1152]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_99: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_165, 0.7071067811865476)
    erf_9: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_99);  mul_99 = None
    add_104: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_181: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_31, [8, 14, 14, 1152]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_107: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_181, 0.7071067811865476)
    erf_10: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_107);  mul_107 = None
    add_111: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_197: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_34, [8, 14, 14, 1152]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_115: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_197, 0.7071067811865476)
    erf_11: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_115);  mul_115 = None
    add_118: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_213: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_37, [8, 14, 14, 1152]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_123: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_213, 0.7071067811865476)
    erf_12: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_123);  mul_123 = None
    add_125: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_229: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_40, [8, 14, 14, 1152]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_131: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_229, 0.7071067811865476)
    erf_13: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_131);  mul_131 = None
    add_132: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_245: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_43, [8, 14, 14, 1152]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_139: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_245, 0.7071067811865476)
    erf_14: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_139);  mul_139 = None
    add_139: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_261: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_46, [8, 14, 14, 1152]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_147: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476)
    erf_15: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_147);  mul_147 = None
    add_146: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_277: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_49, [8, 14, 14, 1152]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_155: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_277, 0.7071067811865476)
    erf_16: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_155);  mul_155 = None
    add_153: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_293: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_52, [8, 14, 14, 1152]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_163: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_293, 0.7071067811865476)
    erf_17: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_163);  mul_163 = None
    add_160: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:286, code: cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
    sub_57: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat, getitem_121);  cat = getitem_121 = None
    mul_165: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_39);  sub_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_313: "f32[8, 1, 1152]" = torch.ops.aten.view.default(addmm_55, [8, 1, 1152]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_171: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(view_313, 0.7071067811865476)
    erf_18: "f32[8, 1, 1152]" = torch.ops.aten.erf.default(mul_171);  mul_171 = None
    add_167: "f32[8, 1, 1152]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:286, code: cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
    sub_60: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_1, getitem_127);  cat_1 = getitem_127 = None
    mul_173: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_41);  sub_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_332: "f32[8, 1, 1152]" = torch.ops.aten.view.default(addmm_58, [8, 1, 1152]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_179: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(view_332, 0.7071067811865476)
    erf_19: "f32[8, 1, 1152]" = torch.ops.aten.erf.default(mul_179);  mul_179 = None
    add_174: "f32[8, 1, 1152]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:703, code: x = self.norm(x)
    sub_63: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_2, getitem_133);  cat_2 = getitem_133 = None
    mul_181: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_43);  sub_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:720, code: out = out + 0.5 * aux.max(1)[0]
    mul_184: "f32[8, 1000]" = torch.ops.aten.mul.Tensor(tangents_1, 0.5)
    unsqueeze_60: "f32[8, 1, 1000]" = torch.ops.aten.unsqueeze.default(mul_184, 1);  mul_184 = None
    full_default_4: "f32[8, 196, 1000]" = torch.ops.aten.full.default([8, 196, 1000], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    scatter: "f32[8, 196, 1000]" = torch.ops.aten.scatter.src(full_default_4, 1, unsqueeze_61, unsqueeze_60);  full_default_4 = unsqueeze_61 = unsqueeze_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:719, code: aux = self.aux_head(x[:, 1:])
    sum_21: "f32[1, 1, 1000]" = torch.ops.aten.sum.dim_IntList(scatter, [0, 1], True)
    view_337: "f32[1000]" = torch.ops.aten.view.default(sum_21, [1000]);  sum_21 = None
    view_338: "f32[1568, 1000]" = torch.ops.aten.view.default(scatter, [1568, 1000]);  scatter = None
    permute_175: "f32[1000, 1568]" = torch.ops.aten.permute.default(view_338, [1, 0])
    mm_27: "f32[1000, 384]" = torch.ops.aten.mm.default(permute_175, view_335);  permute_175 = view_335 = None
    permute_176: "f32[384, 1000]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    mm_28: "f32[1568, 384]" = torch.ops.aten.mm.default(view_338, permute_177);  view_338 = permute_177 = None
    view_339: "f32[8, 196, 384]" = torch.ops.aten.view.default(mm_28, [8, 196, 384]);  mm_28 = None
    permute_178: "f32[1000, 384]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
    full_default_5: "f32[8, 197, 384]" = torch.ops.aten.full.default([8, 197, 384], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default_5, view_339, 1, 1, 9223372036854775807);  view_339 = None
    slice_scatter_1: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default_5, slice_scatter, 0, 0, 9223372036854775807);  slice_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:716, code: out = self.head(out)
    mm_29: "f32[8, 384]" = torch.ops.aten.mm.default(tangents_1, permute_179);  permute_179 = None
    permute_180: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_30: "f32[1000, 384]" = torch.ops.aten.mm.default(permute_180, select);  permute_180 = select = None
    permute_181: "f32[384, 1000]" = torch.ops.aten.permute.default(mm_30, [1, 0]);  mm_30 = None
    sum_22: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_340: "f32[1000]" = torch.ops.aten.view.default(sum_22, [1000]);  sum_22 = None
    permute_182: "f32[1000, 384]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:710, code: out = x[:, 0]
    select_scatter: "f32[8, 197, 384]" = torch.ops.aten.select_scatter.default(full_default_5, mm_29, 1, 0);  mm_29 = None
    slice_scatter_2: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default_5, select_scatter, 0, 0, 9223372036854775807);  select_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:710, code: out = x[:, 0]
    add_180: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(slice_scatter_1, slice_scatter_2);  slice_scatter_1 = slice_scatter_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:703, code: x = self.norm(x)
    mul_186: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_180, primals_246);  primals_246 = None
    mul_187: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_186, 384)
    sum_23: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_186, [2], True)
    mul_188: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_186, mul_181);  mul_186 = None
    sum_24: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_188, [2], True);  mul_188 = None
    mul_189: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_181, sum_24);  sum_24 = None
    sub_65: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(mul_187, sum_23);  mul_187 = sum_23 = None
    sub_66: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(sub_65, mul_189);  sub_65 = mul_189 = None
    div_20: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 384);  rsqrt_43 = None
    mul_190: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(div_20, sub_66);  div_20 = sub_66 = None
    mul_191: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_180, mul_181);  mul_181 = None
    sum_25: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_191, [0, 1]);  mul_191 = None
    sum_26: "f32[384]" = torch.ops.aten.sum.dim_IntList(add_180, [0, 1]);  add_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:288, code: return torch.cat([cls_embed, x[:, 1:]], dim=1)
    slice_26: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(mul_190, 1, 0, 1)
    slice_27: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(mul_190, 1, 1, 197);  mul_190 = None
    slice_scatter_3: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default_5, slice_27, 1, 1, 9223372036854775807);  slice_27 = None
    slice_scatter_4: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default_5, slice_scatter_3, 0, 0, 9223372036854775807);  slice_scatter_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_341: "f32[8, 384]" = torch.ops.aten.view.default(slice_26, [8, 384])
    mm_31: "f32[8, 1152]" = torch.ops.aten.mm.default(view_341, permute_183);  permute_183 = None
    permute_184: "f32[384, 8]" = torch.ops.aten.permute.default(view_341, [1, 0])
    mm_32: "f32[384, 1152]" = torch.ops.aten.mm.default(permute_184, view_333);  permute_184 = view_333 = None
    permute_185: "f32[1152, 384]" = torch.ops.aten.permute.default(mm_32, [1, 0]);  mm_32 = None
    sum_27: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_341, [0], True);  view_341 = None
    view_342: "f32[384]" = torch.ops.aten.view.default(sum_27, [384]);  sum_27 = None
    permute_186: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
    view_343: "f32[8, 1, 1152]" = torch.ops.aten.view.default(mm_31, [8, 1, 1152]);  mm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_193: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(add_174, 0.5);  add_174 = None
    mul_194: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(view_332, view_332)
    mul_195: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(mul_194, -0.5);  mul_194 = None
    exp_20: "f32[8, 1, 1152]" = torch.ops.aten.exp.default(mul_195);  mul_195 = None
    mul_196: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_197: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(view_332, mul_196);  view_332 = mul_196 = None
    add_182: "f32[8, 1, 1152]" = torch.ops.aten.add.Tensor(mul_193, mul_197);  mul_193 = mul_197 = None
    mul_198: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(view_343, add_182);  view_343 = add_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_344: "f32[8, 1152]" = torch.ops.aten.view.default(mul_198, [8, 1152]);  mul_198 = None
    mm_33: "f32[8, 384]" = torch.ops.aten.mm.default(view_344, permute_187);  permute_187 = None
    permute_188: "f32[1152, 8]" = torch.ops.aten.permute.default(view_344, [1, 0])
    mm_34: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_188, view_331);  permute_188 = view_331 = None
    permute_189: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_34, [1, 0]);  mm_34 = None
    sum_28: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_344, [0], True);  view_344 = None
    view_345: "f32[1152]" = torch.ops.aten.view.default(sum_28, [1152]);  sum_28 = None
    permute_190: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_189, [1, 0]);  permute_189 = None
    view_346: "f32[8, 1, 384]" = torch.ops.aten.view.default(mm_33, [8, 1, 384]);  mm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:287, code: cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
    mul_200: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_346, primals_240);  primals_240 = None
    mul_201: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(mul_200, 384)
    sum_29: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_200, [2], True)
    mul_202: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(mul_200, mul_176);  mul_200 = None
    sum_30: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_202, [2], True);  mul_202 = None
    mul_203: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(mul_176, sum_30);  sum_30 = None
    sub_68: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(mul_201, sum_29);  mul_201 = sum_29 = None
    sub_69: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(sub_68, mul_203);  sub_68 = mul_203 = None
    mul_204: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(div_21, sub_69);  div_21 = sub_69 = None
    mul_205: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_346, mul_176);  mul_176 = None
    sum_31: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_205, [0, 1]);  mul_205 = None
    sum_32: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_346, [0, 1]);  view_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:287, code: cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
    add_183: "f32[8, 1, 384]" = torch.ops.aten.add.Tensor(slice_26, mul_204);  slice_26 = mul_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:243, code: cls_embed = self.proj(cls_embed)
    view_347: "f32[8, 384]" = torch.ops.aten.view.default(add_183, [8, 384])
    mm_35: "f32[8, 384]" = torch.ops.aten.mm.default(view_347, permute_191);  permute_191 = None
    permute_192: "f32[384, 8]" = torch.ops.aten.permute.default(view_347, [1, 0])
    mm_36: "f32[384, 384]" = torch.ops.aten.mm.default(permute_192, view_329);  permute_192 = view_329 = None
    permute_193: "f32[384, 384]" = torch.ops.aten.permute.default(mm_36, [1, 0]);  mm_36 = None
    sum_33: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_347, [0], True);  view_347 = None
    view_348: "f32[384]" = torch.ops.aten.view.default(sum_33, [384]);  sum_33 = None
    permute_194: "f32[384, 384]" = torch.ops.aten.permute.default(permute_193, [1, 0]);  permute_193 = None
    view_349: "f32[8, 1, 384]" = torch.ops.aten.view.default(mm_35, [8, 1, 384]);  mm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:242, code: cls_embed = (attn @ v).transpose(1, 2).reshape(B, 1, self.head_dim * self.num_heads)
    view_350: "f32[8, 1, 12, 32]" = torch.ops.aten.view.default(view_349, [8, 1, 12, 32]);  view_349 = None
    permute_195: "f32[8, 12, 1, 32]" = torch.ops.aten.permute.default(view_350, [0, 2, 1, 3]);  view_350 = None
    view_351: "f32[96, 1, 32]" = torch.ops.aten.view.default(permute_195, [96, 1, 32]);  permute_195 = None
    bmm_36: "f32[96, 197, 32]" = torch.ops.aten.bmm.default(permute_196, view_351);  permute_196 = None
    bmm_37: "f32[96, 1, 197]" = torch.ops.aten.bmm.default(view_351, permute_197);  view_351 = permute_197 = None
    view_352: "f32[8, 12, 197, 32]" = torch.ops.aten.view.default(bmm_36, [8, 12, 197, 32]);  bmm_36 = None
    view_353: "f32[8, 12, 1, 197]" = torch.ops.aten.view.default(bmm_37, [8, 12, 1, 197]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:239, code: attn = attn.softmax(dim=-1)
    mul_206: "f32[8, 12, 1, 197]" = torch.ops.aten.mul.Tensor(view_353, alias_23);  view_353 = None
    sum_34: "f32[8, 12, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_206, [-1], True)
    mul_207: "f32[8, 12, 1, 197]" = torch.ops.aten.mul.Tensor(alias_23, sum_34);  alias_23 = sum_34 = None
    sub_70: "f32[8, 12, 1, 197]" = torch.ops.aten.sub.Tensor(mul_206, mul_207);  mul_206 = mul_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:238, code: attn = ((q * self.scale) @ k.transpose(-2, -1))
    view_354: "f32[96, 1, 197]" = torch.ops.aten.view.default(sub_70, [96, 1, 197]);  sub_70 = None
    bmm_38: "f32[96, 32, 197]" = torch.ops.aten.bmm.default(permute_198, view_354);  permute_198 = None
    bmm_39: "f32[96, 1, 32]" = torch.ops.aten.bmm.default(view_354, permute_199);  view_354 = permute_199 = None
    view_355: "f32[8, 12, 32, 197]" = torch.ops.aten.view.default(bmm_38, [8, 12, 32, 197]);  bmm_38 = None
    view_356: "f32[8, 12, 1, 32]" = torch.ops.aten.view.default(bmm_39, [8, 12, 1, 32]);  bmm_39 = None
    permute_200: "f32[8, 12, 197, 32]" = torch.ops.aten.permute.default(view_355, [0, 1, 3, 2]);  view_355 = None
    mul_208: "f32[8, 12, 1, 32]" = torch.ops.aten.mul.Tensor(view_356, 0.1767766952966369);  view_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:237, code: q = self.q(x[:, :1, :]).reshape(B, self.num_heads, 1, self.head_dim)
    view_357: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_208, [8, 1, 384]);  mul_208 = None
    view_358: "f32[8, 384]" = torch.ops.aten.view.default(view_357, [8, 384]);  view_357 = None
    permute_201: "f32[384, 8]" = torch.ops.aten.permute.default(view_358, [1, 0])
    mm_37: "f32[384, 384]" = torch.ops.aten.mm.default(permute_201, view_319);  permute_201 = view_319 = None
    permute_202: "f32[384, 384]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    mm_38: "f32[8, 384]" = torch.ops.aten.mm.default(view_358, permute_203);  view_358 = permute_203 = None
    view_359: "f32[8, 1, 384]" = torch.ops.aten.view.default(mm_38, [8, 1, 384]);  mm_38 = None
    permute_204: "f32[384, 384]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
    full_default_11: "f32[8, 1, 384]" = torch.ops.aten.full.default([8, 1, 384], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_5: "f32[8, 1, 384]" = torch.ops.aten.slice_scatter.default(full_default_11, view_359, 2, 0, 9223372036854775807);  view_359 = None
    slice_scatter_6: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default_5, slice_scatter_5, 1, 0, 1);  slice_scatter_5 = None
    slice_scatter_7: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default_5, slice_scatter_6, 0, 0, 9223372036854775807);  slice_scatter_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:236, code: k, v = kv.unbind(0)
    cat_3: "f32[16, 12, 197, 32]" = torch.ops.aten.cat.default([permute_200, view_352]);  permute_200 = view_352 = None
    view_360: "f32[2, 8, 12, 197, 32]" = torch.ops.aten.view.default(cat_3, [2, 8, 12, 197, 32]);  cat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:235, code: kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_205: "f32[8, 197, 2, 12, 32]" = torch.ops.aten.permute.default(view_360, [1, 3, 0, 2, 4]);  view_360 = None
    clone_199: "f32[8, 197, 2, 12, 32]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
    view_361: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_199, [8, 197, 768]);  clone_199 = None
    view_362: "f32[1576, 768]" = torch.ops.aten.view.default(view_361, [1576, 768]);  view_361 = None
    permute_206: "f32[768, 1576]" = torch.ops.aten.permute.default(view_362, [1, 0])
    mm_39: "f32[768, 384]" = torch.ops.aten.mm.default(permute_206, view_316);  permute_206 = view_316 = None
    permute_207: "f32[384, 768]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    mm_40: "f32[1576, 384]" = torch.ops.aten.mm.default(view_362, permute_208);  view_362 = permute_208 = None
    view_363: "f32[8, 197, 384]" = torch.ops.aten.view.default(mm_40, [8, 197, 384]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:235, code: kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    add_184: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(slice_scatter_7, view_363);  slice_scatter_7 = view_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:235, code: kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_209: "f32[768, 384]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:286, code: cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
    mul_210: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_184, primals_234);  primals_234 = None
    mul_211: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_210, 384)
    sum_35: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_210, [2], True)
    mul_212: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_210, mul_173);  mul_210 = None
    sum_36: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_212, [2], True);  mul_212 = None
    mul_213: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_173, sum_36);  sum_36 = None
    sub_72: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(mul_211, sum_35);  mul_211 = sum_35 = None
    sub_73: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(sub_72, mul_213);  sub_72 = mul_213 = None
    div_22: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 384);  rsqrt_41 = None
    mul_214: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(div_22, sub_73);  div_22 = sub_73 = None
    mul_215: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_184, mul_173);  mul_173 = None
    sum_37: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_215, [0, 1]);  mul_215 = None
    sum_38: "f32[384]" = torch.ops.aten.sum.dim_IntList(add_184, [0, 1]);  add_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:286, code: cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
    add_185: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(slice_scatter_4, mul_214);  slice_scatter_4 = mul_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:285, code: cls_embed = x[:, :1]
    slice_scatter_8: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default_5, add_183, 1, 0, 1);  add_183 = None
    slice_scatter_9: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default_5, slice_scatter_8, 0, 0, 9223372036854775807);  slice_scatter_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:285, code: cls_embed = x[:, :1]
    add_186: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_185, slice_scatter_9);  add_185 = slice_scatter_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:288, code: return torch.cat([cls_embed, x[:, 1:]], dim=1)
    slice_28: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_186, 1, 0, 1)
    slice_29: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_186, 1, 1, 197);  add_186 = None
    slice_scatter_10: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default_5, slice_29, 1, 1, 9223372036854775807);  slice_29 = None
    slice_scatter_11: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default_5, slice_scatter_10, 0, 0, 9223372036854775807);  slice_scatter_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_364: "f32[8, 384]" = torch.ops.aten.view.default(slice_28, [8, 384])
    mm_41: "f32[8, 1152]" = torch.ops.aten.mm.default(view_364, permute_210);  permute_210 = None
    permute_211: "f32[384, 8]" = torch.ops.aten.permute.default(view_364, [1, 0])
    mm_42: "f32[384, 1152]" = torch.ops.aten.mm.default(permute_211, view_314);  permute_211 = view_314 = None
    permute_212: "f32[1152, 384]" = torch.ops.aten.permute.default(mm_42, [1, 0]);  mm_42 = None
    sum_39: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_364, [0], True);  view_364 = None
    view_365: "f32[384]" = torch.ops.aten.view.default(sum_39, [384]);  sum_39 = None
    permute_213: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_212, [1, 0]);  permute_212 = None
    view_366: "f32[8, 1, 1152]" = torch.ops.aten.view.default(mm_41, [8, 1, 1152]);  mm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_217: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(add_167, 0.5);  add_167 = None
    mul_218: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(view_313, view_313)
    mul_219: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(mul_218, -0.5);  mul_218 = None
    exp_21: "f32[8, 1, 1152]" = torch.ops.aten.exp.default(mul_219);  mul_219 = None
    mul_220: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_221: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(view_313, mul_220);  view_313 = mul_220 = None
    add_188: "f32[8, 1, 1152]" = torch.ops.aten.add.Tensor(mul_217, mul_221);  mul_217 = mul_221 = None
    mul_222: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(view_366, add_188);  view_366 = add_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_367: "f32[8, 1152]" = torch.ops.aten.view.default(mul_222, [8, 1152]);  mul_222 = None
    mm_43: "f32[8, 384]" = torch.ops.aten.mm.default(view_367, permute_214);  permute_214 = None
    permute_215: "f32[1152, 8]" = torch.ops.aten.permute.default(view_367, [1, 0])
    mm_44: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_215, view_312);  permute_215 = view_312 = None
    permute_216: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_44, [1, 0]);  mm_44 = None
    sum_40: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_367, [0], True);  view_367 = None
    view_368: "f32[1152]" = torch.ops.aten.view.default(sum_40, [1152]);  sum_40 = None
    permute_217: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_216, [1, 0]);  permute_216 = None
    view_369: "f32[8, 1, 384]" = torch.ops.aten.view.default(mm_43, [8, 1, 384]);  mm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:287, code: cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
    mul_224: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_369, primals_228);  primals_228 = None
    mul_225: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(mul_224, 384)
    sum_41: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_224, [2], True)
    mul_226: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(mul_224, mul_168);  mul_224 = None
    sum_42: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_226, [2], True);  mul_226 = None
    mul_227: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(mul_168, sum_42);  sum_42 = None
    sub_75: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(mul_225, sum_41);  mul_225 = sum_41 = None
    sub_76: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(sub_75, mul_227);  sub_75 = mul_227 = None
    mul_228: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(div_23, sub_76);  div_23 = sub_76 = None
    mul_229: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(view_369, mul_168);  mul_168 = None
    sum_43: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_229, [0, 1]);  mul_229 = None
    sum_44: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_369, [0, 1]);  view_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:287, code: cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
    add_189: "f32[8, 1, 384]" = torch.ops.aten.add.Tensor(slice_28, mul_228);  slice_28 = mul_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:243, code: cls_embed = self.proj(cls_embed)
    view_370: "f32[8, 384]" = torch.ops.aten.view.default(add_189, [8, 384])
    mm_45: "f32[8, 384]" = torch.ops.aten.mm.default(view_370, permute_218);  permute_218 = None
    permute_219: "f32[384, 8]" = torch.ops.aten.permute.default(view_370, [1, 0])
    mm_46: "f32[384, 384]" = torch.ops.aten.mm.default(permute_219, view_310);  permute_219 = view_310 = None
    permute_220: "f32[384, 384]" = torch.ops.aten.permute.default(mm_46, [1, 0]);  mm_46 = None
    sum_45: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_370, [0], True);  view_370 = None
    view_371: "f32[384]" = torch.ops.aten.view.default(sum_45, [384]);  sum_45 = None
    permute_221: "f32[384, 384]" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
    view_372: "f32[8, 1, 384]" = torch.ops.aten.view.default(mm_45, [8, 1, 384]);  mm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:242, code: cls_embed = (attn @ v).transpose(1, 2).reshape(B, 1, self.head_dim * self.num_heads)
    view_373: "f32[8, 1, 12, 32]" = torch.ops.aten.view.default(view_372, [8, 1, 12, 32]);  view_372 = None
    permute_222: "f32[8, 12, 1, 32]" = torch.ops.aten.permute.default(view_373, [0, 2, 1, 3]);  view_373 = None
    view_374: "f32[96, 1, 32]" = torch.ops.aten.view.default(permute_222, [96, 1, 32]);  permute_222 = None
    bmm_40: "f32[96, 197, 32]" = torch.ops.aten.bmm.default(permute_223, view_374);  permute_223 = None
    bmm_41: "f32[96, 1, 197]" = torch.ops.aten.bmm.default(view_374, permute_224);  view_374 = permute_224 = None
    view_375: "f32[8, 12, 197, 32]" = torch.ops.aten.view.default(bmm_40, [8, 12, 197, 32]);  bmm_40 = None
    view_376: "f32[8, 12, 1, 197]" = torch.ops.aten.view.default(bmm_41, [8, 12, 1, 197]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:239, code: attn = attn.softmax(dim=-1)
    mul_230: "f32[8, 12, 1, 197]" = torch.ops.aten.mul.Tensor(view_376, alias_24);  view_376 = None
    sum_46: "f32[8, 12, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_230, [-1], True)
    mul_231: "f32[8, 12, 1, 197]" = torch.ops.aten.mul.Tensor(alias_24, sum_46);  alias_24 = sum_46 = None
    sub_77: "f32[8, 12, 1, 197]" = torch.ops.aten.sub.Tensor(mul_230, mul_231);  mul_230 = mul_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:238, code: attn = ((q * self.scale) @ k.transpose(-2, -1))
    view_377: "f32[96, 1, 197]" = torch.ops.aten.view.default(sub_77, [96, 1, 197]);  sub_77 = None
    bmm_42: "f32[96, 32, 197]" = torch.ops.aten.bmm.default(permute_225, view_377);  permute_225 = None
    bmm_43: "f32[96, 1, 32]" = torch.ops.aten.bmm.default(view_377, permute_226);  view_377 = permute_226 = None
    view_378: "f32[8, 12, 32, 197]" = torch.ops.aten.view.default(bmm_42, [8, 12, 32, 197]);  bmm_42 = None
    view_379: "f32[8, 12, 1, 32]" = torch.ops.aten.view.default(bmm_43, [8, 12, 1, 32]);  bmm_43 = None
    permute_227: "f32[8, 12, 197, 32]" = torch.ops.aten.permute.default(view_378, [0, 1, 3, 2]);  view_378 = None
    mul_232: "f32[8, 12, 1, 32]" = torch.ops.aten.mul.Tensor(view_379, 0.1767766952966369);  view_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:237, code: q = self.q(x[:, :1, :]).reshape(B, self.num_heads, 1, self.head_dim)
    view_380: "f32[8, 1, 384]" = torch.ops.aten.view.default(mul_232, [8, 1, 384]);  mul_232 = None
    view_381: "f32[8, 384]" = torch.ops.aten.view.default(view_380, [8, 384]);  view_380 = None
    permute_228: "f32[384, 8]" = torch.ops.aten.permute.default(view_381, [1, 0])
    mm_47: "f32[384, 384]" = torch.ops.aten.mm.default(permute_228, view_300);  permute_228 = view_300 = None
    permute_229: "f32[384, 384]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    mm_48: "f32[8, 384]" = torch.ops.aten.mm.default(view_381, permute_230);  view_381 = permute_230 = None
    view_382: "f32[8, 1, 384]" = torch.ops.aten.view.default(mm_48, [8, 1, 384]);  mm_48 = None
    permute_231: "f32[384, 384]" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
    slice_scatter_12: "f32[8, 1, 384]" = torch.ops.aten.slice_scatter.default(full_default_11, view_382, 2, 0, 9223372036854775807);  full_default_11 = view_382 = None
    slice_scatter_13: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default_5, slice_scatter_12, 1, 0, 1);  slice_scatter_12 = None
    slice_scatter_14: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default_5, slice_scatter_13, 0, 0, 9223372036854775807);  slice_scatter_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:236, code: k, v = kv.unbind(0)
    cat_4: "f32[16, 12, 197, 32]" = torch.ops.aten.cat.default([permute_227, view_375]);  permute_227 = view_375 = None
    view_383: "f32[2, 8, 12, 197, 32]" = torch.ops.aten.view.default(cat_4, [2, 8, 12, 197, 32]);  cat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:235, code: kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_232: "f32[8, 197, 2, 12, 32]" = torch.ops.aten.permute.default(view_383, [1, 3, 0, 2, 4]);  view_383 = None
    clone_200: "f32[8, 197, 2, 12, 32]" = torch.ops.aten.clone.default(permute_232, memory_format = torch.contiguous_format);  permute_232 = None
    view_384: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_200, [8, 197, 768]);  clone_200 = None
    view_385: "f32[1576, 768]" = torch.ops.aten.view.default(view_384, [1576, 768]);  view_384 = None
    permute_233: "f32[768, 1576]" = torch.ops.aten.permute.default(view_385, [1, 0])
    mm_49: "f32[768, 384]" = torch.ops.aten.mm.default(permute_233, view_297);  permute_233 = view_297 = None
    permute_234: "f32[384, 768]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    mm_50: "f32[1576, 384]" = torch.ops.aten.mm.default(view_385, permute_235);  view_385 = permute_235 = None
    view_386: "f32[8, 197, 384]" = torch.ops.aten.view.default(mm_50, [8, 197, 384]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:235, code: kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    add_190: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(slice_scatter_14, view_386);  slice_scatter_14 = view_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:235, code: kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_236: "f32[768, 384]" = torch.ops.aten.permute.default(permute_234, [1, 0]);  permute_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:286, code: cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
    mul_234: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_190, primals_222);  primals_222 = None
    mul_235: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_234, 384)
    sum_47: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_234, [2], True)
    mul_236: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_234, mul_165);  mul_234 = None
    sum_48: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_236, [2], True);  mul_236 = None
    mul_237: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_165, sum_48);  sum_48 = None
    sub_79: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(mul_235, sum_47);  mul_235 = sum_47 = None
    sub_80: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(sub_79, mul_237);  sub_79 = mul_237 = None
    div_24: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 384);  rsqrt_39 = None
    mul_238: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(div_24, sub_80);  div_24 = sub_80 = None
    mul_239: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_190, mul_165);  mul_165 = None
    sum_49: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_239, [0, 1]);  mul_239 = None
    sum_50: "f32[384]" = torch.ops.aten.sum.dim_IntList(add_190, [0, 1]);  add_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:286, code: cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
    add_191: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(slice_scatter_11, mul_238);  slice_scatter_11 = mul_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:285, code: cls_embed = x[:, :1]
    slice_scatter_15: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default_5, add_189, 1, 0, 1);  add_189 = None
    slice_scatter_16: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default_5, slice_scatter_15, 0, 0, 9223372036854775807);  full_default_5 = slice_scatter_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:285, code: cls_embed = x[:, :1]
    add_192: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_191, slice_scatter_16);  add_191 = slice_scatter_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:634, code: x = torch.cat([cls_tokens, x], dim=1)
    slice_30: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_192, 1, 0, 1)
    slice_31: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_192, 1, 1, 197);  add_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:633, code: cls_tokens = self.cls_token.expand(B, -1, -1)
    sum_51: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(slice_30, [0], True);  slice_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:628, code: x = x.reshape(B, -1, C)
    view_387: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(slice_31, [8, 14, 14, 384]);  slice_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_201: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_387, memory_format = torch.contiguous_format)
    view_388: "f32[1568, 384]" = torch.ops.aten.view.default(clone_201, [1568, 384]);  clone_201 = None
    mm_51: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_388, permute_237);  permute_237 = None
    permute_238: "f32[384, 1568]" = torch.ops.aten.permute.default(view_388, [1, 0])
    mm_52: "f32[384, 1152]" = torch.ops.aten.mm.default(permute_238, view_294);  permute_238 = view_294 = None
    permute_239: "f32[1152, 384]" = torch.ops.aten.permute.default(mm_52, [1, 0]);  mm_52 = None
    sum_52: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_388, [0], True);  view_388 = None
    view_389: "f32[384]" = torch.ops.aten.view.default(sum_52, [384]);  sum_52 = None
    permute_240: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
    view_390: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_51, [8, 14, 14, 1152]);  mm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_241: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(add_160, 0.5);  add_160 = None
    mul_242: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_293, view_293)
    mul_243: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_242, -0.5);  mul_242 = None
    exp_22: "f32[8, 14, 14, 1152]" = torch.ops.aten.exp.default(mul_243);  mul_243 = None
    mul_244: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_245: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_293, mul_244);  view_293 = mul_244 = None
    add_194: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(mul_241, mul_245);  mul_241 = mul_245 = None
    mul_246: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_390, add_194);  view_390 = add_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_391: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_246, [1568, 1152]);  mul_246 = None
    mm_53: "f32[1568, 384]" = torch.ops.aten.mm.default(view_391, permute_241);  permute_241 = None
    permute_242: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_391, [1, 0])
    mm_54: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_242, view_292);  permute_242 = view_292 = None
    permute_243: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_54, [1, 0]);  mm_54 = None
    sum_53: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_391, [0], True);  view_391 = None
    view_392: "f32[1152]" = torch.ops.aten.view.default(sum_53, [1152]);  sum_53 = None
    permute_244: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
    view_393: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_53, [8, 14, 14, 384]);  mm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    mul_248: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_393, primals_216);  primals_216 = None
    mul_249: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_248, 384)
    sum_54: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_248, [3], True)
    mul_250: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_248, mul_160);  mul_248 = None
    sum_55: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_250, [3], True);  mul_250 = None
    mul_251: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_160, sum_55);  sum_55 = None
    sub_82: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_249, sum_54);  mul_249 = sum_54 = None
    sub_83: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_82, mul_251);  sub_82 = mul_251 = None
    mul_252: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_25, sub_83);  div_25 = sub_83 = None
    mul_253: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_393, mul_160);  mul_160 = None
    sum_56: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_253, [0, 1, 2]);  mul_253 = None
    sum_57: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_393, [0, 1, 2]);  view_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_195: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(view_387, mul_252);  view_387 = mul_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_394: "f32[1568, 384]" = torch.ops.aten.view.default(add_195, [1568, 384])
    mm_55: "f32[1568, 384]" = torch.ops.aten.mm.default(view_394, permute_245);  permute_245 = None
    permute_246: "f32[384, 1568]" = torch.ops.aten.permute.default(view_394, [1, 0])
    mm_56: "f32[384, 384]" = torch.ops.aten.mm.default(permute_246, view_290);  permute_246 = view_290 = None
    permute_247: "f32[384, 384]" = torch.ops.aten.permute.default(mm_56, [1, 0]);  mm_56 = None
    sum_58: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_394, [0], True);  view_394 = None
    view_395: "f32[384]" = torch.ops.aten.view.default(sum_58, [384]);  sum_58 = None
    permute_248: "f32[384, 384]" = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None
    view_396: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_55, [8, 14, 14, 384]);  mm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    view_397: "f32[8, 196, 12, 32]" = torch.ops.aten.view.default(view_396, [8, 196, 12, 32]);  view_396 = None
    permute_249: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_397, [0, 2, 1, 3]);  view_397 = None
    clone_203: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
    view_398: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_203, [96, 196, 32]);  clone_203 = None
    bmm_44: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(permute_250, view_398);  permute_250 = None
    bmm_45: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_398, permute_251);  view_398 = permute_251 = None
    view_399: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_44, [8, 12, 196, 32]);  bmm_44 = None
    view_400: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_45, [8, 12, 196, 196]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    mul_254: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_400, alias_25);  view_400 = None
    sum_59: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_254, [-1], True)
    mul_255: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(alias_25, sum_59);  alias_25 = sum_59 = None
    sub_84: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_254, mul_255);  mul_254 = mul_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_256: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(sub_84, 0.1767766952966369);  sub_84 = None
    view_401: "f32[96, 196, 196]" = torch.ops.aten.view.default(mul_256, [96, 196, 196]);  mul_256 = None
    bmm_46: "f32[96, 32, 196]" = torch.ops.aten.bmm.default(permute_252, view_401);  permute_252 = None
    bmm_47: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_401, permute_253);  view_401 = permute_253 = None
    view_402: "f32[8, 12, 32, 196]" = torch.ops.aten.view.default(bmm_46, [8, 12, 32, 196]);  bmm_46 = None
    view_403: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_47, [8, 12, 196, 32]);  bmm_47 = None
    permute_254: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_402, [0, 1, 3, 2]);  view_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    cat_5: "f32[24, 12, 196, 32]" = torch.ops.aten.cat.default([view_403, permute_254, view_399]);  view_403 = permute_254 = view_399 = None
    view_404: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.view.default(cat_5, [3, 8, 12, 196, 32]);  cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_255: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.permute.default(view_404, [1, 3, 0, 2, 4]);  view_404 = None
    clone_204: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.clone.default(permute_255, memory_format = torch.contiguous_format);  permute_255 = None
    view_405: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(clone_204, [8, 14, 14, 1152]);  clone_204 = None
    view_406: "f32[1568, 1152]" = torch.ops.aten.view.default(view_405, [1568, 1152]);  view_405 = None
    permute_256: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_406, [1, 0])
    mm_57: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_256, view_280);  permute_256 = view_280 = None
    permute_257: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    mm_58: "f32[1568, 384]" = torch.ops.aten.mm.default(view_406, permute_258);  view_406 = permute_258 = None
    view_407: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_58, [8, 14, 14, 384]);  mm_58 = None
    permute_259: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_257, [1, 0]);  permute_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    mul_258: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_407, primals_211);  primals_211 = None
    mul_259: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_258, 384)
    sum_60: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_258, [3], True)
    mul_260: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_258, mul_157);  mul_258 = None
    sum_61: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_260, [3], True);  mul_260 = None
    mul_261: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_157, sum_61);  sum_61 = None
    sub_86: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_259, sum_60);  mul_259 = sum_60 = None
    sub_87: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_86, mul_261);  sub_86 = mul_261 = None
    mul_262: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_26, sub_87);  div_26 = sub_87 = None
    mul_263: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_407, mul_157);  mul_157 = None
    sum_62: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_263, [0, 1, 2]);  mul_263 = None
    sum_63: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_407, [0, 1, 2]);  view_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_196: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_195, mul_262);  add_195 = mul_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_408: "f32[1568, 384]" = torch.ops.aten.view.default(add_196, [1568, 384])
    mm_59: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_408, permute_260);  permute_260 = None
    permute_261: "f32[384, 1568]" = torch.ops.aten.permute.default(view_408, [1, 0])
    mm_60: "f32[384, 1152]" = torch.ops.aten.mm.default(permute_261, view_278);  permute_261 = view_278 = None
    permute_262: "f32[1152, 384]" = torch.ops.aten.permute.default(mm_60, [1, 0]);  mm_60 = None
    sum_64: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_408, [0], True);  view_408 = None
    view_409: "f32[384]" = torch.ops.aten.view.default(sum_64, [384]);  sum_64 = None
    permute_263: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
    view_410: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_59, [8, 14, 14, 1152]);  mm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_265: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(add_153, 0.5);  add_153 = None
    mul_266: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_277, view_277)
    mul_267: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_266, -0.5);  mul_266 = None
    exp_23: "f32[8, 14, 14, 1152]" = torch.ops.aten.exp.default(mul_267);  mul_267 = None
    mul_268: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_269: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_277, mul_268);  view_277 = mul_268 = None
    add_198: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(mul_265, mul_269);  mul_265 = mul_269 = None
    mul_270: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_410, add_198);  view_410 = add_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_411: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_270, [1568, 1152]);  mul_270 = None
    mm_61: "f32[1568, 384]" = torch.ops.aten.mm.default(view_411, permute_264);  permute_264 = None
    permute_265: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_411, [1, 0])
    mm_62: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_265, view_276);  permute_265 = view_276 = None
    permute_266: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_62, [1, 0]);  mm_62 = None
    sum_65: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_411, [0], True);  view_411 = None
    view_412: "f32[1152]" = torch.ops.aten.view.default(sum_65, [1152]);  sum_65 = None
    permute_267: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_266, [1, 0]);  permute_266 = None
    view_413: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_61, [8, 14, 14, 384]);  mm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    mul_272: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_413, primals_205);  primals_205 = None
    mul_273: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_272, 384)
    sum_66: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_272, [3], True)
    mul_274: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_272, mul_152);  mul_272 = None
    sum_67: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_274, [3], True);  mul_274 = None
    mul_275: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_152, sum_67);  sum_67 = None
    sub_89: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_273, sum_66);  mul_273 = sum_66 = None
    sub_90: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_89, mul_275);  sub_89 = mul_275 = None
    mul_276: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_27, sub_90);  div_27 = sub_90 = None
    mul_277: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_413, mul_152);  mul_152 = None
    sum_68: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_277, [0, 1, 2]);  mul_277 = None
    sum_69: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_413, [0, 1, 2]);  view_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_199: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_196, mul_276);  add_196 = mul_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_414: "f32[1568, 384]" = torch.ops.aten.view.default(add_199, [1568, 384])
    mm_63: "f32[1568, 384]" = torch.ops.aten.mm.default(view_414, permute_268);  permute_268 = None
    permute_269: "f32[384, 1568]" = torch.ops.aten.permute.default(view_414, [1, 0])
    mm_64: "f32[384, 384]" = torch.ops.aten.mm.default(permute_269, view_274);  permute_269 = view_274 = None
    permute_270: "f32[384, 384]" = torch.ops.aten.permute.default(mm_64, [1, 0]);  mm_64 = None
    sum_70: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_414, [0], True);  view_414 = None
    view_415: "f32[384]" = torch.ops.aten.view.default(sum_70, [384]);  sum_70 = None
    permute_271: "f32[384, 384]" = torch.ops.aten.permute.default(permute_270, [1, 0]);  permute_270 = None
    view_416: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_63, [8, 14, 14, 384]);  mm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    view_417: "f32[8, 196, 12, 32]" = torch.ops.aten.view.default(view_416, [8, 196, 12, 32]);  view_416 = None
    permute_272: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_417, [0, 2, 1, 3]);  view_417 = None
    clone_207: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(permute_272, memory_format = torch.contiguous_format);  permute_272 = None
    view_418: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_207, [96, 196, 32]);  clone_207 = None
    bmm_48: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(permute_273, view_418);  permute_273 = None
    bmm_49: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_418, permute_274);  view_418 = permute_274 = None
    view_419: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_48, [8, 12, 196, 32]);  bmm_48 = None
    view_420: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_49, [8, 12, 196, 196]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    mul_278: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_420, alias_26);  view_420 = None
    sum_71: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_278, [-1], True)
    mul_279: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(alias_26, sum_71);  alias_26 = sum_71 = None
    sub_91: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_278, mul_279);  mul_278 = mul_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_280: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(sub_91, 0.1767766952966369);  sub_91 = None
    view_421: "f32[96, 196, 196]" = torch.ops.aten.view.default(mul_280, [96, 196, 196]);  mul_280 = None
    bmm_50: "f32[96, 32, 196]" = torch.ops.aten.bmm.default(permute_275, view_421);  permute_275 = None
    bmm_51: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_421, permute_276);  view_421 = permute_276 = None
    view_422: "f32[8, 12, 32, 196]" = torch.ops.aten.view.default(bmm_50, [8, 12, 32, 196]);  bmm_50 = None
    view_423: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_51, [8, 12, 196, 32]);  bmm_51 = None
    permute_277: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_422, [0, 1, 3, 2]);  view_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    cat_6: "f32[24, 12, 196, 32]" = torch.ops.aten.cat.default([view_423, permute_277, view_419]);  view_423 = permute_277 = view_419 = None
    view_424: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.view.default(cat_6, [3, 8, 12, 196, 32]);  cat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_278: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.permute.default(view_424, [1, 3, 0, 2, 4]);  view_424 = None
    clone_208: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.clone.default(permute_278, memory_format = torch.contiguous_format);  permute_278 = None
    view_425: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(clone_208, [8, 14, 14, 1152]);  clone_208 = None
    view_426: "f32[1568, 1152]" = torch.ops.aten.view.default(view_425, [1568, 1152]);  view_425 = None
    permute_279: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_426, [1, 0])
    mm_65: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_279, view_264);  permute_279 = view_264 = None
    permute_280: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    mm_66: "f32[1568, 384]" = torch.ops.aten.mm.default(view_426, permute_281);  view_426 = permute_281 = None
    view_427: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_66, [8, 14, 14, 384]);  mm_66 = None
    permute_282: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_280, [1, 0]);  permute_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    mul_282: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_427, primals_200);  primals_200 = None
    mul_283: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_282, 384)
    sum_72: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_282, [3], True)
    mul_284: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_282, mul_149);  mul_282 = None
    sum_73: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_284, [3], True);  mul_284 = None
    mul_285: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_149, sum_73);  sum_73 = None
    sub_93: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_283, sum_72);  mul_283 = sum_72 = None
    sub_94: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_93, mul_285);  sub_93 = mul_285 = None
    mul_286: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_28, sub_94);  div_28 = sub_94 = None
    mul_287: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_427, mul_149);  mul_149 = None
    sum_74: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_287, [0, 1, 2]);  mul_287 = None
    sum_75: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_427, [0, 1, 2]);  view_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_200: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_199, mul_286);  add_199 = mul_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_428: "f32[1568, 384]" = torch.ops.aten.view.default(add_200, [1568, 384])
    mm_67: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_428, permute_283);  permute_283 = None
    permute_284: "f32[384, 1568]" = torch.ops.aten.permute.default(view_428, [1, 0])
    mm_68: "f32[384, 1152]" = torch.ops.aten.mm.default(permute_284, view_262);  permute_284 = view_262 = None
    permute_285: "f32[1152, 384]" = torch.ops.aten.permute.default(mm_68, [1, 0]);  mm_68 = None
    sum_76: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_428, [0], True);  view_428 = None
    view_429: "f32[384]" = torch.ops.aten.view.default(sum_76, [384]);  sum_76 = None
    permute_286: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_285, [1, 0]);  permute_285 = None
    view_430: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_67, [8, 14, 14, 1152]);  mm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_289: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(add_146, 0.5);  add_146 = None
    mul_290: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_261, view_261)
    mul_291: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_290, -0.5);  mul_290 = None
    exp_24: "f32[8, 14, 14, 1152]" = torch.ops.aten.exp.default(mul_291);  mul_291 = None
    mul_292: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_293: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_261, mul_292);  view_261 = mul_292 = None
    add_202: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(mul_289, mul_293);  mul_289 = mul_293 = None
    mul_294: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_430, add_202);  view_430 = add_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_431: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_294, [1568, 1152]);  mul_294 = None
    mm_69: "f32[1568, 384]" = torch.ops.aten.mm.default(view_431, permute_287);  permute_287 = None
    permute_288: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_431, [1, 0])
    mm_70: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_288, view_260);  permute_288 = view_260 = None
    permute_289: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_70, [1, 0]);  mm_70 = None
    sum_77: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_431, [0], True);  view_431 = None
    view_432: "f32[1152]" = torch.ops.aten.view.default(sum_77, [1152]);  sum_77 = None
    permute_290: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_289, [1, 0]);  permute_289 = None
    view_433: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_69, [8, 14, 14, 384]);  mm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    mul_296: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_433, primals_194);  primals_194 = None
    mul_297: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_296, 384)
    sum_78: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_296, [3], True)
    mul_298: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_296, mul_144);  mul_296 = None
    sum_79: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_298, [3], True);  mul_298 = None
    mul_299: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_144, sum_79);  sum_79 = None
    sub_96: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_297, sum_78);  mul_297 = sum_78 = None
    sub_97: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_96, mul_299);  sub_96 = mul_299 = None
    mul_300: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_29, sub_97);  div_29 = sub_97 = None
    mul_301: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_433, mul_144);  mul_144 = None
    sum_80: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_301, [0, 1, 2]);  mul_301 = None
    sum_81: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_433, [0, 1, 2]);  view_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_203: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_200, mul_300);  add_200 = mul_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_434: "f32[1568, 384]" = torch.ops.aten.view.default(add_203, [1568, 384])
    mm_71: "f32[1568, 384]" = torch.ops.aten.mm.default(view_434, permute_291);  permute_291 = None
    permute_292: "f32[384, 1568]" = torch.ops.aten.permute.default(view_434, [1, 0])
    mm_72: "f32[384, 384]" = torch.ops.aten.mm.default(permute_292, view_258);  permute_292 = view_258 = None
    permute_293: "f32[384, 384]" = torch.ops.aten.permute.default(mm_72, [1, 0]);  mm_72 = None
    sum_82: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_434, [0], True);  view_434 = None
    view_435: "f32[384]" = torch.ops.aten.view.default(sum_82, [384]);  sum_82 = None
    permute_294: "f32[384, 384]" = torch.ops.aten.permute.default(permute_293, [1, 0]);  permute_293 = None
    view_436: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_71, [8, 14, 14, 384]);  mm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    view_437: "f32[8, 196, 12, 32]" = torch.ops.aten.view.default(view_436, [8, 196, 12, 32]);  view_436 = None
    permute_295: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_437, [0, 2, 1, 3]);  view_437 = None
    clone_211: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(permute_295, memory_format = torch.contiguous_format);  permute_295 = None
    view_438: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_211, [96, 196, 32]);  clone_211 = None
    bmm_52: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(permute_296, view_438);  permute_296 = None
    bmm_53: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_438, permute_297);  view_438 = permute_297 = None
    view_439: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_52, [8, 12, 196, 32]);  bmm_52 = None
    view_440: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_53, [8, 12, 196, 196]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    mul_302: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_440, alias_27);  view_440 = None
    sum_83: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_302, [-1], True)
    mul_303: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(alias_27, sum_83);  alias_27 = sum_83 = None
    sub_98: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_302, mul_303);  mul_302 = mul_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_304: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(sub_98, 0.1767766952966369);  sub_98 = None
    view_441: "f32[96, 196, 196]" = torch.ops.aten.view.default(mul_304, [96, 196, 196]);  mul_304 = None
    bmm_54: "f32[96, 32, 196]" = torch.ops.aten.bmm.default(permute_298, view_441);  permute_298 = None
    bmm_55: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_441, permute_299);  view_441 = permute_299 = None
    view_442: "f32[8, 12, 32, 196]" = torch.ops.aten.view.default(bmm_54, [8, 12, 32, 196]);  bmm_54 = None
    view_443: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_55, [8, 12, 196, 32]);  bmm_55 = None
    permute_300: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_442, [0, 1, 3, 2]);  view_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    cat_7: "f32[24, 12, 196, 32]" = torch.ops.aten.cat.default([view_443, permute_300, view_439]);  view_443 = permute_300 = view_439 = None
    view_444: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.view.default(cat_7, [3, 8, 12, 196, 32]);  cat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_301: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.permute.default(view_444, [1, 3, 0, 2, 4]);  view_444 = None
    clone_212: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.clone.default(permute_301, memory_format = torch.contiguous_format);  permute_301 = None
    view_445: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(clone_212, [8, 14, 14, 1152]);  clone_212 = None
    view_446: "f32[1568, 1152]" = torch.ops.aten.view.default(view_445, [1568, 1152]);  view_445 = None
    permute_302: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_446, [1, 0])
    mm_73: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_302, view_248);  permute_302 = view_248 = None
    permute_303: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    mm_74: "f32[1568, 384]" = torch.ops.aten.mm.default(view_446, permute_304);  view_446 = permute_304 = None
    view_447: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_74, [8, 14, 14, 384]);  mm_74 = None
    permute_305: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_303, [1, 0]);  permute_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    mul_306: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_447, primals_189);  primals_189 = None
    mul_307: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_306, 384)
    sum_84: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_306, [3], True)
    mul_308: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_306, mul_141);  mul_306 = None
    sum_85: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_308, [3], True);  mul_308 = None
    mul_309: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_141, sum_85);  sum_85 = None
    sub_100: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_307, sum_84);  mul_307 = sum_84 = None
    sub_101: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_100, mul_309);  sub_100 = mul_309 = None
    mul_310: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_30, sub_101);  div_30 = sub_101 = None
    mul_311: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_447, mul_141);  mul_141 = None
    sum_86: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_311, [0, 1, 2]);  mul_311 = None
    sum_87: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_447, [0, 1, 2]);  view_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_204: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_203, mul_310);  add_203 = mul_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_448: "f32[1568, 384]" = torch.ops.aten.view.default(add_204, [1568, 384])
    mm_75: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_448, permute_306);  permute_306 = None
    permute_307: "f32[384, 1568]" = torch.ops.aten.permute.default(view_448, [1, 0])
    mm_76: "f32[384, 1152]" = torch.ops.aten.mm.default(permute_307, view_246);  permute_307 = view_246 = None
    permute_308: "f32[1152, 384]" = torch.ops.aten.permute.default(mm_76, [1, 0]);  mm_76 = None
    sum_88: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_448, [0], True);  view_448 = None
    view_449: "f32[384]" = torch.ops.aten.view.default(sum_88, [384]);  sum_88 = None
    permute_309: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_308, [1, 0]);  permute_308 = None
    view_450: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_75, [8, 14, 14, 1152]);  mm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_313: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(add_139, 0.5);  add_139 = None
    mul_314: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_245, view_245)
    mul_315: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_314, -0.5);  mul_314 = None
    exp_25: "f32[8, 14, 14, 1152]" = torch.ops.aten.exp.default(mul_315);  mul_315 = None
    mul_316: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_317: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_245, mul_316);  view_245 = mul_316 = None
    add_206: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(mul_313, mul_317);  mul_313 = mul_317 = None
    mul_318: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_450, add_206);  view_450 = add_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_451: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_318, [1568, 1152]);  mul_318 = None
    mm_77: "f32[1568, 384]" = torch.ops.aten.mm.default(view_451, permute_310);  permute_310 = None
    permute_311: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_451, [1, 0])
    mm_78: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_311, view_244);  permute_311 = view_244 = None
    permute_312: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_78, [1, 0]);  mm_78 = None
    sum_89: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_451, [0], True);  view_451 = None
    view_452: "f32[1152]" = torch.ops.aten.view.default(sum_89, [1152]);  sum_89 = None
    permute_313: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_312, [1, 0]);  permute_312 = None
    view_453: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_77, [8, 14, 14, 384]);  mm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    mul_320: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_453, primals_183);  primals_183 = None
    mul_321: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_320, 384)
    sum_90: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_320, [3], True)
    mul_322: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_320, mul_136);  mul_320 = None
    sum_91: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_322, [3], True);  mul_322 = None
    mul_323: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_136, sum_91);  sum_91 = None
    sub_103: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_321, sum_90);  mul_321 = sum_90 = None
    sub_104: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_103, mul_323);  sub_103 = mul_323 = None
    mul_324: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_31, sub_104);  div_31 = sub_104 = None
    mul_325: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_453, mul_136);  mul_136 = None
    sum_92: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_325, [0, 1, 2]);  mul_325 = None
    sum_93: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_453, [0, 1, 2]);  view_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_207: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_204, mul_324);  add_204 = mul_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_454: "f32[1568, 384]" = torch.ops.aten.view.default(add_207, [1568, 384])
    mm_79: "f32[1568, 384]" = torch.ops.aten.mm.default(view_454, permute_314);  permute_314 = None
    permute_315: "f32[384, 1568]" = torch.ops.aten.permute.default(view_454, [1, 0])
    mm_80: "f32[384, 384]" = torch.ops.aten.mm.default(permute_315, view_242);  permute_315 = view_242 = None
    permute_316: "f32[384, 384]" = torch.ops.aten.permute.default(mm_80, [1, 0]);  mm_80 = None
    sum_94: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_454, [0], True);  view_454 = None
    view_455: "f32[384]" = torch.ops.aten.view.default(sum_94, [384]);  sum_94 = None
    permute_317: "f32[384, 384]" = torch.ops.aten.permute.default(permute_316, [1, 0]);  permute_316 = None
    view_456: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_79, [8, 14, 14, 384]);  mm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    view_457: "f32[8, 196, 12, 32]" = torch.ops.aten.view.default(view_456, [8, 196, 12, 32]);  view_456 = None
    permute_318: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_457, [0, 2, 1, 3]);  view_457 = None
    clone_215: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(permute_318, memory_format = torch.contiguous_format);  permute_318 = None
    view_458: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_215, [96, 196, 32]);  clone_215 = None
    bmm_56: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(permute_319, view_458);  permute_319 = None
    bmm_57: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_458, permute_320);  view_458 = permute_320 = None
    view_459: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_56, [8, 12, 196, 32]);  bmm_56 = None
    view_460: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_57, [8, 12, 196, 196]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    mul_326: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_460, alias_28);  view_460 = None
    sum_95: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_326, [-1], True)
    mul_327: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(alias_28, sum_95);  alias_28 = sum_95 = None
    sub_105: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_326, mul_327);  mul_326 = mul_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_328: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(sub_105, 0.1767766952966369);  sub_105 = None
    view_461: "f32[96, 196, 196]" = torch.ops.aten.view.default(mul_328, [96, 196, 196]);  mul_328 = None
    bmm_58: "f32[96, 32, 196]" = torch.ops.aten.bmm.default(permute_321, view_461);  permute_321 = None
    bmm_59: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_461, permute_322);  view_461 = permute_322 = None
    view_462: "f32[8, 12, 32, 196]" = torch.ops.aten.view.default(bmm_58, [8, 12, 32, 196]);  bmm_58 = None
    view_463: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_59, [8, 12, 196, 32]);  bmm_59 = None
    permute_323: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_462, [0, 1, 3, 2]);  view_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    cat_8: "f32[24, 12, 196, 32]" = torch.ops.aten.cat.default([view_463, permute_323, view_459]);  view_463 = permute_323 = view_459 = None
    view_464: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.view.default(cat_8, [3, 8, 12, 196, 32]);  cat_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_324: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.permute.default(view_464, [1, 3, 0, 2, 4]);  view_464 = None
    clone_216: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.clone.default(permute_324, memory_format = torch.contiguous_format);  permute_324 = None
    view_465: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(clone_216, [8, 14, 14, 1152]);  clone_216 = None
    view_466: "f32[1568, 1152]" = torch.ops.aten.view.default(view_465, [1568, 1152]);  view_465 = None
    permute_325: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_466, [1, 0])
    mm_81: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_325, view_232);  permute_325 = view_232 = None
    permute_326: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    mm_82: "f32[1568, 384]" = torch.ops.aten.mm.default(view_466, permute_327);  view_466 = permute_327 = None
    view_467: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_82, [8, 14, 14, 384]);  mm_82 = None
    permute_328: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_326, [1, 0]);  permute_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    mul_330: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_467, primals_178);  primals_178 = None
    mul_331: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_330, 384)
    sum_96: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_330, [3], True)
    mul_332: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_330, mul_133);  mul_330 = None
    sum_97: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_332, [3], True);  mul_332 = None
    mul_333: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_133, sum_97);  sum_97 = None
    sub_107: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_331, sum_96);  mul_331 = sum_96 = None
    sub_108: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_107, mul_333);  sub_107 = mul_333 = None
    mul_334: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_32, sub_108);  div_32 = sub_108 = None
    mul_335: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_467, mul_133);  mul_133 = None
    sum_98: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_335, [0, 1, 2]);  mul_335 = None
    sum_99: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_467, [0, 1, 2]);  view_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_208: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_207, mul_334);  add_207 = mul_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_468: "f32[1568, 384]" = torch.ops.aten.view.default(add_208, [1568, 384])
    mm_83: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_468, permute_329);  permute_329 = None
    permute_330: "f32[384, 1568]" = torch.ops.aten.permute.default(view_468, [1, 0])
    mm_84: "f32[384, 1152]" = torch.ops.aten.mm.default(permute_330, view_230);  permute_330 = view_230 = None
    permute_331: "f32[1152, 384]" = torch.ops.aten.permute.default(mm_84, [1, 0]);  mm_84 = None
    sum_100: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_468, [0], True);  view_468 = None
    view_469: "f32[384]" = torch.ops.aten.view.default(sum_100, [384]);  sum_100 = None
    permute_332: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_331, [1, 0]);  permute_331 = None
    view_470: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_83, [8, 14, 14, 1152]);  mm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_337: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(add_132, 0.5);  add_132 = None
    mul_338: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_229, view_229)
    mul_339: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_338, -0.5);  mul_338 = None
    exp_26: "f32[8, 14, 14, 1152]" = torch.ops.aten.exp.default(mul_339);  mul_339 = None
    mul_340: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_341: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_229, mul_340);  view_229 = mul_340 = None
    add_210: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(mul_337, mul_341);  mul_337 = mul_341 = None
    mul_342: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_470, add_210);  view_470 = add_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_471: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_342, [1568, 1152]);  mul_342 = None
    mm_85: "f32[1568, 384]" = torch.ops.aten.mm.default(view_471, permute_333);  permute_333 = None
    permute_334: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_471, [1, 0])
    mm_86: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_334, view_228);  permute_334 = view_228 = None
    permute_335: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_86, [1, 0]);  mm_86 = None
    sum_101: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_471, [0], True);  view_471 = None
    view_472: "f32[1152]" = torch.ops.aten.view.default(sum_101, [1152]);  sum_101 = None
    permute_336: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_335, [1, 0]);  permute_335 = None
    view_473: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_85, [8, 14, 14, 384]);  mm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    mul_344: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_473, primals_172);  primals_172 = None
    mul_345: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_344, 384)
    sum_102: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_344, [3], True)
    mul_346: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_344, mul_128);  mul_344 = None
    sum_103: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_346, [3], True);  mul_346 = None
    mul_347: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_128, sum_103);  sum_103 = None
    sub_110: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_345, sum_102);  mul_345 = sum_102 = None
    sub_111: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_110, mul_347);  sub_110 = mul_347 = None
    mul_348: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_33, sub_111);  div_33 = sub_111 = None
    mul_349: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_473, mul_128);  mul_128 = None
    sum_104: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_349, [0, 1, 2]);  mul_349 = None
    sum_105: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_473, [0, 1, 2]);  view_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_211: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_208, mul_348);  add_208 = mul_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_474: "f32[1568, 384]" = torch.ops.aten.view.default(add_211, [1568, 384])
    mm_87: "f32[1568, 384]" = torch.ops.aten.mm.default(view_474, permute_337);  permute_337 = None
    permute_338: "f32[384, 1568]" = torch.ops.aten.permute.default(view_474, [1, 0])
    mm_88: "f32[384, 384]" = torch.ops.aten.mm.default(permute_338, view_226);  permute_338 = view_226 = None
    permute_339: "f32[384, 384]" = torch.ops.aten.permute.default(mm_88, [1, 0]);  mm_88 = None
    sum_106: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_474, [0], True);  view_474 = None
    view_475: "f32[384]" = torch.ops.aten.view.default(sum_106, [384]);  sum_106 = None
    permute_340: "f32[384, 384]" = torch.ops.aten.permute.default(permute_339, [1, 0]);  permute_339 = None
    view_476: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_87, [8, 14, 14, 384]);  mm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    view_477: "f32[8, 196, 12, 32]" = torch.ops.aten.view.default(view_476, [8, 196, 12, 32]);  view_476 = None
    permute_341: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_477, [0, 2, 1, 3]);  view_477 = None
    clone_219: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(permute_341, memory_format = torch.contiguous_format);  permute_341 = None
    view_478: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_219, [96, 196, 32]);  clone_219 = None
    bmm_60: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(permute_342, view_478);  permute_342 = None
    bmm_61: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_478, permute_343);  view_478 = permute_343 = None
    view_479: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_60, [8, 12, 196, 32]);  bmm_60 = None
    view_480: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_61, [8, 12, 196, 196]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    mul_350: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_480, alias_29);  view_480 = None
    sum_107: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_350, [-1], True)
    mul_351: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(alias_29, sum_107);  alias_29 = sum_107 = None
    sub_112: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_350, mul_351);  mul_350 = mul_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_352: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(sub_112, 0.1767766952966369);  sub_112 = None
    view_481: "f32[96, 196, 196]" = torch.ops.aten.view.default(mul_352, [96, 196, 196]);  mul_352 = None
    bmm_62: "f32[96, 32, 196]" = torch.ops.aten.bmm.default(permute_344, view_481);  permute_344 = None
    bmm_63: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_481, permute_345);  view_481 = permute_345 = None
    view_482: "f32[8, 12, 32, 196]" = torch.ops.aten.view.default(bmm_62, [8, 12, 32, 196]);  bmm_62 = None
    view_483: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_63, [8, 12, 196, 32]);  bmm_63 = None
    permute_346: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_482, [0, 1, 3, 2]);  view_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    cat_9: "f32[24, 12, 196, 32]" = torch.ops.aten.cat.default([view_483, permute_346, view_479]);  view_483 = permute_346 = view_479 = None
    view_484: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.view.default(cat_9, [3, 8, 12, 196, 32]);  cat_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_347: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.permute.default(view_484, [1, 3, 0, 2, 4]);  view_484 = None
    clone_220: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.clone.default(permute_347, memory_format = torch.contiguous_format);  permute_347 = None
    view_485: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(clone_220, [8, 14, 14, 1152]);  clone_220 = None
    view_486: "f32[1568, 1152]" = torch.ops.aten.view.default(view_485, [1568, 1152]);  view_485 = None
    permute_348: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_486, [1, 0])
    mm_89: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_348, view_216);  permute_348 = view_216 = None
    permute_349: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    mm_90: "f32[1568, 384]" = torch.ops.aten.mm.default(view_486, permute_350);  view_486 = permute_350 = None
    view_487: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_90, [8, 14, 14, 384]);  mm_90 = None
    permute_351: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_349, [1, 0]);  permute_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    mul_354: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_487, primals_167);  primals_167 = None
    mul_355: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_354, 384)
    sum_108: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_354, [3], True)
    mul_356: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_354, mul_125);  mul_354 = None
    sum_109: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_356, [3], True);  mul_356 = None
    mul_357: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_125, sum_109);  sum_109 = None
    sub_114: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_355, sum_108);  mul_355 = sum_108 = None
    sub_115: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_114, mul_357);  sub_114 = mul_357 = None
    mul_358: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_34, sub_115);  div_34 = sub_115 = None
    mul_359: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_487, mul_125);  mul_125 = None
    sum_110: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_359, [0, 1, 2]);  mul_359 = None
    sum_111: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_487, [0, 1, 2]);  view_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_212: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_211, mul_358);  add_211 = mul_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_488: "f32[1568, 384]" = torch.ops.aten.view.default(add_212, [1568, 384])
    mm_91: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_488, permute_352);  permute_352 = None
    permute_353: "f32[384, 1568]" = torch.ops.aten.permute.default(view_488, [1, 0])
    mm_92: "f32[384, 1152]" = torch.ops.aten.mm.default(permute_353, view_214);  permute_353 = view_214 = None
    permute_354: "f32[1152, 384]" = torch.ops.aten.permute.default(mm_92, [1, 0]);  mm_92 = None
    sum_112: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_488, [0], True);  view_488 = None
    view_489: "f32[384]" = torch.ops.aten.view.default(sum_112, [384]);  sum_112 = None
    permute_355: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_354, [1, 0]);  permute_354 = None
    view_490: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_91, [8, 14, 14, 1152]);  mm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_361: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(add_125, 0.5);  add_125 = None
    mul_362: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_213, view_213)
    mul_363: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_362, -0.5);  mul_362 = None
    exp_27: "f32[8, 14, 14, 1152]" = torch.ops.aten.exp.default(mul_363);  mul_363 = None
    mul_364: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_365: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_213, mul_364);  view_213 = mul_364 = None
    add_214: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(mul_361, mul_365);  mul_361 = mul_365 = None
    mul_366: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_490, add_214);  view_490 = add_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_491: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_366, [1568, 1152]);  mul_366 = None
    mm_93: "f32[1568, 384]" = torch.ops.aten.mm.default(view_491, permute_356);  permute_356 = None
    permute_357: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_491, [1, 0])
    mm_94: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_357, view_212);  permute_357 = view_212 = None
    permute_358: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_94, [1, 0]);  mm_94 = None
    sum_113: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_491, [0], True);  view_491 = None
    view_492: "f32[1152]" = torch.ops.aten.view.default(sum_113, [1152]);  sum_113 = None
    permute_359: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_358, [1, 0]);  permute_358 = None
    view_493: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_93, [8, 14, 14, 384]);  mm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    mul_368: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_493, primals_161);  primals_161 = None
    mul_369: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_368, 384)
    sum_114: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_368, [3], True)
    mul_370: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_368, mul_120);  mul_368 = None
    sum_115: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_370, [3], True);  mul_370 = None
    mul_371: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_120, sum_115);  sum_115 = None
    sub_117: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_369, sum_114);  mul_369 = sum_114 = None
    sub_118: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_117, mul_371);  sub_117 = mul_371 = None
    mul_372: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_35, sub_118);  div_35 = sub_118 = None
    mul_373: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_493, mul_120);  mul_120 = None
    sum_116: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_373, [0, 1, 2]);  mul_373 = None
    sum_117: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_493, [0, 1, 2]);  view_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_215: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_212, mul_372);  add_212 = mul_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_494: "f32[1568, 384]" = torch.ops.aten.view.default(add_215, [1568, 384])
    mm_95: "f32[1568, 384]" = torch.ops.aten.mm.default(view_494, permute_360);  permute_360 = None
    permute_361: "f32[384, 1568]" = torch.ops.aten.permute.default(view_494, [1, 0])
    mm_96: "f32[384, 384]" = torch.ops.aten.mm.default(permute_361, view_210);  permute_361 = view_210 = None
    permute_362: "f32[384, 384]" = torch.ops.aten.permute.default(mm_96, [1, 0]);  mm_96 = None
    sum_118: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_494, [0], True);  view_494 = None
    view_495: "f32[384]" = torch.ops.aten.view.default(sum_118, [384]);  sum_118 = None
    permute_363: "f32[384, 384]" = torch.ops.aten.permute.default(permute_362, [1, 0]);  permute_362 = None
    view_496: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_95, [8, 14, 14, 384]);  mm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    view_497: "f32[8, 196, 12, 32]" = torch.ops.aten.view.default(view_496, [8, 196, 12, 32]);  view_496 = None
    permute_364: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_497, [0, 2, 1, 3]);  view_497 = None
    clone_223: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(permute_364, memory_format = torch.contiguous_format);  permute_364 = None
    view_498: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_223, [96, 196, 32]);  clone_223 = None
    bmm_64: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(permute_365, view_498);  permute_365 = None
    bmm_65: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_498, permute_366);  view_498 = permute_366 = None
    view_499: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_64, [8, 12, 196, 32]);  bmm_64 = None
    view_500: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_65, [8, 12, 196, 196]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    mul_374: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_500, alias_30);  view_500 = None
    sum_119: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_374, [-1], True)
    mul_375: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(alias_30, sum_119);  alias_30 = sum_119 = None
    sub_119: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_374, mul_375);  mul_374 = mul_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_376: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(sub_119, 0.1767766952966369);  sub_119 = None
    view_501: "f32[96, 196, 196]" = torch.ops.aten.view.default(mul_376, [96, 196, 196]);  mul_376 = None
    bmm_66: "f32[96, 32, 196]" = torch.ops.aten.bmm.default(permute_367, view_501);  permute_367 = None
    bmm_67: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_501, permute_368);  view_501 = permute_368 = None
    view_502: "f32[8, 12, 32, 196]" = torch.ops.aten.view.default(bmm_66, [8, 12, 32, 196]);  bmm_66 = None
    view_503: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_67, [8, 12, 196, 32]);  bmm_67 = None
    permute_369: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_502, [0, 1, 3, 2]);  view_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    cat_10: "f32[24, 12, 196, 32]" = torch.ops.aten.cat.default([view_503, permute_369, view_499]);  view_503 = permute_369 = view_499 = None
    view_504: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.view.default(cat_10, [3, 8, 12, 196, 32]);  cat_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_370: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.permute.default(view_504, [1, 3, 0, 2, 4]);  view_504 = None
    clone_224: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.clone.default(permute_370, memory_format = torch.contiguous_format);  permute_370 = None
    view_505: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(clone_224, [8, 14, 14, 1152]);  clone_224 = None
    view_506: "f32[1568, 1152]" = torch.ops.aten.view.default(view_505, [1568, 1152]);  view_505 = None
    permute_371: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_506, [1, 0])
    mm_97: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_371, view_200);  permute_371 = view_200 = None
    permute_372: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    mm_98: "f32[1568, 384]" = torch.ops.aten.mm.default(view_506, permute_373);  view_506 = permute_373 = None
    view_507: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_98, [8, 14, 14, 384]);  mm_98 = None
    permute_374: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_372, [1, 0]);  permute_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    mul_378: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_507, primals_156);  primals_156 = None
    mul_379: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_378, 384)
    sum_120: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_378, [3], True)
    mul_380: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_378, mul_117);  mul_378 = None
    sum_121: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_380, [3], True);  mul_380 = None
    mul_381: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_117, sum_121);  sum_121 = None
    sub_121: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_379, sum_120);  mul_379 = sum_120 = None
    sub_122: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_121, mul_381);  sub_121 = mul_381 = None
    mul_382: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_36, sub_122);  div_36 = sub_122 = None
    mul_383: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_507, mul_117);  mul_117 = None
    sum_122: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_383, [0, 1, 2]);  mul_383 = None
    sum_123: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_507, [0, 1, 2]);  view_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_216: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_215, mul_382);  add_215 = mul_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_508: "f32[1568, 384]" = torch.ops.aten.view.default(add_216, [1568, 384])
    mm_99: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_508, permute_375);  permute_375 = None
    permute_376: "f32[384, 1568]" = torch.ops.aten.permute.default(view_508, [1, 0])
    mm_100: "f32[384, 1152]" = torch.ops.aten.mm.default(permute_376, view_198);  permute_376 = view_198 = None
    permute_377: "f32[1152, 384]" = torch.ops.aten.permute.default(mm_100, [1, 0]);  mm_100 = None
    sum_124: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_508, [0], True);  view_508 = None
    view_509: "f32[384]" = torch.ops.aten.view.default(sum_124, [384]);  sum_124 = None
    permute_378: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_377, [1, 0]);  permute_377 = None
    view_510: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_99, [8, 14, 14, 1152]);  mm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_385: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(add_118, 0.5);  add_118 = None
    mul_386: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_197, view_197)
    mul_387: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_386, -0.5);  mul_386 = None
    exp_28: "f32[8, 14, 14, 1152]" = torch.ops.aten.exp.default(mul_387);  mul_387 = None
    mul_388: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
    mul_389: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_197, mul_388);  view_197 = mul_388 = None
    add_218: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(mul_385, mul_389);  mul_385 = mul_389 = None
    mul_390: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_510, add_218);  view_510 = add_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_511: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_390, [1568, 1152]);  mul_390 = None
    mm_101: "f32[1568, 384]" = torch.ops.aten.mm.default(view_511, permute_379);  permute_379 = None
    permute_380: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_511, [1, 0])
    mm_102: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_380, view_196);  permute_380 = view_196 = None
    permute_381: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_102, [1, 0]);  mm_102 = None
    sum_125: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_511, [0], True);  view_511 = None
    view_512: "f32[1152]" = torch.ops.aten.view.default(sum_125, [1152]);  sum_125 = None
    permute_382: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_381, [1, 0]);  permute_381 = None
    view_513: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_101, [8, 14, 14, 384]);  mm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    mul_392: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_513, primals_150);  primals_150 = None
    mul_393: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_392, 384)
    sum_126: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_392, [3], True)
    mul_394: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_392, mul_112);  mul_392 = None
    sum_127: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_394, [3], True);  mul_394 = None
    mul_395: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_112, sum_127);  sum_127 = None
    sub_124: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_393, sum_126);  mul_393 = sum_126 = None
    sub_125: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_124, mul_395);  sub_124 = mul_395 = None
    mul_396: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_37, sub_125);  div_37 = sub_125 = None
    mul_397: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_513, mul_112);  mul_112 = None
    sum_128: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_397, [0, 1, 2]);  mul_397 = None
    sum_129: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_513, [0, 1, 2]);  view_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_219: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_216, mul_396);  add_216 = mul_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_514: "f32[1568, 384]" = torch.ops.aten.view.default(add_219, [1568, 384])
    mm_103: "f32[1568, 384]" = torch.ops.aten.mm.default(view_514, permute_383);  permute_383 = None
    permute_384: "f32[384, 1568]" = torch.ops.aten.permute.default(view_514, [1, 0])
    mm_104: "f32[384, 384]" = torch.ops.aten.mm.default(permute_384, view_194);  permute_384 = view_194 = None
    permute_385: "f32[384, 384]" = torch.ops.aten.permute.default(mm_104, [1, 0]);  mm_104 = None
    sum_130: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_514, [0], True);  view_514 = None
    view_515: "f32[384]" = torch.ops.aten.view.default(sum_130, [384]);  sum_130 = None
    permute_386: "f32[384, 384]" = torch.ops.aten.permute.default(permute_385, [1, 0]);  permute_385 = None
    view_516: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_103, [8, 14, 14, 384]);  mm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    view_517: "f32[8, 196, 12, 32]" = torch.ops.aten.view.default(view_516, [8, 196, 12, 32]);  view_516 = None
    permute_387: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_517, [0, 2, 1, 3]);  view_517 = None
    clone_227: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(permute_387, memory_format = torch.contiguous_format);  permute_387 = None
    view_518: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_227, [96, 196, 32]);  clone_227 = None
    bmm_68: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(permute_388, view_518);  permute_388 = None
    bmm_69: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_518, permute_389);  view_518 = permute_389 = None
    view_519: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_68, [8, 12, 196, 32]);  bmm_68 = None
    view_520: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_69, [8, 12, 196, 196]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    mul_398: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_520, alias_31);  view_520 = None
    sum_131: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_398, [-1], True)
    mul_399: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(alias_31, sum_131);  alias_31 = sum_131 = None
    sub_126: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_398, mul_399);  mul_398 = mul_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_400: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(sub_126, 0.1767766952966369);  sub_126 = None
    view_521: "f32[96, 196, 196]" = torch.ops.aten.view.default(mul_400, [96, 196, 196]);  mul_400 = None
    bmm_70: "f32[96, 32, 196]" = torch.ops.aten.bmm.default(permute_390, view_521);  permute_390 = None
    bmm_71: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_521, permute_391);  view_521 = permute_391 = None
    view_522: "f32[8, 12, 32, 196]" = torch.ops.aten.view.default(bmm_70, [8, 12, 32, 196]);  bmm_70 = None
    view_523: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_71, [8, 12, 196, 32]);  bmm_71 = None
    permute_392: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_522, [0, 1, 3, 2]);  view_522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    cat_11: "f32[24, 12, 196, 32]" = torch.ops.aten.cat.default([view_523, permute_392, view_519]);  view_523 = permute_392 = view_519 = None
    view_524: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.view.default(cat_11, [3, 8, 12, 196, 32]);  cat_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_393: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.permute.default(view_524, [1, 3, 0, 2, 4]);  view_524 = None
    clone_228: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.clone.default(permute_393, memory_format = torch.contiguous_format);  permute_393 = None
    view_525: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(clone_228, [8, 14, 14, 1152]);  clone_228 = None
    view_526: "f32[1568, 1152]" = torch.ops.aten.view.default(view_525, [1568, 1152]);  view_525 = None
    permute_394: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_526, [1, 0])
    mm_105: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_394, view_184);  permute_394 = view_184 = None
    permute_395: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    mm_106: "f32[1568, 384]" = torch.ops.aten.mm.default(view_526, permute_396);  view_526 = permute_396 = None
    view_527: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_106, [8, 14, 14, 384]);  mm_106 = None
    permute_397: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_395, [1, 0]);  permute_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    mul_402: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_527, primals_145);  primals_145 = None
    mul_403: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_402, 384)
    sum_132: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_402, [3], True)
    mul_404: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_402, mul_109);  mul_402 = None
    sum_133: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_404, [3], True);  mul_404 = None
    mul_405: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_109, sum_133);  sum_133 = None
    sub_128: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_403, sum_132);  mul_403 = sum_132 = None
    sub_129: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_128, mul_405);  sub_128 = mul_405 = None
    mul_406: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_38, sub_129);  div_38 = sub_129 = None
    mul_407: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_527, mul_109);  mul_109 = None
    sum_134: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_407, [0, 1, 2]);  mul_407 = None
    sum_135: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_527, [0, 1, 2]);  view_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_220: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_219, mul_406);  add_219 = mul_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_528: "f32[1568, 384]" = torch.ops.aten.view.default(add_220, [1568, 384])
    mm_107: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_528, permute_398);  permute_398 = None
    permute_399: "f32[384, 1568]" = torch.ops.aten.permute.default(view_528, [1, 0])
    mm_108: "f32[384, 1152]" = torch.ops.aten.mm.default(permute_399, view_182);  permute_399 = view_182 = None
    permute_400: "f32[1152, 384]" = torch.ops.aten.permute.default(mm_108, [1, 0]);  mm_108 = None
    sum_136: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_528, [0], True);  view_528 = None
    view_529: "f32[384]" = torch.ops.aten.view.default(sum_136, [384]);  sum_136 = None
    permute_401: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_400, [1, 0]);  permute_400 = None
    view_530: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_107, [8, 14, 14, 1152]);  mm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_409: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(add_111, 0.5);  add_111 = None
    mul_410: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_181, view_181)
    mul_411: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_410, -0.5);  mul_410 = None
    exp_29: "f32[8, 14, 14, 1152]" = torch.ops.aten.exp.default(mul_411);  mul_411 = None
    mul_412: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
    mul_413: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_181, mul_412);  view_181 = mul_412 = None
    add_222: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(mul_409, mul_413);  mul_409 = mul_413 = None
    mul_414: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_530, add_222);  view_530 = add_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_531: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_414, [1568, 1152]);  mul_414 = None
    mm_109: "f32[1568, 384]" = torch.ops.aten.mm.default(view_531, permute_402);  permute_402 = None
    permute_403: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_531, [1, 0])
    mm_110: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_403, view_180);  permute_403 = view_180 = None
    permute_404: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_110, [1, 0]);  mm_110 = None
    sum_137: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_531, [0], True);  view_531 = None
    view_532: "f32[1152]" = torch.ops.aten.view.default(sum_137, [1152]);  sum_137 = None
    permute_405: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_404, [1, 0]);  permute_404 = None
    view_533: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_109, [8, 14, 14, 384]);  mm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    mul_416: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_533, primals_139);  primals_139 = None
    mul_417: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_416, 384)
    sum_138: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_416, [3], True)
    mul_418: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_416, mul_104);  mul_416 = None
    sum_139: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_418, [3], True);  mul_418 = None
    mul_419: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_104, sum_139);  sum_139 = None
    sub_131: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_417, sum_138);  mul_417 = sum_138 = None
    sub_132: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_131, mul_419);  sub_131 = mul_419 = None
    mul_420: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_39, sub_132);  div_39 = sub_132 = None
    mul_421: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_533, mul_104);  mul_104 = None
    sum_140: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_421, [0, 1, 2]);  mul_421 = None
    sum_141: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_533, [0, 1, 2]);  view_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_223: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_220, mul_420);  add_220 = mul_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_534: "f32[1568, 384]" = torch.ops.aten.view.default(add_223, [1568, 384])
    mm_111: "f32[1568, 384]" = torch.ops.aten.mm.default(view_534, permute_406);  permute_406 = None
    permute_407: "f32[384, 1568]" = torch.ops.aten.permute.default(view_534, [1, 0])
    mm_112: "f32[384, 384]" = torch.ops.aten.mm.default(permute_407, view_178);  permute_407 = view_178 = None
    permute_408: "f32[384, 384]" = torch.ops.aten.permute.default(mm_112, [1, 0]);  mm_112 = None
    sum_142: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_534, [0], True);  view_534 = None
    view_535: "f32[384]" = torch.ops.aten.view.default(sum_142, [384]);  sum_142 = None
    permute_409: "f32[384, 384]" = torch.ops.aten.permute.default(permute_408, [1, 0]);  permute_408 = None
    view_536: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_111, [8, 14, 14, 384]);  mm_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    view_537: "f32[8, 196, 12, 32]" = torch.ops.aten.view.default(view_536, [8, 196, 12, 32]);  view_536 = None
    permute_410: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_537, [0, 2, 1, 3]);  view_537 = None
    clone_231: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(permute_410, memory_format = torch.contiguous_format);  permute_410 = None
    view_538: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_231, [96, 196, 32]);  clone_231 = None
    bmm_72: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(permute_411, view_538);  permute_411 = None
    bmm_73: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_538, permute_412);  view_538 = permute_412 = None
    view_539: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_72, [8, 12, 196, 32]);  bmm_72 = None
    view_540: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_73, [8, 12, 196, 196]);  bmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    mul_422: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_540, alias_32);  view_540 = None
    sum_143: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_422, [-1], True)
    mul_423: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(alias_32, sum_143);  alias_32 = sum_143 = None
    sub_133: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_422, mul_423);  mul_422 = mul_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_424: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(sub_133, 0.1767766952966369);  sub_133 = None
    view_541: "f32[96, 196, 196]" = torch.ops.aten.view.default(mul_424, [96, 196, 196]);  mul_424 = None
    bmm_74: "f32[96, 32, 196]" = torch.ops.aten.bmm.default(permute_413, view_541);  permute_413 = None
    bmm_75: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_541, permute_414);  view_541 = permute_414 = None
    view_542: "f32[8, 12, 32, 196]" = torch.ops.aten.view.default(bmm_74, [8, 12, 32, 196]);  bmm_74 = None
    view_543: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_75, [8, 12, 196, 32]);  bmm_75 = None
    permute_415: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_542, [0, 1, 3, 2]);  view_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    cat_12: "f32[24, 12, 196, 32]" = torch.ops.aten.cat.default([view_543, permute_415, view_539]);  view_543 = permute_415 = view_539 = None
    view_544: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.view.default(cat_12, [3, 8, 12, 196, 32]);  cat_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_416: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.permute.default(view_544, [1, 3, 0, 2, 4]);  view_544 = None
    clone_232: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.clone.default(permute_416, memory_format = torch.contiguous_format);  permute_416 = None
    view_545: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(clone_232, [8, 14, 14, 1152]);  clone_232 = None
    view_546: "f32[1568, 1152]" = torch.ops.aten.view.default(view_545, [1568, 1152]);  view_545 = None
    permute_417: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_546, [1, 0])
    mm_113: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_417, view_168);  permute_417 = view_168 = None
    permute_418: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    mm_114: "f32[1568, 384]" = torch.ops.aten.mm.default(view_546, permute_419);  view_546 = permute_419 = None
    view_547: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_114, [8, 14, 14, 384]);  mm_114 = None
    permute_420: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_418, [1, 0]);  permute_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    mul_426: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_547, primals_134);  primals_134 = None
    mul_427: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_426, 384)
    sum_144: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_426, [3], True)
    mul_428: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_426, mul_101);  mul_426 = None
    sum_145: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_428, [3], True);  mul_428 = None
    mul_429: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_101, sum_145);  sum_145 = None
    sub_135: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_427, sum_144);  mul_427 = sum_144 = None
    sub_136: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_135, mul_429);  sub_135 = mul_429 = None
    mul_430: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_40, sub_136);  div_40 = sub_136 = None
    mul_431: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_547, mul_101);  mul_101 = None
    sum_146: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_431, [0, 1, 2]);  mul_431 = None
    sum_147: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_547, [0, 1, 2]);  view_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_224: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_223, mul_430);  add_223 = mul_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_548: "f32[1568, 384]" = torch.ops.aten.view.default(add_224, [1568, 384])
    mm_115: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_548, permute_421);  permute_421 = None
    permute_422: "f32[384, 1568]" = torch.ops.aten.permute.default(view_548, [1, 0])
    mm_116: "f32[384, 1152]" = torch.ops.aten.mm.default(permute_422, view_166);  permute_422 = view_166 = None
    permute_423: "f32[1152, 384]" = torch.ops.aten.permute.default(mm_116, [1, 0]);  mm_116 = None
    sum_148: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_548, [0], True);  view_548 = None
    view_549: "f32[384]" = torch.ops.aten.view.default(sum_148, [384]);  sum_148 = None
    permute_424: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_423, [1, 0]);  permute_423 = None
    view_550: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_115, [8, 14, 14, 1152]);  mm_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_433: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(add_104, 0.5);  add_104 = None
    mul_434: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_165, view_165)
    mul_435: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_434, -0.5);  mul_434 = None
    exp_30: "f32[8, 14, 14, 1152]" = torch.ops.aten.exp.default(mul_435);  mul_435 = None
    mul_436: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(exp_30, 0.3989422804014327);  exp_30 = None
    mul_437: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_165, mul_436);  view_165 = mul_436 = None
    add_226: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(mul_433, mul_437);  mul_433 = mul_437 = None
    mul_438: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_550, add_226);  view_550 = add_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_551: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_438, [1568, 1152]);  mul_438 = None
    mm_117: "f32[1568, 384]" = torch.ops.aten.mm.default(view_551, permute_425);  permute_425 = None
    permute_426: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_551, [1, 0])
    mm_118: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_426, view_164);  permute_426 = view_164 = None
    permute_427: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_118, [1, 0]);  mm_118 = None
    sum_149: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_551, [0], True);  view_551 = None
    view_552: "f32[1152]" = torch.ops.aten.view.default(sum_149, [1152]);  sum_149 = None
    permute_428: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_427, [1, 0]);  permute_427 = None
    view_553: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_117, [8, 14, 14, 384]);  mm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    mul_440: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_553, primals_128);  primals_128 = None
    mul_441: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_440, 384)
    sum_150: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_440, [3], True)
    mul_442: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_440, mul_96);  mul_440 = None
    sum_151: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_442, [3], True);  mul_442 = None
    mul_443: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_96, sum_151);  sum_151 = None
    sub_138: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_441, sum_150);  mul_441 = sum_150 = None
    sub_139: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_138, mul_443);  sub_138 = mul_443 = None
    mul_444: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_41, sub_139);  div_41 = sub_139 = None
    mul_445: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_553, mul_96);  mul_96 = None
    sum_152: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_445, [0, 1, 2]);  mul_445 = None
    sum_153: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_553, [0, 1, 2]);  view_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_227: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_224, mul_444);  add_224 = mul_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_554: "f32[1568, 384]" = torch.ops.aten.view.default(add_227, [1568, 384])
    mm_119: "f32[1568, 384]" = torch.ops.aten.mm.default(view_554, permute_429);  permute_429 = None
    permute_430: "f32[384, 1568]" = torch.ops.aten.permute.default(view_554, [1, 0])
    mm_120: "f32[384, 384]" = torch.ops.aten.mm.default(permute_430, view_162);  permute_430 = view_162 = None
    permute_431: "f32[384, 384]" = torch.ops.aten.permute.default(mm_120, [1, 0]);  mm_120 = None
    sum_154: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_554, [0], True);  view_554 = None
    view_555: "f32[384]" = torch.ops.aten.view.default(sum_154, [384]);  sum_154 = None
    permute_432: "f32[384, 384]" = torch.ops.aten.permute.default(permute_431, [1, 0]);  permute_431 = None
    view_556: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_119, [8, 14, 14, 384]);  mm_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    view_557: "f32[8, 196, 12, 32]" = torch.ops.aten.view.default(view_556, [8, 196, 12, 32]);  view_556 = None
    permute_433: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_557, [0, 2, 1, 3]);  view_557 = None
    clone_235: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(permute_433, memory_format = torch.contiguous_format);  permute_433 = None
    view_558: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_235, [96, 196, 32]);  clone_235 = None
    bmm_76: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(permute_434, view_558);  permute_434 = None
    bmm_77: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_558, permute_435);  view_558 = permute_435 = None
    view_559: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_76, [8, 12, 196, 32]);  bmm_76 = None
    view_560: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_77, [8, 12, 196, 196]);  bmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    mul_446: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_560, alias_33);  view_560 = None
    sum_155: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_446, [-1], True)
    mul_447: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(alias_33, sum_155);  alias_33 = sum_155 = None
    sub_140: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_446, mul_447);  mul_446 = mul_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_448: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(sub_140, 0.1767766952966369);  sub_140 = None
    view_561: "f32[96, 196, 196]" = torch.ops.aten.view.default(mul_448, [96, 196, 196]);  mul_448 = None
    bmm_78: "f32[96, 32, 196]" = torch.ops.aten.bmm.default(permute_436, view_561);  permute_436 = None
    bmm_79: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_561, permute_437);  view_561 = permute_437 = None
    view_562: "f32[8, 12, 32, 196]" = torch.ops.aten.view.default(bmm_78, [8, 12, 32, 196]);  bmm_78 = None
    view_563: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_79, [8, 12, 196, 32]);  bmm_79 = None
    permute_438: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_562, [0, 1, 3, 2]);  view_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    cat_13: "f32[24, 12, 196, 32]" = torch.ops.aten.cat.default([view_563, permute_438, view_559]);  view_563 = permute_438 = view_559 = None
    view_564: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.view.default(cat_13, [3, 8, 12, 196, 32]);  cat_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_439: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.permute.default(view_564, [1, 3, 0, 2, 4]);  view_564 = None
    clone_236: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.clone.default(permute_439, memory_format = torch.contiguous_format);  permute_439 = None
    view_565: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(clone_236, [8, 14, 14, 1152]);  clone_236 = None
    view_566: "f32[1568, 1152]" = torch.ops.aten.view.default(view_565, [1568, 1152]);  view_565 = None
    permute_440: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_566, [1, 0])
    mm_121: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_440, view_152);  permute_440 = view_152 = None
    permute_441: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    mm_122: "f32[1568, 384]" = torch.ops.aten.mm.default(view_566, permute_442);  view_566 = permute_442 = None
    view_567: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_122, [8, 14, 14, 384]);  mm_122 = None
    permute_443: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_441, [1, 0]);  permute_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    mul_450: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_567, primals_123);  primals_123 = None
    mul_451: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_450, 384)
    sum_156: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_450, [3], True)
    mul_452: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_450, mul_93);  mul_450 = None
    sum_157: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_452, [3], True);  mul_452 = None
    mul_453: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_93, sum_157);  sum_157 = None
    sub_142: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_451, sum_156);  mul_451 = sum_156 = None
    sub_143: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_142, mul_453);  sub_142 = mul_453 = None
    mul_454: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_42, sub_143);  div_42 = sub_143 = None
    mul_455: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_567, mul_93);  mul_93 = None
    sum_158: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_455, [0, 1, 2]);  mul_455 = None
    sum_159: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_567, [0, 1, 2]);  view_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_228: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_227, mul_454);  add_227 = mul_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_568: "f32[1568, 384]" = torch.ops.aten.view.default(add_228, [1568, 384])
    mm_123: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_568, permute_444);  permute_444 = None
    permute_445: "f32[384, 1568]" = torch.ops.aten.permute.default(view_568, [1, 0])
    mm_124: "f32[384, 1152]" = torch.ops.aten.mm.default(permute_445, view_150);  permute_445 = view_150 = None
    permute_446: "f32[1152, 384]" = torch.ops.aten.permute.default(mm_124, [1, 0]);  mm_124 = None
    sum_160: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_568, [0], True);  view_568 = None
    view_569: "f32[384]" = torch.ops.aten.view.default(sum_160, [384]);  sum_160 = None
    permute_447: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_446, [1, 0]);  permute_446 = None
    view_570: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_123, [8, 14, 14, 1152]);  mm_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_457: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(add_97, 0.5);  add_97 = None
    mul_458: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_149, view_149)
    mul_459: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_458, -0.5);  mul_458 = None
    exp_31: "f32[8, 14, 14, 1152]" = torch.ops.aten.exp.default(mul_459);  mul_459 = None
    mul_460: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(exp_31, 0.3989422804014327);  exp_31 = None
    mul_461: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_149, mul_460);  view_149 = mul_460 = None
    add_230: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(mul_457, mul_461);  mul_457 = mul_461 = None
    mul_462: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_570, add_230);  view_570 = add_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_571: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_462, [1568, 1152]);  mul_462 = None
    mm_125: "f32[1568, 384]" = torch.ops.aten.mm.default(view_571, permute_448);  permute_448 = None
    permute_449: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_571, [1, 0])
    mm_126: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_449, view_148);  permute_449 = view_148 = None
    permute_450: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_126, [1, 0]);  mm_126 = None
    sum_161: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_571, [0], True);  view_571 = None
    view_572: "f32[1152]" = torch.ops.aten.view.default(sum_161, [1152]);  sum_161 = None
    permute_451: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_450, [1, 0]);  permute_450 = None
    view_573: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_125, [8, 14, 14, 384]);  mm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    mul_464: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_573, primals_117);  primals_117 = None
    mul_465: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_464, 384)
    sum_162: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_464, [3], True)
    mul_466: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_464, mul_88);  mul_464 = None
    sum_163: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_466, [3], True);  mul_466 = None
    mul_467: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_88, sum_163);  sum_163 = None
    sub_145: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_465, sum_162);  mul_465 = sum_162 = None
    sub_146: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_145, mul_467);  sub_145 = mul_467 = None
    mul_468: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_43, sub_146);  div_43 = sub_146 = None
    mul_469: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_573, mul_88);  mul_88 = None
    sum_164: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_469, [0, 1, 2]);  mul_469 = None
    sum_165: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_573, [0, 1, 2]);  view_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_231: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_228, mul_468);  add_228 = mul_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_574: "f32[1568, 384]" = torch.ops.aten.view.default(add_231, [1568, 384])
    mm_127: "f32[1568, 384]" = torch.ops.aten.mm.default(view_574, permute_452);  permute_452 = None
    permute_453: "f32[384, 1568]" = torch.ops.aten.permute.default(view_574, [1, 0])
    mm_128: "f32[384, 384]" = torch.ops.aten.mm.default(permute_453, view_146);  permute_453 = view_146 = None
    permute_454: "f32[384, 384]" = torch.ops.aten.permute.default(mm_128, [1, 0]);  mm_128 = None
    sum_166: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_574, [0], True);  view_574 = None
    view_575: "f32[384]" = torch.ops.aten.view.default(sum_166, [384]);  sum_166 = None
    permute_455: "f32[384, 384]" = torch.ops.aten.permute.default(permute_454, [1, 0]);  permute_454 = None
    view_576: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_127, [8, 14, 14, 384]);  mm_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    view_577: "f32[8, 196, 12, 32]" = torch.ops.aten.view.default(view_576, [8, 196, 12, 32]);  view_576 = None
    permute_456: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_577, [0, 2, 1, 3]);  view_577 = None
    clone_239: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(permute_456, memory_format = torch.contiguous_format);  permute_456 = None
    view_578: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_239, [96, 196, 32]);  clone_239 = None
    bmm_80: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(permute_457, view_578);  permute_457 = None
    bmm_81: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_578, permute_458);  view_578 = permute_458 = None
    view_579: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_80, [8, 12, 196, 32]);  bmm_80 = None
    view_580: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_81, [8, 12, 196, 196]);  bmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    mul_470: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_580, alias_34);  view_580 = None
    sum_167: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_470, [-1], True)
    mul_471: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(alias_34, sum_167);  alias_34 = sum_167 = None
    sub_147: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_470, mul_471);  mul_470 = mul_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_472: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(sub_147, 0.1767766952966369);  sub_147 = None
    view_581: "f32[96, 196, 196]" = torch.ops.aten.view.default(mul_472, [96, 196, 196]);  mul_472 = None
    bmm_82: "f32[96, 32, 196]" = torch.ops.aten.bmm.default(permute_459, view_581);  permute_459 = None
    bmm_83: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_581, permute_460);  view_581 = permute_460 = None
    view_582: "f32[8, 12, 32, 196]" = torch.ops.aten.view.default(bmm_82, [8, 12, 32, 196]);  bmm_82 = None
    view_583: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_83, [8, 12, 196, 32]);  bmm_83 = None
    permute_461: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_582, [0, 1, 3, 2]);  view_582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    cat_14: "f32[24, 12, 196, 32]" = torch.ops.aten.cat.default([view_583, permute_461, view_579]);  view_583 = permute_461 = view_579 = None
    view_584: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.view.default(cat_14, [3, 8, 12, 196, 32]);  cat_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_462: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.permute.default(view_584, [1, 3, 0, 2, 4]);  view_584 = None
    clone_240: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.clone.default(permute_462, memory_format = torch.contiguous_format);  permute_462 = None
    view_585: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(clone_240, [8, 14, 14, 1152]);  clone_240 = None
    view_586: "f32[1568, 1152]" = torch.ops.aten.view.default(view_585, [1568, 1152]);  view_585 = None
    permute_463: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_586, [1, 0])
    mm_129: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_463, view_136);  permute_463 = view_136 = None
    permute_464: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    mm_130: "f32[1568, 384]" = torch.ops.aten.mm.default(view_586, permute_465);  view_586 = permute_465 = None
    view_587: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_130, [8, 14, 14, 384]);  mm_130 = None
    permute_466: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_464, [1, 0]);  permute_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    mul_474: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_587, primals_112);  primals_112 = None
    mul_475: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_474, 384)
    sum_168: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_474, [3], True)
    mul_476: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_474, mul_85);  mul_474 = None
    sum_169: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_476, [3], True);  mul_476 = None
    mul_477: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_85, sum_169);  sum_169 = None
    sub_149: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_475, sum_168);  mul_475 = sum_168 = None
    sub_150: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_149, mul_477);  sub_149 = mul_477 = None
    mul_478: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_44, sub_150);  div_44 = sub_150 = None
    mul_479: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_587, mul_85);  mul_85 = None
    sum_170: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_479, [0, 1, 2]);  mul_479 = None
    sum_171: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_587, [0, 1, 2]);  view_587 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_232: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_231, mul_478);  add_231 = mul_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_588: "f32[1568, 384]" = torch.ops.aten.view.default(add_232, [1568, 384])
    mm_131: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_588, permute_467);  permute_467 = None
    permute_468: "f32[384, 1568]" = torch.ops.aten.permute.default(view_588, [1, 0])
    mm_132: "f32[384, 1152]" = torch.ops.aten.mm.default(permute_468, view_134);  permute_468 = view_134 = None
    permute_469: "f32[1152, 384]" = torch.ops.aten.permute.default(mm_132, [1, 0]);  mm_132 = None
    sum_172: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_588, [0], True);  view_588 = None
    view_589: "f32[384]" = torch.ops.aten.view.default(sum_172, [384]);  sum_172 = None
    permute_470: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_469, [1, 0]);  permute_469 = None
    view_590: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_131, [8, 14, 14, 1152]);  mm_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_481: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(add_90, 0.5);  add_90 = None
    mul_482: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_133, view_133)
    mul_483: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_482, -0.5);  mul_482 = None
    exp_32: "f32[8, 14, 14, 1152]" = torch.ops.aten.exp.default(mul_483);  mul_483 = None
    mul_484: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(exp_32, 0.3989422804014327);  exp_32 = None
    mul_485: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_133, mul_484);  view_133 = mul_484 = None
    add_234: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(mul_481, mul_485);  mul_481 = mul_485 = None
    mul_486: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_590, add_234);  view_590 = add_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_591: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_486, [1568, 1152]);  mul_486 = None
    mm_133: "f32[1568, 384]" = torch.ops.aten.mm.default(view_591, permute_471);  permute_471 = None
    permute_472: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_591, [1, 0])
    mm_134: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_472, view_132);  permute_472 = view_132 = None
    permute_473: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_134, [1, 0]);  mm_134 = None
    sum_173: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_591, [0], True);  view_591 = None
    view_592: "f32[1152]" = torch.ops.aten.view.default(sum_173, [1152]);  sum_173 = None
    permute_474: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_473, [1, 0]);  permute_473 = None
    view_593: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_133, [8, 14, 14, 384]);  mm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    mul_488: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_593, primals_106);  primals_106 = None
    mul_489: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_488, 384)
    sum_174: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_488, [3], True)
    mul_490: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_488, mul_80);  mul_488 = None
    sum_175: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_490, [3], True);  mul_490 = None
    mul_491: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_80, sum_175);  sum_175 = None
    sub_152: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_489, sum_174);  mul_489 = sum_174 = None
    sub_153: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_152, mul_491);  sub_152 = mul_491 = None
    mul_492: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_45, sub_153);  div_45 = sub_153 = None
    mul_493: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_593, mul_80);  mul_80 = None
    sum_176: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_493, [0, 1, 2]);  mul_493 = None
    sum_177: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_593, [0, 1, 2]);  view_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_235: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_232, mul_492);  add_232 = mul_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_594: "f32[1568, 384]" = torch.ops.aten.view.default(add_235, [1568, 384])
    mm_135: "f32[1568, 384]" = torch.ops.aten.mm.default(view_594, permute_475);  permute_475 = None
    permute_476: "f32[384, 1568]" = torch.ops.aten.permute.default(view_594, [1, 0])
    mm_136: "f32[384, 384]" = torch.ops.aten.mm.default(permute_476, view_130);  permute_476 = view_130 = None
    permute_477: "f32[384, 384]" = torch.ops.aten.permute.default(mm_136, [1, 0]);  mm_136 = None
    sum_178: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_594, [0], True);  view_594 = None
    view_595: "f32[384]" = torch.ops.aten.view.default(sum_178, [384]);  sum_178 = None
    permute_478: "f32[384, 384]" = torch.ops.aten.permute.default(permute_477, [1, 0]);  permute_477 = None
    view_596: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_135, [8, 14, 14, 384]);  mm_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    view_597: "f32[8, 196, 12, 32]" = torch.ops.aten.view.default(view_596, [8, 196, 12, 32]);  view_596 = None
    permute_479: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_597, [0, 2, 1, 3]);  view_597 = None
    clone_243: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(permute_479, memory_format = torch.contiguous_format);  permute_479 = None
    view_598: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_243, [96, 196, 32]);  clone_243 = None
    bmm_84: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(permute_480, view_598);  permute_480 = None
    bmm_85: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_598, permute_481);  view_598 = permute_481 = None
    view_599: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_84, [8, 12, 196, 32]);  bmm_84 = None
    view_600: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_85, [8, 12, 196, 196]);  bmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    mul_494: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_600, alias_35);  view_600 = None
    sum_179: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_494, [-1], True)
    mul_495: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(alias_35, sum_179);  alias_35 = sum_179 = None
    sub_154: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_494, mul_495);  mul_494 = mul_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_496: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(sub_154, 0.1767766952966369);  sub_154 = None
    view_601: "f32[96, 196, 196]" = torch.ops.aten.view.default(mul_496, [96, 196, 196]);  mul_496 = None
    bmm_86: "f32[96, 32, 196]" = torch.ops.aten.bmm.default(permute_482, view_601);  permute_482 = None
    bmm_87: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_601, permute_483);  view_601 = permute_483 = None
    view_602: "f32[8, 12, 32, 196]" = torch.ops.aten.view.default(bmm_86, [8, 12, 32, 196]);  bmm_86 = None
    view_603: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_87, [8, 12, 196, 32]);  bmm_87 = None
    permute_484: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_602, [0, 1, 3, 2]);  view_602 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    cat_15: "f32[24, 12, 196, 32]" = torch.ops.aten.cat.default([view_603, permute_484, view_599]);  view_603 = permute_484 = view_599 = None
    view_604: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.view.default(cat_15, [3, 8, 12, 196, 32]);  cat_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_485: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.permute.default(view_604, [1, 3, 0, 2, 4]);  view_604 = None
    clone_244: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.clone.default(permute_485, memory_format = torch.contiguous_format);  permute_485 = None
    view_605: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(clone_244, [8, 14, 14, 1152]);  clone_244 = None
    view_606: "f32[1568, 1152]" = torch.ops.aten.view.default(view_605, [1568, 1152]);  view_605 = None
    permute_486: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_606, [1, 0])
    mm_137: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_486, view_120);  permute_486 = view_120 = None
    permute_487: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    mm_138: "f32[1568, 384]" = torch.ops.aten.mm.default(view_606, permute_488);  view_606 = permute_488 = None
    view_607: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_138, [8, 14, 14, 384]);  mm_138 = None
    permute_489: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_487, [1, 0]);  permute_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    mul_498: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_607, primals_101);  primals_101 = None
    mul_499: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_498, 384)
    sum_180: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_498, [3], True)
    mul_500: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_498, mul_77);  mul_498 = None
    sum_181: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_500, [3], True);  mul_500 = None
    mul_501: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_77, sum_181);  sum_181 = None
    sub_156: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_499, sum_180);  mul_499 = sum_180 = None
    sub_157: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_156, mul_501);  sub_156 = mul_501 = None
    mul_502: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_46, sub_157);  div_46 = sub_157 = None
    mul_503: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_607, mul_77);  mul_77 = None
    sum_182: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_503, [0, 1, 2]);  mul_503 = None
    sum_183: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_607, [0, 1, 2]);  view_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_236: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_235, mul_502);  add_235 = mul_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_608: "f32[1568, 384]" = torch.ops.aten.view.default(add_236, [1568, 384])
    mm_139: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_608, permute_490);  permute_490 = None
    permute_491: "f32[384, 1568]" = torch.ops.aten.permute.default(view_608, [1, 0])
    mm_140: "f32[384, 1152]" = torch.ops.aten.mm.default(permute_491, view_118);  permute_491 = view_118 = None
    permute_492: "f32[1152, 384]" = torch.ops.aten.permute.default(mm_140, [1, 0]);  mm_140 = None
    sum_184: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_608, [0], True);  view_608 = None
    view_609: "f32[384]" = torch.ops.aten.view.default(sum_184, [384]);  sum_184 = None
    permute_493: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_492, [1, 0]);  permute_492 = None
    view_610: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_139, [8, 14, 14, 1152]);  mm_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_505: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(add_83, 0.5);  add_83 = None
    mul_506: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_117, view_117)
    mul_507: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_506, -0.5);  mul_506 = None
    exp_33: "f32[8, 14, 14, 1152]" = torch.ops.aten.exp.default(mul_507);  mul_507 = None
    mul_508: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(exp_33, 0.3989422804014327);  exp_33 = None
    mul_509: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_117, mul_508);  view_117 = mul_508 = None
    add_238: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(mul_505, mul_509);  mul_505 = mul_509 = None
    mul_510: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_610, add_238);  view_610 = add_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_611: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_510, [1568, 1152]);  mul_510 = None
    mm_141: "f32[1568, 384]" = torch.ops.aten.mm.default(view_611, permute_494);  permute_494 = None
    permute_495: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_611, [1, 0])
    mm_142: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_495, view_116);  permute_495 = view_116 = None
    permute_496: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_142, [1, 0]);  mm_142 = None
    sum_185: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_611, [0], True);  view_611 = None
    view_612: "f32[1152]" = torch.ops.aten.view.default(sum_185, [1152]);  sum_185 = None
    permute_497: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_496, [1, 0]);  permute_496 = None
    view_613: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_141, [8, 14, 14, 384]);  mm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    mul_512: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_613, primals_95);  primals_95 = None
    mul_513: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_512, 384)
    sum_186: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_512, [3], True)
    mul_514: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_512, mul_72);  mul_512 = None
    sum_187: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_514, [3], True);  mul_514 = None
    mul_515: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_72, sum_187);  sum_187 = None
    sub_159: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_513, sum_186);  mul_513 = sum_186 = None
    sub_160: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_159, mul_515);  sub_159 = mul_515 = None
    mul_516: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_47, sub_160);  div_47 = sub_160 = None
    mul_517: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_613, mul_72);  mul_72 = None
    sum_188: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_517, [0, 1, 2]);  mul_517 = None
    sum_189: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_613, [0, 1, 2]);  view_613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_239: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_236, mul_516);  add_236 = mul_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_614: "f32[1568, 384]" = torch.ops.aten.view.default(add_239, [1568, 384])
    mm_143: "f32[1568, 384]" = torch.ops.aten.mm.default(view_614, permute_498);  permute_498 = None
    permute_499: "f32[384, 1568]" = torch.ops.aten.permute.default(view_614, [1, 0])
    mm_144: "f32[384, 384]" = torch.ops.aten.mm.default(permute_499, view_114);  permute_499 = view_114 = None
    permute_500: "f32[384, 384]" = torch.ops.aten.permute.default(mm_144, [1, 0]);  mm_144 = None
    sum_190: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_614, [0], True);  view_614 = None
    view_615: "f32[384]" = torch.ops.aten.view.default(sum_190, [384]);  sum_190 = None
    permute_501: "f32[384, 384]" = torch.ops.aten.permute.default(permute_500, [1, 0]);  permute_500 = None
    view_616: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_143, [8, 14, 14, 384]);  mm_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    view_617: "f32[8, 196, 12, 32]" = torch.ops.aten.view.default(view_616, [8, 196, 12, 32]);  view_616 = None
    permute_502: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_617, [0, 2, 1, 3]);  view_617 = None
    clone_247: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(permute_502, memory_format = torch.contiguous_format);  permute_502 = None
    view_618: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_247, [96, 196, 32]);  clone_247 = None
    bmm_88: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(permute_503, view_618);  permute_503 = None
    bmm_89: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_618, permute_504);  view_618 = permute_504 = None
    view_619: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_88, [8, 12, 196, 32]);  bmm_88 = None
    view_620: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_89, [8, 12, 196, 196]);  bmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    mul_518: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_620, alias_36);  view_620 = None
    sum_191: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_518, [-1], True)
    mul_519: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(alias_36, sum_191);  alias_36 = sum_191 = None
    sub_161: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_518, mul_519);  mul_518 = mul_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_520: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(sub_161, 0.1767766952966369);  sub_161 = None
    view_621: "f32[96, 196, 196]" = torch.ops.aten.view.default(mul_520, [96, 196, 196]);  mul_520 = None
    bmm_90: "f32[96, 32, 196]" = torch.ops.aten.bmm.default(permute_505, view_621);  permute_505 = None
    bmm_91: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_621, permute_506);  view_621 = permute_506 = None
    view_622: "f32[8, 12, 32, 196]" = torch.ops.aten.view.default(bmm_90, [8, 12, 32, 196]);  bmm_90 = None
    view_623: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_91, [8, 12, 196, 32]);  bmm_91 = None
    permute_507: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_622, [0, 1, 3, 2]);  view_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    cat_16: "f32[24, 12, 196, 32]" = torch.ops.aten.cat.default([view_623, permute_507, view_619]);  view_623 = permute_507 = view_619 = None
    view_624: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.view.default(cat_16, [3, 8, 12, 196, 32]);  cat_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_508: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.permute.default(view_624, [1, 3, 0, 2, 4]);  view_624 = None
    clone_248: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.clone.default(permute_508, memory_format = torch.contiguous_format);  permute_508 = None
    view_625: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(clone_248, [8, 14, 14, 1152]);  clone_248 = None
    view_626: "f32[1568, 1152]" = torch.ops.aten.view.default(view_625, [1568, 1152]);  view_625 = None
    permute_509: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_626, [1, 0])
    mm_145: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_509, view_104);  permute_509 = view_104 = None
    permute_510: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    mm_146: "f32[1568, 384]" = torch.ops.aten.mm.default(view_626, permute_511);  view_626 = permute_511 = None
    view_627: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_146, [8, 14, 14, 384]);  mm_146 = None
    permute_512: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_510, [1, 0]);  permute_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    mul_522: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_627, primals_90);  primals_90 = None
    mul_523: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_522, 384)
    sum_192: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_522, [3], True)
    mul_524: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_522, mul_69);  mul_522 = None
    sum_193: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_524, [3], True);  mul_524 = None
    mul_525: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_69, sum_193);  sum_193 = None
    sub_163: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_523, sum_192);  mul_523 = sum_192 = None
    sub_164: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_163, mul_525);  sub_163 = mul_525 = None
    mul_526: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_48, sub_164);  div_48 = sub_164 = None
    mul_527: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_627, mul_69);  mul_69 = None
    sum_194: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_527, [0, 1, 2]);  mul_527 = None
    sum_195: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_627, [0, 1, 2]);  view_627 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_240: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_239, mul_526);  add_239 = mul_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_628: "f32[1568, 384]" = torch.ops.aten.view.default(add_240, [1568, 384])
    mm_147: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_628, permute_513);  permute_513 = None
    permute_514: "f32[384, 1568]" = torch.ops.aten.permute.default(view_628, [1, 0])
    mm_148: "f32[384, 1152]" = torch.ops.aten.mm.default(permute_514, view_102);  permute_514 = view_102 = None
    permute_515: "f32[1152, 384]" = torch.ops.aten.permute.default(mm_148, [1, 0]);  mm_148 = None
    sum_196: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_628, [0], True);  view_628 = None
    view_629: "f32[384]" = torch.ops.aten.view.default(sum_196, [384]);  sum_196 = None
    permute_516: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_515, [1, 0]);  permute_515 = None
    view_630: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_147, [8, 14, 14, 1152]);  mm_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_529: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(add_76, 0.5);  add_76 = None
    mul_530: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_101, view_101)
    mul_531: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_530, -0.5);  mul_530 = None
    exp_34: "f32[8, 14, 14, 1152]" = torch.ops.aten.exp.default(mul_531);  mul_531 = None
    mul_532: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(exp_34, 0.3989422804014327);  exp_34 = None
    mul_533: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_101, mul_532);  view_101 = mul_532 = None
    add_242: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(mul_529, mul_533);  mul_529 = mul_533 = None
    mul_534: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_630, add_242);  view_630 = add_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_631: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_534, [1568, 1152]);  mul_534 = None
    mm_149: "f32[1568, 384]" = torch.ops.aten.mm.default(view_631, permute_517);  permute_517 = None
    permute_518: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_631, [1, 0])
    mm_150: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_518, view_100);  permute_518 = view_100 = None
    permute_519: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_150, [1, 0]);  mm_150 = None
    sum_197: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_631, [0], True);  view_631 = None
    view_632: "f32[1152]" = torch.ops.aten.view.default(sum_197, [1152]);  sum_197 = None
    permute_520: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_519, [1, 0]);  permute_519 = None
    view_633: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_149, [8, 14, 14, 384]);  mm_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    mul_536: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_633, primals_84);  primals_84 = None
    mul_537: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_536, 384)
    sum_198: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_536, [3], True)
    mul_538: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_536, mul_64);  mul_536 = None
    sum_199: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_538, [3], True);  mul_538 = None
    mul_539: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_64, sum_199);  sum_199 = None
    sub_166: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_537, sum_198);  mul_537 = sum_198 = None
    sub_167: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_166, mul_539);  sub_166 = mul_539 = None
    mul_540: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_49, sub_167);  div_49 = sub_167 = None
    mul_541: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_633, mul_64);  mul_64 = None
    sum_200: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_541, [0, 1, 2]);  mul_541 = None
    sum_201: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_633, [0, 1, 2]);  view_633 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_243: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_240, mul_540);  add_240 = mul_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_634: "f32[1568, 384]" = torch.ops.aten.view.default(add_243, [1568, 384])
    mm_151: "f32[1568, 384]" = torch.ops.aten.mm.default(view_634, permute_521);  permute_521 = None
    permute_522: "f32[384, 1568]" = torch.ops.aten.permute.default(view_634, [1, 0])
    mm_152: "f32[384, 384]" = torch.ops.aten.mm.default(permute_522, view_98);  permute_522 = view_98 = None
    permute_523: "f32[384, 384]" = torch.ops.aten.permute.default(mm_152, [1, 0]);  mm_152 = None
    sum_202: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_634, [0], True);  view_634 = None
    view_635: "f32[384]" = torch.ops.aten.view.default(sum_202, [384]);  sum_202 = None
    permute_524: "f32[384, 384]" = torch.ops.aten.permute.default(permute_523, [1, 0]);  permute_523 = None
    view_636: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_151, [8, 14, 14, 384]);  mm_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    view_637: "f32[8, 196, 12, 32]" = torch.ops.aten.view.default(view_636, [8, 196, 12, 32]);  view_636 = None
    permute_525: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_637, [0, 2, 1, 3]);  view_637 = None
    clone_251: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(permute_525, memory_format = torch.contiguous_format);  permute_525 = None
    view_638: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_251, [96, 196, 32]);  clone_251 = None
    bmm_92: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(permute_526, view_638);  permute_526 = None
    bmm_93: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_638, permute_527);  view_638 = permute_527 = None
    view_639: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_92, [8, 12, 196, 32]);  bmm_92 = None
    view_640: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_93, [8, 12, 196, 196]);  bmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    mul_542: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_640, alias_37);  view_640 = None
    sum_203: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_542, [-1], True)
    mul_543: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(alias_37, sum_203);  alias_37 = sum_203 = None
    sub_168: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_542, mul_543);  mul_542 = mul_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_544: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(sub_168, 0.1767766952966369);  sub_168 = None
    view_641: "f32[96, 196, 196]" = torch.ops.aten.view.default(mul_544, [96, 196, 196]);  mul_544 = None
    bmm_94: "f32[96, 32, 196]" = torch.ops.aten.bmm.default(permute_528, view_641);  permute_528 = None
    bmm_95: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_641, permute_529);  view_641 = permute_529 = None
    view_642: "f32[8, 12, 32, 196]" = torch.ops.aten.view.default(bmm_94, [8, 12, 32, 196]);  bmm_94 = None
    view_643: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_95, [8, 12, 196, 32]);  bmm_95 = None
    permute_530: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_642, [0, 1, 3, 2]);  view_642 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    cat_17: "f32[24, 12, 196, 32]" = torch.ops.aten.cat.default([view_643, permute_530, view_639]);  view_643 = permute_530 = view_639 = None
    view_644: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.view.default(cat_17, [3, 8, 12, 196, 32]);  cat_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_531: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.permute.default(view_644, [1, 3, 0, 2, 4]);  view_644 = None
    clone_252: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.clone.default(permute_531, memory_format = torch.contiguous_format);  permute_531 = None
    view_645: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(clone_252, [8, 14, 14, 1152]);  clone_252 = None
    view_646: "f32[1568, 1152]" = torch.ops.aten.view.default(view_645, [1568, 1152]);  view_645 = None
    permute_532: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_646, [1, 0])
    mm_153: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_532, view_88);  permute_532 = view_88 = None
    permute_533: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_153, [1, 0]);  mm_153 = None
    mm_154: "f32[1568, 384]" = torch.ops.aten.mm.default(view_646, permute_534);  view_646 = permute_534 = None
    view_647: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_154, [8, 14, 14, 384]);  mm_154 = None
    permute_535: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_533, [1, 0]);  permute_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    mul_546: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_647, primals_79);  primals_79 = None
    mul_547: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_546, 384)
    sum_204: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_546, [3], True)
    mul_548: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_546, mul_61);  mul_546 = None
    sum_205: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_548, [3], True);  mul_548 = None
    mul_549: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_61, sum_205);  sum_205 = None
    sub_170: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_547, sum_204);  mul_547 = sum_204 = None
    sub_171: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_170, mul_549);  sub_170 = mul_549 = None
    mul_550: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_50, sub_171);  div_50 = sub_171 = None
    mul_551: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_647, mul_61);  mul_61 = None
    sum_206: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_551, [0, 1, 2]);  mul_551 = None
    sum_207: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_647, [0, 1, 2]);  view_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_244: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_243, mul_550);  add_243 = mul_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_648: "f32[1568, 384]" = torch.ops.aten.view.default(add_244, [1568, 384])
    mm_155: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_648, permute_536);  permute_536 = None
    permute_537: "f32[384, 1568]" = torch.ops.aten.permute.default(view_648, [1, 0])
    mm_156: "f32[384, 1152]" = torch.ops.aten.mm.default(permute_537, view_86);  permute_537 = view_86 = None
    permute_538: "f32[1152, 384]" = torch.ops.aten.permute.default(mm_156, [1, 0]);  mm_156 = None
    sum_208: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_648, [0], True);  view_648 = None
    view_649: "f32[384]" = torch.ops.aten.view.default(sum_208, [384]);  sum_208 = None
    permute_539: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_538, [1, 0]);  permute_538 = None
    view_650: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_155, [8, 14, 14, 1152]);  mm_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_553: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(add_69, 0.5);  add_69 = None
    mul_554: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_85, view_85)
    mul_555: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_554, -0.5);  mul_554 = None
    exp_35: "f32[8, 14, 14, 1152]" = torch.ops.aten.exp.default(mul_555);  mul_555 = None
    mul_556: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(exp_35, 0.3989422804014327);  exp_35 = None
    mul_557: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_85, mul_556);  view_85 = mul_556 = None
    add_246: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(mul_553, mul_557);  mul_553 = mul_557 = None
    mul_558: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_650, add_246);  view_650 = add_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_651: "f32[1568, 1152]" = torch.ops.aten.view.default(mul_558, [1568, 1152]);  mul_558 = None
    mm_157: "f32[1568, 384]" = torch.ops.aten.mm.default(view_651, permute_540);  permute_540 = None
    permute_541: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_651, [1, 0])
    mm_158: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_541, view_84);  permute_541 = view_84 = None
    permute_542: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_158, [1, 0]);  mm_158 = None
    sum_209: "f32[1, 1152]" = torch.ops.aten.sum.dim_IntList(view_651, [0], True);  view_651 = None
    view_652: "f32[1152]" = torch.ops.aten.view.default(sum_209, [1152]);  sum_209 = None
    permute_543: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_542, [1, 0]);  permute_542 = None
    view_653: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_157, [8, 14, 14, 384]);  mm_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    mul_560: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_653, primals_73);  primals_73 = None
    mul_561: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_560, 384)
    sum_210: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_560, [3], True)
    mul_562: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_560, mul_56);  mul_560 = None
    sum_211: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_562, [3], True);  mul_562 = None
    mul_563: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_56, sum_211);  sum_211 = None
    sub_173: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_561, sum_210);  mul_561 = sum_210 = None
    sub_174: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_173, mul_563);  sub_173 = mul_563 = None
    mul_564: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_51, sub_174);  div_51 = sub_174 = None
    mul_565: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_653, mul_56);  mul_56 = None
    sum_212: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_565, [0, 1, 2]);  mul_565 = None
    sum_213: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_653, [0, 1, 2]);  view_653 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_247: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_244, mul_564);  add_244 = mul_564 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_654: "f32[1568, 384]" = torch.ops.aten.view.default(add_247, [1568, 384])
    mm_159: "f32[1568, 384]" = torch.ops.aten.mm.default(view_654, permute_544);  permute_544 = None
    permute_545: "f32[384, 1568]" = torch.ops.aten.permute.default(view_654, [1, 0])
    mm_160: "f32[384, 384]" = torch.ops.aten.mm.default(permute_545, view_82);  permute_545 = view_82 = None
    permute_546: "f32[384, 384]" = torch.ops.aten.permute.default(mm_160, [1, 0]);  mm_160 = None
    sum_214: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_654, [0], True);  view_654 = None
    view_655: "f32[384]" = torch.ops.aten.view.default(sum_214, [384]);  sum_214 = None
    permute_547: "f32[384, 384]" = torch.ops.aten.permute.default(permute_546, [1, 0]);  permute_546 = None
    view_656: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_159, [8, 14, 14, 384]);  mm_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    view_657: "f32[8, 196, 12, 32]" = torch.ops.aten.view.default(view_656, [8, 196, 12, 32]);  view_656 = None
    permute_548: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_657, [0, 2, 1, 3]);  view_657 = None
    clone_255: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(permute_548, memory_format = torch.contiguous_format);  permute_548 = None
    view_658: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_255, [96, 196, 32]);  clone_255 = None
    bmm_96: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(permute_549, view_658);  permute_549 = None
    bmm_97: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_658, permute_550);  view_658 = permute_550 = None
    view_659: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_96, [8, 12, 196, 32]);  bmm_96 = None
    view_660: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_97, [8, 12, 196, 196]);  bmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    mul_566: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_660, alias_38);  view_660 = None
    sum_215: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_566, [-1], True)
    mul_567: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(alias_38, sum_215);  alias_38 = sum_215 = None
    sub_175: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_566, mul_567);  mul_566 = mul_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_568: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(sub_175, 0.1767766952966369);  sub_175 = None
    view_661: "f32[96, 196, 196]" = torch.ops.aten.view.default(mul_568, [96, 196, 196]);  mul_568 = None
    bmm_98: "f32[96, 32, 196]" = torch.ops.aten.bmm.default(permute_551, view_661);  permute_551 = None
    bmm_99: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_661, permute_552);  view_661 = permute_552 = None
    view_662: "f32[8, 12, 32, 196]" = torch.ops.aten.view.default(bmm_98, [8, 12, 32, 196]);  bmm_98 = None
    view_663: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_99, [8, 12, 196, 32]);  bmm_99 = None
    permute_553: "f32[8, 12, 196, 32]" = torch.ops.aten.permute.default(view_662, [0, 1, 3, 2]);  view_662 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    cat_18: "f32[24, 12, 196, 32]" = torch.ops.aten.cat.default([view_663, permute_553, view_659]);  view_663 = permute_553 = view_659 = None
    view_664: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.view.default(cat_18, [3, 8, 12, 196, 32]);  cat_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_554: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.permute.default(view_664, [1, 3, 0, 2, 4]);  view_664 = None
    clone_256: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.clone.default(permute_554, memory_format = torch.contiguous_format);  permute_554 = None
    view_665: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(clone_256, [8, 14, 14, 1152]);  clone_256 = None
    view_666: "f32[1568, 1152]" = torch.ops.aten.view.default(view_665, [1568, 1152]);  view_665 = None
    permute_555: "f32[1152, 1568]" = torch.ops.aten.permute.default(view_666, [1, 0])
    mm_161: "f32[1152, 384]" = torch.ops.aten.mm.default(permute_555, view_72);  permute_555 = view_72 = None
    permute_556: "f32[384, 1152]" = torch.ops.aten.permute.default(mm_161, [1, 0]);  mm_161 = None
    mm_162: "f32[1568, 384]" = torch.ops.aten.mm.default(view_666, permute_557);  view_666 = permute_557 = None
    view_667: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(mm_162, [8, 14, 14, 384]);  mm_162 = None
    permute_558: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_556, [1, 0]);  permute_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    mul_570: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_667, primals_68);  primals_68 = None
    mul_571: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_570, 384)
    sum_216: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_570, [3], True)
    mul_572: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_570, mul_53);  mul_570 = None
    sum_217: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_572, [3], True);  mul_572 = None
    mul_573: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_53, sum_217);  sum_217 = None
    sub_177: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_571, sum_216);  mul_571 = sum_216 = None
    sub_178: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_177, mul_573);  sub_177 = mul_573 = None
    mul_574: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_52, sub_178);  div_52 = sub_178 = None
    mul_575: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(view_667, mul_53);  mul_53 = None
    sum_218: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_575, [0, 1, 2]);  mul_575 = None
    sum_219: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_667, [0, 1, 2]);  view_667 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_248: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_247, mul_574);  add_247 = mul_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:620, code: x = x + self.pos_embed
    sum_220: "f32[1, 14, 14, 384]" = torch.ops.aten.sum.dim_IntList(add_248, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:373, code: x = x.permute(0, 2, 3, 1)
    permute_559: "f32[8, 384, 14, 14]" = torch.ops.aten.permute.default(add_248, [0, 3, 1, 2]);  add_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:372, code: x = self.proj(x)  # B, C, H, W
    sum_221: "f32[384]" = torch.ops.aten.sum.dim_IntList(permute_559, [0, 2, 3])
    convolution_backward = torch.ops.aten.convolution_backward.default(permute_559, permute_57, primals_66, [384], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  permute_559 = permute_57 = primals_66 = None
    getitem_136: "f32[8, 192, 28, 28]" = convolution_backward[0]
    getitem_137: "f32[384, 192, 2, 2]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:371, code: x = x.permute(0, 3, 1, 2)
    permute_560: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(getitem_136, [0, 2, 3, 1]);  getitem_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_258: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(permute_560, memory_format = torch.contiguous_format)
    view_668: "f32[6272, 192]" = torch.ops.aten.view.default(clone_258, [6272, 192]);  clone_258 = None
    mm_163: "f32[6272, 576]" = torch.ops.aten.mm.default(view_668, permute_561);  permute_561 = None
    permute_562: "f32[192, 6272]" = torch.ops.aten.permute.default(view_668, [1, 0])
    mm_164: "f32[192, 576]" = torch.ops.aten.mm.default(permute_562, view_70);  permute_562 = view_70 = None
    permute_563: "f32[576, 192]" = torch.ops.aten.permute.default(mm_164, [1, 0]);  mm_164 = None
    sum_222: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_668, [0], True);  view_668 = None
    view_669: "f32[192]" = torch.ops.aten.view.default(sum_222, [192]);  sum_222 = None
    permute_564: "f32[192, 576]" = torch.ops.aten.permute.default(permute_563, [1, 0]);  permute_563 = None
    view_670: "f32[8, 28, 28, 576]" = torch.ops.aten.view.default(mm_163, [8, 28, 28, 576]);  mm_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_577: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(add_61, 0.5);  add_61 = None
    mul_578: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_69, view_69)
    mul_579: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(mul_578, -0.5);  mul_578 = None
    exp_36: "f32[8, 28, 28, 576]" = torch.ops.aten.exp.default(mul_579);  mul_579 = None
    mul_580: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(exp_36, 0.3989422804014327);  exp_36 = None
    mul_581: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_69, mul_580);  view_69 = mul_580 = None
    add_250: "f32[8, 28, 28, 576]" = torch.ops.aten.add.Tensor(mul_577, mul_581);  mul_577 = mul_581 = None
    mul_582: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_670, add_250);  view_670 = add_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_671: "f32[6272, 576]" = torch.ops.aten.view.default(mul_582, [6272, 576]);  mul_582 = None
    mm_165: "f32[6272, 192]" = torch.ops.aten.mm.default(view_671, permute_565);  permute_565 = None
    permute_566: "f32[576, 6272]" = torch.ops.aten.permute.default(view_671, [1, 0])
    mm_166: "f32[576, 192]" = torch.ops.aten.mm.default(permute_566, view_68);  permute_566 = view_68 = None
    permute_567: "f32[192, 576]" = torch.ops.aten.permute.default(mm_166, [1, 0]);  mm_166 = None
    sum_223: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_671, [0], True);  view_671 = None
    view_672: "f32[576]" = torch.ops.aten.view.default(sum_223, [576]);  sum_223 = None
    permute_568: "f32[576, 192]" = torch.ops.aten.permute.default(permute_567, [1, 0]);  permute_567 = None
    view_673: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_165, [8, 28, 28, 192]);  mm_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    mul_584: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(view_673, primals_60);  primals_60 = None
    mul_585: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_584, 192)
    sum_224: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_584, [3], True)
    mul_586: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_584, mul_48);  mul_584 = None
    sum_225: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_586, [3], True);  mul_586 = None
    mul_587: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_48, sum_225);  sum_225 = None
    sub_180: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(mul_585, sum_224);  mul_585 = sum_224 = None
    sub_181: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(sub_180, mul_587);  sub_180 = mul_587 = None
    mul_588: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(div_53, sub_181);  div_53 = sub_181 = None
    mul_589: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(view_673, mul_48);  mul_48 = None
    sum_226: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_589, [0, 1, 2]);  mul_589 = None
    sum_227: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_673, [0, 1, 2]);  view_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_251: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(permute_560, mul_588);  permute_560 = mul_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:90, code: x = self.proj(x.permute(0, 2, 3, 1))
    sum_228: "f32[1, 1, 1, 192]" = torch.ops.aten.sum.dim_IntList(add_251, [0, 1, 2], True)
    view_674: "f32[192]" = torch.ops.aten.view.default(sum_228, [192]);  sum_228 = None
    clone_260: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_251, memory_format = torch.contiguous_format)
    view_675: "f32[6272, 192]" = torch.ops.aten.view.default(clone_260, [6272, 192]);  clone_260 = None
    permute_569: "f32[192, 6272]" = torch.ops.aten.permute.default(view_675, [1, 0])
    mm_167: "f32[192, 192]" = torch.ops.aten.mm.default(permute_569, view_66);  permute_569 = view_66 = None
    permute_570: "f32[192, 192]" = torch.ops.aten.permute.default(mm_167, [1, 0]);  mm_167 = None
    mm_168: "f32[6272, 192]" = torch.ops.aten.mm.default(view_675, permute_571);  view_675 = permute_571 = None
    view_676: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_168, [8, 28, 28, 192]);  mm_168 = None
    permute_572: "f32[192, 192]" = torch.ops.aten.permute.default(permute_570, [1, 0]);  permute_570 = None
    permute_573: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(view_676, [0, 3, 1, 2]);  view_676 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:88, code: x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
    constant_pad_nd_8: "f32[8, 192, 30, 30]" = torch.ops.aten.constant_pad_nd.default(permute_573, [1, 1, 1, 1], 0.0);  permute_573 = None
    slice_32: "f32[8, 192, 30, 30]" = torch.ops.aten.slice.Tensor(constant_pad_nd_8, 0, 0, 9223372036854775807);  constant_pad_nd_8 = None
    slice_33: "f32[8, 192, 30, 30]" = torch.ops.aten.slice.Tensor(slice_32, 1, 0, 9223372036854775807);  slice_32 = None
    index_4: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.index.Tensor(slice_33, [None, None, unsqueeze_17, add_17]);  slice_33 = None
    permute_574: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.permute.default(index_4, [0, 1, 2, 4, 3, 5]);  index_4 = None
    clone_261: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.clone.default(permute_574, memory_format = torch.contiguous_format);  permute_574 = None
    view_677: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_261, [8, 1728, 196]);  clone_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:87, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
    view_678: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.view.default(view_677, [8, 6, 32, 9, 196]);  view_677 = None
    permute_575: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.permute.default(view_678, [0, 1, 4, 3, 2]);  view_678 = None
    clone_262: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.clone.default(permute_575, memory_format = torch.contiguous_format);  permute_575 = None
    view_679: "f32[9408, 9, 32]" = torch.ops.aten.view.default(clone_262, [9408, 9, 32]);  clone_262 = None
    bmm_100: "f32[9408, 9, 32]" = torch.ops.aten.bmm.default(permute_576, view_679);  permute_576 = None
    bmm_101: "f32[9408, 9, 9]" = torch.ops.aten.bmm.default(view_679, permute_577);  view_679 = permute_577 = None
    view_680: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.view.default(bmm_100, [8, 6, 196, 9, 32]);  bmm_100 = None
    view_681: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.view.default(bmm_101, [8, 6, 196, 9, 9]);  bmm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:84, code: attn = attn.softmax(dim=-1)
    mul_590: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(view_681, alias_39);  view_681 = None
    sum_229: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.sum.dim_IntList(mul_590, [-1], True)
    mul_591: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(alias_39, sum_229);  alias_39 = sum_229 = None
    sub_182: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.sub.Tensor(mul_590, mul_591);  mul_590 = mul_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:83, code: attn = attn * self.scale
    mul_592: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(sub_182, 0.1767766952966369);  sub_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:82, code: self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
    permute_578: "f32[8, 196, 6, 9, 9]" = torch.ops.aten.permute.default(mul_592, [0, 2, 1, 3, 4]);  mul_592 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:80, code: attn = self.attn(attn).reshape(
    clone_263: "f32[8, 196, 6, 9, 9]" = torch.ops.aten.clone.default(permute_578, memory_format = torch.contiguous_format);  permute_578 = None
    view_682: "f32[8, 14, 14, 486]" = torch.ops.aten.view.default(clone_263, [8, 14, 14, 486]);  clone_263 = None
    view_683: "f32[1568, 486]" = torch.ops.aten.view.default(view_682, [1568, 486]);  view_682 = None
    mm_169: "f32[1568, 192]" = torch.ops.aten.mm.default(view_683, permute_579);  permute_579 = None
    permute_580: "f32[486, 1568]" = torch.ops.aten.permute.default(view_683, [1, 0])
    mm_170: "f32[486, 192]" = torch.ops.aten.mm.default(permute_580, view_58);  permute_580 = view_58 = None
    permute_581: "f32[192, 486]" = torch.ops.aten.permute.default(mm_170, [1, 0]);  mm_170 = None
    sum_230: "f32[1, 486]" = torch.ops.aten.sum.dim_IntList(view_683, [0], True);  view_683 = None
    view_684: "f32[486]" = torch.ops.aten.view.default(sum_230, [486]);  sum_230 = None
    permute_582: "f32[486, 192]" = torch.ops.aten.permute.default(permute_581, [1, 0]);  permute_581 = None
    view_685: "f32[8, 14, 14, 192]" = torch.ops.aten.view.default(mm_169, [8, 14, 14, 192]);  mm_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:79, code: attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    permute_583: "f32[8, 192, 14, 14]" = torch.ops.aten.permute.default(view_685, [0, 3, 1, 2]);  view_685 = None
    avg_pool2d_backward: "f32[8, 192, 28, 28]" = torch.ops.aten.avg_pool2d_backward.default(permute_583, permute_47, [2, 2], [2, 2], [0, 0], True, True, None);  permute_583 = permute_47 = None
    permute_584: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(avg_pool2d_backward, [0, 2, 3, 1]);  avg_pool2d_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:77, code: self.kernel_size * self.kernel_size, h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H
    permute_585: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.permute.default(view_680, [0, 1, 4, 3, 2]);  view_680 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:75, code: v = self.unfold(v).reshape(
    clone_264: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.clone.default(permute_585, memory_format = torch.contiguous_format);  permute_585 = None
    view_686: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_264, [8, 1728, 196]);  clone_264 = None
    view_687: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.view.default(view_686, [8, 192, 3, 3, 14, 14]);  view_686 = None
    permute_586: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.permute.default(view_687, [0, 1, 2, 4, 3, 5]);  view_687 = None
    _unsafe_index_put_4: "f32[8, 192, 30, 30]" = torch.ops.aten._unsafe_index_put.default(full_default, [None, None, unsqueeze_17, add_17], permute_586, True);  permute_586 = None
    constant_pad_nd_9: "f32[8, 192, 28, 28]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_4, [-1, -1, -1, -1], 0.0);  _unsafe_index_put_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    permute_587: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(constant_pad_nd_9, [0, 2, 3, 1]);  constant_pad_nd_9 = None
    clone_265: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(permute_587, memory_format = torch.contiguous_format);  permute_587 = None
    view_688: "f32[6272, 192]" = torch.ops.aten.view.default(clone_265, [6272, 192]);  clone_265 = None
    permute_588: "f32[192, 6272]" = torch.ops.aten.permute.default(view_688, [1, 0])
    mm_171: "f32[192, 192]" = torch.ops.aten.mm.default(permute_588, view_54);  permute_588 = view_54 = None
    permute_589: "f32[192, 192]" = torch.ops.aten.permute.default(mm_171, [1, 0]);  mm_171 = None
    mm_172: "f32[6272, 192]" = torch.ops.aten.mm.default(view_688, permute_590);  view_688 = permute_590 = None
    view_689: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_172, [8, 28, 28, 192]);  mm_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    add_256: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(permute_584, view_689);  permute_584 = view_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    permute_591: "f32[192, 192]" = torch.ops.aten.permute.default(permute_589, [1, 0]);  permute_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    mul_594: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(add_256, primals_53);  primals_53 = None
    mul_595: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_594, 192)
    sum_231: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_594, [3], True)
    mul_596: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_594, mul_45);  mul_594 = None
    sum_232: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_596, [3], True);  mul_596 = None
    mul_597: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_45, sum_232);  sum_232 = None
    sub_184: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(mul_595, sum_231);  mul_595 = sum_231 = None
    sub_185: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(sub_184, mul_597);  sub_184 = mul_597 = None
    mul_598: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(div_54, sub_185);  div_54 = sub_185 = None
    mul_599: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(add_256, mul_45);  mul_45 = None
    sum_233: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_599, [0, 1, 2]);  mul_599 = None
    sum_234: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_256, [0, 1, 2]);  add_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_257: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_251, mul_598);  add_251 = mul_598 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_267: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_257, memory_format = torch.contiguous_format)
    view_690: "f32[6272, 192]" = torch.ops.aten.view.default(clone_267, [6272, 192]);  clone_267 = None
    mm_173: "f32[6272, 576]" = torch.ops.aten.mm.default(view_690, permute_592);  permute_592 = None
    permute_593: "f32[192, 6272]" = torch.ops.aten.permute.default(view_690, [1, 0])
    mm_174: "f32[192, 576]" = torch.ops.aten.mm.default(permute_593, view_52);  permute_593 = view_52 = None
    permute_594: "f32[576, 192]" = torch.ops.aten.permute.default(mm_174, [1, 0]);  mm_174 = None
    sum_235: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_690, [0], True);  view_690 = None
    view_691: "f32[192]" = torch.ops.aten.view.default(sum_235, [192]);  sum_235 = None
    permute_595: "f32[192, 576]" = torch.ops.aten.permute.default(permute_594, [1, 0]);  permute_594 = None
    view_692: "f32[8, 28, 28, 576]" = torch.ops.aten.view.default(mm_173, [8, 28, 28, 576]);  mm_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_601: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(add_49, 0.5);  add_49 = None
    mul_602: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_51, view_51)
    mul_603: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(mul_602, -0.5);  mul_602 = None
    exp_37: "f32[8, 28, 28, 576]" = torch.ops.aten.exp.default(mul_603);  mul_603 = None
    mul_604: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(exp_37, 0.3989422804014327);  exp_37 = None
    mul_605: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_51, mul_604);  view_51 = mul_604 = None
    add_259: "f32[8, 28, 28, 576]" = torch.ops.aten.add.Tensor(mul_601, mul_605);  mul_601 = mul_605 = None
    mul_606: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_692, add_259);  view_692 = add_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_693: "f32[6272, 576]" = torch.ops.aten.view.default(mul_606, [6272, 576]);  mul_606 = None
    mm_175: "f32[6272, 192]" = torch.ops.aten.mm.default(view_693, permute_596);  permute_596 = None
    permute_597: "f32[576, 6272]" = torch.ops.aten.permute.default(view_693, [1, 0])
    mm_176: "f32[576, 192]" = torch.ops.aten.mm.default(permute_597, view_50);  permute_597 = view_50 = None
    permute_598: "f32[192, 576]" = torch.ops.aten.permute.default(mm_176, [1, 0]);  mm_176 = None
    sum_236: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_693, [0], True);  view_693 = None
    view_694: "f32[576]" = torch.ops.aten.view.default(sum_236, [576]);  sum_236 = None
    permute_599: "f32[576, 192]" = torch.ops.aten.permute.default(permute_598, [1, 0]);  permute_598 = None
    view_695: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_175, [8, 28, 28, 192]);  mm_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    mul_608: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(view_695, primals_47);  primals_47 = None
    mul_609: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_608, 192)
    sum_237: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_608, [3], True)
    mul_610: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_608, mul_40);  mul_608 = None
    sum_238: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_610, [3], True);  mul_610 = None
    mul_611: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_40, sum_238);  sum_238 = None
    sub_187: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(mul_609, sum_237);  mul_609 = sum_237 = None
    sub_188: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(sub_187, mul_611);  sub_187 = mul_611 = None
    mul_612: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(div_55, sub_188);  div_55 = sub_188 = None
    mul_613: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(view_695, mul_40);  mul_40 = None
    sum_239: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_613, [0, 1, 2]);  mul_613 = None
    sum_240: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_695, [0, 1, 2]);  view_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_260: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_257, mul_612);  add_257 = mul_612 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:90, code: x = self.proj(x.permute(0, 2, 3, 1))
    sum_241: "f32[1, 1, 1, 192]" = torch.ops.aten.sum.dim_IntList(add_260, [0, 1, 2], True)
    view_696: "f32[192]" = torch.ops.aten.view.default(sum_241, [192]);  sum_241 = None
    clone_269: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_260, memory_format = torch.contiguous_format)
    view_697: "f32[6272, 192]" = torch.ops.aten.view.default(clone_269, [6272, 192]);  clone_269 = None
    permute_600: "f32[192, 6272]" = torch.ops.aten.permute.default(view_697, [1, 0])
    mm_177: "f32[192, 192]" = torch.ops.aten.mm.default(permute_600, view_48);  permute_600 = view_48 = None
    permute_601: "f32[192, 192]" = torch.ops.aten.permute.default(mm_177, [1, 0]);  mm_177 = None
    mm_178: "f32[6272, 192]" = torch.ops.aten.mm.default(view_697, permute_602);  view_697 = permute_602 = None
    view_698: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_178, [8, 28, 28, 192]);  mm_178 = None
    permute_603: "f32[192, 192]" = torch.ops.aten.permute.default(permute_601, [1, 0]);  permute_601 = None
    permute_604: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(view_698, [0, 3, 1, 2]);  view_698 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:88, code: x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
    constant_pad_nd_10: "f32[8, 192, 30, 30]" = torch.ops.aten.constant_pad_nd.default(permute_604, [1, 1, 1, 1], 0.0);  permute_604 = None
    slice_34: "f32[8, 192, 30, 30]" = torch.ops.aten.slice.Tensor(constant_pad_nd_10, 0, 0, 9223372036854775807);  constant_pad_nd_10 = None
    slice_35: "f32[8, 192, 30, 30]" = torch.ops.aten.slice.Tensor(slice_34, 1, 0, 9223372036854775807);  slice_34 = None
    index_5: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.index.Tensor(slice_35, [None, None, unsqueeze_17, add_17]);  slice_35 = None
    permute_605: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.permute.default(index_5, [0, 1, 2, 4, 3, 5]);  index_5 = None
    clone_270: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.clone.default(permute_605, memory_format = torch.contiguous_format);  permute_605 = None
    view_699: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_270, [8, 1728, 196]);  clone_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:87, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
    view_700: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.view.default(view_699, [8, 6, 32, 9, 196]);  view_699 = None
    permute_606: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.permute.default(view_700, [0, 1, 4, 3, 2]);  view_700 = None
    clone_271: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.clone.default(permute_606, memory_format = torch.contiguous_format);  permute_606 = None
    view_701: "f32[9408, 9, 32]" = torch.ops.aten.view.default(clone_271, [9408, 9, 32]);  clone_271 = None
    bmm_102: "f32[9408, 9, 32]" = torch.ops.aten.bmm.default(permute_607, view_701);  permute_607 = None
    bmm_103: "f32[9408, 9, 9]" = torch.ops.aten.bmm.default(view_701, permute_608);  view_701 = permute_608 = None
    view_702: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.view.default(bmm_102, [8, 6, 196, 9, 32]);  bmm_102 = None
    view_703: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.view.default(bmm_103, [8, 6, 196, 9, 9]);  bmm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:84, code: attn = attn.softmax(dim=-1)
    mul_614: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(view_703, alias_40);  view_703 = None
    sum_242: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.sum.dim_IntList(mul_614, [-1], True)
    mul_615: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(alias_40, sum_242);  alias_40 = sum_242 = None
    sub_189: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.sub.Tensor(mul_614, mul_615);  mul_614 = mul_615 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:83, code: attn = attn * self.scale
    mul_616: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(sub_189, 0.1767766952966369);  sub_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:82, code: self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
    permute_609: "f32[8, 196, 6, 9, 9]" = torch.ops.aten.permute.default(mul_616, [0, 2, 1, 3, 4]);  mul_616 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:80, code: attn = self.attn(attn).reshape(
    clone_272: "f32[8, 196, 6, 9, 9]" = torch.ops.aten.clone.default(permute_609, memory_format = torch.contiguous_format);  permute_609 = None
    view_704: "f32[8, 14, 14, 486]" = torch.ops.aten.view.default(clone_272, [8, 14, 14, 486]);  clone_272 = None
    view_705: "f32[1568, 486]" = torch.ops.aten.view.default(view_704, [1568, 486]);  view_704 = None
    mm_179: "f32[1568, 192]" = torch.ops.aten.mm.default(view_705, permute_610);  permute_610 = None
    permute_611: "f32[486, 1568]" = torch.ops.aten.permute.default(view_705, [1, 0])
    mm_180: "f32[486, 192]" = torch.ops.aten.mm.default(permute_611, view_40);  permute_611 = view_40 = None
    permute_612: "f32[192, 486]" = torch.ops.aten.permute.default(mm_180, [1, 0]);  mm_180 = None
    sum_243: "f32[1, 486]" = torch.ops.aten.sum.dim_IntList(view_705, [0], True);  view_705 = None
    view_706: "f32[486]" = torch.ops.aten.view.default(sum_243, [486]);  sum_243 = None
    permute_613: "f32[486, 192]" = torch.ops.aten.permute.default(permute_612, [1, 0]);  permute_612 = None
    view_707: "f32[8, 14, 14, 192]" = torch.ops.aten.view.default(mm_179, [8, 14, 14, 192]);  mm_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:79, code: attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    permute_614: "f32[8, 192, 14, 14]" = torch.ops.aten.permute.default(view_707, [0, 3, 1, 2]);  view_707 = None
    avg_pool2d_backward_1: "f32[8, 192, 28, 28]" = torch.ops.aten.avg_pool2d_backward.default(permute_614, permute_33, [2, 2], [2, 2], [0, 0], True, True, None);  permute_614 = permute_33 = None
    permute_615: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(avg_pool2d_backward_1, [0, 2, 3, 1]);  avg_pool2d_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:77, code: self.kernel_size * self.kernel_size, h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H
    permute_616: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.permute.default(view_702, [0, 1, 4, 3, 2]);  view_702 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:75, code: v = self.unfold(v).reshape(
    clone_273: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.clone.default(permute_616, memory_format = torch.contiguous_format);  permute_616 = None
    view_708: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_273, [8, 1728, 196]);  clone_273 = None
    view_709: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.view.default(view_708, [8, 192, 3, 3, 14, 14]);  view_708 = None
    permute_617: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.permute.default(view_709, [0, 1, 2, 4, 3, 5]);  view_709 = None
    _unsafe_index_put_5: "f32[8, 192, 30, 30]" = torch.ops.aten._unsafe_index_put.default(full_default, [None, None, unsqueeze_17, add_17], permute_617, True);  permute_617 = None
    constant_pad_nd_11: "f32[8, 192, 28, 28]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_5, [-1, -1, -1, -1], 0.0);  _unsafe_index_put_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    permute_618: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(constant_pad_nd_11, [0, 2, 3, 1]);  constant_pad_nd_11 = None
    clone_274: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(permute_618, memory_format = torch.contiguous_format);  permute_618 = None
    view_710: "f32[6272, 192]" = torch.ops.aten.view.default(clone_274, [6272, 192]);  clone_274 = None
    permute_619: "f32[192, 6272]" = torch.ops.aten.permute.default(view_710, [1, 0])
    mm_181: "f32[192, 192]" = torch.ops.aten.mm.default(permute_619, view_36);  permute_619 = view_36 = None
    permute_620: "f32[192, 192]" = torch.ops.aten.permute.default(mm_181, [1, 0]);  mm_181 = None
    mm_182: "f32[6272, 192]" = torch.ops.aten.mm.default(view_710, permute_621);  view_710 = permute_621 = None
    view_711: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_182, [8, 28, 28, 192]);  mm_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    add_265: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(permute_615, view_711);  permute_615 = view_711 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    permute_622: "f32[192, 192]" = torch.ops.aten.permute.default(permute_620, [1, 0]);  permute_620 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    mul_618: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(add_265, primals_40);  primals_40 = None
    mul_619: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_618, 192)
    sum_244: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_618, [3], True)
    mul_620: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_618, mul_37);  mul_618 = None
    sum_245: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_620, [3], True);  mul_620 = None
    mul_621: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_37, sum_245);  sum_245 = None
    sub_191: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(mul_619, sum_244);  mul_619 = sum_244 = None
    sub_192: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(sub_191, mul_621);  sub_191 = mul_621 = None
    mul_622: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(div_56, sub_192);  div_56 = sub_192 = None
    mul_623: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(add_265, mul_37);  mul_37 = None
    sum_246: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_623, [0, 1, 2]);  mul_623 = None
    sum_247: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_265, [0, 1, 2]);  add_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_266: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_260, mul_622);  add_260 = mul_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_276: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_266, memory_format = torch.contiguous_format)
    view_712: "f32[6272, 192]" = torch.ops.aten.view.default(clone_276, [6272, 192]);  clone_276 = None
    mm_183: "f32[6272, 576]" = torch.ops.aten.mm.default(view_712, permute_623);  permute_623 = None
    permute_624: "f32[192, 6272]" = torch.ops.aten.permute.default(view_712, [1, 0])
    mm_184: "f32[192, 576]" = torch.ops.aten.mm.default(permute_624, view_34);  permute_624 = view_34 = None
    permute_625: "f32[576, 192]" = torch.ops.aten.permute.default(mm_184, [1, 0]);  mm_184 = None
    sum_248: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_712, [0], True);  view_712 = None
    view_713: "f32[192]" = torch.ops.aten.view.default(sum_248, [192]);  sum_248 = None
    permute_626: "f32[192, 576]" = torch.ops.aten.permute.default(permute_625, [1, 0]);  permute_625 = None
    view_714: "f32[8, 28, 28, 576]" = torch.ops.aten.view.default(mm_183, [8, 28, 28, 576]);  mm_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_625: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(add_37, 0.5);  add_37 = None
    mul_626: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_33, view_33)
    mul_627: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(mul_626, -0.5);  mul_626 = None
    exp_38: "f32[8, 28, 28, 576]" = torch.ops.aten.exp.default(mul_627);  mul_627 = None
    mul_628: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(exp_38, 0.3989422804014327);  exp_38 = None
    mul_629: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_33, mul_628);  view_33 = mul_628 = None
    add_268: "f32[8, 28, 28, 576]" = torch.ops.aten.add.Tensor(mul_625, mul_629);  mul_625 = mul_629 = None
    mul_630: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_714, add_268);  view_714 = add_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_715: "f32[6272, 576]" = torch.ops.aten.view.default(mul_630, [6272, 576]);  mul_630 = None
    mm_185: "f32[6272, 192]" = torch.ops.aten.mm.default(view_715, permute_627);  permute_627 = None
    permute_628: "f32[576, 6272]" = torch.ops.aten.permute.default(view_715, [1, 0])
    mm_186: "f32[576, 192]" = torch.ops.aten.mm.default(permute_628, view_32);  permute_628 = view_32 = None
    permute_629: "f32[192, 576]" = torch.ops.aten.permute.default(mm_186, [1, 0]);  mm_186 = None
    sum_249: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_715, [0], True);  view_715 = None
    view_716: "f32[576]" = torch.ops.aten.view.default(sum_249, [576]);  sum_249 = None
    permute_630: "f32[576, 192]" = torch.ops.aten.permute.default(permute_629, [1, 0]);  permute_629 = None
    view_717: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_185, [8, 28, 28, 192]);  mm_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    mul_632: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(view_717, primals_34);  primals_34 = None
    mul_633: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_632, 192)
    sum_250: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_632, [3], True)
    mul_634: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_632, mul_32);  mul_632 = None
    sum_251: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_634, [3], True);  mul_634 = None
    mul_635: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_32, sum_251);  sum_251 = None
    sub_194: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(mul_633, sum_250);  mul_633 = sum_250 = None
    sub_195: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(sub_194, mul_635);  sub_194 = mul_635 = None
    mul_636: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(div_57, sub_195);  div_57 = sub_195 = None
    mul_637: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(view_717, mul_32);  mul_32 = None
    sum_252: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_637, [0, 1, 2]);  mul_637 = None
    sum_253: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_717, [0, 1, 2]);  view_717 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_269: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_266, mul_636);  add_266 = mul_636 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:90, code: x = self.proj(x.permute(0, 2, 3, 1))
    sum_254: "f32[1, 1, 1, 192]" = torch.ops.aten.sum.dim_IntList(add_269, [0, 1, 2], True)
    view_718: "f32[192]" = torch.ops.aten.view.default(sum_254, [192]);  sum_254 = None
    clone_278: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_269, memory_format = torch.contiguous_format)
    view_719: "f32[6272, 192]" = torch.ops.aten.view.default(clone_278, [6272, 192]);  clone_278 = None
    permute_631: "f32[192, 6272]" = torch.ops.aten.permute.default(view_719, [1, 0])
    mm_187: "f32[192, 192]" = torch.ops.aten.mm.default(permute_631, view_30);  permute_631 = view_30 = None
    permute_632: "f32[192, 192]" = torch.ops.aten.permute.default(mm_187, [1, 0]);  mm_187 = None
    mm_188: "f32[6272, 192]" = torch.ops.aten.mm.default(view_719, permute_633);  view_719 = permute_633 = None
    view_720: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_188, [8, 28, 28, 192]);  mm_188 = None
    permute_634: "f32[192, 192]" = torch.ops.aten.permute.default(permute_632, [1, 0]);  permute_632 = None
    permute_635: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(view_720, [0, 3, 1, 2]);  view_720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:88, code: x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
    constant_pad_nd_12: "f32[8, 192, 30, 30]" = torch.ops.aten.constant_pad_nd.default(permute_635, [1, 1, 1, 1], 0.0);  permute_635 = None
    slice_36: "f32[8, 192, 30, 30]" = torch.ops.aten.slice.Tensor(constant_pad_nd_12, 0, 0, 9223372036854775807);  constant_pad_nd_12 = None
    slice_37: "f32[8, 192, 30, 30]" = torch.ops.aten.slice.Tensor(slice_36, 1, 0, 9223372036854775807);  slice_36 = None
    index_6: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.index.Tensor(slice_37, [None, None, unsqueeze_17, add_17]);  slice_37 = None
    permute_636: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.permute.default(index_6, [0, 1, 2, 4, 3, 5]);  index_6 = None
    clone_279: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.clone.default(permute_636, memory_format = torch.contiguous_format);  permute_636 = None
    view_721: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_279, [8, 1728, 196]);  clone_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:87, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
    view_722: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.view.default(view_721, [8, 6, 32, 9, 196]);  view_721 = None
    permute_637: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.permute.default(view_722, [0, 1, 4, 3, 2]);  view_722 = None
    clone_280: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.clone.default(permute_637, memory_format = torch.contiguous_format);  permute_637 = None
    view_723: "f32[9408, 9, 32]" = torch.ops.aten.view.default(clone_280, [9408, 9, 32]);  clone_280 = None
    bmm_104: "f32[9408, 9, 32]" = torch.ops.aten.bmm.default(permute_638, view_723);  permute_638 = None
    bmm_105: "f32[9408, 9, 9]" = torch.ops.aten.bmm.default(view_723, permute_639);  view_723 = permute_639 = None
    view_724: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.view.default(bmm_104, [8, 6, 196, 9, 32]);  bmm_104 = None
    view_725: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.view.default(bmm_105, [8, 6, 196, 9, 9]);  bmm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:84, code: attn = attn.softmax(dim=-1)
    mul_638: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(view_725, alias_41);  view_725 = None
    sum_255: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.sum.dim_IntList(mul_638, [-1], True)
    mul_639: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(alias_41, sum_255);  alias_41 = sum_255 = None
    sub_196: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.sub.Tensor(mul_638, mul_639);  mul_638 = mul_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:83, code: attn = attn * self.scale
    mul_640: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(sub_196, 0.1767766952966369);  sub_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:82, code: self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
    permute_640: "f32[8, 196, 6, 9, 9]" = torch.ops.aten.permute.default(mul_640, [0, 2, 1, 3, 4]);  mul_640 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:80, code: attn = self.attn(attn).reshape(
    clone_281: "f32[8, 196, 6, 9, 9]" = torch.ops.aten.clone.default(permute_640, memory_format = torch.contiguous_format);  permute_640 = None
    view_726: "f32[8, 14, 14, 486]" = torch.ops.aten.view.default(clone_281, [8, 14, 14, 486]);  clone_281 = None
    view_727: "f32[1568, 486]" = torch.ops.aten.view.default(view_726, [1568, 486]);  view_726 = None
    mm_189: "f32[1568, 192]" = torch.ops.aten.mm.default(view_727, permute_641);  permute_641 = None
    permute_642: "f32[486, 1568]" = torch.ops.aten.permute.default(view_727, [1, 0])
    mm_190: "f32[486, 192]" = torch.ops.aten.mm.default(permute_642, view_22);  permute_642 = view_22 = None
    permute_643: "f32[192, 486]" = torch.ops.aten.permute.default(mm_190, [1, 0]);  mm_190 = None
    sum_256: "f32[1, 486]" = torch.ops.aten.sum.dim_IntList(view_727, [0], True);  view_727 = None
    view_728: "f32[486]" = torch.ops.aten.view.default(sum_256, [486]);  sum_256 = None
    permute_644: "f32[486, 192]" = torch.ops.aten.permute.default(permute_643, [1, 0]);  permute_643 = None
    view_729: "f32[8, 14, 14, 192]" = torch.ops.aten.view.default(mm_189, [8, 14, 14, 192]);  mm_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:79, code: attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    permute_645: "f32[8, 192, 14, 14]" = torch.ops.aten.permute.default(view_729, [0, 3, 1, 2]);  view_729 = None
    avg_pool2d_backward_2: "f32[8, 192, 28, 28]" = torch.ops.aten.avg_pool2d_backward.default(permute_645, permute_19, [2, 2], [2, 2], [0, 0], True, True, None);  permute_645 = permute_19 = None
    permute_646: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(avg_pool2d_backward_2, [0, 2, 3, 1]);  avg_pool2d_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:77, code: self.kernel_size * self.kernel_size, h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H
    permute_647: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.permute.default(view_724, [0, 1, 4, 3, 2]);  view_724 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:75, code: v = self.unfold(v).reshape(
    clone_282: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.clone.default(permute_647, memory_format = torch.contiguous_format);  permute_647 = None
    view_730: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_282, [8, 1728, 196]);  clone_282 = None
    view_731: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.view.default(view_730, [8, 192, 3, 3, 14, 14]);  view_730 = None
    permute_648: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.permute.default(view_731, [0, 1, 2, 4, 3, 5]);  view_731 = None
    _unsafe_index_put_6: "f32[8, 192, 30, 30]" = torch.ops.aten._unsafe_index_put.default(full_default, [None, None, unsqueeze_17, add_17], permute_648, True);  permute_648 = None
    constant_pad_nd_13: "f32[8, 192, 28, 28]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_6, [-1, -1, -1, -1], 0.0);  _unsafe_index_put_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    permute_649: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(constant_pad_nd_13, [0, 2, 3, 1]);  constant_pad_nd_13 = None
    clone_283: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(permute_649, memory_format = torch.contiguous_format);  permute_649 = None
    view_732: "f32[6272, 192]" = torch.ops.aten.view.default(clone_283, [6272, 192]);  clone_283 = None
    permute_650: "f32[192, 6272]" = torch.ops.aten.permute.default(view_732, [1, 0])
    mm_191: "f32[192, 192]" = torch.ops.aten.mm.default(permute_650, view_18);  permute_650 = view_18 = None
    permute_651: "f32[192, 192]" = torch.ops.aten.permute.default(mm_191, [1, 0]);  mm_191 = None
    mm_192: "f32[6272, 192]" = torch.ops.aten.mm.default(view_732, permute_652);  view_732 = permute_652 = None
    view_733: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_192, [8, 28, 28, 192]);  mm_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    add_274: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(permute_646, view_733);  permute_646 = view_733 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    permute_653: "f32[192, 192]" = torch.ops.aten.permute.default(permute_651, [1, 0]);  permute_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    mul_642: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(add_274, primals_27);  primals_27 = None
    mul_643: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_642, 192)
    sum_257: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_642, [3], True)
    mul_644: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_642, mul_29);  mul_642 = None
    sum_258: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_644, [3], True);  mul_644 = None
    mul_645: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_29, sum_258);  sum_258 = None
    sub_198: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(mul_643, sum_257);  mul_643 = sum_257 = None
    sub_199: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(sub_198, mul_645);  sub_198 = mul_645 = None
    mul_646: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(div_58, sub_199);  div_58 = sub_199 = None
    mul_647: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(add_274, mul_29);  mul_29 = None
    sum_259: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_647, [0, 1, 2]);  mul_647 = None
    sum_260: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_274, [0, 1, 2]);  add_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_275: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_269, mul_646);  add_269 = mul_646 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_285: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_275, memory_format = torch.contiguous_format)
    view_734: "f32[6272, 192]" = torch.ops.aten.view.default(clone_285, [6272, 192]);  clone_285 = None
    mm_193: "f32[6272, 576]" = torch.ops.aten.mm.default(view_734, permute_654);  permute_654 = None
    permute_655: "f32[192, 6272]" = torch.ops.aten.permute.default(view_734, [1, 0])
    mm_194: "f32[192, 576]" = torch.ops.aten.mm.default(permute_655, view_16);  permute_655 = view_16 = None
    permute_656: "f32[576, 192]" = torch.ops.aten.permute.default(mm_194, [1, 0]);  mm_194 = None
    sum_261: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_734, [0], True);  view_734 = None
    view_735: "f32[192]" = torch.ops.aten.view.default(sum_261, [192]);  sum_261 = None
    permute_657: "f32[192, 576]" = torch.ops.aten.permute.default(permute_656, [1, 0]);  permute_656 = None
    view_736: "f32[8, 28, 28, 576]" = torch.ops.aten.view.default(mm_193, [8, 28, 28, 576]);  mm_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_649: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(add_25, 0.5);  add_25 = None
    mul_650: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_15, view_15)
    mul_651: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(mul_650, -0.5);  mul_650 = None
    exp_39: "f32[8, 28, 28, 576]" = torch.ops.aten.exp.default(mul_651);  mul_651 = None
    mul_652: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(exp_39, 0.3989422804014327);  exp_39 = None
    mul_653: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_15, mul_652);  view_15 = mul_652 = None
    add_277: "f32[8, 28, 28, 576]" = torch.ops.aten.add.Tensor(mul_649, mul_653);  mul_649 = mul_653 = None
    mul_654: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_736, add_277);  view_736 = add_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_737: "f32[6272, 576]" = torch.ops.aten.view.default(mul_654, [6272, 576]);  mul_654 = None
    mm_195: "f32[6272, 192]" = torch.ops.aten.mm.default(view_737, permute_658);  permute_658 = None
    permute_659: "f32[576, 6272]" = torch.ops.aten.permute.default(view_737, [1, 0])
    mm_196: "f32[576, 192]" = torch.ops.aten.mm.default(permute_659, view_14);  permute_659 = view_14 = None
    permute_660: "f32[192, 576]" = torch.ops.aten.permute.default(mm_196, [1, 0]);  mm_196 = None
    sum_262: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_737, [0], True);  view_737 = None
    view_738: "f32[576]" = torch.ops.aten.view.default(sum_262, [576]);  sum_262 = None
    permute_661: "f32[576, 192]" = torch.ops.aten.permute.default(permute_660, [1, 0]);  permute_660 = None
    view_739: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_195, [8, 28, 28, 192]);  mm_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    mul_656: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(view_739, primals_21);  primals_21 = None
    mul_657: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_656, 192)
    sum_263: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_656, [3], True)
    mul_658: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_656, mul_24);  mul_656 = None
    sum_264: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_658, [3], True);  mul_658 = None
    mul_659: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_24, sum_264);  sum_264 = None
    sub_201: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(mul_657, sum_263);  mul_657 = sum_263 = None
    sub_202: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(sub_201, mul_659);  sub_201 = mul_659 = None
    mul_660: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(div_59, sub_202);  div_59 = sub_202 = None
    mul_661: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(view_739, mul_24);  mul_24 = None
    sum_265: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_661, [0, 1, 2]);  mul_661 = None
    sum_266: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_739, [0, 1, 2]);  view_739 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_278: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_275, mul_660);  add_275 = mul_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:90, code: x = self.proj(x.permute(0, 2, 3, 1))
    sum_267: "f32[1, 1, 1, 192]" = torch.ops.aten.sum.dim_IntList(add_278, [0, 1, 2], True)
    view_740: "f32[192]" = torch.ops.aten.view.default(sum_267, [192]);  sum_267 = None
    clone_287: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_278, memory_format = torch.contiguous_format)
    view_741: "f32[6272, 192]" = torch.ops.aten.view.default(clone_287, [6272, 192]);  clone_287 = None
    permute_662: "f32[192, 6272]" = torch.ops.aten.permute.default(view_741, [1, 0])
    mm_197: "f32[192, 192]" = torch.ops.aten.mm.default(permute_662, view_12);  permute_662 = view_12 = None
    permute_663: "f32[192, 192]" = torch.ops.aten.permute.default(mm_197, [1, 0]);  mm_197 = None
    mm_198: "f32[6272, 192]" = torch.ops.aten.mm.default(view_741, permute_664);  view_741 = permute_664 = None
    view_742: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_198, [8, 28, 28, 192]);  mm_198 = None
    permute_665: "f32[192, 192]" = torch.ops.aten.permute.default(permute_663, [1, 0]);  permute_663 = None
    permute_666: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(view_742, [0, 3, 1, 2]);  view_742 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:88, code: x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
    constant_pad_nd_14: "f32[8, 192, 30, 30]" = torch.ops.aten.constant_pad_nd.default(permute_666, [1, 1, 1, 1], 0.0);  permute_666 = None
    slice_38: "f32[8, 192, 30, 30]" = torch.ops.aten.slice.Tensor(constant_pad_nd_14, 0, 0, 9223372036854775807);  constant_pad_nd_14 = None
    slice_39: "f32[8, 192, 30, 30]" = torch.ops.aten.slice.Tensor(slice_38, 1, 0, 9223372036854775807);  slice_38 = None
    index_7: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.index.Tensor(slice_39, [None, None, unsqueeze_17, add_17]);  slice_39 = None
    permute_667: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.permute.default(index_7, [0, 1, 2, 4, 3, 5]);  index_7 = None
    clone_288: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.clone.default(permute_667, memory_format = torch.contiguous_format);  permute_667 = None
    view_743: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_288, [8, 1728, 196]);  clone_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:87, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
    view_744: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.view.default(view_743, [8, 6, 32, 9, 196]);  view_743 = None
    permute_668: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.permute.default(view_744, [0, 1, 4, 3, 2]);  view_744 = None
    clone_289: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.clone.default(permute_668, memory_format = torch.contiguous_format);  permute_668 = None
    view_745: "f32[9408, 9, 32]" = torch.ops.aten.view.default(clone_289, [9408, 9, 32]);  clone_289 = None
    bmm_106: "f32[9408, 9, 32]" = torch.ops.aten.bmm.default(permute_669, view_745);  permute_669 = None
    bmm_107: "f32[9408, 9, 9]" = torch.ops.aten.bmm.default(view_745, permute_670);  view_745 = permute_670 = None
    view_746: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.view.default(bmm_106, [8, 6, 196, 9, 32]);  bmm_106 = None
    view_747: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.view.default(bmm_107, [8, 6, 196, 9, 9]);  bmm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:84, code: attn = attn.softmax(dim=-1)
    mul_662: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(view_747, alias_42);  view_747 = None
    sum_268: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.sum.dim_IntList(mul_662, [-1], True)
    mul_663: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(alias_42, sum_268);  alias_42 = sum_268 = None
    sub_203: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.sub.Tensor(mul_662, mul_663);  mul_662 = mul_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:83, code: attn = attn * self.scale
    mul_664: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(sub_203, 0.1767766952966369);  sub_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:82, code: self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
    permute_671: "f32[8, 196, 6, 9, 9]" = torch.ops.aten.permute.default(mul_664, [0, 2, 1, 3, 4]);  mul_664 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:80, code: attn = self.attn(attn).reshape(
    clone_290: "f32[8, 196, 6, 9, 9]" = torch.ops.aten.clone.default(permute_671, memory_format = torch.contiguous_format);  permute_671 = None
    view_748: "f32[8, 14, 14, 486]" = torch.ops.aten.view.default(clone_290, [8, 14, 14, 486]);  clone_290 = None
    view_749: "f32[1568, 486]" = torch.ops.aten.view.default(view_748, [1568, 486]);  view_748 = None
    mm_199: "f32[1568, 192]" = torch.ops.aten.mm.default(view_749, permute_672);  permute_672 = None
    permute_673: "f32[486, 1568]" = torch.ops.aten.permute.default(view_749, [1, 0])
    mm_200: "f32[486, 192]" = torch.ops.aten.mm.default(permute_673, view_4);  permute_673 = view_4 = None
    permute_674: "f32[192, 486]" = torch.ops.aten.permute.default(mm_200, [1, 0]);  mm_200 = None
    sum_269: "f32[1, 486]" = torch.ops.aten.sum.dim_IntList(view_749, [0], True);  view_749 = None
    view_750: "f32[486]" = torch.ops.aten.view.default(sum_269, [486]);  sum_269 = None
    permute_675: "f32[486, 192]" = torch.ops.aten.permute.default(permute_674, [1, 0]);  permute_674 = None
    view_751: "f32[8, 14, 14, 192]" = torch.ops.aten.view.default(mm_199, [8, 14, 14, 192]);  mm_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:79, code: attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    permute_676: "f32[8, 192, 14, 14]" = torch.ops.aten.permute.default(view_751, [0, 3, 1, 2]);  view_751 = None
    avg_pool2d_backward_3: "f32[8, 192, 28, 28]" = torch.ops.aten.avg_pool2d_backward.default(permute_676, permute_5, [2, 2], [2, 2], [0, 0], True, True, None);  permute_676 = permute_5 = None
    permute_677: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(avg_pool2d_backward_3, [0, 2, 3, 1]);  avg_pool2d_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:77, code: self.kernel_size * self.kernel_size, h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H
    permute_678: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.permute.default(view_746, [0, 1, 4, 3, 2]);  view_746 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:75, code: v = self.unfold(v).reshape(
    clone_291: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.clone.default(permute_678, memory_format = torch.contiguous_format);  permute_678 = None
    view_752: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_291, [8, 1728, 196]);  clone_291 = None
    view_753: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.view.default(view_752, [8, 192, 3, 3, 14, 14]);  view_752 = None
    permute_679: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.permute.default(view_753, [0, 1, 2, 4, 3, 5]);  view_753 = None
    _unsafe_index_put_7: "f32[8, 192, 30, 30]" = torch.ops.aten._unsafe_index_put.default(full_default, [None, None, unsqueeze_17, add_17], permute_679, True);  full_default = unsqueeze_17 = add_17 = permute_679 = None
    constant_pad_nd_15: "f32[8, 192, 28, 28]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_7, [-1, -1, -1, -1], 0.0);  _unsafe_index_put_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    permute_680: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(constant_pad_nd_15, [0, 2, 3, 1]);  constant_pad_nd_15 = None
    clone_292: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(permute_680, memory_format = torch.contiguous_format);  permute_680 = None
    view_754: "f32[6272, 192]" = torch.ops.aten.view.default(clone_292, [6272, 192]);  clone_292 = None
    permute_681: "f32[192, 6272]" = torch.ops.aten.permute.default(view_754, [1, 0])
    mm_201: "f32[192, 192]" = torch.ops.aten.mm.default(permute_681, view);  permute_681 = view = None
    permute_682: "f32[192, 192]" = torch.ops.aten.permute.default(mm_201, [1, 0]);  mm_201 = None
    mm_202: "f32[6272, 192]" = torch.ops.aten.mm.default(view_754, permute_683);  view_754 = permute_683 = None
    view_755: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_202, [8, 28, 28, 192]);  mm_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    add_283: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(permute_677, view_755);  permute_677 = view_755 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    permute_684: "f32[192, 192]" = torch.ops.aten.permute.default(permute_682, [1, 0]);  permute_682 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    mul_666: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(add_283, primals_14);  primals_14 = None
    mul_667: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_666, 192)
    sum_270: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_666, [3], True)
    mul_668: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_666, mul_21);  mul_666 = None
    sum_271: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_668, [3], True);  mul_668 = None
    mul_669: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_21, sum_271);  sum_271 = None
    sub_205: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(mul_667, sum_270);  mul_667 = sum_270 = None
    sub_206: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(sub_205, mul_669);  sub_205 = mul_669 = None
    mul_670: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(div_60, sub_206);  div_60 = sub_206 = None
    mul_671: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(add_283, mul_21);  mul_21 = None
    sum_272: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_671, [0, 1, 2]);  mul_671 = None
    sum_273: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_283, [0, 1, 2]);  add_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_284: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_278, mul_670);  add_278 = mul_670 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:695, code: x = self.patch_embed(x).permute(0, 2, 3, 1)  # B,C,H,W-> B,H,W,C
    permute_685: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(add_284, [0, 3, 1, 2]);  add_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:358, code: x = self.proj(x)  # B, C, H, W
    sum_274: "f32[192]" = torch.ops.aten.sum.dim_IntList(permute_685, [0, 2, 3])
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(permute_685, relu_2, primals_12, [192], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  permute_685 = primals_12 = None
    getitem_139: "f32[8, 64, 112, 112]" = convolution_backward_1[0]
    getitem_140: "f32[192, 64, 4, 4]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:357, code: x = self.conv(x)
    alias_44: "f32[8, 64, 112, 112]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_45: "f32[8, 64, 112, 112]" = torch.ops.aten.alias.default(alias_44);  alias_44 = None
    le: "b8[8, 64, 112, 112]" = torch.ops.aten.le.Scalar(alias_45, 0);  alias_45 = None
    full_default_27: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "f32[8, 64, 112, 112]" = torch.ops.aten.where.self(le, full_default_27, getitem_139);  le = getitem_139 = None
    sum_275: "f32[64]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_207: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_112);  convolution_2 = unsqueeze_112 = None
    mul_672: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where, sub_207)
    sum_276: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_672, [0, 2, 3]);  mul_672 = None
    mul_673: "f32[64]" = torch.ops.aten.mul.Tensor(sum_275, 9.964923469387754e-06)
    unsqueeze_113: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_673, 0);  mul_673 = None
    unsqueeze_114: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_113, 2);  unsqueeze_113 = None
    unsqueeze_115: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, 3);  unsqueeze_114 = None
    mul_674: "f32[64]" = torch.ops.aten.mul.Tensor(sum_276, 9.964923469387754e-06)
    mul_675: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_676: "f32[64]" = torch.ops.aten.mul.Tensor(mul_674, mul_675);  mul_674 = mul_675 = None
    unsqueeze_116: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_676, 0);  mul_676 = None
    unsqueeze_117: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, 2);  unsqueeze_116 = None
    unsqueeze_118: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_117, 3);  unsqueeze_117 = None
    mul_677: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_10);  primals_10 = None
    unsqueeze_119: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_677, 0);  mul_677 = None
    unsqueeze_120: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_119, 2);  unsqueeze_119 = None
    unsqueeze_121: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, 3);  unsqueeze_120 = None
    mul_678: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_118);  sub_207 = unsqueeze_118 = None
    sub_209: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(where, mul_678);  where = mul_678 = None
    sub_210: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(sub_209, unsqueeze_115);  sub_209 = unsqueeze_115 = None
    mul_679: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_210, unsqueeze_121);  sub_210 = unsqueeze_121 = None
    mul_680: "f32[64]" = torch.ops.aten.mul.Tensor(sum_276, squeeze_7);  sum_276 = squeeze_7 = None
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_679, relu_1, primals_9, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_679 = primals_9 = None
    getitem_142: "f32[8, 64, 112, 112]" = convolution_backward_2[0]
    getitem_143: "f32[64, 64, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
    alias_47: "f32[8, 64, 112, 112]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_48: "f32[8, 64, 112, 112]" = torch.ops.aten.alias.default(alias_47);  alias_47 = None
    le_1: "b8[8, 64, 112, 112]" = torch.ops.aten.le.Scalar(alias_48, 0);  alias_48 = None
    where_1: "f32[8, 64, 112, 112]" = torch.ops.aten.where.self(le_1, full_default_27, getitem_142);  le_1 = getitem_142 = None
    sum_277: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_211: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_124);  convolution_1 = unsqueeze_124 = None
    mul_681: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_1, sub_211)
    sum_278: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_681, [0, 2, 3]);  mul_681 = None
    mul_682: "f32[64]" = torch.ops.aten.mul.Tensor(sum_277, 9.964923469387754e-06)
    unsqueeze_125: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_682, 0);  mul_682 = None
    unsqueeze_126: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_125, 2);  unsqueeze_125 = None
    unsqueeze_127: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, 3);  unsqueeze_126 = None
    mul_683: "f32[64]" = torch.ops.aten.mul.Tensor(sum_278, 9.964923469387754e-06)
    mul_684: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_685: "f32[64]" = torch.ops.aten.mul.Tensor(mul_683, mul_684);  mul_683 = mul_684 = None
    unsqueeze_128: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_685, 0);  mul_685 = None
    unsqueeze_129: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, 2);  unsqueeze_128 = None
    unsqueeze_130: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_129, 3);  unsqueeze_129 = None
    mul_686: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_7);  primals_7 = None
    unsqueeze_131: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_686, 0);  mul_686 = None
    unsqueeze_132: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_131, 2);  unsqueeze_131 = None
    unsqueeze_133: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, 3);  unsqueeze_132 = None
    mul_687: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_130);  sub_211 = unsqueeze_130 = None
    sub_213: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(where_1, mul_687);  where_1 = mul_687 = None
    sub_214: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(sub_213, unsqueeze_127);  sub_213 = unsqueeze_127 = None
    mul_688: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_133);  sub_214 = unsqueeze_133 = None
    mul_689: "f32[64]" = torch.ops.aten.mul.Tensor(sum_278, squeeze_4);  sum_278 = squeeze_4 = None
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_688, relu, primals_6, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_688 = primals_6 = None
    getitem_145: "f32[8, 64, 112, 112]" = convolution_backward_3[0]
    getitem_146: "f32[64, 64, 3, 3]" = convolution_backward_3[1];  convolution_backward_3 = None
    alias_50: "f32[8, 64, 112, 112]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_51: "f32[8, 64, 112, 112]" = torch.ops.aten.alias.default(alias_50);  alias_50 = None
    le_2: "b8[8, 64, 112, 112]" = torch.ops.aten.le.Scalar(alias_51, 0);  alias_51 = None
    where_2: "f32[8, 64, 112, 112]" = torch.ops.aten.where.self(le_2, full_default_27, getitem_145);  le_2 = full_default_27 = getitem_145 = None
    sum_279: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_215: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_136);  convolution = unsqueeze_136 = None
    mul_690: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_2, sub_215)
    sum_280: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_690, [0, 2, 3]);  mul_690 = None
    mul_691: "f32[64]" = torch.ops.aten.mul.Tensor(sum_279, 9.964923469387754e-06)
    unsqueeze_137: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_691, 0);  mul_691 = None
    unsqueeze_138: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_137, 2);  unsqueeze_137 = None
    unsqueeze_139: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, 3);  unsqueeze_138 = None
    mul_692: "f32[64]" = torch.ops.aten.mul.Tensor(sum_280, 9.964923469387754e-06)
    mul_693: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_694: "f32[64]" = torch.ops.aten.mul.Tensor(mul_692, mul_693);  mul_692 = mul_693 = None
    unsqueeze_140: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_694, 0);  mul_694 = None
    unsqueeze_141: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, 2);  unsqueeze_140 = None
    unsqueeze_142: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_141, 3);  unsqueeze_141 = None
    mul_695: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_4);  primals_4 = None
    unsqueeze_143: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_695, 0);  mul_695 = None
    unsqueeze_144: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_143, 2);  unsqueeze_143 = None
    unsqueeze_145: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, 3);  unsqueeze_144 = None
    mul_696: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_215, unsqueeze_142);  sub_215 = unsqueeze_142 = None
    sub_217: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(where_2, mul_696);  where_2 = mul_696 = None
    sub_218: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(sub_217, unsqueeze_139);  sub_217 = unsqueeze_139 = None
    mul_697: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_218, unsqueeze_145);  sub_218 = unsqueeze_145 = None
    mul_698: "f32[64]" = torch.ops.aten.mul.Tensor(sum_280, squeeze_1);  sum_280 = squeeze_1 = None
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_697, primals_261, primals_3, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_697 = primals_261 = primals_3 = None
    getitem_149: "f32[64, 3, 7, 7]" = convolution_backward_4[1];  convolution_backward_4 = None
    return [sum_220, sum_51, getitem_149, mul_698, sum_279, getitem_146, mul_689, sum_277, getitem_143, mul_680, sum_275, getitem_140, sum_274, sum_272, sum_273, permute_684, permute_675, view_750, permute_665, view_740, sum_265, sum_266, permute_661, view_738, permute_657, view_735, sum_259, sum_260, permute_653, permute_644, view_728, permute_634, view_718, sum_252, sum_253, permute_630, view_716, permute_626, view_713, sum_246, sum_247, permute_622, permute_613, view_706, permute_603, view_696, sum_239, sum_240, permute_599, view_694, permute_595, view_691, sum_233, sum_234, permute_591, permute_582, view_684, permute_572, view_674, sum_226, sum_227, permute_568, view_672, permute_564, view_669, getitem_137, sum_221, sum_218, sum_219, permute_558, permute_547, view_655, sum_212, sum_213, permute_543, view_652, permute_539, view_649, sum_206, sum_207, permute_535, permute_524, view_635, sum_200, sum_201, permute_520, view_632, permute_516, view_629, sum_194, sum_195, permute_512, permute_501, view_615, sum_188, sum_189, permute_497, view_612, permute_493, view_609, sum_182, sum_183, permute_489, permute_478, view_595, sum_176, sum_177, permute_474, view_592, permute_470, view_589, sum_170, sum_171, permute_466, permute_455, view_575, sum_164, sum_165, permute_451, view_572, permute_447, view_569, sum_158, sum_159, permute_443, permute_432, view_555, sum_152, sum_153, permute_428, view_552, permute_424, view_549, sum_146, sum_147, permute_420, permute_409, view_535, sum_140, sum_141, permute_405, view_532, permute_401, view_529, sum_134, sum_135, permute_397, permute_386, view_515, sum_128, sum_129, permute_382, view_512, permute_378, view_509, sum_122, sum_123, permute_374, permute_363, view_495, sum_116, sum_117, permute_359, view_492, permute_355, view_489, sum_110, sum_111, permute_351, permute_340, view_475, sum_104, sum_105, permute_336, view_472, permute_332, view_469, sum_98, sum_99, permute_328, permute_317, view_455, sum_92, sum_93, permute_313, view_452, permute_309, view_449, sum_86, sum_87, permute_305, permute_294, view_435, sum_80, sum_81, permute_290, view_432, permute_286, view_429, sum_74, sum_75, permute_282, permute_271, view_415, sum_68, sum_69, permute_267, view_412, permute_263, view_409, sum_62, sum_63, permute_259, permute_248, view_395, sum_56, sum_57, permute_244, view_392, permute_240, view_389, sum_49, sum_50, permute_236, permute_231, permute_221, view_371, sum_43, sum_44, permute_217, view_368, permute_213, view_365, sum_37, sum_38, permute_209, permute_204, permute_194, view_348, sum_31, sum_32, permute_190, view_345, permute_186, view_342, sum_25, sum_26, permute_182, view_340, permute_178, view_337, None, None, None, None, None, None, None, None, None, None]
    