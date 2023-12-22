from __future__ import annotations



def forward(self, primals_2: "f32[128]", primals_4: "f32[128]", primals_6: "f32[128]", primals_8: "f32[128]", primals_10: "f32[256]", primals_13: "f32[256]", primals_15: "f32[256]", primals_17: "f32[256]", primals_19: "f32[256]", primals_21: "f32[512]", primals_24: "f32[512]", primals_26: "f32[512]", primals_28: "f32[512]", primals_30: "f32[512]", primals_32: "f32[512]", primals_34: "f32[512]", primals_36: "f32[512]", primals_38: "f32[512]", primals_40: "f32[512]", primals_42: "f32[512]", primals_44: "f32[512]", primals_46: "f32[512]", primals_48: "f32[512]", primals_50: "f32[512]", primals_52: "f32[512]", primals_54: "f32[512]", primals_56: "f32[512]", primals_58: "f32[512]", primals_60: "f32[512]", primals_62: "f32[512]", primals_64: "f32[512]", primals_66: "f32[512]", primals_68: "f32[512]", primals_70: "f32[512]", primals_72: "f32[512]", primals_74: "f32[512]", primals_76: "f32[512]", primals_78: "f32[512]", primals_80: "f32[512]", primals_82: "f32[512]", primals_84: "f32[512]", primals_86: "f32[512]", primals_88: "f32[512]", primals_90: "f32[512]", primals_92: "f32[512]", primals_94: "f32[512]", primals_96: "f32[512]", primals_98: "f32[512]", primals_100: "f32[512]", primals_102: "f32[512]", primals_104: "f32[512]", primals_106: "f32[128, 3, 4, 4]", primals_124: "f32[256, 128, 3, 3]", primals_142: "f32[512, 256, 3, 3]", primals_306: "f32[8, 3, 224, 224]", mul: "f32[8, 16, 196, 128]", view_2: "f32[25088, 128]", view_12: "f32[25088, 128]", mul_4: "f32[8, 16, 196, 128]", view_14: "f32[25088, 128]", addmm_2: "f32[25088, 512]", view_16: "f32[25088, 512]", mul_9: "f32[8, 16, 196, 128]", view_18: "f32[25088, 128]", view_28: "f32[25088, 128]", bernoulli: "f32[8, 1, 1, 1]", mul_14: "f32[8, 16, 196, 128]", view_30: "f32[25088, 128]", addmm_6: "f32[25088, 512]", view_32: "f32[25088, 512]", bernoulli_1: "f32[8, 1, 1, 1]", permute_17: "f32[8, 128, 56, 56]", mul_20: "f32[8, 56, 56, 256]", constant_pad_nd: "f32[8, 256, 57, 57]", getitem_17: "i64[8, 256, 28, 28]", mul_22: "f32[8, 4, 196, 256]", view_38: "f32[6272, 256]", view_48: "f32[6272, 256]", bernoulli_2: "f32[8, 1, 1, 1]", mul_27: "f32[8, 4, 196, 256]", view_50: "f32[6272, 256]", addmm_10: "f32[6272, 1024]", view_52: "f32[6272, 1024]", bernoulli_3: "f32[8, 1, 1, 1]", mul_33: "f32[8, 4, 196, 256]", view_54: "f32[6272, 256]", view_64: "f32[6272, 256]", bernoulli_4: "f32[8, 1, 1, 1]", mul_38: "f32[8, 4, 196, 256]", view_66: "f32[6272, 256]", addmm_14: "f32[6272, 1024]", view_68: "f32[6272, 1024]", bernoulli_5: "f32[8, 1, 1, 1]", permute_37: "f32[8, 256, 28, 28]", mul_44: "f32[8, 28, 28, 512]", constant_pad_nd_1: "f32[8, 512, 29, 29]", getitem_35: "i64[8, 512, 14, 14]", mul_46: "f32[8, 1, 196, 512]", view_74: "f32[1568, 512]", view_84: "f32[1568, 512]", bernoulli_6: "f32[8, 1, 1, 1]", mul_51: "f32[8, 1, 196, 512]", view_86: "f32[1568, 512]", addmm_18: "f32[1568, 2048]", view_88: "f32[1568, 2048]", bernoulli_7: "f32[8, 1, 1, 1]", mul_57: "f32[8, 1, 196, 512]", view_90: "f32[1568, 512]", view_100: "f32[1568, 512]", bernoulli_8: "f32[8, 1, 1, 1]", mul_62: "f32[8, 1, 196, 512]", view_102: "f32[1568, 512]", addmm_22: "f32[1568, 2048]", view_104: "f32[1568, 2048]", bernoulli_9: "f32[8, 1, 1, 1]", mul_68: "f32[8, 1, 196, 512]", view_106: "f32[1568, 512]", view_116: "f32[1568, 512]", bernoulli_10: "f32[8, 1, 1, 1]", mul_73: "f32[8, 1, 196, 512]", view_118: "f32[1568, 512]", addmm_26: "f32[1568, 2048]", view_120: "f32[1568, 2048]", bernoulli_11: "f32[8, 1, 1, 1]", mul_79: "f32[8, 1, 196, 512]", view_122: "f32[1568, 512]", view_132: "f32[1568, 512]", bernoulli_12: "f32[8, 1, 1, 1]", mul_84: "f32[8, 1, 196, 512]", view_134: "f32[1568, 512]", addmm_30: "f32[1568, 2048]", view_136: "f32[1568, 2048]", bernoulli_13: "f32[8, 1, 1, 1]", mul_90: "f32[8, 1, 196, 512]", view_138: "f32[1568, 512]", view_148: "f32[1568, 512]", bernoulli_14: "f32[8, 1, 1, 1]", mul_95: "f32[8, 1, 196, 512]", view_150: "f32[1568, 512]", addmm_34: "f32[1568, 2048]", view_152: "f32[1568, 2048]", bernoulli_15: "f32[8, 1, 1, 1]", mul_101: "f32[8, 1, 196, 512]", view_154: "f32[1568, 512]", view_164: "f32[1568, 512]", bernoulli_16: "f32[8, 1, 1, 1]", mul_106: "f32[8, 1, 196, 512]", view_166: "f32[1568, 512]", addmm_38: "f32[1568, 2048]", view_168: "f32[1568, 2048]", bernoulli_17: "f32[8, 1, 1, 1]", mul_112: "f32[8, 1, 196, 512]", view_170: "f32[1568, 512]", view_180: "f32[1568, 512]", bernoulli_18: "f32[8, 1, 1, 1]", mul_117: "f32[8, 1, 196, 512]", view_182: "f32[1568, 512]", addmm_42: "f32[1568, 2048]", view_184: "f32[1568, 2048]", bernoulli_19: "f32[8, 1, 1, 1]", mul_123: "f32[8, 1, 196, 512]", view_186: "f32[1568, 512]", view_196: "f32[1568, 512]", bernoulli_20: "f32[8, 1, 1, 1]", mul_128: "f32[8, 1, 196, 512]", view_198: "f32[1568, 512]", addmm_46: "f32[1568, 2048]", view_200: "f32[1568, 2048]", bernoulli_21: "f32[8, 1, 1, 1]", mul_134: "f32[8, 1, 196, 512]", view_202: "f32[1568, 512]", view_212: "f32[1568, 512]", bernoulli_22: "f32[8, 1, 1, 1]", mul_139: "f32[8, 1, 196, 512]", view_214: "f32[1568, 512]", addmm_50: "f32[1568, 2048]", view_216: "f32[1568, 2048]", bernoulli_23: "f32[8, 1, 1, 1]", mul_145: "f32[8, 1, 196, 512]", view_218: "f32[1568, 512]", view_228: "f32[1568, 512]", bernoulli_24: "f32[8, 1, 1, 1]", mul_150: "f32[8, 1, 196, 512]", view_230: "f32[1568, 512]", addmm_54: "f32[1568, 2048]", view_232: "f32[1568, 2048]", bernoulli_25: "f32[8, 1, 1, 1]", mul_156: "f32[8, 1, 196, 512]", view_234: "f32[1568, 512]", view_244: "f32[1568, 512]", bernoulli_26: "f32[8, 1, 1, 1]", mul_161: "f32[8, 1, 196, 512]", view_246: "f32[1568, 512]", addmm_58: "f32[1568, 2048]", view_248: "f32[1568, 2048]", bernoulli_27: "f32[8, 1, 1, 1]", mul_167: "f32[8, 1, 196, 512]", view_250: "f32[1568, 512]", view_260: "f32[1568, 512]", bernoulli_28: "f32[8, 1, 1, 1]", mul_172: "f32[8, 1, 196, 512]", view_262: "f32[1568, 512]", addmm_62: "f32[1568, 2048]", view_264: "f32[1568, 2048]", bernoulli_29: "f32[8, 1, 1, 1]", mul_178: "f32[8, 1, 196, 512]", view_266: "f32[1568, 512]", view_276: "f32[1568, 512]", bernoulli_30: "f32[8, 1, 1, 1]", mul_183: "f32[8, 1, 196, 512]", view_278: "f32[1568, 512]", addmm_66: "f32[1568, 2048]", view_280: "f32[1568, 2048]", bernoulli_31: "f32[8, 1, 1, 1]", mul_189: "f32[8, 1, 196, 512]", view_282: "f32[1568, 512]", view_292: "f32[1568, 512]", bernoulli_32: "f32[8, 1, 1, 1]", mul_194: "f32[8, 1, 196, 512]", view_294: "f32[1568, 512]", addmm_70: "f32[1568, 2048]", view_296: "f32[1568, 2048]", bernoulli_33: "f32[8, 1, 1, 1]", mul_200: "f32[8, 1, 196, 512]", view_298: "f32[1568, 512]", view_308: "f32[1568, 512]", bernoulli_34: "f32[8, 1, 1, 1]", mul_205: "f32[8, 1, 196, 512]", view_310: "f32[1568, 512]", addmm_74: "f32[1568, 2048]", view_312: "f32[1568, 2048]", bernoulli_35: "f32[8, 1, 1, 1]", mul_211: "f32[8, 1, 196, 512]", view_314: "f32[1568, 512]", view_324: "f32[1568, 512]", bernoulli_36: "f32[8, 1, 1, 1]", mul_216: "f32[8, 1, 196, 512]", view_326: "f32[1568, 512]", addmm_78: "f32[1568, 2048]", view_328: "f32[1568, 2048]", bernoulli_37: "f32[8, 1, 1, 1]", mul_222: "f32[8, 1, 196, 512]", view_330: "f32[1568, 512]", view_340: "f32[1568, 512]", bernoulli_38: "f32[8, 1, 1, 1]", mul_227: "f32[8, 1, 196, 512]", view_342: "f32[1568, 512]", addmm_82: "f32[1568, 2048]", view_344: "f32[1568, 2048]", bernoulli_39: "f32[8, 1, 1, 1]", mul_233: "f32[8, 1, 196, 512]", view_346: "f32[1568, 512]", view_356: "f32[1568, 512]", bernoulli_40: "f32[8, 1, 1, 1]", mul_238: "f32[8, 1, 196, 512]", view_358: "f32[1568, 512]", addmm_86: "f32[1568, 2048]", view_360: "f32[1568, 2048]", bernoulli_41: "f32[8, 1, 1, 1]", mul_244: "f32[8, 1, 196, 512]", view_362: "f32[1568, 512]", view_372: "f32[1568, 512]", bernoulli_42: "f32[8, 1, 1, 1]", mul_249: "f32[8, 1, 196, 512]", view_374: "f32[1568, 512]", addmm_90: "f32[1568, 2048]", view_376: "f32[1568, 2048]", bernoulli_43: "f32[8, 1, 1, 1]", mul_255: "f32[8, 1, 196, 512]", view_378: "f32[1568, 512]", view_388: "f32[1568, 512]", bernoulli_44: "f32[8, 1, 1, 1]", mul_260: "f32[8, 1, 196, 512]", view_390: "f32[1568, 512]", addmm_94: "f32[1568, 2048]", view_392: "f32[1568, 2048]", bernoulli_45: "f32[8, 1, 1, 1]", mul_266: "f32[8, 14, 14, 512]", clone_174: "f32[8, 512]", permute_187: "f32[1000, 512]", div_71: "f32[8, 14, 14, 1]", permute_195: "f32[512, 2048]", permute_199: "f32[2048, 512]", div_72: "f32[8, 1, 196, 1]", permute_203: "f32[512, 512]", permute_208: "f32[128, 196, 196]", permute_209: "f32[128, 32, 196]", alias_24: "f32[8, 16, 1, 196, 196]", permute_210: "f32[128, 32, 196]", permute_211: "f32[128, 196, 32]", permute_214: "f32[1536, 512]", div_73: "f32[8, 1, 196, 1]", permute_218: "f32[512, 2048]", permute_222: "f32[2048, 512]", div_74: "f32[8, 1, 196, 1]", permute_226: "f32[512, 512]", permute_231: "f32[128, 196, 196]", permute_232: "f32[128, 32, 196]", alias_25: "f32[8, 16, 1, 196, 196]", permute_233: "f32[128, 32, 196]", permute_234: "f32[128, 196, 32]", permute_237: "f32[1536, 512]", div_75: "f32[8, 1, 196, 1]", permute_241: "f32[512, 2048]", permute_245: "f32[2048, 512]", div_76: "f32[8, 1, 196, 1]", permute_249: "f32[512, 512]", permute_254: "f32[128, 196, 196]", permute_255: "f32[128, 32, 196]", alias_26: "f32[8, 16, 1, 196, 196]", permute_256: "f32[128, 32, 196]", permute_257: "f32[128, 196, 32]", permute_260: "f32[1536, 512]", div_77: "f32[8, 1, 196, 1]", permute_264: "f32[512, 2048]", permute_268: "f32[2048, 512]", div_78: "f32[8, 1, 196, 1]", permute_272: "f32[512, 512]", permute_277: "f32[128, 196, 196]", permute_278: "f32[128, 32, 196]", alias_27: "f32[8, 16, 1, 196, 196]", permute_279: "f32[128, 32, 196]", permute_280: "f32[128, 196, 32]", permute_283: "f32[1536, 512]", div_79: "f32[8, 1, 196, 1]", permute_287: "f32[512, 2048]", permute_291: "f32[2048, 512]", div_80: "f32[8, 1, 196, 1]", permute_295: "f32[512, 512]", permute_300: "f32[128, 196, 196]", permute_301: "f32[128, 32, 196]", alias_28: "f32[8, 16, 1, 196, 196]", permute_302: "f32[128, 32, 196]", permute_303: "f32[128, 196, 32]", permute_306: "f32[1536, 512]", div_81: "f32[8, 1, 196, 1]", permute_310: "f32[512, 2048]", permute_314: "f32[2048, 512]", div_82: "f32[8, 1, 196, 1]", permute_318: "f32[512, 512]", permute_323: "f32[128, 196, 196]", permute_324: "f32[128, 32, 196]", alias_29: "f32[8, 16, 1, 196, 196]", permute_325: "f32[128, 32, 196]", permute_326: "f32[128, 196, 32]", permute_329: "f32[1536, 512]", div_83: "f32[8, 1, 196, 1]", permute_333: "f32[512, 2048]", permute_337: "f32[2048, 512]", div_84: "f32[8, 1, 196, 1]", permute_341: "f32[512, 512]", permute_346: "f32[128, 196, 196]", permute_347: "f32[128, 32, 196]", alias_30: "f32[8, 16, 1, 196, 196]", permute_348: "f32[128, 32, 196]", permute_349: "f32[128, 196, 32]", permute_352: "f32[1536, 512]", div_85: "f32[8, 1, 196, 1]", permute_356: "f32[512, 2048]", permute_360: "f32[2048, 512]", div_86: "f32[8, 1, 196, 1]", permute_364: "f32[512, 512]", permute_369: "f32[128, 196, 196]", permute_370: "f32[128, 32, 196]", alias_31: "f32[8, 16, 1, 196, 196]", permute_371: "f32[128, 32, 196]", permute_372: "f32[128, 196, 32]", permute_375: "f32[1536, 512]", div_87: "f32[8, 1, 196, 1]", permute_379: "f32[512, 2048]", permute_383: "f32[2048, 512]", div_88: "f32[8, 1, 196, 1]", permute_387: "f32[512, 512]", permute_392: "f32[128, 196, 196]", permute_393: "f32[128, 32, 196]", alias_32: "f32[8, 16, 1, 196, 196]", permute_394: "f32[128, 32, 196]", permute_395: "f32[128, 196, 32]", permute_398: "f32[1536, 512]", div_89: "f32[8, 1, 196, 1]", permute_402: "f32[512, 2048]", permute_406: "f32[2048, 512]", div_90: "f32[8, 1, 196, 1]", permute_410: "f32[512, 512]", permute_415: "f32[128, 196, 196]", permute_416: "f32[128, 32, 196]", alias_33: "f32[8, 16, 1, 196, 196]", permute_417: "f32[128, 32, 196]", permute_418: "f32[128, 196, 32]", permute_421: "f32[1536, 512]", div_91: "f32[8, 1, 196, 1]", permute_425: "f32[512, 2048]", permute_429: "f32[2048, 512]", div_92: "f32[8, 1, 196, 1]", permute_433: "f32[512, 512]", permute_438: "f32[128, 196, 196]", permute_439: "f32[128, 32, 196]", alias_34: "f32[8, 16, 1, 196, 196]", permute_440: "f32[128, 32, 196]", permute_441: "f32[128, 196, 32]", permute_444: "f32[1536, 512]", div_93: "f32[8, 1, 196, 1]", permute_448: "f32[512, 2048]", permute_452: "f32[2048, 512]", div_94: "f32[8, 1, 196, 1]", permute_456: "f32[512, 512]", permute_461: "f32[128, 196, 196]", permute_462: "f32[128, 32, 196]", alias_35: "f32[8, 16, 1, 196, 196]", permute_463: "f32[128, 32, 196]", permute_464: "f32[128, 196, 32]", permute_467: "f32[1536, 512]", div_95: "f32[8, 1, 196, 1]", permute_471: "f32[512, 2048]", permute_475: "f32[2048, 512]", div_96: "f32[8, 1, 196, 1]", permute_479: "f32[512, 512]", permute_484: "f32[128, 196, 196]", permute_485: "f32[128, 32, 196]", alias_36: "f32[8, 16, 1, 196, 196]", permute_486: "f32[128, 32, 196]", permute_487: "f32[128, 196, 32]", permute_490: "f32[1536, 512]", div_97: "f32[8, 1, 196, 1]", permute_494: "f32[512, 2048]", permute_498: "f32[2048, 512]", div_98: "f32[8, 1, 196, 1]", permute_502: "f32[512, 512]", permute_507: "f32[128, 196, 196]", permute_508: "f32[128, 32, 196]", alias_37: "f32[8, 16, 1, 196, 196]", permute_509: "f32[128, 32, 196]", permute_510: "f32[128, 196, 32]", permute_513: "f32[1536, 512]", div_99: "f32[8, 1, 196, 1]", permute_517: "f32[512, 2048]", permute_521: "f32[2048, 512]", div_100: "f32[8, 1, 196, 1]", permute_525: "f32[512, 512]", permute_530: "f32[128, 196, 196]", permute_531: "f32[128, 32, 196]", alias_38: "f32[8, 16, 1, 196, 196]", permute_532: "f32[128, 32, 196]", permute_533: "f32[128, 196, 32]", permute_536: "f32[1536, 512]", div_101: "f32[8, 1, 196, 1]", permute_540: "f32[512, 2048]", permute_544: "f32[2048, 512]", div_102: "f32[8, 1, 196, 1]", permute_548: "f32[512, 512]", permute_553: "f32[128, 196, 196]", permute_554: "f32[128, 32, 196]", alias_39: "f32[8, 16, 1, 196, 196]", permute_555: "f32[128, 32, 196]", permute_556: "f32[128, 196, 32]", permute_559: "f32[1536, 512]", div_103: "f32[8, 1, 196, 1]", permute_563: "f32[512, 2048]", permute_567: "f32[2048, 512]", div_104: "f32[8, 1, 196, 1]", permute_571: "f32[512, 512]", permute_576: "f32[128, 196, 196]", permute_577: "f32[128, 32, 196]", alias_40: "f32[8, 16, 1, 196, 196]", permute_578: "f32[128, 32, 196]", permute_579: "f32[128, 196, 32]", permute_582: "f32[1536, 512]", div_105: "f32[8, 1, 196, 1]", permute_586: "f32[512, 2048]", permute_590: "f32[2048, 512]", div_106: "f32[8, 1, 196, 1]", permute_594: "f32[512, 512]", permute_599: "f32[128, 196, 196]", permute_600: "f32[128, 32, 196]", alias_41: "f32[8, 16, 1, 196, 196]", permute_601: "f32[128, 32, 196]", permute_602: "f32[128, 196, 32]", permute_605: "f32[1536, 512]", div_107: "f32[8, 1, 196, 1]", permute_609: "f32[512, 2048]", permute_613: "f32[2048, 512]", div_108: "f32[8, 1, 196, 1]", permute_617: "f32[512, 512]", permute_622: "f32[128, 196, 196]", permute_623: "f32[128, 32, 196]", alias_42: "f32[8, 16, 1, 196, 196]", permute_624: "f32[128, 32, 196]", permute_625: "f32[128, 196, 32]", permute_628: "f32[1536, 512]", div_109: "f32[8, 1, 196, 1]", permute_632: "f32[512, 2048]", permute_636: "f32[2048, 512]", div_110: "f32[8, 1, 196, 1]", permute_640: "f32[512, 512]", permute_645: "f32[128, 196, 196]", permute_646: "f32[128, 32, 196]", alias_43: "f32[8, 16, 1, 196, 196]", permute_647: "f32[128, 32, 196]", permute_648: "f32[128, 196, 32]", permute_651: "f32[1536, 512]", div_111: "f32[8, 1, 196, 1]", div_112: "f32[8, 28, 28, 1]", permute_661: "f32[256, 1024]", permute_665: "f32[1024, 256]", div_113: "f32[8, 4, 196, 1]", permute_669: "f32[256, 256]", permute_674: "f32[256, 196, 196]", permute_675: "f32[256, 32, 196]", alias_44: "f32[8, 8, 4, 196, 196]", permute_676: "f32[256, 32, 196]", permute_677: "f32[256, 196, 32]", permute_680: "f32[768, 256]", div_114: "f32[8, 4, 196, 1]", permute_684: "f32[256, 1024]", permute_688: "f32[1024, 256]", div_115: "f32[8, 4, 196, 1]", permute_692: "f32[256, 256]", permute_697: "f32[256, 196, 196]", permute_698: "f32[256, 32, 196]", alias_45: "f32[8, 8, 4, 196, 196]", permute_699: "f32[256, 32, 196]", permute_700: "f32[256, 196, 32]", permute_703: "f32[768, 256]", div_116: "f32[8, 4, 196, 1]", div_117: "f32[8, 56, 56, 1]", permute_713: "f32[128, 512]", permute_717: "f32[512, 128]", div_118: "f32[8, 16, 196, 1]", permute_721: "f32[128, 128]", permute_726: "f32[512, 196, 196]", permute_727: "f32[512, 32, 196]", alias_46: "f32[8, 4, 16, 196, 196]", permute_728: "f32[512, 32, 196]", permute_729: "f32[512, 196, 32]", permute_732: "f32[384, 128]", div_119: "f32[8, 16, 196, 1]", permute_736: "f32[128, 512]", permute_740: "f32[512, 128]", div_120: "f32[8, 16, 196, 1]", permute_744: "f32[128, 128]", permute_749: "f32[512, 196, 196]", permute_750: "f32[512, 32, 196]", alias_47: "f32[8, 4, 16, 196, 196]", permute_751: "f32[512, 32, 196]", permute_752: "f32[512, 196, 32]", permute_755: "f32[384, 128]", div_121: "f32[8, 16, 196, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_15: "f32[8, 16, 196, 512]" = torch.ops.aten.view.default(addmm_2, [8, 16, 196, 512]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_7: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(view_15, 0.7071067811865476)
    erf: "f32[8, 16, 196, 512]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
    add_6: "f32[8, 16, 196, 512]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_2: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli, 0.9782608691602945);  bernoulli = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_31: "f32[8, 16, 196, 512]" = torch.ops.aten.view.default(addmm_6, [8, 16, 196, 512]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_17: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(view_31, 0.7071067811865476)
    erf_1: "f32[8, 16, 196, 512]" = torch.ops.aten.erf.default(mul_17);  mul_17 = None
    add_13: "f32[8, 16, 196, 512]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_3: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_1, 0.9782608691602945);  bernoulli_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_5: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_2, 0.9565217383205891);  bernoulli_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_51: "f32[8, 4, 196, 1024]" = torch.ops.aten.view.default(addmm_10, [8, 4, 196, 1024]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_30: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(view_51, 0.7071067811865476)
    erf_2: "f32[8, 4, 196, 1024]" = torch.ops.aten.erf.default(mul_30);  mul_30 = None
    add_23: "f32[8, 4, 196, 1024]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_6: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_3, 0.9565217383205891);  bernoulli_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_8: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_4, 0.9347826093435287);  bernoulli_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_67: "f32[8, 4, 196, 1024]" = torch.ops.aten.view.default(addmm_14, [8, 4, 196, 1024]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_41: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(view_67, 0.7071067811865476)
    erf_3: "f32[8, 4, 196, 1024]" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
    add_30: "f32[8, 4, 196, 1024]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_9: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_5, 0.9347826093435287);  bernoulli_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_11: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_6, 0.9130434766411781);  bernoulli_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_87: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_18, [8, 1, 196, 2048]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_54: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_87, 0.7071067811865476)
    erf_4: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
    add_40: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_12: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_7, 0.9130434766411781);  bernoulli_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_14: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_8, 0.8913043439388275);  bernoulli_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_103: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_22, [8, 1, 196, 2048]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_65: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_103, 0.7071067811865476)
    erf_5: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_65);  mul_65 = None
    add_47: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_15: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_9, 0.8913043439388275);  bernoulli_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_17: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_10, 0.8695652186870575);  bernoulli_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_119: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_26, [8, 1, 196, 2048]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_76: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_119, 0.7071067811865476)
    erf_6: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
    add_54: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_18: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_11, 0.8695652186870575);  bernoulli_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_20: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_12, 0.8478260785341263);  bernoulli_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_135: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_30, [8, 1, 196, 2048]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_87: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_135, 0.7071067811865476)
    erf_7: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
    add_61: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_21: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_13, 0.8478260785341263);  bernoulli_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_23: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_14, 0.8260869532823563);  bernoulli_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_151: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_34, [8, 1, 196, 2048]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_98: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476)
    erf_8: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_98);  mul_98 = None
    add_68: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_24: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_15, 0.8260869532823563);  bernoulli_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_26: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_16, 0.8043478280305862);  bernoulli_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_167: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_38, [8, 1, 196, 2048]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_109: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_167, 0.7071067811865476)
    erf_9: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_109);  mul_109 = None
    add_75: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_27: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_17, 0.8043478280305862);  bernoulli_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_29: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_18, 0.782608687877655);  bernoulli_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_183: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_42, [8, 1, 196, 2048]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_120: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_183, 0.7071067811865476)
    erf_10: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_120);  mul_120 = None
    add_82: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_30: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_19, 0.782608687877655);  bernoulli_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_32: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_20, 0.760869562625885);  bernoulli_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_199: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_46, [8, 1, 196, 2048]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_131: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_199, 0.7071067811865476)
    erf_11: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_131);  mul_131 = None
    add_89: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_33: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_21, 0.760869562625885);  bernoulli_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_35: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_22, 0.739130437374115);  bernoulli_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_215: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_50, [8, 1, 196, 2048]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_142: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_215, 0.7071067811865476)
    erf_12: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_142);  mul_142 = None
    add_96: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_36: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_23, 0.739130437374115);  bernoulli_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_38: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_24, 0.717391312122345);  bernoulli_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_231: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_54, [8, 1, 196, 2048]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_153: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_231, 0.7071067811865476)
    erf_13: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_153);  mul_153 = None
    add_103: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_39: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_25, 0.717391312122345);  bernoulli_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_41: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_26, 0.695652186870575);  bernoulli_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_247: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_58, [8, 1, 196, 2048]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_164: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_247, 0.7071067811865476)
    erf_14: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_164);  mul_164 = None
    add_110: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_42: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_27, 0.695652186870575);  bernoulli_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_44: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_28, 0.6739130616188049);  bernoulli_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_263: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_62, [8, 1, 196, 2048]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_175: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_263, 0.7071067811865476)
    erf_15: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_175);  mul_175 = None
    add_117: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_45: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_29, 0.6739130616188049);  bernoulli_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_47: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_30, 0.6521739065647125);  bernoulli_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_279: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_66, [8, 1, 196, 2048]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_186: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_279, 0.7071067811865476)
    erf_16: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_186);  mul_186 = None
    add_124: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_48: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_31, 0.6521739065647125);  bernoulli_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_50: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_32, 0.6304347813129425);  bernoulli_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_295: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_70, [8, 1, 196, 2048]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_197: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_295, 0.7071067811865476)
    erf_17: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_197);  mul_197 = None
    add_131: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_51: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_33, 0.6304347813129425);  bernoulli_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_53: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_34, 0.6086956560611725);  bernoulli_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_311: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_74, [8, 1, 196, 2048]);  addmm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_208: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_311, 0.7071067811865476)
    erf_18: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_208);  mul_208 = None
    add_138: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_54: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_35, 0.6086956560611725);  bernoulli_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_56: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_36, 0.5869565308094025);  bernoulli_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_327: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_78, [8, 1, 196, 2048]);  addmm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_219: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_327, 0.7071067811865476)
    erf_19: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_219);  mul_219 = None
    add_145: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_57: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_37, 0.5869565308094025);  bernoulli_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_59: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_38, 0.5652174055576324);  bernoulli_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_343: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_82, [8, 1, 196, 2048]);  addmm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_230: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_343, 0.7071067811865476)
    erf_20: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_230);  mul_230 = None
    add_152: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_60: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_39, 0.5652174055576324);  bernoulli_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_62: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_40, 0.54347825050354);  bernoulli_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_359: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_86, [8, 1, 196, 2048]);  addmm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_241: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_359, 0.7071067811865476)
    erf_21: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_241);  mul_241 = None
    add_159: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_63: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_41, 0.54347825050354);  bernoulli_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_65: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_42, 0.52173912525177);  bernoulli_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_375: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_90, [8, 1, 196, 2048]);  addmm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_252: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_375, 0.7071067811865476)
    erf_22: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_252);  mul_252 = None
    add_166: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_66: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_43, 0.52173912525177);  bernoulli_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_68: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_44, 0.5);  bernoulli_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_391: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(addmm_94, [8, 1, 196, 2048]);  addmm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_263: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_391, 0.7071067811865476)
    erf_23: "f32[8, 1, 196, 2048]" = torch.ops.aten.erf.default(mul_263);  mul_263 = None
    add_173: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_69: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_45, 0.5);  bernoulli_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:433, code: return x if pre_logits else self.head(x)
    mm: "f32[8, 512]" = torch.ops.aten.mm.default(tangents_1, permute_187);  permute_187 = None
    permute_188: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 512]" = torch.ops.aten.mm.default(permute_188, clone_174);  permute_188 = clone_174 = None
    permute_189: "f32[512, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_25: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_397: "f32[1000]" = torch.ops.aten.view.default(sum_25, [1000]);  sum_25 = None
    permute_190: "f32[1000, 512]" = torch.ops.aten.permute.default(permute_189, [1, 0]);  permute_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_398: "f32[8, 512, 1, 1]" = torch.ops.aten.view.default(mm, [8, 512, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    squeeze: "f32[8, 512, 1]" = torch.ops.aten.squeeze.dim(view_398, 3);  view_398 = None
    squeeze_1: "f32[8, 512]" = torch.ops.aten.squeeze.dim(squeeze, 2);  squeeze = None
    full: "f32[4096]" = torch.ops.aten.full.default([4096], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    as_strided_1: "f32[8, 512]" = torch.ops.aten.as_strided.default(full, [8, 512], [512, 1], 0)
    copy: "f32[8, 512]" = torch.ops.aten.copy.default(as_strided_1, squeeze_1);  as_strided_1 = squeeze_1 = None
    as_strided_scatter: "f32[4096]" = torch.ops.aten.as_strided_scatter.default(full, copy, [8, 512], [512, 1], 0);  full = copy = None
    as_strided_4: "f32[8, 512, 1, 1]" = torch.ops.aten.as_strided.default(as_strided_scatter, [8, 512, 1, 1], [512, 1, 1, 1], 0);  as_strided_scatter = None
    expand_97: "f32[8, 512, 14, 14]" = torch.ops.aten.expand.default(as_strided_4, [8, 512, 14, 14]);  as_strided_4 = None
    div_70: "f32[8, 512, 14, 14]" = torch.ops.aten.div.Scalar(expand_97, 196);  expand_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:427, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_191: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(div_70, [0, 2, 3, 1]);  div_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone_175: "f32[8, 14, 14, 512]" = torch.ops.aten.clone.default(permute_191, memory_format = torch.contiguous_format);  permute_191 = None
    mul_269: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(clone_175, primals_104);  primals_104 = None
    mul_270: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_269, 512)
    sum_26: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_269, [3], True)
    mul_271: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_269, mul_266);  mul_269 = None
    sum_27: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_271, [3], True);  mul_271 = None
    mul_272: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_266, sum_27);  sum_27 = None
    sub_76: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_270, sum_26);  mul_270 = sum_26 = None
    sub_77: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_76, mul_272);  sub_76 = mul_272 = None
    mul_273: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_71, sub_77);  div_71 = sub_77 = None
    mul_274: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(clone_175, mul_266);  mul_266 = None
    sum_28: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_274, [0, 1, 2]);  mul_274 = None
    sum_29: "f32[512]" = torch.ops.aten.sum.dim_IntList(clone_175, [0, 1, 2]);  clone_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:427, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_192: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_273, [0, 3, 1, 2]);  mul_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:242, code: return x.permute(0, 3, 1, 2)  # (B, C, H', W')
    permute_193: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(permute_192, [0, 2, 3, 1]);  permute_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:175, code: x = x.transpose(2, 3).reshape(B, height, width, C)
    view_399: "f32[8, 1, 14, 1, 14, 512]" = torch.ops.aten.view.default(permute_193, [8, 1, 14, 1, 14, 512]);  permute_193 = None
    permute_194: "f32[8, 1, 1, 14, 14, 512]" = torch.ops.aten.permute.default(view_399, [0, 1, 3, 2, 4, 5]);  view_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:174, code: x = x.reshape(B, grid_size, grid_size, block_size, block_size, C)
    view_400: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(permute_194, [8, 1, 196, 512]);  permute_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_275: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_400, div_69);  div_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_401: "f32[1568, 512]" = torch.ops.aten.view.default(mul_275, [1568, 512]);  mul_275 = None
    mm_2: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_401, permute_195);  permute_195 = None
    permute_196: "f32[512, 1568]" = torch.ops.aten.permute.default(view_401, [1, 0])
    mm_3: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_196, view_392);  permute_196 = view_392 = None
    permute_197: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_30: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_401, [0], True);  view_401 = None
    view_402: "f32[512]" = torch.ops.aten.view.default(sum_30, [512]);  sum_30 = None
    permute_198: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
    view_403: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_2, [8, 1, 196, 2048]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_277: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_173, 0.5);  add_173 = None
    mul_278: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_391, view_391)
    mul_279: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_278, -0.5);  mul_278 = None
    exp_24: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_279);  mul_279 = None
    mul_280: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_281: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_391, mul_280);  view_391 = mul_280 = None
    add_178: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_277, mul_281);  mul_277 = mul_281 = None
    mul_282: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_403, add_178);  view_403 = add_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_404: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_282, [1568, 2048]);  mul_282 = None
    mm_4: "f32[1568, 512]" = torch.ops.aten.mm.default(view_404, permute_199);  permute_199 = None
    permute_200: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_404, [1, 0])
    mm_5: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_200, view_390);  permute_200 = view_390 = None
    permute_201: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_31: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_404, [0], True);  view_404 = None
    view_405: "f32[2048]" = torch.ops.aten.view.default(sum_31, [2048]);  sum_31 = None
    permute_202: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
    view_406: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_4, [8, 1, 196, 512]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_284: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_406, primals_102);  primals_102 = None
    mul_285: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_284, 512)
    sum_32: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_284, [3], True)
    mul_286: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_284, mul_260);  mul_284 = None
    sum_33: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_286, [3], True);  mul_286 = None
    mul_287: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_260, sum_33);  sum_33 = None
    sub_79: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_285, sum_32);  mul_285 = sum_32 = None
    sub_80: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_79, mul_287);  sub_79 = mul_287 = None
    mul_288: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_72, sub_80);  div_72 = sub_80 = None
    mul_289: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_406, mul_260);  mul_260 = None
    sum_34: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_289, [0, 1, 2]);  mul_289 = None
    sum_35: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_406, [0, 1, 2]);  view_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_179: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(view_400, mul_288);  view_400 = mul_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_290: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_179, div_68);  div_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_407: "f32[1568, 512]" = torch.ops.aten.view.default(mul_290, [1568, 512]);  mul_290 = None
    mm_6: "f32[1568, 512]" = torch.ops.aten.mm.default(view_407, permute_203);  permute_203 = None
    permute_204: "f32[512, 1568]" = torch.ops.aten.permute.default(view_407, [1, 0])
    mm_7: "f32[512, 512]" = torch.ops.aten.mm.default(permute_204, view_388);  permute_204 = view_388 = None
    permute_205: "f32[512, 512]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_36: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_407, [0], True);  view_407 = None
    view_408: "f32[512]" = torch.ops.aten.view.default(sum_36, [512]);  sum_36 = None
    permute_206: "f32[512, 512]" = torch.ops.aten.permute.default(permute_205, [1, 0]);  permute_205 = None
    view_409: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_6, [8, 1, 196, 512]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_410: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_409, [8, 1, 196, 32, 16]);  view_409 = None
    permute_207: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_410, [0, 4, 1, 2, 3]);  view_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_176: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_207, memory_format = torch.contiguous_format);  permute_207 = None
    view_411: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_176, [128, 196, 32]);  clone_176 = None
    bmm_48: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_208, view_411);  permute_208 = None
    bmm_49: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_411, permute_209);  view_411 = permute_209 = None
    view_412: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_48, [8, 16, 1, 196, 32]);  bmm_48 = None
    view_413: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_49, [8, 16, 1, 196, 196]);  bmm_49 = None
    mul_291: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_413, alias_24);  view_413 = None
    sum_37: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_291, [-1], True)
    mul_292: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_24, sum_37);  alias_24 = sum_37 = None
    sub_81: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_291, mul_292);  mul_291 = mul_292 = None
    view_414: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_81, [128, 196, 196]);  sub_81 = None
    bmm_50: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_210, view_414);  permute_210 = None
    bmm_51: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_414, permute_211);  view_414 = permute_211 = None
    view_415: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_50, [8, 16, 1, 32, 196]);  bmm_50 = None
    view_416: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_51, [8, 16, 1, 196, 32]);  bmm_51 = None
    mul_293: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_415, 0.42044820762685725);  view_415 = None
    permute_212: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_293, [0, 1, 2, 4, 3]);  mul_293 = None
    mul_294: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_416, 0.42044820762685725);  view_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_294, permute_212, view_412]);  mul_294 = permute_212 = view_412 = None
    view_417: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat, [3, 8, 16, 1, 196, 32]);  cat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_213: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_417, [1, 3, 4, 0, 2, 5]);  view_417 = None
    clone_177: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_213, memory_format = torch.contiguous_format);  permute_213 = None
    view_418: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_177, [8, 1, 196, 1536]);  clone_177 = None
    view_419: "f32[1568, 1536]" = torch.ops.aten.view.default(view_418, [1568, 1536]);  view_418 = None
    mm_8: "f32[1568, 512]" = torch.ops.aten.mm.default(view_419, permute_214);  permute_214 = None
    permute_215: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_419, [1, 0])
    mm_9: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_215, view_378);  permute_215 = view_378 = None
    permute_216: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_38: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_419, [0], True);  view_419 = None
    view_420: "f32[1536]" = torch.ops.aten.view.default(sum_38, [1536]);  sum_38 = None
    permute_217: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_216, [1, 0]);  permute_216 = None
    view_421: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_8, [8, 1, 196, 512]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_296: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_421, primals_100);  primals_100 = None
    mul_297: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_296, 512)
    sum_39: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_296, [3], True)
    mul_298: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_296, mul_255);  mul_296 = None
    sum_40: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_298, [3], True);  mul_298 = None
    mul_299: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_255, sum_40);  sum_40 = None
    sub_83: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_297, sum_39);  mul_297 = sum_39 = None
    sub_84: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_83, mul_299);  sub_83 = mul_299 = None
    mul_300: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_73, sub_84);  div_73 = sub_84 = None
    mul_301: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_421, mul_255);  mul_255 = None
    sum_41: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_301, [0, 1, 2]);  mul_301 = None
    sum_42: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_421, [0, 1, 2]);  view_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_180: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_179, mul_300);  add_179 = mul_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_302: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_180, div_66);  div_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_422: "f32[1568, 512]" = torch.ops.aten.view.default(mul_302, [1568, 512]);  mul_302 = None
    mm_10: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_422, permute_218);  permute_218 = None
    permute_219: "f32[512, 1568]" = torch.ops.aten.permute.default(view_422, [1, 0])
    mm_11: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_219, view_376);  permute_219 = view_376 = None
    permute_220: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_43: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_422, [0], True);  view_422 = None
    view_423: "f32[512]" = torch.ops.aten.view.default(sum_43, [512]);  sum_43 = None
    permute_221: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
    view_424: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_10, [8, 1, 196, 2048]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_304: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_166, 0.5);  add_166 = None
    mul_305: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_375, view_375)
    mul_306: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_305, -0.5);  mul_305 = None
    exp_25: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_306);  mul_306 = None
    mul_307: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_308: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_375, mul_307);  view_375 = mul_307 = None
    add_182: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_304, mul_308);  mul_304 = mul_308 = None
    mul_309: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_424, add_182);  view_424 = add_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_425: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_309, [1568, 2048]);  mul_309 = None
    mm_12: "f32[1568, 512]" = torch.ops.aten.mm.default(view_425, permute_222);  permute_222 = None
    permute_223: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_425, [1, 0])
    mm_13: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_223, view_374);  permute_223 = view_374 = None
    permute_224: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_44: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_425, [0], True);  view_425 = None
    view_426: "f32[2048]" = torch.ops.aten.view.default(sum_44, [2048]);  sum_44 = None
    permute_225: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_224, [1, 0]);  permute_224 = None
    view_427: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_12, [8, 1, 196, 512]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_311: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_427, primals_98);  primals_98 = None
    mul_312: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_311, 512)
    sum_45: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_311, [3], True)
    mul_313: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_311, mul_249);  mul_311 = None
    sum_46: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_313, [3], True);  mul_313 = None
    mul_314: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_249, sum_46);  sum_46 = None
    sub_86: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_312, sum_45);  mul_312 = sum_45 = None
    sub_87: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_86, mul_314);  sub_86 = mul_314 = None
    mul_315: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_74, sub_87);  div_74 = sub_87 = None
    mul_316: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_427, mul_249);  mul_249 = None
    sum_47: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_316, [0, 1, 2]);  mul_316 = None
    sum_48: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_427, [0, 1, 2]);  view_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_183: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_180, mul_315);  add_180 = mul_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_317: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_183, div_65);  div_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_428: "f32[1568, 512]" = torch.ops.aten.view.default(mul_317, [1568, 512]);  mul_317 = None
    mm_14: "f32[1568, 512]" = torch.ops.aten.mm.default(view_428, permute_226);  permute_226 = None
    permute_227: "f32[512, 1568]" = torch.ops.aten.permute.default(view_428, [1, 0])
    mm_15: "f32[512, 512]" = torch.ops.aten.mm.default(permute_227, view_372);  permute_227 = view_372 = None
    permute_228: "f32[512, 512]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_49: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_428, [0], True);  view_428 = None
    view_429: "f32[512]" = torch.ops.aten.view.default(sum_49, [512]);  sum_49 = None
    permute_229: "f32[512, 512]" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
    view_430: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_14, [8, 1, 196, 512]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_431: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_430, [8, 1, 196, 32, 16]);  view_430 = None
    permute_230: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_431, [0, 4, 1, 2, 3]);  view_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_178: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_230, memory_format = torch.contiguous_format);  permute_230 = None
    view_432: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_178, [128, 196, 32]);  clone_178 = None
    bmm_52: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_231, view_432);  permute_231 = None
    bmm_53: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_432, permute_232);  view_432 = permute_232 = None
    view_433: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_52, [8, 16, 1, 196, 32]);  bmm_52 = None
    view_434: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_53, [8, 16, 1, 196, 196]);  bmm_53 = None
    mul_318: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_434, alias_25);  view_434 = None
    sum_50: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_318, [-1], True)
    mul_319: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_25, sum_50);  alias_25 = sum_50 = None
    sub_88: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_318, mul_319);  mul_318 = mul_319 = None
    view_435: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_88, [128, 196, 196]);  sub_88 = None
    bmm_54: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_233, view_435);  permute_233 = None
    bmm_55: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_435, permute_234);  view_435 = permute_234 = None
    view_436: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_54, [8, 16, 1, 32, 196]);  bmm_54 = None
    view_437: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_55, [8, 16, 1, 196, 32]);  bmm_55 = None
    mul_320: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_436, 0.42044820762685725);  view_436 = None
    permute_235: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_320, [0, 1, 2, 4, 3]);  mul_320 = None
    mul_321: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_437, 0.42044820762685725);  view_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_1: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_321, permute_235, view_433]);  mul_321 = permute_235 = view_433 = None
    view_438: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_1, [3, 8, 16, 1, 196, 32]);  cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_236: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_438, [1, 3, 4, 0, 2, 5]);  view_438 = None
    clone_179: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_236, memory_format = torch.contiguous_format);  permute_236 = None
    view_439: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_179, [8, 1, 196, 1536]);  clone_179 = None
    view_440: "f32[1568, 1536]" = torch.ops.aten.view.default(view_439, [1568, 1536]);  view_439 = None
    mm_16: "f32[1568, 512]" = torch.ops.aten.mm.default(view_440, permute_237);  permute_237 = None
    permute_238: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_440, [1, 0])
    mm_17: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_238, view_362);  permute_238 = view_362 = None
    permute_239: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_51: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_440, [0], True);  view_440 = None
    view_441: "f32[1536]" = torch.ops.aten.view.default(sum_51, [1536]);  sum_51 = None
    permute_240: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
    view_442: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_16, [8, 1, 196, 512]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_323: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_442, primals_96);  primals_96 = None
    mul_324: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_323, 512)
    sum_52: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_323, [3], True)
    mul_325: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_323, mul_244);  mul_323 = None
    sum_53: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_325, [3], True);  mul_325 = None
    mul_326: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_244, sum_53);  sum_53 = None
    sub_90: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_324, sum_52);  mul_324 = sum_52 = None
    sub_91: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_90, mul_326);  sub_90 = mul_326 = None
    mul_327: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_75, sub_91);  div_75 = sub_91 = None
    mul_328: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_442, mul_244);  mul_244 = None
    sum_54: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_328, [0, 1, 2]);  mul_328 = None
    sum_55: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_442, [0, 1, 2]);  view_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_184: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_183, mul_327);  add_183 = mul_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_329: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_184, div_63);  div_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_443: "f32[1568, 512]" = torch.ops.aten.view.default(mul_329, [1568, 512]);  mul_329 = None
    mm_18: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_443, permute_241);  permute_241 = None
    permute_242: "f32[512, 1568]" = torch.ops.aten.permute.default(view_443, [1, 0])
    mm_19: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_242, view_360);  permute_242 = view_360 = None
    permute_243: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_56: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_443, [0], True);  view_443 = None
    view_444: "f32[512]" = torch.ops.aten.view.default(sum_56, [512]);  sum_56 = None
    permute_244: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
    view_445: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_18, [8, 1, 196, 2048]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_331: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_159, 0.5);  add_159 = None
    mul_332: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_359, view_359)
    mul_333: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_332, -0.5);  mul_332 = None
    exp_26: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_333);  mul_333 = None
    mul_334: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_335: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_359, mul_334);  view_359 = mul_334 = None
    add_186: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_331, mul_335);  mul_331 = mul_335 = None
    mul_336: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_445, add_186);  view_445 = add_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_446: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_336, [1568, 2048]);  mul_336 = None
    mm_20: "f32[1568, 512]" = torch.ops.aten.mm.default(view_446, permute_245);  permute_245 = None
    permute_246: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_446, [1, 0])
    mm_21: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_246, view_358);  permute_246 = view_358 = None
    permute_247: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_57: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_446, [0], True);  view_446 = None
    view_447: "f32[2048]" = torch.ops.aten.view.default(sum_57, [2048]);  sum_57 = None
    permute_248: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None
    view_448: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_20, [8, 1, 196, 512]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_338: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_448, primals_94);  primals_94 = None
    mul_339: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_338, 512)
    sum_58: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_338, [3], True)
    mul_340: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_338, mul_238);  mul_338 = None
    sum_59: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_340, [3], True);  mul_340 = None
    mul_341: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_238, sum_59);  sum_59 = None
    sub_93: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_339, sum_58);  mul_339 = sum_58 = None
    sub_94: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_93, mul_341);  sub_93 = mul_341 = None
    mul_342: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_76, sub_94);  div_76 = sub_94 = None
    mul_343: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_448, mul_238);  mul_238 = None
    sum_60: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_343, [0, 1, 2]);  mul_343 = None
    sum_61: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_448, [0, 1, 2]);  view_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_187: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_184, mul_342);  add_184 = mul_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_344: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_187, div_62);  div_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_449: "f32[1568, 512]" = torch.ops.aten.view.default(mul_344, [1568, 512]);  mul_344 = None
    mm_22: "f32[1568, 512]" = torch.ops.aten.mm.default(view_449, permute_249);  permute_249 = None
    permute_250: "f32[512, 1568]" = torch.ops.aten.permute.default(view_449, [1, 0])
    mm_23: "f32[512, 512]" = torch.ops.aten.mm.default(permute_250, view_356);  permute_250 = view_356 = None
    permute_251: "f32[512, 512]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_62: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_449, [0], True);  view_449 = None
    view_450: "f32[512]" = torch.ops.aten.view.default(sum_62, [512]);  sum_62 = None
    permute_252: "f32[512, 512]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
    view_451: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_22, [8, 1, 196, 512]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_452: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_451, [8, 1, 196, 32, 16]);  view_451 = None
    permute_253: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_452, [0, 4, 1, 2, 3]);  view_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_180: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_253, memory_format = torch.contiguous_format);  permute_253 = None
    view_453: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_180, [128, 196, 32]);  clone_180 = None
    bmm_56: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_254, view_453);  permute_254 = None
    bmm_57: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_453, permute_255);  view_453 = permute_255 = None
    view_454: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_56, [8, 16, 1, 196, 32]);  bmm_56 = None
    view_455: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_57, [8, 16, 1, 196, 196]);  bmm_57 = None
    mul_345: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_455, alias_26);  view_455 = None
    sum_63: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_345, [-1], True)
    mul_346: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_26, sum_63);  alias_26 = sum_63 = None
    sub_95: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_345, mul_346);  mul_345 = mul_346 = None
    view_456: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_95, [128, 196, 196]);  sub_95 = None
    bmm_58: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_256, view_456);  permute_256 = None
    bmm_59: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_456, permute_257);  view_456 = permute_257 = None
    view_457: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_58, [8, 16, 1, 32, 196]);  bmm_58 = None
    view_458: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_59, [8, 16, 1, 196, 32]);  bmm_59 = None
    mul_347: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_457, 0.42044820762685725);  view_457 = None
    permute_258: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_347, [0, 1, 2, 4, 3]);  mul_347 = None
    mul_348: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_458, 0.42044820762685725);  view_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_2: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_348, permute_258, view_454]);  mul_348 = permute_258 = view_454 = None
    view_459: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_2, [3, 8, 16, 1, 196, 32]);  cat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_259: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_459, [1, 3, 4, 0, 2, 5]);  view_459 = None
    clone_181: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_259, memory_format = torch.contiguous_format);  permute_259 = None
    view_460: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_181, [8, 1, 196, 1536]);  clone_181 = None
    view_461: "f32[1568, 1536]" = torch.ops.aten.view.default(view_460, [1568, 1536]);  view_460 = None
    mm_24: "f32[1568, 512]" = torch.ops.aten.mm.default(view_461, permute_260);  permute_260 = None
    permute_261: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_461, [1, 0])
    mm_25: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_261, view_346);  permute_261 = view_346 = None
    permute_262: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_64: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_461, [0], True);  view_461 = None
    view_462: "f32[1536]" = torch.ops.aten.view.default(sum_64, [1536]);  sum_64 = None
    permute_263: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
    view_463: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_24, [8, 1, 196, 512]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_350: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_463, primals_92);  primals_92 = None
    mul_351: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_350, 512)
    sum_65: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_350, [3], True)
    mul_352: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_350, mul_233);  mul_350 = None
    sum_66: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_352, [3], True);  mul_352 = None
    mul_353: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_233, sum_66);  sum_66 = None
    sub_97: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_351, sum_65);  mul_351 = sum_65 = None
    sub_98: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_97, mul_353);  sub_97 = mul_353 = None
    mul_354: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_77, sub_98);  div_77 = sub_98 = None
    mul_355: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_463, mul_233);  mul_233 = None
    sum_67: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_355, [0, 1, 2]);  mul_355 = None
    sum_68: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_463, [0, 1, 2]);  view_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_188: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_187, mul_354);  add_187 = mul_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_356: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_188, div_60);  div_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_464: "f32[1568, 512]" = torch.ops.aten.view.default(mul_356, [1568, 512]);  mul_356 = None
    mm_26: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_464, permute_264);  permute_264 = None
    permute_265: "f32[512, 1568]" = torch.ops.aten.permute.default(view_464, [1, 0])
    mm_27: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_265, view_344);  permute_265 = view_344 = None
    permute_266: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_69: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_464, [0], True);  view_464 = None
    view_465: "f32[512]" = torch.ops.aten.view.default(sum_69, [512]);  sum_69 = None
    permute_267: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_266, [1, 0]);  permute_266 = None
    view_466: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_26, [8, 1, 196, 2048]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_358: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_152, 0.5);  add_152 = None
    mul_359: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_343, view_343)
    mul_360: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_359, -0.5);  mul_359 = None
    exp_27: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_360);  mul_360 = None
    mul_361: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_362: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_343, mul_361);  view_343 = mul_361 = None
    add_190: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_358, mul_362);  mul_358 = mul_362 = None
    mul_363: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_466, add_190);  view_466 = add_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_467: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_363, [1568, 2048]);  mul_363 = None
    mm_28: "f32[1568, 512]" = torch.ops.aten.mm.default(view_467, permute_268);  permute_268 = None
    permute_269: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_467, [1, 0])
    mm_29: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_269, view_342);  permute_269 = view_342 = None
    permute_270: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_70: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_467, [0], True);  view_467 = None
    view_468: "f32[2048]" = torch.ops.aten.view.default(sum_70, [2048]);  sum_70 = None
    permute_271: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_270, [1, 0]);  permute_270 = None
    view_469: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_28, [8, 1, 196, 512]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_365: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_469, primals_90);  primals_90 = None
    mul_366: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_365, 512)
    sum_71: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_365, [3], True)
    mul_367: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_365, mul_227);  mul_365 = None
    sum_72: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_367, [3], True);  mul_367 = None
    mul_368: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_227, sum_72);  sum_72 = None
    sub_100: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_366, sum_71);  mul_366 = sum_71 = None
    sub_101: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_100, mul_368);  sub_100 = mul_368 = None
    mul_369: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_78, sub_101);  div_78 = sub_101 = None
    mul_370: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_469, mul_227);  mul_227 = None
    sum_73: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_370, [0, 1, 2]);  mul_370 = None
    sum_74: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_469, [0, 1, 2]);  view_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_191: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_188, mul_369);  add_188 = mul_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_371: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_191, div_59);  div_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_470: "f32[1568, 512]" = torch.ops.aten.view.default(mul_371, [1568, 512]);  mul_371 = None
    mm_30: "f32[1568, 512]" = torch.ops.aten.mm.default(view_470, permute_272);  permute_272 = None
    permute_273: "f32[512, 1568]" = torch.ops.aten.permute.default(view_470, [1, 0])
    mm_31: "f32[512, 512]" = torch.ops.aten.mm.default(permute_273, view_340);  permute_273 = view_340 = None
    permute_274: "f32[512, 512]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_75: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_470, [0], True);  view_470 = None
    view_471: "f32[512]" = torch.ops.aten.view.default(sum_75, [512]);  sum_75 = None
    permute_275: "f32[512, 512]" = torch.ops.aten.permute.default(permute_274, [1, 0]);  permute_274 = None
    view_472: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_30, [8, 1, 196, 512]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_473: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_472, [8, 1, 196, 32, 16]);  view_472 = None
    permute_276: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_473, [0, 4, 1, 2, 3]);  view_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_182: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_276, memory_format = torch.contiguous_format);  permute_276 = None
    view_474: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_182, [128, 196, 32]);  clone_182 = None
    bmm_60: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_277, view_474);  permute_277 = None
    bmm_61: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_474, permute_278);  view_474 = permute_278 = None
    view_475: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_60, [8, 16, 1, 196, 32]);  bmm_60 = None
    view_476: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_61, [8, 16, 1, 196, 196]);  bmm_61 = None
    mul_372: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_476, alias_27);  view_476 = None
    sum_76: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_372, [-1], True)
    mul_373: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_27, sum_76);  alias_27 = sum_76 = None
    sub_102: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_372, mul_373);  mul_372 = mul_373 = None
    view_477: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_102, [128, 196, 196]);  sub_102 = None
    bmm_62: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_279, view_477);  permute_279 = None
    bmm_63: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_477, permute_280);  view_477 = permute_280 = None
    view_478: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_62, [8, 16, 1, 32, 196]);  bmm_62 = None
    view_479: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_63, [8, 16, 1, 196, 32]);  bmm_63 = None
    mul_374: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_478, 0.42044820762685725);  view_478 = None
    permute_281: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_374, [0, 1, 2, 4, 3]);  mul_374 = None
    mul_375: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_479, 0.42044820762685725);  view_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_3: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_375, permute_281, view_475]);  mul_375 = permute_281 = view_475 = None
    view_480: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_3, [3, 8, 16, 1, 196, 32]);  cat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_282: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_480, [1, 3, 4, 0, 2, 5]);  view_480 = None
    clone_183: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_282, memory_format = torch.contiguous_format);  permute_282 = None
    view_481: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_183, [8, 1, 196, 1536]);  clone_183 = None
    view_482: "f32[1568, 1536]" = torch.ops.aten.view.default(view_481, [1568, 1536]);  view_481 = None
    mm_32: "f32[1568, 512]" = torch.ops.aten.mm.default(view_482, permute_283);  permute_283 = None
    permute_284: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_482, [1, 0])
    mm_33: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_284, view_330);  permute_284 = view_330 = None
    permute_285: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_77: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_482, [0], True);  view_482 = None
    view_483: "f32[1536]" = torch.ops.aten.view.default(sum_77, [1536]);  sum_77 = None
    permute_286: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_285, [1, 0]);  permute_285 = None
    view_484: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_32, [8, 1, 196, 512]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_377: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_484, primals_88);  primals_88 = None
    mul_378: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_377, 512)
    sum_78: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_377, [3], True)
    mul_379: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_377, mul_222);  mul_377 = None
    sum_79: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_379, [3], True);  mul_379 = None
    mul_380: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_222, sum_79);  sum_79 = None
    sub_104: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_378, sum_78);  mul_378 = sum_78 = None
    sub_105: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_104, mul_380);  sub_104 = mul_380 = None
    mul_381: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_79, sub_105);  div_79 = sub_105 = None
    mul_382: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_484, mul_222);  mul_222 = None
    sum_80: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_382, [0, 1, 2]);  mul_382 = None
    sum_81: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_484, [0, 1, 2]);  view_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_192: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_191, mul_381);  add_191 = mul_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_383: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_192, div_57);  div_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_485: "f32[1568, 512]" = torch.ops.aten.view.default(mul_383, [1568, 512]);  mul_383 = None
    mm_34: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_485, permute_287);  permute_287 = None
    permute_288: "f32[512, 1568]" = torch.ops.aten.permute.default(view_485, [1, 0])
    mm_35: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_288, view_328);  permute_288 = view_328 = None
    permute_289: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_82: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_485, [0], True);  view_485 = None
    view_486: "f32[512]" = torch.ops.aten.view.default(sum_82, [512]);  sum_82 = None
    permute_290: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_289, [1, 0]);  permute_289 = None
    view_487: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_34, [8, 1, 196, 2048]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_385: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_145, 0.5);  add_145 = None
    mul_386: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_327, view_327)
    mul_387: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_386, -0.5);  mul_386 = None
    exp_28: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_387);  mul_387 = None
    mul_388: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
    mul_389: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_327, mul_388);  view_327 = mul_388 = None
    add_194: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_385, mul_389);  mul_385 = mul_389 = None
    mul_390: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_487, add_194);  view_487 = add_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_488: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_390, [1568, 2048]);  mul_390 = None
    mm_36: "f32[1568, 512]" = torch.ops.aten.mm.default(view_488, permute_291);  permute_291 = None
    permute_292: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_488, [1, 0])
    mm_37: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_292, view_326);  permute_292 = view_326 = None
    permute_293: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_83: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_488, [0], True);  view_488 = None
    view_489: "f32[2048]" = torch.ops.aten.view.default(sum_83, [2048]);  sum_83 = None
    permute_294: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_293, [1, 0]);  permute_293 = None
    view_490: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_36, [8, 1, 196, 512]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_392: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_490, primals_86);  primals_86 = None
    mul_393: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_392, 512)
    sum_84: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_392, [3], True)
    mul_394: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_392, mul_216);  mul_392 = None
    sum_85: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_394, [3], True);  mul_394 = None
    mul_395: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_216, sum_85);  sum_85 = None
    sub_107: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_393, sum_84);  mul_393 = sum_84 = None
    sub_108: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_107, mul_395);  sub_107 = mul_395 = None
    mul_396: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_80, sub_108);  div_80 = sub_108 = None
    mul_397: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_490, mul_216);  mul_216 = None
    sum_86: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_397, [0, 1, 2]);  mul_397 = None
    sum_87: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_490, [0, 1, 2]);  view_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_195: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_192, mul_396);  add_192 = mul_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_398: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_195, div_56);  div_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_491: "f32[1568, 512]" = torch.ops.aten.view.default(mul_398, [1568, 512]);  mul_398 = None
    mm_38: "f32[1568, 512]" = torch.ops.aten.mm.default(view_491, permute_295);  permute_295 = None
    permute_296: "f32[512, 1568]" = torch.ops.aten.permute.default(view_491, [1, 0])
    mm_39: "f32[512, 512]" = torch.ops.aten.mm.default(permute_296, view_324);  permute_296 = view_324 = None
    permute_297: "f32[512, 512]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_88: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_491, [0], True);  view_491 = None
    view_492: "f32[512]" = torch.ops.aten.view.default(sum_88, [512]);  sum_88 = None
    permute_298: "f32[512, 512]" = torch.ops.aten.permute.default(permute_297, [1, 0]);  permute_297 = None
    view_493: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_38, [8, 1, 196, 512]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_494: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_493, [8, 1, 196, 32, 16]);  view_493 = None
    permute_299: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_494, [0, 4, 1, 2, 3]);  view_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_184: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_299, memory_format = torch.contiguous_format);  permute_299 = None
    view_495: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_184, [128, 196, 32]);  clone_184 = None
    bmm_64: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_300, view_495);  permute_300 = None
    bmm_65: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_495, permute_301);  view_495 = permute_301 = None
    view_496: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_64, [8, 16, 1, 196, 32]);  bmm_64 = None
    view_497: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_65, [8, 16, 1, 196, 196]);  bmm_65 = None
    mul_399: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_497, alias_28);  view_497 = None
    sum_89: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_399, [-1], True)
    mul_400: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_28, sum_89);  alias_28 = sum_89 = None
    sub_109: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_399, mul_400);  mul_399 = mul_400 = None
    view_498: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_109, [128, 196, 196]);  sub_109 = None
    bmm_66: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_302, view_498);  permute_302 = None
    bmm_67: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_498, permute_303);  view_498 = permute_303 = None
    view_499: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_66, [8, 16, 1, 32, 196]);  bmm_66 = None
    view_500: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_67, [8, 16, 1, 196, 32]);  bmm_67 = None
    mul_401: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_499, 0.42044820762685725);  view_499 = None
    permute_304: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_401, [0, 1, 2, 4, 3]);  mul_401 = None
    mul_402: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_500, 0.42044820762685725);  view_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_4: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_402, permute_304, view_496]);  mul_402 = permute_304 = view_496 = None
    view_501: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_4, [3, 8, 16, 1, 196, 32]);  cat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_305: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_501, [1, 3, 4, 0, 2, 5]);  view_501 = None
    clone_185: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_305, memory_format = torch.contiguous_format);  permute_305 = None
    view_502: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_185, [8, 1, 196, 1536]);  clone_185 = None
    view_503: "f32[1568, 1536]" = torch.ops.aten.view.default(view_502, [1568, 1536]);  view_502 = None
    mm_40: "f32[1568, 512]" = torch.ops.aten.mm.default(view_503, permute_306);  permute_306 = None
    permute_307: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_503, [1, 0])
    mm_41: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_307, view_314);  permute_307 = view_314 = None
    permute_308: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_90: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_503, [0], True);  view_503 = None
    view_504: "f32[1536]" = torch.ops.aten.view.default(sum_90, [1536]);  sum_90 = None
    permute_309: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_308, [1, 0]);  permute_308 = None
    view_505: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_40, [8, 1, 196, 512]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_404: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_505, primals_84);  primals_84 = None
    mul_405: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_404, 512)
    sum_91: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_404, [3], True)
    mul_406: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_404, mul_211);  mul_404 = None
    sum_92: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_406, [3], True);  mul_406 = None
    mul_407: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_211, sum_92);  sum_92 = None
    sub_111: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_405, sum_91);  mul_405 = sum_91 = None
    sub_112: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_111, mul_407);  sub_111 = mul_407 = None
    mul_408: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_81, sub_112);  div_81 = sub_112 = None
    mul_409: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_505, mul_211);  mul_211 = None
    sum_93: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_409, [0, 1, 2]);  mul_409 = None
    sum_94: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_505, [0, 1, 2]);  view_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_196: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_195, mul_408);  add_195 = mul_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_410: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_196, div_54);  div_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_506: "f32[1568, 512]" = torch.ops.aten.view.default(mul_410, [1568, 512]);  mul_410 = None
    mm_42: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_506, permute_310);  permute_310 = None
    permute_311: "f32[512, 1568]" = torch.ops.aten.permute.default(view_506, [1, 0])
    mm_43: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_311, view_312);  permute_311 = view_312 = None
    permute_312: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_95: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_506, [0], True);  view_506 = None
    view_507: "f32[512]" = torch.ops.aten.view.default(sum_95, [512]);  sum_95 = None
    permute_313: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_312, [1, 0]);  permute_312 = None
    view_508: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_42, [8, 1, 196, 2048]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_412: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_138, 0.5);  add_138 = None
    mul_413: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_311, view_311)
    mul_414: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_413, -0.5);  mul_413 = None
    exp_29: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_414);  mul_414 = None
    mul_415: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
    mul_416: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_311, mul_415);  view_311 = mul_415 = None
    add_198: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_412, mul_416);  mul_412 = mul_416 = None
    mul_417: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_508, add_198);  view_508 = add_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_509: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_417, [1568, 2048]);  mul_417 = None
    mm_44: "f32[1568, 512]" = torch.ops.aten.mm.default(view_509, permute_314);  permute_314 = None
    permute_315: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_509, [1, 0])
    mm_45: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_315, view_310);  permute_315 = view_310 = None
    permute_316: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_96: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_509, [0], True);  view_509 = None
    view_510: "f32[2048]" = torch.ops.aten.view.default(sum_96, [2048]);  sum_96 = None
    permute_317: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_316, [1, 0]);  permute_316 = None
    view_511: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_44, [8, 1, 196, 512]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_419: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_511, primals_82);  primals_82 = None
    mul_420: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_419, 512)
    sum_97: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_419, [3], True)
    mul_421: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_419, mul_205);  mul_419 = None
    sum_98: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_421, [3], True);  mul_421 = None
    mul_422: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_205, sum_98);  sum_98 = None
    sub_114: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_420, sum_97);  mul_420 = sum_97 = None
    sub_115: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_114, mul_422);  sub_114 = mul_422 = None
    mul_423: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_82, sub_115);  div_82 = sub_115 = None
    mul_424: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_511, mul_205);  mul_205 = None
    sum_99: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_424, [0, 1, 2]);  mul_424 = None
    sum_100: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_511, [0, 1, 2]);  view_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_199: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_196, mul_423);  add_196 = mul_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_425: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_199, div_53);  div_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_512: "f32[1568, 512]" = torch.ops.aten.view.default(mul_425, [1568, 512]);  mul_425 = None
    mm_46: "f32[1568, 512]" = torch.ops.aten.mm.default(view_512, permute_318);  permute_318 = None
    permute_319: "f32[512, 1568]" = torch.ops.aten.permute.default(view_512, [1, 0])
    mm_47: "f32[512, 512]" = torch.ops.aten.mm.default(permute_319, view_308);  permute_319 = view_308 = None
    permute_320: "f32[512, 512]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_101: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_512, [0], True);  view_512 = None
    view_513: "f32[512]" = torch.ops.aten.view.default(sum_101, [512]);  sum_101 = None
    permute_321: "f32[512, 512]" = torch.ops.aten.permute.default(permute_320, [1, 0]);  permute_320 = None
    view_514: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_46, [8, 1, 196, 512]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_515: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_514, [8, 1, 196, 32, 16]);  view_514 = None
    permute_322: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_515, [0, 4, 1, 2, 3]);  view_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_186: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_322, memory_format = torch.contiguous_format);  permute_322 = None
    view_516: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_186, [128, 196, 32]);  clone_186 = None
    bmm_68: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_323, view_516);  permute_323 = None
    bmm_69: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_516, permute_324);  view_516 = permute_324 = None
    view_517: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_68, [8, 16, 1, 196, 32]);  bmm_68 = None
    view_518: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_69, [8, 16, 1, 196, 196]);  bmm_69 = None
    mul_426: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_518, alias_29);  view_518 = None
    sum_102: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_426, [-1], True)
    mul_427: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_29, sum_102);  alias_29 = sum_102 = None
    sub_116: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_426, mul_427);  mul_426 = mul_427 = None
    view_519: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_116, [128, 196, 196]);  sub_116 = None
    bmm_70: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_325, view_519);  permute_325 = None
    bmm_71: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_519, permute_326);  view_519 = permute_326 = None
    view_520: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_70, [8, 16, 1, 32, 196]);  bmm_70 = None
    view_521: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_71, [8, 16, 1, 196, 32]);  bmm_71 = None
    mul_428: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_520, 0.42044820762685725);  view_520 = None
    permute_327: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_428, [0, 1, 2, 4, 3]);  mul_428 = None
    mul_429: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_521, 0.42044820762685725);  view_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_5: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_429, permute_327, view_517]);  mul_429 = permute_327 = view_517 = None
    view_522: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_5, [3, 8, 16, 1, 196, 32]);  cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_328: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_522, [1, 3, 4, 0, 2, 5]);  view_522 = None
    clone_187: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_328, memory_format = torch.contiguous_format);  permute_328 = None
    view_523: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_187, [8, 1, 196, 1536]);  clone_187 = None
    view_524: "f32[1568, 1536]" = torch.ops.aten.view.default(view_523, [1568, 1536]);  view_523 = None
    mm_48: "f32[1568, 512]" = torch.ops.aten.mm.default(view_524, permute_329);  permute_329 = None
    permute_330: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_524, [1, 0])
    mm_49: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_330, view_298);  permute_330 = view_298 = None
    permute_331: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_103: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_524, [0], True);  view_524 = None
    view_525: "f32[1536]" = torch.ops.aten.view.default(sum_103, [1536]);  sum_103 = None
    permute_332: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_331, [1, 0]);  permute_331 = None
    view_526: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_48, [8, 1, 196, 512]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_431: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_526, primals_80);  primals_80 = None
    mul_432: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_431, 512)
    sum_104: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_431, [3], True)
    mul_433: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_431, mul_200);  mul_431 = None
    sum_105: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_433, [3], True);  mul_433 = None
    mul_434: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_200, sum_105);  sum_105 = None
    sub_118: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_432, sum_104);  mul_432 = sum_104 = None
    sub_119: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_118, mul_434);  sub_118 = mul_434 = None
    mul_435: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_83, sub_119);  div_83 = sub_119 = None
    mul_436: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_526, mul_200);  mul_200 = None
    sum_106: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_436, [0, 1, 2]);  mul_436 = None
    sum_107: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_526, [0, 1, 2]);  view_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_200: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_199, mul_435);  add_199 = mul_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_437: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_200, div_51);  div_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_527: "f32[1568, 512]" = torch.ops.aten.view.default(mul_437, [1568, 512]);  mul_437 = None
    mm_50: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_527, permute_333);  permute_333 = None
    permute_334: "f32[512, 1568]" = torch.ops.aten.permute.default(view_527, [1, 0])
    mm_51: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_334, view_296);  permute_334 = view_296 = None
    permute_335: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_108: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_527, [0], True);  view_527 = None
    view_528: "f32[512]" = torch.ops.aten.view.default(sum_108, [512]);  sum_108 = None
    permute_336: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_335, [1, 0]);  permute_335 = None
    view_529: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_50, [8, 1, 196, 2048]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_439: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_131, 0.5);  add_131 = None
    mul_440: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_295, view_295)
    mul_441: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_440, -0.5);  mul_440 = None
    exp_30: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_441);  mul_441 = None
    mul_442: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_30, 0.3989422804014327);  exp_30 = None
    mul_443: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_295, mul_442);  view_295 = mul_442 = None
    add_202: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_439, mul_443);  mul_439 = mul_443 = None
    mul_444: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_529, add_202);  view_529 = add_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_530: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_444, [1568, 2048]);  mul_444 = None
    mm_52: "f32[1568, 512]" = torch.ops.aten.mm.default(view_530, permute_337);  permute_337 = None
    permute_338: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_530, [1, 0])
    mm_53: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_338, view_294);  permute_338 = view_294 = None
    permute_339: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_109: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_530, [0], True);  view_530 = None
    view_531: "f32[2048]" = torch.ops.aten.view.default(sum_109, [2048]);  sum_109 = None
    permute_340: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_339, [1, 0]);  permute_339 = None
    view_532: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_52, [8, 1, 196, 512]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_446: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_532, primals_78);  primals_78 = None
    mul_447: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_446, 512)
    sum_110: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_446, [3], True)
    mul_448: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_446, mul_194);  mul_446 = None
    sum_111: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_448, [3], True);  mul_448 = None
    mul_449: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_194, sum_111);  sum_111 = None
    sub_121: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_447, sum_110);  mul_447 = sum_110 = None
    sub_122: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_121, mul_449);  sub_121 = mul_449 = None
    mul_450: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_84, sub_122);  div_84 = sub_122 = None
    mul_451: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_532, mul_194);  mul_194 = None
    sum_112: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_451, [0, 1, 2]);  mul_451 = None
    sum_113: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_532, [0, 1, 2]);  view_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_203: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_200, mul_450);  add_200 = mul_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_452: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_203, div_50);  div_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_533: "f32[1568, 512]" = torch.ops.aten.view.default(mul_452, [1568, 512]);  mul_452 = None
    mm_54: "f32[1568, 512]" = torch.ops.aten.mm.default(view_533, permute_341);  permute_341 = None
    permute_342: "f32[512, 1568]" = torch.ops.aten.permute.default(view_533, [1, 0])
    mm_55: "f32[512, 512]" = torch.ops.aten.mm.default(permute_342, view_292);  permute_342 = view_292 = None
    permute_343: "f32[512, 512]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_114: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_533, [0], True);  view_533 = None
    view_534: "f32[512]" = torch.ops.aten.view.default(sum_114, [512]);  sum_114 = None
    permute_344: "f32[512, 512]" = torch.ops.aten.permute.default(permute_343, [1, 0]);  permute_343 = None
    view_535: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_54, [8, 1, 196, 512]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_536: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_535, [8, 1, 196, 32, 16]);  view_535 = None
    permute_345: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_536, [0, 4, 1, 2, 3]);  view_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_188: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_345, memory_format = torch.contiguous_format);  permute_345 = None
    view_537: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_188, [128, 196, 32]);  clone_188 = None
    bmm_72: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_346, view_537);  permute_346 = None
    bmm_73: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_537, permute_347);  view_537 = permute_347 = None
    view_538: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_72, [8, 16, 1, 196, 32]);  bmm_72 = None
    view_539: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_73, [8, 16, 1, 196, 196]);  bmm_73 = None
    mul_453: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_539, alias_30);  view_539 = None
    sum_115: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_453, [-1], True)
    mul_454: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_30, sum_115);  alias_30 = sum_115 = None
    sub_123: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_453, mul_454);  mul_453 = mul_454 = None
    view_540: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_123, [128, 196, 196]);  sub_123 = None
    bmm_74: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_348, view_540);  permute_348 = None
    bmm_75: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_540, permute_349);  view_540 = permute_349 = None
    view_541: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_74, [8, 16, 1, 32, 196]);  bmm_74 = None
    view_542: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_75, [8, 16, 1, 196, 32]);  bmm_75 = None
    mul_455: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_541, 0.42044820762685725);  view_541 = None
    permute_350: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_455, [0, 1, 2, 4, 3]);  mul_455 = None
    mul_456: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_542, 0.42044820762685725);  view_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_6: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_456, permute_350, view_538]);  mul_456 = permute_350 = view_538 = None
    view_543: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_6, [3, 8, 16, 1, 196, 32]);  cat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_351: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_543, [1, 3, 4, 0, 2, 5]);  view_543 = None
    clone_189: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_351, memory_format = torch.contiguous_format);  permute_351 = None
    view_544: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_189, [8, 1, 196, 1536]);  clone_189 = None
    view_545: "f32[1568, 1536]" = torch.ops.aten.view.default(view_544, [1568, 1536]);  view_544 = None
    mm_56: "f32[1568, 512]" = torch.ops.aten.mm.default(view_545, permute_352);  permute_352 = None
    permute_353: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_545, [1, 0])
    mm_57: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_353, view_282);  permute_353 = view_282 = None
    permute_354: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_116: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_545, [0], True);  view_545 = None
    view_546: "f32[1536]" = torch.ops.aten.view.default(sum_116, [1536]);  sum_116 = None
    permute_355: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_354, [1, 0]);  permute_354 = None
    view_547: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_56, [8, 1, 196, 512]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_458: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_547, primals_76);  primals_76 = None
    mul_459: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_458, 512)
    sum_117: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_458, [3], True)
    mul_460: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_458, mul_189);  mul_458 = None
    sum_118: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_460, [3], True);  mul_460 = None
    mul_461: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_189, sum_118);  sum_118 = None
    sub_125: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_459, sum_117);  mul_459 = sum_117 = None
    sub_126: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_125, mul_461);  sub_125 = mul_461 = None
    mul_462: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_85, sub_126);  div_85 = sub_126 = None
    mul_463: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_547, mul_189);  mul_189 = None
    sum_119: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_463, [0, 1, 2]);  mul_463 = None
    sum_120: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_547, [0, 1, 2]);  view_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_204: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_203, mul_462);  add_203 = mul_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_464: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_204, div_48);  div_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_548: "f32[1568, 512]" = torch.ops.aten.view.default(mul_464, [1568, 512]);  mul_464 = None
    mm_58: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_548, permute_356);  permute_356 = None
    permute_357: "f32[512, 1568]" = torch.ops.aten.permute.default(view_548, [1, 0])
    mm_59: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_357, view_280);  permute_357 = view_280 = None
    permute_358: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_121: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_548, [0], True);  view_548 = None
    view_549: "f32[512]" = torch.ops.aten.view.default(sum_121, [512]);  sum_121 = None
    permute_359: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_358, [1, 0]);  permute_358 = None
    view_550: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_58, [8, 1, 196, 2048]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_466: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_124, 0.5);  add_124 = None
    mul_467: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_279, view_279)
    mul_468: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_467, -0.5);  mul_467 = None
    exp_31: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_468);  mul_468 = None
    mul_469: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_31, 0.3989422804014327);  exp_31 = None
    mul_470: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_279, mul_469);  view_279 = mul_469 = None
    add_206: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_466, mul_470);  mul_466 = mul_470 = None
    mul_471: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_550, add_206);  view_550 = add_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_551: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_471, [1568, 2048]);  mul_471 = None
    mm_60: "f32[1568, 512]" = torch.ops.aten.mm.default(view_551, permute_360);  permute_360 = None
    permute_361: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_551, [1, 0])
    mm_61: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_361, view_278);  permute_361 = view_278 = None
    permute_362: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_122: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_551, [0], True);  view_551 = None
    view_552: "f32[2048]" = torch.ops.aten.view.default(sum_122, [2048]);  sum_122 = None
    permute_363: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_362, [1, 0]);  permute_362 = None
    view_553: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_60, [8, 1, 196, 512]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_473: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_553, primals_74);  primals_74 = None
    mul_474: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_473, 512)
    sum_123: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_473, [3], True)
    mul_475: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_473, mul_183);  mul_473 = None
    sum_124: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_475, [3], True);  mul_475 = None
    mul_476: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_183, sum_124);  sum_124 = None
    sub_128: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_474, sum_123);  mul_474 = sum_123 = None
    sub_129: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_128, mul_476);  sub_128 = mul_476 = None
    mul_477: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_86, sub_129);  div_86 = sub_129 = None
    mul_478: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_553, mul_183);  mul_183 = None
    sum_125: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_478, [0, 1, 2]);  mul_478 = None
    sum_126: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_553, [0, 1, 2]);  view_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_207: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_204, mul_477);  add_204 = mul_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_479: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_207, div_47);  div_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_554: "f32[1568, 512]" = torch.ops.aten.view.default(mul_479, [1568, 512]);  mul_479 = None
    mm_62: "f32[1568, 512]" = torch.ops.aten.mm.default(view_554, permute_364);  permute_364 = None
    permute_365: "f32[512, 1568]" = torch.ops.aten.permute.default(view_554, [1, 0])
    mm_63: "f32[512, 512]" = torch.ops.aten.mm.default(permute_365, view_276);  permute_365 = view_276 = None
    permute_366: "f32[512, 512]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_127: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_554, [0], True);  view_554 = None
    view_555: "f32[512]" = torch.ops.aten.view.default(sum_127, [512]);  sum_127 = None
    permute_367: "f32[512, 512]" = torch.ops.aten.permute.default(permute_366, [1, 0]);  permute_366 = None
    view_556: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_62, [8, 1, 196, 512]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_557: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_556, [8, 1, 196, 32, 16]);  view_556 = None
    permute_368: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_557, [0, 4, 1, 2, 3]);  view_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_190: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_368, memory_format = torch.contiguous_format);  permute_368 = None
    view_558: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_190, [128, 196, 32]);  clone_190 = None
    bmm_76: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_369, view_558);  permute_369 = None
    bmm_77: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_558, permute_370);  view_558 = permute_370 = None
    view_559: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_76, [8, 16, 1, 196, 32]);  bmm_76 = None
    view_560: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_77, [8, 16, 1, 196, 196]);  bmm_77 = None
    mul_480: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_560, alias_31);  view_560 = None
    sum_128: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_480, [-1], True)
    mul_481: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_31, sum_128);  alias_31 = sum_128 = None
    sub_130: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_480, mul_481);  mul_480 = mul_481 = None
    view_561: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_130, [128, 196, 196]);  sub_130 = None
    bmm_78: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_371, view_561);  permute_371 = None
    bmm_79: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_561, permute_372);  view_561 = permute_372 = None
    view_562: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_78, [8, 16, 1, 32, 196]);  bmm_78 = None
    view_563: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_79, [8, 16, 1, 196, 32]);  bmm_79 = None
    mul_482: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_562, 0.42044820762685725);  view_562 = None
    permute_373: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_482, [0, 1, 2, 4, 3]);  mul_482 = None
    mul_483: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_563, 0.42044820762685725);  view_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_7: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_483, permute_373, view_559]);  mul_483 = permute_373 = view_559 = None
    view_564: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_7, [3, 8, 16, 1, 196, 32]);  cat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_374: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_564, [1, 3, 4, 0, 2, 5]);  view_564 = None
    clone_191: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_374, memory_format = torch.contiguous_format);  permute_374 = None
    view_565: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_191, [8, 1, 196, 1536]);  clone_191 = None
    view_566: "f32[1568, 1536]" = torch.ops.aten.view.default(view_565, [1568, 1536]);  view_565 = None
    mm_64: "f32[1568, 512]" = torch.ops.aten.mm.default(view_566, permute_375);  permute_375 = None
    permute_376: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_566, [1, 0])
    mm_65: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_376, view_266);  permute_376 = view_266 = None
    permute_377: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_129: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_566, [0], True);  view_566 = None
    view_567: "f32[1536]" = torch.ops.aten.view.default(sum_129, [1536]);  sum_129 = None
    permute_378: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_377, [1, 0]);  permute_377 = None
    view_568: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_64, [8, 1, 196, 512]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_485: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_568, primals_72);  primals_72 = None
    mul_486: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_485, 512)
    sum_130: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_485, [3], True)
    mul_487: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_485, mul_178);  mul_485 = None
    sum_131: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_487, [3], True);  mul_487 = None
    mul_488: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_178, sum_131);  sum_131 = None
    sub_132: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_486, sum_130);  mul_486 = sum_130 = None
    sub_133: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_132, mul_488);  sub_132 = mul_488 = None
    mul_489: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_87, sub_133);  div_87 = sub_133 = None
    mul_490: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_568, mul_178);  mul_178 = None
    sum_132: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_490, [0, 1, 2]);  mul_490 = None
    sum_133: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_568, [0, 1, 2]);  view_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_208: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_207, mul_489);  add_207 = mul_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_491: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_208, div_45);  div_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_569: "f32[1568, 512]" = torch.ops.aten.view.default(mul_491, [1568, 512]);  mul_491 = None
    mm_66: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_569, permute_379);  permute_379 = None
    permute_380: "f32[512, 1568]" = torch.ops.aten.permute.default(view_569, [1, 0])
    mm_67: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_380, view_264);  permute_380 = view_264 = None
    permute_381: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_134: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_569, [0], True);  view_569 = None
    view_570: "f32[512]" = torch.ops.aten.view.default(sum_134, [512]);  sum_134 = None
    permute_382: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_381, [1, 0]);  permute_381 = None
    view_571: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_66, [8, 1, 196, 2048]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_493: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_117, 0.5);  add_117 = None
    mul_494: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_263, view_263)
    mul_495: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_494, -0.5);  mul_494 = None
    exp_32: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_495);  mul_495 = None
    mul_496: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_32, 0.3989422804014327);  exp_32 = None
    mul_497: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_263, mul_496);  view_263 = mul_496 = None
    add_210: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_493, mul_497);  mul_493 = mul_497 = None
    mul_498: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_571, add_210);  view_571 = add_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_572: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_498, [1568, 2048]);  mul_498 = None
    mm_68: "f32[1568, 512]" = torch.ops.aten.mm.default(view_572, permute_383);  permute_383 = None
    permute_384: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_572, [1, 0])
    mm_69: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_384, view_262);  permute_384 = view_262 = None
    permute_385: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_135: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_572, [0], True);  view_572 = None
    view_573: "f32[2048]" = torch.ops.aten.view.default(sum_135, [2048]);  sum_135 = None
    permute_386: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_385, [1, 0]);  permute_385 = None
    view_574: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_68, [8, 1, 196, 512]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_500: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_574, primals_70);  primals_70 = None
    mul_501: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_500, 512)
    sum_136: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_500, [3], True)
    mul_502: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_500, mul_172);  mul_500 = None
    sum_137: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_502, [3], True);  mul_502 = None
    mul_503: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_172, sum_137);  sum_137 = None
    sub_135: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_501, sum_136);  mul_501 = sum_136 = None
    sub_136: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_135, mul_503);  sub_135 = mul_503 = None
    mul_504: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_88, sub_136);  div_88 = sub_136 = None
    mul_505: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_574, mul_172);  mul_172 = None
    sum_138: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_505, [0, 1, 2]);  mul_505 = None
    sum_139: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_574, [0, 1, 2]);  view_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_211: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_208, mul_504);  add_208 = mul_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_506: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_211, div_44);  div_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_575: "f32[1568, 512]" = torch.ops.aten.view.default(mul_506, [1568, 512]);  mul_506 = None
    mm_70: "f32[1568, 512]" = torch.ops.aten.mm.default(view_575, permute_387);  permute_387 = None
    permute_388: "f32[512, 1568]" = torch.ops.aten.permute.default(view_575, [1, 0])
    mm_71: "f32[512, 512]" = torch.ops.aten.mm.default(permute_388, view_260);  permute_388 = view_260 = None
    permute_389: "f32[512, 512]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_140: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_575, [0], True);  view_575 = None
    view_576: "f32[512]" = torch.ops.aten.view.default(sum_140, [512]);  sum_140 = None
    permute_390: "f32[512, 512]" = torch.ops.aten.permute.default(permute_389, [1, 0]);  permute_389 = None
    view_577: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_70, [8, 1, 196, 512]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_578: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_577, [8, 1, 196, 32, 16]);  view_577 = None
    permute_391: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_578, [0, 4, 1, 2, 3]);  view_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_192: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_391, memory_format = torch.contiguous_format);  permute_391 = None
    view_579: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_192, [128, 196, 32]);  clone_192 = None
    bmm_80: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_392, view_579);  permute_392 = None
    bmm_81: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_579, permute_393);  view_579 = permute_393 = None
    view_580: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_80, [8, 16, 1, 196, 32]);  bmm_80 = None
    view_581: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_81, [8, 16, 1, 196, 196]);  bmm_81 = None
    mul_507: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_581, alias_32);  view_581 = None
    sum_141: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_507, [-1], True)
    mul_508: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_32, sum_141);  alias_32 = sum_141 = None
    sub_137: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_507, mul_508);  mul_507 = mul_508 = None
    view_582: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_137, [128, 196, 196]);  sub_137 = None
    bmm_82: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_394, view_582);  permute_394 = None
    bmm_83: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_582, permute_395);  view_582 = permute_395 = None
    view_583: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_82, [8, 16, 1, 32, 196]);  bmm_82 = None
    view_584: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_83, [8, 16, 1, 196, 32]);  bmm_83 = None
    mul_509: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_583, 0.42044820762685725);  view_583 = None
    permute_396: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_509, [0, 1, 2, 4, 3]);  mul_509 = None
    mul_510: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_584, 0.42044820762685725);  view_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_8: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_510, permute_396, view_580]);  mul_510 = permute_396 = view_580 = None
    view_585: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_8, [3, 8, 16, 1, 196, 32]);  cat_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_397: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_585, [1, 3, 4, 0, 2, 5]);  view_585 = None
    clone_193: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_397, memory_format = torch.contiguous_format);  permute_397 = None
    view_586: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_193, [8, 1, 196, 1536]);  clone_193 = None
    view_587: "f32[1568, 1536]" = torch.ops.aten.view.default(view_586, [1568, 1536]);  view_586 = None
    mm_72: "f32[1568, 512]" = torch.ops.aten.mm.default(view_587, permute_398);  permute_398 = None
    permute_399: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_587, [1, 0])
    mm_73: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_399, view_250);  permute_399 = view_250 = None
    permute_400: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_142: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_587, [0], True);  view_587 = None
    view_588: "f32[1536]" = torch.ops.aten.view.default(sum_142, [1536]);  sum_142 = None
    permute_401: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_400, [1, 0]);  permute_400 = None
    view_589: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_72, [8, 1, 196, 512]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_512: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_589, primals_68);  primals_68 = None
    mul_513: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_512, 512)
    sum_143: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_512, [3], True)
    mul_514: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_512, mul_167);  mul_512 = None
    sum_144: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_514, [3], True);  mul_514 = None
    mul_515: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_167, sum_144);  sum_144 = None
    sub_139: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_513, sum_143);  mul_513 = sum_143 = None
    sub_140: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_139, mul_515);  sub_139 = mul_515 = None
    mul_516: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_89, sub_140);  div_89 = sub_140 = None
    mul_517: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_589, mul_167);  mul_167 = None
    sum_145: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_517, [0, 1, 2]);  mul_517 = None
    sum_146: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_589, [0, 1, 2]);  view_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_212: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_211, mul_516);  add_211 = mul_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_518: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_212, div_42);  div_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_590: "f32[1568, 512]" = torch.ops.aten.view.default(mul_518, [1568, 512]);  mul_518 = None
    mm_74: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_590, permute_402);  permute_402 = None
    permute_403: "f32[512, 1568]" = torch.ops.aten.permute.default(view_590, [1, 0])
    mm_75: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_403, view_248);  permute_403 = view_248 = None
    permute_404: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_147: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_590, [0], True);  view_590 = None
    view_591: "f32[512]" = torch.ops.aten.view.default(sum_147, [512]);  sum_147 = None
    permute_405: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_404, [1, 0]);  permute_404 = None
    view_592: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_74, [8, 1, 196, 2048]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_520: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_110, 0.5);  add_110 = None
    mul_521: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_247, view_247)
    mul_522: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_521, -0.5);  mul_521 = None
    exp_33: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_522);  mul_522 = None
    mul_523: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_33, 0.3989422804014327);  exp_33 = None
    mul_524: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_247, mul_523);  view_247 = mul_523 = None
    add_214: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_520, mul_524);  mul_520 = mul_524 = None
    mul_525: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_592, add_214);  view_592 = add_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_593: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_525, [1568, 2048]);  mul_525 = None
    mm_76: "f32[1568, 512]" = torch.ops.aten.mm.default(view_593, permute_406);  permute_406 = None
    permute_407: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_593, [1, 0])
    mm_77: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_407, view_246);  permute_407 = view_246 = None
    permute_408: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_148: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_593, [0], True);  view_593 = None
    view_594: "f32[2048]" = torch.ops.aten.view.default(sum_148, [2048]);  sum_148 = None
    permute_409: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_408, [1, 0]);  permute_408 = None
    view_595: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_76, [8, 1, 196, 512]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_527: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_595, primals_66);  primals_66 = None
    mul_528: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_527, 512)
    sum_149: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_527, [3], True)
    mul_529: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_527, mul_161);  mul_527 = None
    sum_150: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_529, [3], True);  mul_529 = None
    mul_530: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_161, sum_150);  sum_150 = None
    sub_142: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_528, sum_149);  mul_528 = sum_149 = None
    sub_143: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_142, mul_530);  sub_142 = mul_530 = None
    mul_531: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_90, sub_143);  div_90 = sub_143 = None
    mul_532: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_595, mul_161);  mul_161 = None
    sum_151: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_532, [0, 1, 2]);  mul_532 = None
    sum_152: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_595, [0, 1, 2]);  view_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_215: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_212, mul_531);  add_212 = mul_531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_533: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_215, div_41);  div_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_596: "f32[1568, 512]" = torch.ops.aten.view.default(mul_533, [1568, 512]);  mul_533 = None
    mm_78: "f32[1568, 512]" = torch.ops.aten.mm.default(view_596, permute_410);  permute_410 = None
    permute_411: "f32[512, 1568]" = torch.ops.aten.permute.default(view_596, [1, 0])
    mm_79: "f32[512, 512]" = torch.ops.aten.mm.default(permute_411, view_244);  permute_411 = view_244 = None
    permute_412: "f32[512, 512]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_153: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_596, [0], True);  view_596 = None
    view_597: "f32[512]" = torch.ops.aten.view.default(sum_153, [512]);  sum_153 = None
    permute_413: "f32[512, 512]" = torch.ops.aten.permute.default(permute_412, [1, 0]);  permute_412 = None
    view_598: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_78, [8, 1, 196, 512]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_599: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_598, [8, 1, 196, 32, 16]);  view_598 = None
    permute_414: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_599, [0, 4, 1, 2, 3]);  view_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_194: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_414, memory_format = torch.contiguous_format);  permute_414 = None
    view_600: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_194, [128, 196, 32]);  clone_194 = None
    bmm_84: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_415, view_600);  permute_415 = None
    bmm_85: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_600, permute_416);  view_600 = permute_416 = None
    view_601: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_84, [8, 16, 1, 196, 32]);  bmm_84 = None
    view_602: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_85, [8, 16, 1, 196, 196]);  bmm_85 = None
    mul_534: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_602, alias_33);  view_602 = None
    sum_154: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_534, [-1], True)
    mul_535: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_33, sum_154);  alias_33 = sum_154 = None
    sub_144: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_534, mul_535);  mul_534 = mul_535 = None
    view_603: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_144, [128, 196, 196]);  sub_144 = None
    bmm_86: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_417, view_603);  permute_417 = None
    bmm_87: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_603, permute_418);  view_603 = permute_418 = None
    view_604: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_86, [8, 16, 1, 32, 196]);  bmm_86 = None
    view_605: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_87, [8, 16, 1, 196, 32]);  bmm_87 = None
    mul_536: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_604, 0.42044820762685725);  view_604 = None
    permute_419: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_536, [0, 1, 2, 4, 3]);  mul_536 = None
    mul_537: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_605, 0.42044820762685725);  view_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_9: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_537, permute_419, view_601]);  mul_537 = permute_419 = view_601 = None
    view_606: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_9, [3, 8, 16, 1, 196, 32]);  cat_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_420: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_606, [1, 3, 4, 0, 2, 5]);  view_606 = None
    clone_195: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_420, memory_format = torch.contiguous_format);  permute_420 = None
    view_607: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_195, [8, 1, 196, 1536]);  clone_195 = None
    view_608: "f32[1568, 1536]" = torch.ops.aten.view.default(view_607, [1568, 1536]);  view_607 = None
    mm_80: "f32[1568, 512]" = torch.ops.aten.mm.default(view_608, permute_421);  permute_421 = None
    permute_422: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_608, [1, 0])
    mm_81: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_422, view_234);  permute_422 = view_234 = None
    permute_423: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_155: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_608, [0], True);  view_608 = None
    view_609: "f32[1536]" = torch.ops.aten.view.default(sum_155, [1536]);  sum_155 = None
    permute_424: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_423, [1, 0]);  permute_423 = None
    view_610: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_80, [8, 1, 196, 512]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_539: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_610, primals_64);  primals_64 = None
    mul_540: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_539, 512)
    sum_156: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_539, [3], True)
    mul_541: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_539, mul_156);  mul_539 = None
    sum_157: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_541, [3], True);  mul_541 = None
    mul_542: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_156, sum_157);  sum_157 = None
    sub_146: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_540, sum_156);  mul_540 = sum_156 = None
    sub_147: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_146, mul_542);  sub_146 = mul_542 = None
    mul_543: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_91, sub_147);  div_91 = sub_147 = None
    mul_544: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_610, mul_156);  mul_156 = None
    sum_158: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_544, [0, 1, 2]);  mul_544 = None
    sum_159: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_610, [0, 1, 2]);  view_610 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_216: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_215, mul_543);  add_215 = mul_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_545: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_216, div_39);  div_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_611: "f32[1568, 512]" = torch.ops.aten.view.default(mul_545, [1568, 512]);  mul_545 = None
    mm_82: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_611, permute_425);  permute_425 = None
    permute_426: "f32[512, 1568]" = torch.ops.aten.permute.default(view_611, [1, 0])
    mm_83: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_426, view_232);  permute_426 = view_232 = None
    permute_427: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_160: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_611, [0], True);  view_611 = None
    view_612: "f32[512]" = torch.ops.aten.view.default(sum_160, [512]);  sum_160 = None
    permute_428: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_427, [1, 0]);  permute_427 = None
    view_613: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_82, [8, 1, 196, 2048]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_547: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_103, 0.5);  add_103 = None
    mul_548: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_231, view_231)
    mul_549: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_548, -0.5);  mul_548 = None
    exp_34: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_549);  mul_549 = None
    mul_550: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_34, 0.3989422804014327);  exp_34 = None
    mul_551: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_231, mul_550);  view_231 = mul_550 = None
    add_218: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_547, mul_551);  mul_547 = mul_551 = None
    mul_552: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_613, add_218);  view_613 = add_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_614: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_552, [1568, 2048]);  mul_552 = None
    mm_84: "f32[1568, 512]" = torch.ops.aten.mm.default(view_614, permute_429);  permute_429 = None
    permute_430: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_614, [1, 0])
    mm_85: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_430, view_230);  permute_430 = view_230 = None
    permute_431: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_161: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_614, [0], True);  view_614 = None
    view_615: "f32[2048]" = torch.ops.aten.view.default(sum_161, [2048]);  sum_161 = None
    permute_432: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_431, [1, 0]);  permute_431 = None
    view_616: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_84, [8, 1, 196, 512]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_554: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_616, primals_62);  primals_62 = None
    mul_555: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_554, 512)
    sum_162: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_554, [3], True)
    mul_556: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_554, mul_150);  mul_554 = None
    sum_163: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_556, [3], True);  mul_556 = None
    mul_557: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_150, sum_163);  sum_163 = None
    sub_149: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_555, sum_162);  mul_555 = sum_162 = None
    sub_150: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_149, mul_557);  sub_149 = mul_557 = None
    mul_558: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_92, sub_150);  div_92 = sub_150 = None
    mul_559: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_616, mul_150);  mul_150 = None
    sum_164: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_559, [0, 1, 2]);  mul_559 = None
    sum_165: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_616, [0, 1, 2]);  view_616 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_219: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_216, mul_558);  add_216 = mul_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_560: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_219, div_38);  div_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_617: "f32[1568, 512]" = torch.ops.aten.view.default(mul_560, [1568, 512]);  mul_560 = None
    mm_86: "f32[1568, 512]" = torch.ops.aten.mm.default(view_617, permute_433);  permute_433 = None
    permute_434: "f32[512, 1568]" = torch.ops.aten.permute.default(view_617, [1, 0])
    mm_87: "f32[512, 512]" = torch.ops.aten.mm.default(permute_434, view_228);  permute_434 = view_228 = None
    permute_435: "f32[512, 512]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_166: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_617, [0], True);  view_617 = None
    view_618: "f32[512]" = torch.ops.aten.view.default(sum_166, [512]);  sum_166 = None
    permute_436: "f32[512, 512]" = torch.ops.aten.permute.default(permute_435, [1, 0]);  permute_435 = None
    view_619: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_86, [8, 1, 196, 512]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_620: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_619, [8, 1, 196, 32, 16]);  view_619 = None
    permute_437: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_620, [0, 4, 1, 2, 3]);  view_620 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_196: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_437, memory_format = torch.contiguous_format);  permute_437 = None
    view_621: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_196, [128, 196, 32]);  clone_196 = None
    bmm_88: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_438, view_621);  permute_438 = None
    bmm_89: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_621, permute_439);  view_621 = permute_439 = None
    view_622: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_88, [8, 16, 1, 196, 32]);  bmm_88 = None
    view_623: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_89, [8, 16, 1, 196, 196]);  bmm_89 = None
    mul_561: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_623, alias_34);  view_623 = None
    sum_167: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_561, [-1], True)
    mul_562: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_34, sum_167);  alias_34 = sum_167 = None
    sub_151: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_561, mul_562);  mul_561 = mul_562 = None
    view_624: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_151, [128, 196, 196]);  sub_151 = None
    bmm_90: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_440, view_624);  permute_440 = None
    bmm_91: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_624, permute_441);  view_624 = permute_441 = None
    view_625: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_90, [8, 16, 1, 32, 196]);  bmm_90 = None
    view_626: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_91, [8, 16, 1, 196, 32]);  bmm_91 = None
    mul_563: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_625, 0.42044820762685725);  view_625 = None
    permute_442: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_563, [0, 1, 2, 4, 3]);  mul_563 = None
    mul_564: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_626, 0.42044820762685725);  view_626 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_10: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_564, permute_442, view_622]);  mul_564 = permute_442 = view_622 = None
    view_627: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_10, [3, 8, 16, 1, 196, 32]);  cat_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_443: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_627, [1, 3, 4, 0, 2, 5]);  view_627 = None
    clone_197: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_443, memory_format = torch.contiguous_format);  permute_443 = None
    view_628: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_197, [8, 1, 196, 1536]);  clone_197 = None
    view_629: "f32[1568, 1536]" = torch.ops.aten.view.default(view_628, [1568, 1536]);  view_628 = None
    mm_88: "f32[1568, 512]" = torch.ops.aten.mm.default(view_629, permute_444);  permute_444 = None
    permute_445: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_629, [1, 0])
    mm_89: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_445, view_218);  permute_445 = view_218 = None
    permute_446: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_168: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_629, [0], True);  view_629 = None
    view_630: "f32[1536]" = torch.ops.aten.view.default(sum_168, [1536]);  sum_168 = None
    permute_447: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_446, [1, 0]);  permute_446 = None
    view_631: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_88, [8, 1, 196, 512]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_566: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_631, primals_60);  primals_60 = None
    mul_567: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_566, 512)
    sum_169: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_566, [3], True)
    mul_568: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_566, mul_145);  mul_566 = None
    sum_170: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_568, [3], True);  mul_568 = None
    mul_569: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_145, sum_170);  sum_170 = None
    sub_153: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_567, sum_169);  mul_567 = sum_169 = None
    sub_154: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_153, mul_569);  sub_153 = mul_569 = None
    mul_570: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_93, sub_154);  div_93 = sub_154 = None
    mul_571: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_631, mul_145);  mul_145 = None
    sum_171: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_571, [0, 1, 2]);  mul_571 = None
    sum_172: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_631, [0, 1, 2]);  view_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_220: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_219, mul_570);  add_219 = mul_570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_572: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_220, div_36);  div_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_632: "f32[1568, 512]" = torch.ops.aten.view.default(mul_572, [1568, 512]);  mul_572 = None
    mm_90: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_632, permute_448);  permute_448 = None
    permute_449: "f32[512, 1568]" = torch.ops.aten.permute.default(view_632, [1, 0])
    mm_91: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_449, view_216);  permute_449 = view_216 = None
    permute_450: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_173: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_632, [0], True);  view_632 = None
    view_633: "f32[512]" = torch.ops.aten.view.default(sum_173, [512]);  sum_173 = None
    permute_451: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_450, [1, 0]);  permute_450 = None
    view_634: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_90, [8, 1, 196, 2048]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_574: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_96, 0.5);  add_96 = None
    mul_575: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_215, view_215)
    mul_576: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_575, -0.5);  mul_575 = None
    exp_35: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_576);  mul_576 = None
    mul_577: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_35, 0.3989422804014327);  exp_35 = None
    mul_578: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_215, mul_577);  view_215 = mul_577 = None
    add_222: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_574, mul_578);  mul_574 = mul_578 = None
    mul_579: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_634, add_222);  view_634 = add_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_635: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_579, [1568, 2048]);  mul_579 = None
    mm_92: "f32[1568, 512]" = torch.ops.aten.mm.default(view_635, permute_452);  permute_452 = None
    permute_453: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_635, [1, 0])
    mm_93: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_453, view_214);  permute_453 = view_214 = None
    permute_454: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_174: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_635, [0], True);  view_635 = None
    view_636: "f32[2048]" = torch.ops.aten.view.default(sum_174, [2048]);  sum_174 = None
    permute_455: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_454, [1, 0]);  permute_454 = None
    view_637: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_92, [8, 1, 196, 512]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_581: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_637, primals_58);  primals_58 = None
    mul_582: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_581, 512)
    sum_175: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_581, [3], True)
    mul_583: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_581, mul_139);  mul_581 = None
    sum_176: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_583, [3], True);  mul_583 = None
    mul_584: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_139, sum_176);  sum_176 = None
    sub_156: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_582, sum_175);  mul_582 = sum_175 = None
    sub_157: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_156, mul_584);  sub_156 = mul_584 = None
    mul_585: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_94, sub_157);  div_94 = sub_157 = None
    mul_586: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_637, mul_139);  mul_139 = None
    sum_177: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_586, [0, 1, 2]);  mul_586 = None
    sum_178: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_637, [0, 1, 2]);  view_637 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_223: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_220, mul_585);  add_220 = mul_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_587: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_223, div_35);  div_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_638: "f32[1568, 512]" = torch.ops.aten.view.default(mul_587, [1568, 512]);  mul_587 = None
    mm_94: "f32[1568, 512]" = torch.ops.aten.mm.default(view_638, permute_456);  permute_456 = None
    permute_457: "f32[512, 1568]" = torch.ops.aten.permute.default(view_638, [1, 0])
    mm_95: "f32[512, 512]" = torch.ops.aten.mm.default(permute_457, view_212);  permute_457 = view_212 = None
    permute_458: "f32[512, 512]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_179: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_638, [0], True);  view_638 = None
    view_639: "f32[512]" = torch.ops.aten.view.default(sum_179, [512]);  sum_179 = None
    permute_459: "f32[512, 512]" = torch.ops.aten.permute.default(permute_458, [1, 0]);  permute_458 = None
    view_640: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_94, [8, 1, 196, 512]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_641: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_640, [8, 1, 196, 32, 16]);  view_640 = None
    permute_460: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_641, [0, 4, 1, 2, 3]);  view_641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_198: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_460, memory_format = torch.contiguous_format);  permute_460 = None
    view_642: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_198, [128, 196, 32]);  clone_198 = None
    bmm_92: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_461, view_642);  permute_461 = None
    bmm_93: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_642, permute_462);  view_642 = permute_462 = None
    view_643: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_92, [8, 16, 1, 196, 32]);  bmm_92 = None
    view_644: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_93, [8, 16, 1, 196, 196]);  bmm_93 = None
    mul_588: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_644, alias_35);  view_644 = None
    sum_180: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_588, [-1], True)
    mul_589: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_35, sum_180);  alias_35 = sum_180 = None
    sub_158: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_588, mul_589);  mul_588 = mul_589 = None
    view_645: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_158, [128, 196, 196]);  sub_158 = None
    bmm_94: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_463, view_645);  permute_463 = None
    bmm_95: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_645, permute_464);  view_645 = permute_464 = None
    view_646: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_94, [8, 16, 1, 32, 196]);  bmm_94 = None
    view_647: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_95, [8, 16, 1, 196, 32]);  bmm_95 = None
    mul_590: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_646, 0.42044820762685725);  view_646 = None
    permute_465: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_590, [0, 1, 2, 4, 3]);  mul_590 = None
    mul_591: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_647, 0.42044820762685725);  view_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_11: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_591, permute_465, view_643]);  mul_591 = permute_465 = view_643 = None
    view_648: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_11, [3, 8, 16, 1, 196, 32]);  cat_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_466: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_648, [1, 3, 4, 0, 2, 5]);  view_648 = None
    clone_199: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_466, memory_format = torch.contiguous_format);  permute_466 = None
    view_649: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_199, [8, 1, 196, 1536]);  clone_199 = None
    view_650: "f32[1568, 1536]" = torch.ops.aten.view.default(view_649, [1568, 1536]);  view_649 = None
    mm_96: "f32[1568, 512]" = torch.ops.aten.mm.default(view_650, permute_467);  permute_467 = None
    permute_468: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_650, [1, 0])
    mm_97: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_468, view_202);  permute_468 = view_202 = None
    permute_469: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_181: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_650, [0], True);  view_650 = None
    view_651: "f32[1536]" = torch.ops.aten.view.default(sum_181, [1536]);  sum_181 = None
    permute_470: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_469, [1, 0]);  permute_469 = None
    view_652: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_96, [8, 1, 196, 512]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_593: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_652, primals_56);  primals_56 = None
    mul_594: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_593, 512)
    sum_182: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_593, [3], True)
    mul_595: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_593, mul_134);  mul_593 = None
    sum_183: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_595, [3], True);  mul_595 = None
    mul_596: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_134, sum_183);  sum_183 = None
    sub_160: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_594, sum_182);  mul_594 = sum_182 = None
    sub_161: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_160, mul_596);  sub_160 = mul_596 = None
    mul_597: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_95, sub_161);  div_95 = sub_161 = None
    mul_598: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_652, mul_134);  mul_134 = None
    sum_184: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_598, [0, 1, 2]);  mul_598 = None
    sum_185: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_652, [0, 1, 2]);  view_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_224: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_223, mul_597);  add_223 = mul_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_599: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_224, div_33);  div_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_653: "f32[1568, 512]" = torch.ops.aten.view.default(mul_599, [1568, 512]);  mul_599 = None
    mm_98: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_653, permute_471);  permute_471 = None
    permute_472: "f32[512, 1568]" = torch.ops.aten.permute.default(view_653, [1, 0])
    mm_99: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_472, view_200);  permute_472 = view_200 = None
    permute_473: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_186: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_653, [0], True);  view_653 = None
    view_654: "f32[512]" = torch.ops.aten.view.default(sum_186, [512]);  sum_186 = None
    permute_474: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_473, [1, 0]);  permute_473 = None
    view_655: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_98, [8, 1, 196, 2048]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_601: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_89, 0.5);  add_89 = None
    mul_602: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_199, view_199)
    mul_603: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_602, -0.5);  mul_602 = None
    exp_36: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_603);  mul_603 = None
    mul_604: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_36, 0.3989422804014327);  exp_36 = None
    mul_605: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_199, mul_604);  view_199 = mul_604 = None
    add_226: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_601, mul_605);  mul_601 = mul_605 = None
    mul_606: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_655, add_226);  view_655 = add_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_656: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_606, [1568, 2048]);  mul_606 = None
    mm_100: "f32[1568, 512]" = torch.ops.aten.mm.default(view_656, permute_475);  permute_475 = None
    permute_476: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_656, [1, 0])
    mm_101: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_476, view_198);  permute_476 = view_198 = None
    permute_477: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_187: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_656, [0], True);  view_656 = None
    view_657: "f32[2048]" = torch.ops.aten.view.default(sum_187, [2048]);  sum_187 = None
    permute_478: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_477, [1, 0]);  permute_477 = None
    view_658: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_100, [8, 1, 196, 512]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_608: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_658, primals_54);  primals_54 = None
    mul_609: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_608, 512)
    sum_188: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_608, [3], True)
    mul_610: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_608, mul_128);  mul_608 = None
    sum_189: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_610, [3], True);  mul_610 = None
    mul_611: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_128, sum_189);  sum_189 = None
    sub_163: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_609, sum_188);  mul_609 = sum_188 = None
    sub_164: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_163, mul_611);  sub_163 = mul_611 = None
    mul_612: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_96, sub_164);  div_96 = sub_164 = None
    mul_613: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_658, mul_128);  mul_128 = None
    sum_190: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_613, [0, 1, 2]);  mul_613 = None
    sum_191: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_658, [0, 1, 2]);  view_658 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_227: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_224, mul_612);  add_224 = mul_612 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_614: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_227, div_32);  div_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_659: "f32[1568, 512]" = torch.ops.aten.view.default(mul_614, [1568, 512]);  mul_614 = None
    mm_102: "f32[1568, 512]" = torch.ops.aten.mm.default(view_659, permute_479);  permute_479 = None
    permute_480: "f32[512, 1568]" = torch.ops.aten.permute.default(view_659, [1, 0])
    mm_103: "f32[512, 512]" = torch.ops.aten.mm.default(permute_480, view_196);  permute_480 = view_196 = None
    permute_481: "f32[512, 512]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_192: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_659, [0], True);  view_659 = None
    view_660: "f32[512]" = torch.ops.aten.view.default(sum_192, [512]);  sum_192 = None
    permute_482: "f32[512, 512]" = torch.ops.aten.permute.default(permute_481, [1, 0]);  permute_481 = None
    view_661: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_102, [8, 1, 196, 512]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_662: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_661, [8, 1, 196, 32, 16]);  view_661 = None
    permute_483: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_662, [0, 4, 1, 2, 3]);  view_662 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_200: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_483, memory_format = torch.contiguous_format);  permute_483 = None
    view_663: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_200, [128, 196, 32]);  clone_200 = None
    bmm_96: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_484, view_663);  permute_484 = None
    bmm_97: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_663, permute_485);  view_663 = permute_485 = None
    view_664: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_96, [8, 16, 1, 196, 32]);  bmm_96 = None
    view_665: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_97, [8, 16, 1, 196, 196]);  bmm_97 = None
    mul_615: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_665, alias_36);  view_665 = None
    sum_193: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_615, [-1], True)
    mul_616: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_36, sum_193);  alias_36 = sum_193 = None
    sub_165: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_615, mul_616);  mul_615 = mul_616 = None
    view_666: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_165, [128, 196, 196]);  sub_165 = None
    bmm_98: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_486, view_666);  permute_486 = None
    bmm_99: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_666, permute_487);  view_666 = permute_487 = None
    view_667: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_98, [8, 16, 1, 32, 196]);  bmm_98 = None
    view_668: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_99, [8, 16, 1, 196, 32]);  bmm_99 = None
    mul_617: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_667, 0.42044820762685725);  view_667 = None
    permute_488: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_617, [0, 1, 2, 4, 3]);  mul_617 = None
    mul_618: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_668, 0.42044820762685725);  view_668 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_12: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_618, permute_488, view_664]);  mul_618 = permute_488 = view_664 = None
    view_669: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_12, [3, 8, 16, 1, 196, 32]);  cat_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_489: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_669, [1, 3, 4, 0, 2, 5]);  view_669 = None
    clone_201: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_489, memory_format = torch.contiguous_format);  permute_489 = None
    view_670: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_201, [8, 1, 196, 1536]);  clone_201 = None
    view_671: "f32[1568, 1536]" = torch.ops.aten.view.default(view_670, [1568, 1536]);  view_670 = None
    mm_104: "f32[1568, 512]" = torch.ops.aten.mm.default(view_671, permute_490);  permute_490 = None
    permute_491: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_671, [1, 0])
    mm_105: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_491, view_186);  permute_491 = view_186 = None
    permute_492: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_194: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_671, [0], True);  view_671 = None
    view_672: "f32[1536]" = torch.ops.aten.view.default(sum_194, [1536]);  sum_194 = None
    permute_493: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_492, [1, 0]);  permute_492 = None
    view_673: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_104, [8, 1, 196, 512]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_620: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_673, primals_52);  primals_52 = None
    mul_621: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_620, 512)
    sum_195: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_620, [3], True)
    mul_622: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_620, mul_123);  mul_620 = None
    sum_196: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_622, [3], True);  mul_622 = None
    mul_623: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_123, sum_196);  sum_196 = None
    sub_167: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_621, sum_195);  mul_621 = sum_195 = None
    sub_168: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_167, mul_623);  sub_167 = mul_623 = None
    mul_624: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_97, sub_168);  div_97 = sub_168 = None
    mul_625: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_673, mul_123);  mul_123 = None
    sum_197: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_625, [0, 1, 2]);  mul_625 = None
    sum_198: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_673, [0, 1, 2]);  view_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_228: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_227, mul_624);  add_227 = mul_624 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_626: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_228, div_30);  div_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_674: "f32[1568, 512]" = torch.ops.aten.view.default(mul_626, [1568, 512]);  mul_626 = None
    mm_106: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_674, permute_494);  permute_494 = None
    permute_495: "f32[512, 1568]" = torch.ops.aten.permute.default(view_674, [1, 0])
    mm_107: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_495, view_184);  permute_495 = view_184 = None
    permute_496: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_199: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_674, [0], True);  view_674 = None
    view_675: "f32[512]" = torch.ops.aten.view.default(sum_199, [512]);  sum_199 = None
    permute_497: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_496, [1, 0]);  permute_496 = None
    view_676: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_106, [8, 1, 196, 2048]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_628: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_82, 0.5);  add_82 = None
    mul_629: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_183, view_183)
    mul_630: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_629, -0.5);  mul_629 = None
    exp_37: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_630);  mul_630 = None
    mul_631: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_37, 0.3989422804014327);  exp_37 = None
    mul_632: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_183, mul_631);  view_183 = mul_631 = None
    add_230: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_628, mul_632);  mul_628 = mul_632 = None
    mul_633: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_676, add_230);  view_676 = add_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_677: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_633, [1568, 2048]);  mul_633 = None
    mm_108: "f32[1568, 512]" = torch.ops.aten.mm.default(view_677, permute_498);  permute_498 = None
    permute_499: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_677, [1, 0])
    mm_109: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_499, view_182);  permute_499 = view_182 = None
    permute_500: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_200: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_677, [0], True);  view_677 = None
    view_678: "f32[2048]" = torch.ops.aten.view.default(sum_200, [2048]);  sum_200 = None
    permute_501: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_500, [1, 0]);  permute_500 = None
    view_679: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_108, [8, 1, 196, 512]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_635: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_679, primals_50);  primals_50 = None
    mul_636: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_635, 512)
    sum_201: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_635, [3], True)
    mul_637: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_635, mul_117);  mul_635 = None
    sum_202: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_637, [3], True);  mul_637 = None
    mul_638: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_117, sum_202);  sum_202 = None
    sub_170: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_636, sum_201);  mul_636 = sum_201 = None
    sub_171: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_170, mul_638);  sub_170 = mul_638 = None
    mul_639: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_98, sub_171);  div_98 = sub_171 = None
    mul_640: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_679, mul_117);  mul_117 = None
    sum_203: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_640, [0, 1, 2]);  mul_640 = None
    sum_204: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_679, [0, 1, 2]);  view_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_231: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_228, mul_639);  add_228 = mul_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_641: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_231, div_29);  div_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_680: "f32[1568, 512]" = torch.ops.aten.view.default(mul_641, [1568, 512]);  mul_641 = None
    mm_110: "f32[1568, 512]" = torch.ops.aten.mm.default(view_680, permute_502);  permute_502 = None
    permute_503: "f32[512, 1568]" = torch.ops.aten.permute.default(view_680, [1, 0])
    mm_111: "f32[512, 512]" = torch.ops.aten.mm.default(permute_503, view_180);  permute_503 = view_180 = None
    permute_504: "f32[512, 512]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_205: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_680, [0], True);  view_680 = None
    view_681: "f32[512]" = torch.ops.aten.view.default(sum_205, [512]);  sum_205 = None
    permute_505: "f32[512, 512]" = torch.ops.aten.permute.default(permute_504, [1, 0]);  permute_504 = None
    view_682: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_110, [8, 1, 196, 512]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_683: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_682, [8, 1, 196, 32, 16]);  view_682 = None
    permute_506: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_683, [0, 4, 1, 2, 3]);  view_683 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_202: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_506, memory_format = torch.contiguous_format);  permute_506 = None
    view_684: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_202, [128, 196, 32]);  clone_202 = None
    bmm_100: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_507, view_684);  permute_507 = None
    bmm_101: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_684, permute_508);  view_684 = permute_508 = None
    view_685: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_100, [8, 16, 1, 196, 32]);  bmm_100 = None
    view_686: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_101, [8, 16, 1, 196, 196]);  bmm_101 = None
    mul_642: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_686, alias_37);  view_686 = None
    sum_206: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_642, [-1], True)
    mul_643: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_37, sum_206);  alias_37 = sum_206 = None
    sub_172: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_642, mul_643);  mul_642 = mul_643 = None
    view_687: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_172, [128, 196, 196]);  sub_172 = None
    bmm_102: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_509, view_687);  permute_509 = None
    bmm_103: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_687, permute_510);  view_687 = permute_510 = None
    view_688: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_102, [8, 16, 1, 32, 196]);  bmm_102 = None
    view_689: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_103, [8, 16, 1, 196, 32]);  bmm_103 = None
    mul_644: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_688, 0.42044820762685725);  view_688 = None
    permute_511: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_644, [0, 1, 2, 4, 3]);  mul_644 = None
    mul_645: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_689, 0.42044820762685725);  view_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_13: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_645, permute_511, view_685]);  mul_645 = permute_511 = view_685 = None
    view_690: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_13, [3, 8, 16, 1, 196, 32]);  cat_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_512: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_690, [1, 3, 4, 0, 2, 5]);  view_690 = None
    clone_203: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_512, memory_format = torch.contiguous_format);  permute_512 = None
    view_691: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_203, [8, 1, 196, 1536]);  clone_203 = None
    view_692: "f32[1568, 1536]" = torch.ops.aten.view.default(view_691, [1568, 1536]);  view_691 = None
    mm_112: "f32[1568, 512]" = torch.ops.aten.mm.default(view_692, permute_513);  permute_513 = None
    permute_514: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_692, [1, 0])
    mm_113: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_514, view_170);  permute_514 = view_170 = None
    permute_515: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    sum_207: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_692, [0], True);  view_692 = None
    view_693: "f32[1536]" = torch.ops.aten.view.default(sum_207, [1536]);  sum_207 = None
    permute_516: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_515, [1, 0]);  permute_515 = None
    view_694: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_112, [8, 1, 196, 512]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_647: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_694, primals_48);  primals_48 = None
    mul_648: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_647, 512)
    sum_208: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_647, [3], True)
    mul_649: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_647, mul_112);  mul_647 = None
    sum_209: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_649, [3], True);  mul_649 = None
    mul_650: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_112, sum_209);  sum_209 = None
    sub_174: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_648, sum_208);  mul_648 = sum_208 = None
    sub_175: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_174, mul_650);  sub_174 = mul_650 = None
    mul_651: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_99, sub_175);  div_99 = sub_175 = None
    mul_652: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_694, mul_112);  mul_112 = None
    sum_210: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_652, [0, 1, 2]);  mul_652 = None
    sum_211: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_694, [0, 1, 2]);  view_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_232: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_231, mul_651);  add_231 = mul_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_653: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_232, div_27);  div_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_695: "f32[1568, 512]" = torch.ops.aten.view.default(mul_653, [1568, 512]);  mul_653 = None
    mm_114: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_695, permute_517);  permute_517 = None
    permute_518: "f32[512, 1568]" = torch.ops.aten.permute.default(view_695, [1, 0])
    mm_115: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_518, view_168);  permute_518 = view_168 = None
    permute_519: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_212: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_695, [0], True);  view_695 = None
    view_696: "f32[512]" = torch.ops.aten.view.default(sum_212, [512]);  sum_212 = None
    permute_520: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_519, [1, 0]);  permute_519 = None
    view_697: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_114, [8, 1, 196, 2048]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_655: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_75, 0.5);  add_75 = None
    mul_656: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_167, view_167)
    mul_657: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_656, -0.5);  mul_656 = None
    exp_38: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_657);  mul_657 = None
    mul_658: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_38, 0.3989422804014327);  exp_38 = None
    mul_659: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_167, mul_658);  view_167 = mul_658 = None
    add_234: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_655, mul_659);  mul_655 = mul_659 = None
    mul_660: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_697, add_234);  view_697 = add_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_698: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_660, [1568, 2048]);  mul_660 = None
    mm_116: "f32[1568, 512]" = torch.ops.aten.mm.default(view_698, permute_521);  permute_521 = None
    permute_522: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_698, [1, 0])
    mm_117: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_522, view_166);  permute_522 = view_166 = None
    permute_523: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_213: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_698, [0], True);  view_698 = None
    view_699: "f32[2048]" = torch.ops.aten.view.default(sum_213, [2048]);  sum_213 = None
    permute_524: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_523, [1, 0]);  permute_523 = None
    view_700: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_116, [8, 1, 196, 512]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_662: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_700, primals_46);  primals_46 = None
    mul_663: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_662, 512)
    sum_214: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_662, [3], True)
    mul_664: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_662, mul_106);  mul_662 = None
    sum_215: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_664, [3], True);  mul_664 = None
    mul_665: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_106, sum_215);  sum_215 = None
    sub_177: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_663, sum_214);  mul_663 = sum_214 = None
    sub_178: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_177, mul_665);  sub_177 = mul_665 = None
    mul_666: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_100, sub_178);  div_100 = sub_178 = None
    mul_667: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_700, mul_106);  mul_106 = None
    sum_216: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_667, [0, 1, 2]);  mul_667 = None
    sum_217: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_700, [0, 1, 2]);  view_700 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_235: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_232, mul_666);  add_232 = mul_666 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_668: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_235, div_26);  div_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_701: "f32[1568, 512]" = torch.ops.aten.view.default(mul_668, [1568, 512]);  mul_668 = None
    mm_118: "f32[1568, 512]" = torch.ops.aten.mm.default(view_701, permute_525);  permute_525 = None
    permute_526: "f32[512, 1568]" = torch.ops.aten.permute.default(view_701, [1, 0])
    mm_119: "f32[512, 512]" = torch.ops.aten.mm.default(permute_526, view_164);  permute_526 = view_164 = None
    permute_527: "f32[512, 512]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_218: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_701, [0], True);  view_701 = None
    view_702: "f32[512]" = torch.ops.aten.view.default(sum_218, [512]);  sum_218 = None
    permute_528: "f32[512, 512]" = torch.ops.aten.permute.default(permute_527, [1, 0]);  permute_527 = None
    view_703: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_118, [8, 1, 196, 512]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_704: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_703, [8, 1, 196, 32, 16]);  view_703 = None
    permute_529: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_704, [0, 4, 1, 2, 3]);  view_704 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_204: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_529, memory_format = torch.contiguous_format);  permute_529 = None
    view_705: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_204, [128, 196, 32]);  clone_204 = None
    bmm_104: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_530, view_705);  permute_530 = None
    bmm_105: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_705, permute_531);  view_705 = permute_531 = None
    view_706: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_104, [8, 16, 1, 196, 32]);  bmm_104 = None
    view_707: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_105, [8, 16, 1, 196, 196]);  bmm_105 = None
    mul_669: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_707, alias_38);  view_707 = None
    sum_219: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_669, [-1], True)
    mul_670: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_38, sum_219);  alias_38 = sum_219 = None
    sub_179: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_669, mul_670);  mul_669 = mul_670 = None
    view_708: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_179, [128, 196, 196]);  sub_179 = None
    bmm_106: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_532, view_708);  permute_532 = None
    bmm_107: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_708, permute_533);  view_708 = permute_533 = None
    view_709: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_106, [8, 16, 1, 32, 196]);  bmm_106 = None
    view_710: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_107, [8, 16, 1, 196, 32]);  bmm_107 = None
    mul_671: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_709, 0.42044820762685725);  view_709 = None
    permute_534: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_671, [0, 1, 2, 4, 3]);  mul_671 = None
    mul_672: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_710, 0.42044820762685725);  view_710 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_14: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_672, permute_534, view_706]);  mul_672 = permute_534 = view_706 = None
    view_711: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_14, [3, 8, 16, 1, 196, 32]);  cat_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_535: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_711, [1, 3, 4, 0, 2, 5]);  view_711 = None
    clone_205: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_535, memory_format = torch.contiguous_format);  permute_535 = None
    view_712: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_205, [8, 1, 196, 1536]);  clone_205 = None
    view_713: "f32[1568, 1536]" = torch.ops.aten.view.default(view_712, [1568, 1536]);  view_712 = None
    mm_120: "f32[1568, 512]" = torch.ops.aten.mm.default(view_713, permute_536);  permute_536 = None
    permute_537: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_713, [1, 0])
    mm_121: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_537, view_154);  permute_537 = view_154 = None
    permute_538: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    sum_220: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_713, [0], True);  view_713 = None
    view_714: "f32[1536]" = torch.ops.aten.view.default(sum_220, [1536]);  sum_220 = None
    permute_539: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_538, [1, 0]);  permute_538 = None
    view_715: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_120, [8, 1, 196, 512]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_674: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_715, primals_44);  primals_44 = None
    mul_675: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_674, 512)
    sum_221: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_674, [3], True)
    mul_676: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_674, mul_101);  mul_674 = None
    sum_222: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_676, [3], True);  mul_676 = None
    mul_677: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_101, sum_222);  sum_222 = None
    sub_181: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_675, sum_221);  mul_675 = sum_221 = None
    sub_182: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_181, mul_677);  sub_181 = mul_677 = None
    mul_678: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_101, sub_182);  div_101 = sub_182 = None
    mul_679: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_715, mul_101);  mul_101 = None
    sum_223: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_679, [0, 1, 2]);  mul_679 = None
    sum_224: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_715, [0, 1, 2]);  view_715 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_236: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_235, mul_678);  add_235 = mul_678 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_680: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_236, div_24);  div_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_716: "f32[1568, 512]" = torch.ops.aten.view.default(mul_680, [1568, 512]);  mul_680 = None
    mm_122: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_716, permute_540);  permute_540 = None
    permute_541: "f32[512, 1568]" = torch.ops.aten.permute.default(view_716, [1, 0])
    mm_123: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_541, view_152);  permute_541 = view_152 = None
    permute_542: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_225: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_716, [0], True);  view_716 = None
    view_717: "f32[512]" = torch.ops.aten.view.default(sum_225, [512]);  sum_225 = None
    permute_543: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_542, [1, 0]);  permute_542 = None
    view_718: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_122, [8, 1, 196, 2048]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_682: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_68, 0.5);  add_68 = None
    mul_683: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_151, view_151)
    mul_684: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_683, -0.5);  mul_683 = None
    exp_39: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_684);  mul_684 = None
    mul_685: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_39, 0.3989422804014327);  exp_39 = None
    mul_686: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_151, mul_685);  view_151 = mul_685 = None
    add_238: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_682, mul_686);  mul_682 = mul_686 = None
    mul_687: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_718, add_238);  view_718 = add_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_719: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_687, [1568, 2048]);  mul_687 = None
    mm_124: "f32[1568, 512]" = torch.ops.aten.mm.default(view_719, permute_544);  permute_544 = None
    permute_545: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_719, [1, 0])
    mm_125: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_545, view_150);  permute_545 = view_150 = None
    permute_546: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    sum_226: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_719, [0], True);  view_719 = None
    view_720: "f32[2048]" = torch.ops.aten.view.default(sum_226, [2048]);  sum_226 = None
    permute_547: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_546, [1, 0]);  permute_546 = None
    view_721: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_124, [8, 1, 196, 512]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_689: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_721, primals_42);  primals_42 = None
    mul_690: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_689, 512)
    sum_227: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_689, [3], True)
    mul_691: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_689, mul_95);  mul_689 = None
    sum_228: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_691, [3], True);  mul_691 = None
    mul_692: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_95, sum_228);  sum_228 = None
    sub_184: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_690, sum_227);  mul_690 = sum_227 = None
    sub_185: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_184, mul_692);  sub_184 = mul_692 = None
    mul_693: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_102, sub_185);  div_102 = sub_185 = None
    mul_694: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_721, mul_95);  mul_95 = None
    sum_229: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_694, [0, 1, 2]);  mul_694 = None
    sum_230: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_721, [0, 1, 2]);  view_721 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_239: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_236, mul_693);  add_236 = mul_693 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_695: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_239, div_23);  div_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_722: "f32[1568, 512]" = torch.ops.aten.view.default(mul_695, [1568, 512]);  mul_695 = None
    mm_126: "f32[1568, 512]" = torch.ops.aten.mm.default(view_722, permute_548);  permute_548 = None
    permute_549: "f32[512, 1568]" = torch.ops.aten.permute.default(view_722, [1, 0])
    mm_127: "f32[512, 512]" = torch.ops.aten.mm.default(permute_549, view_148);  permute_549 = view_148 = None
    permute_550: "f32[512, 512]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_231: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_722, [0], True);  view_722 = None
    view_723: "f32[512]" = torch.ops.aten.view.default(sum_231, [512]);  sum_231 = None
    permute_551: "f32[512, 512]" = torch.ops.aten.permute.default(permute_550, [1, 0]);  permute_550 = None
    view_724: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_126, [8, 1, 196, 512]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_725: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_724, [8, 1, 196, 32, 16]);  view_724 = None
    permute_552: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_725, [0, 4, 1, 2, 3]);  view_725 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_206: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_552, memory_format = torch.contiguous_format);  permute_552 = None
    view_726: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_206, [128, 196, 32]);  clone_206 = None
    bmm_108: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_553, view_726);  permute_553 = None
    bmm_109: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_726, permute_554);  view_726 = permute_554 = None
    view_727: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_108, [8, 16, 1, 196, 32]);  bmm_108 = None
    view_728: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_109, [8, 16, 1, 196, 196]);  bmm_109 = None
    mul_696: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_728, alias_39);  view_728 = None
    sum_232: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_696, [-1], True)
    mul_697: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_39, sum_232);  alias_39 = sum_232 = None
    sub_186: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_696, mul_697);  mul_696 = mul_697 = None
    view_729: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_186, [128, 196, 196]);  sub_186 = None
    bmm_110: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_555, view_729);  permute_555 = None
    bmm_111: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_729, permute_556);  view_729 = permute_556 = None
    view_730: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_110, [8, 16, 1, 32, 196]);  bmm_110 = None
    view_731: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_111, [8, 16, 1, 196, 32]);  bmm_111 = None
    mul_698: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_730, 0.42044820762685725);  view_730 = None
    permute_557: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_698, [0, 1, 2, 4, 3]);  mul_698 = None
    mul_699: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_731, 0.42044820762685725);  view_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_15: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_699, permute_557, view_727]);  mul_699 = permute_557 = view_727 = None
    view_732: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_15, [3, 8, 16, 1, 196, 32]);  cat_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_558: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_732, [1, 3, 4, 0, 2, 5]);  view_732 = None
    clone_207: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_558, memory_format = torch.contiguous_format);  permute_558 = None
    view_733: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_207, [8, 1, 196, 1536]);  clone_207 = None
    view_734: "f32[1568, 1536]" = torch.ops.aten.view.default(view_733, [1568, 1536]);  view_733 = None
    mm_128: "f32[1568, 512]" = torch.ops.aten.mm.default(view_734, permute_559);  permute_559 = None
    permute_560: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_734, [1, 0])
    mm_129: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_560, view_138);  permute_560 = view_138 = None
    permute_561: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_233: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_734, [0], True);  view_734 = None
    view_735: "f32[1536]" = torch.ops.aten.view.default(sum_233, [1536]);  sum_233 = None
    permute_562: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_561, [1, 0]);  permute_561 = None
    view_736: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_128, [8, 1, 196, 512]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_701: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_736, primals_40);  primals_40 = None
    mul_702: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_701, 512)
    sum_234: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_701, [3], True)
    mul_703: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_701, mul_90);  mul_701 = None
    sum_235: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_703, [3], True);  mul_703 = None
    mul_704: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_90, sum_235);  sum_235 = None
    sub_188: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_702, sum_234);  mul_702 = sum_234 = None
    sub_189: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_188, mul_704);  sub_188 = mul_704 = None
    mul_705: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_103, sub_189);  div_103 = sub_189 = None
    mul_706: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_736, mul_90);  mul_90 = None
    sum_236: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_706, [0, 1, 2]);  mul_706 = None
    sum_237: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_736, [0, 1, 2]);  view_736 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_240: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_239, mul_705);  add_239 = mul_705 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_707: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_240, div_21);  div_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_737: "f32[1568, 512]" = torch.ops.aten.view.default(mul_707, [1568, 512]);  mul_707 = None
    mm_130: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_737, permute_563);  permute_563 = None
    permute_564: "f32[512, 1568]" = torch.ops.aten.permute.default(view_737, [1, 0])
    mm_131: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_564, view_136);  permute_564 = view_136 = None
    permute_565: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    sum_238: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_737, [0], True);  view_737 = None
    view_738: "f32[512]" = torch.ops.aten.view.default(sum_238, [512]);  sum_238 = None
    permute_566: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_565, [1, 0]);  permute_565 = None
    view_739: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_130, [8, 1, 196, 2048]);  mm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_709: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_61, 0.5);  add_61 = None
    mul_710: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_135, view_135)
    mul_711: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_710, -0.5);  mul_710 = None
    exp_40: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_711);  mul_711 = None
    mul_712: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_40, 0.3989422804014327);  exp_40 = None
    mul_713: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_135, mul_712);  view_135 = mul_712 = None
    add_242: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_709, mul_713);  mul_709 = mul_713 = None
    mul_714: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_739, add_242);  view_739 = add_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_740: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_714, [1568, 2048]);  mul_714 = None
    mm_132: "f32[1568, 512]" = torch.ops.aten.mm.default(view_740, permute_567);  permute_567 = None
    permute_568: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_740, [1, 0])
    mm_133: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_568, view_134);  permute_568 = view_134 = None
    permute_569: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_239: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_740, [0], True);  view_740 = None
    view_741: "f32[2048]" = torch.ops.aten.view.default(sum_239, [2048]);  sum_239 = None
    permute_570: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_569, [1, 0]);  permute_569 = None
    view_742: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_132, [8, 1, 196, 512]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_716: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_742, primals_38);  primals_38 = None
    mul_717: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_716, 512)
    sum_240: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_716, [3], True)
    mul_718: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_716, mul_84);  mul_716 = None
    sum_241: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_718, [3], True);  mul_718 = None
    mul_719: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_84, sum_241);  sum_241 = None
    sub_191: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_717, sum_240);  mul_717 = sum_240 = None
    sub_192: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_191, mul_719);  sub_191 = mul_719 = None
    mul_720: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_104, sub_192);  div_104 = sub_192 = None
    mul_721: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_742, mul_84);  mul_84 = None
    sum_242: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_721, [0, 1, 2]);  mul_721 = None
    sum_243: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_742, [0, 1, 2]);  view_742 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_243: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_240, mul_720);  add_240 = mul_720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_722: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_243, div_20);  div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_743: "f32[1568, 512]" = torch.ops.aten.view.default(mul_722, [1568, 512]);  mul_722 = None
    mm_134: "f32[1568, 512]" = torch.ops.aten.mm.default(view_743, permute_571);  permute_571 = None
    permute_572: "f32[512, 1568]" = torch.ops.aten.permute.default(view_743, [1, 0])
    mm_135: "f32[512, 512]" = torch.ops.aten.mm.default(permute_572, view_132);  permute_572 = view_132 = None
    permute_573: "f32[512, 512]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_244: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_743, [0], True);  view_743 = None
    view_744: "f32[512]" = torch.ops.aten.view.default(sum_244, [512]);  sum_244 = None
    permute_574: "f32[512, 512]" = torch.ops.aten.permute.default(permute_573, [1, 0]);  permute_573 = None
    view_745: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_134, [8, 1, 196, 512]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_746: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_745, [8, 1, 196, 32, 16]);  view_745 = None
    permute_575: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_746, [0, 4, 1, 2, 3]);  view_746 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_208: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_575, memory_format = torch.contiguous_format);  permute_575 = None
    view_747: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_208, [128, 196, 32]);  clone_208 = None
    bmm_112: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_576, view_747);  permute_576 = None
    bmm_113: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_747, permute_577);  view_747 = permute_577 = None
    view_748: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_112, [8, 16, 1, 196, 32]);  bmm_112 = None
    view_749: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_113, [8, 16, 1, 196, 196]);  bmm_113 = None
    mul_723: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_749, alias_40);  view_749 = None
    sum_245: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_723, [-1], True)
    mul_724: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_40, sum_245);  alias_40 = sum_245 = None
    sub_193: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_723, mul_724);  mul_723 = mul_724 = None
    view_750: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_193, [128, 196, 196]);  sub_193 = None
    bmm_114: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_578, view_750);  permute_578 = None
    bmm_115: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_750, permute_579);  view_750 = permute_579 = None
    view_751: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_114, [8, 16, 1, 32, 196]);  bmm_114 = None
    view_752: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_115, [8, 16, 1, 196, 32]);  bmm_115 = None
    mul_725: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_751, 0.42044820762685725);  view_751 = None
    permute_580: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_725, [0, 1, 2, 4, 3]);  mul_725 = None
    mul_726: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_752, 0.42044820762685725);  view_752 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_16: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_726, permute_580, view_748]);  mul_726 = permute_580 = view_748 = None
    view_753: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_16, [3, 8, 16, 1, 196, 32]);  cat_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_581: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_753, [1, 3, 4, 0, 2, 5]);  view_753 = None
    clone_209: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_581, memory_format = torch.contiguous_format);  permute_581 = None
    view_754: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_209, [8, 1, 196, 1536]);  clone_209 = None
    view_755: "f32[1568, 1536]" = torch.ops.aten.view.default(view_754, [1568, 1536]);  view_754 = None
    mm_136: "f32[1568, 512]" = torch.ops.aten.mm.default(view_755, permute_582);  permute_582 = None
    permute_583: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_755, [1, 0])
    mm_137: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_583, view_122);  permute_583 = view_122 = None
    permute_584: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    sum_246: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_755, [0], True);  view_755 = None
    view_756: "f32[1536]" = torch.ops.aten.view.default(sum_246, [1536]);  sum_246 = None
    permute_585: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_584, [1, 0]);  permute_584 = None
    view_757: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_136, [8, 1, 196, 512]);  mm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_728: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_757, primals_36);  primals_36 = None
    mul_729: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_728, 512)
    sum_247: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_728, [3], True)
    mul_730: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_728, mul_79);  mul_728 = None
    sum_248: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_730, [3], True);  mul_730 = None
    mul_731: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_79, sum_248);  sum_248 = None
    sub_195: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_729, sum_247);  mul_729 = sum_247 = None
    sub_196: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_195, mul_731);  sub_195 = mul_731 = None
    mul_732: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_105, sub_196);  div_105 = sub_196 = None
    mul_733: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_757, mul_79);  mul_79 = None
    sum_249: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_733, [0, 1, 2]);  mul_733 = None
    sum_250: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_757, [0, 1, 2]);  view_757 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_244: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_243, mul_732);  add_243 = mul_732 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_734: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_244, div_18);  div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_758: "f32[1568, 512]" = torch.ops.aten.view.default(mul_734, [1568, 512]);  mul_734 = None
    mm_138: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_758, permute_586);  permute_586 = None
    permute_587: "f32[512, 1568]" = torch.ops.aten.permute.default(view_758, [1, 0])
    mm_139: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_587, view_120);  permute_587 = view_120 = None
    permute_588: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_251: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_758, [0], True);  view_758 = None
    view_759: "f32[512]" = torch.ops.aten.view.default(sum_251, [512]);  sum_251 = None
    permute_589: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_588, [1, 0]);  permute_588 = None
    view_760: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_138, [8, 1, 196, 2048]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_736: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_54, 0.5);  add_54 = None
    mul_737: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_119, view_119)
    mul_738: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_737, -0.5);  mul_737 = None
    exp_41: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_738);  mul_738 = None
    mul_739: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_41, 0.3989422804014327);  exp_41 = None
    mul_740: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_119, mul_739);  view_119 = mul_739 = None
    add_246: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_736, mul_740);  mul_736 = mul_740 = None
    mul_741: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_760, add_246);  view_760 = add_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_761: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_741, [1568, 2048]);  mul_741 = None
    mm_140: "f32[1568, 512]" = torch.ops.aten.mm.default(view_761, permute_590);  permute_590 = None
    permute_591: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_761, [1, 0])
    mm_141: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_591, view_118);  permute_591 = view_118 = None
    permute_592: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_252: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_761, [0], True);  view_761 = None
    view_762: "f32[2048]" = torch.ops.aten.view.default(sum_252, [2048]);  sum_252 = None
    permute_593: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_592, [1, 0]);  permute_592 = None
    view_763: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_140, [8, 1, 196, 512]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_743: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_763, primals_34);  primals_34 = None
    mul_744: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_743, 512)
    sum_253: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_743, [3], True)
    mul_745: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_743, mul_73);  mul_743 = None
    sum_254: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_745, [3], True);  mul_745 = None
    mul_746: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_73, sum_254);  sum_254 = None
    sub_198: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_744, sum_253);  mul_744 = sum_253 = None
    sub_199: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_198, mul_746);  sub_198 = mul_746 = None
    mul_747: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_106, sub_199);  div_106 = sub_199 = None
    mul_748: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_763, mul_73);  mul_73 = None
    sum_255: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_748, [0, 1, 2]);  mul_748 = None
    sum_256: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_763, [0, 1, 2]);  view_763 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_247: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_244, mul_747);  add_244 = mul_747 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_749: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_247, div_17);  div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_764: "f32[1568, 512]" = torch.ops.aten.view.default(mul_749, [1568, 512]);  mul_749 = None
    mm_142: "f32[1568, 512]" = torch.ops.aten.mm.default(view_764, permute_594);  permute_594 = None
    permute_595: "f32[512, 1568]" = torch.ops.aten.permute.default(view_764, [1, 0])
    mm_143: "f32[512, 512]" = torch.ops.aten.mm.default(permute_595, view_116);  permute_595 = view_116 = None
    permute_596: "f32[512, 512]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_257: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_764, [0], True);  view_764 = None
    view_765: "f32[512]" = torch.ops.aten.view.default(sum_257, [512]);  sum_257 = None
    permute_597: "f32[512, 512]" = torch.ops.aten.permute.default(permute_596, [1, 0]);  permute_596 = None
    view_766: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_142, [8, 1, 196, 512]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_767: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_766, [8, 1, 196, 32, 16]);  view_766 = None
    permute_598: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_767, [0, 4, 1, 2, 3]);  view_767 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_210: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_598, memory_format = torch.contiguous_format);  permute_598 = None
    view_768: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_210, [128, 196, 32]);  clone_210 = None
    bmm_116: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_599, view_768);  permute_599 = None
    bmm_117: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_768, permute_600);  view_768 = permute_600 = None
    view_769: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_116, [8, 16, 1, 196, 32]);  bmm_116 = None
    view_770: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_117, [8, 16, 1, 196, 196]);  bmm_117 = None
    mul_750: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_770, alias_41);  view_770 = None
    sum_258: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_750, [-1], True)
    mul_751: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_41, sum_258);  alias_41 = sum_258 = None
    sub_200: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_750, mul_751);  mul_750 = mul_751 = None
    view_771: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_200, [128, 196, 196]);  sub_200 = None
    bmm_118: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_601, view_771);  permute_601 = None
    bmm_119: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_771, permute_602);  view_771 = permute_602 = None
    view_772: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_118, [8, 16, 1, 32, 196]);  bmm_118 = None
    view_773: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_119, [8, 16, 1, 196, 32]);  bmm_119 = None
    mul_752: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_772, 0.42044820762685725);  view_772 = None
    permute_603: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_752, [0, 1, 2, 4, 3]);  mul_752 = None
    mul_753: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_773, 0.42044820762685725);  view_773 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_17: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_753, permute_603, view_769]);  mul_753 = permute_603 = view_769 = None
    view_774: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_17, [3, 8, 16, 1, 196, 32]);  cat_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_604: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_774, [1, 3, 4, 0, 2, 5]);  view_774 = None
    clone_211: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_604, memory_format = torch.contiguous_format);  permute_604 = None
    view_775: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_211, [8, 1, 196, 1536]);  clone_211 = None
    view_776: "f32[1568, 1536]" = torch.ops.aten.view.default(view_775, [1568, 1536]);  view_775 = None
    mm_144: "f32[1568, 512]" = torch.ops.aten.mm.default(view_776, permute_605);  permute_605 = None
    permute_606: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_776, [1, 0])
    mm_145: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_606, view_106);  permute_606 = view_106 = None
    permute_607: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    sum_259: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_776, [0], True);  view_776 = None
    view_777: "f32[1536]" = torch.ops.aten.view.default(sum_259, [1536]);  sum_259 = None
    permute_608: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_607, [1, 0]);  permute_607 = None
    view_778: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_144, [8, 1, 196, 512]);  mm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_755: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_778, primals_32);  primals_32 = None
    mul_756: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_755, 512)
    sum_260: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_755, [3], True)
    mul_757: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_755, mul_68);  mul_755 = None
    sum_261: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_757, [3], True);  mul_757 = None
    mul_758: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_68, sum_261);  sum_261 = None
    sub_202: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_756, sum_260);  mul_756 = sum_260 = None
    sub_203: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_202, mul_758);  sub_202 = mul_758 = None
    mul_759: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_107, sub_203);  div_107 = sub_203 = None
    mul_760: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_778, mul_68);  mul_68 = None
    sum_262: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_760, [0, 1, 2]);  mul_760 = None
    sum_263: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_778, [0, 1, 2]);  view_778 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_248: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_247, mul_759);  add_247 = mul_759 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_761: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_248, div_15);  div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_779: "f32[1568, 512]" = torch.ops.aten.view.default(mul_761, [1568, 512]);  mul_761 = None
    mm_146: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_779, permute_609);  permute_609 = None
    permute_610: "f32[512, 1568]" = torch.ops.aten.permute.default(view_779, [1, 0])
    mm_147: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_610, view_104);  permute_610 = view_104 = None
    permute_611: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    sum_264: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_779, [0], True);  view_779 = None
    view_780: "f32[512]" = torch.ops.aten.view.default(sum_264, [512]);  sum_264 = None
    permute_612: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_611, [1, 0]);  permute_611 = None
    view_781: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_146, [8, 1, 196, 2048]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_763: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_47, 0.5);  add_47 = None
    mul_764: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_103, view_103)
    mul_765: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_764, -0.5);  mul_764 = None
    exp_42: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_765);  mul_765 = None
    mul_766: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_42, 0.3989422804014327);  exp_42 = None
    mul_767: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_103, mul_766);  view_103 = mul_766 = None
    add_250: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_763, mul_767);  mul_763 = mul_767 = None
    mul_768: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_781, add_250);  view_781 = add_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_782: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_768, [1568, 2048]);  mul_768 = None
    mm_148: "f32[1568, 512]" = torch.ops.aten.mm.default(view_782, permute_613);  permute_613 = None
    permute_614: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_782, [1, 0])
    mm_149: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_614, view_102);  permute_614 = view_102 = None
    permute_615: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_149, [1, 0]);  mm_149 = None
    sum_265: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_782, [0], True);  view_782 = None
    view_783: "f32[2048]" = torch.ops.aten.view.default(sum_265, [2048]);  sum_265 = None
    permute_616: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_615, [1, 0]);  permute_615 = None
    view_784: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_148, [8, 1, 196, 512]);  mm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_770: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_784, primals_30);  primals_30 = None
    mul_771: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_770, 512)
    sum_266: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_770, [3], True)
    mul_772: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_770, mul_62);  mul_770 = None
    sum_267: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_772, [3], True);  mul_772 = None
    mul_773: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_62, sum_267);  sum_267 = None
    sub_205: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_771, sum_266);  mul_771 = sum_266 = None
    sub_206: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_205, mul_773);  sub_205 = mul_773 = None
    mul_774: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_108, sub_206);  div_108 = sub_206 = None
    mul_775: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_784, mul_62);  mul_62 = None
    sum_268: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_775, [0, 1, 2]);  mul_775 = None
    sum_269: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_784, [0, 1, 2]);  view_784 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_251: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_248, mul_774);  add_248 = mul_774 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_776: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_251, div_14);  div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_785: "f32[1568, 512]" = torch.ops.aten.view.default(mul_776, [1568, 512]);  mul_776 = None
    mm_150: "f32[1568, 512]" = torch.ops.aten.mm.default(view_785, permute_617);  permute_617 = None
    permute_618: "f32[512, 1568]" = torch.ops.aten.permute.default(view_785, [1, 0])
    mm_151: "f32[512, 512]" = torch.ops.aten.mm.default(permute_618, view_100);  permute_618 = view_100 = None
    permute_619: "f32[512, 512]" = torch.ops.aten.permute.default(mm_151, [1, 0]);  mm_151 = None
    sum_270: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_785, [0], True);  view_785 = None
    view_786: "f32[512]" = torch.ops.aten.view.default(sum_270, [512]);  sum_270 = None
    permute_620: "f32[512, 512]" = torch.ops.aten.permute.default(permute_619, [1, 0]);  permute_619 = None
    view_787: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_150, [8, 1, 196, 512]);  mm_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_788: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_787, [8, 1, 196, 32, 16]);  view_787 = None
    permute_621: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_788, [0, 4, 1, 2, 3]);  view_788 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_212: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_621, memory_format = torch.contiguous_format);  permute_621 = None
    view_789: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_212, [128, 196, 32]);  clone_212 = None
    bmm_120: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_622, view_789);  permute_622 = None
    bmm_121: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_789, permute_623);  view_789 = permute_623 = None
    view_790: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_120, [8, 16, 1, 196, 32]);  bmm_120 = None
    view_791: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_121, [8, 16, 1, 196, 196]);  bmm_121 = None
    mul_777: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_791, alias_42);  view_791 = None
    sum_271: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_777, [-1], True)
    mul_778: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_42, sum_271);  alias_42 = sum_271 = None
    sub_207: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_777, mul_778);  mul_777 = mul_778 = None
    view_792: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_207, [128, 196, 196]);  sub_207 = None
    bmm_122: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_624, view_792);  permute_624 = None
    bmm_123: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_792, permute_625);  view_792 = permute_625 = None
    view_793: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_122, [8, 16, 1, 32, 196]);  bmm_122 = None
    view_794: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_123, [8, 16, 1, 196, 32]);  bmm_123 = None
    mul_779: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_793, 0.42044820762685725);  view_793 = None
    permute_626: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_779, [0, 1, 2, 4, 3]);  mul_779 = None
    mul_780: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_794, 0.42044820762685725);  view_794 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_18: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_780, permute_626, view_790]);  mul_780 = permute_626 = view_790 = None
    view_795: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_18, [3, 8, 16, 1, 196, 32]);  cat_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_627: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_795, [1, 3, 4, 0, 2, 5]);  view_795 = None
    clone_213: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_627, memory_format = torch.contiguous_format);  permute_627 = None
    view_796: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_213, [8, 1, 196, 1536]);  clone_213 = None
    view_797: "f32[1568, 1536]" = torch.ops.aten.view.default(view_796, [1568, 1536]);  view_796 = None
    mm_152: "f32[1568, 512]" = torch.ops.aten.mm.default(view_797, permute_628);  permute_628 = None
    permute_629: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_797, [1, 0])
    mm_153: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_629, view_90);  permute_629 = view_90 = None
    permute_630: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_153, [1, 0]);  mm_153 = None
    sum_272: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_797, [0], True);  view_797 = None
    view_798: "f32[1536]" = torch.ops.aten.view.default(sum_272, [1536]);  sum_272 = None
    permute_631: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_630, [1, 0]);  permute_630 = None
    view_799: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_152, [8, 1, 196, 512]);  mm_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_782: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_799, primals_28);  primals_28 = None
    mul_783: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_782, 512)
    sum_273: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_782, [3], True)
    mul_784: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_782, mul_57);  mul_782 = None
    sum_274: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_784, [3], True);  mul_784 = None
    mul_785: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_57, sum_274);  sum_274 = None
    sub_209: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_783, sum_273);  mul_783 = sum_273 = None
    sub_210: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_209, mul_785);  sub_209 = mul_785 = None
    mul_786: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_109, sub_210);  div_109 = sub_210 = None
    mul_787: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_799, mul_57);  mul_57 = None
    sum_275: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_787, [0, 1, 2]);  mul_787 = None
    sum_276: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_799, [0, 1, 2]);  view_799 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_252: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_251, mul_786);  add_251 = mul_786 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_788: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_252, div_12);  div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_800: "f32[1568, 512]" = torch.ops.aten.view.default(mul_788, [1568, 512]);  mul_788 = None
    mm_154: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_800, permute_632);  permute_632 = None
    permute_633: "f32[512, 1568]" = torch.ops.aten.permute.default(view_800, [1, 0])
    mm_155: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_633, view_88);  permute_633 = view_88 = None
    permute_634: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_155, [1, 0]);  mm_155 = None
    sum_277: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_800, [0], True);  view_800 = None
    view_801: "f32[512]" = torch.ops.aten.view.default(sum_277, [512]);  sum_277 = None
    permute_635: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_634, [1, 0]);  permute_634 = None
    view_802: "f32[8, 1, 196, 2048]" = torch.ops.aten.view.default(mm_154, [8, 1, 196, 2048]);  mm_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_790: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(add_40, 0.5);  add_40 = None
    mul_791: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_87, view_87)
    mul_792: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_791, -0.5);  mul_791 = None
    exp_43: "f32[8, 1, 196, 2048]" = torch.ops.aten.exp.default(mul_792);  mul_792 = None
    mul_793: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_43, 0.3989422804014327);  exp_43 = None
    mul_794: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_87, mul_793);  view_87 = mul_793 = None
    add_254: "f32[8, 1, 196, 2048]" = torch.ops.aten.add.Tensor(mul_790, mul_794);  mul_790 = mul_794 = None
    mul_795: "f32[8, 1, 196, 2048]" = torch.ops.aten.mul.Tensor(view_802, add_254);  view_802 = add_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_803: "f32[1568, 2048]" = torch.ops.aten.view.default(mul_795, [1568, 2048]);  mul_795 = None
    mm_156: "f32[1568, 512]" = torch.ops.aten.mm.default(view_803, permute_636);  permute_636 = None
    permute_637: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_803, [1, 0])
    mm_157: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_637, view_86);  permute_637 = view_86 = None
    permute_638: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_157, [1, 0]);  mm_157 = None
    sum_278: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_803, [0], True);  view_803 = None
    view_804: "f32[2048]" = torch.ops.aten.view.default(sum_278, [2048]);  sum_278 = None
    permute_639: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_638, [1, 0]);  permute_638 = None
    view_805: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_156, [8, 1, 196, 512]);  mm_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_797: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_805, primals_26);  primals_26 = None
    mul_798: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_797, 512)
    sum_279: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_797, [3], True)
    mul_799: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_797, mul_51);  mul_797 = None
    sum_280: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_799, [3], True);  mul_799 = None
    mul_800: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_51, sum_280);  sum_280 = None
    sub_212: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_798, sum_279);  mul_798 = sum_279 = None
    sub_213: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_212, mul_800);  sub_212 = mul_800 = None
    mul_801: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_110, sub_213);  div_110 = sub_213 = None
    mul_802: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_805, mul_51);  mul_51 = None
    sum_281: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_802, [0, 1, 2]);  mul_802 = None
    sum_282: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_805, [0, 1, 2]);  view_805 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_255: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_252, mul_801);  add_252 = mul_801 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_803: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(add_255, div_11);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_806: "f32[1568, 512]" = torch.ops.aten.view.default(mul_803, [1568, 512]);  mul_803 = None
    mm_158: "f32[1568, 512]" = torch.ops.aten.mm.default(view_806, permute_640);  permute_640 = None
    permute_641: "f32[512, 1568]" = torch.ops.aten.permute.default(view_806, [1, 0])
    mm_159: "f32[512, 512]" = torch.ops.aten.mm.default(permute_641, view_84);  permute_641 = view_84 = None
    permute_642: "f32[512, 512]" = torch.ops.aten.permute.default(mm_159, [1, 0]);  mm_159 = None
    sum_283: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_806, [0], True);  view_806 = None
    view_807: "f32[512]" = torch.ops.aten.view.default(sum_283, [512]);  sum_283 = None
    permute_643: "f32[512, 512]" = torch.ops.aten.permute.default(permute_642, [1, 0]);  permute_642 = None
    view_808: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_158, [8, 1, 196, 512]);  mm_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_809: "f32[8, 1, 196, 32, 16]" = torch.ops.aten.view.default(view_808, [8, 1, 196, 32, 16]);  view_808 = None
    permute_644: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(view_809, [0, 4, 1, 2, 3]);  view_809 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_214: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.clone.default(permute_644, memory_format = torch.contiguous_format);  permute_644 = None
    view_810: "f32[128, 196, 32]" = torch.ops.aten.view.default(clone_214, [128, 196, 32]);  clone_214 = None
    bmm_124: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(permute_645, view_810);  permute_645 = None
    bmm_125: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_810, permute_646);  view_810 = permute_646 = None
    view_811: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_124, [8, 16, 1, 196, 32]);  bmm_124 = None
    view_812: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.view.default(bmm_125, [8, 16, 1, 196, 196]);  bmm_125 = None
    mul_804: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(view_812, alias_43);  view_812 = None
    sum_284: "f32[8, 16, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_804, [-1], True)
    mul_805: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.mul.Tensor(alias_43, sum_284);  alias_43 = sum_284 = None
    sub_214: "f32[8, 16, 1, 196, 196]" = torch.ops.aten.sub.Tensor(mul_804, mul_805);  mul_804 = mul_805 = None
    view_813: "f32[128, 196, 196]" = torch.ops.aten.view.default(sub_214, [128, 196, 196]);  sub_214 = None
    bmm_126: "f32[128, 32, 196]" = torch.ops.aten.bmm.default(permute_647, view_813);  permute_647 = None
    bmm_127: "f32[128, 196, 32]" = torch.ops.aten.bmm.default(view_813, permute_648);  view_813 = permute_648 = None
    view_814: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.view.default(bmm_126, [8, 16, 1, 32, 196]);  bmm_126 = None
    view_815: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.view.default(bmm_127, [8, 16, 1, 196, 32]);  bmm_127 = None
    mul_806: "f32[8, 16, 1, 32, 196]" = torch.ops.aten.mul.Scalar(view_814, 0.42044820762685725);  view_814 = None
    permute_649: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.permute.default(mul_806, [0, 1, 2, 4, 3]);  mul_806 = None
    mul_807: "f32[8, 16, 1, 196, 32]" = torch.ops.aten.mul.Scalar(view_815, 0.42044820762685725);  view_815 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_19: "f32[24, 16, 1, 196, 32]" = torch.ops.aten.cat.default([mul_807, permute_649, view_811]);  mul_807 = permute_649 = view_811 = None
    view_816: "f32[3, 8, 16, 1, 196, 32]" = torch.ops.aten.view.default(cat_19, [3, 8, 16, 1, 196, 32]);  cat_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_650: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.permute.default(view_816, [1, 3, 4, 0, 2, 5]);  view_816 = None
    clone_215: "f32[8, 1, 196, 3, 16, 32]" = torch.ops.aten.clone.default(permute_650, memory_format = torch.contiguous_format);  permute_650 = None
    view_817: "f32[8, 1, 196, 1536]" = torch.ops.aten.view.default(clone_215, [8, 1, 196, 1536]);  clone_215 = None
    view_818: "f32[1568, 1536]" = torch.ops.aten.view.default(view_817, [1568, 1536]);  view_817 = None
    mm_160: "f32[1568, 512]" = torch.ops.aten.mm.default(view_818, permute_651);  permute_651 = None
    permute_652: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_818, [1, 0])
    mm_161: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_652, view_74);  permute_652 = view_74 = None
    permute_653: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_161, [1, 0]);  mm_161 = None
    sum_285: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_818, [0], True);  view_818 = None
    view_819: "f32[1536]" = torch.ops.aten.view.default(sum_285, [1536]);  sum_285 = None
    permute_654: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_653, [1, 0]);  permute_653 = None
    view_820: "f32[8, 1, 196, 512]" = torch.ops.aten.view.default(mm_160, [8, 1, 196, 512]);  mm_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_809: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_820, primals_24);  primals_24 = None
    mul_810: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_809, 512)
    sum_286: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_809, [3], True)
    mul_811: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_809, mul_46);  mul_809 = None
    sum_287: "f32[8, 1, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_811, [3], True);  mul_811 = None
    mul_812: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(mul_46, sum_287);  sum_287 = None
    sub_216: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(mul_810, sum_286);  mul_810 = sum_286 = None
    sub_217: "f32[8, 1, 196, 512]" = torch.ops.aten.sub.Tensor(sub_216, mul_812);  sub_216 = mul_812 = None
    mul_813: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(div_111, sub_217);  div_111 = sub_217 = None
    mul_814: "f32[8, 1, 196, 512]" = torch.ops.aten.mul.Tensor(view_820, mul_46);  mul_46 = None
    sum_288: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_814, [0, 1, 2]);  mul_814 = None
    sum_289: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_820, [0, 1, 2]);  view_820 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_256: "f32[8, 1, 196, 512]" = torch.ops.aten.add.Tensor(add_255, mul_813);  add_255 = mul_813 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:235, code: x = x + self.pos_embed
    sum_290: "f32[1, 1, 196, 512]" = torch.ops.aten.sum.dim_IntList(add_256, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:160, code: x = x.transpose(2, 3).reshape(B, grid_height * grid_width, -1, C)
    view_821: "f32[8, 1, 1, 14, 14, 512]" = torch.ops.aten.view.default(add_256, [8, 1, 1, 14, 14, 512]);  add_256 = None
    permute_655: "f32[8, 1, 14, 1, 14, 512]" = torch.ops.aten.permute.default(view_821, [0, 1, 3, 2, 4, 5]);  view_821 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:159, code: x = x.reshape(B, grid_height, block_size, grid_width, block_size, C)
    view_822: "f32[8, 14, 14, 512]" = torch.ops.aten.view.default(permute_655, [8, 14, 14, 512]);  permute_655 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:233, code: x = x.permute(0, 2, 3, 1)  # (B, H', W', C), switch to channels last for transformer
    permute_656: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(view_822, [0, 3, 1, 2]);  view_822 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    max_pool2d_with_indices_backward: "f32[8, 512, 29, 29]" = torch.ops.aten.max_pool2d_with_indices_backward.default(permute_656, constant_pad_nd_1, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_35);  permute_656 = constant_pad_nd_1 = getitem_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_2: "f32[8, 512, 28, 28]" = torch.ops.aten.constant_pad_nd.default(max_pool2d_with_indices_backward, [0, -1, 0, -1]);  max_pool2d_with_indices_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:143, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_657: "f32[8, 28, 28, 512]" = torch.ops.aten.permute.default(constant_pad_nd_2, [0, 2, 3, 1]);  constant_pad_nd_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_816: "f32[8, 28, 28, 512]" = torch.ops.aten.mul.Tensor(permute_657, primals_21);  primals_21 = None
    mul_817: "f32[8, 28, 28, 512]" = torch.ops.aten.mul.Tensor(mul_816, 512)
    sum_291: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_816, [3], True)
    mul_818: "f32[8, 28, 28, 512]" = torch.ops.aten.mul.Tensor(mul_816, mul_44);  mul_816 = None
    sum_292: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_818, [3], True);  mul_818 = None
    mul_819: "f32[8, 28, 28, 512]" = torch.ops.aten.mul.Tensor(mul_44, sum_292);  sum_292 = None
    sub_219: "f32[8, 28, 28, 512]" = torch.ops.aten.sub.Tensor(mul_817, sum_291);  mul_817 = sum_291 = None
    sub_220: "f32[8, 28, 28, 512]" = torch.ops.aten.sub.Tensor(sub_219, mul_819);  sub_219 = mul_819 = None
    mul_820: "f32[8, 28, 28, 512]" = torch.ops.aten.mul.Tensor(div_112, sub_220);  div_112 = sub_220 = None
    mul_821: "f32[8, 28, 28, 512]" = torch.ops.aten.mul.Tensor(permute_657, mul_44);  mul_44 = None
    sum_293: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_821, [0, 1, 2]);  mul_821 = None
    sum_294: "f32[512]" = torch.ops.aten.sum.dim_IntList(permute_657, [0, 1, 2]);  permute_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:143, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_658: "f32[8, 512, 28, 28]" = torch.ops.aten.permute.default(mul_820, [0, 3, 1, 2]);  mul_820 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:141, code: x = self.conv(x)
    sum_295: "f32[512]" = torch.ops.aten.sum.dim_IntList(permute_658, [0, 2, 3])
    convolution_backward = torch.ops.aten.convolution_backward.default(permute_658, permute_37, primals_142, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  permute_658 = permute_37 = primals_142 = None
    getitem_178: "f32[8, 256, 28, 28]" = convolution_backward[0]
    getitem_179: "f32[512, 256, 3, 3]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:242, code: return x.permute(0, 3, 1, 2)  # (B, C, H', W')
    permute_659: "f32[8, 28, 28, 256]" = torch.ops.aten.permute.default(getitem_178, [0, 2, 3, 1]);  getitem_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:175, code: x = x.transpose(2, 3).reshape(B, height, width, C)
    view_823: "f32[8, 2, 14, 2, 14, 256]" = torch.ops.aten.view.default(permute_659, [8, 2, 14, 2, 14, 256]);  permute_659 = None
    permute_660: "f32[8, 2, 2, 14, 14, 256]" = torch.ops.aten.permute.default(view_823, [0, 1, 3, 2, 4, 5]);  view_823 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:174, code: x = x.reshape(B, grid_size, grid_size, block_size, block_size, C)
    clone_217: "f32[8, 2, 2, 14, 14, 256]" = torch.ops.aten.clone.default(permute_660, memory_format = torch.contiguous_format);  permute_660 = None
    view_824: "f32[8, 4, 196, 256]" = torch.ops.aten.view.default(clone_217, [8, 4, 196, 256]);  clone_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_822: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(view_824, div_9);  div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_825: "f32[6272, 256]" = torch.ops.aten.view.default(mul_822, [6272, 256]);  mul_822 = None
    mm_162: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_825, permute_661);  permute_661 = None
    permute_662: "f32[256, 6272]" = torch.ops.aten.permute.default(view_825, [1, 0])
    mm_163: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_662, view_68);  permute_662 = view_68 = None
    permute_663: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_163, [1, 0]);  mm_163 = None
    sum_296: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_825, [0], True);  view_825 = None
    view_826: "f32[256]" = torch.ops.aten.view.default(sum_296, [256]);  sum_296 = None
    permute_664: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_663, [1, 0]);  permute_663 = None
    view_827: "f32[8, 4, 196, 1024]" = torch.ops.aten.view.default(mm_162, [8, 4, 196, 1024]);  mm_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_824: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(add_30, 0.5);  add_30 = None
    mul_825: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(view_67, view_67)
    mul_826: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(mul_825, -0.5);  mul_825 = None
    exp_44: "f32[8, 4, 196, 1024]" = torch.ops.aten.exp.default(mul_826);  mul_826 = None
    mul_827: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(exp_44, 0.3989422804014327);  exp_44 = None
    mul_828: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(view_67, mul_827);  view_67 = mul_827 = None
    add_258: "f32[8, 4, 196, 1024]" = torch.ops.aten.add.Tensor(mul_824, mul_828);  mul_824 = mul_828 = None
    mul_829: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(view_827, add_258);  view_827 = add_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_828: "f32[6272, 1024]" = torch.ops.aten.view.default(mul_829, [6272, 1024]);  mul_829 = None
    mm_164: "f32[6272, 256]" = torch.ops.aten.mm.default(view_828, permute_665);  permute_665 = None
    permute_666: "f32[1024, 6272]" = torch.ops.aten.permute.default(view_828, [1, 0])
    mm_165: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_666, view_66);  permute_666 = view_66 = None
    permute_667: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_165, [1, 0]);  mm_165 = None
    sum_297: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_828, [0], True);  view_828 = None
    view_829: "f32[1024]" = torch.ops.aten.view.default(sum_297, [1024]);  sum_297 = None
    permute_668: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_667, [1, 0]);  permute_667 = None
    view_830: "f32[8, 4, 196, 256]" = torch.ops.aten.view.default(mm_164, [8, 4, 196, 256]);  mm_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_831: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(view_830, primals_19);  primals_19 = None
    mul_832: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_831, 256)
    sum_298: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_831, [3], True)
    mul_833: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_831, mul_38);  mul_831 = None
    sum_299: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_833, [3], True);  mul_833 = None
    mul_834: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_38, sum_299);  sum_299 = None
    sub_222: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(mul_832, sum_298);  mul_832 = sum_298 = None
    sub_223: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(sub_222, mul_834);  sub_222 = mul_834 = None
    mul_835: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(div_113, sub_223);  div_113 = sub_223 = None
    mul_836: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(view_830, mul_38);  mul_38 = None
    sum_300: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_836, [0, 1, 2]);  mul_836 = None
    sum_301: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_830, [0, 1, 2]);  view_830 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_259: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(view_824, mul_835);  view_824 = mul_835 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_837: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(add_259, div_8);  div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_831: "f32[6272, 256]" = torch.ops.aten.view.default(mul_837, [6272, 256]);  mul_837 = None
    mm_166: "f32[6272, 256]" = torch.ops.aten.mm.default(view_831, permute_669);  permute_669 = None
    permute_670: "f32[256, 6272]" = torch.ops.aten.permute.default(view_831, [1, 0])
    mm_167: "f32[256, 256]" = torch.ops.aten.mm.default(permute_670, view_64);  permute_670 = view_64 = None
    permute_671: "f32[256, 256]" = torch.ops.aten.permute.default(mm_167, [1, 0]);  mm_167 = None
    sum_302: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_831, [0], True);  view_831 = None
    view_832: "f32[256]" = torch.ops.aten.view.default(sum_302, [256]);  sum_302 = None
    permute_672: "f32[256, 256]" = torch.ops.aten.permute.default(permute_671, [1, 0]);  permute_671 = None
    view_833: "f32[8, 4, 196, 256]" = torch.ops.aten.view.default(mm_166, [8, 4, 196, 256]);  mm_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_834: "f32[8, 4, 196, 32, 8]" = torch.ops.aten.view.default(view_833, [8, 4, 196, 32, 8]);  view_833 = None
    permute_673: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.permute.default(view_834, [0, 4, 1, 2, 3]);  view_834 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_218: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.clone.default(permute_673, memory_format = torch.contiguous_format);  permute_673 = None
    view_835: "f32[256, 196, 32]" = torch.ops.aten.view.default(clone_218, [256, 196, 32]);  clone_218 = None
    bmm_128: "f32[256, 196, 32]" = torch.ops.aten.bmm.default(permute_674, view_835);  permute_674 = None
    bmm_129: "f32[256, 196, 196]" = torch.ops.aten.bmm.default(view_835, permute_675);  view_835 = permute_675 = None
    view_836: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.view.default(bmm_128, [8, 8, 4, 196, 32]);  bmm_128 = None
    view_837: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.view.default(bmm_129, [8, 8, 4, 196, 196]);  bmm_129 = None
    mul_838: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(view_837, alias_44);  view_837 = None
    sum_303: "f32[8, 8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_838, [-1], True)
    mul_839: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(alias_44, sum_303);  alias_44 = sum_303 = None
    sub_224: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(mul_838, mul_839);  mul_838 = mul_839 = None
    view_838: "f32[256, 196, 196]" = torch.ops.aten.view.default(sub_224, [256, 196, 196]);  sub_224 = None
    bmm_130: "f32[256, 32, 196]" = torch.ops.aten.bmm.default(permute_676, view_838);  permute_676 = None
    bmm_131: "f32[256, 196, 32]" = torch.ops.aten.bmm.default(view_838, permute_677);  view_838 = permute_677 = None
    view_839: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.view.default(bmm_130, [8, 8, 4, 32, 196]);  bmm_130 = None
    view_840: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.view.default(bmm_131, [8, 8, 4, 196, 32]);  bmm_131 = None
    mul_840: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.mul.Scalar(view_839, 0.42044820762685725);  view_839 = None
    permute_678: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.permute.default(mul_840, [0, 1, 2, 4, 3]);  mul_840 = None
    mul_841: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.mul.Scalar(view_840, 0.42044820762685725);  view_840 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_20: "f32[24, 8, 4, 196, 32]" = torch.ops.aten.cat.default([mul_841, permute_678, view_836]);  mul_841 = permute_678 = view_836 = None
    view_841: "f32[3, 8, 8, 4, 196, 32]" = torch.ops.aten.view.default(cat_20, [3, 8, 8, 4, 196, 32]);  cat_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_679: "f32[8, 4, 196, 3, 8, 32]" = torch.ops.aten.permute.default(view_841, [1, 3, 4, 0, 2, 5]);  view_841 = None
    clone_219: "f32[8, 4, 196, 3, 8, 32]" = torch.ops.aten.clone.default(permute_679, memory_format = torch.contiguous_format);  permute_679 = None
    view_842: "f32[8, 4, 196, 768]" = torch.ops.aten.view.default(clone_219, [8, 4, 196, 768]);  clone_219 = None
    view_843: "f32[6272, 768]" = torch.ops.aten.view.default(view_842, [6272, 768]);  view_842 = None
    mm_168: "f32[6272, 256]" = torch.ops.aten.mm.default(view_843, permute_680);  permute_680 = None
    permute_681: "f32[768, 6272]" = torch.ops.aten.permute.default(view_843, [1, 0])
    mm_169: "f32[768, 256]" = torch.ops.aten.mm.default(permute_681, view_54);  permute_681 = view_54 = None
    permute_682: "f32[256, 768]" = torch.ops.aten.permute.default(mm_169, [1, 0]);  mm_169 = None
    sum_304: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_843, [0], True);  view_843 = None
    view_844: "f32[768]" = torch.ops.aten.view.default(sum_304, [768]);  sum_304 = None
    permute_683: "f32[768, 256]" = torch.ops.aten.permute.default(permute_682, [1, 0]);  permute_682 = None
    view_845: "f32[8, 4, 196, 256]" = torch.ops.aten.view.default(mm_168, [8, 4, 196, 256]);  mm_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_843: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(view_845, primals_17);  primals_17 = None
    mul_844: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_843, 256)
    sum_305: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_843, [3], True)
    mul_845: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_843, mul_33);  mul_843 = None
    sum_306: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_845, [3], True);  mul_845 = None
    mul_846: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_33, sum_306);  sum_306 = None
    sub_226: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(mul_844, sum_305);  mul_844 = sum_305 = None
    sub_227: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(sub_226, mul_846);  sub_226 = mul_846 = None
    mul_847: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(div_114, sub_227);  div_114 = sub_227 = None
    mul_848: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(view_845, mul_33);  mul_33 = None
    sum_307: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_848, [0, 1, 2]);  mul_848 = None
    sum_308: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_845, [0, 1, 2]);  view_845 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_260: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(add_259, mul_847);  add_259 = mul_847 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_849: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(add_260, div_6);  div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_846: "f32[6272, 256]" = torch.ops.aten.view.default(mul_849, [6272, 256]);  mul_849 = None
    mm_170: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_846, permute_684);  permute_684 = None
    permute_685: "f32[256, 6272]" = torch.ops.aten.permute.default(view_846, [1, 0])
    mm_171: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_685, view_52);  permute_685 = view_52 = None
    permute_686: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_171, [1, 0]);  mm_171 = None
    sum_309: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_846, [0], True);  view_846 = None
    view_847: "f32[256]" = torch.ops.aten.view.default(sum_309, [256]);  sum_309 = None
    permute_687: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_686, [1, 0]);  permute_686 = None
    view_848: "f32[8, 4, 196, 1024]" = torch.ops.aten.view.default(mm_170, [8, 4, 196, 1024]);  mm_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_851: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(add_23, 0.5);  add_23 = None
    mul_852: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(view_51, view_51)
    mul_853: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(mul_852, -0.5);  mul_852 = None
    exp_45: "f32[8, 4, 196, 1024]" = torch.ops.aten.exp.default(mul_853);  mul_853 = None
    mul_854: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(exp_45, 0.3989422804014327);  exp_45 = None
    mul_855: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(view_51, mul_854);  view_51 = mul_854 = None
    add_262: "f32[8, 4, 196, 1024]" = torch.ops.aten.add.Tensor(mul_851, mul_855);  mul_851 = mul_855 = None
    mul_856: "f32[8, 4, 196, 1024]" = torch.ops.aten.mul.Tensor(view_848, add_262);  view_848 = add_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_849: "f32[6272, 1024]" = torch.ops.aten.view.default(mul_856, [6272, 1024]);  mul_856 = None
    mm_172: "f32[6272, 256]" = torch.ops.aten.mm.default(view_849, permute_688);  permute_688 = None
    permute_689: "f32[1024, 6272]" = torch.ops.aten.permute.default(view_849, [1, 0])
    mm_173: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_689, view_50);  permute_689 = view_50 = None
    permute_690: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_173, [1, 0]);  mm_173 = None
    sum_310: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_849, [0], True);  view_849 = None
    view_850: "f32[1024]" = torch.ops.aten.view.default(sum_310, [1024]);  sum_310 = None
    permute_691: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_690, [1, 0]);  permute_690 = None
    view_851: "f32[8, 4, 196, 256]" = torch.ops.aten.view.default(mm_172, [8, 4, 196, 256]);  mm_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_858: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(view_851, primals_15);  primals_15 = None
    mul_859: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_858, 256)
    sum_311: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_858, [3], True)
    mul_860: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_858, mul_27);  mul_858 = None
    sum_312: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_860, [3], True);  mul_860 = None
    mul_861: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_27, sum_312);  sum_312 = None
    sub_229: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(mul_859, sum_311);  mul_859 = sum_311 = None
    sub_230: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(sub_229, mul_861);  sub_229 = mul_861 = None
    mul_862: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(div_115, sub_230);  div_115 = sub_230 = None
    mul_863: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(view_851, mul_27);  mul_27 = None
    sum_313: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_863, [0, 1, 2]);  mul_863 = None
    sum_314: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_851, [0, 1, 2]);  view_851 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_263: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(add_260, mul_862);  add_260 = mul_862 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_864: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(add_263, div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_852: "f32[6272, 256]" = torch.ops.aten.view.default(mul_864, [6272, 256]);  mul_864 = None
    mm_174: "f32[6272, 256]" = torch.ops.aten.mm.default(view_852, permute_692);  permute_692 = None
    permute_693: "f32[256, 6272]" = torch.ops.aten.permute.default(view_852, [1, 0])
    mm_175: "f32[256, 256]" = torch.ops.aten.mm.default(permute_693, view_48);  permute_693 = view_48 = None
    permute_694: "f32[256, 256]" = torch.ops.aten.permute.default(mm_175, [1, 0]);  mm_175 = None
    sum_315: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_852, [0], True);  view_852 = None
    view_853: "f32[256]" = torch.ops.aten.view.default(sum_315, [256]);  sum_315 = None
    permute_695: "f32[256, 256]" = torch.ops.aten.permute.default(permute_694, [1, 0]);  permute_694 = None
    view_854: "f32[8, 4, 196, 256]" = torch.ops.aten.view.default(mm_174, [8, 4, 196, 256]);  mm_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_855: "f32[8, 4, 196, 32, 8]" = torch.ops.aten.view.default(view_854, [8, 4, 196, 32, 8]);  view_854 = None
    permute_696: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.permute.default(view_855, [0, 4, 1, 2, 3]);  view_855 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_220: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.clone.default(permute_696, memory_format = torch.contiguous_format);  permute_696 = None
    view_856: "f32[256, 196, 32]" = torch.ops.aten.view.default(clone_220, [256, 196, 32]);  clone_220 = None
    bmm_132: "f32[256, 196, 32]" = torch.ops.aten.bmm.default(permute_697, view_856);  permute_697 = None
    bmm_133: "f32[256, 196, 196]" = torch.ops.aten.bmm.default(view_856, permute_698);  view_856 = permute_698 = None
    view_857: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.view.default(bmm_132, [8, 8, 4, 196, 32]);  bmm_132 = None
    view_858: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.view.default(bmm_133, [8, 8, 4, 196, 196]);  bmm_133 = None
    mul_865: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(view_858, alias_45);  view_858 = None
    sum_316: "f32[8, 8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_865, [-1], True)
    mul_866: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(alias_45, sum_316);  alias_45 = sum_316 = None
    sub_231: "f32[8, 8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(mul_865, mul_866);  mul_865 = mul_866 = None
    view_859: "f32[256, 196, 196]" = torch.ops.aten.view.default(sub_231, [256, 196, 196]);  sub_231 = None
    bmm_134: "f32[256, 32, 196]" = torch.ops.aten.bmm.default(permute_699, view_859);  permute_699 = None
    bmm_135: "f32[256, 196, 32]" = torch.ops.aten.bmm.default(view_859, permute_700);  view_859 = permute_700 = None
    view_860: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.view.default(bmm_134, [8, 8, 4, 32, 196]);  bmm_134 = None
    view_861: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.view.default(bmm_135, [8, 8, 4, 196, 32]);  bmm_135 = None
    mul_867: "f32[8, 8, 4, 32, 196]" = torch.ops.aten.mul.Scalar(view_860, 0.42044820762685725);  view_860 = None
    permute_701: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.permute.default(mul_867, [0, 1, 2, 4, 3]);  mul_867 = None
    mul_868: "f32[8, 8, 4, 196, 32]" = torch.ops.aten.mul.Scalar(view_861, 0.42044820762685725);  view_861 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_21: "f32[24, 8, 4, 196, 32]" = torch.ops.aten.cat.default([mul_868, permute_701, view_857]);  mul_868 = permute_701 = view_857 = None
    view_862: "f32[3, 8, 8, 4, 196, 32]" = torch.ops.aten.view.default(cat_21, [3, 8, 8, 4, 196, 32]);  cat_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_702: "f32[8, 4, 196, 3, 8, 32]" = torch.ops.aten.permute.default(view_862, [1, 3, 4, 0, 2, 5]);  view_862 = None
    clone_221: "f32[8, 4, 196, 3, 8, 32]" = torch.ops.aten.clone.default(permute_702, memory_format = torch.contiguous_format);  permute_702 = None
    view_863: "f32[8, 4, 196, 768]" = torch.ops.aten.view.default(clone_221, [8, 4, 196, 768]);  clone_221 = None
    view_864: "f32[6272, 768]" = torch.ops.aten.view.default(view_863, [6272, 768]);  view_863 = None
    mm_176: "f32[6272, 256]" = torch.ops.aten.mm.default(view_864, permute_703);  permute_703 = None
    permute_704: "f32[768, 6272]" = torch.ops.aten.permute.default(view_864, [1, 0])
    mm_177: "f32[768, 256]" = torch.ops.aten.mm.default(permute_704, view_38);  permute_704 = view_38 = None
    permute_705: "f32[256, 768]" = torch.ops.aten.permute.default(mm_177, [1, 0]);  mm_177 = None
    sum_317: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_864, [0], True);  view_864 = None
    view_865: "f32[768]" = torch.ops.aten.view.default(sum_317, [768]);  sum_317 = None
    permute_706: "f32[768, 256]" = torch.ops.aten.permute.default(permute_705, [1, 0]);  permute_705 = None
    view_866: "f32[8, 4, 196, 256]" = torch.ops.aten.view.default(mm_176, [8, 4, 196, 256]);  mm_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_870: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(view_866, primals_13);  primals_13 = None
    mul_871: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_870, 256)
    sum_318: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_870, [3], True)
    mul_872: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_870, mul_22);  mul_870 = None
    sum_319: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_872, [3], True);  mul_872 = None
    mul_873: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(mul_22, sum_319);  sum_319 = None
    sub_233: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(mul_871, sum_318);  mul_871 = sum_318 = None
    sub_234: "f32[8, 4, 196, 256]" = torch.ops.aten.sub.Tensor(sub_233, mul_873);  sub_233 = mul_873 = None
    mul_874: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(div_116, sub_234);  div_116 = sub_234 = None
    mul_875: "f32[8, 4, 196, 256]" = torch.ops.aten.mul.Tensor(view_866, mul_22);  mul_22 = None
    sum_320: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_875, [0, 1, 2]);  mul_875 = None
    sum_321: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_866, [0, 1, 2]);  view_866 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_264: "f32[8, 4, 196, 256]" = torch.ops.aten.add.Tensor(add_263, mul_874);  add_263 = mul_874 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:235, code: x = x + self.pos_embed
    sum_322: "f32[1, 4, 196, 256]" = torch.ops.aten.sum.dim_IntList(add_264, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:160, code: x = x.transpose(2, 3).reshape(B, grid_height * grid_width, -1, C)
    view_867: "f32[8, 2, 2, 14, 14, 256]" = torch.ops.aten.view.default(add_264, [8, 2, 2, 14, 14, 256]);  add_264 = None
    permute_707: "f32[8, 2, 14, 2, 14, 256]" = torch.ops.aten.permute.default(view_867, [0, 1, 3, 2, 4, 5]);  view_867 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:159, code: x = x.reshape(B, grid_height, block_size, grid_width, block_size, C)
    clone_222: "f32[8, 2, 14, 2, 14, 256]" = torch.ops.aten.clone.default(permute_707, memory_format = torch.contiguous_format);  permute_707 = None
    view_868: "f32[8, 28, 28, 256]" = torch.ops.aten.view.default(clone_222, [8, 28, 28, 256]);  clone_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:233, code: x = x.permute(0, 2, 3, 1)  # (B, H', W', C), switch to channels last for transformer
    permute_708: "f32[8, 256, 28, 28]" = torch.ops.aten.permute.default(view_868, [0, 3, 1, 2]);  view_868 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    max_pool2d_with_indices_backward_1: "f32[8, 256, 57, 57]" = torch.ops.aten.max_pool2d_with_indices_backward.default(permute_708, constant_pad_nd, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_17);  permute_708 = constant_pad_nd = getitem_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_3: "f32[8, 256, 56, 56]" = torch.ops.aten.constant_pad_nd.default(max_pool2d_with_indices_backward_1, [0, -1, 0, -1]);  max_pool2d_with_indices_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:143, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_709: "f32[8, 56, 56, 256]" = torch.ops.aten.permute.default(constant_pad_nd_3, [0, 2, 3, 1]);  constant_pad_nd_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_877: "f32[8, 56, 56, 256]" = torch.ops.aten.mul.Tensor(permute_709, primals_10);  primals_10 = None
    mul_878: "f32[8, 56, 56, 256]" = torch.ops.aten.mul.Tensor(mul_877, 256)
    sum_323: "f32[8, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_877, [3], True)
    mul_879: "f32[8, 56, 56, 256]" = torch.ops.aten.mul.Tensor(mul_877, mul_20);  mul_877 = None
    sum_324: "f32[8, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_879, [3], True);  mul_879 = None
    mul_880: "f32[8, 56, 56, 256]" = torch.ops.aten.mul.Tensor(mul_20, sum_324);  sum_324 = None
    sub_236: "f32[8, 56, 56, 256]" = torch.ops.aten.sub.Tensor(mul_878, sum_323);  mul_878 = sum_323 = None
    sub_237: "f32[8, 56, 56, 256]" = torch.ops.aten.sub.Tensor(sub_236, mul_880);  sub_236 = mul_880 = None
    mul_881: "f32[8, 56, 56, 256]" = torch.ops.aten.mul.Tensor(div_117, sub_237);  div_117 = sub_237 = None
    mul_882: "f32[8, 56, 56, 256]" = torch.ops.aten.mul.Tensor(permute_709, mul_20);  mul_20 = None
    sum_325: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_882, [0, 1, 2]);  mul_882 = None
    sum_326: "f32[256]" = torch.ops.aten.sum.dim_IntList(permute_709, [0, 1, 2]);  permute_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:143, code: x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_710: "f32[8, 256, 56, 56]" = torch.ops.aten.permute.default(mul_881, [0, 3, 1, 2]);  mul_881 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:141, code: x = self.conv(x)
    sum_327: "f32[256]" = torch.ops.aten.sum.dim_IntList(permute_710, [0, 2, 3])
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(permute_710, permute_17, primals_124, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  permute_710 = permute_17 = primals_124 = None
    getitem_181: "f32[8, 128, 56, 56]" = convolution_backward_1[0]
    getitem_182: "f32[256, 128, 3, 3]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:242, code: return x.permute(0, 3, 1, 2)  # (B, C, H', W')
    permute_711: "f32[8, 56, 56, 128]" = torch.ops.aten.permute.default(getitem_181, [0, 2, 3, 1]);  getitem_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:175, code: x = x.transpose(2, 3).reshape(B, height, width, C)
    view_869: "f32[8, 4, 14, 4, 14, 128]" = torch.ops.aten.view.default(permute_711, [8, 4, 14, 4, 14, 128]);  permute_711 = None
    permute_712: "f32[8, 4, 4, 14, 14, 128]" = torch.ops.aten.permute.default(view_869, [0, 1, 3, 2, 4, 5]);  view_869 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:174, code: x = x.reshape(B, grid_size, grid_size, block_size, block_size, C)
    clone_224: "f32[8, 4, 4, 14, 14, 128]" = torch.ops.aten.clone.default(permute_712, memory_format = torch.contiguous_format);  permute_712 = None
    view_870: "f32[8, 16, 196, 128]" = torch.ops.aten.view.default(clone_224, [8, 16, 196, 128]);  clone_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_883: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(view_870, div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_871: "f32[25088, 128]" = torch.ops.aten.view.default(mul_883, [25088, 128]);  mul_883 = None
    mm_178: "f32[25088, 512]" = torch.ops.aten.mm.default(view_871, permute_713);  permute_713 = None
    permute_714: "f32[128, 25088]" = torch.ops.aten.permute.default(view_871, [1, 0])
    mm_179: "f32[128, 512]" = torch.ops.aten.mm.default(permute_714, view_32);  permute_714 = view_32 = None
    permute_715: "f32[512, 128]" = torch.ops.aten.permute.default(mm_179, [1, 0]);  mm_179 = None
    sum_328: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_871, [0], True);  view_871 = None
    view_872: "f32[128]" = torch.ops.aten.view.default(sum_328, [128]);  sum_328 = None
    permute_716: "f32[128, 512]" = torch.ops.aten.permute.default(permute_715, [1, 0]);  permute_715 = None
    view_873: "f32[8, 16, 196, 512]" = torch.ops.aten.view.default(mm_178, [8, 16, 196, 512]);  mm_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_885: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(add_13, 0.5);  add_13 = None
    mul_886: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(view_31, view_31)
    mul_887: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(mul_886, -0.5);  mul_886 = None
    exp_46: "f32[8, 16, 196, 512]" = torch.ops.aten.exp.default(mul_887);  mul_887 = None
    mul_888: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(exp_46, 0.3989422804014327);  exp_46 = None
    mul_889: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(view_31, mul_888);  view_31 = mul_888 = None
    add_266: "f32[8, 16, 196, 512]" = torch.ops.aten.add.Tensor(mul_885, mul_889);  mul_885 = mul_889 = None
    mul_890: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(view_873, add_266);  view_873 = add_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_874: "f32[25088, 512]" = torch.ops.aten.view.default(mul_890, [25088, 512]);  mul_890 = None
    mm_180: "f32[25088, 128]" = torch.ops.aten.mm.default(view_874, permute_717);  permute_717 = None
    permute_718: "f32[512, 25088]" = torch.ops.aten.permute.default(view_874, [1, 0])
    mm_181: "f32[512, 128]" = torch.ops.aten.mm.default(permute_718, view_30);  permute_718 = view_30 = None
    permute_719: "f32[128, 512]" = torch.ops.aten.permute.default(mm_181, [1, 0]);  mm_181 = None
    sum_329: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_874, [0], True);  view_874 = None
    view_875: "f32[512]" = torch.ops.aten.view.default(sum_329, [512]);  sum_329 = None
    permute_720: "f32[512, 128]" = torch.ops.aten.permute.default(permute_719, [1, 0]);  permute_719 = None
    view_876: "f32[8, 16, 196, 128]" = torch.ops.aten.view.default(mm_180, [8, 16, 196, 128]);  mm_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_892: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(view_876, primals_8);  primals_8 = None
    mul_893: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_892, 128)
    sum_330: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_892, [3], True)
    mul_894: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_892, mul_14);  mul_892 = None
    sum_331: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_894, [3], True);  mul_894 = None
    mul_895: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_14, sum_331);  sum_331 = None
    sub_239: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(mul_893, sum_330);  mul_893 = sum_330 = None
    sub_240: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(sub_239, mul_895);  sub_239 = mul_895 = None
    mul_896: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(div_118, sub_240);  div_118 = sub_240 = None
    mul_897: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(view_876, mul_14);  mul_14 = None
    sum_332: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_897, [0, 1, 2]);  mul_897 = None
    sum_333: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_876, [0, 1, 2]);  view_876 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_267: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(view_870, mul_896);  view_870 = mul_896 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_898: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(add_267, div_2);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_877: "f32[25088, 128]" = torch.ops.aten.view.default(mul_898, [25088, 128]);  mul_898 = None
    mm_182: "f32[25088, 128]" = torch.ops.aten.mm.default(view_877, permute_721);  permute_721 = None
    permute_722: "f32[128, 25088]" = torch.ops.aten.permute.default(view_877, [1, 0])
    mm_183: "f32[128, 128]" = torch.ops.aten.mm.default(permute_722, view_28);  permute_722 = view_28 = None
    permute_723: "f32[128, 128]" = torch.ops.aten.permute.default(mm_183, [1, 0]);  mm_183 = None
    sum_334: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_877, [0], True);  view_877 = None
    view_878: "f32[128]" = torch.ops.aten.view.default(sum_334, [128]);  sum_334 = None
    permute_724: "f32[128, 128]" = torch.ops.aten.permute.default(permute_723, [1, 0]);  permute_723 = None
    view_879: "f32[8, 16, 196, 128]" = torch.ops.aten.view.default(mm_182, [8, 16, 196, 128]);  mm_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_880: "f32[8, 16, 196, 32, 4]" = torch.ops.aten.view.default(view_879, [8, 16, 196, 32, 4]);  view_879 = None
    permute_725: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.permute.default(view_880, [0, 4, 1, 2, 3]);  view_880 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_225: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.clone.default(permute_725, memory_format = torch.contiguous_format);  permute_725 = None
    view_881: "f32[512, 196, 32]" = torch.ops.aten.view.default(clone_225, [512, 196, 32]);  clone_225 = None
    bmm_136: "f32[512, 196, 32]" = torch.ops.aten.bmm.default(permute_726, view_881);  permute_726 = None
    bmm_137: "f32[512, 196, 196]" = torch.ops.aten.bmm.default(view_881, permute_727);  view_881 = permute_727 = None
    view_882: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.view.default(bmm_136, [8, 4, 16, 196, 32]);  bmm_136 = None
    view_883: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.view.default(bmm_137, [8, 4, 16, 196, 196]);  bmm_137 = None
    mul_899: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_883, alias_46);  view_883 = None
    sum_335: "f32[8, 4, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_899, [-1], True)
    mul_900: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.mul.Tensor(alias_46, sum_335);  alias_46 = sum_335 = None
    sub_241: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_899, mul_900);  mul_899 = mul_900 = None
    view_884: "f32[512, 196, 196]" = torch.ops.aten.view.default(sub_241, [512, 196, 196]);  sub_241 = None
    bmm_138: "f32[512, 32, 196]" = torch.ops.aten.bmm.default(permute_728, view_884);  permute_728 = None
    bmm_139: "f32[512, 196, 32]" = torch.ops.aten.bmm.default(view_884, permute_729);  view_884 = permute_729 = None
    view_885: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.view.default(bmm_138, [8, 4, 16, 32, 196]);  bmm_138 = None
    view_886: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.view.default(bmm_139, [8, 4, 16, 196, 32]);  bmm_139 = None
    mul_901: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.mul.Scalar(view_885, 0.42044820762685725);  view_885 = None
    permute_730: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.permute.default(mul_901, [0, 1, 2, 4, 3]);  mul_901 = None
    mul_902: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.mul.Scalar(view_886, 0.42044820762685725);  view_886 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_22: "f32[24, 4, 16, 196, 32]" = torch.ops.aten.cat.default([mul_902, permute_730, view_882]);  mul_902 = permute_730 = view_882 = None
    view_887: "f32[3, 8, 4, 16, 196, 32]" = torch.ops.aten.view.default(cat_22, [3, 8, 4, 16, 196, 32]);  cat_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_731: "f32[8, 16, 196, 3, 4, 32]" = torch.ops.aten.permute.default(view_887, [1, 3, 4, 0, 2, 5]);  view_887 = None
    clone_226: "f32[8, 16, 196, 3, 4, 32]" = torch.ops.aten.clone.default(permute_731, memory_format = torch.contiguous_format);  permute_731 = None
    view_888: "f32[8, 16, 196, 384]" = torch.ops.aten.view.default(clone_226, [8, 16, 196, 384]);  clone_226 = None
    view_889: "f32[25088, 384]" = torch.ops.aten.view.default(view_888, [25088, 384]);  view_888 = None
    mm_184: "f32[25088, 128]" = torch.ops.aten.mm.default(view_889, permute_732);  permute_732 = None
    permute_733: "f32[384, 25088]" = torch.ops.aten.permute.default(view_889, [1, 0])
    mm_185: "f32[384, 128]" = torch.ops.aten.mm.default(permute_733, view_18);  permute_733 = view_18 = None
    permute_734: "f32[128, 384]" = torch.ops.aten.permute.default(mm_185, [1, 0]);  mm_185 = None
    sum_336: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_889, [0], True);  view_889 = None
    view_890: "f32[384]" = torch.ops.aten.view.default(sum_336, [384]);  sum_336 = None
    permute_735: "f32[384, 128]" = torch.ops.aten.permute.default(permute_734, [1, 0]);  permute_734 = None
    view_891: "f32[8, 16, 196, 128]" = torch.ops.aten.view.default(mm_184, [8, 16, 196, 128]);  mm_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_904: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(view_891, primals_6);  primals_6 = None
    mul_905: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_904, 128)
    sum_337: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_904, [3], True)
    mul_906: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_904, mul_9);  mul_904 = None
    sum_338: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_906, [3], True);  mul_906 = None
    mul_907: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_9, sum_338);  sum_338 = None
    sub_243: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(mul_905, sum_337);  mul_905 = sum_337 = None
    sub_244: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(sub_243, mul_907);  sub_243 = mul_907 = None
    mul_908: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(div_119, sub_244);  div_119 = sub_244 = None
    mul_909: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(view_891, mul_9);  mul_9 = None
    sum_339: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_909, [0, 1, 2]);  mul_909 = None
    sum_340: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_891, [0, 1, 2]);  view_891 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_268: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(add_267, mul_908);  add_267 = mul_908 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_892: "f32[25088, 128]" = torch.ops.aten.view.default(add_268, [25088, 128])
    mm_186: "f32[25088, 512]" = torch.ops.aten.mm.default(view_892, permute_736);  permute_736 = None
    permute_737: "f32[128, 25088]" = torch.ops.aten.permute.default(view_892, [1, 0])
    mm_187: "f32[128, 512]" = torch.ops.aten.mm.default(permute_737, view_16);  permute_737 = view_16 = None
    permute_738: "f32[512, 128]" = torch.ops.aten.permute.default(mm_187, [1, 0]);  mm_187 = None
    sum_341: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_892, [0], True);  view_892 = None
    view_893: "f32[128]" = torch.ops.aten.view.default(sum_341, [128]);  sum_341 = None
    permute_739: "f32[128, 512]" = torch.ops.aten.permute.default(permute_738, [1, 0]);  permute_738 = None
    view_894: "f32[8, 16, 196, 512]" = torch.ops.aten.view.default(mm_186, [8, 16, 196, 512]);  mm_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_911: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(add_6, 0.5);  add_6 = None
    mul_912: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(view_15, view_15)
    mul_913: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(mul_912, -0.5);  mul_912 = None
    exp_47: "f32[8, 16, 196, 512]" = torch.ops.aten.exp.default(mul_913);  mul_913 = None
    mul_914: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(exp_47, 0.3989422804014327);  exp_47 = None
    mul_915: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(view_15, mul_914);  view_15 = mul_914 = None
    add_270: "f32[8, 16, 196, 512]" = torch.ops.aten.add.Tensor(mul_911, mul_915);  mul_911 = mul_915 = None
    mul_916: "f32[8, 16, 196, 512]" = torch.ops.aten.mul.Tensor(view_894, add_270);  view_894 = add_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_895: "f32[25088, 512]" = torch.ops.aten.view.default(mul_916, [25088, 512]);  mul_916 = None
    mm_188: "f32[25088, 128]" = torch.ops.aten.mm.default(view_895, permute_740);  permute_740 = None
    permute_741: "f32[512, 25088]" = torch.ops.aten.permute.default(view_895, [1, 0])
    mm_189: "f32[512, 128]" = torch.ops.aten.mm.default(permute_741, view_14);  permute_741 = view_14 = None
    permute_742: "f32[128, 512]" = torch.ops.aten.permute.default(mm_189, [1, 0]);  mm_189 = None
    sum_342: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_895, [0], True);  view_895 = None
    view_896: "f32[512]" = torch.ops.aten.view.default(sum_342, [512]);  sum_342 = None
    permute_743: "f32[512, 128]" = torch.ops.aten.permute.default(permute_742, [1, 0]);  permute_742 = None
    view_897: "f32[8, 16, 196, 128]" = torch.ops.aten.view.default(mm_188, [8, 16, 196, 128]);  mm_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_918: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(view_897, primals_4);  primals_4 = None
    mul_919: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_918, 128)
    sum_343: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_918, [3], True)
    mul_920: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_918, mul_4);  mul_918 = None
    sum_344: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_920, [3], True);  mul_920 = None
    mul_921: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_4, sum_344);  sum_344 = None
    sub_246: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(mul_919, sum_343);  mul_919 = sum_343 = None
    sub_247: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(sub_246, mul_921);  sub_246 = mul_921 = None
    mul_922: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(div_120, sub_247);  div_120 = sub_247 = None
    mul_923: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(view_897, mul_4);  mul_4 = None
    sum_345: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_923, [0, 1, 2]);  mul_923 = None
    sum_346: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_897, [0, 1, 2]);  view_897 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_271: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(add_268, mul_922);  add_268 = mul_922 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:79, code: x = self.proj(x)
    view_898: "f32[25088, 128]" = torch.ops.aten.view.default(add_271, [25088, 128])
    mm_190: "f32[25088, 128]" = torch.ops.aten.mm.default(view_898, permute_744);  permute_744 = None
    permute_745: "f32[128, 25088]" = torch.ops.aten.permute.default(view_898, [1, 0])
    mm_191: "f32[128, 128]" = torch.ops.aten.mm.default(permute_745, view_12);  permute_745 = view_12 = None
    permute_746: "f32[128, 128]" = torch.ops.aten.permute.default(mm_191, [1, 0]);  mm_191 = None
    sum_347: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_898, [0], True);  view_898 = None
    view_899: "f32[128]" = torch.ops.aten.view.default(sum_347, [128]);  sum_347 = None
    permute_747: "f32[128, 128]" = torch.ops.aten.permute.default(permute_746, [1, 0]);  permute_746 = None
    view_900: "f32[8, 16, 196, 128]" = torch.ops.aten.view.default(mm_190, [8, 16, 196, 128]);  mm_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:78, code: x = x.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
    view_901: "f32[8, 16, 196, 32, 4]" = torch.ops.aten.view.default(view_900, [8, 16, 196, 32, 4]);  view_900 = None
    permute_748: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.permute.default(view_901, [0, 4, 1, 2, 3]);  view_901 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:69, code: x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
    clone_227: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.clone.default(permute_748, memory_format = torch.contiguous_format);  permute_748 = None
    view_902: "f32[512, 196, 32]" = torch.ops.aten.view.default(clone_227, [512, 196, 32]);  clone_227 = None
    bmm_140: "f32[512, 196, 32]" = torch.ops.aten.bmm.default(permute_749, view_902);  permute_749 = None
    bmm_141: "f32[512, 196, 196]" = torch.ops.aten.bmm.default(view_902, permute_750);  view_902 = permute_750 = None
    view_903: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.view.default(bmm_140, [8, 4, 16, 196, 32]);  bmm_140 = None
    view_904: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.view.default(bmm_141, [8, 4, 16, 196, 196]);  bmm_141 = None
    mul_924: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.mul.Tensor(view_904, alias_47);  view_904 = None
    sum_348: "f32[8, 4, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_924, [-1], True)
    mul_925: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.mul.Tensor(alias_47, sum_348);  alias_47 = sum_348 = None
    sub_248: "f32[8, 4, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_924, mul_925);  mul_924 = mul_925 = None
    view_905: "f32[512, 196, 196]" = torch.ops.aten.view.default(sub_248, [512, 196, 196]);  sub_248 = None
    bmm_142: "f32[512, 32, 196]" = torch.ops.aten.bmm.default(permute_751, view_905);  permute_751 = None
    bmm_143: "f32[512, 196, 32]" = torch.ops.aten.bmm.default(view_905, permute_752);  view_905 = permute_752 = None
    view_906: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.view.default(bmm_142, [8, 4, 16, 32, 196]);  bmm_142 = None
    view_907: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.view.default(bmm_143, [8, 4, 16, 196, 32]);  bmm_143 = None
    mul_926: "f32[8, 4, 16, 32, 196]" = torch.ops.aten.mul.Scalar(view_906, 0.42044820762685725);  view_906 = None
    permute_753: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.permute.default(mul_926, [0, 1, 2, 4, 3]);  mul_926 = None
    mul_927: "f32[8, 4, 16, 196, 32]" = torch.ops.aten.mul.Scalar(view_907, 0.42044820762685725);  view_907 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:66, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_23: "f32[24, 4, 16, 196, 32]" = torch.ops.aten.cat.default([mul_927, permute_753, view_903]);  mul_927 = permute_753 = view_903 = None
    view_908: "f32[3, 8, 4, 16, 196, 32]" = torch.ops.aten.view.default(cat_23, [3, 8, 4, 16, 196, 32]);  cat_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:65, code: qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
    permute_754: "f32[8, 16, 196, 3, 4, 32]" = torch.ops.aten.permute.default(view_908, [1, 3, 4, 0, 2, 5]);  view_908 = None
    clone_228: "f32[8, 16, 196, 3, 4, 32]" = torch.ops.aten.clone.default(permute_754, memory_format = torch.contiguous_format);  permute_754 = None
    view_909: "f32[8, 16, 196, 384]" = torch.ops.aten.view.default(clone_228, [8, 16, 196, 384]);  clone_228 = None
    view_910: "f32[25088, 384]" = torch.ops.aten.view.default(view_909, [25088, 384]);  view_909 = None
    mm_192: "f32[25088, 128]" = torch.ops.aten.mm.default(view_910, permute_755);  permute_755 = None
    permute_756: "f32[384, 25088]" = torch.ops.aten.permute.default(view_910, [1, 0])
    mm_193: "f32[384, 128]" = torch.ops.aten.mm.default(permute_756, view_2);  permute_756 = view_2 = None
    permute_757: "f32[128, 384]" = torch.ops.aten.permute.default(mm_193, [1, 0]);  mm_193 = None
    sum_349: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_910, [0], True);  view_910 = None
    view_911: "f32[384]" = torch.ops.aten.view.default(sum_349, [384]);  sum_349 = None
    permute_758: "f32[384, 128]" = torch.ops.aten.permute.default(permute_757, [1, 0]);  permute_757 = None
    view_912: "f32[8, 16, 196, 128]" = torch.ops.aten.view.default(mm_192, [8, 16, 196, 128]);  mm_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_929: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(view_912, primals_2);  primals_2 = None
    mul_930: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_929, 128)
    sum_350: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_929, [3], True)
    mul_931: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul_929, mul);  mul_929 = None
    sum_351: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_931, [3], True);  mul_931 = None
    mul_932: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(mul, sum_351);  sum_351 = None
    sub_250: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(mul_930, sum_350);  mul_930 = sum_350 = None
    sub_251: "f32[8, 16, 196, 128]" = torch.ops.aten.sub.Tensor(sub_250, mul_932);  sub_250 = mul_932 = None
    mul_933: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(div_121, sub_251);  div_121 = sub_251 = None
    mul_934: "f32[8, 16, 196, 128]" = torch.ops.aten.mul.Tensor(view_912, mul);  mul = None
    sum_352: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_934, [0, 1, 2]);  mul_934 = None
    sum_353: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_912, [0, 1, 2]);  view_912 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_272: "f32[8, 16, 196, 128]" = torch.ops.aten.add.Tensor(add_271, mul_933);  add_271 = mul_933 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:235, code: x = x + self.pos_embed
    sum_354: "f32[1, 16, 196, 128]" = torch.ops.aten.sum.dim_IntList(add_272, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:160, code: x = x.transpose(2, 3).reshape(B, grid_height * grid_width, -1, C)
    view_913: "f32[8, 4, 4, 14, 14, 128]" = torch.ops.aten.view.default(add_272, [8, 4, 4, 14, 14, 128]);  add_272 = None
    permute_759: "f32[8, 4, 14, 4, 14, 128]" = torch.ops.aten.permute.default(view_913, [0, 1, 3, 2, 4, 5]);  view_913 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:159, code: x = x.reshape(B, grid_height, block_size, grid_width, block_size, C)
    clone_229: "f32[8, 4, 14, 4, 14, 128]" = torch.ops.aten.clone.default(permute_759, memory_format = torch.contiguous_format);  permute_759 = None
    view_914: "f32[8, 56, 56, 128]" = torch.ops.aten.view.default(clone_229, [8, 56, 56, 128]);  clone_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/nest.py:233, code: x = x.permute(0, 2, 3, 1)  # (B, H', W', C), switch to channels last for transformer
    permute_760: "f32[8, 128, 56, 56]" = torch.ops.aten.permute.default(view_914, [0, 3, 1, 2]);  view_914 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    sum_355: "f32[128]" = torch.ops.aten.sum.dim_IntList(permute_760, [0, 2, 3])
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(permute_760, primals_306, primals_106, [128], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  permute_760 = primals_306 = primals_106 = None
    getitem_185: "f32[128, 3, 4, 4]" = convolution_backward_2[1];  convolution_backward_2 = None
    return [sum_354, sum_352, sum_353, sum_345, sum_346, sum_339, sum_340, sum_332, sum_333, sum_325, sum_326, sum_322, sum_320, sum_321, sum_313, sum_314, sum_307, sum_308, sum_300, sum_301, sum_293, sum_294, sum_290, sum_288, sum_289, sum_281, sum_282, sum_275, sum_276, sum_268, sum_269, sum_262, sum_263, sum_255, sum_256, sum_249, sum_250, sum_242, sum_243, sum_236, sum_237, sum_229, sum_230, sum_223, sum_224, sum_216, sum_217, sum_210, sum_211, sum_203, sum_204, sum_197, sum_198, sum_190, sum_191, sum_184, sum_185, sum_177, sum_178, sum_171, sum_172, sum_164, sum_165, sum_158, sum_159, sum_151, sum_152, sum_145, sum_146, sum_138, sum_139, sum_132, sum_133, sum_125, sum_126, sum_119, sum_120, sum_112, sum_113, sum_106, sum_107, sum_99, sum_100, sum_93, sum_94, sum_86, sum_87, sum_80, sum_81, sum_73, sum_74, sum_67, sum_68, sum_60, sum_61, sum_54, sum_55, sum_47, sum_48, sum_41, sum_42, sum_34, sum_35, sum_28, sum_29, getitem_185, sum_355, permute_758, view_911, permute_747, view_899, permute_743, view_896, permute_739, view_893, permute_735, view_890, permute_724, view_878, permute_720, view_875, permute_716, view_872, getitem_182, sum_327, permute_706, view_865, permute_695, view_853, permute_691, view_850, permute_687, view_847, permute_683, view_844, permute_672, view_832, permute_668, view_829, permute_664, view_826, getitem_179, sum_295, permute_654, view_819, permute_643, view_807, permute_639, view_804, permute_635, view_801, permute_631, view_798, permute_620, view_786, permute_616, view_783, permute_612, view_780, permute_608, view_777, permute_597, view_765, permute_593, view_762, permute_589, view_759, permute_585, view_756, permute_574, view_744, permute_570, view_741, permute_566, view_738, permute_562, view_735, permute_551, view_723, permute_547, view_720, permute_543, view_717, permute_539, view_714, permute_528, view_702, permute_524, view_699, permute_520, view_696, permute_516, view_693, permute_505, view_681, permute_501, view_678, permute_497, view_675, permute_493, view_672, permute_482, view_660, permute_478, view_657, permute_474, view_654, permute_470, view_651, permute_459, view_639, permute_455, view_636, permute_451, view_633, permute_447, view_630, permute_436, view_618, permute_432, view_615, permute_428, view_612, permute_424, view_609, permute_413, view_597, permute_409, view_594, permute_405, view_591, permute_401, view_588, permute_390, view_576, permute_386, view_573, permute_382, view_570, permute_378, view_567, permute_367, view_555, permute_363, view_552, permute_359, view_549, permute_355, view_546, permute_344, view_534, permute_340, view_531, permute_336, view_528, permute_332, view_525, permute_321, view_513, permute_317, view_510, permute_313, view_507, permute_309, view_504, permute_298, view_492, permute_294, view_489, permute_290, view_486, permute_286, view_483, permute_275, view_471, permute_271, view_468, permute_267, view_465, permute_263, view_462, permute_252, view_450, permute_248, view_447, permute_244, view_444, permute_240, view_441, permute_229, view_429, permute_225, view_426, permute_221, view_423, permute_217, view_420, permute_206, view_408, permute_202, view_405, permute_198, view_402, permute_190, view_397, None]
    