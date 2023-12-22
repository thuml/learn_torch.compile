from __future__ import annotations



def forward(self, primals_25: "f32[128, 3, 4, 4]", primals_27: "f32[128]", primals_29: "f32[128]", primals_35: "f32[128]", primals_41: "f32[128]", primals_47: "f32[128]", primals_53: "f32[512]", primals_56: "f32[256]", primals_62: "f32[256]", primals_68: "f32[256]", primals_74: "f32[256]", primals_80: "f32[1024]", primals_83: "f32[512]", primals_89: "f32[512]", primals_95: "f32[512]", primals_101: "f32[512]", primals_107: "f32[512]", primals_113: "f32[512]", primals_119: "f32[512]", primals_125: "f32[512]", primals_131: "f32[512]", primals_137: "f32[512]", primals_143: "f32[512]", primals_149: "f32[512]", primals_155: "f32[512]", primals_161: "f32[512]", primals_167: "f32[512]", primals_173: "f32[512]", primals_179: "f32[512]", primals_185: "f32[512]", primals_191: "f32[512]", primals_197: "f32[512]", primals_203: "f32[512]", primals_209: "f32[512]", primals_215: "f32[512]", primals_221: "f32[512]", primals_227: "f32[512]", primals_233: "f32[512]", primals_239: "f32[512]", primals_245: "f32[512]", primals_251: "f32[512]", primals_257: "f32[512]", primals_263: "f32[512]", primals_269: "f32[512]", primals_275: "f32[512]", primals_281: "f32[512]", primals_287: "f32[512]", primals_293: "f32[512]", primals_299: "f32[2048]", primals_302: "f32[1024]", primals_308: "f32[1024]", primals_314: "f32[1024]", primals_320: "f32[1024]", primals_326: "f32[1024]", primals_365: "f32[8, 3, 224, 224]", mul: "f32[8, 56, 56, 128]", mul_2: "f32[8, 56, 56, 128]", view_3: "f32[25088, 128]", view_9: "i64[2401]", view_15: "f32[25088, 128]", mul_5: "f32[8, 3136, 128]", view_21: "f32[25088, 128]", addmm_2: "f32[25088, 512]", view_23: "f32[25088, 512]", mul_10: "f32[8, 56, 56, 128]", view_29: "f32[25088, 128]", view_35: "i64[2401]", view_43: "f32[25088, 128]", bernoulli: "f32[8, 1, 1, 1]", mul_14: "f32[8, 3136, 128]", view_49: "f32[25088, 128]", addmm_6: "f32[25088, 512]", view_51: "f32[25088, 512]", bernoulli_1: "f32[8, 1, 1]", mul_20: "f32[8, 28, 28, 512]", view_56: "f32[6272, 512]", mm: "f32[6272, 256]", getitem_19: "f32[8, 28, 28, 1]", rsqrt_6: "f32[8, 28, 28, 1]", view_61: "f32[6272, 256]", view_67: "i64[2401]", view_73: "f32[6272, 256]", bernoulli_2: "f32[8, 1, 1, 1]", mul_26: "f32[8, 784, 256]", view_79: "f32[6272, 256]", addmm_10: "f32[6272, 1024]", view_81: "f32[6272, 1024]", bernoulli_3: "f32[8, 1, 1]", mul_32: "f32[8, 28, 28, 256]", view_87: "f32[6272, 256]", view_93: "i64[2401]", view_101: "f32[6272, 256]", bernoulli_4: "f32[8, 1, 1, 1]", mul_36: "f32[8, 784, 256]", view_107: "f32[6272, 256]", addmm_14: "f32[6272, 1024]", view_109: "f32[6272, 1024]", bernoulli_5: "f32[8, 1, 1]", mul_42: "f32[8, 14, 14, 1024]", view_114: "f32[1568, 1024]", mm_1: "f32[1568, 512]", getitem_35: "f32[8, 14, 14, 1]", rsqrt_11: "f32[8, 14, 14, 1]", view_119: "f32[1568, 512]", view_125: "i64[2401]", view_131: "f32[1568, 512]", bernoulli_6: "f32[8, 1, 1, 1]", mul_48: "f32[8, 196, 512]", view_137: "f32[1568, 512]", addmm_18: "f32[1568, 2048]", view_139: "f32[1568, 2048]", bernoulli_7: "f32[8, 1, 1]", mul_54: "f32[8, 14, 14, 512]", view_145: "f32[1568, 512]", view_151: "i64[2401]", view_159: "f32[1568, 512]", bernoulli_8: "f32[8, 1, 1, 1]", mul_58: "f32[8, 196, 512]", view_165: "f32[1568, 512]", addmm_22: "f32[1568, 2048]", view_167: "f32[1568, 2048]", bernoulli_9: "f32[8, 1, 1]", mul_64: "f32[8, 14, 14, 512]", view_173: "f32[1568, 512]", view_179: "i64[2401]", view_185: "f32[1568, 512]", bernoulli_10: "f32[8, 1, 1, 1]", mul_68: "f32[8, 196, 512]", view_191: "f32[1568, 512]", addmm_26: "f32[1568, 2048]", view_193: "f32[1568, 2048]", bernoulli_11: "f32[8, 1, 1]", mul_74: "f32[8, 14, 14, 512]", view_199: "f32[1568, 512]", view_205: "i64[2401]", view_213: "f32[1568, 512]", bernoulli_12: "f32[8, 1, 1, 1]", mul_78: "f32[8, 196, 512]", view_219: "f32[1568, 512]", addmm_30: "f32[1568, 2048]", view_221: "f32[1568, 2048]", bernoulli_13: "f32[8, 1, 1]", mul_84: "f32[8, 14, 14, 512]", view_227: "f32[1568, 512]", view_233: "i64[2401]", view_239: "f32[1568, 512]", bernoulli_14: "f32[8, 1, 1, 1]", mul_88: "f32[8, 196, 512]", view_245: "f32[1568, 512]", addmm_34: "f32[1568, 2048]", view_247: "f32[1568, 2048]", bernoulli_15: "f32[8, 1, 1]", mul_94: "f32[8, 14, 14, 512]", view_253: "f32[1568, 512]", view_259: "i64[2401]", view_267: "f32[1568, 512]", bernoulli_16: "f32[8, 1, 1, 1]", mul_98: "f32[8, 196, 512]", view_273: "f32[1568, 512]", addmm_38: "f32[1568, 2048]", view_275: "f32[1568, 2048]", bernoulli_17: "f32[8, 1, 1]", mul_104: "f32[8, 14, 14, 512]", view_281: "f32[1568, 512]", view_287: "i64[2401]", view_293: "f32[1568, 512]", bernoulli_18: "f32[8, 1, 1, 1]", mul_108: "f32[8, 196, 512]", view_299: "f32[1568, 512]", addmm_42: "f32[1568, 2048]", view_301: "f32[1568, 2048]", bernoulli_19: "f32[8, 1, 1]", mul_114: "f32[8, 14, 14, 512]", view_307: "f32[1568, 512]", view_313: "i64[2401]", view_321: "f32[1568, 512]", bernoulli_20: "f32[8, 1, 1, 1]", mul_118: "f32[8, 196, 512]", view_327: "f32[1568, 512]", addmm_46: "f32[1568, 2048]", view_329: "f32[1568, 2048]", bernoulli_21: "f32[8, 1, 1]", mul_124: "f32[8, 14, 14, 512]", view_335: "f32[1568, 512]", view_341: "i64[2401]", view_347: "f32[1568, 512]", bernoulli_22: "f32[8, 1, 1, 1]", mul_128: "f32[8, 196, 512]", view_353: "f32[1568, 512]", addmm_50: "f32[1568, 2048]", view_355: "f32[1568, 2048]", bernoulli_23: "f32[8, 1, 1]", mul_134: "f32[8, 14, 14, 512]", view_361: "f32[1568, 512]", view_367: "i64[2401]", view_375: "f32[1568, 512]", bernoulli_24: "f32[8, 1, 1, 1]", mul_138: "f32[8, 196, 512]", view_381: "f32[1568, 512]", addmm_54: "f32[1568, 2048]", view_383: "f32[1568, 2048]", bernoulli_25: "f32[8, 1, 1]", mul_144: "f32[8, 14, 14, 512]", view_389: "f32[1568, 512]", view_395: "i64[2401]", view_401: "f32[1568, 512]", bernoulli_26: "f32[8, 1, 1, 1]", mul_148: "f32[8, 196, 512]", view_407: "f32[1568, 512]", addmm_58: "f32[1568, 2048]", view_409: "f32[1568, 2048]", bernoulli_27: "f32[8, 1, 1]", mul_154: "f32[8, 14, 14, 512]", view_415: "f32[1568, 512]", view_421: "i64[2401]", view_429: "f32[1568, 512]", bernoulli_28: "f32[8, 1, 1, 1]", mul_158: "f32[8, 196, 512]", view_435: "f32[1568, 512]", addmm_62: "f32[1568, 2048]", view_437: "f32[1568, 2048]", bernoulli_29: "f32[8, 1, 1]", mul_164: "f32[8, 14, 14, 512]", view_443: "f32[1568, 512]", view_449: "i64[2401]", view_455: "f32[1568, 512]", bernoulli_30: "f32[8, 1, 1, 1]", mul_168: "f32[8, 196, 512]", view_461: "f32[1568, 512]", addmm_66: "f32[1568, 2048]", view_463: "f32[1568, 2048]", bernoulli_31: "f32[8, 1, 1]", mul_174: "f32[8, 14, 14, 512]", view_469: "f32[1568, 512]", view_475: "i64[2401]", view_483: "f32[1568, 512]", bernoulli_32: "f32[8, 1, 1, 1]", mul_178: "f32[8, 196, 512]", view_489: "f32[1568, 512]", addmm_70: "f32[1568, 2048]", view_491: "f32[1568, 2048]", bernoulli_33: "f32[8, 1, 1]", mul_184: "f32[8, 14, 14, 512]", view_497: "f32[1568, 512]", view_503: "i64[2401]", view_509: "f32[1568, 512]", bernoulli_34: "f32[8, 1, 1, 1]", mul_188: "f32[8, 196, 512]", view_515: "f32[1568, 512]", addmm_74: "f32[1568, 2048]", view_517: "f32[1568, 2048]", bernoulli_35: "f32[8, 1, 1]", mul_194: "f32[8, 14, 14, 512]", view_523: "f32[1568, 512]", view_529: "i64[2401]", view_537: "f32[1568, 512]", bernoulli_36: "f32[8, 1, 1, 1]", mul_198: "f32[8, 196, 512]", view_543: "f32[1568, 512]", addmm_78: "f32[1568, 2048]", view_545: "f32[1568, 2048]", bernoulli_37: "f32[8, 1, 1]", mul_204: "f32[8, 14, 14, 512]", view_551: "f32[1568, 512]", view_557: "i64[2401]", view_563: "f32[1568, 512]", bernoulli_38: "f32[8, 1, 1, 1]", mul_208: "f32[8, 196, 512]", view_569: "f32[1568, 512]", addmm_82: "f32[1568, 2048]", view_571: "f32[1568, 2048]", bernoulli_39: "f32[8, 1, 1]", mul_214: "f32[8, 14, 14, 512]", view_577: "f32[1568, 512]", view_583: "i64[2401]", view_591: "f32[1568, 512]", bernoulli_40: "f32[8, 1, 1, 1]", mul_218: "f32[8, 196, 512]", view_597: "f32[1568, 512]", addmm_86: "f32[1568, 2048]", view_599: "f32[1568, 2048]", bernoulli_41: "f32[8, 1, 1]", mul_224: "f32[8, 7, 7, 2048]", view_604: "f32[392, 2048]", mm_2: "f32[392, 1024]", getitem_163: "f32[8, 7, 7, 1]", rsqrt_48: "f32[8, 7, 7, 1]", view_609: "f32[392, 1024]", view_615: "i64[2401]", view_621: "f32[392, 1024]", bernoulli_42: "f32[8, 1, 1, 1]", mul_230: "f32[8, 49, 1024]", view_627: "f32[392, 1024]", addmm_90: "f32[392, 4096]", view_629: "f32[392, 4096]", bernoulli_43: "f32[8, 1, 1]", mul_236: "f32[8, 7, 7, 1024]", view_635: "f32[392, 1024]", view_641: "i64[2401]", view_647: "f32[392, 1024]", bernoulli_44: "f32[8, 1, 1, 1]", mul_240: "f32[8, 49, 1024]", view_653: "f32[392, 1024]", addmm_94: "f32[392, 4096]", view_655: "f32[392, 4096]", bernoulli_45: "f32[8, 1, 1]", mul_246: "f32[8, 7, 7, 1024]", clone_264: "f32[8, 1024]", permute_248: "f32[1000, 1024]", div_71: "f32[8, 7, 7, 1]", permute_252: "f32[1024, 4096]", permute_256: "f32[4096, 1024]", div_72: "f32[8, 49, 1]", permute_261: "f32[1024, 1024]", permute_266: "f32[256, 49, 49]", permute_267: "f32[256, 32, 49]", alias_24: "f32[8, 32, 49, 49]", permute_269: "f32[256, 32, 49]", permute_270: "f32[256, 49, 32]", permute_273: "f32[3072, 1024]", div_73: "f32[8, 7, 7, 1]", permute_278: "f32[1024, 4096]", permute_282: "f32[4096, 1024]", div_74: "f32[8, 49, 1]", permute_287: "f32[1024, 1024]", permute_292: "f32[256, 49, 49]", permute_293: "f32[256, 32, 49]", alias_25: "f32[8, 32, 49, 49]", permute_295: "f32[256, 32, 49]", permute_296: "f32[256, 49, 32]", permute_299: "f32[3072, 1024]", permute_306: "f32[1024, 2048]", div_76: "f32[8, 7, 7, 1]", permute_309: "f32[512, 2048]", permute_313: "f32[2048, 512]", div_77: "f32[8, 196, 1]", permute_318: "f32[512, 512]", permute_323: "f32[512, 49, 49]", permute_324: "f32[512, 32, 49]", alias_26: "f32[32, 16, 49, 49]", permute_326: "f32[512, 32, 49]", permute_327: "f32[512, 49, 32]", permute_330: "f32[1536, 512]", div_78: "f32[8, 14, 14, 1]", permute_335: "f32[512, 2048]", permute_339: "f32[2048, 512]", div_79: "f32[8, 196, 1]", permute_344: "f32[512, 512]", permute_349: "f32[512, 49, 49]", permute_350: "f32[512, 32, 49]", alias_27: "f32[32, 16, 49, 49]", permute_352: "f32[512, 32, 49]", permute_353: "f32[512, 49, 32]", permute_356: "f32[1536, 512]", div_80: "f32[8, 14, 14, 1]", permute_361: "f32[512, 2048]", permute_365: "f32[2048, 512]", div_81: "f32[8, 196, 1]", permute_370: "f32[512, 512]", permute_375: "f32[512, 49, 49]", permute_376: "f32[512, 32, 49]", alias_28: "f32[32, 16, 49, 49]", permute_378: "f32[512, 32, 49]", permute_379: "f32[512, 49, 32]", permute_382: "f32[1536, 512]", div_82: "f32[8, 14, 14, 1]", permute_387: "f32[512, 2048]", permute_391: "f32[2048, 512]", div_83: "f32[8, 196, 1]", permute_396: "f32[512, 512]", permute_401: "f32[512, 49, 49]", permute_402: "f32[512, 32, 49]", alias_29: "f32[32, 16, 49, 49]", permute_404: "f32[512, 32, 49]", permute_405: "f32[512, 49, 32]", permute_408: "f32[1536, 512]", div_84: "f32[8, 14, 14, 1]", permute_413: "f32[512, 2048]", permute_417: "f32[2048, 512]", div_85: "f32[8, 196, 1]", permute_422: "f32[512, 512]", permute_427: "f32[512, 49, 49]", permute_428: "f32[512, 32, 49]", alias_30: "f32[32, 16, 49, 49]", permute_430: "f32[512, 32, 49]", permute_431: "f32[512, 49, 32]", permute_434: "f32[1536, 512]", div_86: "f32[8, 14, 14, 1]", permute_439: "f32[512, 2048]", permute_443: "f32[2048, 512]", div_87: "f32[8, 196, 1]", permute_448: "f32[512, 512]", permute_453: "f32[512, 49, 49]", permute_454: "f32[512, 32, 49]", alias_31: "f32[32, 16, 49, 49]", permute_456: "f32[512, 32, 49]", permute_457: "f32[512, 49, 32]", permute_460: "f32[1536, 512]", div_88: "f32[8, 14, 14, 1]", permute_465: "f32[512, 2048]", permute_469: "f32[2048, 512]", div_89: "f32[8, 196, 1]", permute_474: "f32[512, 512]", permute_479: "f32[512, 49, 49]", permute_480: "f32[512, 32, 49]", alias_32: "f32[32, 16, 49, 49]", permute_482: "f32[512, 32, 49]", permute_483: "f32[512, 49, 32]", permute_486: "f32[1536, 512]", div_90: "f32[8, 14, 14, 1]", permute_491: "f32[512, 2048]", permute_495: "f32[2048, 512]", div_91: "f32[8, 196, 1]", permute_500: "f32[512, 512]", permute_505: "f32[512, 49, 49]", permute_506: "f32[512, 32, 49]", alias_33: "f32[32, 16, 49, 49]", permute_508: "f32[512, 32, 49]", permute_509: "f32[512, 49, 32]", permute_512: "f32[1536, 512]", div_92: "f32[8, 14, 14, 1]", permute_517: "f32[512, 2048]", permute_521: "f32[2048, 512]", div_93: "f32[8, 196, 1]", permute_526: "f32[512, 512]", permute_531: "f32[512, 49, 49]", permute_532: "f32[512, 32, 49]", alias_34: "f32[32, 16, 49, 49]", permute_534: "f32[512, 32, 49]", permute_535: "f32[512, 49, 32]", permute_538: "f32[1536, 512]", div_94: "f32[8, 14, 14, 1]", permute_543: "f32[512, 2048]", permute_547: "f32[2048, 512]", div_95: "f32[8, 196, 1]", permute_552: "f32[512, 512]", permute_557: "f32[512, 49, 49]", permute_558: "f32[512, 32, 49]", alias_35: "f32[32, 16, 49, 49]", permute_560: "f32[512, 32, 49]", permute_561: "f32[512, 49, 32]", permute_564: "f32[1536, 512]", div_96: "f32[8, 14, 14, 1]", permute_569: "f32[512, 2048]", permute_573: "f32[2048, 512]", div_97: "f32[8, 196, 1]", permute_578: "f32[512, 512]", permute_583: "f32[512, 49, 49]", permute_584: "f32[512, 32, 49]", alias_36: "f32[32, 16, 49, 49]", permute_586: "f32[512, 32, 49]", permute_587: "f32[512, 49, 32]", permute_590: "f32[1536, 512]", div_98: "f32[8, 14, 14, 1]", permute_595: "f32[512, 2048]", permute_599: "f32[2048, 512]", div_99: "f32[8, 196, 1]", permute_604: "f32[512, 512]", permute_609: "f32[512, 49, 49]", permute_610: "f32[512, 32, 49]", alias_37: "f32[32, 16, 49, 49]", permute_612: "f32[512, 32, 49]", permute_613: "f32[512, 49, 32]", permute_616: "f32[1536, 512]", div_100: "f32[8, 14, 14, 1]", permute_621: "f32[512, 2048]", permute_625: "f32[2048, 512]", div_101: "f32[8, 196, 1]", permute_630: "f32[512, 512]", permute_635: "f32[512, 49, 49]", permute_636: "f32[512, 32, 49]", alias_38: "f32[32, 16, 49, 49]", permute_638: "f32[512, 32, 49]", permute_639: "f32[512, 49, 32]", permute_642: "f32[1536, 512]", div_102: "f32[8, 14, 14, 1]", permute_647: "f32[512, 2048]", permute_651: "f32[2048, 512]", div_103: "f32[8, 196, 1]", permute_656: "f32[512, 512]", permute_661: "f32[512, 49, 49]", permute_662: "f32[512, 32, 49]", alias_39: "f32[32, 16, 49, 49]", permute_664: "f32[512, 32, 49]", permute_665: "f32[512, 49, 32]", permute_668: "f32[1536, 512]", div_104: "f32[8, 14, 14, 1]", permute_673: "f32[512, 2048]", permute_677: "f32[2048, 512]", div_105: "f32[8, 196, 1]", permute_682: "f32[512, 512]", permute_687: "f32[512, 49, 49]", permute_688: "f32[512, 32, 49]", alias_40: "f32[32, 16, 49, 49]", permute_690: "f32[512, 32, 49]", permute_691: "f32[512, 49, 32]", permute_694: "f32[1536, 512]", div_106: "f32[8, 14, 14, 1]", permute_699: "f32[512, 2048]", permute_703: "f32[2048, 512]", div_107: "f32[8, 196, 1]", permute_708: "f32[512, 512]", permute_713: "f32[512, 49, 49]", permute_714: "f32[512, 32, 49]", alias_41: "f32[32, 16, 49, 49]", permute_716: "f32[512, 32, 49]", permute_717: "f32[512, 49, 32]", permute_720: "f32[1536, 512]", div_108: "f32[8, 14, 14, 1]", permute_725: "f32[512, 2048]", permute_729: "f32[2048, 512]", div_109: "f32[8, 196, 1]", permute_734: "f32[512, 512]", permute_739: "f32[512, 49, 49]", permute_740: "f32[512, 32, 49]", alias_42: "f32[32, 16, 49, 49]", permute_742: "f32[512, 32, 49]", permute_743: "f32[512, 49, 32]", permute_746: "f32[1536, 512]", div_110: "f32[8, 14, 14, 1]", permute_751: "f32[512, 2048]", permute_755: "f32[2048, 512]", div_111: "f32[8, 196, 1]", permute_760: "f32[512, 512]", permute_765: "f32[512, 49, 49]", permute_766: "f32[512, 32, 49]", alias_43: "f32[32, 16, 49, 49]", permute_768: "f32[512, 32, 49]", permute_769: "f32[512, 49, 32]", permute_772: "f32[1536, 512]", permute_779: "f32[512, 1024]", div_113: "f32[8, 14, 14, 1]", permute_782: "f32[256, 1024]", permute_786: "f32[1024, 256]", div_114: "f32[8, 784, 1]", permute_791: "f32[256, 256]", permute_796: "f32[1024, 49, 49]", permute_797: "f32[1024, 32, 49]", alias_44: "f32[128, 8, 49, 49]", permute_799: "f32[1024, 32, 49]", permute_800: "f32[1024, 49, 32]", permute_803: "f32[768, 256]", div_115: "f32[8, 28, 28, 1]", permute_808: "f32[256, 1024]", permute_812: "f32[1024, 256]", div_116: "f32[8, 784, 1]", permute_817: "f32[256, 256]", permute_822: "f32[1024, 49, 49]", permute_823: "f32[1024, 32, 49]", alias_45: "f32[128, 8, 49, 49]", permute_825: "f32[1024, 32, 49]", permute_826: "f32[1024, 49, 32]", permute_829: "f32[768, 256]", permute_836: "f32[256, 512]", div_118: "f32[8, 28, 28, 1]", permute_839: "f32[128, 512]", permute_843: "f32[512, 128]", div_119: "f32[8, 3136, 1]", permute_848: "f32[128, 128]", permute_853: "f32[2048, 49, 49]", permute_854: "f32[2048, 32, 49]", alias_46: "f32[512, 4, 49, 49]", permute_856: "f32[2048, 32, 49]", permute_857: "f32[2048, 49, 32]", permute_860: "f32[384, 128]", div_120: "f32[8, 56, 56, 1]", permute_865: "f32[128, 512]", permute_869: "f32[512, 128]", div_121: "f32[8, 3136, 1]", permute_874: "f32[128, 128]", permute_879: "f32[2048, 49, 49]", permute_880: "f32[2048, 32, 49]", alias_47: "f32[512, 4, 49, 49]", permute_882: "f32[2048, 32, 49]", permute_883: "f32[2048, 49, 32]", permute_886: "f32[384, 128]", div_122: "f32[8, 56, 56, 1]", div_123: "f32[8, 56, 56, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_22: "f32[8, 3136, 512]" = torch.ops.aten.reshape.default(addmm_2, [8, 3136, 512]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_8: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_22, 0.7071067811865476)
    erf: "f32[8, 3136, 512]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
    add_8: "f32[8, 3136, 512]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_2: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli, 0.9956521736457944);  bernoulli = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_50: "f32[8, 3136, 512]" = torch.ops.aten.reshape.default(addmm_6, [8, 3136, 512]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_17: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_50, 0.7071067811865476)
    erf_1: "f32[8, 3136, 512]" = torch.ops.aten.erf.default(mul_17);  mul_17 = None
    add_17: "f32[8, 3136, 512]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_3: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_1, 0.9956521736457944);  bernoulli_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:358, code: x = self.reduction(x)
    view_57: "f32[8, 28, 28, 256]" = torch.ops.aten.reshape.default(mm, [8, 28, 28, 256]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    sub_8: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(view_57, getitem_19);  view_57 = getitem_19 = None
    mul_22: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_6);  sub_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_5: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_2, 0.9913043472915888);  bernoulli_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_80: "f32[8, 784, 1024]" = torch.ops.aten.reshape.default(addmm_10, [8, 784, 1024]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_29: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_80, 0.7071067811865476)
    erf_2: "f32[8, 784, 1024]" = torch.ops.aten.erf.default(mul_29);  mul_29 = None
    add_27: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_6: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_3, 0.9913043472915888);  bernoulli_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_8: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_4, 0.9869565209373832);  bernoulli_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_108: "f32[8, 784, 1024]" = torch.ops.aten.reshape.default(addmm_14, [8, 784, 1024]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_39: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_108, 0.7071067811865476)
    erf_3: "f32[8, 784, 1024]" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
    add_36: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_9: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_5, 0.9869565209373832);  bernoulli_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:358, code: x = self.reduction(x)
    view_115: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(mm_1, [8, 14, 14, 512]);  mm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    sub_15: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_115, getitem_35);  view_115 = getitem_35 = None
    mul_44: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_11);  sub_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_11: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_6, 0.9826086945831776);  bernoulli_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_138: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(addmm_18, [8, 196, 2048]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_51: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_138, 0.7071067811865476)
    erf_4: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
    add_46: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_12: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_7, 0.9826086945831776);  bernoulli_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_14: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_8, 0.9782608672976494);  bernoulli_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_166: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(addmm_22, [8, 196, 2048]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_61: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_166, 0.7071067811865476)
    erf_5: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
    add_55: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_15: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_9, 0.9782608672976494);  bernoulli_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_17: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_10, 0.9739130418747663);  bernoulli_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_192: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(addmm_26, [8, 196, 2048]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_71: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_192, 0.7071067811865476)
    erf_6: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_71);  mul_71 = None
    add_63: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_18: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_11, 0.9739130418747663);  bernoulli_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_20: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_12, 0.9695652164518833);  bernoulli_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_220: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(addmm_30, [8, 196, 2048]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_81: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_220, 0.7071067811865476)
    erf_7: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_81);  mul_81 = None
    add_72: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_21: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_13, 0.9695652164518833);  bernoulli_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_23: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_14, 0.9652173891663551);  bernoulli_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_246: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(addmm_34, [8, 196, 2048]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_91: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_246, 0.7071067811865476)
    erf_8: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_91);  mul_91 = None
    add_80: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_24: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_15, 0.9652173891663551);  bernoulli_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_26: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_16, 0.960869561880827);  bernoulli_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_274: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(addmm_38, [8, 196, 2048]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_101: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_274, 0.7071067811865476)
    erf_9: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_101);  mul_101 = None
    add_89: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_27: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_17, 0.960869561880827);  bernoulli_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_29: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_18, 0.9565217345952988);  bernoulli_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_300: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(addmm_42, [8, 196, 2048]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_111: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_300, 0.7071067811865476)
    erf_10: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_111);  mul_111 = None
    add_97: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_30: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_19, 0.9565217345952988);  bernoulli_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_32: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_20, 0.9521739110350609);  bernoulli_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_328: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(addmm_46, [8, 196, 2048]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_121: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_328, 0.7071067811865476)
    erf_11: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_121);  mul_121 = None
    add_106: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_33: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_21, 0.9521739110350609);  bernoulli_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_35: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_22, 0.947826087474823);  bernoulli_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_354: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(addmm_50, [8, 196, 2048]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_131: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_354, 0.7071067811865476)
    erf_12: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_131);  mul_131 = None
    add_114: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_36: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_23, 0.947826087474823);  bernoulli_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_38: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_24, 0.9434782639145851);  bernoulli_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_382: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(addmm_54, [8, 196, 2048]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_141: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_382, 0.7071067811865476)
    erf_13: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_141);  mul_141 = None
    add_123: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_39: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_25, 0.9434782639145851);  bernoulli_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_41: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_26, 0.9391304366290569);  bernoulli_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_408: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(addmm_58, [8, 196, 2048]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_151: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_408, 0.7071067811865476)
    erf_14: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_151);  mul_151 = None
    add_131: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_42: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_27, 0.9391304366290569);  bernoulli_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_44: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_28, 0.9347826093435287);  bernoulli_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_436: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(addmm_62, [8, 196, 2048]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_161: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_436, 0.7071067811865476)
    erf_15: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_161);  mul_161 = None
    add_140: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_45: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_29, 0.9347826093435287);  bernoulli_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_47: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_30, 0.9304347857832909);  bernoulli_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_462: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(addmm_66, [8, 196, 2048]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_171: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_462, 0.7071067811865476)
    erf_16: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_171);  mul_171 = None
    add_148: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_48: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_31, 0.9304347857832909);  bernoulli_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_50: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_32, 0.9260869547724724);  bernoulli_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_490: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(addmm_70, [8, 196, 2048]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_181: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_490, 0.7071067811865476)
    erf_17: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_181);  mul_181 = None
    add_157: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_51: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_33, 0.9260869547724724);  bernoulli_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_53: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_34, 0.9217391312122345);  bernoulli_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_516: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(addmm_74, [8, 196, 2048]);  addmm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_191: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_516, 0.7071067811865476)
    erf_18: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_191);  mul_191 = None
    add_165: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_54: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_35, 0.9217391312122345);  bernoulli_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_56: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_36, 0.917391300201416);  bernoulli_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_544: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(addmm_78, [8, 196, 2048]);  addmm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_201: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_544, 0.7071067811865476)
    erf_19: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_201);  mul_201 = None
    add_174: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_57: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_37, 0.917391300201416);  bernoulli_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_59: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_38, 0.9130434766411781);  bernoulli_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_570: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(addmm_82, [8, 196, 2048]);  addmm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_211: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_570, 0.7071067811865476)
    erf_20: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_211);  mul_211 = None
    add_182: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_60: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_39, 0.9130434766411781);  bernoulli_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_62: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_40, 0.9086956530809402);  bernoulli_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_598: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(addmm_86, [8, 196, 2048]);  addmm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_221: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_598, 0.7071067811865476)
    erf_21: "f32[8, 196, 2048]" = torch.ops.aten.erf.default(mul_221);  mul_221 = None
    add_191: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_63: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_41, 0.9086956530809402);  bernoulli_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:358, code: x = self.reduction(x)
    view_605: "f32[8, 7, 7, 1024]" = torch.ops.aten.reshape.default(mm_2, [8, 7, 7, 1024]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    sub_70: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(view_605, getitem_163);  view_605 = getitem_163 = None
    mul_226: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_48);  sub_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_65: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_42, 0.9043478220701218);  bernoulli_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_628: "f32[8, 49, 4096]" = torch.ops.aten.reshape.default(addmm_90, [8, 49, 4096]);  addmm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_233: "f32[8, 49, 4096]" = torch.ops.aten.mul.Tensor(view_628, 0.7071067811865476)
    erf_22: "f32[8, 49, 4096]" = torch.ops.aten.erf.default(mul_233);  mul_233 = None
    add_201: "f32[8, 49, 4096]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_66: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_43, 0.9043478220701218);  bernoulli_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_68: "f32[8, 1, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_44, 0.8999999985098839);  bernoulli_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_654: "f32[8, 49, 4096]" = torch.ops.aten.reshape.default(addmm_94, [8, 49, 4096]);  addmm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_243: "f32[8, 49, 4096]" = torch.ops.aten.mul.Tensor(view_654, 0.7071067811865476)
    erf_23: "f32[8, 49, 4096]" = torch.ops.aten.erf.default(mul_243);  mul_243 = None
    add_209: "f32[8, 49, 4096]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:153, code: random_tensor.div_(keep_prob)
    div_69: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(bernoulli_45, 0.8999999985098839);  bernoulli_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    mm_3: "f32[8, 1024]" = torch.ops.aten.mm.default(tangents_1, permute_248);  permute_248 = None
    permute_249: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_4: "f32[1000, 1024]" = torch.ops.aten.mm.default(permute_249, clone_264);  permute_249 = clone_264 = None
    permute_250: "f32[1024, 1000]" = torch.ops.aten.permute.default(mm_4, [1, 0]);  mm_4 = None
    sum_25: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_658: "f32[1000]" = torch.ops.aten.reshape.default(sum_25, [1000]);  sum_25 = None
    permute_251: "f32[1000, 1024]" = torch.ops.aten.permute.default(permute_250, [1, 0]);  permute_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:65, code: return x.mean(self.dim, keepdim=not self.flatten)
    unsqueeze_46: "f32[8, 1, 1024]" = torch.ops.aten.unsqueeze.default(mm_3, 1);  mm_3 = None
    unsqueeze_47: "f32[8, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, 2);  unsqueeze_46 = None
    expand_96: "f32[8, 7, 7, 1024]" = torch.ops.aten.expand.default(unsqueeze_47, [8, 7, 7, 1024]);  unsqueeze_47 = None
    div_70: "f32[8, 7, 7, 1024]" = torch.ops.aten.div.Scalar(expand_96, 49);  expand_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:610, code: x = self.norm(x)
    mul_249: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(div_70, primals_326);  primals_326 = None
    mul_250: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_249, 1024)
    sum_26: "f32[8, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_249, [3], True)
    mul_251: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_249, mul_246);  mul_249 = None
    sum_27: "f32[8, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [3], True);  mul_251 = None
    mul_252: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_246, sum_27);  sum_27 = None
    sub_78: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(mul_250, sum_26);  mul_250 = sum_26 = None
    sub_79: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(sub_78, mul_252);  sub_78 = mul_252 = None
    mul_253: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(div_71, sub_79);  div_71 = sub_79 = None
    mul_254: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(div_70, mul_246);  mul_246 = None
    sum_28: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_254, [0, 1, 2]);  mul_254 = None
    sum_29: "f32[1024]" = torch.ops.aten.sum.dim_IntList(div_70, [0, 1, 2]);  div_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_659: "f32[8, 49, 1024]" = torch.ops.aten.reshape.default(mul_253, [8, 49, 1024]);  mul_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_255: "f32[8, 49, 1024]" = torch.ops.aten.mul.Tensor(view_659, div_69);  div_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_660: "f32[392, 1024]" = torch.ops.aten.reshape.default(mul_255, [392, 1024]);  mul_255 = None
    mm_5: "f32[392, 4096]" = torch.ops.aten.mm.default(view_660, permute_252);  permute_252 = None
    permute_253: "f32[1024, 392]" = torch.ops.aten.permute.default(view_660, [1, 0])
    mm_6: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_253, view_655);  permute_253 = view_655 = None
    permute_254: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_6, [1, 0]);  mm_6 = None
    sum_30: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_660, [0], True);  view_660 = None
    view_661: "f32[1024]" = torch.ops.aten.reshape.default(sum_30, [1024]);  sum_30 = None
    permute_255: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_254, [1, 0]);  permute_254 = None
    view_662: "f32[8, 49, 4096]" = torch.ops.aten.reshape.default(mm_5, [8, 49, 4096]);  mm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_257: "f32[8, 49, 4096]" = torch.ops.aten.mul.Tensor(add_209, 0.5);  add_209 = None
    mul_258: "f32[8, 49, 4096]" = torch.ops.aten.mul.Tensor(view_654, view_654)
    mul_259: "f32[8, 49, 4096]" = torch.ops.aten.mul.Tensor(mul_258, -0.5);  mul_258 = None
    exp_24: "f32[8, 49, 4096]" = torch.ops.aten.exp.default(mul_259);  mul_259 = None
    mul_260: "f32[8, 49, 4096]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_261: "f32[8, 49, 4096]" = torch.ops.aten.mul.Tensor(view_654, mul_260);  view_654 = mul_260 = None
    add_214: "f32[8, 49, 4096]" = torch.ops.aten.add.Tensor(mul_257, mul_261);  mul_257 = mul_261 = None
    mul_262: "f32[8, 49, 4096]" = torch.ops.aten.mul.Tensor(view_662, add_214);  view_662 = add_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_663: "f32[392, 4096]" = torch.ops.aten.reshape.default(mul_262, [392, 4096]);  mul_262 = None
    mm_7: "f32[392, 1024]" = torch.ops.aten.mm.default(view_663, permute_256);  permute_256 = None
    permute_257: "f32[4096, 392]" = torch.ops.aten.permute.default(view_663, [1, 0])
    mm_8: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_257, view_653);  permute_257 = view_653 = None
    permute_258: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_8, [1, 0]);  mm_8 = None
    sum_31: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_663, [0], True);  view_663 = None
    view_664: "f32[4096]" = torch.ops.aten.reshape.default(sum_31, [4096]);  sum_31 = None
    permute_259: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_258, [1, 0]);  permute_258 = None
    view_665: "f32[8, 49, 1024]" = torch.ops.aten.reshape.default(mm_7, [8, 49, 1024]);  mm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_264: "f32[8, 49, 1024]" = torch.ops.aten.mul.Tensor(view_665, primals_320);  primals_320 = None
    mul_265: "f32[8, 49, 1024]" = torch.ops.aten.mul.Tensor(mul_264, 1024)
    sum_32: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_264, [2], True)
    mul_266: "f32[8, 49, 1024]" = torch.ops.aten.mul.Tensor(mul_264, mul_240);  mul_264 = None
    sum_33: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_266, [2], True);  mul_266 = None
    mul_267: "f32[8, 49, 1024]" = torch.ops.aten.mul.Tensor(mul_240, sum_33);  sum_33 = None
    sub_81: "f32[8, 49, 1024]" = torch.ops.aten.sub.Tensor(mul_265, sum_32);  mul_265 = sum_32 = None
    sub_82: "f32[8, 49, 1024]" = torch.ops.aten.sub.Tensor(sub_81, mul_267);  sub_81 = mul_267 = None
    mul_268: "f32[8, 49, 1024]" = torch.ops.aten.mul.Tensor(div_72, sub_82);  div_72 = sub_82 = None
    mul_269: "f32[8, 49, 1024]" = torch.ops.aten.mul.Tensor(view_665, mul_240);  mul_240 = None
    sum_34: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_269, [0, 1]);  mul_269 = None
    sum_35: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_665, [0, 1]);  view_665 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_215: "f32[8, 49, 1024]" = torch.ops.aten.add.Tensor(view_659, mul_268);  view_659 = mul_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_666: "f32[8, 7, 7, 1024]" = torch.ops.aten.reshape.default(add_215, [8, 7, 7, 1024]);  add_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_270: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_666, div_68);  div_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    full_default: "f32[8, 7, 7, 1024]" = torch.ops.aten.full.default([8, 7, 7, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_667: "f32[8, 1, 7, 1, 7, 1024]" = torch.ops.aten.reshape.default(mul_270, [8, 1, 7, 1, 7, 1024]);  mul_270 = None
    permute_260: "f32[8, 1, 1, 7, 7, 1024]" = torch.ops.aten.permute.default(view_667, [0, 1, 3, 2, 4, 5]);  view_667 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    view_668: "f32[8, 7, 7, 1024]" = torch.ops.aten.reshape.default(permute_260, [8, 7, 7, 1024]);  permute_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_669: "f32[8, 49, 1024]" = torch.ops.aten.reshape.default(view_668, [8, 49, 1024]);  view_668 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_670: "f32[392, 1024]" = torch.ops.aten.reshape.default(view_669, [392, 1024]);  view_669 = None
    mm_9: "f32[392, 1024]" = torch.ops.aten.mm.default(view_670, permute_261);  permute_261 = None
    permute_262: "f32[1024, 392]" = torch.ops.aten.permute.default(view_670, [1, 0])
    mm_10: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_262, view_647);  permute_262 = view_647 = None
    permute_263: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_10, [1, 0]);  mm_10 = None
    sum_36: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_670, [0], True);  view_670 = None
    view_671: "f32[1024]" = torch.ops.aten.reshape.default(sum_36, [1024]);  sum_36 = None
    permute_264: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
    view_672: "f32[8, 49, 1024]" = torch.ops.aten.reshape.default(mm_9, [8, 49, 1024]);  mm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_673: "f32[8, 49, 32, 32]" = torch.ops.aten.reshape.default(view_672, [8, 49, 32, 32]);  view_672 = None
    permute_265: "f32[8, 32, 49, 32]" = torch.ops.aten.permute.default(view_673, [0, 2, 1, 3]);  view_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_265: "f32[8, 32, 49, 32]" = torch.ops.aten.clone.default(permute_265, memory_format = torch.contiguous_format);  permute_265 = None
    view_674: "f32[256, 49, 32]" = torch.ops.aten.reshape.default(clone_265, [256, 49, 32]);  clone_265 = None
    bmm_48: "f32[256, 49, 32]" = torch.ops.aten.bmm.default(permute_266, view_674);  permute_266 = None
    bmm_49: "f32[256, 49, 49]" = torch.ops.aten.bmm.default(view_674, permute_267);  view_674 = permute_267 = None
    view_675: "f32[8, 32, 49, 32]" = torch.ops.aten.reshape.default(bmm_48, [8, 32, 49, 32]);  bmm_48 = None
    view_676: "f32[8, 32, 49, 49]" = torch.ops.aten.reshape.default(bmm_49, [8, 32, 49, 49]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    mul_271: "f32[8, 32, 49, 49]" = torch.ops.aten.mul.Tensor(view_676, alias_24);  view_676 = None
    sum_37: "f32[8, 32, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_271, [-1], True)
    mul_272: "f32[8, 32, 49, 49]" = torch.ops.aten.mul.Tensor(alias_24, sum_37);  alias_24 = sum_37 = None
    sub_83: "f32[8, 32, 49, 49]" = torch.ops.aten.sub.Tensor(mul_271, mul_272);  mul_271 = mul_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_38: "f32[1, 32, 49, 49]" = torch.ops.aten.sum.dim_IntList(sub_83, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze: "f32[32, 49, 49]" = torch.ops.aten.squeeze.dim(sum_38, 0);  sum_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_268: "f32[49, 49, 32]" = torch.ops.aten.permute.default(squeeze, [1, 2, 0]);  squeeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_677: "f32[2401, 32]" = torch.ops.aten.reshape.default(permute_268, [2401, 32]);  permute_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    full_default_2: "f32[169, 32]" = torch.ops.aten.full.default([169, 32], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    index_put: "f32[169, 32]" = torch.ops.aten.index_put.default(full_default_2, [view_641], view_677, True);  view_641 = view_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_678: "f32[256, 49, 49]" = torch.ops.aten.reshape.default(sub_83, [256, 49, 49]);  sub_83 = None
    bmm_50: "f32[256, 32, 49]" = torch.ops.aten.bmm.default(permute_269, view_678);  permute_269 = None
    bmm_51: "f32[256, 49, 32]" = torch.ops.aten.bmm.default(view_678, permute_270);  view_678 = permute_270 = None
    view_679: "f32[8, 32, 32, 49]" = torch.ops.aten.reshape.default(bmm_50, [8, 32, 32, 49]);  bmm_50 = None
    view_680: "f32[8, 32, 49, 32]" = torch.ops.aten.reshape.default(bmm_51, [8, 32, 49, 32]);  bmm_51 = None
    permute_271: "f32[8, 32, 49, 32]" = torch.ops.aten.permute.default(view_679, [0, 1, 3, 2]);  view_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_273: "f32[8, 32, 49, 32]" = torch.ops.aten.mul.Tensor(view_680, 0.1767766952966369);  view_680 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    cat: "f32[24, 32, 49, 32]" = torch.ops.aten.cat.default([mul_273, permute_271, view_675]);  mul_273 = permute_271 = view_675 = None
    view_681: "f32[3, 8, 32, 49, 32]" = torch.ops.aten.reshape.default(cat, [3, 8, 32, 49, 32]);  cat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_272: "f32[8, 49, 3, 32, 32]" = torch.ops.aten.permute.default(view_681, [1, 3, 0, 2, 4]);  view_681 = None
    clone_266: "f32[8, 49, 3, 32, 32]" = torch.ops.aten.clone.default(permute_272, memory_format = torch.contiguous_format);  permute_272 = None
    view_682: "f32[8, 49, 3072]" = torch.ops.aten.reshape.default(clone_266, [8, 49, 3072]);  clone_266 = None
    view_683: "f32[392, 3072]" = torch.ops.aten.reshape.default(view_682, [392, 3072]);  view_682 = None
    mm_11: "f32[392, 1024]" = torch.ops.aten.mm.default(view_683, permute_273);  permute_273 = None
    permute_274: "f32[3072, 392]" = torch.ops.aten.permute.default(view_683, [1, 0])
    mm_12: "f32[3072, 1024]" = torch.ops.aten.mm.default(permute_274, view_635);  permute_274 = view_635 = None
    permute_275: "f32[1024, 3072]" = torch.ops.aten.permute.default(mm_12, [1, 0]);  mm_12 = None
    sum_39: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_683, [0], True);  view_683 = None
    view_684: "f32[3072]" = torch.ops.aten.reshape.default(sum_39, [3072]);  sum_39 = None
    permute_276: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_275, [1, 0]);  permute_275 = None
    view_685: "f32[8, 49, 1024]" = torch.ops.aten.reshape.default(mm_11, [8, 49, 1024]);  mm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_686: "f32[8, 7, 7, 1024]" = torch.ops.aten.reshape.default(view_685, [8, 7, 7, 1024]);  view_685 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_687: "f32[8, 1, 1, 7, 7, 1024]" = torch.ops.aten.reshape.default(view_686, [8, 1, 1, 7, 7, 1024]);  view_686 = None
    permute_277: "f32[8, 1, 7, 1, 7, 1024]" = torch.ops.aten.permute.default(view_687, [0, 1, 3, 2, 4, 5]);  view_687 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    view_688: "f32[8, 7, 7, 1024]" = torch.ops.aten.reshape.default(permute_277, [8, 7, 7, 1024]);  permute_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    mul_275: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_688, primals_314);  primals_314 = None
    mul_276: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_275, 1024)
    sum_40: "f32[8, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_275, [3], True)
    mul_277: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_275, mul_236);  mul_275 = None
    sum_41: "f32[8, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_277, [3], True);  mul_277 = None
    mul_278: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_236, sum_41);  sum_41 = None
    sub_85: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(mul_276, sum_40);  mul_276 = sum_40 = None
    sub_86: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(sub_85, mul_278);  sub_85 = mul_278 = None
    mul_279: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(div_73, sub_86);  div_73 = sub_86 = None
    mul_280: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_688, mul_236);  mul_236 = None
    sum_42: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_280, [0, 1, 2]);  mul_280 = None
    sum_43: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_688, [0, 1, 2]);  view_688 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_216: "f32[8, 7, 7, 1024]" = torch.ops.aten.add.Tensor(view_666, mul_279);  view_666 = mul_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_689: "f32[8, 49, 1024]" = torch.ops.aten.reshape.default(add_216, [8, 49, 1024]);  add_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_281: "f32[8, 49, 1024]" = torch.ops.aten.mul.Tensor(view_689, div_66);  div_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_690: "f32[392, 1024]" = torch.ops.aten.reshape.default(mul_281, [392, 1024]);  mul_281 = None
    mm_13: "f32[392, 4096]" = torch.ops.aten.mm.default(view_690, permute_278);  permute_278 = None
    permute_279: "f32[1024, 392]" = torch.ops.aten.permute.default(view_690, [1, 0])
    mm_14: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_279, view_629);  permute_279 = view_629 = None
    permute_280: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_14, [1, 0]);  mm_14 = None
    sum_44: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_690, [0], True);  view_690 = None
    view_691: "f32[1024]" = torch.ops.aten.reshape.default(sum_44, [1024]);  sum_44 = None
    permute_281: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_280, [1, 0]);  permute_280 = None
    view_692: "f32[8, 49, 4096]" = torch.ops.aten.reshape.default(mm_13, [8, 49, 4096]);  mm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_283: "f32[8, 49, 4096]" = torch.ops.aten.mul.Tensor(add_201, 0.5);  add_201 = None
    mul_284: "f32[8, 49, 4096]" = torch.ops.aten.mul.Tensor(view_628, view_628)
    mul_285: "f32[8, 49, 4096]" = torch.ops.aten.mul.Tensor(mul_284, -0.5);  mul_284 = None
    exp_25: "f32[8, 49, 4096]" = torch.ops.aten.exp.default(mul_285);  mul_285 = None
    mul_286: "f32[8, 49, 4096]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_287: "f32[8, 49, 4096]" = torch.ops.aten.mul.Tensor(view_628, mul_286);  view_628 = mul_286 = None
    add_218: "f32[8, 49, 4096]" = torch.ops.aten.add.Tensor(mul_283, mul_287);  mul_283 = mul_287 = None
    mul_288: "f32[8, 49, 4096]" = torch.ops.aten.mul.Tensor(view_692, add_218);  view_692 = add_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_693: "f32[392, 4096]" = torch.ops.aten.reshape.default(mul_288, [392, 4096]);  mul_288 = None
    mm_15: "f32[392, 1024]" = torch.ops.aten.mm.default(view_693, permute_282);  permute_282 = None
    permute_283: "f32[4096, 392]" = torch.ops.aten.permute.default(view_693, [1, 0])
    mm_16: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_283, view_627);  permute_283 = view_627 = None
    permute_284: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_16, [1, 0]);  mm_16 = None
    sum_45: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_693, [0], True);  view_693 = None
    view_694: "f32[4096]" = torch.ops.aten.reshape.default(sum_45, [4096]);  sum_45 = None
    permute_285: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_284, [1, 0]);  permute_284 = None
    view_695: "f32[8, 49, 1024]" = torch.ops.aten.reshape.default(mm_15, [8, 49, 1024]);  mm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_290: "f32[8, 49, 1024]" = torch.ops.aten.mul.Tensor(view_695, primals_308);  primals_308 = None
    mul_291: "f32[8, 49, 1024]" = torch.ops.aten.mul.Tensor(mul_290, 1024)
    sum_46: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_290, [2], True)
    mul_292: "f32[8, 49, 1024]" = torch.ops.aten.mul.Tensor(mul_290, mul_230);  mul_290 = None
    sum_47: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_292, [2], True);  mul_292 = None
    mul_293: "f32[8, 49, 1024]" = torch.ops.aten.mul.Tensor(mul_230, sum_47);  sum_47 = None
    sub_88: "f32[8, 49, 1024]" = torch.ops.aten.sub.Tensor(mul_291, sum_46);  mul_291 = sum_46 = None
    sub_89: "f32[8, 49, 1024]" = torch.ops.aten.sub.Tensor(sub_88, mul_293);  sub_88 = mul_293 = None
    mul_294: "f32[8, 49, 1024]" = torch.ops.aten.mul.Tensor(div_74, sub_89);  div_74 = sub_89 = None
    mul_295: "f32[8, 49, 1024]" = torch.ops.aten.mul.Tensor(view_695, mul_230);  mul_230 = None
    sum_48: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_295, [0, 1]);  mul_295 = None
    sum_49: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_695, [0, 1]);  view_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_219: "f32[8, 49, 1024]" = torch.ops.aten.add.Tensor(view_689, mul_294);  view_689 = mul_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_696: "f32[8, 7, 7, 1024]" = torch.ops.aten.reshape.default(add_219, [8, 7, 7, 1024]);  add_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_296: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_696, div_65);  div_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_697: "f32[8, 1, 7, 1, 7, 1024]" = torch.ops.aten.reshape.default(mul_296, [8, 1, 7, 1, 7, 1024]);  mul_296 = None
    permute_286: "f32[8, 1, 1, 7, 7, 1024]" = torch.ops.aten.permute.default(view_697, [0, 1, 3, 2, 4, 5]);  view_697 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    view_698: "f32[8, 7, 7, 1024]" = torch.ops.aten.reshape.default(permute_286, [8, 7, 7, 1024]);  permute_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_699: "f32[8, 49, 1024]" = torch.ops.aten.reshape.default(view_698, [8, 49, 1024]);  view_698 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_700: "f32[392, 1024]" = torch.ops.aten.reshape.default(view_699, [392, 1024]);  view_699 = None
    mm_17: "f32[392, 1024]" = torch.ops.aten.mm.default(view_700, permute_287);  permute_287 = None
    permute_288: "f32[1024, 392]" = torch.ops.aten.permute.default(view_700, [1, 0])
    mm_18: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_288, view_621);  permute_288 = view_621 = None
    permute_289: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_18, [1, 0]);  mm_18 = None
    sum_50: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_700, [0], True);  view_700 = None
    view_701: "f32[1024]" = torch.ops.aten.reshape.default(sum_50, [1024]);  sum_50 = None
    permute_290: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_289, [1, 0]);  permute_289 = None
    view_702: "f32[8, 49, 1024]" = torch.ops.aten.reshape.default(mm_17, [8, 49, 1024]);  mm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_703: "f32[8, 49, 32, 32]" = torch.ops.aten.reshape.default(view_702, [8, 49, 32, 32]);  view_702 = None
    permute_291: "f32[8, 32, 49, 32]" = torch.ops.aten.permute.default(view_703, [0, 2, 1, 3]);  view_703 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_267: "f32[8, 32, 49, 32]" = torch.ops.aten.clone.default(permute_291, memory_format = torch.contiguous_format);  permute_291 = None
    view_704: "f32[256, 49, 32]" = torch.ops.aten.reshape.default(clone_267, [256, 49, 32]);  clone_267 = None
    bmm_52: "f32[256, 49, 32]" = torch.ops.aten.bmm.default(permute_292, view_704);  permute_292 = None
    bmm_53: "f32[256, 49, 49]" = torch.ops.aten.bmm.default(view_704, permute_293);  view_704 = permute_293 = None
    view_705: "f32[8, 32, 49, 32]" = torch.ops.aten.reshape.default(bmm_52, [8, 32, 49, 32]);  bmm_52 = None
    view_706: "f32[8, 32, 49, 49]" = torch.ops.aten.reshape.default(bmm_53, [8, 32, 49, 49]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    mul_297: "f32[8, 32, 49, 49]" = torch.ops.aten.mul.Tensor(view_706, alias_25);  view_706 = None
    sum_51: "f32[8, 32, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_297, [-1], True)
    mul_298: "f32[8, 32, 49, 49]" = torch.ops.aten.mul.Tensor(alias_25, sum_51);  alias_25 = sum_51 = None
    sub_90: "f32[8, 32, 49, 49]" = torch.ops.aten.sub.Tensor(mul_297, mul_298);  mul_297 = mul_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_52: "f32[1, 32, 49, 49]" = torch.ops.aten.sum.dim_IntList(sub_90, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_1: "f32[32, 49, 49]" = torch.ops.aten.squeeze.dim(sum_52, 0);  sum_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_294: "f32[49, 49, 32]" = torch.ops.aten.permute.default(squeeze_1, [1, 2, 0]);  squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_707: "f32[2401, 32]" = torch.ops.aten.reshape.default(permute_294, [2401, 32]);  permute_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_put_1: "f32[169, 32]" = torch.ops.aten.index_put_.default(full_default_2, [view_615], view_707, True);  full_default_2 = view_615 = view_707 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_708: "f32[256, 49, 49]" = torch.ops.aten.reshape.default(sub_90, [256, 49, 49]);  sub_90 = None
    bmm_54: "f32[256, 32, 49]" = torch.ops.aten.bmm.default(permute_295, view_708);  permute_295 = None
    bmm_55: "f32[256, 49, 32]" = torch.ops.aten.bmm.default(view_708, permute_296);  view_708 = permute_296 = None
    view_709: "f32[8, 32, 32, 49]" = torch.ops.aten.reshape.default(bmm_54, [8, 32, 32, 49]);  bmm_54 = None
    view_710: "f32[8, 32, 49, 32]" = torch.ops.aten.reshape.default(bmm_55, [8, 32, 49, 32]);  bmm_55 = None
    permute_297: "f32[8, 32, 49, 32]" = torch.ops.aten.permute.default(view_709, [0, 1, 3, 2]);  view_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_299: "f32[8, 32, 49, 32]" = torch.ops.aten.mul.Tensor(view_710, 0.1767766952966369);  view_710 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    cat_1: "f32[24, 32, 49, 32]" = torch.ops.aten.cat.default([mul_299, permute_297, view_705]);  mul_299 = permute_297 = view_705 = None
    view_711: "f32[3, 8, 32, 49, 32]" = torch.ops.aten.reshape.default(cat_1, [3, 8, 32, 49, 32]);  cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_298: "f32[8, 49, 3, 32, 32]" = torch.ops.aten.permute.default(view_711, [1, 3, 0, 2, 4]);  view_711 = None
    clone_268: "f32[8, 49, 3, 32, 32]" = torch.ops.aten.clone.default(permute_298, memory_format = torch.contiguous_format);  permute_298 = None
    view_712: "f32[8, 49, 3072]" = torch.ops.aten.reshape.default(clone_268, [8, 49, 3072]);  clone_268 = None
    view_713: "f32[392, 3072]" = torch.ops.aten.reshape.default(view_712, [392, 3072]);  view_712 = None
    mm_19: "f32[392, 1024]" = torch.ops.aten.mm.default(view_713, permute_299);  permute_299 = None
    permute_300: "f32[3072, 392]" = torch.ops.aten.permute.default(view_713, [1, 0])
    mm_20: "f32[3072, 1024]" = torch.ops.aten.mm.default(permute_300, view_609);  permute_300 = view_609 = None
    permute_301: "f32[1024, 3072]" = torch.ops.aten.permute.default(mm_20, [1, 0]);  mm_20 = None
    sum_53: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_713, [0], True);  view_713 = None
    view_714: "f32[3072]" = torch.ops.aten.reshape.default(sum_53, [3072]);  sum_53 = None
    permute_302: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_301, [1, 0]);  permute_301 = None
    view_715: "f32[8, 49, 1024]" = torch.ops.aten.reshape.default(mm_19, [8, 49, 1024]);  mm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_716: "f32[8, 7, 7, 1024]" = torch.ops.aten.reshape.default(view_715, [8, 7, 7, 1024]);  view_715 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_717: "f32[8, 1, 1, 7, 7, 1024]" = torch.ops.aten.reshape.default(view_716, [8, 1, 1, 7, 7, 1024]);  view_716 = None
    permute_303: "f32[8, 1, 7, 1, 7, 1024]" = torch.ops.aten.permute.default(view_717, [0, 1, 3, 2, 4, 5]);  view_717 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    view_718: "f32[8, 7, 7, 1024]" = torch.ops.aten.reshape.default(permute_303, [8, 7, 7, 1024]);  permute_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    mul_301: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_718, primals_302);  primals_302 = None
    mul_302: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_301, 1024)
    sum_54: "f32[8, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_301, [3], True)
    mul_303: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_301, mul_226);  mul_301 = None
    sum_55: "f32[8, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_303, [3], True);  mul_303 = None
    mul_304: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_226, sum_55);  sum_55 = None
    sub_92: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(mul_302, sum_54);  mul_302 = sum_54 = None
    sub_93: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(sub_92, mul_304);  sub_92 = mul_304 = None
    div_75: "f32[8, 7, 7, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 1024);  rsqrt_48 = None
    mul_305: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(div_75, sub_93);  div_75 = sub_93 = None
    mul_306: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_718, mul_226);  mul_226 = None
    sum_56: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_306, [0, 1, 2]);  mul_306 = None
    sum_57: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_718, [0, 1, 2]);  view_718 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_220: "f32[8, 7, 7, 1024]" = torch.ops.aten.add.Tensor(view_696, mul_305);  view_696 = mul_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:358, code: x = self.reduction(x)
    view_719: "f32[392, 1024]" = torch.ops.aten.reshape.default(add_220, [392, 1024]);  add_220 = None
    permute_304: "f32[1024, 392]" = torch.ops.aten.permute.default(view_719, [1, 0])
    mm_21: "f32[1024, 2048]" = torch.ops.aten.mm.default(permute_304, view_604);  permute_304 = view_604 = None
    permute_305: "f32[2048, 1024]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    mm_22: "f32[392, 2048]" = torch.ops.aten.mm.default(view_719, permute_306);  view_719 = permute_306 = None
    view_720: "f32[8, 7, 7, 2048]" = torch.ops.aten.reshape.default(mm_22, [8, 7, 7, 2048]);  mm_22 = None
    permute_307: "f32[1024, 2048]" = torch.ops.aten.permute.default(permute_305, [1, 0]);  permute_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:357, code: x = self.norm(x)
    mul_308: "f32[8, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(view_720, primals_299);  primals_299 = None
    mul_309: "f32[8, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(mul_308, 2048)
    sum_58: "f32[8, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_308, [3], True)
    mul_310: "f32[8, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(mul_308, mul_224);  mul_308 = None
    sum_59: "f32[8, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_310, [3], True);  mul_310 = None
    mul_311: "f32[8, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(mul_224, sum_59);  sum_59 = None
    sub_95: "f32[8, 7, 7, 2048]" = torch.ops.aten.sub.Tensor(mul_309, sum_58);  mul_309 = sum_58 = None
    sub_96: "f32[8, 7, 7, 2048]" = torch.ops.aten.sub.Tensor(sub_95, mul_311);  sub_95 = mul_311 = None
    mul_312: "f32[8, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(div_76, sub_96);  div_76 = sub_96 = None
    mul_313: "f32[8, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(view_720, mul_224);  mul_224 = None
    sum_60: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_313, [0, 1, 2]);  mul_313 = None
    sum_61: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_720, [0, 1, 2]);  view_720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:356, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
    view_721: "f32[8, 7, 7, 2, 2, 512]" = torch.ops.aten.reshape.default(mul_312, [8, 7, 7, 2, 2, 512]);  mul_312 = None
    permute_308: "f32[8, 7, 2, 7, 2, 512]" = torch.ops.aten.permute.default(view_721, [0, 1, 4, 2, 3, 5]);  view_721 = None
    clone_269: "f32[8, 7, 2, 7, 2, 512]" = torch.ops.aten.clone.default(permute_308, memory_format = torch.contiguous_format);  permute_308 = None
    view_722: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_269, [8, 14, 14, 512]);  clone_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_723: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(view_722, [8, 196, 512]);  view_722 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_314: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_723, div_63);  div_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_724: "f32[1568, 512]" = torch.ops.aten.reshape.default(mul_314, [1568, 512]);  mul_314 = None
    mm_23: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_724, permute_309);  permute_309 = None
    permute_310: "f32[512, 1568]" = torch.ops.aten.permute.default(view_724, [1, 0])
    mm_24: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_310, view_599);  permute_310 = view_599 = None
    permute_311: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_24, [1, 0]);  mm_24 = None
    sum_62: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_724, [0], True);  view_724 = None
    view_725: "f32[512]" = torch.ops.aten.reshape.default(sum_62, [512]);  sum_62 = None
    permute_312: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_311, [1, 0]);  permute_311 = None
    view_726: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(mm_23, [8, 196, 2048]);  mm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_316: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(add_191, 0.5);  add_191 = None
    mul_317: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_598, view_598)
    mul_318: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_317, -0.5);  mul_317 = None
    exp_26: "f32[8, 196, 2048]" = torch.ops.aten.exp.default(mul_318);  mul_318 = None
    mul_319: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_320: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_598, mul_319);  view_598 = mul_319 = None
    add_222: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(mul_316, mul_320);  mul_316 = mul_320 = None
    mul_321: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_726, add_222);  view_726 = add_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_727: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_321, [1568, 2048]);  mul_321 = None
    mm_25: "f32[1568, 512]" = torch.ops.aten.mm.default(view_727, permute_313);  permute_313 = None
    permute_314: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_727, [1, 0])
    mm_26: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_314, view_597);  permute_314 = view_597 = None
    permute_315: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_26, [1, 0]);  mm_26 = None
    sum_63: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_727, [0], True);  view_727 = None
    view_728: "f32[2048]" = torch.ops.aten.reshape.default(sum_63, [2048]);  sum_63 = None
    permute_316: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_315, [1, 0]);  permute_315 = None
    view_729: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(mm_25, [8, 196, 512]);  mm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_323: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_729, primals_293);  primals_293 = None
    mul_324: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_323, 512)
    sum_64: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_323, [2], True)
    mul_325: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_323, mul_218);  mul_323 = None
    sum_65: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_325, [2], True);  mul_325 = None
    mul_326: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_218, sum_65);  sum_65 = None
    sub_98: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(mul_324, sum_64);  mul_324 = sum_64 = None
    sub_99: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(sub_98, mul_326);  sub_98 = mul_326 = None
    mul_327: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(div_77, sub_99);  div_77 = sub_99 = None
    mul_328: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_729, mul_218);  mul_218 = None
    sum_66: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_328, [0, 1]);  mul_328 = None
    sum_67: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_729, [0, 1]);  view_729 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_223: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_723, mul_327);  view_723 = mul_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_730: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(add_223, [8, 14, 14, 512]);  add_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_329: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_730, div_62);  div_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    roll_22: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(mul_329, [-3, -3], [2, 1]);  mul_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    full_default_6: "f32[8, 14, 14, 512]" = torch.ops.aten.full.default([8, 14, 14, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_731: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(roll_22, [8, 2, 7, 2, 7, 512]);  roll_22 = None
    permute_317: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_731, [0, 1, 3, 2, 4, 5]);  view_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_270: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_317, memory_format = torch.contiguous_format);  permute_317 = None
    view_732: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_270, [32, 7, 7, 512]);  clone_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_733: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(view_732, [32, 49, 512]);  view_732 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_734: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_733, [1568, 512]);  view_733 = None
    mm_27: "f32[1568, 512]" = torch.ops.aten.mm.default(view_734, permute_318);  permute_318 = None
    permute_319: "f32[512, 1568]" = torch.ops.aten.permute.default(view_734, [1, 0])
    mm_28: "f32[512, 512]" = torch.ops.aten.mm.default(permute_319, view_591);  permute_319 = view_591 = None
    permute_320: "f32[512, 512]" = torch.ops.aten.permute.default(mm_28, [1, 0]);  mm_28 = None
    sum_68: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_734, [0], True);  view_734 = None
    view_735: "f32[512]" = torch.ops.aten.reshape.default(sum_68, [512]);  sum_68 = None
    permute_321: "f32[512, 512]" = torch.ops.aten.permute.default(permute_320, [1, 0]);  permute_320 = None
    view_736: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_27, [32, 49, 512]);  mm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_737: "f32[32, 49, 16, 32]" = torch.ops.aten.reshape.default(view_736, [32, 49, 16, 32]);  view_736 = None
    permute_322: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_737, [0, 2, 1, 3]);  view_737 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_271: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(permute_322, memory_format = torch.contiguous_format);  permute_322 = None
    view_738: "f32[512, 49, 32]" = torch.ops.aten.reshape.default(clone_271, [512, 49, 32]);  clone_271 = None
    bmm_56: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(permute_323, view_738);  permute_323 = None
    bmm_57: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_738, permute_324);  view_738 = permute_324 = None
    view_739: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_56, [32, 16, 49, 32]);  bmm_56 = None
    view_740: "f32[32, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_57, [32, 16, 49, 49]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    mul_330: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_740, alias_26);  view_740 = None
    sum_69: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_330, [-1], True)
    mul_331: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(alias_26, sum_69);  alias_26 = sum_69 = None
    sub_100: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(mul_330, mul_331);  mul_330 = mul_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    view_741: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.reshape.default(sub_100, [8, 4, 16, 49, 49]);  sub_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_742: "f32[32, 16, 49, 49]" = torch.ops.aten.reshape.default(view_741, [32, 16, 49, 49]);  view_741 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_70: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(view_742, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_2: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_70, 0);  sum_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_325: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_2, [1, 2, 0]);  squeeze_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_743: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_325, [2401, 16]);  permute_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    full_default_8: "f32[169, 16]" = torch.ops.aten.full.default([169, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    index_put_2: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_8, [view_583], view_743, True);  view_583 = view_743 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_744: "f32[512, 49, 49]" = torch.ops.aten.reshape.default(view_742, [512, 49, 49]);  view_742 = None
    bmm_58: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(permute_326, view_744);  permute_326 = None
    bmm_59: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_744, permute_327);  view_744 = permute_327 = None
    view_745: "f32[32, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_58, [32, 16, 32, 49]);  bmm_58 = None
    view_746: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_59, [32, 16, 49, 32]);  bmm_59 = None
    permute_328: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_745, [0, 1, 3, 2]);  view_745 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_332: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_746, 0.1767766952966369);  view_746 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    cat_2: "f32[96, 16, 49, 32]" = torch.ops.aten.cat.default([mul_332, permute_328, view_739]);  mul_332 = permute_328 = view_739 = None
    view_747: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_2, [3, 32, 16, 49, 32]);  cat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_329: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_747, [1, 3, 0, 2, 4]);  view_747 = None
    clone_272: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_329, memory_format = torch.contiguous_format);  permute_329 = None
    view_748: "f32[32, 49, 1536]" = torch.ops.aten.reshape.default(clone_272, [32, 49, 1536]);  clone_272 = None
    view_749: "f32[1568, 1536]" = torch.ops.aten.reshape.default(view_748, [1568, 1536]);  view_748 = None
    mm_29: "f32[1568, 512]" = torch.ops.aten.mm.default(view_749, permute_330);  permute_330 = None
    permute_331: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_749, [1, 0])
    mm_30: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_331, view_577);  permute_331 = view_577 = None
    permute_332: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_30, [1, 0]);  mm_30 = None
    sum_71: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_749, [0], True);  view_749 = None
    view_750: "f32[1536]" = torch.ops.aten.reshape.default(sum_71, [1536]);  sum_71 = None
    permute_333: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_332, [1, 0]);  permute_332 = None
    view_751: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_29, [32, 49, 512]);  mm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_752: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(view_751, [32, 7, 7, 512]);  view_751 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_753: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_752, [8, 2, 2, 7, 7, 512]);  view_752 = None
    permute_334: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_753, [0, 1, 3, 2, 4, 5]);  view_753 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_273: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_334, memory_format = torch.contiguous_format);  permute_334 = None
    view_754: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_273, [8, 14, 14, 512]);  clone_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    roll_23: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(view_754, [3, 3], [2, 1]);  view_754 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    mul_334: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(roll_23, primals_287);  primals_287 = None
    mul_335: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_334, 512)
    sum_72: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_334, [3], True)
    mul_336: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_334, mul_214);  mul_334 = None
    sum_73: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_336, [3], True);  mul_336 = None
    mul_337: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_214, sum_73);  sum_73 = None
    sub_102: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_335, sum_72);  mul_335 = sum_72 = None
    sub_103: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_102, mul_337);  sub_102 = mul_337 = None
    mul_338: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_78, sub_103);  div_78 = sub_103 = None
    mul_339: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(roll_23, mul_214);  mul_214 = None
    sum_74: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_339, [0, 1, 2]);  mul_339 = None
    sum_75: "f32[512]" = torch.ops.aten.sum.dim_IntList(roll_23, [0, 1, 2]);  roll_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_224: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_730, mul_338);  view_730 = mul_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_755: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(add_224, [8, 196, 512]);  add_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_340: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_755, div_60);  div_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_756: "f32[1568, 512]" = torch.ops.aten.reshape.default(mul_340, [1568, 512]);  mul_340 = None
    mm_31: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_756, permute_335);  permute_335 = None
    permute_336: "f32[512, 1568]" = torch.ops.aten.permute.default(view_756, [1, 0])
    mm_32: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_336, view_571);  permute_336 = view_571 = None
    permute_337: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_32, [1, 0]);  mm_32 = None
    sum_76: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_756, [0], True);  view_756 = None
    view_757: "f32[512]" = torch.ops.aten.reshape.default(sum_76, [512]);  sum_76 = None
    permute_338: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_337, [1, 0]);  permute_337 = None
    view_758: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(mm_31, [8, 196, 2048]);  mm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_342: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(add_182, 0.5);  add_182 = None
    mul_343: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_570, view_570)
    mul_344: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_343, -0.5);  mul_343 = None
    exp_27: "f32[8, 196, 2048]" = torch.ops.aten.exp.default(mul_344);  mul_344 = None
    mul_345: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_346: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_570, mul_345);  view_570 = mul_345 = None
    add_226: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(mul_342, mul_346);  mul_342 = mul_346 = None
    mul_347: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_758, add_226);  view_758 = add_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_759: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_347, [1568, 2048]);  mul_347 = None
    mm_33: "f32[1568, 512]" = torch.ops.aten.mm.default(view_759, permute_339);  permute_339 = None
    permute_340: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_759, [1, 0])
    mm_34: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_340, view_569);  permute_340 = view_569 = None
    permute_341: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_34, [1, 0]);  mm_34 = None
    sum_77: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_759, [0], True);  view_759 = None
    view_760: "f32[2048]" = torch.ops.aten.reshape.default(sum_77, [2048]);  sum_77 = None
    permute_342: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_341, [1, 0]);  permute_341 = None
    view_761: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(mm_33, [8, 196, 512]);  mm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_349: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_761, primals_281);  primals_281 = None
    mul_350: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_349, 512)
    sum_78: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_349, [2], True)
    mul_351: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_349, mul_208);  mul_349 = None
    sum_79: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_351, [2], True);  mul_351 = None
    mul_352: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_208, sum_79);  sum_79 = None
    sub_105: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(mul_350, sum_78);  mul_350 = sum_78 = None
    sub_106: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(sub_105, mul_352);  sub_105 = mul_352 = None
    mul_353: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(div_79, sub_106);  div_79 = sub_106 = None
    mul_354: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_761, mul_208);  mul_208 = None
    sum_80: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_354, [0, 1]);  mul_354 = None
    sum_81: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_761, [0, 1]);  view_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_227: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_755, mul_353);  view_755 = mul_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_762: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(add_227, [8, 14, 14, 512]);  add_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_355: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_762, div_59);  div_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_763: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(mul_355, [8, 2, 7, 2, 7, 512]);  mul_355 = None
    permute_343: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_763, [0, 1, 3, 2, 4, 5]);  view_763 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_274: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_343, memory_format = torch.contiguous_format);  permute_343 = None
    view_764: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_274, [32, 7, 7, 512]);  clone_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_765: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(view_764, [32, 49, 512]);  view_764 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_766: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_765, [1568, 512]);  view_765 = None
    mm_35: "f32[1568, 512]" = torch.ops.aten.mm.default(view_766, permute_344);  permute_344 = None
    permute_345: "f32[512, 1568]" = torch.ops.aten.permute.default(view_766, [1, 0])
    mm_36: "f32[512, 512]" = torch.ops.aten.mm.default(permute_345, view_563);  permute_345 = view_563 = None
    permute_346: "f32[512, 512]" = torch.ops.aten.permute.default(mm_36, [1, 0]);  mm_36 = None
    sum_82: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_766, [0], True);  view_766 = None
    view_767: "f32[512]" = torch.ops.aten.reshape.default(sum_82, [512]);  sum_82 = None
    permute_347: "f32[512, 512]" = torch.ops.aten.permute.default(permute_346, [1, 0]);  permute_346 = None
    view_768: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_35, [32, 49, 512]);  mm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_769: "f32[32, 49, 16, 32]" = torch.ops.aten.reshape.default(view_768, [32, 49, 16, 32]);  view_768 = None
    permute_348: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_769, [0, 2, 1, 3]);  view_769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_275: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(permute_348, memory_format = torch.contiguous_format);  permute_348 = None
    view_770: "f32[512, 49, 32]" = torch.ops.aten.reshape.default(clone_275, [512, 49, 32]);  clone_275 = None
    bmm_60: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(permute_349, view_770);  permute_349 = None
    bmm_61: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_770, permute_350);  view_770 = permute_350 = None
    view_771: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_60, [32, 16, 49, 32]);  bmm_60 = None
    view_772: "f32[32, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_61, [32, 16, 49, 49]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    mul_356: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_772, alias_27);  view_772 = None
    sum_83: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_356, [-1], True)
    mul_357: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(alias_27, sum_83);  alias_27 = sum_83 = None
    sub_107: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(mul_356, mul_357);  mul_356 = mul_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_84: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(sub_107, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_3: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_84, 0);  sum_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_351: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_3, [1, 2, 0]);  squeeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_773: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_351, [2401, 16]);  permute_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_put_3: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_8, [view_557], view_773, True);  view_557 = view_773 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_774: "f32[512, 49, 49]" = torch.ops.aten.reshape.default(sub_107, [512, 49, 49]);  sub_107 = None
    bmm_62: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(permute_352, view_774);  permute_352 = None
    bmm_63: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_774, permute_353);  view_774 = permute_353 = None
    view_775: "f32[32, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_62, [32, 16, 32, 49]);  bmm_62 = None
    view_776: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_63, [32, 16, 49, 32]);  bmm_63 = None
    permute_354: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_775, [0, 1, 3, 2]);  view_775 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_358: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_776, 0.1767766952966369);  view_776 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    cat_3: "f32[96, 16, 49, 32]" = torch.ops.aten.cat.default([mul_358, permute_354, view_771]);  mul_358 = permute_354 = view_771 = None
    view_777: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_3, [3, 32, 16, 49, 32]);  cat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_355: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_777, [1, 3, 0, 2, 4]);  view_777 = None
    clone_276: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_355, memory_format = torch.contiguous_format);  permute_355 = None
    view_778: "f32[32, 49, 1536]" = torch.ops.aten.reshape.default(clone_276, [32, 49, 1536]);  clone_276 = None
    view_779: "f32[1568, 1536]" = torch.ops.aten.reshape.default(view_778, [1568, 1536]);  view_778 = None
    mm_37: "f32[1568, 512]" = torch.ops.aten.mm.default(view_779, permute_356);  permute_356 = None
    permute_357: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_779, [1, 0])
    mm_38: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_357, view_551);  permute_357 = view_551 = None
    permute_358: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_38, [1, 0]);  mm_38 = None
    sum_85: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_779, [0], True);  view_779 = None
    view_780: "f32[1536]" = torch.ops.aten.reshape.default(sum_85, [1536]);  sum_85 = None
    permute_359: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_358, [1, 0]);  permute_358 = None
    view_781: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_37, [32, 49, 512]);  mm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_782: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(view_781, [32, 7, 7, 512]);  view_781 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_783: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_782, [8, 2, 2, 7, 7, 512]);  view_782 = None
    permute_360: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_783, [0, 1, 3, 2, 4, 5]);  view_783 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_277: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_360, memory_format = torch.contiguous_format);  permute_360 = None
    view_784: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_277, [8, 14, 14, 512]);  clone_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    mul_360: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_784, primals_275);  primals_275 = None
    mul_361: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_360, 512)
    sum_86: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_360, [3], True)
    mul_362: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_360, mul_204);  mul_360 = None
    sum_87: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_362, [3], True);  mul_362 = None
    mul_363: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_204, sum_87);  sum_87 = None
    sub_109: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_361, sum_86);  mul_361 = sum_86 = None
    sub_110: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_109, mul_363);  sub_109 = mul_363 = None
    mul_364: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_80, sub_110);  div_80 = sub_110 = None
    mul_365: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_784, mul_204);  mul_204 = None
    sum_88: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_365, [0, 1, 2]);  mul_365 = None
    sum_89: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_784, [0, 1, 2]);  view_784 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_228: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_762, mul_364);  view_762 = mul_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_785: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(add_228, [8, 196, 512]);  add_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_366: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_785, div_57);  div_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_786: "f32[1568, 512]" = torch.ops.aten.reshape.default(mul_366, [1568, 512]);  mul_366 = None
    mm_39: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_786, permute_361);  permute_361 = None
    permute_362: "f32[512, 1568]" = torch.ops.aten.permute.default(view_786, [1, 0])
    mm_40: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_362, view_545);  permute_362 = view_545 = None
    permute_363: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_40, [1, 0]);  mm_40 = None
    sum_90: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_786, [0], True);  view_786 = None
    view_787: "f32[512]" = torch.ops.aten.reshape.default(sum_90, [512]);  sum_90 = None
    permute_364: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_363, [1, 0]);  permute_363 = None
    view_788: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(mm_39, [8, 196, 2048]);  mm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_368: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(add_174, 0.5);  add_174 = None
    mul_369: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_544, view_544)
    mul_370: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_369, -0.5);  mul_369 = None
    exp_28: "f32[8, 196, 2048]" = torch.ops.aten.exp.default(mul_370);  mul_370 = None
    mul_371: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
    mul_372: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_544, mul_371);  view_544 = mul_371 = None
    add_230: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(mul_368, mul_372);  mul_368 = mul_372 = None
    mul_373: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_788, add_230);  view_788 = add_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_789: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_373, [1568, 2048]);  mul_373 = None
    mm_41: "f32[1568, 512]" = torch.ops.aten.mm.default(view_789, permute_365);  permute_365 = None
    permute_366: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_789, [1, 0])
    mm_42: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_366, view_543);  permute_366 = view_543 = None
    permute_367: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_42, [1, 0]);  mm_42 = None
    sum_91: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_789, [0], True);  view_789 = None
    view_790: "f32[2048]" = torch.ops.aten.reshape.default(sum_91, [2048]);  sum_91 = None
    permute_368: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_367, [1, 0]);  permute_367 = None
    view_791: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(mm_41, [8, 196, 512]);  mm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_375: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_791, primals_269);  primals_269 = None
    mul_376: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_375, 512)
    sum_92: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_375, [2], True)
    mul_377: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_375, mul_198);  mul_375 = None
    sum_93: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_377, [2], True);  mul_377 = None
    mul_378: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_198, sum_93);  sum_93 = None
    sub_112: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(mul_376, sum_92);  mul_376 = sum_92 = None
    sub_113: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(sub_112, mul_378);  sub_112 = mul_378 = None
    mul_379: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(div_81, sub_113);  div_81 = sub_113 = None
    mul_380: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_791, mul_198);  mul_198 = None
    sum_94: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_380, [0, 1]);  mul_380 = None
    sum_95: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_791, [0, 1]);  view_791 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_231: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_785, mul_379);  view_785 = mul_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_792: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(add_231, [8, 14, 14, 512]);  add_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_381: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_792, div_56);  div_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    roll_24: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(mul_381, [-3, -3], [2, 1]);  mul_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_793: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(roll_24, [8, 2, 7, 2, 7, 512]);  roll_24 = None
    permute_369: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_793, [0, 1, 3, 2, 4, 5]);  view_793 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_278: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_369, memory_format = torch.contiguous_format);  permute_369 = None
    view_794: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_278, [32, 7, 7, 512]);  clone_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_795: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(view_794, [32, 49, 512]);  view_794 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_796: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_795, [1568, 512]);  view_795 = None
    mm_43: "f32[1568, 512]" = torch.ops.aten.mm.default(view_796, permute_370);  permute_370 = None
    permute_371: "f32[512, 1568]" = torch.ops.aten.permute.default(view_796, [1, 0])
    mm_44: "f32[512, 512]" = torch.ops.aten.mm.default(permute_371, view_537);  permute_371 = view_537 = None
    permute_372: "f32[512, 512]" = torch.ops.aten.permute.default(mm_44, [1, 0]);  mm_44 = None
    sum_96: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_796, [0], True);  view_796 = None
    view_797: "f32[512]" = torch.ops.aten.reshape.default(sum_96, [512]);  sum_96 = None
    permute_373: "f32[512, 512]" = torch.ops.aten.permute.default(permute_372, [1, 0]);  permute_372 = None
    view_798: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_43, [32, 49, 512]);  mm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_799: "f32[32, 49, 16, 32]" = torch.ops.aten.reshape.default(view_798, [32, 49, 16, 32]);  view_798 = None
    permute_374: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_799, [0, 2, 1, 3]);  view_799 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_279: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(permute_374, memory_format = torch.contiguous_format);  permute_374 = None
    view_800: "f32[512, 49, 32]" = torch.ops.aten.reshape.default(clone_279, [512, 49, 32]);  clone_279 = None
    bmm_64: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(permute_375, view_800);  permute_375 = None
    bmm_65: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_800, permute_376);  view_800 = permute_376 = None
    view_801: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_64, [32, 16, 49, 32]);  bmm_64 = None
    view_802: "f32[32, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_65, [32, 16, 49, 49]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    mul_382: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_802, alias_28);  view_802 = None
    sum_97: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_382, [-1], True)
    mul_383: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(alias_28, sum_97);  alias_28 = sum_97 = None
    sub_114: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(mul_382, mul_383);  mul_382 = mul_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    view_803: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.reshape.default(sub_114, [8, 4, 16, 49, 49]);  sub_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_804: "f32[32, 16, 49, 49]" = torch.ops.aten.reshape.default(view_803, [32, 16, 49, 49]);  view_803 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_98: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(view_804, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_4: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_98, 0);  sum_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_377: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_4, [1, 2, 0]);  squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_805: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_377, [2401, 16]);  permute_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_put_4: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_8, [view_529], view_805, True);  view_529 = view_805 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_806: "f32[512, 49, 49]" = torch.ops.aten.reshape.default(view_804, [512, 49, 49]);  view_804 = None
    bmm_66: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(permute_378, view_806);  permute_378 = None
    bmm_67: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_806, permute_379);  view_806 = permute_379 = None
    view_807: "f32[32, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_66, [32, 16, 32, 49]);  bmm_66 = None
    view_808: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_67, [32, 16, 49, 32]);  bmm_67 = None
    permute_380: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_807, [0, 1, 3, 2]);  view_807 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_384: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_808, 0.1767766952966369);  view_808 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    cat_4: "f32[96, 16, 49, 32]" = torch.ops.aten.cat.default([mul_384, permute_380, view_801]);  mul_384 = permute_380 = view_801 = None
    view_809: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_4, [3, 32, 16, 49, 32]);  cat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_381: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_809, [1, 3, 0, 2, 4]);  view_809 = None
    clone_280: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_381, memory_format = torch.contiguous_format);  permute_381 = None
    view_810: "f32[32, 49, 1536]" = torch.ops.aten.reshape.default(clone_280, [32, 49, 1536]);  clone_280 = None
    view_811: "f32[1568, 1536]" = torch.ops.aten.reshape.default(view_810, [1568, 1536]);  view_810 = None
    mm_45: "f32[1568, 512]" = torch.ops.aten.mm.default(view_811, permute_382);  permute_382 = None
    permute_383: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_811, [1, 0])
    mm_46: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_383, view_523);  permute_383 = view_523 = None
    permute_384: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_46, [1, 0]);  mm_46 = None
    sum_99: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_811, [0], True);  view_811 = None
    view_812: "f32[1536]" = torch.ops.aten.reshape.default(sum_99, [1536]);  sum_99 = None
    permute_385: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_384, [1, 0]);  permute_384 = None
    view_813: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_45, [32, 49, 512]);  mm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_814: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(view_813, [32, 7, 7, 512]);  view_813 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_815: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_814, [8, 2, 2, 7, 7, 512]);  view_814 = None
    permute_386: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_815, [0, 1, 3, 2, 4, 5]);  view_815 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_281: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_386, memory_format = torch.contiguous_format);  permute_386 = None
    view_816: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_281, [8, 14, 14, 512]);  clone_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    roll_25: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(view_816, [3, 3], [2, 1]);  view_816 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    mul_386: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(roll_25, primals_263);  primals_263 = None
    mul_387: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_386, 512)
    sum_100: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_386, [3], True)
    mul_388: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_386, mul_194);  mul_386 = None
    sum_101: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_388, [3], True);  mul_388 = None
    mul_389: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_194, sum_101);  sum_101 = None
    sub_116: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_387, sum_100);  mul_387 = sum_100 = None
    sub_117: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_116, mul_389);  sub_116 = mul_389 = None
    mul_390: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_82, sub_117);  div_82 = sub_117 = None
    mul_391: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(roll_25, mul_194);  mul_194 = None
    sum_102: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_391, [0, 1, 2]);  mul_391 = None
    sum_103: "f32[512]" = torch.ops.aten.sum.dim_IntList(roll_25, [0, 1, 2]);  roll_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_232: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_792, mul_390);  view_792 = mul_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_817: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(add_232, [8, 196, 512]);  add_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_392: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_817, div_54);  div_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_818: "f32[1568, 512]" = torch.ops.aten.reshape.default(mul_392, [1568, 512]);  mul_392 = None
    mm_47: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_818, permute_387);  permute_387 = None
    permute_388: "f32[512, 1568]" = torch.ops.aten.permute.default(view_818, [1, 0])
    mm_48: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_388, view_517);  permute_388 = view_517 = None
    permute_389: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_48, [1, 0]);  mm_48 = None
    sum_104: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_818, [0], True);  view_818 = None
    view_819: "f32[512]" = torch.ops.aten.reshape.default(sum_104, [512]);  sum_104 = None
    permute_390: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_389, [1, 0]);  permute_389 = None
    view_820: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(mm_47, [8, 196, 2048]);  mm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_394: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(add_165, 0.5);  add_165 = None
    mul_395: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_516, view_516)
    mul_396: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_395, -0.5);  mul_395 = None
    exp_29: "f32[8, 196, 2048]" = torch.ops.aten.exp.default(mul_396);  mul_396 = None
    mul_397: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
    mul_398: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_516, mul_397);  view_516 = mul_397 = None
    add_234: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(mul_394, mul_398);  mul_394 = mul_398 = None
    mul_399: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_820, add_234);  view_820 = add_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_821: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_399, [1568, 2048]);  mul_399 = None
    mm_49: "f32[1568, 512]" = torch.ops.aten.mm.default(view_821, permute_391);  permute_391 = None
    permute_392: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_821, [1, 0])
    mm_50: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_392, view_515);  permute_392 = view_515 = None
    permute_393: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_50, [1, 0]);  mm_50 = None
    sum_105: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_821, [0], True);  view_821 = None
    view_822: "f32[2048]" = torch.ops.aten.reshape.default(sum_105, [2048]);  sum_105 = None
    permute_394: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_393, [1, 0]);  permute_393 = None
    view_823: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(mm_49, [8, 196, 512]);  mm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_401: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_823, primals_257);  primals_257 = None
    mul_402: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_401, 512)
    sum_106: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_401, [2], True)
    mul_403: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_401, mul_188);  mul_401 = None
    sum_107: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_403, [2], True);  mul_403 = None
    mul_404: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_188, sum_107);  sum_107 = None
    sub_119: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(mul_402, sum_106);  mul_402 = sum_106 = None
    sub_120: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(sub_119, mul_404);  sub_119 = mul_404 = None
    mul_405: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(div_83, sub_120);  div_83 = sub_120 = None
    mul_406: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_823, mul_188);  mul_188 = None
    sum_108: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_406, [0, 1]);  mul_406 = None
    sum_109: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_823, [0, 1]);  view_823 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_235: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_817, mul_405);  view_817 = mul_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_824: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(add_235, [8, 14, 14, 512]);  add_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_407: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_824, div_53);  div_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_825: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(mul_407, [8, 2, 7, 2, 7, 512]);  mul_407 = None
    permute_395: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_825, [0, 1, 3, 2, 4, 5]);  view_825 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_282: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_395, memory_format = torch.contiguous_format);  permute_395 = None
    view_826: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_282, [32, 7, 7, 512]);  clone_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_827: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(view_826, [32, 49, 512]);  view_826 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_828: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_827, [1568, 512]);  view_827 = None
    mm_51: "f32[1568, 512]" = torch.ops.aten.mm.default(view_828, permute_396);  permute_396 = None
    permute_397: "f32[512, 1568]" = torch.ops.aten.permute.default(view_828, [1, 0])
    mm_52: "f32[512, 512]" = torch.ops.aten.mm.default(permute_397, view_509);  permute_397 = view_509 = None
    permute_398: "f32[512, 512]" = torch.ops.aten.permute.default(mm_52, [1, 0]);  mm_52 = None
    sum_110: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_828, [0], True);  view_828 = None
    view_829: "f32[512]" = torch.ops.aten.reshape.default(sum_110, [512]);  sum_110 = None
    permute_399: "f32[512, 512]" = torch.ops.aten.permute.default(permute_398, [1, 0]);  permute_398 = None
    view_830: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_51, [32, 49, 512]);  mm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_831: "f32[32, 49, 16, 32]" = torch.ops.aten.reshape.default(view_830, [32, 49, 16, 32]);  view_830 = None
    permute_400: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_831, [0, 2, 1, 3]);  view_831 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_283: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(permute_400, memory_format = torch.contiguous_format);  permute_400 = None
    view_832: "f32[512, 49, 32]" = torch.ops.aten.reshape.default(clone_283, [512, 49, 32]);  clone_283 = None
    bmm_68: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(permute_401, view_832);  permute_401 = None
    bmm_69: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_832, permute_402);  view_832 = permute_402 = None
    view_833: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_68, [32, 16, 49, 32]);  bmm_68 = None
    view_834: "f32[32, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_69, [32, 16, 49, 49]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    mul_408: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_834, alias_29);  view_834 = None
    sum_111: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_408, [-1], True)
    mul_409: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(alias_29, sum_111);  alias_29 = sum_111 = None
    sub_121: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(mul_408, mul_409);  mul_408 = mul_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_112: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(sub_121, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_5: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_112, 0);  sum_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_403: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_5, [1, 2, 0]);  squeeze_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_835: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_403, [2401, 16]);  permute_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_put_5: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_8, [view_503], view_835, True);  view_503 = view_835 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_836: "f32[512, 49, 49]" = torch.ops.aten.reshape.default(sub_121, [512, 49, 49]);  sub_121 = None
    bmm_70: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(permute_404, view_836);  permute_404 = None
    bmm_71: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_836, permute_405);  view_836 = permute_405 = None
    view_837: "f32[32, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_70, [32, 16, 32, 49]);  bmm_70 = None
    view_838: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_71, [32, 16, 49, 32]);  bmm_71 = None
    permute_406: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_837, [0, 1, 3, 2]);  view_837 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_410: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_838, 0.1767766952966369);  view_838 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    cat_5: "f32[96, 16, 49, 32]" = torch.ops.aten.cat.default([mul_410, permute_406, view_833]);  mul_410 = permute_406 = view_833 = None
    view_839: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_5, [3, 32, 16, 49, 32]);  cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_407: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_839, [1, 3, 0, 2, 4]);  view_839 = None
    clone_284: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_407, memory_format = torch.contiguous_format);  permute_407 = None
    view_840: "f32[32, 49, 1536]" = torch.ops.aten.reshape.default(clone_284, [32, 49, 1536]);  clone_284 = None
    view_841: "f32[1568, 1536]" = torch.ops.aten.reshape.default(view_840, [1568, 1536]);  view_840 = None
    mm_53: "f32[1568, 512]" = torch.ops.aten.mm.default(view_841, permute_408);  permute_408 = None
    permute_409: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_841, [1, 0])
    mm_54: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_409, view_497);  permute_409 = view_497 = None
    permute_410: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_54, [1, 0]);  mm_54 = None
    sum_113: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_841, [0], True);  view_841 = None
    view_842: "f32[1536]" = torch.ops.aten.reshape.default(sum_113, [1536]);  sum_113 = None
    permute_411: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_410, [1, 0]);  permute_410 = None
    view_843: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_53, [32, 49, 512]);  mm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_844: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(view_843, [32, 7, 7, 512]);  view_843 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_845: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_844, [8, 2, 2, 7, 7, 512]);  view_844 = None
    permute_412: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_845, [0, 1, 3, 2, 4, 5]);  view_845 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_285: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_412, memory_format = torch.contiguous_format);  permute_412 = None
    view_846: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_285, [8, 14, 14, 512]);  clone_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    mul_412: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_846, primals_251);  primals_251 = None
    mul_413: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_412, 512)
    sum_114: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_412, [3], True)
    mul_414: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_412, mul_184);  mul_412 = None
    sum_115: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_414, [3], True);  mul_414 = None
    mul_415: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_184, sum_115);  sum_115 = None
    sub_123: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_413, sum_114);  mul_413 = sum_114 = None
    sub_124: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_123, mul_415);  sub_123 = mul_415 = None
    mul_416: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_84, sub_124);  div_84 = sub_124 = None
    mul_417: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_846, mul_184);  mul_184 = None
    sum_116: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_417, [0, 1, 2]);  mul_417 = None
    sum_117: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_846, [0, 1, 2]);  view_846 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_236: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_824, mul_416);  view_824 = mul_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_847: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(add_236, [8, 196, 512]);  add_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_418: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_847, div_51);  div_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_848: "f32[1568, 512]" = torch.ops.aten.reshape.default(mul_418, [1568, 512]);  mul_418 = None
    mm_55: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_848, permute_413);  permute_413 = None
    permute_414: "f32[512, 1568]" = torch.ops.aten.permute.default(view_848, [1, 0])
    mm_56: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_414, view_491);  permute_414 = view_491 = None
    permute_415: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_56, [1, 0]);  mm_56 = None
    sum_118: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_848, [0], True);  view_848 = None
    view_849: "f32[512]" = torch.ops.aten.reshape.default(sum_118, [512]);  sum_118 = None
    permute_416: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_415, [1, 0]);  permute_415 = None
    view_850: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(mm_55, [8, 196, 2048]);  mm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_420: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(add_157, 0.5);  add_157 = None
    mul_421: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_490, view_490)
    mul_422: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_421, -0.5);  mul_421 = None
    exp_30: "f32[8, 196, 2048]" = torch.ops.aten.exp.default(mul_422);  mul_422 = None
    mul_423: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_30, 0.3989422804014327);  exp_30 = None
    mul_424: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_490, mul_423);  view_490 = mul_423 = None
    add_238: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(mul_420, mul_424);  mul_420 = mul_424 = None
    mul_425: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_850, add_238);  view_850 = add_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_851: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_425, [1568, 2048]);  mul_425 = None
    mm_57: "f32[1568, 512]" = torch.ops.aten.mm.default(view_851, permute_417);  permute_417 = None
    permute_418: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_851, [1, 0])
    mm_58: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_418, view_489);  permute_418 = view_489 = None
    permute_419: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_58, [1, 0]);  mm_58 = None
    sum_119: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_851, [0], True);  view_851 = None
    view_852: "f32[2048]" = torch.ops.aten.reshape.default(sum_119, [2048]);  sum_119 = None
    permute_420: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_419, [1, 0]);  permute_419 = None
    view_853: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(mm_57, [8, 196, 512]);  mm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_427: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_853, primals_245);  primals_245 = None
    mul_428: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_427, 512)
    sum_120: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_427, [2], True)
    mul_429: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_427, mul_178);  mul_427 = None
    sum_121: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_429, [2], True);  mul_429 = None
    mul_430: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_178, sum_121);  sum_121 = None
    sub_126: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(mul_428, sum_120);  mul_428 = sum_120 = None
    sub_127: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(sub_126, mul_430);  sub_126 = mul_430 = None
    mul_431: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(div_85, sub_127);  div_85 = sub_127 = None
    mul_432: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_853, mul_178);  mul_178 = None
    sum_122: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_432, [0, 1]);  mul_432 = None
    sum_123: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_853, [0, 1]);  view_853 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_239: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_847, mul_431);  view_847 = mul_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_854: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(add_239, [8, 14, 14, 512]);  add_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_433: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_854, div_50);  div_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    roll_26: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(mul_433, [-3, -3], [2, 1]);  mul_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_855: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(roll_26, [8, 2, 7, 2, 7, 512]);  roll_26 = None
    permute_421: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_855, [0, 1, 3, 2, 4, 5]);  view_855 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_286: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_421, memory_format = torch.contiguous_format);  permute_421 = None
    view_856: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_286, [32, 7, 7, 512]);  clone_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_857: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(view_856, [32, 49, 512]);  view_856 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_858: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_857, [1568, 512]);  view_857 = None
    mm_59: "f32[1568, 512]" = torch.ops.aten.mm.default(view_858, permute_422);  permute_422 = None
    permute_423: "f32[512, 1568]" = torch.ops.aten.permute.default(view_858, [1, 0])
    mm_60: "f32[512, 512]" = torch.ops.aten.mm.default(permute_423, view_483);  permute_423 = view_483 = None
    permute_424: "f32[512, 512]" = torch.ops.aten.permute.default(mm_60, [1, 0]);  mm_60 = None
    sum_124: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_858, [0], True);  view_858 = None
    view_859: "f32[512]" = torch.ops.aten.reshape.default(sum_124, [512]);  sum_124 = None
    permute_425: "f32[512, 512]" = torch.ops.aten.permute.default(permute_424, [1, 0]);  permute_424 = None
    view_860: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_59, [32, 49, 512]);  mm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_861: "f32[32, 49, 16, 32]" = torch.ops.aten.reshape.default(view_860, [32, 49, 16, 32]);  view_860 = None
    permute_426: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_861, [0, 2, 1, 3]);  view_861 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_287: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(permute_426, memory_format = torch.contiguous_format);  permute_426 = None
    view_862: "f32[512, 49, 32]" = torch.ops.aten.reshape.default(clone_287, [512, 49, 32]);  clone_287 = None
    bmm_72: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(permute_427, view_862);  permute_427 = None
    bmm_73: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_862, permute_428);  view_862 = permute_428 = None
    view_863: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_72, [32, 16, 49, 32]);  bmm_72 = None
    view_864: "f32[32, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_73, [32, 16, 49, 49]);  bmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    mul_434: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_864, alias_30);  view_864 = None
    sum_125: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_434, [-1], True)
    mul_435: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(alias_30, sum_125);  alias_30 = sum_125 = None
    sub_128: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(mul_434, mul_435);  mul_434 = mul_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    view_865: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.reshape.default(sub_128, [8, 4, 16, 49, 49]);  sub_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_866: "f32[32, 16, 49, 49]" = torch.ops.aten.reshape.default(view_865, [32, 16, 49, 49]);  view_865 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_126: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(view_866, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_6: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_126, 0);  sum_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_429: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_6, [1, 2, 0]);  squeeze_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_867: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_429, [2401, 16]);  permute_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_put_6: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_8, [view_475], view_867, True);  view_475 = view_867 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_868: "f32[512, 49, 49]" = torch.ops.aten.reshape.default(view_866, [512, 49, 49]);  view_866 = None
    bmm_74: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(permute_430, view_868);  permute_430 = None
    bmm_75: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_868, permute_431);  view_868 = permute_431 = None
    view_869: "f32[32, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_74, [32, 16, 32, 49]);  bmm_74 = None
    view_870: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_75, [32, 16, 49, 32]);  bmm_75 = None
    permute_432: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_869, [0, 1, 3, 2]);  view_869 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_436: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_870, 0.1767766952966369);  view_870 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    cat_6: "f32[96, 16, 49, 32]" = torch.ops.aten.cat.default([mul_436, permute_432, view_863]);  mul_436 = permute_432 = view_863 = None
    view_871: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_6, [3, 32, 16, 49, 32]);  cat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_433: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_871, [1, 3, 0, 2, 4]);  view_871 = None
    clone_288: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_433, memory_format = torch.contiguous_format);  permute_433 = None
    view_872: "f32[32, 49, 1536]" = torch.ops.aten.reshape.default(clone_288, [32, 49, 1536]);  clone_288 = None
    view_873: "f32[1568, 1536]" = torch.ops.aten.reshape.default(view_872, [1568, 1536]);  view_872 = None
    mm_61: "f32[1568, 512]" = torch.ops.aten.mm.default(view_873, permute_434);  permute_434 = None
    permute_435: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_873, [1, 0])
    mm_62: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_435, view_469);  permute_435 = view_469 = None
    permute_436: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_62, [1, 0]);  mm_62 = None
    sum_127: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_873, [0], True);  view_873 = None
    view_874: "f32[1536]" = torch.ops.aten.reshape.default(sum_127, [1536]);  sum_127 = None
    permute_437: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_436, [1, 0]);  permute_436 = None
    view_875: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_61, [32, 49, 512]);  mm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_876: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(view_875, [32, 7, 7, 512]);  view_875 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_877: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_876, [8, 2, 2, 7, 7, 512]);  view_876 = None
    permute_438: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_877, [0, 1, 3, 2, 4, 5]);  view_877 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_289: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_438, memory_format = torch.contiguous_format);  permute_438 = None
    view_878: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_289, [8, 14, 14, 512]);  clone_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    roll_27: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(view_878, [3, 3], [2, 1]);  view_878 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    mul_438: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(roll_27, primals_239);  primals_239 = None
    mul_439: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_438, 512)
    sum_128: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_438, [3], True)
    mul_440: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_438, mul_174);  mul_438 = None
    sum_129: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_440, [3], True);  mul_440 = None
    mul_441: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_174, sum_129);  sum_129 = None
    sub_130: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_439, sum_128);  mul_439 = sum_128 = None
    sub_131: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_130, mul_441);  sub_130 = mul_441 = None
    mul_442: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_86, sub_131);  div_86 = sub_131 = None
    mul_443: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(roll_27, mul_174);  mul_174 = None
    sum_130: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_443, [0, 1, 2]);  mul_443 = None
    sum_131: "f32[512]" = torch.ops.aten.sum.dim_IntList(roll_27, [0, 1, 2]);  roll_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_240: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_854, mul_442);  view_854 = mul_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_879: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(add_240, [8, 196, 512]);  add_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_444: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_879, div_48);  div_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_880: "f32[1568, 512]" = torch.ops.aten.reshape.default(mul_444, [1568, 512]);  mul_444 = None
    mm_63: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_880, permute_439);  permute_439 = None
    permute_440: "f32[512, 1568]" = torch.ops.aten.permute.default(view_880, [1, 0])
    mm_64: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_440, view_463);  permute_440 = view_463 = None
    permute_441: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_64, [1, 0]);  mm_64 = None
    sum_132: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_880, [0], True);  view_880 = None
    view_881: "f32[512]" = torch.ops.aten.reshape.default(sum_132, [512]);  sum_132 = None
    permute_442: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_441, [1, 0]);  permute_441 = None
    view_882: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(mm_63, [8, 196, 2048]);  mm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_446: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(add_148, 0.5);  add_148 = None
    mul_447: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_462, view_462)
    mul_448: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_447, -0.5);  mul_447 = None
    exp_31: "f32[8, 196, 2048]" = torch.ops.aten.exp.default(mul_448);  mul_448 = None
    mul_449: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_31, 0.3989422804014327);  exp_31 = None
    mul_450: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_462, mul_449);  view_462 = mul_449 = None
    add_242: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(mul_446, mul_450);  mul_446 = mul_450 = None
    mul_451: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_882, add_242);  view_882 = add_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_883: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_451, [1568, 2048]);  mul_451 = None
    mm_65: "f32[1568, 512]" = torch.ops.aten.mm.default(view_883, permute_443);  permute_443 = None
    permute_444: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_883, [1, 0])
    mm_66: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_444, view_461);  permute_444 = view_461 = None
    permute_445: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_66, [1, 0]);  mm_66 = None
    sum_133: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_883, [0], True);  view_883 = None
    view_884: "f32[2048]" = torch.ops.aten.reshape.default(sum_133, [2048]);  sum_133 = None
    permute_446: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_445, [1, 0]);  permute_445 = None
    view_885: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(mm_65, [8, 196, 512]);  mm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_453: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_885, primals_233);  primals_233 = None
    mul_454: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_453, 512)
    sum_134: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_453, [2], True)
    mul_455: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_453, mul_168);  mul_453 = None
    sum_135: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_455, [2], True);  mul_455 = None
    mul_456: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_168, sum_135);  sum_135 = None
    sub_133: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(mul_454, sum_134);  mul_454 = sum_134 = None
    sub_134: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(sub_133, mul_456);  sub_133 = mul_456 = None
    mul_457: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(div_87, sub_134);  div_87 = sub_134 = None
    mul_458: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_885, mul_168);  mul_168 = None
    sum_136: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_458, [0, 1]);  mul_458 = None
    sum_137: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_885, [0, 1]);  view_885 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_243: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_879, mul_457);  view_879 = mul_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_886: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(add_243, [8, 14, 14, 512]);  add_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_459: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_886, div_47);  div_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_887: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(mul_459, [8, 2, 7, 2, 7, 512]);  mul_459 = None
    permute_447: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_887, [0, 1, 3, 2, 4, 5]);  view_887 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_290: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_447, memory_format = torch.contiguous_format);  permute_447 = None
    view_888: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_290, [32, 7, 7, 512]);  clone_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_889: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(view_888, [32, 49, 512]);  view_888 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_890: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_889, [1568, 512]);  view_889 = None
    mm_67: "f32[1568, 512]" = torch.ops.aten.mm.default(view_890, permute_448);  permute_448 = None
    permute_449: "f32[512, 1568]" = torch.ops.aten.permute.default(view_890, [1, 0])
    mm_68: "f32[512, 512]" = torch.ops.aten.mm.default(permute_449, view_455);  permute_449 = view_455 = None
    permute_450: "f32[512, 512]" = torch.ops.aten.permute.default(mm_68, [1, 0]);  mm_68 = None
    sum_138: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_890, [0], True);  view_890 = None
    view_891: "f32[512]" = torch.ops.aten.reshape.default(sum_138, [512]);  sum_138 = None
    permute_451: "f32[512, 512]" = torch.ops.aten.permute.default(permute_450, [1, 0]);  permute_450 = None
    view_892: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_67, [32, 49, 512]);  mm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_893: "f32[32, 49, 16, 32]" = torch.ops.aten.reshape.default(view_892, [32, 49, 16, 32]);  view_892 = None
    permute_452: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_893, [0, 2, 1, 3]);  view_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_291: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(permute_452, memory_format = torch.contiguous_format);  permute_452 = None
    view_894: "f32[512, 49, 32]" = torch.ops.aten.reshape.default(clone_291, [512, 49, 32]);  clone_291 = None
    bmm_76: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(permute_453, view_894);  permute_453 = None
    bmm_77: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_894, permute_454);  view_894 = permute_454 = None
    view_895: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_76, [32, 16, 49, 32]);  bmm_76 = None
    view_896: "f32[32, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_77, [32, 16, 49, 49]);  bmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    mul_460: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_896, alias_31);  view_896 = None
    sum_139: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_460, [-1], True)
    mul_461: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(alias_31, sum_139);  alias_31 = sum_139 = None
    sub_135: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(mul_460, mul_461);  mul_460 = mul_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_140: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(sub_135, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_7: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_140, 0);  sum_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_455: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_7, [1, 2, 0]);  squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_897: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_455, [2401, 16]);  permute_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_put_7: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_8, [view_449], view_897, True);  view_449 = view_897 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_898: "f32[512, 49, 49]" = torch.ops.aten.reshape.default(sub_135, [512, 49, 49]);  sub_135 = None
    bmm_78: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(permute_456, view_898);  permute_456 = None
    bmm_79: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_898, permute_457);  view_898 = permute_457 = None
    view_899: "f32[32, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_78, [32, 16, 32, 49]);  bmm_78 = None
    view_900: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_79, [32, 16, 49, 32]);  bmm_79 = None
    permute_458: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_899, [0, 1, 3, 2]);  view_899 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_462: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_900, 0.1767766952966369);  view_900 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    cat_7: "f32[96, 16, 49, 32]" = torch.ops.aten.cat.default([mul_462, permute_458, view_895]);  mul_462 = permute_458 = view_895 = None
    view_901: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_7, [3, 32, 16, 49, 32]);  cat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_459: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_901, [1, 3, 0, 2, 4]);  view_901 = None
    clone_292: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_459, memory_format = torch.contiguous_format);  permute_459 = None
    view_902: "f32[32, 49, 1536]" = torch.ops.aten.reshape.default(clone_292, [32, 49, 1536]);  clone_292 = None
    view_903: "f32[1568, 1536]" = torch.ops.aten.reshape.default(view_902, [1568, 1536]);  view_902 = None
    mm_69: "f32[1568, 512]" = torch.ops.aten.mm.default(view_903, permute_460);  permute_460 = None
    permute_461: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_903, [1, 0])
    mm_70: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_461, view_443);  permute_461 = view_443 = None
    permute_462: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_70, [1, 0]);  mm_70 = None
    sum_141: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_903, [0], True);  view_903 = None
    view_904: "f32[1536]" = torch.ops.aten.reshape.default(sum_141, [1536]);  sum_141 = None
    permute_463: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_462, [1, 0]);  permute_462 = None
    view_905: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_69, [32, 49, 512]);  mm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_906: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(view_905, [32, 7, 7, 512]);  view_905 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_907: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_906, [8, 2, 2, 7, 7, 512]);  view_906 = None
    permute_464: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_907, [0, 1, 3, 2, 4, 5]);  view_907 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_293: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_464, memory_format = torch.contiguous_format);  permute_464 = None
    view_908: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_293, [8, 14, 14, 512]);  clone_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    mul_464: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_908, primals_227);  primals_227 = None
    mul_465: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_464, 512)
    sum_142: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_464, [3], True)
    mul_466: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_464, mul_164);  mul_464 = None
    sum_143: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_466, [3], True);  mul_466 = None
    mul_467: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_164, sum_143);  sum_143 = None
    sub_137: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_465, sum_142);  mul_465 = sum_142 = None
    sub_138: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_137, mul_467);  sub_137 = mul_467 = None
    mul_468: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_88, sub_138);  div_88 = sub_138 = None
    mul_469: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_908, mul_164);  mul_164 = None
    sum_144: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_469, [0, 1, 2]);  mul_469 = None
    sum_145: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_908, [0, 1, 2]);  view_908 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_244: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_886, mul_468);  view_886 = mul_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_909: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(add_244, [8, 196, 512]);  add_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_470: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_909, div_45);  div_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_910: "f32[1568, 512]" = torch.ops.aten.reshape.default(mul_470, [1568, 512]);  mul_470 = None
    mm_71: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_910, permute_465);  permute_465 = None
    permute_466: "f32[512, 1568]" = torch.ops.aten.permute.default(view_910, [1, 0])
    mm_72: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_466, view_437);  permute_466 = view_437 = None
    permute_467: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_72, [1, 0]);  mm_72 = None
    sum_146: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_910, [0], True);  view_910 = None
    view_911: "f32[512]" = torch.ops.aten.reshape.default(sum_146, [512]);  sum_146 = None
    permute_468: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_467, [1, 0]);  permute_467 = None
    view_912: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(mm_71, [8, 196, 2048]);  mm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_472: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(add_140, 0.5);  add_140 = None
    mul_473: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_436, view_436)
    mul_474: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_473, -0.5);  mul_473 = None
    exp_32: "f32[8, 196, 2048]" = torch.ops.aten.exp.default(mul_474);  mul_474 = None
    mul_475: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_32, 0.3989422804014327);  exp_32 = None
    mul_476: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_436, mul_475);  view_436 = mul_475 = None
    add_246: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(mul_472, mul_476);  mul_472 = mul_476 = None
    mul_477: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_912, add_246);  view_912 = add_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_913: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_477, [1568, 2048]);  mul_477 = None
    mm_73: "f32[1568, 512]" = torch.ops.aten.mm.default(view_913, permute_469);  permute_469 = None
    permute_470: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_913, [1, 0])
    mm_74: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_470, view_435);  permute_470 = view_435 = None
    permute_471: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_74, [1, 0]);  mm_74 = None
    sum_147: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_913, [0], True);  view_913 = None
    view_914: "f32[2048]" = torch.ops.aten.reshape.default(sum_147, [2048]);  sum_147 = None
    permute_472: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_471, [1, 0]);  permute_471 = None
    view_915: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(mm_73, [8, 196, 512]);  mm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_479: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_915, primals_221);  primals_221 = None
    mul_480: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_479, 512)
    sum_148: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_479, [2], True)
    mul_481: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_479, mul_158);  mul_479 = None
    sum_149: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_481, [2], True);  mul_481 = None
    mul_482: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_158, sum_149);  sum_149 = None
    sub_140: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(mul_480, sum_148);  mul_480 = sum_148 = None
    sub_141: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(sub_140, mul_482);  sub_140 = mul_482 = None
    mul_483: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(div_89, sub_141);  div_89 = sub_141 = None
    mul_484: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_915, mul_158);  mul_158 = None
    sum_150: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_484, [0, 1]);  mul_484 = None
    sum_151: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_915, [0, 1]);  view_915 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_247: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_909, mul_483);  view_909 = mul_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_916: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(add_247, [8, 14, 14, 512]);  add_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_485: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_916, div_44);  div_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    roll_28: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(mul_485, [-3, -3], [2, 1]);  mul_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_917: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(roll_28, [8, 2, 7, 2, 7, 512]);  roll_28 = None
    permute_473: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_917, [0, 1, 3, 2, 4, 5]);  view_917 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_294: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_473, memory_format = torch.contiguous_format);  permute_473 = None
    view_918: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_294, [32, 7, 7, 512]);  clone_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_919: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(view_918, [32, 49, 512]);  view_918 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_920: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_919, [1568, 512]);  view_919 = None
    mm_75: "f32[1568, 512]" = torch.ops.aten.mm.default(view_920, permute_474);  permute_474 = None
    permute_475: "f32[512, 1568]" = torch.ops.aten.permute.default(view_920, [1, 0])
    mm_76: "f32[512, 512]" = torch.ops.aten.mm.default(permute_475, view_429);  permute_475 = view_429 = None
    permute_476: "f32[512, 512]" = torch.ops.aten.permute.default(mm_76, [1, 0]);  mm_76 = None
    sum_152: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_920, [0], True);  view_920 = None
    view_921: "f32[512]" = torch.ops.aten.reshape.default(sum_152, [512]);  sum_152 = None
    permute_477: "f32[512, 512]" = torch.ops.aten.permute.default(permute_476, [1, 0]);  permute_476 = None
    view_922: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_75, [32, 49, 512]);  mm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_923: "f32[32, 49, 16, 32]" = torch.ops.aten.reshape.default(view_922, [32, 49, 16, 32]);  view_922 = None
    permute_478: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_923, [0, 2, 1, 3]);  view_923 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_295: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(permute_478, memory_format = torch.contiguous_format);  permute_478 = None
    view_924: "f32[512, 49, 32]" = torch.ops.aten.reshape.default(clone_295, [512, 49, 32]);  clone_295 = None
    bmm_80: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(permute_479, view_924);  permute_479 = None
    bmm_81: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_924, permute_480);  view_924 = permute_480 = None
    view_925: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_80, [32, 16, 49, 32]);  bmm_80 = None
    view_926: "f32[32, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_81, [32, 16, 49, 49]);  bmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    mul_486: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_926, alias_32);  view_926 = None
    sum_153: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_486, [-1], True)
    mul_487: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(alias_32, sum_153);  alias_32 = sum_153 = None
    sub_142: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(mul_486, mul_487);  mul_486 = mul_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    view_927: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.reshape.default(sub_142, [8, 4, 16, 49, 49]);  sub_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_928: "f32[32, 16, 49, 49]" = torch.ops.aten.reshape.default(view_927, [32, 16, 49, 49]);  view_927 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_154: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(view_928, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_8: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_154, 0);  sum_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_481: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_8, [1, 2, 0]);  squeeze_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_929: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_481, [2401, 16]);  permute_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_put_8: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_8, [view_421], view_929, True);  view_421 = view_929 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_930: "f32[512, 49, 49]" = torch.ops.aten.reshape.default(view_928, [512, 49, 49]);  view_928 = None
    bmm_82: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(permute_482, view_930);  permute_482 = None
    bmm_83: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_930, permute_483);  view_930 = permute_483 = None
    view_931: "f32[32, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_82, [32, 16, 32, 49]);  bmm_82 = None
    view_932: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_83, [32, 16, 49, 32]);  bmm_83 = None
    permute_484: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_931, [0, 1, 3, 2]);  view_931 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_488: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_932, 0.1767766952966369);  view_932 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    cat_8: "f32[96, 16, 49, 32]" = torch.ops.aten.cat.default([mul_488, permute_484, view_925]);  mul_488 = permute_484 = view_925 = None
    view_933: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_8, [3, 32, 16, 49, 32]);  cat_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_485: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_933, [1, 3, 0, 2, 4]);  view_933 = None
    clone_296: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_485, memory_format = torch.contiguous_format);  permute_485 = None
    view_934: "f32[32, 49, 1536]" = torch.ops.aten.reshape.default(clone_296, [32, 49, 1536]);  clone_296 = None
    view_935: "f32[1568, 1536]" = torch.ops.aten.reshape.default(view_934, [1568, 1536]);  view_934 = None
    mm_77: "f32[1568, 512]" = torch.ops.aten.mm.default(view_935, permute_486);  permute_486 = None
    permute_487: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_935, [1, 0])
    mm_78: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_487, view_415);  permute_487 = view_415 = None
    permute_488: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_78, [1, 0]);  mm_78 = None
    sum_155: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_935, [0], True);  view_935 = None
    view_936: "f32[1536]" = torch.ops.aten.reshape.default(sum_155, [1536]);  sum_155 = None
    permute_489: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_488, [1, 0]);  permute_488 = None
    view_937: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_77, [32, 49, 512]);  mm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_938: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(view_937, [32, 7, 7, 512]);  view_937 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_939: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_938, [8, 2, 2, 7, 7, 512]);  view_938 = None
    permute_490: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_939, [0, 1, 3, 2, 4, 5]);  view_939 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_297: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_490, memory_format = torch.contiguous_format);  permute_490 = None
    view_940: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_297, [8, 14, 14, 512]);  clone_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    roll_29: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(view_940, [3, 3], [2, 1]);  view_940 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    mul_490: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(roll_29, primals_215);  primals_215 = None
    mul_491: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_490, 512)
    sum_156: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_490, [3], True)
    mul_492: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_490, mul_154);  mul_490 = None
    sum_157: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_492, [3], True);  mul_492 = None
    mul_493: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_154, sum_157);  sum_157 = None
    sub_144: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_491, sum_156);  mul_491 = sum_156 = None
    sub_145: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_144, mul_493);  sub_144 = mul_493 = None
    mul_494: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_90, sub_145);  div_90 = sub_145 = None
    mul_495: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(roll_29, mul_154);  mul_154 = None
    sum_158: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_495, [0, 1, 2]);  mul_495 = None
    sum_159: "f32[512]" = torch.ops.aten.sum.dim_IntList(roll_29, [0, 1, 2]);  roll_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_248: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_916, mul_494);  view_916 = mul_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_941: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(add_248, [8, 196, 512]);  add_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_496: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_941, div_42);  div_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_942: "f32[1568, 512]" = torch.ops.aten.reshape.default(mul_496, [1568, 512]);  mul_496 = None
    mm_79: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_942, permute_491);  permute_491 = None
    permute_492: "f32[512, 1568]" = torch.ops.aten.permute.default(view_942, [1, 0])
    mm_80: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_492, view_409);  permute_492 = view_409 = None
    permute_493: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_80, [1, 0]);  mm_80 = None
    sum_160: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_942, [0], True);  view_942 = None
    view_943: "f32[512]" = torch.ops.aten.reshape.default(sum_160, [512]);  sum_160 = None
    permute_494: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_493, [1, 0]);  permute_493 = None
    view_944: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(mm_79, [8, 196, 2048]);  mm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_498: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(add_131, 0.5);  add_131 = None
    mul_499: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_408, view_408)
    mul_500: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_499, -0.5);  mul_499 = None
    exp_33: "f32[8, 196, 2048]" = torch.ops.aten.exp.default(mul_500);  mul_500 = None
    mul_501: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_33, 0.3989422804014327);  exp_33 = None
    mul_502: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_408, mul_501);  view_408 = mul_501 = None
    add_250: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(mul_498, mul_502);  mul_498 = mul_502 = None
    mul_503: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_944, add_250);  view_944 = add_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_945: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_503, [1568, 2048]);  mul_503 = None
    mm_81: "f32[1568, 512]" = torch.ops.aten.mm.default(view_945, permute_495);  permute_495 = None
    permute_496: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_945, [1, 0])
    mm_82: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_496, view_407);  permute_496 = view_407 = None
    permute_497: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_82, [1, 0]);  mm_82 = None
    sum_161: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_945, [0], True);  view_945 = None
    view_946: "f32[2048]" = torch.ops.aten.reshape.default(sum_161, [2048]);  sum_161 = None
    permute_498: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_497, [1, 0]);  permute_497 = None
    view_947: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(mm_81, [8, 196, 512]);  mm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_505: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_947, primals_209);  primals_209 = None
    mul_506: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_505, 512)
    sum_162: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_505, [2], True)
    mul_507: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_505, mul_148);  mul_505 = None
    sum_163: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_507, [2], True);  mul_507 = None
    mul_508: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_148, sum_163);  sum_163 = None
    sub_147: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(mul_506, sum_162);  mul_506 = sum_162 = None
    sub_148: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(sub_147, mul_508);  sub_147 = mul_508 = None
    mul_509: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(div_91, sub_148);  div_91 = sub_148 = None
    mul_510: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_947, mul_148);  mul_148 = None
    sum_164: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_510, [0, 1]);  mul_510 = None
    sum_165: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_947, [0, 1]);  view_947 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_251: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_941, mul_509);  view_941 = mul_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_948: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(add_251, [8, 14, 14, 512]);  add_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_511: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_948, div_41);  div_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_949: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(mul_511, [8, 2, 7, 2, 7, 512]);  mul_511 = None
    permute_499: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_949, [0, 1, 3, 2, 4, 5]);  view_949 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_298: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_499, memory_format = torch.contiguous_format);  permute_499 = None
    view_950: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_298, [32, 7, 7, 512]);  clone_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_951: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(view_950, [32, 49, 512]);  view_950 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_952: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_951, [1568, 512]);  view_951 = None
    mm_83: "f32[1568, 512]" = torch.ops.aten.mm.default(view_952, permute_500);  permute_500 = None
    permute_501: "f32[512, 1568]" = torch.ops.aten.permute.default(view_952, [1, 0])
    mm_84: "f32[512, 512]" = torch.ops.aten.mm.default(permute_501, view_401);  permute_501 = view_401 = None
    permute_502: "f32[512, 512]" = torch.ops.aten.permute.default(mm_84, [1, 0]);  mm_84 = None
    sum_166: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_952, [0], True);  view_952 = None
    view_953: "f32[512]" = torch.ops.aten.reshape.default(sum_166, [512]);  sum_166 = None
    permute_503: "f32[512, 512]" = torch.ops.aten.permute.default(permute_502, [1, 0]);  permute_502 = None
    view_954: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_83, [32, 49, 512]);  mm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_955: "f32[32, 49, 16, 32]" = torch.ops.aten.reshape.default(view_954, [32, 49, 16, 32]);  view_954 = None
    permute_504: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_955, [0, 2, 1, 3]);  view_955 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_299: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(permute_504, memory_format = torch.contiguous_format);  permute_504 = None
    view_956: "f32[512, 49, 32]" = torch.ops.aten.reshape.default(clone_299, [512, 49, 32]);  clone_299 = None
    bmm_84: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(permute_505, view_956);  permute_505 = None
    bmm_85: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_956, permute_506);  view_956 = permute_506 = None
    view_957: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_84, [32, 16, 49, 32]);  bmm_84 = None
    view_958: "f32[32, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_85, [32, 16, 49, 49]);  bmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    mul_512: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_958, alias_33);  view_958 = None
    sum_167: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_512, [-1], True)
    mul_513: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(alias_33, sum_167);  alias_33 = sum_167 = None
    sub_149: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(mul_512, mul_513);  mul_512 = mul_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_168: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(sub_149, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_9: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_168, 0);  sum_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_507: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_9, [1, 2, 0]);  squeeze_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_959: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_507, [2401, 16]);  permute_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_put_9: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_8, [view_395], view_959, True);  view_395 = view_959 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_960: "f32[512, 49, 49]" = torch.ops.aten.reshape.default(sub_149, [512, 49, 49]);  sub_149 = None
    bmm_86: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(permute_508, view_960);  permute_508 = None
    bmm_87: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_960, permute_509);  view_960 = permute_509 = None
    view_961: "f32[32, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_86, [32, 16, 32, 49]);  bmm_86 = None
    view_962: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_87, [32, 16, 49, 32]);  bmm_87 = None
    permute_510: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_961, [0, 1, 3, 2]);  view_961 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_514: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_962, 0.1767766952966369);  view_962 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    cat_9: "f32[96, 16, 49, 32]" = torch.ops.aten.cat.default([mul_514, permute_510, view_957]);  mul_514 = permute_510 = view_957 = None
    view_963: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_9, [3, 32, 16, 49, 32]);  cat_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_511: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_963, [1, 3, 0, 2, 4]);  view_963 = None
    clone_300: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_511, memory_format = torch.contiguous_format);  permute_511 = None
    view_964: "f32[32, 49, 1536]" = torch.ops.aten.reshape.default(clone_300, [32, 49, 1536]);  clone_300 = None
    view_965: "f32[1568, 1536]" = torch.ops.aten.reshape.default(view_964, [1568, 1536]);  view_964 = None
    mm_85: "f32[1568, 512]" = torch.ops.aten.mm.default(view_965, permute_512);  permute_512 = None
    permute_513: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_965, [1, 0])
    mm_86: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_513, view_389);  permute_513 = view_389 = None
    permute_514: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_86, [1, 0]);  mm_86 = None
    sum_169: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_965, [0], True);  view_965 = None
    view_966: "f32[1536]" = torch.ops.aten.reshape.default(sum_169, [1536]);  sum_169 = None
    permute_515: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_514, [1, 0]);  permute_514 = None
    view_967: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_85, [32, 49, 512]);  mm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_968: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(view_967, [32, 7, 7, 512]);  view_967 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_969: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_968, [8, 2, 2, 7, 7, 512]);  view_968 = None
    permute_516: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_969, [0, 1, 3, 2, 4, 5]);  view_969 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_301: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_516, memory_format = torch.contiguous_format);  permute_516 = None
    view_970: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_301, [8, 14, 14, 512]);  clone_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    mul_516: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_970, primals_203);  primals_203 = None
    mul_517: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_516, 512)
    sum_170: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_516, [3], True)
    mul_518: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_516, mul_144);  mul_516 = None
    sum_171: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_518, [3], True);  mul_518 = None
    mul_519: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_144, sum_171);  sum_171 = None
    sub_151: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_517, sum_170);  mul_517 = sum_170 = None
    sub_152: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_151, mul_519);  sub_151 = mul_519 = None
    mul_520: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_92, sub_152);  div_92 = sub_152 = None
    mul_521: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_970, mul_144);  mul_144 = None
    sum_172: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_521, [0, 1, 2]);  mul_521 = None
    sum_173: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_970, [0, 1, 2]);  view_970 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_252: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_948, mul_520);  view_948 = mul_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_971: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(add_252, [8, 196, 512]);  add_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_522: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_971, div_39);  div_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_972: "f32[1568, 512]" = torch.ops.aten.reshape.default(mul_522, [1568, 512]);  mul_522 = None
    mm_87: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_972, permute_517);  permute_517 = None
    permute_518: "f32[512, 1568]" = torch.ops.aten.permute.default(view_972, [1, 0])
    mm_88: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_518, view_383);  permute_518 = view_383 = None
    permute_519: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_88, [1, 0]);  mm_88 = None
    sum_174: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_972, [0], True);  view_972 = None
    view_973: "f32[512]" = torch.ops.aten.reshape.default(sum_174, [512]);  sum_174 = None
    permute_520: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_519, [1, 0]);  permute_519 = None
    view_974: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(mm_87, [8, 196, 2048]);  mm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_524: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(add_123, 0.5);  add_123 = None
    mul_525: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_382, view_382)
    mul_526: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_525, -0.5);  mul_525 = None
    exp_34: "f32[8, 196, 2048]" = torch.ops.aten.exp.default(mul_526);  mul_526 = None
    mul_527: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_34, 0.3989422804014327);  exp_34 = None
    mul_528: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_382, mul_527);  view_382 = mul_527 = None
    add_254: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(mul_524, mul_528);  mul_524 = mul_528 = None
    mul_529: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_974, add_254);  view_974 = add_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_975: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_529, [1568, 2048]);  mul_529 = None
    mm_89: "f32[1568, 512]" = torch.ops.aten.mm.default(view_975, permute_521);  permute_521 = None
    permute_522: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_975, [1, 0])
    mm_90: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_522, view_381);  permute_522 = view_381 = None
    permute_523: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_90, [1, 0]);  mm_90 = None
    sum_175: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_975, [0], True);  view_975 = None
    view_976: "f32[2048]" = torch.ops.aten.reshape.default(sum_175, [2048]);  sum_175 = None
    permute_524: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_523, [1, 0]);  permute_523 = None
    view_977: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(mm_89, [8, 196, 512]);  mm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_531: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_977, primals_197);  primals_197 = None
    mul_532: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_531, 512)
    sum_176: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_531, [2], True)
    mul_533: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_531, mul_138);  mul_531 = None
    sum_177: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_533, [2], True);  mul_533 = None
    mul_534: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_138, sum_177);  sum_177 = None
    sub_154: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(mul_532, sum_176);  mul_532 = sum_176 = None
    sub_155: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(sub_154, mul_534);  sub_154 = mul_534 = None
    mul_535: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(div_93, sub_155);  div_93 = sub_155 = None
    mul_536: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_977, mul_138);  mul_138 = None
    sum_178: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_536, [0, 1]);  mul_536 = None
    sum_179: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_977, [0, 1]);  view_977 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_255: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_971, mul_535);  view_971 = mul_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_978: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(add_255, [8, 14, 14, 512]);  add_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_537: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_978, div_38);  div_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    roll_30: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(mul_537, [-3, -3], [2, 1]);  mul_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_979: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(roll_30, [8, 2, 7, 2, 7, 512]);  roll_30 = None
    permute_525: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_979, [0, 1, 3, 2, 4, 5]);  view_979 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_302: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_525, memory_format = torch.contiguous_format);  permute_525 = None
    view_980: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_302, [32, 7, 7, 512]);  clone_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_981: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(view_980, [32, 49, 512]);  view_980 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_982: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_981, [1568, 512]);  view_981 = None
    mm_91: "f32[1568, 512]" = torch.ops.aten.mm.default(view_982, permute_526);  permute_526 = None
    permute_527: "f32[512, 1568]" = torch.ops.aten.permute.default(view_982, [1, 0])
    mm_92: "f32[512, 512]" = torch.ops.aten.mm.default(permute_527, view_375);  permute_527 = view_375 = None
    permute_528: "f32[512, 512]" = torch.ops.aten.permute.default(mm_92, [1, 0]);  mm_92 = None
    sum_180: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_982, [0], True);  view_982 = None
    view_983: "f32[512]" = torch.ops.aten.reshape.default(sum_180, [512]);  sum_180 = None
    permute_529: "f32[512, 512]" = torch.ops.aten.permute.default(permute_528, [1, 0]);  permute_528 = None
    view_984: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_91, [32, 49, 512]);  mm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_985: "f32[32, 49, 16, 32]" = torch.ops.aten.reshape.default(view_984, [32, 49, 16, 32]);  view_984 = None
    permute_530: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_985, [0, 2, 1, 3]);  view_985 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_303: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(permute_530, memory_format = torch.contiguous_format);  permute_530 = None
    view_986: "f32[512, 49, 32]" = torch.ops.aten.reshape.default(clone_303, [512, 49, 32]);  clone_303 = None
    bmm_88: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(permute_531, view_986);  permute_531 = None
    bmm_89: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_986, permute_532);  view_986 = permute_532 = None
    view_987: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_88, [32, 16, 49, 32]);  bmm_88 = None
    view_988: "f32[32, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_89, [32, 16, 49, 49]);  bmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    mul_538: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_988, alias_34);  view_988 = None
    sum_181: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_538, [-1], True)
    mul_539: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(alias_34, sum_181);  alias_34 = sum_181 = None
    sub_156: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(mul_538, mul_539);  mul_538 = mul_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    view_989: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.reshape.default(sub_156, [8, 4, 16, 49, 49]);  sub_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_990: "f32[32, 16, 49, 49]" = torch.ops.aten.reshape.default(view_989, [32, 16, 49, 49]);  view_989 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_182: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(view_990, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_10: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_182, 0);  sum_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_533: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_10, [1, 2, 0]);  squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_991: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_533, [2401, 16]);  permute_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_put_10: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_8, [view_367], view_991, True);  view_367 = view_991 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_992: "f32[512, 49, 49]" = torch.ops.aten.reshape.default(view_990, [512, 49, 49]);  view_990 = None
    bmm_90: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(permute_534, view_992);  permute_534 = None
    bmm_91: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_992, permute_535);  view_992 = permute_535 = None
    view_993: "f32[32, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_90, [32, 16, 32, 49]);  bmm_90 = None
    view_994: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_91, [32, 16, 49, 32]);  bmm_91 = None
    permute_536: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_993, [0, 1, 3, 2]);  view_993 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_540: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_994, 0.1767766952966369);  view_994 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    cat_10: "f32[96, 16, 49, 32]" = torch.ops.aten.cat.default([mul_540, permute_536, view_987]);  mul_540 = permute_536 = view_987 = None
    view_995: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_10, [3, 32, 16, 49, 32]);  cat_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_537: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_995, [1, 3, 0, 2, 4]);  view_995 = None
    clone_304: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_537, memory_format = torch.contiguous_format);  permute_537 = None
    view_996: "f32[32, 49, 1536]" = torch.ops.aten.reshape.default(clone_304, [32, 49, 1536]);  clone_304 = None
    view_997: "f32[1568, 1536]" = torch.ops.aten.reshape.default(view_996, [1568, 1536]);  view_996 = None
    mm_93: "f32[1568, 512]" = torch.ops.aten.mm.default(view_997, permute_538);  permute_538 = None
    permute_539: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_997, [1, 0])
    mm_94: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_539, view_361);  permute_539 = view_361 = None
    permute_540: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_94, [1, 0]);  mm_94 = None
    sum_183: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_997, [0], True);  view_997 = None
    view_998: "f32[1536]" = torch.ops.aten.reshape.default(sum_183, [1536]);  sum_183 = None
    permute_541: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_540, [1, 0]);  permute_540 = None
    view_999: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_93, [32, 49, 512]);  mm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_1000: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(view_999, [32, 7, 7, 512]);  view_999 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_1001: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1000, [8, 2, 2, 7, 7, 512]);  view_1000 = None
    permute_542: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1001, [0, 1, 3, 2, 4, 5]);  view_1001 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_305: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_542, memory_format = torch.contiguous_format);  permute_542 = None
    view_1002: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_305, [8, 14, 14, 512]);  clone_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    roll_31: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(view_1002, [3, 3], [2, 1]);  view_1002 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    mul_542: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(roll_31, primals_191);  primals_191 = None
    mul_543: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_542, 512)
    sum_184: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_542, [3], True)
    mul_544: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_542, mul_134);  mul_542 = None
    sum_185: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_544, [3], True);  mul_544 = None
    mul_545: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_134, sum_185);  sum_185 = None
    sub_158: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_543, sum_184);  mul_543 = sum_184 = None
    sub_159: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_158, mul_545);  sub_158 = mul_545 = None
    mul_546: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_94, sub_159);  div_94 = sub_159 = None
    mul_547: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(roll_31, mul_134);  mul_134 = None
    sum_186: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_547, [0, 1, 2]);  mul_547 = None
    sum_187: "f32[512]" = torch.ops.aten.sum.dim_IntList(roll_31, [0, 1, 2]);  roll_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_256: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_978, mul_546);  view_978 = mul_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_1003: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(add_256, [8, 196, 512]);  add_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_548: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_1003, div_36);  div_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1004: "f32[1568, 512]" = torch.ops.aten.reshape.default(mul_548, [1568, 512]);  mul_548 = None
    mm_95: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_1004, permute_543);  permute_543 = None
    permute_544: "f32[512, 1568]" = torch.ops.aten.permute.default(view_1004, [1, 0])
    mm_96: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_544, view_355);  permute_544 = view_355 = None
    permute_545: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_96, [1, 0]);  mm_96 = None
    sum_188: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1004, [0], True);  view_1004 = None
    view_1005: "f32[512]" = torch.ops.aten.reshape.default(sum_188, [512]);  sum_188 = None
    permute_546: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_545, [1, 0]);  permute_545 = None
    view_1006: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(mm_95, [8, 196, 2048]);  mm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_550: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(add_114, 0.5);  add_114 = None
    mul_551: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_354, view_354)
    mul_552: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_551, -0.5);  mul_551 = None
    exp_35: "f32[8, 196, 2048]" = torch.ops.aten.exp.default(mul_552);  mul_552 = None
    mul_553: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_35, 0.3989422804014327);  exp_35 = None
    mul_554: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_354, mul_553);  view_354 = mul_553 = None
    add_258: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(mul_550, mul_554);  mul_550 = mul_554 = None
    mul_555: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1006, add_258);  view_1006 = add_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1007: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_555, [1568, 2048]);  mul_555 = None
    mm_97: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1007, permute_547);  permute_547 = None
    permute_548: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_1007, [1, 0])
    mm_98: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_548, view_353);  permute_548 = view_353 = None
    permute_549: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_98, [1, 0]);  mm_98 = None
    sum_189: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1007, [0], True);  view_1007 = None
    view_1008: "f32[2048]" = torch.ops.aten.reshape.default(sum_189, [2048]);  sum_189 = None
    permute_550: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_549, [1, 0]);  permute_549 = None
    view_1009: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(mm_97, [8, 196, 512]);  mm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_557: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_1009, primals_185);  primals_185 = None
    mul_558: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_557, 512)
    sum_190: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_557, [2], True)
    mul_559: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_557, mul_128);  mul_557 = None
    sum_191: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_559, [2], True);  mul_559 = None
    mul_560: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_128, sum_191);  sum_191 = None
    sub_161: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(mul_558, sum_190);  mul_558 = sum_190 = None
    sub_162: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(sub_161, mul_560);  sub_161 = mul_560 = None
    mul_561: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(div_95, sub_162);  div_95 = sub_162 = None
    mul_562: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_1009, mul_128);  mul_128 = None
    sum_192: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_562, [0, 1]);  mul_562 = None
    sum_193: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1009, [0, 1]);  view_1009 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_259: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_1003, mul_561);  view_1003 = mul_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_1010: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(add_259, [8, 14, 14, 512]);  add_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_563: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1010, div_35);  div_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_1011: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(mul_563, [8, 2, 7, 2, 7, 512]);  mul_563 = None
    permute_551: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_1011, [0, 1, 3, 2, 4, 5]);  view_1011 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_306: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_551, memory_format = torch.contiguous_format);  permute_551 = None
    view_1012: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_306, [32, 7, 7, 512]);  clone_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_1013: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(view_1012, [32, 49, 512]);  view_1012 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_1014: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_1013, [1568, 512]);  view_1013 = None
    mm_99: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1014, permute_552);  permute_552 = None
    permute_553: "f32[512, 1568]" = torch.ops.aten.permute.default(view_1014, [1, 0])
    mm_100: "f32[512, 512]" = torch.ops.aten.mm.default(permute_553, view_347);  permute_553 = view_347 = None
    permute_554: "f32[512, 512]" = torch.ops.aten.permute.default(mm_100, [1, 0]);  mm_100 = None
    sum_194: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1014, [0], True);  view_1014 = None
    view_1015: "f32[512]" = torch.ops.aten.reshape.default(sum_194, [512]);  sum_194 = None
    permute_555: "f32[512, 512]" = torch.ops.aten.permute.default(permute_554, [1, 0]);  permute_554 = None
    view_1016: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_99, [32, 49, 512]);  mm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_1017: "f32[32, 49, 16, 32]" = torch.ops.aten.reshape.default(view_1016, [32, 49, 16, 32]);  view_1016 = None
    permute_556: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_1017, [0, 2, 1, 3]);  view_1017 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_307: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(permute_556, memory_format = torch.contiguous_format);  permute_556 = None
    view_1018: "f32[512, 49, 32]" = torch.ops.aten.reshape.default(clone_307, [512, 49, 32]);  clone_307 = None
    bmm_92: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(permute_557, view_1018);  permute_557 = None
    bmm_93: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_1018, permute_558);  view_1018 = permute_558 = None
    view_1019: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_92, [32, 16, 49, 32]);  bmm_92 = None
    view_1020: "f32[32, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_93, [32, 16, 49, 49]);  bmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    mul_564: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_1020, alias_35);  view_1020 = None
    sum_195: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_564, [-1], True)
    mul_565: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(alias_35, sum_195);  alias_35 = sum_195 = None
    sub_163: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(mul_564, mul_565);  mul_564 = mul_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_196: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(sub_163, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_11: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_196, 0);  sum_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_559: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_11, [1, 2, 0]);  squeeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_1021: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_559, [2401, 16]);  permute_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_put_11: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_8, [view_341], view_1021, True);  view_341 = view_1021 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_1022: "f32[512, 49, 49]" = torch.ops.aten.reshape.default(sub_163, [512, 49, 49]);  sub_163 = None
    bmm_94: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(permute_560, view_1022);  permute_560 = None
    bmm_95: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_1022, permute_561);  view_1022 = permute_561 = None
    view_1023: "f32[32, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_94, [32, 16, 32, 49]);  bmm_94 = None
    view_1024: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_95, [32, 16, 49, 32]);  bmm_95 = None
    permute_562: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_1023, [0, 1, 3, 2]);  view_1023 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_566: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_1024, 0.1767766952966369);  view_1024 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    cat_11: "f32[96, 16, 49, 32]" = torch.ops.aten.cat.default([mul_566, permute_562, view_1019]);  mul_566 = permute_562 = view_1019 = None
    view_1025: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_11, [3, 32, 16, 49, 32]);  cat_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_563: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_1025, [1, 3, 0, 2, 4]);  view_1025 = None
    clone_308: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_563, memory_format = torch.contiguous_format);  permute_563 = None
    view_1026: "f32[32, 49, 1536]" = torch.ops.aten.reshape.default(clone_308, [32, 49, 1536]);  clone_308 = None
    view_1027: "f32[1568, 1536]" = torch.ops.aten.reshape.default(view_1026, [1568, 1536]);  view_1026 = None
    mm_101: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1027, permute_564);  permute_564 = None
    permute_565: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_1027, [1, 0])
    mm_102: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_565, view_335);  permute_565 = view_335 = None
    permute_566: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_102, [1, 0]);  mm_102 = None
    sum_197: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1027, [0], True);  view_1027 = None
    view_1028: "f32[1536]" = torch.ops.aten.reshape.default(sum_197, [1536]);  sum_197 = None
    permute_567: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_566, [1, 0]);  permute_566 = None
    view_1029: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_101, [32, 49, 512]);  mm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_1030: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1029, [32, 7, 7, 512]);  view_1029 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_1031: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1030, [8, 2, 2, 7, 7, 512]);  view_1030 = None
    permute_568: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1031, [0, 1, 3, 2, 4, 5]);  view_1031 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_309: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_568, memory_format = torch.contiguous_format);  permute_568 = None
    view_1032: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_309, [8, 14, 14, 512]);  clone_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    mul_568: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1032, primals_179);  primals_179 = None
    mul_569: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_568, 512)
    sum_198: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_568, [3], True)
    mul_570: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_568, mul_124);  mul_568 = None
    sum_199: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_570, [3], True);  mul_570 = None
    mul_571: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_124, sum_199);  sum_199 = None
    sub_165: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_569, sum_198);  mul_569 = sum_198 = None
    sub_166: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_165, mul_571);  sub_165 = mul_571 = None
    mul_572: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_96, sub_166);  div_96 = sub_166 = None
    mul_573: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1032, mul_124);  mul_124 = None
    sum_200: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_573, [0, 1, 2]);  mul_573 = None
    sum_201: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1032, [0, 1, 2]);  view_1032 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_260: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_1010, mul_572);  view_1010 = mul_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_1033: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(add_260, [8, 196, 512]);  add_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_574: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_1033, div_33);  div_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1034: "f32[1568, 512]" = torch.ops.aten.reshape.default(mul_574, [1568, 512]);  mul_574 = None
    mm_103: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_1034, permute_569);  permute_569 = None
    permute_570: "f32[512, 1568]" = torch.ops.aten.permute.default(view_1034, [1, 0])
    mm_104: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_570, view_329);  permute_570 = view_329 = None
    permute_571: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_104, [1, 0]);  mm_104 = None
    sum_202: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1034, [0], True);  view_1034 = None
    view_1035: "f32[512]" = torch.ops.aten.reshape.default(sum_202, [512]);  sum_202 = None
    permute_572: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_571, [1, 0]);  permute_571 = None
    view_1036: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(mm_103, [8, 196, 2048]);  mm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_576: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(add_106, 0.5);  add_106 = None
    mul_577: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_328, view_328)
    mul_578: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_577, -0.5);  mul_577 = None
    exp_36: "f32[8, 196, 2048]" = torch.ops.aten.exp.default(mul_578);  mul_578 = None
    mul_579: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_36, 0.3989422804014327);  exp_36 = None
    mul_580: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_328, mul_579);  view_328 = mul_579 = None
    add_262: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(mul_576, mul_580);  mul_576 = mul_580 = None
    mul_581: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1036, add_262);  view_1036 = add_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1037: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_581, [1568, 2048]);  mul_581 = None
    mm_105: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1037, permute_573);  permute_573 = None
    permute_574: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_1037, [1, 0])
    mm_106: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_574, view_327);  permute_574 = view_327 = None
    permute_575: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_106, [1, 0]);  mm_106 = None
    sum_203: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1037, [0], True);  view_1037 = None
    view_1038: "f32[2048]" = torch.ops.aten.reshape.default(sum_203, [2048]);  sum_203 = None
    permute_576: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_575, [1, 0]);  permute_575 = None
    view_1039: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(mm_105, [8, 196, 512]);  mm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_583: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_1039, primals_173);  primals_173 = None
    mul_584: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_583, 512)
    sum_204: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_583, [2], True)
    mul_585: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_583, mul_118);  mul_583 = None
    sum_205: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_585, [2], True);  mul_585 = None
    mul_586: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_118, sum_205);  sum_205 = None
    sub_168: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(mul_584, sum_204);  mul_584 = sum_204 = None
    sub_169: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(sub_168, mul_586);  sub_168 = mul_586 = None
    mul_587: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(div_97, sub_169);  div_97 = sub_169 = None
    mul_588: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_1039, mul_118);  mul_118 = None
    sum_206: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_588, [0, 1]);  mul_588 = None
    sum_207: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1039, [0, 1]);  view_1039 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_263: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_1033, mul_587);  view_1033 = mul_587 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_1040: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(add_263, [8, 14, 14, 512]);  add_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_589: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1040, div_32);  div_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    roll_32: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(mul_589, [-3, -3], [2, 1]);  mul_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_1041: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(roll_32, [8, 2, 7, 2, 7, 512]);  roll_32 = None
    permute_577: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_1041, [0, 1, 3, 2, 4, 5]);  view_1041 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_310: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_577, memory_format = torch.contiguous_format);  permute_577 = None
    view_1042: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_310, [32, 7, 7, 512]);  clone_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_1043: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(view_1042, [32, 49, 512]);  view_1042 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_1044: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_1043, [1568, 512]);  view_1043 = None
    mm_107: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1044, permute_578);  permute_578 = None
    permute_579: "f32[512, 1568]" = torch.ops.aten.permute.default(view_1044, [1, 0])
    mm_108: "f32[512, 512]" = torch.ops.aten.mm.default(permute_579, view_321);  permute_579 = view_321 = None
    permute_580: "f32[512, 512]" = torch.ops.aten.permute.default(mm_108, [1, 0]);  mm_108 = None
    sum_208: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1044, [0], True);  view_1044 = None
    view_1045: "f32[512]" = torch.ops.aten.reshape.default(sum_208, [512]);  sum_208 = None
    permute_581: "f32[512, 512]" = torch.ops.aten.permute.default(permute_580, [1, 0]);  permute_580 = None
    view_1046: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_107, [32, 49, 512]);  mm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_1047: "f32[32, 49, 16, 32]" = torch.ops.aten.reshape.default(view_1046, [32, 49, 16, 32]);  view_1046 = None
    permute_582: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_1047, [0, 2, 1, 3]);  view_1047 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_311: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(permute_582, memory_format = torch.contiguous_format);  permute_582 = None
    view_1048: "f32[512, 49, 32]" = torch.ops.aten.reshape.default(clone_311, [512, 49, 32]);  clone_311 = None
    bmm_96: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(permute_583, view_1048);  permute_583 = None
    bmm_97: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_1048, permute_584);  view_1048 = permute_584 = None
    view_1049: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_96, [32, 16, 49, 32]);  bmm_96 = None
    view_1050: "f32[32, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_97, [32, 16, 49, 49]);  bmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    mul_590: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_1050, alias_36);  view_1050 = None
    sum_209: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_590, [-1], True)
    mul_591: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(alias_36, sum_209);  alias_36 = sum_209 = None
    sub_170: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(mul_590, mul_591);  mul_590 = mul_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    view_1051: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.reshape.default(sub_170, [8, 4, 16, 49, 49]);  sub_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_1052: "f32[32, 16, 49, 49]" = torch.ops.aten.reshape.default(view_1051, [32, 16, 49, 49]);  view_1051 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_210: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(view_1052, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_12: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_210, 0);  sum_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_585: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_12, [1, 2, 0]);  squeeze_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_1053: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_585, [2401, 16]);  permute_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_put_12: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_8, [view_313], view_1053, True);  view_313 = view_1053 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_1054: "f32[512, 49, 49]" = torch.ops.aten.reshape.default(view_1052, [512, 49, 49]);  view_1052 = None
    bmm_98: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(permute_586, view_1054);  permute_586 = None
    bmm_99: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_1054, permute_587);  view_1054 = permute_587 = None
    view_1055: "f32[32, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_98, [32, 16, 32, 49]);  bmm_98 = None
    view_1056: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_99, [32, 16, 49, 32]);  bmm_99 = None
    permute_588: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_1055, [0, 1, 3, 2]);  view_1055 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_592: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_1056, 0.1767766952966369);  view_1056 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    cat_12: "f32[96, 16, 49, 32]" = torch.ops.aten.cat.default([mul_592, permute_588, view_1049]);  mul_592 = permute_588 = view_1049 = None
    view_1057: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_12, [3, 32, 16, 49, 32]);  cat_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_589: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_1057, [1, 3, 0, 2, 4]);  view_1057 = None
    clone_312: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_589, memory_format = torch.contiguous_format);  permute_589 = None
    view_1058: "f32[32, 49, 1536]" = torch.ops.aten.reshape.default(clone_312, [32, 49, 1536]);  clone_312 = None
    view_1059: "f32[1568, 1536]" = torch.ops.aten.reshape.default(view_1058, [1568, 1536]);  view_1058 = None
    mm_109: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1059, permute_590);  permute_590 = None
    permute_591: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_1059, [1, 0])
    mm_110: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_591, view_307);  permute_591 = view_307 = None
    permute_592: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_110, [1, 0]);  mm_110 = None
    sum_211: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1059, [0], True);  view_1059 = None
    view_1060: "f32[1536]" = torch.ops.aten.reshape.default(sum_211, [1536]);  sum_211 = None
    permute_593: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_592, [1, 0]);  permute_592 = None
    view_1061: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_109, [32, 49, 512]);  mm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_1062: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1061, [32, 7, 7, 512]);  view_1061 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_1063: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1062, [8, 2, 2, 7, 7, 512]);  view_1062 = None
    permute_594: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1063, [0, 1, 3, 2, 4, 5]);  view_1063 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_313: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_594, memory_format = torch.contiguous_format);  permute_594 = None
    view_1064: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_313, [8, 14, 14, 512]);  clone_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    roll_33: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(view_1064, [3, 3], [2, 1]);  view_1064 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    mul_594: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(roll_33, primals_167);  primals_167 = None
    mul_595: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_594, 512)
    sum_212: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_594, [3], True)
    mul_596: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_594, mul_114);  mul_594 = None
    sum_213: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_596, [3], True);  mul_596 = None
    mul_597: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_114, sum_213);  sum_213 = None
    sub_172: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_595, sum_212);  mul_595 = sum_212 = None
    sub_173: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_172, mul_597);  sub_172 = mul_597 = None
    mul_598: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_98, sub_173);  div_98 = sub_173 = None
    mul_599: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(roll_33, mul_114);  mul_114 = None
    sum_214: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_599, [0, 1, 2]);  mul_599 = None
    sum_215: "f32[512]" = torch.ops.aten.sum.dim_IntList(roll_33, [0, 1, 2]);  roll_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_264: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_1040, mul_598);  view_1040 = mul_598 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_1065: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(add_264, [8, 196, 512]);  add_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_600: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_1065, div_30);  div_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1066: "f32[1568, 512]" = torch.ops.aten.reshape.default(mul_600, [1568, 512]);  mul_600 = None
    mm_111: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_1066, permute_595);  permute_595 = None
    permute_596: "f32[512, 1568]" = torch.ops.aten.permute.default(view_1066, [1, 0])
    mm_112: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_596, view_301);  permute_596 = view_301 = None
    permute_597: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_112, [1, 0]);  mm_112 = None
    sum_216: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1066, [0], True);  view_1066 = None
    view_1067: "f32[512]" = torch.ops.aten.reshape.default(sum_216, [512]);  sum_216 = None
    permute_598: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_597, [1, 0]);  permute_597 = None
    view_1068: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(mm_111, [8, 196, 2048]);  mm_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_602: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(add_97, 0.5);  add_97 = None
    mul_603: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_300, view_300)
    mul_604: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_603, -0.5);  mul_603 = None
    exp_37: "f32[8, 196, 2048]" = torch.ops.aten.exp.default(mul_604);  mul_604 = None
    mul_605: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_37, 0.3989422804014327);  exp_37 = None
    mul_606: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_300, mul_605);  view_300 = mul_605 = None
    add_266: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(mul_602, mul_606);  mul_602 = mul_606 = None
    mul_607: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1068, add_266);  view_1068 = add_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1069: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_607, [1568, 2048]);  mul_607 = None
    mm_113: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1069, permute_599);  permute_599 = None
    permute_600: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_1069, [1, 0])
    mm_114: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_600, view_299);  permute_600 = view_299 = None
    permute_601: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_114, [1, 0]);  mm_114 = None
    sum_217: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1069, [0], True);  view_1069 = None
    view_1070: "f32[2048]" = torch.ops.aten.reshape.default(sum_217, [2048]);  sum_217 = None
    permute_602: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_601, [1, 0]);  permute_601 = None
    view_1071: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(mm_113, [8, 196, 512]);  mm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_609: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_1071, primals_161);  primals_161 = None
    mul_610: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_609, 512)
    sum_218: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_609, [2], True)
    mul_611: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_609, mul_108);  mul_609 = None
    sum_219: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_611, [2], True);  mul_611 = None
    mul_612: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_108, sum_219);  sum_219 = None
    sub_175: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(mul_610, sum_218);  mul_610 = sum_218 = None
    sub_176: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(sub_175, mul_612);  sub_175 = mul_612 = None
    mul_613: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(div_99, sub_176);  div_99 = sub_176 = None
    mul_614: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_1071, mul_108);  mul_108 = None
    sum_220: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_614, [0, 1]);  mul_614 = None
    sum_221: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1071, [0, 1]);  view_1071 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_267: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_1065, mul_613);  view_1065 = mul_613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_1072: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(add_267, [8, 14, 14, 512]);  add_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_615: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1072, div_29);  div_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_1073: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(mul_615, [8, 2, 7, 2, 7, 512]);  mul_615 = None
    permute_603: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_1073, [0, 1, 3, 2, 4, 5]);  view_1073 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_314: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_603, memory_format = torch.contiguous_format);  permute_603 = None
    view_1074: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_314, [32, 7, 7, 512]);  clone_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_1075: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(view_1074, [32, 49, 512]);  view_1074 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_1076: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_1075, [1568, 512]);  view_1075 = None
    mm_115: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1076, permute_604);  permute_604 = None
    permute_605: "f32[512, 1568]" = torch.ops.aten.permute.default(view_1076, [1, 0])
    mm_116: "f32[512, 512]" = torch.ops.aten.mm.default(permute_605, view_293);  permute_605 = view_293 = None
    permute_606: "f32[512, 512]" = torch.ops.aten.permute.default(mm_116, [1, 0]);  mm_116 = None
    sum_222: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1076, [0], True);  view_1076 = None
    view_1077: "f32[512]" = torch.ops.aten.reshape.default(sum_222, [512]);  sum_222 = None
    permute_607: "f32[512, 512]" = torch.ops.aten.permute.default(permute_606, [1, 0]);  permute_606 = None
    view_1078: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_115, [32, 49, 512]);  mm_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_1079: "f32[32, 49, 16, 32]" = torch.ops.aten.reshape.default(view_1078, [32, 49, 16, 32]);  view_1078 = None
    permute_608: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_1079, [0, 2, 1, 3]);  view_1079 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_315: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(permute_608, memory_format = torch.contiguous_format);  permute_608 = None
    view_1080: "f32[512, 49, 32]" = torch.ops.aten.reshape.default(clone_315, [512, 49, 32]);  clone_315 = None
    bmm_100: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(permute_609, view_1080);  permute_609 = None
    bmm_101: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_1080, permute_610);  view_1080 = permute_610 = None
    view_1081: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_100, [32, 16, 49, 32]);  bmm_100 = None
    view_1082: "f32[32, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_101, [32, 16, 49, 49]);  bmm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    mul_616: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_1082, alias_37);  view_1082 = None
    sum_223: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_616, [-1], True)
    mul_617: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(alias_37, sum_223);  alias_37 = sum_223 = None
    sub_177: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(mul_616, mul_617);  mul_616 = mul_617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_224: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(sub_177, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_13: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_224, 0);  sum_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_611: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_13, [1, 2, 0]);  squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_1083: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_611, [2401, 16]);  permute_611 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_put_13: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_8, [view_287], view_1083, True);  view_287 = view_1083 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_1084: "f32[512, 49, 49]" = torch.ops.aten.reshape.default(sub_177, [512, 49, 49]);  sub_177 = None
    bmm_102: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(permute_612, view_1084);  permute_612 = None
    bmm_103: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_1084, permute_613);  view_1084 = permute_613 = None
    view_1085: "f32[32, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_102, [32, 16, 32, 49]);  bmm_102 = None
    view_1086: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_103, [32, 16, 49, 32]);  bmm_103 = None
    permute_614: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_1085, [0, 1, 3, 2]);  view_1085 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_618: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_1086, 0.1767766952966369);  view_1086 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    cat_13: "f32[96, 16, 49, 32]" = torch.ops.aten.cat.default([mul_618, permute_614, view_1081]);  mul_618 = permute_614 = view_1081 = None
    view_1087: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_13, [3, 32, 16, 49, 32]);  cat_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_615: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_1087, [1, 3, 0, 2, 4]);  view_1087 = None
    clone_316: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_615, memory_format = torch.contiguous_format);  permute_615 = None
    view_1088: "f32[32, 49, 1536]" = torch.ops.aten.reshape.default(clone_316, [32, 49, 1536]);  clone_316 = None
    view_1089: "f32[1568, 1536]" = torch.ops.aten.reshape.default(view_1088, [1568, 1536]);  view_1088 = None
    mm_117: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1089, permute_616);  permute_616 = None
    permute_617: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_1089, [1, 0])
    mm_118: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_617, view_281);  permute_617 = view_281 = None
    permute_618: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_118, [1, 0]);  mm_118 = None
    sum_225: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1089, [0], True);  view_1089 = None
    view_1090: "f32[1536]" = torch.ops.aten.reshape.default(sum_225, [1536]);  sum_225 = None
    permute_619: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_618, [1, 0]);  permute_618 = None
    view_1091: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_117, [32, 49, 512]);  mm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_1092: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1091, [32, 7, 7, 512]);  view_1091 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_1093: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1092, [8, 2, 2, 7, 7, 512]);  view_1092 = None
    permute_620: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1093, [0, 1, 3, 2, 4, 5]);  view_1093 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_317: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_620, memory_format = torch.contiguous_format);  permute_620 = None
    view_1094: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_317, [8, 14, 14, 512]);  clone_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    mul_620: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1094, primals_155);  primals_155 = None
    mul_621: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_620, 512)
    sum_226: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_620, [3], True)
    mul_622: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_620, mul_104);  mul_620 = None
    sum_227: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_622, [3], True);  mul_622 = None
    mul_623: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_104, sum_227);  sum_227 = None
    sub_179: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_621, sum_226);  mul_621 = sum_226 = None
    sub_180: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_179, mul_623);  sub_179 = mul_623 = None
    mul_624: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_100, sub_180);  div_100 = sub_180 = None
    mul_625: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1094, mul_104);  mul_104 = None
    sum_228: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_625, [0, 1, 2]);  mul_625 = None
    sum_229: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1094, [0, 1, 2]);  view_1094 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_268: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_1072, mul_624);  view_1072 = mul_624 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_1095: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(add_268, [8, 196, 512]);  add_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_626: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_1095, div_27);  div_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1096: "f32[1568, 512]" = torch.ops.aten.reshape.default(mul_626, [1568, 512]);  mul_626 = None
    mm_119: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_1096, permute_621);  permute_621 = None
    permute_622: "f32[512, 1568]" = torch.ops.aten.permute.default(view_1096, [1, 0])
    mm_120: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_622, view_275);  permute_622 = view_275 = None
    permute_623: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_120, [1, 0]);  mm_120 = None
    sum_230: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1096, [0], True);  view_1096 = None
    view_1097: "f32[512]" = torch.ops.aten.reshape.default(sum_230, [512]);  sum_230 = None
    permute_624: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_623, [1, 0]);  permute_623 = None
    view_1098: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(mm_119, [8, 196, 2048]);  mm_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_628: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(add_89, 0.5);  add_89 = None
    mul_629: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_274, view_274)
    mul_630: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_629, -0.5);  mul_629 = None
    exp_38: "f32[8, 196, 2048]" = torch.ops.aten.exp.default(mul_630);  mul_630 = None
    mul_631: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_38, 0.3989422804014327);  exp_38 = None
    mul_632: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_274, mul_631);  view_274 = mul_631 = None
    add_270: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(mul_628, mul_632);  mul_628 = mul_632 = None
    mul_633: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1098, add_270);  view_1098 = add_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1099: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_633, [1568, 2048]);  mul_633 = None
    mm_121: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1099, permute_625);  permute_625 = None
    permute_626: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_1099, [1, 0])
    mm_122: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_626, view_273);  permute_626 = view_273 = None
    permute_627: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_122, [1, 0]);  mm_122 = None
    sum_231: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1099, [0], True);  view_1099 = None
    view_1100: "f32[2048]" = torch.ops.aten.reshape.default(sum_231, [2048]);  sum_231 = None
    permute_628: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_627, [1, 0]);  permute_627 = None
    view_1101: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(mm_121, [8, 196, 512]);  mm_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_635: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_1101, primals_149);  primals_149 = None
    mul_636: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_635, 512)
    sum_232: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_635, [2], True)
    mul_637: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_635, mul_98);  mul_635 = None
    sum_233: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_637, [2], True);  mul_637 = None
    mul_638: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_98, sum_233);  sum_233 = None
    sub_182: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(mul_636, sum_232);  mul_636 = sum_232 = None
    sub_183: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(sub_182, mul_638);  sub_182 = mul_638 = None
    mul_639: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(div_101, sub_183);  div_101 = sub_183 = None
    mul_640: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_1101, mul_98);  mul_98 = None
    sum_234: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_640, [0, 1]);  mul_640 = None
    sum_235: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1101, [0, 1]);  view_1101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_271: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_1095, mul_639);  view_1095 = mul_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_1102: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(add_271, [8, 14, 14, 512]);  add_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_641: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1102, div_26);  div_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    roll_34: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(mul_641, [-3, -3], [2, 1]);  mul_641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_1103: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(roll_34, [8, 2, 7, 2, 7, 512]);  roll_34 = None
    permute_629: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_1103, [0, 1, 3, 2, 4, 5]);  view_1103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_318: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_629, memory_format = torch.contiguous_format);  permute_629 = None
    view_1104: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_318, [32, 7, 7, 512]);  clone_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_1105: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(view_1104, [32, 49, 512]);  view_1104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_1106: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_1105, [1568, 512]);  view_1105 = None
    mm_123: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1106, permute_630);  permute_630 = None
    permute_631: "f32[512, 1568]" = torch.ops.aten.permute.default(view_1106, [1, 0])
    mm_124: "f32[512, 512]" = torch.ops.aten.mm.default(permute_631, view_267);  permute_631 = view_267 = None
    permute_632: "f32[512, 512]" = torch.ops.aten.permute.default(mm_124, [1, 0]);  mm_124 = None
    sum_236: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1106, [0], True);  view_1106 = None
    view_1107: "f32[512]" = torch.ops.aten.reshape.default(sum_236, [512]);  sum_236 = None
    permute_633: "f32[512, 512]" = torch.ops.aten.permute.default(permute_632, [1, 0]);  permute_632 = None
    view_1108: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_123, [32, 49, 512]);  mm_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_1109: "f32[32, 49, 16, 32]" = torch.ops.aten.reshape.default(view_1108, [32, 49, 16, 32]);  view_1108 = None
    permute_634: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_1109, [0, 2, 1, 3]);  view_1109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_319: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(permute_634, memory_format = torch.contiguous_format);  permute_634 = None
    view_1110: "f32[512, 49, 32]" = torch.ops.aten.reshape.default(clone_319, [512, 49, 32]);  clone_319 = None
    bmm_104: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(permute_635, view_1110);  permute_635 = None
    bmm_105: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_1110, permute_636);  view_1110 = permute_636 = None
    view_1111: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_104, [32, 16, 49, 32]);  bmm_104 = None
    view_1112: "f32[32, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_105, [32, 16, 49, 49]);  bmm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    mul_642: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_1112, alias_38);  view_1112 = None
    sum_237: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_642, [-1], True)
    mul_643: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(alias_38, sum_237);  alias_38 = sum_237 = None
    sub_184: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(mul_642, mul_643);  mul_642 = mul_643 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    view_1113: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.reshape.default(sub_184, [8, 4, 16, 49, 49]);  sub_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_1114: "f32[32, 16, 49, 49]" = torch.ops.aten.reshape.default(view_1113, [32, 16, 49, 49]);  view_1113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_238: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(view_1114, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_14: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_238, 0);  sum_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_637: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_14, [1, 2, 0]);  squeeze_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_1115: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_637, [2401, 16]);  permute_637 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_put_14: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_8, [view_259], view_1115, True);  view_259 = view_1115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_1116: "f32[512, 49, 49]" = torch.ops.aten.reshape.default(view_1114, [512, 49, 49]);  view_1114 = None
    bmm_106: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(permute_638, view_1116);  permute_638 = None
    bmm_107: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_1116, permute_639);  view_1116 = permute_639 = None
    view_1117: "f32[32, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_106, [32, 16, 32, 49]);  bmm_106 = None
    view_1118: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_107, [32, 16, 49, 32]);  bmm_107 = None
    permute_640: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_1117, [0, 1, 3, 2]);  view_1117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_644: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_1118, 0.1767766952966369);  view_1118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    cat_14: "f32[96, 16, 49, 32]" = torch.ops.aten.cat.default([mul_644, permute_640, view_1111]);  mul_644 = permute_640 = view_1111 = None
    view_1119: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_14, [3, 32, 16, 49, 32]);  cat_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_641: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_1119, [1, 3, 0, 2, 4]);  view_1119 = None
    clone_320: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_641, memory_format = torch.contiguous_format);  permute_641 = None
    view_1120: "f32[32, 49, 1536]" = torch.ops.aten.reshape.default(clone_320, [32, 49, 1536]);  clone_320 = None
    view_1121: "f32[1568, 1536]" = torch.ops.aten.reshape.default(view_1120, [1568, 1536]);  view_1120 = None
    mm_125: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1121, permute_642);  permute_642 = None
    permute_643: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_1121, [1, 0])
    mm_126: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_643, view_253);  permute_643 = view_253 = None
    permute_644: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_126, [1, 0]);  mm_126 = None
    sum_239: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1121, [0], True);  view_1121 = None
    view_1122: "f32[1536]" = torch.ops.aten.reshape.default(sum_239, [1536]);  sum_239 = None
    permute_645: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_644, [1, 0]);  permute_644 = None
    view_1123: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_125, [32, 49, 512]);  mm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_1124: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1123, [32, 7, 7, 512]);  view_1123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_1125: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1124, [8, 2, 2, 7, 7, 512]);  view_1124 = None
    permute_646: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1125, [0, 1, 3, 2, 4, 5]);  view_1125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_321: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_646, memory_format = torch.contiguous_format);  permute_646 = None
    view_1126: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_321, [8, 14, 14, 512]);  clone_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    roll_35: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(view_1126, [3, 3], [2, 1]);  view_1126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    mul_646: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(roll_35, primals_143);  primals_143 = None
    mul_647: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_646, 512)
    sum_240: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_646, [3], True)
    mul_648: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_646, mul_94);  mul_646 = None
    sum_241: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_648, [3], True);  mul_648 = None
    mul_649: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_94, sum_241);  sum_241 = None
    sub_186: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_647, sum_240);  mul_647 = sum_240 = None
    sub_187: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_186, mul_649);  sub_186 = mul_649 = None
    mul_650: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_102, sub_187);  div_102 = sub_187 = None
    mul_651: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(roll_35, mul_94);  mul_94 = None
    sum_242: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_651, [0, 1, 2]);  mul_651 = None
    sum_243: "f32[512]" = torch.ops.aten.sum.dim_IntList(roll_35, [0, 1, 2]);  roll_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_272: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_1102, mul_650);  view_1102 = mul_650 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_1127: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(add_272, [8, 196, 512]);  add_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_652: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_1127, div_24);  div_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1128: "f32[1568, 512]" = torch.ops.aten.reshape.default(mul_652, [1568, 512]);  mul_652 = None
    mm_127: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_1128, permute_647);  permute_647 = None
    permute_648: "f32[512, 1568]" = torch.ops.aten.permute.default(view_1128, [1, 0])
    mm_128: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_648, view_247);  permute_648 = view_247 = None
    permute_649: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_128, [1, 0]);  mm_128 = None
    sum_244: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1128, [0], True);  view_1128 = None
    view_1129: "f32[512]" = torch.ops.aten.reshape.default(sum_244, [512]);  sum_244 = None
    permute_650: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_649, [1, 0]);  permute_649 = None
    view_1130: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(mm_127, [8, 196, 2048]);  mm_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_654: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(add_80, 0.5);  add_80 = None
    mul_655: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_246, view_246)
    mul_656: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_655, -0.5);  mul_655 = None
    exp_39: "f32[8, 196, 2048]" = torch.ops.aten.exp.default(mul_656);  mul_656 = None
    mul_657: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_39, 0.3989422804014327);  exp_39 = None
    mul_658: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_246, mul_657);  view_246 = mul_657 = None
    add_274: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(mul_654, mul_658);  mul_654 = mul_658 = None
    mul_659: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1130, add_274);  view_1130 = add_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1131: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_659, [1568, 2048]);  mul_659 = None
    mm_129: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1131, permute_651);  permute_651 = None
    permute_652: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_1131, [1, 0])
    mm_130: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_652, view_245);  permute_652 = view_245 = None
    permute_653: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_130, [1, 0]);  mm_130 = None
    sum_245: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1131, [0], True);  view_1131 = None
    view_1132: "f32[2048]" = torch.ops.aten.reshape.default(sum_245, [2048]);  sum_245 = None
    permute_654: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_653, [1, 0]);  permute_653 = None
    view_1133: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(mm_129, [8, 196, 512]);  mm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_661: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_1133, primals_137);  primals_137 = None
    mul_662: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_661, 512)
    sum_246: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_661, [2], True)
    mul_663: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_661, mul_88);  mul_661 = None
    sum_247: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_663, [2], True);  mul_663 = None
    mul_664: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_88, sum_247);  sum_247 = None
    sub_189: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(mul_662, sum_246);  mul_662 = sum_246 = None
    sub_190: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(sub_189, mul_664);  sub_189 = mul_664 = None
    mul_665: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(div_103, sub_190);  div_103 = sub_190 = None
    mul_666: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_1133, mul_88);  mul_88 = None
    sum_248: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_666, [0, 1]);  mul_666 = None
    sum_249: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1133, [0, 1]);  view_1133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_275: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_1127, mul_665);  view_1127 = mul_665 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_1134: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(add_275, [8, 14, 14, 512]);  add_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_667: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1134, div_23);  div_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_1135: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(mul_667, [8, 2, 7, 2, 7, 512]);  mul_667 = None
    permute_655: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_1135, [0, 1, 3, 2, 4, 5]);  view_1135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_322: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_655, memory_format = torch.contiguous_format);  permute_655 = None
    view_1136: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_322, [32, 7, 7, 512]);  clone_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_1137: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(view_1136, [32, 49, 512]);  view_1136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_1138: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_1137, [1568, 512]);  view_1137 = None
    mm_131: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1138, permute_656);  permute_656 = None
    permute_657: "f32[512, 1568]" = torch.ops.aten.permute.default(view_1138, [1, 0])
    mm_132: "f32[512, 512]" = torch.ops.aten.mm.default(permute_657, view_239);  permute_657 = view_239 = None
    permute_658: "f32[512, 512]" = torch.ops.aten.permute.default(mm_132, [1, 0]);  mm_132 = None
    sum_250: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1138, [0], True);  view_1138 = None
    view_1139: "f32[512]" = torch.ops.aten.reshape.default(sum_250, [512]);  sum_250 = None
    permute_659: "f32[512, 512]" = torch.ops.aten.permute.default(permute_658, [1, 0]);  permute_658 = None
    view_1140: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_131, [32, 49, 512]);  mm_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_1141: "f32[32, 49, 16, 32]" = torch.ops.aten.reshape.default(view_1140, [32, 49, 16, 32]);  view_1140 = None
    permute_660: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_1141, [0, 2, 1, 3]);  view_1141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_323: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(permute_660, memory_format = torch.contiguous_format);  permute_660 = None
    view_1142: "f32[512, 49, 32]" = torch.ops.aten.reshape.default(clone_323, [512, 49, 32]);  clone_323 = None
    bmm_108: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(permute_661, view_1142);  permute_661 = None
    bmm_109: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_1142, permute_662);  view_1142 = permute_662 = None
    view_1143: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_108, [32, 16, 49, 32]);  bmm_108 = None
    view_1144: "f32[32, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_109, [32, 16, 49, 49]);  bmm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    mul_668: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_1144, alias_39);  view_1144 = None
    sum_251: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_668, [-1], True)
    mul_669: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(alias_39, sum_251);  alias_39 = sum_251 = None
    sub_191: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(mul_668, mul_669);  mul_668 = mul_669 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_252: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(sub_191, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_15: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_252, 0);  sum_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_663: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_15, [1, 2, 0]);  squeeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_1145: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_663, [2401, 16]);  permute_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_put_15: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_8, [view_233], view_1145, True);  view_233 = view_1145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_1146: "f32[512, 49, 49]" = torch.ops.aten.reshape.default(sub_191, [512, 49, 49]);  sub_191 = None
    bmm_110: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(permute_664, view_1146);  permute_664 = None
    bmm_111: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_1146, permute_665);  view_1146 = permute_665 = None
    view_1147: "f32[32, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_110, [32, 16, 32, 49]);  bmm_110 = None
    view_1148: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_111, [32, 16, 49, 32]);  bmm_111 = None
    permute_666: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_1147, [0, 1, 3, 2]);  view_1147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_670: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_1148, 0.1767766952966369);  view_1148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    cat_15: "f32[96, 16, 49, 32]" = torch.ops.aten.cat.default([mul_670, permute_666, view_1143]);  mul_670 = permute_666 = view_1143 = None
    view_1149: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_15, [3, 32, 16, 49, 32]);  cat_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_667: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_1149, [1, 3, 0, 2, 4]);  view_1149 = None
    clone_324: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_667, memory_format = torch.contiguous_format);  permute_667 = None
    view_1150: "f32[32, 49, 1536]" = torch.ops.aten.reshape.default(clone_324, [32, 49, 1536]);  clone_324 = None
    view_1151: "f32[1568, 1536]" = torch.ops.aten.reshape.default(view_1150, [1568, 1536]);  view_1150 = None
    mm_133: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1151, permute_668);  permute_668 = None
    permute_669: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_1151, [1, 0])
    mm_134: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_669, view_227);  permute_669 = view_227 = None
    permute_670: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_134, [1, 0]);  mm_134 = None
    sum_253: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1151, [0], True);  view_1151 = None
    view_1152: "f32[1536]" = torch.ops.aten.reshape.default(sum_253, [1536]);  sum_253 = None
    permute_671: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_670, [1, 0]);  permute_670 = None
    view_1153: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_133, [32, 49, 512]);  mm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_1154: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1153, [32, 7, 7, 512]);  view_1153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_1155: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1154, [8, 2, 2, 7, 7, 512]);  view_1154 = None
    permute_672: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1155, [0, 1, 3, 2, 4, 5]);  view_1155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_325: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_672, memory_format = torch.contiguous_format);  permute_672 = None
    view_1156: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_325, [8, 14, 14, 512]);  clone_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    mul_672: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1156, primals_131);  primals_131 = None
    mul_673: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_672, 512)
    sum_254: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_672, [3], True)
    mul_674: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_672, mul_84);  mul_672 = None
    sum_255: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_674, [3], True);  mul_674 = None
    mul_675: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_84, sum_255);  sum_255 = None
    sub_193: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_673, sum_254);  mul_673 = sum_254 = None
    sub_194: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_193, mul_675);  sub_193 = mul_675 = None
    mul_676: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_104, sub_194);  div_104 = sub_194 = None
    mul_677: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1156, mul_84);  mul_84 = None
    sum_256: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_677, [0, 1, 2]);  mul_677 = None
    sum_257: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1156, [0, 1, 2]);  view_1156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_276: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_1134, mul_676);  view_1134 = mul_676 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_1157: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(add_276, [8, 196, 512]);  add_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_678: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_1157, div_21);  div_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1158: "f32[1568, 512]" = torch.ops.aten.reshape.default(mul_678, [1568, 512]);  mul_678 = None
    mm_135: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_1158, permute_673);  permute_673 = None
    permute_674: "f32[512, 1568]" = torch.ops.aten.permute.default(view_1158, [1, 0])
    mm_136: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_674, view_221);  permute_674 = view_221 = None
    permute_675: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_136, [1, 0]);  mm_136 = None
    sum_258: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1158, [0], True);  view_1158 = None
    view_1159: "f32[512]" = torch.ops.aten.reshape.default(sum_258, [512]);  sum_258 = None
    permute_676: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_675, [1, 0]);  permute_675 = None
    view_1160: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(mm_135, [8, 196, 2048]);  mm_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_680: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(add_72, 0.5);  add_72 = None
    mul_681: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_220, view_220)
    mul_682: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_681, -0.5);  mul_681 = None
    exp_40: "f32[8, 196, 2048]" = torch.ops.aten.exp.default(mul_682);  mul_682 = None
    mul_683: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_40, 0.3989422804014327);  exp_40 = None
    mul_684: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_220, mul_683);  view_220 = mul_683 = None
    add_278: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(mul_680, mul_684);  mul_680 = mul_684 = None
    mul_685: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1160, add_278);  view_1160 = add_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1161: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_685, [1568, 2048]);  mul_685 = None
    mm_137: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1161, permute_677);  permute_677 = None
    permute_678: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_1161, [1, 0])
    mm_138: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_678, view_219);  permute_678 = view_219 = None
    permute_679: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_138, [1, 0]);  mm_138 = None
    sum_259: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1161, [0], True);  view_1161 = None
    view_1162: "f32[2048]" = torch.ops.aten.reshape.default(sum_259, [2048]);  sum_259 = None
    permute_680: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_679, [1, 0]);  permute_679 = None
    view_1163: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(mm_137, [8, 196, 512]);  mm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_687: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_1163, primals_125);  primals_125 = None
    mul_688: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_687, 512)
    sum_260: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_687, [2], True)
    mul_689: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_687, mul_78);  mul_687 = None
    sum_261: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_689, [2], True);  mul_689 = None
    mul_690: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_78, sum_261);  sum_261 = None
    sub_196: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(mul_688, sum_260);  mul_688 = sum_260 = None
    sub_197: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(sub_196, mul_690);  sub_196 = mul_690 = None
    mul_691: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(div_105, sub_197);  div_105 = sub_197 = None
    mul_692: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_1163, mul_78);  mul_78 = None
    sum_262: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_692, [0, 1]);  mul_692 = None
    sum_263: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1163, [0, 1]);  view_1163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_279: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_1157, mul_691);  view_1157 = mul_691 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_1164: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(add_279, [8, 14, 14, 512]);  add_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_693: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1164, div_20);  div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    roll_36: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(mul_693, [-3, -3], [2, 1]);  mul_693 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_1165: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(roll_36, [8, 2, 7, 2, 7, 512]);  roll_36 = None
    permute_681: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_1165, [0, 1, 3, 2, 4, 5]);  view_1165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_326: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_681, memory_format = torch.contiguous_format);  permute_681 = None
    view_1166: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_326, [32, 7, 7, 512]);  clone_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_1167: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(view_1166, [32, 49, 512]);  view_1166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_1168: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_1167, [1568, 512]);  view_1167 = None
    mm_139: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1168, permute_682);  permute_682 = None
    permute_683: "f32[512, 1568]" = torch.ops.aten.permute.default(view_1168, [1, 0])
    mm_140: "f32[512, 512]" = torch.ops.aten.mm.default(permute_683, view_213);  permute_683 = view_213 = None
    permute_684: "f32[512, 512]" = torch.ops.aten.permute.default(mm_140, [1, 0]);  mm_140 = None
    sum_264: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1168, [0], True);  view_1168 = None
    view_1169: "f32[512]" = torch.ops.aten.reshape.default(sum_264, [512]);  sum_264 = None
    permute_685: "f32[512, 512]" = torch.ops.aten.permute.default(permute_684, [1, 0]);  permute_684 = None
    view_1170: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_139, [32, 49, 512]);  mm_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_1171: "f32[32, 49, 16, 32]" = torch.ops.aten.reshape.default(view_1170, [32, 49, 16, 32]);  view_1170 = None
    permute_686: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_1171, [0, 2, 1, 3]);  view_1171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_327: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(permute_686, memory_format = torch.contiguous_format);  permute_686 = None
    view_1172: "f32[512, 49, 32]" = torch.ops.aten.reshape.default(clone_327, [512, 49, 32]);  clone_327 = None
    bmm_112: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(permute_687, view_1172);  permute_687 = None
    bmm_113: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_1172, permute_688);  view_1172 = permute_688 = None
    view_1173: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_112, [32, 16, 49, 32]);  bmm_112 = None
    view_1174: "f32[32, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_113, [32, 16, 49, 49]);  bmm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    mul_694: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_1174, alias_40);  view_1174 = None
    sum_265: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_694, [-1], True)
    mul_695: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(alias_40, sum_265);  alias_40 = sum_265 = None
    sub_198: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(mul_694, mul_695);  mul_694 = mul_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    view_1175: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.reshape.default(sub_198, [8, 4, 16, 49, 49]);  sub_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_1176: "f32[32, 16, 49, 49]" = torch.ops.aten.reshape.default(view_1175, [32, 16, 49, 49]);  view_1175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_266: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(view_1176, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_16: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_266, 0);  sum_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_689: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_16, [1, 2, 0]);  squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_1177: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_689, [2401, 16]);  permute_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_put_16: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_8, [view_205], view_1177, True);  view_205 = view_1177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_1178: "f32[512, 49, 49]" = torch.ops.aten.reshape.default(view_1176, [512, 49, 49]);  view_1176 = None
    bmm_114: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(permute_690, view_1178);  permute_690 = None
    bmm_115: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_1178, permute_691);  view_1178 = permute_691 = None
    view_1179: "f32[32, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_114, [32, 16, 32, 49]);  bmm_114 = None
    view_1180: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_115, [32, 16, 49, 32]);  bmm_115 = None
    permute_692: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_1179, [0, 1, 3, 2]);  view_1179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_696: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_1180, 0.1767766952966369);  view_1180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    cat_16: "f32[96, 16, 49, 32]" = torch.ops.aten.cat.default([mul_696, permute_692, view_1173]);  mul_696 = permute_692 = view_1173 = None
    view_1181: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_16, [3, 32, 16, 49, 32]);  cat_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_693: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_1181, [1, 3, 0, 2, 4]);  view_1181 = None
    clone_328: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_693, memory_format = torch.contiguous_format);  permute_693 = None
    view_1182: "f32[32, 49, 1536]" = torch.ops.aten.reshape.default(clone_328, [32, 49, 1536]);  clone_328 = None
    view_1183: "f32[1568, 1536]" = torch.ops.aten.reshape.default(view_1182, [1568, 1536]);  view_1182 = None
    mm_141: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1183, permute_694);  permute_694 = None
    permute_695: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_1183, [1, 0])
    mm_142: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_695, view_199);  permute_695 = view_199 = None
    permute_696: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_142, [1, 0]);  mm_142 = None
    sum_267: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1183, [0], True);  view_1183 = None
    view_1184: "f32[1536]" = torch.ops.aten.reshape.default(sum_267, [1536]);  sum_267 = None
    permute_697: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_696, [1, 0]);  permute_696 = None
    view_1185: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_141, [32, 49, 512]);  mm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_1186: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1185, [32, 7, 7, 512]);  view_1185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_1187: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1186, [8, 2, 2, 7, 7, 512]);  view_1186 = None
    permute_698: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1187, [0, 1, 3, 2, 4, 5]);  view_1187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_329: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_698, memory_format = torch.contiguous_format);  permute_698 = None
    view_1188: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_329, [8, 14, 14, 512]);  clone_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    roll_37: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(view_1188, [3, 3], [2, 1]);  view_1188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    mul_698: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(roll_37, primals_119);  primals_119 = None
    mul_699: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_698, 512)
    sum_268: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_698, [3], True)
    mul_700: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_698, mul_74);  mul_698 = None
    sum_269: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_700, [3], True);  mul_700 = None
    mul_701: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_74, sum_269);  sum_269 = None
    sub_200: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_699, sum_268);  mul_699 = sum_268 = None
    sub_201: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_200, mul_701);  sub_200 = mul_701 = None
    mul_702: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_106, sub_201);  div_106 = sub_201 = None
    mul_703: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(roll_37, mul_74);  mul_74 = None
    sum_270: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_703, [0, 1, 2]);  mul_703 = None
    sum_271: "f32[512]" = torch.ops.aten.sum.dim_IntList(roll_37, [0, 1, 2]);  roll_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_280: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_1164, mul_702);  view_1164 = mul_702 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_1189: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(add_280, [8, 196, 512]);  add_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_704: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_1189, div_18);  div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1190: "f32[1568, 512]" = torch.ops.aten.reshape.default(mul_704, [1568, 512]);  mul_704 = None
    mm_143: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_1190, permute_699);  permute_699 = None
    permute_700: "f32[512, 1568]" = torch.ops.aten.permute.default(view_1190, [1, 0])
    mm_144: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_700, view_193);  permute_700 = view_193 = None
    permute_701: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_144, [1, 0]);  mm_144 = None
    sum_272: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1190, [0], True);  view_1190 = None
    view_1191: "f32[512]" = torch.ops.aten.reshape.default(sum_272, [512]);  sum_272 = None
    permute_702: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_701, [1, 0]);  permute_701 = None
    view_1192: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(mm_143, [8, 196, 2048]);  mm_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_706: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(add_63, 0.5);  add_63 = None
    mul_707: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_192, view_192)
    mul_708: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_707, -0.5);  mul_707 = None
    exp_41: "f32[8, 196, 2048]" = torch.ops.aten.exp.default(mul_708);  mul_708 = None
    mul_709: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_41, 0.3989422804014327);  exp_41 = None
    mul_710: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_192, mul_709);  view_192 = mul_709 = None
    add_282: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(mul_706, mul_710);  mul_706 = mul_710 = None
    mul_711: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1192, add_282);  view_1192 = add_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1193: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_711, [1568, 2048]);  mul_711 = None
    mm_145: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1193, permute_703);  permute_703 = None
    permute_704: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_1193, [1, 0])
    mm_146: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_704, view_191);  permute_704 = view_191 = None
    permute_705: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_146, [1, 0]);  mm_146 = None
    sum_273: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1193, [0], True);  view_1193 = None
    view_1194: "f32[2048]" = torch.ops.aten.reshape.default(sum_273, [2048]);  sum_273 = None
    permute_706: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_705, [1, 0]);  permute_705 = None
    view_1195: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(mm_145, [8, 196, 512]);  mm_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_713: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_1195, primals_113);  primals_113 = None
    mul_714: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_713, 512)
    sum_274: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_713, [2], True)
    mul_715: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_713, mul_68);  mul_713 = None
    sum_275: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_715, [2], True);  mul_715 = None
    mul_716: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_68, sum_275);  sum_275 = None
    sub_203: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(mul_714, sum_274);  mul_714 = sum_274 = None
    sub_204: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(sub_203, mul_716);  sub_203 = mul_716 = None
    mul_717: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(div_107, sub_204);  div_107 = sub_204 = None
    mul_718: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_1195, mul_68);  mul_68 = None
    sum_276: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_718, [0, 1]);  mul_718 = None
    sum_277: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1195, [0, 1]);  view_1195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_283: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_1189, mul_717);  view_1189 = mul_717 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_1196: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(add_283, [8, 14, 14, 512]);  add_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_719: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1196, div_17);  div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_1197: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(mul_719, [8, 2, 7, 2, 7, 512]);  mul_719 = None
    permute_707: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_1197, [0, 1, 3, 2, 4, 5]);  view_1197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_330: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_707, memory_format = torch.contiguous_format);  permute_707 = None
    view_1198: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_330, [32, 7, 7, 512]);  clone_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_1199: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(view_1198, [32, 49, 512]);  view_1198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_1200: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_1199, [1568, 512]);  view_1199 = None
    mm_147: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1200, permute_708);  permute_708 = None
    permute_709: "f32[512, 1568]" = torch.ops.aten.permute.default(view_1200, [1, 0])
    mm_148: "f32[512, 512]" = torch.ops.aten.mm.default(permute_709, view_185);  permute_709 = view_185 = None
    permute_710: "f32[512, 512]" = torch.ops.aten.permute.default(mm_148, [1, 0]);  mm_148 = None
    sum_278: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1200, [0], True);  view_1200 = None
    view_1201: "f32[512]" = torch.ops.aten.reshape.default(sum_278, [512]);  sum_278 = None
    permute_711: "f32[512, 512]" = torch.ops.aten.permute.default(permute_710, [1, 0]);  permute_710 = None
    view_1202: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_147, [32, 49, 512]);  mm_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_1203: "f32[32, 49, 16, 32]" = torch.ops.aten.reshape.default(view_1202, [32, 49, 16, 32]);  view_1202 = None
    permute_712: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_1203, [0, 2, 1, 3]);  view_1203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_331: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(permute_712, memory_format = torch.contiguous_format);  permute_712 = None
    view_1204: "f32[512, 49, 32]" = torch.ops.aten.reshape.default(clone_331, [512, 49, 32]);  clone_331 = None
    bmm_116: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(permute_713, view_1204);  permute_713 = None
    bmm_117: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_1204, permute_714);  view_1204 = permute_714 = None
    view_1205: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_116, [32, 16, 49, 32]);  bmm_116 = None
    view_1206: "f32[32, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_117, [32, 16, 49, 49]);  bmm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    mul_720: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_1206, alias_41);  view_1206 = None
    sum_279: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_720, [-1], True)
    mul_721: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(alias_41, sum_279);  alias_41 = sum_279 = None
    sub_205: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(mul_720, mul_721);  mul_720 = mul_721 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_280: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(sub_205, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_17: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_280, 0);  sum_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_715: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_17, [1, 2, 0]);  squeeze_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_1207: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_715, [2401, 16]);  permute_715 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_put_17: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_8, [view_179], view_1207, True);  view_179 = view_1207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_1208: "f32[512, 49, 49]" = torch.ops.aten.reshape.default(sub_205, [512, 49, 49]);  sub_205 = None
    bmm_118: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(permute_716, view_1208);  permute_716 = None
    bmm_119: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_1208, permute_717);  view_1208 = permute_717 = None
    view_1209: "f32[32, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_118, [32, 16, 32, 49]);  bmm_118 = None
    view_1210: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_119, [32, 16, 49, 32]);  bmm_119 = None
    permute_718: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_1209, [0, 1, 3, 2]);  view_1209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_722: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_1210, 0.1767766952966369);  view_1210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    cat_17: "f32[96, 16, 49, 32]" = torch.ops.aten.cat.default([mul_722, permute_718, view_1205]);  mul_722 = permute_718 = view_1205 = None
    view_1211: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_17, [3, 32, 16, 49, 32]);  cat_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_719: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_1211, [1, 3, 0, 2, 4]);  view_1211 = None
    clone_332: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_719, memory_format = torch.contiguous_format);  permute_719 = None
    view_1212: "f32[32, 49, 1536]" = torch.ops.aten.reshape.default(clone_332, [32, 49, 1536]);  clone_332 = None
    view_1213: "f32[1568, 1536]" = torch.ops.aten.reshape.default(view_1212, [1568, 1536]);  view_1212 = None
    mm_149: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1213, permute_720);  permute_720 = None
    permute_721: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_1213, [1, 0])
    mm_150: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_721, view_173);  permute_721 = view_173 = None
    permute_722: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_150, [1, 0]);  mm_150 = None
    sum_281: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1213, [0], True);  view_1213 = None
    view_1214: "f32[1536]" = torch.ops.aten.reshape.default(sum_281, [1536]);  sum_281 = None
    permute_723: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_722, [1, 0]);  permute_722 = None
    view_1215: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_149, [32, 49, 512]);  mm_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_1216: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1215, [32, 7, 7, 512]);  view_1215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_1217: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1216, [8, 2, 2, 7, 7, 512]);  view_1216 = None
    permute_724: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1217, [0, 1, 3, 2, 4, 5]);  view_1217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_333: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_724, memory_format = torch.contiguous_format);  permute_724 = None
    view_1218: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_333, [8, 14, 14, 512]);  clone_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    mul_724: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1218, primals_107);  primals_107 = None
    mul_725: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_724, 512)
    sum_282: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_724, [3], True)
    mul_726: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_724, mul_64);  mul_724 = None
    sum_283: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_726, [3], True);  mul_726 = None
    mul_727: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_64, sum_283);  sum_283 = None
    sub_207: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_725, sum_282);  mul_725 = sum_282 = None
    sub_208: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_207, mul_727);  sub_207 = mul_727 = None
    mul_728: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_108, sub_208);  div_108 = sub_208 = None
    mul_729: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1218, mul_64);  mul_64 = None
    sum_284: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_729, [0, 1, 2]);  mul_729 = None
    sum_285: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1218, [0, 1, 2]);  view_1218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_284: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_1196, mul_728);  view_1196 = mul_728 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_1219: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(add_284, [8, 196, 512]);  add_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_730: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_1219, div_15);  div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1220: "f32[1568, 512]" = torch.ops.aten.reshape.default(mul_730, [1568, 512]);  mul_730 = None
    mm_151: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_1220, permute_725);  permute_725 = None
    permute_726: "f32[512, 1568]" = torch.ops.aten.permute.default(view_1220, [1, 0])
    mm_152: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_726, view_167);  permute_726 = view_167 = None
    permute_727: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_152, [1, 0]);  mm_152 = None
    sum_286: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1220, [0], True);  view_1220 = None
    view_1221: "f32[512]" = torch.ops.aten.reshape.default(sum_286, [512]);  sum_286 = None
    permute_728: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_727, [1, 0]);  permute_727 = None
    view_1222: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(mm_151, [8, 196, 2048]);  mm_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_732: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(add_55, 0.5);  add_55 = None
    mul_733: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_166, view_166)
    mul_734: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_733, -0.5);  mul_733 = None
    exp_42: "f32[8, 196, 2048]" = torch.ops.aten.exp.default(mul_734);  mul_734 = None
    mul_735: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_42, 0.3989422804014327);  exp_42 = None
    mul_736: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_166, mul_735);  view_166 = mul_735 = None
    add_286: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(mul_732, mul_736);  mul_732 = mul_736 = None
    mul_737: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1222, add_286);  view_1222 = add_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1223: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_737, [1568, 2048]);  mul_737 = None
    mm_153: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1223, permute_729);  permute_729 = None
    permute_730: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_1223, [1, 0])
    mm_154: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_730, view_165);  permute_730 = view_165 = None
    permute_731: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_154, [1, 0]);  mm_154 = None
    sum_287: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1223, [0], True);  view_1223 = None
    view_1224: "f32[2048]" = torch.ops.aten.reshape.default(sum_287, [2048]);  sum_287 = None
    permute_732: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_731, [1, 0]);  permute_731 = None
    view_1225: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(mm_153, [8, 196, 512]);  mm_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_739: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_1225, primals_101);  primals_101 = None
    mul_740: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_739, 512)
    sum_288: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_739, [2], True)
    mul_741: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_739, mul_58);  mul_739 = None
    sum_289: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_741, [2], True);  mul_741 = None
    mul_742: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_58, sum_289);  sum_289 = None
    sub_210: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(mul_740, sum_288);  mul_740 = sum_288 = None
    sub_211: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(sub_210, mul_742);  sub_210 = mul_742 = None
    mul_743: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(div_109, sub_211);  div_109 = sub_211 = None
    mul_744: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_1225, mul_58);  mul_58 = None
    sum_290: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_744, [0, 1]);  mul_744 = None
    sum_291: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1225, [0, 1]);  view_1225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_287: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_1219, mul_743);  view_1219 = mul_743 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_1226: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(add_287, [8, 14, 14, 512]);  add_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_745: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1226, div_14);  div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    roll_38: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(mul_745, [-3, -3], [2, 1]);  mul_745 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_1227: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(roll_38, [8, 2, 7, 2, 7, 512]);  roll_38 = None
    permute_733: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_1227, [0, 1, 3, 2, 4, 5]);  view_1227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_334: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_733, memory_format = torch.contiguous_format);  permute_733 = None
    view_1228: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_334, [32, 7, 7, 512]);  clone_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_1229: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(view_1228, [32, 49, 512]);  view_1228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_1230: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_1229, [1568, 512]);  view_1229 = None
    mm_155: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1230, permute_734);  permute_734 = None
    permute_735: "f32[512, 1568]" = torch.ops.aten.permute.default(view_1230, [1, 0])
    mm_156: "f32[512, 512]" = torch.ops.aten.mm.default(permute_735, view_159);  permute_735 = view_159 = None
    permute_736: "f32[512, 512]" = torch.ops.aten.permute.default(mm_156, [1, 0]);  mm_156 = None
    sum_292: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1230, [0], True);  view_1230 = None
    view_1231: "f32[512]" = torch.ops.aten.reshape.default(sum_292, [512]);  sum_292 = None
    permute_737: "f32[512, 512]" = torch.ops.aten.permute.default(permute_736, [1, 0]);  permute_736 = None
    view_1232: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_155, [32, 49, 512]);  mm_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_1233: "f32[32, 49, 16, 32]" = torch.ops.aten.reshape.default(view_1232, [32, 49, 16, 32]);  view_1232 = None
    permute_738: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_1233, [0, 2, 1, 3]);  view_1233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_335: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(permute_738, memory_format = torch.contiguous_format);  permute_738 = None
    view_1234: "f32[512, 49, 32]" = torch.ops.aten.reshape.default(clone_335, [512, 49, 32]);  clone_335 = None
    bmm_120: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(permute_739, view_1234);  permute_739 = None
    bmm_121: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_1234, permute_740);  view_1234 = permute_740 = None
    view_1235: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_120, [32, 16, 49, 32]);  bmm_120 = None
    view_1236: "f32[32, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_121, [32, 16, 49, 49]);  bmm_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    mul_746: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_1236, alias_42);  view_1236 = None
    sum_293: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_746, [-1], True)
    mul_747: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(alias_42, sum_293);  alias_42 = sum_293 = None
    sub_212: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(mul_746, mul_747);  mul_746 = mul_747 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    view_1237: "f32[8, 4, 16, 49, 49]" = torch.ops.aten.reshape.default(sub_212, [8, 4, 16, 49, 49]);  sub_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_1238: "f32[32, 16, 49, 49]" = torch.ops.aten.reshape.default(view_1237, [32, 16, 49, 49]);  view_1237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_294: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(view_1238, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_18: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_294, 0);  sum_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_741: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_18, [1, 2, 0]);  squeeze_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_1239: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_741, [2401, 16]);  permute_741 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_put_18: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_8, [view_151], view_1239, True);  view_151 = view_1239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_1240: "f32[512, 49, 49]" = torch.ops.aten.reshape.default(view_1238, [512, 49, 49]);  view_1238 = None
    bmm_122: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(permute_742, view_1240);  permute_742 = None
    bmm_123: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_1240, permute_743);  view_1240 = permute_743 = None
    view_1241: "f32[32, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_122, [32, 16, 32, 49]);  bmm_122 = None
    view_1242: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_123, [32, 16, 49, 32]);  bmm_123 = None
    permute_744: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_1241, [0, 1, 3, 2]);  view_1241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_748: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_1242, 0.1767766952966369);  view_1242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    cat_18: "f32[96, 16, 49, 32]" = torch.ops.aten.cat.default([mul_748, permute_744, view_1235]);  mul_748 = permute_744 = view_1235 = None
    view_1243: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_18, [3, 32, 16, 49, 32]);  cat_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_745: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_1243, [1, 3, 0, 2, 4]);  view_1243 = None
    clone_336: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_745, memory_format = torch.contiguous_format);  permute_745 = None
    view_1244: "f32[32, 49, 1536]" = torch.ops.aten.reshape.default(clone_336, [32, 49, 1536]);  clone_336 = None
    view_1245: "f32[1568, 1536]" = torch.ops.aten.reshape.default(view_1244, [1568, 1536]);  view_1244 = None
    mm_157: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1245, permute_746);  permute_746 = None
    permute_747: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_1245, [1, 0])
    mm_158: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_747, view_145);  permute_747 = view_145 = None
    permute_748: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_158, [1, 0]);  mm_158 = None
    sum_295: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1245, [0], True);  view_1245 = None
    view_1246: "f32[1536]" = torch.ops.aten.reshape.default(sum_295, [1536]);  sum_295 = None
    permute_749: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_748, [1, 0]);  permute_748 = None
    view_1247: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_157, [32, 49, 512]);  mm_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_1248: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1247, [32, 7, 7, 512]);  view_1247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_1249: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1248, [8, 2, 2, 7, 7, 512]);  view_1248 = None
    permute_750: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1249, [0, 1, 3, 2, 4, 5]);  view_1249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_337: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_750, memory_format = torch.contiguous_format);  permute_750 = None
    view_1250: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_337, [8, 14, 14, 512]);  clone_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    roll_39: "f32[8, 14, 14, 512]" = torch.ops.aten.roll.default(view_1250, [3, 3], [2, 1]);  view_1250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    mul_750: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(roll_39, primals_95);  primals_95 = None
    mul_751: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_750, 512)
    sum_296: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_750, [3], True)
    mul_752: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_750, mul_54);  mul_750 = None
    sum_297: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_752, [3], True);  mul_752 = None
    mul_753: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_54, sum_297);  sum_297 = None
    sub_214: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_751, sum_296);  mul_751 = sum_296 = None
    sub_215: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_214, mul_753);  sub_214 = mul_753 = None
    mul_754: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_110, sub_215);  div_110 = sub_215 = None
    mul_755: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(roll_39, mul_54);  mul_54 = None
    sum_298: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_755, [0, 1, 2]);  mul_755 = None
    sum_299: "f32[512]" = torch.ops.aten.sum.dim_IntList(roll_39, [0, 1, 2]);  roll_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_288: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_1226, mul_754);  view_1226 = mul_754 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_1251: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(add_288, [8, 196, 512]);  add_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_756: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_1251, div_12);  div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1252: "f32[1568, 512]" = torch.ops.aten.reshape.default(mul_756, [1568, 512]);  mul_756 = None
    mm_159: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_1252, permute_751);  permute_751 = None
    permute_752: "f32[512, 1568]" = torch.ops.aten.permute.default(view_1252, [1, 0])
    mm_160: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_752, view_139);  permute_752 = view_139 = None
    permute_753: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_160, [1, 0]);  mm_160 = None
    sum_300: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1252, [0], True);  view_1252 = None
    view_1253: "f32[512]" = torch.ops.aten.reshape.default(sum_300, [512]);  sum_300 = None
    permute_754: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_753, [1, 0]);  permute_753 = None
    view_1254: "f32[8, 196, 2048]" = torch.ops.aten.reshape.default(mm_159, [8, 196, 2048]);  mm_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_758: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(add_46, 0.5);  add_46 = None
    mul_759: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_138, view_138)
    mul_760: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_759, -0.5);  mul_759 = None
    exp_43: "f32[8, 196, 2048]" = torch.ops.aten.exp.default(mul_760);  mul_760 = None
    mul_761: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_43, 0.3989422804014327);  exp_43 = None
    mul_762: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_138, mul_761);  view_138 = mul_761 = None
    add_290: "f32[8, 196, 2048]" = torch.ops.aten.add.Tensor(mul_758, mul_762);  mul_758 = mul_762 = None
    mul_763: "f32[8, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1254, add_290);  view_1254 = add_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1255: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_763, [1568, 2048]);  mul_763 = None
    mm_161: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1255, permute_755);  permute_755 = None
    permute_756: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_1255, [1, 0])
    mm_162: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_756, view_137);  permute_756 = view_137 = None
    permute_757: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_162, [1, 0]);  mm_162 = None
    sum_301: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1255, [0], True);  view_1255 = None
    view_1256: "f32[2048]" = torch.ops.aten.reshape.default(sum_301, [2048]);  sum_301 = None
    permute_758: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_757, [1, 0]);  permute_757 = None
    view_1257: "f32[8, 196, 512]" = torch.ops.aten.reshape.default(mm_161, [8, 196, 512]);  mm_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_765: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_1257, primals_89);  primals_89 = None
    mul_766: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_765, 512)
    sum_302: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_765, [2], True)
    mul_767: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_765, mul_48);  mul_765 = None
    sum_303: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_767, [2], True);  mul_767 = None
    mul_768: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(mul_48, sum_303);  sum_303 = None
    sub_217: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(mul_766, sum_302);  mul_766 = sum_302 = None
    sub_218: "f32[8, 196, 512]" = torch.ops.aten.sub.Tensor(sub_217, mul_768);  sub_217 = mul_768 = None
    mul_769: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(div_111, sub_218);  div_111 = sub_218 = None
    mul_770: "f32[8, 196, 512]" = torch.ops.aten.mul.Tensor(view_1257, mul_48);  mul_48 = None
    sum_304: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_770, [0, 1]);  mul_770 = None
    sum_305: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1257, [0, 1]);  view_1257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_291: "f32[8, 196, 512]" = torch.ops.aten.add.Tensor(view_1251, mul_769);  view_1251 = mul_769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_1258: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(add_291, [8, 14, 14, 512]);  add_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_771: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1258, div_11);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_1259: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(mul_771, [8, 2, 7, 2, 7, 512]);  mul_771 = None
    permute_759: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_1259, [0, 1, 3, 2, 4, 5]);  view_1259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_338: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_759, memory_format = torch.contiguous_format);  permute_759 = None
    view_1260: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_338, [32, 7, 7, 512]);  clone_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_1261: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(view_1260, [32, 49, 512]);  view_1260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_1262: "f32[1568, 512]" = torch.ops.aten.reshape.default(view_1261, [1568, 512]);  view_1261 = None
    mm_163: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1262, permute_760);  permute_760 = None
    permute_761: "f32[512, 1568]" = torch.ops.aten.permute.default(view_1262, [1, 0])
    mm_164: "f32[512, 512]" = torch.ops.aten.mm.default(permute_761, view_131);  permute_761 = view_131 = None
    permute_762: "f32[512, 512]" = torch.ops.aten.permute.default(mm_164, [1, 0]);  mm_164 = None
    sum_306: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1262, [0], True);  view_1262 = None
    view_1263: "f32[512]" = torch.ops.aten.reshape.default(sum_306, [512]);  sum_306 = None
    permute_763: "f32[512, 512]" = torch.ops.aten.permute.default(permute_762, [1, 0]);  permute_762 = None
    view_1264: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_163, [32, 49, 512]);  mm_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_1265: "f32[32, 49, 16, 32]" = torch.ops.aten.reshape.default(view_1264, [32, 49, 16, 32]);  view_1264 = None
    permute_764: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_1265, [0, 2, 1, 3]);  view_1265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_339: "f32[32, 16, 49, 32]" = torch.ops.aten.clone.default(permute_764, memory_format = torch.contiguous_format);  permute_764 = None
    view_1266: "f32[512, 49, 32]" = torch.ops.aten.reshape.default(clone_339, [512, 49, 32]);  clone_339 = None
    bmm_124: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(permute_765, view_1266);  permute_765 = None
    bmm_125: "f32[512, 49, 49]" = torch.ops.aten.bmm.default(view_1266, permute_766);  view_1266 = permute_766 = None
    view_1267: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_124, [32, 16, 49, 32]);  bmm_124 = None
    view_1268: "f32[32, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_125, [32, 16, 49, 49]);  bmm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    mul_772: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_1268, alias_43);  view_1268 = None
    sum_307: "f32[32, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_772, [-1], True)
    mul_773: "f32[32, 16, 49, 49]" = torch.ops.aten.mul.Tensor(alias_43, sum_307);  alias_43 = sum_307 = None
    sub_219: "f32[32, 16, 49, 49]" = torch.ops.aten.sub.Tensor(mul_772, mul_773);  mul_772 = mul_773 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_308: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(sub_219, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_19: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_308, 0);  sum_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_767: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_19, [1, 2, 0]);  squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_1269: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_767, [2401, 16]);  permute_767 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_put_19: "f32[169, 16]" = torch.ops.aten.index_put_.default(full_default_8, [view_125], view_1269, True);  full_default_8 = view_125 = view_1269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_1270: "f32[512, 49, 49]" = torch.ops.aten.reshape.default(sub_219, [512, 49, 49]);  sub_219 = None
    bmm_126: "f32[512, 32, 49]" = torch.ops.aten.bmm.default(permute_768, view_1270);  permute_768 = None
    bmm_127: "f32[512, 49, 32]" = torch.ops.aten.bmm.default(view_1270, permute_769);  view_1270 = permute_769 = None
    view_1271: "f32[32, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_126, [32, 16, 32, 49]);  bmm_126 = None
    view_1272: "f32[32, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_127, [32, 16, 49, 32]);  bmm_127 = None
    permute_770: "f32[32, 16, 49, 32]" = torch.ops.aten.permute.default(view_1271, [0, 1, 3, 2]);  view_1271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_774: "f32[32, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_1272, 0.1767766952966369);  view_1272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    cat_19: "f32[96, 16, 49, 32]" = torch.ops.aten.cat.default([mul_774, permute_770, view_1267]);  mul_774 = permute_770 = view_1267 = None
    view_1273: "f32[3, 32, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_19, [3, 32, 16, 49, 32]);  cat_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_771: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_1273, [1, 3, 0, 2, 4]);  view_1273 = None
    clone_340: "f32[32, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_771, memory_format = torch.contiguous_format);  permute_771 = None
    view_1274: "f32[32, 49, 1536]" = torch.ops.aten.reshape.default(clone_340, [32, 49, 1536]);  clone_340 = None
    view_1275: "f32[1568, 1536]" = torch.ops.aten.reshape.default(view_1274, [1568, 1536]);  view_1274 = None
    mm_165: "f32[1568, 512]" = torch.ops.aten.mm.default(view_1275, permute_772);  permute_772 = None
    permute_773: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_1275, [1, 0])
    mm_166: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_773, view_119);  permute_773 = view_119 = None
    permute_774: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_166, [1, 0]);  mm_166 = None
    sum_309: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1275, [0], True);  view_1275 = None
    view_1276: "f32[1536]" = torch.ops.aten.reshape.default(sum_309, [1536]);  sum_309 = None
    permute_775: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_774, [1, 0]);  permute_774 = None
    view_1277: "f32[32, 49, 512]" = torch.ops.aten.reshape.default(mm_165, [32, 49, 512]);  mm_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_1278: "f32[32, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1277, [32, 7, 7, 512]);  view_1277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_1279: "f32[8, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1278, [8, 2, 2, 7, 7, 512]);  view_1278 = None
    permute_776: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1279, [0, 1, 3, 2, 4, 5]);  view_1279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_341: "f32[8, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_776, memory_format = torch.contiguous_format);  permute_776 = None
    view_1280: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_341, [8, 14, 14, 512]);  clone_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    mul_776: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1280, primals_83);  primals_83 = None
    mul_777: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_776, 512)
    sum_310: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_776, [3], True)
    mul_778: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_776, mul_44);  mul_776 = None
    sum_311: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_778, [3], True);  mul_778 = None
    mul_779: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_44, sum_311);  sum_311 = None
    sub_221: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_777, sum_310);  mul_777 = sum_310 = None
    sub_222: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_221, mul_779);  sub_221 = mul_779 = None
    div_112: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 512);  rsqrt_11 = None
    mul_780: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_112, sub_222);  div_112 = sub_222 = None
    mul_781: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1280, mul_44);  mul_44 = None
    sum_312: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_781, [0, 1, 2]);  mul_781 = None
    sum_313: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1280, [0, 1, 2]);  view_1280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_292: "f32[8, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_1258, mul_780);  view_1258 = mul_780 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:358, code: x = self.reduction(x)
    view_1281: "f32[1568, 512]" = torch.ops.aten.reshape.default(add_292, [1568, 512]);  add_292 = None
    permute_777: "f32[512, 1568]" = torch.ops.aten.permute.default(view_1281, [1, 0])
    mm_167: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_777, view_114);  permute_777 = view_114 = None
    permute_778: "f32[1024, 512]" = torch.ops.aten.permute.default(mm_167, [1, 0]);  mm_167 = None
    mm_168: "f32[1568, 1024]" = torch.ops.aten.mm.default(view_1281, permute_779);  view_1281 = permute_779 = None
    view_1282: "f32[8, 14, 14, 1024]" = torch.ops.aten.reshape.default(mm_168, [8, 14, 14, 1024]);  mm_168 = None
    permute_780: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_778, [1, 0]);  permute_778 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:357, code: x = self.norm(x)
    mul_783: "f32[8, 14, 14, 1024]" = torch.ops.aten.mul.Tensor(view_1282, primals_80);  primals_80 = None
    mul_784: "f32[8, 14, 14, 1024]" = torch.ops.aten.mul.Tensor(mul_783, 1024)
    sum_314: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_783, [3], True)
    mul_785: "f32[8, 14, 14, 1024]" = torch.ops.aten.mul.Tensor(mul_783, mul_42);  mul_783 = None
    sum_315: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_785, [3], True);  mul_785 = None
    mul_786: "f32[8, 14, 14, 1024]" = torch.ops.aten.mul.Tensor(mul_42, sum_315);  sum_315 = None
    sub_224: "f32[8, 14, 14, 1024]" = torch.ops.aten.sub.Tensor(mul_784, sum_314);  mul_784 = sum_314 = None
    sub_225: "f32[8, 14, 14, 1024]" = torch.ops.aten.sub.Tensor(sub_224, mul_786);  sub_224 = mul_786 = None
    mul_787: "f32[8, 14, 14, 1024]" = torch.ops.aten.mul.Tensor(div_113, sub_225);  div_113 = sub_225 = None
    mul_788: "f32[8, 14, 14, 1024]" = torch.ops.aten.mul.Tensor(view_1282, mul_42);  mul_42 = None
    sum_316: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_788, [0, 1, 2]);  mul_788 = None
    sum_317: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1282, [0, 1, 2]);  view_1282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:356, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
    view_1283: "f32[8, 14, 14, 2, 2, 256]" = torch.ops.aten.reshape.default(mul_787, [8, 14, 14, 2, 2, 256]);  mul_787 = None
    permute_781: "f32[8, 14, 2, 14, 2, 256]" = torch.ops.aten.permute.default(view_1283, [0, 1, 4, 2, 3, 5]);  view_1283 = None
    clone_342: "f32[8, 14, 2, 14, 2, 256]" = torch.ops.aten.clone.default(permute_781, memory_format = torch.contiguous_format);  permute_781 = None
    view_1284: "f32[8, 28, 28, 256]" = torch.ops.aten.reshape.default(clone_342, [8, 28, 28, 256]);  clone_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_1285: "f32[8, 784, 256]" = torch.ops.aten.reshape.default(view_1284, [8, 784, 256]);  view_1284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_789: "f32[8, 784, 256]" = torch.ops.aten.mul.Tensor(view_1285, div_9);  div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1286: "f32[6272, 256]" = torch.ops.aten.reshape.default(mul_789, [6272, 256]);  mul_789 = None
    mm_169: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_1286, permute_782);  permute_782 = None
    permute_783: "f32[256, 6272]" = torch.ops.aten.permute.default(view_1286, [1, 0])
    mm_170: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_783, view_109);  permute_783 = view_109 = None
    permute_784: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_170, [1, 0]);  mm_170 = None
    sum_318: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_1286, [0], True);  view_1286 = None
    view_1287: "f32[256]" = torch.ops.aten.reshape.default(sum_318, [256]);  sum_318 = None
    permute_785: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_784, [1, 0]);  permute_784 = None
    view_1288: "f32[8, 784, 1024]" = torch.ops.aten.reshape.default(mm_169, [8, 784, 1024]);  mm_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_791: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(add_36, 0.5);  add_36 = None
    mul_792: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_108, view_108)
    mul_793: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_792, -0.5);  mul_792 = None
    exp_44: "f32[8, 784, 1024]" = torch.ops.aten.exp.default(mul_793);  mul_793 = None
    mul_794: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(exp_44, 0.3989422804014327);  exp_44 = None
    mul_795: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_108, mul_794);  view_108 = mul_794 = None
    add_294: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(mul_791, mul_795);  mul_791 = mul_795 = None
    mul_796: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_1288, add_294);  view_1288 = add_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1289: "f32[6272, 1024]" = torch.ops.aten.reshape.default(mul_796, [6272, 1024]);  mul_796 = None
    mm_171: "f32[6272, 256]" = torch.ops.aten.mm.default(view_1289, permute_786);  permute_786 = None
    permute_787: "f32[1024, 6272]" = torch.ops.aten.permute.default(view_1289, [1, 0])
    mm_172: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_787, view_107);  permute_787 = view_107 = None
    permute_788: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_172, [1, 0]);  mm_172 = None
    sum_319: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1289, [0], True);  view_1289 = None
    view_1290: "f32[1024]" = torch.ops.aten.reshape.default(sum_319, [1024]);  sum_319 = None
    permute_789: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_788, [1, 0]);  permute_788 = None
    view_1291: "f32[8, 784, 256]" = torch.ops.aten.reshape.default(mm_171, [8, 784, 256]);  mm_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_798: "f32[8, 784, 256]" = torch.ops.aten.mul.Tensor(view_1291, primals_74);  primals_74 = None
    mul_799: "f32[8, 784, 256]" = torch.ops.aten.mul.Tensor(mul_798, 256)
    sum_320: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_798, [2], True)
    mul_800: "f32[8, 784, 256]" = torch.ops.aten.mul.Tensor(mul_798, mul_36);  mul_798 = None
    sum_321: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_800, [2], True);  mul_800 = None
    mul_801: "f32[8, 784, 256]" = torch.ops.aten.mul.Tensor(mul_36, sum_321);  sum_321 = None
    sub_227: "f32[8, 784, 256]" = torch.ops.aten.sub.Tensor(mul_799, sum_320);  mul_799 = sum_320 = None
    sub_228: "f32[8, 784, 256]" = torch.ops.aten.sub.Tensor(sub_227, mul_801);  sub_227 = mul_801 = None
    mul_802: "f32[8, 784, 256]" = torch.ops.aten.mul.Tensor(div_114, sub_228);  div_114 = sub_228 = None
    mul_803: "f32[8, 784, 256]" = torch.ops.aten.mul.Tensor(view_1291, mul_36);  mul_36 = None
    sum_322: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_803, [0, 1]);  mul_803 = None
    sum_323: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_1291, [0, 1]);  view_1291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_295: "f32[8, 784, 256]" = torch.ops.aten.add.Tensor(view_1285, mul_802);  view_1285 = mul_802 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_1292: "f32[8, 28, 28, 256]" = torch.ops.aten.reshape.default(add_295, [8, 28, 28, 256]);  add_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_804: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(view_1292, div_8);  div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    roll_40: "f32[8, 28, 28, 256]" = torch.ops.aten.roll.default(mul_804, [-3, -3], [2, 1]);  mul_804 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    full_default_60: "f32[8, 28, 28, 256]" = torch.ops.aten.full.default([8, 28, 28, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_1293: "f32[8, 4, 7, 4, 7, 256]" = torch.ops.aten.reshape.default(roll_40, [8, 4, 7, 4, 7, 256]);  roll_40 = None
    permute_790: "f32[8, 4, 4, 7, 7, 256]" = torch.ops.aten.permute.default(view_1293, [0, 1, 3, 2, 4, 5]);  view_1293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_343: "f32[8, 4, 4, 7, 7, 256]" = torch.ops.aten.clone.default(permute_790, memory_format = torch.contiguous_format);  permute_790 = None
    view_1294: "f32[128, 7, 7, 256]" = torch.ops.aten.reshape.default(clone_343, [128, 7, 7, 256]);  clone_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_1295: "f32[128, 49, 256]" = torch.ops.aten.reshape.default(view_1294, [128, 49, 256]);  view_1294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_1296: "f32[6272, 256]" = torch.ops.aten.reshape.default(view_1295, [6272, 256]);  view_1295 = None
    mm_173: "f32[6272, 256]" = torch.ops.aten.mm.default(view_1296, permute_791);  permute_791 = None
    permute_792: "f32[256, 6272]" = torch.ops.aten.permute.default(view_1296, [1, 0])
    mm_174: "f32[256, 256]" = torch.ops.aten.mm.default(permute_792, view_101);  permute_792 = view_101 = None
    permute_793: "f32[256, 256]" = torch.ops.aten.permute.default(mm_174, [1, 0]);  mm_174 = None
    sum_324: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_1296, [0], True);  view_1296 = None
    view_1297: "f32[256]" = torch.ops.aten.reshape.default(sum_324, [256]);  sum_324 = None
    permute_794: "f32[256, 256]" = torch.ops.aten.permute.default(permute_793, [1, 0]);  permute_793 = None
    view_1298: "f32[128, 49, 256]" = torch.ops.aten.reshape.default(mm_173, [128, 49, 256]);  mm_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_1299: "f32[128, 49, 8, 32]" = torch.ops.aten.reshape.default(view_1298, [128, 49, 8, 32]);  view_1298 = None
    permute_795: "f32[128, 8, 49, 32]" = torch.ops.aten.permute.default(view_1299, [0, 2, 1, 3]);  view_1299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_344: "f32[128, 8, 49, 32]" = torch.ops.aten.clone.default(permute_795, memory_format = torch.contiguous_format);  permute_795 = None
    view_1300: "f32[1024, 49, 32]" = torch.ops.aten.reshape.default(clone_344, [1024, 49, 32]);  clone_344 = None
    bmm_128: "f32[1024, 49, 32]" = torch.ops.aten.bmm.default(permute_796, view_1300);  permute_796 = None
    bmm_129: "f32[1024, 49, 49]" = torch.ops.aten.bmm.default(view_1300, permute_797);  view_1300 = permute_797 = None
    view_1301: "f32[128, 8, 49, 32]" = torch.ops.aten.reshape.default(bmm_128, [128, 8, 49, 32]);  bmm_128 = None
    view_1302: "f32[128, 8, 49, 49]" = torch.ops.aten.reshape.default(bmm_129, [128, 8, 49, 49]);  bmm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    mul_805: "f32[128, 8, 49, 49]" = torch.ops.aten.mul.Tensor(view_1302, alias_44);  view_1302 = None
    sum_325: "f32[128, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_805, [-1], True)
    mul_806: "f32[128, 8, 49, 49]" = torch.ops.aten.mul.Tensor(alias_44, sum_325);  alias_44 = sum_325 = None
    sub_229: "f32[128, 8, 49, 49]" = torch.ops.aten.sub.Tensor(mul_805, mul_806);  mul_805 = mul_806 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    view_1303: "f32[8, 16, 8, 49, 49]" = torch.ops.aten.reshape.default(sub_229, [8, 16, 8, 49, 49]);  sub_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_1304: "f32[128, 8, 49, 49]" = torch.ops.aten.reshape.default(view_1303, [128, 8, 49, 49]);  view_1303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_326: "f32[1, 8, 49, 49]" = torch.ops.aten.sum.dim_IntList(view_1304, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_20: "f32[8, 49, 49]" = torch.ops.aten.squeeze.dim(sum_326, 0);  sum_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_798: "f32[49, 49, 8]" = torch.ops.aten.permute.default(squeeze_20, [1, 2, 0]);  squeeze_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_1305: "f32[2401, 8]" = torch.ops.aten.reshape.default(permute_798, [2401, 8]);  permute_798 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    full_default_62: "f32[169, 8]" = torch.ops.aten.full.default([169, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    index_put_20: "f32[169, 8]" = torch.ops.aten.index_put.default(full_default_62, [view_93], view_1305, True);  view_93 = view_1305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_1306: "f32[1024, 49, 49]" = torch.ops.aten.reshape.default(view_1304, [1024, 49, 49]);  view_1304 = None
    bmm_130: "f32[1024, 32, 49]" = torch.ops.aten.bmm.default(permute_799, view_1306);  permute_799 = None
    bmm_131: "f32[1024, 49, 32]" = torch.ops.aten.bmm.default(view_1306, permute_800);  view_1306 = permute_800 = None
    view_1307: "f32[128, 8, 32, 49]" = torch.ops.aten.reshape.default(bmm_130, [128, 8, 32, 49]);  bmm_130 = None
    view_1308: "f32[128, 8, 49, 32]" = torch.ops.aten.reshape.default(bmm_131, [128, 8, 49, 32]);  bmm_131 = None
    permute_801: "f32[128, 8, 49, 32]" = torch.ops.aten.permute.default(view_1307, [0, 1, 3, 2]);  view_1307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_807: "f32[128, 8, 49, 32]" = torch.ops.aten.mul.Tensor(view_1308, 0.1767766952966369);  view_1308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    cat_20: "f32[384, 8, 49, 32]" = torch.ops.aten.cat.default([mul_807, permute_801, view_1301]);  mul_807 = permute_801 = view_1301 = None
    view_1309: "f32[3, 128, 8, 49, 32]" = torch.ops.aten.reshape.default(cat_20, [3, 128, 8, 49, 32]);  cat_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_802: "f32[128, 49, 3, 8, 32]" = torch.ops.aten.permute.default(view_1309, [1, 3, 0, 2, 4]);  view_1309 = None
    clone_345: "f32[128, 49, 3, 8, 32]" = torch.ops.aten.clone.default(permute_802, memory_format = torch.contiguous_format);  permute_802 = None
    view_1310: "f32[128, 49, 768]" = torch.ops.aten.reshape.default(clone_345, [128, 49, 768]);  clone_345 = None
    view_1311: "f32[6272, 768]" = torch.ops.aten.reshape.default(view_1310, [6272, 768]);  view_1310 = None
    mm_175: "f32[6272, 256]" = torch.ops.aten.mm.default(view_1311, permute_803);  permute_803 = None
    permute_804: "f32[768, 6272]" = torch.ops.aten.permute.default(view_1311, [1, 0])
    mm_176: "f32[768, 256]" = torch.ops.aten.mm.default(permute_804, view_87);  permute_804 = view_87 = None
    permute_805: "f32[256, 768]" = torch.ops.aten.permute.default(mm_176, [1, 0]);  mm_176 = None
    sum_327: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1311, [0], True);  view_1311 = None
    view_1312: "f32[768]" = torch.ops.aten.reshape.default(sum_327, [768]);  sum_327 = None
    permute_806: "f32[768, 256]" = torch.ops.aten.permute.default(permute_805, [1, 0]);  permute_805 = None
    view_1313: "f32[128, 49, 256]" = torch.ops.aten.reshape.default(mm_175, [128, 49, 256]);  mm_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_1314: "f32[128, 7, 7, 256]" = torch.ops.aten.reshape.default(view_1313, [128, 7, 7, 256]);  view_1313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_1315: "f32[8, 4, 4, 7, 7, 256]" = torch.ops.aten.reshape.default(view_1314, [8, 4, 4, 7, 7, 256]);  view_1314 = None
    permute_807: "f32[8, 4, 7, 4, 7, 256]" = torch.ops.aten.permute.default(view_1315, [0, 1, 3, 2, 4, 5]);  view_1315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_346: "f32[8, 4, 7, 4, 7, 256]" = torch.ops.aten.clone.default(permute_807, memory_format = torch.contiguous_format);  permute_807 = None
    view_1316: "f32[8, 28, 28, 256]" = torch.ops.aten.reshape.default(clone_346, [8, 28, 28, 256]);  clone_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    roll_41: "f32[8, 28, 28, 256]" = torch.ops.aten.roll.default(view_1316, [3, 3], [2, 1]);  view_1316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    mul_809: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(roll_41, primals_68);  primals_68 = None
    mul_810: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_809, 256)
    sum_328: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_809, [3], True)
    mul_811: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_809, mul_32);  mul_809 = None
    sum_329: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_811, [3], True);  mul_811 = None
    mul_812: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_32, sum_329);  sum_329 = None
    sub_231: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(mul_810, sum_328);  mul_810 = sum_328 = None
    sub_232: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(sub_231, mul_812);  sub_231 = mul_812 = None
    mul_813: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(div_115, sub_232);  div_115 = sub_232 = None
    mul_814: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(roll_41, mul_32);  mul_32 = None
    sum_330: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_814, [0, 1, 2]);  mul_814 = None
    sum_331: "f32[256]" = torch.ops.aten.sum.dim_IntList(roll_41, [0, 1, 2]);  roll_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_296: "f32[8, 28, 28, 256]" = torch.ops.aten.add.Tensor(view_1292, mul_813);  view_1292 = mul_813 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_1317: "f32[8, 784, 256]" = torch.ops.aten.reshape.default(add_296, [8, 784, 256]);  add_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_815: "f32[8, 784, 256]" = torch.ops.aten.mul.Tensor(view_1317, div_6);  div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1318: "f32[6272, 256]" = torch.ops.aten.reshape.default(mul_815, [6272, 256]);  mul_815 = None
    mm_177: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_1318, permute_808);  permute_808 = None
    permute_809: "f32[256, 6272]" = torch.ops.aten.permute.default(view_1318, [1, 0])
    mm_178: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_809, view_81);  permute_809 = view_81 = None
    permute_810: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_178, [1, 0]);  mm_178 = None
    sum_332: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_1318, [0], True);  view_1318 = None
    view_1319: "f32[256]" = torch.ops.aten.reshape.default(sum_332, [256]);  sum_332 = None
    permute_811: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_810, [1, 0]);  permute_810 = None
    view_1320: "f32[8, 784, 1024]" = torch.ops.aten.reshape.default(mm_177, [8, 784, 1024]);  mm_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_817: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(add_27, 0.5);  add_27 = None
    mul_818: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_80, view_80)
    mul_819: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_818, -0.5);  mul_818 = None
    exp_45: "f32[8, 784, 1024]" = torch.ops.aten.exp.default(mul_819);  mul_819 = None
    mul_820: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(exp_45, 0.3989422804014327);  exp_45 = None
    mul_821: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_80, mul_820);  view_80 = mul_820 = None
    add_298: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(mul_817, mul_821);  mul_817 = mul_821 = None
    mul_822: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_1320, add_298);  view_1320 = add_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1321: "f32[6272, 1024]" = torch.ops.aten.reshape.default(mul_822, [6272, 1024]);  mul_822 = None
    mm_179: "f32[6272, 256]" = torch.ops.aten.mm.default(view_1321, permute_812);  permute_812 = None
    permute_813: "f32[1024, 6272]" = torch.ops.aten.permute.default(view_1321, [1, 0])
    mm_180: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_813, view_79);  permute_813 = view_79 = None
    permute_814: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_180, [1, 0]);  mm_180 = None
    sum_333: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1321, [0], True);  view_1321 = None
    view_1322: "f32[1024]" = torch.ops.aten.reshape.default(sum_333, [1024]);  sum_333 = None
    permute_815: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_814, [1, 0]);  permute_814 = None
    view_1323: "f32[8, 784, 256]" = torch.ops.aten.reshape.default(mm_179, [8, 784, 256]);  mm_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_824: "f32[8, 784, 256]" = torch.ops.aten.mul.Tensor(view_1323, primals_62);  primals_62 = None
    mul_825: "f32[8, 784, 256]" = torch.ops.aten.mul.Tensor(mul_824, 256)
    sum_334: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_824, [2], True)
    mul_826: "f32[8, 784, 256]" = torch.ops.aten.mul.Tensor(mul_824, mul_26);  mul_824 = None
    sum_335: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_826, [2], True);  mul_826 = None
    mul_827: "f32[8, 784, 256]" = torch.ops.aten.mul.Tensor(mul_26, sum_335);  sum_335 = None
    sub_234: "f32[8, 784, 256]" = torch.ops.aten.sub.Tensor(mul_825, sum_334);  mul_825 = sum_334 = None
    sub_235: "f32[8, 784, 256]" = torch.ops.aten.sub.Tensor(sub_234, mul_827);  sub_234 = mul_827 = None
    mul_828: "f32[8, 784, 256]" = torch.ops.aten.mul.Tensor(div_116, sub_235);  div_116 = sub_235 = None
    mul_829: "f32[8, 784, 256]" = torch.ops.aten.mul.Tensor(view_1323, mul_26);  mul_26 = None
    sum_336: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_829, [0, 1]);  mul_829 = None
    sum_337: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_1323, [0, 1]);  view_1323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_299: "f32[8, 784, 256]" = torch.ops.aten.add.Tensor(view_1317, mul_828);  view_1317 = mul_828 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_1324: "f32[8, 28, 28, 256]" = torch.ops.aten.reshape.default(add_299, [8, 28, 28, 256]);  add_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_830: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(view_1324, div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_1325: "f32[8, 4, 7, 4, 7, 256]" = torch.ops.aten.reshape.default(mul_830, [8, 4, 7, 4, 7, 256]);  mul_830 = None
    permute_816: "f32[8, 4, 4, 7, 7, 256]" = torch.ops.aten.permute.default(view_1325, [0, 1, 3, 2, 4, 5]);  view_1325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_347: "f32[8, 4, 4, 7, 7, 256]" = torch.ops.aten.clone.default(permute_816, memory_format = torch.contiguous_format);  permute_816 = None
    view_1326: "f32[128, 7, 7, 256]" = torch.ops.aten.reshape.default(clone_347, [128, 7, 7, 256]);  clone_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_1327: "f32[128, 49, 256]" = torch.ops.aten.reshape.default(view_1326, [128, 49, 256]);  view_1326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_1328: "f32[6272, 256]" = torch.ops.aten.reshape.default(view_1327, [6272, 256]);  view_1327 = None
    mm_181: "f32[6272, 256]" = torch.ops.aten.mm.default(view_1328, permute_817);  permute_817 = None
    permute_818: "f32[256, 6272]" = torch.ops.aten.permute.default(view_1328, [1, 0])
    mm_182: "f32[256, 256]" = torch.ops.aten.mm.default(permute_818, view_73);  permute_818 = view_73 = None
    permute_819: "f32[256, 256]" = torch.ops.aten.permute.default(mm_182, [1, 0]);  mm_182 = None
    sum_338: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_1328, [0], True);  view_1328 = None
    view_1329: "f32[256]" = torch.ops.aten.reshape.default(sum_338, [256]);  sum_338 = None
    permute_820: "f32[256, 256]" = torch.ops.aten.permute.default(permute_819, [1, 0]);  permute_819 = None
    view_1330: "f32[128, 49, 256]" = torch.ops.aten.reshape.default(mm_181, [128, 49, 256]);  mm_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_1331: "f32[128, 49, 8, 32]" = torch.ops.aten.reshape.default(view_1330, [128, 49, 8, 32]);  view_1330 = None
    permute_821: "f32[128, 8, 49, 32]" = torch.ops.aten.permute.default(view_1331, [0, 2, 1, 3]);  view_1331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_348: "f32[128, 8, 49, 32]" = torch.ops.aten.clone.default(permute_821, memory_format = torch.contiguous_format);  permute_821 = None
    view_1332: "f32[1024, 49, 32]" = torch.ops.aten.reshape.default(clone_348, [1024, 49, 32]);  clone_348 = None
    bmm_132: "f32[1024, 49, 32]" = torch.ops.aten.bmm.default(permute_822, view_1332);  permute_822 = None
    bmm_133: "f32[1024, 49, 49]" = torch.ops.aten.bmm.default(view_1332, permute_823);  view_1332 = permute_823 = None
    view_1333: "f32[128, 8, 49, 32]" = torch.ops.aten.reshape.default(bmm_132, [128, 8, 49, 32]);  bmm_132 = None
    view_1334: "f32[128, 8, 49, 49]" = torch.ops.aten.reshape.default(bmm_133, [128, 8, 49, 49]);  bmm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    mul_831: "f32[128, 8, 49, 49]" = torch.ops.aten.mul.Tensor(view_1334, alias_45);  view_1334 = None
    sum_339: "f32[128, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_831, [-1], True)
    mul_832: "f32[128, 8, 49, 49]" = torch.ops.aten.mul.Tensor(alias_45, sum_339);  alias_45 = sum_339 = None
    sub_236: "f32[128, 8, 49, 49]" = torch.ops.aten.sub.Tensor(mul_831, mul_832);  mul_831 = mul_832 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_340: "f32[1, 8, 49, 49]" = torch.ops.aten.sum.dim_IntList(sub_236, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_21: "f32[8, 49, 49]" = torch.ops.aten.squeeze.dim(sum_340, 0);  sum_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_824: "f32[49, 49, 8]" = torch.ops.aten.permute.default(squeeze_21, [1, 2, 0]);  squeeze_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_1335: "f32[2401, 8]" = torch.ops.aten.reshape.default(permute_824, [2401, 8]);  permute_824 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_put_21: "f32[169, 8]" = torch.ops.aten.index_put_.default(full_default_62, [view_67], view_1335, True);  full_default_62 = view_67 = view_1335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_1336: "f32[1024, 49, 49]" = torch.ops.aten.reshape.default(sub_236, [1024, 49, 49]);  sub_236 = None
    bmm_134: "f32[1024, 32, 49]" = torch.ops.aten.bmm.default(permute_825, view_1336);  permute_825 = None
    bmm_135: "f32[1024, 49, 32]" = torch.ops.aten.bmm.default(view_1336, permute_826);  view_1336 = permute_826 = None
    view_1337: "f32[128, 8, 32, 49]" = torch.ops.aten.reshape.default(bmm_134, [128, 8, 32, 49]);  bmm_134 = None
    view_1338: "f32[128, 8, 49, 32]" = torch.ops.aten.reshape.default(bmm_135, [128, 8, 49, 32]);  bmm_135 = None
    permute_827: "f32[128, 8, 49, 32]" = torch.ops.aten.permute.default(view_1337, [0, 1, 3, 2]);  view_1337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_833: "f32[128, 8, 49, 32]" = torch.ops.aten.mul.Tensor(view_1338, 0.1767766952966369);  view_1338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    cat_21: "f32[384, 8, 49, 32]" = torch.ops.aten.cat.default([mul_833, permute_827, view_1333]);  mul_833 = permute_827 = view_1333 = None
    view_1339: "f32[3, 128, 8, 49, 32]" = torch.ops.aten.reshape.default(cat_21, [3, 128, 8, 49, 32]);  cat_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_828: "f32[128, 49, 3, 8, 32]" = torch.ops.aten.permute.default(view_1339, [1, 3, 0, 2, 4]);  view_1339 = None
    clone_349: "f32[128, 49, 3, 8, 32]" = torch.ops.aten.clone.default(permute_828, memory_format = torch.contiguous_format);  permute_828 = None
    view_1340: "f32[128, 49, 768]" = torch.ops.aten.reshape.default(clone_349, [128, 49, 768]);  clone_349 = None
    view_1341: "f32[6272, 768]" = torch.ops.aten.reshape.default(view_1340, [6272, 768]);  view_1340 = None
    mm_183: "f32[6272, 256]" = torch.ops.aten.mm.default(view_1341, permute_829);  permute_829 = None
    permute_830: "f32[768, 6272]" = torch.ops.aten.permute.default(view_1341, [1, 0])
    mm_184: "f32[768, 256]" = torch.ops.aten.mm.default(permute_830, view_61);  permute_830 = view_61 = None
    permute_831: "f32[256, 768]" = torch.ops.aten.permute.default(mm_184, [1, 0]);  mm_184 = None
    sum_341: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1341, [0], True);  view_1341 = None
    view_1342: "f32[768]" = torch.ops.aten.reshape.default(sum_341, [768]);  sum_341 = None
    permute_832: "f32[768, 256]" = torch.ops.aten.permute.default(permute_831, [1, 0]);  permute_831 = None
    view_1343: "f32[128, 49, 256]" = torch.ops.aten.reshape.default(mm_183, [128, 49, 256]);  mm_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_1344: "f32[128, 7, 7, 256]" = torch.ops.aten.reshape.default(view_1343, [128, 7, 7, 256]);  view_1343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_1345: "f32[8, 4, 4, 7, 7, 256]" = torch.ops.aten.reshape.default(view_1344, [8, 4, 4, 7, 7, 256]);  view_1344 = None
    permute_833: "f32[8, 4, 7, 4, 7, 256]" = torch.ops.aten.permute.default(view_1345, [0, 1, 3, 2, 4, 5]);  view_1345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_350: "f32[8, 4, 7, 4, 7, 256]" = torch.ops.aten.clone.default(permute_833, memory_format = torch.contiguous_format);  permute_833 = None
    view_1346: "f32[8, 28, 28, 256]" = torch.ops.aten.reshape.default(clone_350, [8, 28, 28, 256]);  clone_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    mul_835: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(view_1346, primals_56);  primals_56 = None
    mul_836: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_835, 256)
    sum_342: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_835, [3], True)
    mul_837: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_835, mul_22);  mul_835 = None
    sum_343: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_837, [3], True);  mul_837 = None
    mul_838: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_22, sum_343);  sum_343 = None
    sub_238: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(mul_836, sum_342);  mul_836 = sum_342 = None
    sub_239: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(sub_238, mul_838);  sub_238 = mul_838 = None
    div_117: "f32[8, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 256);  rsqrt_6 = None
    mul_839: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(div_117, sub_239);  div_117 = sub_239 = None
    mul_840: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(view_1346, mul_22);  mul_22 = None
    sum_344: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_840, [0, 1, 2]);  mul_840 = None
    sum_345: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_1346, [0, 1, 2]);  view_1346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_300: "f32[8, 28, 28, 256]" = torch.ops.aten.add.Tensor(view_1324, mul_839);  view_1324 = mul_839 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:358, code: x = self.reduction(x)
    view_1347: "f32[6272, 256]" = torch.ops.aten.reshape.default(add_300, [6272, 256]);  add_300 = None
    permute_834: "f32[256, 6272]" = torch.ops.aten.permute.default(view_1347, [1, 0])
    mm_185: "f32[256, 512]" = torch.ops.aten.mm.default(permute_834, view_56);  permute_834 = view_56 = None
    permute_835: "f32[512, 256]" = torch.ops.aten.permute.default(mm_185, [1, 0]);  mm_185 = None
    mm_186: "f32[6272, 512]" = torch.ops.aten.mm.default(view_1347, permute_836);  view_1347 = permute_836 = None
    view_1348: "f32[8, 28, 28, 512]" = torch.ops.aten.reshape.default(mm_186, [8, 28, 28, 512]);  mm_186 = None
    permute_837: "f32[256, 512]" = torch.ops.aten.permute.default(permute_835, [1, 0]);  permute_835 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:357, code: x = self.norm(x)
    mul_842: "f32[8, 28, 28, 512]" = torch.ops.aten.mul.Tensor(view_1348, primals_53);  primals_53 = None
    mul_843: "f32[8, 28, 28, 512]" = torch.ops.aten.mul.Tensor(mul_842, 512)
    sum_346: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_842, [3], True)
    mul_844: "f32[8, 28, 28, 512]" = torch.ops.aten.mul.Tensor(mul_842, mul_20);  mul_842 = None
    sum_347: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_844, [3], True);  mul_844 = None
    mul_845: "f32[8, 28, 28, 512]" = torch.ops.aten.mul.Tensor(mul_20, sum_347);  sum_347 = None
    sub_241: "f32[8, 28, 28, 512]" = torch.ops.aten.sub.Tensor(mul_843, sum_346);  mul_843 = sum_346 = None
    sub_242: "f32[8, 28, 28, 512]" = torch.ops.aten.sub.Tensor(sub_241, mul_845);  sub_241 = mul_845 = None
    mul_846: "f32[8, 28, 28, 512]" = torch.ops.aten.mul.Tensor(div_118, sub_242);  div_118 = sub_242 = None
    mul_847: "f32[8, 28, 28, 512]" = torch.ops.aten.mul.Tensor(view_1348, mul_20);  mul_20 = None
    sum_348: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_847, [0, 1, 2]);  mul_847 = None
    sum_349: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1348, [0, 1, 2]);  view_1348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:356, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
    view_1349: "f32[8, 28, 28, 2, 2, 128]" = torch.ops.aten.reshape.default(mul_846, [8, 28, 28, 2, 2, 128]);  mul_846 = None
    permute_838: "f32[8, 28, 2, 28, 2, 128]" = torch.ops.aten.permute.default(view_1349, [0, 1, 4, 2, 3, 5]);  view_1349 = None
    clone_351: "f32[8, 28, 2, 28, 2, 128]" = torch.ops.aten.clone.default(permute_838, memory_format = torch.contiguous_format);  permute_838 = None
    view_1350: "f32[8, 56, 56, 128]" = torch.ops.aten.reshape.default(clone_351, [8, 56, 56, 128]);  clone_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_1351: "f32[8, 3136, 128]" = torch.ops.aten.reshape.default(view_1350, [8, 3136, 128]);  view_1350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_848: "f32[8, 3136, 128]" = torch.ops.aten.mul.Tensor(view_1351, div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1352: "f32[25088, 128]" = torch.ops.aten.reshape.default(mul_848, [25088, 128]);  mul_848 = None
    mm_187: "f32[25088, 512]" = torch.ops.aten.mm.default(view_1352, permute_839);  permute_839 = None
    permute_840: "f32[128, 25088]" = torch.ops.aten.permute.default(view_1352, [1, 0])
    mm_188: "f32[128, 512]" = torch.ops.aten.mm.default(permute_840, view_51);  permute_840 = view_51 = None
    permute_841: "f32[512, 128]" = torch.ops.aten.permute.default(mm_188, [1, 0]);  mm_188 = None
    sum_350: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_1352, [0], True);  view_1352 = None
    view_1353: "f32[128]" = torch.ops.aten.reshape.default(sum_350, [128]);  sum_350 = None
    permute_842: "f32[128, 512]" = torch.ops.aten.permute.default(permute_841, [1, 0]);  permute_841 = None
    view_1354: "f32[8, 3136, 512]" = torch.ops.aten.reshape.default(mm_187, [8, 3136, 512]);  mm_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_850: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(add_17, 0.5);  add_17 = None
    mul_851: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_50, view_50)
    mul_852: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(mul_851, -0.5);  mul_851 = None
    exp_46: "f32[8, 3136, 512]" = torch.ops.aten.exp.default(mul_852);  mul_852 = None
    mul_853: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(exp_46, 0.3989422804014327);  exp_46 = None
    mul_854: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_50, mul_853);  view_50 = mul_853 = None
    add_302: "f32[8, 3136, 512]" = torch.ops.aten.add.Tensor(mul_850, mul_854);  mul_850 = mul_854 = None
    mul_855: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_1354, add_302);  view_1354 = add_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1355: "f32[25088, 512]" = torch.ops.aten.reshape.default(mul_855, [25088, 512]);  mul_855 = None
    mm_189: "f32[25088, 128]" = torch.ops.aten.mm.default(view_1355, permute_843);  permute_843 = None
    permute_844: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1355, [1, 0])
    mm_190: "f32[512, 128]" = torch.ops.aten.mm.default(permute_844, view_49);  permute_844 = view_49 = None
    permute_845: "f32[128, 512]" = torch.ops.aten.permute.default(mm_190, [1, 0]);  mm_190 = None
    sum_351: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1355, [0], True);  view_1355 = None
    view_1356: "f32[512]" = torch.ops.aten.reshape.default(sum_351, [512]);  sum_351 = None
    permute_846: "f32[512, 128]" = torch.ops.aten.permute.default(permute_845, [1, 0]);  permute_845 = None
    view_1357: "f32[8, 3136, 128]" = torch.ops.aten.reshape.default(mm_189, [8, 3136, 128]);  mm_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_857: "f32[8, 3136, 128]" = torch.ops.aten.mul.Tensor(view_1357, primals_47);  primals_47 = None
    mul_858: "f32[8, 3136, 128]" = torch.ops.aten.mul.Tensor(mul_857, 128)
    sum_352: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_857, [2], True)
    mul_859: "f32[8, 3136, 128]" = torch.ops.aten.mul.Tensor(mul_857, mul_14);  mul_857 = None
    sum_353: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_859, [2], True);  mul_859 = None
    mul_860: "f32[8, 3136, 128]" = torch.ops.aten.mul.Tensor(mul_14, sum_353);  sum_353 = None
    sub_244: "f32[8, 3136, 128]" = torch.ops.aten.sub.Tensor(mul_858, sum_352);  mul_858 = sum_352 = None
    sub_245: "f32[8, 3136, 128]" = torch.ops.aten.sub.Tensor(sub_244, mul_860);  sub_244 = mul_860 = None
    mul_861: "f32[8, 3136, 128]" = torch.ops.aten.mul.Tensor(div_119, sub_245);  div_119 = sub_245 = None
    mul_862: "f32[8, 3136, 128]" = torch.ops.aten.mul.Tensor(view_1357, mul_14);  mul_14 = None
    sum_354: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_862, [0, 1]);  mul_862 = None
    sum_355: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_1357, [0, 1]);  view_1357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_303: "f32[8, 3136, 128]" = torch.ops.aten.add.Tensor(view_1351, mul_861);  view_1351 = mul_861 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_1358: "f32[8, 56, 56, 128]" = torch.ops.aten.reshape.default(add_303, [8, 56, 56, 128]);  add_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/drop.py:154, code: return x * random_tensor
    mul_863: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(view_1358, div_2);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:316, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    roll_42: "f32[8, 56, 56, 128]" = torch.ops.aten.roll.default(mul_863, [-3, -3], [2, 1]);  mul_863 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:312, code: shifted_x = shifted_x[:, :H, :W, :].contiguous()
    full_default_66: "f32[8, 56, 56, 128]" = torch.ops.aten.full.default([8, 56, 56, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_1359: "f32[8, 8, 7, 8, 7, 128]" = torch.ops.aten.reshape.default(roll_42, [8, 8, 7, 8, 7, 128]);  roll_42 = None
    permute_847: "f32[8, 8, 8, 7, 7, 128]" = torch.ops.aten.permute.default(view_1359, [0, 1, 3, 2, 4, 5]);  view_1359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_352: "f32[8, 8, 8, 7, 7, 128]" = torch.ops.aten.clone.default(permute_847, memory_format = torch.contiguous_format);  permute_847 = None
    view_1360: "f32[512, 7, 7, 128]" = torch.ops.aten.reshape.default(clone_352, [512, 7, 7, 128]);  clone_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_1361: "f32[512, 49, 128]" = torch.ops.aten.reshape.default(view_1360, [512, 49, 128]);  view_1360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_1362: "f32[25088, 128]" = torch.ops.aten.reshape.default(view_1361, [25088, 128]);  view_1361 = None
    mm_191: "f32[25088, 128]" = torch.ops.aten.mm.default(view_1362, permute_848);  permute_848 = None
    permute_849: "f32[128, 25088]" = torch.ops.aten.permute.default(view_1362, [1, 0])
    mm_192: "f32[128, 128]" = torch.ops.aten.mm.default(permute_849, view_43);  permute_849 = view_43 = None
    permute_850: "f32[128, 128]" = torch.ops.aten.permute.default(mm_192, [1, 0]);  mm_192 = None
    sum_356: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_1362, [0], True);  view_1362 = None
    view_1363: "f32[128]" = torch.ops.aten.reshape.default(sum_356, [128]);  sum_356 = None
    permute_851: "f32[128, 128]" = torch.ops.aten.permute.default(permute_850, [1, 0]);  permute_850 = None
    view_1364: "f32[512, 49, 128]" = torch.ops.aten.reshape.default(mm_191, [512, 49, 128]);  mm_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_1365: "f32[512, 49, 4, 32]" = torch.ops.aten.reshape.default(view_1364, [512, 49, 4, 32]);  view_1364 = None
    permute_852: "f32[512, 4, 49, 32]" = torch.ops.aten.permute.default(view_1365, [0, 2, 1, 3]);  view_1365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_353: "f32[512, 4, 49, 32]" = torch.ops.aten.clone.default(permute_852, memory_format = torch.contiguous_format);  permute_852 = None
    view_1366: "f32[2048, 49, 32]" = torch.ops.aten.reshape.default(clone_353, [2048, 49, 32]);  clone_353 = None
    bmm_136: "f32[2048, 49, 32]" = torch.ops.aten.bmm.default(permute_853, view_1366);  permute_853 = None
    bmm_137: "f32[2048, 49, 49]" = torch.ops.aten.bmm.default(view_1366, permute_854);  view_1366 = permute_854 = None
    view_1367: "f32[512, 4, 49, 32]" = torch.ops.aten.reshape.default(bmm_136, [512, 4, 49, 32]);  bmm_136 = None
    view_1368: "f32[512, 4, 49, 49]" = torch.ops.aten.reshape.default(bmm_137, [512, 4, 49, 49]);  bmm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    mul_864: "f32[512, 4, 49, 49]" = torch.ops.aten.mul.Tensor(view_1368, alias_46);  view_1368 = None
    sum_357: "f32[512, 4, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_864, [-1], True)
    mul_865: "f32[512, 4, 49, 49]" = torch.ops.aten.mul.Tensor(alias_46, sum_357);  alias_46 = sum_357 = None
    sub_246: "f32[512, 4, 49, 49]" = torch.ops.aten.sub.Tensor(mul_864, mul_865);  mul_864 = mul_865 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:176, code: attn = attn.view(-1, self.num_heads, N, N)
    view_1369: "f32[8, 64, 4, 49, 49]" = torch.ops.aten.reshape.default(sub_246, [8, 64, 4, 49, 49]);  sub_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:175, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    view_1370: "f32[512, 4, 49, 49]" = torch.ops.aten.reshape.default(view_1369, [512, 4, 49, 49]);  view_1369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_358: "f32[1, 4, 49, 49]" = torch.ops.aten.sum.dim_IntList(view_1370, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_22: "f32[4, 49, 49]" = torch.ops.aten.squeeze.dim(sum_358, 0);  sum_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_855: "f32[49, 49, 4]" = torch.ops.aten.permute.default(squeeze_22, [1, 2, 0]);  squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_1371: "f32[2401, 4]" = torch.ops.aten.reshape.default(permute_855, [2401, 4]);  permute_855 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    full_default_68: "f32[169, 4]" = torch.ops.aten.full.default([169, 4], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    index_put_22: "f32[169, 4]" = torch.ops.aten.index_put.default(full_default_68, [view_35], view_1371, True);  view_35 = view_1371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_1372: "f32[2048, 49, 49]" = torch.ops.aten.reshape.default(view_1370, [2048, 49, 49]);  view_1370 = None
    bmm_138: "f32[2048, 32, 49]" = torch.ops.aten.bmm.default(permute_856, view_1372);  permute_856 = None
    bmm_139: "f32[2048, 49, 32]" = torch.ops.aten.bmm.default(view_1372, permute_857);  view_1372 = permute_857 = None
    view_1373: "f32[512, 4, 32, 49]" = torch.ops.aten.reshape.default(bmm_138, [512, 4, 32, 49]);  bmm_138 = None
    view_1374: "f32[512, 4, 49, 32]" = torch.ops.aten.reshape.default(bmm_139, [512, 4, 49, 32]);  bmm_139 = None
    permute_858: "f32[512, 4, 49, 32]" = torch.ops.aten.permute.default(view_1373, [0, 1, 3, 2]);  view_1373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_866: "f32[512, 4, 49, 32]" = torch.ops.aten.mul.Tensor(view_1374, 0.1767766952966369);  view_1374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    cat_22: "f32[1536, 4, 49, 32]" = torch.ops.aten.cat.default([mul_866, permute_858, view_1367]);  mul_866 = permute_858 = view_1367 = None
    view_1375: "f32[3, 512, 4, 49, 32]" = torch.ops.aten.reshape.default(cat_22, [3, 512, 4, 49, 32]);  cat_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_859: "f32[512, 49, 3, 4, 32]" = torch.ops.aten.permute.default(view_1375, [1, 3, 0, 2, 4]);  view_1375 = None
    clone_354: "f32[512, 49, 3, 4, 32]" = torch.ops.aten.clone.default(permute_859, memory_format = torch.contiguous_format);  permute_859 = None
    view_1376: "f32[512, 49, 384]" = torch.ops.aten.reshape.default(clone_354, [512, 49, 384]);  clone_354 = None
    view_1377: "f32[25088, 384]" = torch.ops.aten.reshape.default(view_1376, [25088, 384]);  view_1376 = None
    mm_193: "f32[25088, 128]" = torch.ops.aten.mm.default(view_1377, permute_860);  permute_860 = None
    permute_861: "f32[384, 25088]" = torch.ops.aten.permute.default(view_1377, [1, 0])
    mm_194: "f32[384, 128]" = torch.ops.aten.mm.default(permute_861, view_29);  permute_861 = view_29 = None
    permute_862: "f32[128, 384]" = torch.ops.aten.permute.default(mm_194, [1, 0]);  mm_194 = None
    sum_359: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_1377, [0], True);  view_1377 = None
    view_1378: "f32[384]" = torch.ops.aten.reshape.default(sum_359, [384]);  sum_359 = None
    permute_863: "f32[384, 128]" = torch.ops.aten.permute.default(permute_862, [1, 0]);  permute_862 = None
    view_1379: "f32[512, 49, 128]" = torch.ops.aten.reshape.default(mm_193, [512, 49, 128]);  mm_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_1380: "f32[512, 7, 7, 128]" = torch.ops.aten.reshape.default(view_1379, [512, 7, 7, 128]);  view_1379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_1381: "f32[8, 8, 8, 7, 7, 128]" = torch.ops.aten.reshape.default(view_1380, [8, 8, 8, 7, 7, 128]);  view_1380 = None
    permute_864: "f32[8, 8, 7, 8, 7, 128]" = torch.ops.aten.permute.default(view_1381, [0, 1, 3, 2, 4, 5]);  view_1381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_355: "f32[8, 8, 7, 8, 7, 128]" = torch.ops.aten.clone.default(permute_864, memory_format = torch.contiguous_format);  permute_864 = None
    view_1382: "f32[8, 56, 56, 128]" = torch.ops.aten.reshape.default(clone_355, [8, 56, 56, 128]);  clone_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:292, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    roll_43: "f32[8, 56, 56, 128]" = torch.ops.aten.roll.default(view_1382, [3, 3], [2, 1]);  view_1382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    mul_868: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(roll_43, primals_41);  primals_41 = None
    mul_869: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_868, 128)
    sum_360: "f32[8, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_868, [3], True)
    mul_870: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_868, mul_10);  mul_868 = None
    sum_361: "f32[8, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_870, [3], True);  mul_870 = None
    mul_871: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_10, sum_361);  sum_361 = None
    sub_248: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(mul_869, sum_360);  mul_869 = sum_360 = None
    sub_249: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(sub_248, mul_871);  sub_248 = mul_871 = None
    mul_872: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(div_120, sub_249);  div_120 = sub_249 = None
    mul_873: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(roll_43, mul_10);  mul_10 = None
    sum_362: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_873, [0, 1, 2]);  mul_873 = None
    sum_363: "f32[128]" = torch.ops.aten.sum.dim_IntList(roll_43, [0, 1, 2]);  roll_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_304: "f32[8, 56, 56, 128]" = torch.ops.aten.add.Tensor(view_1358, mul_872);  view_1358 = mul_872 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:326, code: x = x.reshape(B, H, W, C)
    view_1383: "f32[8, 3136, 128]" = torch.ops.aten.reshape.default(add_304, [8, 3136, 128]);  add_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1384: "f32[25088, 128]" = torch.ops.aten.reshape.default(view_1383, [25088, 128])
    mm_195: "f32[25088, 512]" = torch.ops.aten.mm.default(view_1384, permute_865);  permute_865 = None
    permute_866: "f32[128, 25088]" = torch.ops.aten.permute.default(view_1384, [1, 0])
    mm_196: "f32[128, 512]" = torch.ops.aten.mm.default(permute_866, view_23);  permute_866 = view_23 = None
    permute_867: "f32[512, 128]" = torch.ops.aten.permute.default(mm_196, [1, 0]);  mm_196 = None
    sum_364: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_1384, [0], True);  view_1384 = None
    view_1385: "f32[128]" = torch.ops.aten.reshape.default(sum_364, [128]);  sum_364 = None
    permute_868: "f32[128, 512]" = torch.ops.aten.permute.default(permute_867, [1, 0]);  permute_867 = None
    view_1386: "f32[8, 3136, 512]" = torch.ops.aten.reshape.default(mm_195, [8, 3136, 512]);  mm_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_875: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(add_8, 0.5);  add_8 = None
    mul_876: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_22, view_22)
    mul_877: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(mul_876, -0.5);  mul_876 = None
    exp_47: "f32[8, 3136, 512]" = torch.ops.aten.exp.default(mul_877);  mul_877 = None
    mul_878: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(exp_47, 0.3989422804014327);  exp_47 = None
    mul_879: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_22, mul_878);  view_22 = mul_878 = None
    add_306: "f32[8, 3136, 512]" = torch.ops.aten.add.Tensor(mul_875, mul_879);  mul_875 = mul_879 = None
    mul_880: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_1386, add_306);  view_1386 = add_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1387: "f32[25088, 512]" = torch.ops.aten.reshape.default(mul_880, [25088, 512]);  mul_880 = None
    mm_197: "f32[25088, 128]" = torch.ops.aten.mm.default(view_1387, permute_869);  permute_869 = None
    permute_870: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1387, [1, 0])
    mm_198: "f32[512, 128]" = torch.ops.aten.mm.default(permute_870, view_21);  permute_870 = view_21 = None
    permute_871: "f32[128, 512]" = torch.ops.aten.permute.default(mm_198, [1, 0]);  mm_198 = None
    sum_365: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1387, [0], True);  view_1387 = None
    view_1388: "f32[512]" = torch.ops.aten.reshape.default(sum_365, [512]);  sum_365 = None
    permute_872: "f32[512, 128]" = torch.ops.aten.permute.default(permute_871, [1, 0]);  permute_871 = None
    view_1389: "f32[8, 3136, 128]" = torch.ops.aten.reshape.default(mm_197, [8, 3136, 128]);  mm_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_882: "f32[8, 3136, 128]" = torch.ops.aten.mul.Tensor(view_1389, primals_35);  primals_35 = None
    mul_883: "f32[8, 3136, 128]" = torch.ops.aten.mul.Tensor(mul_882, 128)
    sum_366: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_882, [2], True)
    mul_884: "f32[8, 3136, 128]" = torch.ops.aten.mul.Tensor(mul_882, mul_5);  mul_882 = None
    sum_367: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_884, [2], True);  mul_884 = None
    mul_885: "f32[8, 3136, 128]" = torch.ops.aten.mul.Tensor(mul_5, sum_367);  sum_367 = None
    sub_251: "f32[8, 3136, 128]" = torch.ops.aten.sub.Tensor(mul_883, sum_366);  mul_883 = sum_366 = None
    sub_252: "f32[8, 3136, 128]" = torch.ops.aten.sub.Tensor(sub_251, mul_885);  sub_251 = mul_885 = None
    mul_886: "f32[8, 3136, 128]" = torch.ops.aten.mul.Tensor(div_121, sub_252);  div_121 = sub_252 = None
    mul_887: "f32[8, 3136, 128]" = torch.ops.aten.mul.Tensor(view_1389, mul_5);  mul_5 = None
    sum_368: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_887, [0, 1]);  mul_887 = None
    sum_369: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_1389, [0, 1]);  view_1389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:325, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_307: "f32[8, 3136, 128]" = torch.ops.aten.add.Tensor(view_1383, mul_886);  view_1383 = mul_886 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:324, code: x = x.reshape(B, -1, C)
    view_1390: "f32[8, 56, 56, 128]" = torch.ops.aten.reshape.default(add_307, [8, 56, 56, 128]);  add_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:75, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    view_1391: "f32[8, 8, 7, 8, 7, 128]" = torch.ops.aten.reshape.default(view_1390, [8, 8, 7, 8, 7, 128])
    permute_873: "f32[8, 8, 8, 7, 7, 128]" = torch.ops.aten.permute.default(view_1391, [0, 1, 3, 2, 4, 5]);  view_1391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:74, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    clone_356: "f32[8, 8, 8, 7, 7, 128]" = torch.ops.aten.clone.default(permute_873, memory_format = torch.contiguous_format);  permute_873 = None
    view_1392: "f32[512, 7, 7, 128]" = torch.ops.aten.reshape.default(clone_356, [512, 7, 7, 128]);  clone_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:310, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    view_1393: "f32[512, 49, 128]" = torch.ops.aten.reshape.default(view_1392, [512, 49, 128]);  view_1392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:182, code: x = self.proj(x)
    view_1394: "f32[25088, 128]" = torch.ops.aten.reshape.default(view_1393, [25088, 128]);  view_1393 = None
    mm_199: "f32[25088, 128]" = torch.ops.aten.mm.default(view_1394, permute_874);  permute_874 = None
    permute_875: "f32[128, 25088]" = torch.ops.aten.permute.default(view_1394, [1, 0])
    mm_200: "f32[128, 128]" = torch.ops.aten.mm.default(permute_875, view_15);  permute_875 = view_15 = None
    permute_876: "f32[128, 128]" = torch.ops.aten.permute.default(mm_200, [1, 0]);  mm_200 = None
    sum_370: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_1394, [0], True);  view_1394 = None
    view_1395: "f32[128]" = torch.ops.aten.reshape.default(sum_370, [128]);  sum_370 = None
    permute_877: "f32[128, 128]" = torch.ops.aten.permute.default(permute_876, [1, 0]);  permute_876 = None
    view_1396: "f32[512, 49, 128]" = torch.ops.aten.reshape.default(mm_199, [512, 49, 128]);  mm_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:181, code: x = x.transpose(1, 2).reshape(B_, N, -1)
    view_1397: "f32[512, 49, 4, 32]" = torch.ops.aten.reshape.default(view_1396, [512, 49, 4, 32]);  view_1396 = None
    permute_878: "f32[512, 4, 49, 32]" = torch.ops.aten.permute.default(view_1397, [0, 2, 1, 3]);  view_1397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:179, code: x = attn @ v
    clone_357: "f32[512, 4, 49, 32]" = torch.ops.aten.clone.default(permute_878, memory_format = torch.contiguous_format);  permute_878 = None
    view_1398: "f32[2048, 49, 32]" = torch.ops.aten.reshape.default(clone_357, [2048, 49, 32]);  clone_357 = None
    bmm_140: "f32[2048, 49, 32]" = torch.ops.aten.bmm.default(permute_879, view_1398);  permute_879 = None
    bmm_141: "f32[2048, 49, 49]" = torch.ops.aten.bmm.default(view_1398, permute_880);  view_1398 = permute_880 = None
    view_1399: "f32[512, 4, 49, 32]" = torch.ops.aten.reshape.default(bmm_140, [512, 4, 49, 32]);  bmm_140 = None
    view_1400: "f32[512, 4, 49, 49]" = torch.ops.aten.reshape.default(bmm_141, [512, 4, 49, 49]);  bmm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:177, code: attn = self.softmax(attn)
    mul_888: "f32[512, 4, 49, 49]" = torch.ops.aten.mul.Tensor(view_1400, alias_47);  view_1400 = None
    sum_371: "f32[512, 4, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_888, [-1], True)
    mul_889: "f32[512, 4, 49, 49]" = torch.ops.aten.mul.Tensor(alias_47, sum_371);  alias_47 = sum_371 = None
    sub_253: "f32[512, 4, 49, 49]" = torch.ops.aten.sub.Tensor(mul_888, mul_889);  mul_888 = mul_889 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:172, code: attn = attn + self._get_rel_pos_bias()
    sum_372: "f32[1, 4, 49, 49]" = torch.ops.aten.sum.dim_IntList(sub_253, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:146, code: return relative_position_bias.unsqueeze(0)
    squeeze_23: "f32[4, 49, 49]" = torch.ops.aten.squeeze.dim(sum_372, 0);  sum_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:145, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_881: "f32[49, 49, 4]" = torch.ops.aten.permute.default(squeeze_23, [1, 2, 0]);  squeeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:144, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
    view_1401: "f32[2401, 4]" = torch.ops.aten.reshape.default(permute_881, [2401, 4]);  permute_881 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:143, code: relative_position_bias = self.relative_position_bias_table[
    index_put_23: "f32[169, 4]" = torch.ops.aten.index_put_.default(full_default_68, [view_9], view_1401, True);  full_default_68 = view_9 = view_1401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:171, code: attn = q @ k.transpose(-2, -1)
    view_1402: "f32[2048, 49, 49]" = torch.ops.aten.reshape.default(sub_253, [2048, 49, 49]);  sub_253 = None
    bmm_142: "f32[2048, 32, 49]" = torch.ops.aten.bmm.default(permute_882, view_1402);  permute_882 = None
    bmm_143: "f32[2048, 49, 32]" = torch.ops.aten.bmm.default(view_1402, permute_883);  view_1402 = permute_883 = None
    view_1403: "f32[512, 4, 32, 49]" = torch.ops.aten.reshape.default(bmm_142, [512, 4, 32, 49]);  bmm_142 = None
    view_1404: "f32[512, 4, 49, 32]" = torch.ops.aten.reshape.default(bmm_143, [512, 4, 49, 32]);  bmm_143 = None
    permute_884: "f32[512, 4, 49, 32]" = torch.ops.aten.permute.default(view_1403, [0, 1, 3, 2]);  view_1403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:170, code: q = q * self.scale
    mul_890: "f32[512, 4, 49, 32]" = torch.ops.aten.mul.Tensor(view_1404, 0.1767766952966369);  view_1404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:156, code: q, k, v = qkv.unbind(0)
    cat_23: "f32[1536, 4, 49, 32]" = torch.ops.aten.cat.default([mul_890, permute_884, view_1399]);  mul_890 = permute_884 = view_1399 = None
    view_1405: "f32[3, 512, 4, 49, 32]" = torch.ops.aten.reshape.default(cat_23, [3, 512, 4, 49, 32]);  cat_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:155, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_885: "f32[512, 49, 3, 4, 32]" = torch.ops.aten.permute.default(view_1405, [1, 3, 0, 2, 4]);  view_1405 = None
    clone_358: "f32[512, 49, 3, 4, 32]" = torch.ops.aten.clone.default(permute_885, memory_format = torch.contiguous_format);  permute_885 = None
    view_1406: "f32[512, 49, 384]" = torch.ops.aten.reshape.default(clone_358, [512, 49, 384]);  clone_358 = None
    view_1407: "f32[25088, 384]" = torch.ops.aten.reshape.default(view_1406, [25088, 384]);  view_1406 = None
    mm_201: "f32[25088, 128]" = torch.ops.aten.mm.default(view_1407, permute_886);  permute_886 = None
    permute_887: "f32[384, 25088]" = torch.ops.aten.permute.default(view_1407, [1, 0])
    mm_202: "f32[384, 128]" = torch.ops.aten.mm.default(permute_887, view_3);  permute_887 = view_3 = None
    permute_888: "f32[128, 384]" = torch.ops.aten.permute.default(mm_202, [1, 0]);  mm_202 = None
    sum_373: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_1407, [0], True);  view_1407 = None
    view_1408: "f32[384]" = torch.ops.aten.reshape.default(sum_373, [384]);  sum_373 = None
    permute_889: "f32[384, 128]" = torch.ops.aten.permute.default(permute_888, [1, 0]);  permute_888 = None
    view_1409: "f32[512, 49, 128]" = torch.ops.aten.reshape.default(mm_201, [512, 49, 128]);  mm_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:304, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
    view_1410: "f32[512, 7, 7, 128]" = torch.ops.aten.reshape.default(view_1409, [512, 7, 7, 128]);  view_1409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:57, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    view_1411: "f32[8, 8, 8, 7, 7, 128]" = torch.ops.aten.reshape.default(view_1410, [8, 8, 8, 7, 7, 128]);  view_1410 = None
    permute_890: "f32[8, 8, 7, 8, 7, 128]" = torch.ops.aten.permute.default(view_1411, [0, 1, 3, 2, 4, 5]);  view_1411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:56, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    clone_359: "f32[8, 8, 7, 8, 7, 128]" = torch.ops.aten.clone.default(permute_890, memory_format = torch.contiguous_format);  permute_890 = None
    view_1412: "f32[8, 56, 56, 128]" = torch.ops.aten.reshape.default(clone_359, [8, 56, 56, 128]);  clone_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    mul_892: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(view_1412, primals_29);  primals_29 = None
    mul_893: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_892, 128)
    sum_374: "f32[8, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_892, [3], True)
    mul_894: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_892, mul_2);  mul_892 = None
    sum_375: "f32[8, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_894, [3], True);  mul_894 = None
    mul_895: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_2, sum_375);  sum_375 = None
    sub_255: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(mul_893, sum_374);  mul_893 = sum_374 = None
    sub_256: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(sub_255, mul_895);  sub_255 = mul_895 = None
    mul_896: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(div_122, sub_256);  div_122 = sub_256 = None
    mul_897: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(view_1412, mul_2);  mul_2 = None
    sum_376: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_897, [0, 1, 2]);  mul_897 = None
    sum_377: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_1412, [0, 1, 2]);  view_1412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/swin_transformer.py:323, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
    add_308: "f32[8, 56, 56, 128]" = torch.ops.aten.add.Tensor(view_1390, mul_896);  view_1390 = mul_896 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    mul_899: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(add_308, primals_27);  primals_27 = None
    mul_900: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_899, 128)
    sum_378: "f32[8, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_899, [3], True)
    mul_901: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_899, mul);  mul_899 = None
    sum_379: "f32[8, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_901, [3], True);  mul_901 = None
    mul_902: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul, sum_379);  sum_379 = None
    sub_258: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(mul_900, sum_378);  mul_900 = sum_378 = None
    sub_259: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(sub_258, mul_902);  sub_258 = mul_902 = None
    mul_903: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(div_123, sub_259);  div_123 = sub_259 = None
    mul_904: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(add_308, mul);  mul = None
    sum_380: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_904, [0, 1, 2]);  mul_904 = None
    sum_381: "f32[128]" = torch.ops.aten.sum.dim_IntList(add_308, [0, 1, 2]);  add_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/format.py:43, code: x = x.permute(0, 2, 3, 1)
    permute_891: "f32[8, 128, 56, 56]" = torch.ops.aten.permute.default(mul_903, [0, 3, 1, 2]);  mul_903 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    sum_382: "f32[128]" = torch.ops.aten.sum.dim_IntList(permute_891, [0, 2, 3])
    convolution_backward = torch.ops.aten.convolution_backward.default(permute_891, primals_365, primals_25, [128], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  permute_891 = primals_365 = primals_25 = None
    getitem_179: "f32[128, 3, 4, 4]" = convolution_backward[1];  convolution_backward = None
    return [index_put_23, index_put_22, index_put_21, index_put_20, index_put_19, index_put_18, index_put_17, index_put_16, index_put_15, index_put_14, index_put_13, index_put_12, index_put_11, index_put_10, index_put_9, index_put_8, index_put_7, index_put_6, index_put_5, index_put_4, index_put_3, index_put_2, index_put_1, index_put, getitem_179, sum_382, sum_380, sum_381, sum_376, sum_377, permute_889, view_1408, permute_877, view_1395, sum_368, sum_369, permute_872, view_1388, permute_868, view_1385, sum_362, sum_363, permute_863, view_1378, permute_851, view_1363, sum_354, sum_355, permute_846, view_1356, permute_842, view_1353, sum_348, sum_349, permute_837, sum_344, sum_345, permute_832, view_1342, permute_820, view_1329, sum_336, sum_337, permute_815, view_1322, permute_811, view_1319, sum_330, sum_331, permute_806, view_1312, permute_794, view_1297, sum_322, sum_323, permute_789, view_1290, permute_785, view_1287, sum_316, sum_317, permute_780, sum_312, sum_313, permute_775, view_1276, permute_763, view_1263, sum_304, sum_305, permute_758, view_1256, permute_754, view_1253, sum_298, sum_299, permute_749, view_1246, permute_737, view_1231, sum_290, sum_291, permute_732, view_1224, permute_728, view_1221, sum_284, sum_285, permute_723, view_1214, permute_711, view_1201, sum_276, sum_277, permute_706, view_1194, permute_702, view_1191, sum_270, sum_271, permute_697, view_1184, permute_685, view_1169, sum_262, sum_263, permute_680, view_1162, permute_676, view_1159, sum_256, sum_257, permute_671, view_1152, permute_659, view_1139, sum_248, sum_249, permute_654, view_1132, permute_650, view_1129, sum_242, sum_243, permute_645, view_1122, permute_633, view_1107, sum_234, sum_235, permute_628, view_1100, permute_624, view_1097, sum_228, sum_229, permute_619, view_1090, permute_607, view_1077, sum_220, sum_221, permute_602, view_1070, permute_598, view_1067, sum_214, sum_215, permute_593, view_1060, permute_581, view_1045, sum_206, sum_207, permute_576, view_1038, permute_572, view_1035, sum_200, sum_201, permute_567, view_1028, permute_555, view_1015, sum_192, sum_193, permute_550, view_1008, permute_546, view_1005, sum_186, sum_187, permute_541, view_998, permute_529, view_983, sum_178, sum_179, permute_524, view_976, permute_520, view_973, sum_172, sum_173, permute_515, view_966, permute_503, view_953, sum_164, sum_165, permute_498, view_946, permute_494, view_943, sum_158, sum_159, permute_489, view_936, permute_477, view_921, sum_150, sum_151, permute_472, view_914, permute_468, view_911, sum_144, sum_145, permute_463, view_904, permute_451, view_891, sum_136, sum_137, permute_446, view_884, permute_442, view_881, sum_130, sum_131, permute_437, view_874, permute_425, view_859, sum_122, sum_123, permute_420, view_852, permute_416, view_849, sum_116, sum_117, permute_411, view_842, permute_399, view_829, sum_108, sum_109, permute_394, view_822, permute_390, view_819, sum_102, sum_103, permute_385, view_812, permute_373, view_797, sum_94, sum_95, permute_368, view_790, permute_364, view_787, sum_88, sum_89, permute_359, view_780, permute_347, view_767, sum_80, sum_81, permute_342, view_760, permute_338, view_757, sum_74, sum_75, permute_333, view_750, permute_321, view_735, sum_66, sum_67, permute_316, view_728, permute_312, view_725, sum_60, sum_61, permute_307, sum_56, sum_57, permute_302, view_714, permute_290, view_701, sum_48, sum_49, permute_285, view_694, permute_281, view_691, sum_42, sum_43, permute_276, view_684, permute_264, view_671, sum_34, sum_35, permute_259, view_664, permute_255, view_661, sum_28, sum_29, permute_251, view_658, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    