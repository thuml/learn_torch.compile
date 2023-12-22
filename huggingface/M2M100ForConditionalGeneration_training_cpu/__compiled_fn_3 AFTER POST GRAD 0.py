from __future__ import annotations



def forward(self, primals_1: "f32[128112, 1024]", primals_2: "f32[1024]", primals_3: "f32[1024]", primals_4: "f32[1024, 1024]", primals_5: "f32[1024]", primals_6: "f32[1024, 1024]", primals_7: "f32[1024]", primals_8: "f32[1024, 1024]", primals_9: "f32[1024]", primals_10: "f32[1024, 1024]", primals_11: "f32[1024]", primals_12: "f32[1024]", primals_13: "f32[1024]", primals_14: "f32[4096, 1024]", primals_15: "f32[4096]", primals_16: "f32[1024, 4096]", primals_17: "f32[1024]", primals_18: "f32[1024]", primals_19: "f32[1024]", primals_20: "f32[1024, 1024]", primals_21: "f32[1024]", primals_22: "f32[1024, 1024]", primals_23: "f32[1024]", primals_24: "f32[1024, 1024]", primals_25: "f32[1024]", primals_26: "f32[1024, 1024]", primals_27: "f32[1024]", primals_28: "f32[1024]", primals_29: "f32[1024]", primals_30: "f32[4096, 1024]", primals_31: "f32[4096]", primals_32: "f32[1024, 4096]", primals_33: "f32[1024]", primals_34: "f32[1024]", primals_35: "f32[1024]", primals_36: "f32[1024, 1024]", primals_37: "f32[1024]", primals_38: "f32[1024, 1024]", primals_39: "f32[1024]", primals_40: "f32[1024, 1024]", primals_41: "f32[1024]", primals_42: "f32[1024, 1024]", primals_43: "f32[1024]", primals_44: "f32[1024]", primals_45: "f32[1024]", primals_46: "f32[4096, 1024]", primals_47: "f32[4096]", primals_48: "f32[1024, 4096]", primals_49: "f32[1024]", primals_50: "f32[1024]", primals_51: "f32[1024]", primals_52: "f32[1024, 1024]", primals_53: "f32[1024]", primals_54: "f32[1024, 1024]", primals_55: "f32[1024]", primals_56: "f32[1024, 1024]", primals_57: "f32[1024]", primals_58: "f32[1024, 1024]", primals_59: "f32[1024]", primals_60: "f32[1024]", primals_61: "f32[1024]", primals_62: "f32[4096, 1024]", primals_63: "f32[4096]", primals_64: "f32[1024, 4096]", primals_65: "f32[1024]", primals_66: "f32[1024]", primals_67: "f32[1024]", primals_68: "f32[1024, 1024]", primals_69: "f32[1024]", primals_70: "f32[1024, 1024]", primals_71: "f32[1024]", primals_72: "f32[1024, 1024]", primals_73: "f32[1024]", primals_74: "f32[1024, 1024]", primals_75: "f32[1024]", primals_76: "f32[1024]", primals_77: "f32[1024]", primals_78: "f32[4096, 1024]", primals_79: "f32[4096]", primals_80: "f32[1024, 4096]", primals_81: "f32[1024]", primals_82: "f32[1024]", primals_83: "f32[1024]", primals_84: "f32[1024, 1024]", primals_85: "f32[1024]", primals_86: "f32[1024, 1024]", primals_87: "f32[1024]", primals_88: "f32[1024, 1024]", primals_89: "f32[1024]", primals_90: "f32[1024, 1024]", primals_91: "f32[1024]", primals_92: "f32[1024]", primals_93: "f32[1024]", primals_94: "f32[4096, 1024]", primals_95: "f32[4096]", primals_96: "f32[1024, 4096]", primals_97: "f32[1024]", primals_98: "f32[1024]", primals_99: "f32[1024]", primals_100: "f32[1024, 1024]", primals_101: "f32[1024]", primals_102: "f32[1024, 1024]", primals_103: "f32[1024]", primals_104: "f32[1024, 1024]", primals_105: "f32[1024]", primals_106: "f32[1024, 1024]", primals_107: "f32[1024]", primals_108: "f32[1024]", primals_109: "f32[1024]", primals_110: "f32[4096, 1024]", primals_111: "f32[4096]", primals_112: "f32[1024, 4096]", primals_113: "f32[1024]", primals_114: "f32[1024]", primals_115: "f32[1024]", primals_116: "f32[1024, 1024]", primals_117: "f32[1024]", primals_118: "f32[1024, 1024]", primals_119: "f32[1024]", primals_120: "f32[1024, 1024]", primals_121: "f32[1024]", primals_122: "f32[1024, 1024]", primals_123: "f32[1024]", primals_124: "f32[1024]", primals_125: "f32[1024]", primals_126: "f32[4096, 1024]", primals_127: "f32[4096]", primals_128: "f32[1024, 4096]", primals_129: "f32[1024]", primals_130: "f32[1024]", primals_131: "f32[1024]", primals_132: "f32[1024, 1024]", primals_133: "f32[1024]", primals_134: "f32[1024, 1024]", primals_135: "f32[1024]", primals_136: "f32[1024, 1024]", primals_137: "f32[1024]", primals_138: "f32[1024, 1024]", primals_139: "f32[1024]", primals_140: "f32[1024]", primals_141: "f32[1024]", primals_142: "f32[4096, 1024]", primals_143: "f32[4096]", primals_144: "f32[1024, 4096]", primals_145: "f32[1024]", primals_146: "f32[1024]", primals_147: "f32[1024]", primals_148: "f32[1024, 1024]", primals_149: "f32[1024]", primals_150: "f32[1024, 1024]", primals_151: "f32[1024]", primals_152: "f32[1024, 1024]", primals_153: "f32[1024]", primals_154: "f32[1024, 1024]", primals_155: "f32[1024]", primals_156: "f32[1024]", primals_157: "f32[1024]", primals_158: "f32[4096, 1024]", primals_159: "f32[4096]", primals_160: "f32[1024, 4096]", primals_161: "f32[1024]", primals_162: "f32[1024]", primals_163: "f32[1024]", primals_164: "f32[1024, 1024]", primals_165: "f32[1024]", primals_166: "f32[1024, 1024]", primals_167: "f32[1024]", primals_168: "f32[1024, 1024]", primals_169: "f32[1024]", primals_170: "f32[1024, 1024]", primals_171: "f32[1024]", primals_172: "f32[1024]", primals_173: "f32[1024]", primals_174: "f32[4096, 1024]", primals_175: "f32[4096]", primals_176: "f32[1024, 4096]", primals_177: "f32[1024]", primals_178: "f32[1024]", primals_179: "f32[1024]", primals_180: "f32[1024, 1024]", primals_181: "f32[1024]", primals_182: "f32[1024, 1024]", primals_183: "f32[1024]", primals_184: "f32[1024, 1024]", primals_185: "f32[1024]", primals_186: "f32[1024, 1024]", primals_187: "f32[1024]", primals_188: "f32[1024]", primals_189: "f32[1024]", primals_190: "f32[4096, 1024]", primals_191: "f32[4096]", primals_192: "f32[1024, 4096]", primals_193: "f32[1024]", primals_194: "f32[1024]", primals_195: "f32[1024]", primals_196: "f32[128112, 1024]", primals_197: "f32[1024]", primals_198: "f32[1024]", primals_199: "f32[1024, 1024]", primals_200: "f32[1024]", primals_201: "f32[1024, 1024]", primals_202: "f32[1024]", primals_203: "f32[1024, 1024]", primals_204: "f32[1024]", primals_205: "f32[1024, 1024]", primals_206: "f32[1024]", primals_207: "f32[1024]", primals_208: "f32[1024]", primals_209: "f32[1024, 1024]", primals_210: "f32[1024]", primals_211: "f32[1024, 1024]", primals_212: "f32[1024]", primals_213: "f32[1024, 1024]", primals_214: "f32[1024]", primals_215: "f32[1024, 1024]", primals_216: "f32[1024]", primals_217: "f32[1024]", primals_218: "f32[1024]", primals_219: "f32[4096, 1024]", primals_220: "f32[4096]", primals_221: "f32[1024, 4096]", primals_222: "f32[1024]", primals_223: "f32[1024]", primals_224: "f32[1024]", primals_225: "f32[1024, 1024]", primals_226: "f32[1024]", primals_227: "f32[1024, 1024]", primals_228: "f32[1024]", primals_229: "f32[1024, 1024]", primals_230: "f32[1024]", primals_231: "f32[1024, 1024]", primals_232: "f32[1024]", primals_233: "f32[1024]", primals_234: "f32[1024]", primals_235: "f32[1024, 1024]", primals_236: "f32[1024]", primals_237: "f32[1024, 1024]", primals_238: "f32[1024]", primals_239: "f32[1024, 1024]", primals_240: "f32[1024]", primals_241: "f32[1024, 1024]", primals_242: "f32[1024]", primals_243: "f32[1024]", primals_244: "f32[1024]", primals_245: "f32[4096, 1024]", primals_246: "f32[4096]", primals_247: "f32[1024, 4096]", primals_248: "f32[1024]", primals_249: "f32[1024]", primals_250: "f32[1024]", primals_251: "f32[1024, 1024]", primals_252: "f32[1024]", primals_253: "f32[1024, 1024]", primals_254: "f32[1024]", primals_255: "f32[1024, 1024]", primals_256: "f32[1024]", primals_257: "f32[1024, 1024]", primals_258: "f32[1024]", primals_259: "f32[1024]", primals_260: "f32[1024]", primals_261: "f32[1024, 1024]", primals_262: "f32[1024]", primals_263: "f32[1024, 1024]", primals_264: "f32[1024]", primals_265: "f32[1024, 1024]", primals_266: "f32[1024]", primals_267: "f32[1024, 1024]", primals_268: "f32[1024]", primals_269: "f32[1024]", primals_270: "f32[1024]", primals_271: "f32[4096, 1024]", primals_272: "f32[4096]", primals_273: "f32[1024, 4096]", primals_274: "f32[1024]", primals_275: "f32[1024]", primals_276: "f32[1024]", primals_277: "f32[1024, 1024]", primals_278: "f32[1024]", primals_279: "f32[1024, 1024]", primals_280: "f32[1024]", primals_281: "f32[1024, 1024]", primals_282: "f32[1024]", primals_283: "f32[1024, 1024]", primals_284: "f32[1024]", primals_285: "f32[1024]", primals_286: "f32[1024]", primals_287: "f32[1024, 1024]", primals_288: "f32[1024]", primals_289: "f32[1024, 1024]", primals_290: "f32[1024]", primals_291: "f32[1024, 1024]", primals_292: "f32[1024]", primals_293: "f32[1024, 1024]", primals_294: "f32[1024]", primals_295: "f32[1024]", primals_296: "f32[1024]", primals_297: "f32[4096, 1024]", primals_298: "f32[4096]", primals_299: "f32[1024, 4096]", primals_300: "f32[1024]", primals_301: "f32[1024]", primals_302: "f32[1024]", primals_303: "f32[1024, 1024]", primals_304: "f32[1024]", primals_305: "f32[1024, 1024]", primals_306: "f32[1024]", primals_307: "f32[1024, 1024]", primals_308: "f32[1024]", primals_309: "f32[1024, 1024]", primals_310: "f32[1024]", primals_311: "f32[1024]", primals_312: "f32[1024]", primals_313: "f32[1024, 1024]", primals_314: "f32[1024]", primals_315: "f32[1024, 1024]", primals_316: "f32[1024]", primals_317: "f32[1024, 1024]", primals_318: "f32[1024]", primals_319: "f32[1024, 1024]", primals_320: "f32[1024]", primals_321: "f32[1024]", primals_322: "f32[1024]", primals_323: "f32[4096, 1024]", primals_324: "f32[4096]", primals_325: "f32[1024, 4096]", primals_326: "f32[1024]", primals_327: "f32[1024]", primals_328: "f32[1024]", primals_329: "f32[1024, 1024]", primals_330: "f32[1024]", primals_331: "f32[1024, 1024]", primals_332: "f32[1024]", primals_333: "f32[1024, 1024]", primals_334: "f32[1024]", primals_335: "f32[1024, 1024]", primals_336: "f32[1024]", primals_337: "f32[1024]", primals_338: "f32[1024]", primals_339: "f32[1024, 1024]", primals_340: "f32[1024]", primals_341: "f32[1024, 1024]", primals_342: "f32[1024]", primals_343: "f32[1024, 1024]", primals_344: "f32[1024]", primals_345: "f32[1024, 1024]", primals_346: "f32[1024]", primals_347: "f32[1024]", primals_348: "f32[1024]", primals_349: "f32[4096, 1024]", primals_350: "f32[4096]", primals_351: "f32[1024, 4096]", primals_352: "f32[1024]", primals_353: "f32[1024]", primals_354: "f32[1024]", primals_355: "f32[1024, 1024]", primals_356: "f32[1024]", primals_357: "f32[1024, 1024]", primals_358: "f32[1024]", primals_359: "f32[1024, 1024]", primals_360: "f32[1024]", primals_361: "f32[1024, 1024]", primals_362: "f32[1024]", primals_363: "f32[1024]", primals_364: "f32[1024]", primals_365: "f32[1024, 1024]", primals_366: "f32[1024]", primals_367: "f32[1024, 1024]", primals_368: "f32[1024]", primals_369: "f32[1024, 1024]", primals_370: "f32[1024]", primals_371: "f32[1024, 1024]", primals_372: "f32[1024]", primals_373: "f32[1024]", primals_374: "f32[1024]", primals_375: "f32[4096, 1024]", primals_376: "f32[4096]", primals_377: "f32[1024, 4096]", primals_378: "f32[1024]", primals_379: "f32[1024]", primals_380: "f32[1024]", primals_381: "f32[1024, 1024]", primals_382: "f32[1024]", primals_383: "f32[1024, 1024]", primals_384: "f32[1024]", primals_385: "f32[1024, 1024]", primals_386: "f32[1024]", primals_387: "f32[1024, 1024]", primals_388: "f32[1024]", primals_389: "f32[1024]", primals_390: "f32[1024]", primals_391: "f32[1024, 1024]", primals_392: "f32[1024]", primals_393: "f32[1024, 1024]", primals_394: "f32[1024]", primals_395: "f32[1024, 1024]", primals_396: "f32[1024]", primals_397: "f32[1024, 1024]", primals_398: "f32[1024]", primals_399: "f32[1024]", primals_400: "f32[1024]", primals_401: "f32[4096, 1024]", primals_402: "f32[4096]", primals_403: "f32[1024, 4096]", primals_404: "f32[1024]", primals_405: "f32[1024]", primals_406: "f32[1024]", primals_407: "f32[1024, 1024]", primals_408: "f32[1024]", primals_409: "f32[1024, 1024]", primals_410: "f32[1024]", primals_411: "f32[1024, 1024]", primals_412: "f32[1024]", primals_413: "f32[1024, 1024]", primals_414: "f32[1024]", primals_415: "f32[1024]", primals_416: "f32[1024]", primals_417: "f32[1024, 1024]", primals_418: "f32[1024]", primals_419: "f32[1024, 1024]", primals_420: "f32[1024]", primals_421: "f32[1024, 1024]", primals_422: "f32[1024]", primals_423: "f32[1024, 1024]", primals_424: "f32[1024]", primals_425: "f32[1024]", primals_426: "f32[1024]", primals_427: "f32[4096, 1024]", primals_428: "f32[4096]", primals_429: "f32[1024, 4096]", primals_430: "f32[1024]", primals_431: "f32[1024]", primals_432: "f32[1024]", primals_433: "f32[1024, 1024]", primals_434: "f32[1024]", primals_435: "f32[1024, 1024]", primals_436: "f32[1024]", primals_437: "f32[1024, 1024]", primals_438: "f32[1024]", primals_439: "f32[1024, 1024]", primals_440: "f32[1024]", primals_441: "f32[1024]", primals_442: "f32[1024]", primals_443: "f32[1024, 1024]", primals_444: "f32[1024]", primals_445: "f32[1024, 1024]", primals_446: "f32[1024]", primals_447: "f32[1024, 1024]", primals_448: "f32[1024]", primals_449: "f32[1024, 1024]", primals_450: "f32[1024]", primals_451: "f32[1024]", primals_452: "f32[1024]", primals_453: "f32[4096, 1024]", primals_454: "f32[4096]", primals_455: "f32[1024, 4096]", primals_456: "f32[1024]", primals_457: "f32[1024]", primals_458: "f32[1024]", primals_459: "f32[1024, 1024]", primals_460: "f32[1024]", primals_461: "f32[1024, 1024]", primals_462: "f32[1024]", primals_463: "f32[1024, 1024]", primals_464: "f32[1024]", primals_465: "f32[1024, 1024]", primals_466: "f32[1024]", primals_467: "f32[1024]", primals_468: "f32[1024]", primals_469: "f32[1024, 1024]", primals_470: "f32[1024]", primals_471: "f32[1024, 1024]", primals_472: "f32[1024]", primals_473: "f32[1024, 1024]", primals_474: "f32[1024]", primals_475: "f32[1024, 1024]", primals_476: "f32[1024]", primals_477: "f32[1024]", primals_478: "f32[1024]", primals_479: "f32[4096, 1024]", primals_480: "f32[4096]", primals_481: "f32[1024, 4096]", primals_482: "f32[1024]", primals_483: "f32[1024]", primals_484: "f32[1024]", primals_485: "f32[1024, 1024]", primals_486: "f32[1024]", primals_487: "f32[1024, 1024]", primals_488: "f32[1024]", primals_489: "f32[1024, 1024]", primals_490: "f32[1024]", primals_491: "f32[1024, 1024]", primals_492: "f32[1024]", primals_493: "f32[1024]", primals_494: "f32[1024]", primals_495: "f32[1024, 1024]", primals_496: "f32[1024]", primals_497: "f32[1024, 1024]", primals_498: "f32[1024]", primals_499: "f32[1024, 1024]", primals_500: "f32[1024]", primals_501: "f32[1024, 1024]", primals_502: "f32[1024]", primals_503: "f32[1024]", primals_504: "f32[1024]", primals_505: "f32[4096, 1024]", primals_506: "f32[4096]", primals_507: "f32[1024, 4096]", primals_508: "f32[1024]", primals_509: "f32[1024]", primals_510: "f32[1024]", primals_511: "f32[128112, 1024]", primals_512: "f32[1026, 1024]", primals_513: "f32[1026, 1024]", primals_514: "i64[1, 128]", primals_515: "i64[1, 128]", primals_516: "i64[1, 128]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:779, code: input_ids = input_ids.view(-1, input_shape[-1])
    view: "i64[1, 128]" = torch.ops.aten.reshape.default(primals_516, [-1, 128]);  primals_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:786, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    embedding: "f32[1, 128, 1024]" = torch.ops.aten.embedding.default(primals_1, view, 1);  primals_1 = None
    mul: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(embedding, 32.0);  embedding = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:113, code: mask = input_ids.ne(padding_idx).int()
    ne: "b8[1, 128]" = torch.ops.aten.ne.Scalar(view, 1)
    convert_element_type: "i32[1, 128]" = torch.ops.prims.convert_element_type.default(ne, torch.int32);  ne = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:114, code: incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    cumsum: "i64[1, 128]" = torch.ops.aten.cumsum.default(convert_element_type, 1)
    convert_element_type_1: "i32[1, 128]" = torch.ops.prims.convert_element_type.default(cumsum, torch.int32);  cumsum = None
    add: "i32[1, 128]" = torch.ops.aten.add.Tensor(convert_element_type_1, 0);  convert_element_type_1 = None
    mul_1: "i32[1, 128]" = torch.ops.aten.mul.Tensor(add, convert_element_type);  add = convert_element_type = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:115, code: return incremental_indices.long() + padding_idx
    convert_element_type_2: "i64[1, 128]" = torch.ops.prims.convert_element_type.default(mul_1, torch.int64);  mul_1 = None
    add_1: "i64[1, 128]" = torch.ops.aten.add.Tensor(convert_element_type_2, 1);  convert_element_type_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:176, code: return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()
    view_1: "i64[128]" = torch.ops.aten.reshape.default(add_1, [-1]);  add_1 = None
    index: "f32[128, 1024]" = torch.ops.aten.index.Tensor(primals_512, [view_1]);  primals_512 = view_1 = None
    view_2: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(index, [1, 128, 1024]);  index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:791, code: hidden_states = inputs_embeds + embed_pos
    add_2: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul, view_2);  mul = view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(add_2, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 128, 1]" = var_mean[0]
    getitem_1: "f32[1, 128, 1]" = var_mean[1];  var_mean = None
    add_3: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
    sub: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_2, getitem_1);  getitem_1 = None
    mul_2: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_3: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_2, primals_2)
    add_4: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_3, primals_3);  mul_3 = primals_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_3: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_4, [128, 1024]);  add_4 = None
    permute: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
    addmm: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_5, view_3, permute);  primals_5 = None
    view_4: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm, [1, 128, 1024]);  addmm = None
    mul_4: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_4, 0.125);  view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
    addmm_1: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_7, view_3, permute_1);  primals_7 = None
    view_6: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_1, [1, 128, 1024]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_7: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_6, [1, -1, 16, 64]);  view_6 = None
    permute_2: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
    clone_1: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_3: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
    addmm_2: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_9, view_3, permute_3);  primals_9 = None
    view_9: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_2, [1, 128, 1024]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_10: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_9, [1, -1, 16, 64]);  view_9 = None
    permute_4: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
    clone_2: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_11: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_4, [1, 128, 16, 64]);  mul_4 = None
    permute_5: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None
    clone_3: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_12: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_3, [16, -1, 64]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_13: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_1, [16, -1, 64]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_14: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_2, [16, -1, 64]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_6: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
    bmm: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_12, permute_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm, [-1], True)
    sub_1: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm, amax)
    exp: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_1: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div, view_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_15: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_1, [1, 16, 128, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_7: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_5: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_16: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_5, [1, 128, 1024]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_17: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_16, [128, 1024]);  view_16 = None
    permute_8: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
    addmm_3: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_11, view_17, permute_8);  primals_11 = None
    view_18: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_3, [1, 128, 1024]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    add_5: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_2, view_18);  add_2 = view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 128, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 128, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt_1: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_2: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_5, getitem_3);  getitem_3 = None
    mul_5: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
    mul_6: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_5, primals_12)
    add_7: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_6, primals_13);  mul_6 = primals_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_19: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_7, [128, 1024]);  add_7 = None
    permute_9: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
    addmm_4: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_15, view_19, permute_9);  primals_15 = None
    view_20: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_4, [1, 128, 4096]);  addmm_4 = None
    relu: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_20);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_21: "f32[128, 4096]" = torch.ops.aten.reshape.default(relu, [128, 4096])
    permute_10: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
    addmm_5: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_17, view_21, permute_10);  primals_17 = None
    view_22: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_5, [1, 128, 1024]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    add_8: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_5, view_22);  add_5 = view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 128, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 128, 1]" = var_mean_2[1];  var_mean_2 = None
    add_9: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_2: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
    sub_3: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_8, getitem_5);  getitem_5 = None
    mul_7: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
    mul_8: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_7, primals_18)
    add_10: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_8, primals_19);  mul_8 = primals_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_23: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_10, [128, 1024]);  add_10 = None
    permute_11: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
    addmm_6: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_21, view_23, permute_11);  primals_21 = None
    view_24: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_6, [1, 128, 1024]);  addmm_6 = None
    mul_9: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_24, 0.125);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_12: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
    addmm_7: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_23, view_23, permute_12);  primals_23 = None
    view_26: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_7, [1, 128, 1024]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_27: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_26, [1, -1, 16, 64]);  view_26 = None
    permute_13: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None
    clone_9: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_14: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
    addmm_8: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_25, view_23, permute_14);  primals_25 = None
    view_29: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_8, [1, 128, 1024]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_30: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_29, [1, -1, 16, 64]);  view_29 = None
    permute_15: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    clone_10: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_31: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_9, [1, 128, 16, 64]);  mul_9 = None
    permute_16: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
    clone_11: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_32: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_11, [16, -1, 64]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_33: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_9, [16, -1, 64]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_34: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_10, [16, -1, 64]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_17: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None
    bmm_2: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_32, permute_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_1: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_2, [-1], True)
    sub_4: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_2, amax_1)
    exp_1: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_2: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_3: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_1, view_34)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_35: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_3, [1, 16, 128, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_18: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_35, [0, 2, 1, 3]);  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_13: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    view_36: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_13, [1, 128, 1024]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_37: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_36, [128, 1024]);  view_36 = None
    permute_19: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
    addmm_9: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_27, view_37, permute_19);  primals_27 = None
    view_38: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_9, [1, 128, 1024]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    add_11: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_8, view_38);  add_8 = view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
    getitem_6: "f32[1, 128, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 128, 1]" = var_mean_3[1];  var_mean_3 = None
    add_12: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
    rsqrt_3: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    sub_5: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_11, getitem_7);  getitem_7 = None
    mul_10: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
    mul_11: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_10, primals_28)
    add_13: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_11, primals_29);  mul_11 = primals_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_39: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_13, [128, 1024]);  add_13 = None
    permute_20: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_30, [1, 0]);  primals_30 = None
    addmm_10: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_31, view_39, permute_20);  primals_31 = None
    view_40: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_10, [1, 128, 4096]);  addmm_10 = None
    relu_1: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_40);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_41: "f32[128, 4096]" = torch.ops.aten.reshape.default(relu_1, [128, 4096])
    permute_21: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
    addmm_11: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_33, view_41, permute_21);  primals_33 = None
    view_42: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_11, [1, 128, 1024]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    add_14: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_11, view_42);  add_11 = view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 128, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 128, 1]" = var_mean_4[1];  var_mean_4 = None
    add_15: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_4: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
    sub_6: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_14, getitem_9);  getitem_9 = None
    mul_12: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
    mul_13: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_12, primals_34)
    add_16: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_13, primals_35);  mul_13 = primals_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_43: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_16, [128, 1024]);  add_16 = None
    permute_22: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
    addmm_12: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_37, view_43, permute_22);  primals_37 = None
    view_44: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_12, [1, 128, 1024]);  addmm_12 = None
    mul_14: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_44, 0.125);  view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_23: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
    addmm_13: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_39, view_43, permute_23);  primals_39 = None
    view_46: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_13, [1, 128, 1024]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_47: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_46, [1, -1, 16, 64]);  view_46 = None
    permute_24: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_47, [0, 2, 1, 3]);  view_47 = None
    clone_17: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_25: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
    addmm_14: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_41, view_43, permute_25);  primals_41 = None
    view_49: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_14, [1, 128, 1024]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_50: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_49, [1, -1, 16, 64]);  view_49 = None
    permute_26: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
    clone_18: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_51: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_14, [1, 128, 16, 64]);  mul_14 = None
    permute_27: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
    clone_19: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_52: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_19, [16, -1, 64]);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_53: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_17, [16, -1, 64]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_54: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_18, [16, -1, 64]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_28: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
    bmm_4: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_52, permute_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_2: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_4, [-1], True)
    sub_7: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_4, amax_2)
    exp_2: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_3: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_5: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_2, view_54)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_55: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_5, [1, 16, 128, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_29: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_55, [0, 2, 1, 3]);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_21: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_56: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_21, [1, 128, 1024]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_57: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_56, [128, 1024]);  view_56 = None
    permute_30: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
    addmm_15: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_43, view_57, permute_30);  primals_43 = None
    view_58: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_15, [1, 128, 1024]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    add_17: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_14, view_58);  add_14 = view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
    getitem_10: "f32[1, 128, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 128, 1]" = var_mean_5[1];  var_mean_5 = None
    add_18: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_5: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_8: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_17, getitem_11);  getitem_11 = None
    mul_15: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = None
    mul_16: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_15, primals_44)
    add_19: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_16, primals_45);  mul_16 = primals_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_59: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_19, [128, 1024]);  add_19 = None
    permute_31: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_46, [1, 0]);  primals_46 = None
    addmm_16: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_47, view_59, permute_31);  primals_47 = None
    view_60: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_16, [1, 128, 4096]);  addmm_16 = None
    relu_2: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_60);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_61: "f32[128, 4096]" = torch.ops.aten.reshape.default(relu_2, [128, 4096])
    permute_32: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
    addmm_17: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_49, view_61, permute_32);  primals_49 = None
    view_62: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_17, [1, 128, 1024]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    add_20: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_17, view_62);  add_17 = view_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_20, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 128, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 128, 1]" = var_mean_6[1];  var_mean_6 = None
    add_21: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_6: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_9: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_20, getitem_13);  getitem_13 = None
    mul_17: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = None
    mul_18: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_17, primals_50)
    add_22: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_18, primals_51);  mul_18 = primals_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_63: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_22, [128, 1024]);  add_22 = None
    permute_33: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_52, [1, 0]);  primals_52 = None
    addmm_18: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_53, view_63, permute_33);  primals_53 = None
    view_64: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_18, [1, 128, 1024]);  addmm_18 = None
    mul_19: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_64, 0.125);  view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_34: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
    addmm_19: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_55, view_63, permute_34);  primals_55 = None
    view_66: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_19, [1, 128, 1024]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_67: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_66, [1, -1, 16, 64]);  view_66 = None
    permute_35: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_67, [0, 2, 1, 3]);  view_67 = None
    clone_25: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_36: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
    addmm_20: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_57, view_63, permute_36);  primals_57 = None
    view_69: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_20, [1, 128, 1024]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_70: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_69, [1, -1, 16, 64]);  view_69 = None
    permute_37: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
    clone_26: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_71: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_19, [1, 128, 16, 64]);  mul_19 = None
    permute_38: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
    clone_27: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_72: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_27, [16, -1, 64]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_73: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_25, [16, -1, 64]);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_74: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_26, [16, -1, 64]);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_39: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_73, [0, 2, 1]);  view_73 = None
    bmm_6: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_72, permute_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_3: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_6, [-1], True)
    sub_10: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_6, amax_3)
    exp_3: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_4: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_3: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_7: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_3, view_74)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_75: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_7, [1, 16, 128, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_40: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_29: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    view_76: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_29, [1, 128, 1024]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_77: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_76, [128, 1024]);  view_76 = None
    permute_41: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
    addmm_21: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_59, view_77, permute_41);  primals_59 = None
    view_78: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_21, [1, 128, 1024]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    add_23: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_20, view_78);  add_20 = view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_23, [2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 128, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 128, 1]" = var_mean_7[1];  var_mean_7 = None
    add_24: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_7: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    sub_11: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_23, getitem_15);  getitem_15 = None
    mul_20: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = None
    mul_21: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_20, primals_60)
    add_25: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_21, primals_61);  mul_21 = primals_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_79: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_25, [128, 1024]);  add_25 = None
    permute_42: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
    addmm_22: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_63, view_79, permute_42);  primals_63 = None
    view_80: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_22, [1, 128, 4096]);  addmm_22 = None
    relu_3: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_80);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_81: "f32[128, 4096]" = torch.ops.aten.reshape.default(relu_3, [128, 4096])
    permute_43: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
    addmm_23: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_65, view_81, permute_43);  primals_65 = None
    view_82: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_23, [1, 128, 1024]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    add_26: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_23, view_82);  add_23 = view_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_26, [2], correction = 0, keepdim = True)
    getitem_16: "f32[1, 128, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 128, 1]" = var_mean_8[1];  var_mean_8 = None
    add_27: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_8: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    sub_12: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_26, getitem_17);  getitem_17 = None
    mul_22: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = None
    mul_23: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_22, primals_66)
    add_28: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_23, primals_67);  mul_23 = primals_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_83: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_28, [128, 1024]);  add_28 = None
    permute_44: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_68, [1, 0]);  primals_68 = None
    addmm_24: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_69, view_83, permute_44);  primals_69 = None
    view_84: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_24, [1, 128, 1024]);  addmm_24 = None
    mul_24: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_84, 0.125);  view_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_45: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
    addmm_25: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_71, view_83, permute_45);  primals_71 = None
    view_86: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_25, [1, 128, 1024]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_87: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_86, [1, -1, 16, 64]);  view_86 = None
    permute_46: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None
    clone_33: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_47: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
    addmm_26: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_73, view_83, permute_47);  primals_73 = None
    view_89: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_26, [1, 128, 1024]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_90: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_89, [1, -1, 16, 64]);  view_89 = None
    permute_48: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
    clone_34: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_91: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_24, [1, 128, 16, 64]);  mul_24 = None
    permute_49: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_91, [0, 2, 1, 3]);  view_91 = None
    clone_35: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_92: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_35, [16, -1, 64]);  clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_93: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_33, [16, -1, 64]);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_94: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_34, [16, -1, 64]);  clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_50: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_93, [0, 2, 1]);  view_93 = None
    bmm_8: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_92, permute_50)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_4: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_8, [-1], True)
    sub_13: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_8, amax_4)
    exp_4: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_5: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_4: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_9: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_4, view_94)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_95: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_9, [1, 16, 128, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_51: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_37: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    view_96: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_37, [1, 128, 1024]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_97: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_96, [128, 1024]);  view_96 = None
    permute_52: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
    addmm_27: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_75, view_97, permute_52);  primals_75 = None
    view_98: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_27, [1, 128, 1024]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    add_29: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_26, view_98);  add_26 = view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 128, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 128, 1]" = var_mean_9[1];  var_mean_9 = None
    add_30: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
    rsqrt_9: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
    sub_14: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_29, getitem_19);  getitem_19 = None
    mul_25: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = None
    mul_26: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_25, primals_76)
    add_31: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_26, primals_77);  mul_26 = primals_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_99: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_31, [128, 1024]);  add_31 = None
    permute_53: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
    addmm_28: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_79, view_99, permute_53);  primals_79 = None
    view_100: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_28, [1, 128, 4096]);  addmm_28 = None
    relu_4: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_100);  view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_101: "f32[128, 4096]" = torch.ops.aten.reshape.default(relu_4, [128, 4096])
    permute_54: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
    addmm_29: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_81, view_101, permute_54);  primals_81 = None
    view_102: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_29, [1, 128, 1024]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    add_32: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_29, view_102);  add_29 = view_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
    getitem_20: "f32[1, 128, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 128, 1]" = var_mean_10[1];  var_mean_10 = None
    add_33: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
    rsqrt_10: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
    sub_15: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_32, getitem_21);  getitem_21 = None
    mul_27: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = None
    mul_28: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_27, primals_82)
    add_34: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_28, primals_83);  mul_28 = primals_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_103: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_34, [128, 1024]);  add_34 = None
    permute_55: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
    addmm_30: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_85, view_103, permute_55);  primals_85 = None
    view_104: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_30, [1, 128, 1024]);  addmm_30 = None
    mul_29: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_104, 0.125);  view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_56: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
    addmm_31: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_87, view_103, permute_56);  primals_87 = None
    view_106: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_31, [1, 128, 1024]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_107: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_106, [1, -1, 16, 64]);  view_106 = None
    permute_57: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_107, [0, 2, 1, 3]);  view_107 = None
    clone_41: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_58: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
    addmm_32: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_89, view_103, permute_58);  primals_89 = None
    view_109: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_32, [1, 128, 1024]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_110: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_109, [1, -1, 16, 64]);  view_109 = None
    permute_59: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_110, [0, 2, 1, 3]);  view_110 = None
    clone_42: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_111: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_29, [1, 128, 16, 64]);  mul_29 = None
    permute_60: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_111, [0, 2, 1, 3]);  view_111 = None
    clone_43: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_112: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_43, [16, -1, 64]);  clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_113: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_41, [16, -1, 64]);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_114: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_42, [16, -1, 64]);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_61: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_113, [0, 2, 1]);  view_113 = None
    bmm_10: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_112, permute_61)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_5: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_10, [-1], True)
    sub_16: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_10, amax_5)
    exp_5: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_6: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_5: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_11: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_5, view_114)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_115: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_11, [1, 16, 128, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_62: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_115, [0, 2, 1, 3]);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_45: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_116: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_45, [1, 128, 1024]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_117: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_116, [128, 1024]);  view_116 = None
    permute_63: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
    addmm_33: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_91, view_117, permute_63);  primals_91 = None
    view_118: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_33, [1, 128, 1024]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    add_35: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_32, view_118);  add_32 = view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 128, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 128, 1]" = var_mean_11[1];  var_mean_11 = None
    add_36: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_11: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_17: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_35, getitem_23);  getitem_23 = None
    mul_30: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = None
    mul_31: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_30, primals_92)
    add_37: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_31, primals_93);  mul_31 = primals_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_119: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_37, [128, 1024]);  add_37 = None
    permute_64: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_94, [1, 0]);  primals_94 = None
    addmm_34: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_95, view_119, permute_64);  primals_95 = None
    view_120: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_34, [1, 128, 4096]);  addmm_34 = None
    relu_5: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_120);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_121: "f32[128, 4096]" = torch.ops.aten.reshape.default(relu_5, [128, 4096])
    permute_65: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
    addmm_35: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_97, view_121, permute_65);  primals_97 = None
    view_122: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_35, [1, 128, 1024]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    add_38: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_35, view_122);  add_35 = view_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 128, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 128, 1]" = var_mean_12[1];  var_mean_12 = None
    add_39: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_12: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
    sub_18: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_38, getitem_25);  getitem_25 = None
    mul_32: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = None
    mul_33: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_32, primals_98)
    add_40: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_33, primals_99);  mul_33 = primals_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_123: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_40, [128, 1024]);  add_40 = None
    permute_66: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_100, [1, 0]);  primals_100 = None
    addmm_36: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_101, view_123, permute_66);  primals_101 = None
    view_124: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_36, [1, 128, 1024]);  addmm_36 = None
    mul_34: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_124, 0.125);  view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_67: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_102, [1, 0]);  primals_102 = None
    addmm_37: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_103, view_123, permute_67);  primals_103 = None
    view_126: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_37, [1, 128, 1024]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_127: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_126, [1, -1, 16, 64]);  view_126 = None
    permute_68: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_127, [0, 2, 1, 3]);  view_127 = None
    clone_49: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_69: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_104, [1, 0]);  primals_104 = None
    addmm_38: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_105, view_123, permute_69);  primals_105 = None
    view_129: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_38, [1, 128, 1024]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_130: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_129, [1, -1, 16, 64]);  view_129 = None
    permute_70: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
    clone_50: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_131: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_34, [1, 128, 16, 64]);  mul_34 = None
    permute_71: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_131, [0, 2, 1, 3]);  view_131 = None
    clone_51: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_132: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_51, [16, -1, 64]);  clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_133: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_49, [16, -1, 64]);  clone_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_134: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_50, [16, -1, 64]);  clone_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_72: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_133, [0, 2, 1]);  view_133 = None
    bmm_12: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_132, permute_72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_6: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_12, [-1], True)
    sub_19: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_12, amax_6)
    exp_6: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_7: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_6: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_13: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_6, view_134)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_135: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_13, [1, 16, 128, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_73: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_135, [0, 2, 1, 3]);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_53: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    view_136: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_53, [1, 128, 1024]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_137: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_136, [128, 1024]);  view_136 = None
    permute_74: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
    addmm_39: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_107, view_137, permute_74);  primals_107 = None
    view_138: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_39, [1, 128, 1024]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    add_41: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_38, view_138);  add_38 = view_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
    getitem_26: "f32[1, 128, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 128, 1]" = var_mean_13[1];  var_mean_13 = None
    add_42: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
    rsqrt_13: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_20: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_41, getitem_27);  getitem_27 = None
    mul_35: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = None
    mul_36: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_35, primals_108)
    add_43: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_36, primals_109);  mul_36 = primals_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_139: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_43, [128, 1024]);  add_43 = None
    permute_75: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_110, [1, 0]);  primals_110 = None
    addmm_40: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_111, view_139, permute_75);  primals_111 = None
    view_140: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_40, [1, 128, 4096]);  addmm_40 = None
    relu_6: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_140);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_141: "f32[128, 4096]" = torch.ops.aten.reshape.default(relu_6, [128, 4096])
    permute_76: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_112, [1, 0]);  primals_112 = None
    addmm_41: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_113, view_141, permute_76);  primals_113 = None
    view_142: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_41, [1, 128, 1024]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    add_44: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_41, view_142);  add_41 = view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_44, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 128, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 128, 1]" = var_mean_14[1];  var_mean_14 = None
    add_45: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_14: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
    sub_21: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_44, getitem_29);  getitem_29 = None
    mul_37: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = None
    mul_38: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_37, primals_114)
    add_46: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_38, primals_115);  mul_38 = primals_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_143: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_46, [128, 1024]);  add_46 = None
    permute_77: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_116, [1, 0]);  primals_116 = None
    addmm_42: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_117, view_143, permute_77);  primals_117 = None
    view_144: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_42, [1, 128, 1024]);  addmm_42 = None
    mul_39: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_144, 0.125);  view_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_78: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
    addmm_43: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_119, view_143, permute_78);  primals_119 = None
    view_146: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_43, [1, 128, 1024]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_147: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_146, [1, -1, 16, 64]);  view_146 = None
    permute_79: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_147, [0, 2, 1, 3]);  view_147 = None
    clone_57: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_80: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
    addmm_44: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_121, view_143, permute_80);  primals_121 = None
    view_149: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_44, [1, 128, 1024]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_150: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_149, [1, -1, 16, 64]);  view_149 = None
    permute_81: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_150, [0, 2, 1, 3]);  view_150 = None
    clone_58: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_151: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_39, [1, 128, 16, 64]);  mul_39 = None
    permute_82: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_151, [0, 2, 1, 3]);  view_151 = None
    clone_59: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_152: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_59, [16, -1, 64]);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_153: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_57, [16, -1, 64]);  clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_154: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_58, [16, -1, 64]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_83: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_153, [0, 2, 1]);  view_153 = None
    bmm_14: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_152, permute_83)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_7: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_14, [-1], True)
    sub_22: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_14, amax_7)
    exp_7: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_8: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_7: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_15: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_7, view_154)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_155: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_15, [1, 16, 128, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_84: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_155, [0, 2, 1, 3]);  view_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_61: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    view_156: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_61, [1, 128, 1024]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_157: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_156, [128, 1024]);  view_156 = None
    permute_85: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
    addmm_45: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_123, view_157, permute_85);  primals_123 = None
    view_158: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_45, [1, 128, 1024]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    add_47: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_44, view_158);  add_44 = view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_47, [2], correction = 0, keepdim = True)
    getitem_30: "f32[1, 128, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 128, 1]" = var_mean_15[1];  var_mean_15 = None
    add_48: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
    rsqrt_15: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
    sub_23: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_47, getitem_31);  getitem_31 = None
    mul_40: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = None
    mul_41: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_40, primals_124)
    add_49: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_41, primals_125);  mul_41 = primals_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_159: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_49, [128, 1024]);  add_49 = None
    permute_86: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
    addmm_46: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_127, view_159, permute_86);  primals_127 = None
    view_160: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_46, [1, 128, 4096]);  addmm_46 = None
    relu_7: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_160);  view_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_161: "f32[128, 4096]" = torch.ops.aten.reshape.default(relu_7, [128, 4096])
    permute_87: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
    addmm_47: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_129, view_161, permute_87);  primals_129 = None
    view_162: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_47, [1, 128, 1024]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    add_50: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_47, view_162);  add_47 = view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_50, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 128, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 128, 1]" = var_mean_16[1];  var_mean_16 = None
    add_51: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
    rsqrt_16: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
    sub_24: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_50, getitem_33);  getitem_33 = None
    mul_42: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = None
    mul_43: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_42, primals_130)
    add_52: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_43, primals_131);  mul_43 = primals_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_163: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_52, [128, 1024]);  add_52 = None
    permute_88: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_132, [1, 0]);  primals_132 = None
    addmm_48: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_133, view_163, permute_88);  primals_133 = None
    view_164: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_48, [1, 128, 1024]);  addmm_48 = None
    mul_44: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_164, 0.125);  view_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_89: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_134, [1, 0]);  primals_134 = None
    addmm_49: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_135, view_163, permute_89);  primals_135 = None
    view_166: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_49, [1, 128, 1024]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_167: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_166, [1, -1, 16, 64]);  view_166 = None
    permute_90: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_167, [0, 2, 1, 3]);  view_167 = None
    clone_65: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_91: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
    addmm_50: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_137, view_163, permute_91);  primals_137 = None
    view_169: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_50, [1, 128, 1024]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_170: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_169, [1, -1, 16, 64]);  view_169 = None
    permute_92: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
    clone_66: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_171: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_44, [1, 128, 16, 64]);  mul_44 = None
    permute_93: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_171, [0, 2, 1, 3]);  view_171 = None
    clone_67: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_172: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_67, [16, -1, 64]);  clone_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_173: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_65, [16, -1, 64]);  clone_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_174: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_66, [16, -1, 64]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_94: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_173, [0, 2, 1]);  view_173 = None
    bmm_16: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_172, permute_94)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_8: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_16, [-1], True)
    sub_25: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_16, amax_8)
    exp_8: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
    sum_9: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_8: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_17: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_8, view_174)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_175: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_17, [1, 16, 128, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_95: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_175, [0, 2, 1, 3]);  view_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_69: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    view_176: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_69, [1, 128, 1024]);  clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_177: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_176, [128, 1024]);  view_176 = None
    permute_96: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
    addmm_51: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_139, view_177, permute_96);  primals_139 = None
    view_178: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_51, [1, 128, 1024]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    add_53: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_50, view_178);  add_50 = view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
    getitem_34: "f32[1, 128, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 128, 1]" = var_mean_17[1];  var_mean_17 = None
    add_54: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
    rsqrt_17: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
    sub_26: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_53, getitem_35);  getitem_35 = None
    mul_45: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = None
    mul_46: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_45, primals_140)
    add_55: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_46, primals_141);  mul_46 = primals_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_179: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_55, [128, 1024]);  add_55 = None
    permute_97: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_142, [1, 0]);  primals_142 = None
    addmm_52: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_143, view_179, permute_97);  primals_143 = None
    view_180: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_52, [1, 128, 4096]);  addmm_52 = None
    relu_8: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_180);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_181: "f32[128, 4096]" = torch.ops.aten.reshape.default(relu_8, [128, 4096])
    permute_98: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
    addmm_53: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_145, view_181, permute_98);  primals_145 = None
    view_182: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_53, [1, 128, 1024]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    add_56: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_53, view_182);  add_53 = view_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
    getitem_36: "f32[1, 128, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 128, 1]" = var_mean_18[1];  var_mean_18 = None
    add_57: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
    rsqrt_18: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    sub_27: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_56, getitem_37);  getitem_37 = None
    mul_47: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = None
    mul_48: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_47, primals_146)
    add_58: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_48, primals_147);  mul_48 = primals_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_183: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_58, [128, 1024]);  add_58 = None
    permute_99: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_148, [1, 0]);  primals_148 = None
    addmm_54: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_149, view_183, permute_99);  primals_149 = None
    view_184: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_54, [1, 128, 1024]);  addmm_54 = None
    mul_49: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_184, 0.125);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_100: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
    addmm_55: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_151, view_183, permute_100);  primals_151 = None
    view_186: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_55, [1, 128, 1024]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_187: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_186, [1, -1, 16, 64]);  view_186 = None
    permute_101: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_187, [0, 2, 1, 3]);  view_187 = None
    clone_73: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_101, memory_format = torch.contiguous_format);  permute_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_102: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
    addmm_56: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_153, view_183, permute_102);  primals_153 = None
    view_189: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_56, [1, 128, 1024]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_190: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_189, [1, -1, 16, 64]);  view_189 = None
    permute_103: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
    clone_74: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_191: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_49, [1, 128, 16, 64]);  mul_49 = None
    permute_104: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_191, [0, 2, 1, 3]);  view_191 = None
    clone_75: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_192: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_75, [16, -1, 64]);  clone_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_193: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_73, [16, -1, 64]);  clone_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_194: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_74, [16, -1, 64]);  clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_105: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_193, [0, 2, 1]);  view_193 = None
    bmm_18: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_192, permute_105)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_9: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_18, [-1], True)
    sub_28: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_18, amax_9)
    exp_9: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
    sum_10: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_9: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_19: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_9, view_194)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_195: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_19, [1, 16, 128, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_106: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_195, [0, 2, 1, 3]);  view_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_77: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    view_196: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_77, [1, 128, 1024]);  clone_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_197: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_196, [128, 1024]);  view_196 = None
    permute_107: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
    addmm_57: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_155, view_197, permute_107);  primals_155 = None
    view_198: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_57, [1, 128, 1024]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    add_59: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_56, view_198);  add_56 = view_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 128, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 128, 1]" = var_mean_19[1];  var_mean_19 = None
    add_60: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
    rsqrt_19: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_29: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_59, getitem_39);  getitem_39 = None
    mul_50: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = None
    mul_51: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_50, primals_156)
    add_61: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_51, primals_157);  mul_51 = primals_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_199: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_61, [128, 1024]);  add_61 = None
    permute_108: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_158, [1, 0]);  primals_158 = None
    addmm_58: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_159, view_199, permute_108);  primals_159 = None
    view_200: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_58, [1, 128, 4096]);  addmm_58 = None
    relu_9: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_200);  view_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_201: "f32[128, 4096]" = torch.ops.aten.reshape.default(relu_9, [128, 4096])
    permute_109: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
    addmm_59: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_161, view_201, permute_109);  primals_161 = None
    view_202: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_59, [1, 128, 1024]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    add_62: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_59, view_202);  add_59 = view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_62, [2], correction = 0, keepdim = True)
    getitem_40: "f32[1, 128, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 128, 1]" = var_mean_20[1];  var_mean_20 = None
    add_63: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_20: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_30: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_62, getitem_41);  getitem_41 = None
    mul_52: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = None
    mul_53: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_52, primals_162)
    add_64: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_53, primals_163);  mul_53 = primals_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_203: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_64, [128, 1024]);  add_64 = None
    permute_110: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_164, [1, 0]);  primals_164 = None
    addmm_60: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_165, view_203, permute_110);  primals_165 = None
    view_204: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_60, [1, 128, 1024]);  addmm_60 = None
    mul_54: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_204, 0.125);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_111: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_166, [1, 0]);  primals_166 = None
    addmm_61: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_167, view_203, permute_111);  primals_167 = None
    view_206: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_61, [1, 128, 1024]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_207: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_206, [1, -1, 16, 64]);  view_206 = None
    permute_112: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None
    clone_81: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_113: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
    addmm_62: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_169, view_203, permute_113);  primals_169 = None
    view_209: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_62, [1, 128, 1024]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_210: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_209, [1, -1, 16, 64]);  view_209 = None
    permute_114: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_210, [0, 2, 1, 3]);  view_210 = None
    clone_82: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_211: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_54, [1, 128, 16, 64]);  mul_54 = None
    permute_115: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_211, [0, 2, 1, 3]);  view_211 = None
    clone_83: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_212: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_83, [16, -1, 64]);  clone_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_213: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_81, [16, -1, 64]);  clone_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_214: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_82, [16, -1, 64]);  clone_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_116: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_213, [0, 2, 1]);  view_213 = None
    bmm_20: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_212, permute_116)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_10: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_20, [-1], True)
    sub_31: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_20, amax_10)
    exp_10: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_11: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_10: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_21: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_10, view_214)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_215: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_21, [1, 16, 128, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_117: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_215, [0, 2, 1, 3]);  view_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_85: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    view_216: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_85, [1, 128, 1024]);  clone_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_217: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_216, [128, 1024]);  view_216 = None
    permute_118: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
    addmm_63: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_171, view_217, permute_118);  primals_171 = None
    view_218: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_63, [1, 128, 1024]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    add_65: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_62, view_218);  add_62 = view_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 128, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 128, 1]" = var_mean_21[1];  var_mean_21 = None
    add_66: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
    rsqrt_21: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_32: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_65, getitem_43);  getitem_43 = None
    mul_55: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = None
    mul_56: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_55, primals_172)
    add_67: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_56, primals_173);  mul_56 = primals_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_219: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_67, [128, 1024]);  add_67 = None
    permute_119: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_174, [1, 0]);  primals_174 = None
    addmm_64: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_175, view_219, permute_119);  primals_175 = None
    view_220: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_64, [1, 128, 4096]);  addmm_64 = None
    relu_10: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_220);  view_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_221: "f32[128, 4096]" = torch.ops.aten.reshape.default(relu_10, [128, 4096])
    permute_120: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
    addmm_65: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_177, view_221, permute_120);  primals_177 = None
    view_222: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_65, [1, 128, 1024]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    add_68: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_65, view_222);  add_65 = view_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_68, [2], correction = 0, keepdim = True)
    getitem_44: "f32[1, 128, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 128, 1]" = var_mean_22[1];  var_mean_22 = None
    add_69: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
    rsqrt_22: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
    sub_33: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_68, getitem_45);  getitem_45 = None
    mul_57: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = None
    mul_58: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_57, primals_178)
    add_70: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_58, primals_179);  mul_58 = primals_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_223: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_70, [128, 1024]);  add_70 = None
    permute_121: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_180, [1, 0]);  primals_180 = None
    addmm_66: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_181, view_223, permute_121);  primals_181 = None
    view_224: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_66, [1, 128, 1024]);  addmm_66 = None
    mul_59: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_224, 0.125);  view_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_122: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_182, [1, 0]);  primals_182 = None
    addmm_67: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_183, view_223, permute_122);  primals_183 = None
    view_226: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_67, [1, 128, 1024]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_227: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_226, [1, -1, 16, 64]);  view_226 = None
    permute_123: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None
    clone_89: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_123, memory_format = torch.contiguous_format);  permute_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_124: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
    addmm_68: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_185, view_223, permute_124);  primals_185 = None
    view_229: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_68, [1, 128, 1024]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_230: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_229, [1, -1, 16, 64]);  view_229 = None
    permute_125: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None
    clone_90: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_231: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_59, [1, 128, 16, 64]);  mul_59 = None
    permute_126: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_231, [0, 2, 1, 3]);  view_231 = None
    clone_91: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_232: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_91, [16, -1, 64]);  clone_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_233: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_89, [16, -1, 64]);  clone_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_234: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_90, [16, -1, 64]);  clone_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_127: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_233, [0, 2, 1]);  view_233 = None
    bmm_22: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_232, permute_127)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_11: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_22, [-1], True)
    sub_34: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_22, amax_11)
    exp_11: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
    sum_12: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_11: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_23: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_11, view_234)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_235: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_23, [1, 16, 128, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_128: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_235, [0, 2, 1, 3]);  view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_93: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    view_236: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_93, [1, 128, 1024]);  clone_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_237: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_236, [128, 1024]);  view_236 = None
    permute_129: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_186, [1, 0]);  primals_186 = None
    addmm_69: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_187, view_237, permute_129);  primals_187 = None
    view_238: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_69, [1, 128, 1024]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:396, code: hidden_states = residual + hidden_states
    add_71: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_68, view_238);  add_68 = view_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
    getitem_46: "f32[1, 128, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 128, 1]" = var_mean_23[1];  var_mean_23 = None
    add_72: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
    rsqrt_23: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
    sub_35: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_71, getitem_47);  getitem_47 = None
    mul_60: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = None
    mul_61: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_60, primals_188)
    add_73: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_61, primals_189);  mul_61 = primals_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_239: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_73, [128, 1024]);  add_73 = None
    permute_130: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_190, [1, 0]);  primals_190 = None
    addmm_70: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_191, view_239, permute_130);  primals_191 = None
    view_240: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_70, [1, 128, 4096]);  addmm_70 = None
    relu_11: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_240);  view_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_241: "f32[128, 4096]" = torch.ops.aten.reshape.default(relu_11, [128, 4096])
    permute_131: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_192, [1, 0]);  primals_192 = None
    addmm_71: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_193, view_241, permute_131);  primals_193 = None
    view_242: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_71, [1, 128, 1024]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:404, code: hidden_states = residual + hidden_states
    add_74: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_71, view_242);  add_71 = view_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:852, code: hidden_states = self.layer_norm(hidden_states)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_74, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 128, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 128, 1]" = var_mean_24[1];  var_mean_24 = None
    add_75: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
    rsqrt_24: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
    sub_36: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_74, getitem_49);  add_74 = getitem_49 = None
    mul_62: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = None
    mul_63: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_62, primals_194)
    add_76: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_63, primals_195);  mul_63 = primals_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:990, code: input_ids = input_ids.view(-1, input_shape[-1])
    view_243: "i64[1, 128]" = torch.ops.aten.reshape.default(primals_515, [-1, 128]);  primals_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1000, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    embedding_1: "f32[1, 128, 1024]" = torch.ops.aten.embedding.default(primals_196, view_243, 1);  primals_196 = None
    mul_64: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(embedding_1, 32.0);  embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:82, code: mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    full_default: "f32[128, 128]" = torch.ops.aten.full.default([128, 128], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:83, code: mask_cond = torch.arange(mask.size(-1), device=device)
    iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:84, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    add_77: "i64[128]" = torch.ops.aten.add.Tensor(iota, 1)
    view_244: "i64[128, 1]" = torch.ops.aten.reshape.default(add_77, [128, 1]);  add_77 = None
    lt: "b8[128, 128]" = torch.ops.aten.lt.Tensor(iota, view_244);  iota = view_244 = None
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where: "f32[128, 128]" = torch.ops.aten.where.self(lt, full_default_1, full_default);  lt = full_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:113, code: mask = input_ids.ne(padding_idx).int()
    ne_1: "b8[1, 128]" = torch.ops.aten.ne.Scalar(view_243, 1)
    convert_element_type_3: "i32[1, 128]" = torch.ops.prims.convert_element_type.default(ne_1, torch.int32);  ne_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:114, code: incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    cumsum_1: "i64[1, 128]" = torch.ops.aten.cumsum.default(convert_element_type_3, 1)
    convert_element_type_4: "i32[1, 128]" = torch.ops.prims.convert_element_type.default(cumsum_1, torch.int32);  cumsum_1 = None
    add_78: "i32[1, 128]" = torch.ops.aten.add.Tensor(convert_element_type_4, 0);  convert_element_type_4 = None
    mul_65: "i32[1, 128]" = torch.ops.aten.mul.Tensor(add_78, convert_element_type_3);  add_78 = convert_element_type_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:115, code: return incremental_indices.long() + padding_idx
    convert_element_type_5: "i64[1, 128]" = torch.ops.prims.convert_element_type.default(mul_65, torch.int64);  mul_65 = None
    add_79: "i64[1, 128]" = torch.ops.aten.add.Tensor(convert_element_type_5, 1);  convert_element_type_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:176, code: return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()
    view_245: "i64[128]" = torch.ops.aten.reshape.default(add_79, [-1]);  add_79 = None
    index_1: "f32[128, 1024]" = torch.ops.aten.index.Tensor(primals_513, [view_245]);  primals_513 = view_245 = None
    view_246: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(index_1, [1, 128, 1024]);  index_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1028, code: hidden_states = inputs_embeds + positions
    add_80: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_64, view_246);  mul_64 = view_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_25 = torch.ops.aten.var_mean.correction(add_80, [2], correction = 0, keepdim = True)
    getitem_50: "f32[1, 128, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 128, 1]" = var_mean_25[1];  var_mean_25 = None
    add_81: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
    rsqrt_25: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    sub_37: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_80, getitem_51);  getitem_51 = None
    mul_66: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_25);  sub_37 = None
    mul_67: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_66, primals_197)
    add_82: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_67, primals_198);  mul_67 = primals_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_247: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_82, [128, 1024]);  add_82 = None
    permute_132: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_199, [1, 0]);  primals_199 = None
    addmm_72: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_200, view_247, permute_132);  primals_200 = None
    view_248: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_72, [1, 128, 1024]);  addmm_72 = None
    mul_68: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_248, 0.125);  view_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_133: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_201, [1, 0]);  primals_201 = None
    addmm_73: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_202, view_247, permute_133);  primals_202 = None
    view_250: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_73, [1, 128, 1024]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_251: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_250, [1, -1, 16, 64]);  view_250 = None
    permute_134: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None
    clone_98: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_134, memory_format = torch.contiguous_format);  permute_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_135: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_203, [1, 0]);  primals_203 = None
    addmm_74: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_204, view_247, permute_135);  primals_204 = None
    view_253: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_74, [1, 128, 1024]);  addmm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_254: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_253, [1, -1, 16, 64]);  view_253 = None
    permute_136: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_254, [0, 2, 1, 3]);  view_254 = None
    clone_99: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_136, memory_format = torch.contiguous_format);  permute_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_255: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_68, [1, 128, 16, 64]);  mul_68 = None
    permute_137: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_255, [0, 2, 1, 3]);  view_255 = None
    clone_100: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_256: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_100, [16, -1, 64]);  clone_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_257: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_98, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_258: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_99, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_138: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_257, [0, 2, 1]);  view_257 = None
    bmm_24: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_256, permute_138)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_259: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_24, [1, 16, 128, 128]);  bmm_24 = None
    unsqueeze_2: "f32[1, 128, 128]" = torch.ops.aten.unsqueeze.default(where, 0);  where = None
    unsqueeze_3: "f32[1, 1, 128, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 1);  unsqueeze_2 = None
    expand_1: "f32[1, 1, 128, 128]" = torch.ops.aten.expand.default(unsqueeze_3, [1, 1, 128, 128]);  unsqueeze_3 = None
    add_83: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_259, expand_1);  view_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_260: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(add_83, [16, 128, 128]);  add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_12: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_260, [-1], True)
    sub_38: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_260, amax_12);  view_260 = amax_12 = None
    exp_12: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_38);  sub_38 = None
    sum_13: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_12: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    alias_28: "f32[16, 128, 128]" = torch.ops.aten.alias.default(div_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_101: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_12);  div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_25: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_101, view_258)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_261: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_25, [1, 16, 128, 64]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_139: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_261, [0, 2, 1, 3]);  view_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_102: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
    view_262: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_102, [1, 128, 1024]);  clone_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_263: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_262, [128, 1024]);  view_262 = None
    permute_140: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_205, [1, 0]);  primals_205 = None
    addmm_75: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_206, view_263, permute_140);  primals_206 = None
    view_264: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_75, [1, 128, 1024]);  addmm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    add_84: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_80, view_264);  add_80 = view_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_26 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
    getitem_52: "f32[1, 128, 1]" = var_mean_26[0]
    getitem_53: "f32[1, 128, 1]" = var_mean_26[1];  var_mean_26 = None
    add_85: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
    rsqrt_26: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
    sub_39: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_84, getitem_53);  getitem_53 = None
    mul_69: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_26);  sub_39 = None
    mul_70: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_69, primals_207)
    add_86: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_70, primals_208);  mul_70 = primals_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_265: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_86, [128, 1024]);  add_86 = None
    permute_141: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_209, [1, 0]);  primals_209 = None
    addmm_76: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_210, view_265, permute_141);  primals_210 = None
    view_266: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_76, [1, 128, 1024]);  addmm_76 = None
    mul_71: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_266, 0.125);  view_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_267: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_76, [128, 1024])
    permute_142: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_211, [1, 0]);  primals_211 = None
    addmm_77: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_212, view_267, permute_142);  primals_212 = None
    view_268: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_77, [1, 128, 1024]);  addmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_269: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_268, [1, -1, 16, 64]);  view_268 = None
    permute_143: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_269, [0, 2, 1, 3]);  view_269 = None
    clone_104: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_143, memory_format = torch.contiguous_format);  permute_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_144: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_213, [1, 0]);  primals_213 = None
    addmm_78: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_214, view_267, permute_144);  primals_214 = None
    view_271: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_78, [1, 128, 1024]);  addmm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_272: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_271, [1, -1, 16, 64]);  view_271 = None
    permute_145: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None
    clone_105: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_145, memory_format = torch.contiguous_format);  permute_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_273: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_71, [1, 128, 16, 64]);  mul_71 = None
    permute_146: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_273, [0, 2, 1, 3]);  view_273 = None
    clone_106: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_146, memory_format = torch.contiguous_format);  permute_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_274: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_106, [16, -1, 64]);  clone_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_275: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_104, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_276: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_105, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_147: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_275, [0, 2, 1]);  view_275 = None
    bmm_26: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_274, permute_147)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_13: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_26, [-1], True)
    sub_40: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_26, amax_13)
    exp_13: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
    sum_14: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_13: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_27: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_13, view_276)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_277: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_27, [1, 16, 128, 64]);  bmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_148: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_277, [0, 2, 1, 3]);  view_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_108: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
    view_278: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_108, [1, 128, 1024]);  clone_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_279: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_278, [128, 1024]);  view_278 = None
    permute_149: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_215, [1, 0]);  primals_215 = None
    addmm_79: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_216, view_279, permute_149);  primals_216 = None
    view_280: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_79, [1, 128, 1024]);  addmm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    add_87: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_84, view_280);  add_84 = view_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_27 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
    getitem_54: "f32[1, 128, 1]" = var_mean_27[0]
    getitem_55: "f32[1, 128, 1]" = var_mean_27[1];  var_mean_27 = None
    add_88: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
    rsqrt_27: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
    sub_41: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_87, getitem_55);  getitem_55 = None
    mul_72: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_27);  sub_41 = None
    mul_73: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_72, primals_217)
    add_89: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_73, primals_218);  mul_73 = primals_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_281: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_89, [128, 1024]);  add_89 = None
    permute_150: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_219, [1, 0]);  primals_219 = None
    addmm_80: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_220, view_281, permute_150);  primals_220 = None
    view_282: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_80, [1, 128, 4096]);  addmm_80 = None
    relu_12: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_282);  view_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_283: "f32[128, 4096]" = torch.ops.aten.reshape.default(relu_12, [128, 4096])
    permute_151: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_221, [1, 0]);  primals_221 = None
    addmm_81: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_222, view_283, permute_151);  primals_222 = None
    view_284: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_81, [1, 128, 1024]);  addmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    add_90: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_87, view_284);  add_87 = view_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_28 = torch.ops.aten.var_mean.correction(add_90, [2], correction = 0, keepdim = True)
    getitem_56: "f32[1, 128, 1]" = var_mean_28[0]
    getitem_57: "f32[1, 128, 1]" = var_mean_28[1];  var_mean_28 = None
    add_91: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
    rsqrt_28: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
    sub_42: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_90, getitem_57);  getitem_57 = None
    mul_74: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_28);  sub_42 = None
    mul_75: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_74, primals_223)
    add_92: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_75, primals_224);  mul_75 = primals_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_285: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_92, [128, 1024]);  add_92 = None
    permute_152: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_225, [1, 0]);  primals_225 = None
    addmm_82: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_226, view_285, permute_152);  primals_226 = None
    view_286: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_82, [1, 128, 1024]);  addmm_82 = None
    mul_76: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_286, 0.125);  view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_153: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_227, [1, 0]);  primals_227 = None
    addmm_83: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_228, view_285, permute_153);  primals_228 = None
    view_288: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_83, [1, 128, 1024]);  addmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_289: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_288, [1, -1, 16, 64]);  view_288 = None
    permute_154: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_289, [0, 2, 1, 3]);  view_289 = None
    clone_112: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_154, memory_format = torch.contiguous_format);  permute_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_155: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_229, [1, 0]);  primals_229 = None
    addmm_84: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_230, view_285, permute_155);  primals_230 = None
    view_291: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_84, [1, 128, 1024]);  addmm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_292: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_291, [1, -1, 16, 64]);  view_291 = None
    permute_156: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None
    clone_113: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_156, memory_format = torch.contiguous_format);  permute_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_293: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_76, [1, 128, 16, 64]);  mul_76 = None
    permute_157: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_293, [0, 2, 1, 3]);  view_293 = None
    clone_114: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_157, memory_format = torch.contiguous_format);  permute_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_294: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_114, [16, -1, 64]);  clone_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_295: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_112, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_296: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_113, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_158: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_295, [0, 2, 1]);  view_295 = None
    bmm_28: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_294, permute_158)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_297: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_28, [1, 16, 128, 128]);  bmm_28 = None
    add_93: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_297, expand_1);  view_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_298: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(add_93, [16, 128, 128]);  add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_14: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_298, [-1], True)
    sub_43: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_298, amax_14);  view_298 = amax_14 = None
    exp_14: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
    sum_15: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_14: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    alias_31: "f32[16, 128, 128]" = torch.ops.aten.alias.default(div_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_115: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_14);  div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_29: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_115, view_296)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_299: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_29, [1, 16, 128, 64]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_159: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_299, [0, 2, 1, 3]);  view_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_116: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None
    view_300: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_116, [1, 128, 1024]);  clone_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_301: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_300, [128, 1024]);  view_300 = None
    permute_160: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_231, [1, 0]);  primals_231 = None
    addmm_85: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_232, view_301, permute_160);  primals_232 = None
    view_302: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_85, [1, 128, 1024]);  addmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    add_94: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_90, view_302);  add_90 = view_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_29 = torch.ops.aten.var_mean.correction(add_94, [2], correction = 0, keepdim = True)
    getitem_58: "f32[1, 128, 1]" = var_mean_29[0]
    getitem_59: "f32[1, 128, 1]" = var_mean_29[1];  var_mean_29 = None
    add_95: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
    rsqrt_29: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
    sub_44: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_94, getitem_59);  getitem_59 = None
    mul_77: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_29);  sub_44 = None
    mul_78: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_77, primals_233)
    add_96: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_78, primals_234);  mul_78 = primals_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_303: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_96, [128, 1024]);  add_96 = None
    permute_161: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_235, [1, 0]);  primals_235 = None
    addmm_86: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_236, view_303, permute_161);  primals_236 = None
    view_304: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_86, [1, 128, 1024]);  addmm_86 = None
    mul_79: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_304, 0.125);  view_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_162: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_237, [1, 0]);  primals_237 = None
    addmm_87: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_238, view_267, permute_162);  primals_238 = None
    view_306: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_87, [1, 128, 1024]);  addmm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_307: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_306, [1, -1, 16, 64]);  view_306 = None
    permute_163: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_307, [0, 2, 1, 3]);  view_307 = None
    clone_118: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_163, memory_format = torch.contiguous_format);  permute_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_164: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_239, [1, 0]);  primals_239 = None
    addmm_88: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_240, view_267, permute_164);  primals_240 = None
    view_309: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_88, [1, 128, 1024]);  addmm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_310: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_309, [1, -1, 16, 64]);  view_309 = None
    permute_165: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_310, [0, 2, 1, 3]);  view_310 = None
    clone_119: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_165, memory_format = torch.contiguous_format);  permute_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_311: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_79, [1, 128, 16, 64]);  mul_79 = None
    permute_166: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_311, [0, 2, 1, 3]);  view_311 = None
    clone_120: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_166, memory_format = torch.contiguous_format);  permute_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_312: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_120, [16, -1, 64]);  clone_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_313: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_118, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_314: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_119, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_167: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_313, [0, 2, 1]);  view_313 = None
    bmm_30: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_312, permute_167)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_15: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_30, [-1], True)
    sub_45: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_30, amax_15)
    exp_15: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_45);  sub_45 = None
    sum_16: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_15: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_31: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_15, view_314)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_315: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_31, [1, 16, 128, 64]);  bmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_168: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_315, [0, 2, 1, 3]);  view_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_122: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
    view_316: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_122, [1, 128, 1024]);  clone_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_317: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_316, [128, 1024]);  view_316 = None
    permute_169: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_241, [1, 0]);  primals_241 = None
    addmm_89: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_242, view_317, permute_169);  primals_242 = None
    view_318: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_89, [1, 128, 1024]);  addmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    add_97: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_94, view_318);  add_94 = view_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_30 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
    getitem_60: "f32[1, 128, 1]" = var_mean_30[0]
    getitem_61: "f32[1, 128, 1]" = var_mean_30[1];  var_mean_30 = None
    add_98: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
    rsqrt_30: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    sub_46: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_97, getitem_61);  getitem_61 = None
    mul_80: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_30);  sub_46 = None
    mul_81: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_80, primals_243)
    add_99: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_81, primals_244);  mul_81 = primals_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_319: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_99, [128, 1024]);  add_99 = None
    permute_170: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_245, [1, 0]);  primals_245 = None
    addmm_90: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_246, view_319, permute_170);  primals_246 = None
    view_320: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_90, [1, 128, 4096]);  addmm_90 = None
    relu_13: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_320);  view_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_321: "f32[128, 4096]" = torch.ops.aten.reshape.default(relu_13, [128, 4096])
    permute_171: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_247, [1, 0]);  primals_247 = None
    addmm_91: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_248, view_321, permute_171);  primals_248 = None
    view_322: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_91, [1, 128, 1024]);  addmm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    add_100: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_97, view_322);  add_97 = view_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_31 = torch.ops.aten.var_mean.correction(add_100, [2], correction = 0, keepdim = True)
    getitem_62: "f32[1, 128, 1]" = var_mean_31[0]
    getitem_63: "f32[1, 128, 1]" = var_mean_31[1];  var_mean_31 = None
    add_101: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
    rsqrt_31: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    sub_47: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_100, getitem_63);  getitem_63 = None
    mul_82: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_31);  sub_47 = None
    mul_83: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_82, primals_249)
    add_102: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_83, primals_250);  mul_83 = primals_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_323: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_102, [128, 1024]);  add_102 = None
    permute_172: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_251, [1, 0]);  primals_251 = None
    addmm_92: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_252, view_323, permute_172);  primals_252 = None
    view_324: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_92, [1, 128, 1024]);  addmm_92 = None
    mul_84: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_324, 0.125);  view_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_173: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_253, [1, 0]);  primals_253 = None
    addmm_93: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_254, view_323, permute_173);  primals_254 = None
    view_326: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_93, [1, 128, 1024]);  addmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_327: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_326, [1, -1, 16, 64]);  view_326 = None
    permute_174: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None
    clone_126: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_174, memory_format = torch.contiguous_format);  permute_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_175: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_255, [1, 0]);  primals_255 = None
    addmm_94: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_256, view_323, permute_175);  primals_256 = None
    view_329: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_94, [1, 128, 1024]);  addmm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_330: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_329, [1, -1, 16, 64]);  view_329 = None
    permute_176: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None
    clone_127: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_176, memory_format = torch.contiguous_format);  permute_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_331: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_84, [1, 128, 16, 64]);  mul_84 = None
    permute_177: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_331, [0, 2, 1, 3]);  view_331 = None
    clone_128: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_332: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_128, [16, -1, 64]);  clone_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_333: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_126, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_334: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_127, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_178: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_333, [0, 2, 1]);  view_333 = None
    bmm_32: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_332, permute_178)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_335: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_32, [1, 16, 128, 128]);  bmm_32 = None
    add_103: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_335, expand_1);  view_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_336: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(add_103, [16, 128, 128]);  add_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_16: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_336, [-1], True)
    sub_48: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_336, amax_16);  view_336 = amax_16 = None
    exp_16: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_48);  sub_48 = None
    sum_17: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_16: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    alias_34: "f32[16, 128, 128]" = torch.ops.aten.alias.default(div_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_129: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_16);  div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_33: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_129, view_334)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_337: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_33, [1, 16, 128, 64]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_179: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_337, [0, 2, 1, 3]);  view_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_130: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
    view_338: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_130, [1, 128, 1024]);  clone_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_339: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_338, [128, 1024]);  view_338 = None
    permute_180: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_257, [1, 0]);  primals_257 = None
    addmm_95: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_258, view_339, permute_180);  primals_258 = None
    view_340: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_95, [1, 128, 1024]);  addmm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    add_104: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_100, view_340);  add_100 = view_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_32 = torch.ops.aten.var_mean.correction(add_104, [2], correction = 0, keepdim = True)
    getitem_64: "f32[1, 128, 1]" = var_mean_32[0]
    getitem_65: "f32[1, 128, 1]" = var_mean_32[1];  var_mean_32 = None
    add_105: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
    rsqrt_32: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    sub_49: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_104, getitem_65);  getitem_65 = None
    mul_85: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_32);  sub_49 = None
    mul_86: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_85, primals_259)
    add_106: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_86, primals_260);  mul_86 = primals_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_341: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_106, [128, 1024]);  add_106 = None
    permute_181: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_261, [1, 0]);  primals_261 = None
    addmm_96: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_262, view_341, permute_181);  primals_262 = None
    view_342: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_96, [1, 128, 1024]);  addmm_96 = None
    mul_87: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_342, 0.125);  view_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_182: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_263, [1, 0]);  primals_263 = None
    addmm_97: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_264, view_267, permute_182);  primals_264 = None
    view_344: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_97, [1, 128, 1024]);  addmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_345: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_344, [1, -1, 16, 64]);  view_344 = None
    permute_183: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_345, [0, 2, 1, 3]);  view_345 = None
    clone_132: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_184: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_265, [1, 0]);  primals_265 = None
    addmm_98: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_266, view_267, permute_184);  primals_266 = None
    view_347: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_98, [1, 128, 1024]);  addmm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_348: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_347, [1, -1, 16, 64]);  view_347 = None
    permute_185: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_348, [0, 2, 1, 3]);  view_348 = None
    clone_133: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_185, memory_format = torch.contiguous_format);  permute_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_349: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_87, [1, 128, 16, 64]);  mul_87 = None
    permute_186: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_349, [0, 2, 1, 3]);  view_349 = None
    clone_134: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_186, memory_format = torch.contiguous_format);  permute_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_350: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_134, [16, -1, 64]);  clone_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_351: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_132, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_352: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_133, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_187: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_351, [0, 2, 1]);  view_351 = None
    bmm_34: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_350, permute_187)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_17: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_34, [-1], True)
    sub_50: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_34, amax_17)
    exp_17: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_50);  sub_50 = None
    sum_18: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_17: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_35: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_17, view_352)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_353: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_35, [1, 16, 128, 64]);  bmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_188: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_353, [0, 2, 1, 3]);  view_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_136: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
    view_354: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_136, [1, 128, 1024]);  clone_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_355: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_354, [128, 1024]);  view_354 = None
    permute_189: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_267, [1, 0]);  primals_267 = None
    addmm_99: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_268, view_355, permute_189);  primals_268 = None
    view_356: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_99, [1, 128, 1024]);  addmm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    add_107: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_104, view_356);  add_104 = view_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_33 = torch.ops.aten.var_mean.correction(add_107, [2], correction = 0, keepdim = True)
    getitem_66: "f32[1, 128, 1]" = var_mean_33[0]
    getitem_67: "f32[1, 128, 1]" = var_mean_33[1];  var_mean_33 = None
    add_108: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
    rsqrt_33: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    sub_51: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_107, getitem_67);  getitem_67 = None
    mul_88: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_33);  sub_51 = None
    mul_89: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_88, primals_269)
    add_109: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_89, primals_270);  mul_89 = primals_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_357: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_109, [128, 1024]);  add_109 = None
    permute_190: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_271, [1, 0]);  primals_271 = None
    addmm_100: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_272, view_357, permute_190);  primals_272 = None
    view_358: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_100, [1, 128, 4096]);  addmm_100 = None
    relu_14: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_358);  view_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_359: "f32[128, 4096]" = torch.ops.aten.reshape.default(relu_14, [128, 4096])
    permute_191: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_273, [1, 0]);  primals_273 = None
    addmm_101: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_274, view_359, permute_191);  primals_274 = None
    view_360: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_101, [1, 128, 1024]);  addmm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    add_110: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_107, view_360);  add_107 = view_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_34 = torch.ops.aten.var_mean.correction(add_110, [2], correction = 0, keepdim = True)
    getitem_68: "f32[1, 128, 1]" = var_mean_34[0]
    getitem_69: "f32[1, 128, 1]" = var_mean_34[1];  var_mean_34 = None
    add_111: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
    rsqrt_34: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
    sub_52: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_110, getitem_69);  getitem_69 = None
    mul_90: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_34);  sub_52 = None
    mul_91: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_90, primals_275)
    add_112: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_91, primals_276);  mul_91 = primals_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_361: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_112, [128, 1024]);  add_112 = None
    permute_192: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_277, [1, 0]);  primals_277 = None
    addmm_102: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_278, view_361, permute_192);  primals_278 = None
    view_362: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_102, [1, 128, 1024]);  addmm_102 = None
    mul_92: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_362, 0.125);  view_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_193: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_279, [1, 0]);  primals_279 = None
    addmm_103: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_280, view_361, permute_193);  primals_280 = None
    view_364: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_103, [1, 128, 1024]);  addmm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_365: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_364, [1, -1, 16, 64]);  view_364 = None
    permute_194: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_365, [0, 2, 1, 3]);  view_365 = None
    clone_140: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_195: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_281, [1, 0]);  primals_281 = None
    addmm_104: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_282, view_361, permute_195);  primals_282 = None
    view_367: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_104, [1, 128, 1024]);  addmm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_368: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_367, [1, -1, 16, 64]);  view_367 = None
    permute_196: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_368, [0, 2, 1, 3]);  view_368 = None
    clone_141: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_196, memory_format = torch.contiguous_format);  permute_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_369: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_92, [1, 128, 16, 64]);  mul_92 = None
    permute_197: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_369, [0, 2, 1, 3]);  view_369 = None
    clone_142: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_197, memory_format = torch.contiguous_format);  permute_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_370: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_142, [16, -1, 64]);  clone_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_371: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_140, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_372: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_141, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_198: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_371, [0, 2, 1]);  view_371 = None
    bmm_36: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_370, permute_198)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_373: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_36, [1, 16, 128, 128]);  bmm_36 = None
    add_113: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_373, expand_1);  view_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_374: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(add_113, [16, 128, 128]);  add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_18: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_374, [-1], True)
    sub_53: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_374, amax_18);  view_374 = amax_18 = None
    exp_18: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_53);  sub_53 = None
    sum_19: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
    div_18: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
    alias_37: "f32[16, 128, 128]" = torch.ops.aten.alias.default(div_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_143: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_18);  div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_37: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_143, view_372)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_375: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_37, [1, 16, 128, 64]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_199: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_375, [0, 2, 1, 3]);  view_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_144: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None
    view_376: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_144, [1, 128, 1024]);  clone_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_377: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_376, [128, 1024]);  view_376 = None
    permute_200: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_283, [1, 0]);  primals_283 = None
    addmm_105: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_284, view_377, permute_200);  primals_284 = None
    view_378: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_105, [1, 128, 1024]);  addmm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    add_114: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_110, view_378);  add_110 = view_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_35 = torch.ops.aten.var_mean.correction(add_114, [2], correction = 0, keepdim = True)
    getitem_70: "f32[1, 128, 1]" = var_mean_35[0]
    getitem_71: "f32[1, 128, 1]" = var_mean_35[1];  var_mean_35 = None
    add_115: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
    rsqrt_35: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    sub_54: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_114, getitem_71);  getitem_71 = None
    mul_93: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_35);  sub_54 = None
    mul_94: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_93, primals_285)
    add_116: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_94, primals_286);  mul_94 = primals_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_379: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_116, [128, 1024]);  add_116 = None
    permute_201: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_287, [1, 0]);  primals_287 = None
    addmm_106: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_288, view_379, permute_201);  primals_288 = None
    view_380: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_106, [1, 128, 1024]);  addmm_106 = None
    mul_95: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_380, 0.125);  view_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_202: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_289, [1, 0]);  primals_289 = None
    addmm_107: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_290, view_267, permute_202);  primals_290 = None
    view_382: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_107, [1, 128, 1024]);  addmm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_383: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_382, [1, -1, 16, 64]);  view_382 = None
    permute_203: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_383, [0, 2, 1, 3]);  view_383 = None
    clone_146: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_204: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_291, [1, 0]);  primals_291 = None
    addmm_108: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_292, view_267, permute_204);  primals_292 = None
    view_385: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_108, [1, 128, 1024]);  addmm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_386: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_385, [1, -1, 16, 64]);  view_385 = None
    permute_205: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_386, [0, 2, 1, 3]);  view_386 = None
    clone_147: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_387: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_95, [1, 128, 16, 64]);  mul_95 = None
    permute_206: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_387, [0, 2, 1, 3]);  view_387 = None
    clone_148: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_206, memory_format = torch.contiguous_format);  permute_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_388: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_148, [16, -1, 64]);  clone_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_389: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_146, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_390: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_147, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_207: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_389, [0, 2, 1]);  view_389 = None
    bmm_38: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_388, permute_207)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_19: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_38, [-1], True)
    sub_55: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_38, amax_19)
    exp_19: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
    sum_20: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
    div_19: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_39: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_19, view_390)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_391: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_39, [1, 16, 128, 64]);  bmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_208: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_391, [0, 2, 1, 3]);  view_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_150: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_208, memory_format = torch.contiguous_format);  permute_208 = None
    view_392: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_150, [1, 128, 1024]);  clone_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_393: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_392, [128, 1024]);  view_392 = None
    permute_209: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_293, [1, 0]);  primals_293 = None
    addmm_109: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_294, view_393, permute_209);  primals_294 = None
    view_394: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_109, [1, 128, 1024]);  addmm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    add_117: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_114, view_394);  add_114 = view_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_36 = torch.ops.aten.var_mean.correction(add_117, [2], correction = 0, keepdim = True)
    getitem_72: "f32[1, 128, 1]" = var_mean_36[0]
    getitem_73: "f32[1, 128, 1]" = var_mean_36[1];  var_mean_36 = None
    add_118: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
    rsqrt_36: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    sub_56: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_117, getitem_73);  getitem_73 = None
    mul_96: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_36);  sub_56 = None
    mul_97: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_96, primals_295)
    add_119: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_97, primals_296);  mul_97 = primals_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_395: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_119, [128, 1024]);  add_119 = None
    permute_210: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_297, [1, 0]);  primals_297 = None
    addmm_110: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_298, view_395, permute_210);  primals_298 = None
    view_396: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_110, [1, 128, 4096]);  addmm_110 = None
    relu_15: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_396);  view_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_397: "f32[128, 4096]" = torch.ops.aten.reshape.default(relu_15, [128, 4096])
    permute_211: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_299, [1, 0]);  primals_299 = None
    addmm_111: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_300, view_397, permute_211);  primals_300 = None
    view_398: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_111, [1, 128, 1024]);  addmm_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    add_120: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_117, view_398);  add_117 = view_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_37 = torch.ops.aten.var_mean.correction(add_120, [2], correction = 0, keepdim = True)
    getitem_74: "f32[1, 128, 1]" = var_mean_37[0]
    getitem_75: "f32[1, 128, 1]" = var_mean_37[1];  var_mean_37 = None
    add_121: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
    rsqrt_37: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
    sub_57: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_120, getitem_75);  getitem_75 = None
    mul_98: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_37);  sub_57 = None
    mul_99: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_98, primals_301)
    add_122: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_99, primals_302);  mul_99 = primals_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_399: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_122, [128, 1024]);  add_122 = None
    permute_212: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_303, [1, 0]);  primals_303 = None
    addmm_112: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_304, view_399, permute_212);  primals_304 = None
    view_400: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_112, [1, 128, 1024]);  addmm_112 = None
    mul_100: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_400, 0.125);  view_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_213: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_305, [1, 0]);  primals_305 = None
    addmm_113: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_306, view_399, permute_213);  primals_306 = None
    view_402: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_113, [1, 128, 1024]);  addmm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_403: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_402, [1, -1, 16, 64]);  view_402 = None
    permute_214: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_403, [0, 2, 1, 3]);  view_403 = None
    clone_154: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_214, memory_format = torch.contiguous_format);  permute_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_215: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_307, [1, 0]);  primals_307 = None
    addmm_114: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_308, view_399, permute_215);  primals_308 = None
    view_405: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_114, [1, 128, 1024]);  addmm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_406: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_405, [1, -1, 16, 64]);  view_405 = None
    permute_216: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_406, [0, 2, 1, 3]);  view_406 = None
    clone_155: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_407: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_100, [1, 128, 16, 64]);  mul_100 = None
    permute_217: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_407, [0, 2, 1, 3]);  view_407 = None
    clone_156: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_217, memory_format = torch.contiguous_format);  permute_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_408: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_156, [16, -1, 64]);  clone_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_409: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_154, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_410: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_155, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_218: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_409, [0, 2, 1]);  view_409 = None
    bmm_40: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_408, permute_218)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_411: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_40, [1, 16, 128, 128]);  bmm_40 = None
    add_123: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_411, expand_1);  view_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_412: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(add_123, [16, 128, 128]);  add_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_20: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_412, [-1], True)
    sub_58: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_412, amax_20);  view_412 = amax_20 = None
    exp_20: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_58);  sub_58 = None
    sum_21: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
    div_20: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
    alias_40: "f32[16, 128, 128]" = torch.ops.aten.alias.default(div_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_157: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_20);  div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_41: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_157, view_410)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_413: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_41, [1, 16, 128, 64]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_219: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_413, [0, 2, 1, 3]);  view_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_158: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None
    view_414: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_158, [1, 128, 1024]);  clone_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_415: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_414, [128, 1024]);  view_414 = None
    permute_220: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_309, [1, 0]);  primals_309 = None
    addmm_115: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_310, view_415, permute_220);  primals_310 = None
    view_416: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_115, [1, 128, 1024]);  addmm_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    add_124: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_120, view_416);  add_120 = view_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_38 = torch.ops.aten.var_mean.correction(add_124, [2], correction = 0, keepdim = True)
    getitem_76: "f32[1, 128, 1]" = var_mean_38[0]
    getitem_77: "f32[1, 128, 1]" = var_mean_38[1];  var_mean_38 = None
    add_125: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
    rsqrt_38: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_125);  add_125 = None
    sub_59: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_124, getitem_77);  getitem_77 = None
    mul_101: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_38);  sub_59 = None
    mul_102: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_101, primals_311)
    add_126: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_102, primals_312);  mul_102 = primals_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_417: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_126, [128, 1024]);  add_126 = None
    permute_221: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_313, [1, 0]);  primals_313 = None
    addmm_116: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_314, view_417, permute_221);  primals_314 = None
    view_418: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_116, [1, 128, 1024]);  addmm_116 = None
    mul_103: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_418, 0.125);  view_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_222: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_315, [1, 0]);  primals_315 = None
    addmm_117: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_316, view_267, permute_222);  primals_316 = None
    view_420: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_117, [1, 128, 1024]);  addmm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_421: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_420, [1, -1, 16, 64]);  view_420 = None
    permute_223: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_421, [0, 2, 1, 3]);  view_421 = None
    clone_160: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_223, memory_format = torch.contiguous_format);  permute_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_224: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_317, [1, 0]);  primals_317 = None
    addmm_118: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_318, view_267, permute_224);  primals_318 = None
    view_423: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_118, [1, 128, 1024]);  addmm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_424: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_423, [1, -1, 16, 64]);  view_423 = None
    permute_225: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_424, [0, 2, 1, 3]);  view_424 = None
    clone_161: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_225, memory_format = torch.contiguous_format);  permute_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_425: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_103, [1, 128, 16, 64]);  mul_103 = None
    permute_226: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_425, [0, 2, 1, 3]);  view_425 = None
    clone_162: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_226, memory_format = torch.contiguous_format);  permute_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_426: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_162, [16, -1, 64]);  clone_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_427: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_160, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_428: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_161, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_227: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_427, [0, 2, 1]);  view_427 = None
    bmm_42: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_426, permute_227)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_21: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_42, [-1], True)
    sub_60: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_42, amax_21)
    exp_21: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_60);  sub_60 = None
    sum_22: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
    div_21: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_43: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_21, view_428)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_429: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_43, [1, 16, 128, 64]);  bmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_228: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_429, [0, 2, 1, 3]);  view_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_164: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_228, memory_format = torch.contiguous_format);  permute_228 = None
    view_430: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_164, [1, 128, 1024]);  clone_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_431: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_430, [128, 1024]);  view_430 = None
    permute_229: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_319, [1, 0]);  primals_319 = None
    addmm_119: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_320, view_431, permute_229);  primals_320 = None
    view_432: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_119, [1, 128, 1024]);  addmm_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    add_127: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_124, view_432);  add_124 = view_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_39 = torch.ops.aten.var_mean.correction(add_127, [2], correction = 0, keepdim = True)
    getitem_78: "f32[1, 128, 1]" = var_mean_39[0]
    getitem_79: "f32[1, 128, 1]" = var_mean_39[1];  var_mean_39 = None
    add_128: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
    rsqrt_39: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
    sub_61: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_127, getitem_79);  getitem_79 = None
    mul_104: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_39);  sub_61 = None
    mul_105: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_104, primals_321)
    add_129: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_105, primals_322);  mul_105 = primals_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_433: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_129, [128, 1024]);  add_129 = None
    permute_230: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_323, [1, 0]);  primals_323 = None
    addmm_120: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_324, view_433, permute_230);  primals_324 = None
    view_434: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_120, [1, 128, 4096]);  addmm_120 = None
    relu_16: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_434);  view_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_435: "f32[128, 4096]" = torch.ops.aten.reshape.default(relu_16, [128, 4096])
    permute_231: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_325, [1, 0]);  primals_325 = None
    addmm_121: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_326, view_435, permute_231);  primals_326 = None
    view_436: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_121, [1, 128, 1024]);  addmm_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    add_130: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_127, view_436);  add_127 = view_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_40 = torch.ops.aten.var_mean.correction(add_130, [2], correction = 0, keepdim = True)
    getitem_80: "f32[1, 128, 1]" = var_mean_40[0]
    getitem_81: "f32[1, 128, 1]" = var_mean_40[1];  var_mean_40 = None
    add_131: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
    rsqrt_40: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
    sub_62: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_130, getitem_81);  getitem_81 = None
    mul_106: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_40);  sub_62 = None
    mul_107: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_106, primals_327)
    add_132: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_107, primals_328);  mul_107 = primals_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_437: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_132, [128, 1024]);  add_132 = None
    permute_232: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_329, [1, 0]);  primals_329 = None
    addmm_122: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_330, view_437, permute_232);  primals_330 = None
    view_438: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_122, [1, 128, 1024]);  addmm_122 = None
    mul_108: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_438, 0.125);  view_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_233: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_331, [1, 0]);  primals_331 = None
    addmm_123: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_332, view_437, permute_233);  primals_332 = None
    view_440: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_123, [1, 128, 1024]);  addmm_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_441: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_440, [1, -1, 16, 64]);  view_440 = None
    permute_234: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_441, [0, 2, 1, 3]);  view_441 = None
    clone_168: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_234, memory_format = torch.contiguous_format);  permute_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_235: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_333, [1, 0]);  primals_333 = None
    addmm_124: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_334, view_437, permute_235);  primals_334 = None
    view_443: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_124, [1, 128, 1024]);  addmm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_444: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_443, [1, -1, 16, 64]);  view_443 = None
    permute_236: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_444, [0, 2, 1, 3]);  view_444 = None
    clone_169: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_236, memory_format = torch.contiguous_format);  permute_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_445: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_108, [1, 128, 16, 64]);  mul_108 = None
    permute_237: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_445, [0, 2, 1, 3]);  view_445 = None
    clone_170: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_237, memory_format = torch.contiguous_format);  permute_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_446: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_170, [16, -1, 64]);  clone_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_447: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_168, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_448: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_169, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_238: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_447, [0, 2, 1]);  view_447 = None
    bmm_44: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_446, permute_238)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_449: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_44, [1, 16, 128, 128]);  bmm_44 = None
    add_133: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_449, expand_1);  view_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_450: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(add_133, [16, 128, 128]);  add_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_22: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_450, [-1], True)
    sub_63: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_450, amax_22);  view_450 = amax_22 = None
    exp_22: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_63);  sub_63 = None
    sum_23: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
    div_22: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
    alias_43: "f32[16, 128, 128]" = torch.ops.aten.alias.default(div_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_171: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_22);  div_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_45: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_171, view_448)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_451: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_45, [1, 16, 128, 64]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_239: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_451, [0, 2, 1, 3]);  view_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_172: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_239, memory_format = torch.contiguous_format);  permute_239 = None
    view_452: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_172, [1, 128, 1024]);  clone_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_453: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_452, [128, 1024]);  view_452 = None
    permute_240: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_335, [1, 0]);  primals_335 = None
    addmm_125: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_336, view_453, permute_240);  primals_336 = None
    view_454: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_125, [1, 128, 1024]);  addmm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    add_134: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_130, view_454);  add_130 = view_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_41 = torch.ops.aten.var_mean.correction(add_134, [2], correction = 0, keepdim = True)
    getitem_82: "f32[1, 128, 1]" = var_mean_41[0]
    getitem_83: "f32[1, 128, 1]" = var_mean_41[1];  var_mean_41 = None
    add_135: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
    rsqrt_41: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
    sub_64: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_134, getitem_83);  getitem_83 = None
    mul_109: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_41);  sub_64 = None
    mul_110: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_109, primals_337)
    add_136: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_110, primals_338);  mul_110 = primals_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_455: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_136, [128, 1024]);  add_136 = None
    permute_241: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_339, [1, 0]);  primals_339 = None
    addmm_126: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_340, view_455, permute_241);  primals_340 = None
    view_456: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_126, [1, 128, 1024]);  addmm_126 = None
    mul_111: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_456, 0.125);  view_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_242: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_341, [1, 0]);  primals_341 = None
    addmm_127: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_342, view_267, permute_242);  primals_342 = None
    view_458: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_127, [1, 128, 1024]);  addmm_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_459: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_458, [1, -1, 16, 64]);  view_458 = None
    permute_243: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_459, [0, 2, 1, 3]);  view_459 = None
    clone_174: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_243, memory_format = torch.contiguous_format);  permute_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_244: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_343, [1, 0]);  primals_343 = None
    addmm_128: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_344, view_267, permute_244);  primals_344 = None
    view_461: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_128, [1, 128, 1024]);  addmm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_462: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_461, [1, -1, 16, 64]);  view_461 = None
    permute_245: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_462, [0, 2, 1, 3]);  view_462 = None
    clone_175: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_245, memory_format = torch.contiguous_format);  permute_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_463: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_111, [1, 128, 16, 64]);  mul_111 = None
    permute_246: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_463, [0, 2, 1, 3]);  view_463 = None
    clone_176: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_246, memory_format = torch.contiguous_format);  permute_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_464: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_176, [16, -1, 64]);  clone_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_465: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_174, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_466: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_175, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_247: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_465, [0, 2, 1]);  view_465 = None
    bmm_46: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_464, permute_247)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_23: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_46, [-1], True)
    sub_65: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_46, amax_23)
    exp_23: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_65);  sub_65 = None
    sum_24: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
    div_23: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_47: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_23, view_466)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_467: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_47, [1, 16, 128, 64]);  bmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_248: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_467, [0, 2, 1, 3]);  view_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_178: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_248, memory_format = torch.contiguous_format);  permute_248 = None
    view_468: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_178, [1, 128, 1024]);  clone_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_469: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_468, [128, 1024]);  view_468 = None
    permute_249: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_345, [1, 0]);  primals_345 = None
    addmm_129: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_346, view_469, permute_249);  primals_346 = None
    view_470: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_129, [1, 128, 1024]);  addmm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    add_137: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_134, view_470);  add_134 = view_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_42 = torch.ops.aten.var_mean.correction(add_137, [2], correction = 0, keepdim = True)
    getitem_84: "f32[1, 128, 1]" = var_mean_42[0]
    getitem_85: "f32[1, 128, 1]" = var_mean_42[1];  var_mean_42 = None
    add_138: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
    rsqrt_42: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
    sub_66: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_137, getitem_85);  getitem_85 = None
    mul_112: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_42);  sub_66 = None
    mul_113: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_112, primals_347)
    add_139: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_113, primals_348);  mul_113 = primals_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_471: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_139, [128, 1024]);  add_139 = None
    permute_250: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_349, [1, 0]);  primals_349 = None
    addmm_130: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_350, view_471, permute_250);  primals_350 = None
    view_472: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_130, [1, 128, 4096]);  addmm_130 = None
    relu_17: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_472);  view_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_473: "f32[128, 4096]" = torch.ops.aten.reshape.default(relu_17, [128, 4096])
    permute_251: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_351, [1, 0]);  primals_351 = None
    addmm_131: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_352, view_473, permute_251);  primals_352 = None
    view_474: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_131, [1, 128, 1024]);  addmm_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    add_140: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_137, view_474);  add_137 = view_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_43 = torch.ops.aten.var_mean.correction(add_140, [2], correction = 0, keepdim = True)
    getitem_86: "f32[1, 128, 1]" = var_mean_43[0]
    getitem_87: "f32[1, 128, 1]" = var_mean_43[1];  var_mean_43 = None
    add_141: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
    rsqrt_43: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
    sub_67: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_140, getitem_87);  getitem_87 = None
    mul_114: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_43);  sub_67 = None
    mul_115: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_114, primals_353)
    add_142: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_115, primals_354);  mul_115 = primals_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_475: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_142, [128, 1024]);  add_142 = None
    permute_252: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_355, [1, 0]);  primals_355 = None
    addmm_132: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_356, view_475, permute_252);  primals_356 = None
    view_476: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_132, [1, 128, 1024]);  addmm_132 = None
    mul_116: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_476, 0.125);  view_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_253: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_357, [1, 0]);  primals_357 = None
    addmm_133: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_358, view_475, permute_253);  primals_358 = None
    view_478: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_133, [1, 128, 1024]);  addmm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_479: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_478, [1, -1, 16, 64]);  view_478 = None
    permute_254: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_479, [0, 2, 1, 3]);  view_479 = None
    clone_182: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_254, memory_format = torch.contiguous_format);  permute_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_255: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_359, [1, 0]);  primals_359 = None
    addmm_134: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_360, view_475, permute_255);  primals_360 = None
    view_481: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_134, [1, 128, 1024]);  addmm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_482: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_481, [1, -1, 16, 64]);  view_481 = None
    permute_256: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_482, [0, 2, 1, 3]);  view_482 = None
    clone_183: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_256, memory_format = torch.contiguous_format);  permute_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_483: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_116, [1, 128, 16, 64]);  mul_116 = None
    permute_257: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_483, [0, 2, 1, 3]);  view_483 = None
    clone_184: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_257, memory_format = torch.contiguous_format);  permute_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_484: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_184, [16, -1, 64]);  clone_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_485: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_182, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_486: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_183, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_258: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_485, [0, 2, 1]);  view_485 = None
    bmm_48: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_484, permute_258)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_487: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_48, [1, 16, 128, 128]);  bmm_48 = None
    add_143: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_487, expand_1);  view_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_488: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(add_143, [16, 128, 128]);  add_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_24: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_488, [-1], True)
    sub_68: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_488, amax_24);  view_488 = amax_24 = None
    exp_24: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_68);  sub_68 = None
    sum_25: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [-1], True)
    div_24: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_24, sum_25);  exp_24 = sum_25 = None
    alias_46: "f32[16, 128, 128]" = torch.ops.aten.alias.default(div_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_185: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_24);  div_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_49: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_185, view_486)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_489: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_49, [1, 16, 128, 64]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_259: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_489, [0, 2, 1, 3]);  view_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_186: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_259, memory_format = torch.contiguous_format);  permute_259 = None
    view_490: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_186, [1, 128, 1024]);  clone_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_491: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_490, [128, 1024]);  view_490 = None
    permute_260: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_361, [1, 0]);  primals_361 = None
    addmm_135: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_362, view_491, permute_260);  primals_362 = None
    view_492: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_135, [1, 128, 1024]);  addmm_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    add_144: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_140, view_492);  add_140 = view_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_44 = torch.ops.aten.var_mean.correction(add_144, [2], correction = 0, keepdim = True)
    getitem_88: "f32[1, 128, 1]" = var_mean_44[0]
    getitem_89: "f32[1, 128, 1]" = var_mean_44[1];  var_mean_44 = None
    add_145: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
    rsqrt_44: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
    sub_69: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_144, getitem_89);  getitem_89 = None
    mul_117: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_44);  sub_69 = None
    mul_118: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_117, primals_363)
    add_146: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_118, primals_364);  mul_118 = primals_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_493: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_146, [128, 1024]);  add_146 = None
    permute_261: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_365, [1, 0]);  primals_365 = None
    addmm_136: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_366, view_493, permute_261);  primals_366 = None
    view_494: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_136, [1, 128, 1024]);  addmm_136 = None
    mul_119: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_494, 0.125);  view_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_262: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_367, [1, 0]);  primals_367 = None
    addmm_137: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_368, view_267, permute_262);  primals_368 = None
    view_496: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_137, [1, 128, 1024]);  addmm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_497: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_496, [1, -1, 16, 64]);  view_496 = None
    permute_263: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_497, [0, 2, 1, 3]);  view_497 = None
    clone_188: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_263, memory_format = torch.contiguous_format);  permute_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_264: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_369, [1, 0]);  primals_369 = None
    addmm_138: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_370, view_267, permute_264);  primals_370 = None
    view_499: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_138, [1, 128, 1024]);  addmm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_500: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_499, [1, -1, 16, 64]);  view_499 = None
    permute_265: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_500, [0, 2, 1, 3]);  view_500 = None
    clone_189: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_265, memory_format = torch.contiguous_format);  permute_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_501: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_119, [1, 128, 16, 64]);  mul_119 = None
    permute_266: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_501, [0, 2, 1, 3]);  view_501 = None
    clone_190: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_266, memory_format = torch.contiguous_format);  permute_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_502: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_190, [16, -1, 64]);  clone_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_503: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_188, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_504: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_189, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_267: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_503, [0, 2, 1]);  view_503 = None
    bmm_50: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_502, permute_267)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_25: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_50, [-1], True)
    sub_70: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_50, amax_25)
    exp_25: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_70);  sub_70 = None
    sum_26: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_25, [-1], True)
    div_25: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_25, sum_26);  exp_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_51: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_25, view_504)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_505: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_51, [1, 16, 128, 64]);  bmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_268: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_505, [0, 2, 1, 3]);  view_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_192: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_268, memory_format = torch.contiguous_format);  permute_268 = None
    view_506: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_192, [1, 128, 1024]);  clone_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_507: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_506, [128, 1024]);  view_506 = None
    permute_269: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_371, [1, 0]);  primals_371 = None
    addmm_139: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_372, view_507, permute_269);  primals_372 = None
    view_508: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_139, [1, 128, 1024]);  addmm_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    add_147: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_144, view_508);  add_144 = view_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_45 = torch.ops.aten.var_mean.correction(add_147, [2], correction = 0, keepdim = True)
    getitem_90: "f32[1, 128, 1]" = var_mean_45[0]
    getitem_91: "f32[1, 128, 1]" = var_mean_45[1];  var_mean_45 = None
    add_148: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
    rsqrt_45: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
    sub_71: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_147, getitem_91);  getitem_91 = None
    mul_120: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_45);  sub_71 = None
    mul_121: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_120, primals_373)
    add_149: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_121, primals_374);  mul_121 = primals_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_509: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_149, [128, 1024]);  add_149 = None
    permute_270: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_375, [1, 0]);  primals_375 = None
    addmm_140: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_376, view_509, permute_270);  primals_376 = None
    view_510: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_140, [1, 128, 4096]);  addmm_140 = None
    relu_18: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_510);  view_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_511: "f32[128, 4096]" = torch.ops.aten.reshape.default(relu_18, [128, 4096])
    permute_271: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_377, [1, 0]);  primals_377 = None
    addmm_141: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_378, view_511, permute_271);  primals_378 = None
    view_512: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_141, [1, 128, 1024]);  addmm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    add_150: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_147, view_512);  add_147 = view_512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_46 = torch.ops.aten.var_mean.correction(add_150, [2], correction = 0, keepdim = True)
    getitem_92: "f32[1, 128, 1]" = var_mean_46[0]
    getitem_93: "f32[1, 128, 1]" = var_mean_46[1];  var_mean_46 = None
    add_151: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
    rsqrt_46: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
    sub_72: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_150, getitem_93);  getitem_93 = None
    mul_122: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_46);  sub_72 = None
    mul_123: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_122, primals_379)
    add_152: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_123, primals_380);  mul_123 = primals_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_513: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_152, [128, 1024]);  add_152 = None
    permute_272: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_381, [1, 0]);  primals_381 = None
    addmm_142: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_382, view_513, permute_272);  primals_382 = None
    view_514: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_142, [1, 128, 1024]);  addmm_142 = None
    mul_124: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_514, 0.125);  view_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_273: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_383, [1, 0]);  primals_383 = None
    addmm_143: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_384, view_513, permute_273);  primals_384 = None
    view_516: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_143, [1, 128, 1024]);  addmm_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_517: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_516, [1, -1, 16, 64]);  view_516 = None
    permute_274: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_517, [0, 2, 1, 3]);  view_517 = None
    clone_196: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_274, memory_format = torch.contiguous_format);  permute_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_275: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_385, [1, 0]);  primals_385 = None
    addmm_144: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_386, view_513, permute_275);  primals_386 = None
    view_519: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_144, [1, 128, 1024]);  addmm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_520: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_519, [1, -1, 16, 64]);  view_519 = None
    permute_276: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_520, [0, 2, 1, 3]);  view_520 = None
    clone_197: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_276, memory_format = torch.contiguous_format);  permute_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_521: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_124, [1, 128, 16, 64]);  mul_124 = None
    permute_277: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_521, [0, 2, 1, 3]);  view_521 = None
    clone_198: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_277, memory_format = torch.contiguous_format);  permute_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_522: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_198, [16, -1, 64]);  clone_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_523: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_196, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_524: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_197, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_278: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_523, [0, 2, 1]);  view_523 = None
    bmm_52: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_522, permute_278)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_525: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_52, [1, 16, 128, 128]);  bmm_52 = None
    add_153: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_525, expand_1);  view_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_526: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(add_153, [16, 128, 128]);  add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_26: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_526, [-1], True)
    sub_73: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_526, amax_26);  view_526 = amax_26 = None
    exp_26: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_73);  sub_73 = None
    sum_27: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_26, [-1], True)
    div_26: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_26, sum_27);  exp_26 = sum_27 = None
    alias_49: "f32[16, 128, 128]" = torch.ops.aten.alias.default(div_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_199: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_26);  div_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_53: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_199, view_524)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_527: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_53, [1, 16, 128, 64]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_279: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_527, [0, 2, 1, 3]);  view_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_200: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_279, memory_format = torch.contiguous_format);  permute_279 = None
    view_528: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_200, [1, 128, 1024]);  clone_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_529: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_528, [128, 1024]);  view_528 = None
    permute_280: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_387, [1, 0]);  primals_387 = None
    addmm_145: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_388, view_529, permute_280);  primals_388 = None
    view_530: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_145, [1, 128, 1024]);  addmm_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    add_154: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_150, view_530);  add_150 = view_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_47 = torch.ops.aten.var_mean.correction(add_154, [2], correction = 0, keepdim = True)
    getitem_94: "f32[1, 128, 1]" = var_mean_47[0]
    getitem_95: "f32[1, 128, 1]" = var_mean_47[1];  var_mean_47 = None
    add_155: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
    rsqrt_47: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
    sub_74: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_154, getitem_95);  getitem_95 = None
    mul_125: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_47);  sub_74 = None
    mul_126: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_125, primals_389)
    add_156: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_126, primals_390);  mul_126 = primals_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_531: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_156, [128, 1024]);  add_156 = None
    permute_281: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_391, [1, 0]);  primals_391 = None
    addmm_146: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_392, view_531, permute_281);  primals_392 = None
    view_532: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_146, [1, 128, 1024]);  addmm_146 = None
    mul_127: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_532, 0.125);  view_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_282: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_393, [1, 0]);  primals_393 = None
    addmm_147: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_394, view_267, permute_282);  primals_394 = None
    view_534: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_147, [1, 128, 1024]);  addmm_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_535: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_534, [1, -1, 16, 64]);  view_534 = None
    permute_283: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_535, [0, 2, 1, 3]);  view_535 = None
    clone_202: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_283, memory_format = torch.contiguous_format);  permute_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_284: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_395, [1, 0]);  primals_395 = None
    addmm_148: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_396, view_267, permute_284);  primals_396 = None
    view_537: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_148, [1, 128, 1024]);  addmm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_538: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_537, [1, -1, 16, 64]);  view_537 = None
    permute_285: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_538, [0, 2, 1, 3]);  view_538 = None
    clone_203: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_285, memory_format = torch.contiguous_format);  permute_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_539: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_127, [1, 128, 16, 64]);  mul_127 = None
    permute_286: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_539, [0, 2, 1, 3]);  view_539 = None
    clone_204: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_286, memory_format = torch.contiguous_format);  permute_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_540: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_204, [16, -1, 64]);  clone_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_541: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_202, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_542: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_203, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_287: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_541, [0, 2, 1]);  view_541 = None
    bmm_54: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_540, permute_287)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_27: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_54, [-1], True)
    sub_75: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_54, amax_27)
    exp_27: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_75);  sub_75 = None
    sum_28: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_27, [-1], True)
    div_27: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_27, sum_28);  exp_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_55: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_27, view_542)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_543: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_55, [1, 16, 128, 64]);  bmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_288: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_543, [0, 2, 1, 3]);  view_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_206: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_288, memory_format = torch.contiguous_format);  permute_288 = None
    view_544: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_206, [1, 128, 1024]);  clone_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_545: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_544, [128, 1024]);  view_544 = None
    permute_289: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_397, [1, 0]);  primals_397 = None
    addmm_149: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_398, view_545, permute_289);  primals_398 = None
    view_546: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_149, [1, 128, 1024]);  addmm_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    add_157: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_154, view_546);  add_154 = view_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_48 = torch.ops.aten.var_mean.correction(add_157, [2], correction = 0, keepdim = True)
    getitem_96: "f32[1, 128, 1]" = var_mean_48[0]
    getitem_97: "f32[1, 128, 1]" = var_mean_48[1];  var_mean_48 = None
    add_158: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
    rsqrt_48: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
    sub_76: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_157, getitem_97);  getitem_97 = None
    mul_128: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_48);  sub_76 = None
    mul_129: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_128, primals_399)
    add_159: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_129, primals_400);  mul_129 = primals_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_547: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_159, [128, 1024]);  add_159 = None
    permute_290: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_401, [1, 0]);  primals_401 = None
    addmm_150: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_402, view_547, permute_290);  primals_402 = None
    view_548: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_150, [1, 128, 4096]);  addmm_150 = None
    relu_19: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_548);  view_548 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_549: "f32[128, 4096]" = torch.ops.aten.reshape.default(relu_19, [128, 4096])
    permute_291: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_403, [1, 0]);  primals_403 = None
    addmm_151: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_404, view_549, permute_291);  primals_404 = None
    view_550: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_151, [1, 128, 1024]);  addmm_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    add_160: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_157, view_550);  add_157 = view_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_49 = torch.ops.aten.var_mean.correction(add_160, [2], correction = 0, keepdim = True)
    getitem_98: "f32[1, 128, 1]" = var_mean_49[0]
    getitem_99: "f32[1, 128, 1]" = var_mean_49[1];  var_mean_49 = None
    add_161: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05);  getitem_98 = None
    rsqrt_49: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    sub_77: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_160, getitem_99);  getitem_99 = None
    mul_130: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_49);  sub_77 = None
    mul_131: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_130, primals_405)
    add_162: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_131, primals_406);  mul_131 = primals_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_551: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_162, [128, 1024]);  add_162 = None
    permute_292: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_407, [1, 0]);  primals_407 = None
    addmm_152: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_408, view_551, permute_292);  primals_408 = None
    view_552: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_152, [1, 128, 1024]);  addmm_152 = None
    mul_132: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_552, 0.125);  view_552 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_293: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_409, [1, 0]);  primals_409 = None
    addmm_153: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_410, view_551, permute_293);  primals_410 = None
    view_554: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_153, [1, 128, 1024]);  addmm_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_555: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_554, [1, -1, 16, 64]);  view_554 = None
    permute_294: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_555, [0, 2, 1, 3]);  view_555 = None
    clone_210: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_294, memory_format = torch.contiguous_format);  permute_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_295: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_411, [1, 0]);  primals_411 = None
    addmm_154: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_412, view_551, permute_295);  primals_412 = None
    view_557: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_154, [1, 128, 1024]);  addmm_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_558: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_557, [1, -1, 16, 64]);  view_557 = None
    permute_296: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_558, [0, 2, 1, 3]);  view_558 = None
    clone_211: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_296, memory_format = torch.contiguous_format);  permute_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_559: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_132, [1, 128, 16, 64]);  mul_132 = None
    permute_297: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_559, [0, 2, 1, 3]);  view_559 = None
    clone_212: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_297, memory_format = torch.contiguous_format);  permute_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_560: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_212, [16, -1, 64]);  clone_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_561: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_210, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_562: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_211, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_298: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_561, [0, 2, 1]);  view_561 = None
    bmm_56: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_560, permute_298)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_563: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_56, [1, 16, 128, 128]);  bmm_56 = None
    add_163: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_563, expand_1);  view_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_564: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(add_163, [16, 128, 128]);  add_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_28: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_564, [-1], True)
    sub_78: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_564, amax_28);  view_564 = amax_28 = None
    exp_28: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_78);  sub_78 = None
    sum_29: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_28, [-1], True)
    div_28: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_28, sum_29);  exp_28 = sum_29 = None
    alias_52: "f32[16, 128, 128]" = torch.ops.aten.alias.default(div_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_213: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_28);  div_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_57: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_213, view_562)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_565: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_57, [1, 16, 128, 64]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_299: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_565, [0, 2, 1, 3]);  view_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_214: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_299, memory_format = torch.contiguous_format);  permute_299 = None
    view_566: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_214, [1, 128, 1024]);  clone_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_567: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_566, [128, 1024]);  view_566 = None
    permute_300: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_413, [1, 0]);  primals_413 = None
    addmm_155: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_414, view_567, permute_300);  primals_414 = None
    view_568: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_155, [1, 128, 1024]);  addmm_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    add_164: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_160, view_568);  add_160 = view_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_50 = torch.ops.aten.var_mean.correction(add_164, [2], correction = 0, keepdim = True)
    getitem_100: "f32[1, 128, 1]" = var_mean_50[0]
    getitem_101: "f32[1, 128, 1]" = var_mean_50[1];  var_mean_50 = None
    add_165: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05);  getitem_100 = None
    rsqrt_50: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
    sub_79: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_164, getitem_101);  getitem_101 = None
    mul_133: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_50);  sub_79 = None
    mul_134: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_133, primals_415)
    add_166: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_134, primals_416);  mul_134 = primals_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_569: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_166, [128, 1024]);  add_166 = None
    permute_301: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_417, [1, 0]);  primals_417 = None
    addmm_156: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_418, view_569, permute_301);  primals_418 = None
    view_570: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_156, [1, 128, 1024]);  addmm_156 = None
    mul_135: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_570, 0.125);  view_570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_302: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_419, [1, 0]);  primals_419 = None
    addmm_157: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_420, view_267, permute_302);  primals_420 = None
    view_572: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_157, [1, 128, 1024]);  addmm_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_573: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_572, [1, -1, 16, 64]);  view_572 = None
    permute_303: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_573, [0, 2, 1, 3]);  view_573 = None
    clone_216: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_303, memory_format = torch.contiguous_format);  permute_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_304: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_421, [1, 0]);  primals_421 = None
    addmm_158: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_422, view_267, permute_304);  primals_422 = None
    view_575: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_158, [1, 128, 1024]);  addmm_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_576: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_575, [1, -1, 16, 64]);  view_575 = None
    permute_305: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_576, [0, 2, 1, 3]);  view_576 = None
    clone_217: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_305, memory_format = torch.contiguous_format);  permute_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_577: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_135, [1, 128, 16, 64]);  mul_135 = None
    permute_306: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_577, [0, 2, 1, 3]);  view_577 = None
    clone_218: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_306, memory_format = torch.contiguous_format);  permute_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_578: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_218, [16, -1, 64]);  clone_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_579: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_216, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_580: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_217, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_307: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_579, [0, 2, 1]);  view_579 = None
    bmm_58: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_578, permute_307)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_29: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_58, [-1], True)
    sub_80: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_58, amax_29)
    exp_29: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_80);  sub_80 = None
    sum_30: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_29, [-1], True)
    div_29: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_29, sum_30);  exp_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_59: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_29, view_580)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_581: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_59, [1, 16, 128, 64]);  bmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_308: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_581, [0, 2, 1, 3]);  view_581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_220: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_308, memory_format = torch.contiguous_format);  permute_308 = None
    view_582: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_220, [1, 128, 1024]);  clone_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_583: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_582, [128, 1024]);  view_582 = None
    permute_309: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_423, [1, 0]);  primals_423 = None
    addmm_159: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_424, view_583, permute_309);  primals_424 = None
    view_584: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_159, [1, 128, 1024]);  addmm_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    add_167: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_164, view_584);  add_164 = view_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_51 = torch.ops.aten.var_mean.correction(add_167, [2], correction = 0, keepdim = True)
    getitem_102: "f32[1, 128, 1]" = var_mean_51[0]
    getitem_103: "f32[1, 128, 1]" = var_mean_51[1];  var_mean_51 = None
    add_168: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
    rsqrt_51: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    sub_81: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_167, getitem_103);  getitem_103 = None
    mul_136: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_51);  sub_81 = None
    mul_137: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_136, primals_425)
    add_169: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_137, primals_426);  mul_137 = primals_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_585: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_169, [128, 1024]);  add_169 = None
    permute_310: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_427, [1, 0]);  primals_427 = None
    addmm_160: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_428, view_585, permute_310);  primals_428 = None
    view_586: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_160, [1, 128, 4096]);  addmm_160 = None
    relu_20: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_586);  view_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_587: "f32[128, 4096]" = torch.ops.aten.reshape.default(relu_20, [128, 4096])
    permute_311: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_429, [1, 0]);  primals_429 = None
    addmm_161: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_430, view_587, permute_311);  primals_430 = None
    view_588: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_161, [1, 128, 1024]);  addmm_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    add_170: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_167, view_588);  add_167 = view_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_52 = torch.ops.aten.var_mean.correction(add_170, [2], correction = 0, keepdim = True)
    getitem_104: "f32[1, 128, 1]" = var_mean_52[0]
    getitem_105: "f32[1, 128, 1]" = var_mean_52[1];  var_mean_52 = None
    add_171: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
    rsqrt_52: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
    sub_82: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_170, getitem_105);  getitem_105 = None
    mul_138: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_52);  sub_82 = None
    mul_139: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_138, primals_431)
    add_172: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_139, primals_432);  mul_139 = primals_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_589: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_172, [128, 1024]);  add_172 = None
    permute_312: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_433, [1, 0]);  primals_433 = None
    addmm_162: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_434, view_589, permute_312);  primals_434 = None
    view_590: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_162, [1, 128, 1024]);  addmm_162 = None
    mul_140: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_590, 0.125);  view_590 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_313: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_435, [1, 0]);  primals_435 = None
    addmm_163: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_436, view_589, permute_313);  primals_436 = None
    view_592: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_163, [1, 128, 1024]);  addmm_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_593: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_592, [1, -1, 16, 64]);  view_592 = None
    permute_314: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_593, [0, 2, 1, 3]);  view_593 = None
    clone_224: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_314, memory_format = torch.contiguous_format);  permute_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_315: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_437, [1, 0]);  primals_437 = None
    addmm_164: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_438, view_589, permute_315);  primals_438 = None
    view_595: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_164, [1, 128, 1024]);  addmm_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_596: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_595, [1, -1, 16, 64]);  view_595 = None
    permute_316: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_596, [0, 2, 1, 3]);  view_596 = None
    clone_225: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_316, memory_format = torch.contiguous_format);  permute_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_597: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_140, [1, 128, 16, 64]);  mul_140 = None
    permute_317: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_597, [0, 2, 1, 3]);  view_597 = None
    clone_226: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_317, memory_format = torch.contiguous_format);  permute_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_598: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_226, [16, -1, 64]);  clone_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_599: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_224, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_600: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_225, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_318: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_599, [0, 2, 1]);  view_599 = None
    bmm_60: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_598, permute_318)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_601: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_60, [1, 16, 128, 128]);  bmm_60 = None
    add_173: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_601, expand_1);  view_601 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_602: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(add_173, [16, 128, 128]);  add_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_30: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_602, [-1], True)
    sub_83: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_602, amax_30);  view_602 = amax_30 = None
    exp_30: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_83);  sub_83 = None
    sum_31: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_30, [-1], True)
    div_30: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_30, sum_31);  exp_30 = sum_31 = None
    alias_55: "f32[16, 128, 128]" = torch.ops.aten.alias.default(div_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_227: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_30);  div_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_61: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_227, view_600)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_603: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_61, [1, 16, 128, 64]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_319: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_603, [0, 2, 1, 3]);  view_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_228: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_319, memory_format = torch.contiguous_format);  permute_319 = None
    view_604: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_228, [1, 128, 1024]);  clone_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_605: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_604, [128, 1024]);  view_604 = None
    permute_320: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_439, [1, 0]);  primals_439 = None
    addmm_165: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_440, view_605, permute_320);  primals_440 = None
    view_606: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_165, [1, 128, 1024]);  addmm_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    add_174: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_170, view_606);  add_170 = view_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_53 = torch.ops.aten.var_mean.correction(add_174, [2], correction = 0, keepdim = True)
    getitem_106: "f32[1, 128, 1]" = var_mean_53[0]
    getitem_107: "f32[1, 128, 1]" = var_mean_53[1];  var_mean_53 = None
    add_175: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05);  getitem_106 = None
    rsqrt_53: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
    sub_84: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_174, getitem_107);  getitem_107 = None
    mul_141: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_53);  sub_84 = None
    mul_142: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_141, primals_441)
    add_176: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_142, primals_442);  mul_142 = primals_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_607: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_176, [128, 1024]);  add_176 = None
    permute_321: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_443, [1, 0]);  primals_443 = None
    addmm_166: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_444, view_607, permute_321);  primals_444 = None
    view_608: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_166, [1, 128, 1024]);  addmm_166 = None
    mul_143: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_608, 0.125);  view_608 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_322: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_445, [1, 0]);  primals_445 = None
    addmm_167: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_446, view_267, permute_322);  primals_446 = None
    view_610: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_167, [1, 128, 1024]);  addmm_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_611: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_610, [1, -1, 16, 64]);  view_610 = None
    permute_323: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_611, [0, 2, 1, 3]);  view_611 = None
    clone_230: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_323, memory_format = torch.contiguous_format);  permute_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_324: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_447, [1, 0]);  primals_447 = None
    addmm_168: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_448, view_267, permute_324);  primals_448 = None
    view_613: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_168, [1, 128, 1024]);  addmm_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_614: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_613, [1, -1, 16, 64]);  view_613 = None
    permute_325: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_614, [0, 2, 1, 3]);  view_614 = None
    clone_231: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_325, memory_format = torch.contiguous_format);  permute_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_615: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_143, [1, 128, 16, 64]);  mul_143 = None
    permute_326: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_615, [0, 2, 1, 3]);  view_615 = None
    clone_232: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_326, memory_format = torch.contiguous_format);  permute_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_616: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_232, [16, -1, 64]);  clone_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_617: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_230, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_618: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_231, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_327: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_617, [0, 2, 1]);  view_617 = None
    bmm_62: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_616, permute_327)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_31: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_62, [-1], True)
    sub_85: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_62, amax_31)
    exp_31: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_85);  sub_85 = None
    sum_32: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_31, [-1], True)
    div_31: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_31, sum_32);  exp_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_63: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_31, view_618)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_619: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_63, [1, 16, 128, 64]);  bmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_328: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_619, [0, 2, 1, 3]);  view_619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_234: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_328, memory_format = torch.contiguous_format);  permute_328 = None
    view_620: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_234, [1, 128, 1024]);  clone_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_621: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_620, [128, 1024]);  view_620 = None
    permute_329: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_449, [1, 0]);  primals_449 = None
    addmm_169: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_450, view_621, permute_329);  primals_450 = None
    view_622: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_169, [1, 128, 1024]);  addmm_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    add_177: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_174, view_622);  add_174 = view_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_54 = torch.ops.aten.var_mean.correction(add_177, [2], correction = 0, keepdim = True)
    getitem_108: "f32[1, 128, 1]" = var_mean_54[0]
    getitem_109: "f32[1, 128, 1]" = var_mean_54[1];  var_mean_54 = None
    add_178: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
    rsqrt_54: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
    sub_86: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_177, getitem_109);  getitem_109 = None
    mul_144: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_54);  sub_86 = None
    mul_145: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_144, primals_451)
    add_179: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_145, primals_452);  mul_145 = primals_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_623: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_179, [128, 1024]);  add_179 = None
    permute_330: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_453, [1, 0]);  primals_453 = None
    addmm_170: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_454, view_623, permute_330);  primals_454 = None
    view_624: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_170, [1, 128, 4096]);  addmm_170 = None
    relu_21: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_624);  view_624 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_625: "f32[128, 4096]" = torch.ops.aten.reshape.default(relu_21, [128, 4096])
    permute_331: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_455, [1, 0]);  primals_455 = None
    addmm_171: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_456, view_625, permute_331);  primals_456 = None
    view_626: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_171, [1, 128, 1024]);  addmm_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    add_180: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_177, view_626);  add_177 = view_626 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_55 = torch.ops.aten.var_mean.correction(add_180, [2], correction = 0, keepdim = True)
    getitem_110: "f32[1, 128, 1]" = var_mean_55[0]
    getitem_111: "f32[1, 128, 1]" = var_mean_55[1];  var_mean_55 = None
    add_181: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
    rsqrt_55: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
    sub_87: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_180, getitem_111);  getitem_111 = None
    mul_146: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_55);  sub_87 = None
    mul_147: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_146, primals_457)
    add_182: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_147, primals_458);  mul_147 = primals_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_627: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_182, [128, 1024]);  add_182 = None
    permute_332: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_459, [1, 0]);  primals_459 = None
    addmm_172: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_460, view_627, permute_332);  primals_460 = None
    view_628: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_172, [1, 128, 1024]);  addmm_172 = None
    mul_148: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_628, 0.125);  view_628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_333: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_461, [1, 0]);  primals_461 = None
    addmm_173: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_462, view_627, permute_333);  primals_462 = None
    view_630: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_173, [1, 128, 1024]);  addmm_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_631: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_630, [1, -1, 16, 64]);  view_630 = None
    permute_334: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_631, [0, 2, 1, 3]);  view_631 = None
    clone_238: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_334, memory_format = torch.contiguous_format);  permute_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_335: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_463, [1, 0]);  primals_463 = None
    addmm_174: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_464, view_627, permute_335);  primals_464 = None
    view_633: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_174, [1, 128, 1024]);  addmm_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_634: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_633, [1, -1, 16, 64]);  view_633 = None
    permute_336: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_634, [0, 2, 1, 3]);  view_634 = None
    clone_239: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_336, memory_format = torch.contiguous_format);  permute_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_635: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_148, [1, 128, 16, 64]);  mul_148 = None
    permute_337: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_635, [0, 2, 1, 3]);  view_635 = None
    clone_240: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_337, memory_format = torch.contiguous_format);  permute_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_636: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_240, [16, -1, 64]);  clone_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_637: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_238, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_638: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_239, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_338: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_637, [0, 2, 1]);  view_637 = None
    bmm_64: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_636, permute_338)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_639: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_64, [1, 16, 128, 128]);  bmm_64 = None
    add_183: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_639, expand_1);  view_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_640: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(add_183, [16, 128, 128]);  add_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_32: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_640, [-1], True)
    sub_88: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_640, amax_32);  view_640 = amax_32 = None
    exp_32: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_88);  sub_88 = None
    sum_33: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_32, [-1], True)
    div_32: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_32, sum_33);  exp_32 = sum_33 = None
    alias_58: "f32[16, 128, 128]" = torch.ops.aten.alias.default(div_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_241: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_32);  div_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_65: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_241, view_638)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_641: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_65, [1, 16, 128, 64]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_339: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_641, [0, 2, 1, 3]);  view_641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_242: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_339, memory_format = torch.contiguous_format);  permute_339 = None
    view_642: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_242, [1, 128, 1024]);  clone_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_643: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_642, [128, 1024]);  view_642 = None
    permute_340: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_465, [1, 0]);  primals_465 = None
    addmm_175: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_466, view_643, permute_340);  primals_466 = None
    view_644: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_175, [1, 128, 1024]);  addmm_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    add_184: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_180, view_644);  add_180 = view_644 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_56 = torch.ops.aten.var_mean.correction(add_184, [2], correction = 0, keepdim = True)
    getitem_112: "f32[1, 128, 1]" = var_mean_56[0]
    getitem_113: "f32[1, 128, 1]" = var_mean_56[1];  var_mean_56 = None
    add_185: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
    rsqrt_56: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
    sub_89: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_184, getitem_113);  getitem_113 = None
    mul_149: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_56);  sub_89 = None
    mul_150: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_149, primals_467)
    add_186: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_150, primals_468);  mul_150 = primals_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_645: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_186, [128, 1024]);  add_186 = None
    permute_341: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_469, [1, 0]);  primals_469 = None
    addmm_176: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_470, view_645, permute_341);  primals_470 = None
    view_646: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_176, [1, 128, 1024]);  addmm_176 = None
    mul_151: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_646, 0.125);  view_646 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_342: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_471, [1, 0]);  primals_471 = None
    addmm_177: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_472, view_267, permute_342);  primals_472 = None
    view_648: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_177, [1, 128, 1024]);  addmm_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_649: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_648, [1, -1, 16, 64]);  view_648 = None
    permute_343: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_649, [0, 2, 1, 3]);  view_649 = None
    clone_244: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_343, memory_format = torch.contiguous_format);  permute_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_344: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_473, [1, 0]);  primals_473 = None
    addmm_178: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_474, view_267, permute_344);  primals_474 = None
    view_651: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_178, [1, 128, 1024]);  addmm_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_652: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_651, [1, -1, 16, 64]);  view_651 = None
    permute_345: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_652, [0, 2, 1, 3]);  view_652 = None
    clone_245: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_345, memory_format = torch.contiguous_format);  permute_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_653: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_151, [1, 128, 16, 64]);  mul_151 = None
    permute_346: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_653, [0, 2, 1, 3]);  view_653 = None
    clone_246: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_346, memory_format = torch.contiguous_format);  permute_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_654: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_246, [16, -1, 64]);  clone_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_655: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_244, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_656: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_245, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_347: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_655, [0, 2, 1]);  view_655 = None
    bmm_66: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_654, permute_347)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_33: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_66, [-1], True)
    sub_90: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_66, amax_33)
    exp_33: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_90);  sub_90 = None
    sum_34: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_33, [-1], True)
    div_33: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_33, sum_34);  exp_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_67: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_33, view_656)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_657: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_67, [1, 16, 128, 64]);  bmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_348: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_657, [0, 2, 1, 3]);  view_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_248: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_348, memory_format = torch.contiguous_format);  permute_348 = None
    view_658: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_248, [1, 128, 1024]);  clone_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_659: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_658, [128, 1024]);  view_658 = None
    permute_349: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_475, [1, 0]);  primals_475 = None
    addmm_179: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_476, view_659, permute_349);  primals_476 = None
    view_660: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_179, [1, 128, 1024]);  addmm_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    add_187: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_184, view_660);  add_184 = view_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_57 = torch.ops.aten.var_mean.correction(add_187, [2], correction = 0, keepdim = True)
    getitem_114: "f32[1, 128, 1]" = var_mean_57[0]
    getitem_115: "f32[1, 128, 1]" = var_mean_57[1];  var_mean_57 = None
    add_188: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-05);  getitem_114 = None
    rsqrt_57: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
    sub_91: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_187, getitem_115);  getitem_115 = None
    mul_152: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_57);  sub_91 = None
    mul_153: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_152, primals_477)
    add_189: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_153, primals_478);  mul_153 = primals_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_661: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_189, [128, 1024]);  add_189 = None
    permute_350: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_479, [1, 0]);  primals_479 = None
    addmm_180: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_480, view_661, permute_350);  primals_480 = None
    view_662: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_180, [1, 128, 4096]);  addmm_180 = None
    relu_22: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_662);  view_662 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_663: "f32[128, 4096]" = torch.ops.aten.reshape.default(relu_22, [128, 4096])
    permute_351: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_481, [1, 0]);  primals_481 = None
    addmm_181: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_482, view_663, permute_351);  primals_482 = None
    view_664: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_181, [1, 128, 1024]);  addmm_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    add_190: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_187, view_664);  add_187 = view_664 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_58 = torch.ops.aten.var_mean.correction(add_190, [2], correction = 0, keepdim = True)
    getitem_116: "f32[1, 128, 1]" = var_mean_58[0]
    getitem_117: "f32[1, 128, 1]" = var_mean_58[1];  var_mean_58 = None
    add_191: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-05);  getitem_116 = None
    rsqrt_58: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
    sub_92: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_190, getitem_117);  getitem_117 = None
    mul_154: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_58);  sub_92 = None
    mul_155: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_154, primals_483)
    add_192: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_155, primals_484);  mul_155 = primals_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_665: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_192, [128, 1024]);  add_192 = None
    permute_352: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_485, [1, 0]);  primals_485 = None
    addmm_182: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_486, view_665, permute_352);  primals_486 = None
    view_666: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_182, [1, 128, 1024]);  addmm_182 = None
    mul_156: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_666, 0.125);  view_666 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_353: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_487, [1, 0]);  primals_487 = None
    addmm_183: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_488, view_665, permute_353);  primals_488 = None
    view_668: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_183, [1, 128, 1024]);  addmm_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_669: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_668, [1, -1, 16, 64]);  view_668 = None
    permute_354: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_669, [0, 2, 1, 3]);  view_669 = None
    clone_252: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_354, memory_format = torch.contiguous_format);  permute_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_355: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_489, [1, 0]);  primals_489 = None
    addmm_184: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_490, view_665, permute_355);  primals_490 = None
    view_671: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_184, [1, 128, 1024]);  addmm_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_672: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_671, [1, -1, 16, 64]);  view_671 = None
    permute_356: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_672, [0, 2, 1, 3]);  view_672 = None
    clone_253: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_356, memory_format = torch.contiguous_format);  permute_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_673: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_156, [1, 128, 16, 64]);  mul_156 = None
    permute_357: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_673, [0, 2, 1, 3]);  view_673 = None
    clone_254: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_357, memory_format = torch.contiguous_format);  permute_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_674: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_254, [16, -1, 64]);  clone_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_675: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_252, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_676: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_253, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_358: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_675, [0, 2, 1]);  view_675 = None
    bmm_68: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_674, permute_358)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_677: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_68, [1, 16, 128, 128]);  bmm_68 = None
    add_193: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_677, expand_1);  view_677 = expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_678: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(add_193, [16, 128, 128]);  add_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_34: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_678, [-1], True)
    sub_93: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_678, amax_34);  view_678 = amax_34 = None
    exp_34: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_93);  sub_93 = None
    sum_35: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_34, [-1], True)
    div_34: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_34, sum_35);  exp_34 = sum_35 = None
    alias_61: "f32[16, 128, 128]" = torch.ops.aten.alias.default(div_34)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:329, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_255: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_34);  div_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_69: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_255, view_676)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_679: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_69, [1, 16, 128, 64]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_359: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_679, [0, 2, 1, 3]);  view_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_256: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_359, memory_format = torch.contiguous_format);  permute_359 = None
    view_680: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_256, [1, 128, 1024]);  clone_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_681: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_680, [128, 1024]);  view_680 = None
    permute_360: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_491, [1, 0]);  primals_491 = None
    addmm_185: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_492, view_681, permute_360);  primals_492 = None
    view_682: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_185, [1, 128, 1024]);  addmm_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:493, code: hidden_states = residual + hidden_states
    add_194: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_190, view_682);  add_190 = view_682 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_59 = torch.ops.aten.var_mean.correction(add_194, [2], correction = 0, keepdim = True)
    getitem_118: "f32[1, 128, 1]" = var_mean_59[0]
    getitem_119: "f32[1, 128, 1]" = var_mean_59[1];  var_mean_59 = None
    add_195: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05);  getitem_118 = None
    rsqrt_59: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_195);  add_195 = None
    sub_94: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_194, getitem_119);  getitem_119 = None
    mul_157: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_59);  sub_94 = None
    mul_158: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_157, primals_493)
    add_196: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_158, primals_494);  mul_158 = primals_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_683: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_196, [128, 1024]);  add_196 = None
    permute_361: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_495, [1, 0]);  primals_495 = None
    addmm_186: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_496, view_683, permute_361);  primals_496 = None
    view_684: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_186, [1, 128, 1024]);  addmm_186 = None
    mul_159: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_684, 0.125);  view_684 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_362: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_497, [1, 0]);  primals_497 = None
    addmm_187: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_498, view_267, permute_362);  primals_498 = None
    view_686: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_187, [1, 128, 1024]);  addmm_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_687: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_686, [1, -1, 16, 64]);  view_686 = None
    permute_363: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_687, [0, 2, 1, 3]);  view_687 = None
    clone_258: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_363, memory_format = torch.contiguous_format);  permute_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_364: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_499, [1, 0]);  primals_499 = None
    addmm_188: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_500, view_267, permute_364);  primals_500 = None
    view_689: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_188, [1, 128, 1024]);  addmm_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_690: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_689, [1, -1, 16, 64]);  view_689 = None
    permute_365: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_690, [0, 2, 1, 3]);  view_690 = None
    clone_259: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_365, memory_format = torch.contiguous_format);  permute_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_691: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_159, [1, 128, 16, 64]);  mul_159 = None
    permute_366: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_691, [0, 2, 1, 3]);  view_691 = None
    clone_260: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_366, memory_format = torch.contiguous_format);  permute_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_692: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_260, [16, -1, 64]);  clone_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_693: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_258, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_694: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(clone_259, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_367: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_693, [0, 2, 1]);  view_693 = None
    bmm_70: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_692, permute_367)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_35: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_70, [-1], True)
    sub_95: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_70, amax_35)
    exp_35: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_95);  sub_95 = None
    sum_36: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_35, [-1], True)
    div_35: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_35, sum_36);  exp_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_71: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(div_35, view_694)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_695: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_71, [1, 16, 128, 64]);  bmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_368: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_695, [0, 2, 1, 3]);  view_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_262: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_368, memory_format = torch.contiguous_format);  permute_368 = None
    view_696: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_262, [1, 128, 1024]);  clone_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_697: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_696, [128, 1024]);  view_696 = None
    permute_369: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_501, [1, 0]);  primals_501 = None
    addmm_189: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_502, view_697, permute_369);  primals_502 = None
    view_698: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_189, [1, 128, 1024]);  addmm_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:513, code: hidden_states = residual + hidden_states
    add_197: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_194, view_698);  add_194 = view_698 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_60 = torch.ops.aten.var_mean.correction(add_197, [2], correction = 0, keepdim = True)
    getitem_120: "f32[1, 128, 1]" = var_mean_60[0]
    getitem_121: "f32[1, 128, 1]" = var_mean_60[1];  var_mean_60 = None
    add_198: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-05);  getitem_120 = None
    rsqrt_60: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_198);  add_198 = None
    sub_96: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_197, getitem_121);  getitem_121 = None
    mul_160: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_60);  sub_96 = None
    mul_161: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_160, primals_503)
    add_199: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_161, primals_504);  mul_161 = primals_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_699: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_199, [128, 1024]);  add_199 = None
    permute_370: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_505, [1, 0]);  primals_505 = None
    addmm_190: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_506, view_699, permute_370);  primals_506 = None
    view_700: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_190, [1, 128, 4096]);  addmm_190 = None
    relu_23: "f32[1, 128, 4096]" = torch.ops.aten.relu.default(view_700);  view_700 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_701: "f32[128, 4096]" = torch.ops.aten.reshape.default(relu_23, [128, 4096])
    permute_371: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_507, [1, 0]);  primals_507 = None
    addmm_191: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_508, view_701, permute_371);  primals_508 = None
    view_702: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(addmm_191, [1, 128, 1024]);  addmm_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:525, code: hidden_states = residual + hidden_states
    add_200: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_197, view_702);  add_197 = view_702 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1114, code: hidden_states = self.layer_norm(hidden_states)
    var_mean_61 = torch.ops.aten.var_mean.correction(add_200, [2], correction = 0, keepdim = True)
    getitem_122: "f32[1, 128, 1]" = var_mean_61[0]
    getitem_123: "f32[1, 128, 1]" = var_mean_61[1];  var_mean_61 = None
    add_201: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-05);  getitem_122 = None
    rsqrt_61: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_201);  add_201 = None
    sub_97: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_200, getitem_123);  add_200 = getitem_123 = None
    mul_162: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_97, rsqrt_61);  sub_97 = None
    mul_163: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_162, primals_509)
    add_202: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_163, primals_510);  mul_163 = primals_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1331, code: lm_logits = self.lm_head(outputs[0])
    permute_372: "f32[1024, 128112]" = torch.ops.aten.permute.default(primals_511, [1, 0]);  primals_511 = None
    view_703: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_202, [128, 1024]);  add_202 = None
    mm: "f32[128, 128112]" = torch.ops.aten.mm.default(view_703, permute_372)
    view_704: "f32[1, 128, 128112]" = torch.ops.aten.reshape.default(mm, [1, 128, 128112]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1338, code: masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
    view_705: "f32[128, 128112]" = torch.ops.aten.reshape.default(view_704, [-1, 128112])
    view_706: "i64[128]" = torch.ops.aten.reshape.default(primals_514, [-1])
    amax_36: "f32[128, 1]" = torch.ops.aten.amax.default(view_705, [1], True)
    sub_98: "f32[128, 128112]" = torch.ops.aten.sub.Tensor(view_705, amax_36);  view_705 = amax_36 = None
    exp_36: "f32[128, 128112]" = torch.ops.aten.exp.default(sub_98)
    sum_37: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(exp_36, [1], True);  exp_36 = None
    log: "f32[128, 1]" = torch.ops.aten.log.default(sum_37);  sum_37 = None
    sub_99: "f32[128, 128112]" = torch.ops.aten.sub.Tensor(sub_98, log);  sub_98 = log = None
    ne_2: "b8[128]" = torch.ops.aten.ne.Scalar(view_706, -100)
    full_default_2: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_1: "i64[128]" = torch.ops.aten.where.self(ne_2, view_706, full_default_2);  view_706 = full_default_2 = None
    unsqueeze_4: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
    gather: "f32[128, 1]" = torch.ops.aten.gather.default(sub_99, 1, unsqueeze_4);  unsqueeze_4 = None
    squeeze: "f32[128]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[128]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    where_2: "f32[128]" = torch.ops.aten.where.self(ne_2, neg, full_default_1);  neg = full_default_1 = None
    sum_38: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type_6: "f32[]" = torch.ops.prims.convert_element_type.default(sum_38, torch.float32);  sum_38 = None
    sum_39: "f32[]" = torch.ops.aten.sum.default(where_2);  where_2 = None
    div_36: "f32[]" = torch.ops.aten.div.Tensor(sum_39, convert_element_type_6);  sum_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1331, code: lm_logits = self.lm_head(outputs[0])
    permute_375: "f32[128112, 1024]" = torch.ops.aten.permute.default(permute_372, [1, 0]);  permute_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1114, code: hidden_states = self.layer_norm(hidden_states)
    div_38: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_61, 1024);  rsqrt_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    permute_377: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_371, [1, 0]);  permute_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    le: "b8[1, 128, 4096]" = torch.ops.aten.le.Scalar(relu_23, 0);  relu_23 = None
    permute_381: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_370, [1, 0]);  permute_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    div_39: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_60, 1024);  rsqrt_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_385: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_369, [1, 0]);  permute_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_390: "f32[16, 128, 128]" = torch.ops.aten.permute.default(div_35, [0, 2, 1]);  div_35 = None
    permute_391: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_694, [0, 2, 1]);  view_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_392: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_692, [0, 2, 1]);  view_692 = None
    permute_393: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_367, [0, 2, 1]);  permute_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_397: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_364, [1, 0]);  permute_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_402: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_362, [1, 0]);  permute_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_406: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_361, [1, 0]);  permute_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    div_40: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_59, 1024);  rsqrt_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_410: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_360, [1, 0]);  permute_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_415: "f32[16, 128, 128]" = torch.ops.aten.permute.default(clone_255, [0, 2, 1]);  clone_255 = None
    permute_416: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_676, [0, 2, 1]);  view_676 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_68: "f32[16, 128, 128]" = torch.ops.aten.alias.default(alias_61);  alias_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_417: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_674, [0, 2, 1]);  view_674 = None
    permute_418: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_358, [0, 2, 1]);  permute_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_422: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_355, [1, 0]);  permute_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_427: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_353, [1, 0]);  permute_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_431: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_352, [1, 0]);  permute_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_41: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_58, 1024);  rsqrt_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    permute_435: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_351, [1, 0]);  permute_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    le_1: "b8[1, 128, 4096]" = torch.ops.aten.le.Scalar(relu_22, 0);  relu_22 = None
    permute_439: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_350, [1, 0]);  permute_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    div_42: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_57, 1024);  rsqrt_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_443: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_349, [1, 0]);  permute_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_448: "f32[16, 128, 128]" = torch.ops.aten.permute.default(div_33, [0, 2, 1]);  div_33 = None
    permute_449: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_656, [0, 2, 1]);  view_656 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_450: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_654, [0, 2, 1]);  view_654 = None
    permute_451: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_347, [0, 2, 1]);  permute_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_455: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_344, [1, 0]);  permute_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_460: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_342, [1, 0]);  permute_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_464: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_341, [1, 0]);  permute_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    div_43: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_56, 1024);  rsqrt_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_468: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_340, [1, 0]);  permute_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_473: "f32[16, 128, 128]" = torch.ops.aten.permute.default(clone_241, [0, 2, 1]);  clone_241 = None
    permute_474: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_638, [0, 2, 1]);  view_638 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_71: "f32[16, 128, 128]" = torch.ops.aten.alias.default(alias_58);  alias_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_475: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_636, [0, 2, 1]);  view_636 = None
    permute_476: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_338, [0, 2, 1]);  permute_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_480: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_335, [1, 0]);  permute_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_485: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_333, [1, 0]);  permute_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_489: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_332, [1, 0]);  permute_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_44: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_55, 1024);  rsqrt_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    permute_493: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_331, [1, 0]);  permute_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    le_2: "b8[1, 128, 4096]" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
    permute_497: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_330, [1, 0]);  permute_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    div_45: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_54, 1024);  rsqrt_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_501: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_329, [1, 0]);  permute_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_506: "f32[16, 128, 128]" = torch.ops.aten.permute.default(div_31, [0, 2, 1]);  div_31 = None
    permute_507: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_618, [0, 2, 1]);  view_618 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_508: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_616, [0, 2, 1]);  view_616 = None
    permute_509: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_327, [0, 2, 1]);  permute_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_513: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_324, [1, 0]);  permute_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_518: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_322, [1, 0]);  permute_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_522: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_321, [1, 0]);  permute_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    div_46: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_53, 1024);  rsqrt_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_526: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_320, [1, 0]);  permute_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_531: "f32[16, 128, 128]" = torch.ops.aten.permute.default(clone_227, [0, 2, 1]);  clone_227 = None
    permute_532: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_600, [0, 2, 1]);  view_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_74: "f32[16, 128, 128]" = torch.ops.aten.alias.default(alias_55);  alias_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_533: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_598, [0, 2, 1]);  view_598 = None
    permute_534: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_318, [0, 2, 1]);  permute_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_538: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_315, [1, 0]);  permute_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_543: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_313, [1, 0]);  permute_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_547: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_312, [1, 0]);  permute_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_47: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_52, 1024);  rsqrt_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    permute_551: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_311, [1, 0]);  permute_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    le_3: "b8[1, 128, 4096]" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
    permute_555: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_310, [1, 0]);  permute_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    div_48: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_51, 1024);  rsqrt_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_559: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_309, [1, 0]);  permute_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_564: "f32[16, 128, 128]" = torch.ops.aten.permute.default(div_29, [0, 2, 1]);  div_29 = None
    permute_565: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_580, [0, 2, 1]);  view_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_566: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_578, [0, 2, 1]);  view_578 = None
    permute_567: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_307, [0, 2, 1]);  permute_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_571: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_304, [1, 0]);  permute_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_576: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_302, [1, 0]);  permute_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_580: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_301, [1, 0]);  permute_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    div_49: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_50, 1024);  rsqrt_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_584: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_300, [1, 0]);  permute_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_589: "f32[16, 128, 128]" = torch.ops.aten.permute.default(clone_213, [0, 2, 1]);  clone_213 = None
    permute_590: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_562, [0, 2, 1]);  view_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_77: "f32[16, 128, 128]" = torch.ops.aten.alias.default(alias_52);  alias_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_591: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_560, [0, 2, 1]);  view_560 = None
    permute_592: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_298, [0, 2, 1]);  permute_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_596: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_295, [1, 0]);  permute_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_601: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_293, [1, 0]);  permute_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_605: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_292, [1, 0]);  permute_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_50: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_49, 1024);  rsqrt_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    permute_609: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_291, [1, 0]);  permute_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    le_4: "b8[1, 128, 4096]" = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
    permute_613: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_290, [1, 0]);  permute_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    div_51: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 1024);  rsqrt_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_617: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_289, [1, 0]);  permute_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_622: "f32[16, 128, 128]" = torch.ops.aten.permute.default(div_27, [0, 2, 1]);  div_27 = None
    permute_623: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_542, [0, 2, 1]);  view_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_624: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_540, [0, 2, 1]);  view_540 = None
    permute_625: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_287, [0, 2, 1]);  permute_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_629: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_284, [1, 0]);  permute_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_634: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_282, [1, 0]);  permute_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_638: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_281, [1, 0]);  permute_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    div_52: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 1024);  rsqrt_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_642: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_280, [1, 0]);  permute_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_647: "f32[16, 128, 128]" = torch.ops.aten.permute.default(clone_199, [0, 2, 1]);  clone_199 = None
    permute_648: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_524, [0, 2, 1]);  view_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_80: "f32[16, 128, 128]" = torch.ops.aten.alias.default(alias_49);  alias_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_649: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_522, [0, 2, 1]);  view_522 = None
    permute_650: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_278, [0, 2, 1]);  permute_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_654: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_275, [1, 0]);  permute_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_659: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_273, [1, 0]);  permute_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_663: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_272, [1, 0]);  permute_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_53: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 1024);  rsqrt_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    permute_667: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_271, [1, 0]);  permute_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    le_5: "b8[1, 128, 4096]" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
    permute_671: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_270, [1, 0]);  permute_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    div_54: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 1024);  rsqrt_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_675: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_269, [1, 0]);  permute_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_680: "f32[16, 128, 128]" = torch.ops.aten.permute.default(div_25, [0, 2, 1]);  div_25 = None
    permute_681: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_504, [0, 2, 1]);  view_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_682: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_502, [0, 2, 1]);  view_502 = None
    permute_683: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_267, [0, 2, 1]);  permute_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_687: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_692: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_696: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_261, [1, 0]);  permute_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    div_55: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 1024);  rsqrt_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_700: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_260, [1, 0]);  permute_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_705: "f32[16, 128, 128]" = torch.ops.aten.permute.default(clone_185, [0, 2, 1]);  clone_185 = None
    permute_706: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_486, [0, 2, 1]);  view_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_83: "f32[16, 128, 128]" = torch.ops.aten.alias.default(alias_46);  alias_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_707: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_484, [0, 2, 1]);  view_484 = None
    permute_708: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_258, [0, 2, 1]);  permute_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_712: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_255, [1, 0]);  permute_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_717: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_721: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_56: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 1024);  rsqrt_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    permute_725: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    le_6: "b8[1, 128, 4096]" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
    permute_729: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_250, [1, 0]);  permute_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    div_57: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 1024);  rsqrt_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_733: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_249, [1, 0]);  permute_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_738: "f32[16, 128, 128]" = torch.ops.aten.permute.default(div_23, [0, 2, 1]);  div_23 = None
    permute_739: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_466, [0, 2, 1]);  view_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_740: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_464, [0, 2, 1]);  view_464 = None
    permute_741: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_247, [0, 2, 1]);  permute_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_745: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_750: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_754: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_241, [1, 0]);  permute_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    div_58: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 1024);  rsqrt_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_758: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_763: "f32[16, 128, 128]" = torch.ops.aten.permute.default(clone_171, [0, 2, 1]);  clone_171 = None
    permute_764: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_448, [0, 2, 1]);  view_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_86: "f32[16, 128, 128]" = torch.ops.aten.alias.default(alias_43);  alias_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_765: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_446, [0, 2, 1]);  view_446 = None
    permute_766: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_238, [0, 2, 1]);  permute_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_770: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_235, [1, 0]);  permute_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_775: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_779: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_59: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 1024);  rsqrt_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    permute_783: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    le_7: "b8[1, 128, 4096]" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
    permute_787: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    div_60: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 1024);  rsqrt_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_791: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_796: "f32[16, 128, 128]" = torch.ops.aten.permute.default(div_21, [0, 2, 1]);  div_21 = None
    permute_797: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_428, [0, 2, 1]);  view_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_798: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_426, [0, 2, 1]);  view_426 = None
    permute_799: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_227, [0, 2, 1]);  permute_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_803: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_224, [1, 0]);  permute_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_808: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_812: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_221, [1, 0]);  permute_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    div_61: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 1024);  rsqrt_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_816: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_821: "f32[16, 128, 128]" = torch.ops.aten.permute.default(clone_157, [0, 2, 1]);  clone_157 = None
    permute_822: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_410, [0, 2, 1]);  view_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_89: "f32[16, 128, 128]" = torch.ops.aten.alias.default(alias_40);  alias_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_823: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_408, [0, 2, 1]);  view_408 = None
    permute_824: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_218, [0, 2, 1]);  permute_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_828: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_215, [1, 0]);  permute_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_833: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_213, [1, 0]);  permute_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_837: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_212, [1, 0]);  permute_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_62: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 1024);  rsqrt_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    permute_841: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    le_8: "b8[1, 128, 4096]" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
    permute_845: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    div_63: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 1024);  rsqrt_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_849: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_209, [1, 0]);  permute_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_854: "f32[16, 128, 128]" = torch.ops.aten.permute.default(div_19, [0, 2, 1]);  div_19 = None
    permute_855: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_390, [0, 2, 1]);  view_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_856: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_388, [0, 2, 1]);  view_388 = None
    permute_857: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_207, [0, 2, 1]);  permute_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_861: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_204, [1, 0]);  permute_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_866: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_870: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    div_64: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 1024);  rsqrt_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_874: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_879: "f32[16, 128, 128]" = torch.ops.aten.permute.default(clone_143, [0, 2, 1]);  clone_143 = None
    permute_880: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_372, [0, 2, 1]);  view_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_92: "f32[16, 128, 128]" = torch.ops.aten.alias.default(alias_37);  alias_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_881: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_370, [0, 2, 1]);  view_370 = None
    permute_882: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_198, [0, 2, 1]);  permute_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_886: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_195, [1, 0]);  permute_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_891: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_193, [1, 0]);  permute_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_895: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_192, [1, 0]);  permute_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_65: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 1024);  rsqrt_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    permute_899: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_191, [1, 0]);  permute_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    le_9: "b8[1, 128, 4096]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
    permute_903: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_190, [1, 0]);  permute_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    div_66: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 1024);  rsqrt_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_907: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_189, [1, 0]);  permute_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_912: "f32[16, 128, 128]" = torch.ops.aten.permute.default(div_17, [0, 2, 1]);  div_17 = None
    permute_913: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_352, [0, 2, 1]);  view_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_914: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_350, [0, 2, 1]);  view_350 = None
    permute_915: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_187, [0, 2, 1]);  permute_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_919: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_924: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_182, [1, 0]);  permute_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_928: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    div_67: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 1024);  rsqrt_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_932: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_180, [1, 0]);  permute_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_937: "f32[16, 128, 128]" = torch.ops.aten.permute.default(clone_129, [0, 2, 1]);  clone_129 = None
    permute_938: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_334, [0, 2, 1]);  view_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_95: "f32[16, 128, 128]" = torch.ops.aten.alias.default(alias_34);  alias_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_939: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_332, [0, 2, 1]);  view_332 = None
    permute_940: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_178, [0, 2, 1]);  permute_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_944: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_949: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_953: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_68: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 1024);  rsqrt_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    permute_957: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_171, [1, 0]);  permute_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    le_10: "b8[1, 128, 4096]" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
    permute_961: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    div_69: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 1024);  rsqrt_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_965: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_970: "f32[16, 128, 128]" = torch.ops.aten.permute.default(div_15, [0, 2, 1]);  div_15 = None
    permute_971: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_314, [0, 2, 1]);  view_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_972: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_312, [0, 2, 1]);  view_312 = None
    permute_973: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_167, [0, 2, 1]);  permute_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_977: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_982: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_986: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    div_70: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 1024);  rsqrt_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_990: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_160, [1, 0]);  permute_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_995: "f32[16, 128, 128]" = torch.ops.aten.permute.default(clone_115, [0, 2, 1]);  clone_115 = None
    permute_996: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_296, [0, 2, 1]);  view_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_98: "f32[16, 128, 128]" = torch.ops.aten.alias.default(alias_31);  alias_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_997: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_294, [0, 2, 1]);  view_294 = None
    permute_998: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_158, [0, 2, 1]);  permute_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_1002: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_155, [1, 0]);  permute_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_1007: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_1011: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_71: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 1024);  rsqrt_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    permute_1015: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    le_11: "b8[1, 128, 4096]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
    permute_1019: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_150, [1, 0]);  permute_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    div_72: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 1024);  rsqrt_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_1023: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_149, [1, 0]);  permute_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_1028: "f32[16, 128, 128]" = torch.ops.aten.permute.default(div_13, [0, 2, 1]);  div_13 = None
    permute_1029: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_276, [0, 2, 1]);  view_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_1030: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_274, [0, 2, 1]);  view_274 = None
    permute_1031: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_147, [0, 2, 1]);  permute_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_1035: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_1040: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_1044: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    div_73: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 1024);  rsqrt_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_1048: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_1053: "f32[16, 128, 128]" = torch.ops.aten.permute.default(clone_101, [0, 2, 1]);  clone_101 = None
    permute_1054: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_258, [0, 2, 1]);  view_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_101: "f32[16, 128, 128]" = torch.ops.aten.alias.default(alias_28);  alias_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_1055: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_256, [0, 2, 1]);  view_256 = None
    permute_1056: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_138, [0, 2, 1]);  permute_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_1060: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_135, [1, 0]);  permute_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_1065: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_1069: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_74: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 1024);  rsqrt_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:852, code: hidden_states = self.layer_norm(hidden_states)
    div_75: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 1024);  rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    permute_1073: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    le_12: "b8[1, 128, 4096]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
    permute_1077: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    div_76: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 1024);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_1081: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_1086: "f32[16, 128, 128]" = torch.ops.aten.permute.default(div_11, [0, 2, 1]);  div_11 = None
    permute_1087: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_234, [0, 2, 1]);  view_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_1088: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
    permute_1089: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_127, [0, 2, 1]);  permute_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_1093: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_1098: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_1102: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_77: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 1024);  rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    permute_1106: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    le_13: "b8[1, 128, 4096]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
    permute_1110: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    div_78: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 1024);  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_1114: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_1119: "f32[16, 128, 128]" = torch.ops.aten.permute.default(div_10, [0, 2, 1]);  div_10 = None
    permute_1120: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_214, [0, 2, 1]);  view_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_1121: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_212, [0, 2, 1]);  view_212 = None
    permute_1122: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_116, [0, 2, 1]);  permute_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_1126: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_1131: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_1135: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_79: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 1024);  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    permute_1139: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    le_14: "b8[1, 128, 4096]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
    permute_1143: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    div_80: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 1024);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_1147: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_1152: "f32[16, 128, 128]" = torch.ops.aten.permute.default(div_9, [0, 2, 1]);  div_9 = None
    permute_1153: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_194, [0, 2, 1]);  view_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_1154: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_192, [0, 2, 1]);  view_192 = None
    permute_1155: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_105, [0, 2, 1]);  permute_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_1159: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_1164: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_1168: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_81: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 1024);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    permute_1172: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    le_15: "b8[1, 128, 4096]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
    permute_1176: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    div_82: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 1024);  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_1180: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_1185: "f32[16, 128, 128]" = torch.ops.aten.permute.default(div_8, [0, 2, 1]);  div_8 = None
    permute_1186: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_174, [0, 2, 1]);  view_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_1187: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_172, [0, 2, 1]);  view_172 = None
    permute_1188: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_94, [0, 2, 1]);  permute_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_1192: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_1197: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_1201: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_83: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 1024);  rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    permute_1205: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    le_16: "b8[1, 128, 4096]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
    permute_1209: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    div_84: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 1024);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_1213: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_1218: "f32[16, 128, 128]" = torch.ops.aten.permute.default(div_7, [0, 2, 1]);  div_7 = None
    permute_1219: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_154, [0, 2, 1]);  view_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_1220: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_152, [0, 2, 1]);  view_152 = None
    permute_1221: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_83, [0, 2, 1]);  permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_1225: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_1230: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_1234: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_85: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 1024);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    permute_1238: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    le_17: "b8[1, 128, 4096]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
    permute_1242: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    div_86: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 1024);  rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_1246: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_1251: "f32[16, 128, 128]" = torch.ops.aten.permute.default(div_6, [0, 2, 1]);  div_6 = None
    permute_1252: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_134, [0, 2, 1]);  view_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_1253: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_132, [0, 2, 1]);  view_132 = None
    permute_1254: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_72, [0, 2, 1]);  permute_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_1258: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_1263: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_1267: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_87: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 1024);  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    permute_1271: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    le_18: "b8[1, 128, 4096]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
    permute_1275: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    div_88: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 1024);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_1279: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_1284: "f32[16, 128, 128]" = torch.ops.aten.permute.default(div_5, [0, 2, 1]);  div_5 = None
    permute_1285: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_114, [0, 2, 1]);  view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_1286: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_112, [0, 2, 1]);  view_112 = None
    permute_1287: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_61, [0, 2, 1]);  permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_1291: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_1296: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_1300: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_89: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 1024);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    permute_1304: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    le_19: "b8[1, 128, 4096]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
    permute_1308: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    div_90: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 1024);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_1312: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_1317: "f32[16, 128, 128]" = torch.ops.aten.permute.default(div_4, [0, 2, 1]);  div_4 = None
    permute_1318: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_94, [0, 2, 1]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_1319: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_92, [0, 2, 1]);  view_92 = None
    permute_1320: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_50, [0, 2, 1]);  permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_1324: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_1329: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_1333: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_91: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 1024);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    permute_1337: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    le_20: "b8[1, 128, 4096]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
    permute_1341: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    div_92: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 1024);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_1345: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_1350: "f32[16, 128, 128]" = torch.ops.aten.permute.default(div_3, [0, 2, 1]);  div_3 = None
    permute_1351: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_74, [0, 2, 1]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_1352: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_72, [0, 2, 1]);  view_72 = None
    permute_1353: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_39, [0, 2, 1]);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_1357: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_1362: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_1366: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_93: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 1024);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    permute_1370: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    le_21: "b8[1, 128, 4096]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
    permute_1374: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    div_94: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 1024);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_1378: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_1383: "f32[16, 128, 128]" = torch.ops.aten.permute.default(div_2, [0, 2, 1]);  div_2 = None
    permute_1384: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_54, [0, 2, 1]);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_1385: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_52, [0, 2, 1]);  view_52 = None
    permute_1386: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_28, [0, 2, 1]);  permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_1390: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_1395: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_1399: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_95: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 1024);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    permute_1403: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    le_22: "b8[1, 128, 4096]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
    permute_1407: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    div_96: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 1024);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_1411: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_1416: "f32[16, 128, 128]" = torch.ops.aten.permute.default(div_1, [0, 2, 1]);  div_1 = None
    permute_1417: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_1418: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None
    permute_1419: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_17, [0, 2, 1]);  permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_1423: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_1428: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_1432: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_97: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 1024);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    permute_1436: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    le_23: "b8[1, 128, 4096]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
    permute_1440: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    div_98: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 1024);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    permute_1444: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_1449: "f32[16, 128, 128]" = torch.ops.aten.permute.default(div, [0, 2, 1]);  div = None
    permute_1450: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_14, [0, 2, 1]);  view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_1451: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
    permute_1452: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_6, [0, 2, 1]);  permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_1456: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_1461: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_1465: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_99: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None
    return [div_36, view_704, clone_98, clone_99, clone_104, clone_105, clone_112, clone_113, clone_118, clone_119, clone_126, clone_127, clone_132, clone_133, clone_140, clone_141, clone_146, clone_147, clone_154, clone_155, clone_160, clone_161, clone_168, clone_169, clone_174, clone_175, clone_182, clone_183, clone_188, clone_189, clone_196, clone_197, clone_202, clone_203, clone_210, clone_211, clone_216, clone_217, clone_224, clone_225, clone_230, clone_231, clone_238, clone_239, clone_244, clone_245, clone_252, clone_253, clone_258, clone_259, add_76, primals_2, primals_12, primals_18, primals_28, primals_34, primals_44, primals_50, primals_60, primals_66, primals_76, primals_82, primals_92, primals_98, primals_108, primals_114, primals_124, primals_130, primals_140, primals_146, primals_156, primals_162, primals_172, primals_178, primals_188, primals_194, primals_197, primals_207, primals_217, primals_223, primals_233, primals_243, primals_249, primals_259, primals_269, primals_275, primals_285, primals_295, primals_301, primals_311, primals_321, primals_327, primals_337, primals_347, primals_353, primals_363, primals_373, primals_379, primals_389, primals_399, primals_405, primals_415, primals_425, primals_431, primals_441, primals_451, primals_457, primals_467, primals_477, primals_483, primals_493, primals_503, primals_509, primals_514, view, mul_2, view_3, bmm, amax, sum_1, view_17, mul_5, view_19, view_21, mul_7, view_23, bmm_2, amax_1, sum_2, view_37, mul_10, view_39, view_41, mul_12, view_43, bmm_4, amax_2, sum_3, view_57, mul_15, view_59, view_61, mul_17, view_63, bmm_6, amax_3, sum_4, view_77, mul_20, view_79, view_81, mul_22, view_83, bmm_8, amax_4, sum_5, view_97, mul_25, view_99, view_101, mul_27, view_103, bmm_10, amax_5, sum_6, view_117, mul_30, view_119, view_121, mul_32, view_123, bmm_12, amax_6, sum_7, view_137, mul_35, view_139, view_141, mul_37, view_143, bmm_14, amax_7, sum_8, view_157, mul_40, view_159, view_161, mul_42, view_163, bmm_16, amax_8, sum_9, view_177, mul_45, view_179, view_181, mul_47, view_183, bmm_18, amax_9, sum_10, view_197, mul_50, view_199, view_201, mul_52, view_203, bmm_20, amax_10, sum_11, view_217, mul_55, view_219, view_221, mul_57, view_223, bmm_22, amax_11, sum_12, view_237, mul_60, view_239, view_241, mul_62, view_243, mul_66, view_247, view_263, mul_69, view_265, view_267, bmm_26, amax_13, sum_14, view_279, mul_72, view_281, view_283, mul_74, view_285, view_301, mul_77, view_303, bmm_30, amax_15, sum_16, view_317, mul_80, view_319, view_321, mul_82, view_323, view_339, mul_85, view_341, bmm_34, amax_17, sum_18, view_355, mul_88, view_357, view_359, mul_90, view_361, view_377, mul_93, view_379, bmm_38, amax_19, sum_20, view_393, mul_96, view_395, view_397, mul_98, view_399, view_415, mul_101, view_417, bmm_42, amax_21, sum_22, view_431, mul_104, view_433, view_435, mul_106, view_437, view_453, mul_109, view_455, bmm_46, amax_23, sum_24, view_469, mul_112, view_471, view_473, mul_114, view_475, view_491, mul_117, view_493, bmm_50, amax_25, sum_26, view_507, mul_120, view_509, view_511, mul_122, view_513, view_529, mul_125, view_531, bmm_54, amax_27, sum_28, view_545, mul_128, view_547, view_549, mul_130, view_551, view_567, mul_133, view_569, bmm_58, amax_29, sum_30, view_583, mul_136, view_585, view_587, mul_138, view_589, view_605, mul_141, view_607, bmm_62, amax_31, sum_32, view_621, mul_144, view_623, view_625, mul_146, view_627, view_643, mul_149, view_645, bmm_66, amax_33, sum_34, view_659, mul_152, view_661, view_663, mul_154, view_665, view_681, mul_157, view_683, bmm_70, amax_35, sum_36, view_697, mul_160, view_699, view_701, mul_162, view_703, sub_99, convert_element_type_6, permute_375, div_38, permute_377, le, permute_381, div_39, permute_385, permute_390, permute_391, permute_392, permute_393, permute_397, permute_402, permute_406, div_40, permute_410, permute_415, permute_416, alias_68, permute_417, permute_418, permute_422, permute_427, permute_431, div_41, permute_435, le_1, permute_439, div_42, permute_443, permute_448, permute_449, permute_450, permute_451, permute_455, permute_460, permute_464, div_43, permute_468, permute_473, permute_474, alias_71, permute_475, permute_476, permute_480, permute_485, permute_489, div_44, permute_493, le_2, permute_497, div_45, permute_501, permute_506, permute_507, permute_508, permute_509, permute_513, permute_518, permute_522, div_46, permute_526, permute_531, permute_532, alias_74, permute_533, permute_534, permute_538, permute_543, permute_547, div_47, permute_551, le_3, permute_555, div_48, permute_559, permute_564, permute_565, permute_566, permute_567, permute_571, permute_576, permute_580, div_49, permute_584, permute_589, permute_590, alias_77, permute_591, permute_592, permute_596, permute_601, permute_605, div_50, permute_609, le_4, permute_613, div_51, permute_617, permute_622, permute_623, permute_624, permute_625, permute_629, permute_634, permute_638, div_52, permute_642, permute_647, permute_648, alias_80, permute_649, permute_650, permute_654, permute_659, permute_663, div_53, permute_667, le_5, permute_671, div_54, permute_675, permute_680, permute_681, permute_682, permute_683, permute_687, permute_692, permute_696, div_55, permute_700, permute_705, permute_706, alias_83, permute_707, permute_708, permute_712, permute_717, permute_721, div_56, permute_725, le_6, permute_729, div_57, permute_733, permute_738, permute_739, permute_740, permute_741, permute_745, permute_750, permute_754, div_58, permute_758, permute_763, permute_764, alias_86, permute_765, permute_766, permute_770, permute_775, permute_779, div_59, permute_783, le_7, permute_787, div_60, permute_791, permute_796, permute_797, permute_798, permute_799, permute_803, permute_808, permute_812, div_61, permute_816, permute_821, permute_822, alias_89, permute_823, permute_824, permute_828, permute_833, permute_837, div_62, permute_841, le_8, permute_845, div_63, permute_849, permute_854, permute_855, permute_856, permute_857, permute_861, permute_866, permute_870, div_64, permute_874, permute_879, permute_880, alias_92, permute_881, permute_882, permute_886, permute_891, permute_895, div_65, permute_899, le_9, permute_903, div_66, permute_907, permute_912, permute_913, permute_914, permute_915, permute_919, permute_924, permute_928, div_67, permute_932, permute_937, permute_938, alias_95, permute_939, permute_940, permute_944, permute_949, permute_953, div_68, permute_957, le_10, permute_961, div_69, permute_965, permute_970, permute_971, permute_972, permute_973, permute_977, permute_982, permute_986, div_70, permute_990, permute_995, permute_996, alias_98, permute_997, permute_998, permute_1002, permute_1007, permute_1011, div_71, permute_1015, le_11, permute_1019, div_72, permute_1023, permute_1028, permute_1029, permute_1030, permute_1031, permute_1035, permute_1040, permute_1044, div_73, permute_1048, permute_1053, permute_1054, alias_101, permute_1055, permute_1056, permute_1060, permute_1065, permute_1069, div_74, div_75, permute_1073, le_12, permute_1077, div_76, permute_1081, permute_1086, permute_1087, permute_1088, permute_1089, permute_1093, permute_1098, permute_1102, div_77, permute_1106, le_13, permute_1110, div_78, permute_1114, permute_1119, permute_1120, permute_1121, permute_1122, permute_1126, permute_1131, permute_1135, div_79, permute_1139, le_14, permute_1143, div_80, permute_1147, permute_1152, permute_1153, permute_1154, permute_1155, permute_1159, permute_1164, permute_1168, div_81, permute_1172, le_15, permute_1176, div_82, permute_1180, permute_1185, permute_1186, permute_1187, permute_1188, permute_1192, permute_1197, permute_1201, div_83, permute_1205, le_16, permute_1209, div_84, permute_1213, permute_1218, permute_1219, permute_1220, permute_1221, permute_1225, permute_1230, permute_1234, div_85, permute_1238, le_17, permute_1242, div_86, permute_1246, permute_1251, permute_1252, permute_1253, permute_1254, permute_1258, permute_1263, permute_1267, div_87, permute_1271, le_18, permute_1275, div_88, permute_1279, permute_1284, permute_1285, permute_1286, permute_1287, permute_1291, permute_1296, permute_1300, div_89, permute_1304, le_19, permute_1308, div_90, permute_1312, permute_1317, permute_1318, permute_1319, permute_1320, permute_1324, permute_1329, permute_1333, div_91, permute_1337, le_20, permute_1341, div_92, permute_1345, permute_1350, permute_1351, permute_1352, permute_1353, permute_1357, permute_1362, permute_1366, div_93, permute_1370, le_21, permute_1374, div_94, permute_1378, permute_1383, permute_1384, permute_1385, permute_1386, permute_1390, permute_1395, permute_1399, div_95, permute_1403, le_22, permute_1407, div_96, permute_1411, permute_1416, permute_1417, permute_1418, permute_1419, permute_1423, permute_1428, permute_1432, div_97, permute_1436, le_23, permute_1440, div_98, permute_1444, permute_1449, permute_1450, permute_1451, permute_1452, permute_1456, permute_1461, permute_1465, div_99]
    