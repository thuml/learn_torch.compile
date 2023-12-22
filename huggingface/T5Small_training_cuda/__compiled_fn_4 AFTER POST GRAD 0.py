from __future__ import annotations



def forward(self, primals_1: "f32[512]", primals_2: "f32[512]", primals_3: "f32[512]", primals_4: "f32[512]", primals_5: "f32[512]", primals_6: "f32[512]", primals_7: "f32[512]", primals_8: "f32[512]", primals_9: "f32[512]", primals_10: "f32[512]", primals_11: "f32[512]", primals_12: "f32[512]", primals_13: "f32[512]", primals_14: "f32[512]", primals_15: "f32[512]", primals_16: "f32[512]", primals_17: "f32[512]", primals_18: "f32[512]", primals_19: "f32[512]", primals_20: "f32[512]", primals_21: "f32[512]", primals_22: "f32[512]", primals_23: "f32[512]", primals_24: "f32[512]", primals_25: "f32[512]", primals_26: "f32[512]", primals_27: "f32[512]", primals_28: "f32[512]", primals_29: "f32[512]", primals_30: "f32[512]", primals_31: "f32[512]", primals_32: "f32[512]", primals_134: "i64[1, 1024]", view: "i64[1, 1024]", getitem: "f32[1, 1024, 512]", getitem_1: "b8[1, 1024, 512]", rsqrt: "f32[1, 1024, 1]", view_1: "f32[1024, 512]", add_3: "i64[1024, 1024]", getitem_3: "b8[1, 8, 1024, 1024]", view_19: "f32[1024, 512]", getitem_5: "b8[1, 1024, 512]", add_6: "f32[1, 1024, 512]", rsqrt_1: "f32[1, 1024, 1]", view_21: "f32[1024, 512]", getitem_7: "b8[1, 1024, 2048]", view_23: "f32[1024, 2048]", getitem_9: "b8[1, 1024, 512]", add_8: "f32[1, 1024, 512]", rsqrt_2: "f32[1, 1024, 1]", view_25: "f32[1024, 512]", getitem_11: "b8[1, 8, 1024, 1024]", view_43: "f32[1024, 512]", getitem_13: "b8[1, 1024, 512]", add_11: "f32[1, 1024, 512]", rsqrt_3: "f32[1, 1024, 1]", view_45: "f32[1024, 512]", getitem_15: "b8[1, 1024, 2048]", view_47: "f32[1024, 2048]", getitem_17: "b8[1, 1024, 512]", add_13: "f32[1, 1024, 512]", rsqrt_4: "f32[1, 1024, 1]", view_49: "f32[1024, 512]", getitem_19: "b8[1, 8, 1024, 1024]", view_67: "f32[1024, 512]", getitem_21: "b8[1, 1024, 512]", add_16: "f32[1, 1024, 512]", rsqrt_5: "f32[1, 1024, 1]", view_69: "f32[1024, 512]", getitem_23: "b8[1, 1024, 2048]", view_71: "f32[1024, 2048]", getitem_25: "b8[1, 1024, 512]", add_18: "f32[1, 1024, 512]", rsqrt_6: "f32[1, 1024, 1]", view_73: "f32[1024, 512]", getitem_27: "b8[1, 8, 1024, 1024]", view_91: "f32[1024, 512]", getitem_29: "b8[1, 1024, 512]", add_21: "f32[1, 1024, 512]", rsqrt_7: "f32[1, 1024, 1]", view_93: "f32[1024, 512]", getitem_31: "b8[1, 1024, 2048]", view_95: "f32[1024, 2048]", getitem_33: "b8[1, 1024, 512]", add_23: "f32[1, 1024, 512]", rsqrt_8: "f32[1, 1024, 1]", view_97: "f32[1024, 512]", getitem_35: "b8[1, 8, 1024, 1024]", view_115: "f32[1024, 512]", getitem_37: "b8[1, 1024, 512]", add_26: "f32[1, 1024, 512]", rsqrt_9: "f32[1, 1024, 1]", view_117: "f32[1024, 512]", getitem_39: "b8[1, 1024, 2048]", view_119: "f32[1024, 2048]", getitem_41: "b8[1, 1024, 512]", add_28: "f32[1, 1024, 512]", rsqrt_10: "f32[1, 1024, 1]", view_121: "f32[1024, 512]", getitem_43: "b8[1, 8, 1024, 1024]", view_139: "f32[1024, 512]", getitem_45: "b8[1, 1024, 512]", add_31: "f32[1, 1024, 512]", rsqrt_11: "f32[1, 1024, 1]", view_141: "f32[1024, 512]", getitem_47: "b8[1, 1024, 2048]", view_143: "f32[1024, 2048]", getitem_49: "b8[1, 1024, 512]", add_33: "f32[1, 1024, 512]", rsqrt_12: "f32[1, 1024, 1]", getitem_51: "b8[1, 1024, 512]", view_145: "i64[1, 1024]", getitem_52: "f32[1, 1024, 512]", getitem_53: "b8[1, 1024, 512]", rsqrt_13: "f32[1, 1024, 1]", view_146: "f32[1024, 512]", add_37: "i64[1024, 1024]", getitem_55: "b8[1, 8, 1024, 1024]", view_164: "f32[1024, 512]", getitem_57: "b8[1, 1024, 512]", add_40: "f32[1, 1024, 512]", rsqrt_14: "f32[1, 1024, 1]", view_166: "f32[1024, 512]", view_169: "f32[1024, 512]", getitem_59: "b8[1, 8, 1024, 1024]", view_184: "f32[1024, 512]", getitem_61: "b8[1, 1024, 512]", add_44: "f32[1, 1024, 512]", rsqrt_15: "f32[1, 1024, 1]", view_186: "f32[1024, 512]", getitem_63: "b8[1, 1024, 2048]", view_188: "f32[1024, 2048]", getitem_65: "b8[1, 1024, 512]", add_46: "f32[1, 1024, 512]", rsqrt_16: "f32[1, 1024, 1]", view_190: "f32[1024, 512]", getitem_67: "b8[1, 8, 1024, 1024]", view_208: "f32[1024, 512]", getitem_69: "b8[1, 1024, 512]", add_49: "f32[1, 1024, 512]", rsqrt_17: "f32[1, 1024, 1]", view_210: "f32[1024, 512]", getitem_71: "b8[1, 8, 1024, 1024]", view_228: "f32[1024, 512]", getitem_73: "b8[1, 1024, 512]", add_52: "f32[1, 1024, 512]", rsqrt_18: "f32[1, 1024, 1]", view_230: "f32[1024, 512]", getitem_75: "b8[1, 1024, 2048]", view_232: "f32[1024, 2048]", getitem_77: "b8[1, 1024, 512]", add_54: "f32[1, 1024, 512]", rsqrt_19: "f32[1, 1024, 1]", view_234: "f32[1024, 512]", getitem_79: "b8[1, 8, 1024, 1024]", view_252: "f32[1024, 512]", getitem_81: "b8[1, 1024, 512]", add_57: "f32[1, 1024, 512]", rsqrt_20: "f32[1, 1024, 1]", view_254: "f32[1024, 512]", getitem_83: "b8[1, 8, 1024, 1024]", view_272: "f32[1024, 512]", getitem_85: "b8[1, 1024, 512]", add_60: "f32[1, 1024, 512]", rsqrt_21: "f32[1, 1024, 1]", view_274: "f32[1024, 512]", getitem_87: "b8[1, 1024, 2048]", view_276: "f32[1024, 2048]", getitem_89: "b8[1, 1024, 512]", add_62: "f32[1, 1024, 512]", rsqrt_22: "f32[1, 1024, 1]", view_278: "f32[1024, 512]", getitem_91: "b8[1, 8, 1024, 1024]", view_296: "f32[1024, 512]", getitem_93: "b8[1, 1024, 512]", add_65: "f32[1, 1024, 512]", rsqrt_23: "f32[1, 1024, 1]", view_298: "f32[1024, 512]", getitem_95: "b8[1, 8, 1024, 1024]", view_316: "f32[1024, 512]", getitem_97: "b8[1, 1024, 512]", add_68: "f32[1, 1024, 512]", rsqrt_24: "f32[1, 1024, 1]", view_318: "f32[1024, 512]", getitem_99: "b8[1, 1024, 2048]", view_320: "f32[1024, 2048]", getitem_101: "b8[1, 1024, 512]", add_70: "f32[1, 1024, 512]", rsqrt_25: "f32[1, 1024, 1]", view_322: "f32[1024, 512]", getitem_103: "b8[1, 8, 1024, 1024]", view_340: "f32[1024, 512]", getitem_105: "b8[1, 1024, 512]", add_73: "f32[1, 1024, 512]", rsqrt_26: "f32[1, 1024, 1]", view_342: "f32[1024, 512]", getitem_107: "b8[1, 8, 1024, 1024]", view_360: "f32[1024, 512]", getitem_109: "b8[1, 1024, 512]", add_76: "f32[1, 1024, 512]", rsqrt_27: "f32[1, 1024, 1]", view_362: "f32[1024, 512]", getitem_111: "b8[1, 1024, 2048]", view_364: "f32[1024, 2048]", getitem_113: "b8[1, 1024, 512]", add_78: "f32[1, 1024, 512]", rsqrt_28: "f32[1, 1024, 1]", view_366: "f32[1024, 512]", getitem_115: "b8[1, 8, 1024, 1024]", view_384: "f32[1024, 512]", getitem_117: "b8[1, 1024, 512]", add_81: "f32[1, 1024, 512]", rsqrt_29: "f32[1, 1024, 1]", view_386: "f32[1024, 512]", getitem_119: "b8[1, 8, 1024, 1024]", view_404: "f32[1024, 512]", getitem_121: "b8[1, 1024, 512]", add_84: "f32[1, 1024, 512]", rsqrt_30: "f32[1, 1024, 1]", view_406: "f32[1024, 512]", getitem_123: "b8[1, 1024, 2048]", view_408: "f32[1024, 2048]", getitem_125: "b8[1, 1024, 512]", add_86: "f32[1, 1024, 512]", rsqrt_31: "f32[1, 1024, 1]", getitem_127: "b8[1, 1024, 512]", view_410: "f32[1024, 512]", sub_24: "f32[1024, 32128]", convert_element_type_7: "f32[]", permute_191: "f32[32128, 512]", permute_195: "f32[512, 2048]", le_1: "b8[1, 1024, 2048]", permute_199: "f32[2048, 512]", permute_203: "f32[512, 512]", permute_206: "f32[8, 1024, 1024]", permute_207: "f32[8, 64, 1024]", alias_67: "f32[1, 8, 1024, 1024]", permute_208: "f32[8, 64, 1024]", permute_209: "f32[8, 1024, 64]", permute_214: "f32[512, 512]", permute_219: "f32[512, 512]", permute_224: "f32[512, 512]", permute_228: "f32[512, 512]", permute_231: "f32[8, 1024, 1024]", permute_232: "f32[8, 64, 1024]", alias_69: "f32[1, 8, 1024, 1024]", permute_233: "f32[8, 64, 1024]", permute_234: "f32[8, 1024, 64]", permute_239: "f32[512, 512]", permute_244: "f32[512, 512]", permute_249: "f32[512, 512]", permute_253: "f32[512, 2048]", le_2: "b8[1, 1024, 2048]", permute_257: "f32[2048, 512]", permute_261: "f32[512, 512]", permute_264: "f32[8, 1024, 1024]", permute_265: "f32[8, 64, 1024]", alias_73: "f32[1, 8, 1024, 1024]", permute_266: "f32[8, 64, 1024]", permute_267: "f32[8, 1024, 64]", permute_272: "f32[512, 512]", permute_277: "f32[512, 512]", permute_282: "f32[512, 512]", permute_286: "f32[512, 512]", permute_289: "f32[8, 1024, 1024]", permute_290: "f32[8, 64, 1024]", alias_75: "f32[1, 8, 1024, 1024]", permute_291: "f32[8, 64, 1024]", permute_292: "f32[8, 1024, 64]", permute_297: "f32[512, 512]", permute_302: "f32[512, 512]", permute_307: "f32[512, 512]", permute_311: "f32[512, 2048]", le_3: "b8[1, 1024, 2048]", permute_315: "f32[2048, 512]", permute_319: "f32[512, 512]", permute_322: "f32[8, 1024, 1024]", permute_323: "f32[8, 64, 1024]", alias_79: "f32[1, 8, 1024, 1024]", permute_324: "f32[8, 64, 1024]", permute_325: "f32[8, 1024, 64]", permute_330: "f32[512, 512]", permute_335: "f32[512, 512]", permute_340: "f32[512, 512]", permute_344: "f32[512, 512]", permute_347: "f32[8, 1024, 1024]", permute_348: "f32[8, 64, 1024]", alias_81: "f32[1, 8, 1024, 1024]", permute_349: "f32[8, 64, 1024]", permute_350: "f32[8, 1024, 64]", permute_355: "f32[512, 512]", permute_360: "f32[512, 512]", permute_365: "f32[512, 512]", permute_369: "f32[512, 2048]", le_4: "b8[1, 1024, 2048]", permute_373: "f32[2048, 512]", permute_377: "f32[512, 512]", permute_380: "f32[8, 1024, 1024]", permute_381: "f32[8, 64, 1024]", alias_85: "f32[1, 8, 1024, 1024]", permute_382: "f32[8, 64, 1024]", permute_383: "f32[8, 1024, 64]", permute_388: "f32[512, 512]", permute_393: "f32[512, 512]", permute_398: "f32[512, 512]", permute_402: "f32[512, 512]", permute_405: "f32[8, 1024, 1024]", permute_406: "f32[8, 64, 1024]", alias_87: "f32[1, 8, 1024, 1024]", permute_407: "f32[8, 64, 1024]", permute_408: "f32[8, 1024, 64]", permute_413: "f32[512, 512]", permute_418: "f32[512, 512]", permute_423: "f32[512, 512]", permute_427: "f32[512, 2048]", le_5: "b8[1, 1024, 2048]", permute_431: "f32[2048, 512]", permute_435: "f32[512, 512]", permute_438: "f32[8, 1024, 1024]", permute_439: "f32[8, 64, 1024]", alias_91: "f32[1, 8, 1024, 1024]", permute_440: "f32[8, 64, 1024]", permute_441: "f32[8, 1024, 64]", permute_446: "f32[512, 512]", permute_451: "f32[512, 512]", permute_456: "f32[512, 512]", permute_460: "f32[512, 512]", permute_463: "f32[8, 1024, 1024]", permute_464: "f32[8, 64, 1024]", alias_93: "f32[1, 8, 1024, 1024]", permute_465: "f32[8, 64, 1024]", permute_466: "f32[8, 1024, 64]", permute_471: "f32[512, 512]", permute_476: "f32[512, 512]", permute_481: "f32[512, 512]", permute_485: "f32[512, 2048]", le_6: "b8[1, 1024, 2048]", permute_489: "f32[2048, 512]", permute_493: "f32[512, 512]", permute_496: "f32[8, 1024, 1024]", permute_497: "f32[8, 64, 1024]", alias_97: "f32[1, 8, 1024, 1024]", permute_498: "f32[8, 64, 1024]", permute_499: "f32[8, 1024, 64]", permute_504: "f32[512, 512]", permute_509: "f32[512, 512]", permute_514: "f32[512, 512]", permute_518: "f32[512, 512]", permute_521: "f32[8, 1024, 1024]", permute_522: "f32[8, 64, 1024]", alias_99: "f32[1, 8, 1024, 1024]", permute_524: "f32[8, 64, 1024]", permute_525: "f32[8, 1024, 64]", permute_530: "f32[512, 512]", permute_535: "f32[512, 512]", permute_540: "f32[512, 512]", permute_544: "f32[512, 2048]", le_7: "b8[1, 1024, 2048]", permute_548: "f32[2048, 512]", permute_552: "f32[512, 512]", permute_555: "f32[8, 1024, 1024]", permute_556: "f32[8, 64, 1024]", alias_104: "f32[1, 8, 1024, 1024]", permute_557: "f32[8, 64, 1024]", permute_558: "f32[8, 1024, 64]", permute_563: "f32[512, 512]", permute_568: "f32[512, 512]", permute_573: "f32[512, 512]", permute_577: "f32[512, 2048]", le_8: "b8[1, 1024, 2048]", permute_581: "f32[2048, 512]", permute_585: "f32[512, 512]", permute_588: "f32[8, 1024, 1024]", permute_589: "f32[8, 64, 1024]", alias_108: "f32[1, 8, 1024, 1024]", permute_590: "f32[8, 64, 1024]", permute_591: "f32[8, 1024, 64]", permute_596: "f32[512, 512]", permute_601: "f32[512, 512]", permute_606: "f32[512, 512]", permute_610: "f32[512, 2048]", le_9: "b8[1, 1024, 2048]", permute_614: "f32[2048, 512]", permute_618: "f32[512, 512]", permute_621: "f32[8, 1024, 1024]", permute_622: "f32[8, 64, 1024]", alias_112: "f32[1, 8, 1024, 1024]", permute_623: "f32[8, 64, 1024]", permute_624: "f32[8, 1024, 64]", permute_629: "f32[512, 512]", permute_634: "f32[512, 512]", permute_639: "f32[512, 512]", permute_643: "f32[512, 2048]", le_10: "b8[1, 1024, 2048]", permute_647: "f32[2048, 512]", permute_651: "f32[512, 512]", permute_654: "f32[8, 1024, 1024]", permute_655: "f32[8, 64, 1024]", alias_116: "f32[1, 8, 1024, 1024]", permute_656: "f32[8, 64, 1024]", permute_657: "f32[8, 1024, 64]", permute_662: "f32[512, 512]", permute_667: "f32[512, 512]", permute_672: "f32[512, 512]", permute_676: "f32[512, 2048]", le_11: "b8[1, 1024, 2048]", permute_680: "f32[2048, 512]", permute_684: "f32[512, 512]", permute_687: "f32[8, 1024, 1024]", permute_688: "f32[8, 64, 1024]", alias_120: "f32[1, 8, 1024, 1024]", permute_689: "f32[8, 64, 1024]", permute_690: "f32[8, 1024, 64]", permute_695: "f32[512, 512]", permute_700: "f32[512, 512]", permute_705: "f32[512, 512]", permute_709: "f32[512, 2048]", le_12: "b8[1, 1024, 2048]", permute_713: "f32[2048, 512]", permute_717: "f32[512, 512]", permute_720: "f32[8, 1024, 1024]", permute_721: "f32[8, 64, 1024]", alias_124: "f32[1, 8, 1024, 1024]", permute_723: "f32[8, 64, 1024]", permute_724: "f32[8, 1024, 64]", permute_729: "f32[512, 512]", permute_734: "f32[512, 512]", permute_739: "f32[512, 512]", tangents_1: "f32[]", tangents_2: "f32[1, 1024, 32128]", tangents_3: "f32[1, 8, 1024, 64]", tangents_4: "f32[1, 8, 1024, 64]", tangents_5: "f32[1, 8, 1024, 64]", tangents_6: "f32[1, 8, 1024, 64]", tangents_7: "f32[1, 8, 1024, 64]", tangents_8: "f32[1, 8, 1024, 64]", tangents_9: "f32[1, 8, 1024, 64]", tangents_10: "f32[1, 8, 1024, 64]", tangents_11: "f32[1, 8, 1024, 64]", tangents_12: "f32[1, 8, 1024, 64]", tangents_13: "f32[1, 8, 1024, 64]", tangents_14: "f32[1, 8, 1024, 64]", tangents_15: "f32[1, 8, 1024, 64]", tangents_16: "f32[1, 8, 1024, 64]", tangents_17: "f32[1, 8, 1024, 64]", tangents_18: "f32[1, 8, 1024, 64]", tangents_19: "f32[1, 8, 1024, 64]", tangents_20: "f32[1, 8, 1024, 64]", tangents_21: "f32[1, 8, 1024, 64]", tangents_22: "f32[1, 8, 1024, 64]", tangents_23: "f32[1, 8, 1024, 64]", tangents_24: "f32[1, 8, 1024, 64]", tangents_25: "f32[1, 8, 1024, 64]", tangents_26: "f32[1, 8, 1024, 64]", tangents_27: "f32[1, 1024, 512]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_1: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(getitem, rsqrt)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_5: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_6, rsqrt_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_7: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_8, rsqrt_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_9: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_11, rsqrt_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_11: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_13, rsqrt_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_13: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_16, rsqrt_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_15: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_18, rsqrt_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_17: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_21, rsqrt_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_19: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_23, rsqrt_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_21: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_26, rsqrt_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_23: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_28, rsqrt_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_25: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_31, rsqrt_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_27: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_33, rsqrt_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_32: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(getitem_52, rsqrt_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_35: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_40, rsqrt_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_37: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_44, rsqrt_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_39: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_46, rsqrt_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_41: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_49, rsqrt_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_43: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_52, rsqrt_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_45: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_54, rsqrt_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_47: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_57, rsqrt_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_49: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_60, rsqrt_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_51: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_62, rsqrt_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_53: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_65, rsqrt_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_55: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_68, rsqrt_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_57: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_70, rsqrt_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_59: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_73, rsqrt_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_61: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_76, rsqrt_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_63: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_78, rsqrt_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_65: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_81, rsqrt_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_67: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_84, rsqrt_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_69: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_86, rsqrt_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1781, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
    view_413: "i64[1024]" = torch.ops.aten.reshape.default(primals_134, [-1]);  primals_134 = None
    full_default_6: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    full_default_7: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    div_23: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_7);  tangents_1 = convert_element_type_7 = None
    unsqueeze_18: "i64[1024, 1]" = torch.ops.aten.unsqueeze.default(view_413, 1);  view_413 = None
    ne_3: "b8[1024, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_18, -100)
    where_4: "i64[1024, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_18, full_default_6);  unsqueeze_18 = full_default_6 = None
    full_default_9: "f32[1024, 32128]" = torch.ops.aten.full.default([1024, 32128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    scatter: "f32[1024, 32128]" = torch.ops.aten.scatter.value(full_default_9, 1, where_4, -1.0);  full_default_9 = where_4 = None
    where_5: "f32[1024, 1]" = torch.ops.aten.where.self(ne_3, div_23, full_default_7);  ne_3 = div_23 = None
    mul_72: "f32[1024, 32128]" = torch.ops.aten.mul.Tensor(scatter, where_5);  scatter = where_5 = None
    exp_19: "f32[1024, 32128]" = torch.ops.aten.exp.default(sub_24);  sub_24 = None
    sum_22: "f32[1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_72, [1], True)
    mul_73: "f32[1024, 32128]" = torch.ops.aten.mul.Tensor(exp_19, sum_22);  exp_19 = sum_22 = None
    sub_25: "f32[1024, 32128]" = torch.ops.aten.sub.Tensor(mul_72, mul_73);  mul_72 = mul_73 = None
    view_414: "f32[1, 1024, 32128]" = torch.ops.aten.reshape.default(sub_25, [1, 1024, 32128]);  sub_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1781, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
    add_88: "f32[1, 1024, 32128]" = torch.ops.aten.add.Tensor(tangents_2, view_414);  tangents_2 = view_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1774, code: lm_logits = self.lm_head(sequence_output)
    view_415: "f32[1024, 32128]" = torch.ops.aten.reshape.default(add_88, [1024, 32128]);  add_88 = None
    permute_189: "f32[32128, 1024]" = torch.ops.aten.permute.default(view_415, [1, 0])
    mm_97: "f32[32128, 512]" = torch.ops.aten.mm.default(permute_189, view_410);  permute_189 = view_410 = None
    permute_190: "f32[512, 32128]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    mm_98: "f32[1024, 512]" = torch.ops.aten.mm.default(view_415, permute_191);  view_415 = permute_191 = None
    view_416: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_98, [1, 1024, 512]);  mm_98 = None
    permute_192: "f32[32128, 512]" = torch.ops.aten.permute.default(permute_190, [1, 0]);  permute_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1772, code: sequence_output = sequence_output * (self.model_dim**-0.5)
    mul_74: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_416, 0.04419417382415922);  view_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1166, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_8: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_127, torch.float32);  getitem_127 = None
    mul_75: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
    mul_76: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_77: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_76, primals_32);  primals_32 = None
    mul_78: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_76, mul_69);  mul_76 = mul_69 = None
    sum_23: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_78, [0, 1], True);  mul_78 = None
    view_417: "f32[512]" = torch.ops.aten.reshape.default(sum_23, [512]);  sum_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_79: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_77, add_86)
    mul_80: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_77, rsqrt_31);  mul_77 = None
    sum_24: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_79, [2], True);  mul_79 = None
    pow_33: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_31, 3);  rsqrt_31 = None
    mul_81: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_24, -0.5);  sum_24 = None
    mul_82: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_81, pow_33);  mul_81 = pow_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_72: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_82, [1, 1024, 512]);  mul_82 = None
    div_24: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_72, 512);  expand_72 = None
    pow_34: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_86, 1.0);  add_86 = None
    mul_83: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_34, 2.0);  pow_34 = None
    mul_84: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_24, mul_83);  div_24 = mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_89: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(mul_80, mul_84);  mul_80 = mul_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_9: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_125, torch.float32);  getitem_125 = None
    mul_85: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.1111111111111112);  convert_element_type_9 = None
    mul_86: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_89, mul_85);  mul_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_418: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_86, [1024, 512]);  mul_86 = None
    permute_193: "f32[512, 1024]" = torch.ops.aten.permute.default(view_418, [1, 0])
    mm_99: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_193, view_408);  permute_193 = view_408 = None
    permute_194: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    mm_100: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_418, permute_195);  view_418 = permute_195 = None
    view_419: "f32[1, 1024, 2048]" = torch.ops.aten.reshape.default(mm_100, [1, 1024, 2048]);  mm_100 = None
    permute_196: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_194, [1, 0]);  permute_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_10: "f32[1, 1024, 2048]" = torch.ops.prims.convert_element_type.default(getitem_123, torch.float32);  getitem_123 = None
    mul_87: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
    mul_88: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_419, mul_87);  view_419 = mul_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    where_6: "f32[1, 1024, 2048]" = torch.ops.aten.where.self(le_1, full_default_7, mul_88);  le_1 = mul_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_420: "f32[1024, 2048]" = torch.ops.aten.reshape.default(where_6, [1024, 2048]);  where_6 = None
    permute_197: "f32[2048, 1024]" = torch.ops.aten.permute.default(view_420, [1, 0])
    mm_101: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_197, view_406);  permute_197 = view_406 = None
    permute_198: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    mm_102: "f32[1024, 512]" = torch.ops.aten.mm.default(view_420, permute_199);  view_420 = permute_199 = None
    view_421: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_102, [1, 1024, 512]);  mm_102 = None
    permute_200: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_89: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_421, primals_31);  primals_31 = None
    mul_90: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_421, mul_67);  view_421 = mul_67 = None
    sum_25: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_90, [0, 1], True);  mul_90 = None
    view_422: "f32[512]" = torch.ops.aten.reshape.default(sum_25, [512]);  sum_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_91: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_89, add_84)
    mul_92: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_89, rsqrt_30);  mul_89 = None
    sum_26: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_91, [2], True);  mul_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_90: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_89, mul_92);  add_89 = mul_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_35: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_30, 3);  rsqrt_30 = None
    mul_93: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_26, -0.5);  sum_26 = None
    mul_94: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_93, pow_35);  mul_93 = pow_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_73: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_94, [1, 1024, 512]);  mul_94 = None
    div_25: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_73, 512);  expand_73 = None
    pow_36: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_84, 1.0);  add_84 = None
    mul_95: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_36, 2.0);  pow_36 = None
    mul_96: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_25, mul_95);  div_25 = mul_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_91: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_90, mul_96);  add_90 = mul_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:647, code: layer_output = hidden_states + self.dropout(attention_output[0])
    convert_element_type_11: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_121, torch.float32);  getitem_121 = None
    mul_97: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
    mul_98: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_91, mul_97);  mul_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_423: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_98, [1024, 512]);  mul_98 = None
    permute_201: "f32[512, 1024]" = torch.ops.aten.permute.default(view_423, [1, 0])
    mm_103: "f32[512, 512]" = torch.ops.aten.mm.default(permute_201, view_404);  permute_201 = view_404 = None
    permute_202: "f32[512, 512]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    mm_104: "f32[1024, 512]" = torch.ops.aten.mm.default(view_423, permute_203);  view_423 = permute_203 = None
    view_424: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_104, [1, 1024, 512]);  mm_104 = None
    permute_204: "f32[512, 512]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_425: "f32[1, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_424, [1, 1024, 8, 64]);  view_424 = None
    permute_205: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_425, [0, 2, 1, 3]);  view_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_426: "f32[8, 1024, 64]" = torch.ops.aten.reshape.default(permute_205, [8, 1024, 64]);  permute_205 = None
    bmm_36: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_206, view_426);  permute_206 = None
    bmm_37: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_426, permute_207);  view_426 = permute_207 = None
    view_427: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_36, [1, 8, 1024, 64]);  bmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_92: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_26, view_427);  tangents_26 = view_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_428: "f32[1, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_37, [1, 8, 1024, 1024]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_12: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_119, torch.float32);  getitem_119 = None
    mul_99: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
    mul_100: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_428, mul_99);  view_428 = mul_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_101: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_100, alias_67);  mul_100 = None
    sum_27: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_101, [-1], True)
    mul_102: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_67, sum_27);  alias_67 = sum_27 = None
    sub_26: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_101, mul_102);  mul_101 = mul_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_1: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_26, 0);  sub_26 = None
    full_8: "f32[8388608]" = torch.ops.aten.full.default([8388608], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    as_strided: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_8, [8, 1024, 1024], [1048576, 1024, 1], 0)
    as_strided_scatter: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_1, [8, 1024, 1024], [1048576, 1024, 1], 0);  squeeze_1 = None
    as_strided_3: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter = None
    new_empty_strided: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_3, [8, 1024, 1024], [1048576, 1024, 1])
    as_strided_5: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_3, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    as_strided_scatter_1: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(as_strided_3, as_strided_5, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  as_strided_3 = as_strided_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    bmm_38: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_208, as_strided_scatter_1);  permute_208 = None
    bmm_39: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_1, permute_209);  as_strided_scatter_1 = permute_209 = None
    view_429: "f32[1, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_38, [1, 8, 64, 1024]);  bmm_38 = None
    view_430: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_39, [1, 8, 1024, 64]);  bmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_210: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_429, [0, 1, 3, 2]);  view_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_93: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_25, permute_210);  tangents_25 = permute_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_211: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_92, [0, 2, 1, 3]);  add_92 = None
    clone_24: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_211, memory_format = torch.contiguous_format);  permute_211 = None
    view_431: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_24, [1, 1024, 512]);  clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_432: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_431, [1024, 512]);  view_431 = None
    permute_212: "f32[512, 1024]" = torch.ops.aten.permute.default(view_432, [1, 0])
    mm_105: "f32[512, 512]" = torch.ops.aten.mm.default(permute_212, view_169);  permute_212 = None
    permute_213: "f32[512, 512]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    mm_106: "f32[1024, 512]" = torch.ops.aten.mm.default(view_432, permute_214);  view_432 = permute_214 = None
    view_433: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_106, [1, 1024, 512]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_94: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(tangents_27, view_433);  tangents_27 = view_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_215: "f32[512, 512]" = torch.ops.aten.permute.default(permute_213, [1, 0]);  permute_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_216: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_93, [0, 2, 1, 3]);  add_93 = None
    clone_25: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
    view_434: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_25, [1, 1024, 512]);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_435: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_434, [1024, 512]);  view_434 = None
    permute_217: "f32[512, 1024]" = torch.ops.aten.permute.default(view_435, [1, 0])
    mm_107: "f32[512, 512]" = torch.ops.aten.mm.default(permute_217, view_169);  permute_217 = None
    permute_218: "f32[512, 512]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    mm_108: "f32[1024, 512]" = torch.ops.aten.mm.default(view_435, permute_219);  view_435 = permute_219 = None
    view_436: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_108, [1, 1024, 512]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_95: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_94, view_436);  add_94 = view_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_220: "f32[512, 512]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_221: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_430, [0, 2, 1, 3]);  view_430 = None
    clone_26: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_221, memory_format = torch.contiguous_format);  permute_221 = None
    view_437: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_26, [1, 1024, 512]);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_438: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_437, [1024, 512]);  view_437 = None
    permute_222: "f32[512, 1024]" = torch.ops.aten.permute.default(view_438, [1, 0])
    mm_109: "f32[512, 512]" = torch.ops.aten.mm.default(permute_222, view_386);  permute_222 = view_386 = None
    permute_223: "f32[512, 512]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    mm_110: "f32[1024, 512]" = torch.ops.aten.mm.default(view_438, permute_224);  view_438 = permute_224 = None
    view_439: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_110, [1, 1024, 512]);  mm_110 = None
    permute_225: "f32[512, 512]" = torch.ops.aten.permute.default(permute_223, [1, 0]);  permute_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_103: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_439, primals_30);  primals_30 = None
    mul_104: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_439, mul_65);  view_439 = mul_65 = None
    sum_28: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_104, [0, 1], True);  mul_104 = None
    view_440: "f32[512]" = torch.ops.aten.reshape.default(sum_28, [512]);  sum_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_105: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_103, add_81)
    mul_106: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_103, rsqrt_29);  mul_103 = None
    sum_29: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_105, [2], True);  mul_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_96: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_91, mul_106);  add_91 = mul_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_37: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_29, 3);  rsqrt_29 = None
    mul_107: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_29, -0.5);  sum_29 = None
    mul_108: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_107, pow_37);  mul_107 = pow_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_74: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_108, [1, 1024, 512]);  mul_108 = None
    div_26: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_74, 512);  expand_74 = None
    pow_38: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_81, 1.0);  add_81 = None
    mul_109: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_38, 2.0);  pow_38 = None
    mul_110: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_26, mul_109);  div_26 = mul_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_97: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_96, mul_110);  add_96 = mul_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_13: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_117, torch.float32);  getitem_117 = None
    mul_111: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
    mul_112: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_97, mul_111);  mul_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_441: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_112, [1024, 512]);  mul_112 = None
    permute_226: "f32[512, 1024]" = torch.ops.aten.permute.default(view_441, [1, 0])
    mm_111: "f32[512, 512]" = torch.ops.aten.mm.default(permute_226, view_384);  permute_226 = view_384 = None
    permute_227: "f32[512, 512]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    mm_112: "f32[1024, 512]" = torch.ops.aten.mm.default(view_441, permute_228);  view_441 = permute_228 = None
    view_442: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_112, [1, 1024, 512]);  mm_112 = None
    permute_229: "f32[512, 512]" = torch.ops.aten.permute.default(permute_227, [1, 0]);  permute_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_443: "f32[1, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_442, [1, 1024, 8, 64]);  view_442 = None
    permute_230: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_443, [0, 2, 1, 3]);  view_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_444: "f32[8, 1024, 64]" = torch.ops.aten.reshape.default(permute_230, [8, 1024, 64]);  permute_230 = None
    bmm_40: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_231, view_444);  permute_231 = None
    bmm_41: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_444, permute_232);  view_444 = permute_232 = None
    view_445: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_40, [1, 8, 1024, 64]);  bmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_98: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_24, view_445);  tangents_24 = view_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_446: "f32[1, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_41, [1, 8, 1024, 1024]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_14: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_115, torch.float32);  getitem_115 = None
    mul_113: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
    mul_114: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_446, mul_113);  view_446 = mul_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_115: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_114, alias_69);  mul_114 = None
    sum_30: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_115, [-1], True)
    mul_116: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_69, sum_30);  alias_69 = sum_30 = None
    sub_27: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_115, mul_116);  mul_115 = mul_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_2: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_27, 0);  sub_27 = None
    as_strided_scatter_2: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_2, [8, 1024, 1024], [1048576, 1024, 1], 0);  squeeze_2 = None
    as_strided_10: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_2, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_2 = None
    new_empty_strided_1: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_10, [8, 1024, 1024], [1048576, 1024, 1])
    as_strided_12: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_10, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    as_strided_scatter_3: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(as_strided_10, as_strided_12, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  as_strided_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    bmm_42: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_233, as_strided_scatter_3);  permute_233 = None
    bmm_43: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_3, permute_234);  as_strided_scatter_3 = permute_234 = None
    view_447: "f32[1, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_42, [1, 8, 64, 1024]);  bmm_42 = None
    view_448: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_43, [1, 8, 1024, 64]);  bmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_235: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_447, [0, 1, 3, 2]);  view_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_99: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_23, permute_235);  tangents_23 = permute_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_236: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_98, [0, 2, 1, 3]);  add_98 = None
    clone_30: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_236, memory_format = torch.contiguous_format);  permute_236 = None
    view_449: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_30, [1, 1024, 512]);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_450: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_449, [1024, 512]);  view_449 = None
    permute_237: "f32[512, 1024]" = torch.ops.aten.permute.default(view_450, [1, 0])
    mm_113: "f32[512, 512]" = torch.ops.aten.mm.default(permute_237, view_366);  permute_237 = None
    permute_238: "f32[512, 512]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    mm_114: "f32[1024, 512]" = torch.ops.aten.mm.default(view_450, permute_239);  view_450 = permute_239 = None
    view_451: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_114, [1, 1024, 512]);  mm_114 = None
    permute_240: "f32[512, 512]" = torch.ops.aten.permute.default(permute_238, [1, 0]);  permute_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_241: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_99, [0, 2, 1, 3]);  add_99 = None
    clone_31: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_241, memory_format = torch.contiguous_format);  permute_241 = None
    view_452: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_31, [1, 1024, 512]);  clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_453: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_452, [1024, 512]);  view_452 = None
    permute_242: "f32[512, 1024]" = torch.ops.aten.permute.default(view_453, [1, 0])
    mm_115: "f32[512, 512]" = torch.ops.aten.mm.default(permute_242, view_366);  permute_242 = None
    permute_243: "f32[512, 512]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    mm_116: "f32[1024, 512]" = torch.ops.aten.mm.default(view_453, permute_244);  view_453 = permute_244 = None
    view_454: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_116, [1, 1024, 512]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_100: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(view_451, view_454);  view_451 = view_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_245: "f32[512, 512]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_246: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None
    clone_32: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_246, memory_format = torch.contiguous_format);  permute_246 = None
    view_455: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_32, [1, 1024, 512]);  clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_456: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_455, [1024, 512]);  view_455 = None
    permute_247: "f32[512, 1024]" = torch.ops.aten.permute.default(view_456, [1, 0])
    mm_117: "f32[512, 512]" = torch.ops.aten.mm.default(permute_247, view_366);  permute_247 = view_366 = None
    permute_248: "f32[512, 512]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    mm_118: "f32[1024, 512]" = torch.ops.aten.mm.default(view_456, permute_249);  view_456 = permute_249 = None
    view_457: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_118, [1, 1024, 512]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_101: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_100, view_457);  add_100 = view_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_250: "f32[512, 512]" = torch.ops.aten.permute.default(permute_248, [1, 0]);  permute_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_117: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_101, primals_29);  primals_29 = None
    mul_118: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_101, mul_63);  add_101 = mul_63 = None
    sum_31: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_118, [0, 1], True);  mul_118 = None
    view_458: "f32[512]" = torch.ops.aten.reshape.default(sum_31, [512]);  sum_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_119: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_117, add_78)
    mul_120: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_117, rsqrt_28);  mul_117 = None
    sum_32: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_119, [2], True);  mul_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_102: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_97, mul_120);  add_97 = mul_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_39: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_28, 3);  rsqrt_28 = None
    mul_121: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_32, -0.5);  sum_32 = None
    mul_122: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_121, pow_39);  mul_121 = pow_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_75: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_122, [1, 1024, 512]);  mul_122 = None
    div_27: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_75, 512);  expand_75 = None
    pow_40: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_78, 1.0);  add_78 = None
    mul_123: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_40, 2.0);  pow_40 = None
    mul_124: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_27, mul_123);  div_27 = mul_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_103: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_102, mul_124);  add_102 = mul_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_15: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_113, torch.float32);  getitem_113 = None
    mul_125: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.1111111111111112);  convert_element_type_15 = None
    mul_126: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_103, mul_125);  mul_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_459: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_126, [1024, 512]);  mul_126 = None
    permute_251: "f32[512, 1024]" = torch.ops.aten.permute.default(view_459, [1, 0])
    mm_119: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_251, view_364);  permute_251 = view_364 = None
    permute_252: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    mm_120: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_459, permute_253);  view_459 = permute_253 = None
    view_460: "f32[1, 1024, 2048]" = torch.ops.aten.reshape.default(mm_120, [1, 1024, 2048]);  mm_120 = None
    permute_254: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_16: "f32[1, 1024, 2048]" = torch.ops.prims.convert_element_type.default(getitem_111, torch.float32);  getitem_111 = None
    mul_127: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
    mul_128: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_460, mul_127);  view_460 = mul_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    where_7: "f32[1, 1024, 2048]" = torch.ops.aten.where.self(le_2, full_default_7, mul_128);  le_2 = mul_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_461: "f32[1024, 2048]" = torch.ops.aten.reshape.default(where_7, [1024, 2048]);  where_7 = None
    permute_255: "f32[2048, 1024]" = torch.ops.aten.permute.default(view_461, [1, 0])
    mm_121: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_255, view_362);  permute_255 = view_362 = None
    permute_256: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    mm_122: "f32[1024, 512]" = torch.ops.aten.mm.default(view_461, permute_257);  view_461 = permute_257 = None
    view_462: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_122, [1, 1024, 512]);  mm_122 = None
    permute_258: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_256, [1, 0]);  permute_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_129: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_462, primals_28);  primals_28 = None
    mul_130: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_462, mul_61);  view_462 = mul_61 = None
    sum_33: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_130, [0, 1], True);  mul_130 = None
    view_463: "f32[512]" = torch.ops.aten.reshape.default(sum_33, [512]);  sum_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_131: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_129, add_76)
    mul_132: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_129, rsqrt_27);  mul_129 = None
    sum_34: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_131, [2], True);  mul_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_104: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_103, mul_132);  add_103 = mul_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_41: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_27, 3);  rsqrt_27 = None
    mul_133: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_34, -0.5);  sum_34 = None
    mul_134: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_133, pow_41);  mul_133 = pow_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_76: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_134, [1, 1024, 512]);  mul_134 = None
    div_28: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_76, 512);  expand_76 = None
    pow_42: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_76, 1.0);  add_76 = None
    mul_135: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_42, 2.0);  pow_42 = None
    mul_136: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_28, mul_135);  div_28 = mul_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_105: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_104, mul_136);  add_104 = mul_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:647, code: layer_output = hidden_states + self.dropout(attention_output[0])
    convert_element_type_17: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_109, torch.float32);  getitem_109 = None
    mul_137: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
    mul_138: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_105, mul_137);  mul_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_464: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_138, [1024, 512]);  mul_138 = None
    permute_259: "f32[512, 1024]" = torch.ops.aten.permute.default(view_464, [1, 0])
    mm_123: "f32[512, 512]" = torch.ops.aten.mm.default(permute_259, view_360);  permute_259 = view_360 = None
    permute_260: "f32[512, 512]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    mm_124: "f32[1024, 512]" = torch.ops.aten.mm.default(view_464, permute_261);  view_464 = permute_261 = None
    view_465: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_124, [1, 1024, 512]);  mm_124 = None
    permute_262: "f32[512, 512]" = torch.ops.aten.permute.default(permute_260, [1, 0]);  permute_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_466: "f32[1, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_465, [1, 1024, 8, 64]);  view_465 = None
    permute_263: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_466, [0, 2, 1, 3]);  view_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_467: "f32[8, 1024, 64]" = torch.ops.aten.reshape.default(permute_263, [8, 1024, 64]);  permute_263 = None
    bmm_44: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_264, view_467);  permute_264 = None
    bmm_45: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_467, permute_265);  view_467 = permute_265 = None
    view_468: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_44, [1, 8, 1024, 64]);  bmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_106: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_22, view_468);  tangents_22 = view_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_469: "f32[1, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_45, [1, 8, 1024, 1024]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_18: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_107, torch.float32);  getitem_107 = None
    mul_139: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.1111111111111112);  convert_element_type_18 = None
    mul_140: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_469, mul_139);  view_469 = mul_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_141: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_140, alias_73);  mul_140 = None
    sum_35: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_141, [-1], True)
    mul_142: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_73, sum_35);  alias_73 = sum_35 = None
    sub_28: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_3: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_28, 0);  sub_28 = None
    as_strided_scatter_4: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_3, [8, 1024, 1024], [1048576, 1024, 1], 0);  squeeze_3 = None
    as_strided_17: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_4, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_4 = None
    new_empty_strided_2: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_17, [8, 1024, 1024], [1048576, 1024, 1])
    as_strided_19: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_17, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    as_strided_scatter_5: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(as_strided_17, as_strided_19, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  as_strided_17 = as_strided_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    bmm_46: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_266, as_strided_scatter_5);  permute_266 = None
    bmm_47: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_5, permute_267);  as_strided_scatter_5 = permute_267 = None
    view_470: "f32[1, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_46, [1, 8, 64, 1024]);  bmm_46 = None
    view_471: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_47, [1, 8, 1024, 64]);  bmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_268: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_470, [0, 1, 3, 2]);  view_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_107: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_21, permute_268);  tangents_21 = permute_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_269: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_106, [0, 2, 1, 3]);  add_106 = None
    clone_38: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_269, memory_format = torch.contiguous_format);  permute_269 = None
    view_472: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_38, [1, 1024, 512]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_473: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_472, [1024, 512]);  view_472 = None
    permute_270: "f32[512, 1024]" = torch.ops.aten.permute.default(view_473, [1, 0])
    mm_125: "f32[512, 512]" = torch.ops.aten.mm.default(permute_270, view_169);  permute_270 = None
    permute_271: "f32[512, 512]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    mm_126: "f32[1024, 512]" = torch.ops.aten.mm.default(view_473, permute_272);  view_473 = permute_272 = None
    view_474: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_126, [1, 1024, 512]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_108: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_95, view_474);  add_95 = view_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_273: "f32[512, 512]" = torch.ops.aten.permute.default(permute_271, [1, 0]);  permute_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_274: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_107, [0, 2, 1, 3]);  add_107 = None
    clone_39: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_274, memory_format = torch.contiguous_format);  permute_274 = None
    view_475: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_39, [1, 1024, 512]);  clone_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_476: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_475, [1024, 512]);  view_475 = None
    permute_275: "f32[512, 1024]" = torch.ops.aten.permute.default(view_476, [1, 0])
    mm_127: "f32[512, 512]" = torch.ops.aten.mm.default(permute_275, view_169);  permute_275 = None
    permute_276: "f32[512, 512]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    mm_128: "f32[1024, 512]" = torch.ops.aten.mm.default(view_476, permute_277);  view_476 = permute_277 = None
    view_477: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_128, [1, 1024, 512]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_109: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_108, view_477);  add_108 = view_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_278: "f32[512, 512]" = torch.ops.aten.permute.default(permute_276, [1, 0]);  permute_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_279: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_471, [0, 2, 1, 3]);  view_471 = None
    clone_40: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_279, memory_format = torch.contiguous_format);  permute_279 = None
    view_478: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_40, [1, 1024, 512]);  clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_479: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_478, [1024, 512]);  view_478 = None
    permute_280: "f32[512, 1024]" = torch.ops.aten.permute.default(view_479, [1, 0])
    mm_129: "f32[512, 512]" = torch.ops.aten.mm.default(permute_280, view_342);  permute_280 = view_342 = None
    permute_281: "f32[512, 512]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    mm_130: "f32[1024, 512]" = torch.ops.aten.mm.default(view_479, permute_282);  view_479 = permute_282 = None
    view_480: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_130, [1, 1024, 512]);  mm_130 = None
    permute_283: "f32[512, 512]" = torch.ops.aten.permute.default(permute_281, [1, 0]);  permute_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_143: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_480, primals_27);  primals_27 = None
    mul_144: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_480, mul_59);  view_480 = mul_59 = None
    sum_36: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_144, [0, 1], True);  mul_144 = None
    view_481: "f32[512]" = torch.ops.aten.reshape.default(sum_36, [512]);  sum_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_145: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_143, add_73)
    mul_146: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_143, rsqrt_26);  mul_143 = None
    sum_37: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_145, [2], True);  mul_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_110: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_105, mul_146);  add_105 = mul_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_43: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_26, 3);  rsqrt_26 = None
    mul_147: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_37, -0.5);  sum_37 = None
    mul_148: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_147, pow_43);  mul_147 = pow_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_77: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_148, [1, 1024, 512]);  mul_148 = None
    div_29: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_77, 512);  expand_77 = None
    pow_44: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_73, 1.0);  add_73 = None
    mul_149: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_44, 2.0);  pow_44 = None
    mul_150: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_29, mul_149);  div_29 = mul_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_111: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_110, mul_150);  add_110 = mul_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_19: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_105, torch.float32);  getitem_105 = None
    mul_151: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
    mul_152: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_111, mul_151);  mul_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_482: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_152, [1024, 512]);  mul_152 = None
    permute_284: "f32[512, 1024]" = torch.ops.aten.permute.default(view_482, [1, 0])
    mm_131: "f32[512, 512]" = torch.ops.aten.mm.default(permute_284, view_340);  permute_284 = view_340 = None
    permute_285: "f32[512, 512]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    mm_132: "f32[1024, 512]" = torch.ops.aten.mm.default(view_482, permute_286);  view_482 = permute_286 = None
    view_483: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_132, [1, 1024, 512]);  mm_132 = None
    permute_287: "f32[512, 512]" = torch.ops.aten.permute.default(permute_285, [1, 0]);  permute_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_484: "f32[1, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_483, [1, 1024, 8, 64]);  view_483 = None
    permute_288: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_484, [0, 2, 1, 3]);  view_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_485: "f32[8, 1024, 64]" = torch.ops.aten.reshape.default(permute_288, [8, 1024, 64]);  permute_288 = None
    bmm_48: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_289, view_485);  permute_289 = None
    bmm_49: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_485, permute_290);  view_485 = permute_290 = None
    view_486: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_48, [1, 8, 1024, 64]);  bmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_112: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_20, view_486);  tangents_20 = view_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_487: "f32[1, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_49, [1, 8, 1024, 1024]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_20: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_103, torch.float32);  getitem_103 = None
    mul_153: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
    mul_154: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_487, mul_153);  view_487 = mul_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_155: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_154, alias_75);  mul_154 = None
    sum_38: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_155, [-1], True)
    mul_156: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_75, sum_38);  alias_75 = sum_38 = None
    sub_29: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_4: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_29, 0);  sub_29 = None
    as_strided_scatter_6: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_4, [8, 1024, 1024], [1048576, 1024, 1], 0);  squeeze_4 = None
    as_strided_24: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_6, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_6 = None
    new_empty_strided_3: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_24, [8, 1024, 1024], [1048576, 1024, 1])
    as_strided_26: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_24, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    as_strided_scatter_7: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(as_strided_24, as_strided_26, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  as_strided_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_113: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(as_strided_12, as_strided_26);  as_strided_12 = as_strided_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    bmm_50: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_291, as_strided_scatter_7);  permute_291 = None
    bmm_51: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_7, permute_292);  as_strided_scatter_7 = permute_292 = None
    view_488: "f32[1, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_50, [1, 8, 64, 1024]);  bmm_50 = None
    view_489: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_51, [1, 8, 1024, 64]);  bmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_293: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_488, [0, 1, 3, 2]);  view_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_114: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_19, permute_293);  tangents_19 = permute_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_294: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_112, [0, 2, 1, 3]);  add_112 = None
    clone_44: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_294, memory_format = torch.contiguous_format);  permute_294 = None
    view_490: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_44, [1, 1024, 512]);  clone_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_491: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_490, [1024, 512]);  view_490 = None
    permute_295: "f32[512, 1024]" = torch.ops.aten.permute.default(view_491, [1, 0])
    mm_133: "f32[512, 512]" = torch.ops.aten.mm.default(permute_295, view_322);  permute_295 = None
    permute_296: "f32[512, 512]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    mm_134: "f32[1024, 512]" = torch.ops.aten.mm.default(view_491, permute_297);  view_491 = permute_297 = None
    view_492: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_134, [1, 1024, 512]);  mm_134 = None
    permute_298: "f32[512, 512]" = torch.ops.aten.permute.default(permute_296, [1, 0]);  permute_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_299: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_114, [0, 2, 1, 3]);  add_114 = None
    clone_45: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_299, memory_format = torch.contiguous_format);  permute_299 = None
    view_493: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_45, [1, 1024, 512]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_494: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_493, [1024, 512]);  view_493 = None
    permute_300: "f32[512, 1024]" = torch.ops.aten.permute.default(view_494, [1, 0])
    mm_135: "f32[512, 512]" = torch.ops.aten.mm.default(permute_300, view_322);  permute_300 = None
    permute_301: "f32[512, 512]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    mm_136: "f32[1024, 512]" = torch.ops.aten.mm.default(view_494, permute_302);  view_494 = permute_302 = None
    view_495: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_136, [1, 1024, 512]);  mm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_115: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(view_492, view_495);  view_492 = view_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_303: "f32[512, 512]" = torch.ops.aten.permute.default(permute_301, [1, 0]);  permute_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_304: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_489, [0, 2, 1, 3]);  view_489 = None
    clone_46: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_304, memory_format = torch.contiguous_format);  permute_304 = None
    view_496: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_46, [1, 1024, 512]);  clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_497: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_496, [1024, 512]);  view_496 = None
    permute_305: "f32[512, 1024]" = torch.ops.aten.permute.default(view_497, [1, 0])
    mm_137: "f32[512, 512]" = torch.ops.aten.mm.default(permute_305, view_322);  permute_305 = view_322 = None
    permute_306: "f32[512, 512]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    mm_138: "f32[1024, 512]" = torch.ops.aten.mm.default(view_497, permute_307);  view_497 = permute_307 = None
    view_498: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_138, [1, 1024, 512]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_116: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_115, view_498);  add_115 = view_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_308: "f32[512, 512]" = torch.ops.aten.permute.default(permute_306, [1, 0]);  permute_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_157: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_116, primals_26);  primals_26 = None
    mul_158: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_116, mul_57);  add_116 = mul_57 = None
    sum_39: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_158, [0, 1], True);  mul_158 = None
    view_499: "f32[512]" = torch.ops.aten.reshape.default(sum_39, [512]);  sum_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_159: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_157, add_70)
    mul_160: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_157, rsqrt_25);  mul_157 = None
    sum_40: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_159, [2], True);  mul_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_117: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_111, mul_160);  add_111 = mul_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_45: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_25, 3);  rsqrt_25 = None
    mul_161: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_40, -0.5);  sum_40 = None
    mul_162: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_161, pow_45);  mul_161 = pow_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_78: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_162, [1, 1024, 512]);  mul_162 = None
    div_30: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_78, 512);  expand_78 = None
    pow_46: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_70, 1.0);  add_70 = None
    mul_163: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_46, 2.0);  pow_46 = None
    mul_164: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_30, mul_163);  div_30 = mul_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_118: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_117, mul_164);  add_117 = mul_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_21: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_101, torch.float32);  getitem_101 = None
    mul_165: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
    mul_166: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_118, mul_165);  mul_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_500: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_166, [1024, 512]);  mul_166 = None
    permute_309: "f32[512, 1024]" = torch.ops.aten.permute.default(view_500, [1, 0])
    mm_139: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_309, view_320);  permute_309 = view_320 = None
    permute_310: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    mm_140: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_500, permute_311);  view_500 = permute_311 = None
    view_501: "f32[1, 1024, 2048]" = torch.ops.aten.reshape.default(mm_140, [1, 1024, 2048]);  mm_140 = None
    permute_312: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_310, [1, 0]);  permute_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_22: "f32[1, 1024, 2048]" = torch.ops.prims.convert_element_type.default(getitem_99, torch.float32);  getitem_99 = None
    mul_167: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
    mul_168: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_501, mul_167);  view_501 = mul_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    where_8: "f32[1, 1024, 2048]" = torch.ops.aten.where.self(le_3, full_default_7, mul_168);  le_3 = mul_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_502: "f32[1024, 2048]" = torch.ops.aten.reshape.default(where_8, [1024, 2048]);  where_8 = None
    permute_313: "f32[2048, 1024]" = torch.ops.aten.permute.default(view_502, [1, 0])
    mm_141: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_313, view_318);  permute_313 = view_318 = None
    permute_314: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    mm_142: "f32[1024, 512]" = torch.ops.aten.mm.default(view_502, permute_315);  view_502 = permute_315 = None
    view_503: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_142, [1, 1024, 512]);  mm_142 = None
    permute_316: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_314, [1, 0]);  permute_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_169: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_503, primals_25);  primals_25 = None
    mul_170: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_503, mul_55);  view_503 = mul_55 = None
    sum_41: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_170, [0, 1], True);  mul_170 = None
    view_504: "f32[512]" = torch.ops.aten.reshape.default(sum_41, [512]);  sum_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_171: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_169, add_68)
    mul_172: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_169, rsqrt_24);  mul_169 = None
    sum_42: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_171, [2], True);  mul_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_119: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_118, mul_172);  add_118 = mul_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_47: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_24, 3);  rsqrt_24 = None
    mul_173: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_42, -0.5);  sum_42 = None
    mul_174: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_173, pow_47);  mul_173 = pow_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_79: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_174, [1, 1024, 512]);  mul_174 = None
    div_31: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_79, 512);  expand_79 = None
    pow_48: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_68, 1.0);  add_68 = None
    mul_175: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_48, 2.0);  pow_48 = None
    mul_176: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_31, mul_175);  div_31 = mul_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_120: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_119, mul_176);  add_119 = mul_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:647, code: layer_output = hidden_states + self.dropout(attention_output[0])
    convert_element_type_23: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_97, torch.float32);  getitem_97 = None
    mul_177: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
    mul_178: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_120, mul_177);  mul_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_505: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_178, [1024, 512]);  mul_178 = None
    permute_317: "f32[512, 1024]" = torch.ops.aten.permute.default(view_505, [1, 0])
    mm_143: "f32[512, 512]" = torch.ops.aten.mm.default(permute_317, view_316);  permute_317 = view_316 = None
    permute_318: "f32[512, 512]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    mm_144: "f32[1024, 512]" = torch.ops.aten.mm.default(view_505, permute_319);  view_505 = permute_319 = None
    view_506: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_144, [1, 1024, 512]);  mm_144 = None
    permute_320: "f32[512, 512]" = torch.ops.aten.permute.default(permute_318, [1, 0]);  permute_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_507: "f32[1, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_506, [1, 1024, 8, 64]);  view_506 = None
    permute_321: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_507, [0, 2, 1, 3]);  view_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_508: "f32[8, 1024, 64]" = torch.ops.aten.reshape.default(permute_321, [8, 1024, 64]);  permute_321 = None
    bmm_52: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_322, view_508);  permute_322 = None
    bmm_53: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_508, permute_323);  view_508 = permute_323 = None
    view_509: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_52, [1, 8, 1024, 64]);  bmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_121: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_18, view_509);  tangents_18 = view_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_510: "f32[1, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_53, [1, 8, 1024, 1024]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_24: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_95, torch.float32);  getitem_95 = None
    mul_179: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.1111111111111112);  convert_element_type_24 = None
    mul_180: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_510, mul_179);  view_510 = mul_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_181: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_180, alias_79);  mul_180 = None
    sum_43: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_181, [-1], True)
    mul_182: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_79, sum_43);  alias_79 = sum_43 = None
    sub_30: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_181, mul_182);  mul_181 = mul_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_5: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_30, 0);  sub_30 = None
    as_strided_scatter_8: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_5, [8, 1024, 1024], [1048576, 1024, 1], 0);  squeeze_5 = None
    as_strided_31: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_8, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_8 = None
    new_empty_strided_4: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_31, [8, 1024, 1024], [1048576, 1024, 1])
    as_strided_33: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_31, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    as_strided_scatter_9: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(as_strided_31, as_strided_33, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  as_strided_31 = as_strided_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    bmm_54: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_324, as_strided_scatter_9);  permute_324 = None
    bmm_55: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_9, permute_325);  as_strided_scatter_9 = permute_325 = None
    view_511: "f32[1, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_54, [1, 8, 64, 1024]);  bmm_54 = None
    view_512: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_55, [1, 8, 1024, 64]);  bmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_326: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_511, [0, 1, 3, 2]);  view_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_122: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_17, permute_326);  tangents_17 = permute_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_327: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_121, [0, 2, 1, 3]);  add_121 = None
    clone_52: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_327, memory_format = torch.contiguous_format);  permute_327 = None
    view_513: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_52, [1, 1024, 512]);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_514: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_513, [1024, 512]);  view_513 = None
    permute_328: "f32[512, 1024]" = torch.ops.aten.permute.default(view_514, [1, 0])
    mm_145: "f32[512, 512]" = torch.ops.aten.mm.default(permute_328, view_169);  permute_328 = None
    permute_329: "f32[512, 512]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    mm_146: "f32[1024, 512]" = torch.ops.aten.mm.default(view_514, permute_330);  view_514 = permute_330 = None
    view_515: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_146, [1, 1024, 512]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_123: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_109, view_515);  add_109 = view_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_331: "f32[512, 512]" = torch.ops.aten.permute.default(permute_329, [1, 0]);  permute_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_332: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_122, [0, 2, 1, 3]);  add_122 = None
    clone_53: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_332, memory_format = torch.contiguous_format);  permute_332 = None
    view_516: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_53, [1, 1024, 512]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_517: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_516, [1024, 512]);  view_516 = None
    permute_333: "f32[512, 1024]" = torch.ops.aten.permute.default(view_517, [1, 0])
    mm_147: "f32[512, 512]" = torch.ops.aten.mm.default(permute_333, view_169);  permute_333 = None
    permute_334: "f32[512, 512]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    mm_148: "f32[1024, 512]" = torch.ops.aten.mm.default(view_517, permute_335);  view_517 = permute_335 = None
    view_518: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_148, [1, 1024, 512]);  mm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_124: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_123, view_518);  add_123 = view_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_336: "f32[512, 512]" = torch.ops.aten.permute.default(permute_334, [1, 0]);  permute_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_337: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None
    clone_54: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_337, memory_format = torch.contiguous_format);  permute_337 = None
    view_519: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_54, [1, 1024, 512]);  clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_520: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_519, [1024, 512]);  view_519 = None
    permute_338: "f32[512, 1024]" = torch.ops.aten.permute.default(view_520, [1, 0])
    mm_149: "f32[512, 512]" = torch.ops.aten.mm.default(permute_338, view_298);  permute_338 = view_298 = None
    permute_339: "f32[512, 512]" = torch.ops.aten.permute.default(mm_149, [1, 0]);  mm_149 = None
    mm_150: "f32[1024, 512]" = torch.ops.aten.mm.default(view_520, permute_340);  view_520 = permute_340 = None
    view_521: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_150, [1, 1024, 512]);  mm_150 = None
    permute_341: "f32[512, 512]" = torch.ops.aten.permute.default(permute_339, [1, 0]);  permute_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_183: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_521, primals_24);  primals_24 = None
    mul_184: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_521, mul_53);  view_521 = mul_53 = None
    sum_44: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_184, [0, 1], True);  mul_184 = None
    view_522: "f32[512]" = torch.ops.aten.reshape.default(sum_44, [512]);  sum_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_185: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_183, add_65)
    mul_186: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_183, rsqrt_23);  mul_183 = None
    sum_45: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_185, [2], True);  mul_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_125: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_120, mul_186);  add_120 = mul_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_49: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_23, 3);  rsqrt_23 = None
    mul_187: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_45, -0.5);  sum_45 = None
    mul_188: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_187, pow_49);  mul_187 = pow_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_80: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_188, [1, 1024, 512]);  mul_188 = None
    div_32: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_80, 512);  expand_80 = None
    pow_50: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_65, 1.0);  add_65 = None
    mul_189: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_50, 2.0);  pow_50 = None
    mul_190: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_32, mul_189);  div_32 = mul_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_126: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_125, mul_190);  add_125 = mul_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_25: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_93, torch.float32);  getitem_93 = None
    mul_191: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
    mul_192: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_126, mul_191);  mul_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_523: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_192, [1024, 512]);  mul_192 = None
    permute_342: "f32[512, 1024]" = torch.ops.aten.permute.default(view_523, [1, 0])
    mm_151: "f32[512, 512]" = torch.ops.aten.mm.default(permute_342, view_296);  permute_342 = view_296 = None
    permute_343: "f32[512, 512]" = torch.ops.aten.permute.default(mm_151, [1, 0]);  mm_151 = None
    mm_152: "f32[1024, 512]" = torch.ops.aten.mm.default(view_523, permute_344);  view_523 = permute_344 = None
    view_524: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_152, [1, 1024, 512]);  mm_152 = None
    permute_345: "f32[512, 512]" = torch.ops.aten.permute.default(permute_343, [1, 0]);  permute_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_525: "f32[1, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_524, [1, 1024, 8, 64]);  view_524 = None
    permute_346: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_525, [0, 2, 1, 3]);  view_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_526: "f32[8, 1024, 64]" = torch.ops.aten.reshape.default(permute_346, [8, 1024, 64]);  permute_346 = None
    bmm_56: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_347, view_526);  permute_347 = None
    bmm_57: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_526, permute_348);  view_526 = permute_348 = None
    view_527: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_56, [1, 8, 1024, 64]);  bmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_127: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_16, view_527);  tangents_16 = view_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_528: "f32[1, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_57, [1, 8, 1024, 1024]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_26: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_91, torch.float32);  getitem_91 = None
    mul_193: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
    mul_194: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_528, mul_193);  view_528 = mul_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_195: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_194, alias_81);  mul_194 = None
    sum_46: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_195, [-1], True)
    mul_196: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_81, sum_46);  alias_81 = sum_46 = None
    sub_31: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_195, mul_196);  mul_195 = mul_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_6: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_31, 0);  sub_31 = None
    as_strided_scatter_10: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_6, [8, 1024, 1024], [1048576, 1024, 1], 0);  squeeze_6 = None
    as_strided_38: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_10, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_10 = None
    new_empty_strided_5: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_38, [8, 1024, 1024], [1048576, 1024, 1])
    as_strided_40: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_38, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    as_strided_scatter_11: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(as_strided_38, as_strided_40, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  as_strided_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_128: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_113, as_strided_40);  add_113 = as_strided_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    bmm_58: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_349, as_strided_scatter_11);  permute_349 = None
    bmm_59: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_11, permute_350);  as_strided_scatter_11 = permute_350 = None
    view_529: "f32[1, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_58, [1, 8, 64, 1024]);  bmm_58 = None
    view_530: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_59, [1, 8, 1024, 64]);  bmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_351: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_529, [0, 1, 3, 2]);  view_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_129: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_15, permute_351);  tangents_15 = permute_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_352: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_127, [0, 2, 1, 3]);  add_127 = None
    clone_58: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_352, memory_format = torch.contiguous_format);  permute_352 = None
    view_531: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_58, [1, 1024, 512]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_532: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_531, [1024, 512]);  view_531 = None
    permute_353: "f32[512, 1024]" = torch.ops.aten.permute.default(view_532, [1, 0])
    mm_153: "f32[512, 512]" = torch.ops.aten.mm.default(permute_353, view_278);  permute_353 = None
    permute_354: "f32[512, 512]" = torch.ops.aten.permute.default(mm_153, [1, 0]);  mm_153 = None
    mm_154: "f32[1024, 512]" = torch.ops.aten.mm.default(view_532, permute_355);  view_532 = permute_355 = None
    view_533: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_154, [1, 1024, 512]);  mm_154 = None
    permute_356: "f32[512, 512]" = torch.ops.aten.permute.default(permute_354, [1, 0]);  permute_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_357: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_129, [0, 2, 1, 3]);  add_129 = None
    clone_59: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_357, memory_format = torch.contiguous_format);  permute_357 = None
    view_534: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_59, [1, 1024, 512]);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_535: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_534, [1024, 512]);  view_534 = None
    permute_358: "f32[512, 1024]" = torch.ops.aten.permute.default(view_535, [1, 0])
    mm_155: "f32[512, 512]" = torch.ops.aten.mm.default(permute_358, view_278);  permute_358 = None
    permute_359: "f32[512, 512]" = torch.ops.aten.permute.default(mm_155, [1, 0]);  mm_155 = None
    mm_156: "f32[1024, 512]" = torch.ops.aten.mm.default(view_535, permute_360);  view_535 = permute_360 = None
    view_536: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_156, [1, 1024, 512]);  mm_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_130: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(view_533, view_536);  view_533 = view_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_361: "f32[512, 512]" = torch.ops.aten.permute.default(permute_359, [1, 0]);  permute_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_362: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_530, [0, 2, 1, 3]);  view_530 = None
    clone_60: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_362, memory_format = torch.contiguous_format);  permute_362 = None
    view_537: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_60, [1, 1024, 512]);  clone_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_538: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_537, [1024, 512]);  view_537 = None
    permute_363: "f32[512, 1024]" = torch.ops.aten.permute.default(view_538, [1, 0])
    mm_157: "f32[512, 512]" = torch.ops.aten.mm.default(permute_363, view_278);  permute_363 = view_278 = None
    permute_364: "f32[512, 512]" = torch.ops.aten.permute.default(mm_157, [1, 0]);  mm_157 = None
    mm_158: "f32[1024, 512]" = torch.ops.aten.mm.default(view_538, permute_365);  view_538 = permute_365 = None
    view_539: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_158, [1, 1024, 512]);  mm_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_131: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_130, view_539);  add_130 = view_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_366: "f32[512, 512]" = torch.ops.aten.permute.default(permute_364, [1, 0]);  permute_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_197: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_131, primals_23);  primals_23 = None
    mul_198: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_131, mul_51);  add_131 = mul_51 = None
    sum_47: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_198, [0, 1], True);  mul_198 = None
    view_540: "f32[512]" = torch.ops.aten.reshape.default(sum_47, [512]);  sum_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_199: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_197, add_62)
    mul_200: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_197, rsqrt_22);  mul_197 = None
    sum_48: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_199, [2], True);  mul_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_132: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_126, mul_200);  add_126 = mul_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_51: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_22, 3);  rsqrt_22 = None
    mul_201: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_48, -0.5);  sum_48 = None
    mul_202: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_201, pow_51);  mul_201 = pow_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_81: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_202, [1, 1024, 512]);  mul_202 = None
    div_33: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_81, 512);  expand_81 = None
    pow_52: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_62, 1.0);  add_62 = None
    mul_203: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_52, 2.0);  pow_52 = None
    mul_204: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_33, mul_203);  div_33 = mul_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_133: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_132, mul_204);  add_132 = mul_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_27: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_89, torch.float32);  getitem_89 = None
    mul_205: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_27, 1.1111111111111112);  convert_element_type_27 = None
    mul_206: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_133, mul_205);  mul_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_541: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_206, [1024, 512]);  mul_206 = None
    permute_367: "f32[512, 1024]" = torch.ops.aten.permute.default(view_541, [1, 0])
    mm_159: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_367, view_276);  permute_367 = view_276 = None
    permute_368: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_159, [1, 0]);  mm_159 = None
    mm_160: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_541, permute_369);  view_541 = permute_369 = None
    view_542: "f32[1, 1024, 2048]" = torch.ops.aten.reshape.default(mm_160, [1, 1024, 2048]);  mm_160 = None
    permute_370: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_368, [1, 0]);  permute_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_28: "f32[1, 1024, 2048]" = torch.ops.prims.convert_element_type.default(getitem_87, torch.float32);  getitem_87 = None
    mul_207: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.1111111111111112);  convert_element_type_28 = None
    mul_208: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_542, mul_207);  view_542 = mul_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    where_9: "f32[1, 1024, 2048]" = torch.ops.aten.where.self(le_4, full_default_7, mul_208);  le_4 = mul_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_543: "f32[1024, 2048]" = torch.ops.aten.reshape.default(where_9, [1024, 2048]);  where_9 = None
    permute_371: "f32[2048, 1024]" = torch.ops.aten.permute.default(view_543, [1, 0])
    mm_161: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_371, view_274);  permute_371 = view_274 = None
    permute_372: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_161, [1, 0]);  mm_161 = None
    mm_162: "f32[1024, 512]" = torch.ops.aten.mm.default(view_543, permute_373);  view_543 = permute_373 = None
    view_544: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_162, [1, 1024, 512]);  mm_162 = None
    permute_374: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_372, [1, 0]);  permute_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_209: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_544, primals_22);  primals_22 = None
    mul_210: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_544, mul_49);  view_544 = mul_49 = None
    sum_49: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_210, [0, 1], True);  mul_210 = None
    view_545: "f32[512]" = torch.ops.aten.reshape.default(sum_49, [512]);  sum_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_211: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_209, add_60)
    mul_212: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_209, rsqrt_21);  mul_209 = None
    sum_50: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_211, [2], True);  mul_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_134: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_133, mul_212);  add_133 = mul_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_53: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_21, 3);  rsqrt_21 = None
    mul_213: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_50, -0.5);  sum_50 = None
    mul_214: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_213, pow_53);  mul_213 = pow_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_82: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_214, [1, 1024, 512]);  mul_214 = None
    div_34: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_82, 512);  expand_82 = None
    pow_54: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_60, 1.0);  add_60 = None
    mul_215: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_54, 2.0);  pow_54 = None
    mul_216: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_34, mul_215);  div_34 = mul_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_135: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_134, mul_216);  add_134 = mul_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:647, code: layer_output = hidden_states + self.dropout(attention_output[0])
    convert_element_type_29: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_85, torch.float32);  getitem_85 = None
    mul_217: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
    mul_218: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_135, mul_217);  mul_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_546: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_218, [1024, 512]);  mul_218 = None
    permute_375: "f32[512, 1024]" = torch.ops.aten.permute.default(view_546, [1, 0])
    mm_163: "f32[512, 512]" = torch.ops.aten.mm.default(permute_375, view_272);  permute_375 = view_272 = None
    permute_376: "f32[512, 512]" = torch.ops.aten.permute.default(mm_163, [1, 0]);  mm_163 = None
    mm_164: "f32[1024, 512]" = torch.ops.aten.mm.default(view_546, permute_377);  view_546 = permute_377 = None
    view_547: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_164, [1, 1024, 512]);  mm_164 = None
    permute_378: "f32[512, 512]" = torch.ops.aten.permute.default(permute_376, [1, 0]);  permute_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_548: "f32[1, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_547, [1, 1024, 8, 64]);  view_547 = None
    permute_379: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_548, [0, 2, 1, 3]);  view_548 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_549: "f32[8, 1024, 64]" = torch.ops.aten.reshape.default(permute_379, [8, 1024, 64]);  permute_379 = None
    bmm_60: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_380, view_549);  permute_380 = None
    bmm_61: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_549, permute_381);  view_549 = permute_381 = None
    view_550: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_60, [1, 8, 1024, 64]);  bmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_136: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_14, view_550);  tangents_14 = view_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_551: "f32[1, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_61, [1, 8, 1024, 1024]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_30: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_83, torch.float32);  getitem_83 = None
    mul_219: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 1.1111111111111112);  convert_element_type_30 = None
    mul_220: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_551, mul_219);  view_551 = mul_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_221: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_220, alias_85);  mul_220 = None
    sum_51: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_221, [-1], True)
    mul_222: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_85, sum_51);  alias_85 = sum_51 = None
    sub_32: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_7: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_32, 0);  sub_32 = None
    as_strided_scatter_12: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_7, [8, 1024, 1024], [1048576, 1024, 1], 0);  squeeze_7 = None
    as_strided_45: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_12, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_12 = None
    new_empty_strided_6: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_45, [8, 1024, 1024], [1048576, 1024, 1])
    as_strided_47: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_45, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    as_strided_scatter_13: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(as_strided_45, as_strided_47, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  as_strided_45 = as_strided_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    bmm_62: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_382, as_strided_scatter_13);  permute_382 = None
    bmm_63: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_13, permute_383);  as_strided_scatter_13 = permute_383 = None
    view_552: "f32[1, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_62, [1, 8, 64, 1024]);  bmm_62 = None
    view_553: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_63, [1, 8, 1024, 64]);  bmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_384: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_552, [0, 1, 3, 2]);  view_552 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_137: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_13, permute_384);  tangents_13 = permute_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_385: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_136, [0, 2, 1, 3]);  add_136 = None
    clone_66: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_385, memory_format = torch.contiguous_format);  permute_385 = None
    view_554: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_66, [1, 1024, 512]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_555: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_554, [1024, 512]);  view_554 = None
    permute_386: "f32[512, 1024]" = torch.ops.aten.permute.default(view_555, [1, 0])
    mm_165: "f32[512, 512]" = torch.ops.aten.mm.default(permute_386, view_169);  permute_386 = None
    permute_387: "f32[512, 512]" = torch.ops.aten.permute.default(mm_165, [1, 0]);  mm_165 = None
    mm_166: "f32[1024, 512]" = torch.ops.aten.mm.default(view_555, permute_388);  view_555 = permute_388 = None
    view_556: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_166, [1, 1024, 512]);  mm_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_138: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_124, view_556);  add_124 = view_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_389: "f32[512, 512]" = torch.ops.aten.permute.default(permute_387, [1, 0]);  permute_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_390: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_137, [0, 2, 1, 3]);  add_137 = None
    clone_67: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_390, memory_format = torch.contiguous_format);  permute_390 = None
    view_557: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_67, [1, 1024, 512]);  clone_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_558: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_557, [1024, 512]);  view_557 = None
    permute_391: "f32[512, 1024]" = torch.ops.aten.permute.default(view_558, [1, 0])
    mm_167: "f32[512, 512]" = torch.ops.aten.mm.default(permute_391, view_169);  permute_391 = None
    permute_392: "f32[512, 512]" = torch.ops.aten.permute.default(mm_167, [1, 0]);  mm_167 = None
    mm_168: "f32[1024, 512]" = torch.ops.aten.mm.default(view_558, permute_393);  view_558 = permute_393 = None
    view_559: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_168, [1, 1024, 512]);  mm_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_139: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_138, view_559);  add_138 = view_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_394: "f32[512, 512]" = torch.ops.aten.permute.default(permute_392, [1, 0]);  permute_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_395: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_553, [0, 2, 1, 3]);  view_553 = None
    clone_68: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_395, memory_format = torch.contiguous_format);  permute_395 = None
    view_560: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_68, [1, 1024, 512]);  clone_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_561: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_560, [1024, 512]);  view_560 = None
    permute_396: "f32[512, 1024]" = torch.ops.aten.permute.default(view_561, [1, 0])
    mm_169: "f32[512, 512]" = torch.ops.aten.mm.default(permute_396, view_254);  permute_396 = view_254 = None
    permute_397: "f32[512, 512]" = torch.ops.aten.permute.default(mm_169, [1, 0]);  mm_169 = None
    mm_170: "f32[1024, 512]" = torch.ops.aten.mm.default(view_561, permute_398);  view_561 = permute_398 = None
    view_562: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_170, [1, 1024, 512]);  mm_170 = None
    permute_399: "f32[512, 512]" = torch.ops.aten.permute.default(permute_397, [1, 0]);  permute_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_223: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_562, primals_21);  primals_21 = None
    mul_224: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_562, mul_47);  view_562 = mul_47 = None
    sum_52: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_224, [0, 1], True);  mul_224 = None
    view_563: "f32[512]" = torch.ops.aten.reshape.default(sum_52, [512]);  sum_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_225: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_223, add_57)
    mul_226: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_223, rsqrt_20);  mul_223 = None
    sum_53: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_225, [2], True);  mul_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_140: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_135, mul_226);  add_135 = mul_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_55: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_20, 3);  rsqrt_20 = None
    mul_227: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_53, -0.5);  sum_53 = None
    mul_228: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_227, pow_55);  mul_227 = pow_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_83: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_228, [1, 1024, 512]);  mul_228 = None
    div_35: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_83, 512);  expand_83 = None
    pow_56: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_57, 1.0);  add_57 = None
    mul_229: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_56, 2.0);  pow_56 = None
    mul_230: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_35, mul_229);  div_35 = mul_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_141: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_140, mul_230);  add_140 = mul_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_31: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_81, torch.float32);  getitem_81 = None
    mul_231: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.1111111111111112);  convert_element_type_31 = None
    mul_232: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_141, mul_231);  mul_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_564: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_232, [1024, 512]);  mul_232 = None
    permute_400: "f32[512, 1024]" = torch.ops.aten.permute.default(view_564, [1, 0])
    mm_171: "f32[512, 512]" = torch.ops.aten.mm.default(permute_400, view_252);  permute_400 = view_252 = None
    permute_401: "f32[512, 512]" = torch.ops.aten.permute.default(mm_171, [1, 0]);  mm_171 = None
    mm_172: "f32[1024, 512]" = torch.ops.aten.mm.default(view_564, permute_402);  view_564 = permute_402 = None
    view_565: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_172, [1, 1024, 512]);  mm_172 = None
    permute_403: "f32[512, 512]" = torch.ops.aten.permute.default(permute_401, [1, 0]);  permute_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_566: "f32[1, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_565, [1, 1024, 8, 64]);  view_565 = None
    permute_404: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_566, [0, 2, 1, 3]);  view_566 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_567: "f32[8, 1024, 64]" = torch.ops.aten.reshape.default(permute_404, [8, 1024, 64]);  permute_404 = None
    bmm_64: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_405, view_567);  permute_405 = None
    bmm_65: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_567, permute_406);  view_567 = permute_406 = None
    view_568: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_64, [1, 8, 1024, 64]);  bmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_142: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_12, view_568);  tangents_12 = view_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_569: "f32[1, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_65, [1, 8, 1024, 1024]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_32: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_79, torch.float32);  getitem_79 = None
    mul_233: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
    mul_234: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_569, mul_233);  view_569 = mul_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_235: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_234, alias_87);  mul_234 = None
    sum_54: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_235, [-1], True)
    mul_236: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_87, sum_54);  alias_87 = sum_54 = None
    sub_33: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_235, mul_236);  mul_235 = mul_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_8: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_33, 0);  sub_33 = None
    as_strided_scatter_14: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_8, [8, 1024, 1024], [1048576, 1024, 1], 0);  squeeze_8 = None
    as_strided_52: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_14, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_14 = None
    new_empty_strided_7: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_52, [8, 1024, 1024], [1048576, 1024, 1])
    as_strided_54: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_52, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    as_strided_scatter_15: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(as_strided_52, as_strided_54, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  as_strided_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_143: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_128, as_strided_54);  add_128 = as_strided_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    bmm_66: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_407, as_strided_scatter_15);  permute_407 = None
    bmm_67: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_15, permute_408);  as_strided_scatter_15 = permute_408 = None
    view_570: "f32[1, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_66, [1, 8, 64, 1024]);  bmm_66 = None
    view_571: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_67, [1, 8, 1024, 64]);  bmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_409: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_570, [0, 1, 3, 2]);  view_570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_144: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_11, permute_409);  tangents_11 = permute_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_410: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_142, [0, 2, 1, 3]);  add_142 = None
    clone_72: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_410, memory_format = torch.contiguous_format);  permute_410 = None
    view_572: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_72, [1, 1024, 512]);  clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_573: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_572, [1024, 512]);  view_572 = None
    permute_411: "f32[512, 1024]" = torch.ops.aten.permute.default(view_573, [1, 0])
    mm_173: "f32[512, 512]" = torch.ops.aten.mm.default(permute_411, view_234);  permute_411 = None
    permute_412: "f32[512, 512]" = torch.ops.aten.permute.default(mm_173, [1, 0]);  mm_173 = None
    mm_174: "f32[1024, 512]" = torch.ops.aten.mm.default(view_573, permute_413);  view_573 = permute_413 = None
    view_574: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_174, [1, 1024, 512]);  mm_174 = None
    permute_414: "f32[512, 512]" = torch.ops.aten.permute.default(permute_412, [1, 0]);  permute_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_415: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_144, [0, 2, 1, 3]);  add_144 = None
    clone_73: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_415, memory_format = torch.contiguous_format);  permute_415 = None
    view_575: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_73, [1, 1024, 512]);  clone_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_576: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_575, [1024, 512]);  view_575 = None
    permute_416: "f32[512, 1024]" = torch.ops.aten.permute.default(view_576, [1, 0])
    mm_175: "f32[512, 512]" = torch.ops.aten.mm.default(permute_416, view_234);  permute_416 = None
    permute_417: "f32[512, 512]" = torch.ops.aten.permute.default(mm_175, [1, 0]);  mm_175 = None
    mm_176: "f32[1024, 512]" = torch.ops.aten.mm.default(view_576, permute_418);  view_576 = permute_418 = None
    view_577: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_176, [1, 1024, 512]);  mm_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_145: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(view_574, view_577);  view_574 = view_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_419: "f32[512, 512]" = torch.ops.aten.permute.default(permute_417, [1, 0]);  permute_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_420: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_571, [0, 2, 1, 3]);  view_571 = None
    clone_74: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_420, memory_format = torch.contiguous_format);  permute_420 = None
    view_578: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_74, [1, 1024, 512]);  clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_579: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_578, [1024, 512]);  view_578 = None
    permute_421: "f32[512, 1024]" = torch.ops.aten.permute.default(view_579, [1, 0])
    mm_177: "f32[512, 512]" = torch.ops.aten.mm.default(permute_421, view_234);  permute_421 = view_234 = None
    permute_422: "f32[512, 512]" = torch.ops.aten.permute.default(mm_177, [1, 0]);  mm_177 = None
    mm_178: "f32[1024, 512]" = torch.ops.aten.mm.default(view_579, permute_423);  view_579 = permute_423 = None
    view_580: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_178, [1, 1024, 512]);  mm_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_146: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_145, view_580);  add_145 = view_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_424: "f32[512, 512]" = torch.ops.aten.permute.default(permute_422, [1, 0]);  permute_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_237: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_146, primals_20);  primals_20 = None
    mul_238: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_146, mul_45);  add_146 = mul_45 = None
    sum_55: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_238, [0, 1], True);  mul_238 = None
    view_581: "f32[512]" = torch.ops.aten.reshape.default(sum_55, [512]);  sum_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_239: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_237, add_54)
    mul_240: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_237, rsqrt_19);  mul_237 = None
    sum_56: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_239, [2], True);  mul_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_147: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_141, mul_240);  add_141 = mul_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_57: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_19, 3);  rsqrt_19 = None
    mul_241: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_56, -0.5);  sum_56 = None
    mul_242: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_241, pow_57);  mul_241 = pow_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_84: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_242, [1, 1024, 512]);  mul_242 = None
    div_36: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_84, 512);  expand_84 = None
    pow_58: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_54, 1.0);  add_54 = None
    mul_243: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_58, 2.0);  pow_58 = None
    mul_244: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_36, mul_243);  div_36 = mul_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_148: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_147, mul_244);  add_147 = mul_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_33: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_77, torch.float32);  getitem_77 = None
    mul_245: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_33, 1.1111111111111112);  convert_element_type_33 = None
    mul_246: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_148, mul_245);  mul_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_582: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_246, [1024, 512]);  mul_246 = None
    permute_425: "f32[512, 1024]" = torch.ops.aten.permute.default(view_582, [1, 0])
    mm_179: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_425, view_232);  permute_425 = view_232 = None
    permute_426: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_179, [1, 0]);  mm_179 = None
    mm_180: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_582, permute_427);  view_582 = permute_427 = None
    view_583: "f32[1, 1024, 2048]" = torch.ops.aten.reshape.default(mm_180, [1, 1024, 2048]);  mm_180 = None
    permute_428: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_426, [1, 0]);  permute_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_34: "f32[1, 1024, 2048]" = torch.ops.prims.convert_element_type.default(getitem_75, torch.float32);  getitem_75 = None
    mul_247: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1.1111111111111112);  convert_element_type_34 = None
    mul_248: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_583, mul_247);  view_583 = mul_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    where_10: "f32[1, 1024, 2048]" = torch.ops.aten.where.self(le_5, full_default_7, mul_248);  le_5 = mul_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_584: "f32[1024, 2048]" = torch.ops.aten.reshape.default(where_10, [1024, 2048]);  where_10 = None
    permute_429: "f32[2048, 1024]" = torch.ops.aten.permute.default(view_584, [1, 0])
    mm_181: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_429, view_230);  permute_429 = view_230 = None
    permute_430: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_181, [1, 0]);  mm_181 = None
    mm_182: "f32[1024, 512]" = torch.ops.aten.mm.default(view_584, permute_431);  view_584 = permute_431 = None
    view_585: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_182, [1, 1024, 512]);  mm_182 = None
    permute_432: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_430, [1, 0]);  permute_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_249: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_585, primals_19);  primals_19 = None
    mul_250: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_585, mul_43);  view_585 = mul_43 = None
    sum_57: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_250, [0, 1], True);  mul_250 = None
    view_586: "f32[512]" = torch.ops.aten.reshape.default(sum_57, [512]);  sum_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_251: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_249, add_52)
    mul_252: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_249, rsqrt_18);  mul_249 = None
    sum_58: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [2], True);  mul_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_149: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_148, mul_252);  add_148 = mul_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_59: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_18, 3);  rsqrt_18 = None
    mul_253: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_58, -0.5);  sum_58 = None
    mul_254: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_253, pow_59);  mul_253 = pow_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_85: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_254, [1, 1024, 512]);  mul_254 = None
    div_37: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_85, 512);  expand_85 = None
    pow_60: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_52, 1.0);  add_52 = None
    mul_255: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_60, 2.0);  pow_60 = None
    mul_256: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_37, mul_255);  div_37 = mul_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_150: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_149, mul_256);  add_149 = mul_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:647, code: layer_output = hidden_states + self.dropout(attention_output[0])
    convert_element_type_35: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_73, torch.float32);  getitem_73 = None
    mul_257: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
    mul_258: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_150, mul_257);  mul_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_587: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_258, [1024, 512]);  mul_258 = None
    permute_433: "f32[512, 1024]" = torch.ops.aten.permute.default(view_587, [1, 0])
    mm_183: "f32[512, 512]" = torch.ops.aten.mm.default(permute_433, view_228);  permute_433 = view_228 = None
    permute_434: "f32[512, 512]" = torch.ops.aten.permute.default(mm_183, [1, 0]);  mm_183 = None
    mm_184: "f32[1024, 512]" = torch.ops.aten.mm.default(view_587, permute_435);  view_587 = permute_435 = None
    view_588: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_184, [1, 1024, 512]);  mm_184 = None
    permute_436: "f32[512, 512]" = torch.ops.aten.permute.default(permute_434, [1, 0]);  permute_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_589: "f32[1, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_588, [1, 1024, 8, 64]);  view_588 = None
    permute_437: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_589, [0, 2, 1, 3]);  view_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_590: "f32[8, 1024, 64]" = torch.ops.aten.reshape.default(permute_437, [8, 1024, 64]);  permute_437 = None
    bmm_68: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_438, view_590);  permute_438 = None
    bmm_69: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_590, permute_439);  view_590 = permute_439 = None
    view_591: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_68, [1, 8, 1024, 64]);  bmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_151: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_10, view_591);  tangents_10 = view_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_592: "f32[1, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_69, [1, 8, 1024, 1024]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_36: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_71, torch.float32);  getitem_71 = None
    mul_259: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_36, 1.1111111111111112);  convert_element_type_36 = None
    mul_260: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_592, mul_259);  view_592 = mul_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_261: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_260, alias_91);  mul_260 = None
    sum_59: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_261, [-1], True)
    mul_262: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_91, sum_59);  alias_91 = sum_59 = None
    sub_34: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_261, mul_262);  mul_261 = mul_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_9: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_34, 0);  sub_34 = None
    as_strided_scatter_16: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_9, [8, 1024, 1024], [1048576, 1024, 1], 0);  squeeze_9 = None
    as_strided_59: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_16, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_16 = None
    new_empty_strided_8: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_59, [8, 1024, 1024], [1048576, 1024, 1])
    as_strided_61: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_59, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    as_strided_scatter_17: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(as_strided_59, as_strided_61, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  as_strided_59 = as_strided_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    bmm_70: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_440, as_strided_scatter_17);  permute_440 = None
    bmm_71: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_17, permute_441);  as_strided_scatter_17 = permute_441 = None
    view_593: "f32[1, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_70, [1, 8, 64, 1024]);  bmm_70 = None
    view_594: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_71, [1, 8, 1024, 64]);  bmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_442: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_593, [0, 1, 3, 2]);  view_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_152: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_9, permute_442);  tangents_9 = permute_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_443: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_151, [0, 2, 1, 3]);  add_151 = None
    clone_80: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_443, memory_format = torch.contiguous_format);  permute_443 = None
    view_595: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_80, [1, 1024, 512]);  clone_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_596: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_595, [1024, 512]);  view_595 = None
    permute_444: "f32[512, 1024]" = torch.ops.aten.permute.default(view_596, [1, 0])
    mm_185: "f32[512, 512]" = torch.ops.aten.mm.default(permute_444, view_169);  permute_444 = None
    permute_445: "f32[512, 512]" = torch.ops.aten.permute.default(mm_185, [1, 0]);  mm_185 = None
    mm_186: "f32[1024, 512]" = torch.ops.aten.mm.default(view_596, permute_446);  view_596 = permute_446 = None
    view_597: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_186, [1, 1024, 512]);  mm_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_153: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_139, view_597);  add_139 = view_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_447: "f32[512, 512]" = torch.ops.aten.permute.default(permute_445, [1, 0]);  permute_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_448: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_152, [0, 2, 1, 3]);  add_152 = None
    clone_81: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_448, memory_format = torch.contiguous_format);  permute_448 = None
    view_598: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_81, [1, 1024, 512]);  clone_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_599: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_598, [1024, 512]);  view_598 = None
    permute_449: "f32[512, 1024]" = torch.ops.aten.permute.default(view_599, [1, 0])
    mm_187: "f32[512, 512]" = torch.ops.aten.mm.default(permute_449, view_169);  permute_449 = None
    permute_450: "f32[512, 512]" = torch.ops.aten.permute.default(mm_187, [1, 0]);  mm_187 = None
    mm_188: "f32[1024, 512]" = torch.ops.aten.mm.default(view_599, permute_451);  view_599 = permute_451 = None
    view_600: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_188, [1, 1024, 512]);  mm_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_154: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_153, view_600);  add_153 = view_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_452: "f32[512, 512]" = torch.ops.aten.permute.default(permute_450, [1, 0]);  permute_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_453: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_594, [0, 2, 1, 3]);  view_594 = None
    clone_82: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_453, memory_format = torch.contiguous_format);  permute_453 = None
    view_601: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_82, [1, 1024, 512]);  clone_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_602: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_601, [1024, 512]);  view_601 = None
    permute_454: "f32[512, 1024]" = torch.ops.aten.permute.default(view_602, [1, 0])
    mm_189: "f32[512, 512]" = torch.ops.aten.mm.default(permute_454, view_210);  permute_454 = view_210 = None
    permute_455: "f32[512, 512]" = torch.ops.aten.permute.default(mm_189, [1, 0]);  mm_189 = None
    mm_190: "f32[1024, 512]" = torch.ops.aten.mm.default(view_602, permute_456);  view_602 = permute_456 = None
    view_603: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_190, [1, 1024, 512]);  mm_190 = None
    permute_457: "f32[512, 512]" = torch.ops.aten.permute.default(permute_455, [1, 0]);  permute_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_263: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_603, primals_18);  primals_18 = None
    mul_264: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_603, mul_41);  view_603 = mul_41 = None
    sum_60: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_264, [0, 1], True);  mul_264 = None
    view_604: "f32[512]" = torch.ops.aten.reshape.default(sum_60, [512]);  sum_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_265: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_263, add_49)
    mul_266: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_263, rsqrt_17);  mul_263 = None
    sum_61: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_265, [2], True);  mul_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_155: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_150, mul_266);  add_150 = mul_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_61: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_17, 3);  rsqrt_17 = None
    mul_267: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_61, -0.5);  sum_61 = None
    mul_268: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_267, pow_61);  mul_267 = pow_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_86: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_268, [1, 1024, 512]);  mul_268 = None
    div_38: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_86, 512);  expand_86 = None
    pow_62: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_49, 1.0);  add_49 = None
    mul_269: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_62, 2.0);  pow_62 = None
    mul_270: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_38, mul_269);  div_38 = mul_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_156: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_155, mul_270);  add_155 = mul_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_37: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_69, torch.float32);  getitem_69 = None
    mul_271: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
    mul_272: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_156, mul_271);  mul_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_605: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_272, [1024, 512]);  mul_272 = None
    permute_458: "f32[512, 1024]" = torch.ops.aten.permute.default(view_605, [1, 0])
    mm_191: "f32[512, 512]" = torch.ops.aten.mm.default(permute_458, view_208);  permute_458 = view_208 = None
    permute_459: "f32[512, 512]" = torch.ops.aten.permute.default(mm_191, [1, 0]);  mm_191 = None
    mm_192: "f32[1024, 512]" = torch.ops.aten.mm.default(view_605, permute_460);  view_605 = permute_460 = None
    view_606: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_192, [1, 1024, 512]);  mm_192 = None
    permute_461: "f32[512, 512]" = torch.ops.aten.permute.default(permute_459, [1, 0]);  permute_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_607: "f32[1, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_606, [1, 1024, 8, 64]);  view_606 = None
    permute_462: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_607, [0, 2, 1, 3]);  view_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_608: "f32[8, 1024, 64]" = torch.ops.aten.reshape.default(permute_462, [8, 1024, 64]);  permute_462 = None
    bmm_72: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_463, view_608);  permute_463 = None
    bmm_73: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_608, permute_464);  view_608 = permute_464 = None
    view_609: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_72, [1, 8, 1024, 64]);  bmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_157: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_8, view_609);  tangents_8 = view_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_610: "f32[1, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_73, [1, 8, 1024, 1024]);  bmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_38: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_67, torch.float32);  getitem_67 = None
    mul_273: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 1.1111111111111112);  convert_element_type_38 = None
    mul_274: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_610, mul_273);  view_610 = mul_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_275: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_274, alias_93);  mul_274 = None
    sum_62: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_275, [-1], True)
    mul_276: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_93, sum_62);  alias_93 = sum_62 = None
    sub_35: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_275, mul_276);  mul_275 = mul_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_10: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_35, 0);  sub_35 = None
    as_strided_scatter_18: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_10, [8, 1024, 1024], [1048576, 1024, 1], 0);  squeeze_10 = None
    as_strided_66: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_18, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_18 = None
    new_empty_strided_9: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_66, [8, 1024, 1024], [1048576, 1024, 1])
    as_strided_68: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_66, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    as_strided_scatter_19: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(as_strided_66, as_strided_68, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  as_strided_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_158: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_143, as_strided_68);  add_143 = as_strided_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    bmm_74: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_465, as_strided_scatter_19);  permute_465 = None
    bmm_75: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_19, permute_466);  as_strided_scatter_19 = permute_466 = None
    view_611: "f32[1, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_74, [1, 8, 64, 1024]);  bmm_74 = None
    view_612: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_75, [1, 8, 1024, 64]);  bmm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_467: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_611, [0, 1, 3, 2]);  view_611 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_159: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_7, permute_467);  tangents_7 = permute_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_468: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_157, [0, 2, 1, 3]);  add_157 = None
    clone_86: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_468, memory_format = torch.contiguous_format);  permute_468 = None
    view_613: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_86, [1, 1024, 512]);  clone_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_614: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_613, [1024, 512]);  view_613 = None
    permute_469: "f32[512, 1024]" = torch.ops.aten.permute.default(view_614, [1, 0])
    mm_193: "f32[512, 512]" = torch.ops.aten.mm.default(permute_469, view_190);  permute_469 = None
    permute_470: "f32[512, 512]" = torch.ops.aten.permute.default(mm_193, [1, 0]);  mm_193 = None
    mm_194: "f32[1024, 512]" = torch.ops.aten.mm.default(view_614, permute_471);  view_614 = permute_471 = None
    view_615: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_194, [1, 1024, 512]);  mm_194 = None
    permute_472: "f32[512, 512]" = torch.ops.aten.permute.default(permute_470, [1, 0]);  permute_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_473: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_159, [0, 2, 1, 3]);  add_159 = None
    clone_87: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_473, memory_format = torch.contiguous_format);  permute_473 = None
    view_616: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_87, [1, 1024, 512]);  clone_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_617: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_616, [1024, 512]);  view_616 = None
    permute_474: "f32[512, 1024]" = torch.ops.aten.permute.default(view_617, [1, 0])
    mm_195: "f32[512, 512]" = torch.ops.aten.mm.default(permute_474, view_190);  permute_474 = None
    permute_475: "f32[512, 512]" = torch.ops.aten.permute.default(mm_195, [1, 0]);  mm_195 = None
    mm_196: "f32[1024, 512]" = torch.ops.aten.mm.default(view_617, permute_476);  view_617 = permute_476 = None
    view_618: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_196, [1, 1024, 512]);  mm_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_160: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(view_615, view_618);  view_615 = view_618 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_477: "f32[512, 512]" = torch.ops.aten.permute.default(permute_475, [1, 0]);  permute_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_478: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_612, [0, 2, 1, 3]);  view_612 = None
    clone_88: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_478, memory_format = torch.contiguous_format);  permute_478 = None
    view_619: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_88, [1, 1024, 512]);  clone_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_620: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_619, [1024, 512]);  view_619 = None
    permute_479: "f32[512, 1024]" = torch.ops.aten.permute.default(view_620, [1, 0])
    mm_197: "f32[512, 512]" = torch.ops.aten.mm.default(permute_479, view_190);  permute_479 = view_190 = None
    permute_480: "f32[512, 512]" = torch.ops.aten.permute.default(mm_197, [1, 0]);  mm_197 = None
    mm_198: "f32[1024, 512]" = torch.ops.aten.mm.default(view_620, permute_481);  view_620 = permute_481 = None
    view_621: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_198, [1, 1024, 512]);  mm_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_161: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_160, view_621);  add_160 = view_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_482: "f32[512, 512]" = torch.ops.aten.permute.default(permute_480, [1, 0]);  permute_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_277: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_161, primals_17);  primals_17 = None
    mul_278: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_161, mul_39);  add_161 = mul_39 = None
    sum_63: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_278, [0, 1], True);  mul_278 = None
    view_622: "f32[512]" = torch.ops.aten.reshape.default(sum_63, [512]);  sum_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_279: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_277, add_46)
    mul_280: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_277, rsqrt_16);  mul_277 = None
    sum_64: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_279, [2], True);  mul_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_162: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_156, mul_280);  add_156 = mul_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_63: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_16, 3);  rsqrt_16 = None
    mul_281: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_64, -0.5);  sum_64 = None
    mul_282: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_281, pow_63);  mul_281 = pow_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_87: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_282, [1, 1024, 512]);  mul_282 = None
    div_39: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_87, 512);  expand_87 = None
    pow_64: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_46, 1.0);  add_46 = None
    mul_283: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_64, 2.0);  pow_64 = None
    mul_284: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_39, mul_283);  div_39 = mul_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_163: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_162, mul_284);  add_162 = mul_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_39: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_65, torch.float32);  getitem_65 = None
    mul_285: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_39, 1.1111111111111112);  convert_element_type_39 = None
    mul_286: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_163, mul_285);  mul_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_623: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_286, [1024, 512]);  mul_286 = None
    permute_483: "f32[512, 1024]" = torch.ops.aten.permute.default(view_623, [1, 0])
    mm_199: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_483, view_188);  permute_483 = view_188 = None
    permute_484: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_199, [1, 0]);  mm_199 = None
    mm_200: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_623, permute_485);  view_623 = permute_485 = None
    view_624: "f32[1, 1024, 2048]" = torch.ops.aten.reshape.default(mm_200, [1, 1024, 2048]);  mm_200 = None
    permute_486: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_484, [1, 0]);  permute_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_40: "f32[1, 1024, 2048]" = torch.ops.prims.convert_element_type.default(getitem_63, torch.float32);  getitem_63 = None
    mul_287: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_40, 1.1111111111111112);  convert_element_type_40 = None
    mul_288: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_624, mul_287);  view_624 = mul_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    where_11: "f32[1, 1024, 2048]" = torch.ops.aten.where.self(le_6, full_default_7, mul_288);  le_6 = mul_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_625: "f32[1024, 2048]" = torch.ops.aten.reshape.default(where_11, [1024, 2048]);  where_11 = None
    permute_487: "f32[2048, 1024]" = torch.ops.aten.permute.default(view_625, [1, 0])
    mm_201: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_487, view_186);  permute_487 = view_186 = None
    permute_488: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_201, [1, 0]);  mm_201 = None
    mm_202: "f32[1024, 512]" = torch.ops.aten.mm.default(view_625, permute_489);  view_625 = permute_489 = None
    view_626: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_202, [1, 1024, 512]);  mm_202 = None
    permute_490: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_488, [1, 0]);  permute_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_289: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_626, primals_16);  primals_16 = None
    mul_290: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_626, mul_37);  view_626 = mul_37 = None
    sum_65: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_290, [0, 1], True);  mul_290 = None
    view_627: "f32[512]" = torch.ops.aten.reshape.default(sum_65, [512]);  sum_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_291: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_289, add_44)
    mul_292: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_289, rsqrt_15);  mul_289 = None
    sum_66: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_291, [2], True);  mul_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_164: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_163, mul_292);  add_163 = mul_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_65: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_15, 3);  rsqrt_15 = None
    mul_293: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_66, -0.5);  sum_66 = None
    mul_294: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_293, pow_65);  mul_293 = pow_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_88: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_294, [1, 1024, 512]);  mul_294 = None
    div_40: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_88, 512);  expand_88 = None
    pow_66: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_44, 1.0);  add_44 = None
    mul_295: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_66, 2.0);  pow_66 = None
    mul_296: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_40, mul_295);  div_40 = mul_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_165: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_164, mul_296);  add_164 = mul_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:647, code: layer_output = hidden_states + self.dropout(attention_output[0])
    convert_element_type_41: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_61, torch.float32);  getitem_61 = None
    mul_297: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_41, 1.1111111111111112);  convert_element_type_41 = None
    mul_298: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_165, mul_297);  mul_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_628: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_298, [1024, 512]);  mul_298 = None
    permute_491: "f32[512, 1024]" = torch.ops.aten.permute.default(view_628, [1, 0])
    mm_203: "f32[512, 512]" = torch.ops.aten.mm.default(permute_491, view_184);  permute_491 = view_184 = None
    permute_492: "f32[512, 512]" = torch.ops.aten.permute.default(mm_203, [1, 0]);  mm_203 = None
    mm_204: "f32[1024, 512]" = torch.ops.aten.mm.default(view_628, permute_493);  view_628 = permute_493 = None
    view_629: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_204, [1, 1024, 512]);  mm_204 = None
    permute_494: "f32[512, 512]" = torch.ops.aten.permute.default(permute_492, [1, 0]);  permute_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_630: "f32[1, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_629, [1, 1024, 8, 64]);  view_629 = None
    permute_495: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_630, [0, 2, 1, 3]);  view_630 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_631: "f32[8, 1024, 64]" = torch.ops.aten.reshape.default(permute_495, [8, 1024, 64]);  permute_495 = None
    bmm_76: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_496, view_631);  permute_496 = None
    bmm_77: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_631, permute_497);  view_631 = permute_497 = None
    view_632: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_76, [1, 8, 1024, 64]);  bmm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_166: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_6, view_632);  tangents_6 = view_632 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_633: "f32[1, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_77, [1, 8, 1024, 1024]);  bmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_42: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_59, torch.float32);  getitem_59 = None
    mul_299: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_42, 1.1111111111111112);  convert_element_type_42 = None
    mul_300: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_633, mul_299);  view_633 = mul_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_301: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_300, alias_97);  mul_300 = None
    sum_67: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_301, [-1], True)
    mul_302: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_97, sum_67);  alias_97 = sum_67 = None
    sub_36: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_301, mul_302);  mul_301 = mul_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_11: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_36, 0);  sub_36 = None
    as_strided_scatter_20: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_11, [8, 1024, 1024], [1048576, 1024, 1], 0);  squeeze_11 = None
    as_strided_73: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_20, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_20 = None
    new_empty_strided_10: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_73, [8, 1024, 1024], [1048576, 1024, 1])
    as_strided_75: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_73, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    as_strided_scatter_21: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(as_strided_73, as_strided_75, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  as_strided_73 = as_strided_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    bmm_78: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_498, as_strided_scatter_21);  permute_498 = None
    bmm_79: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_21, permute_499);  as_strided_scatter_21 = permute_499 = None
    view_634: "f32[1, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_78, [1, 8, 64, 1024]);  bmm_78 = None
    view_635: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_79, [1, 8, 1024, 64]);  bmm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_500: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_634, [0, 1, 3, 2]);  view_634 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_167: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_5, permute_500);  tangents_5 = permute_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_501: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_166, [0, 2, 1, 3]);  add_166 = None
    clone_94: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_501, memory_format = torch.contiguous_format);  permute_501 = None
    view_636: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_94, [1, 1024, 512]);  clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_637: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_636, [1024, 512]);  view_636 = None
    permute_502: "f32[512, 1024]" = torch.ops.aten.permute.default(view_637, [1, 0])
    mm_205: "f32[512, 512]" = torch.ops.aten.mm.default(permute_502, view_169);  permute_502 = None
    permute_503: "f32[512, 512]" = torch.ops.aten.permute.default(mm_205, [1, 0]);  mm_205 = None
    mm_206: "f32[1024, 512]" = torch.ops.aten.mm.default(view_637, permute_504);  view_637 = permute_504 = None
    view_638: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_206, [1, 1024, 512]);  mm_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_168: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_154, view_638);  add_154 = view_638 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_505: "f32[512, 512]" = torch.ops.aten.permute.default(permute_503, [1, 0]);  permute_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_506: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_167, [0, 2, 1, 3]);  add_167 = None
    clone_95: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_506, memory_format = torch.contiguous_format);  permute_506 = None
    view_639: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_95, [1, 1024, 512]);  clone_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_640: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_639, [1024, 512]);  view_639 = None
    permute_507: "f32[512, 1024]" = torch.ops.aten.permute.default(view_640, [1, 0])
    mm_207: "f32[512, 512]" = torch.ops.aten.mm.default(permute_507, view_169);  permute_507 = view_169 = None
    permute_508: "f32[512, 512]" = torch.ops.aten.permute.default(mm_207, [1, 0]);  mm_207 = None
    mm_208: "f32[1024, 512]" = torch.ops.aten.mm.default(view_640, permute_509);  view_640 = permute_509 = None
    view_641: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_208, [1, 1024, 512]);  mm_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_169: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_168, view_641);  add_168 = view_641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_510: "f32[512, 512]" = torch.ops.aten.permute.default(permute_508, [1, 0]);  permute_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_511: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_635, [0, 2, 1, 3]);  view_635 = None
    clone_96: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_511, memory_format = torch.contiguous_format);  permute_511 = None
    view_642: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_96, [1, 1024, 512]);  clone_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_643: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_642, [1024, 512]);  view_642 = None
    permute_512: "f32[512, 1024]" = torch.ops.aten.permute.default(view_643, [1, 0])
    mm_209: "f32[512, 512]" = torch.ops.aten.mm.default(permute_512, view_166);  permute_512 = view_166 = None
    permute_513: "f32[512, 512]" = torch.ops.aten.permute.default(mm_209, [1, 0]);  mm_209 = None
    mm_210: "f32[1024, 512]" = torch.ops.aten.mm.default(view_643, permute_514);  view_643 = permute_514 = None
    view_644: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_210, [1, 1024, 512]);  mm_210 = None
    permute_515: "f32[512, 512]" = torch.ops.aten.permute.default(permute_513, [1, 0]);  permute_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_303: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_644, primals_15);  primals_15 = None
    mul_304: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_644, mul_35);  view_644 = mul_35 = None
    sum_68: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_304, [0, 1], True);  mul_304 = None
    view_645: "f32[512]" = torch.ops.aten.reshape.default(sum_68, [512]);  sum_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_305: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_303, add_40)
    mul_306: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_303, rsqrt_14);  mul_303 = None
    sum_69: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_305, [2], True);  mul_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_170: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_165, mul_306);  add_165 = mul_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_67: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_14, 3);  rsqrt_14 = None
    mul_307: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_69, -0.5);  sum_69 = None
    mul_308: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_307, pow_67);  mul_307 = pow_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_89: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_308, [1, 1024, 512]);  mul_308 = None
    div_41: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_89, 512);  expand_89 = None
    pow_68: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_40, 1.0);  add_40 = None
    mul_309: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_68, 2.0);  pow_68 = None
    mul_310: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_41, mul_309);  div_41 = mul_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_171: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_170, mul_310);  add_170 = mul_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_43: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_57, torch.float32);  getitem_57 = None
    mul_311: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_43, 1.1111111111111112);  convert_element_type_43 = None
    mul_312: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_171, mul_311);  mul_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_646: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_312, [1024, 512]);  mul_312 = None
    permute_516: "f32[512, 1024]" = torch.ops.aten.permute.default(view_646, [1, 0])
    mm_211: "f32[512, 512]" = torch.ops.aten.mm.default(permute_516, view_164);  permute_516 = view_164 = None
    permute_517: "f32[512, 512]" = torch.ops.aten.permute.default(mm_211, [1, 0]);  mm_211 = None
    mm_212: "f32[1024, 512]" = torch.ops.aten.mm.default(view_646, permute_518);  view_646 = permute_518 = None
    view_647: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_212, [1, 1024, 512]);  mm_212 = None
    permute_519: "f32[512, 512]" = torch.ops.aten.permute.default(permute_517, [1, 0]);  permute_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_648: "f32[1, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_647, [1, 1024, 8, 64]);  view_647 = None
    permute_520: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_648, [0, 2, 1, 3]);  view_648 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_649: "f32[8, 1024, 64]" = torch.ops.aten.reshape.default(permute_520, [8, 1024, 64]);  permute_520 = None
    bmm_80: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_521, view_649);  permute_521 = None
    bmm_81: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_649, permute_522);  view_649 = permute_522 = None
    view_650: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_80, [1, 8, 1024, 64]);  bmm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_172: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_4, view_650);  tangents_4 = view_650 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_651: "f32[1, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_81, [1, 8, 1024, 1024]);  bmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_44: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_55, torch.float32);  getitem_55 = None
    mul_313: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_44, 1.1111111111111112);  convert_element_type_44 = None
    mul_314: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_651, mul_313);  view_651 = mul_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_315: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_314, alias_99);  mul_314 = None
    sum_70: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_315, [-1], True)
    mul_316: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_99, sum_70);  alias_99 = sum_70 = None
    sub_37: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_315, mul_316);  mul_315 = mul_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_12: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_37, 0);  sub_37 = None
    as_strided_scatter_22: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_12, [8, 1024, 1024], [1048576, 1024, 1], 0);  squeeze_12 = None
    as_strided_80: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_22, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_22 = None
    new_empty_strided_11: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_80, [8, 1024, 1024], [1048576, 1024, 1])
    as_strided_82: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_80, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    as_strided_scatter_23: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(as_strided_80, as_strided_82, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  as_strided_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_173: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_158, as_strided_82);  add_158 = as_strided_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:451, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
    squeeze_13: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(add_173, 0);  add_173 = None
    permute_523: "f32[1024, 1024, 8]" = torch.ops.aten.permute.default(squeeze_13, [1, 2, 0]);  squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:450, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
    full_default_17: "b8[1024, 1024, 1]" = torch.ops.aten.full.default([1024, 1024, 1], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where_12: "f32[1024, 1024, 8]" = torch.ops.aten.where.self(full_default_17, full_default_7, permute_523);  permute_523 = None
    clone_100: "f32[1024, 1024, 8]" = torch.ops.aten.clone.default(where_12, memory_format = torch.contiguous_format);  where_12 = None
    full_default_19: "f32[32, 8]" = torch.ops.aten.full.default([32, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put: "f32[32, 8]" = torch.ops.aten._unsafe_index_put.default(full_default_19, [add_37], clone_100, True);  add_37 = clone_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    bmm_82: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_524, as_strided_scatter_23);  permute_524 = None
    bmm_83: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_23, permute_525);  as_strided_scatter_23 = permute_525 = None
    view_652: "f32[1, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_82, [1, 8, 64, 1024]);  bmm_82 = None
    view_653: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_83, [1, 8, 1024, 64]);  bmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_526: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_652, [0, 1, 3, 2]);  view_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_174: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_3, permute_526);  tangents_3 = permute_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_527: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_172, [0, 2, 1, 3]);  add_172 = None
    clone_101: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_527, memory_format = torch.contiguous_format);  permute_527 = None
    view_654: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_101, [1, 1024, 512]);  clone_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_655: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_654, [1024, 512]);  view_654 = None
    permute_528: "f32[512, 1024]" = torch.ops.aten.permute.default(view_655, [1, 0])
    mm_213: "f32[512, 512]" = torch.ops.aten.mm.default(permute_528, view_146);  permute_528 = None
    permute_529: "f32[512, 512]" = torch.ops.aten.permute.default(mm_213, [1, 0]);  mm_213 = None
    mm_214: "f32[1024, 512]" = torch.ops.aten.mm.default(view_655, permute_530);  view_655 = permute_530 = None
    view_656: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_214, [1, 1024, 512]);  mm_214 = None
    permute_531: "f32[512, 512]" = torch.ops.aten.permute.default(permute_529, [1, 0]);  permute_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_532: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_174, [0, 2, 1, 3]);  add_174 = None
    clone_102: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_532, memory_format = torch.contiguous_format);  permute_532 = None
    view_657: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_102, [1, 1024, 512]);  clone_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_658: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_657, [1024, 512]);  view_657 = None
    permute_533: "f32[512, 1024]" = torch.ops.aten.permute.default(view_658, [1, 0])
    mm_215: "f32[512, 512]" = torch.ops.aten.mm.default(permute_533, view_146);  permute_533 = None
    permute_534: "f32[512, 512]" = torch.ops.aten.permute.default(mm_215, [1, 0]);  mm_215 = None
    mm_216: "f32[1024, 512]" = torch.ops.aten.mm.default(view_658, permute_535);  view_658 = permute_535 = None
    view_659: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_216, [1, 1024, 512]);  mm_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_175: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(view_656, view_659);  view_656 = view_659 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_536: "f32[512, 512]" = torch.ops.aten.permute.default(permute_534, [1, 0]);  permute_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_537: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_653, [0, 2, 1, 3]);  view_653 = None
    clone_103: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_537, memory_format = torch.contiguous_format);  permute_537 = None
    view_660: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_103, [1, 1024, 512]);  clone_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_661: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_660, [1024, 512]);  view_660 = None
    permute_538: "f32[512, 1024]" = torch.ops.aten.permute.default(view_661, [1, 0])
    mm_217: "f32[512, 512]" = torch.ops.aten.mm.default(permute_538, view_146);  permute_538 = view_146 = None
    permute_539: "f32[512, 512]" = torch.ops.aten.permute.default(mm_217, [1, 0]);  mm_217 = None
    mm_218: "f32[1024, 512]" = torch.ops.aten.mm.default(view_661, permute_540);  view_661 = permute_540 = None
    view_662: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_218, [1, 1024, 512]);  mm_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_176: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_175, view_662);  add_175 = view_662 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_541: "f32[512, 512]" = torch.ops.aten.permute.default(permute_539, [1, 0]);  permute_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_317: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_176, primals_14);  primals_14 = None
    mul_318: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_176, mul_32);  add_176 = mul_32 = None
    sum_71: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_318, [0, 1], True);  mul_318 = None
    view_663: "f32[512]" = torch.ops.aten.reshape.default(sum_71, [512]);  sum_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_319: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_317, getitem_52)
    mul_320: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_317, rsqrt_13);  mul_317 = None
    sum_72: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_319, [2], True);  mul_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_177: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_171, mul_320);  add_171 = mul_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_69: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_13, 3);  rsqrt_13 = None
    mul_321: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_72, -0.5);  sum_72 = None
    mul_322: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_321, pow_69);  mul_321 = pow_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_90: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_322, [1, 1024, 512]);  mul_322 = None
    div_42: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_90, 512);  expand_90 = None
    pow_70: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(getitem_52, 1.0);  getitem_52 = None
    mul_323: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_70, 2.0);  pow_70 = None
    mul_324: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_42, mul_323);  div_42 = mul_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_178: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_177, mul_324);  add_177 = mul_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1076, code: hidden_states = self.dropout(inputs_embeds)
    convert_element_type_45: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_53, torch.float32);  getitem_53 = None
    mul_325: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_45, 1.1111111111111112);  convert_element_type_45 = None
    mul_326: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_178, mul_325);  add_178 = mul_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1021, code: inputs_embeds = self.embed_tokens(input_ids)
    eq_1: "b8[1, 1024]" = torch.ops.aten.eq.Scalar(view_145, -1)
    unsqueeze_20: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    where_13: "f32[1, 1024, 512]" = torch.ops.aten.where.self(unsqueeze_20, full_default_7, mul_326);  unsqueeze_20 = mul_326 = None
    full_default_21: "f32[32128, 512]" = torch.ops.aten.full.default([32128, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_1: "f32[32128, 512]" = torch.ops.aten._unsafe_index_put.default(full_default_21, [view_145], where_13, True);  view_145 = where_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1166, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_46: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_51, torch.float32);  getitem_51 = None
    mul_327: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_46, 1.1111111111111112);  convert_element_type_46 = None
    mul_328: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_169, mul_327);  add_169 = mul_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_329: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_328, primals_13);  primals_13 = None
    mul_330: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_328, mul_27);  mul_328 = mul_27 = None
    sum_73: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_330, [0, 1], True);  mul_330 = None
    view_664: "f32[512]" = torch.ops.aten.reshape.default(sum_73, [512]);  sum_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_331: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_329, add_33)
    mul_332: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_329, rsqrt_12);  mul_329 = None
    sum_74: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_331, [2], True);  mul_331 = None
    pow_71: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_12, 3);  rsqrt_12 = None
    mul_333: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_74, -0.5);  sum_74 = None
    mul_334: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_333, pow_71);  mul_333 = pow_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_91: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_334, [1, 1024, 512]);  mul_334 = None
    div_43: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_91, 512);  expand_91 = None
    pow_72: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_33, 1.0);  add_33 = None
    mul_335: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_72, 2.0);  pow_72 = None
    mul_336: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_43, mul_335);  div_43 = mul_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_179: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(mul_332, mul_336);  mul_332 = mul_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_47: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_49, torch.float32);  getitem_49 = None
    mul_337: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_47, 1.1111111111111112);  convert_element_type_47 = None
    mul_338: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_179, mul_337);  mul_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_665: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_338, [1024, 512]);  mul_338 = None
    permute_542: "f32[512, 1024]" = torch.ops.aten.permute.default(view_665, [1, 0])
    mm_219: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_542, view_143);  permute_542 = view_143 = None
    permute_543: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_219, [1, 0]);  mm_219 = None
    mm_220: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_665, permute_544);  view_665 = permute_544 = None
    view_666: "f32[1, 1024, 2048]" = torch.ops.aten.reshape.default(mm_220, [1, 1024, 2048]);  mm_220 = None
    permute_545: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_543, [1, 0]);  permute_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_48: "f32[1, 1024, 2048]" = torch.ops.prims.convert_element_type.default(getitem_47, torch.float32);  getitem_47 = None
    mul_339: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_48, 1.1111111111111112);  convert_element_type_48 = None
    mul_340: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_666, mul_339);  view_666 = mul_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    where_14: "f32[1, 1024, 2048]" = torch.ops.aten.where.self(le_7, full_default_7, mul_340);  le_7 = mul_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_667: "f32[1024, 2048]" = torch.ops.aten.reshape.default(where_14, [1024, 2048]);  where_14 = None
    permute_546: "f32[2048, 1024]" = torch.ops.aten.permute.default(view_667, [1, 0])
    mm_221: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_546, view_141);  permute_546 = view_141 = None
    permute_547: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_221, [1, 0]);  mm_221 = None
    mm_222: "f32[1024, 512]" = torch.ops.aten.mm.default(view_667, permute_548);  view_667 = permute_548 = None
    view_668: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_222, [1, 1024, 512]);  mm_222 = None
    permute_549: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_547, [1, 0]);  permute_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_341: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_668, primals_12);  primals_12 = None
    mul_342: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_668, mul_25);  view_668 = mul_25 = None
    sum_75: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_342, [0, 1], True);  mul_342 = None
    view_669: "f32[512]" = torch.ops.aten.reshape.default(sum_75, [512]);  sum_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_343: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_341, add_31)
    mul_344: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_341, rsqrt_11);  mul_341 = None
    sum_76: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_343, [2], True);  mul_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_180: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_179, mul_344);  add_179 = mul_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_73: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_11, 3);  rsqrt_11 = None
    mul_345: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_76, -0.5);  sum_76 = None
    mul_346: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_345, pow_73);  mul_345 = pow_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_92: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_346, [1, 1024, 512]);  mul_346 = None
    div_44: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_92, 512);  expand_92 = None
    pow_74: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_31, 1.0);  add_31 = None
    mul_347: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_74, 2.0);  pow_74 = None
    mul_348: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_44, mul_347);  div_44 = mul_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_181: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_180, mul_348);  add_180 = mul_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_49: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_45, torch.float32);  getitem_45 = None
    mul_349: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_49, 1.1111111111111112);  convert_element_type_49 = None
    mul_350: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_181, mul_349);  mul_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_670: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_350, [1024, 512]);  mul_350 = None
    permute_550: "f32[512, 1024]" = torch.ops.aten.permute.default(view_670, [1, 0])
    mm_223: "f32[512, 512]" = torch.ops.aten.mm.default(permute_550, view_139);  permute_550 = view_139 = None
    permute_551: "f32[512, 512]" = torch.ops.aten.permute.default(mm_223, [1, 0]);  mm_223 = None
    mm_224: "f32[1024, 512]" = torch.ops.aten.mm.default(view_670, permute_552);  view_670 = permute_552 = None
    view_671: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_224, [1, 1024, 512]);  mm_224 = None
    permute_553: "f32[512, 512]" = torch.ops.aten.permute.default(permute_551, [1, 0]);  permute_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_672: "f32[1, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_671, [1, 1024, 8, 64]);  view_671 = None
    permute_554: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_672, [0, 2, 1, 3]);  view_672 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_673: "f32[8, 1024, 64]" = torch.ops.aten.reshape.default(permute_554, [8, 1024, 64]);  permute_554 = None
    bmm_84: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_555, view_673);  permute_555 = None
    bmm_85: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_673, permute_556);  view_673 = permute_556 = None
    view_674: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_84, [1, 8, 1024, 64]);  bmm_84 = None
    view_675: "f32[1, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_85, [1, 8, 1024, 1024]);  bmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_50: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_43, torch.float32);  getitem_43 = None
    mul_351: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_50, 1.1111111111111112);  convert_element_type_50 = None
    mul_352: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_675, mul_351);  view_675 = mul_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_353: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_352, alias_104);  mul_352 = None
    sum_77: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_353, [-1], True)
    mul_354: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_104, sum_77);  alias_104 = sum_77 = None
    sub_38: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_353, mul_354);  mul_353 = mul_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_14: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_38, 0);  sub_38 = None
    as_strided_scatter_24: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_14, [8, 1024, 1024], [1048576, 1024, 1], 0);  squeeze_14 = None
    as_strided_87: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_24, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_24 = None
    new_empty_strided_12: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_87, [8, 1024, 1024], [1048576, 1024, 1])
    as_strided_89: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_87, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    as_strided_scatter_25: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(as_strided_87, as_strided_89, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  as_strided_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    bmm_86: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_557, as_strided_scatter_25);  permute_557 = None
    bmm_87: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_25, permute_558);  as_strided_scatter_25 = permute_558 = None
    view_676: "f32[1, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_86, [1, 8, 64, 1024]);  bmm_86 = None
    view_677: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_87, [1, 8, 1024, 64]);  bmm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_559: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_676, [0, 1, 3, 2]);  view_676 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_560: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_674, [0, 2, 1, 3]);  view_674 = None
    clone_111: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_560, memory_format = torch.contiguous_format);  permute_560 = None
    view_678: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_111, [1, 1024, 512]);  clone_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_679: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_678, [1024, 512]);  view_678 = None
    permute_561: "f32[512, 1024]" = torch.ops.aten.permute.default(view_679, [1, 0])
    mm_225: "f32[512, 512]" = torch.ops.aten.mm.default(permute_561, view_121);  permute_561 = None
    permute_562: "f32[512, 512]" = torch.ops.aten.permute.default(mm_225, [1, 0]);  mm_225 = None
    mm_226: "f32[1024, 512]" = torch.ops.aten.mm.default(view_679, permute_563);  view_679 = permute_563 = None
    view_680: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_226, [1, 1024, 512]);  mm_226 = None
    permute_564: "f32[512, 512]" = torch.ops.aten.permute.default(permute_562, [1, 0]);  permute_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_565: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_559, [0, 2, 1, 3]);  permute_559 = None
    view_681: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(permute_565, [1, 1024, 512]);  permute_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_682: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_681, [1024, 512]);  view_681 = None
    permute_566: "f32[512, 1024]" = torch.ops.aten.permute.default(view_682, [1, 0])
    mm_227: "f32[512, 512]" = torch.ops.aten.mm.default(permute_566, view_121);  permute_566 = None
    permute_567: "f32[512, 512]" = torch.ops.aten.permute.default(mm_227, [1, 0]);  mm_227 = None
    mm_228: "f32[1024, 512]" = torch.ops.aten.mm.default(view_682, permute_568);  view_682 = permute_568 = None
    view_683: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_228, [1, 1024, 512]);  mm_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_182: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(view_680, view_683);  view_680 = view_683 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_569: "f32[512, 512]" = torch.ops.aten.permute.default(permute_567, [1, 0]);  permute_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_570: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_677, [0, 2, 1, 3]);  view_677 = None
    clone_112: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_570, memory_format = torch.contiguous_format);  permute_570 = None
    view_684: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_112, [1, 1024, 512]);  clone_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_685: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_684, [1024, 512]);  view_684 = None
    permute_571: "f32[512, 1024]" = torch.ops.aten.permute.default(view_685, [1, 0])
    mm_229: "f32[512, 512]" = torch.ops.aten.mm.default(permute_571, view_121);  permute_571 = view_121 = None
    permute_572: "f32[512, 512]" = torch.ops.aten.permute.default(mm_229, [1, 0]);  mm_229 = None
    mm_230: "f32[1024, 512]" = torch.ops.aten.mm.default(view_685, permute_573);  view_685 = permute_573 = None
    view_686: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_230, [1, 1024, 512]);  mm_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_183: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_182, view_686);  add_182 = view_686 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_574: "f32[512, 512]" = torch.ops.aten.permute.default(permute_572, [1, 0]);  permute_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_355: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_183, primals_11);  primals_11 = None
    mul_356: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_183, mul_23);  add_183 = mul_23 = None
    sum_78: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_356, [0, 1], True);  mul_356 = None
    view_687: "f32[512]" = torch.ops.aten.reshape.default(sum_78, [512]);  sum_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_357: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_355, add_28)
    mul_358: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_355, rsqrt_10);  mul_355 = None
    sum_79: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_357, [2], True);  mul_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_184: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_181, mul_358);  add_181 = mul_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_75: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_10, 3);  rsqrt_10 = None
    mul_359: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_79, -0.5);  sum_79 = None
    mul_360: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_359, pow_75);  mul_359 = pow_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_93: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_360, [1, 1024, 512]);  mul_360 = None
    div_45: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_93, 512);  expand_93 = None
    pow_76: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_28, 1.0);  add_28 = None
    mul_361: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_76, 2.0);  pow_76 = None
    mul_362: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_45, mul_361);  div_45 = mul_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_185: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_184, mul_362);  add_184 = mul_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_51: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_41, torch.float32);  getitem_41 = None
    mul_363: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_51, 1.1111111111111112);  convert_element_type_51 = None
    mul_364: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_185, mul_363);  mul_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_688: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_364, [1024, 512]);  mul_364 = None
    permute_575: "f32[512, 1024]" = torch.ops.aten.permute.default(view_688, [1, 0])
    mm_231: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_575, view_119);  permute_575 = view_119 = None
    permute_576: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_231, [1, 0]);  mm_231 = None
    mm_232: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_688, permute_577);  view_688 = permute_577 = None
    view_689: "f32[1, 1024, 2048]" = torch.ops.aten.reshape.default(mm_232, [1, 1024, 2048]);  mm_232 = None
    permute_578: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_576, [1, 0]);  permute_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_52: "f32[1, 1024, 2048]" = torch.ops.prims.convert_element_type.default(getitem_39, torch.float32);  getitem_39 = None
    mul_365: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_52, 1.1111111111111112);  convert_element_type_52 = None
    mul_366: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_689, mul_365);  view_689 = mul_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    where_15: "f32[1, 1024, 2048]" = torch.ops.aten.where.self(le_8, full_default_7, mul_366);  le_8 = mul_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_690: "f32[1024, 2048]" = torch.ops.aten.reshape.default(where_15, [1024, 2048]);  where_15 = None
    permute_579: "f32[2048, 1024]" = torch.ops.aten.permute.default(view_690, [1, 0])
    mm_233: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_579, view_117);  permute_579 = view_117 = None
    permute_580: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_233, [1, 0]);  mm_233 = None
    mm_234: "f32[1024, 512]" = torch.ops.aten.mm.default(view_690, permute_581);  view_690 = permute_581 = None
    view_691: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_234, [1, 1024, 512]);  mm_234 = None
    permute_582: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_580, [1, 0]);  permute_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_367: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_691, primals_10);  primals_10 = None
    mul_368: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_691, mul_21);  view_691 = mul_21 = None
    sum_80: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_368, [0, 1], True);  mul_368 = None
    view_692: "f32[512]" = torch.ops.aten.reshape.default(sum_80, [512]);  sum_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_369: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_367, add_26)
    mul_370: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_367, rsqrt_9);  mul_367 = None
    sum_81: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_369, [2], True);  mul_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_186: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_185, mul_370);  add_185 = mul_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_77: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_9, 3);  rsqrt_9 = None
    mul_371: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_81, -0.5);  sum_81 = None
    mul_372: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_371, pow_77);  mul_371 = pow_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_94: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_372, [1, 1024, 512]);  mul_372 = None
    div_46: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_94, 512);  expand_94 = None
    pow_78: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_26, 1.0);  add_26 = None
    mul_373: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_78, 2.0);  pow_78 = None
    mul_374: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_46, mul_373);  div_46 = mul_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_187: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_186, mul_374);  add_186 = mul_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_53: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_37, torch.float32);  getitem_37 = None
    mul_375: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_53, 1.1111111111111112);  convert_element_type_53 = None
    mul_376: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_187, mul_375);  mul_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_693: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_376, [1024, 512]);  mul_376 = None
    permute_583: "f32[512, 1024]" = torch.ops.aten.permute.default(view_693, [1, 0])
    mm_235: "f32[512, 512]" = torch.ops.aten.mm.default(permute_583, view_115);  permute_583 = view_115 = None
    permute_584: "f32[512, 512]" = torch.ops.aten.permute.default(mm_235, [1, 0]);  mm_235 = None
    mm_236: "f32[1024, 512]" = torch.ops.aten.mm.default(view_693, permute_585);  view_693 = permute_585 = None
    view_694: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_236, [1, 1024, 512]);  mm_236 = None
    permute_586: "f32[512, 512]" = torch.ops.aten.permute.default(permute_584, [1, 0]);  permute_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_695: "f32[1, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_694, [1, 1024, 8, 64]);  view_694 = None
    permute_587: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_695, [0, 2, 1, 3]);  view_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_696: "f32[8, 1024, 64]" = torch.ops.aten.reshape.default(permute_587, [8, 1024, 64]);  permute_587 = None
    bmm_88: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_588, view_696);  permute_588 = None
    bmm_89: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_696, permute_589);  view_696 = permute_589 = None
    view_697: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_88, [1, 8, 1024, 64]);  bmm_88 = None
    view_698: "f32[1, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_89, [1, 8, 1024, 1024]);  bmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_54: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_35, torch.float32);  getitem_35 = None
    mul_377: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_54, 1.1111111111111112);  convert_element_type_54 = None
    mul_378: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_698, mul_377);  view_698 = mul_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_379: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_378, alias_108);  mul_378 = None
    sum_82: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_379, [-1], True)
    mul_380: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_108, sum_82);  alias_108 = sum_82 = None
    sub_39: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_379, mul_380);  mul_379 = mul_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_15: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_39, 0);  sub_39 = None
    as_strided_scatter_26: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_15, [8, 1024, 1024], [1048576, 1024, 1], 0);  squeeze_15 = None
    as_strided_94: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_26, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_26 = None
    new_empty_strided_13: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_94, [8, 1024, 1024], [1048576, 1024, 1])
    as_strided_96: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_94, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    as_strided_scatter_27: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(as_strided_94, as_strided_96, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  as_strided_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_188: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(as_strided_89, as_strided_96);  as_strided_89 = as_strided_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    bmm_90: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_590, as_strided_scatter_27);  permute_590 = None
    bmm_91: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_27, permute_591);  as_strided_scatter_27 = permute_591 = None
    view_699: "f32[1, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_90, [1, 8, 64, 1024]);  bmm_90 = None
    view_700: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_91, [1, 8, 1024, 64]);  bmm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_592: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_699, [0, 1, 3, 2]);  view_699 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_593: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_697, [0, 2, 1, 3]);  view_697 = None
    clone_118: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_593, memory_format = torch.contiguous_format);  permute_593 = None
    view_701: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_118, [1, 1024, 512]);  clone_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_702: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_701, [1024, 512]);  view_701 = None
    permute_594: "f32[512, 1024]" = torch.ops.aten.permute.default(view_702, [1, 0])
    mm_237: "f32[512, 512]" = torch.ops.aten.mm.default(permute_594, view_97);  permute_594 = None
    permute_595: "f32[512, 512]" = torch.ops.aten.permute.default(mm_237, [1, 0]);  mm_237 = None
    mm_238: "f32[1024, 512]" = torch.ops.aten.mm.default(view_702, permute_596);  view_702 = permute_596 = None
    view_703: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_238, [1, 1024, 512]);  mm_238 = None
    permute_597: "f32[512, 512]" = torch.ops.aten.permute.default(permute_595, [1, 0]);  permute_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_598: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_592, [0, 2, 1, 3]);  permute_592 = None
    view_704: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(permute_598, [1, 1024, 512]);  permute_598 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_705: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_704, [1024, 512]);  view_704 = None
    permute_599: "f32[512, 1024]" = torch.ops.aten.permute.default(view_705, [1, 0])
    mm_239: "f32[512, 512]" = torch.ops.aten.mm.default(permute_599, view_97);  permute_599 = None
    permute_600: "f32[512, 512]" = torch.ops.aten.permute.default(mm_239, [1, 0]);  mm_239 = None
    mm_240: "f32[1024, 512]" = torch.ops.aten.mm.default(view_705, permute_601);  view_705 = permute_601 = None
    view_706: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_240, [1, 1024, 512]);  mm_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_189: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(view_703, view_706);  view_703 = view_706 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_602: "f32[512, 512]" = torch.ops.aten.permute.default(permute_600, [1, 0]);  permute_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_603: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_700, [0, 2, 1, 3]);  view_700 = None
    clone_119: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_603, memory_format = torch.contiguous_format);  permute_603 = None
    view_707: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_119, [1, 1024, 512]);  clone_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_708: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_707, [1024, 512]);  view_707 = None
    permute_604: "f32[512, 1024]" = torch.ops.aten.permute.default(view_708, [1, 0])
    mm_241: "f32[512, 512]" = torch.ops.aten.mm.default(permute_604, view_97);  permute_604 = view_97 = None
    permute_605: "f32[512, 512]" = torch.ops.aten.permute.default(mm_241, [1, 0]);  mm_241 = None
    mm_242: "f32[1024, 512]" = torch.ops.aten.mm.default(view_708, permute_606);  view_708 = permute_606 = None
    view_709: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_242, [1, 1024, 512]);  mm_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_190: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_189, view_709);  add_189 = view_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_607: "f32[512, 512]" = torch.ops.aten.permute.default(permute_605, [1, 0]);  permute_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_381: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_190, primals_9);  primals_9 = None
    mul_382: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_190, mul_19);  add_190 = mul_19 = None
    sum_83: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_382, [0, 1], True);  mul_382 = None
    view_710: "f32[512]" = torch.ops.aten.reshape.default(sum_83, [512]);  sum_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_383: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_381, add_23)
    mul_384: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_381, rsqrt_8);  mul_381 = None
    sum_84: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_383, [2], True);  mul_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_191: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_187, mul_384);  add_187 = mul_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_79: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_8, 3);  rsqrt_8 = None
    mul_385: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_84, -0.5);  sum_84 = None
    mul_386: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_385, pow_79);  mul_385 = pow_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_95: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_386, [1, 1024, 512]);  mul_386 = None
    div_47: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_95, 512);  expand_95 = None
    pow_80: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_23, 1.0);  add_23 = None
    mul_387: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_80, 2.0);  pow_80 = None
    mul_388: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_47, mul_387);  div_47 = mul_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_192: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_191, mul_388);  add_191 = mul_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_55: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_33, torch.float32);  getitem_33 = None
    mul_389: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_55, 1.1111111111111112);  convert_element_type_55 = None
    mul_390: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_192, mul_389);  mul_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_711: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_390, [1024, 512]);  mul_390 = None
    permute_608: "f32[512, 1024]" = torch.ops.aten.permute.default(view_711, [1, 0])
    mm_243: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_608, view_95);  permute_608 = view_95 = None
    permute_609: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_243, [1, 0]);  mm_243 = None
    mm_244: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_711, permute_610);  view_711 = permute_610 = None
    view_712: "f32[1, 1024, 2048]" = torch.ops.aten.reshape.default(mm_244, [1, 1024, 2048]);  mm_244 = None
    permute_611: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_609, [1, 0]);  permute_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_56: "f32[1, 1024, 2048]" = torch.ops.prims.convert_element_type.default(getitem_31, torch.float32);  getitem_31 = None
    mul_391: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_56, 1.1111111111111112);  convert_element_type_56 = None
    mul_392: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_712, mul_391);  view_712 = mul_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    where_16: "f32[1, 1024, 2048]" = torch.ops.aten.where.self(le_9, full_default_7, mul_392);  le_9 = mul_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_713: "f32[1024, 2048]" = torch.ops.aten.reshape.default(where_16, [1024, 2048]);  where_16 = None
    permute_612: "f32[2048, 1024]" = torch.ops.aten.permute.default(view_713, [1, 0])
    mm_245: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_612, view_93);  permute_612 = view_93 = None
    permute_613: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_245, [1, 0]);  mm_245 = None
    mm_246: "f32[1024, 512]" = torch.ops.aten.mm.default(view_713, permute_614);  view_713 = permute_614 = None
    view_714: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_246, [1, 1024, 512]);  mm_246 = None
    permute_615: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_613, [1, 0]);  permute_613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_393: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_714, primals_8);  primals_8 = None
    mul_394: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_714, mul_17);  view_714 = mul_17 = None
    sum_85: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_394, [0, 1], True);  mul_394 = None
    view_715: "f32[512]" = torch.ops.aten.reshape.default(sum_85, [512]);  sum_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_395: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_393, add_21)
    mul_396: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_393, rsqrt_7);  mul_393 = None
    sum_86: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_395, [2], True);  mul_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_193: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_192, mul_396);  add_192 = mul_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_81: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_7, 3);  rsqrt_7 = None
    mul_397: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_86, -0.5);  sum_86 = None
    mul_398: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_397, pow_81);  mul_397 = pow_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_96: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_398, [1, 1024, 512]);  mul_398 = None
    div_48: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_96, 512);  expand_96 = None
    pow_82: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_21, 1.0);  add_21 = None
    mul_399: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_82, 2.0);  pow_82 = None
    mul_400: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_48, mul_399);  div_48 = mul_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_194: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_193, mul_400);  add_193 = mul_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_57: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_29, torch.float32);  getitem_29 = None
    mul_401: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_57, 1.1111111111111112);  convert_element_type_57 = None
    mul_402: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_194, mul_401);  mul_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_716: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_402, [1024, 512]);  mul_402 = None
    permute_616: "f32[512, 1024]" = torch.ops.aten.permute.default(view_716, [1, 0])
    mm_247: "f32[512, 512]" = torch.ops.aten.mm.default(permute_616, view_91);  permute_616 = view_91 = None
    permute_617: "f32[512, 512]" = torch.ops.aten.permute.default(mm_247, [1, 0]);  mm_247 = None
    mm_248: "f32[1024, 512]" = torch.ops.aten.mm.default(view_716, permute_618);  view_716 = permute_618 = None
    view_717: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_248, [1, 1024, 512]);  mm_248 = None
    permute_619: "f32[512, 512]" = torch.ops.aten.permute.default(permute_617, [1, 0]);  permute_617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_718: "f32[1, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_717, [1, 1024, 8, 64]);  view_717 = None
    permute_620: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_718, [0, 2, 1, 3]);  view_718 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_719: "f32[8, 1024, 64]" = torch.ops.aten.reshape.default(permute_620, [8, 1024, 64]);  permute_620 = None
    bmm_92: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_621, view_719);  permute_621 = None
    bmm_93: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_719, permute_622);  view_719 = permute_622 = None
    view_720: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_92, [1, 8, 1024, 64]);  bmm_92 = None
    view_721: "f32[1, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_93, [1, 8, 1024, 1024]);  bmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_58: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_27, torch.float32);  getitem_27 = None
    mul_403: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_58, 1.1111111111111112);  convert_element_type_58 = None
    mul_404: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_721, mul_403);  view_721 = mul_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_405: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_404, alias_112);  mul_404 = None
    sum_87: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_405, [-1], True)
    mul_406: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_112, sum_87);  alias_112 = sum_87 = None
    sub_40: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_405, mul_406);  mul_405 = mul_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_16: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_40, 0);  sub_40 = None
    as_strided_scatter_28: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_16, [8, 1024, 1024], [1048576, 1024, 1], 0);  squeeze_16 = None
    as_strided_101: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_28, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_28 = None
    new_empty_strided_14: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_101, [8, 1024, 1024], [1048576, 1024, 1])
    as_strided_103: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_101, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    as_strided_scatter_29: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(as_strided_101, as_strided_103, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  as_strided_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_195: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_188, as_strided_103);  add_188 = as_strided_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    bmm_94: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_623, as_strided_scatter_29);  permute_623 = None
    bmm_95: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_29, permute_624);  as_strided_scatter_29 = permute_624 = None
    view_722: "f32[1, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_94, [1, 8, 64, 1024]);  bmm_94 = None
    view_723: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_95, [1, 8, 1024, 64]);  bmm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_625: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_722, [0, 1, 3, 2]);  view_722 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_626: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_720, [0, 2, 1, 3]);  view_720 = None
    clone_125: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_626, memory_format = torch.contiguous_format);  permute_626 = None
    view_724: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_125, [1, 1024, 512]);  clone_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_725: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_724, [1024, 512]);  view_724 = None
    permute_627: "f32[512, 1024]" = torch.ops.aten.permute.default(view_725, [1, 0])
    mm_249: "f32[512, 512]" = torch.ops.aten.mm.default(permute_627, view_73);  permute_627 = None
    permute_628: "f32[512, 512]" = torch.ops.aten.permute.default(mm_249, [1, 0]);  mm_249 = None
    mm_250: "f32[1024, 512]" = torch.ops.aten.mm.default(view_725, permute_629);  view_725 = permute_629 = None
    view_726: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_250, [1, 1024, 512]);  mm_250 = None
    permute_630: "f32[512, 512]" = torch.ops.aten.permute.default(permute_628, [1, 0]);  permute_628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_631: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_625, [0, 2, 1, 3]);  permute_625 = None
    view_727: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(permute_631, [1, 1024, 512]);  permute_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_728: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_727, [1024, 512]);  view_727 = None
    permute_632: "f32[512, 1024]" = torch.ops.aten.permute.default(view_728, [1, 0])
    mm_251: "f32[512, 512]" = torch.ops.aten.mm.default(permute_632, view_73);  permute_632 = None
    permute_633: "f32[512, 512]" = torch.ops.aten.permute.default(mm_251, [1, 0]);  mm_251 = None
    mm_252: "f32[1024, 512]" = torch.ops.aten.mm.default(view_728, permute_634);  view_728 = permute_634 = None
    view_729: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_252, [1, 1024, 512]);  mm_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_196: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(view_726, view_729);  view_726 = view_729 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_635: "f32[512, 512]" = torch.ops.aten.permute.default(permute_633, [1, 0]);  permute_633 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_636: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_723, [0, 2, 1, 3]);  view_723 = None
    clone_126: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_636, memory_format = torch.contiguous_format);  permute_636 = None
    view_730: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_126, [1, 1024, 512]);  clone_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_731: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_730, [1024, 512]);  view_730 = None
    permute_637: "f32[512, 1024]" = torch.ops.aten.permute.default(view_731, [1, 0])
    mm_253: "f32[512, 512]" = torch.ops.aten.mm.default(permute_637, view_73);  permute_637 = view_73 = None
    permute_638: "f32[512, 512]" = torch.ops.aten.permute.default(mm_253, [1, 0]);  mm_253 = None
    mm_254: "f32[1024, 512]" = torch.ops.aten.mm.default(view_731, permute_639);  view_731 = permute_639 = None
    view_732: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_254, [1, 1024, 512]);  mm_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_197: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_196, view_732);  add_196 = view_732 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_640: "f32[512, 512]" = torch.ops.aten.permute.default(permute_638, [1, 0]);  permute_638 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_407: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_197, primals_7);  primals_7 = None
    mul_408: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_197, mul_15);  add_197 = mul_15 = None
    sum_88: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_408, [0, 1], True);  mul_408 = None
    view_733: "f32[512]" = torch.ops.aten.reshape.default(sum_88, [512]);  sum_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_409: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_407, add_18)
    mul_410: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_407, rsqrt_6);  mul_407 = None
    sum_89: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_409, [2], True);  mul_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_198: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_194, mul_410);  add_194 = mul_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_83: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_6, 3);  rsqrt_6 = None
    mul_411: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_89, -0.5);  sum_89 = None
    mul_412: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_411, pow_83);  mul_411 = pow_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_97: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_412, [1, 1024, 512]);  mul_412 = None
    div_49: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_97, 512);  expand_97 = None
    pow_84: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_18, 1.0);  add_18 = None
    mul_413: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_84, 2.0);  pow_84 = None
    mul_414: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_49, mul_413);  div_49 = mul_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_199: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_198, mul_414);  add_198 = mul_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_59: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_25, torch.float32);  getitem_25 = None
    mul_415: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_59, 1.1111111111111112);  convert_element_type_59 = None
    mul_416: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_199, mul_415);  mul_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_734: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_416, [1024, 512]);  mul_416 = None
    permute_641: "f32[512, 1024]" = torch.ops.aten.permute.default(view_734, [1, 0])
    mm_255: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_641, view_71);  permute_641 = view_71 = None
    permute_642: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_255, [1, 0]);  mm_255 = None
    mm_256: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_734, permute_643);  view_734 = permute_643 = None
    view_735: "f32[1, 1024, 2048]" = torch.ops.aten.reshape.default(mm_256, [1, 1024, 2048]);  mm_256 = None
    permute_644: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_642, [1, 0]);  permute_642 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_60: "f32[1, 1024, 2048]" = torch.ops.prims.convert_element_type.default(getitem_23, torch.float32);  getitem_23 = None
    mul_417: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_60, 1.1111111111111112);  convert_element_type_60 = None
    mul_418: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_735, mul_417);  view_735 = mul_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    where_17: "f32[1, 1024, 2048]" = torch.ops.aten.where.self(le_10, full_default_7, mul_418);  le_10 = mul_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_736: "f32[1024, 2048]" = torch.ops.aten.reshape.default(where_17, [1024, 2048]);  where_17 = None
    permute_645: "f32[2048, 1024]" = torch.ops.aten.permute.default(view_736, [1, 0])
    mm_257: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_645, view_69);  permute_645 = view_69 = None
    permute_646: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_257, [1, 0]);  mm_257 = None
    mm_258: "f32[1024, 512]" = torch.ops.aten.mm.default(view_736, permute_647);  view_736 = permute_647 = None
    view_737: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_258, [1, 1024, 512]);  mm_258 = None
    permute_648: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_646, [1, 0]);  permute_646 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_419: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_737, primals_6);  primals_6 = None
    mul_420: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_737, mul_13);  view_737 = mul_13 = None
    sum_90: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_420, [0, 1], True);  mul_420 = None
    view_738: "f32[512]" = torch.ops.aten.reshape.default(sum_90, [512]);  sum_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_421: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_419, add_16)
    mul_422: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_419, rsqrt_5);  mul_419 = None
    sum_91: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_421, [2], True);  mul_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_200: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_199, mul_422);  add_199 = mul_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_85: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_5, 3);  rsqrt_5 = None
    mul_423: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_91, -0.5);  sum_91 = None
    mul_424: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_423, pow_85);  mul_423 = pow_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_98: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_424, [1, 1024, 512]);  mul_424 = None
    div_50: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_98, 512);  expand_98 = None
    pow_86: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_16, 1.0);  add_16 = None
    mul_425: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_86, 2.0);  pow_86 = None
    mul_426: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_50, mul_425);  div_50 = mul_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_201: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_200, mul_426);  add_200 = mul_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_61: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_21, torch.float32);  getitem_21 = None
    mul_427: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_61, 1.1111111111111112);  convert_element_type_61 = None
    mul_428: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_201, mul_427);  mul_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_739: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_428, [1024, 512]);  mul_428 = None
    permute_649: "f32[512, 1024]" = torch.ops.aten.permute.default(view_739, [1, 0])
    mm_259: "f32[512, 512]" = torch.ops.aten.mm.default(permute_649, view_67);  permute_649 = view_67 = None
    permute_650: "f32[512, 512]" = torch.ops.aten.permute.default(mm_259, [1, 0]);  mm_259 = None
    mm_260: "f32[1024, 512]" = torch.ops.aten.mm.default(view_739, permute_651);  view_739 = permute_651 = None
    view_740: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_260, [1, 1024, 512]);  mm_260 = None
    permute_652: "f32[512, 512]" = torch.ops.aten.permute.default(permute_650, [1, 0]);  permute_650 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_741: "f32[1, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_740, [1, 1024, 8, 64]);  view_740 = None
    permute_653: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_741, [0, 2, 1, 3]);  view_741 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_742: "f32[8, 1024, 64]" = torch.ops.aten.reshape.default(permute_653, [8, 1024, 64]);  permute_653 = None
    bmm_96: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_654, view_742);  permute_654 = None
    bmm_97: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_742, permute_655);  view_742 = permute_655 = None
    view_743: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_96, [1, 8, 1024, 64]);  bmm_96 = None
    view_744: "f32[1, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_97, [1, 8, 1024, 1024]);  bmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_62: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_19, torch.float32);  getitem_19 = None
    mul_429: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_62, 1.1111111111111112);  convert_element_type_62 = None
    mul_430: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_744, mul_429);  view_744 = mul_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_431: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_430, alias_116);  mul_430 = None
    sum_92: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_431, [-1], True)
    mul_432: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_116, sum_92);  alias_116 = sum_92 = None
    sub_41: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_431, mul_432);  mul_431 = mul_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_17: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_41, 0);  sub_41 = None
    as_strided_scatter_30: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_17, [8, 1024, 1024], [1048576, 1024, 1], 0);  squeeze_17 = None
    as_strided_108: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_30, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_30 = None
    new_empty_strided_15: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_108, [8, 1024, 1024], [1048576, 1024, 1])
    as_strided_110: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_108, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    as_strided_scatter_31: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(as_strided_108, as_strided_110, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  as_strided_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_202: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_195, as_strided_110);  add_195 = as_strided_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    bmm_98: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_656, as_strided_scatter_31);  permute_656 = None
    bmm_99: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_31, permute_657);  as_strided_scatter_31 = permute_657 = None
    view_745: "f32[1, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_98, [1, 8, 64, 1024]);  bmm_98 = None
    view_746: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_99, [1, 8, 1024, 64]);  bmm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_658: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_745, [0, 1, 3, 2]);  view_745 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_659: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_743, [0, 2, 1, 3]);  view_743 = None
    clone_132: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_659, memory_format = torch.contiguous_format);  permute_659 = None
    view_747: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_132, [1, 1024, 512]);  clone_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_748: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_747, [1024, 512]);  view_747 = None
    permute_660: "f32[512, 1024]" = torch.ops.aten.permute.default(view_748, [1, 0])
    mm_261: "f32[512, 512]" = torch.ops.aten.mm.default(permute_660, view_49);  permute_660 = None
    permute_661: "f32[512, 512]" = torch.ops.aten.permute.default(mm_261, [1, 0]);  mm_261 = None
    mm_262: "f32[1024, 512]" = torch.ops.aten.mm.default(view_748, permute_662);  view_748 = permute_662 = None
    view_749: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_262, [1, 1024, 512]);  mm_262 = None
    permute_663: "f32[512, 512]" = torch.ops.aten.permute.default(permute_661, [1, 0]);  permute_661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_664: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_658, [0, 2, 1, 3]);  permute_658 = None
    view_750: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(permute_664, [1, 1024, 512]);  permute_664 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_751: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_750, [1024, 512]);  view_750 = None
    permute_665: "f32[512, 1024]" = torch.ops.aten.permute.default(view_751, [1, 0])
    mm_263: "f32[512, 512]" = torch.ops.aten.mm.default(permute_665, view_49);  permute_665 = None
    permute_666: "f32[512, 512]" = torch.ops.aten.permute.default(mm_263, [1, 0]);  mm_263 = None
    mm_264: "f32[1024, 512]" = torch.ops.aten.mm.default(view_751, permute_667);  view_751 = permute_667 = None
    view_752: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_264, [1, 1024, 512]);  mm_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_203: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(view_749, view_752);  view_749 = view_752 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_668: "f32[512, 512]" = torch.ops.aten.permute.default(permute_666, [1, 0]);  permute_666 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_669: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_746, [0, 2, 1, 3]);  view_746 = None
    clone_133: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_669, memory_format = torch.contiguous_format);  permute_669 = None
    view_753: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_133, [1, 1024, 512]);  clone_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_754: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_753, [1024, 512]);  view_753 = None
    permute_670: "f32[512, 1024]" = torch.ops.aten.permute.default(view_754, [1, 0])
    mm_265: "f32[512, 512]" = torch.ops.aten.mm.default(permute_670, view_49);  permute_670 = view_49 = None
    permute_671: "f32[512, 512]" = torch.ops.aten.permute.default(mm_265, [1, 0]);  mm_265 = None
    mm_266: "f32[1024, 512]" = torch.ops.aten.mm.default(view_754, permute_672);  view_754 = permute_672 = None
    view_755: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_266, [1, 1024, 512]);  mm_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_204: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_203, view_755);  add_203 = view_755 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_673: "f32[512, 512]" = torch.ops.aten.permute.default(permute_671, [1, 0]);  permute_671 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_433: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_204, primals_5);  primals_5 = None
    mul_434: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_204, mul_11);  add_204 = mul_11 = None
    sum_93: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_434, [0, 1], True);  mul_434 = None
    view_756: "f32[512]" = torch.ops.aten.reshape.default(sum_93, [512]);  sum_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_435: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_433, add_13)
    mul_436: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_433, rsqrt_4);  mul_433 = None
    sum_94: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_435, [2], True);  mul_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_205: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_201, mul_436);  add_201 = mul_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_87: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_4, 3);  rsqrt_4 = None
    mul_437: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_94, -0.5);  sum_94 = None
    mul_438: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_437, pow_87);  mul_437 = pow_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_99: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_438, [1, 1024, 512]);  mul_438 = None
    div_51: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_99, 512);  expand_99 = None
    pow_88: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_13, 1.0);  add_13 = None
    mul_439: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_88, 2.0);  pow_88 = None
    mul_440: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_51, mul_439);  div_51 = mul_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_206: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_205, mul_440);  add_205 = mul_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_63: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_17, torch.float32);  getitem_17 = None
    mul_441: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_63, 1.1111111111111112);  convert_element_type_63 = None
    mul_442: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_206, mul_441);  mul_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_757: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_442, [1024, 512]);  mul_442 = None
    permute_674: "f32[512, 1024]" = torch.ops.aten.permute.default(view_757, [1, 0])
    mm_267: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_674, view_47);  permute_674 = view_47 = None
    permute_675: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_267, [1, 0]);  mm_267 = None
    mm_268: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_757, permute_676);  view_757 = permute_676 = None
    view_758: "f32[1, 1024, 2048]" = torch.ops.aten.reshape.default(mm_268, [1, 1024, 2048]);  mm_268 = None
    permute_677: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_675, [1, 0]);  permute_675 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_64: "f32[1, 1024, 2048]" = torch.ops.prims.convert_element_type.default(getitem_15, torch.float32);  getitem_15 = None
    mul_443: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_64, 1.1111111111111112);  convert_element_type_64 = None
    mul_444: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_758, mul_443);  view_758 = mul_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    where_18: "f32[1, 1024, 2048]" = torch.ops.aten.where.self(le_11, full_default_7, mul_444);  le_11 = mul_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_759: "f32[1024, 2048]" = torch.ops.aten.reshape.default(where_18, [1024, 2048]);  where_18 = None
    permute_678: "f32[2048, 1024]" = torch.ops.aten.permute.default(view_759, [1, 0])
    mm_269: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_678, view_45);  permute_678 = view_45 = None
    permute_679: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_269, [1, 0]);  mm_269 = None
    mm_270: "f32[1024, 512]" = torch.ops.aten.mm.default(view_759, permute_680);  view_759 = permute_680 = None
    view_760: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_270, [1, 1024, 512]);  mm_270 = None
    permute_681: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_679, [1, 0]);  permute_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_445: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_760, primals_4);  primals_4 = None
    mul_446: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_760, mul_9);  view_760 = mul_9 = None
    sum_95: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_446, [0, 1], True);  mul_446 = None
    view_761: "f32[512]" = torch.ops.aten.reshape.default(sum_95, [512]);  sum_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_447: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_445, add_11)
    mul_448: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_445, rsqrt_3);  mul_445 = None
    sum_96: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_447, [2], True);  mul_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_207: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_206, mul_448);  add_206 = mul_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_89: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_3, 3);  rsqrt_3 = None
    mul_449: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_96, -0.5);  sum_96 = None
    mul_450: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_449, pow_89);  mul_449 = pow_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_100: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_450, [1, 1024, 512]);  mul_450 = None
    div_52: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_100, 512);  expand_100 = None
    pow_90: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_11, 1.0);  add_11 = None
    mul_451: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_90, 2.0);  pow_90 = None
    mul_452: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_52, mul_451);  div_52 = mul_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_208: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_207, mul_452);  add_207 = mul_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_65: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_13, torch.float32);  getitem_13 = None
    mul_453: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_65, 1.1111111111111112);  convert_element_type_65 = None
    mul_454: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_208, mul_453);  mul_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_762: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_454, [1024, 512]);  mul_454 = None
    permute_682: "f32[512, 1024]" = torch.ops.aten.permute.default(view_762, [1, 0])
    mm_271: "f32[512, 512]" = torch.ops.aten.mm.default(permute_682, view_43);  permute_682 = view_43 = None
    permute_683: "f32[512, 512]" = torch.ops.aten.permute.default(mm_271, [1, 0]);  mm_271 = None
    mm_272: "f32[1024, 512]" = torch.ops.aten.mm.default(view_762, permute_684);  view_762 = permute_684 = None
    view_763: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_272, [1, 1024, 512]);  mm_272 = None
    permute_685: "f32[512, 512]" = torch.ops.aten.permute.default(permute_683, [1, 0]);  permute_683 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_764: "f32[1, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_763, [1, 1024, 8, 64]);  view_763 = None
    permute_686: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_764, [0, 2, 1, 3]);  view_764 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_765: "f32[8, 1024, 64]" = torch.ops.aten.reshape.default(permute_686, [8, 1024, 64]);  permute_686 = None
    bmm_100: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_687, view_765);  permute_687 = None
    bmm_101: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_765, permute_688);  view_765 = permute_688 = None
    view_766: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_100, [1, 8, 1024, 64]);  bmm_100 = None
    view_767: "f32[1, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_101, [1, 8, 1024, 1024]);  bmm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_66: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_11, torch.float32);  getitem_11 = None
    mul_455: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_66, 1.1111111111111112);  convert_element_type_66 = None
    mul_456: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_767, mul_455);  view_767 = mul_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_457: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_456, alias_120);  mul_456 = None
    sum_97: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_457, [-1], True)
    mul_458: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_120, sum_97);  alias_120 = sum_97 = None
    sub_42: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_457, mul_458);  mul_457 = mul_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_18: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_42, 0);  sub_42 = None
    as_strided_scatter_32: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_18, [8, 1024, 1024], [1048576, 1024, 1], 0);  squeeze_18 = None
    as_strided_115: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_32, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_32 = None
    new_empty_strided_16: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_115, [8, 1024, 1024], [1048576, 1024, 1])
    as_strided_117: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_115, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    as_strided_scatter_33: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(as_strided_115, as_strided_117, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  as_strided_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_209: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_202, as_strided_117);  add_202 = as_strided_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    bmm_102: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_689, as_strided_scatter_33);  permute_689 = None
    bmm_103: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_33, permute_690);  as_strided_scatter_33 = permute_690 = None
    view_768: "f32[1, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_102, [1, 8, 64, 1024]);  bmm_102 = None
    view_769: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_103, [1, 8, 1024, 64]);  bmm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_691: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_768, [0, 1, 3, 2]);  view_768 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_692: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_766, [0, 2, 1, 3]);  view_766 = None
    clone_139: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_692, memory_format = torch.contiguous_format);  permute_692 = None
    view_770: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_139, [1, 1024, 512]);  clone_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_771: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_770, [1024, 512]);  view_770 = None
    permute_693: "f32[512, 1024]" = torch.ops.aten.permute.default(view_771, [1, 0])
    mm_273: "f32[512, 512]" = torch.ops.aten.mm.default(permute_693, view_25);  permute_693 = None
    permute_694: "f32[512, 512]" = torch.ops.aten.permute.default(mm_273, [1, 0]);  mm_273 = None
    mm_274: "f32[1024, 512]" = torch.ops.aten.mm.default(view_771, permute_695);  view_771 = permute_695 = None
    view_772: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_274, [1, 1024, 512]);  mm_274 = None
    permute_696: "f32[512, 512]" = torch.ops.aten.permute.default(permute_694, [1, 0]);  permute_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_697: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_691, [0, 2, 1, 3]);  permute_691 = None
    view_773: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(permute_697, [1, 1024, 512]);  permute_697 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_774: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_773, [1024, 512]);  view_773 = None
    permute_698: "f32[512, 1024]" = torch.ops.aten.permute.default(view_774, [1, 0])
    mm_275: "f32[512, 512]" = torch.ops.aten.mm.default(permute_698, view_25);  permute_698 = None
    permute_699: "f32[512, 512]" = torch.ops.aten.permute.default(mm_275, [1, 0]);  mm_275 = None
    mm_276: "f32[1024, 512]" = torch.ops.aten.mm.default(view_774, permute_700);  view_774 = permute_700 = None
    view_775: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_276, [1, 1024, 512]);  mm_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_210: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(view_772, view_775);  view_772 = view_775 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_701: "f32[512, 512]" = torch.ops.aten.permute.default(permute_699, [1, 0]);  permute_699 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_702: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_769, [0, 2, 1, 3]);  view_769 = None
    clone_140: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_702, memory_format = torch.contiguous_format);  permute_702 = None
    view_776: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_140, [1, 1024, 512]);  clone_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_777: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_776, [1024, 512]);  view_776 = None
    permute_703: "f32[512, 1024]" = torch.ops.aten.permute.default(view_777, [1, 0])
    mm_277: "f32[512, 512]" = torch.ops.aten.mm.default(permute_703, view_25);  permute_703 = view_25 = None
    permute_704: "f32[512, 512]" = torch.ops.aten.permute.default(mm_277, [1, 0]);  mm_277 = None
    mm_278: "f32[1024, 512]" = torch.ops.aten.mm.default(view_777, permute_705);  view_777 = permute_705 = None
    view_778: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_278, [1, 1024, 512]);  mm_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_211: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_210, view_778);  add_210 = view_778 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_706: "f32[512, 512]" = torch.ops.aten.permute.default(permute_704, [1, 0]);  permute_704 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_459: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_211, primals_3);  primals_3 = None
    mul_460: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_211, mul_7);  add_211 = mul_7 = None
    sum_98: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_460, [0, 1], True);  mul_460 = None
    view_779: "f32[512]" = torch.ops.aten.reshape.default(sum_98, [512]);  sum_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_461: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_459, add_8)
    mul_462: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_459, rsqrt_2);  mul_459 = None
    sum_99: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_461, [2], True);  mul_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_212: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_208, mul_462);  add_208 = mul_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_91: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_2, 3);  rsqrt_2 = None
    mul_463: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_99, -0.5);  sum_99 = None
    mul_464: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_463, pow_91);  mul_463 = pow_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_101: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_464, [1, 1024, 512]);  mul_464 = None
    div_53: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_101, 512);  expand_101 = None
    pow_92: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_8, 1.0);  add_8 = None
    mul_465: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_92, 2.0);  pow_92 = None
    mul_466: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_53, mul_465);  div_53 = mul_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_213: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_212, mul_466);  add_212 = mul_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_67: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_9, torch.float32);  getitem_9 = None
    mul_467: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_67, 1.1111111111111112);  convert_element_type_67 = None
    mul_468: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_213, mul_467);  mul_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_780: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_468, [1024, 512]);  mul_468 = None
    permute_707: "f32[512, 1024]" = torch.ops.aten.permute.default(view_780, [1, 0])
    mm_279: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_707, view_23);  permute_707 = view_23 = None
    permute_708: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_279, [1, 0]);  mm_279 = None
    mm_280: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_780, permute_709);  view_780 = permute_709 = None
    view_781: "f32[1, 1024, 2048]" = torch.ops.aten.reshape.default(mm_280, [1, 1024, 2048]);  mm_280 = None
    permute_710: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_708, [1, 0]);  permute_708 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_68: "f32[1, 1024, 2048]" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
    mul_469: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_68, 1.1111111111111112);  convert_element_type_68 = None
    mul_470: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_781, mul_469);  view_781 = mul_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    where_19: "f32[1, 1024, 2048]" = torch.ops.aten.where.self(le_12, full_default_7, mul_470);  le_12 = mul_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_782: "f32[1024, 2048]" = torch.ops.aten.reshape.default(where_19, [1024, 2048]);  where_19 = None
    permute_711: "f32[2048, 1024]" = torch.ops.aten.permute.default(view_782, [1, 0])
    mm_281: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_711, view_21);  permute_711 = view_21 = None
    permute_712: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_281, [1, 0]);  mm_281 = None
    mm_282: "f32[1024, 512]" = torch.ops.aten.mm.default(view_782, permute_713);  view_782 = permute_713 = None
    view_783: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_282, [1, 1024, 512]);  mm_282 = None
    permute_714: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_712, [1, 0]);  permute_712 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_471: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_783, primals_2);  primals_2 = None
    mul_472: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_783, mul_5);  view_783 = mul_5 = None
    sum_100: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_472, [0, 1], True);  mul_472 = None
    view_784: "f32[512]" = torch.ops.aten.reshape.default(sum_100, [512]);  sum_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_473: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_471, add_6)
    mul_474: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_471, rsqrt_1);  mul_471 = None
    sum_101: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_473, [2], True);  mul_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_214: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_213, mul_474);  add_213 = mul_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_93: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_1, 3);  rsqrt_1 = None
    mul_475: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_101, -0.5);  sum_101 = None
    mul_476: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_475, pow_93);  mul_475 = pow_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_102: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_476, [1, 1024, 512]);  mul_476 = None
    div_54: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_102, 512);  expand_102 = None
    pow_94: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_6, 1.0);  add_6 = None
    mul_477: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_94, 2.0);  pow_94 = None
    mul_478: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_54, mul_477);  div_54 = mul_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_215: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_214, mul_478);  add_214 = mul_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_69: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_5, torch.float32);  getitem_5 = None
    mul_479: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_69, 1.1111111111111112);  convert_element_type_69 = None
    mul_480: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_215, mul_479);  mul_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_785: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_480, [1024, 512]);  mul_480 = None
    permute_715: "f32[512, 1024]" = torch.ops.aten.permute.default(view_785, [1, 0])
    mm_283: "f32[512, 512]" = torch.ops.aten.mm.default(permute_715, view_19);  permute_715 = view_19 = None
    permute_716: "f32[512, 512]" = torch.ops.aten.permute.default(mm_283, [1, 0]);  mm_283 = None
    mm_284: "f32[1024, 512]" = torch.ops.aten.mm.default(view_785, permute_717);  view_785 = permute_717 = None
    view_786: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_284, [1, 1024, 512]);  mm_284 = None
    permute_718: "f32[512, 512]" = torch.ops.aten.permute.default(permute_716, [1, 0]);  permute_716 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_787: "f32[1, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_786, [1, 1024, 8, 64]);  view_786 = None
    permute_719: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_787, [0, 2, 1, 3]);  view_787 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_788: "f32[8, 1024, 64]" = torch.ops.aten.reshape.default(permute_719, [8, 1024, 64]);  permute_719 = None
    bmm_104: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_720, view_788);  permute_720 = None
    bmm_105: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_788, permute_721);  view_788 = permute_721 = None
    view_789: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_104, [1, 8, 1024, 64]);  bmm_104 = None
    view_790: "f32[1, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_105, [1, 8, 1024, 1024]);  bmm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_70: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
    mul_481: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_70, 1.1111111111111112);  convert_element_type_70 = None
    mul_482: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_790, mul_481);  view_790 = mul_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_483: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_482, alias_124);  mul_482 = None
    sum_102: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_483, [-1], True)
    mul_484: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_124, sum_102);  alias_124 = sum_102 = None
    sub_43: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_483, mul_484);  mul_483 = mul_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_19: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_43, 0);  sub_43 = None
    as_strided_scatter_34: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_19, [8, 1024, 1024], [1048576, 1024, 1], 0);  full_8 = squeeze_19 = None
    as_strided_122: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_34, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_34 = None
    new_empty_strided_17: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_122, [8, 1024, 1024], [1048576, 1024, 1])
    as_strided_124: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_122, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    as_strided_scatter_35: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(as_strided_122, as_strided_124, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  as_strided_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_216: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_209, as_strided_124);  add_209 = as_strided_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:451, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
    squeeze_20: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(add_216, 0);  add_216 = None
    permute_722: "f32[1024, 1024, 8]" = torch.ops.aten.permute.default(squeeze_20, [1, 2, 0]);  squeeze_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:450, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
    where_20: "f32[1024, 1024, 8]" = torch.ops.aten.where.self(full_default_17, full_default_7, permute_722);  full_default_17 = permute_722 = None
    clone_146: "f32[1024, 1024, 8]" = torch.ops.aten.clone.default(where_20, memory_format = torch.contiguous_format);  where_20 = None
    _unsafe_index_put_2: "f32[32, 8]" = torch.ops.prims._unsafe_index_put_.default(full_default_19, [add_3], clone_146, True);  full_default_19 = add_3 = clone_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    bmm_106: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_723, as_strided_scatter_35);  permute_723 = None
    bmm_107: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_35, permute_724);  as_strided_scatter_35 = permute_724 = None
    view_791: "f32[1, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_106, [1, 8, 64, 1024]);  bmm_106 = None
    view_792: "f32[1, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_107, [1, 8, 1024, 64]);  bmm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_725: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_791, [0, 1, 3, 2]);  view_791 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_726: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_789, [0, 2, 1, 3]);  view_789 = None
    clone_147: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_726, memory_format = torch.contiguous_format);  permute_726 = None
    view_793: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_147, [1, 1024, 512]);  clone_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_794: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_793, [1024, 512]);  view_793 = None
    permute_727: "f32[512, 1024]" = torch.ops.aten.permute.default(view_794, [1, 0])
    mm_285: "f32[512, 512]" = torch.ops.aten.mm.default(permute_727, view_1);  permute_727 = None
    permute_728: "f32[512, 512]" = torch.ops.aten.permute.default(mm_285, [1, 0]);  mm_285 = None
    mm_286: "f32[1024, 512]" = torch.ops.aten.mm.default(view_794, permute_729);  view_794 = permute_729 = None
    view_795: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_286, [1, 1024, 512]);  mm_286 = None
    permute_730: "f32[512, 512]" = torch.ops.aten.permute.default(permute_728, [1, 0]);  permute_728 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_731: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_725, [0, 2, 1, 3]);  permute_725 = None
    view_796: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(permute_731, [1, 1024, 512]);  permute_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_797: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_796, [1024, 512]);  view_796 = None
    permute_732: "f32[512, 1024]" = torch.ops.aten.permute.default(view_797, [1, 0])
    mm_287: "f32[512, 512]" = torch.ops.aten.mm.default(permute_732, view_1);  permute_732 = None
    permute_733: "f32[512, 512]" = torch.ops.aten.permute.default(mm_287, [1, 0]);  mm_287 = None
    mm_288: "f32[1024, 512]" = torch.ops.aten.mm.default(view_797, permute_734);  view_797 = permute_734 = None
    view_798: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_288, [1, 1024, 512]);  mm_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_217: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(view_795, view_798);  view_795 = view_798 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_735: "f32[512, 512]" = torch.ops.aten.permute.default(permute_733, [1, 0]);  permute_733 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_736: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_792, [0, 2, 1, 3]);  view_792 = None
    clone_148: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_736, memory_format = torch.contiguous_format);  permute_736 = None
    view_799: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(clone_148, [1, 1024, 512]);  clone_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_800: "f32[1024, 512]" = torch.ops.aten.reshape.default(view_799, [1024, 512]);  view_799 = None
    permute_737: "f32[512, 1024]" = torch.ops.aten.permute.default(view_800, [1, 0])
    mm_289: "f32[512, 512]" = torch.ops.aten.mm.default(permute_737, view_1);  permute_737 = view_1 = None
    permute_738: "f32[512, 512]" = torch.ops.aten.permute.default(mm_289, [1, 0]);  mm_289 = None
    mm_290: "f32[1024, 512]" = torch.ops.aten.mm.default(view_800, permute_739);  view_800 = permute_739 = None
    view_801: "f32[1, 1024, 512]" = torch.ops.aten.reshape.default(mm_290, [1, 1024, 512]);  mm_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_218: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_217, view_801);  add_217 = view_801 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_740: "f32[512, 512]" = torch.ops.aten.permute.default(permute_738, [1, 0]);  permute_738 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_485: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_218, primals_1);  primals_1 = None
    mul_486: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_218, mul_1);  add_218 = mul_1 = None
    sum_103: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_486, [0, 1], True);  mul_486 = None
    view_802: "f32[512]" = torch.ops.aten.reshape.default(sum_103, [512]);  sum_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_487: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_485, getitem)
    mul_488: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_485, rsqrt);  mul_485 = None
    sum_104: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_487, [2], True);  mul_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_219: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_215, mul_488);  add_215 = mul_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_95: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
    mul_489: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_104, -0.5);  sum_104 = None
    mul_490: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_489, pow_95);  mul_489 = pow_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_103: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_490, [1, 1024, 512]);  mul_490 = None
    div_55: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_103, 512);  expand_103 = None
    pow_96: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(getitem, 1.0);  getitem = None
    mul_491: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_96, 2.0);  pow_96 = None
    mul_492: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_55, mul_491);  div_55 = mul_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_220: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_219, mul_492);  add_219 = mul_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1076, code: hidden_states = self.dropout(inputs_embeds)
    convert_element_type_71: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.float32);  getitem_1 = None
    mul_493: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_71, 1.1111111111111112);  convert_element_type_71 = None
    mul_494: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_220, mul_493);  add_220 = mul_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1021, code: inputs_embeds = self.embed_tokens(input_ids)
    eq_3: "b8[1, 1024]" = torch.ops.aten.eq.Scalar(view, -1)
    unsqueeze_22: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_3, -1);  eq_3 = None
    where_21: "f32[1, 1024, 512]" = torch.ops.aten.where.self(unsqueeze_22, full_default_7, mul_494);  unsqueeze_22 = full_default_7 = mul_494 = None
    _unsafe_index_put_3: "f32[32128, 512]" = torch.ops.prims._unsafe_index_put_.default(full_default_21, [view], where_21, True);  full_default_21 = view = where_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1021, code: inputs_embeds = self.embed_tokens(input_ids)
    add_221: "f32[32128, 512]" = torch.ops.aten.add.Tensor(_unsafe_index_put_1, _unsafe_index_put_3);  _unsafe_index_put_1 = _unsafe_index_put_3 = None
    return [view_802, view_784, view_779, view_761, view_756, view_738, view_733, view_715, view_710, view_692, view_687, view_669, view_664, view_663, view_645, view_627, view_622, view_604, view_586, view_581, view_563, view_545, view_540, view_522, view_504, view_499, view_481, view_463, view_458, view_440, view_422, view_417, add_221, permute_740, permute_735, permute_730, _unsafe_index_put_2, permute_718, permute_714, permute_710, permute_706, permute_701, permute_696, permute_685, permute_681, permute_677, permute_673, permute_668, permute_663, permute_652, permute_648, permute_644, permute_640, permute_635, permute_630, permute_619, permute_615, permute_611, permute_607, permute_602, permute_597, permute_586, permute_582, permute_578, permute_574, permute_569, permute_564, permute_553, permute_549, permute_545, permute_541, permute_536, permute_531, _unsafe_index_put, permute_519, permute_515, permute_510, permute_505, permute_494, permute_490, permute_486, permute_482, permute_477, permute_472, permute_461, permute_457, permute_452, permute_447, permute_436, permute_432, permute_428, permute_424, permute_419, permute_414, permute_403, permute_399, permute_394, permute_389, permute_378, permute_374, permute_370, permute_366, permute_361, permute_356, permute_345, permute_341, permute_336, permute_331, permute_320, permute_316, permute_312, permute_308, permute_303, permute_298, permute_287, permute_283, permute_278, permute_273, permute_262, permute_258, permute_254, permute_250, permute_245, permute_240, permute_229, permute_225, permute_220, permute_215, permute_204, permute_200, permute_196, permute_192, None, None, None]
    