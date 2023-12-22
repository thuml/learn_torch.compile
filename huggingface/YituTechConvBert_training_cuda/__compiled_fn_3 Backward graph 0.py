from __future__ import annotations



def forward(self, primals_1: "f32[384, 1]", primals_2: "f32[384, 1]", primals_3: "f32[384, 1]", primals_4: "f32[384, 1]", primals_5: "f32[384, 1]", primals_6: "f32[384, 1]", primals_7: "f32[384, 1]", primals_8: "f32[384, 1]", primals_9: "f32[384, 1]", primals_10: "f32[384, 1]", primals_11: "f32[384, 1]", primals_12: "f32[384, 1]", primals_16: "f32[768]", primals_24: "f32[768, 1, 9]", primals_25: "f32[384, 768, 1]", primals_32: "f32[768]", primals_38: "f32[768]", primals_46: "f32[768, 1, 9]", primals_47: "f32[384, 768, 1]", primals_54: "f32[768]", primals_60: "f32[768]", primals_68: "f32[768, 1, 9]", primals_69: "f32[384, 768, 1]", primals_76: "f32[768]", primals_82: "f32[768]", primals_90: "f32[768, 1, 9]", primals_91: "f32[384, 768, 1]", primals_98: "f32[768]", primals_104: "f32[768]", primals_112: "f32[768, 1, 9]", primals_113: "f32[384, 768, 1]", primals_120: "f32[768]", primals_126: "f32[768]", primals_134: "f32[768, 1, 9]", primals_135: "f32[384, 768, 1]", primals_142: "f32[768]", primals_148: "f32[768]", primals_156: "f32[768, 1, 9]", primals_157: "f32[384, 768, 1]", primals_164: "f32[768]", primals_170: "f32[768]", primals_178: "f32[768, 1, 9]", primals_179: "f32[384, 768, 1]", primals_186: "f32[768]", primals_192: "f32[768]", primals_200: "f32[768, 1, 9]", primals_201: "f32[384, 768, 1]", primals_208: "f32[768]", primals_214: "f32[768]", primals_222: "f32[768, 1, 9]", primals_223: "f32[384, 768, 1]", primals_230: "f32[768]", primals_236: "f32[768]", primals_244: "f32[768, 1, 9]", primals_245: "f32[384, 768, 1]", primals_252: "f32[768]", primals_258: "f32[768]", primals_266: "f32[768, 1, 9]", primals_267: "f32[384, 768, 1]", primals_274: "f32[768]", primals_280: "f32[768]", primals_284: "f32[768]", primals_290: "i64[1, 512]", primals_291: "i64[1, 512]", expand: "i64[1, 512]", slice_4: "i64[1, 512]", mul_1: "f32[1, 512, 768]", getitem_3: "b8[1, 512, 768]", view: "f32[512, 768]", addmm: "f32[512, 384]", permute_3: "f32[1, 768, 512]", convolution: "f32[1, 768, 512]", convolution_1: "f32[1, 384, 512]", permute_9: "f32[384, 54]", view_9: "f32[512, 384]", full_default_1: "i64[1, 1]", unsqueeze_8: "i64[9, 512, 1, 1]", clone_default_33: "f32[1, 6, 512, 64]", clone_default_34: "f32[1, 6, 512, 64]", clone_default_35: "f32[1, 6, 512, 64]", getitem_276: "f32[1, 6, 512]", getitem_277: "i64[]", getitem_278: "i64[]", alias_default_23: "f32[1, 6, 512, 64]", view_30: "f32[512, 768]", getitem_7: "b8[1, 512, 768]", mul_4: "f32[1, 512, 768]", view_32: "f32[512, 768]", addmm_5: "f32[512, 3072]", view_34: "f32[512, 3072]", getitem_11: "b8[1, 512, 768]", mul_9: "f32[1, 512, 768]", view_36: "f32[512, 768]", addmm_7: "f32[512, 384]", permute_22: "f32[1, 768, 512]", convolution_2: "f32[1, 768, 512]", convolution_3: "f32[1, 384, 512]", permute_28: "f32[384, 54]", view_45: "f32[512, 384]", clone_default_30: "f32[1, 6, 512, 64]", clone_default_31: "f32[1, 6, 512, 64]", clone_default_32: "f32[1, 6, 512, 64]", getitem_269: "f32[1, 6, 512]", getitem_270: "i64[]", getitem_271: "i64[]", alias_default_21: "f32[1, 6, 512, 64]", view_66: "f32[512, 768]", getitem_17: "b8[1, 512, 768]", mul_12: "f32[1, 512, 768]", view_68: "f32[512, 768]", addmm_12: "f32[512, 3072]", view_70: "f32[512, 3072]", getitem_21: "b8[1, 512, 768]", mul_17: "f32[1, 512, 768]", view_72: "f32[512, 768]", addmm_14: "f32[512, 384]", permute_41: "f32[1, 768, 512]", convolution_4: "f32[1, 768, 512]", convolution_5: "f32[1, 384, 512]", permute_47: "f32[384, 54]", view_81: "f32[512, 384]", clone_default_27: "f32[1, 6, 512, 64]", clone_default_28: "f32[1, 6, 512, 64]", clone_default_29: "f32[1, 6, 512, 64]", getitem_262: "f32[1, 6, 512]", getitem_263: "i64[]", getitem_264: "i64[]", alias_default_19: "f32[1, 6, 512, 64]", view_102: "f32[512, 768]", getitem_27: "b8[1, 512, 768]", mul_20: "f32[1, 512, 768]", view_104: "f32[512, 768]", addmm_19: "f32[512, 3072]", view_106: "f32[512, 3072]", getitem_31: "b8[1, 512, 768]", mul_25: "f32[1, 512, 768]", view_108: "f32[512, 768]", addmm_21: "f32[512, 384]", permute_60: "f32[1, 768, 512]", convolution_6: "f32[1, 768, 512]", convolution_7: "f32[1, 384, 512]", permute_66: "f32[384, 54]", view_117: "f32[512, 384]", clone_default_24: "f32[1, 6, 512, 64]", clone_default_25: "f32[1, 6, 512, 64]", clone_default_26: "f32[1, 6, 512, 64]", getitem_255: "f32[1, 6, 512]", getitem_256: "i64[]", getitem_257: "i64[]", alias_default_17: "f32[1, 6, 512, 64]", view_138: "f32[512, 768]", getitem_37: "b8[1, 512, 768]", mul_28: "f32[1, 512, 768]", view_140: "f32[512, 768]", addmm_26: "f32[512, 3072]", view_142: "f32[512, 3072]", getitem_41: "b8[1, 512, 768]", mul_33: "f32[1, 512, 768]", view_144: "f32[512, 768]", addmm_28: "f32[512, 384]", permute_79: "f32[1, 768, 512]", convolution_8: "f32[1, 768, 512]", convolution_9: "f32[1, 384, 512]", permute_85: "f32[384, 54]", view_153: "f32[512, 384]", clone_default_21: "f32[1, 6, 512, 64]", clone_default_22: "f32[1, 6, 512, 64]", clone_default_23: "f32[1, 6, 512, 64]", getitem_248: "f32[1, 6, 512]", getitem_249: "i64[]", getitem_250: "i64[]", alias_default_15: "f32[1, 6, 512, 64]", view_174: "f32[512, 768]", getitem_47: "b8[1, 512, 768]", mul_36: "f32[1, 512, 768]", view_176: "f32[512, 768]", addmm_33: "f32[512, 3072]", view_178: "f32[512, 3072]", getitem_51: "b8[1, 512, 768]", mul_41: "f32[1, 512, 768]", view_180: "f32[512, 768]", addmm_35: "f32[512, 384]", permute_98: "f32[1, 768, 512]", convolution_10: "f32[1, 768, 512]", convolution_11: "f32[1, 384, 512]", permute_104: "f32[384, 54]", view_189: "f32[512, 384]", clone_default_18: "f32[1, 6, 512, 64]", clone_default_19: "f32[1, 6, 512, 64]", clone_default_20: "f32[1, 6, 512, 64]", getitem_241: "f32[1, 6, 512]", getitem_242: "i64[]", getitem_243: "i64[]", alias_default_13: "f32[1, 6, 512, 64]", view_210: "f32[512, 768]", getitem_57: "b8[1, 512, 768]", mul_44: "f32[1, 512, 768]", view_212: "f32[512, 768]", addmm_40: "f32[512, 3072]", view_214: "f32[512, 3072]", getitem_61: "b8[1, 512, 768]", mul_49: "f32[1, 512, 768]", view_216: "f32[512, 768]", addmm_42: "f32[512, 384]", permute_117: "f32[1, 768, 512]", convolution_12: "f32[1, 768, 512]", convolution_13: "f32[1, 384, 512]", permute_123: "f32[384, 54]", view_225: "f32[512, 384]", clone_default_15: "f32[1, 6, 512, 64]", clone_default_16: "f32[1, 6, 512, 64]", clone_default_17: "f32[1, 6, 512, 64]", getitem_234: "f32[1, 6, 512]", getitem_235: "i64[]", getitem_236: "i64[]", alias_default_11: "f32[1, 6, 512, 64]", view_246: "f32[512, 768]", getitem_67: "b8[1, 512, 768]", mul_52: "f32[1, 512, 768]", view_248: "f32[512, 768]", addmm_47: "f32[512, 3072]", view_250: "f32[512, 3072]", getitem_71: "b8[1, 512, 768]", mul_57: "f32[1, 512, 768]", view_252: "f32[512, 768]", addmm_49: "f32[512, 384]", permute_136: "f32[1, 768, 512]", convolution_14: "f32[1, 768, 512]", convolution_15: "f32[1, 384, 512]", permute_142: "f32[384, 54]", view_261: "f32[512, 384]", clone_default_12: "f32[1, 6, 512, 64]", clone_default_13: "f32[1, 6, 512, 64]", clone_default_14: "f32[1, 6, 512, 64]", getitem_227: "f32[1, 6, 512]", getitem_228: "i64[]", getitem_229: "i64[]", alias_default_9: "f32[1, 6, 512, 64]", view_282: "f32[512, 768]", getitem_77: "b8[1, 512, 768]", mul_60: "f32[1, 512, 768]", view_284: "f32[512, 768]", addmm_54: "f32[512, 3072]", view_286: "f32[512, 3072]", getitem_81: "b8[1, 512, 768]", mul_65: "f32[1, 512, 768]", view_288: "f32[512, 768]", addmm_56: "f32[512, 384]", permute_155: "f32[1, 768, 512]", convolution_16: "f32[1, 768, 512]", convolution_17: "f32[1, 384, 512]", permute_161: "f32[384, 54]", view_297: "f32[512, 384]", clone_default_9: "f32[1, 6, 512, 64]", clone_default_10: "f32[1, 6, 512, 64]", clone_default_11: "f32[1, 6, 512, 64]", getitem_220: "f32[1, 6, 512]", getitem_221: "i64[]", getitem_222: "i64[]", alias_default_7: "f32[1, 6, 512, 64]", view_318: "f32[512, 768]", getitem_87: "b8[1, 512, 768]", mul_68: "f32[1, 512, 768]", view_320: "f32[512, 768]", addmm_61: "f32[512, 3072]", view_322: "f32[512, 3072]", getitem_91: "b8[1, 512, 768]", mul_73: "f32[1, 512, 768]", view_324: "f32[512, 768]", addmm_63: "f32[512, 384]", permute_174: "f32[1, 768, 512]", convolution_18: "f32[1, 768, 512]", convolution_19: "f32[1, 384, 512]", permute_180: "f32[384, 54]", view_333: "f32[512, 384]", clone_default_6: "f32[1, 6, 512, 64]", clone_default_7: "f32[1, 6, 512, 64]", clone_default_8: "f32[1, 6, 512, 64]", getitem_213: "f32[1, 6, 512]", getitem_214: "i64[]", getitem_215: "i64[]", alias_default_5: "f32[1, 6, 512, 64]", view_354: "f32[512, 768]", getitem_97: "b8[1, 512, 768]", mul_76: "f32[1, 512, 768]", view_356: "f32[512, 768]", addmm_68: "f32[512, 3072]", view_358: "f32[512, 3072]", getitem_101: "b8[1, 512, 768]", mul_81: "f32[1, 512, 768]", view_360: "f32[512, 768]", addmm_70: "f32[512, 384]", permute_193: "f32[1, 768, 512]", convolution_20: "f32[1, 768, 512]", convolution_21: "f32[1, 384, 512]", permute_199: "f32[384, 54]", view_369: "f32[512, 384]", clone_default_3: "f32[1, 6, 512, 64]", clone_default_4: "f32[1, 6, 512, 64]", clone_default_5: "f32[1, 6, 512, 64]", getitem_206: "f32[1, 6, 512]", getitem_207: "i64[]", getitem_208: "i64[]", alias_default_3: "f32[1, 6, 512, 64]", view_390: "f32[512, 768]", getitem_107: "b8[1, 512, 768]", mul_84: "f32[1, 512, 768]", view_392: "f32[512, 768]", addmm_75: "f32[512, 3072]", view_394: "f32[512, 3072]", getitem_111: "b8[1, 512, 768]", mul_89: "f32[1, 512, 768]", view_396: "f32[512, 768]", addmm_77: "f32[512, 384]", permute_212: "f32[1, 768, 512]", convolution_22: "f32[1, 768, 512]", convolution_23: "f32[1, 384, 512]", permute_218: "f32[384, 54]", view_405: "f32[512, 384]", clone_default: "f32[1, 6, 512, 64]", clone_default_1: "f32[1, 6, 512, 64]", clone_default_2: "f32[1, 6, 512, 64]", getitem_199: "f32[1, 6, 512]", getitem_200: "i64[]", getitem_201: "i64[]", alias_default_1: "f32[1, 6, 512, 64]", view_426: "f32[512, 768]", getitem_117: "b8[1, 512, 768]", mul_92: "f32[1, 512, 768]", view_428: "f32[512, 768]", addmm_82: "f32[512, 3072]", view_430: "f32[512, 3072]", getitem_121: "b8[1, 512, 768]", mul_97: "f32[1, 512, 768]", view_432: "f32[512, 768]", addmm_84: "f32[512, 768]", mul_102: "f32[1, 512, 768]", view_434: "f32[512, 768]", sub_52: "f32[512, 30522]", convert_element_type: "f32[]", permute_230: "f32[30522, 768]", div_38: "f32[1, 512, 1]", permute_234: "f32[768, 768]", div_39: "f32[1, 512, 1]", permute_238: "f32[768, 3072]", permute_242: "f32[3072, 768]", div_40: "f32[1, 512, 1]", permute_246: "f32[768, 768]", permute_256: "f32[3072, 9, 64]", permute_257: "f32[3072, 1, 9]", permute_261: "f32[384, 768]", alias_27: "f32[3072, 9, 1]", permute_275: "f32[384, 768]", permute_279: "f32[384, 768]", permute_283: "f32[384, 768]", div_42: "f32[1, 512, 1]", permute_287: "f32[768, 3072]", permute_291: "f32[3072, 768]", div_43: "f32[1, 512, 1]", permute_295: "f32[768, 768]", permute_305: "f32[3072, 9, 64]", permute_306: "f32[3072, 1, 9]", permute_310: "f32[384, 768]", alias_29: "f32[3072, 9, 1]", permute_324: "f32[384, 768]", permute_328: "f32[384, 768]", permute_332: "f32[384, 768]", div_45: "f32[1, 512, 1]", permute_336: "f32[768, 3072]", permute_340: "f32[3072, 768]", div_46: "f32[1, 512, 1]", permute_344: "f32[768, 768]", permute_354: "f32[3072, 9, 64]", permute_355: "f32[3072, 1, 9]", permute_359: "f32[384, 768]", alias_31: "f32[3072, 9, 1]", permute_373: "f32[384, 768]", permute_377: "f32[384, 768]", permute_381: "f32[384, 768]", div_48: "f32[1, 512, 1]", permute_385: "f32[768, 3072]", permute_389: "f32[3072, 768]", div_49: "f32[1, 512, 1]", permute_393: "f32[768, 768]", permute_403: "f32[3072, 9, 64]", permute_404: "f32[3072, 1, 9]", permute_408: "f32[384, 768]", alias_33: "f32[3072, 9, 1]", permute_422: "f32[384, 768]", permute_426: "f32[384, 768]", permute_430: "f32[384, 768]", div_51: "f32[1, 512, 1]", permute_434: "f32[768, 3072]", permute_438: "f32[3072, 768]", div_52: "f32[1, 512, 1]", permute_442: "f32[768, 768]", permute_452: "f32[3072, 9, 64]", permute_453: "f32[3072, 1, 9]", permute_457: "f32[384, 768]", alias_35: "f32[3072, 9, 1]", permute_471: "f32[384, 768]", permute_475: "f32[384, 768]", permute_479: "f32[384, 768]", div_54: "f32[1, 512, 1]", permute_483: "f32[768, 3072]", permute_487: "f32[3072, 768]", div_55: "f32[1, 512, 1]", permute_491: "f32[768, 768]", permute_501: "f32[3072, 9, 64]", permute_502: "f32[3072, 1, 9]", permute_506: "f32[384, 768]", alias_37: "f32[3072, 9, 1]", permute_520: "f32[384, 768]", permute_524: "f32[384, 768]", permute_528: "f32[384, 768]", div_57: "f32[1, 512, 1]", permute_532: "f32[768, 3072]", permute_536: "f32[3072, 768]", div_58: "f32[1, 512, 1]", permute_540: "f32[768, 768]", permute_550: "f32[3072, 9, 64]", permute_551: "f32[3072, 1, 9]", permute_555: "f32[384, 768]", alias_39: "f32[3072, 9, 1]", permute_569: "f32[384, 768]", permute_573: "f32[384, 768]", permute_577: "f32[384, 768]", div_60: "f32[1, 512, 1]", permute_581: "f32[768, 3072]", permute_585: "f32[3072, 768]", div_61: "f32[1, 512, 1]", permute_589: "f32[768, 768]", permute_599: "f32[3072, 9, 64]", permute_600: "f32[3072, 1, 9]", permute_604: "f32[384, 768]", alias_41: "f32[3072, 9, 1]", permute_618: "f32[384, 768]", permute_622: "f32[384, 768]", permute_626: "f32[384, 768]", div_63: "f32[1, 512, 1]", permute_630: "f32[768, 3072]", permute_634: "f32[3072, 768]", div_64: "f32[1, 512, 1]", permute_638: "f32[768, 768]", permute_648: "f32[3072, 9, 64]", permute_649: "f32[3072, 1, 9]", permute_653: "f32[384, 768]", alias_43: "f32[3072, 9, 1]", permute_667: "f32[384, 768]", permute_671: "f32[384, 768]", permute_675: "f32[384, 768]", div_66: "f32[1, 512, 1]", permute_679: "f32[768, 3072]", permute_683: "f32[3072, 768]", div_67: "f32[1, 512, 1]", permute_687: "f32[768, 768]", permute_697: "f32[3072, 9, 64]", permute_698: "f32[3072, 1, 9]", permute_702: "f32[384, 768]", alias_45: "f32[3072, 9, 1]", permute_716: "f32[384, 768]", permute_720: "f32[384, 768]", permute_724: "f32[384, 768]", div_69: "f32[1, 512, 1]", permute_728: "f32[768, 3072]", permute_732: "f32[3072, 768]", div_70: "f32[1, 512, 1]", permute_736: "f32[768, 768]", permute_746: "f32[3072, 9, 64]", permute_747: "f32[3072, 1, 9]", permute_751: "f32[384, 768]", alias_47: "f32[3072, 9, 1]", permute_765: "f32[384, 768]", permute_769: "f32[384, 768]", permute_773: "f32[384, 768]", div_72: "f32[1, 512, 1]", permute_777: "f32[768, 3072]", permute_781: "f32[3072, 768]", div_73: "f32[1, 512, 1]", permute_785: "f32[768, 768]", permute_795: "f32[3072, 9, 64]", permute_796: "f32[3072, 1, 9]", permute_800: "f32[384, 768]", alias_49: "f32[3072, 9, 1]", permute_814: "f32[384, 768]", permute_818: "f32[384, 768]", permute_822: "f32[384, 768]", div_75: "f32[1, 512, 1]", tangents_1: "f32[]", tangents_2: "f32[1, 512, 30522]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_1: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm, [1, 512, 384]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_4: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_1, primals_1);  convolution_1 = primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_8: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_4, [0, 2, 1]);  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_33: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_5, [1, 512, 3072]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_7: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_33, 0.7071067811865476)
    erf: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
    add_12: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_37: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_7, [1, 512, 384]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_16: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_3, primals_2);  convolution_3 = primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_27: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_16, [0, 2, 1]);  add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_69: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_12, [1, 512, 3072]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_15: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_69, 0.7071067811865476)
    erf_1: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
    add_24: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_73: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_14, [1, 512, 384]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_28: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_5, primals_3);  convolution_5 = primals_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_46: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_28, [0, 2, 1]);  add_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_105: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_19, [1, 512, 3072]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_23: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_105, 0.7071067811865476)
    erf_2: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_23);  mul_23 = None
    add_36: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_109: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_21, [1, 512, 384]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_40: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_7, primals_4);  convolution_7 = primals_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_65: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_40, [0, 2, 1]);  add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_141: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_26, [1, 512, 3072]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_31: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_141, 0.7071067811865476)
    erf_3: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_31);  mul_31 = None
    add_48: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_145: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_28, [1, 512, 384]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_52: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_9, primals_5);  convolution_9 = primals_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_84: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_52, [0, 2, 1]);  add_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_177: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_33, [1, 512, 3072]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_39: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_177, 0.7071067811865476)
    erf_4: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
    add_60: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_181: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_35, [1, 512, 384]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_64: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_11, primals_6);  convolution_11 = primals_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_103: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_64, [0, 2, 1]);  add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_213: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_40, [1, 512, 3072]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_47: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_213, 0.7071067811865476)
    erf_5: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
    add_72: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_217: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_42, [1, 512, 384]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_76: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_13, primals_7);  convolution_13 = primals_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_122: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_76, [0, 2, 1]);  add_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_249: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_47, [1, 512, 3072]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_55: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_249, 0.7071067811865476)
    erf_6: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
    add_84: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_253: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_49, [1, 512, 384]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_88: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_15, primals_8);  convolution_15 = primals_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_141: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_88, [0, 2, 1]);  add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_285: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_54, [1, 512, 3072]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_63: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_285, 0.7071067811865476)
    erf_7: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
    add_96: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_289: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_56, [1, 512, 384]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_100: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_17, primals_9);  convolution_17 = primals_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_160: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_100, [0, 2, 1]);  add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_321: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_61, [1, 512, 3072]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_71: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_321, 0.7071067811865476)
    erf_8: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_71);  mul_71 = None
    add_108: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_325: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_63, [1, 512, 384]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_112: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_19, primals_10);  convolution_19 = primals_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_179: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_112, [0, 2, 1]);  add_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_357: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_68, [1, 512, 3072]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_79: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_357, 0.7071067811865476)
    erf_9: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_79);  mul_79 = None
    add_120: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_361: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_70, [1, 512, 384]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_124: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_21, primals_11);  convolution_21 = primals_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_198: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_124, [0, 2, 1]);  add_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_393: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_75, [1, 512, 3072]);  addmm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_87: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_393, 0.7071067811865476)
    erf_10: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
    add_132: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_397: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_77, [1, 512, 384]);  addmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_136: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_23, primals_12);  convolution_23 = primals_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_217: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_136, [0, 2, 1]);  add_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_429: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_82, [1, 512, 3072]);  addmm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_95: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_429, 0.7071067811865476)
    erf_11: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_95);  mul_95 = None
    add_144: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:873, code: hidden_states = self.dense(generator_hidden_states)
    view_433: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_84, [1, 512, 768]);  addmm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_100: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_433, 0.7071067811865476)
    erf_12: "f32[1, 512, 768]" = torch.ops.aten.erf.default(mul_100);  mul_100 = None
    add_148: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:947, code: loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    view_437: "i64[512]" = torch.ops.aten.view.default(primals_291, [-1]);  primals_291 = None
    alias_24: "f32[512, 30522]" = torch.ops.aten.alias.default(sub_52);  sub_52 = None
    full_default_13: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    full_default_14: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    div_37: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type);  tangents_1 = convert_element_type = None
    unsqueeze_87: "i64[512, 1]" = torch.ops.aten.unsqueeze.default(view_437, 1);  view_437 = None
    ne_3: "b8[512, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_87, -100)
    where_2: "i64[512, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_87, full_default_13);  unsqueeze_87 = full_default_13 = None
    full_default_16: "f32[512, 30522]" = torch.ops.aten.full.default([512, 30522], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    scatter: "f32[512, 30522]" = torch.ops.aten.scatter.value(full_default_16, 1, where_2, -1.0);  full_default_16 = where_2 = None
    where_3: "f32[512, 1]" = torch.ops.aten.where.self(ne_3, div_37, full_default_14);  ne_3 = div_37 = None
    mul_104: "f32[512, 30522]" = torch.ops.aten.mul.Tensor(scatter, where_3);  scatter = where_3 = None
    alias_25: "f32[512, 30522]" = torch.ops.aten.alias.default(alias_24);  alias_24 = None
    exp_25: "f32[512, 30522]" = torch.ops.aten.exp.default(alias_25);  alias_25 = None
    sum_28: "f32[512, 1]" = torch.ops.aten.sum.dim_IntList(mul_104, [1], True)
    mul_105: "f32[512, 30522]" = torch.ops.aten.mul.Tensor(exp_25, sum_28);  exp_25 = sum_28 = None
    sub_53: "f32[512, 30522]" = torch.ops.aten.sub.Tensor(mul_104, mul_105);  mul_104 = mul_105 = None
    view_438: "f32[1, 512, 30522]" = torch.ops.aten.view.default(sub_53, [1, 512, 30522]);  sub_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:947, code: loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    add_151: "f32[1, 512, 30522]" = torch.ops.aten.add.Tensor(tangents_2, view_438);  tangents_2 = view_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:941, code: prediction_scores = self.generator_lm_head(prediction_scores)
    view_439: "f32[512, 30522]" = torch.ops.aten.view.default(add_151, [512, 30522]);  add_151 = None
    mm_12: "f32[512, 768]" = torch.ops.aten.mm.default(view_439, permute_230);  permute_230 = None
    permute_231: "f32[30522, 512]" = torch.ops.aten.permute.default(view_439, [1, 0])
    mm_13: "f32[30522, 768]" = torch.ops.aten.mm.default(permute_231, view_434);  permute_231 = view_434 = None
    permute_232: "f32[768, 30522]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_29: "f32[1, 30522]" = torch.ops.aten.sum.dim_IntList(view_439, [0], True);  view_439 = None
    view_440: "f32[30522]" = torch.ops.aten.view.default(sum_29, [30522]);  sum_29 = None
    permute_233: "f32[30522, 768]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
    view_441: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_12, [1, 512, 768]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:875, code: hidden_states = self.LayerNorm(hidden_states)
    mul_107: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_441, primals_284);  primals_284 = None
    mul_108: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_107, 768)
    sum_30: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_107, [2], True)
    mul_109: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_107, mul_102);  mul_107 = None
    sum_31: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_109, [2], True);  mul_109 = None
    mul_110: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_102, sum_31);  sum_31 = None
    sub_55: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_108, sum_30);  mul_108 = sum_30 = None
    sub_56: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_55, mul_110);  sub_55 = mul_110 = None
    mul_111: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_38, sub_56);  div_38 = sub_56 = None
    mul_112: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_441, mul_102);  mul_102 = None
    sum_32: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_112, [0, 1]);  mul_112 = None
    sum_33: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_441, [0, 1]);  view_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_114: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_148, 0.5);  add_148 = None
    mul_115: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_433, view_433)
    mul_116: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_115, -0.5);  mul_115 = None
    exp_26: "f32[1, 512, 768]" = torch.ops.aten.exp.default(mul_116);  mul_116 = None
    mul_117: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_118: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_433, mul_117);  view_433 = mul_117 = None
    add_153: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_114, mul_118);  mul_114 = mul_118 = None
    mul_119: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_111, add_153);  mul_111 = add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:873, code: hidden_states = self.dense(generator_hidden_states)
    view_442: "f32[512, 768]" = torch.ops.aten.view.default(mul_119, [512, 768]);  mul_119 = None
    mm_14: "f32[512, 768]" = torch.ops.aten.mm.default(view_442, permute_234);  permute_234 = None
    permute_235: "f32[768, 512]" = torch.ops.aten.permute.default(view_442, [1, 0])
    mm_15: "f32[768, 768]" = torch.ops.aten.mm.default(permute_235, view_432);  permute_235 = view_432 = None
    permute_236: "f32[768, 768]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_34: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_442, [0], True);  view_442 = None
    view_443: "f32[768]" = torch.ops.aten.view.default(sum_34, [768]);  sum_34 = None
    permute_237: "f32[768, 768]" = torch.ops.aten.permute.default(permute_236, [1, 0]);  permute_236 = None
    view_444: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_14, [1, 512, 768]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_121: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_444, primals_280);  primals_280 = None
    mul_122: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_121, 768)
    sum_35: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_121, [2], True)
    mul_123: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_121, mul_97);  mul_121 = None
    sum_36: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_123, [2], True);  mul_123 = None
    mul_124: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_97, sum_36);  sum_36 = None
    sub_58: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_122, sum_35);  mul_122 = sum_35 = None
    sub_59: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_58, mul_124);  sub_58 = mul_124 = None
    mul_125: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_39, sub_59);  div_39 = sub_59 = None
    mul_126: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_444, mul_97);  mul_97 = None
    sum_37: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_126, [0, 1]);  mul_126 = None
    sum_38: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_444, [0, 1]);  view_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_1: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_121, torch.float32);  getitem_121 = None
    mul_127: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
    mul_128: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_125, mul_127);  mul_127 = None
    clone_36: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_128, memory_format = torch.contiguous_format);  mul_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_445: "f32[512, 768]" = torch.ops.aten.view.default(clone_36, [512, 768]);  clone_36 = None
    mm_16: "f32[512, 3072]" = torch.ops.aten.mm.default(view_445, permute_238);  permute_238 = None
    permute_239: "f32[768, 512]" = torch.ops.aten.permute.default(view_445, [1, 0])
    mm_17: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_239, view_430);  permute_239 = view_430 = None
    permute_240: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_39: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_445, [0], True);  view_445 = None
    view_446: "f32[768]" = torch.ops.aten.view.default(sum_39, [768]);  sum_39 = None
    permute_241: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
    view_447: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_16, [1, 512, 3072]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_130: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_144, 0.5);  add_144 = None
    mul_131: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_429, view_429)
    mul_132: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_131, -0.5);  mul_131 = None
    exp_27: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_132);  mul_132 = None
    mul_133: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_134: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_429, mul_133);  view_429 = mul_133 = None
    add_155: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_130, mul_134);  mul_130 = mul_134 = None
    mul_135: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_447, add_155);  view_447 = add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_448: "f32[512, 3072]" = torch.ops.aten.view.default(mul_135, [512, 3072]);  mul_135 = None
    mm_18: "f32[512, 768]" = torch.ops.aten.mm.default(view_448, permute_242);  permute_242 = None
    permute_243: "f32[3072, 512]" = torch.ops.aten.permute.default(view_448, [1, 0])
    mm_19: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_243, view_428);  permute_243 = view_428 = None
    permute_244: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_40: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_448, [0], True);  view_448 = None
    view_449: "f32[3072]" = torch.ops.aten.view.default(sum_40, [3072]);  sum_40 = None
    permute_245: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
    view_450: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_18, [1, 512, 768]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    add_156: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_125, view_450);  mul_125 = view_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_137: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_156, primals_274);  primals_274 = None
    mul_138: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_137, 768)
    sum_41: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_137, [2], True)
    mul_139: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_137, mul_92);  mul_137 = None
    sum_42: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_139, [2], True);  mul_139 = None
    mul_140: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_92, sum_42);  sum_42 = None
    sub_61: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_138, sum_41);  mul_138 = sum_41 = None
    sub_62: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_61, mul_140);  sub_61 = mul_140 = None
    mul_141: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_40, sub_62);  div_40 = sub_62 = None
    mul_142: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_156, mul_92);  mul_92 = None
    sum_43: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_142, [0, 1]);  mul_142 = None
    sum_44: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_156, [0, 1]);  add_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_2: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_117, torch.float32);  getitem_117 = None
    mul_143: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
    mul_144: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_141, mul_143);  mul_143 = None
    clone_37: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_144, memory_format = torch.contiguous_format);  mul_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_451: "f32[512, 768]" = torch.ops.aten.view.default(clone_37, [512, 768]);  clone_37 = None
    mm_20: "f32[512, 768]" = torch.ops.aten.mm.default(view_451, permute_246);  permute_246 = None
    permute_247: "f32[768, 512]" = torch.ops.aten.permute.default(view_451, [1, 0])
    mm_21: "f32[768, 768]" = torch.ops.aten.mm.default(permute_247, view_426);  permute_247 = view_426 = None
    permute_248: "f32[768, 768]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_45: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_451, [0], True);  view_451 = None
    view_452: "f32[768]" = torch.ops.aten.view.default(sum_45, [768]);  sum_45 = None
    permute_249: "f32[768, 768]" = torch.ops.aten.permute.default(permute_248, [1, 0]);  permute_248 = None
    view_453: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_20, [1, 512, 768]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_454: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_453, [1, 512, 12, 64]);  view_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    slice_29: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_454, 2, 0, 6)
    slice_30: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_454, 2, 6, 12);  view_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_455: "f32[512, 384]" = torch.ops.aten.view.default(slice_30, [512, 384]);  slice_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_250: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(slice_29, [0, 2, 1, 3]);  slice_29 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_250, clone_default, clone_default_1, clone_default_2, None, alias_default_1, getitem_199, getitem_200, getitem_201, 0.1, [True, True, True, False], scale = 0.125);  permute_250 = clone_default = clone_default_1 = clone_default_2 = alias_default_1 = getitem_199 = getitem_200 = getitem_201 = None
    getitem_202: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default[0]
    getitem_203: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default[1]
    getitem_204: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default[2];  _scaled_dot_product_efficient_attention_backward_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    clone_39: "f32[512, 384]" = torch.ops.aten.clone.default(view_455, memory_format = torch.contiguous_format);  view_455 = None
    view_462: "f32[3072, 64, 1]" = torch.ops.aten.view.default(clone_39, [3072, 64, 1]);  clone_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    bmm_40: "f32[3072, 9, 1]" = torch.ops.aten.bmm.default(permute_256, view_462);  permute_256 = None
    bmm_41: "f32[3072, 64, 9]" = torch.ops.aten.bmm.default(view_462, permute_257);  view_462 = permute_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    view_466: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(bmm_41, [1, 512, 384, 9]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    view_467: "f32[1, 512, 3456]" = torch.ops.aten.view.default(view_466, [1, 512, 3456]);  view_466 = None
    permute_258: "f32[1, 3456, 512]" = torch.ops.aten.permute.default(view_467, [0, 2, 1]);  view_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    view_468: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.view.default(permute_258, [1, 384, 9, 1, 512, 1]);  permute_258 = None
    permute_259: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.permute.default(view_468, [0, 1, 2, 4, 3, 5]);  view_468 = None
    full_default_19: "f32[1, 384, 520, 1]" = torch.ops.aten.full.default([1, 384, 520, 1], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put: "f32[1, 384, 520, 1]" = torch.ops.aten._unsafe_index_put.default(full_default_19, [None, None, unsqueeze_8, full_default_1], permute_259, True);  permute_259 = None
    constant_pad_nd_12: "f32[1, 384, 512, 1]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put, [0, 0, -4, -4], 0.0);  _unsafe_index_put = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    squeeze_1: "f32[1, 384, 512]" = torch.ops.aten.squeeze.dim(constant_pad_nd_12, -1);  constant_pad_nd_12 = None
    permute_260: "f32[1, 512, 384]" = torch.ops.aten.permute.default(squeeze_1, [0, 2, 1]);  squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_470: "f32[512, 384]" = torch.ops.aten.view.default(permute_260, [512, 384]);  permute_260 = None
    mm_22: "f32[512, 768]" = torch.ops.aten.mm.default(view_470, permute_261);  permute_261 = None
    permute_262: "f32[384, 512]" = torch.ops.aten.permute.default(view_470, [1, 0])
    mm_23: "f32[384, 768]" = torch.ops.aten.mm.default(permute_262, view_396);  permute_262 = None
    permute_263: "f32[768, 384]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_47: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_470, [0], True);  view_470 = None
    view_471: "f32[384]" = torch.ops.aten.view.default(sum_47, [384]);  sum_47 = None
    permute_264: "f32[384, 768]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
    view_472: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_22, [1, 512, 768]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    add_159: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_141, view_472);  mul_141 = view_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    mul_149: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(bmm_40, alias_27);  bmm_40 = None
    sum_48: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_149, [1], True)
    mul_150: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(alias_27, sum_48);  alias_27 = sum_48 = None
    sub_64: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(mul_149, mul_150);  mul_149 = mul_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_473: "f32[1, 512, 54]" = torch.ops.aten.view.default(sub_64, [1, 512, 54]);  sub_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    sum_49: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_473, [0, 1], True)
    view_474: "f32[54]" = torch.ops.aten.view.default(sum_49, [54]);  sum_49 = None
    view_475: "f32[512, 54]" = torch.ops.aten.view.default(view_473, [512, 54]);  view_473 = None
    permute_265: "f32[54, 512]" = torch.ops.aten.permute.default(view_475, [1, 0]);  view_475 = None
    mm_24: "f32[54, 384]" = torch.ops.aten.mm.default(permute_265, view_405);  view_405 = None
    permute_266: "f32[384, 54]" = torch.ops.aten.permute.default(mm_24, [1, 0]);  mm_24 = None
    mm_25: "f32[384, 512]" = torch.ops.aten.mm.default(permute_218, permute_265);  permute_218 = permute_265 = None
    permute_268: "f32[512, 384]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    view_476: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_268, [1, 512, 384]);  permute_268 = None
    permute_269: "f32[54, 384]" = torch.ops.aten.permute.default(permute_266, [1, 0]);  permute_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    mul_151: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_476, permute_217);  permute_217 = None
    mul_152: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_476, view_397);  view_476 = view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_270: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_204, [0, 2, 1, 3]);  getitem_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_40: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_270, memory_format = torch.contiguous_format);  permute_270 = None
    view_477: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_40, [1, 512, 384]);  clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_271: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_203, [0, 2, 1, 3]);  getitem_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_478: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_271, [1, 512, 384]);  permute_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_272: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_202, [0, 2, 1, 3]);  getitem_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_41: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_272, memory_format = torch.contiguous_format);  permute_272 = None
    view_479: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_41, [1, 512, 384]);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    add_160: "f32[1, 512, 384]" = torch.ops.aten.add.Tensor(mul_151, view_479);  mul_151 = view_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    permute_273: "f32[1, 384, 512]" = torch.ops.aten.permute.default(mul_152, [0, 2, 1]);  mul_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    sum_50: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_273, [0, 2], True)
    view_480: "f32[384, 1]" = torch.ops.aten.view.default(sum_50, [384, 1]);  sum_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(permute_273, convolution_22, primals_267, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_273 = convolution_22 = primals_267 = None
    getitem_126: "f32[1, 768, 512]" = convolution_backward[0]
    getitem_127: "f32[384, 768, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(getitem_126, permute_212, primals_266, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_126 = permute_212 = primals_266 = None
    getitem_129: "f32[1, 768, 512]" = convolution_backward_1[0]
    getitem_130: "f32[768, 1, 9]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_274: "f32[1, 512, 768]" = torch.ops.aten.permute.default(getitem_129, [0, 2, 1]);  getitem_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    add_161: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_159, permute_274);  add_159 = permute_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_481: "f32[512, 384]" = torch.ops.aten.view.default(view_477, [512, 384]);  view_477 = None
    mm_26: "f32[512, 768]" = torch.ops.aten.mm.default(view_481, permute_275);  permute_275 = None
    permute_276: "f32[384, 512]" = torch.ops.aten.permute.default(view_481, [1, 0])
    mm_27: "f32[384, 768]" = torch.ops.aten.mm.default(permute_276, view_396);  permute_276 = None
    permute_277: "f32[768, 384]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_51: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_481, [0], True);  view_481 = None
    view_482: "f32[384]" = torch.ops.aten.view.default(sum_51, [384]);  sum_51 = None
    permute_278: "f32[384, 768]" = torch.ops.aten.permute.default(permute_277, [1, 0]);  permute_277 = None
    view_483: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_26, [1, 512, 768]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    add_162: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_161, view_483);  add_161 = view_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_484: "f32[512, 384]" = torch.ops.aten.view.default(view_478, [512, 384]);  view_478 = None
    mm_28: "f32[512, 768]" = torch.ops.aten.mm.default(view_484, permute_279);  permute_279 = None
    permute_280: "f32[384, 512]" = torch.ops.aten.permute.default(view_484, [1, 0])
    mm_29: "f32[384, 768]" = torch.ops.aten.mm.default(permute_280, view_396);  permute_280 = None
    permute_281: "f32[768, 384]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_52: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_484, [0], True);  view_484 = None
    view_485: "f32[384]" = torch.ops.aten.view.default(sum_52, [384]);  sum_52 = None
    permute_282: "f32[384, 768]" = torch.ops.aten.permute.default(permute_281, [1, 0]);  permute_281 = None
    view_486: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_28, [1, 512, 768]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    add_163: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_162, view_486);  add_162 = view_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_487: "f32[512, 384]" = torch.ops.aten.view.default(add_160, [512, 384]);  add_160 = None
    mm_30: "f32[512, 768]" = torch.ops.aten.mm.default(view_487, permute_283);  permute_283 = None
    permute_284: "f32[384, 512]" = torch.ops.aten.permute.default(view_487, [1, 0])
    mm_31: "f32[384, 768]" = torch.ops.aten.mm.default(permute_284, view_396);  permute_284 = view_396 = None
    permute_285: "f32[768, 384]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_53: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_487, [0], True);  view_487 = None
    view_488: "f32[384]" = torch.ops.aten.view.default(sum_53, [384]);  sum_53 = None
    permute_286: "f32[384, 768]" = torch.ops.aten.permute.default(permute_285, [1, 0]);  permute_285 = None
    view_489: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_30, [1, 512, 768]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    add_164: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_163, view_489);  add_163 = view_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_154: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_164, primals_258);  primals_258 = None
    mul_155: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_154, 768)
    sum_54: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_154, [2], True)
    mul_156: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_154, mul_89);  mul_154 = None
    sum_55: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_156, [2], True);  mul_156 = None
    mul_157: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_89, sum_55);  sum_55 = None
    sub_66: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_155, sum_54);  mul_155 = sum_54 = None
    sub_67: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_66, mul_157);  sub_66 = mul_157 = None
    mul_158: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_42, sub_67);  div_42 = sub_67 = None
    mul_159: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_164, mul_89);  mul_89 = None
    sum_56: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_159, [0, 1]);  mul_159 = None
    sum_57: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_164, [0, 1]);  add_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_4: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_111, torch.float32);  getitem_111 = None
    mul_160: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
    mul_161: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_158, mul_160);  mul_160 = None
    clone_42: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_161, memory_format = torch.contiguous_format);  mul_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_490: "f32[512, 768]" = torch.ops.aten.view.default(clone_42, [512, 768]);  clone_42 = None
    mm_32: "f32[512, 3072]" = torch.ops.aten.mm.default(view_490, permute_287);  permute_287 = None
    permute_288: "f32[768, 512]" = torch.ops.aten.permute.default(view_490, [1, 0])
    mm_33: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_288, view_394);  permute_288 = view_394 = None
    permute_289: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_58: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_490, [0], True);  view_490 = None
    view_491: "f32[768]" = torch.ops.aten.view.default(sum_58, [768]);  sum_58 = None
    permute_290: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_289, [1, 0]);  permute_289 = None
    view_492: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_32, [1, 512, 3072]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_163: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_132, 0.5);  add_132 = None
    mul_164: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_393, view_393)
    mul_165: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_164, -0.5);  mul_164 = None
    exp_28: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_165);  mul_165 = None
    mul_166: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
    mul_167: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_393, mul_166);  view_393 = mul_166 = None
    add_166: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_163, mul_167);  mul_163 = mul_167 = None
    mul_168: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_492, add_166);  view_492 = add_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_493: "f32[512, 3072]" = torch.ops.aten.view.default(mul_168, [512, 3072]);  mul_168 = None
    mm_34: "f32[512, 768]" = torch.ops.aten.mm.default(view_493, permute_291);  permute_291 = None
    permute_292: "f32[3072, 512]" = torch.ops.aten.permute.default(view_493, [1, 0])
    mm_35: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_292, view_392);  permute_292 = view_392 = None
    permute_293: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_59: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_493, [0], True);  view_493 = None
    view_494: "f32[3072]" = torch.ops.aten.view.default(sum_59, [3072]);  sum_59 = None
    permute_294: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_293, [1, 0]);  permute_293 = None
    view_495: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_34, [1, 512, 768]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    add_167: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_158, view_495);  mul_158 = view_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_170: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_167, primals_252);  primals_252 = None
    mul_171: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_170, 768)
    sum_60: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_170, [2], True)
    mul_172: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_170, mul_84);  mul_170 = None
    sum_61: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_172, [2], True);  mul_172 = None
    mul_173: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_84, sum_61);  sum_61 = None
    sub_69: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_171, sum_60);  mul_171 = sum_60 = None
    sub_70: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_69, mul_173);  sub_69 = mul_173 = None
    mul_174: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_43, sub_70);  div_43 = sub_70 = None
    mul_175: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_167, mul_84);  mul_84 = None
    sum_62: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_175, [0, 1]);  mul_175 = None
    sum_63: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_167, [0, 1]);  add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_5: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_107, torch.float32);  getitem_107 = None
    mul_176: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
    mul_177: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_174, mul_176);  mul_176 = None
    clone_43: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_177, memory_format = torch.contiguous_format);  mul_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_496: "f32[512, 768]" = torch.ops.aten.view.default(clone_43, [512, 768]);  clone_43 = None
    mm_36: "f32[512, 768]" = torch.ops.aten.mm.default(view_496, permute_295);  permute_295 = None
    permute_296: "f32[768, 512]" = torch.ops.aten.permute.default(view_496, [1, 0])
    mm_37: "f32[768, 768]" = torch.ops.aten.mm.default(permute_296, view_390);  permute_296 = view_390 = None
    permute_297: "f32[768, 768]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_64: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_496, [0], True);  view_496 = None
    view_497: "f32[768]" = torch.ops.aten.view.default(sum_64, [768]);  sum_64 = None
    permute_298: "f32[768, 768]" = torch.ops.aten.permute.default(permute_297, [1, 0]);  permute_297 = None
    view_498: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_36, [1, 512, 768]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_499: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_498, [1, 512, 12, 64]);  view_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    slice_31: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_499, 2, 0, 6)
    slice_32: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_499, 2, 6, 12);  view_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_500: "f32[512, 384]" = torch.ops.aten.view.default(slice_32, [512, 384]);  slice_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_299: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(slice_31, [0, 2, 1, 3]);  slice_31 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_1 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_299, clone_default_3, clone_default_4, clone_default_5, None, alias_default_3, getitem_206, getitem_207, getitem_208, 0.1, [True, True, True, False], scale = 0.125);  permute_299 = clone_default_3 = clone_default_4 = clone_default_5 = alias_default_3 = getitem_206 = getitem_207 = getitem_208 = None
    getitem_209: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_1[0]
    getitem_210: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_1[1]
    getitem_211: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_1[2];  _scaled_dot_product_efficient_attention_backward_default_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    clone_45: "f32[512, 384]" = torch.ops.aten.clone.default(view_500, memory_format = torch.contiguous_format);  view_500 = None
    view_507: "f32[3072, 64, 1]" = torch.ops.aten.view.default(clone_45, [3072, 64, 1]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    bmm_46: "f32[3072, 9, 1]" = torch.ops.aten.bmm.default(permute_305, view_507);  permute_305 = None
    bmm_47: "f32[3072, 64, 9]" = torch.ops.aten.bmm.default(view_507, permute_306);  view_507 = permute_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    view_511: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(bmm_47, [1, 512, 384, 9]);  bmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    view_512: "f32[1, 512, 3456]" = torch.ops.aten.view.default(view_511, [1, 512, 3456]);  view_511 = None
    permute_307: "f32[1, 3456, 512]" = torch.ops.aten.permute.default(view_512, [0, 2, 1]);  view_512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    view_513: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.view.default(permute_307, [1, 384, 9, 1, 512, 1]);  permute_307 = None
    permute_308: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.permute.default(view_513, [0, 1, 2, 4, 3, 5]);  view_513 = None
    _unsafe_index_put_1: "f32[1, 384, 520, 1]" = torch.ops.aten._unsafe_index_put.default(full_default_19, [None, None, unsqueeze_8, full_default_1], permute_308, True);  permute_308 = None
    constant_pad_nd_13: "f32[1, 384, 512, 1]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_1, [0, 0, -4, -4], 0.0);  _unsafe_index_put_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    squeeze_2: "f32[1, 384, 512]" = torch.ops.aten.squeeze.dim(constant_pad_nd_13, -1);  constant_pad_nd_13 = None
    permute_309: "f32[1, 512, 384]" = torch.ops.aten.permute.default(squeeze_2, [0, 2, 1]);  squeeze_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_515: "f32[512, 384]" = torch.ops.aten.view.default(permute_309, [512, 384]);  permute_309 = None
    mm_38: "f32[512, 768]" = torch.ops.aten.mm.default(view_515, permute_310);  permute_310 = None
    permute_311: "f32[384, 512]" = torch.ops.aten.permute.default(view_515, [1, 0])
    mm_39: "f32[384, 768]" = torch.ops.aten.mm.default(permute_311, view_360);  permute_311 = None
    permute_312: "f32[768, 384]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_66: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_515, [0], True);  view_515 = None
    view_516: "f32[384]" = torch.ops.aten.view.default(sum_66, [384]);  sum_66 = None
    permute_313: "f32[384, 768]" = torch.ops.aten.permute.default(permute_312, [1, 0]);  permute_312 = None
    view_517: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_38, [1, 512, 768]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    add_170: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_174, view_517);  mul_174 = view_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    mul_182: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(bmm_46, alias_29);  bmm_46 = None
    sum_67: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_182, [1], True)
    mul_183: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(alias_29, sum_67);  alias_29 = sum_67 = None
    sub_72: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(mul_182, mul_183);  mul_182 = mul_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_518: "f32[1, 512, 54]" = torch.ops.aten.view.default(sub_72, [1, 512, 54]);  sub_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    sum_68: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_518, [0, 1], True)
    view_519: "f32[54]" = torch.ops.aten.view.default(sum_68, [54]);  sum_68 = None
    view_520: "f32[512, 54]" = torch.ops.aten.view.default(view_518, [512, 54]);  view_518 = None
    permute_314: "f32[54, 512]" = torch.ops.aten.permute.default(view_520, [1, 0]);  view_520 = None
    mm_40: "f32[54, 384]" = torch.ops.aten.mm.default(permute_314, view_369);  view_369 = None
    permute_315: "f32[384, 54]" = torch.ops.aten.permute.default(mm_40, [1, 0]);  mm_40 = None
    mm_41: "f32[384, 512]" = torch.ops.aten.mm.default(permute_199, permute_314);  permute_199 = permute_314 = None
    permute_317: "f32[512, 384]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    view_521: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_317, [1, 512, 384]);  permute_317 = None
    permute_318: "f32[54, 384]" = torch.ops.aten.permute.default(permute_315, [1, 0]);  permute_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    mul_184: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_521, permute_198);  permute_198 = None
    mul_185: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_521, view_361);  view_521 = view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_319: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_211, [0, 2, 1, 3]);  getitem_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_46: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_319, memory_format = torch.contiguous_format);  permute_319 = None
    view_522: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_46, [1, 512, 384]);  clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_320: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_210, [0, 2, 1, 3]);  getitem_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_523: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_320, [1, 512, 384]);  permute_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_321: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_209, [0, 2, 1, 3]);  getitem_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_47: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_321, memory_format = torch.contiguous_format);  permute_321 = None
    view_524: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_47, [1, 512, 384]);  clone_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    add_171: "f32[1, 512, 384]" = torch.ops.aten.add.Tensor(mul_184, view_524);  mul_184 = view_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    permute_322: "f32[1, 384, 512]" = torch.ops.aten.permute.default(mul_185, [0, 2, 1]);  mul_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    sum_69: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_322, [0, 2], True)
    view_525: "f32[384, 1]" = torch.ops.aten.view.default(sum_69, [384, 1]);  sum_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(permute_322, convolution_20, primals_245, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_322 = convolution_20 = primals_245 = None
    getitem_132: "f32[1, 768, 512]" = convolution_backward_2[0]
    getitem_133: "f32[384, 768, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(getitem_132, permute_193, primals_244, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_132 = permute_193 = primals_244 = None
    getitem_135: "f32[1, 768, 512]" = convolution_backward_3[0]
    getitem_136: "f32[768, 1, 9]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_323: "f32[1, 512, 768]" = torch.ops.aten.permute.default(getitem_135, [0, 2, 1]);  getitem_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    add_172: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_170, permute_323);  add_170 = permute_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_526: "f32[512, 384]" = torch.ops.aten.view.default(view_522, [512, 384]);  view_522 = None
    mm_42: "f32[512, 768]" = torch.ops.aten.mm.default(view_526, permute_324);  permute_324 = None
    permute_325: "f32[384, 512]" = torch.ops.aten.permute.default(view_526, [1, 0])
    mm_43: "f32[384, 768]" = torch.ops.aten.mm.default(permute_325, view_360);  permute_325 = None
    permute_326: "f32[768, 384]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_70: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_526, [0], True);  view_526 = None
    view_527: "f32[384]" = torch.ops.aten.view.default(sum_70, [384]);  sum_70 = None
    permute_327: "f32[384, 768]" = torch.ops.aten.permute.default(permute_326, [1, 0]);  permute_326 = None
    view_528: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_42, [1, 512, 768]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    add_173: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_172, view_528);  add_172 = view_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_529: "f32[512, 384]" = torch.ops.aten.view.default(view_523, [512, 384]);  view_523 = None
    mm_44: "f32[512, 768]" = torch.ops.aten.mm.default(view_529, permute_328);  permute_328 = None
    permute_329: "f32[384, 512]" = torch.ops.aten.permute.default(view_529, [1, 0])
    mm_45: "f32[384, 768]" = torch.ops.aten.mm.default(permute_329, view_360);  permute_329 = None
    permute_330: "f32[768, 384]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_71: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_529, [0], True);  view_529 = None
    view_530: "f32[384]" = torch.ops.aten.view.default(sum_71, [384]);  sum_71 = None
    permute_331: "f32[384, 768]" = torch.ops.aten.permute.default(permute_330, [1, 0]);  permute_330 = None
    view_531: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_44, [1, 512, 768]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    add_174: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_173, view_531);  add_173 = view_531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_532: "f32[512, 384]" = torch.ops.aten.view.default(add_171, [512, 384]);  add_171 = None
    mm_46: "f32[512, 768]" = torch.ops.aten.mm.default(view_532, permute_332);  permute_332 = None
    permute_333: "f32[384, 512]" = torch.ops.aten.permute.default(view_532, [1, 0])
    mm_47: "f32[384, 768]" = torch.ops.aten.mm.default(permute_333, view_360);  permute_333 = view_360 = None
    permute_334: "f32[768, 384]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_72: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_532, [0], True);  view_532 = None
    view_533: "f32[384]" = torch.ops.aten.view.default(sum_72, [384]);  sum_72 = None
    permute_335: "f32[384, 768]" = torch.ops.aten.permute.default(permute_334, [1, 0]);  permute_334 = None
    view_534: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_46, [1, 512, 768]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    add_175: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_174, view_534);  add_174 = view_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_187: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_175, primals_236);  primals_236 = None
    mul_188: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_187, 768)
    sum_73: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_187, [2], True)
    mul_189: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_187, mul_81);  mul_187 = None
    sum_74: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_189, [2], True);  mul_189 = None
    mul_190: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_81, sum_74);  sum_74 = None
    sub_74: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_188, sum_73);  mul_188 = sum_73 = None
    sub_75: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_74, mul_190);  sub_74 = mul_190 = None
    mul_191: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_45, sub_75);  div_45 = sub_75 = None
    mul_192: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_175, mul_81);  mul_81 = None
    sum_75: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_192, [0, 1]);  mul_192 = None
    sum_76: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_175, [0, 1]);  add_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_7: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_101, torch.float32);  getitem_101 = None
    mul_193: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
    mul_194: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_191, mul_193);  mul_193 = None
    clone_48: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_194, memory_format = torch.contiguous_format);  mul_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_535: "f32[512, 768]" = torch.ops.aten.view.default(clone_48, [512, 768]);  clone_48 = None
    mm_48: "f32[512, 3072]" = torch.ops.aten.mm.default(view_535, permute_336);  permute_336 = None
    permute_337: "f32[768, 512]" = torch.ops.aten.permute.default(view_535, [1, 0])
    mm_49: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_337, view_358);  permute_337 = view_358 = None
    permute_338: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_77: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_535, [0], True);  view_535 = None
    view_536: "f32[768]" = torch.ops.aten.view.default(sum_77, [768]);  sum_77 = None
    permute_339: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_338, [1, 0]);  permute_338 = None
    view_537: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_48, [1, 512, 3072]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_196: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_120, 0.5);  add_120 = None
    mul_197: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_357, view_357)
    mul_198: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_197, -0.5);  mul_197 = None
    exp_29: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_198);  mul_198 = None
    mul_199: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
    mul_200: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_357, mul_199);  view_357 = mul_199 = None
    add_177: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_196, mul_200);  mul_196 = mul_200 = None
    mul_201: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_537, add_177);  view_537 = add_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_538: "f32[512, 3072]" = torch.ops.aten.view.default(mul_201, [512, 3072]);  mul_201 = None
    mm_50: "f32[512, 768]" = torch.ops.aten.mm.default(view_538, permute_340);  permute_340 = None
    permute_341: "f32[3072, 512]" = torch.ops.aten.permute.default(view_538, [1, 0])
    mm_51: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_341, view_356);  permute_341 = view_356 = None
    permute_342: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_78: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_538, [0], True);  view_538 = None
    view_539: "f32[3072]" = torch.ops.aten.view.default(sum_78, [3072]);  sum_78 = None
    permute_343: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_342, [1, 0]);  permute_342 = None
    view_540: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_50, [1, 512, 768]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    add_178: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_191, view_540);  mul_191 = view_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_203: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_178, primals_230);  primals_230 = None
    mul_204: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_203, 768)
    sum_79: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_203, [2], True)
    mul_205: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_203, mul_76);  mul_203 = None
    sum_80: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_205, [2], True);  mul_205 = None
    mul_206: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_76, sum_80);  sum_80 = None
    sub_77: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_204, sum_79);  mul_204 = sum_79 = None
    sub_78: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_77, mul_206);  sub_77 = mul_206 = None
    mul_207: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_46, sub_78);  div_46 = sub_78 = None
    mul_208: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_178, mul_76);  mul_76 = None
    sum_81: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_208, [0, 1]);  mul_208 = None
    sum_82: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_178, [0, 1]);  add_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_8: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_97, torch.float32);  getitem_97 = None
    mul_209: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
    mul_210: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_207, mul_209);  mul_209 = None
    clone_49: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_210, memory_format = torch.contiguous_format);  mul_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_541: "f32[512, 768]" = torch.ops.aten.view.default(clone_49, [512, 768]);  clone_49 = None
    mm_52: "f32[512, 768]" = torch.ops.aten.mm.default(view_541, permute_344);  permute_344 = None
    permute_345: "f32[768, 512]" = torch.ops.aten.permute.default(view_541, [1, 0])
    mm_53: "f32[768, 768]" = torch.ops.aten.mm.default(permute_345, view_354);  permute_345 = view_354 = None
    permute_346: "f32[768, 768]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_83: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_541, [0], True);  view_541 = None
    view_542: "f32[768]" = torch.ops.aten.view.default(sum_83, [768]);  sum_83 = None
    permute_347: "f32[768, 768]" = torch.ops.aten.permute.default(permute_346, [1, 0]);  permute_346 = None
    view_543: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_52, [1, 512, 768]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_544: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_543, [1, 512, 12, 64]);  view_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    slice_33: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_544, 2, 0, 6)
    slice_34: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_544, 2, 6, 12);  view_544 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_545: "f32[512, 384]" = torch.ops.aten.view.default(slice_34, [512, 384]);  slice_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_348: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(slice_33, [0, 2, 1, 3]);  slice_33 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_2 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_348, clone_default_6, clone_default_7, clone_default_8, None, alias_default_5, getitem_213, getitem_214, getitem_215, 0.1, [True, True, True, False], scale = 0.125);  permute_348 = clone_default_6 = clone_default_7 = clone_default_8 = alias_default_5 = getitem_213 = getitem_214 = getitem_215 = None
    getitem_216: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_2[0]
    getitem_217: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_2[1]
    getitem_218: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_2[2];  _scaled_dot_product_efficient_attention_backward_default_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    clone_51: "f32[512, 384]" = torch.ops.aten.clone.default(view_545, memory_format = torch.contiguous_format);  view_545 = None
    view_552: "f32[3072, 64, 1]" = torch.ops.aten.view.default(clone_51, [3072, 64, 1]);  clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    bmm_52: "f32[3072, 9, 1]" = torch.ops.aten.bmm.default(permute_354, view_552);  permute_354 = None
    bmm_53: "f32[3072, 64, 9]" = torch.ops.aten.bmm.default(view_552, permute_355);  view_552 = permute_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    view_556: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(bmm_53, [1, 512, 384, 9]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    view_557: "f32[1, 512, 3456]" = torch.ops.aten.view.default(view_556, [1, 512, 3456]);  view_556 = None
    permute_356: "f32[1, 3456, 512]" = torch.ops.aten.permute.default(view_557, [0, 2, 1]);  view_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    view_558: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.view.default(permute_356, [1, 384, 9, 1, 512, 1]);  permute_356 = None
    permute_357: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.permute.default(view_558, [0, 1, 2, 4, 3, 5]);  view_558 = None
    _unsafe_index_put_2: "f32[1, 384, 520, 1]" = torch.ops.aten._unsafe_index_put.default(full_default_19, [None, None, unsqueeze_8, full_default_1], permute_357, True);  permute_357 = None
    constant_pad_nd_14: "f32[1, 384, 512, 1]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_2, [0, 0, -4, -4], 0.0);  _unsafe_index_put_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    squeeze_3: "f32[1, 384, 512]" = torch.ops.aten.squeeze.dim(constant_pad_nd_14, -1);  constant_pad_nd_14 = None
    permute_358: "f32[1, 512, 384]" = torch.ops.aten.permute.default(squeeze_3, [0, 2, 1]);  squeeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_560: "f32[512, 384]" = torch.ops.aten.view.default(permute_358, [512, 384]);  permute_358 = None
    mm_54: "f32[512, 768]" = torch.ops.aten.mm.default(view_560, permute_359);  permute_359 = None
    permute_360: "f32[384, 512]" = torch.ops.aten.permute.default(view_560, [1, 0])
    mm_55: "f32[384, 768]" = torch.ops.aten.mm.default(permute_360, view_324);  permute_360 = None
    permute_361: "f32[768, 384]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_85: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_560, [0], True);  view_560 = None
    view_561: "f32[384]" = torch.ops.aten.view.default(sum_85, [384]);  sum_85 = None
    permute_362: "f32[384, 768]" = torch.ops.aten.permute.default(permute_361, [1, 0]);  permute_361 = None
    view_562: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_54, [1, 512, 768]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    add_181: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_207, view_562);  mul_207 = view_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    mul_215: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(bmm_52, alias_31);  bmm_52 = None
    sum_86: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_215, [1], True)
    mul_216: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(alias_31, sum_86);  alias_31 = sum_86 = None
    sub_80: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(mul_215, mul_216);  mul_215 = mul_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_563: "f32[1, 512, 54]" = torch.ops.aten.view.default(sub_80, [1, 512, 54]);  sub_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    sum_87: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_563, [0, 1], True)
    view_564: "f32[54]" = torch.ops.aten.view.default(sum_87, [54]);  sum_87 = None
    view_565: "f32[512, 54]" = torch.ops.aten.view.default(view_563, [512, 54]);  view_563 = None
    permute_363: "f32[54, 512]" = torch.ops.aten.permute.default(view_565, [1, 0]);  view_565 = None
    mm_56: "f32[54, 384]" = torch.ops.aten.mm.default(permute_363, view_333);  view_333 = None
    permute_364: "f32[384, 54]" = torch.ops.aten.permute.default(mm_56, [1, 0]);  mm_56 = None
    mm_57: "f32[384, 512]" = torch.ops.aten.mm.default(permute_180, permute_363);  permute_180 = permute_363 = None
    permute_366: "f32[512, 384]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    view_566: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_366, [1, 512, 384]);  permute_366 = None
    permute_367: "f32[54, 384]" = torch.ops.aten.permute.default(permute_364, [1, 0]);  permute_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    mul_217: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_566, permute_179);  permute_179 = None
    mul_218: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_566, view_325);  view_566 = view_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_368: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_218, [0, 2, 1, 3]);  getitem_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_52: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_368, memory_format = torch.contiguous_format);  permute_368 = None
    view_567: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_52, [1, 512, 384]);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_369: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_217, [0, 2, 1, 3]);  getitem_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_568: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_369, [1, 512, 384]);  permute_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_370: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_216, [0, 2, 1, 3]);  getitem_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_53: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_370, memory_format = torch.contiguous_format);  permute_370 = None
    view_569: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_53, [1, 512, 384]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    add_182: "f32[1, 512, 384]" = torch.ops.aten.add.Tensor(mul_217, view_569);  mul_217 = view_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    permute_371: "f32[1, 384, 512]" = torch.ops.aten.permute.default(mul_218, [0, 2, 1]);  mul_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    sum_88: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_371, [0, 2], True)
    view_570: "f32[384, 1]" = torch.ops.aten.view.default(sum_88, [384, 1]);  sum_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(permute_371, convolution_18, primals_223, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_371 = convolution_18 = primals_223 = None
    getitem_138: "f32[1, 768, 512]" = convolution_backward_4[0]
    getitem_139: "f32[384, 768, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(getitem_138, permute_174, primals_222, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_138 = permute_174 = primals_222 = None
    getitem_141: "f32[1, 768, 512]" = convolution_backward_5[0]
    getitem_142: "f32[768, 1, 9]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_372: "f32[1, 512, 768]" = torch.ops.aten.permute.default(getitem_141, [0, 2, 1]);  getitem_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    add_183: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_181, permute_372);  add_181 = permute_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_571: "f32[512, 384]" = torch.ops.aten.view.default(view_567, [512, 384]);  view_567 = None
    mm_58: "f32[512, 768]" = torch.ops.aten.mm.default(view_571, permute_373);  permute_373 = None
    permute_374: "f32[384, 512]" = torch.ops.aten.permute.default(view_571, [1, 0])
    mm_59: "f32[384, 768]" = torch.ops.aten.mm.default(permute_374, view_324);  permute_374 = None
    permute_375: "f32[768, 384]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_89: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_571, [0], True);  view_571 = None
    view_572: "f32[384]" = torch.ops.aten.view.default(sum_89, [384]);  sum_89 = None
    permute_376: "f32[384, 768]" = torch.ops.aten.permute.default(permute_375, [1, 0]);  permute_375 = None
    view_573: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_58, [1, 512, 768]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    add_184: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_183, view_573);  add_183 = view_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_574: "f32[512, 384]" = torch.ops.aten.view.default(view_568, [512, 384]);  view_568 = None
    mm_60: "f32[512, 768]" = torch.ops.aten.mm.default(view_574, permute_377);  permute_377 = None
    permute_378: "f32[384, 512]" = torch.ops.aten.permute.default(view_574, [1, 0])
    mm_61: "f32[384, 768]" = torch.ops.aten.mm.default(permute_378, view_324);  permute_378 = None
    permute_379: "f32[768, 384]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_90: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_574, [0], True);  view_574 = None
    view_575: "f32[384]" = torch.ops.aten.view.default(sum_90, [384]);  sum_90 = None
    permute_380: "f32[384, 768]" = torch.ops.aten.permute.default(permute_379, [1, 0]);  permute_379 = None
    view_576: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_60, [1, 512, 768]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    add_185: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_184, view_576);  add_184 = view_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_577: "f32[512, 384]" = torch.ops.aten.view.default(add_182, [512, 384]);  add_182 = None
    mm_62: "f32[512, 768]" = torch.ops.aten.mm.default(view_577, permute_381);  permute_381 = None
    permute_382: "f32[384, 512]" = torch.ops.aten.permute.default(view_577, [1, 0])
    mm_63: "f32[384, 768]" = torch.ops.aten.mm.default(permute_382, view_324);  permute_382 = view_324 = None
    permute_383: "f32[768, 384]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_91: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_577, [0], True);  view_577 = None
    view_578: "f32[384]" = torch.ops.aten.view.default(sum_91, [384]);  sum_91 = None
    permute_384: "f32[384, 768]" = torch.ops.aten.permute.default(permute_383, [1, 0]);  permute_383 = None
    view_579: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_62, [1, 512, 768]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    add_186: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_185, view_579);  add_185 = view_579 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_220: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_186, primals_214);  primals_214 = None
    mul_221: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_220, 768)
    sum_92: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_220, [2], True)
    mul_222: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_220, mul_73);  mul_220 = None
    sum_93: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_222, [2], True);  mul_222 = None
    mul_223: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_73, sum_93);  sum_93 = None
    sub_82: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_221, sum_92);  mul_221 = sum_92 = None
    sub_83: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_82, mul_223);  sub_82 = mul_223 = None
    mul_224: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_48, sub_83);  div_48 = sub_83 = None
    mul_225: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_186, mul_73);  mul_73 = None
    sum_94: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_225, [0, 1]);  mul_225 = None
    sum_95: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_186, [0, 1]);  add_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_10: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_91, torch.float32);  getitem_91 = None
    mul_226: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
    mul_227: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_224, mul_226);  mul_226 = None
    clone_54: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_227, memory_format = torch.contiguous_format);  mul_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_580: "f32[512, 768]" = torch.ops.aten.view.default(clone_54, [512, 768]);  clone_54 = None
    mm_64: "f32[512, 3072]" = torch.ops.aten.mm.default(view_580, permute_385);  permute_385 = None
    permute_386: "f32[768, 512]" = torch.ops.aten.permute.default(view_580, [1, 0])
    mm_65: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_386, view_322);  permute_386 = view_322 = None
    permute_387: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_96: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_580, [0], True);  view_580 = None
    view_581: "f32[768]" = torch.ops.aten.view.default(sum_96, [768]);  sum_96 = None
    permute_388: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_387, [1, 0]);  permute_387 = None
    view_582: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_64, [1, 512, 3072]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_229: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_108, 0.5);  add_108 = None
    mul_230: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_321, view_321)
    mul_231: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_230, -0.5);  mul_230 = None
    exp_30: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_231);  mul_231 = None
    mul_232: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_30, 0.3989422804014327);  exp_30 = None
    mul_233: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_321, mul_232);  view_321 = mul_232 = None
    add_188: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_229, mul_233);  mul_229 = mul_233 = None
    mul_234: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_582, add_188);  view_582 = add_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_583: "f32[512, 3072]" = torch.ops.aten.view.default(mul_234, [512, 3072]);  mul_234 = None
    mm_66: "f32[512, 768]" = torch.ops.aten.mm.default(view_583, permute_389);  permute_389 = None
    permute_390: "f32[3072, 512]" = torch.ops.aten.permute.default(view_583, [1, 0])
    mm_67: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_390, view_320);  permute_390 = view_320 = None
    permute_391: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_97: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_583, [0], True);  view_583 = None
    view_584: "f32[3072]" = torch.ops.aten.view.default(sum_97, [3072]);  sum_97 = None
    permute_392: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_391, [1, 0]);  permute_391 = None
    view_585: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_66, [1, 512, 768]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    add_189: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_224, view_585);  mul_224 = view_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_236: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_189, primals_208);  primals_208 = None
    mul_237: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_236, 768)
    sum_98: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_236, [2], True)
    mul_238: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_236, mul_68);  mul_236 = None
    sum_99: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_238, [2], True);  mul_238 = None
    mul_239: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_68, sum_99);  sum_99 = None
    sub_85: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_237, sum_98);  mul_237 = sum_98 = None
    sub_86: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_85, mul_239);  sub_85 = mul_239 = None
    mul_240: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_49, sub_86);  div_49 = sub_86 = None
    mul_241: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_189, mul_68);  mul_68 = None
    sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_241, [0, 1]);  mul_241 = None
    sum_101: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_189, [0, 1]);  add_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_11: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_87, torch.float32);  getitem_87 = None
    mul_242: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
    mul_243: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_240, mul_242);  mul_242 = None
    clone_55: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_243, memory_format = torch.contiguous_format);  mul_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_586: "f32[512, 768]" = torch.ops.aten.view.default(clone_55, [512, 768]);  clone_55 = None
    mm_68: "f32[512, 768]" = torch.ops.aten.mm.default(view_586, permute_393);  permute_393 = None
    permute_394: "f32[768, 512]" = torch.ops.aten.permute.default(view_586, [1, 0])
    mm_69: "f32[768, 768]" = torch.ops.aten.mm.default(permute_394, view_318);  permute_394 = view_318 = None
    permute_395: "f32[768, 768]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_102: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_586, [0], True);  view_586 = None
    view_587: "f32[768]" = torch.ops.aten.view.default(sum_102, [768]);  sum_102 = None
    permute_396: "f32[768, 768]" = torch.ops.aten.permute.default(permute_395, [1, 0]);  permute_395 = None
    view_588: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_68, [1, 512, 768]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_589: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_588, [1, 512, 12, 64]);  view_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    slice_35: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_589, 2, 0, 6)
    slice_36: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_589, 2, 6, 12);  view_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_590: "f32[512, 384]" = torch.ops.aten.view.default(slice_36, [512, 384]);  slice_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_397: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(slice_35, [0, 2, 1, 3]);  slice_35 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_3 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_397, clone_default_9, clone_default_10, clone_default_11, None, alias_default_7, getitem_220, getitem_221, getitem_222, 0.1, [True, True, True, False], scale = 0.125);  permute_397 = clone_default_9 = clone_default_10 = clone_default_11 = alias_default_7 = getitem_220 = getitem_221 = getitem_222 = None
    getitem_223: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_3[0]
    getitem_224: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_3[1]
    getitem_225: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_3[2];  _scaled_dot_product_efficient_attention_backward_default_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    clone_57: "f32[512, 384]" = torch.ops.aten.clone.default(view_590, memory_format = torch.contiguous_format);  view_590 = None
    view_597: "f32[3072, 64, 1]" = torch.ops.aten.view.default(clone_57, [3072, 64, 1]);  clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    bmm_58: "f32[3072, 9, 1]" = torch.ops.aten.bmm.default(permute_403, view_597);  permute_403 = None
    bmm_59: "f32[3072, 64, 9]" = torch.ops.aten.bmm.default(view_597, permute_404);  view_597 = permute_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    view_601: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(bmm_59, [1, 512, 384, 9]);  bmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    view_602: "f32[1, 512, 3456]" = torch.ops.aten.view.default(view_601, [1, 512, 3456]);  view_601 = None
    permute_405: "f32[1, 3456, 512]" = torch.ops.aten.permute.default(view_602, [0, 2, 1]);  view_602 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    view_603: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.view.default(permute_405, [1, 384, 9, 1, 512, 1]);  permute_405 = None
    permute_406: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.permute.default(view_603, [0, 1, 2, 4, 3, 5]);  view_603 = None
    _unsafe_index_put_3: "f32[1, 384, 520, 1]" = torch.ops.aten._unsafe_index_put.default(full_default_19, [None, None, unsqueeze_8, full_default_1], permute_406, True);  permute_406 = None
    constant_pad_nd_15: "f32[1, 384, 512, 1]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_3, [0, 0, -4, -4], 0.0);  _unsafe_index_put_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    squeeze_4: "f32[1, 384, 512]" = torch.ops.aten.squeeze.dim(constant_pad_nd_15, -1);  constant_pad_nd_15 = None
    permute_407: "f32[1, 512, 384]" = torch.ops.aten.permute.default(squeeze_4, [0, 2, 1]);  squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_605: "f32[512, 384]" = torch.ops.aten.view.default(permute_407, [512, 384]);  permute_407 = None
    mm_70: "f32[512, 768]" = torch.ops.aten.mm.default(view_605, permute_408);  permute_408 = None
    permute_409: "f32[384, 512]" = torch.ops.aten.permute.default(view_605, [1, 0])
    mm_71: "f32[384, 768]" = torch.ops.aten.mm.default(permute_409, view_288);  permute_409 = None
    permute_410: "f32[768, 384]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_104: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_605, [0], True);  view_605 = None
    view_606: "f32[384]" = torch.ops.aten.view.default(sum_104, [384]);  sum_104 = None
    permute_411: "f32[384, 768]" = torch.ops.aten.permute.default(permute_410, [1, 0]);  permute_410 = None
    view_607: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_70, [1, 512, 768]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    add_192: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_240, view_607);  mul_240 = view_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    mul_248: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(bmm_58, alias_33);  bmm_58 = None
    sum_105: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_248, [1], True)
    mul_249: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(alias_33, sum_105);  alias_33 = sum_105 = None
    sub_88: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(mul_248, mul_249);  mul_248 = mul_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_608: "f32[1, 512, 54]" = torch.ops.aten.view.default(sub_88, [1, 512, 54]);  sub_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    sum_106: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_608, [0, 1], True)
    view_609: "f32[54]" = torch.ops.aten.view.default(sum_106, [54]);  sum_106 = None
    view_610: "f32[512, 54]" = torch.ops.aten.view.default(view_608, [512, 54]);  view_608 = None
    permute_412: "f32[54, 512]" = torch.ops.aten.permute.default(view_610, [1, 0]);  view_610 = None
    mm_72: "f32[54, 384]" = torch.ops.aten.mm.default(permute_412, view_297);  view_297 = None
    permute_413: "f32[384, 54]" = torch.ops.aten.permute.default(mm_72, [1, 0]);  mm_72 = None
    mm_73: "f32[384, 512]" = torch.ops.aten.mm.default(permute_161, permute_412);  permute_161 = permute_412 = None
    permute_415: "f32[512, 384]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    view_611: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_415, [1, 512, 384]);  permute_415 = None
    permute_416: "f32[54, 384]" = torch.ops.aten.permute.default(permute_413, [1, 0]);  permute_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    mul_250: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_611, permute_160);  permute_160 = None
    mul_251: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_611, view_289);  view_611 = view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_417: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_225, [0, 2, 1, 3]);  getitem_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_58: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_417, memory_format = torch.contiguous_format);  permute_417 = None
    view_612: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_58, [1, 512, 384]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_418: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_224, [0, 2, 1, 3]);  getitem_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_613: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_418, [1, 512, 384]);  permute_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_419: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_223, [0, 2, 1, 3]);  getitem_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_59: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_419, memory_format = torch.contiguous_format);  permute_419 = None
    view_614: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_59, [1, 512, 384]);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    add_193: "f32[1, 512, 384]" = torch.ops.aten.add.Tensor(mul_250, view_614);  mul_250 = view_614 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    permute_420: "f32[1, 384, 512]" = torch.ops.aten.permute.default(mul_251, [0, 2, 1]);  mul_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    sum_107: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_420, [0, 2], True)
    view_615: "f32[384, 1]" = torch.ops.aten.view.default(sum_107, [384, 1]);  sum_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(permute_420, convolution_16, primals_201, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_420 = convolution_16 = primals_201 = None
    getitem_144: "f32[1, 768, 512]" = convolution_backward_6[0]
    getitem_145: "f32[384, 768, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(getitem_144, permute_155, primals_200, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_144 = permute_155 = primals_200 = None
    getitem_147: "f32[1, 768, 512]" = convolution_backward_7[0]
    getitem_148: "f32[768, 1, 9]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_421: "f32[1, 512, 768]" = torch.ops.aten.permute.default(getitem_147, [0, 2, 1]);  getitem_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    add_194: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_192, permute_421);  add_192 = permute_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_616: "f32[512, 384]" = torch.ops.aten.view.default(view_612, [512, 384]);  view_612 = None
    mm_74: "f32[512, 768]" = torch.ops.aten.mm.default(view_616, permute_422);  permute_422 = None
    permute_423: "f32[384, 512]" = torch.ops.aten.permute.default(view_616, [1, 0])
    mm_75: "f32[384, 768]" = torch.ops.aten.mm.default(permute_423, view_288);  permute_423 = None
    permute_424: "f32[768, 384]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_108: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_616, [0], True);  view_616 = None
    view_617: "f32[384]" = torch.ops.aten.view.default(sum_108, [384]);  sum_108 = None
    permute_425: "f32[384, 768]" = torch.ops.aten.permute.default(permute_424, [1, 0]);  permute_424 = None
    view_618: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_74, [1, 512, 768]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    add_195: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_194, view_618);  add_194 = view_618 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_619: "f32[512, 384]" = torch.ops.aten.view.default(view_613, [512, 384]);  view_613 = None
    mm_76: "f32[512, 768]" = torch.ops.aten.mm.default(view_619, permute_426);  permute_426 = None
    permute_427: "f32[384, 512]" = torch.ops.aten.permute.default(view_619, [1, 0])
    mm_77: "f32[384, 768]" = torch.ops.aten.mm.default(permute_427, view_288);  permute_427 = None
    permute_428: "f32[768, 384]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_109: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_619, [0], True);  view_619 = None
    view_620: "f32[384]" = torch.ops.aten.view.default(sum_109, [384]);  sum_109 = None
    permute_429: "f32[384, 768]" = torch.ops.aten.permute.default(permute_428, [1, 0]);  permute_428 = None
    view_621: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_76, [1, 512, 768]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    add_196: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_195, view_621);  add_195 = view_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_622: "f32[512, 384]" = torch.ops.aten.view.default(add_193, [512, 384]);  add_193 = None
    mm_78: "f32[512, 768]" = torch.ops.aten.mm.default(view_622, permute_430);  permute_430 = None
    permute_431: "f32[384, 512]" = torch.ops.aten.permute.default(view_622, [1, 0])
    mm_79: "f32[384, 768]" = torch.ops.aten.mm.default(permute_431, view_288);  permute_431 = view_288 = None
    permute_432: "f32[768, 384]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_110: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_622, [0], True);  view_622 = None
    view_623: "f32[384]" = torch.ops.aten.view.default(sum_110, [384]);  sum_110 = None
    permute_433: "f32[384, 768]" = torch.ops.aten.permute.default(permute_432, [1, 0]);  permute_432 = None
    view_624: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_78, [1, 512, 768]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    add_197: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_196, view_624);  add_196 = view_624 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_253: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_197, primals_192);  primals_192 = None
    mul_254: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_253, 768)
    sum_111: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_253, [2], True)
    mul_255: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_253, mul_65);  mul_253 = None
    sum_112: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_255, [2], True);  mul_255 = None
    mul_256: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_65, sum_112);  sum_112 = None
    sub_90: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_254, sum_111);  mul_254 = sum_111 = None
    sub_91: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_90, mul_256);  sub_90 = mul_256 = None
    mul_257: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_51, sub_91);  div_51 = sub_91 = None
    mul_258: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_197, mul_65);  mul_65 = None
    sum_113: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_258, [0, 1]);  mul_258 = None
    sum_114: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_197, [0, 1]);  add_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_13: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_81, torch.float32);  getitem_81 = None
    mul_259: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
    mul_260: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_257, mul_259);  mul_259 = None
    clone_60: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_260, memory_format = torch.contiguous_format);  mul_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_625: "f32[512, 768]" = torch.ops.aten.view.default(clone_60, [512, 768]);  clone_60 = None
    mm_80: "f32[512, 3072]" = torch.ops.aten.mm.default(view_625, permute_434);  permute_434 = None
    permute_435: "f32[768, 512]" = torch.ops.aten.permute.default(view_625, [1, 0])
    mm_81: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_435, view_286);  permute_435 = view_286 = None
    permute_436: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_115: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_625, [0], True);  view_625 = None
    view_626: "f32[768]" = torch.ops.aten.view.default(sum_115, [768]);  sum_115 = None
    permute_437: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_436, [1, 0]);  permute_436 = None
    view_627: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_80, [1, 512, 3072]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_262: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_96, 0.5);  add_96 = None
    mul_263: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_285, view_285)
    mul_264: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_263, -0.5);  mul_263 = None
    exp_31: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_264);  mul_264 = None
    mul_265: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_31, 0.3989422804014327);  exp_31 = None
    mul_266: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_285, mul_265);  view_285 = mul_265 = None
    add_199: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_262, mul_266);  mul_262 = mul_266 = None
    mul_267: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_627, add_199);  view_627 = add_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_628: "f32[512, 3072]" = torch.ops.aten.view.default(mul_267, [512, 3072]);  mul_267 = None
    mm_82: "f32[512, 768]" = torch.ops.aten.mm.default(view_628, permute_438);  permute_438 = None
    permute_439: "f32[3072, 512]" = torch.ops.aten.permute.default(view_628, [1, 0])
    mm_83: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_439, view_284);  permute_439 = view_284 = None
    permute_440: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_116: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_628, [0], True);  view_628 = None
    view_629: "f32[3072]" = torch.ops.aten.view.default(sum_116, [3072]);  sum_116 = None
    permute_441: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_440, [1, 0]);  permute_440 = None
    view_630: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_82, [1, 512, 768]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    add_200: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_257, view_630);  mul_257 = view_630 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_269: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_200, primals_186);  primals_186 = None
    mul_270: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_269, 768)
    sum_117: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_269, [2], True)
    mul_271: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_269, mul_60);  mul_269 = None
    sum_118: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_271, [2], True);  mul_271 = None
    mul_272: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_60, sum_118);  sum_118 = None
    sub_93: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_270, sum_117);  mul_270 = sum_117 = None
    sub_94: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_93, mul_272);  sub_93 = mul_272 = None
    mul_273: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_52, sub_94);  div_52 = sub_94 = None
    mul_274: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_200, mul_60);  mul_60 = None
    sum_119: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_274, [0, 1]);  mul_274 = None
    sum_120: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_200, [0, 1]);  add_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_14: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_77, torch.float32);  getitem_77 = None
    mul_275: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
    mul_276: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_273, mul_275);  mul_275 = None
    clone_61: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_276, memory_format = torch.contiguous_format);  mul_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_631: "f32[512, 768]" = torch.ops.aten.view.default(clone_61, [512, 768]);  clone_61 = None
    mm_84: "f32[512, 768]" = torch.ops.aten.mm.default(view_631, permute_442);  permute_442 = None
    permute_443: "f32[768, 512]" = torch.ops.aten.permute.default(view_631, [1, 0])
    mm_85: "f32[768, 768]" = torch.ops.aten.mm.default(permute_443, view_282);  permute_443 = view_282 = None
    permute_444: "f32[768, 768]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_121: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_631, [0], True);  view_631 = None
    view_632: "f32[768]" = torch.ops.aten.view.default(sum_121, [768]);  sum_121 = None
    permute_445: "f32[768, 768]" = torch.ops.aten.permute.default(permute_444, [1, 0]);  permute_444 = None
    view_633: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_84, [1, 512, 768]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_634: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_633, [1, 512, 12, 64]);  view_633 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    slice_37: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_634, 2, 0, 6)
    slice_38: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_634, 2, 6, 12);  view_634 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_635: "f32[512, 384]" = torch.ops.aten.view.default(slice_38, [512, 384]);  slice_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_446: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(slice_37, [0, 2, 1, 3]);  slice_37 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_4 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_446, clone_default_12, clone_default_13, clone_default_14, None, alias_default_9, getitem_227, getitem_228, getitem_229, 0.1, [True, True, True, False], scale = 0.125);  permute_446 = clone_default_12 = clone_default_13 = clone_default_14 = alias_default_9 = getitem_227 = getitem_228 = getitem_229 = None
    getitem_230: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_4[0]
    getitem_231: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_4[1]
    getitem_232: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_4[2];  _scaled_dot_product_efficient_attention_backward_default_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    clone_63: "f32[512, 384]" = torch.ops.aten.clone.default(view_635, memory_format = torch.contiguous_format);  view_635 = None
    view_642: "f32[3072, 64, 1]" = torch.ops.aten.view.default(clone_63, [3072, 64, 1]);  clone_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    bmm_64: "f32[3072, 9, 1]" = torch.ops.aten.bmm.default(permute_452, view_642);  permute_452 = None
    bmm_65: "f32[3072, 64, 9]" = torch.ops.aten.bmm.default(view_642, permute_453);  view_642 = permute_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    view_646: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(bmm_65, [1, 512, 384, 9]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    view_647: "f32[1, 512, 3456]" = torch.ops.aten.view.default(view_646, [1, 512, 3456]);  view_646 = None
    permute_454: "f32[1, 3456, 512]" = torch.ops.aten.permute.default(view_647, [0, 2, 1]);  view_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    view_648: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.view.default(permute_454, [1, 384, 9, 1, 512, 1]);  permute_454 = None
    permute_455: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.permute.default(view_648, [0, 1, 2, 4, 3, 5]);  view_648 = None
    _unsafe_index_put_4: "f32[1, 384, 520, 1]" = torch.ops.aten._unsafe_index_put.default(full_default_19, [None, None, unsqueeze_8, full_default_1], permute_455, True);  permute_455 = None
    constant_pad_nd_16: "f32[1, 384, 512, 1]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_4, [0, 0, -4, -4], 0.0);  _unsafe_index_put_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    squeeze_5: "f32[1, 384, 512]" = torch.ops.aten.squeeze.dim(constant_pad_nd_16, -1);  constant_pad_nd_16 = None
    permute_456: "f32[1, 512, 384]" = torch.ops.aten.permute.default(squeeze_5, [0, 2, 1]);  squeeze_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_650: "f32[512, 384]" = torch.ops.aten.view.default(permute_456, [512, 384]);  permute_456 = None
    mm_86: "f32[512, 768]" = torch.ops.aten.mm.default(view_650, permute_457);  permute_457 = None
    permute_458: "f32[384, 512]" = torch.ops.aten.permute.default(view_650, [1, 0])
    mm_87: "f32[384, 768]" = torch.ops.aten.mm.default(permute_458, view_252);  permute_458 = None
    permute_459: "f32[768, 384]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_123: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_650, [0], True);  view_650 = None
    view_651: "f32[384]" = torch.ops.aten.view.default(sum_123, [384]);  sum_123 = None
    permute_460: "f32[384, 768]" = torch.ops.aten.permute.default(permute_459, [1, 0]);  permute_459 = None
    view_652: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_86, [1, 512, 768]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    add_203: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_273, view_652);  mul_273 = view_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    mul_281: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(bmm_64, alias_35);  bmm_64 = None
    sum_124: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_281, [1], True)
    mul_282: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(alias_35, sum_124);  alias_35 = sum_124 = None
    sub_96: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_653: "f32[1, 512, 54]" = torch.ops.aten.view.default(sub_96, [1, 512, 54]);  sub_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    sum_125: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_653, [0, 1], True)
    view_654: "f32[54]" = torch.ops.aten.view.default(sum_125, [54]);  sum_125 = None
    view_655: "f32[512, 54]" = torch.ops.aten.view.default(view_653, [512, 54]);  view_653 = None
    permute_461: "f32[54, 512]" = torch.ops.aten.permute.default(view_655, [1, 0]);  view_655 = None
    mm_88: "f32[54, 384]" = torch.ops.aten.mm.default(permute_461, view_261);  view_261 = None
    permute_462: "f32[384, 54]" = torch.ops.aten.permute.default(mm_88, [1, 0]);  mm_88 = None
    mm_89: "f32[384, 512]" = torch.ops.aten.mm.default(permute_142, permute_461);  permute_142 = permute_461 = None
    permute_464: "f32[512, 384]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    view_656: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_464, [1, 512, 384]);  permute_464 = None
    permute_465: "f32[54, 384]" = torch.ops.aten.permute.default(permute_462, [1, 0]);  permute_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    mul_283: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_656, permute_141);  permute_141 = None
    mul_284: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_656, view_253);  view_656 = view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_466: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_232, [0, 2, 1, 3]);  getitem_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_64: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_466, memory_format = torch.contiguous_format);  permute_466 = None
    view_657: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_64, [1, 512, 384]);  clone_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_467: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_231, [0, 2, 1, 3]);  getitem_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_658: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_467, [1, 512, 384]);  permute_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_468: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_230, [0, 2, 1, 3]);  getitem_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_65: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_468, memory_format = torch.contiguous_format);  permute_468 = None
    view_659: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_65, [1, 512, 384]);  clone_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    add_204: "f32[1, 512, 384]" = torch.ops.aten.add.Tensor(mul_283, view_659);  mul_283 = view_659 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    permute_469: "f32[1, 384, 512]" = torch.ops.aten.permute.default(mul_284, [0, 2, 1]);  mul_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    sum_126: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_469, [0, 2], True)
    view_660: "f32[384, 1]" = torch.ops.aten.view.default(sum_126, [384, 1]);  sum_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(permute_469, convolution_14, primals_179, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_469 = convolution_14 = primals_179 = None
    getitem_150: "f32[1, 768, 512]" = convolution_backward_8[0]
    getitem_151: "f32[384, 768, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(getitem_150, permute_136, primals_178, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_150 = permute_136 = primals_178 = None
    getitem_153: "f32[1, 768, 512]" = convolution_backward_9[0]
    getitem_154: "f32[768, 1, 9]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_470: "f32[1, 512, 768]" = torch.ops.aten.permute.default(getitem_153, [0, 2, 1]);  getitem_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    add_205: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_203, permute_470);  add_203 = permute_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_661: "f32[512, 384]" = torch.ops.aten.view.default(view_657, [512, 384]);  view_657 = None
    mm_90: "f32[512, 768]" = torch.ops.aten.mm.default(view_661, permute_471);  permute_471 = None
    permute_472: "f32[384, 512]" = torch.ops.aten.permute.default(view_661, [1, 0])
    mm_91: "f32[384, 768]" = torch.ops.aten.mm.default(permute_472, view_252);  permute_472 = None
    permute_473: "f32[768, 384]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_127: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_661, [0], True);  view_661 = None
    view_662: "f32[384]" = torch.ops.aten.view.default(sum_127, [384]);  sum_127 = None
    permute_474: "f32[384, 768]" = torch.ops.aten.permute.default(permute_473, [1, 0]);  permute_473 = None
    view_663: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_90, [1, 512, 768]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    add_206: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_205, view_663);  add_205 = view_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_664: "f32[512, 384]" = torch.ops.aten.view.default(view_658, [512, 384]);  view_658 = None
    mm_92: "f32[512, 768]" = torch.ops.aten.mm.default(view_664, permute_475);  permute_475 = None
    permute_476: "f32[384, 512]" = torch.ops.aten.permute.default(view_664, [1, 0])
    mm_93: "f32[384, 768]" = torch.ops.aten.mm.default(permute_476, view_252);  permute_476 = None
    permute_477: "f32[768, 384]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_128: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_664, [0], True);  view_664 = None
    view_665: "f32[384]" = torch.ops.aten.view.default(sum_128, [384]);  sum_128 = None
    permute_478: "f32[384, 768]" = torch.ops.aten.permute.default(permute_477, [1, 0]);  permute_477 = None
    view_666: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_92, [1, 512, 768]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    add_207: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_206, view_666);  add_206 = view_666 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_667: "f32[512, 384]" = torch.ops.aten.view.default(add_204, [512, 384]);  add_204 = None
    mm_94: "f32[512, 768]" = torch.ops.aten.mm.default(view_667, permute_479);  permute_479 = None
    permute_480: "f32[384, 512]" = torch.ops.aten.permute.default(view_667, [1, 0])
    mm_95: "f32[384, 768]" = torch.ops.aten.mm.default(permute_480, view_252);  permute_480 = view_252 = None
    permute_481: "f32[768, 384]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_129: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_667, [0], True);  view_667 = None
    view_668: "f32[384]" = torch.ops.aten.view.default(sum_129, [384]);  sum_129 = None
    permute_482: "f32[384, 768]" = torch.ops.aten.permute.default(permute_481, [1, 0]);  permute_481 = None
    view_669: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_94, [1, 512, 768]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    add_208: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_207, view_669);  add_207 = view_669 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_286: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_208, primals_170);  primals_170 = None
    mul_287: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_286, 768)
    sum_130: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_286, [2], True)
    mul_288: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_286, mul_57);  mul_286 = None
    sum_131: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_288, [2], True);  mul_288 = None
    mul_289: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_57, sum_131);  sum_131 = None
    sub_98: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_287, sum_130);  mul_287 = sum_130 = None
    sub_99: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_98, mul_289);  sub_98 = mul_289 = None
    mul_290: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_54, sub_99);  div_54 = sub_99 = None
    mul_291: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_208, mul_57);  mul_57 = None
    sum_132: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_291, [0, 1]);  mul_291 = None
    sum_133: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_208, [0, 1]);  add_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_16: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_71, torch.float32);  getitem_71 = None
    mul_292: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
    mul_293: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_290, mul_292);  mul_292 = None
    clone_66: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_293, memory_format = torch.contiguous_format);  mul_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_670: "f32[512, 768]" = torch.ops.aten.view.default(clone_66, [512, 768]);  clone_66 = None
    mm_96: "f32[512, 3072]" = torch.ops.aten.mm.default(view_670, permute_483);  permute_483 = None
    permute_484: "f32[768, 512]" = torch.ops.aten.permute.default(view_670, [1, 0])
    mm_97: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_484, view_250);  permute_484 = view_250 = None
    permute_485: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_134: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_670, [0], True);  view_670 = None
    view_671: "f32[768]" = torch.ops.aten.view.default(sum_134, [768]);  sum_134 = None
    permute_486: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_485, [1, 0]);  permute_485 = None
    view_672: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_96, [1, 512, 3072]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_295: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_84, 0.5);  add_84 = None
    mul_296: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_249, view_249)
    mul_297: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_296, -0.5);  mul_296 = None
    exp_32: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_297);  mul_297 = None
    mul_298: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_32, 0.3989422804014327);  exp_32 = None
    mul_299: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_249, mul_298);  view_249 = mul_298 = None
    add_210: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_295, mul_299);  mul_295 = mul_299 = None
    mul_300: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_672, add_210);  view_672 = add_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_673: "f32[512, 3072]" = torch.ops.aten.view.default(mul_300, [512, 3072]);  mul_300 = None
    mm_98: "f32[512, 768]" = torch.ops.aten.mm.default(view_673, permute_487);  permute_487 = None
    permute_488: "f32[3072, 512]" = torch.ops.aten.permute.default(view_673, [1, 0])
    mm_99: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_488, view_248);  permute_488 = view_248 = None
    permute_489: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_135: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_673, [0], True);  view_673 = None
    view_674: "f32[3072]" = torch.ops.aten.view.default(sum_135, [3072]);  sum_135 = None
    permute_490: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_489, [1, 0]);  permute_489 = None
    view_675: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_98, [1, 512, 768]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    add_211: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_290, view_675);  mul_290 = view_675 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_302: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_211, primals_164);  primals_164 = None
    mul_303: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_302, 768)
    sum_136: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_302, [2], True)
    mul_304: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_302, mul_52);  mul_302 = None
    sum_137: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_304, [2], True);  mul_304 = None
    mul_305: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_52, sum_137);  sum_137 = None
    sub_101: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_303, sum_136);  mul_303 = sum_136 = None
    sub_102: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_101, mul_305);  sub_101 = mul_305 = None
    mul_306: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_55, sub_102);  div_55 = sub_102 = None
    mul_307: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_211, mul_52);  mul_52 = None
    sum_138: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_307, [0, 1]);  mul_307 = None
    sum_139: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_211, [0, 1]);  add_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_17: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_67, torch.float32);  getitem_67 = None
    mul_308: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
    mul_309: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_306, mul_308);  mul_308 = None
    clone_67: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_309, memory_format = torch.contiguous_format);  mul_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_676: "f32[512, 768]" = torch.ops.aten.view.default(clone_67, [512, 768]);  clone_67 = None
    mm_100: "f32[512, 768]" = torch.ops.aten.mm.default(view_676, permute_491);  permute_491 = None
    permute_492: "f32[768, 512]" = torch.ops.aten.permute.default(view_676, [1, 0])
    mm_101: "f32[768, 768]" = torch.ops.aten.mm.default(permute_492, view_246);  permute_492 = view_246 = None
    permute_493: "f32[768, 768]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_140: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_676, [0], True);  view_676 = None
    view_677: "f32[768]" = torch.ops.aten.view.default(sum_140, [768]);  sum_140 = None
    permute_494: "f32[768, 768]" = torch.ops.aten.permute.default(permute_493, [1, 0]);  permute_493 = None
    view_678: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_100, [1, 512, 768]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_679: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_678, [1, 512, 12, 64]);  view_678 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    slice_39: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_679, 2, 0, 6)
    slice_40: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_679, 2, 6, 12);  view_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_680: "f32[512, 384]" = torch.ops.aten.view.default(slice_40, [512, 384]);  slice_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_495: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(slice_39, [0, 2, 1, 3]);  slice_39 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_5 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_495, clone_default_15, clone_default_16, clone_default_17, None, alias_default_11, getitem_234, getitem_235, getitem_236, 0.1, [True, True, True, False], scale = 0.125);  permute_495 = clone_default_15 = clone_default_16 = clone_default_17 = alias_default_11 = getitem_234 = getitem_235 = getitem_236 = None
    getitem_237: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_5[0]
    getitem_238: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_5[1]
    getitem_239: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_5[2];  _scaled_dot_product_efficient_attention_backward_default_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    clone_69: "f32[512, 384]" = torch.ops.aten.clone.default(view_680, memory_format = torch.contiguous_format);  view_680 = None
    view_687: "f32[3072, 64, 1]" = torch.ops.aten.view.default(clone_69, [3072, 64, 1]);  clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    bmm_70: "f32[3072, 9, 1]" = torch.ops.aten.bmm.default(permute_501, view_687);  permute_501 = None
    bmm_71: "f32[3072, 64, 9]" = torch.ops.aten.bmm.default(view_687, permute_502);  view_687 = permute_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    view_691: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(bmm_71, [1, 512, 384, 9]);  bmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    view_692: "f32[1, 512, 3456]" = torch.ops.aten.view.default(view_691, [1, 512, 3456]);  view_691 = None
    permute_503: "f32[1, 3456, 512]" = torch.ops.aten.permute.default(view_692, [0, 2, 1]);  view_692 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    view_693: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.view.default(permute_503, [1, 384, 9, 1, 512, 1]);  permute_503 = None
    permute_504: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.permute.default(view_693, [0, 1, 2, 4, 3, 5]);  view_693 = None
    _unsafe_index_put_5: "f32[1, 384, 520, 1]" = torch.ops.aten._unsafe_index_put.default(full_default_19, [None, None, unsqueeze_8, full_default_1], permute_504, True);  permute_504 = None
    constant_pad_nd_17: "f32[1, 384, 512, 1]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_5, [0, 0, -4, -4], 0.0);  _unsafe_index_put_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    squeeze_6: "f32[1, 384, 512]" = torch.ops.aten.squeeze.dim(constant_pad_nd_17, -1);  constant_pad_nd_17 = None
    permute_505: "f32[1, 512, 384]" = torch.ops.aten.permute.default(squeeze_6, [0, 2, 1]);  squeeze_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_695: "f32[512, 384]" = torch.ops.aten.view.default(permute_505, [512, 384]);  permute_505 = None
    mm_102: "f32[512, 768]" = torch.ops.aten.mm.default(view_695, permute_506);  permute_506 = None
    permute_507: "f32[384, 512]" = torch.ops.aten.permute.default(view_695, [1, 0])
    mm_103: "f32[384, 768]" = torch.ops.aten.mm.default(permute_507, view_216);  permute_507 = None
    permute_508: "f32[768, 384]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_142: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_695, [0], True);  view_695 = None
    view_696: "f32[384]" = torch.ops.aten.view.default(sum_142, [384]);  sum_142 = None
    permute_509: "f32[384, 768]" = torch.ops.aten.permute.default(permute_508, [1, 0]);  permute_508 = None
    view_697: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_102, [1, 512, 768]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    add_214: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_306, view_697);  mul_306 = view_697 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    mul_314: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(bmm_70, alias_37);  bmm_70 = None
    sum_143: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_314, [1], True)
    mul_315: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(alias_37, sum_143);  alias_37 = sum_143 = None
    sub_104: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(mul_314, mul_315);  mul_314 = mul_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_698: "f32[1, 512, 54]" = torch.ops.aten.view.default(sub_104, [1, 512, 54]);  sub_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    sum_144: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_698, [0, 1], True)
    view_699: "f32[54]" = torch.ops.aten.view.default(sum_144, [54]);  sum_144 = None
    view_700: "f32[512, 54]" = torch.ops.aten.view.default(view_698, [512, 54]);  view_698 = None
    permute_510: "f32[54, 512]" = torch.ops.aten.permute.default(view_700, [1, 0]);  view_700 = None
    mm_104: "f32[54, 384]" = torch.ops.aten.mm.default(permute_510, view_225);  view_225 = None
    permute_511: "f32[384, 54]" = torch.ops.aten.permute.default(mm_104, [1, 0]);  mm_104 = None
    mm_105: "f32[384, 512]" = torch.ops.aten.mm.default(permute_123, permute_510);  permute_123 = permute_510 = None
    permute_513: "f32[512, 384]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    view_701: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_513, [1, 512, 384]);  permute_513 = None
    permute_514: "f32[54, 384]" = torch.ops.aten.permute.default(permute_511, [1, 0]);  permute_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    mul_316: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_701, permute_122);  permute_122 = None
    mul_317: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_701, view_217);  view_701 = view_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_515: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_239, [0, 2, 1, 3]);  getitem_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_70: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_515, memory_format = torch.contiguous_format);  permute_515 = None
    view_702: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_70, [1, 512, 384]);  clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_516: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_238, [0, 2, 1, 3]);  getitem_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_703: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_516, [1, 512, 384]);  permute_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_517: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_237, [0, 2, 1, 3]);  getitem_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_71: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_517, memory_format = torch.contiguous_format);  permute_517 = None
    view_704: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_71, [1, 512, 384]);  clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    add_215: "f32[1, 512, 384]" = torch.ops.aten.add.Tensor(mul_316, view_704);  mul_316 = view_704 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    permute_518: "f32[1, 384, 512]" = torch.ops.aten.permute.default(mul_317, [0, 2, 1]);  mul_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    sum_145: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_518, [0, 2], True)
    view_705: "f32[384, 1]" = torch.ops.aten.view.default(sum_145, [384, 1]);  sum_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(permute_518, convolution_12, primals_157, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_518 = convolution_12 = primals_157 = None
    getitem_156: "f32[1, 768, 512]" = convolution_backward_10[0]
    getitem_157: "f32[384, 768, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(getitem_156, permute_117, primals_156, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_156 = permute_117 = primals_156 = None
    getitem_159: "f32[1, 768, 512]" = convolution_backward_11[0]
    getitem_160: "f32[768, 1, 9]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_519: "f32[1, 512, 768]" = torch.ops.aten.permute.default(getitem_159, [0, 2, 1]);  getitem_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    add_216: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_214, permute_519);  add_214 = permute_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_706: "f32[512, 384]" = torch.ops.aten.view.default(view_702, [512, 384]);  view_702 = None
    mm_106: "f32[512, 768]" = torch.ops.aten.mm.default(view_706, permute_520);  permute_520 = None
    permute_521: "f32[384, 512]" = torch.ops.aten.permute.default(view_706, [1, 0])
    mm_107: "f32[384, 768]" = torch.ops.aten.mm.default(permute_521, view_216);  permute_521 = None
    permute_522: "f32[768, 384]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_146: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_706, [0], True);  view_706 = None
    view_707: "f32[384]" = torch.ops.aten.view.default(sum_146, [384]);  sum_146 = None
    permute_523: "f32[384, 768]" = torch.ops.aten.permute.default(permute_522, [1, 0]);  permute_522 = None
    view_708: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_106, [1, 512, 768]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    add_217: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_216, view_708);  add_216 = view_708 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_709: "f32[512, 384]" = torch.ops.aten.view.default(view_703, [512, 384]);  view_703 = None
    mm_108: "f32[512, 768]" = torch.ops.aten.mm.default(view_709, permute_524);  permute_524 = None
    permute_525: "f32[384, 512]" = torch.ops.aten.permute.default(view_709, [1, 0])
    mm_109: "f32[384, 768]" = torch.ops.aten.mm.default(permute_525, view_216);  permute_525 = None
    permute_526: "f32[768, 384]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_147: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_709, [0], True);  view_709 = None
    view_710: "f32[384]" = torch.ops.aten.view.default(sum_147, [384]);  sum_147 = None
    permute_527: "f32[384, 768]" = torch.ops.aten.permute.default(permute_526, [1, 0]);  permute_526 = None
    view_711: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_108, [1, 512, 768]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    add_218: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_217, view_711);  add_217 = view_711 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_712: "f32[512, 384]" = torch.ops.aten.view.default(add_215, [512, 384]);  add_215 = None
    mm_110: "f32[512, 768]" = torch.ops.aten.mm.default(view_712, permute_528);  permute_528 = None
    permute_529: "f32[384, 512]" = torch.ops.aten.permute.default(view_712, [1, 0])
    mm_111: "f32[384, 768]" = torch.ops.aten.mm.default(permute_529, view_216);  permute_529 = view_216 = None
    permute_530: "f32[768, 384]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_148: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_712, [0], True);  view_712 = None
    view_713: "f32[384]" = torch.ops.aten.view.default(sum_148, [384]);  sum_148 = None
    permute_531: "f32[384, 768]" = torch.ops.aten.permute.default(permute_530, [1, 0]);  permute_530 = None
    view_714: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_110, [1, 512, 768]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    add_219: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_218, view_714);  add_218 = view_714 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_319: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_219, primals_148);  primals_148 = None
    mul_320: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_319, 768)
    sum_149: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_319, [2], True)
    mul_321: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_319, mul_49);  mul_319 = None
    sum_150: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_321, [2], True);  mul_321 = None
    mul_322: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_49, sum_150);  sum_150 = None
    sub_106: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_320, sum_149);  mul_320 = sum_149 = None
    sub_107: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_106, mul_322);  sub_106 = mul_322 = None
    mul_323: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_57, sub_107);  div_57 = sub_107 = None
    mul_324: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_219, mul_49);  mul_49 = None
    sum_151: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_324, [0, 1]);  mul_324 = None
    sum_152: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_219, [0, 1]);  add_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_19: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_61, torch.float32);  getitem_61 = None
    mul_325: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
    mul_326: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_323, mul_325);  mul_325 = None
    clone_72: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_326, memory_format = torch.contiguous_format);  mul_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_715: "f32[512, 768]" = torch.ops.aten.view.default(clone_72, [512, 768]);  clone_72 = None
    mm_112: "f32[512, 3072]" = torch.ops.aten.mm.default(view_715, permute_532);  permute_532 = None
    permute_533: "f32[768, 512]" = torch.ops.aten.permute.default(view_715, [1, 0])
    mm_113: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_533, view_214);  permute_533 = view_214 = None
    permute_534: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    sum_153: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_715, [0], True);  view_715 = None
    view_716: "f32[768]" = torch.ops.aten.view.default(sum_153, [768]);  sum_153 = None
    permute_535: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_534, [1, 0]);  permute_534 = None
    view_717: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_112, [1, 512, 3072]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_328: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_72, 0.5);  add_72 = None
    mul_329: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_213, view_213)
    mul_330: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_329, -0.5);  mul_329 = None
    exp_33: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_330);  mul_330 = None
    mul_331: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_33, 0.3989422804014327);  exp_33 = None
    mul_332: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_213, mul_331);  view_213 = mul_331 = None
    add_221: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_328, mul_332);  mul_328 = mul_332 = None
    mul_333: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_717, add_221);  view_717 = add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_718: "f32[512, 3072]" = torch.ops.aten.view.default(mul_333, [512, 3072]);  mul_333 = None
    mm_114: "f32[512, 768]" = torch.ops.aten.mm.default(view_718, permute_536);  permute_536 = None
    permute_537: "f32[3072, 512]" = torch.ops.aten.permute.default(view_718, [1, 0])
    mm_115: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_537, view_212);  permute_537 = view_212 = None
    permute_538: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_154: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_718, [0], True);  view_718 = None
    view_719: "f32[3072]" = torch.ops.aten.view.default(sum_154, [3072]);  sum_154 = None
    permute_539: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_538, [1, 0]);  permute_538 = None
    view_720: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_114, [1, 512, 768]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    add_222: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_323, view_720);  mul_323 = view_720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_335: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_222, primals_142);  primals_142 = None
    mul_336: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_335, 768)
    sum_155: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_335, [2], True)
    mul_337: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_335, mul_44);  mul_335 = None
    sum_156: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_337, [2], True);  mul_337 = None
    mul_338: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_44, sum_156);  sum_156 = None
    sub_109: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_336, sum_155);  mul_336 = sum_155 = None
    sub_110: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_109, mul_338);  sub_109 = mul_338 = None
    mul_339: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_58, sub_110);  div_58 = sub_110 = None
    mul_340: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_222, mul_44);  mul_44 = None
    sum_157: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_340, [0, 1]);  mul_340 = None
    sum_158: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_222, [0, 1]);  add_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_20: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_57, torch.float32);  getitem_57 = None
    mul_341: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
    mul_342: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_339, mul_341);  mul_341 = None
    clone_73: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_342, memory_format = torch.contiguous_format);  mul_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_721: "f32[512, 768]" = torch.ops.aten.view.default(clone_73, [512, 768]);  clone_73 = None
    mm_116: "f32[512, 768]" = torch.ops.aten.mm.default(view_721, permute_540);  permute_540 = None
    permute_541: "f32[768, 512]" = torch.ops.aten.permute.default(view_721, [1, 0])
    mm_117: "f32[768, 768]" = torch.ops.aten.mm.default(permute_541, view_210);  permute_541 = view_210 = None
    permute_542: "f32[768, 768]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_159: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_721, [0], True);  view_721 = None
    view_722: "f32[768]" = torch.ops.aten.view.default(sum_159, [768]);  sum_159 = None
    permute_543: "f32[768, 768]" = torch.ops.aten.permute.default(permute_542, [1, 0]);  permute_542 = None
    view_723: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_116, [1, 512, 768]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_724: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_723, [1, 512, 12, 64]);  view_723 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    slice_41: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_724, 2, 0, 6)
    slice_42: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_724, 2, 6, 12);  view_724 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_725: "f32[512, 384]" = torch.ops.aten.view.default(slice_42, [512, 384]);  slice_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_544: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(slice_41, [0, 2, 1, 3]);  slice_41 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_6 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_544, clone_default_18, clone_default_19, clone_default_20, None, alias_default_13, getitem_241, getitem_242, getitem_243, 0.1, [True, True, True, False], scale = 0.125);  permute_544 = clone_default_18 = clone_default_19 = clone_default_20 = alias_default_13 = getitem_241 = getitem_242 = getitem_243 = None
    getitem_244: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_6[0]
    getitem_245: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_6[1]
    getitem_246: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_6[2];  _scaled_dot_product_efficient_attention_backward_default_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    clone_75: "f32[512, 384]" = torch.ops.aten.clone.default(view_725, memory_format = torch.contiguous_format);  view_725 = None
    view_732: "f32[3072, 64, 1]" = torch.ops.aten.view.default(clone_75, [3072, 64, 1]);  clone_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    bmm_76: "f32[3072, 9, 1]" = torch.ops.aten.bmm.default(permute_550, view_732);  permute_550 = None
    bmm_77: "f32[3072, 64, 9]" = torch.ops.aten.bmm.default(view_732, permute_551);  view_732 = permute_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    view_736: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(bmm_77, [1, 512, 384, 9]);  bmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    view_737: "f32[1, 512, 3456]" = torch.ops.aten.view.default(view_736, [1, 512, 3456]);  view_736 = None
    permute_552: "f32[1, 3456, 512]" = torch.ops.aten.permute.default(view_737, [0, 2, 1]);  view_737 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    view_738: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.view.default(permute_552, [1, 384, 9, 1, 512, 1]);  permute_552 = None
    permute_553: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.permute.default(view_738, [0, 1, 2, 4, 3, 5]);  view_738 = None
    _unsafe_index_put_6: "f32[1, 384, 520, 1]" = torch.ops.aten._unsafe_index_put.default(full_default_19, [None, None, unsqueeze_8, full_default_1], permute_553, True);  permute_553 = None
    constant_pad_nd_18: "f32[1, 384, 512, 1]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_6, [0, 0, -4, -4], 0.0);  _unsafe_index_put_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    squeeze_7: "f32[1, 384, 512]" = torch.ops.aten.squeeze.dim(constant_pad_nd_18, -1);  constant_pad_nd_18 = None
    permute_554: "f32[1, 512, 384]" = torch.ops.aten.permute.default(squeeze_7, [0, 2, 1]);  squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_740: "f32[512, 384]" = torch.ops.aten.view.default(permute_554, [512, 384]);  permute_554 = None
    mm_118: "f32[512, 768]" = torch.ops.aten.mm.default(view_740, permute_555);  permute_555 = None
    permute_556: "f32[384, 512]" = torch.ops.aten.permute.default(view_740, [1, 0])
    mm_119: "f32[384, 768]" = torch.ops.aten.mm.default(permute_556, view_180);  permute_556 = None
    permute_557: "f32[768, 384]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_161: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_740, [0], True);  view_740 = None
    view_741: "f32[384]" = torch.ops.aten.view.default(sum_161, [384]);  sum_161 = None
    permute_558: "f32[384, 768]" = torch.ops.aten.permute.default(permute_557, [1, 0]);  permute_557 = None
    view_742: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_118, [1, 512, 768]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    add_225: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_339, view_742);  mul_339 = view_742 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    mul_347: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(bmm_76, alias_39);  bmm_76 = None
    sum_162: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_347, [1], True)
    mul_348: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(alias_39, sum_162);  alias_39 = sum_162 = None
    sub_112: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_743: "f32[1, 512, 54]" = torch.ops.aten.view.default(sub_112, [1, 512, 54]);  sub_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    sum_163: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_743, [0, 1], True)
    view_744: "f32[54]" = torch.ops.aten.view.default(sum_163, [54]);  sum_163 = None
    view_745: "f32[512, 54]" = torch.ops.aten.view.default(view_743, [512, 54]);  view_743 = None
    permute_559: "f32[54, 512]" = torch.ops.aten.permute.default(view_745, [1, 0]);  view_745 = None
    mm_120: "f32[54, 384]" = torch.ops.aten.mm.default(permute_559, view_189);  view_189 = None
    permute_560: "f32[384, 54]" = torch.ops.aten.permute.default(mm_120, [1, 0]);  mm_120 = None
    mm_121: "f32[384, 512]" = torch.ops.aten.mm.default(permute_104, permute_559);  permute_104 = permute_559 = None
    permute_562: "f32[512, 384]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    view_746: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_562, [1, 512, 384]);  permute_562 = None
    permute_563: "f32[54, 384]" = torch.ops.aten.permute.default(permute_560, [1, 0]);  permute_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    mul_349: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_746, permute_103);  permute_103 = None
    mul_350: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_746, view_181);  view_746 = view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_564: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_246, [0, 2, 1, 3]);  getitem_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_76: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_564, memory_format = torch.contiguous_format);  permute_564 = None
    view_747: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_76, [1, 512, 384]);  clone_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_565: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_245, [0, 2, 1, 3]);  getitem_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_748: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_565, [1, 512, 384]);  permute_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_566: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_244, [0, 2, 1, 3]);  getitem_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_77: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_566, memory_format = torch.contiguous_format);  permute_566 = None
    view_749: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_77, [1, 512, 384]);  clone_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    add_226: "f32[1, 512, 384]" = torch.ops.aten.add.Tensor(mul_349, view_749);  mul_349 = view_749 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    permute_567: "f32[1, 384, 512]" = torch.ops.aten.permute.default(mul_350, [0, 2, 1]);  mul_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    sum_164: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_567, [0, 2], True)
    view_750: "f32[384, 1]" = torch.ops.aten.view.default(sum_164, [384, 1]);  sum_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(permute_567, convolution_10, primals_135, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_567 = convolution_10 = primals_135 = None
    getitem_162: "f32[1, 768, 512]" = convolution_backward_12[0]
    getitem_163: "f32[384, 768, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(getitem_162, permute_98, primals_134, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_162 = permute_98 = primals_134 = None
    getitem_165: "f32[1, 768, 512]" = convolution_backward_13[0]
    getitem_166: "f32[768, 1, 9]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_568: "f32[1, 512, 768]" = torch.ops.aten.permute.default(getitem_165, [0, 2, 1]);  getitem_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    add_227: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_225, permute_568);  add_225 = permute_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_751: "f32[512, 384]" = torch.ops.aten.view.default(view_747, [512, 384]);  view_747 = None
    mm_122: "f32[512, 768]" = torch.ops.aten.mm.default(view_751, permute_569);  permute_569 = None
    permute_570: "f32[384, 512]" = torch.ops.aten.permute.default(view_751, [1, 0])
    mm_123: "f32[384, 768]" = torch.ops.aten.mm.default(permute_570, view_180);  permute_570 = None
    permute_571: "f32[768, 384]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_165: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_751, [0], True);  view_751 = None
    view_752: "f32[384]" = torch.ops.aten.view.default(sum_165, [384]);  sum_165 = None
    permute_572: "f32[384, 768]" = torch.ops.aten.permute.default(permute_571, [1, 0]);  permute_571 = None
    view_753: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_122, [1, 512, 768]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    add_228: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_227, view_753);  add_227 = view_753 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_754: "f32[512, 384]" = torch.ops.aten.view.default(view_748, [512, 384]);  view_748 = None
    mm_124: "f32[512, 768]" = torch.ops.aten.mm.default(view_754, permute_573);  permute_573 = None
    permute_574: "f32[384, 512]" = torch.ops.aten.permute.default(view_754, [1, 0])
    mm_125: "f32[384, 768]" = torch.ops.aten.mm.default(permute_574, view_180);  permute_574 = None
    permute_575: "f32[768, 384]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    sum_166: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_754, [0], True);  view_754 = None
    view_755: "f32[384]" = torch.ops.aten.view.default(sum_166, [384]);  sum_166 = None
    permute_576: "f32[384, 768]" = torch.ops.aten.permute.default(permute_575, [1, 0]);  permute_575 = None
    view_756: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_124, [1, 512, 768]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    add_229: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_228, view_756);  add_228 = view_756 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_757: "f32[512, 384]" = torch.ops.aten.view.default(add_226, [512, 384]);  add_226 = None
    mm_126: "f32[512, 768]" = torch.ops.aten.mm.default(view_757, permute_577);  permute_577 = None
    permute_578: "f32[384, 512]" = torch.ops.aten.permute.default(view_757, [1, 0])
    mm_127: "f32[384, 768]" = torch.ops.aten.mm.default(permute_578, view_180);  permute_578 = view_180 = None
    permute_579: "f32[768, 384]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_167: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_757, [0], True);  view_757 = None
    view_758: "f32[384]" = torch.ops.aten.view.default(sum_167, [384]);  sum_167 = None
    permute_580: "f32[384, 768]" = torch.ops.aten.permute.default(permute_579, [1, 0]);  permute_579 = None
    view_759: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_126, [1, 512, 768]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    add_230: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_229, view_759);  add_229 = view_759 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_352: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_230, primals_126);  primals_126 = None
    mul_353: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_352, 768)
    sum_168: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_352, [2], True)
    mul_354: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_352, mul_41);  mul_352 = None
    sum_169: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_354, [2], True);  mul_354 = None
    mul_355: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_41, sum_169);  sum_169 = None
    sub_114: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_353, sum_168);  mul_353 = sum_168 = None
    sub_115: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_114, mul_355);  sub_114 = mul_355 = None
    mul_356: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_60, sub_115);  div_60 = sub_115 = None
    mul_357: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_230, mul_41);  mul_41 = None
    sum_170: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_357, [0, 1]);  mul_357 = None
    sum_171: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_230, [0, 1]);  add_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_22: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_51, torch.float32);  getitem_51 = None
    mul_358: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
    mul_359: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_356, mul_358);  mul_358 = None
    clone_78: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_359, memory_format = torch.contiguous_format);  mul_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_760: "f32[512, 768]" = torch.ops.aten.view.default(clone_78, [512, 768]);  clone_78 = None
    mm_128: "f32[512, 3072]" = torch.ops.aten.mm.default(view_760, permute_581);  permute_581 = None
    permute_582: "f32[768, 512]" = torch.ops.aten.permute.default(view_760, [1, 0])
    mm_129: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_582, view_178);  permute_582 = view_178 = None
    permute_583: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_172: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_760, [0], True);  view_760 = None
    view_761: "f32[768]" = torch.ops.aten.view.default(sum_172, [768]);  sum_172 = None
    permute_584: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_583, [1, 0]);  permute_583 = None
    view_762: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_128, [1, 512, 3072]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_361: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_60, 0.5);  add_60 = None
    mul_362: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_177, view_177)
    mul_363: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_362, -0.5);  mul_362 = None
    exp_34: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_363);  mul_363 = None
    mul_364: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_34, 0.3989422804014327);  exp_34 = None
    mul_365: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_177, mul_364);  view_177 = mul_364 = None
    add_232: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_361, mul_365);  mul_361 = mul_365 = None
    mul_366: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_762, add_232);  view_762 = add_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_763: "f32[512, 3072]" = torch.ops.aten.view.default(mul_366, [512, 3072]);  mul_366 = None
    mm_130: "f32[512, 768]" = torch.ops.aten.mm.default(view_763, permute_585);  permute_585 = None
    permute_586: "f32[3072, 512]" = torch.ops.aten.permute.default(view_763, [1, 0])
    mm_131: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_586, view_176);  permute_586 = view_176 = None
    permute_587: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    sum_173: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_763, [0], True);  view_763 = None
    view_764: "f32[3072]" = torch.ops.aten.view.default(sum_173, [3072]);  sum_173 = None
    permute_588: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_587, [1, 0]);  permute_587 = None
    view_765: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_130, [1, 512, 768]);  mm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    add_233: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_356, view_765);  mul_356 = view_765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_368: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_233, primals_120);  primals_120 = None
    mul_369: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_368, 768)
    sum_174: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_368, [2], True)
    mul_370: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_368, mul_36);  mul_368 = None
    sum_175: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_370, [2], True);  mul_370 = None
    mul_371: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_36, sum_175);  sum_175 = None
    sub_117: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_369, sum_174);  mul_369 = sum_174 = None
    sub_118: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_117, mul_371);  sub_117 = mul_371 = None
    mul_372: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_61, sub_118);  div_61 = sub_118 = None
    mul_373: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_233, mul_36);  mul_36 = None
    sum_176: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_373, [0, 1]);  mul_373 = None
    sum_177: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_233, [0, 1]);  add_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_23: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_47, torch.float32);  getitem_47 = None
    mul_374: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
    mul_375: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_372, mul_374);  mul_374 = None
    clone_79: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_375, memory_format = torch.contiguous_format);  mul_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_766: "f32[512, 768]" = torch.ops.aten.view.default(clone_79, [512, 768]);  clone_79 = None
    mm_132: "f32[512, 768]" = torch.ops.aten.mm.default(view_766, permute_589);  permute_589 = None
    permute_590: "f32[768, 512]" = torch.ops.aten.permute.default(view_766, [1, 0])
    mm_133: "f32[768, 768]" = torch.ops.aten.mm.default(permute_590, view_174);  permute_590 = view_174 = None
    permute_591: "f32[768, 768]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_178: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_766, [0], True);  view_766 = None
    view_767: "f32[768]" = torch.ops.aten.view.default(sum_178, [768]);  sum_178 = None
    permute_592: "f32[768, 768]" = torch.ops.aten.permute.default(permute_591, [1, 0]);  permute_591 = None
    view_768: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_132, [1, 512, 768]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_769: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_768, [1, 512, 12, 64]);  view_768 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    slice_43: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_769, 2, 0, 6)
    slice_44: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_769, 2, 6, 12);  view_769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_770: "f32[512, 384]" = torch.ops.aten.view.default(slice_44, [512, 384]);  slice_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_593: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(slice_43, [0, 2, 1, 3]);  slice_43 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_7 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_593, clone_default_21, clone_default_22, clone_default_23, None, alias_default_15, getitem_248, getitem_249, getitem_250, 0.1, [True, True, True, False], scale = 0.125);  permute_593 = clone_default_21 = clone_default_22 = clone_default_23 = alias_default_15 = getitem_248 = getitem_249 = getitem_250 = None
    getitem_251: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_7[0]
    getitem_252: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_7[1]
    getitem_253: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_7[2];  _scaled_dot_product_efficient_attention_backward_default_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    clone_81: "f32[512, 384]" = torch.ops.aten.clone.default(view_770, memory_format = torch.contiguous_format);  view_770 = None
    view_777: "f32[3072, 64, 1]" = torch.ops.aten.view.default(clone_81, [3072, 64, 1]);  clone_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    bmm_82: "f32[3072, 9, 1]" = torch.ops.aten.bmm.default(permute_599, view_777);  permute_599 = None
    bmm_83: "f32[3072, 64, 9]" = torch.ops.aten.bmm.default(view_777, permute_600);  view_777 = permute_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    view_781: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(bmm_83, [1, 512, 384, 9]);  bmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    view_782: "f32[1, 512, 3456]" = torch.ops.aten.view.default(view_781, [1, 512, 3456]);  view_781 = None
    permute_601: "f32[1, 3456, 512]" = torch.ops.aten.permute.default(view_782, [0, 2, 1]);  view_782 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    view_783: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.view.default(permute_601, [1, 384, 9, 1, 512, 1]);  permute_601 = None
    permute_602: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.permute.default(view_783, [0, 1, 2, 4, 3, 5]);  view_783 = None
    _unsafe_index_put_7: "f32[1, 384, 520, 1]" = torch.ops.aten._unsafe_index_put.default(full_default_19, [None, None, unsqueeze_8, full_default_1], permute_602, True);  permute_602 = None
    constant_pad_nd_19: "f32[1, 384, 512, 1]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_7, [0, 0, -4, -4], 0.0);  _unsafe_index_put_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    squeeze_8: "f32[1, 384, 512]" = torch.ops.aten.squeeze.dim(constant_pad_nd_19, -1);  constant_pad_nd_19 = None
    permute_603: "f32[1, 512, 384]" = torch.ops.aten.permute.default(squeeze_8, [0, 2, 1]);  squeeze_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_785: "f32[512, 384]" = torch.ops.aten.view.default(permute_603, [512, 384]);  permute_603 = None
    mm_134: "f32[512, 768]" = torch.ops.aten.mm.default(view_785, permute_604);  permute_604 = None
    permute_605: "f32[384, 512]" = torch.ops.aten.permute.default(view_785, [1, 0])
    mm_135: "f32[384, 768]" = torch.ops.aten.mm.default(permute_605, view_144);  permute_605 = None
    permute_606: "f32[768, 384]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_180: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_785, [0], True);  view_785 = None
    view_786: "f32[384]" = torch.ops.aten.view.default(sum_180, [384]);  sum_180 = None
    permute_607: "f32[384, 768]" = torch.ops.aten.permute.default(permute_606, [1, 0]);  permute_606 = None
    view_787: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_134, [1, 512, 768]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    add_236: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_372, view_787);  mul_372 = view_787 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    mul_380: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(bmm_82, alias_41);  bmm_82 = None
    sum_181: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_380, [1], True)
    mul_381: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(alias_41, sum_181);  alias_41 = sum_181 = None
    sub_120: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(mul_380, mul_381);  mul_380 = mul_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_788: "f32[1, 512, 54]" = torch.ops.aten.view.default(sub_120, [1, 512, 54]);  sub_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    sum_182: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_788, [0, 1], True)
    view_789: "f32[54]" = torch.ops.aten.view.default(sum_182, [54]);  sum_182 = None
    view_790: "f32[512, 54]" = torch.ops.aten.view.default(view_788, [512, 54]);  view_788 = None
    permute_608: "f32[54, 512]" = torch.ops.aten.permute.default(view_790, [1, 0]);  view_790 = None
    mm_136: "f32[54, 384]" = torch.ops.aten.mm.default(permute_608, view_153);  view_153 = None
    permute_609: "f32[384, 54]" = torch.ops.aten.permute.default(mm_136, [1, 0]);  mm_136 = None
    mm_137: "f32[384, 512]" = torch.ops.aten.mm.default(permute_85, permute_608);  permute_85 = permute_608 = None
    permute_611: "f32[512, 384]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    view_791: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_611, [1, 512, 384]);  permute_611 = None
    permute_612: "f32[54, 384]" = torch.ops.aten.permute.default(permute_609, [1, 0]);  permute_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    mul_382: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_791, permute_84);  permute_84 = None
    mul_383: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_791, view_145);  view_791 = view_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_613: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_253, [0, 2, 1, 3]);  getitem_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_82: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_613, memory_format = torch.contiguous_format);  permute_613 = None
    view_792: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_82, [1, 512, 384]);  clone_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_614: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_252, [0, 2, 1, 3]);  getitem_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_793: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_614, [1, 512, 384]);  permute_614 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_615: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_251, [0, 2, 1, 3]);  getitem_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_83: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_615, memory_format = torch.contiguous_format);  permute_615 = None
    view_794: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_83, [1, 512, 384]);  clone_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    add_237: "f32[1, 512, 384]" = torch.ops.aten.add.Tensor(mul_382, view_794);  mul_382 = view_794 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    permute_616: "f32[1, 384, 512]" = torch.ops.aten.permute.default(mul_383, [0, 2, 1]);  mul_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    sum_183: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_616, [0, 2], True)
    view_795: "f32[384, 1]" = torch.ops.aten.view.default(sum_183, [384, 1]);  sum_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(permute_616, convolution_8, primals_113, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_616 = convolution_8 = primals_113 = None
    getitem_168: "f32[1, 768, 512]" = convolution_backward_14[0]
    getitem_169: "f32[384, 768, 1]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(getitem_168, permute_79, primals_112, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_168 = permute_79 = primals_112 = None
    getitem_171: "f32[1, 768, 512]" = convolution_backward_15[0]
    getitem_172: "f32[768, 1, 9]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_617: "f32[1, 512, 768]" = torch.ops.aten.permute.default(getitem_171, [0, 2, 1]);  getitem_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    add_238: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_236, permute_617);  add_236 = permute_617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_796: "f32[512, 384]" = torch.ops.aten.view.default(view_792, [512, 384]);  view_792 = None
    mm_138: "f32[512, 768]" = torch.ops.aten.mm.default(view_796, permute_618);  permute_618 = None
    permute_619: "f32[384, 512]" = torch.ops.aten.permute.default(view_796, [1, 0])
    mm_139: "f32[384, 768]" = torch.ops.aten.mm.default(permute_619, view_144);  permute_619 = None
    permute_620: "f32[768, 384]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_184: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_796, [0], True);  view_796 = None
    view_797: "f32[384]" = torch.ops.aten.view.default(sum_184, [384]);  sum_184 = None
    permute_621: "f32[384, 768]" = torch.ops.aten.permute.default(permute_620, [1, 0]);  permute_620 = None
    view_798: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_138, [1, 512, 768]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    add_239: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_238, view_798);  add_238 = view_798 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_799: "f32[512, 384]" = torch.ops.aten.view.default(view_793, [512, 384]);  view_793 = None
    mm_140: "f32[512, 768]" = torch.ops.aten.mm.default(view_799, permute_622);  permute_622 = None
    permute_623: "f32[384, 512]" = torch.ops.aten.permute.default(view_799, [1, 0])
    mm_141: "f32[384, 768]" = torch.ops.aten.mm.default(permute_623, view_144);  permute_623 = None
    permute_624: "f32[768, 384]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_185: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_799, [0], True);  view_799 = None
    view_800: "f32[384]" = torch.ops.aten.view.default(sum_185, [384]);  sum_185 = None
    permute_625: "f32[384, 768]" = torch.ops.aten.permute.default(permute_624, [1, 0]);  permute_624 = None
    view_801: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_140, [1, 512, 768]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    add_240: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_239, view_801);  add_239 = view_801 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_802: "f32[512, 384]" = torch.ops.aten.view.default(add_237, [512, 384]);  add_237 = None
    mm_142: "f32[512, 768]" = torch.ops.aten.mm.default(view_802, permute_626);  permute_626 = None
    permute_627: "f32[384, 512]" = torch.ops.aten.permute.default(view_802, [1, 0])
    mm_143: "f32[384, 768]" = torch.ops.aten.mm.default(permute_627, view_144);  permute_627 = view_144 = None
    permute_628: "f32[768, 384]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_186: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_802, [0], True);  view_802 = None
    view_803: "f32[384]" = torch.ops.aten.view.default(sum_186, [384]);  sum_186 = None
    permute_629: "f32[384, 768]" = torch.ops.aten.permute.default(permute_628, [1, 0]);  permute_628 = None
    view_804: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_142, [1, 512, 768]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    add_241: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_240, view_804);  add_240 = view_804 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_385: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_241, primals_104);  primals_104 = None
    mul_386: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_385, 768)
    sum_187: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_385, [2], True)
    mul_387: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_385, mul_33);  mul_385 = None
    sum_188: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_387, [2], True);  mul_387 = None
    mul_388: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_33, sum_188);  sum_188 = None
    sub_122: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_386, sum_187);  mul_386 = sum_187 = None
    sub_123: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_122, mul_388);  sub_122 = mul_388 = None
    mul_389: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_63, sub_123);  div_63 = sub_123 = None
    mul_390: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_241, mul_33);  mul_33 = None
    sum_189: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_390, [0, 1]);  mul_390 = None
    sum_190: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_241, [0, 1]);  add_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_25: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_41, torch.float32);  getitem_41 = None
    mul_391: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
    mul_392: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_389, mul_391);  mul_391 = None
    clone_84: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_392, memory_format = torch.contiguous_format);  mul_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_805: "f32[512, 768]" = torch.ops.aten.view.default(clone_84, [512, 768]);  clone_84 = None
    mm_144: "f32[512, 3072]" = torch.ops.aten.mm.default(view_805, permute_630);  permute_630 = None
    permute_631: "f32[768, 512]" = torch.ops.aten.permute.default(view_805, [1, 0])
    mm_145: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_631, view_142);  permute_631 = view_142 = None
    permute_632: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    sum_191: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_805, [0], True);  view_805 = None
    view_806: "f32[768]" = torch.ops.aten.view.default(sum_191, [768]);  sum_191 = None
    permute_633: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_632, [1, 0]);  permute_632 = None
    view_807: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_144, [1, 512, 3072]);  mm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_394: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_48, 0.5);  add_48 = None
    mul_395: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_141, view_141)
    mul_396: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_395, -0.5);  mul_395 = None
    exp_35: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_396);  mul_396 = None
    mul_397: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_35, 0.3989422804014327);  exp_35 = None
    mul_398: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_141, mul_397);  view_141 = mul_397 = None
    add_243: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_394, mul_398);  mul_394 = mul_398 = None
    mul_399: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_807, add_243);  view_807 = add_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_808: "f32[512, 3072]" = torch.ops.aten.view.default(mul_399, [512, 3072]);  mul_399 = None
    mm_146: "f32[512, 768]" = torch.ops.aten.mm.default(view_808, permute_634);  permute_634 = None
    permute_635: "f32[3072, 512]" = torch.ops.aten.permute.default(view_808, [1, 0])
    mm_147: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_635, view_140);  permute_635 = view_140 = None
    permute_636: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    sum_192: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_808, [0], True);  view_808 = None
    view_809: "f32[3072]" = torch.ops.aten.view.default(sum_192, [3072]);  sum_192 = None
    permute_637: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_636, [1, 0]);  permute_636 = None
    view_810: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_146, [1, 512, 768]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    add_244: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_389, view_810);  mul_389 = view_810 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_401: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_244, primals_98);  primals_98 = None
    mul_402: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_401, 768)
    sum_193: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_401, [2], True)
    mul_403: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_401, mul_28);  mul_401 = None
    sum_194: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_403, [2], True);  mul_403 = None
    mul_404: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_28, sum_194);  sum_194 = None
    sub_125: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_402, sum_193);  mul_402 = sum_193 = None
    sub_126: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_125, mul_404);  sub_125 = mul_404 = None
    mul_405: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_64, sub_126);  div_64 = sub_126 = None
    mul_406: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_244, mul_28);  mul_28 = None
    sum_195: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_406, [0, 1]);  mul_406 = None
    sum_196: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_244, [0, 1]);  add_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_26: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_37, torch.float32);  getitem_37 = None
    mul_407: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
    mul_408: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_405, mul_407);  mul_407 = None
    clone_85: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_408, memory_format = torch.contiguous_format);  mul_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_811: "f32[512, 768]" = torch.ops.aten.view.default(clone_85, [512, 768]);  clone_85 = None
    mm_148: "f32[512, 768]" = torch.ops.aten.mm.default(view_811, permute_638);  permute_638 = None
    permute_639: "f32[768, 512]" = torch.ops.aten.permute.default(view_811, [1, 0])
    mm_149: "f32[768, 768]" = torch.ops.aten.mm.default(permute_639, view_138);  permute_639 = view_138 = None
    permute_640: "f32[768, 768]" = torch.ops.aten.permute.default(mm_149, [1, 0]);  mm_149 = None
    sum_197: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_811, [0], True);  view_811 = None
    view_812: "f32[768]" = torch.ops.aten.view.default(sum_197, [768]);  sum_197 = None
    permute_641: "f32[768, 768]" = torch.ops.aten.permute.default(permute_640, [1, 0]);  permute_640 = None
    view_813: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_148, [1, 512, 768]);  mm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_814: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_813, [1, 512, 12, 64]);  view_813 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    slice_45: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_814, 2, 0, 6)
    slice_46: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_814, 2, 6, 12);  view_814 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_815: "f32[512, 384]" = torch.ops.aten.view.default(slice_46, [512, 384]);  slice_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_642: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(slice_45, [0, 2, 1, 3]);  slice_45 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_8 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_642, clone_default_24, clone_default_25, clone_default_26, None, alias_default_17, getitem_255, getitem_256, getitem_257, 0.1, [True, True, True, False], scale = 0.125);  permute_642 = clone_default_24 = clone_default_25 = clone_default_26 = alias_default_17 = getitem_255 = getitem_256 = getitem_257 = None
    getitem_258: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_8[0]
    getitem_259: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_8[1]
    getitem_260: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_8[2];  _scaled_dot_product_efficient_attention_backward_default_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    clone_87: "f32[512, 384]" = torch.ops.aten.clone.default(view_815, memory_format = torch.contiguous_format);  view_815 = None
    view_822: "f32[3072, 64, 1]" = torch.ops.aten.view.default(clone_87, [3072, 64, 1]);  clone_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    bmm_88: "f32[3072, 9, 1]" = torch.ops.aten.bmm.default(permute_648, view_822);  permute_648 = None
    bmm_89: "f32[3072, 64, 9]" = torch.ops.aten.bmm.default(view_822, permute_649);  view_822 = permute_649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    view_826: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(bmm_89, [1, 512, 384, 9]);  bmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    view_827: "f32[1, 512, 3456]" = torch.ops.aten.view.default(view_826, [1, 512, 3456]);  view_826 = None
    permute_650: "f32[1, 3456, 512]" = torch.ops.aten.permute.default(view_827, [0, 2, 1]);  view_827 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    view_828: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.view.default(permute_650, [1, 384, 9, 1, 512, 1]);  permute_650 = None
    permute_651: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.permute.default(view_828, [0, 1, 2, 4, 3, 5]);  view_828 = None
    _unsafe_index_put_8: "f32[1, 384, 520, 1]" = torch.ops.aten._unsafe_index_put.default(full_default_19, [None, None, unsqueeze_8, full_default_1], permute_651, True);  permute_651 = None
    constant_pad_nd_20: "f32[1, 384, 512, 1]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_8, [0, 0, -4, -4], 0.0);  _unsafe_index_put_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    squeeze_9: "f32[1, 384, 512]" = torch.ops.aten.squeeze.dim(constant_pad_nd_20, -1);  constant_pad_nd_20 = None
    permute_652: "f32[1, 512, 384]" = torch.ops.aten.permute.default(squeeze_9, [0, 2, 1]);  squeeze_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_830: "f32[512, 384]" = torch.ops.aten.view.default(permute_652, [512, 384]);  permute_652 = None
    mm_150: "f32[512, 768]" = torch.ops.aten.mm.default(view_830, permute_653);  permute_653 = None
    permute_654: "f32[384, 512]" = torch.ops.aten.permute.default(view_830, [1, 0])
    mm_151: "f32[384, 768]" = torch.ops.aten.mm.default(permute_654, view_108);  permute_654 = None
    permute_655: "f32[768, 384]" = torch.ops.aten.permute.default(mm_151, [1, 0]);  mm_151 = None
    sum_199: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_830, [0], True);  view_830 = None
    view_831: "f32[384]" = torch.ops.aten.view.default(sum_199, [384]);  sum_199 = None
    permute_656: "f32[384, 768]" = torch.ops.aten.permute.default(permute_655, [1, 0]);  permute_655 = None
    view_832: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_150, [1, 512, 768]);  mm_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    add_247: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_405, view_832);  mul_405 = view_832 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    mul_413: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(bmm_88, alias_43);  bmm_88 = None
    sum_200: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_413, [1], True)
    mul_414: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(alias_43, sum_200);  alias_43 = sum_200 = None
    sub_128: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(mul_413, mul_414);  mul_413 = mul_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_833: "f32[1, 512, 54]" = torch.ops.aten.view.default(sub_128, [1, 512, 54]);  sub_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    sum_201: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_833, [0, 1], True)
    view_834: "f32[54]" = torch.ops.aten.view.default(sum_201, [54]);  sum_201 = None
    view_835: "f32[512, 54]" = torch.ops.aten.view.default(view_833, [512, 54]);  view_833 = None
    permute_657: "f32[54, 512]" = torch.ops.aten.permute.default(view_835, [1, 0]);  view_835 = None
    mm_152: "f32[54, 384]" = torch.ops.aten.mm.default(permute_657, view_117);  view_117 = None
    permute_658: "f32[384, 54]" = torch.ops.aten.permute.default(mm_152, [1, 0]);  mm_152 = None
    mm_153: "f32[384, 512]" = torch.ops.aten.mm.default(permute_66, permute_657);  permute_66 = permute_657 = None
    permute_660: "f32[512, 384]" = torch.ops.aten.permute.default(mm_153, [1, 0]);  mm_153 = None
    view_836: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_660, [1, 512, 384]);  permute_660 = None
    permute_661: "f32[54, 384]" = torch.ops.aten.permute.default(permute_658, [1, 0]);  permute_658 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    mul_415: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_836, permute_65);  permute_65 = None
    mul_416: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_836, view_109);  view_836 = view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_662: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_260, [0, 2, 1, 3]);  getitem_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_88: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_662, memory_format = torch.contiguous_format);  permute_662 = None
    view_837: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_88, [1, 512, 384]);  clone_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_663: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_259, [0, 2, 1, 3]);  getitem_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_838: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_663, [1, 512, 384]);  permute_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_664: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_258, [0, 2, 1, 3]);  getitem_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_89: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_664, memory_format = torch.contiguous_format);  permute_664 = None
    view_839: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_89, [1, 512, 384]);  clone_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    add_248: "f32[1, 512, 384]" = torch.ops.aten.add.Tensor(mul_415, view_839);  mul_415 = view_839 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    permute_665: "f32[1, 384, 512]" = torch.ops.aten.permute.default(mul_416, [0, 2, 1]);  mul_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    sum_202: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_665, [0, 2], True)
    view_840: "f32[384, 1]" = torch.ops.aten.view.default(sum_202, [384, 1]);  sum_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(permute_665, convolution_6, primals_91, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_665 = convolution_6 = primals_91 = None
    getitem_174: "f32[1, 768, 512]" = convolution_backward_16[0]
    getitem_175: "f32[384, 768, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(getitem_174, permute_60, primals_90, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_174 = permute_60 = primals_90 = None
    getitem_177: "f32[1, 768, 512]" = convolution_backward_17[0]
    getitem_178: "f32[768, 1, 9]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_666: "f32[1, 512, 768]" = torch.ops.aten.permute.default(getitem_177, [0, 2, 1]);  getitem_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    add_249: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_247, permute_666);  add_247 = permute_666 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_841: "f32[512, 384]" = torch.ops.aten.view.default(view_837, [512, 384]);  view_837 = None
    mm_154: "f32[512, 768]" = torch.ops.aten.mm.default(view_841, permute_667);  permute_667 = None
    permute_668: "f32[384, 512]" = torch.ops.aten.permute.default(view_841, [1, 0])
    mm_155: "f32[384, 768]" = torch.ops.aten.mm.default(permute_668, view_108);  permute_668 = None
    permute_669: "f32[768, 384]" = torch.ops.aten.permute.default(mm_155, [1, 0]);  mm_155 = None
    sum_203: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_841, [0], True);  view_841 = None
    view_842: "f32[384]" = torch.ops.aten.view.default(sum_203, [384]);  sum_203 = None
    permute_670: "f32[384, 768]" = torch.ops.aten.permute.default(permute_669, [1, 0]);  permute_669 = None
    view_843: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_154, [1, 512, 768]);  mm_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    add_250: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_249, view_843);  add_249 = view_843 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_844: "f32[512, 384]" = torch.ops.aten.view.default(view_838, [512, 384]);  view_838 = None
    mm_156: "f32[512, 768]" = torch.ops.aten.mm.default(view_844, permute_671);  permute_671 = None
    permute_672: "f32[384, 512]" = torch.ops.aten.permute.default(view_844, [1, 0])
    mm_157: "f32[384, 768]" = torch.ops.aten.mm.default(permute_672, view_108);  permute_672 = None
    permute_673: "f32[768, 384]" = torch.ops.aten.permute.default(mm_157, [1, 0]);  mm_157 = None
    sum_204: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_844, [0], True);  view_844 = None
    view_845: "f32[384]" = torch.ops.aten.view.default(sum_204, [384]);  sum_204 = None
    permute_674: "f32[384, 768]" = torch.ops.aten.permute.default(permute_673, [1, 0]);  permute_673 = None
    view_846: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_156, [1, 512, 768]);  mm_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    add_251: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_250, view_846);  add_250 = view_846 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_847: "f32[512, 384]" = torch.ops.aten.view.default(add_248, [512, 384]);  add_248 = None
    mm_158: "f32[512, 768]" = torch.ops.aten.mm.default(view_847, permute_675);  permute_675 = None
    permute_676: "f32[384, 512]" = torch.ops.aten.permute.default(view_847, [1, 0])
    mm_159: "f32[384, 768]" = torch.ops.aten.mm.default(permute_676, view_108);  permute_676 = view_108 = None
    permute_677: "f32[768, 384]" = torch.ops.aten.permute.default(mm_159, [1, 0]);  mm_159 = None
    sum_205: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_847, [0], True);  view_847 = None
    view_848: "f32[384]" = torch.ops.aten.view.default(sum_205, [384]);  sum_205 = None
    permute_678: "f32[384, 768]" = torch.ops.aten.permute.default(permute_677, [1, 0]);  permute_677 = None
    view_849: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_158, [1, 512, 768]);  mm_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    add_252: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_251, view_849);  add_251 = view_849 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_418: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_252, primals_82);  primals_82 = None
    mul_419: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_418, 768)
    sum_206: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_418, [2], True)
    mul_420: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_418, mul_25);  mul_418 = None
    sum_207: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_420, [2], True);  mul_420 = None
    mul_421: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_25, sum_207);  sum_207 = None
    sub_130: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_419, sum_206);  mul_419 = sum_206 = None
    sub_131: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_130, mul_421);  sub_130 = mul_421 = None
    mul_422: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_66, sub_131);  div_66 = sub_131 = None
    mul_423: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_252, mul_25);  mul_25 = None
    sum_208: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_423, [0, 1]);  mul_423 = None
    sum_209: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_252, [0, 1]);  add_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_28: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_31, torch.float32);  getitem_31 = None
    mul_424: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.1111111111111112);  convert_element_type_28 = None
    mul_425: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_422, mul_424);  mul_424 = None
    clone_90: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_425, memory_format = torch.contiguous_format);  mul_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_850: "f32[512, 768]" = torch.ops.aten.view.default(clone_90, [512, 768]);  clone_90 = None
    mm_160: "f32[512, 3072]" = torch.ops.aten.mm.default(view_850, permute_679);  permute_679 = None
    permute_680: "f32[768, 512]" = torch.ops.aten.permute.default(view_850, [1, 0])
    mm_161: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_680, view_106);  permute_680 = view_106 = None
    permute_681: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_161, [1, 0]);  mm_161 = None
    sum_210: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_850, [0], True);  view_850 = None
    view_851: "f32[768]" = torch.ops.aten.view.default(sum_210, [768]);  sum_210 = None
    permute_682: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_681, [1, 0]);  permute_681 = None
    view_852: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_160, [1, 512, 3072]);  mm_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_427: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_36, 0.5);  add_36 = None
    mul_428: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_105, view_105)
    mul_429: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_428, -0.5);  mul_428 = None
    exp_36: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_429);  mul_429 = None
    mul_430: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_36, 0.3989422804014327);  exp_36 = None
    mul_431: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_105, mul_430);  view_105 = mul_430 = None
    add_254: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_427, mul_431);  mul_427 = mul_431 = None
    mul_432: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_852, add_254);  view_852 = add_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_853: "f32[512, 3072]" = torch.ops.aten.view.default(mul_432, [512, 3072]);  mul_432 = None
    mm_162: "f32[512, 768]" = torch.ops.aten.mm.default(view_853, permute_683);  permute_683 = None
    permute_684: "f32[3072, 512]" = torch.ops.aten.permute.default(view_853, [1, 0])
    mm_163: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_684, view_104);  permute_684 = view_104 = None
    permute_685: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_163, [1, 0]);  mm_163 = None
    sum_211: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_853, [0], True);  view_853 = None
    view_854: "f32[3072]" = torch.ops.aten.view.default(sum_211, [3072]);  sum_211 = None
    permute_686: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_685, [1, 0]);  permute_685 = None
    view_855: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_162, [1, 512, 768]);  mm_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    add_255: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_422, view_855);  mul_422 = view_855 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_434: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_255, primals_76);  primals_76 = None
    mul_435: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_434, 768)
    sum_212: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_434, [2], True)
    mul_436: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_434, mul_20);  mul_434 = None
    sum_213: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_436, [2], True);  mul_436 = None
    mul_437: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_20, sum_213);  sum_213 = None
    sub_133: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_435, sum_212);  mul_435 = sum_212 = None
    sub_134: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_133, mul_437);  sub_133 = mul_437 = None
    mul_438: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_67, sub_134);  div_67 = sub_134 = None
    mul_439: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_255, mul_20);  mul_20 = None
    sum_214: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_439, [0, 1]);  mul_439 = None
    sum_215: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_255, [0, 1]);  add_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_29: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_27, torch.float32);  getitem_27 = None
    mul_440: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
    mul_441: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_438, mul_440);  mul_440 = None
    clone_91: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_441, memory_format = torch.contiguous_format);  mul_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_856: "f32[512, 768]" = torch.ops.aten.view.default(clone_91, [512, 768]);  clone_91 = None
    mm_164: "f32[512, 768]" = torch.ops.aten.mm.default(view_856, permute_687);  permute_687 = None
    permute_688: "f32[768, 512]" = torch.ops.aten.permute.default(view_856, [1, 0])
    mm_165: "f32[768, 768]" = torch.ops.aten.mm.default(permute_688, view_102);  permute_688 = view_102 = None
    permute_689: "f32[768, 768]" = torch.ops.aten.permute.default(mm_165, [1, 0]);  mm_165 = None
    sum_216: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_856, [0], True);  view_856 = None
    view_857: "f32[768]" = torch.ops.aten.view.default(sum_216, [768]);  sum_216 = None
    permute_690: "f32[768, 768]" = torch.ops.aten.permute.default(permute_689, [1, 0]);  permute_689 = None
    view_858: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_164, [1, 512, 768]);  mm_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_859: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_858, [1, 512, 12, 64]);  view_858 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    slice_47: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_859, 2, 0, 6)
    slice_48: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_859, 2, 6, 12);  view_859 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_860: "f32[512, 384]" = torch.ops.aten.view.default(slice_48, [512, 384]);  slice_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_691: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(slice_47, [0, 2, 1, 3]);  slice_47 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_9 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_691, clone_default_27, clone_default_28, clone_default_29, None, alias_default_19, getitem_262, getitem_263, getitem_264, 0.1, [True, True, True, False], scale = 0.125);  permute_691 = clone_default_27 = clone_default_28 = clone_default_29 = alias_default_19 = getitem_262 = getitem_263 = getitem_264 = None
    getitem_265: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_9[0]
    getitem_266: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_9[1]
    getitem_267: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_9[2];  _scaled_dot_product_efficient_attention_backward_default_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    clone_93: "f32[512, 384]" = torch.ops.aten.clone.default(view_860, memory_format = torch.contiguous_format);  view_860 = None
    view_867: "f32[3072, 64, 1]" = torch.ops.aten.view.default(clone_93, [3072, 64, 1]);  clone_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    bmm_94: "f32[3072, 9, 1]" = torch.ops.aten.bmm.default(permute_697, view_867);  permute_697 = None
    bmm_95: "f32[3072, 64, 9]" = torch.ops.aten.bmm.default(view_867, permute_698);  view_867 = permute_698 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    view_871: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(bmm_95, [1, 512, 384, 9]);  bmm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    view_872: "f32[1, 512, 3456]" = torch.ops.aten.view.default(view_871, [1, 512, 3456]);  view_871 = None
    permute_699: "f32[1, 3456, 512]" = torch.ops.aten.permute.default(view_872, [0, 2, 1]);  view_872 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    view_873: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.view.default(permute_699, [1, 384, 9, 1, 512, 1]);  permute_699 = None
    permute_700: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.permute.default(view_873, [0, 1, 2, 4, 3, 5]);  view_873 = None
    _unsafe_index_put_9: "f32[1, 384, 520, 1]" = torch.ops.aten._unsafe_index_put.default(full_default_19, [None, None, unsqueeze_8, full_default_1], permute_700, True);  permute_700 = None
    constant_pad_nd_21: "f32[1, 384, 512, 1]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_9, [0, 0, -4, -4], 0.0);  _unsafe_index_put_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    squeeze_10: "f32[1, 384, 512]" = torch.ops.aten.squeeze.dim(constant_pad_nd_21, -1);  constant_pad_nd_21 = None
    permute_701: "f32[1, 512, 384]" = torch.ops.aten.permute.default(squeeze_10, [0, 2, 1]);  squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_875: "f32[512, 384]" = torch.ops.aten.view.default(permute_701, [512, 384]);  permute_701 = None
    mm_166: "f32[512, 768]" = torch.ops.aten.mm.default(view_875, permute_702);  permute_702 = None
    permute_703: "f32[384, 512]" = torch.ops.aten.permute.default(view_875, [1, 0])
    mm_167: "f32[384, 768]" = torch.ops.aten.mm.default(permute_703, view_72);  permute_703 = None
    permute_704: "f32[768, 384]" = torch.ops.aten.permute.default(mm_167, [1, 0]);  mm_167 = None
    sum_218: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_875, [0], True);  view_875 = None
    view_876: "f32[384]" = torch.ops.aten.view.default(sum_218, [384]);  sum_218 = None
    permute_705: "f32[384, 768]" = torch.ops.aten.permute.default(permute_704, [1, 0]);  permute_704 = None
    view_877: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_166, [1, 512, 768]);  mm_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    add_258: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_438, view_877);  mul_438 = view_877 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    mul_446: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(bmm_94, alias_45);  bmm_94 = None
    sum_219: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_446, [1], True)
    mul_447: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(alias_45, sum_219);  alias_45 = sum_219 = None
    sub_136: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(mul_446, mul_447);  mul_446 = mul_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_878: "f32[1, 512, 54]" = torch.ops.aten.view.default(sub_136, [1, 512, 54]);  sub_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    sum_220: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_878, [0, 1], True)
    view_879: "f32[54]" = torch.ops.aten.view.default(sum_220, [54]);  sum_220 = None
    view_880: "f32[512, 54]" = torch.ops.aten.view.default(view_878, [512, 54]);  view_878 = None
    permute_706: "f32[54, 512]" = torch.ops.aten.permute.default(view_880, [1, 0]);  view_880 = None
    mm_168: "f32[54, 384]" = torch.ops.aten.mm.default(permute_706, view_81);  view_81 = None
    permute_707: "f32[384, 54]" = torch.ops.aten.permute.default(mm_168, [1, 0]);  mm_168 = None
    mm_169: "f32[384, 512]" = torch.ops.aten.mm.default(permute_47, permute_706);  permute_47 = permute_706 = None
    permute_709: "f32[512, 384]" = torch.ops.aten.permute.default(mm_169, [1, 0]);  mm_169 = None
    view_881: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_709, [1, 512, 384]);  permute_709 = None
    permute_710: "f32[54, 384]" = torch.ops.aten.permute.default(permute_707, [1, 0]);  permute_707 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    mul_448: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_881, permute_46);  permute_46 = None
    mul_449: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_881, view_73);  view_881 = view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_711: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_267, [0, 2, 1, 3]);  getitem_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_94: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_711, memory_format = torch.contiguous_format);  permute_711 = None
    view_882: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_94, [1, 512, 384]);  clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_712: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_266, [0, 2, 1, 3]);  getitem_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_883: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_712, [1, 512, 384]);  permute_712 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_713: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_265, [0, 2, 1, 3]);  getitem_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_95: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_713, memory_format = torch.contiguous_format);  permute_713 = None
    view_884: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_95, [1, 512, 384]);  clone_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    add_259: "f32[1, 512, 384]" = torch.ops.aten.add.Tensor(mul_448, view_884);  mul_448 = view_884 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    permute_714: "f32[1, 384, 512]" = torch.ops.aten.permute.default(mul_449, [0, 2, 1]);  mul_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    sum_221: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_714, [0, 2], True)
    view_885: "f32[384, 1]" = torch.ops.aten.view.default(sum_221, [384, 1]);  sum_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(permute_714, convolution_4, primals_69, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_714 = convolution_4 = primals_69 = None
    getitem_180: "f32[1, 768, 512]" = convolution_backward_18[0]
    getitem_181: "f32[384, 768, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(getitem_180, permute_41, primals_68, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_180 = permute_41 = primals_68 = None
    getitem_183: "f32[1, 768, 512]" = convolution_backward_19[0]
    getitem_184: "f32[768, 1, 9]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_715: "f32[1, 512, 768]" = torch.ops.aten.permute.default(getitem_183, [0, 2, 1]);  getitem_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    add_260: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_258, permute_715);  add_258 = permute_715 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_886: "f32[512, 384]" = torch.ops.aten.view.default(view_882, [512, 384]);  view_882 = None
    mm_170: "f32[512, 768]" = torch.ops.aten.mm.default(view_886, permute_716);  permute_716 = None
    permute_717: "f32[384, 512]" = torch.ops.aten.permute.default(view_886, [1, 0])
    mm_171: "f32[384, 768]" = torch.ops.aten.mm.default(permute_717, view_72);  permute_717 = None
    permute_718: "f32[768, 384]" = torch.ops.aten.permute.default(mm_171, [1, 0]);  mm_171 = None
    sum_222: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_886, [0], True);  view_886 = None
    view_887: "f32[384]" = torch.ops.aten.view.default(sum_222, [384]);  sum_222 = None
    permute_719: "f32[384, 768]" = torch.ops.aten.permute.default(permute_718, [1, 0]);  permute_718 = None
    view_888: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_170, [1, 512, 768]);  mm_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    add_261: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_260, view_888);  add_260 = view_888 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_889: "f32[512, 384]" = torch.ops.aten.view.default(view_883, [512, 384]);  view_883 = None
    mm_172: "f32[512, 768]" = torch.ops.aten.mm.default(view_889, permute_720);  permute_720 = None
    permute_721: "f32[384, 512]" = torch.ops.aten.permute.default(view_889, [1, 0])
    mm_173: "f32[384, 768]" = torch.ops.aten.mm.default(permute_721, view_72);  permute_721 = None
    permute_722: "f32[768, 384]" = torch.ops.aten.permute.default(mm_173, [1, 0]);  mm_173 = None
    sum_223: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_889, [0], True);  view_889 = None
    view_890: "f32[384]" = torch.ops.aten.view.default(sum_223, [384]);  sum_223 = None
    permute_723: "f32[384, 768]" = torch.ops.aten.permute.default(permute_722, [1, 0]);  permute_722 = None
    view_891: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_172, [1, 512, 768]);  mm_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    add_262: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_261, view_891);  add_261 = view_891 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_892: "f32[512, 384]" = torch.ops.aten.view.default(add_259, [512, 384]);  add_259 = None
    mm_174: "f32[512, 768]" = torch.ops.aten.mm.default(view_892, permute_724);  permute_724 = None
    permute_725: "f32[384, 512]" = torch.ops.aten.permute.default(view_892, [1, 0])
    mm_175: "f32[384, 768]" = torch.ops.aten.mm.default(permute_725, view_72);  permute_725 = view_72 = None
    permute_726: "f32[768, 384]" = torch.ops.aten.permute.default(mm_175, [1, 0]);  mm_175 = None
    sum_224: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_892, [0], True);  view_892 = None
    view_893: "f32[384]" = torch.ops.aten.view.default(sum_224, [384]);  sum_224 = None
    permute_727: "f32[384, 768]" = torch.ops.aten.permute.default(permute_726, [1, 0]);  permute_726 = None
    view_894: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_174, [1, 512, 768]);  mm_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    add_263: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_262, view_894);  add_262 = view_894 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_451: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_263, primals_60);  primals_60 = None
    mul_452: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_451, 768)
    sum_225: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_451, [2], True)
    mul_453: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_451, mul_17);  mul_451 = None
    sum_226: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_453, [2], True);  mul_453 = None
    mul_454: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_17, sum_226);  sum_226 = None
    sub_138: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_452, sum_225);  mul_452 = sum_225 = None
    sub_139: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_138, mul_454);  sub_138 = mul_454 = None
    mul_455: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_69, sub_139);  div_69 = sub_139 = None
    mul_456: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_263, mul_17);  mul_17 = None
    sum_227: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_456, [0, 1]);  mul_456 = None
    sum_228: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_263, [0, 1]);  add_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_31: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_21, torch.float32);  getitem_21 = None
    mul_457: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.1111111111111112);  convert_element_type_31 = None
    mul_458: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_455, mul_457);  mul_457 = None
    clone_96: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_458, memory_format = torch.contiguous_format);  mul_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_895: "f32[512, 768]" = torch.ops.aten.view.default(clone_96, [512, 768]);  clone_96 = None
    mm_176: "f32[512, 3072]" = torch.ops.aten.mm.default(view_895, permute_728);  permute_728 = None
    permute_729: "f32[768, 512]" = torch.ops.aten.permute.default(view_895, [1, 0])
    mm_177: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_729, view_70);  permute_729 = view_70 = None
    permute_730: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_177, [1, 0]);  mm_177 = None
    sum_229: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_895, [0], True);  view_895 = None
    view_896: "f32[768]" = torch.ops.aten.view.default(sum_229, [768]);  sum_229 = None
    permute_731: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_730, [1, 0]);  permute_730 = None
    view_897: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_176, [1, 512, 3072]);  mm_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_460: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_24, 0.5);  add_24 = None
    mul_461: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_69, view_69)
    mul_462: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_461, -0.5);  mul_461 = None
    exp_37: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_462);  mul_462 = None
    mul_463: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_37, 0.3989422804014327);  exp_37 = None
    mul_464: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_69, mul_463);  view_69 = mul_463 = None
    add_265: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_460, mul_464);  mul_460 = mul_464 = None
    mul_465: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_897, add_265);  view_897 = add_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_898: "f32[512, 3072]" = torch.ops.aten.view.default(mul_465, [512, 3072]);  mul_465 = None
    mm_178: "f32[512, 768]" = torch.ops.aten.mm.default(view_898, permute_732);  permute_732 = None
    permute_733: "f32[3072, 512]" = torch.ops.aten.permute.default(view_898, [1, 0])
    mm_179: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_733, view_68);  permute_733 = view_68 = None
    permute_734: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_179, [1, 0]);  mm_179 = None
    sum_230: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_898, [0], True);  view_898 = None
    view_899: "f32[3072]" = torch.ops.aten.view.default(sum_230, [3072]);  sum_230 = None
    permute_735: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_734, [1, 0]);  permute_734 = None
    view_900: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_178, [1, 512, 768]);  mm_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    add_266: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_455, view_900);  mul_455 = view_900 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_467: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_266, primals_54);  primals_54 = None
    mul_468: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_467, 768)
    sum_231: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_467, [2], True)
    mul_469: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_467, mul_12);  mul_467 = None
    sum_232: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_469, [2], True);  mul_469 = None
    mul_470: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_12, sum_232);  sum_232 = None
    sub_141: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_468, sum_231);  mul_468 = sum_231 = None
    sub_142: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_141, mul_470);  sub_141 = mul_470 = None
    mul_471: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_70, sub_142);  div_70 = sub_142 = None
    mul_472: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_266, mul_12);  mul_12 = None
    sum_233: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_472, [0, 1]);  mul_472 = None
    sum_234: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_266, [0, 1]);  add_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_32: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_17, torch.float32);  getitem_17 = None
    mul_473: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
    mul_474: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_471, mul_473);  mul_473 = None
    clone_97: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_474, memory_format = torch.contiguous_format);  mul_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_901: "f32[512, 768]" = torch.ops.aten.view.default(clone_97, [512, 768]);  clone_97 = None
    mm_180: "f32[512, 768]" = torch.ops.aten.mm.default(view_901, permute_736);  permute_736 = None
    permute_737: "f32[768, 512]" = torch.ops.aten.permute.default(view_901, [1, 0])
    mm_181: "f32[768, 768]" = torch.ops.aten.mm.default(permute_737, view_66);  permute_737 = view_66 = None
    permute_738: "f32[768, 768]" = torch.ops.aten.permute.default(mm_181, [1, 0]);  mm_181 = None
    sum_235: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_901, [0], True);  view_901 = None
    view_902: "f32[768]" = torch.ops.aten.view.default(sum_235, [768]);  sum_235 = None
    permute_739: "f32[768, 768]" = torch.ops.aten.permute.default(permute_738, [1, 0]);  permute_738 = None
    view_903: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_180, [1, 512, 768]);  mm_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_904: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_903, [1, 512, 12, 64]);  view_903 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    slice_49: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_904, 2, 0, 6)
    slice_50: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_904, 2, 6, 12);  view_904 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_905: "f32[512, 384]" = torch.ops.aten.view.default(slice_50, [512, 384]);  slice_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_740: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(slice_49, [0, 2, 1, 3]);  slice_49 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_10 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_740, clone_default_30, clone_default_31, clone_default_32, None, alias_default_21, getitem_269, getitem_270, getitem_271, 0.1, [True, True, True, False], scale = 0.125);  permute_740 = clone_default_30 = clone_default_31 = clone_default_32 = alias_default_21 = getitem_269 = getitem_270 = getitem_271 = None
    getitem_272: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_10[0]
    getitem_273: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_10[1]
    getitem_274: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_10[2];  _scaled_dot_product_efficient_attention_backward_default_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    clone_99: "f32[512, 384]" = torch.ops.aten.clone.default(view_905, memory_format = torch.contiguous_format);  view_905 = None
    view_912: "f32[3072, 64, 1]" = torch.ops.aten.view.default(clone_99, [3072, 64, 1]);  clone_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    bmm_100: "f32[3072, 9, 1]" = torch.ops.aten.bmm.default(permute_746, view_912);  permute_746 = None
    bmm_101: "f32[3072, 64, 9]" = torch.ops.aten.bmm.default(view_912, permute_747);  view_912 = permute_747 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    view_916: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(bmm_101, [1, 512, 384, 9]);  bmm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    view_917: "f32[1, 512, 3456]" = torch.ops.aten.view.default(view_916, [1, 512, 3456]);  view_916 = None
    permute_748: "f32[1, 3456, 512]" = torch.ops.aten.permute.default(view_917, [0, 2, 1]);  view_917 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    view_918: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.view.default(permute_748, [1, 384, 9, 1, 512, 1]);  permute_748 = None
    permute_749: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.permute.default(view_918, [0, 1, 2, 4, 3, 5]);  view_918 = None
    _unsafe_index_put_10: "f32[1, 384, 520, 1]" = torch.ops.aten._unsafe_index_put.default(full_default_19, [None, None, unsqueeze_8, full_default_1], permute_749, True);  permute_749 = None
    constant_pad_nd_22: "f32[1, 384, 512, 1]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_10, [0, 0, -4, -4], 0.0);  _unsafe_index_put_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    squeeze_11: "f32[1, 384, 512]" = torch.ops.aten.squeeze.dim(constant_pad_nd_22, -1);  constant_pad_nd_22 = None
    permute_750: "f32[1, 512, 384]" = torch.ops.aten.permute.default(squeeze_11, [0, 2, 1]);  squeeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_920: "f32[512, 384]" = torch.ops.aten.view.default(permute_750, [512, 384]);  permute_750 = None
    mm_182: "f32[512, 768]" = torch.ops.aten.mm.default(view_920, permute_751);  permute_751 = None
    permute_752: "f32[384, 512]" = torch.ops.aten.permute.default(view_920, [1, 0])
    mm_183: "f32[384, 768]" = torch.ops.aten.mm.default(permute_752, view_36);  permute_752 = None
    permute_753: "f32[768, 384]" = torch.ops.aten.permute.default(mm_183, [1, 0]);  mm_183 = None
    sum_237: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_920, [0], True);  view_920 = None
    view_921: "f32[384]" = torch.ops.aten.view.default(sum_237, [384]);  sum_237 = None
    permute_754: "f32[384, 768]" = torch.ops.aten.permute.default(permute_753, [1, 0]);  permute_753 = None
    view_922: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_182, [1, 512, 768]);  mm_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    add_269: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_471, view_922);  mul_471 = view_922 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    mul_479: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(bmm_100, alias_47);  bmm_100 = None
    sum_238: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_479, [1], True)
    mul_480: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(alias_47, sum_238);  alias_47 = sum_238 = None
    sub_144: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(mul_479, mul_480);  mul_479 = mul_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_923: "f32[1, 512, 54]" = torch.ops.aten.view.default(sub_144, [1, 512, 54]);  sub_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    sum_239: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_923, [0, 1], True)
    view_924: "f32[54]" = torch.ops.aten.view.default(sum_239, [54]);  sum_239 = None
    view_925: "f32[512, 54]" = torch.ops.aten.view.default(view_923, [512, 54]);  view_923 = None
    permute_755: "f32[54, 512]" = torch.ops.aten.permute.default(view_925, [1, 0]);  view_925 = None
    mm_184: "f32[54, 384]" = torch.ops.aten.mm.default(permute_755, view_45);  view_45 = None
    permute_756: "f32[384, 54]" = torch.ops.aten.permute.default(mm_184, [1, 0]);  mm_184 = None
    mm_185: "f32[384, 512]" = torch.ops.aten.mm.default(permute_28, permute_755);  permute_28 = permute_755 = None
    permute_758: "f32[512, 384]" = torch.ops.aten.permute.default(mm_185, [1, 0]);  mm_185 = None
    view_926: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_758, [1, 512, 384]);  permute_758 = None
    permute_759: "f32[54, 384]" = torch.ops.aten.permute.default(permute_756, [1, 0]);  permute_756 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    mul_481: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_926, permute_27);  permute_27 = None
    mul_482: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_926, view_37);  view_926 = view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_760: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_274, [0, 2, 1, 3]);  getitem_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_100: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_760, memory_format = torch.contiguous_format);  permute_760 = None
    view_927: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_100, [1, 512, 384]);  clone_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_761: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_273, [0, 2, 1, 3]);  getitem_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_928: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_761, [1, 512, 384]);  permute_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_762: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_272, [0, 2, 1, 3]);  getitem_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_101: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_762, memory_format = torch.contiguous_format);  permute_762 = None
    view_929: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_101, [1, 512, 384]);  clone_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    add_270: "f32[1, 512, 384]" = torch.ops.aten.add.Tensor(mul_481, view_929);  mul_481 = view_929 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    permute_763: "f32[1, 384, 512]" = torch.ops.aten.permute.default(mul_482, [0, 2, 1]);  mul_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    sum_240: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_763, [0, 2], True)
    view_930: "f32[384, 1]" = torch.ops.aten.view.default(sum_240, [384, 1]);  sum_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(permute_763, convolution_2, primals_47, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_763 = convolution_2 = primals_47 = None
    getitem_186: "f32[1, 768, 512]" = convolution_backward_20[0]
    getitem_187: "f32[384, 768, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(getitem_186, permute_22, primals_46, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_186 = permute_22 = primals_46 = None
    getitem_189: "f32[1, 768, 512]" = convolution_backward_21[0]
    getitem_190: "f32[768, 1, 9]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_764: "f32[1, 512, 768]" = torch.ops.aten.permute.default(getitem_189, [0, 2, 1]);  getitem_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    add_271: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_269, permute_764);  add_269 = permute_764 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_931: "f32[512, 384]" = torch.ops.aten.view.default(view_927, [512, 384]);  view_927 = None
    mm_186: "f32[512, 768]" = torch.ops.aten.mm.default(view_931, permute_765);  permute_765 = None
    permute_766: "f32[384, 512]" = torch.ops.aten.permute.default(view_931, [1, 0])
    mm_187: "f32[384, 768]" = torch.ops.aten.mm.default(permute_766, view_36);  permute_766 = None
    permute_767: "f32[768, 384]" = torch.ops.aten.permute.default(mm_187, [1, 0]);  mm_187 = None
    sum_241: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_931, [0], True);  view_931 = None
    view_932: "f32[384]" = torch.ops.aten.view.default(sum_241, [384]);  sum_241 = None
    permute_768: "f32[384, 768]" = torch.ops.aten.permute.default(permute_767, [1, 0]);  permute_767 = None
    view_933: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_186, [1, 512, 768]);  mm_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    add_272: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_271, view_933);  add_271 = view_933 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_934: "f32[512, 384]" = torch.ops.aten.view.default(view_928, [512, 384]);  view_928 = None
    mm_188: "f32[512, 768]" = torch.ops.aten.mm.default(view_934, permute_769);  permute_769 = None
    permute_770: "f32[384, 512]" = torch.ops.aten.permute.default(view_934, [1, 0])
    mm_189: "f32[384, 768]" = torch.ops.aten.mm.default(permute_770, view_36);  permute_770 = None
    permute_771: "f32[768, 384]" = torch.ops.aten.permute.default(mm_189, [1, 0]);  mm_189 = None
    sum_242: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_934, [0], True);  view_934 = None
    view_935: "f32[384]" = torch.ops.aten.view.default(sum_242, [384]);  sum_242 = None
    permute_772: "f32[384, 768]" = torch.ops.aten.permute.default(permute_771, [1, 0]);  permute_771 = None
    view_936: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_188, [1, 512, 768]);  mm_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    add_273: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_272, view_936);  add_272 = view_936 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_937: "f32[512, 384]" = torch.ops.aten.view.default(add_270, [512, 384]);  add_270 = None
    mm_190: "f32[512, 768]" = torch.ops.aten.mm.default(view_937, permute_773);  permute_773 = None
    permute_774: "f32[384, 512]" = torch.ops.aten.permute.default(view_937, [1, 0])
    mm_191: "f32[384, 768]" = torch.ops.aten.mm.default(permute_774, view_36);  permute_774 = view_36 = None
    permute_775: "f32[768, 384]" = torch.ops.aten.permute.default(mm_191, [1, 0]);  mm_191 = None
    sum_243: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_937, [0], True);  view_937 = None
    view_938: "f32[384]" = torch.ops.aten.view.default(sum_243, [384]);  sum_243 = None
    permute_776: "f32[384, 768]" = torch.ops.aten.permute.default(permute_775, [1, 0]);  permute_775 = None
    view_939: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_190, [1, 512, 768]);  mm_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    add_274: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_273, view_939);  add_273 = view_939 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_484: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_274, primals_38);  primals_38 = None
    mul_485: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_484, 768)
    sum_244: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_484, [2], True)
    mul_486: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_484, mul_9);  mul_484 = None
    sum_245: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_486, [2], True);  mul_486 = None
    mul_487: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_9, sum_245);  sum_245 = None
    sub_146: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_485, sum_244);  mul_485 = sum_244 = None
    sub_147: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_146, mul_487);  sub_146 = mul_487 = None
    mul_488: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_72, sub_147);  div_72 = sub_147 = None
    mul_489: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_274, mul_9);  mul_9 = None
    sum_246: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_489, [0, 1]);  mul_489 = None
    sum_247: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_274, [0, 1]);  add_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_34: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_11, torch.float32);  getitem_11 = None
    mul_490: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1.1111111111111112);  convert_element_type_34 = None
    mul_491: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_488, mul_490);  mul_490 = None
    clone_102: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_491, memory_format = torch.contiguous_format);  mul_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_940: "f32[512, 768]" = torch.ops.aten.view.default(clone_102, [512, 768]);  clone_102 = None
    mm_192: "f32[512, 3072]" = torch.ops.aten.mm.default(view_940, permute_777);  permute_777 = None
    permute_778: "f32[768, 512]" = torch.ops.aten.permute.default(view_940, [1, 0])
    mm_193: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_778, view_34);  permute_778 = view_34 = None
    permute_779: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_193, [1, 0]);  mm_193 = None
    sum_248: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_940, [0], True);  view_940 = None
    view_941: "f32[768]" = torch.ops.aten.view.default(sum_248, [768]);  sum_248 = None
    permute_780: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_779, [1, 0]);  permute_779 = None
    view_942: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_192, [1, 512, 3072]);  mm_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_493: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_12, 0.5);  add_12 = None
    mul_494: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_33, view_33)
    mul_495: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_494, -0.5);  mul_494 = None
    exp_38: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_495);  mul_495 = None
    mul_496: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_38, 0.3989422804014327);  exp_38 = None
    mul_497: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_33, mul_496);  view_33 = mul_496 = None
    add_276: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_493, mul_497);  mul_493 = mul_497 = None
    mul_498: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_942, add_276);  view_942 = add_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_943: "f32[512, 3072]" = torch.ops.aten.view.default(mul_498, [512, 3072]);  mul_498 = None
    mm_194: "f32[512, 768]" = torch.ops.aten.mm.default(view_943, permute_781);  permute_781 = None
    permute_782: "f32[3072, 512]" = torch.ops.aten.permute.default(view_943, [1, 0])
    mm_195: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_782, view_32);  permute_782 = view_32 = None
    permute_783: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_195, [1, 0]);  mm_195 = None
    sum_249: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_943, [0], True);  view_943 = None
    view_944: "f32[3072]" = torch.ops.aten.view.default(sum_249, [3072]);  sum_249 = None
    permute_784: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_783, [1, 0]);  permute_783 = None
    view_945: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_194, [1, 512, 768]);  mm_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    add_277: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_488, view_945);  mul_488 = view_945 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_500: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_277, primals_32);  primals_32 = None
    mul_501: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_500, 768)
    sum_250: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_500, [2], True)
    mul_502: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_500, mul_4);  mul_500 = None
    sum_251: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_502, [2], True);  mul_502 = None
    mul_503: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_4, sum_251);  sum_251 = None
    sub_149: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_501, sum_250);  mul_501 = sum_250 = None
    sub_150: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_149, mul_503);  sub_149 = mul_503 = None
    mul_504: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_73, sub_150);  div_73 = sub_150 = None
    mul_505: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_277, mul_4);  mul_4 = None
    sum_252: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_505, [0, 1]);  mul_505 = None
    sum_253: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_277, [0, 1]);  add_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_35: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
    mul_506: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
    mul_507: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_504, mul_506);  mul_506 = None
    clone_103: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_507, memory_format = torch.contiguous_format);  mul_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_946: "f32[512, 768]" = torch.ops.aten.view.default(clone_103, [512, 768]);  clone_103 = None
    mm_196: "f32[512, 768]" = torch.ops.aten.mm.default(view_946, permute_785);  permute_785 = None
    permute_786: "f32[768, 512]" = torch.ops.aten.permute.default(view_946, [1, 0])
    mm_197: "f32[768, 768]" = torch.ops.aten.mm.default(permute_786, view_30);  permute_786 = view_30 = None
    permute_787: "f32[768, 768]" = torch.ops.aten.permute.default(mm_197, [1, 0]);  mm_197 = None
    sum_254: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_946, [0], True);  view_946 = None
    view_947: "f32[768]" = torch.ops.aten.view.default(sum_254, [768]);  sum_254 = None
    permute_788: "f32[768, 768]" = torch.ops.aten.permute.default(permute_787, [1, 0]);  permute_787 = None
    view_948: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_196, [1, 512, 768]);  mm_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_949: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_948, [1, 512, 12, 64]);  view_948 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    slice_51: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_949, 2, 0, 6)
    slice_52: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_949, 2, 6, 12);  view_949 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_950: "f32[512, 384]" = torch.ops.aten.view.default(slice_52, [512, 384]);  slice_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_789: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(slice_51, [0, 2, 1, 3]);  slice_51 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_11 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_789, clone_default_33, clone_default_34, clone_default_35, None, alias_default_23, getitem_276, getitem_277, getitem_278, 0.1, [True, True, True, False], scale = 0.125);  permute_789 = clone_default_33 = clone_default_34 = clone_default_35 = alias_default_23 = getitem_276 = getitem_277 = getitem_278 = None
    getitem_279: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_11[0]
    getitem_280: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_11[1]
    getitem_281: "f32[1, 6, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_11[2];  _scaled_dot_product_efficient_attention_backward_default_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    clone_105: "f32[512, 384]" = torch.ops.aten.clone.default(view_950, memory_format = torch.contiguous_format);  view_950 = None
    view_957: "f32[3072, 64, 1]" = torch.ops.aten.view.default(clone_105, [3072, 64, 1]);  clone_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    bmm_106: "f32[3072, 9, 1]" = torch.ops.aten.bmm.default(permute_795, view_957);  permute_795 = None
    bmm_107: "f32[3072, 64, 9]" = torch.ops.aten.bmm.default(view_957, permute_796);  view_957 = permute_796 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    view_961: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(bmm_107, [1, 512, 384, 9]);  bmm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    view_962: "f32[1, 512, 3456]" = torch.ops.aten.view.default(view_961, [1, 512, 3456]);  view_961 = None
    permute_797: "f32[1, 3456, 512]" = torch.ops.aten.permute.default(view_962, [0, 2, 1]);  view_962 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    view_963: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.view.default(permute_797, [1, 384, 9, 1, 512, 1]);  permute_797 = None
    permute_798: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.permute.default(view_963, [0, 1, 2, 4, 3, 5]);  view_963 = None
    _unsafe_index_put_11: "f32[1, 384, 520, 1]" = torch.ops.aten._unsafe_index_put.default(full_default_19, [None, None, unsqueeze_8, full_default_1], permute_798, True);  full_default_19 = unsqueeze_8 = full_default_1 = permute_798 = None
    constant_pad_nd_23: "f32[1, 384, 512, 1]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_11, [0, 0, -4, -4], 0.0);  _unsafe_index_put_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    squeeze_12: "f32[1, 384, 512]" = torch.ops.aten.squeeze.dim(constant_pad_nd_23, -1);  constant_pad_nd_23 = None
    permute_799: "f32[1, 512, 384]" = torch.ops.aten.permute.default(squeeze_12, [0, 2, 1]);  squeeze_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_965: "f32[512, 384]" = torch.ops.aten.view.default(permute_799, [512, 384]);  permute_799 = None
    mm_198: "f32[512, 768]" = torch.ops.aten.mm.default(view_965, permute_800);  permute_800 = None
    permute_801: "f32[384, 512]" = torch.ops.aten.permute.default(view_965, [1, 0])
    mm_199: "f32[384, 768]" = torch.ops.aten.mm.default(permute_801, view);  permute_801 = None
    permute_802: "f32[768, 384]" = torch.ops.aten.permute.default(mm_199, [1, 0]);  mm_199 = None
    sum_256: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_965, [0], True);  view_965 = None
    view_966: "f32[384]" = torch.ops.aten.view.default(sum_256, [384]);  sum_256 = None
    permute_803: "f32[384, 768]" = torch.ops.aten.permute.default(permute_802, [1, 0]);  permute_802 = None
    view_967: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_198, [1, 512, 768]);  mm_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    add_280: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_504, view_967);  mul_504 = view_967 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    mul_512: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(bmm_106, alias_49);  bmm_106 = None
    sum_257: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_512, [1], True)
    mul_513: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(alias_49, sum_257);  alias_49 = sum_257 = None
    sub_152: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(mul_512, mul_513);  mul_512 = mul_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_968: "f32[1, 512, 54]" = torch.ops.aten.view.default(sub_152, [1, 512, 54]);  sub_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    sum_258: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_968, [0, 1], True)
    view_969: "f32[54]" = torch.ops.aten.view.default(sum_258, [54]);  sum_258 = None
    view_970: "f32[512, 54]" = torch.ops.aten.view.default(view_968, [512, 54]);  view_968 = None
    permute_804: "f32[54, 512]" = torch.ops.aten.permute.default(view_970, [1, 0]);  view_970 = None
    mm_200: "f32[54, 384]" = torch.ops.aten.mm.default(permute_804, view_9);  view_9 = None
    permute_805: "f32[384, 54]" = torch.ops.aten.permute.default(mm_200, [1, 0]);  mm_200 = None
    mm_201: "f32[384, 512]" = torch.ops.aten.mm.default(permute_9, permute_804);  permute_9 = permute_804 = None
    permute_807: "f32[512, 384]" = torch.ops.aten.permute.default(mm_201, [1, 0]);  mm_201 = None
    view_971: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_807, [1, 512, 384]);  permute_807 = None
    permute_808: "f32[54, 384]" = torch.ops.aten.permute.default(permute_805, [1, 0]);  permute_805 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    mul_514: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_971, permute_8);  permute_8 = None
    mul_515: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_971, view_1);  view_971 = view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_809: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_281, [0, 2, 1, 3]);  getitem_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_106: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_809, memory_format = torch.contiguous_format);  permute_809 = None
    view_972: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_106, [1, 512, 384]);  clone_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_810: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_280, [0, 2, 1, 3]);  getitem_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_973: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_810, [1, 512, 384]);  permute_810 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_811: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_279, [0, 2, 1, 3]);  getitem_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_107: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_811, memory_format = torch.contiguous_format);  permute_811 = None
    view_974: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_107, [1, 512, 384]);  clone_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    add_281: "f32[1, 512, 384]" = torch.ops.aten.add.Tensor(mul_514, view_974);  mul_514 = view_974 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    permute_812: "f32[1, 384, 512]" = torch.ops.aten.permute.default(mul_515, [0, 2, 1]);  mul_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    sum_259: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_812, [0, 2], True)
    view_975: "f32[384, 1]" = torch.ops.aten.view.default(sum_259, [384, 1]);  sum_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(permute_812, convolution, primals_25, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_812 = convolution = primals_25 = None
    getitem_192: "f32[1, 768, 512]" = convolution_backward_22[0]
    getitem_193: "f32[384, 768, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(getitem_192, permute_3, primals_24, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_192 = permute_3 = primals_24 = None
    getitem_195: "f32[1, 768, 512]" = convolution_backward_23[0]
    getitem_196: "f32[768, 1, 9]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_813: "f32[1, 512, 768]" = torch.ops.aten.permute.default(getitem_195, [0, 2, 1]);  getitem_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    add_282: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_280, permute_813);  add_280 = permute_813 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_976: "f32[512, 384]" = torch.ops.aten.view.default(view_972, [512, 384]);  view_972 = None
    mm_202: "f32[512, 768]" = torch.ops.aten.mm.default(view_976, permute_814);  permute_814 = None
    permute_815: "f32[384, 512]" = torch.ops.aten.permute.default(view_976, [1, 0])
    mm_203: "f32[384, 768]" = torch.ops.aten.mm.default(permute_815, view);  permute_815 = None
    permute_816: "f32[768, 384]" = torch.ops.aten.permute.default(mm_203, [1, 0]);  mm_203 = None
    sum_260: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_976, [0], True);  view_976 = None
    view_977: "f32[384]" = torch.ops.aten.view.default(sum_260, [384]);  sum_260 = None
    permute_817: "f32[384, 768]" = torch.ops.aten.permute.default(permute_816, [1, 0]);  permute_816 = None
    view_978: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_202, [1, 512, 768]);  mm_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    add_283: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_282, view_978);  add_282 = view_978 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_979: "f32[512, 384]" = torch.ops.aten.view.default(view_973, [512, 384]);  view_973 = None
    mm_204: "f32[512, 768]" = torch.ops.aten.mm.default(view_979, permute_818);  permute_818 = None
    permute_819: "f32[384, 512]" = torch.ops.aten.permute.default(view_979, [1, 0])
    mm_205: "f32[384, 768]" = torch.ops.aten.mm.default(permute_819, view);  permute_819 = None
    permute_820: "f32[768, 384]" = torch.ops.aten.permute.default(mm_205, [1, 0]);  mm_205 = None
    sum_261: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_979, [0], True);  view_979 = None
    view_980: "f32[384]" = torch.ops.aten.view.default(sum_261, [384]);  sum_261 = None
    permute_821: "f32[384, 768]" = torch.ops.aten.permute.default(permute_820, [1, 0]);  permute_820 = None
    view_981: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_204, [1, 512, 768]);  mm_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    add_284: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_283, view_981);  add_283 = view_981 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_982: "f32[512, 384]" = torch.ops.aten.view.default(add_281, [512, 384]);  add_281 = None
    mm_206: "f32[512, 768]" = torch.ops.aten.mm.default(view_982, permute_822);  permute_822 = None
    permute_823: "f32[384, 512]" = torch.ops.aten.permute.default(view_982, [1, 0])
    mm_207: "f32[384, 768]" = torch.ops.aten.mm.default(permute_823, view);  permute_823 = view = None
    permute_824: "f32[768, 384]" = torch.ops.aten.permute.default(mm_207, [1, 0]);  mm_207 = None
    sum_262: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_982, [0], True);  view_982 = None
    view_983: "f32[384]" = torch.ops.aten.view.default(sum_262, [384]);  sum_262 = None
    permute_825: "f32[384, 768]" = torch.ops.aten.permute.default(permute_824, [1, 0]);  permute_824 = None
    view_984: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_206, [1, 512, 768]);  mm_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    add_285: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_284, view_984);  add_284 = view_984 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:236, code: embeddings = self.dropout(embeddings)
    convert_element_type_37: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
    mul_516: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
    mul_517: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_285, mul_516);  add_285 = mul_516 = None
    clone_108: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_517, memory_format = torch.contiguous_format);  mul_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:235, code: embeddings = self.LayerNorm(embeddings)
    mul_519: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(clone_108, primals_16);  primals_16 = None
    mul_520: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_519, 768)
    sum_263: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_519, [2], True)
    mul_521: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_519, mul_1);  mul_519 = None
    sum_264: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_521, [2], True);  mul_521 = None
    mul_522: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_1, sum_264);  sum_264 = None
    sub_154: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_520, sum_263);  mul_520 = sum_263 = None
    sub_155: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_154, mul_522);  sub_154 = mul_522 = None
    mul_523: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_75, sub_155);  div_75 = sub_155 = None
    mul_524: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(clone_108, mul_1);  mul_1 = None
    sum_265: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_524, [0, 1]);  mul_524 = None
    sum_266: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_108, [0, 1]);  clone_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:232, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    eq: "b8[1, 512]" = torch.ops.aten.eq.Scalar(expand, -1)
    unsqueeze_160: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    where_4: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_160, full_default_14, mul_523);  unsqueeze_160 = None
    full_default_43: "f32[2, 768]" = torch.ops.aten.full.default([2, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_12: "f32[2, 768]" = torch.ops.aten._unsafe_index_put.default(full_default_43, [expand], where_4, True);  full_default_43 = expand = where_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:231, code: position_embeddings = self.position_embeddings(position_ids)
    eq_1: "b8[1, 512]" = torch.ops.aten.eq.Scalar(slice_4, -1)
    unsqueeze_161: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    where_5: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_161, full_default_14, mul_523);  unsqueeze_161 = None
    full_default_45: "f32[512, 768]" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_13: "f32[512, 768]" = torch.ops.aten._unsafe_index_put.default(full_default_45, [slice_4], where_5, True);  full_default_45 = slice_4 = where_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:230, code: inputs_embeds = self.word_embeddings(input_ids)
    eq_2: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_290, 0)
    unsqueeze_162: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
    where_6: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_162, full_default_14, mul_523);  unsqueeze_162 = full_default_14 = mul_523 = None
    full_default_47: "f32[30522, 768]" = torch.ops.aten.full.default([30522, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_14: "f32[30522, 768]" = torch.ops.aten._unsafe_index_put.default(full_default_47, [primals_290], where_6, True);  full_default_47 = primals_290 = where_6 = None
    return [view_975, view_930, view_885, view_840, view_795, view_750, view_705, view_660, view_615, view_570, view_525, view_480, _unsafe_index_put_14, _unsafe_index_put_13, _unsafe_index_put_12, sum_265, sum_266, permute_825, view_983, permute_821, view_980, permute_817, view_977, getitem_196, getitem_193, permute_808, view_969, permute_803, view_966, permute_788, view_947, sum_252, sum_253, permute_784, view_944, permute_780, view_941, sum_246, sum_247, permute_776, view_938, permute_772, view_935, permute_768, view_932, getitem_190, getitem_187, permute_759, view_924, permute_754, view_921, permute_739, view_902, sum_233, sum_234, permute_735, view_899, permute_731, view_896, sum_227, sum_228, permute_727, view_893, permute_723, view_890, permute_719, view_887, getitem_184, getitem_181, permute_710, view_879, permute_705, view_876, permute_690, view_857, sum_214, sum_215, permute_686, view_854, permute_682, view_851, sum_208, sum_209, permute_678, view_848, permute_674, view_845, permute_670, view_842, getitem_178, getitem_175, permute_661, view_834, permute_656, view_831, permute_641, view_812, sum_195, sum_196, permute_637, view_809, permute_633, view_806, sum_189, sum_190, permute_629, view_803, permute_625, view_800, permute_621, view_797, getitem_172, getitem_169, permute_612, view_789, permute_607, view_786, permute_592, view_767, sum_176, sum_177, permute_588, view_764, permute_584, view_761, sum_170, sum_171, permute_580, view_758, permute_576, view_755, permute_572, view_752, getitem_166, getitem_163, permute_563, view_744, permute_558, view_741, permute_543, view_722, sum_157, sum_158, permute_539, view_719, permute_535, view_716, sum_151, sum_152, permute_531, view_713, permute_527, view_710, permute_523, view_707, getitem_160, getitem_157, permute_514, view_699, permute_509, view_696, permute_494, view_677, sum_138, sum_139, permute_490, view_674, permute_486, view_671, sum_132, sum_133, permute_482, view_668, permute_478, view_665, permute_474, view_662, getitem_154, getitem_151, permute_465, view_654, permute_460, view_651, permute_445, view_632, sum_119, sum_120, permute_441, view_629, permute_437, view_626, sum_113, sum_114, permute_433, view_623, permute_429, view_620, permute_425, view_617, getitem_148, getitem_145, permute_416, view_609, permute_411, view_606, permute_396, view_587, sum_100, sum_101, permute_392, view_584, permute_388, view_581, sum_94, sum_95, permute_384, view_578, permute_380, view_575, permute_376, view_572, getitem_142, getitem_139, permute_367, view_564, permute_362, view_561, permute_347, view_542, sum_81, sum_82, permute_343, view_539, permute_339, view_536, sum_75, sum_76, permute_335, view_533, permute_331, view_530, permute_327, view_527, getitem_136, getitem_133, permute_318, view_519, permute_313, view_516, permute_298, view_497, sum_62, sum_63, permute_294, view_494, permute_290, view_491, sum_56, sum_57, permute_286, view_488, permute_282, view_485, permute_278, view_482, getitem_130, getitem_127, permute_269, view_474, permute_264, view_471, permute_249, view_452, sum_43, sum_44, permute_245, view_449, permute_241, view_446, sum_37, sum_38, permute_237, view_443, sum_32, sum_33, permute_233, view_440, None, None, None, None]
    